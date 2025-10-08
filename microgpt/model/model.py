import torch
import torch.nn as nn
from torch.nn import functional as F

from microgpt.model.block import Block
from microgpt.config import Config
from microgpt.tokenizer.tokenizer import GPTtokenizer

class GPTModel(nn.Module):
        
    """
    Base GPT model class. 
    Should not be instantiated directly. Inherit from this class to create specific models
    """

    def __init__(self, config: Config, use_lora: bool = True):
        
        """
        Creates the GPT model architecture.

        Args:
            config (Config): Configuration object containing model hyperparameters.
            use_lora (bool): Whether to apply LoRA adapters in the attention mechanism.
        """

        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim),
            dropout = nn.Dropout(p=config.dropout),
            decoder = nn.ModuleList([Block(config, use_lora) for _ in range(config.num_layers)]),
            norm = nn.RMSNorm(config.embedding_dim),
            lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        ))

    def _init_weights(self, module: nn.Module) -> None:

        """
        Initializes the weights of the model.

        Args:
            module (nn.Module): The module to initialize.

        Notes:
            - Linear and Embedding weights are initialized from a normal distribution with mean 0 and std 0.02.
            - Linear bias are initialized as 0.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def calculate_loss(self, xb: torch.Tensor, yb: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        """
        Calculates the loss for the model.

        Args:
            xb (torch.Tensor): The model output tensor of shape (B, T, C).
            yb (torch.Tensor): The target tensor of shape (B, T).
            loss_mask (torch.Tensor | None): Optional mask tensor of shape (B, T)
        
        Returns:
            torch.Tensor: The calculated loss.
        """

        raise NotImplementedError('Subclasses must implement this method')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of the GPT model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T) containing token indices.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """

        # embedding weights
        x = self.transformer.wte(x)
        x = self.transformer.dropout(x)

        # passing through attention & mlp blocks
        for layer in self.transformer.decoder:
            x = layer(x)

        # normalisation
        x = self.transformer.norm(x)

        return x
    
    @torch.no_grad()
    def generate(
        self, 
        tokenizer: GPTtokenizer, 
        text: str, 
        max_new_tokens: int, 
        device: str
    ) -> str:
        
        """
        Generater method to produce text given a prompt.

        Args:
            tokenizer (GPTtokenizer): Tokenizer object for encoding and decoding text.
            text (str): Input prompt text.
            max_new_tokens (int): Maximum number of new tokens to generate.
            device (str): Device to perform computation on ('cpu' or 'cuda').

        Returns:
            str: Generated text.
        """

        raise NotImplementedError('Subclasses must implement this method')
    
class PretrainModel(GPTModel):

    def __init__(self, config: Config, model_dict: dict | None = None, train: bool = True):

        super().__init__(config, use_lora=False)

        if not train and model_dict is not None:
            self.load_state_dict(model_dict, strict=True)
            return

        # weight tying
        self.transformer.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights) 
           
    def calculate_loss(self, xb: torch.Tensor, yb: torch.Tensor, *args) -> torch.Tensor:
        B, T, C = xb.shape
        xb = xb.view(B*T, C)
        yb = yb.view(B*T)
        loss = F.cross_entropy(xb, yb, reduction='none')
        return loss.mean()

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None, *args) -> tuple[torch.Tensor, torch.Tensor | None]:
       
        x = super().forward(x)

        # if training, calculate loss over all tokens
        # if evaluating, only return logits for the last token
        if targets is not None:
            logits = self.transformer.lm_head(x)
            loss = self.calculate_loss(logits, targets)
        else:
            logits = self.transformer.lm_head(x[:, -1, :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        tokenizer: GPTtokenizer, 
        text: str, 
        max_new_tokens: int, 
        device: str
    ) -> str:

        # encodes text and crops to context_size
        context = tokenizer.encode(text)[:, -self.config.context_size+1:].to(device)
        output = []

        for _ in range(max_new_tokens):

            # forward pass and crop to context_size
            context = context[:, -self.config.context_size:]
            logits, _ = self.forward(context)

            # temperature scaling
            logits = logits / self.config.temperature
            
            # top-k sampling 
            probs, idxs = torch.topk(logits, self.config.k)

            probs = F.softmax(probs, dim=-1)
            idx = idxs[0, torch.multinomial(probs, num_samples=1)]

            # stops if eos token is generated
            if idx.item() == tokenizer.eos_token_id:
                return tokenizer.decode(output)

            context = torch.cat([context, idx], dim=1)
            output.append(idx.item())
            
        return tokenizer.decode(output)
    
class FinetuneModel(GPTModel):

    def __init__(self, config: Config, model_dict: dict, train: bool = True):
        super().__init__(config, use_lora=True)

        if not train:
            self.load_state_dict(model_dict, strict=True)
            return

        # freezes all parameters except for layernorm, embedding, lm_head and LoRA adapters
        for param in self.parameters():
            param.requires_grad = False

        for name, param in self.transformer.named_parameters():
            param.requires_grad = any(x in name for x in ["wte", "lm_head", "norm", "A", "B"])

        self.load_state_dict(model_dict, strict=False)

    def calculate_loss(self, xb: torch.Tensor, yb: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = xb.shape
        xb = xb.view(B*T, C)
        yb = yb.view(B*T)
        loss = F.cross_entropy(xb, yb, reduction='none')

        # applies loss mask
        loss_mask = loss_mask.view(B*T)
        loss = loss * loss_mask 
        return loss.sum() / loss_mask.sum().clamp(min=1)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None):

        x = super().forward(x)

        # if training, calculate loss over all tokens
        # if evaluating, only return logits for the last token
        if targets is not None:
            logits = self.transformer.lm_head(x)
            loss = self.calculate_loss(logits, targets, loss_mask=loss_mask)
        else:
            logits = self.transformer.lm_head(x[:, -1, :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        tokenizer: GPTtokenizer, 
        text: str, 
        max_new_tokens: int, 
        device: str
    ) -> str:

        # encodes text and crops to context_size
        context = tokenizer.encode(text)[:, -self.config.context_size+1:].to(device)
        output = []

        for _ in range(max_new_tokens):

            # forward pass and crop to context_size
            context = context[:, -self.config.context_size:]
            logits, _ = self.forward(context)

            # temperature scaling
            logits = logits / self.config.temperature
            
            # top-k sampling
            probs, idxs = torch.topk(logits, self.config.k)

            probs = F.softmax(probs, dim=-1)
            idx = idxs[0, torch.multinomial(probs, num_samples=1)]

            # stops if end_assistant token is generated
            if idx.item() == tokenizer.end_assistant_token_id:
                return tokenizer.decode(output)

            context = torch.cat([context, idx], dim=1)
            output.append(idx.item())
            
        return tokenizer.decode(output)
    
class RewardModel(GPTModel):

    def __init__(self, config: Config, model_dict: dict, train: bool = True):
        super().__init__(config, use_lora=True)

        if not train:
            self.load_state_dict(model_dict, strict=True)
            return

        # replaces lm_head with a linear layer outputting a single scalar value
        self.transformer.lm_head = nn.Linear(self.config.embedding_dim, 1, bias=True)

        # freezes all other parameters except for layernorm, embedding, lm_head and LoRA adapters
        for param in self.parameters():
            param.requires_grad = False

        for name, param in self.transformer.named_parameters():
            param.requires_grad = any(x in name for x in ["lm_head", "norm", "A", "B"])

        self.apply(self._init_weights)
        del model_dict["transformer.lm_head.weight"]
        self.load_state_dict(model_dict, strict=False)

    def calculate_loss(self, sA: torch.Tensor, sB: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(sA - sB).mean()
    
    def masked_mean_pool(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:

        # ignore padding tokens when mean pooling
        mask = (token_ids != self.config.pad_token_id).unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)

        # uses average pooling of non-masked embeddings
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts
    
    def forward(self, accepted: torch.Tensor, rejected: torch.Tensor | None = None, *args) -> tuple[torch.Tensor, torch.Tensor | None]:

        # embeddings for accepted and rejected sequences
        accepted_emb = super().forward(accepted)
        accepted_emb = self.masked_mean_pool(accepted, accepted_emb)
        accepted_emb = self.transformer.lm_head(accepted_emb)

        if rejected is None:
            return accepted_emb, None

        rejected_emb = super().forward(rejected)
        rejected_emb = self.masked_mean_pool(rejected, rejected_emb)
        rejected_emb = self.transformer.lm_head(rejected_emb)

        return (accepted_emb, rejected_emb), self.calculate_loss(accepted_emb, rejected_emb)
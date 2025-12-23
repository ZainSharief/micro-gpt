import torch
import torch.nn as nn
from torch.nn import functional as F

from microgpt.model.block import Block
from microgpt.config import Config
from microgpt.tokenizer.tokenizer import GPTtokenizer

class GPTModel(nn.Module):
        
    def __init__(self, config: Config, use_lora: bool = True):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim),
            dropout = nn.Dropout(p=config.dropout),
            decoder = nn.Sequential(Block(config, use_lora) for _ in range(config.num_layers)),
            norm = nn.RMSNorm(config.embedding_dim),
            lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        ))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer.wte(x)
        x = self.transformer.dropout(x)
        x = self.transformer.decoder(x)
        x = self.transformer.norm(x)
        return x
    
class PretrainModel(GPTModel):

    def __init__(self, config: Config, model_dict: dict | None = None, train: bool = True):
        super().__init__(config, use_lora=False)

        if not train:
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
        
        context = tokenizer.encode(text)[:, -self.config.context_size+1:].to(device)
        output = []

        for _ in range(max_new_tokens):

            context = context[:, -self.config.context_size:]
            logits, _ = self.forward(context)

            logits = logits / self.config.temperature
            probs, idxs = torch.topk(logits, self.config.k)

            probs = F.softmax(probs, dim=-1)
            idx = idxs[0, torch.multinomial(probs, num_samples=1)]

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

        # prevents loss including user tokens
        loss_mask = loss_mask.view(B*T)
        loss = loss * loss_mask 
        return loss.sum() / loss_mask.sum().clamp(min=1)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None):
        x = super().forward(x)

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

        context = tokenizer.encode(text)[:, -self.config.context_size+1:].to(device)
        output = []

        for _ in range(max_new_tokens):

            context = context[:, -self.config.context_size:]
            logits, _ = self.forward(context)

            logits = logits / self.config.temperature
            probs, idxs = torch.topk(logits, self.config.k)

            probs = F.softmax(probs, dim=-1)
            idx = idxs[0, torch.multinomial(probs, num_samples=1)]

            if idx.item() == tokenizer.end_assistant_token_id:
                return tokenizer.decode(output)

            context = torch.cat([context, idx], dim=1)
            output.append(idx.item())
            
        return tokenizer.decode(output)
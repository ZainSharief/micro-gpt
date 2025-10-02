import torch
import torch.nn as nn
from torch.nn import functional as F

from microgpt.modules.block import Block
from microgpt.config import Config

class GPTModel(nn.Module):
        
    def __init__(self, config: Config, use_lora: bool = True) -> None:
        
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim),
            dropout = nn.Dropout(p=config.dropout),
            decoder = nn.ModuleList([Block(config, use_lora) for _ in range(config.num_layers)]),
            norm = nn.RMSNorm(config.embedding_dim),
            lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        ))

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def calculate_loss(self, xb, yb, loss_mask=None):
        raise NotImplementedError()

    def forward(self, x):

        pad_mask = (x != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)

        # embedding weights
        x = self.transformer.wte(x)
        x = self.transformer.dropout(x)

        # passing through attention & mlp blocks
        for layer in self.transformer.decoder:
            x = layer(x)

        # normalisation and out prediction
        x = self.transformer.norm(x)

        return x
    
    @torch.no_grad()
    def generate(self, tokenizer, text, max_new_tokens, device):
        raise NotImplementedError()
    
class PretrainModel(GPTModel):

    def __init__(self, config: Config) -> None:
        super().__init__(config, use_lora=False)

        self.transformer.lm_head.weight = self.transformer.wte.weight # Weight tying     
        self.apply(self._init_weights) 
        assert self.transformer.lm_head.weight.data_ptr() == self.transformer.wte.weight.data_ptr()
           
    def calculate_loss(self, xb, yb, *args):
        B, T, C = xb.shape
        xb = xb.view(B*T, C)
        yb = yb.view(B*T)
        loss = F.cross_entropy(xb, yb, reduction='none')
        return loss.mean()


    def forward(self, x, targets=None, loss_mask=None):

        x = super().forward(x)

        if targets is not None:
            logits = self.transformer.lm_head(x)
            loss = self.calculate_loss(logits, targets)
        else:
            logits = self.transformer.lm_head(x[:, -1, :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, tokenizer, text, max_new_tokens, device):

        # Encodes the text and adjusts size to context_size
        context = tokenizer.encode(text)[:, -self.config.context_size+1:].to(device)
        output = []

        for _ in range(max_new_tokens):

            # Passes through model and takes the last token
            context = context[:, -self.config.context_size:]
            logits, _ = self.forward(context)

            # Uses temperature to scale the logits for softmax
            logits = logits / self.config.temperature
            
            # Uses top-k sampling to get the next token
            probs, idxs = torch.topk(logits, self.config.k)

            probs = F.softmax(probs, dim=-1)
            idx = idxs[:, torch.multinomial(probs, num_samples=1)]

            if idx.item() == tokenizer.eos_token_id:
                return tokenizer.decode(output)

            context = torch.cat([context, idx], dim=1)
            output.append(idx.item())
            
        return tokenizer.decode(output)
    
class FinetuneModel(GPTModel):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        for param in self.parameters():
            param.requires_grad = False

        for name, param in self.transformer.named_parameters():
            param.requires_grad = any(x in name for x in ["wte", "lm_head", "norm", "A", "B"])

    def calculate_loss(self, xb, yb, loss_mask=None):
        B, T, C = xb.shape
        xb = xb.view(B*T, C)
        yb = yb.view(B*T)
        loss = F.cross_entropy(xb, yb, reduction='none')

        loss_mask = loss_mask.view(B*T)
        loss = loss * loss_mask 
        return loss.sum() / loss_mask.sum().clamp(min=1)

    def forward(self, x, targets=None, loss_mask=None):

        x = super().forward(x)

        if targets is not None:
            logits = self.transformer.lm_head(x)
            loss = self.calculate_loss(logits, targets, loss_mask=loss_mask)
        else:
            logits = self.transformer.lm_head(x[:, -1, :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, tokenizer, text, max_new_tokens, device):

        # Encodes the text and adjusts size to context_size
        context = tokenizer.encode(text)[:, -self.config.context_size+1:].to(device)
        output = []

        for _ in range(max_new_tokens):

            # Passes through model and takes the last token
            context = context[:, -self.config.context_size:]
            logits, _ = self.forward(context)

            # Uses temperature to scale the logits for softmax
            logits = logits / self.config.temperature
            
            # Uses top-k sampling to get the next token
            probs, idxs = torch.topk(logits, self.config.k)

            probs = F.softmax(probs, dim=-1)
            idx = idxs[:, torch.multinomial(probs, num_samples=1)].squeeze(0)

            if idx.item() == tokenizer.end_assistant_token_id:
                return tokenizer.decode(output)

            context = torch.cat([context, idx], dim=1)
            output.append(idx.item())
            
        return tokenizer.decode(output)
    
class RewardModel(GPTModel):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        for param in self.parameters():
            param.requires_grad = False

        self.transformer.lm_head = nn.Linear(self.config.embedding_dim, 1, bias=True)

        for name, param in self.transformer.named_parameters():
            param.requires_grad = any(x in name for x in ["wte", "lm_head", "norm", "A", "B"])

        self.apply(self._init_weights)

    def calculate_loss(self, sA, sB):
        return -F.logsigmoid(sA - sB).mean()
    
    def masked_mean_pool(self, token_ids, embeddings):
        mask = (token_ids != self.config.pad_token_id).unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts
    
    def forward(self, accepted, rejected=None, *args):

        accepted_emb = super().forward(accepted)
        accepted_emb = self.masked_mean_pool(accepted, accepted_emb)
        accepted_emb = self.transformer.lm_head(accepted_emb)

        if rejected is None:
            return accepted_emb, None

        rejected_emb = super().forward(rejected)
        rejected_emb = self.masked_mean_pool(rejected, rejected_emb)
        rejected_emb = self.transformer.lm_head(rejected_emb)

        return (accepted_emb, rejected_emb), self.calculate_loss(accepted_emb, rejected_emb)
import torch
import torch.nn as nn
from torch.nn import functional as F

from microgpt.model.attention import MultiHeadAttention
from microgpt.config import Config

class GPTModel(nn.Module):
        
    def __init__(self, config: Config, use_lora: bool = False) -> None:
        
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim),
            dropout = nn.Dropout(p=config.dropout),
            decoder = nn.ModuleList([Block(config, use_lora) for _ in range(config.num_layers)]),
            norm = nn.RMSNorm(config.embedding_dim),
            lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        ))

        # Weight tying
        self.transformer.wte.weight = self.transformer.lm_head.weight

        # Initialise the model weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)       

    def calculate_loss(self, xb, yb):
        B, T, C = xb.shape
        xb = xb.view(B*T, C)
        yb = yb.view(B*T)
        loss = F.cross_entropy(xb, yb, ignore_index=self.pad_token_id)
        return loss 

    def forward(self, x, targets=None):

        pad_mask = (x != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)

        # embedding weights
        x = self.transformer.wte(x)
        x = self.transformer.dropout(x)

        # passing through attention & mlp blocks
        for layer in self.transformer.decoder:
            x = layer(x, pad_mask)

        # normalisation and out prediction
        x = self.transformer.norm(x)
        
        if targets is not None:
            logits = self.transformer.lm_head(x)
            loss = self.calculate_loss(logits, targets)
        else:
            logits = self.transformer.lm_head(x[:, -1, :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, tokeniser, text, temperature, k, max_new_tokens, device):

        # Encodes the text and adjusts size to context_size
        context = tokeniser.encode(text)[:, -self.config.context_size+1:].to(device)
        output = []

        for _ in range(max_new_tokens):

            # Passes through model and takes the last token
            context = context[:, -self.config.context_size:]
            logits, _ = self.forward(context)

            # Uses temperature to scale the logits for softmax
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # Uses top-k sampling to get the next token
            probs, idxs = torch.topk(probs, k)

            idx = idxs[0, torch.multinomial(probs, 1)]
            
            context = torch.cat((context, idx), dim=1)
            output.append(idx[0, 0].int())
            
        return tokeniser.decode(output)

class Block(nn.Module):

    def __init__(self, config: Config, use_lora: bool = False) -> None:
        super().__init__()
        self.rmsnorm_1 = nn.RMSNorm(config.embedding_dim)
        self.multiheadattention = MultiHeadAttention(config, use_lora)
        self.dropout_1 = nn.Dropout(config.dropout)

        self.rmsnorm_2 = nn.RMSNorm(config.embedding_dim)
        self.mlp = MLP(config.embedding_dim, config.projection)
        self.dropout_2 = nn.Dropout(config.dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.dropout_1(self.multiheadattention(self.rmsnorm_1(x), pad_mask))
        x = x + self.dropout_2(self.mlp(self.rmsnorm_2(x)))
        return x    

class MLP(nn.Module):

    def __init__(self, embedding_dim, projection=4):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, projection * embedding_dim),
            nn.GELU(),
            nn.Linear(projection * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.feedforward(x)
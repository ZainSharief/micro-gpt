import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPTModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int, num_heads: int, num_layers: int, device: str, projection: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.context_size = context_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embedding_dim),
            dropout = nn.Dropout(p=dropout),
            decoder = nn.Sequential(*[Block(embedding_dim, num_heads, context_size, device, projection, dropout) for _ in range(num_layers)]),
            norm = nn.RMSNorm(embedding_dim),
            lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        ))

        # Weight tying
        self.transformer.wte.weight = self.transformer.lm_head.weight

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
        loss = F.cross_entropy(xb, yb)
        return loss 

    def forward(self, x, targets=None):

        # embedding weights
        x = self.transformer.wte(x)
        x = self.transformer.dropout(x)

        # passing through attention & mlp blocks
        x = self.transformer.decoder(x)

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
        context = tokeniser.encode(text)[:, -self.context_size:].to(device)
        output = []

        for _ in range(max_new_tokens):

            # Passes through model and takes the last token
            context = context[:, -self.context_size:]
            logits, _ = self.forward(context)

            # Uses temperature to scale the logits for softmax
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # Uses top-k sampling to get the next token
            probs, idxs = torch.topk(probs, k)

            idx = idxs[0][torch.multinomial(probs, 1)]
            
            context = torch.cat((context, idx), dim=1)
            output.append(idx[0][0].int())
            
        return tokeniser.decode(output)

class Block(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size, device, projection=4, dropout=0.1):
        super().__init__()
        self.rmsnorm_1 = nn.RMSNorm(embedding_dim)
        self.multiheadattention = MultiHeadAttention(embedding_dim, num_heads, context_size, device, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)

        self.rmsnorm_2 = nn.RMSNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, projection)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout_1(self.multiheadattention(self.rmsnorm_1(x)))
        x = x + self.dropout_2(self.mlp(self.rmsnorm_2(x)))
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size, device, flash: bool = True, dropout: float = 0.0):
        super().__init__()
        self.flash = flash

        assert embedding_dim % num_heads == 0, 'embedding_dim must be divisible by num_heads'

        self.head_size = embedding_dim // num_heads
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.attn = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_size, context_size, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(1)
        self.register_buffer("mask", mask)

        self.rope = RotaryPositionalEmbedding(embedding_dim, context_size)

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.attn(x).split(self.embedding_dim, dim=2)

        q = self.rope(q)
        k = self.rope(k)

        # (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        if self.flash:
            x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        else:
            wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            wei = wei.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            x = wei @ v
            
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        return x

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()

        theta = (10_000 ** ((-2 * torch.arange(0, dim//2, dtype=torch.float)) / dim)).unsqueeze(0)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        angles = position * theta

        self.register_buffer("cos", torch.cos(angles).unsqueeze(0).unsqueeze(-1))
        self.register_buffer("sin", torch.sin(angles).unsqueeze(0).unsqueeze(-1))

    def forward(self, x):
        B, T, C = x.size()
        x = x.view(B, T, C // 2, 2) 
        x_rot = torch.cat([-x[..., 1:2], x[..., 0:1]], dim=-1)

        x_out = x * self.cos[:, :T] + x_rot * self.sin[:, :T]
        return x_out.view(B, T, C)

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
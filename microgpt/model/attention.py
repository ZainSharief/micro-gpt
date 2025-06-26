import torch
import torch.nn as nn
import torch.nn.functional as F

from microgpt.config import Config

class MultiHeadAttention(nn.Module):

    def __init__(self, config: Config, use_lora: bool = False) -> None:

        super().__init__()
        assert config.embedding_dim % config.num_heads == 0, 'embedding_dim must be divisible by num_heads'

        self.head_size = config.embedding_dim // config.num_heads
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        self.attn = LoRALinear(self.embedding_dim, 3 * self.embedding_dim, config.lora_rank, config.lora_alpha) \
            if use_lora else nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        
        self.rope = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=config.context_size)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x, pad_mask=None):
        B, T, C = x.size()

        q, k, v = self.attn(x).split(self.embedding_dim, dim=2)

        # (B, T, num_heads, head_size)
        q = q.view(B, T, self.num_heads, self.head_size)
        k = k.view(B, T, self.num_heads, self.head_size)
        v = v.view(B, T, self.num_heads, self.head_size)

        q = self.rope(q)
        k = self.rope(k)

        # (B, num_heads, T, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = torch.tril(torch.ones(1, 1, q.size(-2), k.size(-2), dtype=torch.bool, device=x.device))
        attn_mask = attn_mask if pad_mask is None else attn_mask & pad_mask

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        return x

class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos = None) -> torch.Tensor:
       
        seq_len = x.size(1)
        rope_cache = (self.cache[:seq_len] if input_pos is None else self.cache[input_pos])
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack([
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], 
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], -1)

        x_out = x_out.flatten(3)
        return x_out.type_as(x)
    
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, lora_dropout: float = 0.1):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        
        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())
        
        self.A = nn.Parameter(torch.randn(in_features, self.rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(self.rank, out_features))
        self.dropout = nn.Dropout(lora_dropout)
        
    def forward(self, x):
        
        x1 = self.linear(x)
        x2 = self.alpha * (x @ self.A @ self.B)
        x2 = self.dropout(x2)
        return x1 + x2
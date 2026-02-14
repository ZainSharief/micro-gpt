import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import math
from microgpt import Config
from .rope import RotaryPositionalEmbeddings

class MultiHeadAttention(nn.Module):

    def __init__(self, config: Config, use_lora: bool = False, dropout: float = 0.0):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0, 'embedding_dim must be divisible by num_heads'
        
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = dropout
        self.use_lora = use_lora

        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.context_size)

        self.wq = nn.Linear(self.embedding_dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.embedding_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.embedding_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.embedding_dim, bias=False)

        if self.use_lora:
            self.wq = LoRALinear(self.wq, rank=config.lora_rank, alpha=config.lora_alpha, dropout=dropout)
            self.wk = LoRALinear(self.wk, rank=config.lora_rank, alpha=config.lora_alpha, dropout=dropout)
            self.wv = LoRALinear(self.wv, rank=config.lora_rank, alpha=config.lora_alpha, dropout=dropout)

    def repeat_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        B, T, n_kv, d = x.shape
        if num_repeats == 1:
            return x
        
        # (B, T, n_kv, 1, d) -> (B, T, n_kv, num_repeats, d) -> (B, T, n_heads, d)
        return x[:, :, :, None, :].expand(B, T, n_kv, num_repeats, d).reshape(B, T, n_kv * num_repeats, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # (B, T, num_heads, head_dim)
        q = self.rope(self.wq(x).view(B, T, self.num_heads, self.head_dim))

        # (B, T, num_kv_heads, head_dim)
        k = self.rope(self.wk(x).view(B, T, self.num_kv_heads, self.head_dim))
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim)

        # (B, T, num_kv_heads, D) -> (B, T, num_heads, D)
        k = self.repeat_kv(k, self.num_kv_groups)
        v = self.repeat_kv(v, self.num_kv_groups)

        # (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(x)
    
class LoRALinear(nn.Module):

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()

        self.base = base_linear
        self.A = nn.Parameter(torch.randn(base_linear.in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, base_linear.out_features))
        self.scaling = alpha / self.A.size(1)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = (self.dropout(x) @ self.A @ self.B) * self.scaling
        return self.base(x) + lora_out
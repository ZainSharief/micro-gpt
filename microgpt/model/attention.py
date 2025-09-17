import torch
import torch.nn as nn
import torch.nn.functional as F

from microgpt.config import Config
from microgpt.model.rope import RotaryPositionalEmbeddings
from microgpt.model.lora import LoRALinear

class MultiHeadAttention(nn.Module):

    def __init__(self, config: Config, use_lora: bool = False) -> None:

        super().__init__()
        assert config.embedding_dim % config.num_heads == 0, 'embedding_dim must be divisible by num_heads'
        
        self.head_size = config.embedding_dim // config.num_heads
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout
        self.use_lora = use_lora

        self.attn = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        if use_lora:
            self.lora_attn = LoRALinear(self.attn, config.lora_rank, config.lora_alpha) 
        
        self.rope = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=config.context_size)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x, pad_mask=None):
        B, T, C = x.size()

        if self.use_lora:
            q, k, v = self.lora_attn(x).split(self.embedding_dim, dim=2)
        else:
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from microgpt import Config
from .attention import MultiHeadAttention
from .lora import LoRALinear

class Block(nn.Module):

    def __init__(self, config: Config, use_lora: bool = False, dropout: float = 0.0):
        super().__init__()

        self.rmsnorm_1 = nn.RMSNorm(config.embedding_dim)
        self.multiheadattention = MultiHeadAttention(config, use_lora)
        self.dropout_1 = nn.Dropout(dropout)

        self.rmsnorm_2 = nn.RMSNorm(config.embedding_dim)
        self.mlp = MLP(config, use_lora, dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.residual_scale = 1.0 / (2 * config.num_layers) ** 0.5  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (self.residual_scale * self.dropout_1(self.multiheadattention(self.rmsnorm_1(x))))
        x = x + (self.residual_scale * self.dropout_2(self.mlp(self.rmsnorm_2(x))))
        return x    

class MLP(nn.Module):

    def __init__(self, config: Config, use_lora: bool = False, dropout: float = 0.0):
        super().__init__()

        # SwiGLU MLP
        hidden_dim = config.embedding_dim * config.projection
        self.w1 = nn.Linear(config.embedding_dim, hidden_dim, bias=False) # Gate Projection
        self.w2 = nn.Linear(config.embedding_dim, hidden_dim, bias=False) # Value Projection
        self.w3 = nn.Linear(hidden_dim, config.embedding_dim, bias=False) # Output Projection

        if use_lora:
            self.w1 = LoRALinear(self.w1, rank=config.lora_rank, alpha=config.lora_alpha, dropout=dropout)
            self.w2 = LoRALinear(self.w2, rank=config.lora_rank, alpha=config.lora_alpha, dropout=dropout)
            self.w3 = LoRALinear(self.w3, rank=config.lora_rank, alpha=config.lora_alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
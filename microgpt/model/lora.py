import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
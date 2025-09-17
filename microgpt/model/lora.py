import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank, alpha, dropout=0.1):
        super().__init__()

        self.base = base_linear
        self.A = nn.Parameter(torch.randn(base_linear.in_features, rank) * (1 / rank ** 0.5))
        self.B = nn.Parameter(torch.zeros(rank, base_linear.out_features))
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.dropout(x @ self.A @ self.B) * self.alpha
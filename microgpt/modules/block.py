import torch
import torch.nn as nn

from microgpt.config import Config
from microgpt.modules.attention import MultiHeadAttention

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
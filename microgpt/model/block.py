import torch
import torch.nn as nn

from microgpt.config import Config
from microgpt.model.attention import MultiHeadAttention

class Block(nn.Module):

    def __init__(self, config: Config, use_lora: bool = False):

        """
        Transformer block consisting of multi-head self-attention and MLP,
        with RMSNorm and residual connections.

        Args:
            config (Config): Configuration object containing model hyperparameters.
            use_lora (bool): Whether to apply LoRA adapters in the attention mechanism.
        """

        super().__init__()

        self.rmsnorm_1 = nn.RMSNorm(config.embedding_dim)
        self.multiheadattention = MultiHeadAttention(config, use_lora)
        self.dropout_1 = nn.Dropout(config.dropout)

        self.rmsnorm_2 = nn.RMSNorm(config.embedding_dim)
        self.mlp = MLP(config.embedding_dim, config.projection)
        self.dropout_2 = nn.Dropout(config.dropout)

        self.residual_scale = 1.0 / (2 * config.num_layers) ** 0.5  

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:

        """
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            pad_mask (torch.Tensor, optional): Boolean mask to ignore padding tokens, shape (B, T).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).

        Note:
            - When training, I did not remember to use residual scaling. In future work, remove the following line 
              to enable it.
        """
        self.residual_scale = 1.0 # remove this line to enable residual scaling

        x = x + (self.residual_scale * self.dropout_1(self.multiheadattention(self.rmsnorm_1(x), pad_mask)))
        x = x + (self.residual_scale * self.dropout_2(self.mlp(self.rmsnorm_2(x))))
        return x    

class MLP(nn.Module):

    def __init__(self, embedding_dim: int, projection: int = 4):

        """
        Feedforward MLP module.

        Args:
            embedding_dim (int): The input and output dimension of the MLP.
            projection (int): The expansion factor for the hidden layer.
        """

        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, projection * embedding_dim),
            nn.GELU(),
            nn.Linear(projection * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """

        return self.feedforward(x)
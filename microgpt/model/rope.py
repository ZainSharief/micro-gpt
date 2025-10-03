import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):

    """
    This implemenation is an adaptation of the implementation at:
    https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000):

        """
        Rotary Positional Embeddings (RoPE) module.
        
        Args:
            dim (int): Dimension of the embeddings (must be even).
            max_seq_len (int): Maximum sequence length.
            base (int): Base for the rotary embeddings.
        """

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

        """
        Forward pass of the rotary positional embeddings.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, num_heads, head_size).
            input_pos (torch.Tensor, optional): Tensor of shape (T,) indicating the position indices for each token.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, num_heads, head_size).
        """
       
        seq_len = x.size(1)
        rope_cache = (self.cache[:seq_len].unsqueeze(0).unsqueeze(2) if input_pos is None else self.cache[input_pos].unsqueeze(2))
        xshaped = x.reshape(*x.shape[:-1], -1, 2)

        x_out = torch.stack([
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], 
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], -1)

        x_out = x_out.flatten(3)
        return x_out.type_as(x)
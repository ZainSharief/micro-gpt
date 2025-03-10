from dataclasses import dataclass

@dataclass
class config:
    embedding_dim: int = 768
    context_size: int = 256
    num_heads: int = 16
    num_layers: int = 12
    max_norm: float = 1.0
    dropout: float = 0.2

    k: int = 50
    p: float = 0.9
    temperature: float = 1.2

    model_path: str = 'model_final.pth'
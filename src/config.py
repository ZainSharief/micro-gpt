from dataclasses import dataclass

@dataclass
class config:
    embedding_dim: int = 768
    context_size: int = 128
    num_heads: int = 16
    num_layers: int = 12
    max_norm: float = 3.0
    dropout: float = 0.1

    k: int = 20
    p: float = 0.9
    temperature: float = 0.8

    base_model_path: str = 'base_model_final.pth'
    fine_tuned_model_path: str = 'fine_tuned_model_final.pth'
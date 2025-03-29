from dataclasses import dataclass

@dataclass
class config:
    embedding_dim: int = 768
    context_size: int = 256
    num_heads: int = 16
    num_layers: int = 12
    max_norm: float = 1.0
    dropout: float = 0.1

    lora_rank: int = 16
    lora_alpha: float = 16

    k: int = 40
    p: float = 0.9
    temperature: float = 0.6

    base_model_path: str = 'base_model_final.pth'
    fine_tuned_model_path: str = 'fine_tuned_model_final.pth'
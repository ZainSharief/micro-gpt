from dataclasses import dataclass

@dataclass
class Config:
    
    embedding_dim: int = 768
    context_size: int = 1024
    num_heads: int = 16
    num_kv_heads: int = 4
    num_layers: int = 12
    max_norm: float = 1.0
    projection: int = 4

    vocab_size: int = 50304 # round to the nearest 128

    lora_rank: int = 16
    lora_alpha: float = 16

    k: int = 50
    p: float = 0.92
    temperature: float = 0.7

    base_model_path: str = 'weights/base_model.pth'
    fine_tuned_model_path: str = 'weights/fine_tuned_model.pth'

from dataclasses import dataclass

@dataclass
class Config:
    
    embedding_dim: int = 768
    context_size: int = 384
    num_heads: int = 16
    num_layers: int = 12
    max_norm: float = 1.0
    dropout: float = 0.1
    projection: int = 4

    vocab_size: int = 50261
    pad_token_id: int = 0

    lora_rank: int = 16
    lora_alpha: float = 16

    k = 50
    p = 0.92
    temperature = 0.7

    base_model_path: str = 'weights/base_model.pth'
    fine_tuned_model_path: str = 'weights/fine_tuned_model.pth'
from dataclasses import dataclass

@dataclass
class Config:
    
    """
    Configuration class for model hyperparameters and settings.

    Attributes:
        embedding_dim (int): Dimension of the token embeddings.
        context_size (int): Maximum context size (sequence length).
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        max_norm (float): Maximum norm for gradient clipping.
        dropout (float): Dropout rate.
        projection (int): Expansion factor for the MLP hidden layer.
        vocab_size (int): Size of the vocabulary.
        pad_token_id (int): Token ID used for padding.
        lora_rank (int): Rank for LoRA adapters.
        lora_alpha (int): Scaling factor for LoRA adapters.
        k (int): Top-k sampling parameter.
        p (float): Top-p (nucleus) sampling parameter.
        temperature (float): Temperature for sampling.
        base_model_path (str): Path to the base model weights.
        fine_tuned_model_path (str): Path to the fine-tuned model weights.
    """

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
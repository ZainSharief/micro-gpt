import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPTModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, num_heads, num_layers, device, projection=4, dropout=0.1):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, context_size, dropout)
        self.blocks = nn.ModuleList([Block(embedding_dim, num_heads, context_size, device, projection, dropout) for _ in range(num_layers)])
        self.out_projection = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        padding_mask = (x == 0).float()
        padding_mask = padding_mask.masked_fill(padding_mask == 1, float('-inf'))
        padding_mask = padding_mask.unsqueeze(1)
        padding_mask = padding_mask.expand(-1, x.size(1), -1)

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, padding_mask)
        return self.out_projection(x)

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEncoding(context_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        token_embedding += self.positional_embedding(token_embedding)
        return self.dropout(token_embedding)

class PositionalEncoding(nn.Module):

    def __init__(self, context_size, d_model):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(context_size, d_model)
        position = torch.arange(context_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:, :x.size(1), :]
        return x

class Block(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size, device, projection=4, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.multiheadattention = MultiHeadAttention(embedding_dim, num_heads, context_size, device)
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, projection)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        x = x + self.dropout1(self.multiheadattention(self.layernorm1(x), padding_mask))
        x = x + self.dropout2(self.feedforward(self.layernorm2(x)))
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size, device):
        super().__init__()
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        self.attn_mask = torch.tril(torch.ones(context_size, context_size, device=device), diagonal=-1)
        self.attn_mask = self.attn_mask.masked_fill(self.attn_mask == 1, float('-inf'))

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            attn_mask = self.attn_mask + padding_mask
        else:
            attn_mask = self.attn_mask

        out, _ = self.multihead_attention(x, x, x, attn_mask=attn_mask, need_weights=False, is_causal=True)
        return out

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, projection=4):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, projection * embedding_dim),
            nn.GELU(),
            nn.Linear(projection * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.feedforward(x)
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPTModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, num_heads, num_layers, device, projection=4, dropout=0.1):
        super().__init__()
        self.context_size = context_size
        self.device = device

        self.embedding = Embedding(vocab_size, embedding_dim, context_size, dropout)
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads, device, projection, dropout) for _ in range(num_layers)])
        self.out_projection = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        return self.out_projection(x)

    @torch.no_grad()
    def generate(self, tokeniser, text, temperature, k, max_new_tokens):

        # Encodes the text and adjusts size to context_size
        context = tokeniser.encode(text)[-self.context_size:]        
        context.to(self.device)

        output = ''
        for _ in range(max_new_tokens):

            # Passes through model and takes the last token
            context = context[:, -self.context_size:] 
            logits = self.forward(context)[:, -1, :] 
            logits[:, 0] = float('-inf')

            # Uses temperature to scale the logits for softmax
            logits = logits / temperature           
            probs = F.softmax(logits, dim=-1)

            # Uses top-k sampling to get the next token
            probs, idxs = torch.topk(probs, k)   
            idx = idxs[0][torch.multinomial(probs, 1)]
            
            context = torch.cat((context, idx), dim=1)
            idx = idx[0].int()
            output += tokeniser.decode(idx)

            # Returns the sequence if the end of sequence is reached
            if tokeniser.eos_token in output:
                return output.split(tokeniser.eos_token)[0]

        return output

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_embedding = PositionalEncoding(context_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        return self.dropout(x)

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

    def __init__(self, embedding_dim, num_heads, device, projection=4, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.multiheadattention = MultiHeadAttention(embedding_dim, num_heads, dropout, device)
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, projection)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.multiheadattention(self.layernorm1(x)))
        x = x + self.dropout2(self.feedforward(self.layernorm2(x)))
        return x
    
class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout, device):
        super().__init__()
        self.device = device

        assert embedding_dim % num_heads == 0, 'embedding_dim must be divisible by num_heads'

        self.head_size = embedding_dim // num_heads
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.attn = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()

        mask = torch.tril(torch.ones(T, T, device=self.device)).view(1, 1, T, T)

        qkv = self.attn(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)

        # (B, num_heads, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        wei = wei.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        x = wei @ v
        x = x.transpose(1, 2).contiguous().view(B, T, C)

        x = self.proj(x)
        return x

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
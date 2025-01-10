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
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads, context_size, device, projection, dropout) for _ in range(num_layers)])
        self.out_projection = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        return self.out_projection(x)
    
    @torch.no_grad()
    def generate(self, tokeniser, text, temperature, k, max_new_tokens):

        # Encodes the text and adjusts size to context_size
        context = tokeniser.encode(text)[-self.context_size:]        
        context = [0]*(self.context_size - len(context)) + context   

        context = torch.tensor(context).unsqueeze(0)
        context.to(self.device)

        output = []
        for _ in range(max_new_tokens):

            # Passes through model and takes the last token
            context = context[:, -self.context_size:] 
            logits = self.forward(context)[:, -1, :] 

            # Uses temperature to scale the logits for softmax
            logits = logits / temperature           
            probs = F.softmax(logits, dim=-1)

            # Uses top-k sampling to get the next token
            probs, idxs = torch.topk(probs, k)      
            idx = idxs[0][torch.multinomial(probs, 1)]

            # Returns the sequence if the end of sequence is reached
            if idx == tokeniser.eos_token:
                return tokeniser.decode(output)
            
            context = torch.cat((context, idx), dim=1)
            output.append(idx[0].int())

        return tokeniser.decode(output)

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
        self.multiheadattention = MultiHeadAttention(context_size, embedding_dim, num_heads, device)
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, projection)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.multiheadattention(self.layernorm1(x)))
        x = x + self.dropout2(self.feedforward(self.layernorm2(x)))
        return x
    
class MultiHeadAttention(nn.Module):

    def __init__(self, context_size, embedding_dim, num_heads, device):
        super().__init__()

        assert embedding_dim % num_heads == 0, 'embedding_dim must be divisible by num_heads'

        self.head_size = embedding_dim // num_heads
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        mask = torch.tril(torch.ones(context_size, context_size, device=device))
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x).view(B, self.num_heads, T, self.head_size)
        k = self.key(x).view(B, self.num_heads, T, self.head_size)
        v = self.value(x).view(B, self.num_heads, T, self.head_size)

        wei = (q @ k.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
        wei = wei.masked_fill(self.mask == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)

        x = wei @ v
        x = x.transpose(1, 2).contiguous().view(B, T, C)

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
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPTModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, num_heads, num_layers, device, projection=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.context_size = context_size
        self.device = device

        self.embedding = Embedding(vocab_size, embedding_dim, context_size, dropout)
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads, context_size, device, projection, dropout) for _ in range(num_layers)])
        self.out_projection = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()          

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        return self.out_projection(x)

    @torch.no_grad()
    def generate(self, tokeniser, text, temperature, k, max_new_tokens):

        training_mode = self.training 
        self.eval()

        # Encodes the text and adjusts size to context_size
        context = tokeniser.encode(text)[-self.context_size:].to(self.device)

        output = ''
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
            
            context = torch.cat((context, idx), dim=1)
            idx = idx[0].int()
            output += tokeniser.decode(idx)

        self.train(training_mode)
            
        return output

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, dropout):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.token_embedding(x)
        return self.dropout(x)

class Block(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size, device, projection=4, dropout=0.1):
        super().__init__()
        self.rmsnorm1 = nn.RMSNorm(embedding_dim)
        self.multiheadattention = MultiHeadAttention(embedding_dim, num_heads, context_size, dropout, device)
        self.dropout1 = nn.Dropout(dropout)

        self.rmsnorm2 = nn.RMSNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, projection)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.multiheadattention(self.rmsnorm1(x)))
        x = x + self.dropout2(self.feedforward(self.rmsnorm2(x)))
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size, dropout, device):
        super().__init__()
        self.device = device

        assert embedding_dim % num_heads == 0, 'embedding_dim must be divisible by num_heads'

        self.head_size = embedding_dim // num_heads
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.attn = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_size, context_size, device=self.device, dtype=torch.long)).unsqueeze(0).unsqueeze(1)
        self.register_buffer("mask", mask)

        self.rope = RotaryPositionalEmbedding(embedding_dim, context_size)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        q = self.rope(q)
        k = self.rope(k)

        # (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        wei = wei.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        x = wei @ v
        x = x.transpose(1, 2).contiguous().view(B, T, C)

        x = self.proj(x)
        return x

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        theta = (10_000 ** ((-2 * torch.arange(0, dim//2, dtype=torch.float)) / dim)).unsqueeze(0)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        angles = position * theta

        self.register_buffer("cos", torch.cos(angles).unsqueeze(0).unsqueeze(-1))
        self.register_buffer("sin", torch.sin(angles).unsqueeze(0).unsqueeze(-1))

    def forward(self, x):
        B, T, C = x.size()
        x = x.view(B, T, C // 2, 2) 
        x_rot = torch.cat([-x[..., 1:2], x[..., 0:1]], dim=-1)

        x_out = x * self.cos[:, :T] + x_rot * self.sin[:, :T]
        return x_out.view(B, T, C)

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
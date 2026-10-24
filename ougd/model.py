from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F

BLOCKS = ('O', 'U', 'G', 'D')

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 768
    hidden_dim: int = 2048
    layers: int = 12
    heads: int = 12
    kv_heads: int = 4
    context: int = 2048
    rope_theta: float = 10000.

    def __post_init__(self):
        if self.dim % self.heads or self.heads % self.kv_heads or (self.dim // self.heads) % 2:
            raise ValueError('dim/heads must be even; heads must be divisible by kv_heads')
        if min(self.vocab_size, self.dim, self.hidden_dim, self.layers, self.context) < 1:
            raise ValueError('model dimensions must be positive')

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.dim, c.dim, bias=False)
        self.k = nn.Linear(c.dim, c.kv_heads * (c.dim // c.heads), bias=False)
        self.v = nn.Linear(c.dim, c.kv_heads * (c.dim // c.heads), bias=False)
        self.O = nn.Linear(c.dim, c.dim, bias=False)

    def rope(self, x):
        d = x.shape[-1]
        freq = self.c.rope_theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
        phase = torch.arange(x.shape[-2], device=x.device)[:, None] * freq[None, :]
        co, si = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * co - b * si, a * si + b * co), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, _ = x.shape
        h, kh, d = self.c.heads, self.c.kv_heads, self.c.dim // self.c.heads
        q = self.rope(self.q(x).view(b, t, h, d).transpose(1, 2))
        k = self.rope(self.k(x).view(b, t, kh, d).transpose(1, 2)).repeat_interleave(h // kh, 1)
        v = self.v(x).view(b, t, kh, d).transpose(1, 2).repeat_interleave(h // kh, 1)
        # Explicit math attention: supports reverse-over-reverse Hessians on CPU/CUDA.
        score = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        prob = score.masked_fill(mask, float('-inf')).softmax(-1)
        return self.O((prob @ v).transpose(1, 2).reshape(b, t, -1))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.dim)
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c.dim)
        self.U = nn.Linear(c.dim, c.hidden_dim, bias=False)
        self.G = nn.Linear(c.dim, c.hidden_dim, bias=False)
        self.D = nn.Linear(c.hidden_dim, c.dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        h = self.ffn_norm(x)
        return x + self.D(F.silu(self.G(h)) * self.U(h))

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        if ids.shape[1] > self.config.context:
            raise ValueError('sequence exceeds context')
        h = self.embedding(ids)
        for block in self.blocks:
            h = block(h)
        return F.linear(self.norm(h), self.embedding.weight)

    def selected(self, layer):
        b = self.blocks[layer]
        return dict(zip(BLOCKS, (b.attn.O.weight, b.U.weight, b.G.weight, b.D.weight)))

def ce_sum(model, batch):
    logits = model(batch['input_ids'])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch['labels'].reshape(-1),
                           ignore_index=-100, reduction='sum')

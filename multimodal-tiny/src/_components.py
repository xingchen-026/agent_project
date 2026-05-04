#!/usr/bin/env python3
"""Foundation components: RMSNorm, RotaryEmbedding, SwiGLU."""

import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos(), freqs.sin()


def apply_rotary(x, cos, sin):
    T = x.shape[-2]
    half = x.shape[-1] // 2
    cos = cos[:T].reshape(1, 1, T, half).to(x.dtype)
    sin = sin[:T].reshape(1, 1, T, half).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = dim * hidden_mult
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

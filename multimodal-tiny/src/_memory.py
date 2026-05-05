#!/usr/bin/env python3
"""Memory Bank — Perceiver-style cross-modal sensory compression."""

import torch
import torch.nn as nn
from _components import RMSNorm, SwiGLU


class MemoryBank(nn.Module):
    """Compresses sensory patches into fixed-size memory tokens via cross-attention.

    Input:  sensory_patches [B, N_sensory, dim]  (variable length)
    Output: memory_tokens   [B, n_mem, dim]      (fixed size)
    """
    def __init__(self, dim, n_mem=16, n_heads=8, mlp_mult=4):
        super().__init__()
        self.n_mem = n_mem
        self.memory = nn.Parameter(torch.empty(n_mem, dim))
        torch.nn.init.normal_(self.memory, std=0.02)

        head_dim = dim // n_heads
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)

    def forward(self, sensory):
        B = sensory.shape[0]
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(mem, sensory, sensory)
        mem = self.norm1(mem + attn_out)
        mem = self.norm2(mem + self.mlp(mem))
        return mem

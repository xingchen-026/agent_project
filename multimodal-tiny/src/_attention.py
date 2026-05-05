#!/usr/bin/env python3
"""Self-attention with KV cache + QK normalization + RoPE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from _components import RMSNorm, SwiGLU, apply_rotary


class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin, mask=None, past_kv=None, use_cache=False):
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        is_causal = (mask is None and past_kv is None)
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=mask, is_causal=is_causal)
        return self.proj(out.transpose(1, 2).reshape(B, T, D)), present_kv


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, head_dim, mlp_multiplier=4):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_heads, head_dim)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_multiplier)

    def forward(self, x, cos, sin, mask=None, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.attn_norm(x), cos, sin,
                                         mask=mask, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return (x, present_kv) if use_cache else x

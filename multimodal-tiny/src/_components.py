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


class MoE_SwiGLU(nn.Module):
    """Mixture of Experts — SwiGLU FFN with top-k gating.
    Same FLOPs as single SwiGLU, but 4-8x more capacity.
    """
    def __init__(self, dim, n_experts=8, top_k=2, expert_mult=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)

        # Each expert: smaller SwiGLU (expert_mult instead of 4)
        expert_hidden = dim * expert_mult
        self.w1 = nn.Parameter(torch.empty(n_experts, dim, expert_hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, dim, expert_hidden))
        self.w3 = nn.Parameter(torch.empty(n_experts, expert_hidden, dim))
        self._init_weights()

    def _init_weights(self):
        for w in [self.w1, self.w2]:
            nn.init.xavier_uniform_(w, gain=0.02)
        nn.init.xavier_uniform_(self.w3, gain=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        # Routing: [B, T, n_experts]
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)

        # Top-k selection
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Dispatch to experts
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            # Find tokens routed to this expert
            e_mask = (topk_idx == e)  # [B, T, top_k]
            mask = e_mask.any(dim=-1)  # [B, T]
            if not mask.any():
                continue
            # Gather tokens [N_e, D]
            tokens = x[mask]
            expert_out = (torch.nn.functional.silu(tokens @ self.w1[e]) *
                          (tokens @ self.w2[e])) @ self.w3[e]
            # Gather weights for this expert, shape [N_e]
            e_weight = topk_weights.masked_select(e_mask).view(-1, 1)
            out[mask] += expert_out * e_weight

        # Load balancing aux loss
        if self.training:
            # Fraction of tokens routed to each expert
            fraction = probs.mean(dim=(0, 1))  # [n_experts]
            target = torch.ones_like(fraction) / self.n_experts
            aux_loss = (fraction * target.log()).sum() * self.n_experts
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return out, aux_loss

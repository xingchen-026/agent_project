#!/usr/bin/env python3
"""
Tiny Unified Multimodal Transformer — v2
========================================
Architecture improvements over MVP:
- Deeper (6 layers) for better capacity
- SwiGLU MLP (stronger per-param)
- Pre-normalized attention (QKNorm for RoPE stability)
- Better weight init
- Configurable via dataclass
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 50257
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    head_dim: int = 64  # dim // n_heads
    max_seq_len: int = 1024
    num_image_tokens: int = 49  # 7x7 grid

    # Image processing
    image_size: int = 224
    patch_size: int = 32  # 224/32 = 7 → 49 patches
    use_type_embed: bool = True

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # MLP
    mlp_multiplier: int = 4
    mlp_type: str = "swiglu"  # swiglu or relu

    # RoPE
    rope_theta: float = 10000.0

    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        patches_per_side = self.image_size // self.patch_size
        self.num_image_tokens = patches_per_side * patches_per_side


# ── Components ─────────────────────────────────────────────────────────

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
    cos, sin = cos[:T].reshape(1, 1, T, half), sin[:T].reshape(1, 1, T, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.attention_dropout)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # RoPE with QK normalization (improves stability)
        q, k = self.q_norm(q).transpose(1, 2), self.k_norm(k).transpose(1, 2)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v.transpose(1, 2),
                                              attn_mask=mask, is_causal=(mask is None))
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = dim * hidden_mult
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_multiplier)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ── Unified Model ──────────────────────────────────────────────────────

class TinyMultimodal(nn.Module):
    """
    Tiny unified multimodal transformer.
    Processes [image_patches, text_tokens] in one autoregressive sequence.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Text embeddings
        self.text_embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Image patch projection
        self.img_proj = nn.Linear(3 * cfg.patch_size * cfg.patch_size, cfg.dim)
        self.img_norm = RMSNorm(cfg.dim)

        # Type embedding
        if cfg.use_type_embed:
            self.type_embed = nn.Embedding(2, cfg.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim)

        # LM head (tied)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.text_embed.weight = self.lm_head.weight

        # RoPE
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, 0.02)

    def get_num_image_tokens(self):
        ps = self.cfg.patch_size
        isz = self.cfg.image_size
        return (isz // ps) ** 2

    def _image_to_patches(self, images):
        B, C, H, W = images.shape
        p = self.cfg.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        return patches

    def forward(self, text_ids, images, return_all=False):
        B = text_ids.shape[0]
        device = text_ids.device
        n_img_tokens = self.get_num_image_tokens()

        # ── Image tokens ──
        patches = self._image_to_patches(images)
        img_tokens = self.img_norm(self.img_proj(patches))
        if hasattr(self, 'type_embed'):
            img_tokens = img_tokens + self.type_embed(torch.full((B, n_img_tokens), 1, device=device))

        # ── Text tokens ──
        text_tokens = self.text_embed(text_ids)
        if hasattr(self, 'type_embed'):
            text_type = torch.zeros(1, text_ids.shape[1], device=device, dtype=torch.long)
            text_tokens = text_tokens + self.type_embed(text_type)

        # ── Concatenate ──
        x = torch.cat([img_tokens, text_tokens], dim=1)
        total_len = x.shape[1]

        # ── RoPE ──
        cos, sin = self.rope(total_len, device)

        # ── Attention mask: image tokens attend to all images, text is causal ──
        mask = None if not return_all else torch.triu(
            torch.full((total_len, total_len), float('-inf'), device=device), diagonal=1)
        if n_img_tokens > 0:
            mask = torch.full((total_len, total_len), float('-inf'), device=device)
            # Image region: all-to-all
            mask[:n_img_tokens, :n_img_tokens] = 0
            # Text region: causal within text, all-to-all to images
            for i in range(n_img_tokens, total_len):
                mask[i, :i] = 0  # causal

        # ── Transformer ──
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)

        # ── Output ──
        text_logits = self.lm_head(x[:, n_img_tokens:])
        return text_logits

    @torch.no_grad()
    def generate(self, image, tokenizer, max_len=50, temperature=0.8, top_k=50):
        self.eval()
        device = next(self.parameters()).device
        if image.dim() == 3:
            image = image.unsqueeze(0)

        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        text_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

        generated = []
        for _ in range(max_len):
            logits = self(text_ids, image)
            next_logits = logits[0, -1] / temperature

            if top_k > 0:
                vals, _ = next_logits.topk(min(top_k, len(next_logits)))
                next_logits[next_logits < vals[-1]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
            text_ids = torch.cat([text_ids, torch.tensor([[next_id]], device=device)], dim=1)

            if text_ids.shape[1] > 100:
                break

        return tokenizer.decode(generated, skip_special_tokens=True)

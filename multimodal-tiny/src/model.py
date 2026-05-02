#!/usr/bin/env python3
"""
Tiny Unified Multimodal Transformer — v2.1 (Phase 2: Image Generation)
======================================================================
Phase 2 adds an image generation head for multimodal output capability.

Phase 1 base:
- 6-layer SwiGLU transformer with RoPE
- QK-normalized causal self-attention
- Image patch + text token fusion

Phase 2 additions:
- Image decoder head (patch-level pixel reconstruction)
- Combined text CE + image MSE loss training
- Image placeholder tokens for text-to-image inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 10000
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    head_dim: int = 64  # dim // n_heads
    max_seq_len: int = 1024
    num_image_tokens: int = 49  # 7x7 grid

    # Image processing
    image_size: int = 224
    patch_size: int = 32  # 224/32 = 7 -> 49 patches
    use_type_embed: bool = True

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # MLP
    mlp_multiplier: int = 4
    mlp_type: str = "swiglu"  # swiglu or relu

    # RoPE
    rope_theta: float = 10000.0

    # --- Phase 2: Image Generation ---
    img_generation: bool = True        # Enable image generation head
    img_decoder_hidden: int = 512      # Hidden dim of image decoder MLP

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


# ── Image Decoder Head (Phase 2) ─────────────────────────────────────

class ImageDecoderHead(nn.Module):
    """
    Reconstructs image patches from transformer hidden states.
    Takes [B, n_patches, dim] and outputs [B, n_patches, C*P*P].
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        patch_pixels = 3 * cfg.patch_size * cfg.patch_size  # C*H*W per patch
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.img_decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.img_decoder_hidden, patch_pixels),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


def patches_to_image(patches, patch_size, image_size):
    """
    Rearrange patches [B, N, C*P*P] back to image tensor [B, C, H, W].
    Used for visualization and saving generated images.
    """
    B, N, _ = patches.shape
    C = 3
    p = patch_size
    h = w = image_size // p  # patches per side
    img = patches.reshape(B, h, w, C, p, p)
    img = img.permute(0, 3, 1, 4, 2, 5).reshape(B, C, h * p, w * p)
    return img


# ── Unified Model (v2.1) ──────────────────────────────────────────────

class TinyMultimodal(nn.Module):
    """
    Tiny unified multimodal transformer with bidirectional generation.
    Processes [image_patches, text_tokens] in one autoregressive sequence.
    Phase 2: can also reconstruct images from hidden states.

    Inference modes:
      - img_to_text: standard, predicts text from image+text input
      - text_to_img: generates image from text-only input (uses [IMG] placeholder)
      - joint: predicts both text and reconstructs image from input
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Text embeddings
        self.text_embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Image patch projection
        self.img_proj = nn.Linear(3 * cfg.patch_size * cfg.patch_size, cfg.dim)
        self.img_norm = RMSNorm(cfg.dim)

        # Type embedding (0=text, 1=image)
        if cfg.use_type_embed:
            self.type_embed = nn.Embedding(2, cfg.dim)
        # Image placeholder token for text-to-image generation
        if cfg.img_generation:
            self.img_placeholder = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim)

        # LM head (tied)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.text_embed.weight = self.lm_head.weight

        # RoPE
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        # Phase 2: Image generation head
        if cfg.img_generation:
            self.img_decoder = ImageDecoderHead(cfg)

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
        """Convert [B, C, H, W] images to [B, N_patches, C*P*P] patches."""
        B, C, H, W = images.shape
        p = self.cfg.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        return patches

    def _make_attention_mask(self, n_img_tokens, total_len, bidirectional_img=True, device=None):
        """
        Build attention mask.
        - Image region: all-to-all (bidirectional within images)
        - Text region: causal (can attend to all images + previous text)
        """
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        # Image region: full bidirectional
        mask[:n_img_tokens, :n_img_tokens] = 0
        # Text -> all images + causal in text
        for i in range(n_img_tokens, total_len):
            mask[i, :i + 1] = 0  # causal + attend to images (images are at indices < n_img_tokens)
        return mask

    def forward(self, text_ids, images=None, return_img=False, img_gen_mode=False):
        """
        Args:
            text_ids: [B, T] token IDs
            images: [B, C, H, W] or None for text-to-image mode
            return_img: if True, also return image reconstruction
            img_gen_mode: if True, use image placeholder tokens
        Returns:
            text_logits: [B, T, vocab_size]
            If return_img:
                (text_logits, img_recon, target_patches)
        """
        B = text_ids.shape[0]
        device = text_ids.device
        n_img_tokens = self.get_num_image_tokens()

        # ── Image tokens ──
        if img_gen_mode or images is None:
            # Text-to-image generation: use learned [IMG] placeholder
            img_tokens = self.img_placeholder.expand(B, n_img_tokens, -1)
            img_type = torch.full((B, n_img_tokens), 1, device=device)  # type=1: image (placeholder)
            target_patches = None
        else:
            # Normal mode: project real image patches
            patches = self._image_to_patches(images)
            target_patches = patches.detach()  # ground truth for reconstruction loss
            img_tokens = self.img_norm(self.img_proj(patches))
            img_type = torch.full((B, n_img_tokens), 1, device=device)  # type=1: image

        if self.cfg.use_type_embed:
            img_tokens = img_tokens + self.type_embed(img_type)

        # ── Text tokens ──
        text_tokens = self.text_embed(text_ids)
        if self.cfg.use_type_embed:
            text_type = torch.zeros(B, text_ids.shape[1], device=device, dtype=torch.long)
            text_tokens = text_tokens + self.type_embed(text_type)

        # ── Concatenate ──
        x = torch.cat([img_tokens, text_tokens], dim=1)
        total_len = x.shape[1]

        # ── RoPE ──
        cos, sin = self.rope(total_len, device)

        # ── Attention mask ──
        mask = self._make_attention_mask(n_img_tokens, total_len, device=device)

        # ── Transformer ──
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)

        # ── Text output ──
        text_logits = self.lm_head(x[:, n_img_tokens:])

        if return_img and self.cfg.img_generation:
            # Image reconstruction from image-token hidden states
            img_hidden = x[:, :n_img_tokens]  # [B, N_patches, dim]
            img_recon = self.img_decoder(img_hidden)  # [B, N_patches, C*P*P]
            return text_logits, img_recon, target_patches

        return text_logits

    # ── Inference ──

    @torch.no_grad()
    def generate_text(self, image, tokenizer, max_len=50, temperature=0.8, top_k=50):
        """Generate text description from an image (img→text)."""
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

        return tokenizer.decode(generated)

    @torch.no_grad()
    def generate_image(self, text_ids, tokenizer, num_steps=1, temperature=1.0):
        """
        Generate an image from text using iterative refinement.
        (Experimental: applies decoder to [IMG] placeholder hidden states)

        Args:
            text_ids: [B, T] token IDs
            tokenizer: tokenizer for decoding
            num_steps: number of refinement steps (1 = single pass)
        Returns:
            image: [B, C, H, W] tensor in [-1, 1] range
        """
        self.eval()
        device = next(self.parameters()).device

        # Forward with placeholder tokens
        _, img_recon, _ = self(text_ids, images=None, return_img=True, img_gen_mode=True)

        if num_steps > 1:
            # Iterative refinement (placeholder for future work)
            pass

        # Convert patches to image
        p = self.cfg.patch_size
        img_size = self.cfg.image_size
        img = patches_to_image(img_recon, p, img_size)
        return torch.clamp(img, -1.0, 1.0)

    @torch.no_grad()
    def reconstruct_image(self, images):
        """
        Reconstruct input images through the model (autoencoding).
        Useful for checking how well the model preserves visual info.
        """
        self.eval()
        device = next(self.parameters()).device
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(device)

        # Create dummy text (just BOS token)
        bos_id = 0  # pad token used as BOS
        text_ids = torch.full((images.shape[0], 1), bos_id, dtype=torch.long, device=device)

        _, img_recon, _ = self(text_ids, images, return_img=True)

        p = self.cfg.patch_size
        img_size = self.cfg.image_size
        img = patches_to_image(img_recon, p, img_size)
        return torch.clamp(img, -1.0, 1.0)

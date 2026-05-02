#!/usr/bin/env python3
"""
Tiny Unified Multimodal Transformer — v3.0 (Phase 3: Audio Modality)
=====================================================================
Phase 3 adds audio spectrogram patch processing alongside image+text.

Architecture:
  [image_patches | audio_patches | text_tokens] → Transformer → outputs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 10000
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 1024

    # Image processing
    image_size: int = 224
    patch_size: int = 32
    use_type_embed: bool = True

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # MLP
    mlp_multiplier: int = 4

    # RoPE
    rope_theta: float = 10000.0

    # Phase 2: Image Generation
    img_generation: bool = True
    img_decoder_hidden: int = 512

    # --- Phase 3: Audio Modality ---
    use_audio: bool = True
    n_mels: int = 128
    audio_n_fft: int = 512
    audio_hop_length: int = 128
    audio_time_frames: int = 128
    audio_patch_freq: int = 16
    audio_patch_time: int = 16

    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        patches_per_side = self.image_size // self.patch_size
        self.num_image_tokens = patches_per_side * patches_per_side
        self.num_audio_tokens = (self.n_mels // self.audio_patch_freq) * \
                                (self.audio_time_frames // self.audio_patch_time)


# ── Common Components ────────────────────────────────────────────────

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


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
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
        self.attn = SelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_multiplier)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ── Decoder Heads ────────────────────────────────────────────────────

class ImageDecoderHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        patch_pixels = 3 * cfg.patch_size * cfg.patch_size
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.img_decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.img_decoder_hidden, patch_pixels),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)


class AudioDecoderHead(nn.Module):
    """Reconstruct mel spectrogram patches from hidden states."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        patch_pixels = cfg.audio_patch_freq * cfg.audio_patch_time  # 1-channel mel
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 2),
            nn.GELU(),
            nn.Linear(cfg.dim // 2, patch_pixels),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)


# ── Utility ──────────────────────────────────────────────────────────

def patches_to_image(patches, patch_size, image_size):
    B, N, _ = patches.shape
    C = 3
    p = patch_size
    h = w = image_size // p
    try:
        img = patches.reshape(B, h, w, C, p, p)
        img = img.permute(0, 3, 1, 4, 2, 5).reshape(B, C, h * p, w * p)
        return img
    except RuntimeError:
        return patches  # return raw if reshape fails


def mel_patches_to_spectrogram(patches, cfg):
    """Rebuild mel spectrogram from patches: [B, N, F*T] → [B, 1, n_mels, time_frames]."""
    n_freq_patches = cfg.n_mels // cfg.audio_patch_freq
    n_time_patches = cfg.audio_time_frames // cfg.audio_patch_time
    pf, pt = cfg.audio_patch_freq, cfg.audio_patch_time
    spec = patches.reshape(-1, n_freq_patches, n_time_patches, pf, pt)
    spec = spec.permute(0, 1, 3, 2, 4).reshape(-1, 1, cfg.n_mels, cfg.audio_time_frames)
    return spec


# ── Unified Model v3.0 ──────────────────────────────────────────────

class TinyMultimodal(nn.Module):
    """
    Tiny unified multimodal transformer — supports image + audio + text.
    Token order: [image_patches | audio_patches | text_tokens]
    Attention: image↔audio all-to-all, text causal (attends to all modalities)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Text
        self.text_embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Image projection
        self.img_proj = nn.Linear(3 * cfg.patch_size * cfg.patch_size, cfg.dim)
        self.img_norm = RMSNorm(cfg.dim)

        # Audio projection: 1-channel mel patches
        if cfg.use_audio:
            self.audio_proj = nn.Linear(
                cfg.audio_patch_freq * cfg.audio_patch_time, cfg.dim)
            self.audio_norm = RMSNorm(cfg.dim)
            self.audio_decoder = AudioDecoderHead(cfg)

        # Placeholder tokens for text-to-modality generation
        if cfg.img_generation:
            self.img_placeholder = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        if cfg.use_audio:
            self.audio_placeholder = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)

        # Type embedding: 0=text, 1=image, 2=audio, 3=img_placeholder, 4=audio_placeholder
        if cfg.use_type_embed:
            n_types = 5 if cfg.use_audio else 3
            self.type_embed = nn.Embedding(n_types, cfg.dim)

        # Transformer
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim)

        # LM head
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.text_embed.weight = self.lm_head.weight

        # RoPE
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        # Image decoder
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

    def get_num_audio_tokens(self, mels=None):
        if mels is not None:
            # Actual number from mel spectrogram dimensions
            H, W = mels.shape[-2], mels.shape[-1]
            pf, pt = self.cfg.audio_patch_freq, self.cfg.audio_patch_time
            return (H // pf) * (W // pt)
        return (self.cfg.n_mels // self.cfg.audio_patch_freq) * \
               (self.cfg.audio_time_frames // self.cfg.audio_patch_time)

    def _image_to_patches(self, images):
        B, C, H, W = images.shape
        p = self.cfg.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        return patches

    def _spectrogram_to_patches(self, mels):
        """
        Convert [B, 1, n_mels, T] mel spectrogram to [B, N, F*T] patches.
        """
        B, C, H, W = mels.shape
        pf, pt = self.cfg.audio_patch_freq, self.cfg.audio_patch_time
        patches = mels.unfold(2, pf, pf).unfold(3, pt, pt)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, pf * pt)
        return patches

    def _make_attention_mask(self, n_img, n_audio, total_len, device=None):
        """Image↔Audio all-to-all, Text causal."""
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        end_sensory = n_img + n_audio
        # Sensory region (image + audio): all-to-all
        mask[:end_sensory, :end_sensory] = 0
        # Text: causal + attend to all sensory
        for i in range(end_sensory, total_len):
            mask[i, :i + 1] = 0
        return mask

    def forward(self, text_ids, images=None, audios=None,
                return_img=False, return_audio=False,
                img_gen_mode=False, audio_gen_mode=False):
        """
        Args:
            text_ids: [B, T]
            images: [B, 3, H, W] or None
            audios: [B, 1, n_mels, T] or None
        Returns:
            text_logits or (text_logits, recon_dict)
        """
        B = text_ids.shape[0]
        device = text_ids.device
        n_img = self.get_num_image_tokens()
        target_patches = {}
        tokens_list = []
        type_ids_list = []

        # ── Image tokens ──
        if img_gen_mode or images is None:
            img_tokens = self.img_placeholder.expand(B, n_img, -1)
            ttype = torch.full((B, n_img), 3, device=device, dtype=torch.long)
            target_patches['image'] = None
        else:
            patches = self._image_to_patches(images)
            target_patches['image'] = patches.detach()
            img_tokens = self.img_norm(self.img_proj(patches))
            ttype = torch.full((B, n_img), 1, device=device, dtype=torch.long)
        tokens_list.append(img_tokens)
        type_ids_list.append(ttype)

        # ── Audio tokens ──
        n_aud = 0
        if self.cfg.use_audio:
            if audio_gen_mode or audios is None:
                n_aud = self.get_num_audio_tokens()  # from config
                aud_tokens = self.audio_placeholder.expand(B, n_aud, -1)
                atype = torch.full((B, n_aud), 4, device=device, dtype=torch.long)
                target_patches['audio'] = None
            else:
                n_aud = self.get_num_audio_tokens(audios)
                aud_patches = self._spectrogram_to_patches(audios)
                target_patches['audio'] = aud_patches.detach()
                aud_tokens = self.audio_norm(self.audio_proj(aud_patches))
                atype = torch.full((B, n_aud), 2, device=device, dtype=torch.long)
            tokens_list.append(aud_tokens)
            type_ids_list.append(atype)

        # ── Text tokens ──
        text_tokens = self.text_embed(text_ids)
        ttype = torch.zeros(B, text_ids.shape[1], device=device, dtype=torch.long)
        tokens_list.append(text_tokens)
        type_ids_list.append(ttype)

        # ── Combine ──
        x = torch.cat(tokens_list, dim=1)
        if self.cfg.use_type_embed:
            type_ids = torch.cat(type_ids_list, dim=1)
            x = x + self.type_embed(type_ids)

        total_len = x.shape[1]
        cos, sin = self.rope(total_len, device)
        mask = self._make_attention_mask(n_img, n_aud, total_len, device)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)

        # ── Text output ──
        n_sensory = n_img + n_aud
        text_logits = self.lm_head(x[:, n_sensory:])

        # ── Reconstruction outputs ──
        results = {'text_logits': text_logits}

        if return_img and self.cfg.img_generation:
            img_hidden = x[:, :n_img]
            results['img_recon'] = self.img_decoder(img_hidden)
            results['target_img'] = target_patches.get('image')

        if return_audio and self.cfg.use_audio and n_aud > 0 and audios is not None:
            aud_hidden = x[:, n_img:n_img + n_aud]
            results['aud_recon'] = self.audio_decoder(aud_hidden)
            results['target_aud'] = target_patches.get('audio')

        if return_img or return_audio:
            return results
        return text_logits

    # ── Inference ──

    @torch.no_grad()
    def generate_text(self, image, tokenizer, audio=None, max_len=50,
                      temperature=0.8, top_k=50):
        """Generate text description from image (optionally + audio)."""
        self.eval()
        device = next(self.parameters()).device
        if image is not None and image.dim() == 3:
            image = image.unsqueeze(0)
        if audio is not None and audio.dim() == 3:
            audio = audio.unsqueeze(0)

        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        text_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        generated = []

        for _ in range(max_len):
            out = self(text_ids, images=image, audios=audio)
            logits = out if isinstance(out, torch.Tensor) else out['text_logits']
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
    def generate_image(self, text_ids, tokenizer):
        """Generate image from text."""
        self.eval()
        device = next(self.parameters()).device
        out = self(text_ids, images=None, return_img=True, img_gen_mode=True)
        img_recon = out['img_recon']
        p = self.cfg.patch_size
        img = patches_to_image(img_recon, p, self.cfg.image_size)
        return torch.clamp(img, -1.0, 1.0)

    @torch.no_grad()
    def reconstruct_image(self, images):
        """Reconstruct images."""
        self.eval()
        device = next(self.parameters()).device
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(device)
        bos_id = 0
        text_ids = torch.full((images.shape[0], 1), bos_id, dtype=torch.long, device=device)
        out = self(text_ids, images=images, return_img=True)
        img = patches_to_image(out['img_recon'], self.cfg.patch_size, self.cfg.image_size)
        return torch.clamp(img, -1.0, 1.0)

    @torch.no_grad()
    def reconstruct_audio(self, audios):
        """Reconstruct audio mel spectrograms."""
        self.eval()
        device = next(self.parameters()).device
        if audios.dim() == 3:
            audios = audios.unsqueeze(0)
        audios = audios.to(device)
        bos_id = 0
        text_ids = torch.full((audios.shape[0], 1), bos_id, dtype=torch.long, device=device)
        out = self(text_ids, audios=audios, return_audio=True)
        spec = mel_patches_to_spectrogram(out['aud_recon'], self.cfg)
        return torch.clamp(spec, -1.0, 1.0)

#!/usr/bin/env python3
"""
Tiny Unified Multimodal Transformer (v5.0)
Supports video + image + audio + text with shared transformer backbone,
KV cache, and Perceiver-style memory bank.

Architecture:
  [video | image | audio | text] → Transformer → outputs
  [memory(16) | text]            → Transformer → outputs  (with MemoryBank)

Sub-modules:
  _components.py  — RMSNorm, RotaryEmbedding, SwiGLU, apply_rotary
  _attention.py   — SelfAttention (KV cache), TransformerBlock
  _memory.py      — MemoryBank (cross-modal compression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from _components import RMSNorm, RotaryEmbedding, apply_rotary, SwiGLU
from _attention import SelfAttention, TransformerBlock
from _memory import MemoryBank

# Re-export for backward compat
__all__ = ['TinyMultimodal', 'ModelConfig', 'RMSNorm', 'RotaryEmbedding',
           'SwiGLU', 'SelfAttention', 'TransformerBlock', 'MemoryBank',
           'apply_rotary', 'patches_to_image', 'mel_patches_to_spectrogram',
           'video_patches_to_frames']


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


class VideoDecoderHead(nn.Module):
    """Reconstruct video spatiotemporal patches from hidden states."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Each patch: 3 RGB channels x patch_size x patch_size x patch_time
        patch_pixels = 3 * cfg.video_patch_size * cfg.video_patch_size * cfg.video_patch_time
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


def mel_patches_to_spectrogram(patches, cfg, n_time_patches=None, n_time_total=None):
    """
    Rebuild mel spectrogram from patches: [B, N, F*T] → [B, 1, n_mels, time_frames].
    
    If n_time_patches/n_time_total are provided, use actual audio dimensions
    instead of config defaults (handles variable-length audio).
    """
    n_freq_patches = cfg.n_mels // cfg.audio_patch_freq
    n_time = n_time_patches if n_time_patches is not None else (cfg.audio_time_frames // cfg.audio_patch_time)
    time_total = n_time_total if n_time_total is not None else cfg.audio_time_frames
    pf, pt = cfg.audio_patch_freq, cfg.audio_patch_time
    spec = patches.reshape(-1, n_freq_patches, n_time, pf, pt)
    spec = spec.permute(0, 1, 3, 2, 4).reshape(-1, 1, cfg.n_mels, time_total)
    return spec


def video_patches_to_frames(patches, cfg):
    """
    Rebuild video frames from spatiotemporal patches.
    patches: [B, N, C*ps*ps*pt] → [B, 3, T, H, W]
    """
    B = patches.shape[0]
    pt = cfg.video_patch_time
    ps = cfg.video_patch_size
    n_t = cfg.video_frames // pt
    n_h = n_w = cfg.video_resolution // ps
    C = 3
    # [B, n_t, n_h, n_w, C, pt, ps, ps]
    patches_3d = patches.reshape(B, n_t, n_h, n_w, C, pt, ps, ps)
    # [B, C, n_t, pt, n_h, ps, n_w, ps]
    patches_3d = patches_3d.permute(0, 4, 1, 5, 2, 6, 3, 7)
    # [B, C, n_t*pt, n_h*ps, n_w*ps] = [B, 3, T, H, W]
    frames = patches_3d.reshape(B, C, cfg.video_frames, cfg.video_resolution, cfg.video_resolution)
    return frames


# ── Unified Model v3.0 ──────────────────────────────────────────────

class TinyMultimodal(nn.Module):
    """
    Tiny unified multimodal transformer — supports video + image + audio + text.
    Token order: [video_tokens | image_tokens | audio_patches | text_tokens]
    Attention: video↔image↔audio all-to-all, text causal (attends to all modalities)
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

        # Video projection: spatiotemporal patches (C*ps*ps*pt)
        if cfg.use_video:
            vid_patch_pixels = 3 * cfg.video_patch_size * cfg.video_patch_size * cfg.video_patch_time
            self.video_proj = nn.Linear(vid_patch_pixels, cfg.dim)
            self.video_norm = RMSNorm(cfg.dim)
            self.video_decoder = VideoDecoderHead(cfg)

        # Placeholder tokens for text-to-modality generation
        if cfg.img_generation:
            self.img_placeholder = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        if cfg.use_audio:
            self.audio_placeholder = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        if cfg.use_video:
            self.video_placeholder = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)

        # Type embedding: 0=text, 1=image, 2=audio, 3=img_placeholder, 4=audio_placeholder, 5=video, 6=video_placeholder
        if cfg.use_type_embed:
            n_types = 7 if (cfg.use_audio and cfg.use_video) else (
                        6 if cfg.use_video else (
                        5 if cfg.use_audio else 3))
            self.type_embed = nn.Embedding(n_types, cfg.dim)

        # Memory Bank (compresses sensory patches → fixed-size memory tokens)
        self.use_memory_bank = getattr(cfg, 'use_memory_bank', False)
        if self.use_memory_bank:
            n_mem = getattr(cfg, 'n_mem_tokens', 16)
            self.memory_bank = MemoryBank(cfg.dim, n_mem=n_mem, n_heads=cfg.n_heads,
                                          mlp_mult=cfg.mlp_multiplier)

        # Transformer
        self.blocks = nn.ModuleList([TransformerBlock(cfg.dim, cfg.n_heads, cfg.head_dim,
                                                       cfg.mlp_multiplier) for _ in range(cfg.n_layers)])
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
            H, W = mels.shape[-2], mels.shape[-1]
            pf, pt = self.cfg.audio_patch_freq, self.cfg.audio_patch_time
            return (H // pf) * (W // pt)
        return (self.cfg.n_mels // self.cfg.audio_patch_freq) * \
               (self.cfg.audio_time_frames // self.cfg.audio_patch_time)

    def get_num_video_tokens(self, videos=None):
        if videos is not None:
            # [B, 3, T, H, W]
            C, T, H, W = videos.shape[1], videos.shape[2], videos.shape[3], videos.shape[4]
            nt = T // self.cfg.video_patch_time
            nh = H // self.cfg.video_patch_size
            nw = W // self.cfg.video_patch_size
            return nt * nh * nw
        return self.cfg.num_video_tokens

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

    def _video_to_patches(self, videos):
        """
        Convert [B, 3, T, H, W] video to [B, N, C*pt*ps*ps] spatiotemporal patches.
        """
        B, C, T, H, W = videos.shape
        pt = self.cfg.video_patch_time
        ps = self.cfg.video_patch_size
        # unfold temporal, then spatial
        patches = videos.unfold(2, pt, pt).unfold(3, ps, ps).unfold(4, ps, ps)
        # [B, n_t, n_h, n_w, C, pt, ps, ps] → [B, N, C*pt*ps*ps]
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7)
        patches = patches.reshape(B, -1, C * pt * ps * ps)
        return patches

    def _make_attention_mask(self, n_video, n_img, n_audio, total_len, device=None):
        """Video↔Image↔Audio all-to-all, Text causal."""
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        end_sensory = n_video + n_img + n_audio
        # Sensory region: all-to-all
        mask[:end_sensory, :end_sensory] = 0
        # Text: causal + attend to all sensory
        for i in range(end_sensory, total_len):
            mask[i, :i + 1] = 0
        return mask

    def forward(self, text_ids, images=None, audios=None, videos=None,
                return_img=False, return_audio=False, return_video=False,
                img_gen_mode=False, audio_gen_mode=False,
                past_key_values=None, use_cache=False):
        B = text_ids.shape[0]
        device = text_ids.device
        n_img = self.get_num_image_tokens()
        target_patches = {}
        tokens_list = []
        type_ids_list = []

        # ── Sensory encoding (skipped when using KV cache) ──
        if past_key_values is not None:
            # Incremental step: only embed new text token
            n_vid = 0 if not self.cfg.use_video else self.get_num_video_tokens()
            n_aud = 0 if not self.cfg.use_audio else self.get_num_audio_tokens()
            text_tokens = self.text_embed(text_ids)
            ttype = torch.zeros(B, text_ids.shape[1], device=device, dtype=torch.long)
            tokens_list.append(text_tokens)
            type_ids_list.append(ttype)
            x = torch.cat(tokens_list, dim=1)
            if self.cfg.use_type_embed:
                type_ids = torch.cat(type_ids_list, dim=1)
                x = x + self.type_embed(type_ids)
            total_len = x.shape[1]
            cache_len = past_key_values[0][0].shape[2] if past_key_values[0] else 0
            cos, sin = self.rope(cache_len + total_len, device)
            cos, sin = cos[cache_len:], sin[cache_len:]
            mask = None  # incremental: no mask needed (q_len=1, sees all cached)
        else:
            # ── Video tokens ──
            n_vid = 0
            if self.cfg.use_video:
                if videos is not None:
                    n_vid = self.get_num_video_tokens(videos)
                    vid_patches = self._video_to_patches(videos)
                    target_patches['video'] = vid_patches.detach()
                    vid_tokens = self.video_norm(self.video_proj(vid_patches))
                    vtype = torch.full((B, n_vid), 5, device=device, dtype=torch.long)
                else:
                    n_vid = self.get_num_video_tokens()
                    vid_tokens = self.video_placeholder.expand(B, n_vid, -1)
                    vtype = torch.full((B, n_vid), 6, device=device, dtype=torch.long)
                    target_patches['video'] = None
                tokens_list.append(vid_tokens)
                type_ids_list.append(vtype)

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
                    n_aud = self.get_num_audio_tokens()
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

            # ── Memory Bank: compress sensory → fixed-size memory tokens ──
            if self.use_memory_bank and tokens_list[:-1]:  # skip text
                # Concatenate all sensory patches
                sensory = torch.cat(tokens_list[:-1], dim=1)  # [video | image | audio]
                mem_tokens = self.memory_bank(sensory)  # [B, n_mem, dim]
                # Replace sensory tokens with compressed memory tokens
                # New sequence: [memory | text]
                x = torch.cat([mem_tokens, tokens_list[-1]], dim=1)
                n_sensory_for_mask = self.memory_bank.n_mem
                n_vid_mem, n_img_mem, n_aud_mem = n_sensory_for_mask, 0, 0
                # Type embed: mem=1 (image type), text=0
                if self.cfg.use_type_embed:
                    mem_type = torch.full((B, n_sensory_for_mask), 1, device=device, dtype=torch.long)
                    txt_type = torch.zeros(B, tokens_list[-1].shape[1], device=device, dtype=torch.long)
                    type_ids = torch.cat([mem_type, txt_type], dim=1)
                    x = x + self.type_embed(type_ids)
            else:
                x = torch.cat(tokens_list, dim=1)
                if self.cfg.use_type_embed:
                    type_ids = torch.cat(type_ids_list, dim=1)
                    x = x + self.type_embed(type_ids)
                n_vid_mem, n_img_mem, n_aud_mem = n_vid, n_img, n_aud
                n_sensory_for_mask = n_vid + n_img + n_aud

            total_len = x.shape[1]
            cos, sin = self.rope(total_len, device)
            mask = self._make_attention_mask(n_vid_mem, n_img_mem, n_aud_mem, total_len, device)

        # ── Transformer (with mask + KV cache) ──
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_key_values[i] if past_key_values else None
            out = block(x, cos, sin, mask=mask, past_kv=pkv, use_cache=use_cache)
            if use_cache:
                x, present_kv = out
                new_kvs.append(present_kv)
            else:
                x = out
        x = self.final_norm(x)

        # ── Text output ──
        if past_key_values is not None:
            text_logits = self.lm_head(x)  # only text tokens present
        elif self.use_memory_bank:
            n_sensory = self.memory_bank.n_mem
            text_logits = self.lm_head(x[:, n_sensory:])
        else:
            n_sensory = n_vid + n_img + n_aud
            text_logits = self.lm_head(x[:, n_sensory:])

        # ── Reconstruction outputs ──
        results = {'text_logits': text_logits}
        if use_cache:
            results['past_key_values'] = new_kvs

        if return_img and self.cfg.img_generation and past_key_values is None:
            img_hidden = x[:, n_vid:n_vid + n_img]
            results['img_recon'] = self.img_decoder(img_hidden)
            results['target_img'] = target_patches.get('image')

        if return_audio and self.cfg.use_audio and n_aud > 0 and audios is not None and past_key_values is None:
            aud_hidden = x[:, n_vid + n_img:n_vid + n_img + n_aud]
            results['aud_recon'] = self.audio_decoder(aud_hidden)
            results['target_aud'] = target_patches.get('audio')

        if return_video and self.cfg.use_video and n_vid > 0 and videos is not None and past_key_values is None:
            vid_hidden = x[:, :n_vid]
            results['vid_recon'] = self.video_decoder(vid_hidden)
            results['target_vid'] = target_patches.get('video')

        if return_img or return_audio or return_video or use_cache:
            return results
        return text_logits

    # ── Inference ──

    def start_conversation(self, image, audio=None, video=None):
        """Initialize multi-turn conversation with sensory input.
        Returns a 'context' dict with cached memory + KV cache for subsequent chat() calls.
        """
        device = next(self.parameters()).device
        if image is not None and image.dim() == 3:
            image = image.unsqueeze(0)
        if audio is not None and audio.dim() == 3:
            audio = audio.unsqueeze(0)
        if video is not None and video.dim() == 4:
            video = video.unsqueeze(0)

        bos_id = 2  # <bos>
        text_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        out = self(text_ids, images=image, audios=audio, videos=video, use_cache=True)
        return {'past_key_values': out['past_key_values'],
                'context_text': ''}

    @torch.no_grad()
    def chat(self, context, prompt_text, tokenizer, max_len=50, temperature=0.8, top_k=50):
        """Generate response in a multi-turn conversation.
        Uses cached memory + KV from start_conversation().
        Returns (response_text, updated_context).
        """
        self.eval()
        device = next(self.parameters()).device
        past_key_values = context['past_key_values']

        # Encode the prompt text as continuation
        full_text = context.get('context_text', '') + prompt_text
        prompt_ids = torch.tensor([[2] + tokenizer.encode(prompt_text)],
                                  dtype=torch.long, device=device)[:, :64]
        out = self(prompt_ids, use_cache=True, past_key_values=past_key_values)
        past_key_values = out['past_key_values']
        logits = out['text_logits']

        # Sample first response token
        next_logits = logits[0, -1] / temperature
        if top_k > 0:
            vals, _ = next_logits.topk(min(top_k, len(next_logits)))
            next_logits[next_logits < vals[-1]] = float('-inf')
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id == tokenizer.eos_token_id:
            return '', context
        generated = [next_id]

        # Decode
        for _ in range(max_len - 1):
            text_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = self(text_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = out['past_key_values']
            logits = out['text_logits']
            next_logits = logits[0, -1] / temperature
            if top_k > 0:
                vals, _ = next_logits.topk(min(top_k, len(next_logits)))
                next_logits[next_logits < vals[-1]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)

        response = tokenizer.decode(generated)
        new_context = {
            'past_key_values': past_key_values,
            'context_text': context.get('context_text', '') + prompt_text + response,
        }
        return response, new_context

    @torch.no_grad()
    def generate_text(self, image, tokenizer, audio=None, video=None, max_len=50,
                      temperature=0.8, top_k=50):
        """Generate text from image (+ optional audio/video) using KV cache."""
        self.eval()
        device = next(self.parameters()).device
        if image is not None and image.dim() == 3:
            image = image.unsqueeze(0)
        if audio is not None and audio.dim() == 3:
            audio = audio.unsqueeze(0)
        if video is not None and video.dim() == 4:
            video = video.unsqueeze(0)

        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id

        # Step 1: Prefill — encode sensory + BOS token, cache all K/V
        text_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        out = self(text_ids, images=image, audios=audio, videos=video, use_cache=True)
        past_key_values = out['past_key_values']
        logits = out['text_logits']

        # Sample first token
        next_logits = logits[0, -1] / temperature
        if top_k > 0:
            vals, _ = next_logits.topk(min(top_k, len(next_logits)))
            next_logits[next_logits < vals[-1]] = float('-inf')
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id == tokenizer.eos_token_id:
            return ""
        generated = [next_id]

        # Step 2: Incremental decoding with KV cache
        for _ in range(max_len - 1):
            text_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = self(text_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = out['past_key_values']
            logits = out['text_logits']

            next_logits = logits[0, -1] / temperature
            if top_k > 0:
                vals, _ = next_logits.topk(min(top_k, len(next_logits)))
                next_logits[next_logits < vals[-1]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)

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
        # Compute actual audio dimensions for flexible-length reconstruction
        H, W = audios.shape[-2], audios.shape[-1]
        pf, pt = self.cfg.audio_patch_freq, self.cfg.audio_patch_time
        n_fp = H // pf
        n_tp = W // pt
        n_time_total = n_tp * pt
        spec = mel_patches_to_spectrogram(out['aud_recon'], self.cfg,
                                          n_time_patches=n_tp, n_time_total=n_time_total)
        return torch.clamp(spec, -1.0, 1.0)

    @torch.no_grad()
    def reconstruct_video(self, videos):
        """Reconstruct video frames."""
        self.eval()
        device = next(self.parameters()).device
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)
        videos = videos.to(device)
        bos_id = 0
        text_ids = torch.full((videos.shape[0], 1), bos_id, dtype=torch.long, device=device)
        out = self(text_ids, videos=videos, return_video=True)
        frames = video_patches_to_frames(out['vid_recon'], self.cfg)
        return torch.clamp(frames, -1.0, 1.0)

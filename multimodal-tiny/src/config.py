#!/usr/bin/env python3
"""
Single source of truth for all model architecture configuration.
Every parameter affecting model construction lives here.
"""

import os
import json
from dataclasses import dataclass, field, asdict, fields
from typing import Optional


@dataclass
class ModelConfig:
    """All architecture parameters for TinyMultimodal.

    Serialize via to_dict()/save(), deserialize via from_dict()/from_json().
    Use resolve_config() to build from checkpoint + defaults.
    """
    # ── Core architecture ──
    vocab_size: int = 10000
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 1024

    # ── Image processing ──
    image_size: int = 224
    patch_size: int = 32
    use_type_embed: bool = True

    # ── Regularization ──
    dropout: float = 0.0

    # ── MLP ──
    mlp_multiplier: int = 4

    # ── RoPE ──
    rope_theta: float = 10000.0

    # ── Image generation ──
    img_generation: bool = True
    img_decoder_hidden: int = 512

    # ── Audio ──
    use_audio: bool = True
    n_mels: int = 128
    audio_n_fft: int = 512
    audio_hop_length: int = 128
    audio_time_frames: int = 128
    audio_patch_freq: int = 16
    audio_patch_time: int = 16

    # ── Video ──
    use_video: bool = True
    video_frames: int = 4
    video_resolution: int = 64
    video_patch_size: int = 16
    video_patch_time: int = 2

    # ── Version tracking ──
    arch_version: str = "4.1"

    # ── Computed fields (set in __post_init__, not serialized) ──
    head_dim: int = field(init=False, default=64)
    num_image_tokens: int = field(init=False, default=49)
    num_audio_tokens: int = field(init=False, default=64)
    num_video_tokens: int = field(init=False, default=32)

    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        patches_per_side = self.image_size // self.patch_size
        self.num_image_tokens = patches_per_side * patches_per_side
        if self.use_audio:
            self.num_audio_tokens = (self.n_mels // self.audio_patch_freq) * \
                                    (self.audio_time_frames // self.audio_patch_time)
        if self.use_video:
            vfr = self.video_frames // self.video_patch_time
            vsp = self.video_resolution // self.video_patch_size
            self.num_video_tokens = vfr * vsp * vsp

    # ── Serialization ──

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove computed fields — they're rebuilt in __post_init__
        for key in ['head_dim', 'num_image_tokens', 'num_audio_tokens', 'num_video_tokens']:
            d.pop(key, None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())

    def describe(self) -> str:
        """Human-readable: '448d-8L-7h'."""
        return f"{self.dim}d-{self.n_layers}L-{self.n_heads}h"


# ── Configuration helpers ──────────────────────────────────────────

def read_checkpoint_config(checkpoint_path: str) -> Optional[dict]:
    """Quick-read model_config from a checkpoint without loading weights."""
    import torch
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return ckpt.get('model_config', None)


def _infer_config_from_state_dict(state_dict: dict, tokenizer) -> Optional[dict]:
    """Infer architecture params from state_dict shapes (for legacy checkpoints)."""
    try:
        # text_embed: [vocab, dim] -> infer dim
        embed = state_dict.get('text_embed.weight')
        if embed is None:
            return None
        dim = embed.shape[1]

        # Count transformer blocks from keys like 'blocks.N.attn_norm.weight'
        n_layers = 0
        for key in state_dict:
            if key.startswith('blocks.') and '.attn_norm.weight' in key:
                n_layers = max(n_layers, int(key.split('.')[1]) + 1)

        # Infer n_heads from qkv weight: [3*dim, dim] -> dim // (dim/n_heads) = heads
        # Actually look at q_norm weight: [head_dim] = [dim // n_heads]
        n_heads = 6  # default
        q_norm_key = 'blocks.0.attn.q_norm.weight'
        if q_norm_key in state_dict:
            head_dim = state_dict[q_norm_key].shape[0]
            n_heads = dim // head_dim

        # Check modality presence
        use_audio = 'audio_proj.weight' in state_dict
        use_video = 'video_proj.weight' in state_dict
        img_gen = 'img_decoder.net.0.weight' in state_dict

        return {
            'dim': dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'vocab_size': tokenizer.vocab_size,
            'use_audio': use_audio,
            'use_video': use_video,
            'img_generation': img_gen,
        }
    except Exception:
        return None


def resolve_config(checkpoint_path: Optional[str], tokenizer,
                   defaults: Optional[dict] = None) -> ModelConfig:
    """Build ModelConfig by priority:
    1. Config embedded in checkpoint
    2. Inferred from state_dict shapes (legacy checkpoints)
    3. Explicit defaults dict
    4. ModelConfig() class defaults
    """
    override = dict(defaults) if defaults else {}

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Try embedded config first
        ckpt_config = read_checkpoint_config(checkpoint_path)
        if ckpt_config:
            merged = {**ckpt_config, **override}
            merged['vocab_size'] = tokenizer.vocab_size
            return ModelConfig.from_dict(merged)

        # Legacy checkpoint: infer from shapes
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        inferred = _infer_config_from_state_dict(state_dict, tokenizer)
        if inferred:
            merged = {**inferred, **override}
            return ModelConfig.from_dict(merged)

    cfg = ModelConfig(**override)
    cfg.vocab_size = tokenizer.vocab_size
    return cfg


def build_config_from_args(args, tokenizer) -> ModelConfig:
    """Build ModelConfig from argparse Namespace + tokenizer."""
    # Load from config file if specified
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        cfg = ModelConfig.from_json(args.config)
    else:
        cfg = ModelConfig()

    # CLI overrides
    str_keys = ['dim', 'n_layers', 'n_heads', 'image_size', 'patch_size',
                'vocab_size', 'img_decoder_hidden', 'max_seq_len',
                'dropout', 'mlp_multiplier', 'rope_theta']
    for key in str_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))

    bool_keys = ['use_audio', 'use_video', 'img_generation', 'use_type_embed']
    for key in bool_keys:
        if hasattr(args, key):
            setattr(cfg, key, getattr(args, key))

    if hasattr(args, 'img_gen'):
        cfg.img_generation = args.img_gen

    # Map 'layers' arg to 'n_layers' for backward compat
    if hasattr(args, 'layers') and args.layers is not None:
        cfg.n_layers = args.layers

    cfg.vocab_size = tokenizer.vocab_size
    cfg.__post_init__()
    return cfg

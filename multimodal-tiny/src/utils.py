#!/usr/bin/env python3
"""Shared utilities: loss functions, checkpoint loading, data collation, logging."""

import json
import logging
from dataclasses import dataclass
import torch
import torch.nn.functional as F


# ── Default config (deprecated shim, use config.ModelConfig directly) ──

@dataclass
class DefaultConfig:
    """Backward-compatible config proxy. Use ModelConfig from config.py directly."""
    dim: int = 384
    n_layers: int = 6
    image_size: int = 224
    patch_size: int = 32
    img_decoder_hidden: int = 512
    vocab_size: int = 10000
    max_text_len: int = 48

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('multimodal')


# ── Loss functions ───────────────────────────────────────────────────

def compute_text_loss(logits, text_ids, attn_mask):
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = text_ids[:, 1:].reshape(-1)
    shift_mask = attn_mask[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
    return (loss * shift_mask).sum() / shift_mask.sum()


def compute_mse_loss(pred, target):
    if pred is None or target is None:
        return torch.tensor(0.0, device=pred.device if pred is not None else 'cpu')
    return F.mse_loss(pred, target)


# ── Checkpoint ───────────────────────────────────────────────────────

def load_checkpoint_adaptive(model, checkpoint_path, device='cpu', verbose=True):
    """Load checkpoint, adapting to architecture changes (dim/layers/heads/vocab).

    - Same-shape weights: strict copy
    - Different-shape Linear weights: copy overlapping sub-matrix (warm start)
    - Embedding/lm_head: copy overlapping vocab rows, init new rows with mean
    - Falls back to shape-match mode if checkpoint has no embedded config.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    ckpt_config = ckpt.get('model_config', None)

    model_dict = model.state_dict()
    loaded = skipped = 0

    for key in state_dict:
        if key not in model_dict:
            skipped += 1
            continue

        ckpt_shape = state_dict[key].shape
        model_shape = model_dict[key].shape

        if ckpt_shape == model_shape:
            model_dict[key] = state_dict[key].to(device)
            loaded += 1
        elif 'embed' in key or 'lm_head' in key:
            # Vocab and/or dim changed — copy overlapping sub-matrix
            min_dim0 = min(ckpt_shape[0], model_shape[0])
            min_dim1 = min(ckpt_shape[1], model_shape[1]) if len(ckpt_shape) > 1 else 1
            if len(ckpt_shape) == 2:
                model_dict[key][:min_dim0, :min_dim1] = state_dict[key][:min_dim0, :min_dim1].to(device)
            else:
                model_dict[key][:min_dim0] = state_dict[key][:min_dim0].to(device)
            loaded += 1
            if verbose:
                logger.info(f"  Resized {key}: {list(ckpt_shape)} -> {list(model_shape)}")
        elif len(ckpt_shape) == 2 and len(model_shape) == 2:
            # Weight matrix — copy overlapping sub-matrix
            min_0 = min(ckpt_shape[0], model_shape[0])
            min_1 = min(ckpt_shape[1], model_shape[1])
            model_dict[key][:min_0, :min_1] = state_dict[key][:min_0, :min_1].to(device)
            loaded += 1
            if verbose and ckpt_shape != model_shape:
                logger.info(f"  Partial {key}: ({min_0},{min_1}) of {list(model_shape)}")
        elif len(ckpt_shape) == 1 and len(model_shape) == 1:
            # Bias or norm weight — copy overlapping
            min_len = min(ckpt_shape[0], model_shape[0])
            model_dict[key][:min_len] = state_dict[key][:min_len].to(device)
            loaded += 1
            if verbose and ckpt_shape != model_shape:
                logger.info(f"  Partial {key}: {min_len} of {model_shape[0]}")
        else:
            skipped += 1

    # Init new embedding rows/cols when vocab or dim expands
    old_embed = state_dict.get('text_embed.weight', None)
    if old_embed is not None and 'text_embed.weight' in model_dict:
        new_embed = model_dict['text_embed.weight']
        old_vocab, old_dim = old_embed.shape
        new_vocab, new_dim = new_embed.shape

        # Init new vocab rows
        if new_vocab > old_vocab:
            embed_mean = old_embed.mean(dim=0, keepdim=True).to(device)
            embed_std = old_embed.std().item()
            noise_std = embed_std * 0.1
            torch.nn.init.normal_(new_embed[old_vocab:, :old_dim], mean=0.0, std=noise_std)
            new_embed[old_vocab:, :old_dim] += embed_mean
            if verbose:
                logger.info(f"  Init {new_vocab - old_vocab} new embedding rows")

        # Init new dim columns (for architecture expansion)
        if new_dim > old_dim:
            # Scalar mean/std from old embedding for broadcasting
            scalar_mean = new_embed[:, :old_dim].mean().item()
            scalar_std = new_embed[:, :old_dim].std().item()
            torch.nn.init.normal_(new_embed[:, old_dim:], mean=scalar_mean,
                                  std=scalar_std * 0.1)
            if verbose:
                logger.info(f"  Init {new_dim - old_dim} new embedding dim columns")

        model_dict['text_embed.weight'] = new_embed

    # Backward compat: split old qkv weight into q_proj, k_proj, v_proj
    for key in list(state_dict.keys()):
        if '.attn.qkv.weight' in key:
            old_weight = state_dict.pop(key)
            d = old_weight.shape[0] // 3
            prefix = key.replace('.attn.qkv.weight', '.attn')
            state_dict[f'{prefix}.q_proj.weight'] = old_weight[:d]
            state_dict[f'{prefix}.k_proj.weight'] = old_weight[d:2*d]
            state_dict[f'{prefix}.v_proj.weight'] = old_weight[2*d:]
            if verbose:
                logger.info(f"  Split {key} -> q_proj, k_proj, v_proj")

    model.load_state_dict(model_dict, strict=False)

    if verbose:
        logger.info(f"Loaded {loaded}/{len(model_dict)} keys ({skipped} skipped)")

    info = {}
    for key in ['epoch', 'best_loss', 'model_config']:
        if key in ckpt:
            info[key] = ckpt[key]
    return info or {'best_loss': float('inf')}


# Backward-compat alias
def load_checkpoint_flexible(model, checkpoint_path, device='cpu'):
    """Deprecated — use load_checkpoint_adaptive."""
    return load_checkpoint_adaptive(model, checkpoint_path, device)


# ── Data loading helpers ─────────────────────────────────────────────

_KEY_MAP = {'image': 'images', 'audio': 'audios', 'video': 'videos'}


def make_collate(tokenizer, max_text_len, modality='image'):
    """Create a collate function for a given modality (image/audio/video)."""
    data_key = _KEY_MAP[modality]

    def collate(batch):
        data, captions = zip(*batch)
        data = torch.stack(data)
        enc = tokenizer(list(captions), padding='max_length', truncation=True,
                        max_length=max_text_len, return_tensors='pt')
        return {data_key: data, 'text_ids': enc['input_ids'],
                'attn_mask': enc['attention_mask']}

    return collate


def interleave_loaders(*loaders):
    """Round-robin interleave multiple DataLoaders into a flat list of batches."""
    its = [iter(l) for l in loaders]
    done = [False] * len(its)
    result = []
    while not all(done):
        for i, it in enumerate(its):
            if not done[i]:
                try:
                    result.append(next(it))
                except StopIteration:
                    done[i] = True
    return result


# ── Metrics logging ──────────────────────────────────────────────────

def save_metrics(log_dir, metrics_list):
    with open(log_dir / "metrics.json", 'w') as f:
        json.dump(metrics_list, f, indent=2)


def save_config(log_dir, config_dict):
    with open(log_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

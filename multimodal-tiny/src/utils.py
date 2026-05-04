#!/usr/bin/env python3
"""Shared utilities: loss functions, checkpoint loading, data collation, logging."""

import json
import logging
from dataclasses import dataclass
import torch
import torch.nn.functional as F


# ── Default config ───────────────────────────────────────────────────

@dataclass
class DefaultConfig:
    """Single source of truth for architecture defaults."""
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

def load_checkpoint_flexible(model, checkpoint_path, device='cpu'):
    """Load checkpoint matching keys by name+shape, silent on mismatches."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    model_dict = model.state_dict()
    loaded = skipped = 0
    for key in state_dict:
        if key in model_dict and state_dict[key].shape == model_dict[key].shape:
            model_dict[key] = state_dict[key]
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_dict, strict=False)
    logger.info(f"Loaded {loaded}/{len(model_dict)} keys ({skipped} skipped)")

    info = {}
    if 'epoch' in ckpt:
        info = {'epoch': ckpt['epoch'], 'best_loss': ckpt.get('best_loss', float('inf'))}
    return info or {'best_loss': float('inf')}


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

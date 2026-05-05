#!/usr/bin/env python3
"""Training infrastructure — optimizer/scheduler builders, checkpoint, utilities.

Replaces:
  train_utils.py (4 functions, 0 callers — entirely dead code)
  10 inline optimizer setups across training scripts
  10 inline checkpoint saves across training scripts
  6 inline scheduler setups across training scripts
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# ── Optimizer Builders ──────────────────────────────────────────────

def build_standard_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.98)):
    """Single-group AdamW for all parameters."""
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


def build_new_module_optimizer(model, lr, new_module_names, new_lr_mult=5.0,
                                weight_decay=0.1, betas=(0.9, 0.98)):
    """Two-group optimizer: new modules at higher LR, body at base LR.

    Used by: train_joint, train_joint_full, train_clip, train_distill.
    """
    new_params = [p for n, p in model.named_parameters()
                  if any(m in n for m in new_module_names)]
    other_params = [p for n, p in model.named_parameters()
                    if not any(m in n for m in new_module_names)]
    return AdamW([
        {'params': new_params, 'lr': lr * new_lr_mult},
        {'params': other_params, 'lr': lr},
    ], weight_decay=weight_decay, betas=betas)


def build_differential_optimizer(model, lr, embed_mult=3.0, decoder_mult=1.5,
                                  weight_decay=0.05, betas=(0.9, 0.95)):
    """Three-group optimizer for fine-tuning: embed (fast), decoder (moderate), body (slow).

    Partitions by parameter name:
      - embed: text_embed, lm_head (vocabulary-sensitive, fast adaptation)
      - decoder: img_decoder, audio_decoder, video_decoder, *_proj, *_norm
      - body: everything else

    Used by: finetune_cn, finetune_coco_cn, finetune_vqa.
    """
    embed_params, decoder_params, body_params = [], [], []

    for name, param in model.named_parameters():
        if 'text_embed' in name or 'lm_head' in name:
            embed_params.append(param)
        elif any(k in name for k in ['decoder', 'img_', 'audio_', 'video_', '_proj', '_norm']):
            decoder_params.append(param)
        else:
            body_params.append(param)

    return AdamW([
        {'params': embed_params, 'lr': lr * embed_mult},
        {'params': decoder_params, 'lr': lr * decoder_mult},
        {'params': body_params, 'lr': lr},
    ], weight_decay=weight_decay, betas=betas)


# ── Scheduler Builder ───────────────────────────────────────────────

def build_scheduler(optimizer, total_steps, warmup_ratio=0.1, min_warmup=200,
                    scheduler_type='cosine'):
    """Linear warmup + cosine decay (or linear decay).

    If warmup_ratio=0 or total_steps too small, returns cosine-only scheduler.
    """
    warmup_steps = min(max(min_warmup, int(total_steps * warmup_ratio)), total_steps // 2)

    if warmup_steps <= 1 or total_steps <= warmup_steps + 10:
        return CosineAnnealingLR(optimizer, T_max=total_steps)

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    if scheduler_type == 'cosine':
        main = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    else:
        main = LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                        total_iters=total_steps - warmup_steps)

    return SequentialLR(optimizer, [warmup, main], milestones=[warmup_steps])


# ── Checkpoint ──────────────────────────────────────────────────────

def save_checkpoint(output_dir, epoch, model, optimizer, best_loss,
                    model_config=None, extra_state=None, is_best=False):
    """Unified checkpoint saver. Handles best.pt copy automatically.

    Args:
        output_dir: Path or str
        epoch:      current epoch number
        model:      nn.Module
        optimizer:  AdamW or similar
        best_loss:  current best validation loss
        model_config: ModelConfig.to_dict() or dict
        extra_state: dict of extra fields (e.g. distill_head_state_dict)
        is_best:    if True, also copies to best.pt
    Returns:
        ckpt_path:  Path to saved checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    if model_config:
        ckpt_data['model_config'] = model_config
    if extra_state:
        ckpt_data.update(extra_state)

    ckpt_path = output_dir / f'epoch_{epoch}.pt'
    torch.save(ckpt_data, ckpt_path)

    if is_best:
        best_path = output_dir / 'best.pt'
        shutil.copy(ckpt_path, best_path)

    return ckpt_path


# ── Utilities ───────────────────────────────────────────────────────

def setup_output_dirs(output_dir, log_dir):
    """Create output and log directories."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = Path(log_dir)
    log.mkdir(parents=True, exist_ok=True)
    return out, log


def seed_everything(seed=42):
    """Set seed for random, numpy, torch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_metrics(log_dir, metrics):
    """Save metrics list as JSON."""
    with open(Path(log_dir) / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


def print_header(text, width=60):
    """Print a centered header banner."""
    print(f"\n{'=' * width}")
    print(text)
    print(f"{'=' * width}\n")


def count_params(model):
    """Return total parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6

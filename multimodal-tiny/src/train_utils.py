#!/usr/bin/env python3
"""Training infrastructure: checkpoint saving, optimizer/scheduler builders, metrics logging."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path


def build_optimizer_scheduler(model, lr, total_steps, weight_decay=0.1,
                               warmup_ratio=0.1, betas=(0.9, 0.95)):
    """Standard AdamW + linear warmup + cosine decay."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    warmup_steps = min(500, int(total_steps * warmup_ratio))
    warmup = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    return optimizer, scheduler


def build_differential_optimizer(model, lr, weight_decay=0.05, betas=(0.9, 0.95),
                                  embed_mult=5.0, decoder_mult=2.0):
    """Differential LR: embed/lm_head fast, decoder medium, body slow (for Phase 5)."""
    embed_params, body_params, decoder_params = [], [], []
    for name, param in model.named_parameters():
        if 'text_embed' in name or 'lm_head' in name:
            embed_params.append(param)
        elif 'decoder' in name or 'img_' in name or 'audio_' in name or 'video_' in name:
            decoder_params.append(param)
        else:
            body_params.append(param)

    param_groups = [
        {'params': embed_params, 'lr': lr * embed_mult},
        {'params': decoder_params, 'lr': lr * decoder_mult},
        {'params': body_params, 'lr': lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=weight_decay, betas=betas)

    embed_lr = lr * embed_mult
    decoder_lr = lr * decoder_mult
    print(f"  LR: embed={embed_lr:.1e}, decoder={decoder_lr:.1e}, body={lr:.1e}")
    return optimizer, param_groups


def save_checkpoint(output_dir, epoch, model, optimizer, best_loss,
                    val_losses, train_losses, phase=4, is_best=False):
    """Save training checkpoint and optionally update best.pt."""
    output_dir = Path(output_dir)
    ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'phase': phase,
        'model_config': model.cfg.to_dict(),
        'arch_version': model.cfg.arch_version,
    }
    for k, v in {**val_losses, **train_losses}.items():
        save_dict[k] = v
    torch.save(save_dict, ckpt_path)
    print(f"  Saved {ckpt_path}")

    if is_best:
        import shutil
        shutil.copy(ckpt_path, output_dir / "best.pt")
        print(f"  ★ New best")


def print_header(text, width=60):
    print(f"\n{'='*width}")
    print(f"{text}")
    print(f"{'='*width}")

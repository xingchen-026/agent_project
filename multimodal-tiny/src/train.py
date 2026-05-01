#!/usr/bin/env python3
"""
Training script for Tiny Multimodal Model.
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from tqdm import tqdm

from model import TinyMultimodal, ModelConfig
from data import build_loaders
from synthetic_data import SyntheticDataset

import warnings
warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser(description="Train Tiny Multimodal Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train_size", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--max_text_len", type=int, default=48)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--use_synthetic", action='store_true',
                        help='Use synthetic data instead of COCO')
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer (zero-download, fully local) ──
    from tokenizer import SimpleTokenizer
    print("Building local tokenizer...")
    tokenizer = SimpleTokenizer(max_vocab=10000)

    # ── Config ──
    cfg = ModelConfig(
        dim=args.dim,
        n_layers=args.layers,
        image_size=args.image_size,
        patch_size=args.patch_size,
        vocab_size=tokenizer.vocab_size,
    )

    # ── Model ──
    print(f"Building TinyMultimodal: {cfg.n_layers} layers, {cfg.dim} dim")
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")

    # ── Data ──
    print("Building data loaders...")
    if args.use_synthetic:
        print("  Using synthetic data (no download needed)")
        def collate(batch):
            images, captions = zip(*batch)
            images = torch.stack(images)
            enc = tokenizer(list(captions), padding='max_length', truncation=True,
                          max_length=args.max_text_len, return_tensors='pt')
            return images, enc['input_ids'], enc['attention_mask']
        train_ds = SyntheticDataset(num_samples=args.train_size, image_size=args.image_size)
        val_ds   = SyntheticDataset(num_samples=args.val_size, image_size=args.image_size, seed=99)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate, num_workers=0)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)
    else:
        data_config = {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "max_text_len": args.max_text_len,
        }
        train_loader, val_loader = build_loaders(tokenizer, data_config)

    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(500, total_steps // 10)
    warmup_sch = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup_steps])

    # ── Resume ──
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch}")

    # ── Prepare output dirs ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config ──
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config_dict['model_params_m'] = total / 1e6
    with open(log_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # ── Training ──
    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs, {args.train_size} samples, batch={args.batch_size}")
    print(f"  Steps/epoch: {len(train_loader)}")
    print(f"  Warmup: {warmup_steps} steps, Total: {total_steps} steps")
    print(f"{'='*60}\n")

    global_step = 0
    metrics = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, text_ids, attn_mask in pbar:
            images, text_ids, attn_mask = images.to(device), text_ids.to(device), attn_mask.to(device)

            logits = model(text_ids, images)
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = text_ids[:, 1:].reshape(-1)
            shift_mask = attn_mask[:, 1:].reshape(-1)

            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = (loss * shift_mask).sum() / shift_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n  Epoch {epoch+1} avg train loss: {avg_loss:.4f}")

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, text_ids, attn_mask in val_loader:
                images, text_ids, attn_mask = images.to(device), text_ids.to(device), attn_mask.to(device)
                logits = model(text_ids, images)
                shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
                shift_labels = text_ids[:, 1:].reshape(-1)
                shift_mask = attn_mask[:, 1:].reshape(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                loss = (loss * shift_mask).sum() / shift_mask.sum()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"  Epoch {epoch+1} val loss:     {val_loss:.4f}")

        # ── Sample generation ──
        if (epoch + 1) % max(1, args.save_every) == 0 and len(val_loader.dataset) > 0:
            sample_img, _ = val_loader.dataset[0]
            caption = model.generate(sample_img.to(device), tokenizer, max_len=30, temperature=0.8)
            print(f"  Sample caption: {caption[:100]}")

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_every == 0 or val_loss < best_loss:
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss

            ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
            }, ckpt_path)

            if is_best:
                best_path = output_dir / "best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ New best (val: {val_loss:.4f})")

        # ── Log ──
        metrics.append({'epoch': epoch+1, 'train_loss': avg_loss, 'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0]})
        with open(log_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_loss:.4f}")
    print(f"Checkpoints: {output_dir.resolve()}")
    print(f"Logs: {log_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()

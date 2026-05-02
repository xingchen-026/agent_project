#!/usr/bin/env python3
"""
Training script for Tiny Multimodal Model — v2.1 (Phase 2)
==========================================================
Phase 2 adds image generation head with combined loss training.
Supports:
  - Phase 1 style: text-only CE loss (backward compatible)
  - Phase 2: combined text CE + image MSE reconstruction loss
  - Resume from Phase 1 checkpoints (auto-detects missing keys)
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

from model import TinyMultimodal, ModelConfig, patches_to_image
from data import build_loaders
from synthetic_data import SyntheticDataset

import warnings
warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser(description="Train Tiny Multimodal Model")
    # Data
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--max_text_len", type=int, default=48)
    parser.add_argument("--use_synthetic", action='store_true')
    # Model
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=32)
    # Phase 2: Generation
    parser.add_argument("--img_gen", action='store_true', default=True,
                        help="Enable image generation head + combined loss (Phase 2)")
    parser.add_argument("--img_loss_weight", type=float, default=1.0,
                        help="Weight of image MSE loss relative to text CE loss")
    parser.add_argument("--img_decoder_hidden", type=int, default=512,
                        help="Hidden dim of image decoder MLP")
    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Phase 1 checkpoint path for transfer learning")
    return parser.parse_args()


def compute_text_loss(logits, text_ids, attn_mask):
    """Cross-entropy loss for text prediction (shifted)."""
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = text_ids[:, 1:].reshape(-1)
    shift_mask = attn_mask[:, 1:].reshape(-1)

    loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
    return (loss * shift_mask).sum() / shift_mask.sum()


def compute_image_loss(img_recon, target_patches):
    """MSE loss for image patch reconstruction."""
    return F.mse_loss(img_recon, target_patches)


def load_checkpoint_with_flexibility(model, checkpoint_path, device='cpu'):
    """
    Load checkpoint with flexibility for Phase 1→Phase 2 migration.
    Handles missing keys (new Phase 2 weights) and mismatched sizes.
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt  # bare state_dict

    # Try strict first, then flexible
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✓ Loaded strictly (full match)")
        return ckpt
    except Exception as e:
        missing = set()
        unexpected = set()
        for key in model.state_dict():
            if key not in state_dict:
                missing.add(key)
        for key in state_dict:
            if key not in model.state_dict():
                unexpected.add(key)
            elif state_dict[key].shape != model.state_dict()[key].shape:
                print(f"  Shape mismatch: {key} "
                      f"ckpt={list(state_dict[key].shape)} "
                      f"model={list(model.state_dict()[key].shape)}")

        if missing:
            print(f"  Missing keys in checkpoint (will init fresh): {sorted(missing)}")
        if unexpected:
            print(f"  Unexpected keys in checkpoint (skipping): {sorted(unexpected)}")

        # Load matching keys
        model_dict = model.state_dict()
        for key in state_dict:
            if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                model_dict[key] = state_dict[key]
        model.load_state_dict(model_dict, strict=True)
        print(f"  ✓ Loaded {len(state_dict) - len(unexpected)}/{len(model_dict)} keys")

        # Keep only what we need from ckpt meta
        if 'optimizer_state_dict' in ckpt and 'epoch' in ckpt:
            return {
                'epoch': ckpt.get('epoch', -1),
                'optimizer_state_dict': ckpt.get('optimizer_state_dict'),
                'train_loss': ckpt.get('train_loss'),
                'val_loss': ckpt.get('val_loss'),
                'best_loss': ckpt.get('best_loss', float('inf')),
            }
        return {'epoch': -1, 'best_loss': float('inf')}


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase 2 Image Generation: {'ENABLED' if args.img_gen else 'DISABLED'}")

    # ── Tokenizer ──
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
        img_generation=args.img_gen,
        img_decoder_hidden=args.img_decoder_hidden,
    )

    # ── Model ──
    print(f"Building TinyMultimodal v2.1: {cfg.n_layers} layers, {cfg.dim} dim")
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")

    # ── Phase 1 → Phase 2 checkpoint migration ──
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt_info = load_checkpoint_with_flexibility(model, args.resume, device)
        start_epoch = ckpt_info.get('epoch', -1) + 1
        best_loss = ckpt_info.get('best_loss', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")

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

    # ── Resume optimizer state ──
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("  Optimizer state restored")
            except Exception as e:
                print(f"  Optimizer state NOT restored ({e})")

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
    phase_tag = "Phase 2 (text+img gen)" if args.img_gen else "Phase 1 (text only)"
    print(f"\n{'='*60}")
    print(f"{phase_tag}: {args.epochs} epochs, {args.train_size} samples")
    print(f"  Steps/epoch: {len(train_loader)}, Batch: {args.batch_size}")
    if args.img_gen:
        print(f"  Image loss weight: {args.img_loss_weight}")
    print(f"{'='*60}\n")

    global_step = 0
    metrics = []

    use_img_gen = args.img_gen

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss_text = 0.0
        epoch_loss_img = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, text_ids, attn_mask in pbar:
            images, text_ids, attn_mask = images.to(device), text_ids.to(device), attn_mask.to(device)

            # Forward with optional image reconstruction
            if use_img_gen:
                logits, img_recon, target_patches = model(text_ids, images, return_img=True)
                loss_text = compute_text_loss(logits, text_ids, attn_mask)
                loss_img = compute_image_loss(img_recon, target_patches)
                loss = loss_text + args.img_loss_weight * loss_img
                epoch_loss_text += loss_text.item()
                epoch_loss_img += loss_img.item()
                loss_str = f"txt={loss_text.item():.4f} img={loss_img.item():.4f}"
            else:
                logits = model(text_ids, images, return_img=False)
                loss_text = compute_text_loss(logits, text_ids, attn_mask)
                loss = loss_text
                epoch_loss_text += loss_text.item()
                loss_str = f"loss={loss_text.item():.4f}"

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            lr_val = scheduler.get_last_lr()[0]
            pbar.set_postfix_str(f"{loss_str} lr={lr_val:.2e}")

        avg_text_loss = epoch_loss_text / len(train_loader)
        if use_img_gen:
            avg_img_loss = epoch_loss_img / len(train_loader)
            print(f"\n  Epoch {epoch+1}: text_loss={avg_text_loss:.4f}, img_loss={avg_img_loss:.4f}")
        else:
            print(f"\n  Epoch {epoch+1}: train_loss={avg_text_loss:.4f}")

        # ── Validation ──
        model.eval()
        val_text_loss = 0.0
        val_img_loss = 0.0
        with torch.no_grad():
            for images, text_ids, attn_mask in val_loader:
                images, text_ids, attn_mask = images.to(device), text_ids.to(device), attn_mask.to(device)
                if use_img_gen:
                    logits, img_recon, target_patches = model(text_ids, images, return_img=True)
                    val_text_loss += compute_text_loss(logits, text_ids, attn_mask).item()
                    val_img_loss += compute_image_loss(img_recon, target_patches).item()
                else:
                    logits = model(text_ids, images)
                    val_text_loss += compute_text_loss(logits, text_ids, attn_mask).item()

        val_text_loss /= len(val_loader)
        if use_img_gen:
            val_img_loss /= len(val_loader)
            print(f"  Val: text_loss={val_text_loss:.4f}, img_loss={val_img_loss:.4f}")
        else:
            print(f"  Val: loss={val_text_loss:.4f}")

        # Combined validation metric for best checkpoint tracking
        combined_val = val_text_loss + (args.img_loss_weight * val_img_loss if use_img_gen else 0)

        # ── Sample generation ──
        if (epoch + 1) % max(1, args.save_every) == 0 and len(val_loader.dataset) > 0:
            sample_img, sample_caption = val_loader.dataset[0]
            sample_img_t = sample_img.to(device)

            # Generate text
            gen_caption = model.generate_text(sample_img_t, tokenizer, max_len=30, temperature=0.8)
            print(f"  Ground truth: {sample_caption[:80]}")
            print(f"  Generated:    {gen_caption[:80]}")

            # Image reconstruction quality
            if use_img_gen:
                recon_img = model.reconstruct_image(sample_img_t)
                recon_mse = F.mse_loss(recon_img, sample_img_t).item()
                print(f"  Recon MSE: {recon_mse:.6f}")

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_every == 0 or combined_val < best_loss:
            is_best = combined_val < best_loss
            if is_best:
                best_loss = combined_val

            ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_text_loss': avg_text_loss,
                'train_img_loss': avg_img_loss if use_img_gen else None,
                'val_text_loss': val_text_loss,
                'val_img_loss': val_img_loss if use_img_gen else None,
                'best_loss': best_loss,
                'phase': 2 if use_img_gen else 1,
            }, ckpt_path)

            if is_best:
                best_path = output_dir / "best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  \u2713 New best (combined: {combined_val:.4f})")

        # ── Log ──
        log_entry = {
            'epoch': epoch + 1,
            'train_text_loss': avg_text_loss,
            'val_text_loss': val_text_loss,
            'lr': scheduler.get_last_lr()[0],
        }
        if use_img_gen:
            log_entry['train_img_loss'] = avg_img_loss
            log_entry['val_img_loss'] = val_img_loss
        metrics.append(log_entry)
        with open(log_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {output_dir.resolve()}")
    print(f"Logs: {log_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()

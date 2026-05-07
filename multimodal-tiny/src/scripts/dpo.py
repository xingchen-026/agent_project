#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) for caption quality improvement.

Uses cross-image negative sampling:
  - Preferred:  ground-truth COCO caption for the image
  - Rejected:   randomly sampled caption from a different image

This teaches the model to prefer captions that actually describe the image,
without requiring the model to generate good captions first.

Usage:
  python train_dpo.py --resume ../checkpoints_phase6_full/best.pt --epochs 5
"""

import os, sys, json, math, argparse, random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import TinyMultimodal
from core.tokenizer import SimpleTokenizer
from core.config import resolve_config
from utils import load_checkpoint_adaptive

from training.losses import lm_loss
from data.datasets import CocoCaptionDataset, ImageCaptionCollate, encode_captions, split_dataset
from training.optimizers import (build_scheduler, build_new_module_optimizer,
                       save_checkpoint, setup_output_dirs,
                       seed_everything, log_metrics, print_header, count_params)


# ── DPO Dataset ─────────────────────────────────────────────────────

class DpoDataset(Dataset):
    """Builds (image, preferred_caption, rejected_caption) triplets.

    Uses cross-image negative sampling: rejected caption is randomly
    sampled from a different image. Both captions are real COCO text,
    creating a clean preference signal.
    """

    def __init__(self, coco_dir, ann_file, image_size=224,
                 max_images=None, seed=42, pre_cache=True):
        # Load all image-caption pairs
        self.base_ds = CocoCaptionDataset(
            coco_dir, ann_file, image_size=image_size,
            max_images=max_images, max_captions_per_image=1,
            pre_cache=pre_cache, seed=seed,
        )
        self.samples = self.base_ds.samples
        self._cache = self.base_ds._cache
        self.image_size = image_size
        n = len(self.samples)

        # Build negative indices: for each sample, pick a random different-image caption
        random.seed(seed)
        self.neg_indices = []
        for i in range(n):
            pi, _ = self.samples[i]  # path for this sample
            j = i
            while j == i or self.samples[j][0] == pi:  # Ensure different image
                j = random.randrange(n)
            self.neg_indices.append(j)

        print(f"  DPO pairs: {n} (cross-image negatives)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pref_cap = self.samples[idx]
        _, rej_cap = self.samples[self.neg_indices[idx]]

        if self._cache is not None:
            img = self._cache[path]
        else:
            from data.datasets import preprocess_image_path
            img = preprocess_image_path(path, self.image_size)

        return img, pref_cap, rej_cap


# ── DPO Loss ────────────────────────────────────────────────────────

def compute_log_prob_per_token(model, image, text_ids):
    """Compute per-token log probabilities for a caption given an image.

    Returns: [B] tensor of summed log probabilities (one scalar per sample).
    """
    out = model(text_ids, images=image)
    logits = out['text_logits'] if isinstance(out, dict) else out

    B = logits.shape[0]
    shift_logits = logits[:, :-1].contiguous()
    shift_targets = text_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lps = log_probs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)

    return token_lps.sum(dim=-1)


def dpo_loss(policy_model, ref_model, image, pref_ids, rej_ids, beta=0.1):
    """Standard DPO loss.

    L = -log(sigmoid(beta * (log_ratio_pref - log_ratio_rej)))

    where log_ratio = log(π_policy / π_ref)
    """
    with torch.no_grad():
        ref_pref = compute_log_prob_per_token(ref_model, image, pref_ids)
        ref_rej = compute_log_prob_per_token(ref_model, image, rej_ids)

    pol_pref = compute_log_prob_per_token(policy_model, image, pref_ids)
    pol_rej = compute_log_prob_per_token(policy_model, image, rej_ids)

    log_ratio_pref = pol_pref - ref_pref
    log_ratio_rej = pol_rej - ref_rej

    return -F.logsigmoid(beta * (log_ratio_pref - log_ratio_rej)).mean()


def dpo_accuracy(policy_model, ref_model, image, pref_ids, rej_ids):
    """Fraction of samples where policy correctly prefers 'pref' over 'rej'."""
    with torch.no_grad():
        ref_pref = compute_log_prob_per_token(ref_model, image, pref_ids)
        ref_rej = compute_log_prob_per_token(ref_model, image, rej_ids)
        pol_pref = compute_log_prob_per_token(policy_model, image, pref_ids)
        pol_rej = compute_log_prob_per_token(policy_model, image, rej_ids)

    ratio_pref = pol_pref - ref_pref
    ratio_rej = pol_rej - ref_rej
    correct = (ratio_pref > ratio_rej).float().mean()
    return correct.item()


# ── Training ────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="DPO Alignment Training")
    p.add_argument("--resume", default="../checkpoints_phase6_full/best.pt")
    p.add_argument("--coco-dir", default="../coco_data")
    p.add_argument("--ann-file", default="../coco_data/captions_val2017.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature (lower = more conservative)")
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--output-dir", default="../checkpoints_phase6_dpo")
    p.add_argument("--log-dir", default="../logs_phase6_dpo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-text-len", type=int, default=48)
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer(max_vocab=10000)

    # ── Reference model (frozen) ──
    print("\nLoading reference model...")
    # Respect the checkpoint's modality config — don't override architecture
    cfg = resolve_config(args.resume, tokenizer, defaults={
        'use_memory_bank': False,
        'use_contrastive': False,
        'use_diffusion_decoder': False,
    })
    ref_model = TinyMultimodal(cfg).to(device)
    load_checkpoint_adaptive(ref_model, args.resume, device)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()
    print(f"  Reference: {count_params(ref_model):.2f}M params (frozen)")

    # ── Policy model (trainable, fresh load from checkpoint) ──
    print("Creating policy model...")
    policy_model = TinyMultimodal(cfg).to(device)
    load_checkpoint_adaptive(policy_model, args.resume, device)
    policy_model.train()
    print(f"  Policy: {count_params(policy_model):.2f}M params (trainable)")

    # ── Data ──
    print("\nBuilding DPO dataset...")
    ds = DpoDataset(args.coco_dir, args.ann_file, image_size=cfg.image_size,
                    max_images=args.max_images, seed=args.seed,
                    pre_cache=(args.max_images <= 5000))
    train_ds, val_ds = split_dataset(ds, val_frac=0.05, min_val=16, seed=args.seed)

    def dpo_collate(batch):
        imgs = torch.stack([item[0] for item in batch])
        pref_texts = [item[1] for item in batch]
        rej_texts = [item[2] for item in batch]
        return imgs, pref_texts, rej_texts

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               collate_fn=dpo_collate, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                             collate_fn=dpo_collate, num_workers=0)

    print(f"  Train: {len(train_ds)} pairs ({len(train_loader)} batches)")
    print(f"  Val: {len(val_ds)} pairs ({len(val_loader)} batches)")

    # ── Optimizer ──
    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01,
                       betas=(0.9, 0.999))
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"  LR: {args.lr:.1e}, beta: {args.beta}")

    output_dir, log_dir = setup_output_dirs(args.output_dir, args.log_dir)

    # ── Training ──
    print_header(f"DPO Training: {args.epochs} epochs")
    print(f"  Reference: frozen  |  Policy: trainable  |  beta: {args.beta}")
    print(f"  Cross-image negatives: preferred=GT, rejected=other image's caption")

    metrics = []
    best_acc = 0.0

    for epoch in range(args.epochs):
        policy_model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, pref_caps, rej_caps in pbar:
            images = images.to(device)
            pref_ids, _ = encode_captions(tokenizer, pref_caps,
                                           max_len=args.max_text_len, device=device)
            rej_ids, _ = encode_captions(tokenizer, rej_caps,
                                          max_len=args.max_text_len, device=device)

            loss = dpo_loss(policy_model, ref_model, images,
                            pref_ids, rej_ids, beta=args.beta)
            acc = dpo_accuracy(policy_model, ref_model, images,
                               pref_ids, rej_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            n += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.3f}'})

        avg_loss = epoch_loss / n
        avg_acc = epoch_acc / n

        # Validation
        policy_model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_n = 0
        with torch.no_grad():
            for images, pref_caps, rej_caps in val_loader:
                images = images.to(device)
                pref_ids, _ = encode_captions(tokenizer, pref_caps,
                                               max_len=args.max_text_len, device=device)
                rej_ids, _ = encode_captions(tokenizer, rej_caps,
                                              max_len=args.max_text_len, device=device)
                val_loss += dpo_loss(policy_model, ref_model, images,
                                      pref_ids, rej_ids, beta=args.beta).item()
                val_acc += dpo_accuracy(policy_model, ref_model, images,
                                         pref_ids, rej_ids)
                val_n += 1

        avg_val_loss = val_loss / val_n
        avg_val_acc = val_acc / val_n

        record = {
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr()[0],
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
        }
        metrics.append(record)
        log_metrics(log_dir, metrics)

        is_best = avg_val_acc > best_acc
        if is_best:
            best_acc = avg_val_acc

        save_checkpoint(output_dir, epoch + 1, policy_model, optimizer,
                        best_acc if is_best else best_acc,
                        model_config=cfg.to_dict(), is_best=is_best)

        status = " [BEST]" if is_best else ""
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f} acc={avg_acc:.3f}  "
              f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.3f}{status}")

    print(f"\nDPO training complete! Best val_acc: {best_acc:.4f}")

    # Final: generate a few captions with DPO-tuned model vs reference
    print("\n--- Generation comparison (DPO vs Reference) ---")
    from data.datasets import preprocess_image_path

    policy_model.eval()
    test_samples = random.sample(ds.samples, min(5, len(ds.samples)))
    for path, gt in test_samples:
        img = preprocess_image_path(path, cfg.image_size).unsqueeze(0).to(device)

        ref_gen = ref_model.generate_text(img, tokenizer, max_len=40,
                                           temperature=0.8, top_k=50)
        dpo_gen = policy_model.generate_text(img, tokenizer, max_len=40,
                                              temperature=0.8, top_k=50)

        print(f"  GT:   {gt[:80]}")
        print(f"  Ref:  {ref_gen[:80]}")
        print(f"  DPO:  {dpo_gen[:80]}")
        print()


if __name__ == '__main__':
    main()

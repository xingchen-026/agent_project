#!/usr/bin/env python3
"""Unified Training Entry Point — consolidates 8 training scripts via --mode flag.

Modes:
  full       Image+Audio+Video joint training on COCO (replaces train_joint_full.py)
  joint      CLIP+LM+Diffusion joint training on COCO (replaces train_joint.py)
  clip       CLIP contrastive pre-training on COCO (replaces train_clip.py)
  distill    ResNet50 → MemoryBank knowledge distillation (replaces train_distill.py)
  base       From-scratch synthetic multi-modal training (replaces train.py)

Usage:
  python train_unified.py --mode full --resume ../checkpoints_phase6/best.pt --epochs 20
  python train_unified.py --mode joint --resume ../checkpoints_phase6/best.pt --epochs 15
  python train_unified.py --mode clip --resume ../checkpoints_phase6/best.pt --epochs 10
"""

import os, sys, json, math, argparse, random, shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import ModelConfig, resolve_config
from utils import load_checkpoint_adaptive

from losses import lm_loss, mse_loss, clip_contrastive_loss, diffusion_loss_fn, distill_loss, retrieval_accuracy
from data_lib import (CocoCaptionDataset, CachedPairDataset, ImageCaptionCollate,
                       encode_captions, split_dataset, preprocess_image_path, preprocess_image_pil)
from training import (build_new_module_optimizer, build_differential_optimizer,
                       build_scheduler, save_checkpoint, setup_output_dirs,
                       seed_everything, log_metrics, print_header, count_params)


# ── Shared Argparser ────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Unified Multi-Modal Training")
    # Mode
    p.add_argument("--mode", required=True,
                   choices=["full", "joint", "clip", "distill", "base"])
    # Common
    p.add_argument("--resume", default="../checkpoints_phase6/best.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="../checkpoints_phase6_unified")
    p.add_argument("--log-dir", default="../logs_phase6_unified")
    # COCO data
    p.add_argument("--coco-dir", default="../coco_data")
    p.add_argument("--ann-file", default="../coco_data/captions_val2017.json")
    p.add_argument("--max-images", type=int, default=2000)
    # Modality flags
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--use-memory-bank", action="store_true")
    # Joint/multi loss weights
    p.add_argument("--clip-weight", type=float, default=1.0)
    p.add_argument("--lm-weight", type=float, default=0.5)
    p.add_argument("--diff-weight", type=float, default=0.1)
    # CLIP specific
    p.add_argument("--temperature", type=float, default=0.07)
    # Base training
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--val-size", type=int, default=500)
    # Distill
    p.add_argument("--teacher-model", default="resnet50")
    return p.parse_args()


# ── Shared Training Loop ────────────────────────────────────────────

def run_training_loop(model, optimizer, scheduler, train_loader,
                       train_step_fn, val_step_fn, args, mode_name,
                       tokenizer=None, device='cuda'):
    """Generic training loop used by all modes.

    train_step_fn(model, batch, device, optimizer, scheduler) → float loss
    The function handles backward + optimizer step internally.
    """
    output_dir, log_dir = setup_output_dirs(args.output_dir, args.log_dir)

    metrics = []
    best_metric = float('inf')
    metric_name = 'val_loss'

    print_header(f"{mode_name.upper()} Training: {args.epochs} epochs")
    print(f"  Batches: {len(train_loader)}/epoch  LR: {args.lr:.1e}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            loss = train_step_fn(model, batch, device, optimizer, scheduler)
            epoch_loss += loss
            n += 1
            pbar.set_postfix({'loss': f'{loss:.3f}'})

        avg_loss = epoch_loss / max(n, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_result = val_step_fn(model, device)

        # Handle different val return types
        if isinstance(val_result, dict):
            avg_val = val_result.get('val_loss', val_result.get('recall@1', 0.0))
            record = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0],
                      'train_loss': avg_loss, **val_result}
        else:
            avg_val = val_result
            record = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0],
                      'train_loss': avg_loss, 'val_loss': avg_val}

        metrics.append(record)
        log_metrics(log_dir, metrics)

        # For retrieval modes, higher is better
        is_best = avg_val < best_metric
        if is_best:
            best_metric = avg_val

        save_checkpoint(output_dir, epoch + 1, model, optimizer,
                        best_metric, model_config=None, is_best=is_best)

        val_str = " ".join(f"{k}={v:.4f}" for k, v in val_result.items()) \
            if isinstance(val_result, dict) else f"val={avg_val:.4f}"
        status = " [BEST]" if is_best else ""
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}  {val_str}{status}")

    print(f"\n{mode_name} training complete! Best {metric_name}: {best_metric:.4f}")
    return output_dir


# ═════════════════════════════════════════════════════════════════════
# Mode: FULL — Image+Audio+Video Joint Training
# Replaces: train_joint_full.py
# ═════════════════════════════════════════════════════════════════════

def setup_full(args, tokenizer, device):
    """Full multi-modal joint training: COCO images + synthetic audio + video."""
    from audio_synthetic import AudioDataset
    from video_synthetic import VideoDataset

    cfg = resolve_config(args.resume, tokenizer, defaults={
        'img_generation': True,
        'use_audio': not args.no_audio,
        'use_video': not args.no_video,
        'use_memory_bank': args.use_memory_bank,
        'use_contrastive': False,
        'use_diffusion_decoder': False,
    })
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {count_params(model):.2f}M ({cfg.describe()})")
    print(f"  Audio: {cfg.use_audio}  Video: {cfg.use_video}  MemoryBank: {cfg.use_memory_bank}")
    load_checkpoint_adaptive(model, args.resume, device)

    # Data
    print("\nBuilding datasets...")
    img_ds = CocoCaptionDataset(args.coco_dir, args.ann_file,
                                 image_size=cfg.image_size, max_images=args.max_images,
                                 max_captions_per_image=2, pre_cache=True, seed=args.seed)
    img_train, img_val = split_dataset(img_ds, seed=args.seed)
    img_loader = DataLoader(img_train, args.batch_size, shuffle=True,
                            collate_fn=ImageCaptionCollate(), num_workers=0, drop_last=True)
    print(f"  Image: {len(img_train)} train / {len(img_val)} val")

    aud_loader = vid_loader = None
    if cfg.use_audio:
        aud_ds = AudioDataset(num_samples=1000, seed=args.seed)
        aud_train, aud_val = split_dataset(aud_ds, seed=args.seed)
        aud_loader = DataLoader(aud_train, args.batch_size, shuffle=True,
                                collate_fn=ImageCaptionCollate(), num_workers=0, drop_last=True)
        print(f"  Audio: {len(aud_train)} train / {len(aud_val)} val")
    if cfg.use_video:
        vid_ds = VideoDataset(num_samples=800, seed=args.seed)
        vid_train, vid_val = split_dataset(vid_ds, seed=args.seed)
        vid_loader = DataLoader(vid_train, args.batch_size, shuffle=True,
                                collate_fn=ImageCaptionCollate(), num_workers=0, drop_last=True)
        print(f"  Video: {len(vid_train)} train / {len(vid_val)} val")

    # Optimizer
    new_names = ['memory_bank', 'audio_proj', 'audio_decoder', 'video_proj', 'video_decoder', 'img_decoder']
    optimizer = build_new_module_optimizer(model, args.lr, new_names)

    total_steps = len(img_loader) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_ratio=0.03, min_warmup=min(100, total_steps // 10))

    # Loss weights
    W = {'img_lm': 0.5, 'img_rec': 0.3, 'aud_lm': 0.3, 'aud_rec': 0.1,
         'vid_lm': 0.2, 'vid_rec': 0.1}

    # Mutable state for cycling audio/video iterators
    state = {'aud_it': iter(aud_loader) if cfg.use_audio else None,
             'vid_it': iter(vid_loader) if cfg.use_video else None}

    def train_step(model, batch, device, optimizer, scheduler):
        images, captions = batch
        images = images.to(device)
        text_ids, lengths = encode_captions(tokenizer, captions, device=device)
        total_loss = 0.0

        # Image
        out = model(text_ids, images=images, return_img=True)
        loss = lm_loss(out['text_logits'], text_ids, lengths) * W['img_lm']
        if 'img_recon' in out:
            loss = loss + mse_loss(out['img_recon'], model._image_to_patches(images)) * W['img_rec']
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()

        # Audio
        if cfg.use_audio:
            try:
                aud, aud_caps = next(state['aud_it'])
            except StopIteration:
                state['aud_it'] = iter(aud_loader)
                aud, aud_caps = next(state['aud_it'])
            aud = aud.to(device)
            aud_text_ids, aud_len = encode_captions(tokenizer, aud_caps, device=device)
            aud_out = model(aud_text_ids, audios=aud, return_audio=True)
            aud_loss = lm_loss(aud_out['text_logits'], aud_text_ids, aud_len) * W['aud_lm']
            if aud_out.get('target_aud') is not None:
                aud_loss = aud_loss + mse_loss(aud_out['aud_recon'], aud_out['target_aud']) * W['aud_rec']
            aud_loss.backward()
            total_loss += aud_loss.item()

        # Video
        if cfg.use_video:
            try:
                vid, vid_caps = next(state['vid_it'])
            except StopIteration:
                state['vid_it'] = iter(vid_loader)
                vid, vid_caps = next(state['vid_it'])
            vid = vid.to(device)
            vid_text_ids, vid_len = encode_captions(tokenizer, vid_caps, device=device)
            vid_out = model(vid_text_ids, videos=vid, return_video=True)
            vid_loss = lm_loss(vid_out['text_logits'], vid_text_ids, vid_len) * W['vid_lm']
            if vid_out.get('target_vid') is not None:
                vid_loss = vid_loss + mse_loss(vid_out['vid_recon'], vid_out['target_vid']) * W['vid_rec']
            vid_loss.backward()
            total_loss += vid_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return total_loss

    # Validation step
    @torch.no_grad()
    def val_step(model, device):
        total_val = 0.0
        n = 0
        for images, captions in img_val:
            images = images.unsqueeze(0).to(device)
            text_ids, lengths = encode_captions(tokenizer, [captions], device=device)
            out = model(text_ids, images=images)
            logits = out if isinstance(out, torch.Tensor) else out['text_logits']
            total_val += lm_loss(logits, text_ids, lengths).item()
            n += 1
            if n >= 50:
                break
        return {'val_loss': total_val / max(n, 1)}

    return model, img_loader, optimizer, scheduler, train_step, val_step


# ═════════════════════════════════════════════════════════════════════
# Mode: JOINT — CLIP+LM+Diffusion Joint Training
# Replaces: train_joint.py
# ═════════════════════════════════════════════════════════════════════

def setup_joint(args, tokenizer, device):
    """CLIP + LM + Diffusion joint training on COCO."""
    cfg = resolve_config(args.resume, tokenizer, defaults={
        'img_generation': True, 'use_audio': False, 'use_video': False,
        'use_memory_bank': True, 'n_mem_tokens': 16,
        'use_contrastive': True, 'contrastive_dim': 256,
        'use_diffusion_decoder': True,
    })
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {count_params(model):.2f}M ({cfg.describe()})")
    print(f"  MemoryBank: {cfg.use_memory_bank}  Contrastive: {cfg.use_contrastive}  Diffusion: {cfg.use_diffusion_decoder}")
    load_checkpoint_adaptive(model, args.resume, device)

    # Data
    from data_lib import PadCollate
    ds = CocoCaptionDataset(args.coco_dir, args.ann_file,
                             image_size=cfg.image_size, max_images=args.max_images,
                             max_captions_per_image=3, pre_cache=True, seed=args.seed)
    train_ds, val_ds = split_dataset(ds, val_frac=0.05, seed=args.seed)
    collate = PadCollate(tokenizer, max_len=48, add_bos=True, return_dict=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               collate_fn=collate, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=0)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Optimizer
    new_names = ['contrastive_proj', 'diffusion_decoder', 'memory_bank']
    optimizer = build_new_module_optimizer(model, args.lr, new_names)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps)
    print(f"  LR: new={args.lr * 5:.1e} body={args.lr:.1e}")

    def train_step(model, batch, device, optimizer, scheduler):
        images, text_ids, lengths = batch
        images = images.to(device)
        text_ids = text_ids.to(device)

        out = model(text_ids, images=images, return_memory_hidden=True)
        mem_hidden = out.get('memory_hidden')
        patches = model._image_to_patches(images)

        # CLIP loss
        if cfg.use_contrastive and mem_hidden is not None:
            text_hidden = out.get('text_hidden')
            img_emb = model.contrastive_proj(mem_hidden.mean(dim=1))
            text_emb = model.contrastive_proj(text_hidden.mean(dim=1)) if text_hidden is not None else img_emb
            loss_clip = clip_contrastive_loss(img_emb, text_emb) * args.clip_weight
        else:
            loss_clip = torch.tensor(0.0, device=device)

        # LM loss
        loss_lm = lm_loss(out['text_logits'], text_ids, lengths) * args.lm_weight

        # Diffusion loss
        if cfg.use_diffusion_decoder and mem_hidden is not None:
            loss_diff = diffusion_loss_fn(model.img_decoder, patches, mem_hidden) * args.diff_weight
        else:
            loss_diff = torch.tensor(0.0, device=device)

        loss = loss_clip + loss_lm + loss_diff
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return loss.item()

    @torch.no_grad()
    def val_step(model, device):
        total = 0.0
        for images, text_ids, lengths in val_loader:
            images = images.to(device)
            text_ids = text_ids.to(device)
            out = model(text_ids, images=images)
            logits = out if isinstance(out, torch.Tensor) else out['text_logits']
            total += lm_loss(logits, text_ids, lengths).item()
        return {'val_loss': total / len(val_loader)}

    return model, train_loader, optimizer, scheduler, train_step, val_step


# ═════════════════════════════════════════════════════════════════════
# Mode: CLIP — Contrastive Pre-training
# Replaces: train_clip.py
# ═════════════════════════════════════════════════════════════════════

def setup_clip(args, tokenizer, device):
    """CLIP contrastive pre-training on COCO."""
    cfg = resolve_config(args.resume, tokenizer, defaults={
        'img_generation': True, 'use_audio': True, 'use_video': True,
        'use_contrastive': True, 'contrastive_dim': 256,
    })
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {count_params(model):.2f}M ({cfg.describe()})  Contrastive: {cfg.use_contrastive}")
    load_checkpoint_adaptive(model, args.resume, device)

    # Data — split at image level to prevent caption leakage
    print("\nBuilding COCO contrastive dataset...")
    ds = CocoCaptionDataset(args.coco_dir, args.ann_file,
                             image_size=cfg.image_size, max_images=args.max_images,
                             max_captions_per_image=5, pre_cache=True, seed=args.seed)

    # Group by image path for proper split
    img_to_pairs = {}
    for path, cap in ds.samples:
        img_to_pairs.setdefault(path, []).append((path, cap))
    img_paths = sorted(img_to_pairs.keys())
    random.seed(args.seed)
    random.shuffle(img_paths)
    n_val_imgs = max(10, int(len(img_paths) * 0.05))
    val_paths = set(img_paths[:n_val_imgs])
    train_pairs = [(p, c) for p, c in ds.samples if p not in val_paths]
    val_pairs = [(p, c) for p, c in ds.samples if p in val_paths]

    train_ds = CachedPairDataset(train_pairs, image_size=cfg.image_size, cache=ds._cache)
    val_ds = CachedPairDataset(val_pairs, image_size=cfg.image_size, cache=ds._cache)
    print(f"  Train: {len(train_pairs)} pairs  Val: {len(val_pairs)} pairs")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               collate_fn=ImageCaptionCollate(), num_workers=0,
                               drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                             collate_fn=ImageCaptionCollate(), num_workers=0)

    # Optimizer
    optimizer = build_new_module_optimizer(model, args.lr, ['contrastive_proj'])
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"  LR: proj={args.lr * 5:.1e}, body={args.lr:.1e}")

    def train_step(model, batch, device, optimizer, scheduler):
        images, captions = batch
        images = images.to(device)
        bos_id = 2
        encoded = tokenizer(list(captions), padding=True, truncation=True,
                            max_length=48, return_tensors=True)
        text_ids = torch.cat([
            torch.full((len(captions), 1), bos_id, dtype=torch.long),
            encoded['input_ids']
        ], dim=1)[:, :48].to(device)

        img_emb, text_emb = model._encode_contrastive_impl(images, text_ids)
        loss = clip_contrastive_loss(img_emb, text_emb, args.temperature)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return loss.item()

    @torch.no_grad()
    def val_step(model, device):
        all_img, all_text = [], []
        for images, captions in val_loader:
            images = images.to(device)
            bos_id = 2
            encoded = tokenizer(list(captions), padding=True, truncation=True,
                                max_length=48, return_tensors=True)
            text_ids = torch.cat([
                torch.full((len(captions), 1), bos_id, dtype=torch.long),
                encoded['input_ids']
            ], dim=1)[:, :48].to(device)
            ie, te = model._encode_contrastive_impl(images, text_ids)
            all_img.append(ie)
            all_text.append(te)
        return retrieval_accuracy(torch.cat(all_img), torch.cat(all_text))

    return model, train_loader, optimizer, scheduler, train_step, val_step


# ═════════════════════════════════════════════════════════════════════
# Mode: DISTILL — ResNet50 → MemoryBank Knowledge Distillation
# Replaces: train_distill.py
# ═════════════════════════════════════════════════════════════════════

def setup_distill(args, tokenizer, device):
    """ResNet50 teacher → TinyMultimodal MemoryBank distillation."""
    import torchvision.models as models

    # Teacher
    print("Loading teacher: ResNet50 (ImageNet pretrained)...")
    teacher = models.resnet50(weights='IMAGENET1K_V2').to(device)
    teacher.fc = nn.Identity()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student
    cfg = resolve_config(args.resume, tokenizer, defaults={
        'img_generation': True, 'use_audio': True, 'use_video': True,
        'use_memory_bank': True, 'n_mem_tokens': 16,
    })
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {count_params(model):.2f}M ({cfg.describe()})")
    load_checkpoint_adaptive(model, args.resume, device)

    # Distill head: student_dim → 2048
    distill_head = nn.Sequential(
        nn.Linear(cfg.dim, cfg.dim * 2), nn.GELU(),
        nn.Linear(cfg.dim * 2, 2048),
    ).to(device)

    # Teacher normalization
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Data: images only
    ds = CocoCaptionDataset(args.coco_dir, args.ann_file,
                             image_size=cfg.image_size, max_images=args.max_images,
                             max_captions_per_image=1, pre_cache=True, seed=args.seed)
    train_ds, val_ds = split_dataset(ds, val_frac=0.05, min_val=10, seed=args.seed)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               collate_fn=ImageCaptionCollate(), num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                             collate_fn=ImageCaptionCollate(), num_workers=0)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Optimizer: distill_head (5x), memory_bank (3x), body (0.5x)
    distill_params = list(distill_head.parameters())
    mem_params = [p for n, p in model.named_parameters() if 'memory_bank' in n]
    other_params = [p for n, p in model.named_parameters() if 'memory_bank' not in n]

    optimizer = AdamW([
        {'params': distill_params, 'lr': args.lr * 5},
        {'params': mem_params, 'lr': args.lr * 3},
        {'params': other_params, 'lr': args.lr * 0.5},
    ], weight_decay=0.1, betas=(0.9, 0.98))

    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    def train_step(model_and_head, batch, device, optimizer, scheduler):
        _model, _head = model_and_head
        images, _ = batch
        images = images.to(device)

        # Teacher: normalized input
        teacher_input = (images + 1) / 2
        teacher_input = (teacher_input - imagenet_mean) / imagenet_std
        with torch.no_grad():
            teacher_emb = teacher(teacher_input)

        # Student
        bos_id = 2
        text_ids = torch.full((images.shape[0], 1), bos_id, dtype=torch.long, device=device)
        out = _model(text_ids, images=images, return_memory_hidden=True)
        mem_hidden = out.get('memory_hidden')
        if mem_hidden is None:
            return 0.0

        student_emb = _head(mem_hidden.mean(dim=1))
        loss = distill_loss(student_emb, teacher_emb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(_model.parameters()) + list(_head.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        return loss.item()

    @torch.no_grad()
    def val_step(model_and_head, device):
        _model, _head = model_and_head
        cos_sim_total = 0.0
        n = 0
        for images, _ in val_loader:
            images = images.to(device)
            teacher_input = (images + 1) / 2
            teacher_input = (teacher_input - imagenet_mean) / imagenet_std
            teacher_emb = teacher(teacher_input)
            bos_id = 2
            text_ids = torch.full((images.shape[0], 1), bos_id, dtype=torch.long, device=device)
            out = _model(text_ids, images=images, return_memory_hidden=True)
            mem_hidden = out.get('memory_hidden')
            if mem_hidden is not None:
                student_emb = _head(mem_hidden.mean(dim=1))
                cos_sim_total += F.cosine_similarity(student_emb, teacher_emb).mean().item()
            n += 1
        return {'val_cosine': cos_sim_total / max(n, 1)}

    return (model, distill_head), train_loader, optimizer, scheduler, train_step, val_step


# ═════════════════════════════════════════════════════════════════════
# Mode: BASE — From-scratch synthetic training
# Replaces: train.py
# ═════════════════════════════════════════════════════════════════════

def setup_base(args, tokenizer, device):
    """From-scratch multi-modal training on synthetic data."""
    from synthetic_data import SyntheticDataset
    from audio_synthetic import AudioDataset
    from video_synthetic import VideoDataset
    from data_lib import PadCollate, interleave_loaders

    cfg = ModelConfig(
        dim=args.dim, n_layers=args.layers, n_heads=args.n_heads,
        vocab_size=tokenizer.vocab_size,
        img_generation=True,
        use_audio=not args.no_audio,
        use_video=not args.no_video,
    )
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {count_params(model):.2f}M ({cfg.describe()})  "
          f"Audio: {cfg.use_audio}  Video: {cfg.use_video}")

    if os.path.exists(args.resume):
        load_checkpoint_adaptive(model, args.resume, device)

    # Data
    img_ds = SyntheticDataset(num_samples=args.train_size, image_size=cfg.image_size, seed=args.seed)
    val_img = SyntheticDataset(num_samples=args.val_size, image_size=cfg.image_size, seed=args.seed + 1)

    loaders = [DataLoader(img_ds, args.batch_size, shuffle=True,
                          collate_fn=PadCollate(tokenizer, max_len=48, add_bos=False, return_dict=True),
                          num_workers=0)]
    if cfg.use_audio:
        aud_ds = AudioDataset(num_samples=5000, seed=args.seed)
        loaders.append(DataLoader(aud_ds, args.batch_size, shuffle=True,
                                   collate_fn=PadCollate(tokenizer, max_len=48, add_bos=False, return_dict=True),
                                   num_workers=0))
    if cfg.use_video:
        vid_ds = VideoDataset(num_samples=3000, seed=args.seed)
        loaders.append(DataLoader(vid_ds, args.batch_size, shuffle=True,
                                   collate_fn=PadCollate(tokenizer, max_len=48, add_bos=False, return_dict=True),
                                   num_workers=0))

    train_batches = interleave_loaders(*loaders)
    val_loader = DataLoader(val_img, args.batch_size, shuffle=False,
                             collate_fn=PadCollate(tokenizer, max_len=48, add_bos=False, return_dict=True),
                             num_workers=0)
    print(f"  Batches: {len(train_batches)}  Val: {len(val_loader)}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = len(train_batches) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps)

    def train_step(model, batch, device, optimizer, scheduler):
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        images = data.get('images')
        audios = data.get('audios')
        videos = data.get('videos')
        text_ids = data['text_ids']

        out = model(text_ids, images=images, audios=audios, videos=videos,
                    return_img=True,
                    return_audio=audios is not None,
                    return_video=videos is not None)
        if isinstance(out, torch.Tensor):
            loss = lm_loss(out, text_ids, attn_mask=data['attn_mask'])
        else:
            loss = lm_loss(out['text_logits'], text_ids, attn_mask=data['attn_mask'])
            if out.get('img_recon') is not None and images is not None:
                loss = loss + mse_loss(out['img_recon'], model._image_to_patches(images)) * 0.5
            if out.get('aud_recon') is not None:
                loss = loss + mse_loss(out['aud_recon'], out.get('target_aud')) * 0.5
            if out.get('vid_recon') is not None:
                loss = loss + mse_loss(out['vid_recon'], out.get('target_vid')) * 0.5

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return loss.item()

    @torch.no_grad()
    def val_step(model, device):
        total = 0.0
        for batch in val_loader:
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(data['text_ids'], images=data.get('images'))
            logits = out if isinstance(out, torch.Tensor) else out['text_logits']
            total += lm_loss(logits, data['text_ids'], attn_mask=data['attn_mask']).item()
        return {'val_loss': total / len(val_loader)}

    # Wrap train_batches in a simple iterable for the training loop
    class BatchList:
        def __init__(self, batches):
            self.batches = batches
        def __len__(self):
            return len(self.batches)
        def __iter__(self):
            return iter(self.batches)

    return model, BatchList(train_batches), optimizer, scheduler, train_step, val_step


# ═════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: {args.mode}")

    tokenizer = SimpleTokenizer(max_vocab=10000)

    # Mode dispatcher
    setup_fns = {
        'full': setup_full,
        'joint': setup_joint,
        'clip': setup_clip,
        'distill': setup_distill,
        'base': setup_base,
    }

    if args.mode not in setup_fns:
        print(f"Unknown mode: {args.mode}. Choices: {list(setup_fns.keys())}")
        sys.exit(1)

    model_or_tuple, train_loader, optimizer, scheduler, train_step, val_step = \
        setup_fns[args.mode](args, tokenizer, device)

    run_training_loop(model_or_tuple, optimizer, scheduler, train_loader,
                       train_step, val_step, args, args.mode,
                       tokenizer=tokenizer, device=device)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
CLIP-style Contrastive Pre-training for TinyMultimodal.
Aligns image/text embeddings via InfoNCE loss on COCO captions.

Usage:
  python train_clip.py --resume ../checkpoints_phase6/best.pt --epochs 10
"""

import os, sys, json, math, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import ModelConfig, resolve_config
from utils import load_checkpoint_adaptive


# ── COCO Contrastive Dataset ──────────────────────────────────────

class CocoContrastiveDataset(Dataset):
    """COCO images + captions for contrastive learning."""

    def __init__(self, coco_dir, ann_file, image_size=224, max_samples=None):
        from pycocotools.coco import COCO
        self.img_dir = Path(coco_dir) / 'val2017'
        self.image_size = image_size

        coco = COCO(str(ann_file))
        img_ids = sorted(coco.imgs.keys())
        if max_samples:
            img_ids = img_ids[:max_samples]

        self.pairs = []
        for img_id in img_ids:
            img_info = coco.imgs[img_id]
            img_path = self.img_dir / img_info['file_name']
            if not img_path.exists():
                continue
            ann_ids = coco.getAnnIds(imgIds=img_id)
            for ann in coco.loadAnns(ann_ids):
                caption = ann['caption'].strip()
                if caption:
                    self.pairs.append((str(img_path), caption))

        # Deduplicate images: keep up to 5 captions per image
        seen = {}
        dedup = []
        for path, cap in self.pairs:
            n = seen.get(path, 0)
            if n < 5:
                dedup.append((path, cap))
                seen[path] = n + 1
        self.pairs = dedup

        print(f"  Contrastive Dataset: {len(self.pairs)} pairs "
              f"({len(seen)} images, {len(self.pairs)//len(seen):.1f} captions/img)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return img_tensor, caption


# ── InfoNCE Loss ───────────────────────────────────────────────────

def clip_loss(img_emb, text_emb, temperature=0.07):
    """Symmetric InfoNCE loss (CLIP-style)."""
    logits = (text_emb @ img_emb.T) / temperature  # [B, B]
    labels = torch.arange(len(logits), device=logits.device)
    loss_t2i = F.cross_entropy(logits, labels)
    loss_i2t = F.cross_entropy(logits.T, labels)
    return (loss_t2i + loss_i2t) / 2


def retrieval_accuracy(img_emb, text_emb, top_k=(1, 5, 10)):
    """Compute text→image retrieval accuracy."""
    sim = text_emb @ img_emb.T  # [B, B]
    acc = {}
    for k in top_k:
        correct = sum(1 for i in range(len(sim)) if i in sim[i].topk(k).indices)
        acc[f'recall@{k}'] = correct / len(sim)
    return acc


# ── Cached Pair Dataset ────────────────────────────────────────────

class CachedPairDataset(Dataset):
    """Pre-cached image-text pairs for fast training."""
    def __init__(self, pairs, image_size=224, cache=None):
        self.pairs = pairs
        if cache is not None:
            self._cache = cache
        else:
            unique_paths = sorted(set(p for p, _ in pairs))
            self._cache = {}
            print(f"  Pre-caching {len(unique_paths)} images...")
            for i, path in enumerate(unique_paths):
                img = Image.open(path).convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)
                self._cache[path] = torch.from_numpy(np.array(img)).permute(2,0,1).float()/127.5-1.0
                if (i+1) % 1000 == 0:
                    print(f"    {i+1}/{len(unique_paths)}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        return self._cache[img_path], caption


# ── Collate ────────────────────────────────────────────────────────

def contrastive_collate(batch):
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return images, captions


# ── Training ───────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="CLIP Contrastive Pre-training")
    parser.add_argument("--resume", default="../checkpoints_phase6/best.pt")
    parser.add_argument("--coco-dir", default="../coco_data")
    parser.add_argument("--ann-file", default="../coco_data/captions_val2017.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--output-dir", default="../checkpoints_phase6_clip")
    parser.add_argument("--log-dir", default="../logs_phase6_clip")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer(max_vocab=10000)

    # Build model with contrastive head
    cfg_dict = {'img_generation': True, 'use_audio': True, 'use_video': True,
                'use_contrastive': True, 'contrastive_dim': 256}
    if os.path.exists(args.resume):
        cfg = resolve_config(args.resume, tokenizer, defaults=cfg_dict)
    else:
        cfg = ModelConfig(vocab_size=tokenizer.vocab_size, **cfg_dict)
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M ({cfg.describe()})")

    if os.path.exists(args.resume):
        load_checkpoint_adaptive(model, args.resume, device)
        print(f"Loaded: {args.resume}")
    else:
        print("Training from scratch")

    # Data — split at IMAGE level (not pair level) to prevent leakage
    print("\nBuilding COCO contrastive dataset...")
    ds = CocoContrastiveDataset(args.coco_dir, args.ann_file,
                                image_size=cfg.image_size, max_samples=args.max_samples)

    # Group pairs by image path for proper split
    img_to_pairs = {}
    for path, cap in ds.pairs:
        img_to_pairs.setdefault(path, []).append((path, cap))
    img_paths = sorted(img_to_pairs.keys())
    random.shuffle(img_paths)
    n_val_imgs = max(10, int(len(img_paths) * 0.05))
    val_paths = set(img_paths[:n_val_imgs])
    train_pairs = [(p, c) for p, c in ds.pairs if p not in val_paths]
    val_pairs = [(p, c) for p, c in ds.pairs if p in val_paths]

    # Pre-cache all images once, share cache across train/val
    all_paths = sorted(set(p for p, _ in ds.pairs))
    shared_cache = {}
    print(f"  Pre-caching {len(all_paths)} images...")
    for i, path in enumerate(all_paths):
        img = Image.open(path).convert('RGB')
        img = img.resize((cfg.image_size, cfg.image_size), Image.LANCZOS)
        shared_cache[path] = torch.from_numpy(np.array(img)).permute(2,0,1).float()/127.5-1.0
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{len(all_paths)}")

    train_ds = CachedPairDataset(train_pairs, cache=shared_cache)
    val_ds = CachedPairDataset(val_pairs, cache=shared_cache)
    print(f"  Train: {len(train_pairs)} pairs ({len(img_paths)-n_val_imgs} imgs)")
    print(f"  Val: {len(val_pairs)} pairs ({n_val_imgs} imgs)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=contrastive_collate, num_workers=2,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=contrastive_collate, num_workers=2, pin_memory=True)

    # Optimizer — fine-tune body gently, train new projection head faster
    proj_params = [p for n, p in model.named_parameters() if 'contrastive_proj' in n]
    other_params = [p for n, p in model.named_parameters() if 'contrastive_proj' not in n]
    optimizer = AdamW([
        {'params': proj_params, 'lr': args.lr * 5},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=0.1, betas=(0.9, 0.98))
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"  LR: proj={args.lr*5:.1e}, body={args.lr:.1e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\n{'='*60}")
    print(f"CLIP Contrastive Pre-training: {args.epochs} epochs, {len(train_pairs)} pairs")
    print(f"{'='*60}\n")

    metrics = []
    best_recall = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, captions in pbar:
            images = images.to(device)
            bos_id = 2
            encoded = tokenizer(list(captions), padding=True, truncation=True,
                                max_length=48, return_tensors=True)
            text_ids = torch.cat([
                torch.full((len(captions), 1), bos_id, dtype=torch.long),
                encoded['input_ids']
            ], dim=1)[:, :48].to(device)

            img_emb, text_emb = model._encode_contrastive_impl(images, text_ids)
            loss = clip_loss(img_emb, text_emb, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_img_embs, val_text_embs = [], []
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                encoded = tokenizer(list(captions), padding=True, truncation=True,
                                    max_length=48, return_tensors=True)
                text_ids = torch.cat([
                    torch.full((len(captions), 1), bos_id, dtype=torch.long),
                    encoded['input_ids']
                ], dim=1)[:, :48].to(device)
                ie, te = model._encode_contrastive_impl(images, text_ids)
                val_img_embs.append(ie)
                val_text_embs.append(te)

        val_img = torch.cat(val_img_embs)
        val_text = torch.cat(val_text_embs)
        acc = retrieval_accuracy(val_img, val_text)

        record = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0],
                  'train_loss': avg_loss, **acc}
        metrics.append(record)

        with open(log_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        is_best = acc['recall@1'] > best_recall
        if is_best:
            best_recall = acc['recall@1']

        ckpt_path = output_dir / f'epoch_{epoch+1}.pt'
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_recall': best_recall,
                    'model_config': cfg.to_dict()}, ckpt_path)

        status = "[BEST]" if is_best else ""
        recall_str = " ".join(f"R@{k}={v:.3f}" for k, v in acc.items())
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} {recall_str} {status}")

        if is_best:
            import shutil
            shutil.copy(ckpt_path, output_dir / 'best.pt')

    print(f"\nCLIP training complete! Best R@1: {best_recall:.4f}")


if __name__ == '__main__':
    main()

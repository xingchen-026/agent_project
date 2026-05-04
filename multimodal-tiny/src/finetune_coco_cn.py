#!/usr/bin/env python3
"""
Phase 5 v3 — COCO-CN Real Chinese Image Fine-tuning.
Uses COCO val2014 images + Chinese captions from COCO-CN dataset.

Usage:
  cd multimodal-tiny/src
  PYTHONIOENCODING=utf-8 python finetune_coco_cn.py \
    --resume ../checkpoints_phase5_v2/best.pt --epochs 10
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
from config import resolve_config
from utils import compute_text_loss, logger, load_checkpoint_adaptive


# ── COCO-CN Dataset ───────────────────────────────────────────────

class CocoCnDataset(Dataset):
    """COCO val2014 images with Chinese captions from COCO-CN."""

    def __init__(self, coco_dir, captions_file, image_size=224, max_samples=None):
        coco_dir = Path(coco_dir)
        self.img_dir = coco_dir / 'val2014'

        # Parse COCO-CN captions
        self.samples = []
        with open(captions_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                img_id, caption = line.split('\t', 1)
                # COCO-CN format: COCO_val2014_XXXXXXXXXX#N or COCO_train2014_...
                img_name = img_id.split('#')[0]
                if 'val2014' not in img_name:
                    continue
                img_path = self.img_dir / (img_name + '.jpg')
                if img_path.exists():
                    self.samples.append((str(img_path), caption))

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        self.image_size = image_size
        print(f"  COCO-CN Dataset: {len(self.samples)} image-caption pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return img_tensor, caption


# ── Collate ────────────────────────────────────────────────────────

class CocoCnCollate:
    def __init__(self, tokenizer, max_text_len=64):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]

        encoded = self.tokenizer(captions, padding=True, truncation=True,
                                 max_length=self.max_text_len, return_tensors=True)
        return {
            'images': images,
            'text_ids': encoded['input_ids'],
            'attn_mask': encoded['attention_mask'],
        }


# ── Training ───────────────────────────────────────────────────────

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="COCO-CN Chinese Fine-tuning")
    parser.add_argument("--resume", default="../checkpoints_phase5_v2/best.pt")
    parser.add_argument("--coco-dir", default="../coco_data")
    parser.add_argument("--captions-file", default="../coco_data/coco-cn-master/data/coco-cn_ext.icap2020.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output-dir", default="../checkpoints_phase5_v3")
    parser.add_argument("--log-dir", default="../logs_phase5_v3")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = SimpleTokenizer(max_vocab=10000, add_chinese=True)
    new_vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {new_vocab_size} tokens")

    # Model
    # Model
    cfg = resolve_config(args.resume, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)

    # Load checkpoint (handles vocab/dim/layer changes automatically)
    if os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        load_checkpoint_adaptive(model, args.resume, device)
    else:
        print(f"  WARNING: checkpoint not found: {args.resume}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total/1e6:.2f}M")

    # Data
    print("\nBuilding COCO-CN dataset...")
    ds = CocoCnDataset(args.coco_dir, args.captions_file,
                       image_size=DefaultConfig.image_size, max_samples=args.max_samples)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    print(f"  Train: {n_train}, Val: {n_val}")

    collate = CocoCnCollate(tokenizer, args.max_text_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)
    print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}")

    # Optimizer — gentle fine-tuning, lower LR for real data
    embed_params, body_params, decoder_params = [], [], []
    for name, param in model.named_parameters():
        if 'text_embed' in name or 'lm_head' in name:
            embed_params.append(param)
        elif 'decoder' in name or 'img_' in name or 'audio_' in name or 'video_' in name:
            decoder_params.append(param)
        else:
            body_params.append(param)

    lr = args.lr
    optimizer = AdamW([
        {'params': embed_params, 'lr': lr * 3.0},
        {'params': decoder_params, 'lr': lr * 1.5},
        {'params': body_params, 'lr': lr},
    ], weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"  LR: body={lr:.1e}, decoder={lr*1.5:.1e}, embed={lr*3:.1e}")

    # Output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    # Training
    print(f"\n{'='*60}")
    print(f"Phase 5 v3 — COCO-CN Real Chinese Fine-tuning: {args.epochs} epochs")
    print(f"  Real images: {n_train}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*60}\n")

    metrics = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            images = batch['images'].to(device)
            text_ids = batch['text_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)

            out = model(text_ids, images=images, return_img=True)
            loss = compute_text_loss(out['text_logits'], text_ids, attn_mask)
            if 'img_recon' in out and out['target_img'] is not None:
                img_loss = F.mse_loss(out['img_recon'], out['target_img'])
                loss = loss + 0.3 * img_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                continue
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                text_ids = batch['text_ids'].to(device)
                attn_mask = batch['attn_mask'].to(device)

                out = model(text_ids, images=images)
                loss = compute_text_loss(
                    out if isinstance(out, torch.Tensor) else out['text_logits'],
                    text_ids, attn_mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        record = {
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr()[0],
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        metrics.append(record)

        # Save
        with open(log_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss

        ckpt_path = output_dir / f'epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, ckpt_path)

        status = "★" if is_best else ""
        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} {status}")

        if is_best:
            import shutil
            shutil.copy(ckpt_path, output_dir / 'best.pt')

    print(f"\nCOCO-CN fine-tuning complete! Best val_loss: {best_loss:.4f}")
    print(f"  Checkpoints: {output_dir}/")
    print(f"  Logs: {log_dir}/")


if __name__ == '__main__':
    main()

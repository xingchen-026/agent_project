#!/usr/bin/env python3
"""
Unified Training: CLIP contrastive + Language Modeling + Diffusion Decoding.
Single forward → three aligned objectives. Data: COCO 25K captions.

Usage:
  python train_joint.py --resume ../checkpoints_phase6/best.pt --epochs 15
"""

import os, sys, json, math, argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import ModelConfig, resolve_config
from utils import load_checkpoint_adaptive


# ── Dataset ────────────────────────────────────────────────────────

class CocoCaptionDataset(Dataset):
    def __init__(self, coco_dir, ann_file, image_size=224, max_images=None):
        from pycocotools.coco import COCO
        self.img_dir = Path(coco_dir) / 'val2017'
        coco = COCO(str(ann_file))
        img_ids = sorted(coco.imgs.keys())
        if max_images:
            random.seed(42)
            img_ids = random.sample(img_ids, min(max_images, len(img_ids)))

        self.samples = []
        self._cache = {}
        print(f"  Loading {len(img_ids)} COCO images...")
        for i, img_id in enumerate(img_ids):
            info = coco.imgs[img_id]
            path = self.img_dir / info['file_name']
            if not path.exists():
                continue
            # Pre-cache image
            img = Image.open(path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
            self._cache[str(path)] = torch.from_numpy(np.array(img)).permute(2,0,1).float()/127.5-1.0

            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            for ann in anns[:3]:  # up to 3 captions per image
                cap = ann['caption'].strip()
                if cap:
                    self.samples.append((str(path), cap))
            if (i+1) % 500 == 0:
                print(f"    {i+1}/{len(img_ids)}")
        print(f"  Dataset: {len(self.samples)} image-caption pairs ({len(img_ids)} images)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        return self._cache[path], caption


# ── Collate ────────────────────────────────────────────────────────

class JointCollate:
    def __init__(self, tokenizer, max_len=48):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        bos_id = 2  # <bos>
        text_ids = [torch.tensor([bos_id] + self.tok.encode(c)[:self.max_len-1])
                    for c in captions]
        padded = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True,
                                                  padding_value=self.tok.pad_token_id)
        return images, padded, [len(t) for t in text_ids]


# ── Losses ──────────────────────────────────────────────────────────

def clip_contrastive_loss(img_emb, text_emb, temperature=0.07):
    """Symmetric InfoNCE."""
    img_emb = F.normalize(img_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = (text_emb @ img_emb.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def lm_loss_fn(logits, targets, lengths):
    """Masked cross-entropy for language modeling."""
    B, T, V = logits.shape
    loss = 0.0
    for b in range(B):
        l = lengths[b] - 1
        if l <= 0:
            continue
        loss += F.cross_entropy(logits[b, :l], targets[b, 1:l+1])
    return loss / B


def diffusion_loss_fn(diffusion_decoder, patches, memory_hidden):
    """DDPM noise prediction loss."""
    loss, _ = diffusion_decoder(patches, memory_hidden)
    return loss


# ── Training ───────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Unified CLIP+LM+Diff Training")
    parser.add_argument("--resume", default="../checkpoints_phase6/best.pt")
    parser.add_argument("--coco-dir", default="../coco_data")
    parser.add_argument("--ann-file", default="../coco_data/captions_val2017.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--clip-weight", type=float, default=1.0)
    parser.add_argument("--lm-weight", type=float, default=0.5)
    parser.add_argument("--diff-weight", type=float, default=0.1)
    parser.add_argument("--output-dir", default="../checkpoints_phase6_joint")
    parser.add_argument("--log-dir", default="../logs_phase6_joint")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer(max_vocab=10000)

    # Model with all three heads
    cfg_dict = {
        'img_generation': True, 'use_audio': False, 'use_video': False,
        'use_memory_bank': True, 'n_mem_tokens': 16,
        'use_contrastive': True, 'contrastive_dim': 256,
        'use_diffusion_decoder': True,
    }
    if os.path.exists(args.resume):
        cfg = resolve_config(args.resume, tokenizer, defaults=cfg_dict)
    else:
        cfg = ModelConfig(vocab_size=tokenizer.vocab_size, **cfg_dict)

    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.2f}M ({cfg.describe()})")
    print(f"  MemoryBank: {cfg.use_memory_bank} ({cfg.n_mem_tokens} tokens)")
    print(f"  Contrastive: {cfg.use_contrastive} ({cfg.contrastive_dim}d)")
    print(f"  Diffusion: {cfg.use_diffusion_decoder}")

    if os.path.exists(args.resume):
        load_checkpoint_adaptive(model, args.resume, device)
        print(f"Loaded: {args.resume}")

    # Data
    ds = CocoCaptionDataset(args.coco_dir, args.ann_file,
                            image_size=cfg.image_size, max_images=args.max_images)
    n_val = max(32, int(len(ds) * 0.05))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    collate = JointCollate(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    print(f"  Train: {n_train} pairs ({len(train_loader)} batches)")
    print(f"  Val: {n_val} pairs")

    # Optimizer
    new_modules = ['contrastive_proj', 'diffusion_decoder', 'memory_bank']
    new_params = [p for n, p in model.named_parameters()
                  if any(m in n for m in new_modules)]
    other_params = [p for n, p in model.named_parameters()
                    if not any(m in n for m in new_modules)]
    optimizer = AdamW([
        {'params': new_params, 'lr': args.lr * 5},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=0.1, betas=(0.9, 0.98))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(200, total_steps // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    print(f"  LR: new={args.lr*5:.1e} body={args.lr:.1e}, warmup={warmup_steps}")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\n{'='*60}")
    print(f"Unified Training: {args.epochs} epochs")
    print(f"  L_clip={args.clip_weight}  L_lm={args.lm_weight}  L_diff={args.diff_weight}")
    print(f"{'='*60}\n")

    metrics = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {'clip': 0, 'lm': 0, 'diff': 0, 'total': 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, text_ids, lengths in pbar:
            images = images.to(device)
            text_ids = text_ids.to(device)

            # Forward
            out = model(text_ids, images=images, return_memory_hidden=True)
            text_logits = out['text_logits']
            mem_hidden = out.get('memory_hidden')

            # Target patches for diffusion
            patches = model._image_to_patches(images)

            # 1. CLIP loss: align image (memory) with text (text hidden)
            if cfg.use_contrastive and mem_hidden is not None:
                text_hidden = out.get('text_hidden')
                img_emb = model.contrastive_proj(mem_hidden.mean(dim=1))
                if text_hidden is not None and text_hidden.shape[1] > 0:
                    text_emb = model.contrastive_proj(text_hidden.mean(dim=1))
                else:
                    text_emb = img_emb  # fallback
                loss_clip = clip_contrastive_loss(img_emb, text_emb) * args.clip_weight
            else:
                loss_clip = torch.tensor(0.0, device=device)

            # 2. LM loss
            loss_lm = lm_loss_fn(text_logits, text_ids, lengths) * args.lm_weight

            # 3. Diffusion loss
            if cfg.use_diffusion_decoder and mem_hidden is not None:
                loss_diff = diffusion_loss_fn(model.img_decoder, patches, mem_hidden)
                loss_diff = loss_diff * args.diff_weight
            else:
                loss_diff = torch.tensor(0.0, device=device)

            loss = loss_clip + loss_lm + loss_diff

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses['clip'] += loss_clip.item()
            epoch_losses['lm'] += loss_lm.item()
            epoch_losses['diff'] += loss_diff.item()
            epoch_losses['total'] += loss.item()

            pbar.set_postfix({
                'clip': f'{loss_clip.item():.3f}',
                'lm': f'{loss_lm.item():.3f}',
                'diff': f'{loss_diff.item():.3f}',
            })

        n = len(train_loader)
        avg = {k: v/n for k, v in epoch_losses.items()}

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, text_ids, lengths in val_loader:
                images = images.to(device)
                text_ids = text_ids.to(device)
                out = model(text_ids, images=images)
                logits = out if isinstance(out, torch.Tensor) else out['text_logits']
                val_total += lm_loss_fn(logits, text_ids, lengths).item()
        avg_val = val_total / len(val_loader)

        record = {'epoch': epoch+1, 'lr': scheduler.get_last_lr()[0],
                  'train_clip': avg['clip'], 'train_lm': avg['lm'],
                  'train_diff': avg['diff'], 'train_total': avg['total'],
                  'val_loss': avg_val}
        metrics.append(record)

        with open(log_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        is_best = avg_val < best_loss
        if is_best:
            best_loss = avg_val

        ckpt_path = output_dir / f'epoch_{epoch+1}.pt'
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': cfg.to_dict(), 'best_loss': best_loss}, ckpt_path)

        status = "[BEST]" if is_best else ""
        print(f"  Epoch {epoch+1}: clip={avg['clip']:.4f} lm={avg['lm']:.4f} "
              f"diff={avg['diff']:.4f} total={avg['total']:.4f} val={avg_val:.4f} {status}")

        if is_best:
            import shutil
            shutil.copy(ckpt_path, output_dir / 'best.pt')

    print(f"\nJoint training complete! Best val_loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()

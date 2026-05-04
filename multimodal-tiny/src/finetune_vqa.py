#!/usr/bin/env python3
"""
Phase C — Multimodal Instruction Tuning (Chinese VQA).
Converts COCO-CN captions into instruction-following Q&A pairs.

Usage:
  PYTHONIOENCODING=utf-8 python finetune_vqa.py \
    --resume ../checkpoints_phase6_cn/best.pt --epochs 5
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


# ── VQA Templates ──────────────────────────────────────────────────

VQA_TEMPLATES = [
    ("图片里有什么？", "图片里有{caption}"),
    ("描述这张图片", "{caption}"),
    ("图中是什么场景？", "{caption}"),
    ("这张照片展示了什么？", "这张照片展示了{caption}"),
    ("请描述图中的内容", "图中{caption}"),
    ("照片里有什么物体？", "照片里有{caption}"),
    ("画面中能看到什么？", "画面中能看到{caption}"),
    ("简单描述一下这张图", "{caption}"),
    ("图里有什么人/物？", "图里有{caption}"),
    ("这张图片的内容是什么？", "图片内容是{caption}"),
]


# ── VQA Dataset ────────────────────────────────────────────────────

class VqaDataset(Dataset):
    """COCO images + templated Chinese VQA pairs."""

    def __init__(self, coco_dir, captions_file, image_size=224, max_samples=None):
        coco_dir = Path(coco_dir)
        img_dirs = {
            'val2014': coco_dir / 'val2014',
            'train2014': coco_dir / 'train2014',
        }

        self.samples = []
        with open(captions_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                img_id, caption = line.split('\t', 1)
                img_name = img_id.split('#')[0]
                for split_name, img_dir in img_dirs.items():
                    img_path = img_dir / (img_name + '.jpg')
                    if img_path.exists():
                        # For each caption, create multiple Q&A variants
                        for question_t, answer_t in VQA_TEMPLATES:
                            answer = answer_t.format(caption=caption)
                            self.samples.append((str(img_path), question_t, answer))
                        break

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        self.image_size = image_size
        print(f"  VQA Dataset: {len(self.samples)} Q&A pairs "
              f"({len(self.samples)//len(VQA_TEMPLATES)} images x {len(VQA_TEMPLATES)} templates)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, question, answer = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        # Instruction format: Q + A (model learns to answer after seeing image+question)
        text = f"问：{question}\n答：{answer}"
        return img_tensor, text


# ── Collate ────────────────────────────────────────────────────────

class VqaCollate:
    def __init__(self, tokenizer, max_text_len=64):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_text_len, return_tensors=True)
        return {
            'images': images,
            'text_ids': encoded['input_ids'],
            'attn_mask': encoded['attention_mask'],
        }


# ── Training ───────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="VQA Instruction Tuning")
    parser.add_argument("--resume", default="../checkpoints_phase6_cn/best.pt")
    parser.add_argument("--coco-dir", default="../coco_data")
    parser.add_argument("--captions-file", default="../coco_data/coco-cn-master/data/coco-cn_ext.icap2020.txt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--output-dir", default="../checkpoints_phase6_vqa")
    parser.add_argument("--log-dir", default="../logs_phase6_vqa")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer(max_vocab=10000, add_chinese=True)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # Model
    cfg = resolve_config(args.resume, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.2f}M params ({cfg.describe()})")

    if os.path.exists(args.resume):
        load_checkpoint_adaptive(model, args.resume, device)
        print(f"Loaded: {args.resume}")
    else:
        print(f"WARNING: checkpoint not found: {args.resume}")

    # Data
    print("\nBuilding VQA dataset...")
    ds = VqaDataset(args.coco_dir, args.captions_file,
                    image_size=cfg.image_size, max_samples=args.max_samples if args.max_samples > 0 else None)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    print(f"  Train: {n_train}, Val: {n_val}")

    collate = VqaCollate(tokenizer, args.max_text_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)
    print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}")

    # Optimizer
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    # Training
    print(f"\n{'='*60}")
    print(f"VQA Instruction Tuning: {args.epochs} epochs, {n_train} samples")
    print(f"  {len(VQA_TEMPLATES)} question templates, LR={args.lr}")
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

            out = model(text_ids, images=images)
            loss = compute_text_loss(
                out if isinstance(out, torch.Tensor) else out['text_logits'],
                text_ids, attn_mask)

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
        record = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0],
                  'train_loss': avg_train_loss, 'val_loss': avg_val_loss}
        metrics.append(record)

        with open(log_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss

        ckpt_path = output_dir / f'epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss,
            'model_config': cfg.to_dict(), 'arch_version': cfg.arch_version,
        }, ckpt_path)

        status = "[BEST]" if is_best else ""
        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} {status}")

        if is_best:
            import shutil
            shutil.copy(ckpt_path, output_dir / 'best.pt')

    print(f"\nVQA tuning complete! Best val_loss: {best_loss:.4f}")
    print(f"  Checkpoints: {output_dir}/")


if __name__ == '__main__':
    main()

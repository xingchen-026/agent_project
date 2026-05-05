#!/usr/bin/env python3
"""
Knowledge Distillation: CLIP-ViT → TinyMultimodal MemoryBank.
Uses CLIP as frozen teacher to teach real-world visual concepts.

Usage:
  python train_distill.py --resume ../checkpoints_phase6/best.pt --epochs 10
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


# ── Image Dataset ──────────────────────────────────────────────────

class ImageOnlyDataset(Dataset):
    """COCO images only (no captions needed)."""
    def __init__(self, img_dir, image_size=224, max_images=None):
        paths = sorted(Path(img_dir).glob('*.jpg'))
        if max_images:
            paths = paths[:max_images]
        self.paths = paths
        self.image_size = image_size

        # Pre-cache
        self._cache = {}
        print(f"  Loading {len(paths)} images...")
        for i, p in enumerate(paths):
            img = Image.open(p).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
            self._cache[str(p)] = torch.from_numpy(np.array(img)).permute(2,0,1).float()/127.5-1.0
            if (i+1) % 500 == 0:
                print(f"    {i+1}/{len(paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        return self._cache[path]


# ── Training ───────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="CLIP Knowledge Distillation")
    parser.add_argument("--resume", default="../checkpoints_phase6/best.pt")
    parser.add_argument("--coco-dir", default="../coco_data/val2017")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--teacher-model", default="ViT-B-32")
    parser.add_argument("--output-dir", default="../checkpoints_phase6_distill")
    parser.add_argument("--log-dir", default="../logs_phase6_distill")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Teacher: ResNet50 (torchvision, cached locally) ──
    print("Loading teacher: ResNet50 (ImageNet pretrained)...")
    import torchvision.models as models
    teacher = models.resnet50(weights='IMAGENET1K_V2').to(device)
    teacher.fc = torch.nn.Identity()  # Remove classifier, use 2048-dim features
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_dim = 2048
    print(f"  Teacher: ResNet50, output dim: {teacher_dim}")

    # ── Student: TinyMultimodal ──
    tokenizer = SimpleTokenizer(max_vocab=10000)
    cfg_dict = {'img_generation': True, 'use_audio': True, 'use_video': True,
                'use_memory_bank': True, 'n_mem_tokens': 16}
    if os.path.exists(args.resume):
        cfg = resolve_config(args.resume, tokenizer, defaults=cfg_dict)
    else:
        cfg = ModelConfig(vocab_size=tokenizer.vocab_size, **cfg_dict)

    # Add distillation projection head
    class DistillHead(torch.nn.Module):
        def __init__(self, student_dim, teacher_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(student_dim, student_dim),
                torch.nn.GELU(),
                torch.nn.Linear(student_dim, teacher_dim),
            )
        def forward(self, x):
            return F.normalize(self.net(x), dim=-1)

    model = TinyMultimodal(cfg).to(device)
    distill_head = DistillHead(cfg.dim, teacher_dim).to(device)
    print(f"Student: {sum(p.numel() for p in model.parameters())/1e6:.2f}M ({cfg.describe()})")

    if os.path.exists(args.resume):
        load_checkpoint_adaptive(model, args.resume, device)
        print(f"Loaded: {args.resume}")

    # ── Data ──
    print(f"\nLoading COCO images from {args.coco_dir}...")
    ds = ImageOnlyDataset(args.coco_dir, image_size=cfg.image_size,
                          max_images=args.max_images)
    n_val = max(32, int(len(ds) * 0.05))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ── Optimizer ──
    distill_params = list(distill_head.parameters())
    mem_params = [p for n, p in model.named_parameters() if 'memory_bank' in n]
    other_params = [p for n, p in model.named_parameters()
                    if 'memory_bank' not in n and not any(p is dp for dp in distill_params)]

    optimizer = AdamW([
        {'params': distill_params, 'lr': args.lr * 5},    # New head trains fast
        {'params': mem_params, 'lr': args.lr * 3},         # Memory bank adapts
        {'params': other_params, 'lr': args.lr * 0.5},     # Body barely moves
    ], weight_decay=0.1, betas=(0.9, 0.98))
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"  LR: distill={args.lr*5:.1e} mem={args.lr*3:.1e} body={args.lr*0.5:.1e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Loss: Cosine + MSE ──
    def distill_loss(student_emb, teacher_emb):
        cosine = 1 - F.cosine_similarity(student_emb, teacher_emb).mean()
        mse = F.mse_loss(student_emb, teacher_emb)
        return cosine + 0.5 * mse

    # ── Training ──
    print(f"\n{'='*60}")
    print(f"CLIP Distillation: {args.epochs} epochs, {n_train} images")
    print(f"{'='*60}\n")

    metrics = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        distill_head.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images in pbar:
            images = images.to(device)

            # Teacher: ResNet50 features (convert [-1,1] → ImageNet normalization)
            with torch.no_grad():
                imgs_imagenet = (images + 1) / 2  # [-1,1] → [0,1]
                imgs_imagenet = (imgs_imagenet - torch.tensor([0.485, 0.456, 0.406],
                    device=device).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225],
                    device=device).view(1,3,1,1)
                teacher_emb = F.normalize(teacher(imgs_imagenet), dim=-1)

            # Student: encode through memory bank
            B, C, H, W = images.shape
            n_img = model.get_num_image_tokens()
            patches = model._image_to_patches(images)
            img_tokens = model.img_norm(model.img_proj(patches))
            mem_tokens = model.memory_bank(img_tokens)  # [B, n_mem, dim]
            student_hidden = mem_tokens.mean(dim=1)       # [B, dim]
            student_emb = distill_head(student_hidden)     # [B, teacher_dim]

            loss = distill_loss(student_emb, teacher_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(distill_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        distill_head.eval()
        val_loss = 0.0
        val_cosine = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                imgs_imagenet = (images + 1) / 2
                imgs_imagenet = (imgs_imagenet - torch.tensor([0.485, 0.456, 0.406],
                    device=device).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225],
                    device=device).view(1,3,1,1)
                teacher_emb = F.normalize(teacher(imgs_imagenet), dim=-1)

                B = images.shape[0]
                patches = model._image_to_patches(images)
                img_tokens = model.img_norm(model.img_proj(patches))
                mem_tokens = model.memory_bank(img_tokens)
                student_emb = distill_head(mem_tokens.mean(dim=1))

                val_loss += distill_loss(student_emb, teacher_emb).item()
                val_cosine += F.cosine_similarity(student_emb, teacher_emb).mean().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cosine = val_cosine / len(val_loader)

        record = {'epoch': epoch+1, 'lr': scheduler.get_last_lr()[0],
                  'train_loss': avg_loss, 'val_loss': avg_val_loss,
                  'val_cosine_sim': avg_val_cosine}
        metrics.append(record)

        with open(log_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss

        ckpt_path = output_dir / f'epoch_{epoch+1}.pt'
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'distill_head': distill_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': cfg.to_dict(), 'best_loss': best_loss},
                   ckpt_path)

        status = "[BEST]" if is_best else ""
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} val={avg_val_loss:.4f} "
              f"cosine={avg_val_cosine:.4f} {status}")

        if is_best:
            import shutil
            shutil.copy(ckpt_path, output_dir / 'best.pt')

    print(f"\nDistillation complete! Best val_loss: {best_loss:.4f}")
    print(f"  Best cosine similarity: {1 - best_loss:.4f}")


if __name__ == '__main__':
    main()

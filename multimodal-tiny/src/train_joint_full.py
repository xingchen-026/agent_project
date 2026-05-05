#!/usr/bin/env python3
"""Full Multi-Modal Joint Training: Image + Audio + Video.
Independent modality loaders, round-robin, gradient accumulation per modality.
Focus: LM cross-entropy + reconstruction (MSE) for all modalities.
"""

import os, sys, json, math, argparse, random, shutil
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
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
from synthetic_data import SyntheticDataset
from audio_synthetic import AudioDataset
from video_synthetic import VideoDataset


# ── COCO Dataset ────────────────────────────────────────────────────

class CocoCacheDataset(Dataset):
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
            img = Image.open(path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
            self._cache[str(path)] = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id))[:2]:
                cap = ann['caption'].strip()
                if cap:
                    self.samples.append((str(path), cap))
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(img_ids)}")
        print(f"  COCO: {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cap = self.samples[idx]
        return self._cache[path], cap


# ── Collate ────────────────────────────────────────────────────────

def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return imgs, captions


# ── Helpers ─────────────────────────────────────────────────────────

def encode_captions(tokenizer, captions, max_len=48, device='cuda'):
    bos = 2
    ids = [[bos] + tokenizer.encode(c)[:max_len - 1] for c in captions]
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i) for i in ids], batch_first=True, padding_value=0
    ).to(device)
    return padded, [len(i) for i in ids]


def lm_loss(logits, targets, lengths):
    B = logits.shape[0]
    loss = 0.0
    for b in range(B):
        l = lengths[b] - 1
        if l > 0:
            loss += F.cross_entropy(logits[b, :l], targets[b, 1:l + 1])
    return loss / B


# ── Training ───────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Full Multi-Modal Joint Training")
    p.add_argument("--resume", default="../checkpoints_phase6/best.pt")
    p.add_argument("--coco-dir", default="../coco_data")
    p.add_argument("--ann-file", default="../coco_data/captions_val2017.json")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--output-dir", default="../checkpoints_phase6_full")
    p.add_argument("--log-dir", default="../logs_phase6_full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-memory-bank", action="store_true",
                   help="Enable MemoryBank (disabled for position-based reconstruction)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = SimpleTokenizer(max_vocab=10000)

    # MemoryBank OFF by default — reconstruction heads need per-modality hidden states
    use_audio = not args.no_audio
    use_video = not args.no_video
    cfg = resolve_config(args.resume, tok, defaults={
        'img_generation': True,
        'use_audio': use_audio,
        'use_video': use_video,
        'use_memory_bank': args.use_memory_bank,
        'n_mem_tokens': 16,
        'use_contrastive': False,
        'use_diffusion_decoder': False,
    })
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total / 1e6:.2f}M ({cfg.describe()})")
    print(f"  Audio: {cfg.use_audio}  Video: {cfg.use_video}  MemoryBank: {cfg.use_memory_bank}")
    load_checkpoint_adaptive(model, args.resume, device)

    # ── Datasets ──
    print("\nBuilding datasets...")
    img_ds = CocoCacheDataset(args.coco_dir, args.ann_file,
                              image_size=cfg.image_size, max_images=args.max_images)

    def split_ds(ds, val_frac=0.05):
        n = len(ds)
        nv = max(4, int(n * val_frac))
        idx = list(range(n))
        random.shuffle(idx)
        return Subset(ds, idx[nv:]), Subset(ds, idx[:nv])

    img_train, img_val = split_ds(img_ds)
    img_loader = DataLoader(img_train, args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=0, drop_last=True)
    print(f"  Image: {len(img_train)} train / {len(img_val)} val")

    if cfg.use_audio:
        aud_ds = AudioDataset(num_samples=1000, seed=args.seed)
        aud_train, aud_val = split_ds(aud_ds)
        aud_loader = DataLoader(aud_train, args.batch_size, shuffle=True,
                                collate_fn=collate_fn, num_workers=0, drop_last=True)
        print(f"  Audio: {len(aud_train)} train / {len(aud_val)} val")

    if cfg.use_video:
        vid_ds = VideoDataset(num_samples=800, seed=args.seed)
        vid_train, vid_val = split_ds(vid_ds)
        vid_loader = DataLoader(vid_train, args.batch_size, shuffle=True,
                                collate_fn=collate_fn, num_workers=0, drop_last=True)
        print(f"  Video: {len(vid_train)} train / {len(vid_val)} val")

    # ── Optimizer ──
    new_param_names = ['memory_bank', 'audio_proj', 'audio_decoder',
                       'video_proj', 'video_decoder', 'img_decoder']
    new_params = [p for n, p in model.named_parameters()
                  if any(m in n for m in new_param_names)]
    other_params = [p for n, p in model.named_parameters()
                    if not any(m in n for m in new_param_names)]
    optimizer = AdamW([
        {'params': new_params, 'lr': args.lr * 5},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=0.1, betas=(0.9, 0.98))

    n_batches = len(img_loader)
    total_steps = n_batches * args.epochs
    warmup_steps = min(100, total_steps // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    print(f"  LR: new={args.lr * 5:.1e} body={args.lr:.1e}, warmup={warmup_steps}, steps={total_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Weights ──
    W = {'img_lm': 0.5, 'img_rec': 0.3, 'aud_lm': 0.3, 'aud_rec': 0.1,
         'vid_lm': 0.2, 'vid_rec': 0.1}

    print(f"\n{'=' * 60}")
    print(f"Full Multi-Modal Joint Training: {args.epochs} epochs")
    print(f"  Modalities: image{' + audio' if cfg.use_audio else ''}{' + video' if cfg.use_video else ''}")
    print(f"  Image: LM×{W['img_lm']} + Recon×{W['img_rec']}")
    if cfg.use_audio:
        print(f"  Audio: LM×{W['aud_lm']} + Recon×{W['aud_rec']}")
    if cfg.use_video:
        print(f"  Video: LM×{W['vid_lm']} + Recon×{W['vid_rec']}")
    print(f"{'=' * 60}\n")

    metrics = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = {'img': 0.0, 'aud': 0.0, 'vid': 0.0, 'total': 0.0}
        n = 0

        aud_it = iter(aud_loader) if cfg.use_audio else None
        vid_it = iter(vid_loader) if cfg.use_video else None
        pbar = tqdm(img_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, captions in pbar:
            images = images.to(device)
            text_ids, lengths = encode_captions(tok, captions, device=device)

            # ── Image forward + backward ──
            out = model(text_ids, images=images, return_img=True)
            loss = lm_loss(out['text_logits'], text_ids, lengths) * W['img_lm']
            if 'img_recon' in out:
                loss = loss + F.mse_loss(out['img_recon'],
                                         model._image_to_patches(images)) * W['img_rec']
            loss.backward()
            epoch_loss['img'] += loss.item()

            # ── Audio forward + backward ──
            if cfg.use_audio:
                try:
                    aud, aud_caps = next(aud_it)
                except StopIteration:
                    aud_it = iter(aud_loader)
                    aud, aud_caps = next(aud_it)
                aud = aud.to(device)
                aud_text_ids, aud_len = encode_captions(tok, aud_caps, device=device)
                aud_out = model(aud_text_ids, audios=aud, return_audio=True)
                aud_loss = lm_loss(aud_out['text_logits'], aud_text_ids, aud_len) * W['aud_lm']
                if 'aud_recon' in aud_out and aud_out.get('target_aud') is not None:
                    aud_loss = aud_loss + F.mse_loss(aud_out['aud_recon'],
                                                     aud_out['target_aud']) * W['aud_rec']
                aud_loss.backward()
                epoch_loss['aud'] += aud_loss.item()
            else:
                aud_loss = torch.tensor(0.0, device=device)

            # ── Video forward + backward ──
            if cfg.use_video:
                try:
                    vid, vid_caps = next(vid_it)
                except StopIteration:
                    vid_it = iter(vid_loader)
                    vid, vid_caps = next(vid_it)
                vid = vid.to(device)
                vid_text_ids, vid_len = encode_captions(tok, vid_caps, device=device)
                vid_out = model(vid_text_ids, videos=vid, return_video=True)
                vid_loss = lm_loss(vid_out['text_logits'], vid_text_ids, vid_len) * W['vid_lm']
                if 'vid_recon' in vid_out and vid_out.get('target_vid') is not None:
                    vid_loss = vid_loss + F.mse_loss(vid_out['vid_recon'],
                                                     vid_out['target_vid']) * W['vid_rec']
                vid_loss.backward()
                epoch_loss['vid'] += vid_loss.item()
            else:
                vid_loss = torch.tensor(0.0, device=device)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_step_loss = loss.item() + (aud_loss.item() if cfg.use_audio else 0) + \
                              (vid_loss.item() if cfg.use_video else 0)
            epoch_loss['total'] += total_step_loss
            n += 1

            postfix = {'loss': f'{total_step_loss:.3f}'}
            if cfg.use_audio:
                postfix['aud'] = f'{aud_loss.item():.3f}'
            if cfg.use_video:
                postfix['vid'] = f'{vid_loss.item():.3f}'
            pbar.set_postfix(postfix)

        avg_loss = {k: v / max(n, 1) for k, v in epoch_loss.items()}

        # ── Validation (image + audio + video) ──
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            # Image validation
            for images, captions in img_val:
                images = images.unsqueeze(0).to(device)
                text_ids, lengths = encode_captions(tok, [captions], device=device)
                out = model(text_ids, images=images)
                logits = out if isinstance(out, torch.Tensor) else out['text_logits']
                val_loss += lm_loss(logits, text_ids, lengths).item()
                val_n += 1

            # Audio validation (subset)
            if cfg.use_audio:
                for aud, aud_caps in aud_val:
                    aud = aud.unsqueeze(0).to(device)
                    aud_text_ids, aud_len = encode_captions(tok, [aud_caps], device=device)
                    aud_out = model(aud_text_ids, audios=aud)
                    alogits = aud_out if isinstance(aud_out, torch.Tensor) else aud_out['text_logits']
                    val_loss += lm_loss(alogits, aud_text_ids, aud_len).item()
                    val_n += 1

            # Video validation (subset)
            if cfg.use_video:
                for vid, vid_caps in vid_val:
                    vid = vid.unsqueeze(0).to(device)
                    vid_text_ids, vid_len = encode_captions(tok, [vid_caps], device=device)
                    vid_out = model(vid_text_ids, videos=vid)
                    vlogits = vid_out if isinstance(vid_out, torch.Tensor) else vid_out['text_logits']
                    val_loss += lm_loss(vlogits, vid_text_ids, vid_len).item()
                    val_n += 1

        avg_val = val_loss / max(val_n, 1)
        record = {
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr()[0],
            'train_img': avg_loss['img'],
            'train_aud': avg_loss['aud'],
            'train_vid': avg_loss['vid'],
            'train_total': avg_loss['total'],
            'val_loss': avg_val,
        }
        metrics.append(record)
        with open(log_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        is_best = avg_val < best_loss
        if is_best:
            best_loss = avg_val

        ckpt_path = output_dir / f'epoch_{epoch + 1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': cfg.to_dict(),
            'best_loss': best_loss,
        }, ckpt_path)

        parts = [f"img={avg_loss['img']:.4f}"]
        if cfg.use_audio:
            parts.append(f"aud={avg_loss['aud']:.4f}")
        if cfg.use_video:
            parts.append(f"vid={avg_loss['vid']:.4f}")
        parts.append(f"total={avg_loss['total']:.4f}")
        parts.append(f"val={avg_val:.4f}")
        status = " [BEST]" if is_best else ""
        print(f"  Epoch {epoch + 1}: " + "  ".join(parts) + status)

        if is_best:
            shutil.copy(ckpt_path, output_dir / 'best.pt')

    print(f"\nFull multi-modal complete! Best val: {best_loss:.4f}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()

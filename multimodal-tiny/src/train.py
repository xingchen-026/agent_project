#!/usr/bin/env python3
"""
Training script — v3.0 (Phase 3: Audio Modality)
=================================================
Supports Phase 1 (text), Phase 2 (text+image), Phase 3 (text+image+audio).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from tqdm import tqdm

from model import TinyMultimodal, ModelConfig, patches_to_image, mel_patches_to_spectrogram
from data import build_loaders
from synthetic_data import SyntheticDataset
from audio_synthetic import AudioDataset, get_audio_tokenizer_description

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

    # Phase 2: Image generation
    parser.add_argument("--img_gen", action='store_true', default=True)
    parser.add_argument("--img_loss_weight", type=float, default=1.0)
    parser.add_argument("--img_decoder_hidden", type=int, default=512)

    # Phase 3: Audio
    parser.add_argument("--use_audio", action='store_true',
                        help="Enable audio modality training")
    parser.add_argument("--aud_loss_weight", type=float, default=1.0)
    parser.add_argument("--aud_train_size", type=int, default=5000)
    parser.add_argument("--aud_val_size", type=int, default=200)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def compute_text_loss(logits, text_ids, attn_mask):
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = text_ids[:, 1:].reshape(-1)
    shift_mask = attn_mask[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
    return (loss * shift_mask).sum() / shift_mask.sum()


def compute_mse_loss(pred, target):
    if pred is None or target is None:
        return torch.tensor(0.0, device=pred.device if pred is not None else 'cpu')
    return F.mse_loss(pred, target)


def load_checkpoint_flexible(model, checkpoint_path, device='cpu'):
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    model_dict = model.state_dict()
    loaded = 0
    for key in state_dict:
        if key in model_dict and state_dict[key].shape == model_dict[key].shape:
            model_dict[key] = state_dict[key]
            loaded += 1

    model.load_state_dict(model_dict, strict=True)
    print(f"  ✓ Loaded {loaded}/{len(model_dict)} keys")

    info = {}
    if 'epoch' in ckpt:
        info = {'epoch': ckpt['epoch'], 'best_loss': ckpt.get('best_loss', float('inf'))}
    return info or {'best_loss': float('inf')}


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    phase = "Phase 3 (text+img+audio)" if args.use_audio else \
            "Phase 2 (text+img gen)" if args.img_gen else \
            "Phase 1 (text only)"
    print(f"Mode: {phase}")

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
        use_audio=args.use_audio,
    )

    # ── Model ──
    print(f"Building TinyMultimodal v3.0: {cfg.n_layers} layers, {cfg.dim} dim")
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")
    if total > 30_000_000:
        print(f"  ⚠️  Over 30M budget!")

    # ── Resume ──
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        info = load_checkpoint_flexible(model, args.resume, device)
        start_epoch = info.get('epoch', -1) + 1
        best_loss = info.get('best_loss', float('inf'))
        print(f"  Resumed at epoch {start_epoch}")

    # ── Data ──
    print("Building data loaders...")
    use_img = args.img_gen or not args.use_audio

    def image_collate(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        enc = tokenizer(list(captions), padding='max_length', truncation=True,
                        max_length=args.max_text_len, return_tensors='pt')
        return {'images': images, 'text_ids': enc['input_ids'],
                'attn_mask': enc['attention_mask']}

    if args.use_synthetic:
        print("  Using synthetic data (no download needed)")
        train_ds_img = SyntheticDataset(
            num_samples=args.train_size, image_size=args.image_size)
        val_ds_img = SyntheticDataset(
            num_samples=args.val_size, image_size=args.image_size, seed=99)

        train_loader_img = torch.utils.data.DataLoader(
            train_ds_img, batch_size=args.batch_size,
            shuffle=True, collate_fn=image_collate, num_workers=0)
        val_loader_img = torch.utils.data.DataLoader(
            val_ds_img, batch_size=args.batch_size,
            shuffle=False, collate_fn=image_collate, num_workers=0)

        train_loader = train_loader_img
        val_loader = val_loader_img
    else:
        data_config = {"train_size": args.train_size, "val_size": args.val_size,
                       "batch_size": args.batch_size, "image_size": args.image_size,
                       "max_text_len": args.max_text_len}
        train_loader, val_loader = build_loaders(tokenizer, data_config)

    # Audio data
    if args.use_audio:
        def audio_collate(batch):
            mels, captions = zip(*batch)
            mels = torch.stack(mels)
            enc = tokenizer(list(captions), padding='max_length', truncation=True,
                            max_length=args.max_text_len, return_tensors='pt')
            return {'audios': mels, 'text_ids': enc['input_ids'],
                    'attn_mask': enc['attention_mask']}

        train_ds_aud = AudioDataset(num_samples=args.aud_train_size)
        val_ds_aud = AudioDataset(num_samples=args.aud_val_size, seed=99)
        train_loader_aud = torch.utils.data.DataLoader(
            train_ds_aud, batch_size=args.batch_size,
            shuffle=True, collate_fn=audio_collate, num_workers=0)
        val_loader_aud = torch.utils.data.DataLoader(
            val_ds_aud, batch_size=args.batch_size,
            shuffle=False, collate_fn=audio_collate, num_workers=0)

    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(500, total_steps // 10)
    warmup_sch = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup_steps])

    # ── Output dirs ──
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
    print(f"{phase}: {args.epochs} epochs, {args.train_size} samples")
    print(f"  Steps/epoch: {len(train_loader)}, Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    global_step = 0
    metrics = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            text_ids = batch['text_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            images = batch.get('images', None)
            if images is not None:
                images = images.to(device)
            audios = batch.get('audios', None)
            if audios is not None:
                audios = audios.to(device)

            # Forward
            return_img = images is not None and args.img_gen
            return_audio = audios is not None and args.use_audio

            out = model(text_ids, images=images, audios=audios,
                        return_img=return_img, return_audio=return_audio)

            text_logits = out if isinstance(out, torch.Tensor) else out['text_logits']
            loss = compute_text_loss(text_logits, text_ids, attn_mask)
            epoch_losses['text'] = epoch_losses.get('text', 0) + loss.item()

            loss_str = f"txt={loss.item():.4f}"

            if return_img and 'img_recon' in out:
                loss_img = compute_mse_loss(out['img_recon'], out['target_img'])
                loss = loss + args.img_loss_weight * loss_img
                epoch_losses['img'] = epoch_losses.get('img', 0) + loss_img.item()
                loss_str += f" img={loss_img.item():.4f}"

            if return_audio and 'aud_recon' in out:
                loss_aud = compute_mse_loss(out['aud_recon'], out['target_aud'])
                loss = loss + args.aud_loss_weight * loss_aud
                epoch_losses['aud'] = epoch_losses.get('aud', 0) + loss_aud.item()
                loss_str += f" aud={loss_aud.item():.4f}"

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            lr_val = scheduler.get_last_lr()[0]
            pbar.set_postfix_str(f"{loss_str} lr={lr_val:.2e}")

        # ── Validation ──
        model.eval()
        val_losses = {}
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(device)
                attn_mask = batch['attn_mask'].to(device)
                images = batch.get('images', None)
                if images is not None: images = images.to(device)
                audios = batch.get('audios', None)
                if audios is not None: audios = audios.to(device)

                return_img = images is not None and args.img_gen
                return_audio = audios is not None and args.use_audio

                out = model(text_ids, images=images, audios=audios,
                            return_img=return_img, return_audio=return_audio)
                text_logits = out if isinstance(out, torch.Tensor) else out['text_logits']
                vl = compute_text_loss(text_logits, text_ids, attn_mask).item()
                val_losses['text'] = val_losses.get('text', 0) + vl

                if return_img and 'img_recon' in out:
                    val_losses['img'] = val_losses.get('img', 0) + \
                        compute_mse_loss(out['img_recon'], out['target_img']).item()
                if return_audio and 'aud_recon' in out:
                    val_losses['aud'] = val_losses.get('aud', 0) + \
                        compute_mse_loss(out['aud_recon'], out['target_aud']).item()

        # Average
        for k in val_losses:
            val_losses[k] /= len(val_loader)

        result_str = f"  Epoch {epoch+1}: "
        for k in ['text', 'img', 'aud']:
            if k in epoch_losses:
                epoch_losses[k] /= len(train_loader)
                result_str += f"{k}={epoch_losses[k]:.4f}({val_losses.get(k,0):.4f}) "

        print(result_str)

        # ── Sample generation ──
        if (epoch + 1) % args.save_every == 0 and not args.use_audio:
            if len(train_loader.dataset) > 0:
                sample_img, sample_caption = train_loader.dataset[0]
                sample_img_t = sample_img.unsqueeze(0).to(device)
                gen_caption = model.generate_text(sample_img_t, tokenizer, max_len=30,
                                                   temperature=0.8)
                print(f"  GT: {sample_caption[:60]}")
                print(f"  Gen: {gen_caption[:60]}")

        # ── Combined validation metric ──
        combined = val_losses.get('text', 0) + \
                   args.img_loss_weight * val_losses.get('img', 0) + \
                   args.aud_loss_weight * val_losses.get('aud', 0)

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_every == 0 or combined < best_loss:
            is_best = combined < best_loss
            if is_best:
                best_loss = combined

            ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
            save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'best_loss': best_loss, 'phase': 3 if args.use_audio else 2}
            for k in ['text', 'img', 'aud']:
                if k in val_losses:
                    save_dict[f'val_{k}_loss'] = val_losses[k]
                if k in epoch_losses:
                    save_dict[f'train_{k}_loss'] = epoch_losses[k]
            torch.save(save_dict, ckpt_path)

            if is_best:
                best_path = output_dir / "best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  \u2713 New best (combined: {combined:.4f})")

        # ── Log ──
        log_entry = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0]}
        for k in ['text', 'img', 'aud']:
            if k in epoch_losses:
                log_entry[f'train_{k}_loss'] = epoch_losses[k]
            if k in val_losses:
                log_entry[f'val_{k}_loss'] = val_losses[k]
        metrics.append(log_entry)
        with open(log_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()

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
from utils import compute_text_loss, compute_mse_loss, load_checkpoint_flexible, \
    make_collate, interleave_loaders, logger
from data import build_loaders
from synthetic_data import SyntheticDataset
from audio_synthetic import AudioDataset
from video_synthetic import VideoDataset

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
    parser.add_argument("--n_heads", type=int, default=None,
                        help="Number of attention heads (default: dim//64)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.json (overrides individual args)")
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

    # Phase 4: Video
    parser.add_argument("--use_video", action='store_true',
                        help="Enable video modality training")
    parser.add_argument("--vid_loss_weight", type=float, default=0.5)
    parser.add_argument("--vid_train_size", type=int, default=3000)
    parser.add_argument("--vid_val_size", type=int, default=100)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    phase = "Phase 4 (text+img+audio+video)" if args.use_video else \
            "Phase 3 (text+img+audio)" if args.use_audio else \
            "Phase 2 (text+img gen)" if args.img_gen else \
            "Phase 1 (text only)"
    print(f"Mode: {phase}")

    # ── Tokenizer ──
    from tokenizer import SimpleTokenizer
    print("Building local tokenizer...")
    tokenizer = SimpleTokenizer(max_vocab=10000)

    # ── Config ──
    if args.config:
        from config import ModelConfig as MC
        cfg = MC.from_json(args.config)
        print(f"Loaded config from {args.config}: {cfg.describe()}")
    else:
        n_heads = args.n_heads if args.n_heads else args.dim // 64
        cfg = ModelConfig(
            dim=args.dim,
            n_layers=args.layers,
            n_heads=n_heads,
            image_size=args.image_size,
            patch_size=args.patch_size,
            vocab_size=tokenizer.vocab_size,
            img_generation=args.img_gen,
            img_decoder_hidden=args.img_decoder_hidden,
            use_audio=args.use_audio,
            use_video=args.use_video,
        )

    # ── Model ──
    print(f"Building TinyMultimodal: {cfg.describe()}")
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")
    if total > 30_000_000:
        print(f"  Over 30M budget — adjust dim/layers")

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

    img_collate = make_collate(tokenizer, args.max_text_len, 'image')

    if args.use_synthetic:
        print("  Using synthetic data (no download needed)")
        train_ds_img = SyntheticDataset(
            num_samples=args.train_size, image_size=args.image_size)
        val_ds_img = SyntheticDataset(
            num_samples=args.val_size, image_size=args.image_size, seed=99)

        train_loader_img = torch.utils.data.DataLoader(
            train_ds_img, batch_size=args.batch_size,
            shuffle=True, collate_fn=img_collate, num_workers=0)
        val_loader_img = torch.utils.data.DataLoader(
            val_ds_img, batch_size=args.batch_size,
            shuffle=False, collate_fn=img_collate, num_workers=0)

        train_loader = train_loader_img
        val_loader = val_loader_img
    else:
        data_config = {"train_size": args.train_size, "val_size": args.val_size,
                       "batch_size": args.batch_size, "image_size": args.image_size,
                       "max_text_len": args.max_text_len}
        train_loader, val_loader = build_loaders(tokenizer, data_config)

    # ── Video data ──
    if args.use_video:
        vid_collate = make_collate(tokenizer, args.max_text_len, 'video')
        train_ds_vid = VideoDataset(num_samples=args.vid_train_size)
        val_ds_vid = VideoDataset(num_samples=args.vid_val_size, seed=99)
        train_loader_vid = torch.utils.data.DataLoader(
            train_ds_vid, batch_size=args.batch_size,
            shuffle=True, collate_fn=vid_collate, num_workers=0)
        val_loader_vid = torch.utils.data.DataLoader(
            val_ds_vid, batch_size=args.batch_size,
            shuffle=False, collate_fn=vid_collate, num_workers=0)

    # Audio data
    if args.use_audio:
        aud_collate = make_collate(tokenizer, args.max_text_len, 'audio')
        train_ds_aud = AudioDataset(num_samples=args.aud_train_size)
        val_ds_aud = AudioDataset(num_samples=args.aud_val_size, seed=99)
        train_loader_aud = torch.utils.data.DataLoader(
            train_ds_aud, batch_size=args.batch_size,
            shuffle=True, collate_fn=aud_collate, num_workers=0)
        val_loader_aud = torch.utils.data.DataLoader(
            val_ds_aud, batch_size=args.batch_size,
            shuffle=False, collate_fn=aud_collate, num_workers=0)

    # ── Interleave loaders ──
    loaders_to_interleave = [train_loader_img, val_loader_img]
    loaders_val_to_interleave = [val_loader_img]
    loader_labels = ['img']

    if args.use_audio:
        loaders_to_interleave.append(train_loader_aud)
        loaders_val_to_interleave.append(val_loader_aud)
        loader_labels.append('aud')
    if args.use_video:
        loaders_to_interleave.append(train_loader_vid)
        loaders_val_to_interleave.append(val_loader_vid)
        loader_labels.append('vid')

    if len(loaders_to_interleave) > 1:
        train_loader = interleave_loaders(*loaders_to_interleave)
        val_loader = interleave_loaders(*loaders_val_to_interleave)
        sizes = [len(l) for l in loaders_to_interleave]
        print(f"  Combined: {len(train_loader)} train batches "
              f"({' + '.join(f'{s} {lbl}' for s, lbl in zip(sizes, loader_labels))})")

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
    print(f"{phase}: {args.epochs} epochs")
    if args.use_video:
        print(f"  Img samples: {args.train_size}, Aud samples: {args.aud_train_size}, Vid samples: {args.vid_train_size}")
    elif args.use_audio:
        print(f"  Img samples: {args.train_size}, Aud samples: {args.aud_train_size}")
    else:
        print(f"  Train samples: {args.train_size}")
    print(f"  Steps/epoch: {len(train_loader)}, Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    global_step = 0
    metrics = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {}
        train_counts = {'text': 0}
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
            videos = batch.get('videos', None)
            if videos is not None:
                videos = videos.to(device)

            # Forward
            return_img = images is not None and args.img_gen
            return_audio = audios is not None and args.use_audio
            return_video = videos is not None and args.use_video

            out = model(text_ids, images=images, audios=audios, videos=videos,
                        return_img=return_img, return_audio=return_audio,
                        return_video=return_video)

            text_logits = out if isinstance(out, torch.Tensor) else out['text_logits']
            loss = compute_text_loss(text_logits, text_ids, attn_mask)
            epoch_losses['text'] = epoch_losses.get('text', 0) + loss.item()
            train_counts['text'] += 1

            loss_str = f"txt={loss.item():.4f}"

            if return_img and 'img_recon' in out:
                loss_img = compute_mse_loss(out['img_recon'], out['target_img'])
                loss = loss + args.img_loss_weight * loss_img
                epoch_losses['img'] = epoch_losses.get('img', 0) + loss_img.item()
                train_counts['img'] = train_counts.get('img', 0) + 1
                loss_str += f" img={loss_img.item():.4f}"

            if return_audio and 'aud_recon' in out:
                loss_aud = compute_mse_loss(out['aud_recon'], out['target_aud'])
                loss = loss + args.aud_loss_weight * loss_aud
                epoch_losses['aud'] = epoch_losses.get('aud', 0) + loss_aud.item()
                train_counts['aud'] = train_counts.get('aud', 0) + 1
                loss_str += f" aud={loss_aud.item():.4f}"

            if return_video and 'vid_recon' in out:
                loss_vid = compute_mse_loss(out['vid_recon'], out['target_vid'])
                loss = loss + args.vid_loss_weight * loss_vid
                epoch_losses['vid'] = epoch_losses.get('vid', 0) + loss_vid.item()
                train_counts['vid'] = train_counts.get('vid', 0) + 1
                loss_str += f" vid={loss_vid.item():.4f}"

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"Gradient NaN/Inf (norm={grad_norm}), skipping step")
                optimizer.zero_grad()
                continue
            optimizer.step()
            scheduler.step()

            global_step += 1
            lr_val = scheduler.get_last_lr()[0]
            pbar.set_postfix_str(f"{loss_str} lr={lr_val:.2e}")

        # ── Validation ──
        model.eval()
        val_losses = {}
        val_counts = {'text': 0}
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(device)
                attn_mask = batch['attn_mask'].to(device)
                images = batch.get('images', None)
                if images is not None: images = images.to(device)
                audios = batch.get('audios', None)
                if audios is not None: audios = audios.to(device)
                videos = batch.get('videos', None)
                if videos is not None: videos = videos.to(device)

                return_img = images is not None and args.img_gen
                return_audio = audios is not None and args.use_audio
                return_video = videos is not None and args.use_video

                out = model(text_ids, images=images, audios=audios, videos=videos,
                            return_img=return_img, return_audio=return_audio,
                            return_video=return_video)
                text_logits = out if isinstance(out, torch.Tensor) else out['text_logits']
                vl = compute_text_loss(text_logits, text_ids, attn_mask).item()
                val_losses['text'] = val_losses.get('text', 0) + vl
                val_counts['text'] += 1

                if return_img and 'img_recon' in out:
                    val_losses['img'] = val_losses.get('img', 0) + \
                        compute_mse_loss(out['img_recon'], out['target_img']).item()
                    val_counts['img'] = val_counts.get('img', 0) + 1
                if return_audio and 'aud_recon' in out:
                    val_losses['aud'] = val_losses.get('aud', 0) + \
                        compute_mse_loss(out['aud_recon'], out['target_aud']).item()
                    val_counts['aud'] = val_counts.get('aud', 0) + 1
                if return_video and 'vid_recon' in out:
                    val_losses['vid'] = val_losses.get('vid', 0) + \
                        compute_mse_loss(out['vid_recon'], out['target_vid']).item()
                    val_counts['vid'] = val_counts.get('vid', 0) + 1

        # Average (by modality count, not total batches)
        for k in val_losses:
            val_losses[k] /= max(val_counts.get(k, 1), 1)

        result_str = f"  Epoch {epoch+1}: "
        for k in ['text', 'img', 'aud', 'vid']:
            if k in epoch_losses:
                epoch_losses[k] /= max(train_counts.get(k, 1), 1)
                result_str += f"{k}={epoch_losses[k]:.4f}({val_losses.get(k,0):.4f}) "

        print(result_str)

        # ── Sample generation (from image) ──
        if (epoch + 1) % args.save_every == 0:
            if hasattr(train_loader_img, 'dataset') and len(train_loader_img.dataset) > 0:
                sample_img, sample_caption = train_loader_img.dataset[0]
                sample_img_t = sample_img.unsqueeze(0).to(device)
                gen_caption = model.generate_text(sample_img_t, tokenizer, max_len=30,
                                                   temperature=0.8)
                print(f"  Img→Txt GT: {sample_caption[:60]}")
                print(f"  Img→Txt Gen: {gen_caption[:60]}")

        # ── Combined validation metric ──
        combined = val_losses.get('text', 0) + \
                   args.img_loss_weight * val_losses.get('img', 0) + \
                   args.aud_loss_weight * val_losses.get('aud', 0) + \
                   args.vid_loss_weight * val_losses.get('vid', 0)

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_every == 0 or combined < best_loss:
            is_best = combined < best_loss
            if is_best:
                best_loss = combined

            ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
            save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'best_loss': best_loss, 'phase': 3 if args.use_audio else 2,
                         'model_config': cfg.to_dict(),
                         'arch_version': cfg.arch_version}
            for k in ['text', 'img', 'aud', 'vid']:
                if k in val_losses:
                    save_dict[f'val_{k}_loss'] = val_losses[k]
                if k in epoch_losses:
                    save_dict[f'train_{k}_loss'] = epoch_losses[k]
            torch.save(save_dict, ckpt_path)

            if is_best:
                best_path = output_dir / "best.pt"
                torch.save({'model_state_dict': model.state_dict(),
                            'model_config': cfg.to_dict(),
                            'arch_version': cfg.arch_version,
                            'epoch': epoch, 'best_loss': best_loss}, best_path)
                print(f"  [OK] New best (combined: {combined:.4f})")

        # ── Log ──
        log_entry = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0]}
        for k in ['text', 'img', 'aud', 'vid']:
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

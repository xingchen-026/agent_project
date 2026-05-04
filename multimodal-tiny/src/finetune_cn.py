#!/usr/bin/env python3
"""
Phase 5 — Chinese Language Fine-tuning
Adds Chinese tokens to vocabulary, resizes model embeddings, trains on Chinese captions.
"""

import os, sys, json, math, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config
from utils import compute_text_loss, compute_mse_loss, make_collate, interleave_loaders, logger, load_checkpoint_adaptive


def get_args():
    parser = argparse.ArgumentParser(description="Phase 5 — Chinese Fine-tuning")
    parser.add_argument("--resume", type=str, default="./checkpoints_phase5/best.pt",
                        help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_warmup", action="store_true",
                        help="Skip warmup (use for continued fine-tuning)")
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--aud_train_size", type=int, default=2000)
    parser.add_argument("--aud_val_size", type=int, default=100)
    parser.add_argument("--vid_train_size", type=int, default=2000)
    parser.add_argument("--vid_val_size", type=int, default=100)
    parser.add_argument("--max_text_len", type=int, default=48)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_phase5_v2")
    parser.add_argument("--log_dir", type=str, default="./logs_phase5_v2")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Tokenizer (with Chinese) ──
    tokenizer = SimpleTokenizer(max_vocab=10000, add_chinese=True)
    new_vocab_size = tokenizer.vocab_size
    # old_vocab_size will be read from checkpoint embedding shape
    print(f"Tokenizer: new vocab = {new_vocab_size} (with Chinese ngrams)")
    
    # ── Model ──
    cfg = resolve_config(args.resume, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)

    # ── Load checkpoint (handles vocab/dim/layer changes automatically) ──
    if os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        load_checkpoint_adaptive(model, args.resume, device)
    else:
        print(f"  ⚠️  Checkpoint not found: {args.resume}, training from scratch")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")
    
    # ── Chinese Data ──
    print("Building Chinese data loaders...")
    from cn_data import ZhImageDataset, ZhAudioDataset, ZhVideoDataset
    
    img_collate = make_collate(tokenizer, args.max_text_len, 'image')
    aud_collate = make_collate(tokenizer, args.max_text_len, 'audio')
    vid_collate = make_collate(tokenizer, args.max_text_len, 'video')

    train_ds_img = ZhImageDataset(num_samples=args.train_size)
    val_ds_img = ZhImageDataset(num_samples=args.val_size, seed=99)
    train_ds_aud = ZhAudioDataset(num_samples=args.aud_train_size)
    val_ds_aud = ZhAudioDataset(num_samples=args.aud_val_size, seed=99)
    train_ds_vid = ZhVideoDataset(num_samples=args.vid_train_size)
    val_ds_vid = ZhVideoDataset(num_samples=args.vid_val_size, seed=99)
    
    train_loader_img = torch.utils.data.DataLoader(
        train_ds_img, batch_size=args.batch_size,
        shuffle=True, collate_fn=img_collate, num_workers=0)
    train_loader_aud = torch.utils.data.DataLoader(
        train_ds_aud, batch_size=args.batch_size,
        shuffle=True, collate_fn=aud_collate, num_workers=0)
    train_loader_vid = torch.utils.data.DataLoader(
        train_ds_vid, batch_size=args.batch_size,
        shuffle=True, collate_fn=vid_collate, num_workers=0)
    
    val_loader_img = torch.utils.data.DataLoader(
        val_ds_img, batch_size=args.batch_size,
        shuffle=False, collate_fn=img_collate, num_workers=0)
    val_loader_aud = torch.utils.data.DataLoader(
        val_ds_aud, batch_size=args.batch_size,
        shuffle=False, collate_fn=aud_collate, num_workers=0)
    val_loader_vid = torch.utils.data.DataLoader(
        val_ds_vid, batch_size=args.batch_size,
        shuffle=False, collate_fn=vid_collate, num_workers=0)
    
    # Interleave: img:aud:vid round-robin
    loaders = [train_loader_img, train_loader_aud, train_loader_vid]
    labels = ['img', 'aud', 'vid']
    train_loader = interleave_loaders(*loaders)
    val_loader = interleave_loaders(val_loader_img, val_loader_aud, val_loader_vid)
    
    sizes = [len(l) for l in loaders]
    print(f"  Train: {' + '.join(f'{s} {lbl}' for s, lbl in zip(sizes, labels))} "
          f"= {len(train_loader)} batches")
    
    # ── Optimizer (differential LR: faster for new embeddings, slower for transformer) ──
    embed_params = []
    body_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'text_embed' in name or 'lm_head' in name:
            embed_params.append(param)
        elif 'decoder' in name or 'img_' in name or 'audio_' in name or 'video_' in name:
            decoder_params.append(param)
        else:
            body_params.append(param)

    param_groups = [
        {'params': embed_params, 'lr': args.lr * 5.0},     # 5e-4 — fast adapt for new tokens
        {'params': decoder_params, 'lr': args.lr * 2.0},   # 2e-4 — moderate
        {'params': body_params, 'lr': args.lr},             # 1e-4 — slow for transformer body
    ]
    optimizer = AdamW(param_groups, weight_decay=0.05, betas=(0.9, 0.95))
    print(f"  LR: embed={args.lr*5:.1e}, decoder={args.lr*2:.1e}, body={args.lr:.1e}")
    total_steps = len(train_loader) * args.epochs
    if args.no_warmup:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        print(f"  LR schedule: CosineAnnealing (no warmup)")
    else:
        warmup_steps = min(200, total_steps // 10)
        warmup_sch = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_sch = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup_steps])
        print(f"  LR schedule: warmup={warmup_steps} + cosine")
    
    # ── Output ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config_dict['model_params_m'] = total / 1e6
    with open(log_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # ── Training ──
    print(f"\n{'='*60}")
    print(f"Phase 5 — Chinese Fine-tuning: {args.epochs} epochs")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Steps/epoch: {len(train_loader)}")
    print(f"{'='*60}\n")
    
    metrics = []
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {}
        train_counts = {'text': 0, 'img': 0, 'aud': 0, 'vid': 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            text_ids = batch['text_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            images = batch.get('images', None)
            audios = batch.get('audios', None)
            videos = batch.get('videos', None)

            if images is not None:
                images = images.to(device)
            if audios is not None:
                audios = audios.to(device)
            if videos is not None:
                videos = videos.to(device)

            return_img = images is not None
            return_audio = audios is not None
            return_video = videos is not None

            out = model(text_ids, images=images, audios=audios, videos=videos,
                        return_img=return_img, return_audio=return_audio,
                        return_video=return_video)

            text_logits = out if isinstance(out, torch.Tensor) else out['text_logits']
            loss = compute_text_loss(text_logits, text_ids, attn_mask)
            epoch_losses['text'] = epoch_losses.get('text', 0) + loss.item()
            train_counts['text'] += 1

            # Reconstruction losses
            if return_img and 'img_recon' in out:
                loss_img = compute_mse_loss(out['img_recon'], out['target_img'])
                loss = loss + 0.5 * loss_img
                epoch_losses['img'] = epoch_losses.get('img', 0) + loss_img.item()
                train_counts['img'] += 1

            if return_audio and 'aud_recon' in out:
                loss_aud = compute_mse_loss(out['aud_recon'], out['target_aud'])
                loss = loss + 0.5 * loss_aud
                epoch_losses['aud'] = epoch_losses.get('aud', 0) + loss_aud.item()
                train_counts['aud'] += 1

            if return_video and 'vid_recon' in out:
                loss_vid = compute_mse_loss(out['vid_recon'], out['target_vid'])
                loss = loss + 0.5 * loss_vid
                epoch_losses['vid'] = epoch_losses.get('vid', 0) + loss_vid.item()
                train_counts['vid'] += 1

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"Gradient NaN/Inf (norm={grad_norm}), skipping step")
                optimizer.zero_grad()
                continue
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({'txt': f'{loss.item():.4f}'})
        
        # ── Validation ──
        model.eval()
        val_losses = {'text': 0, 'img': 0, 'aud': 0, 'vid': 0}
        val_counts = {'text': 0, 'img': 0, 'aud': 0, 'vid': 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Validating"):
                text_ids = batch['text_ids'].to(device)
                attn_mask = batch['attn_mask'].to(device)
                images = batch.get('images', None)
                audios = batch.get('audios', None)
                videos = batch.get('videos', None)
                
                if images is not None:
                    images = images.to(device)
                if audios is not None:
                    audios = audios.to(device)
                if videos is not None:
                    videos = videos.to(device)
                
                return_img = images is not None
                return_audio = audios is not None
                return_video = videos is not None
                
                out = model(text_ids, images=images, audios=audios, videos=videos,
                            return_img=return_img, return_audio=return_audio,
                            return_video=return_video)
                
                text_logits = out if isinstance(out, torch.Tensor) else out['text_logits']
                loss = compute_text_loss(text_logits, text_ids, attn_mask)
                val_losses['text'] += loss.item()
                val_counts['text'] += 1
                
                if return_img and 'img_recon' in out:
                    loss_img = compute_mse_loss(out['img_recon'], out['target_img'])
                    val_losses['img'] += loss_img.item()
                    val_counts['img'] += 1
                if return_audio and 'aud_recon' in out:
                    loss_aud = compute_mse_loss(out['aud_recon'], out['target_aud'])
                    val_losses['aud'] += loss_aud.item()
                    val_counts['aud'] += 1
                if return_video and 'vid_recon' in out:
                    loss_vid = compute_mse_loss(out['vid_recon'], out['target_vid'])
                    val_losses['vid'] += loss_vid.item()
                    val_counts['vid'] += 1
        
        # Log
        record = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0]}
        for key in ['text', 'img', 'aud', 'vid']:
            if train_counts.get(key, 0) > 0:
                record[f'train_{key}_loss'] = epoch_losses.get(key, 0) / train_counts[key]
            if val_counts.get(key, 0) > 0:
                record[f'val_{key}_loss'] = val_losses[key] / val_counts[key]
        
        combined_val = sum(record.get(f'val_{k}_loss', 0) for k in ['text', 'img', 'aud', 'vid'])
        combined_train = sum(record.get(f'train_{k}_loss', 0) for k in ['text', 'img', 'aud', 'vid'])
        record['combined_val'] = combined_val
        record['combined_train'] = combined_train
        metrics.append(record)
        
        # Save
        with open(log_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Checkpoint
        is_best = combined_val < best_loss
        if is_best:
            best_loss = combined_val
        
        ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, ckpt_path)
        print(f"  Saved {ckpt_path}")
        
        if is_best:
            import shutil
            shutil.copy(ckpt_path, output_dir / "best.pt")
            print(f"  ★ New best: {combined_val:.4f}")
        
        # Print summary
        val_str = " | ".join(
            f"{k}={record.get(f'val_{k}_loss', 0):.4f}" for k in ['text', 'img', 'aud', 'vid']
            if f'val_{k}_loss' in record
        )
        print(f"  Val: {val_str} | combined={combined_val:.4f}")
    
    print(f"\n✅ Phase 5 complete! Best combined val loss: {best_loss:.4f}")
    print(f"  Checkpoints: {output_dir}/")
    print(f"  Logs: {log_dir}/")


if __name__ == '__main__':
    main()

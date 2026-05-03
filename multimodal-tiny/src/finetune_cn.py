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
from src.model import TinyMultimodal, ModelConfig
from src.tokenizer import SimpleTokenizer
from src.train import compute_text_loss, compute_mse_loss


def get_args():
    parser = argparse.ArgumentParser(description="Phase 5 — Chinese Fine-tuning")
    parser.add_argument("--resume", type=str, default="./checkpoints_phase4_5/best.pt",
                        help="Phase 4.5 checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--aud_train_size", type=int, default=2000)
    parser.add_argument("--aud_val_size", type=int, default=100)
    parser.add_argument("--vid_train_size", type=int, default=2000)
    parser.add_argument("--vid_val_size", type=int, default=100)
    parser.add_argument("--max_text_len", type=int, default=48)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_phase5")
    parser.add_argument("--log_dir", type=str, default="./logs_phase5")
    return parser.parse_args()


def resize_embeddings(model, old_vocab_size, new_vocab_size):
    """
    Resize model's embedding + lm_head to accommodate new vocab size.
    Copies old weights for existing tokens, random init for new ones.
    """
    device = next(model.parameters()).device
    dim = model.cfg.dim
    
    # Get old weights
    old_embed_weight = model.text_embed.weight.data  # [old_vocab, dim]
    old_lm_weight = model.lm_head.weight.data        # [old_vocab, dim]
    
    # Create new embedding
    new_embed = torch.nn.Embedding(new_vocab_size, dim, device=device)
    
    # Copy old weights
    new_embed.weight.data[:old_vocab_size] = old_embed_weight[:old_vocab_size]
    
    # Init new tokens with small random values (scaled to match existing)
    std = old_embed_weight.std().item()
    nn.init = torch.nn.init
    nn.init.normal_(new_embed.weight.data[old_vocab_size:], mean=0.0, std=std * 0.3)
    
    # Replace model's embedding and lm_head
    model.text_embed = new_embed
    model.lm_head = torch.nn.Linear(dim, new_vocab_size, bias=False, device=device)
    model.lm_head.weight.data[:old_vocab_size] = old_lm_weight[:old_vocab_size]
    nn.init.normal_(model.lm_head.weight.data[old_vocab_size:], mean=0.0, std=std * 0.3)
    
    # Re-tie weights
    model.text_embed.weight = model.lm_head.weight
    
    # Update config
    model.cfg.vocab_size = new_vocab_size
    
    print(f"  Embedding resized: {old_vocab_size} → {new_vocab_size}")
    return model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Tokenizer (with Chinese) ──
    tokenizer = SimpleTokenizer(max_vocab=10000, add_chinese=True)
    old_vocab_size = 1171  # Phase 4/4.5 vocab without Chinese
    new_vocab_size = tokenizer.vocab_size  # ~1337
    print(f"Tokenizer: {old_vocab_size} → {new_vocab_size} tokens")
    
    # ── Model ──
    cfg = ModelConfig(
        dim=384, n_layers=6,
        image_size=224, patch_size=32,
        vocab_size=new_vocab_size,
        img_generation=True, img_decoder_hidden=512,
        use_audio=True, use_video=True,
    )
    model = TinyMultimodal(cfg).to(device)
    
    # ── Load checkpoint and resize ──
    if os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        # Load what we can (will fail on embedding/lm_head size mismatch)
        # Build a temp model with old vocab to load transformer weights
        temp_cfg = ModelConfig(
            dim=384, n_layers=6, image_size=224, patch_size=32,
            vocab_size=old_vocab_size,
            img_generation=True, img_decoder_hidden=512,
            use_audio=True, use_video=True,
        )
        from src.model import TinyMultimodal as TinyMultimodalOld
        temp_model = TinyMultimodalOld(temp_cfg).to(device)
        
        missing, unexpected = temp_model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded from checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
        
        # Now copy all non-embedding weights from temp to real model
        # (temp has smaller vocab, so only copy layers that match)
        model_dict = model.state_dict()
        for key, val in temp_model.state_dict().items():
            if key in model_dict and val.shape == model_dict[key].shape:
                model_dict[key] = val
            elif key in model_dict:
                # Shape mismatch - likely embedding/lm_head
                if 'embed' in key or 'lm_head' in key:
                    # We'll handle this below
                    pass
        
        model.load_state_dict(model_dict, strict=False)
        
        # Now resize embeddings
        old_embed = temp_model.text_embed.weight.data
        old_lm = temp_model.lm_head.weight.data
        
        # Copy old weights to new model
        new_embed_data = model.text_embed.weight.data
        new_lm_data = model.lm_head.weight.data
        
        new_embed_data[:old_vocab_size] = old_embed[:old_vocab_size]
        new_lm_data[:old_vocab_size] = old_lm[:old_vocab_size]
        
        # Init new Chinese tokens
        std = old_embed.std().item()
        torch.nn.init.normal_(new_embed_data[old_vocab_size:], mean=0.0, std=std * 0.3)
        torch.nn.init.normal_(new_lm_data[old_vocab_size:], mean=0.0, std=std * 0.3)
        
        # Re-tie
        model.text_embed.weight = model.lm_head.weight
        
        print(f"  ✓ Embeddings resized, Chinese tokens initialized")
        del temp_model
    else:
        print(f"  ⚠️  Checkpoint not found: {args.resume}, training from scratch")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")
    
    # ── Chinese Data ──
    print("Building Chinese data loaders...")
    from src.cn_data import ZhImageDataset, ZhAudioDataset, ZhVideoDataset
    
    def img_collate(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        enc = tokenizer(list(captions), padding='max_length', truncation=True,
                        max_length=args.max_text_len, return_tensors='pt')
        return {'images': images, 'text_ids': enc['input_ids'],
                'attn_mask': enc['attention_mask']}
    
    def aud_collate(batch):
        mels, captions = zip(*batch)
        mels = torch.stack(mels)
        enc = tokenizer(list(captions), padding='max_length', truncation=True,
                        max_length=args.max_text_len, return_tensors='pt')
        return {'audios': mels, 'text_ids': enc['input_ids'],
                'attn_mask': enc['attention_mask']}
    
    def vid_collate(batch):
        videos, captions = zip(*batch)
        videos = torch.stack(videos)
        enc = tokenizer(list(captions), padding='max_length', truncation=True,
                        max_length=args.max_text_len, return_tensors='pt')
        return {'videos': videos, 'text_ids': enc['input_ids'],
                'attn_mask': enc['attention_mask']}
    
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
    
    # Interleave: img:aud:vid = 5:2:2 ratio
    loaders = [train_loader_img, train_loader_aud, train_loader_vid]
    labels = ['img', 'aud', 'vid']
    
    its = [iter(l) for l in loaders]
    done = [False] * len(its)
    train_loader = []
    while not all(done):
        for i, it in enumerate(its):
            if not done[i]:
                try:
                    train_loader.append(next(it))
                except StopIteration:
                    done[i] = True
    
    val_its = [iter(val_loader_img), iter(val_loader_aud), iter(val_loader_vid)]
    val_done = [False] * 3
    val_loader = []
    while not all(val_done):
        for i, it in enumerate(val_its):
            if not val_done[i]:
                try:
                    val_loader.append(next(it))
                except StopIteration:
                    val_done[i] = True
    
    sizes = [len(l) for l in loaders]
    print(f"  Train: {' + '.join(f'{s} {lbl}' for s, lbl in zip(sizes, labels))} "
          f"= {len(train_loader)} batches")
    
    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(200, total_steps // 10)
    warmup_sch = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup_steps])
    
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
            
            # Reconstruction losses
            if return_img and 'img_recon' in out:
                loss_img = compute_mse_loss(out['img_recon'], out['target_img'])
                loss = loss + 0.5 * loss_img
                epoch_losses['img'] = epoch_losses.get('img', 0) + loss_img.item()
            
            if return_audio and 'aud_recon' in out:
                loss_aud = compute_mse_loss(out['aud_recon'], out['target_aud'])
                loss = loss + 0.5 * loss_aud
                epoch_losses['aud'] = epoch_losses.get('aud', 0) + loss_aud.item()
            
            if return_video and 'vid_recon' in out:
                loss_vid = compute_mse_loss(out['vid_recon'], out['target_vid'])
                loss = loss + 0.5 * loss_vid
                epoch_losses['vid'] = epoch_losses.get('vid', 0) + loss_vid.item()
            
            epoch_losses['text'] = epoch_losses.get('text', 0) + loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            if val_counts[key] > 0:
                record[f'train_{key}_loss'] = epoch_losses.get(key, 0) / max(val_counts[key], 1)
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

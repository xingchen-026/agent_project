#!/usr/bin/env python3
"""
MVP: Tiny Unified Multimodal Transformer
========================================
Proof-of-concept for a small (<50M param) native multimodal model that
processes text + image tokens in the same autoregressive sequence.

Architecture:
  - 4-layer Transformer with RoPE, 384 dim, 6 heads
  - Text: GPT-2 tokenizer (BPE, 50K vocab)
  - Image: ViT-style patch embedding (32x32 patches from 224x224)
  - Learnable type embeddings (text vs image)
  - Causal attention for next-text-token prediction

Training:
  - COCO 2017 val subset: image → caption prediction
  - Loss: CE on text tokens only (simpler for MVP)
  - CPU-friendly, batch_size depends on available RAM

Usage:
  python train_mvp.py --epochs 5 --batch_size 4 --data_size 2000
"""

import os
import sys
import math
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer

# ---------------------------------------------------------------------------
# 1. RoPE Positional Encoding
# ---------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """RoPE: applies rotation to query and key in attention."""

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        return torch.cos(freqs), torch.sin(freqs)

def apply_rotary(x, cos, sin):
    """x: (batch, heads, seq, head_dim), cos/sin: (seq, head_dim//2)"""
    T = x.shape[-2]
    half = x.shape[-1] // 2
    cos = cos[:T].view(1, 1, T, -1)  # (1, 1, seq, half)
    sin = sin[:T].view(1, 1, T, -1)

    x1 = x[..., :half]   # (..., half)
    x2 = x[..., half:]   # (..., half)
    rotated = torch.stack(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
    ).flatten(-2)
    return rotated

# ---------------------------------------------------------------------------
# 2. Transformer Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, T, n_heads, head_dim)

        # Apply RoPE
        q = apply_rotary(q.transpose(1, 2), cos, sin)  # (B, n_heads, T, head_dim)
        k = apply_rotary(k.transpose(1, 2), cos, sin)

        out = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = dim * hidden_mult
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x

# ---------------------------------------------------------------------------
# 3. Unified Multimodal Transformer
# ---------------------------------------------------------------------------

class UnifiedMultimodalTransformer(nn.Module):
    """
    Processes text tokens + image patch tokens in one sequence.

    Input format: [IMG_PATCH_0..IMG_PATCH_48][BOS] caption tokens [EOS]
    Training target: predict caption tokens (cross-entropy on text tokens only)
    """

    def __init__(
        self,
        vocab_size=50257,
        dim=384,
        n_layers=4,
        n_heads=6,
        max_seq_len=1024,
        patch_size=32,
        image_size=224,
        num_image_tokens=49,
        rope_base=10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_image_tokens = num_image_tokens
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.text_embed = nn.Embedding(vocab_size, dim)
        # Image patch projection (ViT-style)
        patch_dim = 3 * patch_size * patch_size  # RGB
        self.img_proj = nn.Linear(patch_dim, dim, bias=True)
        self.img_norm = RMSNorm(dim)

        # Learnable type embeddings
        self.type_embed = nn.Embedding(2, dim)  # 0=text, 1=image

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)

        # Output projection
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tie weights
        self.text_embed.weight = self.lm_head.weight

        # RoPE
        self.rope = RotaryPositionalEncoding(dim // n_heads, max_seq_len, rope_base)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.02)

    def forward(self, text_ids, images, text_mask=None):
        """
        Args:
            text_ids: (B, T_text) — token IDs for text portion
            images:   (B, 3, H, W) — input images
            text_mask: (B, T_text) — padding mask for text
        Returns:
            logits: (B, total_seq, vocab_size)
        """
        B = text_ids.shape[0]
        device = text_ids.device

        # ---- Process image patches ----
        # Convert image to patches
        patches = self._image_to_patches(images)  # (B, num_patches, patch_dim)
        img_tokens = self.img_proj(patches)        # (B, num_patches, dim)
        img_tokens = self.img_norm(img_tokens)

        # Add image type embedding
        img_type = torch.full((B, self.num_image_tokens), 1, device=device, dtype=torch.long)
        img_tokens = img_tokens + self.type_embed(img_type)

        # ---- Process text tokens ----
        text_tokens = self.text_embed(text_ids)  # (B, T_text, dim)
        text_type = torch.full((text_ids.shape[1],), 0, device=device, dtype=torch.long)
        text_tokens = text_tokens + self.type_embed(text_type)

        # ---- Concatenate: [image patches] + [text tokens] ----
        x = torch.cat([img_tokens, text_tokens], dim=1)  # (B, num_patches + T_text, dim)
        total_len = x.shape[1]

        # ---- RoPE ----
        cos, sin = self.rope(total_len, device)

        # ---- Causal mask for full sequence ----
        # Image patches see each other, text tokens see all patches and previous text
        # For next-token prediction on text, use standard causal mask
        mask = torch.triu(
            torch.full((total_len, total_len), float('-inf'), device=device),
            diagonal=1
        )
        # Allow all image tokens to see each other (no causal within images)
        if self.num_image_tokens > 0:
            mask[:self.num_image_tokens, :self.num_image_tokens] = 0

        # ---- Forward through transformer ----
        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.final_norm(x)

        # ---- LM head (only text portion for efficiency) ----
        text_start = self.num_image_tokens
        text_logits = self.lm_head(x[:, text_start:, :])  # (B, T_text, vocab)

        return text_logits

    def _image_to_patches(self, images):
        """Convert (B, 3, H, W) images to (B, num_patches, patch_dim) patches."""
        B, C, H, W = images.shape
        ps = int(math.sqrt(self.num_image_tokens * C * H * W / (3 * H * W)))  # auto compute
        # Simple: reshape to patches
        p = int(math.sqrt(self.num_image_tokens))
        assert p * p == self.num_image_tokens, f"{self.num_image_tokens} must be perfect square"
        patch_h, patch_w = H // p, W // p
        patches = images.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_image_tokens, -1)
        return patches

    @torch.no_grad()
    def generate(self, image, max_new_tokens=50, temperature=1.0, top_k=50):
        """Generate caption from image greedily."""
        self.eval()
        device = next(self.parameters()).device
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # We need a starting token; use BOS token (GPT-2 uses <|endoftext|> for this)
        bos_id = 50256  # GPT-2 EOS token used as BOS
        text_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

        generated = []
        for _ in range(max_new_tokens):
            logits = self(text_ids, image)  # (1, T_text, vocab)
            next_logits = logits[0, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, len(next_logits)))
                next_logits[next_logits < values[-1]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

            token = next_id.item()
            if token == 50256:  # EOS
                break
            generated.append(token)
            text_ids = torch.cat([text_ids, next_id.unsqueeze(0)], dim=1)

            if text_ids.shape[1] > self.max_seq_len - self.num_image_tokens:
                break

        return generated

# ---------------------------------------------------------------------------
# 4. Dataset: COCO Captions
# ---------------------------------------------------------------------------

class COCOSubset(Dataset):
    """COCO validation subset for quick MVP training."""

    def __init__(self, root="./data/coco", split="val2017", max_samples=5000, image_size=224):
        self.root = Path(root)
        self.image_dir = self.root / split
        self.image_size = image_size
        self.max_samples = max_samples

        # Download/verify COCO val2017
        self._ensure_data()

        # Load annotations
        ann_file = self.root / "annotations" / "captions_val2017.json"
        with open(ann_file) as f:
            data = json.load(f)

        # Build image_id -> captions mapping
        img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        img_ids = list(img_id_to_file.keys())[:max_samples]

        self.samples = []
        for img_id in img_ids:
            caps = [ann["caption"] for ann in data["annotations"] if ann["image_id"] == img_id]
            if caps:
                # Pick first caption for simplicity
                self.samples.append((img_id_to_file[img_id], caps[0]))

        print(f"  Loaded {len(self.samples)} samples from COCO {split}")

    def _ensure_data(self):
        """Auto-download COCO val2017 if not present."""
        if (self.image_dir).exists() and len(list(self.image_dir.glob("*.jpg"))) > 100:
            return

        print("  Downloading COCO val2017 (will take a moment)...")
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "annotations").mkdir(parents=True, exist_ok=True)

        urls = [
            ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
            ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
        ]

        for url, fname in urls:
            dest = self.root / fname
            if not dest.exists():
                print(f"    Downloading {fname}...")
                import urllib.request
                urllib.request.urlretrieve(url, dest)
                print(f"    Extracting {fname}...")
                import zipfile
                with zipfile.ZipFile(dest, 'r') as zf:
                    zf.extractall(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        img_path = self.image_dir / fname

        image = Image.open(img_path).convert("RGB")
        # Resize to expected size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return img_tensor, caption


def collate_fn(batch, tokenizer, max_text_len=64):
    """Collate: images stay as tensors, texts get tokenized."""
    images, captions = zip(*batch)
    images = torch.stack(images)  # (B, 3, H, W)

    # Tokenize with padding
    enc = tokenizer(
        list(captions),
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )

    return images, enc["input_ids"], enc["attention_mask"]

# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print("Loading COCO dataset...")
    dataset = COCOSubset(
        root=args.data_dir,
        max_samples=args.data_size,
        image_size=224,
    )
    collate = lambda b: collate_fn(b, tokenizer, args.max_text_len)

    # Model
    print(f"Building UnifiedMultimodalTransformer...")
    model = UnifiedMultimodalTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=384,
        n_layers=4,
        n_heads=6,
        max_seq_len=512,
        num_image_tokens=49,  # 7x7 patches from 224x224
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.2f}M")

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.999))

    total_steps = len(dataset) // args.batch_size * args.epochs
    warmup_steps = min(500, total_steps // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    # Training loop
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, {args.data_size} samples")
    print(f"  Warmup: {warmup_steps} steps, Total: {total_steps} steps")
    print(f"  Expected time: ~{(total_steps * 0.15):.0f}s per step on CPU = {(total_steps * 0.15 / 60):.0f} min total\n")

    global_step = 0
    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        model.train()
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate, num_workers=0,
        )
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for images, text_ids, attn_mask in pbar:
            images, text_ids = images.to(device), text_ids.to(device)

            # Forward
            logits = model(text_ids, images)  # (B, T_text, vocab)

            # Loss: cross-entropy on text tokens only
            # Shift: predict text_ids[:, 1:] from logits[:, :-1, :]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = text_ids[:, 1:].contiguous()
            shift_mask = attn_mask[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
            )
            loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / epoch_steps
        print(f"  Avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'args': vars(args),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  New best model! Loss: {best_loss:.4f}")

        # Generate sample
        print(f"\n  Generating sample caption...")
        model.eval()
        with torch.no_grad():
            sample_img, _ = dataset[0]
            sample_img = sample_img.to(device)
            tokens = model.generate(sample_img, max_new_tokens=30, temperature=0.8)
            caption = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  Generated: {caption[:100]}")

    print(f"\n{'='*60}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MVP: Tiny Unified Multimodal Transformer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (CPU-friendly)")
    parser.add_argument("--data_size", type=int, default=2000, help="Number of COCO samples to use")
    parser.add_argument("--max_text_len", type=int, default=64, max_length=128)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data/coco", help="COCO data root")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Model output dir")
    args = parser.parse_args()

    train(args)

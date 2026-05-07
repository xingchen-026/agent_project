#!/usr/bin/env python3
"""
Evaluation script for Tiny Multimodal Model — v2.1
===================================================
Tests text generation, image reconstruction, and text-to-image generation.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import numpy as np

from core.model import TinyMultimodal, patches_to_image
from core.tokenizer import SimpleTokenizer
from core.config import resolve_config
from utils import load_checkpoint_adaptive
from data.synthetic import generate_sample, SyntheticDataset


def tensor_to_pil(tensor):
    """Convert [-1, 1] tensor to PIL Image."""
    img = tensor.detach().cpu().clamp(-1, 1)
    img = (img + 1) / 2 * 255
    img = img.byte().permute(1, 2, 0).numpy()
    return PILImage.fromarray(img)


@torch.no_grad()
def evaluate_text_generation(model, tokenizer, num_samples=5, max_len=40, device='cpu'):
    """Evaluate text generation from synthetic images."""
    print(f"\n{'='*60}")
    print("Text Generation from Images")
    print(f"{'='*60}")
    for i in range(num_samples):
        img, caption = generate_sample(image_size=model.cfg.image_size, seed=100 + i)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

        generated = model.generate_text(img_t.to(device), tokenizer, max_len=max_len, temperature=0.7)
        print(f"\n  [{i}]")
        print(f"    GT:      {caption}")
        print(f"    Gen:     {generated}")


@torch.no_grad()
def evaluate_image_reconstruction(model, num_samples=3, device='cpu'):
    """Reconstruct images and measure quality."""
    print(f"\n{'='*60}")
    print("Image Reconstruction Quality")
    print(f"{'='*60}")
    total_mse = 0.0
    for i in range(num_samples):
        img, caption = generate_sample(image_size=model.cfg.image_size, seed=200 + i)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

        recon = model.reconstruct_image(img_t.unsqueeze(0).to(device))
        mse = F.mse_loss(recon, img_t.unsqueeze(0).to(device)).item()
        total_mse += mse

        # Save original and reconstructed for visual comparison
        save_dir = Path("eval_output")
        save_dir.mkdir(exist_ok=True)
        tensor_to_pil(img_t).save(save_dir / f"recon_{i}_orig.png")
        tensor_to_pil(recon[0]).save(save_dir / f"recon_{i}_recon.png")
        print(f"  [{i}] MSE: {mse:.6f} — saved to eval_output/")

    print(f"  Average MSE: {total_mse/num_samples:.6f}")


@torch.no_grad()
def evaluate_text_to_image(model, tokenizer, prompts, device='cpu'):
    """Generate images from text prompts using [IMG] placeholder."""
    if not model.cfg.img_generation:
        print("Model does not have image generation head enabled.")
        return

    print(f"\n{'='*60}")
    print("Text-to-Image Generation (Experimental)")
    print(f"{'='*60}")

    for i, prompt in enumerate(prompts):
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        text_ids = torch.tensor([tokens], device=device, dtype=torch.long)

        # Generate image
        img = model.generate_image(text_ids, tokenizer)
        mse = float('nan')  # No ground truth for text-only generation

        save_dir = Path("eval_output")
        save_dir.mkdir(exist_ok=True)
        tensor_to_pil(img[0]).save(save_dir / f"t2i_{i}_generated.png")
        print(f"  [{i}] Prompt: {prompt[:60]}...")
        print(f"       Saved: eval_output/t2i_{i}_generated.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tiny Multimodal Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Model checkpoint (.pt file)")
    parser.add_argument("--test", choices=["all", "text_gen", "recon", "t2i"],
                        default="all", help="Which test to run")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ──
    tokenizer = SimpleTokenizer(max_vocab=10000)

    # ── Config (auto-detected from checkpoint) ──
    cfg = resolve_config(args.checkpoint, tokenizer,
        defaults={'img_generation': True, 'use_audio': False, 'use_video': False})
    model = TinyMultimodal(cfg).to(device)
    load_checkpoint_adaptive(model, args.checkpoint, device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.describe()} ({params/1e6:.2f}M)")

    # ── Run tests ──
    if args.test in ("all", "text_gen"):
        evaluate_text_generation(model, tokenizer, args.num_samples, device=device)

    if args.test in ("all", "recon"):
        evaluate_image_reconstruction(model, args.num_samples, device=device)

    if args.test in ("all", "t2i"):
        prompts = [
            "A large red circle on a black background.",
            "A small blue square on a dark blue background.",
            "An image containing a yellow star, and a green triangle on a black background.",
        ]
        evaluate_text_to_image(model, tokenizer, prompts, device=device)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

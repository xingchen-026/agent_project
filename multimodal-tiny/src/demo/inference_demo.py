#!/usr/bin/env python3
"""
TinyMultimodal Inference Demo — Interactive CLI + Comprehensive Tests
Usage:
  python3 src/inference_demo.py                   # Interactive mode
  python3 src/inference_demo.py --test-all         # Full automated tests
  python3 src/inference_demo.py --checkpoint ./checkpoints_phase4_5/best.pt
"""

import os, sys, json, math, random, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from model import TinyMultimodal, patches_to_image, mel_patches_to_spectrogram, video_patches_to_frames
from tokenizer import SimpleTokenizer
from synthetic_data import SyntheticDataset
from audio_synthetic import AudioDataset
from video_synthetic import VideoDataset
from config import resolve_config
from utils import load_checkpoint_adaptive


def get_args():
    parser = argparse.ArgumentParser(description="TinyMultimodal Inference Demo")
    parser.add_argument("--checkpoint", default="./checkpoints_phase5_v2/best.pt",
                        help="Model checkpoint")
    parser.add_argument("--output", default="./demo_output",
                        help="Output directory for figures")
    parser.add_argument("--test-all", action='store_true',
                        help="Run all automated tests")
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Samples per test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Device & Model Setup ────────────────────────────────────────────

def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer(max_vocab=10000)
    cfg = resolve_config(checkpoint_path, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    info = {}
    if os.path.exists(checkpoint_path):
        info = load_checkpoint_adaptive(model, checkpoint_path, device)
    print(f"Model: {total/1e6:.2f}M parameters ({cfg.describe()}, {'CUDA' if torch.cuda.is_available() else 'CPU'})")
    return model, tokenizer, device


# ── Utility Functions ───────────────────────────────────────────────

def tensor_to_numpy(tensor):
    arr = ((tensor.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    return arr

def compute_psnr(orig, recon, max_val=2.0):
    mse = F.mse_loss(orig, recon).item()
    if mse < 1e-10:
        return float('inf')
    return 10 * math.log10((max_val ** 2) / mse)

def print_header(text):
    w = 60
    print(f"\n{'='*w}")
    print(f"  {text}")
    print(f"{'='*w}")

def print_result(label, value, unit=""):
    print(f"  {label}: {value}{unit}")


# ── Test Suites ─────────────────────────────────────────────────────

def test_image_reconstruction(model, tokenizer, device, args):
    """Diverse image reconstruction test."""
    print_header("📷  IMAGE RECONSTRUCTION (Diverse Test)")
    output_dir = Path(args.output)
    n = args.num_samples
    
    # Test with synthetic data
    ds = SyntheticDataset(num_samples=n, image_size=224, seed=args.seed)
    psnrs = []
    
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
    
    for i in range(n):
        img, caption = ds[i]
        img = img.unsqueeze(0).to(device)
        recon = model.reconstruct_image(img)
        psnr = compute_psnr(img, recon)
        psnrs.append(psnr)
        
        orig_np = tensor_to_numpy(img[0])
        recon_np = tensor_to_numpy(recon[0])
        
        axes[0, i].imshow(orig_np)
        axes[0, i].set_title(f'Original', fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_np)
        axes[1, i].set_title(f'{psnr:.1f}dB', fontsize=8, color='green' if psnr > 22 else 'orange')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recon", fontsize=10)
    fig.suptitle(f'Image Reconstruction (Avg PSNR: {np.mean(psnrs):.2f} dB)', fontsize=13)
    plt.tight_layout()
    path = output_dir / 'test_image_recon.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Avg PSNR: {np.mean(psnrs):.2f} dB")
    print(f"  Range: [{min(psnrs):.2f}, {max(psnrs):.2f}] dB")
    print(f"  → Saved: {path}")
    return {'avg_psnr': float(np.mean(psnrs)), 'psnrs': psnrs}


def test_audio_reconstruction(model, tokenizer, device, args):
    """Audio reconstruction test."""
    print_header("🔊  AUDIO RECONSTRUCTION")
    output_dir = Path(args.output)
    n = min(args.num_samples, 4)
    
    ds = AudioDataset(num_samples=n, seed=args.seed)
    mses, snrs = [], []
    
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 8))
    
    for i in range(n):
        mel, caption = ds[i]
        mel = mel.unsqueeze(0).to(device)
        recon = model.reconstruct_audio(mel)
        
        min_t = min(mel.shape[-1], recon.shape[-1])
        mse = F.mse_loss(mel[..., :min_t], recon[..., :min_t]).item()
        snr = 10 * math.log10(mel[..., :min_t].pow(2).mean().item() / max(mse, 1e-10))
        mses.append(mse)
        snrs.append(snr)
        
        orig_np = mel[0, 0, :, :min_t].cpu().numpy()
        recon_np = recon[0, 0, :, :min_t].cpu().numpy()
        
        im0 = axes[0, i].imshow(orig_np, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Original', fontsize=8)
        axes[0, i].axis('off')
        im1 = axes[1, i].imshow(recon_np, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[1, i].set_title(f'MSE={mse:.4f}', fontsize=8)
        axes[1, i].axis('off')
        axes[2, i].plot(np.mean(np.abs(orig_np), axis=1), 'b-', alpha=0.7, label='Orig', lw=1)
        axes[2, i].plot(np.mean(np.abs(recon_np), axis=1), 'r--', alpha=0.7, label='Recon', lw=1)
        axes[2, i].set_title(f'SNR={snr:.1f}dB', fontsize=8)
        axes[2, i].legend(fontsize=5)
    
    avg_mse = np.mean(mses)
    avg_snr = np.mean(snrs)
    fig.suptitle(f'Audio Reconstruction (Avg MSE: {avg_mse:.5f}, Avg SNR: {avg_snr:.1f} dB)', fontsize=13)
    plt.tight_layout()
    path = output_dir / 'test_audio_recon.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Avg MSE: {avg_mse:.5f}  |  Avg SNR: {avg_snr:.1f} dB")
    print(f"  → Saved: {path}")
    return {'avg_mse': float(avg_mse), 'avg_snr': float(avg_snr)}


def test_video_reconstruction(model, tokenizer, device, args):
    """Video reconstruction with per-frame analysis."""
    print_header("🎬  VIDEO RECONSTRUCTION (4-Frame Analysis)")
    output_dir = Path(args.output)
    n = min(args.num_samples, 4)
    
    ds = VideoDataset(num_samples=n, seed=args.seed)
    psnrs = []
    
    fig, axes = plt.subplots(n, 9, figsize=(22, n * 2.5))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        vid, caption = ds[i]
        vid = vid.unsqueeze(0).to(device)
        recon = model.reconstruct_video(vid)
        psnr = compute_psnr(vid, recon)
        psnrs.append(psnr)
        
        # Video: [C,T,H,W] -> permute(1,0,2,3) -> [T,C,H,W] -> each frame [H,W,3]
        vid_np = ((vid[0].permute(1, 0, 2, 3).detach().cpu().numpy() + 1) / 2).clip(0, 1)
        recon_vid_np = ((recon[0].permute(1, 0, 2, 3).detach().cpu().numpy() + 1) / 2).clip(0, 1)
        orig_frames = [vid_np[t].transpose(1, 2, 0) for t in range(4)]
        recon_frames_list = [recon_vid_np[t].transpose(1, 2, 0) for t in range(4)]
        
        for t in range(4):
            axes[i, t].imshow(orig_frames[t])
            axes[i, t].set_title(f'Frame {t+1} Orig', fontsize=8)
            axes[i, t].axis('off')
            # Pixel diff
            diff = np.abs(orig_frames[t].astype(float) - recon_frames_list[t].astype(float))
            axes[i, t+4].imshow(diff.mean(axis=2), cmap='hot', vmin=0, vmax=0.5)
            axes[i, t+4].set_title(f'Error t={t+1}', fontsize=8, color='red')
            axes[i, t+4].axis('off')
        
        # Caption
        axes[i, 8].text(0.1, 0.5, f'PSNR:\n{psnr:.1f} dB\n\n{shorten(caption, 60)}',
                       transform=axes[i, 8].transAxes, fontsize=8, va='center')
        axes[i, 8].axis('off')
    
    fig.suptitle(f'Video Reconstruction — Original→Error Map (Avg PSNR: {np.mean(psnrs):.2f} dB)',
                 fontsize=14)
    plt.tight_layout()
    path = output_dir / 'test_video_recon.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Avg PSNR: {np.mean(psnrs):.2f} dB  |  Range: [{min(psnrs):.2f}, {max(psnrs):.2f}]")
    print(f"  → Saved: {path}")
    return {'avg_psnr': float(np.mean(psnrs)), 'psnrs': psnrs}


def test_text_generation(model, tokenizer, device, args):
    """Text generation from image (with optional cross-modal video/audio)."""
    print_header("📝  TEXT GENERATION FROM IMAGE")
    output_dir = Path(args.output)
    n = min(args.num_samples, 6)
    
    ds_img = SyntheticDataset(num_samples=n, image_size=224, seed=args.seed)
    results = []
    
    for i in range(n):
        img, gt_caption = ds_img[i]
        img = img.to(device)
        
        # Generate text from image only
        gen_text = model.generate_text(img, tokenizer, max_len=48, temperature=0.7, top_k=30)
        
        results.append({'type': 'image_only', 'gt': gt_caption, 'gen': gen_text})
        match = simple_match(gt_caption, gen_text)
        print(f"\n  [{i+1}] Image → Text:")
        print(f"    GT:  {gt_caption[:80]}")
        print(f"    Gen: {gen_text[:80]}")
        print(f"    Match: {match:.0f}%", end="")
    
    # Cross-modal: Image + Video → Text
    print_header("🎯  CROSS-MODAL TEST (Image + Video → Text)")
    ds_vid = VideoDataset(num_samples=min(n, 3), seed=args.seed + 100)
    for i in range(min(n, 3)):
        vid, vid_caption = ds_vid[i]
        # Get a random image for context
        img2, _ = SyntheticDataset(num_samples=1, image_size=224, seed=args.seed + 200 + i * 7)[0]
        
        vid = vid.to(device)
        img2 = img2.to(device)
        gen_text = model.generate_text(img2, tokenizer, video=vid, max_len=48, temperature=0.7, top_k=30)
        
        results.append({'type': 'image_video', 'gt': vid_caption, 'gen': gen_text})
        print(f"\n  [{i+1}] Image+Video → Text:")
        print(f"    Video GT: {vid_caption[:80]}")
        print(f"    Gen:      {gen_text[:80]}")
    
    # Cross-modal: Audio + Image → Text  
    print_header("🎯  CROSS-MODAL TEST (Image + Audio → Text)")
    ds_aud = AudioDataset(num_samples=min(n, 3), seed=args.seed + 300)
    for i in range(min(n, 3)):
        aud, aud_caption = ds_aud[i]
        img3, _ = SyntheticDataset(num_samples=1, image_size=224, seed=args.seed + 400 + i * 7)[0]
        
        aud = aud.unsqueeze(0).to(device)
        img3 = img3.to(device)
        gen_text = model.generate_text(img3, tokenizer, audio=aud, max_len=48, temperature=0.7, top_k=30)
        
        results.append({'type': 'image_audio', 'gt': aud_caption, 'gen': gen_text})
        print(f"\n  [{i+1}] Image+Audio → Text:")
        print(f"    Audio GT: {aud_caption[:80]}")
        print(f"    Gen:      {gen_text[:80]}")
    
    # Save results
    with open(output_dir / 'text_generation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved: {output_dir / 'text_generation_results.json'}")
    return results


def test_text_to_image(model, tokenizer, device, args):
    """Test text-to-image generation."""
    print_header("🎨  TEXT → IMAGE GENERATION")
    output_dir = Path(args.output)
    
    test_prompts = [
        "red circle",
        "blue square",
        "green triangle",
        "small yellow circle on black background",
        "three circles in different colors",
    ]
    
    fig, axes = plt.subplots(1, len(test_prompts), figsize=(len(test_prompts) * 3, 3))
    
    for i, prompt in enumerate(test_prompts):
        tokens = tokenizer.encode(prompt)
        bos_id = 0
        text_ids = torch.tensor([[bos_id] + tokens], dtype=torch.long, device=device)
        # Truncate to max reasonable length
        text_ids = text_ids[:, :48]
        
        try:
            gen_img = model.generate_image(text_ids, tokenizer)
            axes[i].imshow(tensor_to_numpy(gen_img[0]))
            axes[i].set_title(shorten(prompt, 25), fontsize=8)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', fontsize=7)
            axes[i].set_title(shorten(prompt, 25), fontsize=7, color='red')
        axes[i].axis('off')
    
    fig.suptitle('Text → Image Generation', fontsize=14)
    plt.tight_layout()
    path = output_dir / 'test_text_to_image.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Generated {len(test_prompts)} text→image samples")
    print(f"  → Saved: {path}")
    return {'prompts': test_prompts}


def test_edge_cases(model, tokenizer, device, args):
    """Test edge cases: empty input, extreme captions, multi-object, etc."""
    print_header("🧪  EDGE CASE TESTS")
    output_dir = Path(args.output)
    results = []
    
    # 1. Very long caption
    print("\n  1. Very long caption → Image reconstruction still works?")
    ds = SyntheticDataset(num_samples=1, image_size=224, seed=42)
    img, _ = ds[0]
    img = img.unsqueeze(0).to(device)
    long_caption = "A " + "very " * 30 + "long caption that exceeds typical max length"
    tokens = tokenizer.encode(long_caption[:200])
    gen_text = model.generate_text(img, tokenizer, max_len=100, temperature=0.5, top_k=20)
    print(f"     Long caption gen: {gen_text[:60]}...")
    results.append({'test': 'long_caption', 'gen': gen_text[:100]})
    
    # 2. No visual input (text only)
    print("\n  2. Text only (no image) → generate_text with None image")
    # This tests the model's behavior when no sensory input is provided
    try:
        fake_img = torch.zeros(1, 3, 224, 224, device=device)  # black image
        gen_text = model.generate_text(fake_img, tokenizer, max_len=20, temperature=0.8)
        print(f"     Black image → {gen_text[:60]}")
        results.append({'test': 'black_image', 'gen': gen_text[:60]})
    except Exception as e:
        print(f"     Error: {e}")
    
    # 3. Multiple objects in one image (test that description covers multiple shapes)
    print("\n  3. Multi-object caption generation")
    ds_multi = SyntheticDataset(num_samples=2, image_size=224, seed=77)
    for i in range(2):
        img2, gt = ds_multi[i]
        img2 = img2.to(device)
        gen = model.generate_text(img2, tokenizer, max_len=48, temperature=0.6, top_k=25)
        print(f"     [{i+1}] GT:  {gt[:80]}")
        print(f"         Gen: {gen[:80]}")
        results.append({'test': f'multi_object_{i+1}', 'gt': gt, 'gen': gen})
    
    # 4. Video with all-black frames
    print("\n  4. Empty/blank video frames")
    try:
        blank_vid = torch.full((1, 3, 4, 64, 64), -1.0, device=device)  # all black
        recon = model.reconstruct_video(blank_vid)
        psnr = compute_psnr(blank_vid, recon)
        print(f"     Blank video PSNR: {psnr:.2f} dB (should be high if correctly passed through)")
        results.append({'test': 'blank_video', 'psnr': psnr})
    except Exception as e:
        print(f"     Error: {e}")
    
    # Save edge case results
    with open(output_dir / 'edge_case_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved: {output_dir / 'edge_case_results.json'}")
    return results


def shorten(text, max_len=50):
    return text[:max_len] + "..." if len(text) > max_len else text


def simple_match(gt, gen):
    """Rough similarity score (0-100) based on word overlap."""
    gt_words = set(gt.lower().replace('.', '').split())
    gen_words = set(gen.lower().replace('.', '').split())
    if not gt_words:
        return 100.0
    intersection = gt_words & gen_words
    recall = len(intersection) / len(gt_words) * 100
    precision = len(intersection) / max(len(gen_words), 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1)
    return f1


# ── Interactive Mode ────────────────────────────────────────────────

def interactive_mode(model, tokenizer, device, args):
    """Interactive CLI demo."""
    import readline  # for arrow key history support
    
    print_header("🦎  TinyMultimodal Interactive Demo")
    print("  Model: 18.75M params, 4 modalities (text+image+audio+video)")
    print("  Type 'help' for commands, 'quit' to exit\n")
    
    # Pre-generate some test data
    ds_img = SyntheticDataset(num_samples=10, image_size=224, seed=42)
    ds_aud = AudioDataset(num_samples=10, seed=42)
    ds_vid = VideoDataset(num_samples=10, seed=42)
    
    while True:
        try:
            cmd = input("\n>>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not cmd:
            continue
        
        if cmd in ('q', 'quit', 'exit'):
            print("Bye!")
            break
        
        elif cmd == 'help':
            print("""
  Commands:
    help          — this help
    img [n]       — reconstruct image #n (0-9)
    aud [n]       — reconstruct audio #n (0-9)
    vid [n]       — reconstruct video #n (0-9)
    caption [n]   — generate caption from image #n
    img+vid [n]   — generate caption from image+video #n
    img+aud [n]   — generate caption from image+audio #n
    text2img      — generate image from text
    all           — run all tests on first 3 samples
    ls            — list all available test samples
    q/quit/exit   — exit
            """)
        
        elif cmd == 'ls':
            print("\n  Available test samples:")
            for i in range(5):
                _, c1 = ds_img[i]
                _, c2 = ds_aud[i]
                _, c3 = ds_vid[i]
                print(f"  [{i}] Image: {c1[:50]}")
                print(f"      Audio: {c2[:50]}")
                print(f"      Video: {c3[:50]}")
                print()
        
        elif cmd == 'all':
            print_header("Running all tests on first 3 samples...")
            for i in range(3):
                print(f"\n--- Sample {i} ---")
                run_img_demo(model, ds_img, i, device, tokenizer)
                run_aud_demo(model, ds_aud, i, device, tokenizer)
                run_vid_demo(model, ds_vid, i, device, tokenizer)
                run_caption_demo(model, ds_img, i, device, tokenizer)
        
        elif cmd.startswith('img'):
            idx = _parse_idx(cmd)
            run_img_demo(model, ds_img, idx, device, tokenizer)
        
        elif cmd.startswith('aud'):
            idx = _parse_idx(cmd)
            run_aud_demo(model, ds_aud, idx, device, tokenizer)
        
        elif cmd.startswith('vid'):
            idx = _parse_idx(cmd)
            run_vid_demo(model, ds_vid, idx, device, tokenizer)
        
        elif cmd.startswith('caption'):
            idx = _parse_idx(cmd)
            run_caption_demo(model, ds_img, idx, device, tokenizer)
        
        elif cmd.startswith('img+vid'):
            idx = _parse_idx(cmd)
            run_cross_demo(model, ds_img, ds_vid, idx, device, tokenizer, 'video')
        
        elif cmd.startswith('img+aud'):
            idx = _parse_idx(cmd)
            run_cross_demo(model, ds_img, ds_aud, idx, device, tokenizer, 'audio')
        
        elif cmd == 'text2img':
            run_text2img_demo(model, tokenizer, device)
        
        else:
            print(f"  Unknown command: {cmd} (try 'help')")


def _parse_idx(cmd):
    parts = cmd.split()
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return 0


def run_img_demo(model, ds, idx, device, tokenizer):
    img, caption = ds[idx]
    img_batch = img.unsqueeze(0).to(device)
    recon = model.reconstruct_image(img_batch)
    psnr = compute_psnr(img_batch, recon)
    print(f"  📷 Image #{idx}: {caption[:60]}")
    print(f"     PSNR: {psnr:.2f} dB")


def run_aud_demo(model, ds, idx, device, tokenizer):
    mel, caption = ds[idx]
    mel_batch = mel.unsqueeze(0).to(device)
    recon = model.reconstruct_audio(mel_batch)
    min_t = min(mel_batch.shape[-1], recon.shape[-1])
    mse = F.mse_loss(mel_batch[..., :min_t], recon[..., :min_t]).item()
    print(f"  🔊 Audio #{idx}: {caption[:60]}")
    print(f"     MSE: {mse:.5f}")


def run_vid_demo(model, ds, idx, device, tokenizer):
    vid, caption = ds[idx]
    vid_batch = vid.unsqueeze(0).to(device)
    recon = model.reconstruct_video(vid_batch)
    psnr = compute_psnr(vid_batch, recon)
    print(f"  🎬 Video #{idx}: {caption[:60]}")
    print(f"     PSNR: {psnr:.2f} dB")


def run_caption_demo(model, ds, idx, device, tokenizer):
    img, gt_caption = ds[idx]
    img = img.to(device)
    gen_text = model.generate_text(img, tokenizer, max_len=48, temperature=0.7, top_k=30)
    score = simple_match(gt_caption, gen_text)
    print(f"  📝 Caption #{idx}:")
    print(f"     GT:  {gt_caption[:80]}")
    print(f"     Gen: {gen_text[:80]}")
    print(f"     Match: {score:.0f}%")
    return gen_text


def run_cross_demo(model, ds_img, ds_other, idx, device, tokenizer, modality):
    img, _ = ds_img[idx]
    other, gt_other = ds_other[idx]
    img = img.to(device)
    if modality == 'video':
        other = other.to(device)
        gen = model.generate_text(img, tokenizer, video=other, max_len=48, temperature=0.7, top_k=30)
    else:
        other = other.unsqueeze(0).to(device)
        gen = model.generate_text(img, tokenizer, audio=other, max_len=48, temperature=0.7, top_k=30)
    print(f"  🎯 {modality.upper()} Cross-Modal #{idx}:")
    print(f"     {modality.title()} GT: {gt_other[:60]}")
    print(f"     Gen: {gen[:80]}")


def run_text2img_demo(model, tokenizer, device):
    prompt = input("  Enter text prompt: ").strip()
    if not prompt:
        prompt = "red circle"
    tokens = tokenizer.encode(prompt)
    bos_id = 0
    text_ids = torch.tensor([[bos_id] + tokens[:47]], dtype=torch.long, device=device)
    
    try:
        gen_img = model.generate_image(text_ids, tokenizer)
        # Save the generated image
        img_np = tensor_to_numpy(gen_img[0])
        output_path = Path(args.output) / f'text2img_{prompt[:20].replace(" ", "_")}.png'
        plt.imsave(output_path, img_np)
        print(f"  Generated image saved to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    global args
    args = get_args()
    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer, device = load_model(args.checkpoint)
    
    if args.test_all:
        results = {}
        results['image'] = test_image_reconstruction(model, tokenizer, device, args)
        results['audio'] = test_audio_reconstruction(model, tokenizer, device, args)
        results['video'] = test_video_reconstruction(model, tokenizer, device, args)
        results['text_gen'] = test_text_generation(model, tokenizer, device, args)
        results['text2img'] = test_text_to_image(model, tokenizer, device, args)
        results['edge_cases'] = test_edge_cases(model, tokenizer, device, args)
    
        # Summary
        print_header("📊  COMPREHENSIVE TEST SUMMARY")
        img_psnr = results['image']['avg_psnr']
        aud_mse = results['audio']['avg_mse']
        aud_snr = results['audio']['avg_snr']
        vid_psnr = results['video']['avg_psnr']
        print(f"  Image Recon PSNR : {img_psnr:.2f} dB")
        print(f"  Audio Recon MSE  : {aud_mse:.5f}  |  SNR: {aud_snr:.1f} dB")
        print(f"  Video Recon PSNR : {vid_psnr:.2f} dB")
        print(f"\n  Results saved to: {args.output}/")
        
        with open(args.output / 'comprehensive_test_results.json', 'w') as f:
            # Make results JSON-serializable
            serializable = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    serializable[k] = {kk: (vv if not isinstance(vv, (np.ndarray,)) else vv.tolist())
                                      for kk, vv in v.items()}
                else:
                    serializable[k] = str(v)
            json.dump(serializable, f, indent=2)
        print(f"  → Saved: {args.output / 'comprehensive_test_results.json'}")
    
    else:
        interactive_mode(model, tokenizer, device, args)


if __name__ == '__main__':
    main()

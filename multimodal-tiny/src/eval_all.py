#!/usr/bin/env python3
"""
Comprehensive Evaluation + Visualization for Phase 4 Multimodal Model.

Tests all 4 modalities (text, image, audio, video) and generates visualizations.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import TinyMultimodal, ModelConfig, patches_to_image, mel_patches_to_spectrogram, video_patches_to_frames
from src.tokenizer import SimpleTokenizer
from src.synthetic_data import SyntheticDataset
from src.audio_synthetic import AudioDataset
from src.video_synthetic import VideoDataset
from src.train import load_checkpoint_flexible


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Tiny Multimodal Model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_phase4/best.pt")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--img_train_size", type=int, default=500)
    parser.add_argument("--aud_train_size", type=int, default=200)
    parser.add_argument("--vid_train_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_to_numpy(tensor):
    """Convert a [-1, 1] tensor to [0, 1] numpy image (HWC)."""
    arr = ((tensor.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    return arr


def compute_psnr(original, reconstructed, max_val=2.0):
    """Compute PSNR between original and reconstructed tensors."""
    mse = F.mse_loss(original, reconstructed).item()
    if mse < 1e-10:
        return float('inf')
    psnr = 10 * math.log10((max_val ** 2) / mse)
    return psnr


def compute_snr(signal, noise):
    """Compute Signal-to-Noise Ratio (dB)."""
    signal_power = signal.pow(2).mean().item()
    noise_power = (signal - noise).pow(2).mean().item()
    if noise_power < 1e-10:
        return float('inf')
    return 10 * math.log10(signal_power / noise_power)


def save_metrics_json(results, path):
    """Save evaluation metrics to JSON."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved to {path}")


def plot_training_curves(metric_paths, save_path):
    """Plot training curves from metrics.json files."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_flat = axes.flatten()
    
    modalities = [
        ('text', 'val_text_loss', 'Text Loss', 0),
        ('image', 'val_img_loss', 'Image MSE Loss', 1),
        ('audio', 'val_aud_loss', 'Audio MSE Loss', 2),
        ('video', 'val_vid_loss', 'Video MSE Loss', 3),
    ]
    
    # Plot consolidated combined loss
    ax_combined = ax_flat[4]
    
    colors = {'text': '#1f77b4', 'image': '#ff7f0e', 'audio': '#2ca02c', 'video': '#d62728'}
    
    combined_vals = {}
    
    for phase_name, path in metric_paths.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        if not metrics:
            continue
        
        epochs = [m['epoch'] for m in metrics]
        
        for mod_name, key, title, idx in modalities:
            ax = ax_flat[idx]
            if key in metrics[0]:
                vals = [m[key] for m in metrics]
                ax.plot(epochs, vals, 'o-', label=phase_name, color=colors.get(mod_name, None), 
                       alpha=0.7, markersize=4)
                ax.set_title(title, fontsize=13, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        
        # Combined loss
        losses = []
        for m in metrics:
            l = 0
            if 'val_text_loss' in m:
                l += m['val_text_loss']
            if 'val_img_loss' in m:
                l += m['val_img_loss']
            if 'val_aud_loss' in m:
                l += m['val_aud_loss']
            if 'val_vid_loss' in m:
                l += m['val_vid_loss']
            losses.append(l)
        
        combined_vals[phase_name] = (epochs, losses)
    
    # Plot combined (each phase on its own x-axis)
    for phase, (epochs, vals) in combined_vals.items():
        if vals:
            ax_combined.plot(epochs, vals, 'o-', label=phase, markersize=4)
    ax_combined.set_title('Combined Validation Loss', fontsize=13, fontweight='bold')
    ax_combined.set_xlabel('Epoch')
    ax_combined.set_ylabel('Combined Loss')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend(fontsize=8)
    
    # Learning rate
    ax_lr = ax_flat[5]
    for phase_name, path in metric_paths.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        if not metrics or 'lr' not in metrics[0]:
            continue
        epochs = [m['epoch'] for m in metrics]
        lrs = [m['lr'] for m in metrics]
        ax_lr.plot(epochs, lrs, 'o-', label=phase_name, markersize=4)
    ax_lr.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('LR')
    ax_lr.grid(True, alpha=0.3)
    ax_lr.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to {save_path}")


def visualize_reconstructions(model, device, tokenizer, args):
    """Run and visualize reconstruction tests for all modalities."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    num_samples = min(args.num_samples, 8)
    seed = args.seed
    
    # ── 1. Image Reconstruction ──
    print("\n" + "="*60)
    print("📷  IMAGE RECONSTRUCTION EVALUATION")
    print("="*60)
    
    ds_img = SyntheticDataset(num_samples=num_samples, image_size=224, seed=seed)
    orig_images = []
    recon_images = []
    img_psnr = []
    
    for i in range(num_samples):
        img_tensor, caption = ds_img[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        recon = model.reconstruct_image(img_tensor)
        psnr = compute_psnr(img_tensor, recon)
        img_psnr.append(psnr)
        orig_images.append(tensor_to_numpy(img_tensor[0]))
        recon_images.append(tensor_to_numpy(recon[0]))
    
    avg_psnr = np.mean(img_psnr)
    print(f"  Avg PSNR: {avg_psnr:.2f} dB (per-sample: {[f'{p:.2f}' for p in img_psnr]})")
    results['image'] = {'psnr_dB': avg_psnr, 'per_sample_psnr': img_psnr}
    
    # Save image comparison figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))
    for i in range(num_samples):
        axes[0, i].imshow(orig_images[i])
        axes[0, i].set_title(f'Original {i+1}', fontsize=9)
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_images[i])
        axes[1, i].set_title(f'Recon {img_psnr[i]:.1f}dB', fontsize=9)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=11)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=11)
    fig.suptitle(f'Image Reconstruction (Avg PSNR: {avg_psnr:.2f} dB)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    img_fig_path = output_dir / 'image_reconstruction.png'
    plt.savefig(img_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved {img_fig_path}")
    
    # ── 2. Audio Reconstruction ──
    print("\n" + "="*60)
    print("🔊  AUDIO RECONSTRUCTION EVALUATION")
    print("="*60)
    
    ds_aud = AudioDataset(num_samples=num_samples, seed=seed)
    orig_mels = []
    recon_mels = []
    aud_mse = []
    aud_snr = []
    
    for i in range(num_samples):
        mel_tensor, caption = ds_aud[i]  # [1, n_mels, T]
        mel_tensor = mel_tensor.unsqueeze(0).to(device)  # [1, 1, n_mels, T]
        recon = model.reconstruct_audio(mel_tensor)
        
        # Handle length mismatch (audio may not be multiple of patch time)
        min_t = min(mel_tensor.shape[-1], recon.shape[-1])
        mel_trimmed = mel_tensor[..., :min_t]
        recon_trimmed = recon[..., :min_t]
        
        mse = F.mse_loss(mel_trimmed, recon_trimmed).item()
        snr = compute_snr(mel_trimmed, recon_trimmed)
        aud_mse.append(mse)
        aud_snr.append(snr)
        orig_mels.append(mel_trimmed[0].detach().cpu().numpy())
        recon_mels.append(recon_trimmed[0].detach().cpu().numpy())
    
    avg_aud_mse = np.mean(aud_mse)
    avg_aud_snr = np.mean(aud_snr)
    print(f"  Avg MSE: {avg_aud_mse:.6f}")
    print(f"  Avg SNR: {avg_aud_snr:.2f} dB")
    results['audio'] = {'mse': float(avg_aud_mse), 'snr_dB': float(avg_aud_snr),
                        'per_sample_mse': aud_mse, 'per_sample_snr': aud_snr}
    
    # Save audio comparison figure
    n_cols = min(num_samples, 4)
    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 3, 10))
    for i in range(n_cols):
        # Original spectrogram
        im0 = axes[0, i].imshow(orig_mels[i][0], aspect='auto', cmap='viridis',
                                vmin=-1, vmax=1)
        axes[0, i].set_title(f'Original {i+1}', fontsize=9)
        axes[0, i].axis('off')
        # Reconstructed
        im1 = axes[1, i].imshow(recon_mels[i][0], aspect='auto', cmap='viridis',
                                vmin=-1, vmax=1)
        axes[1, i].set_title(f'Recon MSE={aud_mse[i]:.4f}', fontsize=9)
        axes[1, i].axis('off')
        # Difference
        diff = orig_mels[i][0] - recon_mels[i][0]
        im2 = axes[2, i].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[2, i].set_title('Error', fontsize=9)
        axes[2, i].axis('off')
        # Frequency projection
        freq_profile_orig = np.mean(np.abs(orig_mels[i][0]), axis=1)
        freq_profile_recon = np.mean(np.abs(recon_mels[i][0]), axis=1)
        axes[3, i].plot(freq_profile_orig, 'b-', alpha=0.7, label='Orig', linewidth=1)
        axes[3, i].plot(freq_profile_recon, 'r--', alpha=0.7, label='Recon', linewidth=1)
        axes[3, i].set_title(f'Freq Profile (SNR={aud_snr[i]:.1f}dB)', fontsize=8)
        axes[3, i].legend(fontsize=6)
    
    fig.suptitle(f'Audio Mel-Spectrogram Reconstruction (Avg MSE: {avg_aud_mse:.6f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    aud_fig_path = output_dir / 'audio_reconstruction.png'
    plt.savefig(aud_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved {aud_fig_path}")
    
    # ── 3. Video Reconstruction ──
    print("\n" + "="*60)
    print("🎬  VIDEO RECONSTRUCTION EVALUATION")
    print("="*60)
    
    ds_vid = VideoDataset(num_samples=num_samples, seed=seed)
    orig_vids = []
    recon_vids = []
    vid_psnr = []
    vid_frame_psnr = []
    
    for i in range(num_samples):
        vid_tensor, caption = ds_vid[i]  # [3, 4, 64, 64]
        vid_tensor = vid_tensor.unsqueeze(0).to(device)  # [1, 3, 4, 64, 64]
        recon = model.reconstruct_video(vid_tensor)
        
        psnr = compute_psnr(vid_tensor, recon)
        vid_psnr.append(psnr)
        
        # Per-frame PSNR
        frame_psnrs = []
        # Video: [T, C, H, W] -> [T, H, W, C]
        vid_orig_np = ((vid_tensor[0].permute(1, 0, 2, 3).detach().cpu().numpy() + 1) / 2).clip(0, 1).transpose(0, 2, 3, 1)
        vid_recon_np = ((recon[0].permute(1, 0, 2, 3).detach().cpu().numpy() + 1) / 2).clip(0, 1).transpose(0, 2, 3, 1)
        
        for f in range(4):
            frame_psnr = compute_psnr(vid_tensor[0, :, f], recon[0, :, f], max_val=2.0)
            frame_psnrs.append(frame_psnr)
        
        vid_frame_psnr.append(frame_psnrs)
        orig_vids.append(vid_orig_np)
        recon_vids.append(vid_recon_np)
    
    avg_vid_psnr = np.mean(vid_psnr)
    avg_frame_psnr = np.mean([np.mean(fp) for fp in vid_frame_psnr])
    print(f"  Avg Video PSNR: {avg_vid_psnr:.2f} dB")
    print(f"  Avg Frame PSNR: {avg_frame_psnr:.2f} dB")
    results['video'] = {'psnr_dB': avg_vid_psnr, 'avg_frame_psnr_dB': avg_frame_psnr,
                        'per_sample_psnr': vid_psnr,
                        'per_frame_psnr': vid_frame_psnr}
    
    # Save video comparison (show 4 frames for first 4 samples)
    n_show = min(num_samples, 4)
    fig, axes = plt.subplots(n_show, 8, figsize=(20, n_show * 2.5))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        for t in range(4):
            # Original frame
            axes[i, t].imshow(orig_vids[i][t])
            axes[i, t].set_title(f'S{i+1} Orig t={t+1}', fontsize=8)
            axes[i, t].axis('off')
            # Reconstructed frame
            axes[i, t + 4].imshow(recon_vids[i][t])
            axes[i, t + 4].set_title(f'S{i+1} Recon t={t+1}', fontsize=8)
            axes[i, t + 4].axis('off')
    
    fig.suptitle(f'Video Reconstruction — 4 Frames (Avg PSNR: {avg_vid_psnr:.2f} dB)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    vid_fig_path = output_dir / 'video_reconstruction.png'
    plt.savefig(vid_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved {vid_fig_path}")
    
    # ── 4. Text Generation from Image ──
    print("\n" + "="*60)
    print("📝  TEXT GENERATION FROM IMAGE")
    print("="*60)
    
    gen_results = []
    ds_gen = SyntheticDataset(num_samples=min(num_samples, 5), image_size=224, seed=seed+100)
    
    for i in range(min(num_samples, 5)):
        img_tensor, gt_caption = ds_gen[i]
        img_tensor = img_tensor.to(device)
        gen_text = model.generate_text(img_tensor, tokenizer, max_len=48, temperature=0.7, top_k=30)
        gen_results.append({'gt': gt_caption, 'gen': gen_text})
        print(f"\n  Sample {i+1}:")
        print(f"    GT:  {gt_caption[:80]}")
        print(f"    Gen: {gen_text[:80]}")
    
    results['text_generation'] = gen_results
    
    # ── 5. Cross-modal text generation (image + video) ──
    print("\n" + "="*60)
    print("🎯  CROSS-MODAL TEXT GENERATION (Image + Video)")
    print("="*60)
    
    ds_vid2 = VideoDataset(num_samples=min(num_samples, 3), seed=seed+200)
    for i in range(min(num_samples, 3)):
        vid_tensor, gt_vid_caption = ds_vid2[i]
        # Also get an image for context
        img_tensor2, _ = SyntheticDataset(num_samples=1, image_size=224, seed=seed+300+i)[0]
        
        vid_tensor = vid_tensor.to(device)
        img_tensor2 = img_tensor2.to(device)
        gen_text2 = model.generate_text(img_tensor2, tokenizer, video=vid_tensor, max_len=48, temperature=0.7, top_k=30)
        
        print(f"\n  Cross Sample {i+1}:")
        print(f"    Video GT: {gt_vid_caption[:80]}")
        print(f"    Gen:      {gen_text2[:80]}")
        
        if 'cross_modal_gen' not in results:
            results['cross_modal_gen'] = []
        results['cross_modal_gen'].append({'video_gt': gt_vid_caption, 'gen': gen_text2})
    
    return results


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # ── Tokenizer ──
    tokenizer = SimpleTokenizer(max_vocab=10000)
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")
    
    # ── Model ──
    cfg = ModelConfig(
        dim=384, n_layers=6,
        image_size=224, patch_size=32,
        vocab_size=tokenizer.vocab_size,
        img_generation=True, img_decoder_hidden=512,
        use_audio=True, use_video=True,
    )
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.2f}M parameters")
    
    # ── Load checkpoint ──
    if os.path.exists(args.checkpoint):
        info = load_checkpoint_flexible(model, args.checkpoint, device)
        print(f"Checkpoint info: {info}")
    else:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        return
    
    # ── Plot training curves ──
    print("\n" + "="*60)
    print("📈  TRAINING CURVES")
    print("="*60)
    metric_paths = {
        'Phase1': './logs/metrics.json',
        'Phase2': './logs_phase2/metrics.json',
        'Phase3a': './logs_phase3a/metrics.json',
        'Phase4': './logs_phase4/metrics.json',
    }
    plot_training_curves(metric_paths, os.path.join(args.output_dir, 'training_curves.png'))
    
    # ── Run evaluations ──
    results = visualize_reconstructions(model, device, tokenizer, args)
    
    # ── Save all results ──
    save_metrics_json(results, os.path.join(args.output_dir, 'eval_metrics.json'))
    
    # ── Summary ──
    print("\n" + "="*60)
    print("📊  EVALUATION SUMMARY")
    print("="*60)
    print(f"  Image:  PSNR = {results.get('image', {}).get('psnr_dB', 'N/A'):.2f} dB")
    print(f"  Audio:  MSE  = {results.get('audio', {}).get('mse', 'N/A'):.6f}")
    print(f"          SNR  = {results.get('audio', {}).get('snr_dB', 'N/A'):.2f} dB")
    print(f"  Video:  PSNR = {results.get('video', {}).get('psnr_dB', 'N/A'):.2f} dB")
    print(f"  Params: {total/1e6:.2f}M / 30M budget")
    print(f"\n  Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

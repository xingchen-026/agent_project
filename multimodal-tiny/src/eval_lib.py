#!/usr/bin/env python3
"""Shared evaluation and demo infrastructure — model loading, metrics, visualization.

Consolidates:
  10 copies of 6-step model loading → load_eval_model()
  2 copies of tensor_to_numpy         → in data_lib (reused here)
  2 copies of compute_psnr/snr        → in losses (reused here)
  5 copies of matplotlib 'Agg' setup  → set once at module level
"""

import os, sys, math
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure src/ is importable
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config, ModelConfig
from utils import load_checkpoint_adaptive
from losses import compute_psnr, compute_snr, bleu_score, rouge_l, mse_loss


# ── Model Loading ───────────────────────────────────────────────────

def load_eval_model(checkpoint_path, tokenizer=None, device=None,
                    defaults=None):
    """Load model for evaluation — single source for the 6-step bootstrap.

    Args:
        checkpoint_path: path to .pt checkpoint
        tokenizer:       existing SimpleTokenizer (created if None)
        device:          torch device (auto-detected if None)
        defaults:        dict of config overrides (e.g. {'use_audio': True})
    Returns:
        (model, tokenizer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = SimpleTokenizer(max_vocab=10000)

    if defaults is None:
        defaults = {'img_generation': True, 'use_audio': True, 'use_video': True,
                    'use_memory_bank': True, 'n_mem_tokens': 16}

    cfg = resolve_config(checkpoint_path, tokenizer, defaults=defaults)
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {total:.2f}M ({cfg.describe()})")

    load_checkpoint_adaptive(model, checkpoint_path, device)
    model.eval()

    return model, tokenizer, device


# ── Image Helpers ───────────────────────────────────────────────────

def load_image_tensor(path_or_array, size=224):
    """Load PIL path or numpy array → [C, H, W] tensor in [-1, 1]."""
    from data_lib import preprocess_image_path, preprocess_image_pil
    from PIL import Image as PILImage

    if isinstance(path_or_array, (str, Path)):
        return preprocess_image_path(str(path_or_array), size)
    if isinstance(path_or_array, np.ndarray):
        img = PILImage.fromarray(path_or_array).resize((size, size), PILImage.LANCZOS)
        return preprocess_image_pil(img)
    if isinstance(path_or_array, PILImage.Image):
        img = path_or_array.resize((size, size), PILImage.LANCZOS)
        return preprocess_image_pil(img)
    raise TypeError(f"Expected path, np.ndarray, or PIL.Image, got {type(path_or_array)}")


# ── Evaluation Functions ────────────────────────────────────────────

@torch.no_grad()
def evaluate_coco_generation(model, tokenizer, dataset, device,
                              max_gen_len=48, num_samples=None):
    """Evaluate image→text generation on COCO: BLEU + ROUGE.

    Returns:
        dict with keys: bleu1, bleu4, rouge_l, samples (list of (img_path, gt, gen))
    """
    from torch.utils.data import DataLoader, Subset
    from data_lib import ImageCaptionCollate

    if num_samples:
        ds = Subset(dataset, range(min(num_samples, len(dataset))))
    else:
        ds = dataset

    results = {'bleu1': 0.0, 'bleu4': 0.0, 'rouge_l': 0.0, 'samples': []}
    n = 0

    for img_tensor, caption in ds:
        img_tensor = img_tensor.unsqueeze(0).to(device)
        bos_id = 2
        text_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        out = model(text_ids, images=img_tensor)
        logits = out if isinstance(out, torch.Tensor) else out['text_logits']
        gen_ids = logits[0].argmax(dim=-1)[:max_gen_len]
        gen_text = tokenizer.decode(gen_ids.tolist())

        b1, b4 = bleu_score(caption, gen_text)
        rl = rouge_l(caption, gen_text)
        results['bleu1'] += b1
        results['bleu4'] += b4
        results['rouge_l'] += rl
        results['samples'].append((caption, gen_text))
        n += 1

    results['bleu1'] /= max(n, 1)
    results['bleu4'] /= max(n, 1)
    results['rouge_l'] /= max(n, 1)
    results['n'] = n
    return results


@torch.no_grad()
def evaluate_audio_recon(model, mel_batch, device):
    """Evaluate audio reconstruction MSE and SNR.

    Args:
        model:     TinyMultimodal
        mel_batch: [B, 1, n_mels, T] mel spectrogram batch
        device:    torch device
    Returns:
        dict with mse, snr
    """
    mel_batch = mel_batch.to(device)
    bos_id = 2
    text_ids = torch.full((mel_batch.shape[0], 1), bos_id, dtype=torch.long, device=device)

    out = model(text_ids, audios=mel_batch, return_audio=True)
    recon = out.get('aud_recon')
    target = out.get('target_aud')

    if recon is None or target is None:
        return {'mse': float('inf'), 'snr': -float('inf')}

    mse_val = mse_loss(recon, target).item()
    snr_val = compute_snr(target, recon)
    return {'mse': mse_val, 'snr': snr_val}


@torch.no_grad()
def evaluate_retrieval(model, tokenizer, device, num_images=100, seed=42):
    """Cross-modal retrieval: text-to-image and image-to-text Recall@K.

    Returns:
        dict with recall@1, recall@5, recall@10, mrr for both directions
    """
    from losses import retrieval_accuracy, clip_contrastive_loss

    # Use synthetic images for quick retrieval benchmark
    from synthetic_data import SyntheticDataset
    from data_lib import encode_captions

    ds = SyntheticDataset(num_samples=num_images, image_size=224, seed=seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=min(16, num_images),
                                          shuffle=False, collate_fn=lambda batch: (
        torch.stack([b[0] for b in batch]), [b[1] for b in batch]))

    all_img_embs, all_text_embs = [], []

    for images, captions in loader:
        images = images.to(device)
        text_ids, lengths = encode_captions(tokenizer, captions, device=device)
        img_emb, text_emb = model._encode_contrastive_impl(images, text_ids)

        img_emb = F.normalize(img_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        all_img_embs.append(img_emb)
        all_text_embs.append(text_emb)

    img_embs = torch.cat(all_img_embs)
    text_embs = torch.cat(all_text_embs)

    t2i = retrieval_accuracy(img_embs, text_embs)
    i2t = retrieval_accuracy(text_embs, img_embs)

    return {'t2i': t2i, 'i2t': i2t}


# ── Visualization ───────────────────────────────────────────────────

def plot_image_reconstructions(orig_images, recon_images, psnrs, save_path,
                                num_show=8, figsize=(16, 8)):
    """Plot original vs reconstructed images side by side."""
    from data_lib import tensor_to_numpy

    num_show = min(num_show, len(orig_images))
    fig, axes = plt.subplots(2, num_show, figsize=figsize)

    for i in range(num_show):
        axes[0, i].imshow(tensor_to_numpy(orig_images[i]))
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(tensor_to_numpy(recon_images[i]))
        psnr_str = f"{psnrs[i]:.1f}" if psnrs[i] != float('inf') else "inf"
        axes[1, i].set_title(f"Recon (PSNR={psnr_str})")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


def plot_audio_reconstructions(orig_mels, recon_mels, mses, snrs, save_path,
                                num_show=4):
    """Plot original vs reconstructed mel spectrograms."""
    from data_lib import tensor_to_numpy

    num_show = min(num_show, len(orig_mels))
    fig, axes = plt.subplots(2, num_show, figsize=(4 * num_show, 6))

    for i in range(num_show):
        axes[0, i].imshow(tensor_to_numpy(orig_mels[i]).squeeze(), aspect='auto',
                          origin='lower', cmap='magma')
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(tensor_to_numpy(recon_mels[i]).squeeze(), aspect='auto',
                          origin='lower', cmap='magma')
        axes[1, i].set_title(f"Recon (SNR={snrs[i]:.1f}dB)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


def plot_training_curves(metric_paths, save_path, title="Training Curves"):
    """Plot loss curves from one or more metrics.json files.

    Args:
        metric_paths: dict of {label: path_to_metrics.json}
        save_path:    output PNG path
        title:        plot title
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, path in metric_paths.items():
        with open(path) as f:
            metrics = json.load(f)
        epochs = [m['epoch'] for m in metrics]
        vals = [m.get('val_loss', m.get('loss', 0)) for m in metrics]
        ax.plot(epochs, vals, 'o-', label=label, markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


# ── Demo Functions ──────────────────────────────────────────────────

def run_simple_demo(model, tokenizer, device, num_samples=3):
    """Run a quick all-modality test and print results."""
    from synthetic_data import SyntheticDataset
    from audio_synthetic import AudioDataset
    from video_synthetic import VideoDataset
    from data_lib import encode_captions, tensor_to_numpy

    print(f"\n{'=' * 50}")
    print("Running Demo Tests")
    print(f"{'=' * 50}")

    # Image test
    img_ds = SyntheticDataset(num_samples=num_samples, seed=42)
    print(f"\n--- Image Reconstruction ---")
    for i, (img, caption) in enumerate(img_ds):
        img = img.unsqueeze(0).to(device)
        text_ids, _ = encode_captions(tokenizer, [caption], device=device)
        out = model(text_ids, images=img, return_img=True)
        if 'img_recon' in out:
            psnr = compute_psnr(model._image_to_patches(img), out['img_recon'])
            print(f"  [{i}] PSNR={psnr:.1f}dB  caption: {caption[:40]}...")

    # Audio test
    aud_ds = AudioDataset(num_samples=num_samples, seed=42)
    print(f"\n--- Audio Reconstruction ---")
    for i, (aud, caption) in enumerate(aud_ds):
        aud = aud.unsqueeze(0).to(device)
        text_ids, _ = encode_captions(tokenizer, [caption], device=device)
        out = model(text_ids, audios=aud, return_audio=True)
        if out.get('aud_recon') is not None:
            snr = compute_snr(out['target_aud'], out['aud_recon'])
            print(f"  [{i}] SNR={snr:.1f}dB  caption: {caption[:40]}...")

    # Video test
    vid_ds = VideoDataset(num_samples=num_samples, seed=42)
    print(f"\n--- Video Reconstruction ---")
    for i, (vid, caption) in enumerate(vid_ds):
        vid = vid.unsqueeze(0).to(device)
        text_ids, _ = encode_captions(tokenizer, [caption], device=device)
        out = model(text_ids, videos=vid, return_video=True)
        if out.get('vid_recon') is not None:
            psnr = compute_psnr(out['target_vid'], out['vid_recon'])
            print(f"  [{i}] PSNR={psnr:.1f}dB  caption: {caption[:40]}...")

    print(f"\nDemo complete.\n")


# Convenience
import json

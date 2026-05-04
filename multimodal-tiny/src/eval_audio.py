#!/usr/bin/env python3
"""
Real Audio Evaluation for TinyMultimodal.
Tests audio reconstruction + generation on ESC-50 environmental sounds.

Usage:
  python src/eval_audio.py --checkpoint checkpoints_phase5_v2/best.pt
"""

import os, sys, json, math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMultimodal, ModelConfig
from tokenizer import SimpleTokenizer
from utils import load_checkpoint_flexible, DefaultConfig

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 128
DURATION = 1.0
N_FRAMES = int(SAMPLE_RATE * DURATION // HOP_LENGTH)


def get_args():
    parser = argparse.ArgumentParser(description="Real Audio Evaluation")
    parser.add_argument("--checkpoint", default="./checkpoints_phase5_v2/best.pt")
    parser.add_argument("--esc50-dir", default="../esc50_data")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--output-dir", default="./eval_audio_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Audio loading & preprocessing ─────────────────────────────────

def load_audio_file(path, target_sr=SAMPLE_RATE):
    """Load audio file, resample to target sample rate."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform[0]
    except Exception:
        pass

    # Fallback: scipy
    import scipy.io.wavfile as wav
    sr, data = wav.read(str(path))
    if data.dtype == np.int16 or data.dtype == np.int32:
        data = data.astype(np.float32) / 32768.0
    data = torch.from_numpy(data).float()
    if data.dim() == 0:
        return data
    if data.dim() == 2:
        data = data.mean(dim=1)
    if sr != target_sr:
        from torchaudio.transforms import Resample
        data = data.unsqueeze(0)
        resampler = Resample(sr, target_sr)
        data = resampler(data)
        data = data[0]
    return data


def waveform_to_mel(waveform, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert [T] waveform to [1, n_mels, T'] mel spectrogram."""
    window = torch.hann_window(n_fft)
    spec = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    mag = spec.abs()

    # Mel filterbank
    n_freqs = n_fft // 2 + 1
    mel_min, mel_max = 0.0, 2595.0 * math.log10(1.0 + SAMPLE_RATE / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / SAMPLE_RATE).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for f in range(left, center):
            fb[m - 1, f] = (f - left) / (center - left)
        for f in range(center, right):
            fb[m - 1, f] = (right - f) / (right - center)

    mel_spec = fb @ mag[:n_freqs, :]
    mel_spec = torch.clamp(mel_spec, min=1e-10)
    mel_spec = torch.log(mel_spec)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    mel_spec = mel_spec.clamp(-3, 3) / 3.0

    # Pad/truncate to N_FRAMES
    if mel_spec.shape[-1] < N_FRAMES:
        mel_spec = F.pad(mel_spec, (0, N_FRAMES - mel_spec.shape[-1]))
    else:
        mel_spec = mel_spec[:, :N_FRAMES]

    return mel_spec.unsqueeze(0)  # [1, n_mels, N_FRAMES]


# ── Evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_audio_recon(model, mel_batch, device):
    """Run audio reconstruction and compute metrics."""
    mel_batch = mel_batch.unsqueeze(0).to(device)  # [1, 1, n_mels, T]
    recon = model.reconstruct_audio(mel_batch)

    min_t = min(mel_batch.shape[-1], recon.shape[-1])
    mel_trimmed = mel_batch[..., :min_t]
    recon_trimmed = recon[..., :min_t]

    mse = F.mse_loss(mel_trimmed, recon_trimmed).item()
    signal_power = mel_trimmed.pow(2).mean().item()
    snr = 10 * math.log10(signal_power / max(mse, 1e-10))

    return {
        'mse': mse,
        'snr_db': snr,
        'original': mel_trimmed[0].cpu(),
        'reconstructed': recon_trimmed[0].cpu(),
    }


def evaluate(model, tokenizer, device, samples, args):
    """Run full evaluation on audio samples."""
    results = []
    mses, snrs = [], []

    print(f"Evaluating {len(samples)} audio samples...")
    for i, sample in enumerate(samples):
        try:
            if 'mel' in sample:
                mel = sample['mel']
            else:
                waveform = load_audio_file(sample['path'])
                # Take center segment of ~1 second
                center = len(waveform) // 2
                half = int(SAMPLE_RATE * DURATION) // 2
                start = max(0, center - half)
                end = min(len(waveform), center + half)
                segment = waveform[start:end]
                if len(segment) < SAMPLE_RATE * 0.5:
                    continue
                mel = waveform_to_mel(segment)
        except Exception as e:
            print(f"  [{i}] Error loading {sample.get('path', '?')}: {e}")
            continue

        result = evaluate_audio_recon(model, mel, device)
        result['index'] = i
        result['label'] = sample.get('label', f'sample_{i}')
        results.append(result)
        mses.append(result['mse'])
        snrs.append(result['snr_db'])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] MSE={np.mean(mses[-10:]):.4f} SNR={np.mean(snrs[-10:]):.1f}dB")

    metrics = {
        'avg_mse': float(np.mean(mses)) if mses else 0,
        'avg_snr_db': float(np.mean(snrs)) if snrs else 0,
        'num_evaluated': len(results),
        'per_sample_mse': mses,
        'per_sample_snr': snrs,
    }
    return results, metrics


# ── Synthetic baseline ─────────────────────────────────────────────

def synthetic_baseline(model, device, num_samples=30, seed=42):
    """Evaluate on synthetic audio for comparison."""
    from audio_synthetic import AudioDataset
    ds = AudioDataset(num_samples=num_samples, seed=seed)
    samples = []
    for i in range(num_samples):
        mel, caption = ds[i]
        samples.append({'mel': mel, 'label': caption[:40], 'caption': caption})
    return samples


# ── Visualization ──────────────────────────────────────────────────

def visualize(results, metrics, output_dir, title_prefix=""):
    """Create visualization of audio reconstruction results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_show = min(8, len(results))
    if n_show == 0:
        print(f"  No results to visualize")
        return
    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 2.5, 8))

    for i in range(n_show):
        r = results[i]
        orig = r['original']  # [1, n_mels, T]
        recon = r['reconstructed']

        # Original spectrogram
        axes[0, i].imshow(orig[0].numpy() if orig.dim() == 3 else orig.numpy(),
                         aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[0, i].set_title(r.get('label', '')[:30], fontsize=7)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original')

        # Reconstructed
        axes[1, i].imshow(recon[0].numpy() if recon.dim() == 3 else recon.numpy(),
                         aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[1, i].set_title(f'MSE={r["mse"]:.4f}', fontsize=7)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed')

        # Frequency profile comparison
        orig_np = orig[0].numpy() if orig.dim() == 3 else orig.numpy()
        recon_np = recon[0].numpy() if recon.dim() == 3 else recon.numpy()
        axes[2, i].plot(np.mean(np.abs(orig_np), axis=1), 'b-', alpha=0.7, lw=1, label='Orig')
        axes[2, i].plot(np.mean(np.abs(recon_np), axis=1), 'r--', alpha=0.7, lw=1, label='Recon')
        axes[2, i].set_title(f'SNR={r["snr_db"]:.1f}dB', fontsize=7)
        axes[2, i].legend(fontsize=5)
        if i == 0:
            axes[2, i].set_ylabel('Freq Profile')

    suptitle = f'{title_prefix} Audio Reconstruction'
    if metrics:
        suptitle += f'  |  MSE={metrics.get("avg_mse", 0):.5f}  SNR={metrics.get("avg_snr_db", 0):.1f}dB'
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    path = output_dir / f'audio_eval_{title_prefix.lower().replace(" ", "_")}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── ESC-50 Data ────────────────────────────────────────────────────

def load_esc50_samples(esc50_dir, num_samples=30, seed=42):
    """Load samples from ESC-50 dataset."""
    esc50_dir = Path(esc50_dir)
    audio_dir = esc50_dir / 'audio'
    meta_file = esc50_dir / 'meta' / 'esc50.csv'

    if not audio_dir.exists():
        return None

    # Read metadata
    samples = []
    if meta_file.exists():
        with open(meta_file) as f:
            lines = f.readlines()[1:]  # skip header
        for line in lines[:num_samples]:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                fname, category = parts[0], parts[2]
                path = audio_dir / fname
                if path.exists():
                    samples.append({'path': str(path), 'label': category})
    else:
        audio_files = sorted(audio_dir.glob('*.wav'))[:num_samples]
        for f in audio_files:
            samples.append({'path': str(f), 'label': f.stem[:30]})

    return samples if samples else None


# ── Main ───────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer(max_vocab=10000)
    cfg = ModelConfig(
        dim=DefaultConfig.dim, n_layers=DefaultConfig.n_layers,
        image_size=DefaultConfig.image_size, patch_size=DefaultConfig.patch_size,
        vocab_size=tokenizer.vocab_size,
        img_generation=True, img_decoder_hidden=DefaultConfig.img_decoder_hidden,
        use_audio=True, use_video=True,
    )
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    if os.path.exists(args.checkpoint):
        load_checkpoint_flexible(model, args.checkpoint, device)
    else:
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return

    print(f"\n{'='*60}")
    print("AUDIO EVALUATION")
    print(f"{'='*60}")

    # 1. Synthetic baseline
    print("\n--- Synthetic Audio Baseline ---")
    synthetic_samples = synthetic_baseline(model, device, args.num_samples, args.seed)
    syn_results, syn_metrics = evaluate(model, tokenizer, device, synthetic_samples, args)
    print(f"  Synthetic MSE={syn_metrics['avg_mse']:.5f} SNR={syn_metrics['avg_snr_db']:.1f}dB")
    visualize(syn_results, syn_metrics, args.output_dir, "Synthetic")

    # 2. ESC-50 real audio
    esc50_samples = load_esc50_samples(args.esc50_dir, args.num_samples, args.seed)
    if esc50_samples:
        print(f"\n--- ESC-50 Real Audio ({len(esc50_samples)} samples) ---")
        real_results, real_metrics = evaluate(model, tokenizer, device, esc50_samples, args)
        print(f"  ESC-50 MSE={real_metrics['avg_mse']:.5f} SNR={real_metrics['avg_snr_db']:.1f}dB")
        visualize(real_results, real_metrics, args.output_dir, "ESC50")

        # Comparison
        print(f"\n{'='*60}")
        print("COMPARISON: Synthetic vs Real Audio")
        print(f"{'='*60}")
        print(f"  Synthetic MSE:  {syn_metrics['avg_mse']:.5f}")
        print(f"  ESC-50 MSE:     {real_metrics['avg_mse']:.5f}")
        ratio = real_metrics['avg_mse'] / max(syn_metrics['avg_mse'], 1e-10)
        print(f"  Ratio (real/syn): {ratio:.1f}x")
        if ratio < 3:
            print(f"  ✓ Audio encoder generalizes well to real sounds!")
        elif ratio < 10:
            print(f"  ~ Moderate generalization gap")
        else:
            print(f"  ⚠ Large gap — model overfits to synthetic audio patterns")

        # Save combined
        with open(Path(args.output_dir) / 'audio_eval_metrics.json', 'w') as f:
            json.dump({
                'synthetic': syn_metrics,
                'esc50': real_metrics,
                'real_to_synthetic_mse_ratio': ratio,
            }, f, indent=2)
    else:
        print(f"\n  ESC-50 not found at {args.esc50_dir} — only synthetic baseline run")
        print(f"  Clone: git clone https://github.com/karolpiczak/ESC-50.git {args.esc50_dir}")
        print(f"  Then: cd {args.esc50_dir} && git lfs pull")


if __name__ == '__main__':
    main()

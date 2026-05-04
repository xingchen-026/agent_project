#!/usr/bin/env python3
"""
Quantization + Deployment Evaluation for TinyMultimodal.
Tests: dynamic INT8 quantization, model size, inference speed, accuracy.

Usage:
  python src/quantize_eval.py --checkpoint checkpoints_phase5_v2/best.pt
"""

import os, sys, json, math, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config
from utils import load_checkpoint_adaptive


def get_args():
    parser = argparse.ArgumentParser(description="Quantization + Deployment Eval")
    parser.add_argument("--checkpoint", default="./checkpoints_phase5_v2/best.pt")
    parser.add_argument("--output-dir", default="./quantize_results")
    parser.add_argument("--num-iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--onnx-export", action="store_true", help="Also export ONNX")
    return parser.parse_args()


# ── Model loading ──────────────────────────────────────────────────

def load_fp32_model(checkpoint_path, device='cpu'):
    tokenizer = SimpleTokenizer(max_vocab=10000)
    cfg = resolve_config(checkpoint_path, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.2f}M params ({cfg.describe()})")

    if os.path.exists(checkpoint_path):
        load_checkpoint_adaptive(model, checkpoint_path, device)
    model.eval()
    return model, tokenizer


# ── Size measurement ───────────────────────────────────────────────

def get_model_size_mb(model):
    """Get model size in MB (state_dict)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = Path(f.name).stat().st_size / (1024 * 1024)
    Path(f.name).unlink()
    return size_mb


def count_quantizable_params(model):
    """Count parameters in Linear layers (quantizable via dynamic q)."""
    linear_params = 0
    total_params = 0
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        total_params += params
        if isinstance(module, nn.Linear):
            linear_params += params
    return linear_params, total_params


# ── Accuracy test ──────────────────────────────────────────────────

def _model_dtype(model):
    """Get the dtype of model parameters."""
    return next(model.parameters()).dtype


@torch.no_grad()
def test_accuracy(model, tokenizer, device, num_samples=20):
    """Test reconstruction accuracy on all modalities."""
    from synthetic_data import SyntheticDataset
    from audio_synthetic import AudioDataset
    from video_synthetic import VideoDataset

    dtype = _model_dtype(model)
    results = {}

    # Image reconstruction
    ds_img = SyntheticDataset(num_samples=num_samples, image_size=224, seed=42)
    img_mses = []
    for i in range(num_samples):
        img, _ = ds_img[i]
        img = img.unsqueeze(0).to(device=device, dtype=dtype)
        recon = model.reconstruct_image(img)
        mse = F.mse_loss(img.float(), recon.float()).item()
        img_mses.append(mse)
    results['img_mse'] = float(np.mean(img_mses))

    # Audio reconstruction
    ds_aud = AudioDataset(num_samples=num_samples, seed=42)
    aud_mses = []
    for i in range(num_samples):
        mel, _ = ds_aud[i]
        mel = mel.unsqueeze(0).to(device=device, dtype=dtype)
        recon = model.reconstruct_audio(mel)
        min_t = min(mel.shape[-1], recon.shape[-1])
        mse = F.mse_loss(mel[..., :min_t].float(), recon[..., :min_t].float()).item()
        aud_mses.append(mse)
    results['aud_mse'] = float(np.mean(aud_mses))

    # Video reconstruction
    ds_vid = VideoDataset(num_samples=num_samples, seed=42)
    vid_mses = []
    for i in range(num_samples):
        vid, _ = ds_vid[i]
        vid = vid.unsqueeze(0).to(device=device, dtype=dtype)
        recon = model.reconstruct_video(vid)
        mse = F.mse_loss(vid.float(), recon.float()).item()
        vid_mses.append(mse)
    results['vid_mse'] = float(np.mean(vid_mses))

    return results


# ── Speed benchmark ────────────────────────────────────────────────

@torch.no_grad()
def benchmark_speed(model, tokenizer, device, num_iters=50):
    """Measure inference latency for each modality."""
    from synthetic_data import SyntheticDataset
    from audio_synthetic import AudioDataset
    from video_synthetic import VideoDataset

    latencies = {}

    dtype = _model_dtype(model)

    # Warmup
    img, _ = SyntheticDataset(num_samples=1, image_size=224, seed=0)[0]
    img = img.unsqueeze(0).to(device=device, dtype=dtype)
    for _ in range(5):
        _ = model.reconstruct_image(img)

    # Image benchmark
    times = []
    for i in range(num_iters):
        img, _ = SyntheticDataset(num_samples=1, image_size=224, seed=i)[0]
        img = img.unsqueeze(0).to(device=device, dtype=dtype)
        t0 = time.perf_counter()
        _ = model.reconstruct_image(img)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    latencies['image_ms'] = float(np.mean(times) * 1000)

    # Audio benchmark
    mel, _ = AudioDataset(num_samples=1, seed=0)[0]
    mel = mel.unsqueeze(0).to(device=device, dtype=dtype)
    for _ in range(5):
        _ = model.reconstruct_audio(mel)

    times = []
    for i in range(num_iters):
        mel, _ = AudioDataset(num_samples=1, seed=i)[0]
        mel = mel.unsqueeze(0).to(device=device, dtype=dtype)
        t0 = time.perf_counter()
        _ = model.reconstruct_audio(mel)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    latencies['audio_ms'] = float(np.mean(times) * 1000)

    # Video benchmark
    vid, _ = VideoDataset(num_samples=1, seed=0)[0]
    vid = vid.unsqueeze(0).to(device=device, dtype=dtype)
    for _ in range(5):
        _ = model.reconstruct_video(vid)

    times = []
    for i in range(num_iters):
        vid, _ = VideoDataset(num_samples=1, seed=i)[0]
        vid = vid.unsqueeze(0).to(device=device, dtype=dtype)
        t0 = time.perf_counter()
        _ = model.reconstruct_video(vid)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    latencies['video_ms'] = float(np.mean(times) * 1000)

    return latencies


# ── Quantization ───────────────────────────────────────────────────

def quantize_model_int8(model):
    """INT8 dynamic quantization — body only, decoders FP32."""
    model_q = torch.quantization.quantize_dynamic(
        model, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False)
    for attr in ['img_proj', 'img_decoder', 'audio_proj', 'audio_decoder',
                 'video_proj', 'video_decoder']:
        if hasattr(model, attr) and hasattr(model_q, attr):
            setattr(model_q, attr, getattr(model, attr))
    return model_q


def quantize_model_fp16(model):
    """FP16 half-precision — simple, near-lossless, 2x compression."""
    return model.half()


# ── Main ───────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device("cpu")  # Quantization works best on CPU
    print(f"Device: {device} (CPU for quantization benchmarks)")

    # Load FP32 model
    print(f"\n{'='*60}")
    print("1. Loading FP32 Model")
    print(f"{'='*60}")
    model_fp32, tokenizer = load_fp32_model(args.checkpoint, device)
    fp32_size = get_model_size_mb(model_fp32)
    lin_params, total_params = count_quantizable_params(model_fp32)
    print(f"  FP32 model size: {fp32_size:.1f} MB")
    print(f"  Total params: {total_params:,}")
    print(f"  Linear params (quantizable): {lin_params:,} ({100*lin_params/total_params:.1f}%)")

    # FP32 accuracy
    print(f"\n{'='*60}")
    print("2. FP32 Accuracy (reconstruction MSE)")
    print(f"{'='*60}")
    acc_fp32 = test_accuracy(model_fp32, tokenizer, device)
    for k, v in acc_fp32.items():
        print(f"  {k}: {v:.6f}")

    # FP32 speed
    print(f"\n{'='*60}")
    print("3. FP32 Inference Speed ({0} iters)".format(args.num_iters))
    print(f"{'='*60}")
    speed_fp32 = benchmark_speed(model_fp32, tokenizer, device, args.num_iters)
    for k, v in speed_fp32.items():
        print(f"  {k}: {v:.2f} ms")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {'fp32': {'size_mb': fp32_size, 'accuracy': acc_fp32, 'speed_ms': speed_fp32}}

    # ── INT8 ──
    print(f"\n{'='*60}")
    print("4. INT8 Dynamic Quantization (transformer body only)")
    print(f"{'='*60}")
    model_int8 = quantize_model_int8(model_fp32)
    int8_size = get_model_size_mb(model_int8)
    print(f"  INT8 size: {int8_size:.1f} MB ({fp32_size/int8_size:.1f}x compression)")

    acc_int8 = test_accuracy(model_int8, tokenizer, device)
    for k, v in acc_int8.items():
        delta_pct = 100 * (v - acc_fp32[k]) / max(acc_fp32[k], 1e-10)
        print(f"  {k}: {v:.6f} ({delta_pct:+.1f}%)")

    speed_int8 = benchmark_speed(model_int8, tokenizer, device, args.num_iters)
    results['int8'] = {'size_mb': int8_size, 'accuracy': acc_int8, 'speed_ms': speed_int8}

    # ── FP16 ──
    print(f"\n{'='*60}")
    print("5. FP16 Half-Precision")
    print(f"{'='*60}")
    model_fp16 = quantize_model_fp16(model_fp32)
    fp16_size = get_model_size_mb(model_fp16)
    print(f"  FP16 size: {fp16_size:.1f} MB ({fp32_size/fp16_size:.1f}x compression)")

    acc_fp16 = test_accuracy(model_fp16, tokenizer, device)
    for k, v in acc_fp16.items():
        delta_pct = 100 * (v - acc_fp32[k]) / max(acc_fp32[k], 1e-10)
        print(f"  {k}: {v:.6f} ({delta_pct:+.1f}%)")

    speed_fp16 = benchmark_speed(model_fp16, tokenizer, device, args.num_iters)
    for k, v in speed_fp16.items():
        delta_pct = 100 * (v - speed_fp32[k]) / max(speed_fp32[k], 1e-10)
        print(f"  {k}: {v:.2f} ms ({delta_pct:+.1f}%)")
    results['fp16'] = {'size_mb': fp16_size, 'accuracy': acc_fp16, 'speed_ms': speed_fp16}

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'':>10} {'FP32':>10} {'INT8':>10} {'FP16':>10}")
    print(f"  {'Size':>8}: {fp32_size:>8.1f} MB {int8_size:>8.1f} MB {fp16_size:>8.1f} MB")
    for mod in ['img_mse', 'aud_mse', 'vid_mse']:
        label = mod.replace('_mse', '')
        print(f"  {label:>8}: {acc_fp32[mod]:>8.5f}  {acc_int8[mod]:>8.5f}  {acc_fp16[mod]:>8.5f}")

    # Best recommendation
    print(f"\n  Compression: INT8={fp32_size/int8_size:.1f}x  FP16={fp32_size/fp16_size:.1f}x")
    int8_acc_loss = max(abs(acc_int8[k]/max(acc_fp32[k],1e-10)-1) for k in acc_fp32)
    fp16_acc_loss = max(abs(acc_fp16[k]/max(acc_fp32[k],1e-10)-1) for k in acc_fp32)
    print(f"  Max accuracy loss: INT8={int8_acc_loss*100:.1f}%  FP16={fp16_acc_loss*100:.1f}%")

    if fp16_acc_loss < 0.02 and fp16_size <= 40:
        print(f"\n  ★ FP16 recommended: near-lossless ({fp16_acc_loss*100:.1f}% loss), {fp32_size/fp16_size:.1f}x compression, GPU-friendly")
    if int8_size <= 25 and int8_acc_loss < 0.10:
        print(f"  ★ INT8 suitable for CPU deployment (size priority)")

    results['recommendation'] = {
        'int8_acc_loss_pct': round(int8_acc_loss * 100, 2),
        'fp16_acc_loss_pct': round(fp16_acc_loss * 100, 2),
        'fp32_to_int8_ratio': round(fp32_size / int8_size, 1),
        'fp32_to_fp16_ratio': round(fp32_size / fp16_size, 1),
    }

    with open(output_dir / 'quantize_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'quantize_results.json'}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Inference Optimization Benchmark: torch.compile + FP16 + throughput.
Usage:
  python benchmark_compile.py --checkpoint ../checkpoints_phase6/best.pt
"""

import os, sys, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config
from utils import load_checkpoint_adaptive


@torch.no_grad()
def benchmark(model, fn, warmup=5, iters=50):
    """Run benchmark on a function, return mean ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints_phase6/best.pt")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"torch.cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    tok = SimpleTokenizer(max_vocab=10000)
    cfg = resolve_config(args.checkpoint, tok,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    print(f"Config: {cfg.describe()}")

    # Build FP32 model
    model = TinyMultimodal(cfg).to(device)
    load_checkpoint_adaptive(model, args.checkpoint, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.2f}M params")
    model.eval()

    B = args.batch_size

    # Input tensors
    img = torch.randn(B, 3, 224, 224, device=device)
    mel = torch.randn(B, 1, 128, 125, device=device)
    vid = torch.randn(B, 3, 4, 64, 64, device=device)
    text_ids = torch.randint(0, 500, (B, 32), device=device)

    # ── Benchmarks ──
    results = {}
    model_names = []

    # 1. FP32 eager
    print(f"\n{'='*60}")
    print("Benchmarking FP32 (eager)...")
    print(f"{'='*60}")
    model_fp32 = model

    def run_img_fp32():
        model_fp32.reconstruct_image(img)
    def run_aud_fp32():
        model_fp32.reconstruct_audio(mel)
    def run_vid_fp32():
        model_fp32.reconstruct_video(vid)
    def run_text_fp32():
        model_fp32(text_ids, images=img)

    fp32_times = {
        'image': benchmark(model_fp32, run_img_fp32, iters=args.iters),
        'audio': benchmark(model_fp32, run_aud_fp32, iters=args.iters),
        'video': benchmark(model_fp32, run_vid_fp32, iters=args.iters),
        'text+image': benchmark(model_fp32, run_text_fp32, iters=args.iters),
    }
    results['fp32_eager'] = fp32_times
    model_names.append('fp32_eager')
    for k, v in fp32_times.items():
        print(f"  {k:>12}: {v:.2f} ms")

    # 2. FP32 + torch.compile
    print(f"\n{'='*60}")
    print("Benchmarking torch.compile (FP32)...")
    print(f"{'='*60}")
    try:
        torch._dynamo.config.suppress_errors = True
        model_compiled = torch.compile(model, mode="reduce-overhead")
        # Warmup compile
        _ = model_compiled.reconstruct_image(img)
        _ = model_compiled.reconstruct_audio(mel)
        _ = model_compiled.reconstruct_video(vid)

        def run_img_comp():
            model_compiled.reconstruct_image(img)
        def run_aud_comp():
            model_compiled.reconstruct_audio(mel)
        def run_vid_comp():
            model_compiled.reconstruct_video(vid)
        def run_text_comp():
            model_compiled(text_ids, images=img)

        comp_times = {
            'image': benchmark(model_compiled, run_img_comp, iters=args.iters),
            'audio': benchmark(model_compiled, run_aud_comp, iters=args.iters),
            'video': benchmark(model_compiled, run_vid_comp, iters=args.iters),
            'text+image': benchmark(model_compiled, run_text_comp, iters=args.iters),
        }
        results['fp32_compile'] = comp_times
        model_names.append('fp32_compile')
        for k, v in comp_times.items():
            speedup = fp32_times[k] / max(v, 1e-10)
            print(f"  {k:>12}: {v:.2f} ms ({speedup:.1f}x)")
    except Exception as e:
        print(f"  torch.compile failed: {e}")

    # 3. FP16 eager
    print(f"\n{'='*60}")
    print("Benchmarking FP16 (eager)...")
    print(f"{'='*60}")
    try:
        model_fp16 = model.half()
        img_h = img.half()
        mel_h = mel.half()
        vid_h = vid.half()

        def run_img_fp16():
            model_fp16.reconstruct_image(img_h)
        def run_aud_fp16():
            model_fp16.reconstruct_audio(mel_h)
        def run_vid_fp16():
            model_fp16.reconstruct_video(vid_h)

        fp16_times = {
            'image': benchmark(model_fp16, run_img_fp16, iters=args.iters),
            'audio': benchmark(model_fp16, run_aud_fp16, iters=args.iters),
            'video': benchmark(model_fp16, run_vid_fp16, iters=args.iters),
        }
        results['fp16_eager'] = fp16_times
        model_names.append('fp16_eager')
        for k, v in fp16_times.items():
            speedup = fp32_times[k] / max(v, 1e-10)
            print(f"  {k:>12}: {v:.2f} ms ({speedup:.1f}x)")
    except Exception as e:
        print(f"  FP16 failed: {e}")

    # Summary
    if len(model_names) >= 2:
        print(f"\n{'='*60}")
        print("SUMMARY: Speedup vs FP32 Eager")
        print(f"{'='*60}")
        header = f"{'':>12}"
        for mn in model_names:
            header += f"  {mn:>14}"
        print(header)
        for task in ['image', 'audio', 'video']:
            base = results['fp32_eager'][task]
            row = f"  {task:>10}"
            for mn in model_names:
                val = results[mn][task]
                spd = base / max(val, 1e-10)
                row += f"  {val:>7.2f}ms ({spd:.1f}x)"
            print(row)

    print(f"\nDone.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
COCO / Real-Image Evaluation for TinyMultimodal.
Evaluates image→text generation on real images with BLEU + ROUGE metrics.

Usage:
  python src/eval_coco.py --checkpoint checkpoints_phase5_v2/best.pt
  python src/eval_coco.py --checkpoint checkpoints_phase5_v2/best.pt --max-images 100
  python src/eval_coco.py --checkpoint checkpoints_phase5_v2/best.pt --image-dir /path/to/images
"""

import os, sys, json, math, argparse, urllib.request, zipfile, io
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config
from utils import load_checkpoint_adaptive


# ── BLEU / ROUGE Metrics ─────────────────────────────────────────

def ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_score(reference, candidate, max_n=4):
    """Compute BLEU-1 through BLEU-4. Returns (bleu1, bleu4)."""
    ref_tokens = reference.lower().replace('.', '').split()
    cand_tokens = candidate.lower().replace('.', '').split()
    if not ref_tokens or not cand_tokens:
        return 0.0, 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        cand_ngrams = Counter(ngrams(cand_tokens, n))
        matches = sum((ref_ngrams & cand_ngrams).values())
        total = max(len(cand_tokens) - n + 1, 1)
        precisions.append(matches / total if total > 0 else 0.0)

    bp = min(1.0, math.exp(1.0 - len(ref_tokens) / max(len(cand_tokens), 1)))
    bleu1 = precisions[0]

    if any(p == 0 for p in precisions):
        bleu4 = 0.0
    else:
        bleu4 = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bleu1, bleu4


def rouge_l(reference, candidate):
    """ROUGE-L: longest common subsequence based F-measure."""
    ref_tokens = reference.lower().replace('.', '').split()
    cand_tokens = candidate.lower().replace('.', '').split()
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    if recall + precision < 1e-10:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


# ── Data Loading ─────────────────────────────────────────────────

COCO_BASE_URL = 'http://images.cocodataset.org/val2017'
COCO_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


def load_coco_subset(data_dir, max_images=200):
    """Load COCO val2017 images + captions. Downloads individual images on demand."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    ann_file = data_dir / 'captions_val2017.json'
    img_dir = data_dir / 'val2017'
    img_dir.mkdir(parents=True, exist_ok=True)

    # Download annotations if missing (only ~8MB captions file from 241MB zip)
    if not ann_file.exists():
        print("Downloading COCO captions annotations (~8MB)...")
        _download_zip_file(COCO_ANN_URL, data_dir, 'annotations/captions_val2017.json',
                           ann_file)

    # Load annotations
    from pycocotools.coco import COCO
    coco = COCO(str(ann_file))
    img_ids = sorted(coco.imgs.keys())[:max_images]

    # Download individual images on demand (each ~100-200KB)
    samples = []
    for i, img_id in enumerate(img_ids):
        img_info = coco.imgs[img_id]
        fname = img_info['file_name']
        img_path = img_dir / fname

        if not img_path.exists():
            if i == 0:
                print(f"Downloading COCO images (up to {max_images}, ~100-200KB each)...")
            try:
                _download_file(f"{COCO_BASE_URL}/{fname}", img_path)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                continue
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(img_ids)} images...")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        samples.append({
            'image_path': str(img_path),
            'captions': [a['caption'] for a in anns],
            'image_id': img_id,
        })

    print(f"  Loaded {len(samples)} COCO images with {sum(len(s['captions']) for s in samples)} captions")
    return samples


def _download_zip_file(url, dest_dir, member_path, target_path):
    """Download a zip and extract a single member file."""
    resp = urllib.request.urlopen(url)
    total = int(resp.headers.get('content-length', 0))
    data = io.BytesIO()
    downloaded = 0
    while True:
        chunk = resp.read(65536)
        if not chunk:
            break
        data.write(chunk)
        downloaded += len(chunk)
        if total:
            print(f"\r  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB", end='', flush=True)
    print()
    data.seek(0)

    with zipfile.ZipFile(data) as zf:
        zf.extract(member_path, dest_dir)
        extracted = Path(dest_dir) / member_path
        if extracted != target_path:
            extracted.rename(target_path)
        # Clean up empty annotation dir
        ann_dir = Path(dest_dir) / 'annotations'
        if ann_dir.exists():
            try:
                ann_dir.rmdir()
            except OSError:
                pass
    print(f"  Extracted: {target_path.name}")


def _download_file(url, dest_path):
    """Download a single file."""
    resp = urllib.request.urlopen(url)
    with open(dest_path, 'wb') as f:
        f.write(resp.read())


def load_images_from_dir(image_dir, max_images=100):
    """Load images from a directory (no captions — generation-only mode)."""
    image_dir = Path(image_dir)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)[:max_images]
    samples = [{'image_path': str(p), 'captions': [], 'image_id': i} for i, p in enumerate(paths)]
    print(f"  Loaded {len(samples)} images from {image_dir}")
    return samples


# ── Image Preprocessing ──────────────────────────────────────────

def preprocess_image(image_path, image_size=224):
    """Load and preprocess an image for the model."""
    img = PILImage.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), PILImage.LANCZOS)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor


# ── Evaluation ───────────────────────────────────────────────────

def evaluate(model, tokenizer, device, samples, max_gen_len=48, check_recon=True):
    """Run image→text generation + reconstruction PSNR."""
    results = []
    bleu1s, bleu4s, rougels = [], [], []
    psnrs = []

    for i, sample in enumerate(samples):
        try:
            img_tensor = preprocess_image(sample['image_path']).to(device)
        except Exception as e:
            print(f"  [{i}] Error loading {sample['image_path']}: {e}")
            continue

        entry = {
            'image_path': sample['image_path'],
            'reference_captions': sample.get('captions', []),
        }

        # Image reconstruction PSNR
        if check_recon:
            with torch.no_grad():
                img_batch = img_tensor.unsqueeze(0)
                recon = model.reconstruct_image(img_batch)
                mse = torch.nn.functional.mse_loss(img_batch, recon).item()
                psnr = 10 * math.log10(4.0 / max(mse, 1e-10))
                entry['recon_psnr'] = psnr
                psnrs.append(psnr)

        # Text generation
        gen_text = model.generate_text(img_tensor, tokenizer, max_len=max_gen_len,
                                       temperature=0.7, top_k=30)
        entry['generated'] = gen_text

        if sample.get('captions'):
            ref = sample['captions'][0]
            b1, b4 = bleu_score(ref, gen_text)
            rl = rouge_l(ref, gen_text)
            entry['bleu1'] = b1
            entry['bleu4'] = b4
            entry['rouge_l'] = rl
            bleu1s.append(b1)
            bleu4s.append(b4)
            rougels.append(rl)

        results.append(entry)

        if (i + 1) % 25 == 0:
            parts = [f"{i+1}/{len(samples)}"]
            if psnrs:
                parts.append(f"PSNR={np.mean(psnrs[-25:]):.1f}dB")
            if bleu1s:
                parts.append(f"BLEU-1={np.mean(bleu1s[-25:]):.3f}")
            print(f"  [{' | '.join(parts)}]")

    metrics = {}
    if bleu1s:
        metrics['bleu1'] = float(np.mean(bleu1s))
        metrics['bleu4'] = float(np.mean(bleu4s))
        metrics['rouge_l'] = float(np.mean(rougels))
    if psnrs:
        metrics['recon_psnr_db'] = float(np.mean(psnrs))
    metrics['num_evaluated'] = len(results)
    return results, metrics


# ── Visualization ────────────────────────────────────────────────

def visualize(results, metrics, output_dir, num_show=12):
    """Create a visualization grid of generated captions vs references."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(num_show, len(results))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for i in range(n):
        r = results[i]
        try:
            img = PILImage.open(r['image_path']).convert('RGB').resize((224, 224))
            axes[i].imshow(img)
        except:
            axes[i].text(0.5, 0.5, 'Load Error', ha='center', va='center')

        title = r.get('generated', '')[:40]
        if r.get('reference_captions'):
            ref = r['reference_captions'][0][:40]
            b1 = r.get('bleu1', 0)
            title = f"Gen: {title}\nRef: {ref}\nBLEU-1={b1:.3f}"
        axes[i].set_title(title, fontsize=7)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    suptitle = f'COCO Image→Text Generation'
    if metrics:
        suptitle += f'  |  BLEU-1={metrics.get("bleu1",0):.4f}  BLEU-4={metrics.get("bleu4",0):.4f}  ROUGE-L={metrics.get("rouge_l",0):.4f}'
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    path = output_dir / 'coco_generation.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {path}")


# ── Main ─────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="COCO / Real-Image Evaluation")
    parser.add_argument("--checkpoint", default="./checkpoints_phase5_v2/best.pt")
    parser.add_argument("--coco-dir", default="./coco_data")
    parser.add_argument("--image-dir", default=None, help="Evaluate arbitrary images (no captions)")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--max-gen-len", type=int, default=48)
    parser.add_argument("--output-dir", default="./eval_coco_results")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    tokenizer = SimpleTokenizer(max_vocab=10000)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    cfg = resolve_config(args.checkpoint, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.2f}M params ({cfg.describe()})")

    ckpt_path = args.checkpoint
    if os.path.exists(ckpt_path):
        load_checkpoint_adaptive(model, ckpt_path, device)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    # Data
    if args.image_dir:
        samples = load_images_from_dir(args.image_dir, args.max_images)
        has_captions = False
    elif args.no_download:
        # Try local COCO dir first
        img_dir = Path(args.coco_dir) / 'val2017'
        ann_file = Path(args.coco_dir) / 'captions_val2017.json'
        if img_dir.exists() and ann_file.exists():
            samples = load_coco_subset(args.coco_dir, args.max_images)
            has_captions = True
        else:
            print("No COCO data found and --no-download set. Use --image-dir for local images.")
            return
    else:
        try:
            samples = load_coco_subset(args.coco_dir, args.max_images)
            has_captions = True
        except Exception as e:
            print(f"COCO download failed: {e}")
            print("Falling back to local image directory mode...")
            # Try to find local images
            local_dirs = [
                os.path.expanduser('~/Pictures'),
                os.path.expanduser('~/Desktop'),
                '.',
            ]
            for d in local_dirs:
                if os.path.isdir(d):
                    samples = load_images_from_dir(d, args.max_images)
                    if samples:
                        has_captions = False
                        break
            if not samples:
                print("No images found. Aborting.")
                return

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating {len(samples)} images...")
    print(f"{'='*60}")

    model.eval()
    results, metrics = evaluate(model, tokenizer, device, samples, args.max_gen_len)

    # Print metrics
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    if 'recon_psnr_db' in metrics:
        print(f"  Recon PSNR: {metrics['recon_psnr_db']:.1f} dB")
    if has_captions:
        print(f"  BLEU-1:     {metrics.get('bleu1', 0):.4f}")
        print(f"  BLEU-4:     {metrics.get('bleu4', 0):.4f}")
        print(f"  ROUGE-L:    {metrics.get('rouge_l', 0):.4f}")
    print(f"  Images:     {metrics['num_evaluated']}")

    if has_captions and metrics.get('bleu1', 0) >= 0.15:
        print(f"\n  ✓ BLEU-1 ≥ 0.15 — ROADMAP target achieved!")
    elif has_captions:
        print(f"\n  Note: Model trained on synthetic geometric shapes only.")
        print(f"  Real photo captioning requires real-image training data.")
    # Print sample generations
    for r in results[:3]:
        print(f"\n  [{Path(r['image_path']).name}]")
        print(f"    Gen: {r['generated'][:100]}")
        if r.get('reference_captions'):
            print(f"    Ref: {r['reference_captions'][0][:100]}")
        if 'recon_psnr' in r:
            print(f"    PSNR: {r['recon_psnr']:.1f} dB")

    # Visualize
    visualize(results, metrics, args.output_dir)

    # Save results
    output_dir = Path(args.output_dir)
    with open(output_dir / 'coco_eval_results.json', 'w') as f:
        json.dump({'metrics': metrics, 'samples': [
            {k: v for k, v in r.items()} for r in results[:50]
        ]}, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {output_dir / 'coco_eval_results.json'}")


if __name__ == '__main__':
    main()

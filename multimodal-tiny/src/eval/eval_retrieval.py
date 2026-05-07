#!/usr/bin/env python3
"""
Cross-Modal Retrieval Evaluation for TinyMultimodal.
Tests text→image retrieval using shared latent space cosine similarity.

Usage:
  python src/eval_retrieval.py --checkpoint checkpoints_phase5_v2/best.pt
"""

import os, sys, json, math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import TinyMultimodal
from core.tokenizer import SimpleTokenizer
from data.cn_data import ZhImageDataset
from core.config import resolve_config
from utils import load_checkpoint_adaptive


# ── Embedding extraction ──────────────────────────────────────────

@torch.no_grad()
def encode_image(model, image_tensor, device):
    """Encode an image → pooled embedding vector [dim]."""
    # Pass image through model with minimal text
    bos_id = 2  # <bos>
    text_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    return _extract_sensory_embedding(model, text_ids, images=image_tensor, device=device,
                                      sensory_slice=slice(0, 49))  # 49 image tokens


@torch.no_grad()
def encode_text(model, text_ids, device):
    """Encode text → pooled embedding vector [dim]."""
    # Pass text with placeholder image (no real image)
    return _extract_text_embedding(model, text_ids, device=device)


def _extract_sensory_embedding(model, text_ids, images, device, sensory_slice):
    """Run partial forward pass and extract sensory hidden states."""
    model.eval()

    # Manual forward to get hidden states (simplified from model.forward)
    B = 1
    n_img = model.get_num_image_tokens()
    patches = model._image_to_patches(images)
    img_tokens = model.img_norm(model.img_proj(patches))
    text_tokens = model.text_embed(text_ids)

    x = torch.cat([img_tokens, text_tokens], dim=1)
    total_len = x.shape[1]

    # Type embeddings
    if model.cfg.use_type_embed:
        img_type = torch.full((B, n_img), 1, device=device, dtype=torch.long)
        txt_type = torch.zeros(B, text_ids.shape[1], device=device, dtype=torch.long)
        type_ids = torch.cat([img_type, txt_type], dim=1)
        x = x + model.type_embed(type_ids)

    cos, sin = model.rope(total_len, device)
    mask = model._make_attention_mask(0, n_img, 0, total_len, device)

    for block in model.blocks:
        x = block(x, cos, sin)
    x = model.final_norm(x)

    # Pool image region
    img_hidden = x[:, sensory_slice]
    return img_hidden.mean(dim=(0, 1))  # [dim]


def _extract_text_embedding(model, text_ids, device):
    """Run forward pass with placeholder image, extract text hidden states."""
    model.eval()
    B = 1
    n_img = model.get_num_image_tokens()

    # Use placeholder for absent image
    img_tokens = model.img_placeholder.expand(B, n_img, -1)
    text_tokens = model.text_embed(text_ids)

    x = torch.cat([img_tokens, text_tokens], dim=1)
    total_len = x.shape[1]

    if model.cfg.use_type_embed:
        img_type = torch.full((B, n_img), 3, device=device, dtype=torch.long)  # placeholder type
        txt_type = torch.zeros(B, text_ids.shape[1], device=device, dtype=torch.long)
        type_ids = torch.cat([img_type, txt_type], dim=1)
        x = x + model.type_embed(type_ids)

    cos, sin = model.rope(total_len, device)
    mask = model._make_attention_mask(0, n_img, 0, total_len, device)

    for block in model.blocks:
        x = block(x, cos, sin)
    x = model.final_norm(x)

    # Pool text region
    text_hidden = x[:, n_img:]
    return text_hidden.mean(dim=(0, 1))  # [dim]


# ── Retrieval evaluation ──────────────────────────────────────────

def evaluate_retrieval(model, tokenizer, device, num_images=100, seed=42):
    """Build image-text pairs, compute embeddings, evaluate retrieval."""
    print(f"Generating {num_images} synthetic image-text pairs...")

    ds = ZhImageDataset(num_samples=num_images, image_size=224, seed=seed)
    img_embeddings = []
    text_embeddings = []
    captions = []

    for i in range(num_images):
        img_tensor, caption = ds[i]
        img_tensor = img_tensor.to(device)

        # Encode image
        img_emb = encode_image(model, img_tensor.unsqueeze(0), device)
        img_embeddings.append(img_emb)

        # Encode text
        text_ids = torch.tensor([[2] + tokenizer.encode(caption)], dtype=torch.long, device=device)
        text_ids = text_ids[:, :48]
        text_emb = encode_text(model, text_ids, device)
        text_embeddings.append(text_emb)

        captions.append(caption)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{num_images}] encoded")

    # Stack embeddings
    img_emb = torch.stack(img_embeddings)  # [N, dim]
    text_emb = torch.stack(text_embeddings)  # [N, dim]

    # Normalize for cosine similarity
    img_emb = F.normalize(img_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    # Similarity matrix: text→image
    sim_matrix = text_emb @ img_emb.T  # [N, N]

    # Compute Recall@K
    total = num_images
    recall_at_k = {}
    for k in [1, 3, 5, 10]:
        # For each text query, is the matching image in top-K?
        correct = 0
        for i in range(total):
            top_k = sim_matrix[i].topk(k).indices
            if i in top_k:
                correct += 1
        recall_at_k[f'recall@{k}'] = correct / total

    # Mean reciprocal rank
    mrr = 0.0
    for i in range(total):
        rank = (sim_matrix[i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1.0 / rank
    mrr /= total

    metrics = {**recall_at_k, 'mrr': mrr, 'num_pairs': total}

    return metrics, sim_matrix, captions, img_emb, text_emb


# ── Visualization ─────────────────────────────────────────────────

def visualize_retrieval(sim_matrix, captions, output_dir, num_queries=5):
    """Show top-3 retrieval results for several text queries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_queries, 4, figsize=(16, num_queries * 3.5))
    indices = np.random.choice(len(captions), num_queries, replace=False)

    for row, idx in enumerate(indices):
        query = captions[idx]
        _, top3 = sim_matrix[idx].topk(3)

        axes[row, 0].text(0.1, 0.5, f'Query:\n"{query}"',
                         transform=axes[row, 0].transAxes, fontsize=9,
                         va='center', wrap=True)
        axes[row, 0].axis('off')
        axes[row, 0].set_title('Text Query', fontsize=10)

        for j, match_idx in enumerate(top3):
            matched_cap = captions[match_idx.item()]
            score = sim_matrix[idx][match_idx].item()
            is_correct = match_idx.item() == idx
            color = 'green' if is_correct else 'red'
            axes[row, j+1].text(0.5, 0.5, f'#{j+1} (sim={score:.3f})\n{matched_cap[:60]}',
                               transform=axes[row, j+1].transAxes, fontsize=7,
                               va='center', ha='center', wrap=True,
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))
            axes[row, j+1].axis('off')

    fig.suptitle('Cross-Modal Retrieval: Text → Image Caption', fontsize=13)
    plt.tight_layout()
    path = output_dir / 'retrieval_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {path}")

    # Similarity heatmap
    fig2, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix[:30, :30].cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Text Query Index')
    ax.set_title('Text→Image Cosine Similarity (first 30 pairs)')
    plt.colorbar(im, ax=ax)
    heatmap_path = output_dir / 'retrieval_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved to {heatmap_path}")


# ── Main ──────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Cross-Modal Retrieval Evaluation")
    parser.add_argument("--checkpoint", default="./checkpoints_phase5_v2/best.pt")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--output-dir", default="./eval_retrieval_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer(max_vocab=10000)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    cfg = resolve_config(args.checkpoint, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params ({cfg.describe()})")

    if os.path.exists(args.checkpoint):
        load_checkpoint_adaptive(model, args.checkpoint, device)
        print(f"Loaded: {args.checkpoint}")
    else:
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return

    print(f"\n{'='*60}")
    print(f"Cross-Modal Retrieval: {args.num_images} image-text pairs")
    print(f"{'='*60}")

    metrics, sim_matrix, captions, img_emb, text_emb = evaluate_retrieval(
        model, tokenizer, device, args.num_images, args.seed)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for k in ['recall@1', 'recall@3', 'recall@5', 'recall@10', 'mrr']:
        print(f"  {k}: {metrics[k]:.4f}")

    # Sanity: is diagonal dominant?
    diag_mean = sim_matrix.diagonal().mean().item()
    off_diag_mean = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (sim_matrix.numel() - len(sim_matrix))
    print(f"\n  Diag mean sim: {diag_mean:.4f}")
    print(f"  Off-diag mean sim: {off_diag_mean:.4f}")
    print(f"  Diag/Off-diag ratio: {diag_mean/max(off_diag_mean, 1e-10):.1f}x")

    if diag_mean > off_diag_mean * 1.5:
        print(f"\n  ✓ Cross-modal alignment detected (diagonal dominates)")
    else:
        print(f"\n  ⚠ Weak cross-modal alignment (diagonal not dominant)")

    visualize_retrieval(sim_matrix, captions, args.output_dir, num_queries=5)

    # Save
    with open(Path(args.output_dir) / 'retrieval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {Path(args.output_dir) / 'retrieval_metrics.json'}")


if __name__ == '__main__':
    main()

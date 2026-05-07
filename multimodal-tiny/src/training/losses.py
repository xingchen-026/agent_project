#!/usr/bin/env python3
"""All loss functions and evaluation metrics — single source of truth.

Consolidates (by original location):
  utils.py:        compute_text_loss, compute_mse_loss
  train_joint.py:  lm_loss_fn, clip_contrastive_loss, diffusion_loss_fn
  train_joint_full.py: lm_loss
  train_clip.py:   clip_loss, retrieval_accuracy
  train_distill.py: distill_loss
  train_dpo.py:    dpo_loss
  eval_coco.py:    bleu_score, rouge_l
  eval_all.py:     compute_psnr, compute_snr
"""

import math
from collections import Counter
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── LM Loss ─────────────────────────────────────────────────────────

def lm_loss(logits, targets, lengths=None, attn_mask=None):
    """Masked cross-entropy for language modeling.

    Supports two calling conventions:
      1) lengths=[...] — per-sample sequence lengths (BART-style)
      2) attn_mask=... — attention mask, shift-by-1 for next-token prediction
    If neither is provided, raises ValueError.
    """
    if lengths is not None:
        B = logits.shape[0]
        loss = 0.0
        for b in range(B):
            l = lengths[b] - 1
            if l > 0:
                loss += F.cross_entropy(logits[b, :l], targets[b, 1:l + 1])
        return loss / B

    if attn_mask is not None:
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_labels = targets[:, 1:].reshape(-1)
        shift_mask = attn_mask[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        return (loss * shift_mask).sum() / shift_mask.sum()

    raise ValueError("lm_loss requires either `lengths` or `attn_mask`")


# ── Reconstruction Loss ─────────────────────────────────────────────

def mse_loss(pred, target):
    """Safe MSE that returns 0.0 when either input is None."""
    if pred is None or target is None:
        return torch.tensor(0.0, device=pred.device if pred is not None else 'cpu')
    return F.mse_loss(pred, target)


# ── CLIP Contrastive Loss ───────────────────────────────────────────

def clip_contrastive_loss(img_emb, text_emb, temperature=0.07):
    """Symmetric InfoNCE loss (CLIP-style). L2-normalizes inputs internally."""
    img_emb = F.normalize(img_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = (text_emb @ img_emb.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Diffusion Loss ──────────────────────────────────────────────────

def diffusion_loss_fn(diffusion_decoder, patches, memory_hidden):
    """DDPM noise prediction loss."""
    loss, _ = diffusion_decoder(patches, memory_hidden)
    return loss


# ── Distillation Loss ───────────────────────────────────────────────

def distill_loss(student_emb, teacher_emb):
    """Cosine distance + 0.5 * MSE for teacher-student distillation."""
    cosine = 1 - F.cosine_similarity(student_emb, teacher_emb).mean()
    mse = F.mse_loss(student_emb, teacher_emb)
    return cosine + 0.5 * mse


# ── DPO Loss ────────────────────────────────────────────────────────

def dpo_loss(pref_logps, rej_logps, beta=0.1):
    """DPO loss: -log(sigmoid(beta * (pref_logp - rej_logp)))."""
    return -F.logsigmoid(beta * (pref_logps - rej_logps))


# ── Retrieval Accuracy ──────────────────────────────────────────────

def retrieval_accuracy(img_emb, text_emb, top_k=(1, 5, 10)):
    """Compute text-to-image retrieval Recall@K."""
    sim = text_emb @ img_emb.T
    acc = {}
    for k in top_k:
        correct = sum(1 for i in range(len(sim)) if i in sim[i].topk(k).indices)
        acc[f'recall@{k}'] = correct / len(sim)
    return acc


# ── BLEU / ROUGE ────────────────────────────────────────────────────

def _ngrams(tokens, n):
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference, candidate, max_n=4):
    """Compute BLEU-1 and BLEU-4. Returns (bleu1, bleu4)."""
    ref_tokens = reference.lower().replace('.', '').split()
    cand_tokens = candidate.lower().replace('.', '').split()
    if not ref_tokens or not cand_tokens:
        return 0.0, 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        cand_ngrams = Counter(_ngrams(cand_tokens, n))
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
    """ROUGE-L: longest common subsequence F-measure."""
    ref_tokens = reference.lower().replace('.', '').split()
    cand_tokens = candidate.lower().replace('.', '').split()
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    if recall + precision < 1e-10:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


# ── Image Quality Metrics ───────────────────────────────────────────

def compute_psnr(original, reconstructed, max_val=2.0):
    """Peak Signal-to-Noise Ratio (dB). Expects tensors in [-1, 1] range (max_val=2.0)."""
    mse = F.mse_loss(original, reconstructed).item()
    if mse < 1e-10:
        return float('inf')
    return 10 * math.log10((max_val ** 2) / mse)


def compute_snr(signal, noise_or_reconstructed):
    """Signal-to-Noise Ratio (dB). noise_or_reconstructed = reconstructed signal."""
    signal_power = signal.pow(2).mean().item()
    noise_power = (signal - noise_or_reconstructed).pow(2).mean().item()
    if noise_power < 1e-10:
        return float('inf')
    return 10 * math.log10(signal_power / noise_power)

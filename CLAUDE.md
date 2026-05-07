# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tiny Multimodal Transformer — 30.97M param native multimodal (text+image+audio+video+Chinese) model. Built with PyTorch 2.11+cu128, Python 3.12. RTX 4060 Laptop (8GB VRAM).

**Architecture**: 8-layer SwiGLU transformer (MoE optional), 448 dim, 7 heads, RMSNorm, RoPE, QK Norm. MemoryBank (16 mem tokens), CLIP contrastive head, Diffusion DDIM decoder.

## Rules

- **Data downloading is the user's responsibility.** When external data is needed, provide the download URL and instructions to the user — never attempt to download datasets, annotation files, or pretrained weights yourself. Wait for the user to confirm the data is in place before proceeding.

## Commands

```powershell
# Environment (PowerShell)
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"

# Environment (Git Bash)
source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8
```

### Training (from multimodal-tiny/)

```powershell
# ── Unified entry (recommended) ──
cd src
python train_unified.py --mode full --resume ../checkpoints_phase6/best.pt --epochs 20
python train_unified.py --mode joint --resume ../checkpoints_phase6/best.pt --epochs 15
python train_unified.py --mode clip --resume ../checkpoints_phase6/best.pt --epochs 10
python train_unified.py --mode distill --resume ../checkpoints_phase6/best.pt --epochs 10
python train_unified.py --mode audio_clip --resume ../checkpoints_phase6/best.pt --epochs 10

# ── COCO LM (large-scale, use --no-pre-cache for >5000 images) ──
python train_unified.py --mode coco_lm \
  --resume ../checkpoints_phase6/best.pt --epochs 20 \
  --ann-file ../coco_data/annotations/captions_train2017.json \
  --val-ann-file ../coco_data/captions_val2017.json \
  --max-images 50000 --no-pre-cache

# ── DPO alignment ──
python train_dpo.py --resume ../checkpoints_phase6_coco_lm_v2/best.pt --epochs 5

# ── Run scripts ──
..\run.ps1 phase6_full         # Full multi-modal joint (image+audio+video)
..\run.ps1 phase6_joint        # CLIP+LM+Diffusion
..\run.ps1 phase6_clip         # CLIP contrastive
..\run.ps1 phase6_distill      # Knowledge distillation
..\run.ps1 phase6_audio_clip   # Audio-CLIP on ESC-50
..\run.ps1 phase6              # Base from scratch (448d-8L-7h)

# ── Finetune (separate scripts) ──
python finetune_coco_cn.py --resume ../checkpoints_phase6/best.pt --epochs 15
python finetune_vqa.py --resume ../checkpoints_phase6_cn/best.pt --epochs 5
```

### Evaluation (from multimodal-tiny/src/)

```powershell
python eval_all.py --checkpoint ../checkpoints_phase6_full/best.pt
python eval_coco.py --checkpoint ../checkpoints_phase6_full/best.pt --coco-dir ../coco_data --max-images 200
python eval_retrieval.py --checkpoint ../checkpoints_phase6_clip/best.pt
python eval_audio.py --checkpoint ../checkpoints_phase6_full/best.pt --esc50-dir ../esc50_data
python quantize_eval.py --checkpoint ../checkpoints_phase6/best.pt
```

## Architecture

### Model (`src/model.py`): `TinyMultimodal`

**Submodules**: `_components.py` (RMSNorm, RoPE, SwiGLU, MoE), `_attention.py` (SelfAttention with QK Norm + KV cache, TransformerBlock), `_memory.py` (MemoryBank cross-attention compressor, DiffusionImageDecoder DDIM).

**Token sequence**: `[video_tokens | image_tokens | audio_patches | text_tokens]`

**Attention mask**: Sensory regions all-to-all; text causal + attends to all sensory. Type embeddings (7 types).

**Forward() returns**: text_logits always; optionally img_recon/aud_recon/vid_recon, memory_hidden/text_hidden, past_key_values (KV cache), aux_loss (MoE).

**Key constraint**: When MemoryBank is active, position-based reconstruction (img_recon, aud_recon, vid_recon) is INCORRECT — set `use_memory_bank=False` for reconstruction training.

### Shared Library Modules

| Module | Purpose |
|--------|---------|
| `losses.py` | All loss functions + metrics (lm_loss, clip_contrastive_loss, mse_loss, bleu_score, compute_psnr, etc.) |
| `data_lib.py` | All datasets (CocoCaptionDataset, CocoCnDataset, VqaDataset), preprocessing, collation |
| `training.py` | Optimizer/scheduler builders, checkpoint save/load, seed, logging |
| `eval_lib.py` | Model loading (load_eval_model), eval functions, visualization, demo runners |
| `train_unified.py` | Single training entry via `--mode {full,joint,clip,distill,base}` |

## Key Conventions

- Sensory data normalized to `[-1, 1]`
- `SimpleTokenizer`: character-level + common English/Chinese ngrams, offline, 1510 tokens with Chinese
- `load_checkpoint_adaptive()` in `utils.py`: key-name+shape matching, handles architecture changes
- `resolve_config()` in `config.py`: priority chain — checkpoint config → inferred from shapes → defaults → class defaults
- Training data is synthetic by default; real COCO used for joint/full/DPO training
- Multi-modal batches interleaved round-robin (full mode uses per-modality gradient accumulation)

## Checkpoint Inventory

| Checkpoint | Size | Best Metric | Description |
|-----------|------|-------------|-------------|
| `checkpoints_phase6/best.pt` | 118 MB | text=0.012 | Base 30.97M |
| `checkpoints_phase6_clip/` | 346 MB | R@1=20% | CLIP contrastive |
| `checkpoints_phase6_audio_clip/` | — | **R@1=88%** | Audio-CLIP ESC-50 |
| `checkpoints_phase6_distill/` | 178 MB | cos=0.45 | Distillation |
| `checkpoints_phase6_joint/` | 380 MB | val=1.33 | CLIP+LM+Diff |
| `checkpoints_phase6_full/best.pt` | ~380 MB | val=1.24 | Full joint (20ep) |
| `checkpoints_phase6_coco_lm_v2/` | — | **val=0.98** | COCO LM 50K图 20ep |
| `checkpoints_phase6_dpo_v3/` | — | **acc=70.7%** | DPO from coco_lm |
| `checkpoints_phase6_vqa/` | 330 MB | — | VQA finetuned |

val≈0.98 is the current hard ceiling at 50K images × 27.75M params (verified 3x).

## Dataset Inventory

| Dataset | Location | Size | Status |
|---------|----------|------|--------|
| COCO val2017 images | `coco_data/val2017/` | 5000 imgs | Ready |
| COCO val2017 captions | `coco_data/captions_val2017.json` | 3.7MB | Ready |
| COCO val2014 images | `coco_data/val2014/` | 1563 imgs | Ready |
| COCO train2014 images | `coco_data/train2014/` | — | Ready |
| COCO-CN Chinese captions | `coco_data/coco-cn-master/data/` | 4712 captions | Ready |
| ESC-50 audio | `esc50_data/audio/` | 2000 clips | Ready |

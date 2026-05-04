# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Tiny Multimodal Transformer — validating sub-30M parameter native multimodal (text+image+audio+video+Chinese) models. Built with PyTorch 2.11+cu128, Python 3.12. Currently 18.88M params. Windows 11 + RTX 4060 (8GB) development environment.

## Commands (Windows)

```powershell
# Environment (PowerShell)
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"

# Environment (Git Bash)
source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8

# ── Training ──
cd multimodal-tiny
.\run.ps1 phase5_v2                     # Phase 5 v2 (Chinese continued FT)
.\run.ps1 phase5_v3                     # Phase 5 v3 (COCO-CN real Chinese FT)

cd src
# COCO-CN real Chinese fine-tuning
python finetune_coco_cn.py --resume ..\checkpoints_phase5_v2\best.pt --epochs 10

# ── Evaluation ──
python eval_all.py --checkpoint ..\checkpoints_phase5_v2\best.pt
python eval_coco.py --checkpoint ..\checkpoints_phase5_v2\best.pt `
    --coco-dir ..\coco_data --no-download --max-images 200
python eval_retrieval.py --checkpoint ..\checkpoints_phase5_v2\best.pt
python eval_audio.py --checkpoint ..\checkpoints_phase5_v2\best.pt `
    --esc50-dir ..\esc50_data
python quantize_eval.py --checkpoint ..\checkpoints_phase5_v2\best.pt
python inference_demo.py --checkpoint ..\checkpoints_phase5_v2\best.pt --test-all
```

## Architecture

**Model** (`src/model.py`): `TinyMultimodal` — 6-layer SwiGLU transformer, RMSNorm, RoPE (FP16-safe), QK normalization. 384 dim, 6 heads, 224px images / 32px patches.

**Token sequence**: `[video_tokens | image_tokens | audio_patches | text_tokens]`

**Attention mask**: Sensory regions all-to-all; text causal + attends to all sensory. Type embeddings (7 types) distinguish modalities and placeholder vs real tokens.

**Modality projections**: unfold → Linear → RMSNorm. Image (3×32×32), Audio (16×16 mel freq×time), Video (3×2×16×16 spatiotemporal).

**Decoder heads**: 2-3 layer MLPs per modality for reconstruction. Learned placeholder tokens for absent modalities.

## Key conventions

- All sensory data normalized to `[-1, 1]`
- `SimpleTokenizer`: character-level + common English/Chinese ngrams, offline, 1510 tokens
- Checkpoint loading via `load_checkpoint_flexible()` in `utils.py` — matches by key name + shape, silent on missing/new keys
- `finetune_cn.py`: reads old vocab from checkpoint embedding, differential LR (embed 5×/decoder 2×/body 1×), mean-anchored init for new tokens, `--no_warmup` flag
- `finetune_coco_cn.py`: real Chinese image-caption FT using COCO-CN + COCO val2014 images
- Training data is synthetic by default: `SyntheticDataset`, `AudioDataset`, `VideoDataset` + `cn_data.py` native Chinese generators (10 image / 23 audio / 10 video patterns)
- Multi-modal batches interleaved round-robin
- RoPE `apply_rotary` casts cos/sin to input dtype (required for FP16 compatibility)

## Phase progression

| Phase | Modalities | Best val_loss | Checkpoint dir | Status |
|-------|-----------|---------------|----------------|--------|
| 1 | text+image | text=0.18 | `checkpoints/` | archived |
| 2 | +image gen head | text=0.127, img=0.022 | `checkpoints_phase2/` | archived |
| 3 | +audio | — | `checkpoints_phase3/` | archived |
| 4 | +video | text=0.037, img=0.017, aud=0.0003, vid=0.044 | `checkpoints_phase4/` | archived |
| 4.5 | balanced all | text=0.036, img=0.015, aud=0.0003, vid=0.036 | `checkpoints_phase4_5/` | kept (English baseline) |
| 5 | +Chinese | text=0.079, img=0.015, aud=0.0002, vid=0.036 | `checkpoints_phase5/` | archived |
| 5 v2 | native CN templates, +132 tokens, 20ep | text=0.048, img=0.015, aud=0.0003, vid=0.029 | `checkpoints_phase5_v2/` | **current best** |
| 5 v3 | COCO-CN real Chinese images | val_loss=2.03 (COCO-CN, 1500 imgs) | `checkpoints_phase5_v3/` | complete |

## Source files

```
src/
├── model.py              # Model architecture (SwiGLU, RoPE, RMSNorm, 4 modalities)
├── tokenizer.py           # SimpleTokenizer (1510 tokens, Chinese ngrams)
├── train.py               # Phase 1-4 training loop
├── train_utils.py         # Training utilities
├── finetune_cn.py         # Phase 5 Chinese FT (synthetic data)
├── finetune_coco_cn.py    # Phase 5 v3 COCO-CN real Chinese FT
├── utils.py               # Shared utilities (load_checkpoint_flexible, DefaultConfig, etc.)
├── data.py                # COCO data loading
├── synthetic_data.py      # Synthetic image generator
├── audio_synthetic.py     # Synthetic audio generator
├── video_synthetic.py     # Synthetic video generator
├── cn_data.py             # Native Chinese caption generators (10/23/10 patterns)
├── eval_all.py            # Full modality eval + visualization
├── eval_coco.py           # COCO real-image evaluation (BLEU/ROUGE/recon PSNR)
├── eval_audio.py          # ESC-50 real audio evaluation (MSE/SNR)
├── eval_retrieval.py      # Cross-modal retrieval (Recall@K/MRR)
├── evaluate.py            # Legacy Phase 1-2 evaluation
├── quantize_eval.py       # INT8/FP16 quantization comparison
└── inference_demo.py      # Interactive CLI + comprehensive test suite
```

## Evaluation summary (Phase 5 v2)

| Task | Metric | Result |
|------|--------|--------|
| COCO image recon | PSNR | 14.2 dB |
| COCO text generation | BLEU-1 | 0.00 (synthetic data limitation) |
| ESC-50 audio recon | MSE ratio vs synthetic | 7.6× |
| Cross-modal retrieval | Recall@1 | 3.0% |
| INT8 quantization | Size / Accuracy | 32.1 MB (2.2×) / +26% MSE |
| FP16 quantization | Size / Accuracy | 36.0 MB (2.0×) / 0.0% loss ★ |

## Dataset inventory

| Dataset | Path | Count |
|---------|------|-------|
| COCO val2017 images | `coco_data/val2017/` | 5,000 |
| COCO val2017 captions (EN) | `coco_data/captions_val2017.json` | 25,010 |
| COCO val2014 images | `coco_data/val2014/` | 1,563 |
| COCO-CN captions (ZH) | `coco_data/coco-cn-master/data/` | 4,712 |
| ESC-50 audio | `esc50_data/audio/` | 2,000 |

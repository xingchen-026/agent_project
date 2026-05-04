# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tiny Multimodal Transformer — validating sub-30M parameter native multimodal (text+image+audio+video+Chinese) models. Built with PyTorch 2.11+cu128, Python 3.12. Currently 18.88M params.

## Commands (Windows PowerShell)

```powershell
# Environment (PowerShell)
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"         # Always set for Chinese text output

# Environment (Git Bash)
source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8

# ── Training ──
cd multimodal-tiny
.\run.ps1 phase4                        # Phase 4 (all modalities from scratch)
.\run.ps1 phase5_v2                     # Phase 5 v2 (Chinese continued FT)
.\run.ps1 phase5_v3                     # Phase 5 v3 (COCO-CN real Chinese FT)
# Or with Git Bash: ./run.sh phase5_v2

cd src
# COCO-CN real Chinese fine-tuning
$env:PYTHONIOENCODING = "utf-8"
python finetune_coco_cn.py --resume ..\checkpoints_phase5_v2\best.pt --epochs 10

# ── Evaluation ──
$env:PYTHONIOENCODING = "utf-8"
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

**Model** (`src/model.py`): `TinyMultimodal` — 6-layer SwiGLU transformer, RMSNorm, RoPE (with FP16-safe dtype casting), QK normalization. 384 dim, 6 heads, 224px images / 32px patches.

**Token sequence**: `[video_tokens | image_tokens | audio_patches | text_tokens]`

**Attention mask**: Sensory regions all-to-all; text causal + attends to all sensory. Type embeddings (7 types) distinguish modalities and placeholder vs real tokens.

**Modality projections**: unfold → Linear → RMSNorm. Image (3×32×32), Audio (16×16 mel freq×time), Video (3×2×16×16 spatiotemporal).

**Decoder heads**: 2-3 layer MLPs per modality for reconstruction. Separate learned placeholder tokens for absent modalities.

## Key conventions

- Sensory data normalized to `[-1, 1]`
- `SimpleTokenizer`: character-level + common English/Chinese ngrams, offline, 1510 tokens with Chinese
- `load_checkpoint_flexible()` in `utils.py`: key-name+shape matching for phase-to-phase transfer
- `finetune_cn.py`: reads old vocab from checkpoint embedding shape, uses differential LR (embed 5×, decoder 2×, body 1×), mean-anchored init for new tokens. Supports `--no_warmup`.
- `finetune_coco_cn.py`: real Chinese image-caption fine-tuning using COCO-CN dataset + COCO val2014 images
- Training data is synthetic by default: `SyntheticDataset`, `AudioDataset`, `VideoDataset` + `cn_data.py` native Chinese generators (10 image / 23 audio / 10 video template patterns)
- Multi-modal batches interleaved round-robin

## Phase progression

| Phase | Modalities | Best val_loss | Checkpoint dir |
|-------|-----------|---------------|----------------|
| 1 | text+image | text=0.18 | `checkpoints/` |
| 2 | +image gen head | text=0.127, img=0.022 | `checkpoints_phase2/` |
| 3 | +audio | — | `checkpoints_phase3/` |
| 4 | +video | text=0.037, img=0.017, aud=0.0003, vid=0.044 | `checkpoints_phase4/` |
| 4.5 | balanced all | text=0.036, img=0.015, aud=0.0003, vid=0.036 | `checkpoints_phase4_5/` |
| 5 | +Chinese | text=0.079, img=0.015, aud=0.0002, vid=0.036 | `checkpoints_phase5/` |
| 5 v2 | native CN templates, +132 tokens, 20ep | text=0.048, img=0.015, aud=0.0003, vid=0.029 | `checkpoints_phase5_v2/` |
| 5 v3 | COCO-CN real Chinese images, 10ep | val_loss=2.03 (COCO-CN) | `checkpoints_phase5_v3/` |

## Evaluation results (Phase 5 v2)

| Task | Metric | Result |
|------|--------|--------|
| COCO image recon | PSNR | 14.2 dB |
| COCO text generation | BLEU-1 | 0.00 (synthetic data limitation) |
| ESC-50 audio recon | MSE ratio vs synthetic | 7.6× |
| Cross-modal retrieval | Recall@1 | 3.0% (vs 1.0% random) |
| INT8 quantization | Size | 32.1 MB (2.2× compression) |
| FP16 quantization | Size / Accuracy loss | 36.0 MB (2.0×) / 0.0% |

## Dataset inventory

| Dataset | Location | Size | Status |
|---------|----------|------|--------|
| COCO val2017 images | `coco_data/val2017/` | 5000 imgs | Ready |
| COCO val2017 captions | `coco_data/captions_val2017.json` | 3.7MB | Ready |
| COCO val2014 images | `coco_data/val2014/` | 1563 imgs | Ready |
| COCO-CN Chinese captions | `coco_data/coco-cn-master/data/` | 4712 captions | Ready |
| ESC-50 audio | `esc50_data/audio/` | 2000 clips | Ready |

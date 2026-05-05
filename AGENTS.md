# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Tiny Multimodal Transformer — 30.97M param native multimodal (text+image+audio+video+Chinese) model. Built with PyTorch 2.11+cu128, Python 3.12. Windows 11 + RTX 4060 (8GB VRAM).

**Integrated architecture**: 5 shared library modules + unified training entry point consolidating 10 separate scripts.

## Commands

```powershell
# Environment (PowerShell)
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"

# Environment (Git Bash)
source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8
```

### Training (from multimodal-tiny/src/)

```powershell
# ── Unified training (all modes) ──
python train_unified.py --mode full --resume ../checkpoints_phase6/best.pt --epochs 20
python train_unified.py --mode joint --resume ../checkpoints_phase6/best.pt --epochs 15
python train_unified.py --mode clip --resume ../checkpoints_phase6/best.pt --epochs 10
python train_unified.py --mode distill --resume ../checkpoints_phase6/best.pt --epochs 10
python train_unified.py --mode base --dim 448 --layers 8 --n_heads 7 --epochs 15

# ── DPO ──
python train_dpo.py --resume ../checkpoints_phase6_full/best.pt --epochs 5

# ── Finetune (standalone scripts) ──
python finetune_coco_cn.py --resume ../checkpoints_phase6/best.pt --epochs 15
python finetune_vqa.py --resume ../checkpoints_phase6_cn/best.pt --epochs 5
```

### Run Scripts (from multimodal-tiny/)

```powershell
.\run.ps1 phase6               # Base from scratch
.\run.ps1 phase6_full          # Full multi-modal joint
.\run.ps1 phase6_joint         # CLIP+LM+Diffusion
.\run.ps1 phase6_clip          # CLIP contrastive
.\run.ps1 phase6_distill       # Knowledge distillation
.\run.ps1 phase6_cn            # COCO-CN Chinese FT
.\run.ps1 phase6_vqa           # VQA instruction tuning
```

### Evaluation (from multimodal-tiny/src/)

```powershell
python eval_all.py --checkpoint ../checkpoints_phase6_full/best.pt
python eval_coco.py --checkpoint ../checkpoints_phase6_full/best.pt --coco-dir ../coco_data --max-images 200
python eval_audio.py --checkpoint ../checkpoints_phase6_full/best.pt --esc50-dir ../esc50_data
python eval_retrieval.py --checkpoint ../checkpoints_phase6_clip/best.pt
python quantize_eval.py --checkpoint ../checkpoints_phase6/best.pt
```

## Architecture

**Model** (`src/model.py`): `TinyMultimodal` — 8-layer SwiGLU/MoE transformer, 448 dim, 7 heads, RMSNorm, RoPE (FP16-safe), QK normalization. MemoryBank (16 mem tokens), CLIP contrastive head, Diffusion DDIM decoder, KV cache + speculative decoding.

**Token sequence**: `[video_tokens | image_tokens | audio_patches | text_tokens]` (MemoryBank compresses sensory → `[memory | text]`).

**Attention mask**: Sensory regions all-to-all; text causal + attends to all sensory. Type embeddings (7 types).

**Shared modules** (eliminate code duplication):
- `losses.py` — lm_loss, clip_contrastive_loss, mse_loss, dpo_loss, BLEU/ROUGE/PSNR
- `data_lib.py` — CocoCaptionDataset, CocoCnDataset, VqaDataset, PadCollate, preprocessing
- `training.py` — build_*_optimizer, build_scheduler, save_checkpoint, seed_everything
- `eval_lib.py` — load_eval_model, evaluate_* functions, visualization, demo runners
- `train_unified.py` — single entry point, `--mode {full,joint,clip,distill,base}`

## Key Conventions

- All sensory data normalized to `[-1, 1]`
- `SimpleTokenizer`: character-level + English/Chinese ngrams, offline, 1510 tokens
- `load_checkpoint_adaptive()` in `utils.py`: key-name+shape matching, handles architecture changes
- `resolve_config()` in `config.py`: checkpoint config → inferred shapes → defaults → class defaults
- MemoryBank INCOMPATIBLE with position-based reconstruction (img_recon/aud_recon/vid_recon). Set `use_memory_bank=False` for reconstruction training.
- Training data is synthetic by default; COCO real data used for full/joint/clip/DPO modes
- Multi-modal gradient accumulation: each modality does separate forward+backward, combined optimizer step
- RoPE `apply_rotary` casts cos/sin to input dtype (required for FP16 compatibility)

## Checkpoint Inventory

| Checkpoint | Size | Best | Description |
|-----------|------|------|-------------|
| `checkpoints_phase6/best.pt` | 118 MB | text=0.012 | Base 30.97M |
| `checkpoints_phase6_clip/` | 346 MB | R@1=20% | CLIP contrastive |
| `checkpoints_phase6_distill/` | 178 MB | cos=0.45 | Distillation |
| `checkpoints_phase6_joint/` | 380 MB | val=1.33 | CLIP+LM+Diff |
| `checkpoints_phase6_full/best.pt` | ~380 MB | val=1.24 | Full joint (20ep) |
| `checkpoints_phase6_dpo/best.pt` | ~380 MB | acc=65.4% | DPO tuned |
| `checkpoints_phase6_vqa/` | 330 MB | — | VQA finetuned |

## Source Files

```
src/
├── model.py              # Core model (753 lines)
├── _components.py        # RMSNorm, RoPE, SwiGLU, MoE
├── _attention.py         # SelfAttention + TransformerBlock
├── _memory.py            # MemoryBank + DiffusionImageDecoder
├── config.py             # ModelConfig dataclass
├── tokenizer.py          # SimpleTokenizer (1510 tokens)
├── utils.py              # Checkpoint loading (load_checkpoint_adaptive)
│
├── losses.py             # ★ Shared: loss functions + metrics
├── data_lib.py           # ★ Shared: datasets, preprocessing, collation
├── training.py           # ★ Shared: optimizer, scheduler, checkpoint
├── eval_lib.py           # ★ Shared: eval infrastructure, demo runners
├── train_unified.py      # ★ Unified training (--mode)
│
├── train_dpo.py          # DPO alignment training
├── train_utils.py        # DEPRECATED: replaced by training.py
│
├── finetune_cn.py        # Chinese FT (synthetic)
├── finetune_coco_cn.py   # COCO-CN real Chinese FT
├── finetune_vqa.py       # VQA instruction tuning
│
├── synthetic_data.py     # Synthetic image generator
├── audio_synthetic.py    # Synthetic audio generator
├── video_synthetic.py    # Synthetic video generator
├── cn_data.py            # Chinese caption templates (457 lines)
├── data.py               # DEPRECATED: replaced by data_lib.py
│
├── eval/                 # Evaluation scripts
│   ├── eval_all.py       #   Full modality eval + visualization
│   ├── eval_coco.py      #   COCO real-image (BLEU/ROUGE/recon PSNR)
│   ├── eval_audio.py     #   ESC-50 real audio (MSE/SNR)
│   ├── eval_retrieval.py #   Cross-modal retrieval (Recall@K/MRR)
│   ├── quantize_eval.py  #   INT8/FP16 quantization
│   └── benchmark_compile.py  # torch.compile benchmarks
│
└── demo/                 # Demo applications
    ├── inference_demo.py #   Interactive CLI + test suite
    ├── demo_vqa.py       #   VQA demo
    └── web_app.py        #   Gradio Web UI
```

## Dataset Inventory

| Dataset | Path | Count |
|---------|------|-------|
| COCO val2017 images | `coco_data/val2017/` | 5,000 |
| COCO val2017 captions (EN) | `coco_data/captions_val2017.json` | 25,010 |
| COCO val2014 images | `coco_data/val2014/` | 1,563 |
| COCO train2014 images | `coco_data/train2014/` | — |
| COCO-CN captions (ZH) | `coco_data/coco-cn-master/data/` | 4,712 |
| ESC-50 audio | `esc50_data/audio/` | 2,000 |

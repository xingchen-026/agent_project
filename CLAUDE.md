# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tiny Multimodal Transformer — a research project validating sub-30M parameter native multimodal (text+image+audio+video) models. Built with PyTorch, designed for CPU/edge training within a 20GB data budget.

## Commands

```bash
# CPU training (default: Phase 2, synthetic data, 10 epochs)
./multimodal-tiny/run.sh

# GPU training — Phase 4 (all modalities: text+image+audio+video)
./multimodal-tiny/run_gpu.sh

# Phase-specific GPU runs
./multimodal-tiny/run_phase3_audio.sh    # Phase 3: +audio
./multimodal-tiny/run_phase4_video.sh    # Phase 4: +video
./multimodal-tiny/run_phase4_5.sh        # Phase 4.5: balanced

# Evaluation
python multimodal-tiny/src/evaluate.py --checkpoint checkpoints/best.pt
python multimodal-tiny/src/evaluate.py --checkpoint checkpoints/best.pt --test recon   # image recon only
python multimodal-tiny/src/evaluate.py --checkpoint checkpoints/best.pt --test text_gen # text gen only
python multimodal-tiny/src/evaluate.py --checkpoint checkpoints/best.pt --test t2i      # text-to-image only

# Full eval with visualization (Phase 4+)
python multimodal-tiny/src/eval_all.py --checkpoint checkpoints_phase4/best.pt

# Interactive inference demo
python multimodal-tiny/src/inference_demo.py
python multimodal-tiny/src/inference_demo.py --test-all

# Chinese fine-tuning
python multimodal-tiny/src/finetune_cn.py --resume checkpoints_phase4_5/best.pt --epochs 5
```

## Architecture

**Model** (`src/model.py`): `TinyMultimodal` — 6-layer transformer with SwiGLU MLP, RMSNorm, RoPE, QK normalization. Default: 384 dim, 6 heads, 10k vocab, 224px images / 32px patches.

**Token sequence order**: `[video_tokens | image_tokens | audio_patches | text_tokens]`

**Attention mask**: Sensory regions (video/image/audio) are all-to-all; text is causal but attends to all preceding sensory tokens. Type embeddings distinguish source modalities.

**Modality processing**:
- Text: `nn.Embedding` tied with `lm_head`
- Image: unfold into patches → linear projection → RMSNorm
- Audio: mel spectrogram patches → linear projection → RMSNorm
- Video: spatiotemporal patches (C×pt×ps×ps) → linear projection → RMSNorm

**Decoder heads**: Separate lightweight heads (2-3 layer MLP) for image, audio, and video reconstruction from hidden states.

**Placeholder tokens**: Learned `nn.Parameter` vectors used when a modality is absent (e.g., text-to-image generation, or audio-gen-mode).

## Key conventions

- All sensory data normalized to `[-1, 1]` range
- Tokenizer is custom `SimpleTokenizer` — character-level with common ngrams, 10k vocab, offline, handles English + Chinese
- Checkpoint loading via `load_checkpoint_flexible()` — matches by key name + shape, silent on missing/new keys (enables phase-to-phase transfer)
- Training data is **synthetic by default** (no downloads): `SyntheticDataset` (geometric shapes), `AudioDataset` (sine/square/FM/AM tones), `VideoDataset` (moving shapes)
- Loss combination: `total = text_CE + img_weight*img_MSE + aud_weight*aud_MSE + vid_weight*vid_MSE`
- Multi-modal batches interleaved round-robin when training multiple modalities

## Phase progression

| Phase | Branch | Modalities | Checkpoint dir |
|-------|--------|-----------|----------------|
| 1 | main | text+image | `checkpoints/` |
| 2 | `phase2-generation` | +image generation head | `checkpoints_phase2/` |
| 3 | — | +audio | `checkpoints_phase3/` |
| 4 | — | +video | `checkpoints_phase4/` |
| 4.5 | — | balanced all-modality | `checkpoints_phase4_5/` |
| 5 | — | +Chinese fine-tuning | `checkpoints_phase5/` |

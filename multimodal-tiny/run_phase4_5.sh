#!/bin/bash
# Phase 4.5 — Video-focused fine-tuning
# Increases video data & loss weight; fine-tunes from Phase 4 best

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/../.venv/Scripts/activate"
export CUDA_VISIBLE_DEVICES=0

python src/train.py \
  --epochs 5 \
  --batch_size 24 \
  --use_synthetic \
  --train_size 10000 \
  --val_size 500 \
  --use_audio \
  --aud_train_size 5000 \
  --aud_val_size 200 \
  --use_video \
  --vid_train_size 5000 \
  --vid_val_size 200 \
  --aud_loss_weight 0.5 \
  --vid_loss_weight 1.0 \
  --img_loss_weight 0.5 \
  --output_dir checkpoints_phase4_5 \
  --log_dir logs_phase4_5 \
  --resume checkpoints_phase4/best.pt

echo "Phase 4.5 complete!"

#!/bin/bash
# Phase 4 GPU run — text + image + audio + video training
# Resumes from Phase 3a best checkpoint

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/../.venv/Scripts/activate"
export CUDA_VISIBLE_DEVICES=0

python src/train.py \
  --epochs 10 \
  --batch_size 24 \
  --use_synthetic \
  --train_size 10000 \
  --val_size 500 \
  --use_audio \
  --aud_train_size 5000 \
  --aud_val_size 200 \
  --use_video \
  --vid_train_size 3000 \
  --vid_val_size 100 \
  --aud_loss_weight 0.5 \
  --vid_loss_weight 0.5 \
  --img_loss_weight 0.5 \
  --output_dir checkpoints_phase4 \
  --log_dir logs_phase4 \
  --resume checkpoints_phase3a/best.pt

echo "Phase 4 complete!"

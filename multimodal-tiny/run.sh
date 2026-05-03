#!/bin/bash
# Run Tiny Multimodal Training — Phase 2
# Usage: ./run.sh [options]
#   --resume checkpoints/best.pt    Continue from Phase 1 checkpoint

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/../.venv"
source "$VENV/Scripts/activate"

echo "=== Tiny Multimodal Phase 2 Training ==="
echo ""

# Default: Phase 2 synthetic data training
python src/train.py \
  --use_synthetic \
  --epochs 10 \
  --batch_size 8 \
  --train_size 10000 \
  --val_size 500 \
  --layers 6 \
  --dim 384 \
  --patch_size 32 \
  --img_gen \
  --img_loss_weight 1.0 \
  --img_decoder_hidden 512 \
  --output_dir checkpoints \
  --log_dir logs \
  "$@"

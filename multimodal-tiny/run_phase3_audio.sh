#!/bin/bash
# Phase 3: Audio Modality Training
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/../.venv/Scripts/activate"

echo "=== Phase 3: Audio Modality Training ==="
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

python src/train.py \
  --use_synthetic \
  --use_audio \
  --epochs 8 \
  --batch_size 32 \
  --train_size 10000 \
  --val_size 500 \
  --aud_train_size 5000 \
  --aud_val_size 200 \
  --layers 6 \
  --dim 384 \
  --patch_size 32 \
  --img_loss_weight 0.5 \
  --aud_loss_weight 0.5 \
  --output_dir checkpoints_phase3a \
  --log_dir logs_phase3a \
  --resume checkpoints/best.pt

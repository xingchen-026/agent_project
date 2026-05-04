#!/bin/bash
# GPU Training — Phase 4 from Phase 4.5 checkpoint
# Usage: ./run_gpu.sh [extra python args]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/../.venv/Scripts/activate"
export CUDA_VISIBLE_DEVICES=0

echo "=== GPU Training ==="
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Mem: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"

python src/train.py \
  --use_synthetic \
  --epochs 10 \
  --batch_size 24 \
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
  --resume checkpoints_phase4_5/best.pt \
  "$@"

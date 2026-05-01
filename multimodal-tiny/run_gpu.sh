#!/bin/bash
# Run Tiny Multimodal Training on GPU
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source /home/xilc/.openclaw/workspace/mvp-venv/bin/activate
export LD_LIBRARY_PATH=/home/xilc/.openclaw/workspace/mvp-venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

echo "=== GPU Training ==="
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

python src/train.py \
  --use_synthetic \
  --epochs 1 \
  --batch_size 32 \
  --train_size 10000 \
  --val_size 500 \
  --layers 6 \
  --dim 384 \
  --patch_size 32 \
  --output_dir checkpoints \
  --log_dir logs \
  "$@"

#!/bin/bash
# Phase 5 v2 — Continued Chinese fine-tuning with expanded templates
# Usage: ./run_phase5_v2.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/../.venv/Scripts/activate"
export PYTHONIOENCODING=utf-8

echo "=== Phase 5 v2 — Chinese Fine-tuning (continued) ==="
echo "Resuming from Phase 5 best checkpoint"
echo "20 epochs, LR=1e-4, no warmup"
echo "Expanded Chinese templates + 1510 token vocab"
echo ""

python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

python src/finetune_cn.py \
  --resume checkpoints_phase5/best.pt \
  --epochs 20 \
  --batch_size 24 \
  --lr 1e-4 \
  --no_warmup \
  --train_size 5000 \
  --val_size 200 \
  --aud_train_size 2000 \
  --aud_val_size 100 \
  --vid_train_size 2000 \
  --vid_val_size 100 \
  --max_text_len 48 \
  --output_dir checkpoints_phase5_v2 \
  --log_dir logs_phase5_v2 \
  "$@"

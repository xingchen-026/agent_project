#!/bin/bash
# Run Tiny Multimodal Training
# Usage: ./run.sh [options]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use shared mvp-venv
VENV="$(dirname "$SCRIPT_DIR")/../mvp-venv"
if [ ! -f "$VENV/bin/python3" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Quick dep check
python3 -c "import torch" 2>/dev/null || pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
python3 -c "import tqdm" 2>/dev/null || pip install tqdm -q

echo "=== Tiny Multimodal Training ==="
echo ""

# Default: synthetic data, moderate run (no download needed)
python src/train.py \
  --use_synthetic \
  --epochs 10 \
  --batch_size 8 \
  --train_size 10000 \
  --val_size 500 \
  --layers 6 \
  --dim 384 \
  --patch_size 32 \
  --output_dir checkpoints \
  --log_dir logs \
  "$@"

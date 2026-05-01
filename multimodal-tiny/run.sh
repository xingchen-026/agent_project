#!/bin/bash
# Run Tiny Multimodal Training
# Usage: ./run.sh [options]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
VENV_DIR="../mvp-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install deps if missing
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null
pip install transformers datasets pillow tqdm -q 2>/dev/null

echo "=== Tiny Multimodal Training ==="
echo ""

# Default: moderate run
python src/train.py \
  --epochs 10 \
  --batch_size 8 \
  --train_size 10000 \
  --val_size 500 \
  --layers 4 \
  --dim 384 \
  --patch_size 32 \
  --output_dir checkpoints \
  --log_dir logs \
  "$@"

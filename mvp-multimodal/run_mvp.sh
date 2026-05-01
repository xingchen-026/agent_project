#!/bin/bash
# Run the multimodal MVP training
# Usage: ./run_mvp.sh [options]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source ../mvp-venv/bin/activate

echo "=== Tiny Unified Multimodal Transformer MVP ==="
echo "Model: ~30M params | CPU training | COCO val2017"
echo ""

# Default: small run for quick validation
python train_mvp.py \
  --epochs 5 \
  --batch_size 4 \
  --data_size 2000 \
  --max_text_len 48 \
  "$@"

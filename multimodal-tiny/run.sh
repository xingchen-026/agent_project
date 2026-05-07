#!/bin/bash
# Multimodal Tiny — Unified Training Launcher
# Usage:
#   ./run.sh phase2      # Phase 2 (text+image gen, CPU)
#   ./run.sh phase4      # Phase 4 (text+img+audio+video, GPU)
#   ./run.sh phase4_5    # Phase 4.5 (balanced all-modality)
#   ./run.sh phase5      # Phase 5 (Chinese fine-tuning)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/../.venv/Scripts/activate"

PHASE="${1:-phase2}"
shift 2>/dev/null

case "$PHASE" in
  phase2)
    echo "=== Phase 2: Text+Image Generation ==="
    python src/train.py \
      --use_synthetic --epochs 10 --batch_size 8 \
      --train_size 10000 --val_size 500 \
      --img_gen --img_loss_weight 1.0 --img_decoder_hidden 512 \
      --output_dir checkpoints --log_dir logs \
      "$@"
    ;;
  phase3)
    echo "=== Phase 3: Text+Image+Audio ==="
    export CUDA_VISIBLE_DEVICES=0
    python src/train.py \
      --use_synthetic --use_audio \
      --epochs 8 --batch_size 32 \
      --train_size 10000 --val_size 500 \
      --aud_train_size 5000 --aud_val_size 200 \
      --img_loss_weight 0.5 --aud_loss_weight 0.5 \
      --output_dir checkpoints_phase3a --log_dir logs_phase3a \
      --resume checkpoints/best.pt \
      "$@"
    ;;
  phase4)
    echo "=== Phase 4: Text+Image+Audio+Video ==="
    export CUDA_VISIBLE_DEVICES=0
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || true
    python src/train.py \
      --use_synthetic --use_audio --use_video \
      --epochs 10 --batch_size 24 \
      --train_size 10000 --val_size 500 \
      --aud_train_size 5000 --aud_val_size 200 \
      --vid_train_size 3000 --vid_val_size 100 \
      --aud_loss_weight 0.5 --vid_loss_weight 0.5 --img_loss_weight 0.5 \
      --output_dir checkpoints_phase4 --log_dir logs_phase4 \
      --resume checkpoints_phase3a/best.pt \
      "$@"
    ;;
  phase4_5)
    echo "=== Phase 4.5: Balanced All-Modality ==="
    export CUDA_VISIBLE_DEVICES=0
    python src/train.py \
      --use_synthetic --use_audio --use_video \
      --epochs 5 --batch_size 24 \
      --train_size 10000 --val_size 500 \
      --aud_train_size 5000 --aud_val_size 200 \
      --vid_train_size 5000 --vid_val_size 200 \
      --aud_loss_weight 0.5 --vid_loss_weight 1.0 --img_loss_weight 0.5 \
      --output_dir checkpoints_phase4_5 --log_dir logs_phase4_5 \
      --resume checkpoints_phase4/best.pt \
      "$@"
    ;;
  phase5)
    echo "=== Phase 5: Chinese Fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0
    PYTHONIOENCODING=utf-8 python src/scripts/finetune_cn.py \
      --resume checkpoints_phase4_5/best.pt --epochs 10 \
      "$@"
    ;;
  phase5_v2)
    echo "=== Phase 5 v2: Continued Chinese Fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONIOENCODING=utf-8
    python src/scripts/finetune_cn.py \
      --resume checkpoints_phase5/best.pt --epochs 20 \
      --batch_size 24 --lr 1e-4 --no_warmup \
      --train_size 5000 --val_size 200 \
      --aud_train_size 2000 --aud_val_size 100 \
      --vid_train_size 2000 --vid_val_size 100 \
      --max_text_len 48 \
      --output_dir checkpoints_phase5_v2 --log_dir logs_phase5_v2 \
      "$@"
    ;;
  phase6)
    echo "=== Phase 6: Scaled-Up Architecture (448d-8L-7h, ~31M) ==="
    export CUDA_VISIBLE_DEVICES=0
    python src/train.py \
      --use_synthetic --use_audio --use_video \
      --dim 448 --layers 8 --n_heads 7 \
      --epochs 15 --batch_size 16 \
      --train_size 10000 --val_size 500 \
      --aud_train_size 5000 --aud_val_size 200 \
      --vid_train_size 3000 --vid_val_size 100 \
      --aud_loss_weight 0.5 --vid_loss_weight 0.5 --img_loss_weight 0.5 \
      --output_dir checkpoints_phase6 --log_dir logs_phase6 \
      "$@"
    ;;
  phase6_cn)
    echo "=== Phase 6 + COCO-CN: Real Chinese Image Fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONIOENCODING=utf-8
    python src/scripts/finetune_coco_cn.py \
      --resume checkpoints_phase6/best.pt --epochs 15 \
      "$@"
    ;;
  phase6_vqa)
    echo "=== Phase 6 + VQA: Chinese Instruction Tuning ==="
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONIOENCODING=utf-8
    python src/scripts/finetune_vqa.py \
      --resume checkpoints_phase6_cn/best.pt --epochs 5 \
      "$@"
    ;;
  phase6_clip)
    echo "=== Phase 6: CLIP Contrastive Pre-training ==="
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONIOENCODING=utf-8
    python src/scripts/train.py --mode clip \
      --resume checkpoints_phase6/best.pt --epochs 10 \
      "$@"
    ;;
  phase6_joint)
    echo "=== Phase 6: CLIP+LM+Diffusion Joint Training ==="
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONIOENCODING=utf-8
    python src/scripts/train.py --mode joint \
      --resume checkpoints_phase6/best.pt --epochs 15 \
      "$@"
    ;;
  phase6_full)
    echo "=== Phase 6: Full Multi-Modal Joint Training (Image+Audio+Video) ==="
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONIOENCODING=utf-8
    python src/scripts/train.py --mode full \
      --resume checkpoints_phase6/best.pt --epochs 20 \
      "$@"
    ;;
  phase6_distill)
    echo "=== Phase 6: Knowledge Distillation (ResNet50 -> MemoryBank) ==="
    export CUDA_VISIBLE_DEVICES=0
    python src/scripts/train.py --mode distill \
      --resume checkpoints_phase6/best.pt --epochs 10 \
      "$@"
    ;;
  phase6_audio_clip)
    echo "=== Phase 6: Audio-CLIP Contrastive (ESC-50) ==="
    export CUDA_VISIBLE_DEVICES=0
    python src/scripts/train.py --mode audio_clip \
      --resume checkpoints_phase6/best.pt --epochs 10 \
      "$@"
    ;;
  *)
    echo "Usage: ./run.sh {phase2|...|phase6|phase6_cn|phase6_vqa|phase6_joint|phase6_full|phase6_clip|phase6_distill|phase6_audio_clip} [extra args]"
    exit 1
    ;;
esac

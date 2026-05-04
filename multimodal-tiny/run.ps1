# Multimodal Tiny — Unified Training Launcher (Windows PowerShell)
# Usage:
#   .\run.ps1 phase2      # Phase 2 (text+image gen, CPU)
#   .\run.ps1 phase4      # Phase 4 (text+img+audio+video, GPU)
#   .\run.ps1 phase4_5    # Phase 4.5 (balanced all-modality)
#   .\run.ps1 phase5      # Phase 5 (Chinese fine-tuning)
param([string]$Phase = "phase2", [string[]]$ExtraArgs)

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR
. "$SCRIPT_DIR\..\.venv\Scripts\Activate.ps1"
$env:PYTHONIOENCODING = "utf-8"

switch ($Phase) {
  "phase2" {
    Write-Host "=== Phase 2: Text+Image Generation ==="
    python src/train.py `
      --use_synthetic --epochs 10 --batch_size 8 `
      --train_size 10000 --val_size 500 `
      --img_gen --img_loss_weight 1.0 --img_decoder_hidden 512 `
      --output_dir checkpoints --log_dir logs `
      @ExtraArgs
  }
  "phase3" {
    Write-Host "=== Phase 3: Text+Image+Audio ==="
    python src/train.py `
      --use_synthetic --use_audio `
      --epochs 8 --batch_size 32 `
      --train_size 10000 --val_size 500 `
      --aud_train_size 5000 --aud_val_size 200 `
      --img_loss_weight 0.5 --aud_loss_weight 0.5 `
      --output_dir checkpoints_phase3a --log_dir logs_phase3a `
      --resume checkpoints/best.pt `
      @ExtraArgs
  }
  "phase4" {
    Write-Host "=== Phase 4: Text+Image+Audio+Video ==="
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python src/train.py `
      --use_synthetic --use_audio --use_video `
      --epochs 10 --batch_size 24 `
      --train_size 10000 --val_size 500 `
      --aud_train_size 5000 --aud_val_size 200 `
      --vid_train_size 3000 --vid_val_size 100 `
      --aud_loss_weight 0.5 --vid_loss_weight 0.5 --img_loss_weight 0.5 `
      --output_dir checkpoints_phase4 --log_dir logs_phase4 `
      --resume checkpoints_phase3a/best.pt `
      @ExtraArgs
  }
  "phase4_5" {
    Write-Host "=== Phase 4.5: Balanced All-Modality ==="
    python src/train.py `
      --use_synthetic --use_audio --use_video `
      --epochs 5 --batch_size 24 `
      --train_size 10000 --val_size 500 `
      --aud_train_size 5000 --aud_val_size 200 `
      --vid_train_size 5000 --vid_val_size 200 `
      --aud_loss_weight 0.5 --vid_loss_weight 1.0 --img_loss_weight 0.5 `
      --output_dir checkpoints_phase4_5 --log_dir logs_phase4_5 `
      --resume checkpoints_phase4/best.pt `
      @ExtraArgs
  }
  "phase5" {
    Write-Host "=== Phase 5: Chinese Fine-tuning ==="
    python src/finetune_cn.py `
      --resume checkpoints_phase4_5/best.pt --epochs 10 `
      @ExtraArgs
  }
  "phase5_v2" {
    Write-Host "=== Phase 5 v2: Continued Chinese Fine-tuning ==="
    python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
    python src/finetune_cn.py `
      --resume checkpoints_phase5/best.pt `
      --epochs 20 --batch_size 24 --lr 1e-4 --no_warmup `
      --train_size 5000 --val_size 200 `
      --aud_train_size 2000 --aud_val_size 100 `
      --vid_train_size 2000 --vid_val_size 100 `
      --max_text_len 48 `
      --output_dir checkpoints_phase5_v2 --log_dir logs_phase5_v2 `
      @ExtraArgs
  }
  "phase6" {
    Write-Host "=== Phase 6: Scaled-Up Architecture (448d-8L-7h, ~31M) ==="
    python src/train.py `
      --use_synthetic --use_audio --use_video `
      --dim 448 --layers 8 --n_heads 7 `
      --epochs 15 --batch_size 16 `
      --train_size 10000 --val_size 500 `
      --aud_train_size 5000 --aud_val_size 200 `
      --vid_train_size 3000 --vid_val_size 100 `
      --aud_loss_weight 0.5 --vid_loss_weight 0.5 --img_loss_weight 0.5 `
      --output_dir checkpoints_phase6 --log_dir logs_phase6 `
      @ExtraArgs
  }
  default {
    Write-Host "Usage: .\run.ps1 {phase2|phase3|phase4|phase4_5|phase5|phase5_v2|phase6} [extra args]"
  }
}

# Multimodal Tiny

**用小参数模型实现原生全模态输入输出（文本+图像+音频+视频+中文）**

## 项目状态

当前最佳：**Phase 5 v2** — 18.88M 参数，全模态 + 中文，val_text=0.048

| 里程碑 | 目标 | 状态 |
|--------|------|:----:|
| 中文 val_text ≤ 0.05 | 提升中文文本生成 | ✅ 0.048 |
| COCO BLEU-1 ≥ 0.15 | 真实图像描述 | ❌ 合成数据限制 |
| 模型 ≤ 20MB (INT8) | 量化部署 | ⚠️ 32MB (FP16=36MB 推荐) |

## 环境

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
pip install torch torchvision torchaudio matplotlib tqdm numpy scipy pycocotools
```

## 快速开始

```powershell
# 训练
cd multimodal-tiny
.\run.ps1 phase5_v2          # 中文微调 (合成数据)
.\run.ps1 phase4             # 全模态从零训练

# 评估
cd src
$env:PYTHONIOENCODING = "utf-8"
python eval_all.py --checkpoint ..\checkpoints_phase5_v2\best.pt
python inference_demo.py --checkpoint ..\checkpoints_phase5_v2\best.pt --test-all

# COCO-CN 真实中文微调
python finetune_coco_cn.py --resume ..\checkpoints_phase5_v2\best.pt --epochs 10
```

## 项目结构

```
src/
├── model.py              # 模型架构 (SwiGLU, RoPE, RMSNorm, 4模态)
├── tokenizer.py           # SimpleTokenizer (1510 tokens, 中文ngram)
├── train.py               # Phase 1-4 训练
├── finetune_cn.py         # Phase 5 中文微调 (合成数据)
├── finetune_coco_cn.py    # Phase 5 v3 COCO-CN 真实中文微调
├── utils.py / train_utils.py  # 工具函数
├── synthetic_data.py      # 合成图像
├── audio_synthetic.py     # 合成音频
├── video_synthetic.py     # 合成视频
├── cn_data.py             # 中文标注 (原生模板, 10/23/10 句式)
├── eval_all.py            # 全模态评估+可视化
├── eval_coco.py           # COCO 真实图像评估 (BLEU/ROUGE)
├── eval_audio.py          # ESC-50 真实音频评估
├── eval_retrieval.py      # 跨模态检索评估
├── quantize_eval.py       # INT8/FP16 量化对比
└── inference_demo.py      # 交互推理+测试套件
```

## 阶段演进

| Phase | 内容 | val_text | 备注 |
|-------|------|:------:|------|
| 1 | text+image | 0.18 | 基础架构 |
| 2 | +image gen head | 0.127 | 图像重建 |
| 3 | +audio | — | 音频模态 |
| 4 | +video | 0.037 | 视频模态 |
| 4.5 | balanced all | 0.036 | 英文最佳 |
| 5 | +Chinese | 0.079 | 首次中文 |
| **5 v2** | **中文模板+token+20ep** | **0.048** | **当前最佳** |
| 5 v3 | COCO-CN 真实中文 | TBD | 训练中 |

## 技术架构

- **Transformer**: 6层 SwiGLU, 384维, 6头, RMSNorm, RoPE (FP16兼容)
- **图像**: 224×224 → 7×7=49 patch, 32px patch size
- **音频**: 16kHz, 128 mel bands, patch 16×16 梅尔时频
- **视频**: 4帧 × 64×64, patch 3×2×16×16 时空
- **文本**: 1510 token vocab (中英字符+ngram)
- **序列**: [video | image | audio | text], 因果+感官双向注意力

## 评估结果 (Phase 5 v2)

| 任务 | 指标 | 结果 |
|------|------|:----:|
| COCO 图像重建 | PSNR | 14.2 dB |
| COCO 文本生成 | BLEU-1 | 0.00 |
| ESC-50 音频 | MSE ratio | 7.6× |
| 跨模态检索 | Recall@1 | 3.0% |
| FP16 量化 | 大小/精度 | 36MB / 零损失 |

## 数据清单

| 数据集 | 路径 | 数量 |
|--------|------|:---:|
| COCO val2017 | `coco_data/val2017/` | 5000 |
| COCO-CN 中文标注 | `coco_data/coco-cn-master/data/` | 4712 |
| COCO val2014 | `coco_data/val2014/` | 1563 |
| ESC-50 环境音 | `esc50_data/audio/` | 2000 |

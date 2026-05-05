# Multimodal Tiny

**30.97M 参数原生多模态模型 — 文本 + 图像 + 音频 + 视频 + 中文**

## 项目状态

当前：**Phase 6+ Generation** — CLIP + MoE + Diffusion + DPO 全能力

| 能力 | 指标 | 状态 |
|------|------|:----:|
| 基座多模态 | 448d-8L-7h, text+image+audio+video | ✅ |
| CLIP 对比学习 | R@1=20% (COCO) | ✅ |
| 知识蒸馏 | ResNet50→MemoryBank cos=0.45 | ✅ |
| Diffusion 解码 | Latent DDIM 10 步采样 | ✅ |
| MoE 推理 | 8 专家, 2.3× 参数/FLOPs | ✅ |
| DPO 对齐 | val_acc=65.4% | ✅ |
| 中文支持 | 1510 tokens, COCO-CN, VQA | ✅ |
| 全模态联合训练 | val=1.24 (20 epoch) | ✅ |

## 环境

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
pip install torch torchvision torchaudio matplotlib tqdm numpy scipy pycocotools pillow
```

## 快速开始

```powershell
cd multimodal-tiny

# ── 训练 ──
.\run.ps1 phase6           # Phase 6 基座 (从零)
.\run.ps1 phase6_full      # 全模态联合 (图像+音频+视频, 推荐)
.\run.ps1 phase6_joint     # CLIP+LM+Diffusion
.\run.ps1 phase6_clip      # CLIP 对比预训练
.\run.ps1 phase6_distill   # 知识蒸馏

# ── 统一入口 ──
cd src
python train_unified.py --mode full --resume ../checkpoints_phase6/best.pt --epochs 20

# ── DPO 训练 ──
python train_dpo.py --resume ../checkpoints_phase6_full/best.pt --epochs 5

# ── 评估 ──
python eval_all.py --checkpoint ../checkpoints_phase6_full/best.pt
python eval_coco.py --checkpoint ../checkpoints_phase6_full/best.pt --coco-dir ../coco_data --max-images 200
```

## 技术架构

- **Transformer**: 8 层 SwiGLU (MoE 可选), 448 维, 7 头, RMSNorm, RoPE, QK Norm
- **图像**: 224×224 → 7×7=49 patch, 32px, 3 通道
- **音频**: 16kHz, 128 mel bands, patch 16×16 梅尔时频
- **视频**: 4 帧 × 64×64, patch 3×2×16×16 时空
- **文本**: 1510 token vocab (中英字符+ngram)
- **MemoryBank**: 16 token 感知器压缩
- **序列**: `[video | image | audio | text]`, 因果+双向注意力

## 项目结构

```
src/
├── model.py              # 核心模型 (753 行)
├── _components.py        # RMSNorm, RoPE, SwiGLU, MoE
├── _attention.py         # SelfAttention + TransformerBlock
├── _memory.py            # MemoryBank + DiffusionImageDecoder
├── config.py             # ModelConfig 数据中心
├── tokenizer.py          # SimpleTokenizer (1510 tokens)
│
├── train_unified.py      # ★ 统一训练入口 (--mode)
├── train_dpo.py          # DPO 对齐训练
│
├── losses.py             # ★ 损失函数 + 评估指标
├── data_lib.py           # ★ 数据集 + 预处理 + 校对
├── training.py           # ★ 优化器/调度器/checkpoint
├── eval_lib.py           # ★ 评估基础设施
│
├── synthetic_data.py     # 合成图像生成
├── audio_synthetic.py    # 合成音频生成
├── video_synthetic.py    # 合成视频生成
├── cn_data.py            # 中文标注模板 (457 行)
├── utils.py              # checkpoint 加载工具
│
├── finetune_cn.py        # 中文微调 (合成)
├── finetune_coco_cn.py   # COCO-CN 真实中文微调
├── finetune_vqa.py       # VQA 指令微调
│
├── eval/                 # 评估脚本
│   ├── eval_all.py       #   全模态评估+可视化
│   ├── eval_coco.py      #   COCO 真实图像 (BLEU/ROUGE)
│   ├── eval_audio.py     #   ESC-50 音频
│   ├── eval_retrieval.py #   跨模态检索
│   ├── quantize_eval.py  #   INT8/FP16 量化
│   └── benchmark_compile.py  # torch.compile 基准
│
└── demo/                 # 演示
    ├── inference_demo.py #   交互 CLI + 测试套件
    ├── demo_vqa.py       #   VQA 演示
    └── web_app.py        #   Gradio Web UI
```

## 阶段演进

| Phase | 内容 | 参数 | 关键指标 |
|-------|------|:---:|------|
| 1-5 | 基础多模态 + 中文 | 18.88M | text=0.048 |
| 6 | 架构扩展 (448d-8L-7h) | 30.97M | text=0.012 |
| +CLIP | 对比预训练 | +256d proj | R@1=20% |
| +Distill | 知识蒸馏 | +distill head | cos=0.45 |
| +MoE | 混合专家 | +8 experts | 2.3× params |
| +Diffusion | 图像生成 | +DDIM | 10 step |
| **+Full** | **全模态联合** | **30.97M** | **val=1.24** |
| +DPO | 偏好对齐 | — | acc=65.4% |

## 评估结果

| 任务 | 指标 | Phase 5 v2 | Phase 6+ Full |
|------|------|:---:|:---:|
| 文本生成 (合成) | val_loss | 0.048 | 0.012 |
| COCO 图像重建 | PSNR | 14.2 dB | — |
| COCO 文本生成 | LM loss | — | 0.53 |
| ESC-50 音频重建 | MSE | 0.00024 | 0.000006 |
| 跨模态检索 | Recall@1 | 3.0% | 20.0% |
| DPO 偏好 | val_acc | — | 65.4% |
| FP16 量化 | 大小 | 36 MB | — |

## 数据清单

| 数据集 | 路径 | 数量 |
|--------|------|:---:|
| COCO val2017 | `coco_data/val2017/` | 5000 |
| COCO val2017 captions | `coco_data/captions_val2017.json` | 25010 |
| COCO val2014 + train2014 | `coco_data/` | 1563+ |
| COCO-CN 中文标注 | `coco_data/coco-cn-master/data/` | 4712 |
| ESC-50 环境音 | `esc50_data/audio/` | 2000 |

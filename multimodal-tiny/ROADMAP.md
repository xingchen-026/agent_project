# 开发路线图

当前状态：**Phase 6+ Generation** — 30.97M 参数，全模态 + 中文 + CLIP + MoE + Diffusion + DPO

## 完成情况总览

### 方向一：架构增强 ✅

| 任务 | 状态 | 结果 |
|------|:----:|------|
| 1.1 Phase 6 扩展 (448d-8L-7h) | ✅ | 18.88M → 30.97M, val_text=0.012 |
| 1.2 MoE 混合专家 | ✅ | 8 专家 top-2 门控, 2.3× 参数同 FLOPs |
| 1.3 MemoryBank 记忆压缩 | ✅ | 16 token 感知器式交叉注意力 |
| 1.4 KV Cache + 推测性解码 | ✅ | 多轮对话, 3 步 draft 解码 |
| 1.5 Diffusion 图像解码器 | ✅ | Latent DDIM, 1000 步训练/10 步采样 |

### 方向二：理解 + 对齐 ✅

| 任务 | 状态 | 结果 |
|------|:----:|------|
| 2.1 CLIP 对比预训练 | ✅ | R@1=20% (随机基线 1%), 10 epochs |
| 2.2 ResNet50 知识蒸馏 | ✅ | cos=0.45 (基线 0.41), MemoryBank 对齐 |
| 2.3 联合训练 (CLIP+LM+Diff) | ✅ | CLIP 98%, LM 67%, val 37% 提升 |
| 2.4 全模态联合训练 | ✅ | 20 epoch, img=0.53, aud=0.006, vid=0.002, val=1.24 |
| 2.5 DPO 对齐训练 | ✅ | val_acc=65.4% (随机 50%), 5 epochs |

### 方向三：中文效果 ✅

| 任务 | 状态 | 结果 |
|------|:----:|------|
| 3.1 中文模板 + token | ✅ | 1510 tokens, 10/23/10 句式 |
| 3.2 COCO-CN 真实中文 | ✅ | val_loss=2.03 (1500 张图) |
| 3.3 VQA 指令微调 | ✅ | 10 种中文问答模板 |

### 方向四：模块整合 ✅

| 任务 | 状态 | 结果 |
|------|:----:|------|
| 4.1 共享库模块 | ✅ | losses/data_lib/training/eval_lib (1731 行) |
| 4.2 统一训练入口 | ✅ | train_unified.py --mode (full/joint/clip/distill/base) |
| 4.3 运行脚本同步 | ✅ | run.ps1/run.sh 一致 |

### 里程碑

| 里程碑 | 目标 | 实际 |
|--------|------|:----:|
| 中文 val_text ≤ 0.05 | 中文效果 | ✅ 0.048 |
| 全模态联合训练 | 真实 COCO 数据 | ✅ val=1.24 |
| DPO 偏好对齐 | val_acc > 60% | ✅ 65.4% |
| COCO BLEU-1 ≥ 0.15 | 图像描述 | ❌ 合成数据限制 |
| INT8 ≤ 20MB | 量化部署 | ⚠️ 32MB |
| 模块整合 | 消除重复代码 | ✅ 10→1 脚本 |

---

## Phase 6+ 全模态联合训练结果 (20 epochs)

| Epoch | img_lm | aud_lm | vid_lm | val_loss |
|-------|--------|--------|--------|----------|
| 1 | 2.50 | 0.41 | 0.26 | 2.04 |
| 5 | 0.92 | 0.007 | 0.022 | 1.34 |
| 10 | 0.72 | 0.006 | 0.009 | 1.24 |
| 15 | 0.59 | 0.006 | 0.003 | 1.24 |
| 20 | 0.53 | 0.006 | 0.002 | 1.24 |

最佳 val_loss=1.23 (epoch 13), checkpoint: `checkpoints_phase6_full/best.pt`

## DPO 对齐训练结果 (5 epochs)

| Epoch | train_loss | train_acc | val_acc |
|-------|-----------|-----------|---------|
| 1 | 0.691 | 55.1% | 64.4% |
| 3 | 0.484 | 82.1% | 65.4% |
| 5 | 0.411 | 89.4% | 63.5% |

最佳 val_acc=65.4%, checkpoint: `checkpoints_phase6_dpo/best.pt`

---

## 瓶颈分析

1. **生成质量受限**：模型 90% 训练数据为合成几何图形，COCO 真实文本仅 20 epochs
2. **LM 能力不足**：COCO 图像 LM loss=0.53，远高于合成数据 (0.012)，语义理解弱
3. **语言多样性低**：合成模板句式单一 (10 种)，远不及自然语言多样性

## 后续方向（建议优先级）

### P0: 提升生成质量
- 延长 COCO LM 预训练 50-100 epochs
- 混合更多真实数据集 (Flickr30k, TextCaps)
- 预期：BLEU-1 突破 0.0, DPO 效果显著提升

### P1: Audio-CLIP
- ESC-50 数据已就绪，桩代码 `train_audio_clip.py`
- 在 `train_unified.py` 添加 `--mode audio_clip`

### P2: Eval/Demo 迁移
- `eval_lib.py` 已创建共享函数
- 旧 eval/demo 脚本迁移为薄包装

### P3: 推理优化
- INT8 ≤ 20MB, ONNX 导出, 流式推理

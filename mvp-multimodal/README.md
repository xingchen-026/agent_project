# MVP: Tiny Unified Multimodal Transformer

**目标：验证小参数模型的原生多模态可行性**

一个 <30M 参数的 Transformer，统一处理文本和图像 token，纯自回归 next-token prediction。

## 架构

```
图像 (224x224 RGB)
    ↓
ViT-style Patch Embedding (32x32 patches → 7×7=49 tokens)
    ↓
文本 (BPE tokens, GPT-2 vocab)
    ↓
[49 图像 tokens | text tokens] → 4层 Transformer (384d, 6头, RoPE)
    ↓
predict next text token
```

- **~30M 参数**，全 CPU 可训
- **RoPE 位置编码**（混合序列友好）
- **因果注意力** + 图像区域全可见
- **可学习类型嵌入**（text vs image）

## 用法

```bash
# 激活环境
source ../mvp-venv/bin/activate

# 训练（CPU）
python train_mvp.py --epochs 5 --batch_size 4 --data_size 2000

# 更多选项
python train_mvp.py --help
```

## 数据

自动下载 COCO val2017 数据集（~5GB）。

## 项目结构

```
mvp-multimodal/
├── train_mvp.py       # 训练脚本
├── checkpoints/       # 模型检查点
├── data/coco/         # COCO 数据
└── README.md
```

## 验证指标

- 训练 loss 从 ~10.8 → 稳定下降
- 每个 epoch 后生成样本描述
- 最终：模型能在给定图像后预测合理的文本 tokens

## 参考论文

- Chameleon (arXiv:2405.09818) — 早融合多模态
- Show-o (arXiv:2408.12528) — 统一 AR+扩散
- MobileLLM (arXiv:2402.14905) — 小模型架构
- Emu3 (BAAI 2024) — 纯自回归全模态

# Multimodal Tiny

**用小参数模型实现原生全模态输入输出（文本+图像）**

## 项目目标
在 20GB 数据预算和 CPU-only 训练条件下，验证 <30M 参数模型的原生全模态可行性。

## 约束
- 训练数据存储：≤ 20GB
- 训练设备：CPU (Intel 16核)
- 推理环境：CPU，最终目标边缘设备

## 数据策略
- COCO val2017 (~5GB): 评估集
- COCO train2017 subset (~10GB): 训练集
- 合成数据: 程序生成几何图形+描述 (0下载)
- 总计: ~15GB (budget 20GB)

## 项目结构
```
src/
├── model.py          # 模型架构 (v2.1)
├── data.py           # 数据加载 (COCO)
├── synthetic_data.py # 合成数据生成 (几何图形)
├── tokenizer.py      # 本地 tokenizer (0下载)
├── train.py          # 训练循环 (Phase 1+2)
└── evaluate.py       # 评估与生成测试
```

## 阶段

### Phase 1 ✅ — 基础架构 + 训练 (文本+图像)
- 6层 SwiGLU Transformer, 384维, RoPE, QK归一化
- 图像 patch (224/32=7, 49 patches) + 文本 token 序列融合
- 因果注意力掩码: 图像区全连接, 文本区因果
- 合成数据训 10 epoch: val_loss 0.18
- 参数: 15.79M

### Phase 2 🚧 — 多模态生成 (当前)
- 添加图像解码头 (patch级像素重建)
- 联合损失训练: 文本 CE + 图像 MSE
- 文本↔图像双生成模式
- 图像占位 token 支持从文本生成图像
- 支持从 Phase 1 checkpoint 迁移学习

### Phase 3 (计划) — 音频/视频扩展
### Phase 4 (计划) — 量化 + 部署

## 使用方法

### 训练 (Phase 2, 从 Phase 1 继续)
```bash
# Phase 2 联合训练 (从最佳 checkpoint 继续)
./run.sh --resume checkpoints/best.pt

# 从头开始 Phase 2 训练
./run.sh
```

### 训练 (Phase 1, 纯文本损失)
```bash
python src/train.py --use_synthetic --epochs 10 --batch_size 32
```

### 评估
```bash
# 全部测试
python src/evaluate.py --checkpoint checkpoints/best.pt

# 只测试文本生成
python src/evaluate.py --test text_gen

# 只测试图像重建
python src/evaluate.py --test recon

# 只测试文本→图像生成
python src/evaluate.py --test t2i
```

### GPU 训练
```bash
./run_gpu.sh
```

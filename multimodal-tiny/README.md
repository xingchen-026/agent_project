# Multimodal Tiny

**用小参数模型实现原生全模态输入输出（文本+图像+音频+视频）**

## 项目目标
在 20GB 数据预算和 CPU-only 训练条件下，验证 <30M 参数模型的原生全模态可行性。

## 约束
- 训练数据存储：≤ 20GB
- 训练设备：CPU (Intel 16核)
- 推理环境：CPU，最终目标边缘设备

## 数据策略
- COCO val2017 (~5GB): 评估集
- COCO train2017 subset (~10GB): 训练集
- 总计: ~15GB (budget 20GB)

## 项目结构
```
src/
├── model.py          # 模型架构
├── data.py           # 数据加载
├── train.py          # 训练循环
├── config.py         # 配置
├── evaluate.py       # 评估
└── utils.py          # 工具
```

## 阶段
- Phase 1: 基础架构 + 训练 (文本+图像)
- Phase 2: 多模态生成 (加生成头)
- Phase 3: 音频/视频扩展
- Phase 4: 量化 + 部署

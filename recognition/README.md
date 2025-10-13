# Radiology Report Simplification

## 项目描述
本项目使用预训练的T5/FLAN-T5模型，通过微调将专业的放射学报告转换为普通人易于理解的摘要。项目基于BioLaySumm 2025数据集，使用LoRA进行参数高效微调。

## 问题描述
放射学报告包含大量专业医学术语，普通患者难以理解。本项目旨在通过自然语言处理技术，将这些专业报告转换为通俗易懂的文本，帮助患者更好地理解自己的医疗状况。

## 工作原理
1. **模型架构**: 使用encoder-decoder架构的T5/FLAN-T5模型
2. **微调策略**: 采用LoRA进行参数高效微调
3. **评估指标**: 使用ROUGE分数评估生成质量

## 依赖环境
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 其他依赖详见requirements.txt

## 使用方法

### 训练模型
```bash
python train.py
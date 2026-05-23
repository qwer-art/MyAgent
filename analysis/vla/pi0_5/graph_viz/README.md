# Pi0.5 架构框图索引

## 概述

Pi0.5 (Physical Intelligence) 是一个基于 PaliGemma 2B VLM + Action Expert 的机器人基础模型，使用 Flow Matching 进行动作去噪生成。

## 核心架构特点

1. **Dual-Expert Transformer**: PaliGemma 2B (frozen) 处理视觉语言，Action Expert 300M (trainable) 生成动作
2. **adaRMSNorm**: 时间条件通过自适应 RMSNorm 注入 Action Expert 的每层，而非简单拼接
3. **Flow Matching**: 使用线性插值 + Euler 积分器，10步去噪生成50步动作序列
4. **State Discretization**: 机器人状态离散化为 token 字符串，与语言 prompt 拼接输入 VLM

## 图的索引

### 顶层流程图

| 图 | 文件 | 说明 |
|----|------|------|
| 推理流程 | `inference.pdf` / `inference.png` | 从观测到动作的完整推理流程，含去噪循环 |
| 训练流程 | `training.pdf` / `training.png` | Flow Matching 训练：噪声采样→插值→loss |

### 子模块展开图 (modules/)

| 图 | 文件 | 说明 |
|----|------|------|
| Gemma Block | `gemma_block.pdf` / `gemma_block.png` | Dual-Expert Block 内部结构：PaliGemma + Action Expert 分支 |
| SigLIP ViT | `siglip_vit.pdf` / `siglip_vit.png` | 视觉编码器：So400m/14, 27层, 1152维 |
| adaRMSNorm | `adaRMSNorm.pdf` / `adaRMSNorm.png` | 时间条件自适应归一化，pi0.5核心创新 |

## 关键参数

- PaliGemma: width=2048, depth=18, 8Q/1KV heads, dim_head=256
- Action Expert: width=1024, FFN=4096, 8Q/1KV heads, dim_head=128
- SigLIP: width=1152, depth=27, 16 heads, patch=14, mlp_dim=4304
- Flow Matching: action_horizon=50, action_dim=32, denoise_steps=10
- LoRA: rank=32, alpha=32.0 (optional fine-tuning)
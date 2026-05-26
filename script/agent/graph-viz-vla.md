# Graph Viz VLA — VLA 框架推理/训练流程可视化

生成 VLA (Vision-Language-Action) 模型的推理和训练流程架构图，使用 graphviz 将数据流拆解为时间阶段 + 特征空间子分类的模块框图。

## 输入

用户需提供:
1. **VLA 框架名称** (如 pi0, pi0.5, openvla, rt2 等)
2. **模式**: `inference` (推理) 或 `training` (训练) 或 `both`
3. **源码路径** (可选，如有本地代码则提供，否则搜索已知的开源仓库)
4. **输出路径** (可选，默认在当前工作目录生成)

## 可视化规范

### 时间主轴

**推理**: Phase 1 (初始化, run ×1) → Phase 2 (递推, repeat ×N)
- Phase 1: 编码 prefix (image + language + state) → 建 KV Cache
- Phase 2: 注入 suffix (noisy action + timestep) → 复用 Cache → 去噪循环

**训练**: 单次前向 + Loss 计算 (无阶段分离)
- Joint Attention: prefix 和 suffix 在每层联合计算 (非分阶段)
- 插值构造: x_t = t·ε + (1-t)·a, 目标 u_t = ε - a
- Loss 回传: MSE(v_t, u_t) → 梯度流

### 6 类颜色 — 数据可解释性

| # | 类别 | 语义 | fillcolor | 边色 | shape |
|---|------|------|-----------|------|-------|
| 1 | 输入(观测空间) | 人可理解的具体数据: 图像/语言/状态/动作 | `#E8F5E9` | `#2E7D32` | box, rounded |
| 2 | KV Cache | 键值缓存, 存储 prefix 上下文, 有动态维度 | `#F3E5F5` | `#7B1FA2` | cylinder |
| 3 | 编码(特征向量) | 人不可理解的高维张量, dim≈1024/2048 | `#E3F2FD` | `#1565C0` | box, rounded |
| 4 | 融合 | 编码特征的交汇点, 见下方融合策略说明 | `#FFF3E0` | `#E65100` | box, rounded |
| 5 | 控制信号 | 时间步条件, 驱动去噪行为, 非观测来源 | `#FFFDE7` | `#F57F17` | box, rounded |
| 6 | 输出(动作空间) | 人可理解的具体动作: 7-DoF 序列 | `#FCE4EC` | `#C62828` | box, rounded |

**训练新增**:

| # | 类别 | 语义 | fillcolor | shape |
|---|------|------|-----------|-------|
| 7 | 采样 | 从分布中抽取: ε~N(0,I), t~Beta(α,β) | `#F1F8E9` | box, rounded |
| 8 | Loss | 训练目标: MSE(v_t, u_t) | `#FFCDD2` | box, rounded |
| 9 | 梯度边界 | 可训练 vs 冻结参数的分界 | 注释/边样式 | dashed line |

### 每个节点的标注规范

每个节点必须包含 3 项信息, 繁简有度:

1. **泛化处理方式** — 功能类别 (1~3 词), 如 `vision-encoder` / `traj-tokenizer` / `action-head`
2. **具体实现** — 行业术语或模型名 (1~3 词), 如 `SigLIP2` / `Qwen3-VL` / `DCT+BPE` / `FM (flow matching)`; 速查表见下方
3. **输出张量** — 名称 + shape, 如 `img_tokens, [B,160,1280]` / `traj_tokens, [B,48,1024]`

**格式**: 节点标签按 `泛化处理方式\n具体实现\n输出张量` 三行排列。

**示例**:
```
Vision Encoder
SigLIP2 + 2x downsample
img_tokens, [B,960,1280]
```
```
DeltaTrajectoryTokenizer
delta-quantize, vocab=1000
traj_tokens, [B,48,1024]
```

### 边的标注规范

每条边必须标注:
1. **张量维度** (传递的数据 shape)
2. **关键变化** (如 "uint8→float32", "HWC→CHW", "712 prefix tokens")

### 融合策略术语

融合节点不仅要标注泛称（如 "Token Fusion"），还必须注明具体的融合方式。常见策略:

| 术语 | 做法 | 代表模型 |
|------|------|----------|
| **Early Fusion** (via Token Concatenation) | 各模态 token 在进入 Transformer 前顺序拼接为单一序列, 由 self-attention 自学跨模态关系 | LLaVA, RT-2, Alpamayo |
| **Late Fusion** | 各模态独立编码, 在输出层融合特征向量 | 双流网络 |
| **Cross-Attention Fusion** | 一个模态通过 cross-attention 访问另一模态的 KV, 非对称交互 | Flamingo, Pi0 (action expert) |
| **Gated Fusion** | 可学习门控系数加权混合各模态特征 | — |

**节点标注要求**: 节点名称写泛称 (如 "Token Fusion"), 节点描述中用术语标注具体策略 (如 "Early Fusion via Token Concatenation: img\|text\|traj → input_ids")。

### 行业术语速查表

节点中涉及的模块如有行业共识的标准缩写, 必须在节点描述中标注 (格式: `缩写 全称`), 不可只用自造名称。按功能分类:

**视觉编码器:**

| 缩写 | 全称 | 典型用途 |
|------|------|----------|
| ViT | Vision Transformer | 通用视觉编码 backbone |
| SigLIP | Sigmoid Loss Language-Image Pre-training | 对比学习视觉编码, 替代 CLIP |
| CLIP | Contrastive Language-Image Pre-training | 对比学习视觉编码 |
| DINOv2 | Self-distillation with No Labels v2 | 自监督视觉编码, 无需文本 |
| ViTDet | Vision Transformer Detector | 检测专用 ViT |
| SAM | Segment Anything Model | 通用分割 |

**语言模型 backbone:**

| 缩写 | 全称 | 典型用途 |
|------|------|----------|
| LLM | Large Language Model | 通用语言 backbone |
| VLM | Vision-Language Model | 多模态语言 backbone |
| LLaMA | Large Language Model Meta AI | Meta 开源 LLM 系列 |
| Qwen-VL | 通义千问-VL | 阿里多模态 LLM |
| PaLM-E | Pathways Language Model - Embodied | Google 具身 LLM |

**动作/轨迹编码:**

| 缩写 | 全称 | 典型用途 |
|------|------|----------|
| FM | Flow Matching | 连续归一化流, 用于动作去噪 |
| DDPM | Denoising Diffusion Probabilistic Model | 离散扩散去噪 |
| CVAE | Conditional Variational Autoencoder | 条件 VAE, 动作建模 |
| ACT | Action Chunking with Transformers | 分块动作预测 |
| DiT | Diffusion Transformer | Transformer 架构的扩散模型 |

**位置/时间编码:**

| 缩写 | 全称 | 典型用途 |
|------|------|----------|
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| PE | Position Embedding | 位置编码 (泛称) |
| Fourier PE | Fourier Position Embedding | 傅里叶位置编码, 用于连续值 (时间步/坐标) |

**注意力/融合:**

| 缩写 | 全称 | 典型用途 |
|------|------|----------|
| GQA | Grouped-Query Attention | 分组查询注意力, KV 共享 |
| MLA | Multi-head Latent Attention | DeepSeek 多头潜注意力 |
| Cross-Attn | Cross-Attention | 跨模态注意力 |
| Self-Attn | Self-Attention | 自注意力 |
| Flash-Attn | Flash Attention | 高效注意力实现 |

**归一化/激活:**

| 缩写 | 全称 | 典型用途 |
|------|------|----------|
| RMSNorm | Root Mean Square Normalization | RMS 归一化, 替代 LayerNorm |
| adaLN | Adaptive Layer Normalization | 自适应归一化, 条件注入 |
| GeGLU | GeLU Gated Linear Unit | 门控激活函数 |
| SiLU | Sigmoid Linear Unit | SiLU 激活函数 |

**量化/分词:**

| 缩写 | 全称 | 典型用途 |
|------|------|----------|
| BPE | Byte-Pair Encoding | 文本分词 |
| VQ-VAE | Vector Quantized VAE | 离散化编码, 动作 token 化 |
| FSQ | Finite Scalar Quantization | 有限标量量化 |

**标注示例**: 节点名称 `Vision Encoder`, 描述中写 `SigLIP2 + 2x downsample` 而非仅写 `视觉编码器 + 下采样`。

### 跨阶段连接

- KV Cache: 紫色粗箭头, 标注 "frozen, clone per step"
- 梯度流: 红色虚线箭头, 标注哪些模块参与反传
- Joint Attention vs 两阶段分离: 用注释框对比

### 训练 vs 推理的关键差异 (必须在训练图中标注)

1. **Joint Attention vs 两阶段分离**: 训练中 prefix 和 suffix 每层联合计算 Q/K/V; 推理中分两阶段 + KV Cache
2. **GT Actions 分支**: 训练中 GT actions 参与插值 (构造 x_t) 和目标构造 (u_t = ε - a)
3. **采样策略**: t ~ Beta(1.5, 1.0) 偏向噪声端; ε ~ N(0,I)
4. **梯度边界**: freeze_vision_encoder / train_expert_only / LoRA 配置决定哪些参数可训练
5. **Loss**: MSE(v_t, u_t), 对所有去噪步取均值

## 执行步骤

1. **阅读源码**: 读取用户提供的 VLA 框架源码, 重点找:
   - `select_action` / `forward` 方法 (推理流程)
   - `compute_loss` / `forward` 方法 (训练流程)
   - `preprocess` / `postprocess` 方法 (数据变换)
   - Transformer / Attention 实现 (计算结构)

2. **提取数据流**: 沿时间主轴, 逐模块记录:
   - 输入张量 (shape, dtype, 值域)
   - 中间张量 (shape, dtype)
   - 输出张量 (shape, dtype, 值域)
   - 控制信号 (timestep, attention mask 等)

3. **生成 graphviz 代码**: 使用 Python + graphviz 库, 按 6/9 类颜色规范生成框图

4. **渲染输出**: 生成 PDF + PNG, 300 DPI

## 输出文件

- `{framework}_inference.py` / `{framework}_inference.png` / `.pdf`
- `{framework}_training.py` / `{framework}_training.png` / `.pdf`

## 代码模板

```python
from graphviz import Digraph

# 6 类颜色节点样式
INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'fontsize': '11'}
CACHE_NODE = {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#F3E5F5', 'fontsize': '12'}
ENCODING_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'fontsize': '11'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'fontsize': '11'}
CONTROL_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'fontsize': '11'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'fontsize': '12'}
# 训练新增
SAMPLE_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#F1F8E9', 'fontsize': '11'}
LOSS_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFCDD2', 'fontsize': '12'}

# 每个节点三行: 泛化处理方式 / 具体实现 / 输出张量
# 示例: "Vision Encoder\nSigLIP2 + 2x downsample\nimg_tokens, [B,960,1280]"
# 融合节点具体实现必须标注策略术语: Early Fusion / Cross-Attn Fusion / Late Fusion / Gated Fusion
# 每条边必须标注: 张量维度 + 关键变化
# 时间主轴: 推理用 Phase 1/2, 训练用单次前向+Loss
# 特征子分类: Image Stream / Language Stream / Action Stream / Time Control
```

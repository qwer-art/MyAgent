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
| 4 | 融合 | 编码特征的交汇点: concat/mask/插值/积分 | `#FFF3E0` | `#E65100` | box, rounded |
| 5 | 控制信号 | 时间步条件, 驱动去噪行为, 非观测来源 | `#FFFDE7` | `#F57F17` | box, rounded |
| 6 | 输出(动作空间) | 人可理解的具体动作: 7-DoF 序列 | `#FCE4EC` | `#C62828` | box, rounded |

**训练新增**:

| # | 类别 | 语义 | fillcolor | shape |
|---|------|------|-----------|-------|
| 7 | 采样 | 从分布中抽取: ε~N(0,I), t~Beta(α,β) | `#F1F8E9` | box, rounded |
| 8 | Loss | 训练目标: MSE(v_t, u_t) | `#FFCDD2` | box, rounded |
| 9 | 梯度边界 | 可训练 vs 冻结参数的分界 | 注释/边样式 | dashed line |

### 每个节点的标注规范

每个节点必须包含:
1. **模块名称** (人可理解的功能描述)
2. **输入→输出维度** (张量 shape, 含具体数值)
3. **数据范围** (值域, 如 [0,1], [-1,1], N(0,I) 等)
4. **关键操作** (如 GQA 8Q/1KV, GeGLU, adaRMSNorm 等结构细节)

### 边的标注规范

每条边必须标注:
1. **张量维度** (传递的数据 shape)
2. **关键变化** (如 "uint8→float32", "HWC→CHW", "712 prefix tokens")

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

# 每个节点必须标注: 模块名 + 输入→输出维度 + 值域 + 关键操作
# 每条边必须标注: 张量维度 + 关键变化
# 时间主轴: 推理用 Phase 1/2, 训练用单次前向+Loss
# 特征子分类: Image Stream / Language Stream / Action Stream / Time Control
```

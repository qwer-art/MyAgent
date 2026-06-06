# Graph Viz VLA — VLA 框架推理/训练流程可视化

## 目标

通用技能：为任意 VLA 框架生成 graphviz 架构图，让人 **30 秒内**看清：
- 输入什么、经过哪些模块、输出什么
- 关键张量维度多少
- 训练与推理有何不同

默认同时产出推理图和训练图，用于横向对比不同模型（pi0.5、openvla、alpamayo 等）。

## 输入

用户需提供:
1. **VLA 框架名称** (如 pi0, pi0.5, openvla, rt2, alpamayo 等)
2. **模式**: `inference` / `training` / `both` (默认 `both`)
3. **源码路径** (可选，有本地代码则提供，否则搜索开源仓库)
4. **输出路径** (可选，默认当前工作目录)

## 核心规范（5 条硬约束）

| # | 约束 | 说明 |
|---|------|------|
| 1 | **每节点 3 行** | `模块名` / `具体实现 (+dtype)` / `输出, [shape]` |
| 2 | **关键边带 shape** | 模态汇合、阶段/Stage 边界、最终输出处必须标注 shape；同流内部边可省略 |
| 3 | **时间/阶段主轴** | 推理: Phase1(prefill) → Phase2(denoise ×N)；训练: 按源码划 Stage cluster，标注 frozen/trainable |
| 4 | **维度速查** | 每张图一个 compact note，列出 5~8 个常量 (`D_model`, `L_prefix`, `act_dim`, `N_action` 等) |
| 5 | **5 色节点** | Input / Encode / Fusion / Output / Boundary (见下) |

其余为推荐项，不强制。主图建议 ≤15 个节点；过复杂时用户明确要求再拆详图。

## 图的结构

### 推理：Phase 1 → Phase 2

**Phase 1** (run ×1): 编码 prefix (image + language + state) → 建 KV Cache
**Phase 2** (repeat ×N): 注入 suffix (noisy action + timestep) → 复用 Cache → 去噪循环

Phase2 cluster label 写 `repeat ×N` 即可，无需 diamond 循环节点。

### 训练：单阶段 or 多 Stage 链

按源码实际训练流程划分，不强行套用推理的 Phase1/Phase2。

**单阶段** (如 pi0.5): 一个 cluster，label 注明 `Joint Attention, 无 KV Cache`，prefix + suffix 每层联合计算。

**多阶段** (如 alpamayo Stage2): 每个 Stage = 一个 dashed cluster，纵向串联:

```
Stage 1: VLM Prefill (frozen, no_grad)
  Inputs → Vision/Lang Encode → KV Cache (stop-grad)
       ↓ COND_EDGE: KV Cache / hidden states
Stage 2: Expert FM (trainable)
  Sample(ε, t) → FM Interpolation → Action Expert → v_θ
       ↓
Stage 3: Loss (可选独立 cluster)
  MSE(v_θ, u_t)
```

规则:
- 每个 Stage label 含 `(frozen)` / `(trainable)` / `(no_grad)`
- Stage 间用 `COND_EDGE` 标注传递物 (KV Cache、hidden states、GT actions)
- FM 插值公式从 `compute_loss` 提取，写入插值节点第 2 行

### 模态子流（推荐）

Phase/Stage 内部可用 dotted subgraph 区分模态，命名自由:
- Image Stream / Language Stream / Action Stream / Time Control

## 节点与边

### 5 色节点

| 类别 | 语义 | fillcolor | shape |
|------|------|-----------|-------|
| Input | 输入、训练采样 (ε, t) | `#E8F5E9` | box, rounded |
| Encode | 编码器、Transformer、KV Cache | `#E3F2FD` | box; Cache 用 cylinder `#F3E5F5` |
| Fusion | 融合、控制信号 (timestep) | `#FFF3E0` / Control `#FFFDE7` | box, rounded |
| Output | 动作输出、Loss | `#FCE4EC` / Loss `#FFCDD2` | box, rounded |
| Boundary | 梯度边界 (训练，单个 note) | 白底红虚线框 | box, dashed |

节点描述使用行业标准缩写 (如 SigLIP、GQA、FM、Cross-Attn)，融合节点第 2 行注明策略 (Early Fusion / Cross-Attn Fusion 等)。

**节点示例**:
```
Vision Encoder
SigLIP2 + 2x downsample
img_tokens, [B, N_img, D_vis]
```

```
Multi-Cam Images
uint8, RGB 0~255
[B, N_cam, 3, H, W]
```

### 3 种边

| 样式 | 线型 | 用于 |
|------|------|------|
| `DATA_EDGE` | 实线 `#333` | 张量数据流 |
| `COND_EDGE` | 虚线/粗线 `#7B1FA2` 或 `#F57F17` | 控制信号、KV Cache 跨阶段 |
| (无逐边梯度) | — | 训练图用单个 `grad_boundary` note 列出 frozen/trainable |

### 维度速查 note 示例

```
D_model=4096  D_vis=1280  L_prefix=712
act_dim=32  N_action=50  denoise_steps=10
```

## 训练 vs 推理差异（checklist）

训练图必须能让人看出以下差异（用 Stage label、grad_boundary note、COND_EDGE 标注）:

- [ ] Joint Attention (训练) vs 两阶段 + KV Cache (推理)
- [ ] GT actions 参与插值构造 x_t 和目标 u_t
- [ ] 采样: ε~N(0,I), t~Beta — 分布从源码提取
- [ ] 哪些模块 frozen / trainable (grad_boundary note)
- [ ] Loss: 从源码提取 (如 MSE(v_θ, u_t))

## 执行步骤

1. **读源码**: `select_action` / `forward` (推理), `compute_loss` / `forward` (训练)
2. **提取**: 模块链 + 关键 shape + 阶段划分 (推理 phase / 训练 stage)
3. **生成 graphviz**: 按 5 条硬约束画图
4. **渲染**: PNG + PDF, 300 DPI

## 输出文件

- `{framework}_inference.py` / `.png` / `.pdf`
- `{framework}_training.py` / `.png` / `.pdf`

## 代码模板

```python
"""{Framework} {Inference|Training} — pip install graphviz"""
from graphviz import Digraph

GRAPH_ATTR = {'dpi': '300', 'rankdir': 'TB', 'compound': 'true', 'fontname': 'Helvetica'}
CLUSTER = {'style': 'dashed', 'fontname': 'Helvetica-Bold', 'fontsize': '14'}

INPUT_NODE  = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'fontsize': '11'}
ENCODE_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'fontsize': '11'}
CACHE_NODE  = {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#F3E5F5', 'fontsize': '12'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'fontsize': '11'}
CONTROL_NODE= {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'fontsize': '11'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'fontsize': '12'}
LOSS_NODE   = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFCDD2', 'fontsize': '12'}
GRAD_BORDER = {'shape': 'box', 'style': 'dashed,filled', 'fillcolor': '#FFFFFF', 'color': '#C62828', 'fontsize': '10'}

DATA_EDGE = {'color': '#333333', 'fontsize': '9'}
COND_EDGE = {'color': '#7B1FA2', 'fontsize': '9', 'style': 'bold'}  # KV Cache 跨阶段


def build_graph():
    dot = Digraph(graph_attr=GRAPH_ATTR)

    dot.node('dims', 'D_model=...  L_prefix=...  act_dim=...', shape='note', fillcolor='#FAFAFA')

    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', **CLUSTER)
        c.node('images', 'Multi-Cam Images\nuint8\n[B,N_cam,3,H,W]', **INPUT_NODE)

    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: Prefill (run x1)', **CLUSTER)
        p1.node('vis_enc', 'Vision Encoder\nSigLIP2\nimg_tokens, [B,N_img,D_vis]', **ENCODE_NODE)
        p1.node('kv_cache', 'KV Cache\nfrozen\n[N_layer,2,B,S_prefix,D_h]', **CACHE_NODE)
        p1.edge('vis_enc', 'kv_cache', label='[B,L_prefix,D]', **DATA_EDGE)

    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Denoise (repeat xN)', **CLUSTER)
        p2.node('expert', 'Action Expert\nCross-Attn\nhidden, [B,N_action,D]', **ENCODE_NODE)
        p2.node('actions', 'Output Actions\n[B,N_action,act_dim]', **OUTPUT_NODE)

    dot.edge('kv_cache', 'expert', label='KV Cache', **COND_EDGE)
    dot.edge('expert', 'actions', label='[B,N_action,act_dim]', **DATA_EDGE)

    # 训练图: 用 Stage cluster 链替代 Phase1/2; 加 grad_boundary note
    # dot.node('grad_boundary', 'FROZEN: VLM\nTRAINABLE: Expert', **GRAD_BORDER)

    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g = build_graph()
    g.render('framework_inference', format='pdf', cleanup=True)
    g.render('framework_inference', format='png', cleanup=True)
```

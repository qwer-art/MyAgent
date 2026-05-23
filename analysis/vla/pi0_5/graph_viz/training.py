"""
Pi0.5 训练流程架构图
使用 Python graphviz 库生成

依赖: pip install graphviz
运行: python training.py
输出: training.pdf, training.png
"""

from graphviz import Digraph

# ---- 样式定义 ----
GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'label': 'Pi0.5 Training Flow\n(Flow Matching Loss)',
    'labelloc': 't',
    'fontsize': '20',
}

MODULE_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#E8F4FD',
    'fontsize': '14',
    'width': '3',
    'height': '1',
}

OP_NODE = {
    'shape': 'box',
    'style': 'rounded',
    'fillcolor': '#FFF3E0',
    'fontsize': '11',
}

LOSS_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#E8F5E9',
    'fontsize': '13',
}

DATA_EDGE = {
    'color': '#333333',
    'fontsize': '10',
    'fontcolor': '#666666',
}

CLUSTER_ATTR = {
    'style': 'dashed',
    'color': '#999999',
    'fontname': 'Helvetica-Bold',
    'fontsize': '16',
}

# ---- 图构建 ----
def build_graph():
    dot = Digraph(
        name='pi05_training',
        graph_attr=GRAPH_ATTR,
    )

    # ========== 训练输入 ==========

    dot.node('obs', 'Observation\nimages [B,224,224,3] x3\nstate [B,32]\nprompt [B,200]', **MODULE_NODE)
    dot.node('action_gt', 'Ground Truth Actions\n[B, 50, 32]', **MODULE_NODE)

    # ========== 噪声与时间采样 ==========

    with dot.subgraph(name='cluster_sample') as c:
        c.attr(label='Noise & Time Sampling', **CLUSTER_ATTR)

        c.node('noise', 'ε ~ N(0, I)\nshape: [B, 50, 32]', **OP_NODE)
        c.node('time', 't ~ Beta(1.5, 1.0)\nt = t*0.999 + 0.001\nrange: [0.001, 1.0]', **OP_NODE)

    # ========== Flow Matching 插值 ==========

    with dot.subgraph(name='cluster_interp') as c:
        c.attr(label='Flow Matching Interpolation', **CLUSTER_ATTR)

        c.node('x_t', 'x_t = t·ε + (1-t)·a\n(线性插值: 噪声→动作)', **OP_NODE)
        c.node('u_t', 'u_t = ε - a\n(目标速度场)', **OP_NODE)

    # ========== Prefix 编码 ==========

    with dot.subgraph(name='cluster_prefix') as c:
        c.attr(label='Prefix Encoding', **CLUSTER_ATTR)

        c.node('siglip', 'SigLIP ViT So400m/14\nimage tokens [B,256,2048]', **MODULE_NODE)
        c.node('lang_embed', 'Gemma Embedder\nlang+state tokens\n[B, max_len, 2048]', **MODULE_NODE)

    # ========== Suffix 编码 ==========

    with dot.subgraph(name='cluster_suffix') as c:
        c.attr(label='Suffix Encoding', **CLUSTER_ATTR)

        c.node('action_in', 'action_in_proj\nLinear(32→1024)\n[B, 50, 1024]', **OP_NODE)
        c.node('time_emb', 'Time Sincos + MLP\n[B, 1024] → adaRMS cond', **OP_NODE)

    # ========== Dual-Expert Transformer ==========

    with dot.subgraph(name='cluster_llm') as c:
        c.attr(label='Gemma Dual-Expert Transformer\nPaliGemma 2B (frozen) + Action Expert 300M (trainable)', **CLUSTER_ATTR)

        c.node('gemma_block', 'Gemma Block x18\nAttention (GQA: 8Q, 1KV)\nFFN (GeGLU)\nadaRMSNorm (Action Expert only)', **MODULE_NODE)
        c.node('prefix_out', 'prefix_out\n[B, prefix_len, 2048/1024]', **OP_NODE)
        c.node('suffix_out', 'suffix_out\n[B, 50, 1024]', **OP_NODE)

    # ========== Loss 计算 ==========

    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='Loss Computation', **CLUSTER_ATTR)

        c.node('v_t', 'v_t = action_out_proj(suffix_out)\nLinear(1024→32)\n[B, 50, 32]', **OP_NODE)
        c.node('loss', 'L = MSE(v_t, u_t)\n= mean((v_t - u_t)², axis=-1)\nper-token loss: [*b, 50]', **LOSS_NODE)

    # ========== LoRA 微调 ==========

    dot.node('lora_note', 'LoRA Fine-tuning (optional)\n- rank=32, alpha=32.0\n- Applied to attn + FFN\n- Only Action Expert LoRA trainable\n- PaliGemma LoRA → freeze base weights',
             shape='note', style='filled', fillcolor='#F5F5F5', fontsize='10')

    # ========== 连接关系 ==========

    # 观测 → 编码
    dot.edge('obs', 'siglip', label='images', **DATA_EDGE)
    dot.edge('obs', 'lang_embed', label='prompt+state', **DATA_EDGE)
    dot.edge('obs', 'action_in', label='x_t', **DATA_EDGE)
    dot.edge('obs', 'time_emb', label='t', **DATA_EDGE)

    # GT 动作 → 插值
    dot.edge('action_gt', 'x_t', label='a', **DATA_EDGE)
    dot.edge('noise', 'x_t', label='ε', **DATA_EDGE)
    dot.edge('noise', 'u_t', label='ε', **DATA_EDGE)
    dot.edge('action_gt', 'u_t', label='a', **DATA_EDGE)
    dot.edge('time', 'x_t', label='t', **DATA_EDGE)

    # Prefix → Transformer
    dot.edge('siglip', 'gemma_block', label='image tokens', **DATA_EDGE)
    dot.edge('lang_embed', 'gemma_block', label='lang tokens', **DATA_EDGE)

    # Suffix → Transformer
    dot.edge('action_in', 'gemma_block', label='action tokens', **DATA_EDGE)
    dot.edge('time_emb', 'gemma_block', label='adaRMS cond', **DATA_EDGE)

    # Transformer → Output
    dot.edge('gemma_block', 'prefix_out', **DATA_EDGE)
    dot.edge('gemma_block', 'suffix_out', **DATA_EDGE)

    # Loss
    dot.edge('suffix_out', 'v_t', **DATA_EDGE)
    dot.edge('u_t', 'loss', label='target', **DATA_EDGE)
    dot.edge('v_t', 'loss', label='prediction', **DATA_EDGE)

    return dot


# ---- 渲染 ----
if __name__ == '__main__':
    import os
    graph = build_graph()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graph.render(os.path.join(out_dir, 'training'), format='pdf', cleanup=True)
    graph.render(os.path.join(out_dir, 'training'), format='png', cleanup=True)
    print(f"已生成: {os.path.join(out_dir, 'training.pdf')}, {os.path.join(out_dir, 'training.png')}")
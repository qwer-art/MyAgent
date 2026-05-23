"""
Pi0.5 推理流程架构图
使用 Python graphviz 库生成

依赖: pip install graphviz
运行: python inference.py
输出: inference.pdf, inference.png
"""

from graphviz import Digraph

# ---- 样式定义 ----
GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'label': 'Pi0.5 Inference Flow\n(Flow Matching Denoising, num_steps=10)',
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

STEP_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#FFEBEE',
    'fontsize': '12',
}

# ---- 图构建 ----
def build_graph():
    dot = Digraph(
        name='pi05_inference',
        graph_attr=GRAPH_ATTR,
    )

    # ========== 输入层 ==========

    # 观测输入
    dot.node('input_images', 'Images\n[B, 224, 224, 3] x 3 views\n(base/wrist_l/wrist_r)', **MODULE_NODE)
    dot.node('input_lang', 'Language Prompt\n[B, max_token_len]\n(e.g. "pick up the cup")', **MODULE_NODE)
    dot.node('input_state', 'Robot State\n[B, action_dim=32]', **MODULE_NODE)
    dot.node('input_noise', 'Initial Noise x_1\n[B, action_horizon=50, action_dim=32]', **OP_NODE)

    # ========== Prefix 编码 (一次性) ==========

    with dot.subgraph(name='cluster_prefix') as c:
        c.attr(label='Prefix Encoding (one-time, KV cache)', **CLUSTER_ATTR)

        # SigLIP 视觉编码器
        c.node('siglip', 'SigLIP ViT\nSo400m/14\n输入: [B,224,224,3]\n输出: [B,256,2048]\n参数: 400M', **MODULE_NODE)

        # 语言 token 嵌入
        c.node('lang_embed', 'Gemma Embedder\nvocab_size=257152\n输入: [B, max_token_len]\n输出: [B, max_token_len, 2048]\n参数: PaliGemma 2B', **MODULE_NODE)

        # 状态离散化 (pi05特有)
        c.node('state_discretize', 'State Discretization\n256 bins, [-1,1]\n→ "Task: ..., State: ...;"\n→ tokenized string', **OP_NODE)

    # ========== Gemma Transformer (PaliGemma + Action Expert) ==========

    with dot.subgraph(name='cluster_llm') as c:
        c.attr(label='Gemma Dual-Expert Transformer\n(PaliGemma 2B + Action Expert 300M)', **CLUSTER_ATTR)

        c.node('gemma_block', 'Gemma Block x18\nAttention: 8 heads, 256 dim/head, 1 KV head\n(GQA)\nFFN: 16384 / 4096 hidden\nPaliGemma: width=2048\nAction Expert: width=1024', **MODULE_NODE)

        c.node('adaRMS', 'adaRMSNorm\n(cond=time_emb)\n→ scale, shift, gate\n调制 Attention + FFN', **OP_NODE)

        c.node('prefix_cache', 'KV Cache\n(prefix tokens)\n存储 image+lang\n的 key/value', **STEP_NODE)

    # ========== Suffix 编码 (每步循环) ==========

    with dot.subgraph(name='cluster_suffix') as c:
        c.attr(label='Suffix Encoding (per denoising step)', **CLUSTER_ATTR)

        c.node('action_in_proj', 'action_in_proj\nLinear(32→1024)\n输入: [B,50,32]\n输出: [B,50,1024]', **OP_NODE)

        c.node('time_sincos', 'Time Sincos Embedding\nmin_period=4e-3, max_period=4.0\n输入: t [B]\n输出: [B, 1024]', **OP_NODE)

        c.node('time_mlp', 'Time MLP\nLinear(1024→1024) + Swish\n+ Linear(1024→1024) + Swish\n输入: [B,1024]\n输出: [B,1024] (adaRMS cond)', **OP_NODE)

    # ========== 流匹配去噪循环 ==========

    with dot.subgraph(name='cluster_denoise') as c:
        c.attr(label='Flow Matching Denoising Loop (10 steps)', **CLUSTER_ATTR)

        c.node('step_init', 'x_1 = noise, t = 1.0\nEuler: dt = -1/10', **STEP_NODE)

        c.node('step_compute', 'v_t = action_out_proj(suffix_out)\nLinear(1024→32)\n输出: [B, 50, 32]', **OP_NODE)

        c.node('step_update', 'x_t = x_t + dt * v_t\nt = t + dt', **STEP_NODE)

        c.node('step_cond', 't >= -dt/2 ?', shape='diamond', style='filled', fillcolor='#E8E8E8', fontsize='12')

        c.node('step_output', 'Output: x_0\n[B, 50, 32]\n(action sequence)', **MODULE_NODE)

    # ========== 连接关系 ==========

    # 输入 → Prefix
    dot.edge('input_images', 'siglip', label='[B,224,224,3]', **DATA_EDGE)
    dot.edge('input_lang', 'lang_embed', label='[B,max_token_len]', **DATA_EDGE)
    dot.edge('input_state', 'state_discretize', label='[B,32]', **DATA_EDGE)

    # 状态离散化 → 语言拼接
    dot.edge('state_discretize', 'lang_embed', label='tokenized string\n+ state bins', **DATA_EDGE)

    # Prefix → KV cache
    dot.edge('siglip', 'gemma_block', label='[B,256,2048]\nimage tokens', **DATA_EDGE)
    dot.edge('lang_embed', 'gemma_block', label='[B,max_token_len,2048]\nlang+state tokens', **DATA_EDGE)
    dot.edge('gemma_block', 'prefix_cache', label='KV cache', **DATA_EDGE)

    # 去噪循环输入
    dot.edge('input_noise', 'step_init', label='x_1', **DATA_EDGE)

    # 每步: action + time → suffix
    dot.edge('step_init', 'action_in_proj', label='x_t [B,50,32]', **DATA_EDGE)
    dot.edge('step_init', 'time_sincos', label='t [B]', **DATA_EDGE)
    dot.edge('action_in_proj', 'gemma_block', label='action_tokens\n[B,50,1024]', **DATA_EDGE)
    dot.edge('time_sincos', 'time_mlp', label='[B,1024]', **DATA_EDGE)
    dot.edge('time_mlp', 'adaRMS', label='adarms_cond\n[B,1024]', **DATA_EDGE)

    # KV cache → 每步 attention
    dot.edge('prefix_cache', 'gemma_block', label='KV cache\n(prefix keys/values)', **DATA_EDGE)

    # Gemma output → action projection
    dot.edge('gemma_block', 'step_compute', label='suffix_out\n[B,50,1024]', **DATA_EDGE)

    # 去噪循环
    dot.edge('step_compute', 'step_update', label='v_t [B,50,32]', **DATA_EDGE)
    dot.edge('step_update', 'step_cond', label='x_t, t', **DATA_EDGE)
    dot.edge('step_cond', 'step_init', label='loop back\n(if t >= -dt/2)', **DATA_EDGE)
    dot.edge('step_cond', 'step_output', label='exit\n(if t < -dt/2)', **DATA_EDGE)

    # 注意力掩码说明
    dot.node('attn_note', 'Attention Mask:\n- Prefix: bidirectional (ar_mask=0)\n  image↔image, lang↔lang, image↔lang\n- Suffix: causal on actions (ar_mask=1,0,...0)\n  prefix→suffix: full attend\n  suffix→prefix: full attend',
             shape='note', style='filled', fillcolor='#F5F5F5', fontsize='10')

    return dot


# ---- 渲染 ----
if __name__ == '__main__':
    import os
    graph = build_graph()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graph.render(os.path.join(out_dir, 'inference'), format='pdf', cleanup=True)
    graph.render(os.path.join(out_dir, 'inference'), format='png', cleanup=True)
    print(f"已生成: {os.path.join(out_dir, 'inference.pdf')}, {os.path.join(out_dir, 'inference.png')}")
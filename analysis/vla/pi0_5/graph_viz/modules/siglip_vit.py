"""
Pi0.5 SigLIP ViT 编码器展开图
So400m/14 变体的内部结构

依赖: pip install graphviz
运行: python siglip_vit.py
输出: siglip_vit.pdf, siglip_vit.png
"""

from graphviz import Digraph

GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'label': 'SigLIP ViT So400m/14 Encoder',
    'labelloc': 't',
    'fontsize': '18',
}

MODULE_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#E8F4FD',
    'fontsize': '12',
    'width': '2.8',
    'height': '0.7',
}

OP_NODE = {
    'shape': 'box',
    'style': 'rounded',
    'fillcolor': '#FFF3E0',
    'fontsize': '11',
}

DATA_EDGE = {
    'color': '#333333',
    'fontsize': '9',
    'fontcolor': '#666666',
}

CLUSTER_ATTR = {
    'style': 'dashed',
    'color': '#999999',
    'fontname': 'Helvetica-Bold',
    'fontsize': '14',
}

def build_graph():
    dot = Digraph(
        name='siglip_vit',
        graph_attr=GRAPH_ATTR,
    )

    # 输入
    dot.node('input', 'Input Image\n[B, 224, 224, 3]', **MODULE_NODE)

    # Patch Embedding
    dot.node('patch_conv', 'Conv2d Patch Embed\n3→1152, k=14, s=14\n[B,16,16,1152]', **OP_NODE)
    dot.node('reshape', 'Reshape\n[B, 256, 1152]', **OP_NODE)
    dot.node('posemb', '+ Pos Emb\n(sincos2d)', **OP_NODE)

    # Transformer Encoder Blocks
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder Block x27 (scan)', **CLUSTER_ATTR)

        c.node('ln1', 'LayerNorm', **OP_NODE)
        c.node('mhsa', 'MultiHeadDotProductAttention\n16 heads, 1152 dim\n[B, 256, 1152]', **MODULE_NODE)
        c.node('res1', '⊕ Residual', shape='plaintext', fontsize='14')
        c.node('ln2', 'LayerNorm', **OP_NODE)
        c.node('mlp', 'MlpBlock\nDense(1152→4304) + Gelu\n+ Dense(4304→1152)', **MODULE_NODE)
        c.node('res2', '⊕ Residual', shape='plaintext', fontsize='14')

    # Final LayerNorm
    dot.node('final_ln', 'LayerNorm\n(final)', **OP_NODE)

    # 输出
    dot.node('output', 'Output Tokens\n[B, 256, 1152]\n→ 投影到 PaliGemma width=2048', **MODULE_NODE)

    # 连接
    dot.edge('input', 'patch_conv', label='[B,224,224,3]', **DATA_EDGE)
    dot.edge('patch_conv', 'reshape', **DATA_EDGE)
    dot.edge('reshape', 'posemb', **DATA_EDGE)
    dot.edge('posemb', 'ln1', label='[B,256,1152]', **DATA_EDGE)
    dot.edge('ln1', 'mhsa', **DATA_EDGE)
    dot.edge('mhsa', 'res1', **DATA_EDGE)
    dot.edge('res1', 'ln2', **DATA_EDGE)
    dot.edge('ln2', 'mlp', **DATA_EDGE)
    dot.edge('mlp', 'res2', **DATA_EDGE)
    dot.edge('res2', 'final_ln', **DATA_EDGE)
    dot.edge('final_ln', 'output', label='[B,256,1152]', **DATA_EDGE)

    # 架构说明
    dot.node('arch_note', 'SigLIP So400m Architecture:\n- width=1152, depth=27, heads=16\n- patch_size=14, mlp_dim=4304\n- pool_type=none (保留空间token)\n- Scan模式: 参数共享, 节省内存\n→ 输出投影: Linear(1152→2048)',
             shape='note', style='filled', fillcolor='#F5F5F5', fontsize='10')

    return dot


if __name__ == '__main__':
    import os
    graph = build_graph()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graph.render(os.path.join(out_dir, 'siglip_vit'), format='pdf', cleanup=True)
    graph.render(os.path.join(out_dir, 'siglip_vit'), format='png', cleanup=True)
    print(f"已生成: {os.path.join(out_dir, 'siglip_vit.pdf')}, {os.path.join(out_dir, 'siglip_vit.png')}")
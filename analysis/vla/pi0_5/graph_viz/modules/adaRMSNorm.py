"""
Pi0.5 adaRMSNorm 模块展开图
pi0.5 的核心创新: 时间条件注入

依赖: pip install graphviz
运行: python adaRMSNorm.py
输出: adaRMSNorm.pdf, adaRMSNorm.png
"""

from graphviz import Digraph

GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'label': 'adaRMSNorm Module\n(Adaptive RMS Normalization with Time Conditioning)',
    'labelloc': 't',
    'fontsize': '18',
}

MODULE_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#E8F4FD',
    'fontsize': '13',
    'width': '2.8',
    'height': '0.7',
}

OP_NODE = {
    'shape': 'box',
    'style': 'rounded',
    'fillcolor': '#FFF3E0',
    'fontsize': '11',
}

SPECIAL_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#FFEBEE',
    'fontsize': '12',
}

DATA_EDGE = {
    'color': '#333333',
    'fontsize': '10',
    'fontcolor': '#666666',
}

def build_graph():
    dot = Digraph(
        name='adaRMSNorm',
        graph_attr=GRAPH_ATTR,
    )

    # 输入
    dot.node('x_in', 'x (hidden state)\n[B, S, D=1024]', **MODULE_NODE)
    dot.node('cond_in', 'time_emb\n[B, 1024]', **MODULE_NODE)

    # RMSNorm 计算
    dot.node('rms', 'RMS Norm\nvar = mean(x²)\nnormed = x / √(var + ε)\nε = 1e-6', **OP_NODE)

    # Adaptive 调制
    dot.node('dense', 'Dense(1024→3×1024)\nkernel_init=zeros\n→ 全零初始化，训练初期为恒等映射', **SPECIAL_NODE)

    dot.node('split', 'Split(3, axis=-1)\n→ scale, shift, gate\n各 [B, 1, 1024]', **OP_NODE)

    dot.node('modulate', 'normed * (1 + scale) + shift\n→ scale 初始化为零\n→ 训练初期约等于 RMSNorm', **OP_NODE)

    dot.node('scale_param', 'learnable scale\nshape=[1024]\ninit=zeros\n→ (1 + scale) 渐进调制', **OP_NODE)

    # 输出
    dot.node('norm_out', 'Normalized Output\n[B, S, 1024]', **MODULE_NODE)
    dot.node('gate_out', 'Gate Signal\n[B, 1, 1024]\n→ 用于残差连接门控', **MODULE_NODE)

    # 对比说明
    dot.node('compare_note', 'pi0 vs pi0.5 对比:\n\npi0: 时间信息通过 MLP 与 action 拼接\n  action_time = MLP(concat(action, time_sincos))\n  → 时间与动作信息耦合\n\npi0.5: 时间信息通过 adaRMSNorm 注入\n  每个 Block 的 RMSNorm 接受 time_emb 条件\n  → 时间信息独立调制每层特征\n  → 更好的时间条件控制\n  → 类似 DiT 的 adaptive layer norm',
             shape='note', style='filled', fillcolor='#F5F5F5', fontsize='10')

    # 连接
    dot.edge('x_in', 'rms', **DATA_EDGE)
    dot.edge('rms', 'modulate', label='normed_x', **DATA_EDGE)
    dot.edge('cond_in', 'dense', **DATA_EDGE)
    dot.edge('dense', 'split', **DATA_EDGE)
    dot.edge('split', 'modulate', label='scale, shift', **DATA_EDGE)
    dot.edge('scale_param', 'modulate', label='learnable scale\n(1 + s)', **DATA_EDGE)
    dot.edge('modulate', 'norm_out', **DATA_EDGE)
    dot.edge('split', 'gate_out', label='gate', **DATA_EDGE)

    # 门控残差说明
    dot.node('gate_usage', 'Gate Usage in Residual:\nout = x + gate ⊙ sublayer_out\n→ 时间相关的门控残差连接',
             shape='note', style='filled', fillcolor='#F5F5F5', fontsize='10')
    dot.edge('gate_out', 'gate_usage', style='dashed', color='#999999')

    return dot


if __name__ == '__main__':
    import os
    graph = build_graph()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graph.render(os.path.join(out_dir, 'adaRMSNorm'), format='pdf', cleanup=True)
    graph.render(os.path.join(out_dir, 'adaRMSNorm'), format='png', cleanup=True)
    print(f"已生成: {os.path.join(out_dir, 'adaRMSNorm.pdf')}, {os.path.join(out_dir, 'adaRMSNorm.png')}")
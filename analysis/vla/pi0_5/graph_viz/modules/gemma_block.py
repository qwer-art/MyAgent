"""
Pi0.5 Gemma Dual-Expert Block 展开图
每个 Gemma Block 的内部结构

依赖: pip install graphviz
运行: python gemma_block.py
输出: gemma_block.pdf, gemma_block.png
"""

from graphviz import Digraph

GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'label': 'Gemma Dual-Expert Block\n(PaliGemma Branch + Action Expert Branch)',
    'labelloc': 't',
    'fontsize': '18',
}

MODULE_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#E8F4FD',
    'fontsize': '13',
    'width': '2.8',
    'height': '0.8',
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
    'fontsize': '14',
}

def build_graph():
    dot = Digraph(
        name='gemma_block',
        graph_attr=GRAPH_ATTR,
    )

    # 输入
    dot.node('input', 'Input Hidden States\nprefix: [B, P, 2048]\nsuffix: [B, 50, 1024]', **MODULE_NODE)

    # ===== adaRMSNorm 条件 =====
    dot.node('time_cond', 'Time Embedding\n[B, 1024]', **OP_NODE)

    # ===== PaliGemma Branch (prefix) =====
    with dot.subgraph(name='cluster_pali') as c:
        c.attr(label='PaliGemma Branch (frozen, width=2048)', **CLUSTER_ATTR)

        c.node('pali_rms1', 'RMSNorm\n(2048)', **OP_NODE)
        c.node('pali_attn', 'Grouped Query Attention\n8 Q-heads, 1 KV-head\ndim_head=256, width=2048\nrotary_pos_emb', **MODULE_NODE)
        c.node('pali_res1', '⊕ Residual', shape='plaintext', fontsize='14')
        c.node('pali_rms2', 'RMSNorm\n(2048)', **OP_NODE)
        c.node('pali_ffn', 'GeGLU FFN\nLinear(2048→16384)\nGelu → Linear(16384→2048)', **MODULE_NODE)
        c.node('pali_res2', '⊕ Residual', shape='plaintext', fontsize='14')

    # ===== Action Expert Branch (suffix) =====
    with dot.subgraph(name='cluster_action') as c:
        c.attr(label='Action Expert Branch (trainable, width=1024)', **CLUSTER_ATTR)

        c.node('act_adanorm1', 'adaRMSNorm\ncond=time_emb\n→ scale, shift, gate\n(3 × 1024)', **MODULE_NODE)
        c.node('act_attn', 'Grouped Query Attention\n8 Q-heads, 1 KV-head\ndim_head=128, width=1024\nrotary_pos_emb\nQ from Action Expert\nKV from both branches (joint)', **MODULE_NODE)
        c.node('act_res1', '⊕ Residual\n(gate modulation)', shape='plaintext', fontsize='14')
        c.node('act_adanorm2', 'adaRMSNorm\ncond=time_emb\n→ scale, shift, gate', **MODULE_NODE)
        c.node('act_ffn', 'GeGLU FFN\nLinear(1024→4096)\nGelu → Linear(4096→1024)', **MODULE_NODE)
        c.node('act_res2', '⊕ Residual\n(gate modulation)', shape='plaintext', fontsize='14')

    # 输出
    dot.node('output', 'Output Hidden States\nprefix: [B, P, 2048]\nsuffix: [B, 50, 1024]', **MODULE_NODE)

    # ===== 连接关系 =====

    # 输入 → 两个分支
    dot.edge('input', 'pali_rms1', label='prefix', **DATA_EDGE)
    dot.edge('input', 'act_adanorm1', label='suffix', **DATA_EDGE)

    # PaliGemma 分支
    dot.edge('pali_rms1', 'pali_attn', **DATA_EDGE)
    dot.edge('pali_attn', 'pali_res1', **DATA_EDGE)
    dot.edge('pali_res1', 'pali_rms2', **DATA_EDGE)
    dot.edge('pali_rms2', 'pali_ffn', **DATA_EDGE)
    dot.edge('pali_ffn', 'pali_res2', **DATA_EDGE)

    # Action Expert 分支
    dot.edge('time_cond', 'act_adanorm1', label='adaRMS cond', **DATA_EDGE)
    dot.edge('time_cond', 'act_adanorm2', label='adaRMS cond', **DATA_EDGE)
    dot.edge('act_adanorm1', 'act_attn', **DATA_EDGE)
    dot.edge('act_attn', 'act_res1', **DATA_EDGE)
    dot.edge('act_res1', 'act_adanorm2', **DATA_EDGE)
    dot.edge('act_adanorm2', 'act_ffn', **DATA_EDGE)
    dot.edge('act_ffn', 'act_res2', **DATA_EDGE)

    # 跨分支 KV 共享 (关键创新)
    dot.node('kv_note', 'Key-Value Sharing:\nPaliGemma & Action Expert\nshare same KV projections\n(prefix tokens only)\n→ Action Expert attends to\n  both image & lang features',
             shape='note', style='filled', fillcolor='#F5F5F5', fontsize='10')

    # 输出
    dot.edge('pali_res2', 'output', label='prefix out', **DATA_EDGE)
    dot.edge('act_res2', 'output', label='suffix out', **DATA_EDGE)

    return dot


if __name__ == '__main__':
    import os
    graph = build_graph()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graph.render(os.path.join(out_dir, 'gemma_block'), format='pdf', cleanup=True)
    graph.render(os.path.join(out_dir, 'gemma_block'), format='png', cleanup=True)
    print(f"已生成: {os.path.join(out_dir, 'gemma_block.pdf')}, {os.path.join(out_dir, 'gemma_block.png')}")
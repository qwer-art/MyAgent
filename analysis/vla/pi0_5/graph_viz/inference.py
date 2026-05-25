"""
Pi0.5 推理流程架构图
时间为主轴、特征空间为子分类，6 类颜色标注数据可解释性

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
    'label': 'Pi0.5 Inference Flow\n(Phase 1: Build KV Cache ×1  |  Phase 2: Flow Matching Denoise ×10)',
    'labelloc': 't',
    'fontsize': '20',
}

# 6 类颜色: 输入(绿) / KV Cache(紫) / 编码(蓝) / 融合(橙) / 控制信号(黄) / 输出(粉)
INPUT_NODE = {
    'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9',
    'fontsize': '11', 'width': '2.6', 'height': '0.5',
}
CACHE_NODE = {
    'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#F3E5F5',
    'fontsize': '12', 'width': '2.6', 'height': '0.8',
}
ENCODING_NODE = {
    'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD',
    'fontsize': '11', 'width': '3', 'height': '0.8',
}
FUSION_NODE = {
    'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0',
    'fontsize': '11', 'width': '2.6', 'height': '0.5',
}
CONTROL_NODE = {
    'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7',
    'fontsize': '11', 'width': '2.8', 'height': '0.5',
}
OUTPUT_NODE = {
    'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC',
    'fontsize': '12', 'width': '2.8', 'height': '0.8',
}

DATA_EDGE = {
    'color': '#333333', 'fontsize': '9', 'fontcolor': '#666666',
}
CACHE_EDGE = {
    'color': '#7B1FA2', 'fontsize': '9', 'fontcolor': '#7B1FA2',
    'style': 'bold',
}
CONTROL_EDGE = {
    'color': '#F9A825', 'fontsize': '9', 'fontcolor': '#F57F17',
    'style': 'dashed',
}

PHASE1_CLUSTER = {
    'style': 'dashed', 'color': '#1565C0',
    'fontname': 'Helvetica-Bold', 'fontsize': '18',
    'bgcolor': '#F5F9FF',
}
PHASE2_CLUSTER = {
    'style': 'dashed', 'color': '#C62828',
    'fontname': 'Helvetica-Bold', 'fontsize': '18',
    'bgcolor': '#FFF5F5',
}
INPUT_CLUSTER = {
    'style': 'dashed', 'color': '#2E7D32',
    'fontname': 'Helvetica-Bold', 'fontsize': '14',
}


def build_graph():
    dot = Digraph(name='pi05_inference', graph_attr=GRAPH_ATTR)

    # ================================================================
    # 输入 (人可理解的观测数据)
    # ================================================================
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs — 人可理解的观测数据', **INPUT_CLUSTER)

        c.node('in_img1', 'image (base cam)\n[1, 360, 360, 3] uint8\nRGB 0~255', **INPUT_NODE)
        c.node('in_img2', 'image2 (wrist cam)\n[1, 360, 360, 3] uint8\nRGB 0~238', **INPUT_NODE)
        c.node('in_lang', 'language prompt\n"pick up the black bowl\nbetween the plate and\nthe ramekin and place\nit on the plate"', **INPUT_NODE)
        c.node('in_state', 'robot state\n[1, 8] float32\neef_pos[3] + gripper[1]\n+ joints_pos[7] → 8-dim', **INPUT_NODE)
        c.node('in_noise', 'initial noise x₁\n[1, 50, 32] float32\nN(0, I) 纯随机噪声', **INPUT_NODE)

    # ================================================================
    # Phase 1: 初始化 — 建 KV Cache (只跑一次)
    # ================================================================
    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: Initialization — Build KV Cache (run ×1)', **PHASE1_CLUSTER)

        # -- Image 子流 --
        with p1.subgraph(name='cluster_p1_img') as c:
            c.attr(label='Image Stream', style='dotted', color='#90CAF9', fontsize='12',
                   fontname='Helvetica-Bold')

            c.node('p1_resize', 'resize + pad\n[1,3,360,360]→[1,3,224,224]\nnorm [0,1]→[-1,1]', **FUSION_NODE)
            c.node('p1_siglip', 'SigLIP ViT\nSo400m/14, 27 blocks\npatch=14, dim=1152\n→ MultiModalProjector\n→ dim 2048', **ENCODING_NODE)
            c.node('p1_img_tok', 'image tokens\n[1, 256, 2048] ×2 views\n= [1, 512, 2048]', **ENCODING_NODE)

        # -- Language + State 子流 --
        with p1.subgraph(name='cluster_p1_lang') as c:
            c.attr(label='Language + State Stream', style='dotted', color='#90CAF9', fontsize='12',
                   fontname='Helvetica-Bold')

            c.node('p1_discretize', 'State Discretization\n256 bins ∈ [-1, 1]\n[1,8] float → "155 248 213\n166 115 174 -1 255"', **FUSION_NODE)
            c.node('p1_prompt', "Prompt Assembly\n'Task: pick up..., State:\n155 248...;\\nAction: '", **FUSION_NODE)
            c.node('p1_tokenizer', 'Tokenizer\nvocab=257152, max_len=200\n→ tokens [1, 200] int64\n+ attn_mask [1, 200]', **ENCODING_NODE)
            c.node('p1_embed', 'Gemma Embedder\ntokens → embeddings\n[1, 200] → [1, 200, 2048]', **ENCODING_NODE)
            c.node('p1_lang_tok', 'lang+state tokens\n[1, 200, 2048]', **ENCODING_NODE)

        # -- Prefix 拼接 --
        p1.node('p1_concat', 'Concat Prefix\nimg [1,512,2048] + lang [1,200,2048]\n→ [1, 712, 2048]', **FUSION_NODE)

        # -- PaliGemma Transformer --
        p1.node('p1_paligemma', 'PaliGemma Transformer\n18 blocks, width=2048\nGQA: 8Q/1KV, dim_head=256\nGeGLU FFN: 2048→16384→2048\nbidirectional prefix attention\n(所有 prefix token 互相可见)', **ENCODING_NODE)

        # -- KV Cache 输出 --
        p1.node('p1_cache', 'KV Cache (frozen, 不增长, 不FIFO)\n18 layers × [1, 712, 1, 256] per layer\nkeys:   [18, 1, 712, 1, 256] = 3.3M bf16 ≈ 6.3MB\nvalues: [18, 1, 712, 1, 256] = 3.3M bf16 ≈ 6.3MB\ntotal ≈ 12.6MB bfloat16\n每帧重建, Phase 2 中 frozen 复用', **CACHE_NODE)

    # ================================================================
    # Phase 2: 递推去噪 — Flow Matching (循环10次)
    # ================================================================
    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Recurrent Denoising — Flow Matching (repeat ×10)', **PHASE2_CLUSTER)

        # -- Time 控制信号子流 --
        with p2.subgraph(name='cluster_p2_time') as c:
            c.attr(label='Time Control Signal', style='dotted', color='#FFF176', fontsize='12',
                   fontname='Helvetica-Bold')

            c.node('p2_t', 't = 1.0 + step×(-0.1)\nt ∈ {1.0, 0.9, ..., 0.1}\n当前去噪进度', **CONTROL_NODE)
            c.node('p2_sincos', 'Time Sincos Embed\nmin=4e-3, max=4.0\nt [1] → [1, 1024]', **CONTROL_NODE)
            c.node('p2_tmlp', 'Time MLP\nL(1024)+Swish+L(1024)+Swish\n[1,1024] → [1,1024]\n= adaRMS 条件向量', **CONTROL_NODE)

        # -- Action 子流 --
        with p2.subgraph(name='cluster_p2_action') as c:
            c.attr(label='Action Stream', style='dotted', color='#90CAF9', fontsize='12',
                   fontname='Helvetica-Bold')

            c.node('p2_ainproj', 'action_in_proj\nLinear(32→1024)\nx_t [1,50,32] → [1,50,1024]\n噪声动作→高维隐空间', **ENCODING_NODE)
            c.node('p2_suffix', 'suffix tokens\n[1, 50, 1024]\ncausal attention on actions\n(第1个token可见后续所有)', **ENCODING_NODE)

        # -- Attention mask --
        p2.node('p2_mask', 'Attention Mask\n4D shape: [1, 1, 50, 762]\n(第2维=1, broadcast到GQA的8个head)\n= prefix_col [1,1,50,712]\n  + suffix_col [1,1,50,50]\n\nprefix↔prefix: 不参与(Q只有suffix)\nsuffix→prefix: 全True (cross attend)\nsuffix↔suffix: causal下三角\n\n构造: prefix_col = repeat(prefix_mask)\n       suffix_col = make_attn_mask(\n         suffix_mask,\n         suffix_ar_mask=[T,F,...,F])\n       → concat → [:,None,:,:] 扩4D', **FUSION_NODE)

        # -- Action Expert Transformer --
        p2.node('p2_expert', 'Action Expert Transformer\n18 blocks, width=1024\nGQA: 8Q/1KV, head_dim=256 (与PaliGemma相同)\nGeGLU FFN: 1024→4096→1024\nadaRMSNorm: 每层用 time_cond 调制\n  out = normed × (1 + scale) + shift\n  gate 控制残差强度\nQ: suffix [1,50,8,256] (action expert)\nKV (整体): [18, 1, 712, 1, 256]\nKV (每层slice): [1, 712, 1, 256]\n  + suffix K/V [1, 50, 1, 256]\n  → 临时 concat [1, 762, 1, 256]\n  (计算完即弃, 不写入 cache)', **ENCODING_NODE)

        # -- 输出 --
        p2.node('p2_aoutproj', 'action_out_proj\nLinear(1024→32)\n[1,50,1024] → v_t [1,50,32]\n隐空间→速度场', **ENCODING_NODE)
        p2.node('p2_euler', 'x_t = x_t + dt × v_t\nEuler 积分, dt = -0.1', **FUSION_NODE)
        p2.node('p2_cond', 'step < 10 ?', shape='diamond', style='filled', fillcolor='#FFCDD2', fontsize='12')

        # -- 最终输出 --
        p2.node('p2_output', 'Output: x₀\n[1, 50, 32] float32\n50 步动作序列\n每步 7-DoF (+ padding)', **OUTPUT_NODE)

    # ================================================================
    # 连接: Inputs → Phase 1
    # ================================================================
    dot.edge('in_img1', 'p1_resize', label='uint8', **DATA_EDGE)
    dot.edge('in_img2', 'p1_resize', label='uint8', **DATA_EDGE)
    dot.edge('in_state', 'p1_discretize', label='[1,8]', **DATA_EDGE)
    dot.edge('in_lang', 'p1_prompt', **DATA_EDGE)

    # ================================================================
    # 连接: Phase 1 内部
    # ================================================================
    dot.edge('p1_resize', 'p1_siglip', label='[1,3,224,224]\n[-1,1]', **DATA_EDGE)
    dot.edge('p1_siglip', 'p1_img_tok', **DATA_EDGE)

    dot.edge('p1_discretize', 'p1_prompt', label='discretized\nstate bins', **DATA_EDGE)
    dot.edge('p1_prompt', 'p1_tokenizer', **DATA_EDGE)
    dot.edge('p1_tokenizer', 'p1_embed', label='tokens\n[1,200]', **DATA_EDGE)
    dot.edge('p1_embed', 'p1_lang_tok', **DATA_EDGE)

    # Image + Lang 汇合
    dot.edge('p1_img_tok', 'p1_concat', **DATA_EDGE)
    dot.edge('p1_lang_tok', 'p1_concat', **DATA_EDGE)
    dot.edge('p1_concat', 'p1_paligemma', label='[1, 712, 2048]', **DATA_EDGE)
    dot.edge('p1_paligemma', 'p1_cache', label='use_cache=True\nfill KV', **DATA_EDGE)

    # ================================================================
    # 连接: Phase 1 → Phase 2 (跨阶段: KV Cache 传递)
    # ================================================================
    dot.edge('p1_cache', 'p2_expert', label='KV Cache (frozen)\nclone per step', **CACHE_EDGE)

    # ================================================================
    # 连接: Inputs → Phase 2
    # ================================================================
    dot.edge('in_noise', 'p2_euler', label='x₁ = noise\n(first step)', **DATA_EDGE)

    # ================================================================
    # 连接: Phase 2 内部 — Time 控制信号
    # ================================================================
    dot.edge('p2_t', 'p2_sincos', label='t [1]', **CONTROL_EDGE)
    dot.edge('p2_sincos', 'p2_tmlp', label='[1, 1024]', **CONTROL_EDGE)
    dot.edge('p2_tmlp', 'p2_expert', label='adaRMS cond\n[1, 1024]\n调制每层行为', **CONTROL_EDGE)

    # ================================================================
    # 连接: Phase 2 内部 — Action 流
    # ================================================================
    dot.edge('p2_euler', 'p2_ainproj', label='x_t [1,50,32]\n当前噪声动作', **DATA_EDGE)
    dot.edge('p2_ainproj', 'p2_suffix', **DATA_EDGE)
    dot.edge('p2_suffix', 'p2_mask', style='invis')  # layout hint
    dot.edge('p2_mask', 'p2_expert', label='mask\n[1,1,50,762]', **DATA_EDGE)
    dot.edge('p2_suffix', 'p2_expert', label='suffix\n[1,50,1024]', **DATA_EDGE)
    dot.edge('p2_expert', 'p2_aoutproj', label='suffix_out\n[1,50,1024]', **DATA_EDGE)
    dot.edge('p2_aoutproj', 'p2_euler', label='v_t [1,50,32]\n预测速度场', **DATA_EDGE)

    # 循环
    dot.edge('p2_euler', 'p2_cond', **DATA_EDGE)
    dot.edge('p2_cond', 'p2_ainproj', label='loop\n(t -= 0.1)', color='#C62828', fontsize='9', fontcolor='#C62828')
    dot.edge('p2_cond', 'p2_output', label='exit', **DATA_EDGE)

    # ================================================================
    # 颜色图例
    # ================================================================
    dot.node('legend',
             '颜色编码 — 数据可解释性:\n'
             '─────────────────────────────\n'
             '🟢 输入(观测空间): 人可理解\n'
             '    图像/语言/状态/噪声\n'
             '🟣 KV Cache: 键值缓存\n'
             '    存储 prefix 上下文\n'
             '🔵 编码(特征向量): 人不可理解\n'
             '    高维张量, 维度≈1024/2048\n'
             '🟠 融合: 编码特征的交汇点\n'
             '    concat / mask / 积分\n'
             '🟡 控制信号: 时间步条件\n'
             '    驱动每步去噪行为\n'
             '🔴 输出(动作空间): 人可理解\n'
             '    50步 × 7-DoF 动作序列\n'
             '─────────────────────────────\n'
             'Phase 1: PaliGemma (width=2048)\n'
             'Phase 2: Action Expert (width=1024)\n'
             '计算量 Phase 1 >> Phase 2\n'
             '  P≈712 prefix vs 50 suffix tokens',
             shape='note', style='filled', fillcolor='#FAFAFA', fontsize='10')

    return dot


if __name__ == '__main__':
    import os
    graph = build_graph()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graph.render(os.path.join(out_dir, 'inference'), format='pdf', cleanup=True)
    graph.render(os.path.join(out_dir, 'inference'), format='png', cleanup=True)
    print(f"已生成: {os.path.join(out_dir, 'inference.pdf')}, {os.path.join(out_dir, 'inference.png')}")
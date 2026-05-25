"""
Pi0.5 训练流程架构图
时间主轴: 单次前向 + Loss (Joint Attention, 无阶段分离)
特征子分类: Image Stream / Lang+State Stream / Action Stream / Time Control
9 类颜色标注数据可解释性 (在推理 6 类基础上 + 采样/Loss/梯度边界)

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
    'label': 'Pi0.5 Training Flow\n(Joint Attention: prefix + suffix 每层联合计算, 无 KV Cache)',
    'labelloc': 't',
    'fontsize': '20',
}

# 9 类颜色: 推理6类 + 训练3类(采样/采样/梯度边界)
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
# 训练新增
SAMPLE_NODE = {
    'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#F1F8E9',
    'fontsize': '11', 'width': '2.8', 'height': '0.5',
}
LOSS_NODE = {
    'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFCDD2',
    'fontsize': '12', 'width': '3', 'height': '0.8',
}

DATA_EDGE = {
    'color': '#333333', 'fontsize': '9', 'fontcolor': '#666666',
}
CONTROL_EDGE = {
    'color': '#F9A825', 'fontsize': '9', 'fontcolor': '#F57F17',
    'style': 'dashed',
}
GRAD_EDGE = {
    'color': '#C62828', 'fontsize': '9', 'fontcolor': '#C62828',
    'style': 'dashed',
}
FREEZE_EDGE = {
    'color': '#9E9E9E', 'fontsize': '9', 'fontcolor': '#9E9E9E',
    'style': 'dotted',
}


def build_graph():
    dot = Digraph(name='pi05_training', graph_attr=GRAPH_ATTR)

    # ================================================================
    # 输入 (人可理解的观测数据 + GT动作)
    # ================================================================
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs — 人可理解的观测与标注数据', style='dashed',
               color='#2E7D32', fontname='Helvetica-Bold', fontsize='14')

        c.node('in_img1', 'image (base cam)\n[1, 360, 360, 3] uint8\nRGB 0~255', **INPUT_NODE)
        c.node('in_img2', 'image2 (wrist cam)\n[1, 360, 360, 3] uint8\nRGB 0~238', **INPUT_NODE)
        c.node('in_lang', 'language prompt\n"pick up the black bowl..."', **INPUT_NODE)
        c.node('in_state', 'robot state\n[1, 8] float32\n归一化后 ∈ [-1, 1]', **INPUT_NODE)
        c.node('in_action', 'GT actions (数据集标注)\n[1, 50, 32] float32\n归一化后 ∈ [-1, 1]', **INPUT_NODE)

    # ================================================================
    # 采样区 (训练独有: 噪声 + 时间步)
    # ================================================================
    with dot.subgraph(name='cluster_sampling') as c:
        c.attr(label='Sampling — 训练独有: 从分布中采样', style='dashed',
               color='#558B2F', fontname='Helvetica-Bold', fontsize='14',
               bgcolor='#FAFFF0')

        c.node('sample_eps', 'ε ~ N(0, I)\n[1, 50, 32] float32\n标准高斯噪声', **SAMPLE_NODE)
        c.node('sample_t', 't ~ Beta(1.5, 1.0)\nt = t×0.999 + 0.001\nt ∈ [0.001, 1.0]\n偏向 t=1 (噪声端)', **SAMPLE_NODE)

    # ================================================================
    # Flow Matching 插值 (训练独有: 构造训练输入和目标)
    # ================================================================
    with dot.subgraph(name='cluster_interpolation') as c:
        c.attr(label='Flow Matching Interpolation — 构造训练对 (x_t, u_t)', style='dashed',
               color='#E65100', fontname='Helvetica-Bold', fontsize='14',
               bgcolor='#FFF8F0')

        c.node('interp_xt', 'x_t = t·ε + (1-t)·a\n线性插值: 噪声↔动作\nt=1 → 纯噪声, t=0 → 干净动作\n[1, 50, 32] float32', **FUSION_NODE)
        c.node('interp_ut', 'u_t = ε - a\n目标速度场: 从噪声指向动作\n与 t 无关 (线性插值的性质)\n[1, 50, 32] float32', **FUSION_NODE)

    # ================================================================
    # Image Stream (编码)
    # ================================================================
    with dot.subgraph(name='cluster_img') as c:
        c.attr(label='Image Stream', style='dotted', color='#90CAF9',
               fontsize='12', fontname='Helvetica-Bold')

        c.node('img_resize', 'resize + pad\n[1,3,360,360]→[1,3,224,224]\nnorm [0,1]→[-1,1]', **FUSION_NODE)
        c.node('img_siglip', 'SigLIP ViT\nSo400m/14, 27 blocks\npatch=14, dim=1152\n→ MultiModalProjector → 2048', **ENCODING_NODE)
        c.node('img_tokens', 'image tokens\n[1, 256, 2048] ×2 views\n= [1, 512, 2048]', **ENCODING_NODE)

    # ================================================================
    # Language + State Stream (编码)
    # ================================================================
    with dot.subgraph(name='cluster_lang') as c:
        c.attr(label='Language + State Stream', style='dotted', color='#90CAF9',
               fontsize='12', fontname='Helvetica-Bold')

        c.node('lang_discretize', 'State Discretization\n256 bins ∈ [-1, 1]\n[1,8] float → "155 248 213\n166 115 174 -1 255"', **FUSION_NODE)
        c.node('lang_prompt', "Prompt Assembly\n'Task: pick up..., State:\n155 248...;\\nAction: '", **FUSION_NODE)
        c.node('lang_tokenizer', 'Tokenizer\nvocab=257152, max_len=200\n→ tokens [1, 200] int64\n+ attn_mask [1, 200]', **ENCODING_NODE)
        c.node('lang_embed', 'Gemma Embedder\n[1, 200] → [1, 200, 2048]', **ENCODING_NODE)
        c.node('lang_tokens', 'lang+state tokens\n[1, 200, 2048]', **ENCODING_NODE)

    # ================================================================
    # Prefix 拼接 + Joint Attention Transformer
    # ================================================================
    with dot.subgraph(name='cluster_transformer') as c:
        c.attr(label='Dual-Expert Joint Attention Transformer\n(训练: 每层联合计算, 推理: 分两阶段 + KV Cache)',
               style='dashed', color='#1565C0', fontname='Helvetica-Bold',
               fontsize='16', bgcolor='#F5F9FF')

        c.node('prefix_concat', 'Concat Prefix\nimg [1,512,2048] + lang [1,200,2048]\n→ [1, 712, 2048]', **FUSION_NODE)

        c.node('joint_attn', 'Joint Attention Block x18\n\nPaliGemma (frozen, width=2048):\n  RMSNorm → GQA(8Q/1KV, dim=256) → Residual\n  RMSNorm → GeGLU FFN(2048→16384→2048) → Residual\n\nAction Expert (trainable, width=1024):\n  adaRMSNorm(time_cond) → GQA(8Q/1KV, dim=128) → Residual(gate)\n  adaRMSNorm(time_cond) → GeGLU FFN(1024→4096→1024) → Residual(gate)\n\nJoint Attention:\n  Q = [Q_pali; Q_act], K = [K_pali; K_act], V = [V_pali; V_act]\n  concat → softmax → split → O_proj各自维度', **ENCODING_NODE)

    # ================================================================
    # Action Stream (编码)
    # ================================================================
    with dot.subgraph(name='cluster_action') as c:
        c.attr(label='Action Stream', style='dotted', color='#90CAF9',
               fontsize='12', fontname='Helvetica-Bold')

        c.node('act_inproj', 'action_in_proj\nLinear(32→1024)\nx_t [1,50,32] → [1,50,1024]\n插值后的噪声动作→高维隐空间', **ENCODING_NODE)
        c.node('act_suffix', 'suffix tokens\n[1, 50, 1024]\ncausal attention on actions', **ENCODING_NODE)

    # ================================================================
    # Time Control Signal (编码)
    # ================================================================
    with dot.subgraph(name='cluster_time') as c:
        c.attr(label='Time Control Signal', style='dotted', color='#FFF176',
               fontsize='12', fontname='Helvetica-Bold')

        c.node('time_sincos', 'Time Sincos Embed\nmin=4e-3, max=4.0\nt [1] → [1, 1024]', **CONTROL_NODE)
        c.node('time_mlp', 'Time MLP\nL(1024)+Swish+L(1024)+Swish\n[1,1024] → [1,1024]\n= adaRMS 条件向量', **CONTROL_NODE)

    # ================================================================
    # Loss 计算 (训练独有)
    # ================================================================
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='Loss Computation', style='dashed',
               color='#C62828', fontname='Helvetica-Bold', fontsize='14',
               bgcolor='#FFF5F5')

        c.node('out_proj', 'action_out_proj\nLinear(1024→32)\nsuffix_out [1,50,1024] → v_t [1,50,32]\n模型预测的速度场', **ENCODING_NODE)
        c.node('loss', 'L = MSE(v_t, u_t)\n= mean((v_t - u_t)²)\nper-token loss: [B, 50, 32]\n最终: mean over all dims', **LOSS_NODE)

    # ================================================================
    # 梯度边界标注 (训练独有)
    # ================================================================
    dot.node('grad_note',
             '梯度边界 (默认配置):\n'
             '─────────────────────────────\n'
             '✅ 可训练 (梯度流过):\n'
             '   Action Expert 全部参数\n'
             '   action_in_proj / action_out_proj\n'
             '   time_mlp_in / time_mlp_out\n'
             '   (LoRA: gemma_expert.q/v_proj)\n'
             '─────────────────────────────\n'
             '🔒 冻结 (梯度不流过):\n'
             '   SigLIP ViT 全部参数\n'
             '   PaliGemma Transformer 全部参数\n'
             '   Gemma Embedder 全部参数\n'
             '─────────────────────────────\n'
             '推理对比:\n'
             '  训练: Joint Attention 联合计算\n'
             '  推理: 分两阶段 + KV Cache\n'
             '  原因: 推理时 prefix 不变,\n'
             '        可缓存 KV 避免重复计算',
             shape='note', style='filled', fillcolor='#FAFAFA', fontsize='10')

    # ================================================================
    # 连接: Inputs → 各流
    # ================================================================
    dot.edge('in_img1', 'img_resize', label='uint8', **DATA_EDGE)
    dot.edge('in_img2', 'img_resize', label='uint8', **DATA_EDGE)
    dot.edge('in_state', 'lang_discretize', label='[1,8]\n归一化', **DATA_EDGE)
    dot.edge('in_lang', 'lang_prompt', **DATA_EDGE)

    # ================================================================
    # 连接: Inputs → 采样 → 插值
    # ================================================================
    dot.edge('in_action', 'interp_xt', label='a (GT actions)\n[1,50,32]', **DATA_EDGE)
    dot.edge('in_action', 'interp_ut', label='a (GT actions)\n[1,50,32]', **DATA_EDGE)
    dot.edge('sample_eps', 'interp_xt', label='ε\n[1,50,32]', **DATA_EDGE)
    dot.edge('sample_eps', 'interp_ut', label='ε\n[1,50,32]', **DATA_EDGE)
    dot.edge('sample_t', 'interp_xt', label='t', **CONTROL_EDGE)
    dot.edge('sample_t', 'time_sincos', label='t [1]', **CONTROL_EDGE)

    # ================================================================
    # 连接: Image Stream 内部
    # ================================================================
    dot.edge('img_resize', 'img_siglip', label='[1,3,224,224]\n[-1,1]', **DATA_EDGE)
    dot.edge('img_siglip', 'img_tokens', **DATA_EDGE)

    # ================================================================
    # 连接: Lang+State Stream 内部
    # ================================================================
    dot.edge('lang_discretize', 'lang_prompt', label='discretized\nstate bins', **DATA_EDGE)
    dot.edge('lang_prompt', 'lang_tokenizer', **DATA_EDGE)
    dot.edge('lang_tokenizer', 'lang_embed', label='tokens\n[1,200]', **DATA_EDGE)
    dot.edge('lang_embed', 'lang_tokens', **DATA_EDGE)

    # ================================================================
    # 连接: Prefix 拼接
    # ================================================================
    dot.edge('img_tokens', 'prefix_concat', label='[1,512,2048]', **DATA_EDGE)
    dot.edge('lang_tokens', 'prefix_concat', label='[1,200,2048]', **DATA_EDGE)
    dot.edge('prefix_concat', 'joint_attn', label='prefix [1,712,2048]', **DATA_EDGE)

    # ================================================================
    # 连接: 插值 → Action Stream
    # ================================================================
    dot.edge('interp_xt', 'act_inproj', label='x_t [1,50,32]\n噪声动作', **DATA_EDGE)
    dot.edge('act_inproj', 'act_suffix', **DATA_EDGE)

    # ================================================================
    # 连接: Time Control 内部
    # ================================================================
    dot.edge('time_sincos', 'time_mlp', label='[1, 1024]', **CONTROL_EDGE)
    dot.edge('time_mlp', 'joint_attn', label='adaRMS cond\n[1, 1024]\n调制 Action Expert\n每层 scale/shift/gate', **CONTROL_EDGE)

    # ================================================================
    # 连接: Action Stream → Transformer
    # ================================================================
    dot.edge('act_suffix', 'joint_attn', label='suffix [1,50,1024]\n(与 prefix 在 attention 层\n联合计算 Q/K/V)', **DATA_EDGE)

    # ================================================================
    # 连接: Transformer → Loss
    # ================================================================
    dot.edge('joint_attn', 'out_proj', label='suffix_out [1,50,1024]', **DATA_EDGE)
    dot.edge('out_proj', 'loss', label='v_t [1,50,32]\n预测速度场', **DATA_EDGE)
    dot.edge('interp_ut', 'loss', label='u_t [1,50,32]\n目标速度场', **DATA_EDGE)

    # ================================================================
    # 梯度流标注 (红色虚线: 可训练, 灰色点线: 冻结)
    # ================================================================
    # Action Expert 链路: loss → out_proj → joint_attn(act) → act_inproj → interp_xt
    dot.edge('loss', 'out_proj', label='∇ grad', style='dashed', color='#C62828', fontsize='8', fontcolor='#C62828', dir='back')
    dot.edge('out_proj', 'joint_attn', label='∇', style='dashed', color='#C62828', fontsize='8', fontcolor='#C62828', dir='back')
    dot.edge('joint_attn', 'act_inproj', label='∇', style='dashed', color='#C62828', fontsize='8', fontcolor='#C62828', dir='back')

    # Time MLP 梯度
    dot.edge('joint_attn', 'time_mlp', label='∇', style='dashed', color='#C62828', fontsize='8', fontcolor='#C62828', dir='back')

    # 冻结边界: prefix 流无梯度
    dot.edge('joint_attn', 'prefix_concat', label='⊘ frozen', style='dotted', color='#9E9E9E', fontsize='8', fontcolor='#9E9E9E', dir='back')
    dot.edge('joint_attn', 'img_siglip', label='⊘ frozen', style='dotted', color='#9E9E9E', fontsize='8', fontcolor='#9E9E9E', dir='back')
    dot.edge('joint_attn', 'lang_embed', label='⊘ frozen', style='dotted', color='#9E9E9E', fontsize='8', fontcolor='#9E9E9E', dir='back')

    return dot


if __name__ == '__main__':
    import os
    graph = build_graph()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graph.render(os.path.join(out_dir, 'training'), format='pdf', cleanup=True)
    graph.render(os.path.join(out_dir, 'training'), format='png', cleanup=True)
    print(f"已生成: {os.path.join(out_dir, 'training.pdf')}, {os.path.join(out_dir, 'training.png')}")
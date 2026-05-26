"""
Alpamayo-R1 Stage 2 Training Architecture Diagram

Node spec (concise, 3-line):
  Line 1: 模块名
  Line 2: 兆体实现, 关键ops
  Line 3: 输出变量名, [B,shape]

Fusion nodes must annotate strategy term.

Training vs Inference:
  - VLM: teacher forcing (no_grad), not autoregressive generate
  - KV Cache: stop-gradient (detach K,V), not frozen clone
  - Expert input: noisy_x = t*a + (1-t)*eps, not x_0 ~ N(0,I)
  - Timestep: t ~ Beta(1.5,1.0) shifted per sample, not linspace
  - Loss: MSE(v_theta, a - eps), no Euler loop
  - Gradient: VLM frozen, expert+proj trainable

Dependencies: pip install graphviz
Run: python alpamayo_training.py
Output: alpamayo_training.pdf, alpamayo_training.png
"""

from graphviz import Digraph

# ---- 9-class color scheme ----
INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'color': '#2E7D32', 'fontsize': '11', 'fontname': 'Helvetica'}
CACHE_NODE = {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#F3E5F5', 'color': '#7B1FA2', 'fontsize': '12', 'fontname': 'Helvetica'}
ENCODING_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'color': '#1565C0', 'fontsize': '11', 'fontname': 'Helvetica'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'color': '#E65100', 'fontsize': '11', 'fontname': 'Helvetica'}
CONTROL_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'color': '#F57F17', 'fontsize': '11', 'fontname': 'Helvetica'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'color': '#C62828', 'fontsize': '12', 'fontname': 'Helvetica'}
SAMPLE_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#F1F8E9', 'color': '#558B2F', 'fontsize': '11', 'fontname': 'Helvetica'}
LOSS_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFCDD2', 'color': '#C62828', 'fontsize': '12', 'fontname': 'Helvetica'}
GRAD_BORDER = {'shape': 'box', 'style': 'dashed,filled', 'fillcolor': '#FFFFFF', 'color': '#C62828', 'fontsize': '10', 'fontname': 'Helvetica'}

DATA_EDGE = {'color': '#333333', 'fontsize': '9', 'fontcolor': '#666666', 'fontname': 'Helvetica'}
CACHE_EDGE = {'color': '#7B1FA2', 'fontsize': '9', 'fontcolor': '#7B1FA2', 'style': 'bold', 'fontname': 'Helvetica'}
CONTROL_EDGE = {'color': '#F57F17', 'fontsize': '9', 'fontcolor': '#F57F17', 'style': 'dashed', 'fontname': 'Helvetica'}
SAMPLE_EDGE = {'color': '#558B2F', 'fontsize': '9', 'fontcolor': '#558B2F', 'fontname': 'Helvetica'}
LOSS_EDGE = {'color': '#C62828', 'fontsize': '9', 'fontcolor': '#C62828', 'style': 'dashed', 'fontname': 'Helvetica'}
GRAD_EDGE = {'color': '#C62828', 'fontsize': '10', 'fontcolor': '#C62828', 'style': 'dashed', 'arrowhead': 'none', 'fontname': 'Helvetica'}

GRAPH_ATTR = {
    'dpi': '300', 'rankdir': 'TB', 'fontname': 'Helvetica',
    'bgcolor': 'white', 'compound': 'true', 'nodesep': '0.4', 'ranksep': '0.6',
}
CLUSTER_ATTR = {
    'style': 'dashed', 'color': '#999999', 'fontname': 'Helvetica-Bold', 'fontsize': '14',
}

LEGEND_HTML = '''<
<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#F5F5F5" color="#757575">
<tr><td colspan="2"><b>DIMENSION REFERENCE (RL config)</b></td></tr>
<tr><td colspan="2" height="1" bgcolor="#999999"></td></tr>
<tr><td align="left">D_vis</td><td align="left">1280</td></tr>
<tr><td align="left">D_model</td><td align="left">4096</td></tr>
<tr><td align="left">D_expert</td><td align="left">4096</td></tr>
<tr><td align="left">N_layer</td><td align="left">36</td></tr>
<tr><td colspan="2" height="1" bgcolor="#999999"></td></tr>
<tr><td align="left">T_hist</td><td align="left">16 (1.6s @10Hz)</td></tr>
<tr><td align="left">hist_tok</td><td align="left">48 (16x3dim)</td></tr>
<tr><td align="left">num_bins</td><td align="left">1000 (per-scalar)</td></tr>
<tr><td align="left">traj_vocab</td><td align="left">4000 (VLM expand)</td></tr>
<tr><td colspan="2" height="1" bgcolor="#999999"></td></tr>
<tr><td align="left">n_waypt</td><td align="left">64</td></tr>
<tr><td align="left">act_dim</td><td align="left">2</td></tr>
<tr><td align="left">t_sampler</td><td align="left">Beta(1.5,1.0) shifted</td></tr>
<tr><td align="left">target</td><td align="left">u_t = a - eps (OT)</td></tr>
<tr><td align="left">loss</td><td align="left">MSE(v_theta, u_t)</td></tr>
</table>
>'''


def build_graph():
    dot = Digraph(name='Alpamayo_R1_Training', graph_attr=GRAPH_ATTR)

    # Legend + gradient boundary
    with dot.subgraph(name='cluster_top_row') as top:
        top.attr(rank='same')
        top.node('legend', LEGEND_HTML, shape='plaintext')
        top.node('inputs_anchor', '', shape='point', width='0')

    dot.node('grad_boundary',
        'GRADIENT BOUNDARY (Stage 2)\n'
        'FROZEN: VLM, SigLIP2\n'
        'TRAINABLE: Expert, in_proj, out_proj\n'
        'KV Cache: stop-grad (detach K,V)',
        **GRAD_BORDER)

    # =====================================================================
    # Phase 1: VLM Prefill (teacher forcing, no_grad)
    # =====================================================================
    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: VLM Prefill (teacher forcing, no_grad)', **CLUSTER_ATTR)

        p1.node('multi_cam', 'Multi-Cam Images\nuint8, N_cam=4\n[B,16,3,448,280]', **INPUT_NODE)
        p1.node('language', 'Language Instruction\nBPE token IDs\n[B,L_text]', **INPUT_NODE)
        p1.node('egomotion', 'Egomotion History\ndelta xyz\n[B,16,3]', **INPUT_NODE)

        p1.node('vis_enc', 'Vision Encoder\nSigLIP2 + 2x downsample\nimg_tokens, [B,2560,1280]', **ENCODING_NODE)
        p1.node('hist_tokenizer', 'Traj Tokenizer\nper-scalar bin, vocab=1000\ntraj_tokens, [B,48]', **ENCODING_NODE)
        p1.node('token_fusion', 'Token Fusion\nEarly Fusion, Token Concatenation\ninput_ids, [B,L_prefix]', **FUSION_NODE)
        p1.node('vlm', 'VLM Backbone\nQwen3-VL-8B, GQA(kv=8)\nhidden, [B,L_prefix,4096]', **ENCODING_NODE)

        p1.node('coc', 'CoC Reasoning\nQwen3-VL text tokens\ntext_ids, [B,L_coc]', **OUTPUT_NODE)
        p1.node('meta_action', 'Meta-Action\nQwen3-VL text tokens\ntext_ids, [B,L_meta]', **OUTPUT_NODE)
        p1.node('kv_cache', 'KV Cache\nPrefix, stop-grad\n[36,2,B,S_prefix,128]', **CACHE_NODE)

        dot.edge('inputs_anchor', 'multi_cam', style='invis')
        dot.edge('inputs_anchor', 'legend', style='invis')
        dot.edge('inputs_anchor', 'grad_boundary', style='invis')

        p1.edge('multi_cam', 'vis_enc', label='uint8 -> float32', **DATA_EDGE)
        p1.edge('egomotion', 'hist_tokenizer', label='delta xyz\ncontinuous R', **DATA_EDGE)
        p1.edge('vis_enc', 'token_fusion', label='[B,2560,1280]\ncontinuous feat', **DATA_EDGE)
        p1.edge('language', 'token_fusion', label='[B,L_text]  BPE IDs', **DATA_EDGE)
        p1.edge('hist_tokenizer', 'token_fusion', label='[B,48]  int IDs\nvocab=1000', **DATA_EDGE)
        p1.edge('token_fusion', 'vlm', label='embed [B,L_prefix,4096]\n+4000 new traj embeds', **DATA_EDGE)
        p1.edge('vlm', 'coc', label='CoC text', **DATA_EDGE)
        p1.edge('vlm', 'meta_action', label='meta-action text', **DATA_EDGE)
        p1.edge('vlm', 'kv_cache', label='past_key_values\nstop-grad: detach(K,V)', **CACHE_EDGE)

    # =====================================================================
    # FM Sampling & Interpolation
    # =====================================================================
    with dot.subgraph(name='cluster_fm_data') as fmd:
        fmd.attr(label='FM Sampling & Interpolation', **CLUSTER_ATTR)

        fmd.node('gt_action', 'GT Action\nUnicycle accel+kappa\na, [B,64,2]', **INPUT_NODE)
        fmd.node('eps_sample', 'Noise\nN(0,I)\neps, [B,64,2]', **SAMPLE_NODE)
        fmd.node('t_sample', 'Timestep\nBeta(1.5,1.0) shifted\nt, [B] scalar', **SAMPLE_NODE)
        fmd.node('fm_interp', 'FM Interpolation\nx_t = t*a + (1-t)*eps\nnoisy_x, [B,64,2]', **FUSION_NODE)
        fmd.node('fm_target', 'FM Target\nu_t = a - eps\ntarget, [B,64,2]', **OUTPUT_NODE)

        fmd.edge('gt_action', 'fm_interp', label='a: [B,64,2]', **DATA_EDGE)
        fmd.edge('eps_sample', 'fm_interp', label='eps: [B,64,2]', **SAMPLE_EDGE)
        fmd.edge('t_sample', 'fm_interp', label='t: [B] scalar', **CONTROL_EDGE)
        fmd.edge('gt_action', 'fm_target', label='a', **DATA_EDGE)
        fmd.edge('eps_sample', 'fm_target', label='eps', **SAMPLE_EDGE)

    # =====================================================================
    # Phase 2: Expert Forward (trainable)
    # =====================================================================
    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Expert Forward (trainable)', **CLUSTER_ATTR)

        p2.node('action_in_proj', 'Action In Proj\nFourierPE+MLP, 4layer SiLU\nembed, [B,64,4096]', **ENCODING_NODE)
        p2.node('expert', 'Expert Transformer\nCross-Attn Fusion, bidirectional\nhidden, [B,64,4096]', **ENCODING_NODE)
        p2.node('action_out_proj', 'Action Out Proj\nLinear(4096,2)\nv_theta, [B,64,2]', **ENCODING_NODE)
        p2.node('loss', 'MSE Loss\nMSE(v_theta, u_t)\nscalar, reduction=mean', **LOSS_NODE)

        p2.edge('action_in_proj', 'expert', label='[B,64,4096]\nembed', **DATA_EDGE)
        p2.edge('expert', 'action_out_proj', label='[B,64,4096]', **DATA_EDGE)
        p2.edge('action_out_proj', 'loss', label='v_theta: [B,64,2]', **LOSS_EDGE)

    # Cross connections
    dot.edge('fm_interp', 'action_in_proj', label='noisy_x: [B,64,2]', **DATA_EDGE)
    dot.edge('t_sample', 'action_in_proj', label='t: [B]  Fourier PE', **CONTROL_EDGE)
    dot.edge('fm_target', 'loss', label='u_t: [B,64,2]', **LOSS_EDGE)
    dot.edge('kv_cache', 'expert', label='KV Cache stop-grad\n[36,2,B,S_prefix,128]\nCross-Attn prefix->suffix\nNO grad -> VLM', **CACHE_EDGE)

    # Gradient boundary annotations
    dot.edge('grad_boundary', 'vlm', label='FROZEN', **GRAD_EDGE)
    dot.edge('grad_boundary', 'vis_enc', label='FROZEN', **GRAD_EDGE)
    dot.edge('grad_boundary', 'expert', label='TRAINABLE', style='dashed', color='#2E7D32', fontcolor='#2E7D32', arrowhead='none', fontsize='10', fontname='Helvetica')
    dot.edge('grad_boundary', 'action_in_proj', label='TRAINABLE', style='dashed', color='#2E7D32', fontcolor='#2E7D32', arrowhead='none', fontsize='10', fontname='Helvetica')
    dot.edge('grad_boundary', 'action_out_proj', label='TRAINABLE', style='dashed', color='#2E7D32', fontcolor='#2E7D32', arrowhead='none', fontsize='10', fontname='Helvetica')

    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    graph = build_graph()
    graph.render(filename='alpamayo_training', format='pdf', cleanup=True)
    graph.render(filename='alpamayo_training', format='png', cleanup=True)
    print("Generated: alpamayo_training.pdf, alpamayo_training.png")
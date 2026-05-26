"""
Alpamayo-R1 Inference Architecture Diagram

Node spec (concise):
  1. 泛化方式 | 2. 兆体实现 | 3. 输出张量
  Example: Vision Encoder | SigLIP2 | [B,2560,1280]

Fusion nodes must annotate strategy term.

Dependencies: pip install graphviz
Run: python alpamayo_inference.py
Output: alpamayo_inference.pdf, alpamayo_inference.png
"""

from graphviz import Digraph

# ---- 6-class color scheme ----
INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'color': '#2E7D32', 'fontsize': '11', 'fontname': 'Helvetica'}
CACHE_NODE = {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#F3E5F5', 'color': '#7B1FA2', 'fontsize': '12', 'fontname': 'Helvetica'}
ENCODING_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'color': '#1565C0', 'fontsize': '11', 'fontname': 'Helvetica'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'color': '#E65100', 'fontsize': '11', 'fontname': 'Helvetica'}
CONTROL_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'color': '#F57F17', 'fontsize': '11', 'fontname': 'Helvetica'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'color': '#C62828', 'fontsize': '12', 'fontname': 'Helvetica'}

DATA_EDGE = {'color': '#333333', 'fontsize': '9', 'fontcolor': '#666666', 'fontname': 'Helvetica'}
CACHE_EDGE = {'color': '#7B1FA2', 'fontsize': '9', 'fontcolor': '#7B1FA2', 'style': 'bold', 'fontname': 'Helvetica'}
CONTROL_EDGE = {'color': '#F57F17', 'fontsize': '9', 'fontcolor': '#F57F17', 'style': 'dashed', 'fontname': 'Helvetica'}

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
<tr><td align="left">VLM</td><td align="left">Qwen3-VL-8B / Cosmos-Reason2-8B</td></tr>
<tr><td align="left">D_vis</td><td align="left">1280</td></tr>
<tr><td align="left">D_model</td><td align="left">4096</td></tr>
<tr><td align="left">D_head</td><td align="left">128 (GQA N_kv=8)</td></tr>
<tr><td align="left">D_expert</td><td align="left">4096</td></tr>
<tr><td align="left">N_layer</td><td align="left">36</td></tr>
<tr><td colspan="2" height="1" bgcolor="#999999"></td></tr>
<tr><td align="left">N_cam</td><td align="left">4 (7 avail)</td></tr>
<tr><td align="left">N_frames</td><td align="left">4</td></tr>
<tr><td align="left">N_img_tok</td><td align="left">160/img</td></tr>
<tr><td colspan="2" height="1" bgcolor="#999999"></td></tr>
<tr><td align="left">T_hist</td><td align="left">16 (1.6s @10Hz)</td></tr>
<tr><td align="left">hist_tok</td><td align="left">48 (16x3dim)</td></tr>
<tr><td align="left">num_bins</td><td align="left">1000 (per-scalar)</td></tr>
<tr><td align="left">traj_vocab</td><td align="left">4000 (VLM expand)</td></tr>
<tr><td colspan="2" height="1" bgcolor="#999999"></td></tr>
<tr><td align="left">n_waypt</td><td align="left">64 (6.4s @10Hz)</td></tr>
<tr><td align="left">act_dim</td><td align="left">2 (accel,kappa)</td></tr>
<tr><td align="left">N_traj</td><td align="left">6</td></tr>
<tr><td align="left">Euler</td><td align="left">10 steps</td></tr>
</table>
>'''


def build_graph():
    dot = Digraph(name='Alpamayo_R1_Inference', graph_attr=GRAPH_ATTR)

    # Legend
    with dot.subgraph(name='cluster_top_row') as top:
        top.attr(rank='same')
        top.node('legend', LEGEND_HTML, shape='plaintext')
        top.node('inputs_anchor', '', shape='point', width='0')

    # =====================================================================
    # Phase 1: VLM Prefill
    # =====================================================================
    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: VLM Prefill (run x1)', **CLUSTER_ATTR)

        p1.node('multi_cam', 'Multi-Cam Images\nuint8, N_cam=4\n[B,16,3,448,280]', **INPUT_NODE)
        p1.node('language', 'Language Instruction\nBPE token IDs\n[B,L_text]', **INPUT_NODE)
        p1.node('egomotion', 'Egomotion History\ndelta xyz\n[B,16,3]', **INPUT_NODE)

        p1.node('vis_enc', 'Vision Encoder\nSigLIP2 + 2x downsample\nimg_tokens, [B,2560,1280]', **ENCODING_NODE)
        p1.node('hist_tokenizer', 'Traj Tokenizer\nper-scalar bin, vocab=1000\ntraj_tokens, [B,48]', **ENCODING_NODE)
        p1.node('token_fusion', 'Token Fusion\nEarly Fusion, Token Concatenation\ninput_ids, [B,L_prefix]', **FUSION_NODE)
        p1.node('vlm', 'VLM Backbone\nQwen3-VL-8B, GQA(kv=8)\nhidden, [B,L_prefix,4096]', **ENCODING_NODE)

        p1.node('coc', 'CoC Reasoning\nQwen3-VL text tokens\ntext_ids, [B,L_coc]', **OUTPUT_NODE)
        p1.node('meta_action', 'Meta-Action\nQwen3-VL text tokens\ntext_ids, [B,L_meta]', **OUTPUT_NODE)
        p1.node('kv_cache', 'KV Cache\nPrefix, frozen\n[36,2,B,S_prefix,128]', **CACHE_NODE)

        dot.edge('inputs_anchor', 'multi_cam', style='invis')
        dot.edge('inputs_anchor', 'legend', style='invis')

        p1.edge('multi_cam', 'vis_enc', label='uint8 -> float32', **DATA_EDGE)
        p1.edge('egomotion', 'hist_tokenizer', label='delta xyz\ncontinuous R', **DATA_EDGE)
        p1.edge('vis_enc', 'token_fusion', label='[B,2560,1280]\ncontinuous feat', **DATA_EDGE)
        p1.edge('language', 'token_fusion', label='[B,L_text]  BPE IDs', **DATA_EDGE)
        p1.edge('hist_tokenizer', 'token_fusion', label='[B,48]  int IDs\nvocab=1000', **DATA_EDGE)
        p1.edge('token_fusion', 'vlm', label='embed [B,L_prefix,4096]\n+4000 new traj embeds', **DATA_EDGE)
        p1.edge('vlm', 'coc', label='CoC text', **DATA_EDGE)
        p1.edge('vlm', 'meta_action', label='meta-action text', **DATA_EDGE)
        p1.edge('vlm', 'kv_cache', label='past_key_values\nfrozen prefix', **CACHE_EDGE)

    # =====================================================================
    # Phase 2: Action Expert Decoding (FM)
    # =====================================================================
    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Action Expert Decoding | FM (x10 Euler)', **CLUSTER_ATTR)

        p2.node('noise_init', 'Noise Sample\nx_0 ~ N(0,I)\n[6B,64,2] accel,kappa', **INPUT_NODE)
        p2.node('flow_time', 'Flow Time\nFM, linspace(0,1,11)\ndt=0.1', **CONTROL_NODE)
        p2.node('action_in_proj', 'Action In Proj\nFourierPE + MLP(4layer)\naction_embed, [6B,64,4096]', **ENCODING_NODE)
        p2.node('expert', 'Expert Transformer\nCross-Attn Fusion, GQA(kv=8)\nhidden, [6B,64,4096]', **ENCODING_NODE)
        p2.node('action_out_proj', 'Action Out Proj\nLinear(4096,2)\nv_field, [6B,64,2]', **ENCODING_NODE)
        p2.node('euler_step', 'Euler Integration\nFM Decoding, x_{t+dt}=x_t+dt*v\naction, [6B,64,2]', **FUSION_NODE)
        p2.node('unicycle_decode', 'Kinematic Decoder\nUnicycle, accel+kappa\nxyz [6B,64,3], rot [6B,64,3,3]', **OUTPUT_NODE)

        p2.edge('noise_init', 'action_in_proj', label='x_t: [6B,64,2]\nnoisy action', **DATA_EDGE)
        p2.edge('flow_time', 'action_in_proj', label='t: scalar\n[0,1]', **CONTROL_EDGE)
        p2.edge('action_in_proj', 'expert', label='[6B,64,4096]\nembed', **DATA_EDGE)
        p2.edge('expert', 'action_out_proj', label='[6B,64,4096]', **DATA_EDGE)
        p2.edge('action_out_proj', 'euler_step', label='v: [6B,64,2]\nvector field', **DATA_EDGE)
        p2.edge('noise_init', 'euler_step', label='x_t', **DATA_EDGE)
        p2.edge('flow_time', 'euler_step', label='dt=0.1', **CONTROL_EDGE)
        p2.edge('euler_step', 'unicycle_decode', label='[6B,64,2]\naccel,kappa', **DATA_EDGE)

    # Cross-phase
    dot.edge('kv_cache', 'expert', label='KV Cache frozen\n[36,2,B,S_prefix,128]\nCross-Attn prefix->suffix', **CACHE_EDGE)

    # Final output
    dot.node('traj_output', 'Predicted Trajectory\n[B,6,64,3] xyz  [B,6,64,3,3] rot\n6 samples x 64 wp  6.4s @10Hz', **OUTPUT_NODE)
    dot.edge('unicycle_decode', 'traj_output', label='reshape 6B->B,6', **DATA_EDGE)

    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    graph = build_graph()
    graph.render(filename='alpamayo_inference', format='pdf', cleanup=True)
    graph.render(filename='alpamayo_inference', format='png', cleanup=True)
    print("Generated: alpamayo_inference.pdf, alpamayo_inference.png")
"""DreamZero Inference — graph-viz-vla simplified (Wan2.2-TI2V World-Action Model)"""
from graphviz import Digraph

GRAPH_ATTR = {'dpi': '300', 'rankdir': 'TB', 'compound': 'true', 'fontname': 'Helvetica'}
CLUSTER = {'style': 'dashed', 'fontname': 'Helvetica-Bold', 'fontsize': '14'}
INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'fontsize': '11'}
ENCODE_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'fontsize': '11'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'fontsize': '11'}
CONTROL_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'fontsize': '11'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'fontsize': '11'}
DATA_EDGE = {'color': '#333333', 'fontsize': '9'}
COND_EDGE = {'color': '#7B1FA2', 'fontsize': '9', 'style': 'bold'}


def build_graph():
    dot = Digraph('dreamzero_inference', graph_attr=GRAPH_ATTR)
    dot.node('dims',
             'D_dit=5120  C_latent=16  FM_steps=16\n'
             'act_dim=14  chunk=32  DiT_layers=40  LoRA_r=4',
             shape='note', fillcolor='#FAFAFA', fontsize='10')

    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', **CLUSTER)
        c.node('text', 'Text Prompt\nT5 token ids\n[B,L_text]', **INPUT_NODE)
        c.node('image', 'Reference Image\nRGB 336x336\n[B,3,336,336]', **INPUT_NODE)
        c.node('noise', 'Initial Noise x_T\nGaussian\n[B,16,T,H,W]', **INPUT_NODE)

    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: Condition Encode (run x1, frozen)', **CLUSTER)
        p1.node('t5', 'T5 Text Encoder\nfrozen\ncontext, [B,L,4096]', **ENCODE_NODE)
        p1.node('clip', 'CLIP ViT-H/14\nfrozen\n[B,1,1280]', **ENCODE_NODE)
        p1.node('ctx_fuse', 'Context Fusion\nproject to D_dit\n[B,L,5120]', **FUSION_NODE)
        p1.edge('t5', 'ctx_fuse', label='[B,L,4096]', **DATA_EDGE)
        p1.edge('clip', 'ctx_fuse', label='[B,1,1280]', **DATA_EDGE)

    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Shifted Flow Matching (repeat x16)', **CLUSTER)
        p2.node('fm_sched', 'FlowMatch Scheduler\nshifted sigma [0.25,1.0]\nEuler step', **CONTROL_NODE)
        p2.node('wan_dit', 'CausalWanModel x40\nCausal Chunk Attn + Cross-Attn\nv_pred, [B,N,16]', **ENCODE_NODE)
        p2.node('x_update', 'Latent Update\nx += v*(sigma_next-sigma)\n[B,16,T,H,W]', **FUSION_NODE)
        p2.edge('fm_sched', 'wan_dit', label='timestep t', **COND_EDGE)
        p2.edge('wan_dit', 'x_update', label='v_pred', **DATA_EDGE)
        p2.edge('x_update', 'wan_dit', label='x_t loop', **COND_EDGE)

    with dot.subgraph(name='cluster_output') as o:
        o.attr(label='Outputs', **CLUSTER)
        o.node('vae_dec', 'Wan VAE Decoder\nCausalConv3d frozen\nvideo, [B,3,T,H,W]', **ENCODE_NODE)
        o.node('act_dec', 'Action Decoder\nCategorySpecific\nactions, [B,32,14]', **OUTPUT_NODE)
        o.node('video_out', 'Predicted Video\n[B,3,T,H,W]', **OUTPUT_NODE)

    dot.edge('text', 't5', **DATA_EDGE)
    dot.edge('image', 'clip', **DATA_EDGE)
    dot.edge('ctx_fuse', 'wan_dit', label='Cross-Attn context', **COND_EDGE)
    dot.edge('noise', 'wan_dit', label='x_T', **DATA_EDGE)
    dot.edge('x_update', 'vae_dec', label='x_0 [B,16,T,H,W]', **DATA_EDGE)
    dot.edge('x_update', 'act_dec', label='action tokens', **DATA_EDGE)
    dot.edge('vae_dec', 'video_out', label='RGB decode', **DATA_EDGE)
    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g = build_graph()
    g.render('dreamzero_inference', format='pdf', cleanup=True)
    g.render('dreamzero_inference', format='png', cleanup=True)
    print('Done: dreamzero_inference.pdf, .png')

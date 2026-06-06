"""VGGT-Omega Inference — graph-viz-vla simplified"""
from graphviz import Digraph

GRAPH_ATTR = {'dpi': '300', 'rankdir': 'TB', 'compound': 'true', 'fontname': 'Helvetica'}
CLUSTER = {'style': 'dashed', 'fontname': 'Helvetica-Bold', 'fontsize': '14'}
INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'fontsize': '11'}
ENCODE_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'fontsize': '11'}
CACHE_NODE = {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#F3E5F5', 'fontsize': '11'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'fontsize': '11'}
CONTROL_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'fontsize': '11'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'fontsize': '11'}
DATA_EDGE = {'color': '#333333', 'fontsize': '9'}
COND_EDGE = {'color': '#F57F17', 'fontsize': '9', 'style': 'dashed'}


def build_graph():
    dot = Digraph('vggt_omega_inference', graph_attr=GRAPH_ATTR)
    dot.node('dims', 'D=1024  D_cat=2048  F=frames  patch=16\nheads=16  AA_layers=24  pose_dim=9',
             shape='note', fillcolor='#FAFAFA', fontsize='10')

    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', **CLUSTER)
        c.node('images', 'Multi-view Images\nRGB uint8->float32\n[B,F,3,H,W]', **INPUT_NODE)

    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: Tokenization', **CLUSTER)
        p1.node('patch', 'Patch Embedding\nConv2d k=16 + LayerNorm\n[B*F,HW,1024]', **ENCODE_NODE)
        p1.node('tokens', 'Token Fusion\nEarly Fusion: cam|reg|patch\n[B*F,1+16+HW,1024]', **FUSION_NODE)
        p1.edge('patch', 'tokens', label='[B*F,HW,1024]', **DATA_EDGE)

    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Alternating-Attention Aggregator (x24)', **CLUSTER)
        p2.node('frame_attn', 'Frame Self-Attn\nper-frame + RoPE\n[B*F,N,1024]', **ENCODE_NODE)
        p2.node('global_attn', 'Inter-Frame Global Attn\nflatten [B,F*N,D]\n[B,F,N,1024]', **ENCODE_NODE)
        p2.node('cached', 'Cached Layer Outputs\ncat frame|global\n[B,F,N,2048]@4,11,17,23', **CACHE_NODE)
        p2.edge('frame_attn', 'global_attn', label='[B,F,N,1024]', **DATA_EDGE)
        p2.edge('global_attn', 'cached', label='[B,F,N,2048]', **DATA_EDGE)

    with dot.subgraph(name='cluster_heads') as h:
        h.attr(label='Phase 3: Prediction Heads', **CLUSTER)
        h.node('cam_head', 'Camera Head\n4x Self-Attn + Linear\npose_enc, [B,F,9]', **ENCODE_NODE)
        h.node('dense_head', 'Dense Head\nDPT fuse + PixelShuffle\ndepth, [B,F,H,W,1]', **ENCODE_NODE)
        h.node('pose_out', 'Extrinsic+Intrinsic\n[B,F,3,4] + [B,F,3,3]', **OUTPUT_NODE)
        h.node('depth_out', 'Depth + Confidence\n[B,F,H,W]', **OUTPUT_NODE)

    dot.edge('images', 'patch', label='ImageNet norm', **DATA_EDGE)
    dot.edge('tokens', 'frame_attn', label='[B*F,N,1024]', **DATA_EDGE)
    dot.node('rope', 'RoPE 2D axial\nbase=100', **CONTROL_NODE)
    dot.edge('rope', 'frame_attn', label='(sin,cos)', **COND_EDGE)
    dot.edge('cached', 'cam_head', label='cam+reg [B,F*17,2048]', **DATA_EDGE)
    dot.edge('cached', 'dense_head', label='patch tokens', **DATA_EDGE)
    dot.edge('cam_head', 'pose_out', label='9D->R,T,FoV', **DATA_EDGE)
    dot.edge('dense_head', 'depth_out', label='exp(depth)', **DATA_EDGE)
    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g = build_graph()
    g.render('vggt_omega_inference', format='pdf', cleanup=True)
    g.render('vggt_omega_inference', format='png', cleanup=True)
    print('Done: vggt_omega_inference.pdf, .png')

"""LingBot-Map (GCT) Streaming Inference — graph-viz-vla simplified"""
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
COND_EDGE = {'color': '#7B1FA2', 'fontsize': '9', 'style': 'bold'}


def build_graph():
    dot = Digraph('lingbot_map_inference', graph_attr=GRAPH_ATTR)
    dot.node('dims', 'D=1024  D_out=2048  S_scale=8  H=378 W=518\nN_tok=1005  window=64kf  AA_layers=24',
             shape='note', fillcolor='#FAFAFA', fontsize='10')

    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', **CLUSTER)
        c.node('images', 'Raw Images\nJPEG/PNG, resize 518x378\n[S,3,378,518]', **INPUT_NODE)

    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: Scale Frames (bidirectional, run x1)', **CLUSTER)
        p1.node('dino_p1', 'Vision Encoder\nDINOv2 ViT-L/14 frozen\npatch, [8,999,1024]', **ENCODE_NODE)
        p1.node('fusion_p1', 'Token Fusion\nEarly Fusion: special|patch\n[8,1005,1024]', **FUSION_NODE)
        p1.node('gct_p1', 'GCT Alternating Attn\nFrame+Global SDPA x24\n[8,1005,2048]', **ENCODE_NODE)
        p1.node('kv_cache', 'Paged KV Cache\nscale_pages=8 frozen\n[8 frames cached]', **CACHE_NODE)
        p1.edge('dino_p1', 'fusion_p1', label='[8,999,1024]', **DATA_EDGE)
        p1.edge('fusion_p1', 'gct_p1', label='[8,1005,1024]', **DATA_EDGE)
        p1.edge('gct_p1', 'kv_cache', label='append K,V', **COND_EDGE)

    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Streaming (causal, repeat xN)', **CLUSTER)
        p2.node('kf', 'Keyframe Gate\nflow or fixed-interval\nbool', **CONTROL_NODE)
        p2.node('gct_p2', 'GCT Streaming Attn\nFlashInfer causal x24\n[1,1005,2048]', **ENCODE_NODE)
        p2.edge('kf', 'gct_p2', label='cache policy', **COND_EDGE)

    with dot.subgraph(name='cluster_heads') as h:
        h.attr(label='Prediction Heads', **CLUSTER)
        h.node('cam_head', 'Camera Head\nCameraCausalHead x4 iter\npose_enc, [B,S,9]', **ENCODE_NODE)
        h.node('depth_head', 'Depth Head\nDPT multi-scale fusion\ndepth, [B,S,1,H,W]', **ENCODE_NODE)
        h.node('point_head', 'Point Head\nDPT multi-scale fusion\nworld_pts, [B,S,H,W,3]', **ENCODE_NODE)

    with dot.subgraph(name='cluster_output') as o:
        o.attr(label='Outputs', **CLUSTER)
        o.node('out_pose', 'Camera Poses c2w\n[B,S,3,4]', **OUTPUT_NODE)
        o.node('out_depth', 'Depth + Confidence\n[B,S,1,H,W]', **OUTPUT_NODE)

    dot.edge('images', 'dino_p1', label='[:8] scale frames', **DATA_EDGE)
    dot.edge('images', 'kf', label='[i] stream frame', **DATA_EDGE)
    dot.edge('kv_cache', 'gct_p2', label='KV Cache clone', **COND_EDGE)
    dot.edge('gct_p1', 'cam_head', label='[1,8,1005,2048]', **DATA_EDGE)
    dot.edge('gct_p2', 'cam_head', label='[1,1,1005,2048]', **DATA_EDGE)
    dot.edge('gct_p1', 'depth_head', **DATA_EDGE)
    dot.edge('gct_p2', 'depth_head', **DATA_EDGE)
    dot.edge('cam_head', 'out_pose', label='[B,S,9]->c2w', **DATA_EDGE)
    dot.edge('depth_head', 'out_depth', label='exp activation', **DATA_EDGE)
    dot.edge('depth_head', 'point_head', label='shared DPT', **DATA_EDGE)
    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g = build_graph()
    g.render('lingbot_map_inference', format='pdf', cleanup=True)
    g.render('lingbot_map_inference', format='png', cleanup=True)
    print('Done: lingbot_map_inference.pdf, .png')

"""NWM Training — graph-viz-vla simplified"""
from graphviz import Digraph

GRAPH_ATTR = {'dpi': '300', 'rankdir': 'TB', 'compound': 'true', 'fontname': 'Helvetica'}
CLUSTER = {'style': 'dashed', 'fontname': 'Helvetica-Bold', 'fontsize': '14'}
INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'fontsize': '11'}
ENCODE_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'fontsize': '11'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'fontsize': '11'}
CONTROL_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'fontsize': '11'}
LOSS_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFCDD2', 'fontsize': '11'}
GRAD_BORDER = {'shape': 'box', 'style': 'dashed,filled', 'fillcolor': '#FFFFFF', 'color': '#C62828', 'fontsize': '10'}
DATA_EDGE = {'color': '#333333', 'fontsize': '9'}
COND_EDGE = {'color': '#F57F17', 'fontsize': '9', 'style': 'dashed'}


def build_graph():
    dot = Digraph('nwm_training', graph_attr=GRAPH_ATTR)
    dot.node('dims', 'D=1152  B=batch  goals=4  T_diff=1000\nx_t=sqrt(a_bar)*x0+sqrt(1-a_bar)*eps',
             shape='note', fillcolor='#FAFAFA', fontsize='10')
    dot.node('grad_boundary', 'FROZEN: VAE Encoder\nTRAINABLE: CDiT x28, Embedders, Final Layer\n(~1B params, EMA decay=0.9999)', **GRAD_BORDER)

    with dot.subgraph(name='cluster_stage1') as s1:
        s1.attr(label='Stage 1: VAE Encode (frozen, no_grad)', **CLUSTER)
        s1.node('images', 'Training Images\n8 frames: 4 ctx + 4 goal\n[B,8,3,224,224]', **INPUT_NODE)
        s1.node('vae', 'VAE Encoder\nSD-VAE x0.18215\n[B*8,4,32,32]', **ENCODE_NODE)
        s1.node('split', 'Context/Goal Split\nx_cond [B*4,4,32,32]\nx_start [B*4,4,32,32]', **FUSION_NODE)
        s1.edge('images', 'vae', **DATA_EDGE)
        s1.edge('vae', 'split', label='[B*8,4,32,32]', **DATA_EDGE)

    with dot.subgraph(name='cluster_stage2') as s2:
        s2.attr(label='Stage 2: CDiT Forward (trainable)', **CLUSTER)
        s2.node('sample', 'Sample eps,t\neps~N(0,I) t~U[0,999]\n[B*4,4,32,32] [B*4]', **INPUT_NODE)
        s2.node('noisy', 'Noisy Latent\nq_sample x_t\n[B*4,4,32,32]', **FUSION_NODE)
        s2.node('embed', 'Patch Embed + Condition\nctx KV + target Q\nxi=psi_a+psi_k+psi_t', **ENCODE_NODE)
        s2.node('cdit', 'CDiT x28\nCross-Attn (not Joint)\n[B*4,256,1152]', **ENCODE_NODE)
        s2.node('out', 'Final Layer\n8ch: eps + sigma\n[B*4,8,32,32]', **ENCODE_NODE)
        s2.edge('sample', 'noisy', label='x0+eps+t', **DATA_EDGE)
        s2.edge('noisy', 'embed', label='x_t', **DATA_EDGE)
        s2.edge('embed', 'cdit', label='Q+KV+xi', **DATA_EDGE)
        s2.edge('cdit', 'out', label='[B*4,256,1152]', **DATA_EDGE)

    with dot.subgraph(name='cluster_stage3') as s3:
        s3.attr(label='Stage 3: Loss', **CLUSTER)
        s3.node('loss', 'Total Loss\nL_mse + L_vlb\nAdamW lr=8e-5', **LOSS_NODE)

    dot.edge('split', 'embed', label='ctx latents [B*4,4,32,32]', **DATA_EDGE)
    dot.edge('split', 'noisy', label='goal x0', **DATA_EDGE)
    dot.edge('out', 'loss', label='pred eps,sigma', **DATA_EDGE)
    dot.edge('sample', 'loss', label='GT eps', **COND_EDGE)
    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g = build_graph()
    g.render('nwm_training', format='pdf', cleanup=True)
    g.render('nwm_training', format='png', cleanup=True)
    print('Done: nwm_training.pdf, .png')

"""DreamZero Training — graph-viz-vla simplified (Teacher Forcing + Flow Matching)"""
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
    dot = Digraph('dreamzero_training', graph_attr=GRAPH_ATTR)
    dot.node('dims',
             'D_dit=5120  x_t=(1-sigma)*x0+sigma*noise\n'
             'loss=MSE(v_pred, noise-x0)  TF: causal chunk attn',
             shape='note', fillcolor='#FAFAFA', fontsize='10')
    dot.node('grad_boundary',
             'FROZEN: T5, CLIP, Wan VAE (~5.3B)\n'
             'TRAINABLE: LoRA on DiT (~20M), Action Enc, Camera Ctrl (~30M total)',
             **GRAD_BORDER)

    with dot.subgraph(name='cluster_stage1') as s1:
        s1.attr(label='Stage 1: Encoders (frozen, no_grad)', **CLUSTER)
        s1.node('video', 'Training Video\n[B,3,T,H,W]', **INPUT_NODE)
        s1.node('text', 'Text + Image\nT5 ids + CLIP ref', **INPUT_NODE)
        s1.node('encoders', 'T5 + CLIP + VAE Enc\nfrozen\ncontext, clean_latent', **ENCODE_NODE)
        s1.edge('video', 'encoders', label='[B,3,T,H,W]', **DATA_EDGE)
        s1.edge('text', 'encoders', **DATA_EDGE)

    with dot.subgraph(name='cluster_stage2') as s2:
        s2.attr(label='Stage 2: Teacher Forcing Forward (trainable, LoRA)', **CLUSTER)
        s2.node('sample', 'Sample noise,sigma\nnoise~N(0,I) t~FM sched\n[B,16,T,H,W]', **INPUT_NODE)
        s2.node('noisy', 'Noisy Latent\nx_t=(1-sigma)*x0+sigma*eps\n[B,16,T,H,W]', **FUSION_NODE)
        s2.node('act_enc', 'Multi-Embodiment\nAction Encoder\ncat_ids, [B,32,64]', **ENCODE_NODE)
        s2.node('concat', 'Token Concat\nclean|noisy|action|state\n[B,1793,5120]', **FUSION_NODE)
        s2.node('wan_dit', 'CausalWanModel x40\nCausal Chunk Self-Attn\n+ Cross-Attn(context)', **ENCODE_NODE)
        s2.node('v_pred', 'Velocity Field\nv_pred, [B,1793,16]', **ENCODE_NODE)
        s2.edge('sample', 'noisy', label='x0+eps+sigma', **DATA_EDGE)
        s2.edge('noisy', 'concat', label='noisy tokens', **DATA_EDGE)
        s2.edge('act_enc', 'concat', label='action tokens', **DATA_EDGE)
        s2.edge('concat', 'wan_dit', label='[B,1793,5120]', **DATA_EDGE)
        s2.edge('wan_dit', 'v_pred', label='[B,1793,16]', **DATA_EDGE)

    with dot.subgraph(name='cluster_stage3') as s3:
        s3.attr(label='Stage 3: Loss', **CLUSTER)
        s3.node('loss', 'Flow Matching Loss\nMSE(v_pred, noise-x0)\nvideo+action joint', **LOSS_NODE)

    dot.edge('encoders', 'noisy', label='clean_latent x0', **DATA_EDGE)
    dot.edge('encoders', 'concat', label='clean tokens + context', **COND_EDGE)
    dot.edge('encoders', 'wan_dit', label='Cross-Attn KV', **COND_EDGE)
    dot.edge('v_pred', 'loss', label='pred v', **DATA_EDGE)
    dot.edge('sample', 'loss', label='target noise-x0', **COND_EDGE)
    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g = build_graph()
    g.render('dreamzero_training', format='pdf', cleanup=True)
    g.render('dreamzero_training', format='png', cleanup=True)
    print('Done: dreamzero_training.pdf, .png')

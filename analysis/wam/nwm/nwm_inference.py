#!/usr/bin/env python3
"""NWM inference graph — follows script/agent/graph-viz-vla.md"""
from graphviz import Digraph

INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'fontsize': '11', 'fontcolor': '#2E7D32'}
ENCODING_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'fontsize': '11', 'fontcolor': '#1565C0'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'fontsize': '11', 'fontcolor': '#E65100'}
CONTROL_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFFDE7', 'fontsize': '11', 'fontcolor': '#F57F17'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'fontsize': '12', 'fontcolor': '#C62828'}
EDGE = {'color': '#333333', 'fontsize': '9', 'fontcolor': '#333333'}
EDGE_LOOP = {'color': '#E65100', 'style': 'dashed', 'fontsize': '9', 'fontcolor': '#E65100'}

OUT = '/home/jerett/OpenProject/MyAgent/analysis/wam/nwm/nwm_inference'

dot = Digraph('NWM_Inference', format='pdf')
dot.attr(rankdir='TB', size='12,18', dpi='300')
dot.attr('node', fontname='DejaVu Sans Mono')

with dot.subgraph(name='cluster_p1') as c:
    c.attr(label='Phase 1: Encode Context (run x1)', style='dashed', color='#1565C0', fontsize='13')
    c.node('ctx', 'Context Images\nPIL RGB frames\nctx, [B,4,3,224,224]', **INPUT_NODE)
    c.node('act', 'Navigation Action\nnormalized delta\naction, [B,3]', **INPUT_NODE)
    c.node('rel', 'Time Shift\nrel_t scalar\nrel_t, [B]', **INPUT_NODE)
    c.node('vae_e', 'VAE Encoder\nSD-VAE-ft-ema\nlatent, [B,4,4,32,32]', **ENCODING_NODE)
    c.node('pe_ctx', 'Context PatchEmbed\nViT PE + patch=2\nctx_tokens, [B,G,1024,1152]', **ENCODING_NODE)
    c.node('emb_a', 'Action Embedder\nFourier PE + MLP\npsi_a, [B,G,1152]', **CONTROL_NODE)
    c.node('emb_k', 'Time Embedder\nFourier PE + MLP\npsi_k, [B,G,1152]', **CONTROL_NODE)
    c.edge('ctx', 'vae_e', label='uint8→float [-1,1]\nHWC→BCHW', **EDGE)
    c.edge('vae_e', 'pe_ctx', label='x0.18215 scale\n4 frames→1024 KV tokens', **EDGE)
    c.edge('act', 'emb_a', label='3 scalars', **EDGE)
    c.edge('rel', 'emb_k', label='k/128', **EDGE)

with dot.subgraph(name='cluster_p2') as c:
    c.attr(label='Phase 2: DDPM Denoise Loop x250 (repeat t=249→0)', style='dashed', color='#E65100', fontsize='13')
    c.node('noise', 'Gaussian Noise\nDDPM init\nz, [B,G,4,32,32]', **INPUT_NODE)
    c.node('pe_tgt', 'Target PatchEmbed\npos_embed[target]\ntgt_tokens, [B,G,256,1152]', **ENCODING_NODE)
    c.node('emb_t', 'Diffusion Timestep\nFourier PE + MLP\npsi_t, [B,G,1152]', **CONTROL_NODE)
    c.node('cond', 'Condition Fusion\nadditive sum\nxi, [B,G,1152]', **CONTROL_NODE)
    c.node('cdit', 'CDiT Block xN\nCross-Attn Fusion\nout, [B,G,256,1152]', **FUSION_NODE)
    c.node('unpatch', 'Unpatchify + sigma\nlearned variance\neps, [B,G,8,32,32]', **ENCODING_NODE)
    c.node('sample', 'DDPM p_sample\nsingle denoise step\nz_{t-1}, [B,G,4,32,32]', **ENCODING_NODE)
    c.node('vae_d', 'VAE Decoder\nSD-VAE-ft-ema\nrgb, [B,G,3,224,224]', **ENCODING_NODE)
    c.node('out', 'Predicted Frame\nclip to [-1,1]\npred, [B,G,3,224,224]', **OUTPUT_NODE)
    c.edge('emb_a', 'cond', label='psi_a', **EDGE)
    c.edge('emb_k', 'cond', label='psi_k', **EDGE)
    c.edge('emb_t', 'cond', label='psi_t per step', **EDGE)
    c.edge('noise', 'pe_tgt', label='z→tokens', **EDGE)
    c.edge('pe_tgt', 'cdit', label='Self-Attn Q\n256 tokens', **EDGE)
    c.edge('pe_ctx', 'cdit', label='Cross-Attn KV\n1024 context tokens', **EDGE)
    c.edge('cond', 'cdit', label='adaLN-Zero modulate', **EDGE)
    c.edge('cdit', 'unpatch', label='transformed tokens', **EDGE)
    c.edge('unpatch', 'sample', label='eps + sigma', **EDGE)
    c.edge('sample', 'vae_d', label='t=0 latent /0.18215', **EDGE)
    c.edge('vae_d', 'out', label='decode RGB', **EDGE)
    c.edge('sample', 'pe_tgt', label='loop: update z\n250 steps', **EDGE_LOOP)
    c.edge('sample', 'emb_t', label='t decrements', **EDGE_LOOP)

for fmt in ('pdf', 'png'):
    g = Digraph('NWM_Inference', format=fmt)
    g.body = dot.body
    g.render(OUT, cleanup=True)
print(f'Saved {OUT}.pdf / .png')

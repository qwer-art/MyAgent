"""LeggedGym (Cassie) Inference — graph-viz-vla simplified"""
from graphviz import Digraph

GRAPH_ATTR = {'dpi': '300', 'rankdir': 'TB', 'compound': 'true', 'fontname': 'Helvetica'}
CLUSTER = {'style': 'dashed', 'fontname': 'Helvetica-Bold', 'fontsize': '14'}
INPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9', 'fontsize': '11'}
ENCODE_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'fontsize': '11'}
FUSION_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0', 'fontsize': '11'}
OUTPUT_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FCE4EC', 'fontsize': '11'}
DATA_EDGE = {'color': '#333333', 'fontsize': '9'}
COND_EDGE = {'color': '#9E9E9E', 'fontsize': '9', 'style': 'dashed'}


def build_graph():
    dot = Digraph('legged_gym_inference', graph_attr=GRAPH_ATTR)
    dot.node('dims', 'N=4096 envs  obs_dim=169  act_dim=12\npolicy_dt=0.02s  decimation=4  dt=0.005s',
             shape='note', fillcolor='#FAFAFA', fontsize='10')

    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs (IsaacGym sensors)', **CLUSTER)
        c.node('sensors', 'Robot State\nlin_vel, ang_vel, gravity\ncommands, dof, height\nraw -> [N,169]', **INPUT_NODE)

    with dot.subgraph(name='cluster_phase1') as p1:
        p1.attr(label='Phase 1: Policy (run x1 per step)', **CLUSTER)
        p1.node('obs', 'Observation Build\nscale + concat + noise\nobs, [N,169]', **FUSION_NODE)
        p1.node('actor', 'Actor MLP\n[512,256,128] ELU\naction, [N,12]', **ENCODE_NODE)
        p1.edge('obs', 'actor', label='[N,169]', **DATA_EDGE)

    with dot.subgraph(name='cluster_phase2') as p2:
        p2.attr(label='Phase 2: Control + Sim (repeat x4 substeps)', **CLUSTER)
        p2.node('pd', 'PD Controller\nKp/Kd torque\n[N,12]', **FUSION_NODE)
        p2.node('sim', 'IsaacGym PhysX\ngym.simulate x4\nrefresh state', **ENCODE_NODE)
        p2.edge('pd', 'sim', label='torques [N,12]', **DATA_EDGE)

    dot.node('actions', 'Joint Targets\n[N,12] scaled', **OUTPUT_NODE)
    dot.edge('sensors', 'obs', label='sensor tensors', **DATA_EDGE)
    dot.edge('actor', 'pd', label='action_mean [N,12]', **DATA_EDGE)
    dot.edge('actor', 'actions', label='[N,12]', **DATA_EDGE)
    dot.edge('sim', 'sensors', label='state feedback', **COND_EDGE)
    return dot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g = build_graph()
    g.render('inference', format='pdf', cleanup=True)
    g.render('inference', format='png', cleanup=True)
    print('Done: inference.pdf, .png')

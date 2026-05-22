"""
ActorCritic Module Detail Diagram (Cassie Config)
Uses Python graphviz library to generate

Dependency: pip install graphviz
Run: python actor_critic.py
Output: actor_critic.pdf, actor_critic.png

Classic Config: CassieRoughCfgPPO
- num_actor_obs = 169, num_critic_obs = 169 (no privileged obs)
- num_actions = 12
- actor_hidden_dims = [512, 256, 128], critic_hidden_dims = [512, 256, 128]
- activation = ELU, init_noise_std = 1.0
"""

from graphviz import Digraph

# ---- Style Definitions ----
GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'nodesep': '0.4',
    'ranksep': '0.5',
}

MODULE_NODE = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#E8F4FD',
    'fontsize': '14',
    'width': '3',
    'height': '1',
}

OP_NODE = {
    'shape': 'box',
    'style': 'rounded',
    'fillcolor': '#FFF3E0',
    'fontsize': '11',
}

DATA_EDGE = {
    'color': '#333333',
    'fontsize': '10',
    'fontcolor': '#666666',
}

CLUSTER_ATTR = {
    'style': 'dashed',
    'color': '#999999',
    'fontname': 'Helvetica-Bold',
    'fontsize': '16',
}

INPUT_NODE = {
    'shape': 'ellipse',
    'style': 'filled',
    'fillcolor': '#E8F5E9',
    'fontsize': '12',
}

OUTPUT_NODE = {
    'shape': 'ellipse',
    'style': 'filled',
    'fillcolor': '#FFEBEE',
    'fontsize': '12',
}

# ---- Graph Construction ----
def build_graph():
    dot = Digraph(
        name='ActorCritic_Detail',
        graph_attr=GRAPH_ATTR,
    )

    # ========== Actor MLP ==========
    with dot.subgraph(name='cluster_actor') as c:
        c.attr(label='Actor MLP (self.actor)\nSource: actor_critic.py:58-67', **CLUSTER_ATTR)

        c.node('actor_input', 'Input\n[N, 169]\nnum_actor_obs', **INPUT_NODE)

        # Layer 0
        c.node('actor_l0', 'Linear(169 -> 512)\nweight: [512, 169]\nbias: [512]', **OP_NODE)
        c.node('actor_a0', 'ELU', **OP_NODE)

        # Layer 1
        c.node('actor_l1', 'Linear(512 -> 256)\nweight: [256, 512]\nbias: [256]', **OP_NODE)
        c.node('actor_a1', 'ELU', **OP_NODE)

        # Layer 2
        c.node('actor_l2', 'Linear(256 -> 128)\nweight: [128, 256]\nbias: [128]', **OP_NODE)
        c.node('actor_a2', 'ELU', **OP_NODE)

        # Output
        c.node('actor_lout', 'Linear(128 -> 12)\nweight: [12, 128]\nbias: [12]\nnum_actions', **OP_NODE)

        c.edge('actor_input', 'actor_l0', label='[N,169]', **DATA_EDGE)
        c.edge('actor_l0', 'actor_a0', label='[N,512]', **DATA_EDGE)
        c.edge('actor_a0', 'actor_l1', label='[N,512]', **DATA_EDGE)
        c.edge('actor_l1', 'actor_a1', label='[N,256]', **DATA_EDGE)
        c.edge('actor_a1', 'actor_l2', label='[N,256]', **DATA_EDGE)
        c.edge('actor_l2', 'actor_a2', label='[N,128]', **DATA_EDGE)
        c.edge('actor_a2', 'actor_lout', label='[N,128]', **DATA_EDGE)

    # ========== Learnable Std ==========
    with dot.subgraph(name='cluster_std') as c:
        c.attr(label='Learnable Std (self.std)', **CLUSTER_ATTR)

        c.node('std_param', 'nn.Parameter\ninit_noise_std * ones(12)\n[12]\n(state-independent)', **OP_NODE)

    # ========== Action Distribution ==========
    with dot.subgraph(name='cluster_dist') as c:
        c.attr(label='Action Distribution\n(update_distribution, act, act_inference)', **CLUSTER_ATTR)

        c.node('mean', 'action_mean\n= self.actor(obs)\n[N, 12]', **OP_NODE)
        c.node('dist', 'Normal(mean, std)\nself.distribution', **OP_NODE)

        # Training path
        c.node('sample', '.sample()\n(Training: stochastic)', **OP_NODE)
        c.node('log_prob', '.log_prob(actions)\n.sum(dim=-1)\n[N] -> [N,1]', **OP_NODE)

        # Inference path
        c.node('mean_only', 'mean only\n(Inference: deterministic)\nact_inference()', **OP_NODE)

        c.edge('actor_lout', 'mean', label='[N,12]', **DATA_EDGE)
        c.edge('std_param', 'dist', label='[12]', **DATA_EDGE)
        c.edge('mean', 'dist', label='[N,12]', **DATA_EDGE)
        c.edge('dist', 'sample', label='Normal', **DATA_EDGE)
        c.edge('dist', 'mean_only', label='Normal', **DATA_EDGE)
        c.edge('sample', 'log_prob', label='actions [N,12]', **DATA_EDGE)

    # ========== Critic MLP ==========
    with dot.subgraph(name='cluster_critic') as c:
        c.attr(label='Critic MLP (self.critic)\nSource: actor_critic.py:69-79', **CLUSTER_ATTR)

        c.node('critic_input', 'Input\n[N, 169]\nnum_critic_obs\n(= num_actor_obs for Cassie)', **INPUT_NODE)

        # Layer 0
        c.node('critic_l0', 'Linear(169 -> 512)\nweight: [512, 169]\nbias: [512]', **OP_NODE)
        c.node('critic_a0', 'ELU', **OP_NODE)

        # Layer 1
        c.node('critic_l1', 'Linear(512 -> 256)\nweight: [256, 512]\nbias: [256]', **OP_NODE)
        c.node('critic_a1', 'ELU', **OP_NODE)

        # Layer 2
        c.node('critic_l2', 'Linear(256 -> 128)\nweight: [128, 256]\nbias: [128]', **OP_NODE)
        c.node('critic_a2', 'ELU', **OP_NODE)

        # Output
        c.node('critic_lout', 'Linear(128 -> 1)\nweight: [1, 128]\nbias: [1]\n(scalar value)', **OP_NODE)

        c.edge('critic_input', 'critic_l0', label='[N,169]', **DATA_EDGE)
        c.edge('critic_l0', 'critic_a0', label='[N,512]', **DATA_EDGE)
        c.edge('critic_a0', 'critic_l1', label='[N,512]', **DATA_EDGE)
        c.edge('critic_l1', 'critic_a1', label='[N,256]', **DATA_EDGE)
        c.edge('critic_a1', 'critic_l2', label='[N,256]', **DATA_EDGE)
        c.edge('critic_l2', 'critic_a2', label='[N,128]', **DATA_EDGE)
        c.edge('critic_a2', 'critic_lout', label='[N,128]', **DATA_EDGE)

    # ========== Outputs ==========
    dot.node('action_output', 'Action Output\n[N, 12]', **OUTPUT_NODE)
    dot.node('value_output', 'Value Output\n[N, 1]\nV(s)', **OUTPUT_NODE)

    dot.edge('sample', 'action_output', label='[N,12]', **DATA_EDGE)
    dot.edge('mean_only', 'action_output', label='[N,12]', **DATA_EDGE)
    dot.edge('critic_lout', 'value_output', label='[N,1]', **DATA_EDGE)

    # ========== Parameter Count Estimate ==========
    # Actor: 169*512 + 512 + 512*256 + 256 + 256*128 + 128 + 128*12 + 12 = 86,916 + 131,328 + 32,896 + 1,548 = ~252,688
    # Critic: same structure = ~252,688
    # Std: 12
    # Total: ~505,388 (0.51M)
    dot.node('param_count', 'Total Parameters\nActor: ~252K\nCritic: ~252K\nStd: 12\nTotal: ~0.51M',
             shape='note', style='filled', fillcolor='#F5F5F5', fontsize='11')

    return dot

# ---- Render ----
if __name__ == '__main__':
    graph = build_graph()
    graph.render(format='pdf', cleanup=True)
    graph.render(format='png', cleanup=True)
    print(f"Generated: {graph.filename}.pdf, {graph.filename}.png")
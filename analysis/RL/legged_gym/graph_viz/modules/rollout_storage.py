"""
RolloutStorage & GAE Detail Diagram
Uses Python graphviz library to generate

Dependency: pip install graphviz
Run: python rollout_storage.py
Output: rollout_storage.pdf, rollout_storage.png

Classic Config: CassieRoughCfgPPO
- num_envs = 4096, num_steps_per_env = 24
- gamma = 0.99, lam (gae_lambda) = 0.95
- Total transitions per iteration: 24 * 4096 = 98,304
- Mini-batch size: 98,304 / 4 = 24,576
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
        name='RolloutStorage_Detail',
        graph_attr=GRAPH_ATTR,
    )

    # ========== Transition Data Structure ==========
    with dot.subgraph(name='cluster_transition') as c:
        c.attr(label='Transition (per step)\nSource: rollout_storage.py:37-51', **CLUSTER_ATTR)

        c.node('trans_obs', 'observations\n[N, 169]', **INPUT_NODE)
        c.node('trans_critic_obs', 'critic_observations\n[N, 169]', **INPUT_NODE)
        c.node('trans_actions', 'actions\n[N, 12]', **INPUT_NODE)
        c.node('trans_rewards', 'rewards\n[N]', **INPUT_NODE)
        c.node('trans_dones', 'dones\n[N]', **INPUT_NODE)
        c.node('trans_values', 'values\n[N, 1]', **INPUT_NODE)
        c.node('trans_log_prob', 'actions_log_prob\n[N, 1]', **INPUT_NODE)
        c.node('trans_mean', 'action_mean\n[N, 12]', **INPUT_NODE)
        c.node('trans_sigma', 'action_sigma\n[N, 12]', **INPUT_NODE)
        c.node('trans_hidden', 'hidden_states\n(None for MLP)', **INPUT_NODE)

    # ========== Storage Buffers ==========
    with dot.subgraph(name='cluster_storage') as c:
        c.attr(label='RolloutStorage Buffers\nSource: rollout_storage.py:53-86\nShape: [num_steps=24, num_envs=4096, *]', **CLUSTER_ATTR)

        c.node('buf_obs', 'observations\n[24, 4096, 169]', **OP_NODE)
        c.node('buf_critic_obs', 'privileged_observations\n[24, 4096, 169]\n(or None)', **OP_NODE)
        c.node('buf_actions', 'actions\n[24, 4096, 12]', **OP_NODE)
        c.node('buf_rewards', 'rewards\n[24, 4096, 1]', **OP_NODE)
        c.node('buf_dones', 'dones\n[24, 4096, 1]\n(byte)', **OP_NODE)
        c.node('buf_values', 'values\n[24, 4096, 1]', **OP_NODE)
        c.node('buf_log_prob', 'actions_log_prob\n[24, 4096, 1]', **OP_NODE)
        c.node('buf_returns', 'returns\n[24, 4096, 1]\n(computed by GAE)', **OP_NODE)
        c.node('buf_advantages', 'advantages\n[24, 4096, 1]\n(computed by GAE)', **OP_NODE)
        c.node('buf_mu', 'mu\n[24, 4096, 12]', **OP_NODE)
        c.node('buf_sigma', 'sigma\n[24, 4096, 12]', **OP_NODE)

    # ========== GAE Computation ==========
    with dot.subgraph(name='cluster_gae') as c:
        c.attr(label='GAE Computation\nSource: rollout_storage.py:123-137\ncompute_returns(last_values, gamma=0.99, lam=0.95)', **CLUSTER_ATTR)

        c.node('gae_loop', 'GAE Loop (reversed t=23..0)\nfor step in reversed(range(24)):\n  next_values = values[step+1]\n  (or last_values for t=23)\n  next_not_terminal = 1 - dones[step]', **OP_NODE)

        c.node('delta', 'delta = rewards[step]\n  + gamma * next_values * next_not_terminal\n  - values[step]\n[4096, 1]', **OP_NODE)

        c.node('advantage', 'advantage = delta\n  + next_not_terminal * gamma * lam * advantage\n  (recursive, lam=0.95)\n[4096, 1]', **OP_NODE)

        c.node('returns', 'returns[step] = advantage\n  + values[step]\n[4096, 1]', **OP_NODE)

        c.node('norm_adv', 'advantages = returns - values\nadvantages = (A - mean)\n  / (std + 1e-8)\n[24, 4096, 1]', **OP_NODE)

        c.edge('gae_loop', 'delta', label='', **DATA_EDGE)
        c.edge('delta', 'advantage', label='', **DATA_EDGE)
        c.edge('advantage', 'returns', label='', **DATA_EDGE)
        c.edge('returns', 'norm_adv', label='', **DATA_EDGE)

    # ========== Mini-Batch Generator ==========
    with dot.subgraph(name='cluster_minibatch') as c:
        c.attr(label='Mini-Batch Generator\nSource: rollout_storage.py:147-183\nmini_batch_generator(num_mini_batches=4, num_epochs=5)', **CLUSTER_ATTR)

        c.node('flatten', 'Flatten\n[24, 4096, *] -> [98304, *]', **OP_NODE)
        c.node('shuffle', 'Random Shuffle\ntorch.randperm(98304)', **OP_NODE)
        c.node('split', 'Split into 4 Mini-Batches\neach: [24576, *]', **OP_NODE)

        c.node('mb_obs', 'obs_batch\n[24576, 169]', **OP_NODE)
        c.node('mb_critic_obs', 'critic_obs_batch\n[24576, 169]', **OP_NODE)
        c.node('mb_actions', 'actions_batch\n[24576, 12]', **OP_NODE)
        c.node('mb_values', 'target_values_batch\n[24576, 1]', **OP_NODE)
        c.node('mb_returns', 'returns_batch\n[24576, 1]', **OP_NODE)
        c.node('mb_advantages', 'advantages_batch\n[24576, 1]', **OP_NODE)
        c.node('mb_log_prob', 'old_log_prob_batch\n[24576, 1]', **OP_NODE)
        c.node('mb_mu', 'old_mu_batch\n[24576, 12]', **OP_NODE)
        c.node('mb_sigma', 'old_sigma_batch\n[24576, 12]', **OP_NODE)

        c.edge('flatten', 'shuffle', label='[98304, *]', **DATA_EDGE)
        c.edge('shuffle', 'split', label='[98304, *]', **DATA_EDGE)
        c.edge('split', 'mb_obs', label='', **DATA_EDGE)
        c.edge('split', 'mb_critic_obs', label='', **DATA_EDGE)
        c.edge('split', 'mb_actions', label='', **DATA_EDGE)
        c.edge('split', 'mb_values', label='', **DATA_EDGE)
        c.edge('split', 'mb_returns', label='', **DATA_EDGE)
        c.edge('split', 'mb_advantages', label='', **DATA_EDGE)
        c.edge('split', 'mb_log_prob', label='', **DATA_EDGE)
        c.edge('split', 'mb_mu', label='', **DATA_EDGE)
        c.edge('split', 'mb_sigma', label='', **DATA_EDGE)

    # ========== Cross-cluster edges ==========
    # Transition -> Storage
    dot.edge('trans_obs', 'buf_obs', label='add_transitions()', **DATA_EDGE)
    dot.edge('trans_actions', 'buf_actions', label='', **DATA_EDGE)
    dot.edge('trans_rewards', 'buf_rewards', label='', **DATA_EDGE)
    dot.edge('trans_dones', 'buf_dones', label='', **DATA_EDGE)
    dot.edge('trans_values', 'buf_values', label='', **DATA_EDGE)
    dot.edge('trans_log_prob', 'buf_log_prob', label='', **DATA_EDGE)
    dot.edge('trans_mean', 'buf_mu', label='', **DATA_EDGE)
    dot.edge('trans_sigma', 'buf_sigma', label='', **DATA_EDGE)

    # Storage -> GAE
    dot.edge('buf_rewards', 'gae_loop', label='[24,4096,1]', **DATA_EDGE)
    dot.edge('buf_values', 'gae_loop', label='[24,4096,1]', **DATA_EDGE)
    dot.edge('buf_dones', 'gae_loop', label='[24,4096,1]', **DATA_EDGE)

    # GAE -> Mini-batch
    dot.edge('norm_adv', 'flatten', label='advantages', **DATA_EDGE)
    dot.edge('buf_returns', 'flatten', label='returns', **DATA_EDGE)

    # Output
    dot.node('output', 'Mini-Batch Yield\n(obs, critic_obs, actions,\nvalues, advantages, returns,\nold_log_prob, old_mu, old_sigma,\n(None, None), None)', **OUTPUT_NODE)
    dot.edge('mb_obs', 'output', label='', **DATA_EDGE)

    return dot

# ---- Render ----
if __name__ == '__main__':
    graph = build_graph()
    graph.render(format='pdf', cleanup=True)
    graph.render(format='png', cleanup=True)
    print(f"Generated: {graph.filename}.pdf, {graph.filename}.png")
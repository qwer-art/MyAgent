"""
PPO Algorithm Detail Diagram (Cassie Config)
Uses Python graphviz library to generate

Dependency: pip install graphviz
Run: python ppo.py
Output: ppo.pdf, ppo.png

Classic Config: LeggedRobotCfgPPO
- clip_param = 0.2, gamma = 0.99, lam (gae_lambda) = 0.95
- desired_kl = 0.01, entropy_coef = 0.01, value_loss_coef = 1.0
- max_grad_norm = 1.0, num_learning_epochs = 5, num_mini_batches = 4
- schedule = 'adaptive' (learning rate)
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

LOSS_NODE = {
    'shape': 'diamond',
    'style': 'filled',
    'fillcolor': '#FFCDD2',
    'fontsize': '11',
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
        name='PPO_Detail',
        graph_attr=GRAPH_ATTR,
    )

    # ========== Step 1: Re-evaluate with Current Policy ==========
    with dot.subgraph(name='cluster_reeval') as c:
        c.attr(label='Step 1: Re-evaluate with Current Policy\n(PPO.update -> actor_critic.evaluate)', **CLUSTER_ATTR)

        c.node('obs_batch', 'obs_batch\n[mini_N, 169]', **INPUT_NODE)
        c.node('critic_obs_batch', 'critic_obs_batch\n[mini_N, 169]', **INPUT_NODE)
        c.node('actions_batch', 'actions_batch\n[mini_N, 12]', **INPUT_NODE)

        c.node('actor_fwd', 'Actor Forward\nactor MLP -> action_mean\n[N, 12]', **MODULE_NODE)
        c.node('critic_fwd', 'Critic Forward\ncritic MLP -> value\n[N, 1]', **MODULE_NODE)

        c.node('new_dist', 'Normal(mean, std)\nupdate_distribution()', **OP_NODE)
        c.node('new_log_prob', 'new_log_prob\n= dist.log_prob(actions)\n.sum(dim=-1)\n[mini_N, 1]', **OP_NODE)
        c.node('new_entropy', 'entropy\n= dist.entropy()\n.sum(dim=-1)\n[mini_N]', **OP_NODE)

        c.edge('obs_batch', 'actor_fwd', label='[mini_N,169]', **DATA_EDGE)
        c.edge('critic_obs_batch', 'critic_fwd', label='[mini_N,169]', **DATA_EDGE)
        c.edge('actor_fwd', 'new_dist', label='mean [mini_N,12]', **DATA_EDGE)
        c.edge('actions_batch', 'new_log_prob', label='[mini_N,12]', **DATA_EDGE)
        c.edge('new_dist', 'new_log_prob', label='Normal', **DATA_EDGE)
        c.edge('new_dist', 'new_entropy', label='Normal', **DATA_EDGE)

    # ========== Step 2: Importance Ratio & Clipping ==========
    with dot.subgraph(name='cluster_surrogate') as c:
        c.attr(label='Step 2: Importance Ratio & Clipping\n(PPO.update:128-141)', **CLUSTER_ATTR)

        c.node('old_log_prob', 'old_log_prob\n[mini_N, 1]\n(from storage)', **INPUT_NODE)
        c.node('advantages', 'advantages\n[mini_N, 1]\n(normalized)', **INPUT_NODE)

        c.node('ratio', 'ratio = exp(new_log_prob\n- old_log_prob)\n[mini_N, 1]', **OP_NODE)
        c.node('clipped_ratio', 'clamp(ratio,\n1 - 0.2,\n1 + 0.2)\n[mini_N, 1]', **OP_NODE)
        c.node('surrogate_unclipped', 'ratio * advantages\n[mini_N, 1]', **OP_NODE)
        c.node('surrogate_clipped', 'clipped_ratio * advantages\n[mini_N, 1]', **OP_NODE)
        c.node('surrogate_loss', 'Surrogate Loss\n= -min(unclipped, clipped)\n.mean()\n(scalar)', **LOSS_NODE)

        c.edge('new_log_prob', 'ratio', label='', **DATA_EDGE)
        c.edge('old_log_prob', 'ratio', label='', **DATA_EDGE)
        c.edge('ratio', 'clipped_ratio', label='', **DATA_EDGE)
        c.edge('ratio', 'surrogate_unclipped', label='', **DATA_EDGE)
        c.edge('advantages', 'surrogate_unclipped', label='', **DATA_EDGE)
        c.edge('clipped_ratio', 'surrogate_clipped', label='', **DATA_EDGE)
        c.edge('advantages', 'surrogate_clipped', label='', **DATA_EDGE)
        c.edge('surrogate_unclipped', 'surrogate_loss', label='', **DATA_EDGE)
        c.edge('surrogate_clipped', 'surrogate_loss', label='', **DATA_EDGE)

    # ========== Step 3: Value Loss ==========
    with dot.subgraph(name='cluster_value') as c:
        c.attr(label='Step 3: Value Loss (Clipped)\n(PPO.update:143-152)', **CLUSTER_ATTR)

        c.node('returns', 'returns\n[mini_N, 1]\n(from GAE)', **INPUT_NODE)
        c.node('old_values', 'values (old)\n[mini_N, 1]\n(from storage)', **INPUT_NODE)

        c.node('value_pred', 'value_pred\n= critic(obs)\n[mini_N, 1]', **OP_NODE)
        c.node('value_clipped', 'value_clipped\n= old_values + clamp(\n  value_pred - old_values,\n  -0.2, +0.2)\n[mini_N, 1]', **OP_NODE)
        c.node('v_loss_unclipped', '(V_pred - R)^2\n[mini_N, 1]', **OP_NODE)
        c.node('v_loss_clipped', '(V_clipped - R)^2\n[mini_N, 1]', **OP_NODE)
        c.node('value_loss', 'Value Loss\n= max(v_unclipped, v_clipped)\n.mean()\n(scalar)', **LOSS_NODE)

        c.edge('critic_fwd', 'value_pred', label='[mini_N,1]', **DATA_EDGE)
        c.edge('value_pred', 'v_loss_unclipped', label='', **DATA_EDGE)
        c.edge('returns', 'v_loss_unclipped', label='', **DATA_EDGE)
        c.edge('value_pred', 'value_clipped', label='', **DATA_EDGE)
        c.edge('old_values', 'value_clipped', label='', **DATA_EDGE)
        c.edge('value_clipped', 'v_loss_clipped', label='', **DATA_EDGE)
        c.edge('returns', 'v_loss_clipped', label='', **DATA_EDGE)
        c.edge('v_loss_unclipped', 'value_loss', label='', **DATA_EDGE)
        c.edge('v_loss_clipped', 'value_loss', label='', **DATA_EDGE)

    # ========== Step 4: KL Divergence & Adaptive LR ==========
    with dot.subgraph(name='cluster_kl') as c:
        c.attr(label='Step 4: KL Divergence & Adaptive LR\n(PPO.update:154-171)', **CLUSTER_ATTR)

        c.node('old_mu', 'old_mu\n[mini_N, 12]\n(from storage)', **INPUT_NODE)
        c.node('old_sigma', 'old_sigma\n[mini_N, 12]\n(from storage)', **INPUT_NODE)

        c.node('new_mu', 'new_mu\n= actor_mean\n[mini_N, 12]', **OP_NODE)
        c.node('new_sigma', 'new_sigma\n= std.expand\n[mini_N, 12]', **OP_NODE)

        c.node('kl_div', 'KL(old || new)\n= log(sigma_new/sigma_old)\n+ (sigma_old^2 + (mu_old-mu_new)^2)\n  / (2*sigma_new^2) - 0.5\n.mean()', **OP_NODE)

        c.node('adaptive_lr', 'Adaptive LR\nif kl > 2*0.01: lr /= 1.5\nif kl < 0.01/2: lr *= 1.5\nschedule=adaptive', **OP_NODE)

        c.edge('old_mu', 'kl_div', label='', **DATA_EDGE)
        c.edge('old_sigma', 'kl_div', label='', **DATA_EDGE)
        c.edge('new_mu', 'kl_div', label='', **DATA_EDGE)
        c.edge('new_sigma', 'kl_div', label='', **DATA_EDGE)
        c.edge('actor_fwd', 'new_mu', label='', **DATA_EDGE)
        c.edge('kl_div', 'adaptive_lr', label='kl_mean', **DATA_EDGE)

    # ========== Step 5: Total Loss & Gradient Step ==========
    with dot.subgraph(name='cluster_total') as c:
        c.attr(label='Step 5: Total Loss & Gradient Step\n(PPO.update:173-185)', **CLUSTER_ATTR)

        c.node('entropy_loss', 'Entropy Bonus\n= -0.01 * entropy.mean()\n(encourage exploration)', **LOSS_NODE)
        c.node('total_loss', 'Total Loss\n= surrogate_loss\n+ 1.0 * value_loss\n- 0.01 * entropy\n(scalar)', **LOSS_NODE)
        c.node('backward', 'loss.backward()', **OP_NODE)
        c.node('clip_grad', 'nn.utils.clip_grad_norm_\nmodel.parameters()\nmax_norm=1.0', **OP_NODE)
        c.node('optimizer_step', 'optimizer.step()\n(Adam, adaptive LR)', **OP_NODE)

        c.edge('surrogate_loss', 'total_loss', label='', **DATA_EDGE)
        c.edge('value_loss', 'total_loss', label='x 1.0', **DATA_EDGE)
        c.edge('new_entropy', 'entropy_loss', label='', **DATA_EDGE)
        c.edge('entropy_loss', 'total_loss', label='-0.01', **DATA_EDGE)
        c.edge('total_loss', 'backward', label='', **DATA_EDGE)
        c.edge('backward', 'clip_grad', label='gradients', **DATA_EDGE)
        c.edge('clip_grad', 'optimizer_step', label='clipped grads', **DATA_EDGE)
        c.edge('adaptive_lr', 'optimizer_step', label='lr', **DATA_EDGE)

    # ========== Output ==========
    dot.node('ppo_output', 'PPO Update Output\nmean_value_loss, mean_surrogate_loss\nmean_entropy, mean_kl', **OUTPUT_NODE)
    dot.edge('optimizer_step', 'ppo_output', label='', **DATA_EDGE)

    return dot

# ---- Render ----
if __name__ == '__main__':
    graph = build_graph()
    graph.render(format='pdf', cleanup=True)
    graph.render(format='png', cleanup=True)
    print(f"Generated: {graph.filename}.pdf, {graph.filename}.png")
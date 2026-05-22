"""
LeggedGym (Cassie) Training Architecture Diagram
Uses Python graphviz library to generate

Dependency: pip install graphviz
Run: python training.py
Output: training.pdf, training.png

Classic Config: CassieRoughCfg + LeggedRobotCfgPPO
- num_actor_obs = 169, num_critic_obs = 169 (no privileged obs)
- num_actions = 12, num_envs = 4096
- num_steps_per_env = 24, num_learning_epochs = 5, num_mini_batches = 4
- actor_hidden_dims = [512, 256, 128], critic_hidden_dims = [512, 256, 128]
- clip_param = 0.2, gamma = 0.99, lam = 0.95, desired_kl = 0.01
- learning_rate = 1e-3 (adaptive), entropy_coef = 0.01, value_loss_coef = 1.0
"""

from graphviz import Digraph

# ---- Style Definitions ----
GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'compound': 'true',
    'nodesep': '0.5',
    'ranksep': '0.7',
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

LOSS_NODE = {
    'shape': 'diamond',
    'style': 'filled',
    'fillcolor': '#FFCDD2',
    'fontsize': '11',
}

# ---- Graph Construction ----
def build_graph():
    dot = Digraph(
        name='LeggedGym_Cassie_Training',
        graph_attr=GRAPH_ATTR,
    )

    # ========== Phase 1: Rollout Collection (OnPolicyRunner.learn) ==========
    with dot.subgraph(name='cluster_rollout') as c:
        c.attr(label='Phase 1: Rollout Collection\n(OnPolicyRunner.learn, num_steps=24)', **CLUSTER_ATTR)

        # Environment
        c.node('env_step', 'LeggedRobot.step()\nIsaacGym env\n[N=4096 envs]', **MODULE_NODE)

        # Actor-Critic act
        c.node('actor_act', 'ActorCritic.act()\nobs -> actor MLP -> Normal(mean, std)\n-> sample()', **MODULE_NODE)
        c.node('critic_eval', 'ActorCritic.evaluate()\ncritic_obs -> critic MLP -> V(s)', **MODULE_NODE)

        # Store transition
        c.node('store_trans', 'Store Transition\n(RolloutStorage.add_transitions)', **OP_NODE)

        # Data flow in rollout
        c.node('obs_input', 'obs [N, 169]', **INPUT_NODE)
        c.node('critic_obs_input', 'critic_obs [N, 169]', **INPUT_NODE)

        c.edge('obs_input', 'actor_act', label='[N,169]', **DATA_EDGE)
        c.edge('critic_obs_input', 'critic_eval', label='[N,169]', **DATA_EDGE)
        c.edge('actor_act', 'env_step', label='actions [N,12]', **DATA_EDGE)
        c.edge('env_step', 'store_trans', label='obs, rew, done', **DATA_EDGE)
        c.edge('actor_act', 'store_trans', label='actions, log_prob,\nmean, sigma, values', **DATA_EDGE)
        c.edge('critic_eval', 'store_trans', label='values [N,1]', **DATA_EDGE)

        # Timeout bootstrapping
        c.node('timeout_boot', 'Timeout Bootstrapping\nrew += gamma*V*time_outs\n(if time_outs in infos)', **OP_NODE)
        c.edge('env_step', 'timeout_boot', label='time_outs', **DATA_EDGE)
        c.edge('timeout_boot', 'store_trans', label='augmented rew', **DATA_EDGE)

    # ========== Phase 2: Compute Returns (GAE) ==========
    with dot.subgraph(name='cluster_returns') as c:
        c.attr(label='Phase 2: Compute Returns (GAE)\n(RolloutStorage.compute_returns)', **CLUSTER_ATTR)

        c.node('gae_loop', 'GAE Loop (reversed)\ndelta = rew + gamma*V_next*(1-done) - V\nA_t = delta + (1-done)*gamma*lam*A_{t+1}\nreturns = A + V', **OP_NODE)
        c.node('norm_adv', 'Normalize Advantages\nA = (A - mean) / (std + 1e-8)', **OP_NODE)
        c.node('last_values', 'Last Values\nV(critic_obs_last)', **MODULE_NODE)

        c.edge('store_trans', 'gae_loop', label='rewards, values, dones\n[24, N, 1]', **DATA_EDGE)
        c.edge('last_values', 'gae_loop', label='V_last [N,1]', **DATA_EDGE)
        c.edge('gae_loop', 'norm_adv', label='advantages [24,N,1]', **DATA_EDGE)

    # ========== Phase 3: PPO Update ==========
    with dot.subgraph(name='cluster_ppo') as c:
        c.attr(label='Phase 3: PPO Update\n(PPO.update, epochs=5, mini_batches=4)', **CLUSTER_ATTR)

        # Mini-batch generator
        c.node('mini_batch', 'Mini-Batch Generator\nflatten [24,N] -> [24*N]\nshuffle -> 4 mini-batches', **OP_NODE)

        # Re-evaluate with current policy
        c.node('re_eval_actor', 'Re-evaluate Actor\nactor MLP -> Normal(mean, std)\nnew_log_prob, new_mu, new_sigma', **MODULE_NODE)
        c.node('re_eval_critic', 'Re-evaluate Critic\ncritic MLP -> V(s)', **MODULE_NODE)

        # KL divergence (adaptive LR)
        c.node('kl_div', 'KL Divergence\nKL(old || new)\n= sum(log(sigma_new/sigma_old)\n+ (sigma_old^2 + (mu_old-mu_new)^2)\n/ (2*sigma_new^2) - 0.5)', **OP_NODE)
        c.node('adaptive_lr', 'Adaptive LR\nkl > 2*desired_kl: lr /= 1.5\nkl < desired_kl/2: lr *= 1.5\n(desired_kl=0.01)', **OP_NODE)

        # Surrogate loss
        c.node('ratio', 'ratio = exp(new_log_prob\n- old_log_prob)', **OP_NODE)
        c.node('surrogate', 'Surrogate Loss\nL = max(-A*ratio,\n-A*clamp(ratio, 1-0.2, 1+0.2))\n.mean()', **LOSS_NODE)

        # Value loss
        c.node('value_loss', 'Value Loss (clipped)\nL = max((V-R)^2,\n(V_clip-R)^2).mean()\nclip_param=0.2', **LOSS_NODE)

        # Entropy bonus
        c.node('entropy', 'Entropy Bonus\nH = Normal.entropy()\n.sum(dim=-1)', **OP_NODE)

        # Total loss
        c.node('total_loss', 'Total Loss\nL = surrogate_loss\n+ 1.0 * value_loss\n- 0.01 * entropy', **LOSS_NODE)

        # Gradient step
        c.node('grad_step', 'Gradient Step\noptimizer.zero_grad()\nloss.backward()\nclip_grad_norm(max=1.0)\noptimizer.step()', **OP_NODE)

        # Edges within PPO
        c.edge('norm_adv', 'mini_batch', label='advantages, returns\n[24*N, 1]', **DATA_EDGE)
        c.edge('mini_batch', 're_eval_actor', label='obs_batch [mini_N, 169]', **DATA_EDGE)
        c.edge('mini_batch', 're_eval_critic', label='critic_obs_batch', **DATA_EDGE)

        c.edge('re_eval_actor', 'ratio', label='new_log_prob', **DATA_EDGE)
        c.edge('mini_batch', 'ratio', label='old_log_prob, advantages', **DATA_EDGE)
        c.edge('ratio', 'surrogate', label='ratio, clipped_ratio', **DATA_EDGE)

        c.edge('re_eval_actor', 'kl_div', label='new_mu, new_sigma', **DATA_EDGE)
        c.edge('mini_batch', 'kl_div', label='old_mu, old_sigma', **DATA_EDGE)
        c.edge('kl_div', 'adaptive_lr', label='kl_mean', **DATA_EDGE)

        c.edge('re_eval_critic', 'value_loss', label='V_batch', **DATA_EDGE)
        c.edge('mini_batch', 'value_loss', label='returns, target_values', **DATA_EDGE)

        c.edge('re_eval_actor', 'entropy', label='distribution', **DATA_EDGE)

        c.edge('surrogate', 'total_loss', label='', **DATA_EDGE)
        c.edge('value_loss', 'total_loss', label='x 1.0', **DATA_EDGE)
        c.edge('entropy', 'total_loss', label='x 0.01', **DATA_EDGE)

        c.edge('total_loss', 'grad_step', label='', **DATA_EDGE)
        c.edge('adaptive_lr', 'grad_step', label='lr', **DATA_EDGE)

    # ========== Logging & Saving ==========
    with dot.subgraph(name='cluster_log') as c:
        c.attr(label='Logging & Saving\n(OnPolicyRunner)', **CLUSTER_ATTR)

        c.node('tensorboard', 'TensorBoard\nLoss/value, Loss/surrogate\nPolicy/mean_noise_std\nTrain/mean_reward', **OP_NODE)
        c.node('save_ckpt', 'Save Checkpoint\nmodel_{iter}.pt\n(every 50 iters)', **OP_NODE)

        c.edge('grad_step', 'tensorboard', label='mean_value_loss,\nmean_surrogate_loss', **DATA_EDGE)
        c.edge('grad_step', 'save_ckpt', label='model_state_dict,\noptimizer_state_dict', **DATA_EDGE)

    # ========== Cross-cluster edges ==========
    # Rollout -> Returns
    dot.edge('store_trans', 'gae_loop', label='stored transitions\n[24, N, *]', **DATA_EDGE)

    return dot

# ---- Render ----
if __name__ == '__main__':
    graph = build_graph()
    graph.render(format='pdf', cleanup=True)
    graph.render(format='png', cleanup=True)
    print(f"Generated: {graph.filename}.pdf, {graph.filename}.png")
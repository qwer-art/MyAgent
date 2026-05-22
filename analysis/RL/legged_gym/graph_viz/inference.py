"""
LeggedGym (Cassie) Inference Architecture Diagram
Uses Python graphviz library to generate

Dependency: pip install graphviz
Run: python inference.py
Output: inference.pdf, inference.png

Classic Config: CassieRoughCfg
- num_actor_obs = 169, num_actions = 12
- actor_hidden_dims = [512, 256, 128], activation = ELU
- control_type = 'P', action_scale = 0.5, decimation = 4
- sim.dt = 0.005s, policy_dt = 0.02s (50Hz)
"""

from graphviz import Digraph

# ---- Style Definitions ----
GRAPH_ATTR = {
    'dpi': '300',
    'rankdir': 'TB',
    'fontname': 'Helvetica',
    'bgcolor': 'white',
    'compound': 'true',
    'nodesep': '0.6',
    'ranksep': '0.8',
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
        name='LeggedGym_Cassie_Inference',
        graph_attr=GRAPH_ATTR,
    )

    # ========== Observation Construction (LeggedRobot.compute_observations) ==========
    with dot.subgraph(name='cluster_obs') as c:
        c.attr(label='Observation Construction\n(LeggedRobot.compute_observations)', **CLUSTER_ATTR)

        # Raw sensor inputs
        c.node('base_lin_vel', 'base_lin_vel\n[N, 3]', **INPUT_NODE)
        c.node('base_ang_vel', 'base_ang_vel\n[N, 3]', **INPUT_NODE)
        c.node('projected_gravity', 'projected_gravity\n[N, 3]', **INPUT_NODE)
        c.node('commands', 'commands[:3]\n(vx, vy, vyaw)\n[N, 3]', **INPUT_NODE)
        c.node('dof_pos', 'dof_pos - default\n[N, 12]', **INPUT_NODE)
        c.node('dof_vel', 'dof_vel\n[N, 12]', **INPUT_NODE)
        c.node('prev_actions', 'prev_actions\n[N, 12]', **INPUT_NODE)
        c.node('height_meas', 'height_measurements\n[N, 121]', **INPUT_NODE)

        # Scaling operations
        c.node('scale_lin_vel', 'x obs_scales.lin_vel\n(2.0)', **OP_NODE)
        c.node('scale_ang_vel', 'x obs_scales.ang_vel\n(0.25)', **OP_NODE)
        c.node('scale_commands', 'x commands_scale\n(2.0, 2.0, 0.25)', **OP_NODE)
        c.node('scale_dof_pos', 'x obs_scales.dof_pos\n(1.0)', **OP_NODE)
        c.node('scale_dof_vel', 'x obs_scales.dof_vel\n(0.05)', **OP_NODE)
        c.node('scale_height', 'x obs_scales.height\n(5.0)', **OP_NODE)

        # Concat
        c.node('concat_obs', 'Concat\n[N, 48]', **OP_NODE)
        c.node('concat_height', 'Concat\n[N, 169]', **OP_NODE)

        # Noise
        c.node('add_noise', '+ Uniform Noise\nnoise_level=1.0', **OP_NODE)
        c.node('clip_obs', 'Clip [-100, 100]', **OP_NODE)

        # Edges within observation cluster
        c.edge('base_lin_vel', 'scale_lin_vel', label='[N,3]', **DATA_EDGE)
        c.edge('base_ang_vel', 'scale_ang_vel', label='[N,3]', **DATA_EDGE)
        c.edge('commands', 'scale_commands', label='[N,3]', **DATA_EDGE)
        c.edge('dof_pos', 'scale_dof_pos', label='[N,12]', **DATA_EDGE)
        c.edge('dof_vel', 'scale_dof_vel', label='[N,12]', **DATA_EDGE)
        c.edge('height_meas', 'scale_height', label='[N,121]', **DATA_EDGE)

        c.edge('scale_lin_vel', 'concat_obs', label='[N,3]', **DATA_EDGE)
        c.edge('scale_ang_vel', 'concat_obs', label='[N,3]', **DATA_EDGE)
        c.edge('projected_gravity', 'concat_obs', label='[N,3]', **DATA_EDGE)
        c.edge('scale_commands', 'concat_obs', label='[N,3]', **DATA_EDGE)
        c.edge('scale_dof_pos', 'concat_obs', label='[N,12]', **DATA_EDGE)
        c.edge('scale_dof_vel', 'concat_obs', label='[N,12]', **DATA_EDGE)
        c.edge('prev_actions', 'concat_obs', label='[N,12]', **DATA_EDGE)

        c.edge('concat_obs', 'concat_height', label='[N,48]', **DATA_EDGE)
        c.edge('scale_height', 'concat_height', label='[N,121]', **DATA_EDGE)
        c.edge('concat_height', 'add_noise', label='[N,169]', **DATA_EDGE)
        c.edge('add_noise', 'clip_obs', label='[N,169]', **DATA_EDGE)

    # ========== Actor Network (ActorCritic.act_inference) ==========
    with dot.subgraph(name='cluster_actor') as c:
        c.attr(label='ActorCritic (Inference)\nact_inference()', **CLUSTER_ATTR)

        c.node('actor_mlp', 'Actor MLP\n[512, 256, 128] + ELU\nInput: [N, 169]\nOutput: [N, 12]\nParams: ~0.11M', **MODULE_NODE)

    # ========== PD Controller (LeggedRobot._compute_torques) ==========
    with dot.subgraph(name='cluster_pd') as c:
        c.attr(label='PD Controller\n(LeggedRobot._compute_torques)', **CLUSTER_ATTR)

        c.node('scale_action', 'x action_scale\n(0.5)', **OP_NODE)
        c.node('add_default', '+ default_dof_pos', **OP_NODE)
        c.node('sub_dof_pos', '- dof_pos', **OP_NODE)
        c.node('kp_gain', 'x Kp\n[100,100,200,200,200,40]\nx2 (L/R)', **OP_NODE)
        c.node('kd_gain', 'x Kd x dof_vel\n[3,3,6,6,6,1]\nx2 (L/R)', **OP_NODE)
        c.node('sum_torque', 'Kp*(target-pos)\n- Kd*vel\n(Torque)', **OP_NODE)
        c.node('clip_torque', 'Clip to\ntorque_limits', **OP_NODE)

        c.edge('scale_action', 'add_default', label='[N,12]', **DATA_EDGE)
        c.edge('add_default', 'sub_dof_pos', label='target [N,12]', **DATA_EDGE)
        c.edge('sub_dof_pos', 'kp_gain', label='error [N,12]', **DATA_EDGE)
        c.edge('kp_gain', 'sum_torque', label='[N,12]', **DATA_EDGE)
        c.edge('kd_gain', 'sum_torque', label='[N,12]', **DATA_EDGE)
        c.edge('sum_torque', 'clip_torque', label='[N,12]', **DATA_EDGE)

    # ========== IsaacGym Simulation ==========
    with dot.subgraph(name='cluster_sim') as c:
        c.attr(label='IsaacGym Simulation\n(PhysX, decimation=4)', **CLUSTER_ATTR)

        c.node('set_torque', 'set_dof_actuation\n_force_tensor', **OP_NODE)
        c.node('simulate', 'gym.simulate()\nx4 substeps\ndt=0.005s each', **OP_NODE)
        c.node('refresh', 'Refresh state tensors\n(root_states, dof_state,\ncontact_forces)', **OP_NODE)

        c.edge('set_torque', 'simulate', label='torques [N,12]', **DATA_EDGE)
        c.edge('simulate', 'refresh', label='', **DATA_EDGE)

    # ========== Post-Physics Step ==========
    with dot.subgraph(name='cluster_post') as c:
        c.attr(label='Post-Physics Step\n(LeggedRobot.post_physics_step)', **CLUSTER_ATTR)

        c.node('compute_obs', 'compute_observations()', **OP_NODE)
        c.node('check_term', 'check_termination()\npelvis contact?', **OP_NODE)
        c.node('compute_rew', 'compute_reward()', **OP_NODE)
        c.node('reset_envs', 'reset_idx()\n(terminated envs)', **OP_NODE)

        c.edge('refresh', 'compute_obs', label='sensor data', **DATA_EDGE)
        c.edge('refresh', 'check_term', label='contact_forces', **DATA_EDGE)
        c.edge('check_term', 'compute_rew', label='', **DATA_EDGE)
        c.edge('compute_rew', 'reset_envs', label='', **DATA_EDGE)

    # ========== Cross-cluster edges ==========
    # Observation -> Actor
    dot.edge('clip_obs', 'actor_mlp', label='obs [N, 169]', **DATA_EDGE)

    # Actor -> PD Controller
    dot.edge('actor_mlp', 'scale_action', label='action_mean [N, 12]', **DATA_EDGE)

    # PD Controller -> Simulation
    dot.edge('clip_torque', 'set_torque', label='torques [N, 12]', **DATA_EDGE)

    # Post-physics step feedback loop
    dot.edge('compute_obs', 'base_lin_vel', label='obs feedback\n(loop back)', style='dashed', color='#999999', fontsize='10', fontcolor='#666666')

    # ========== Output node ==========
    dot.node('output_action', 'Action Output\n[N, 12]\n(12 joint targets)', **OUTPUT_NODE)
    dot.edge('actor_mlp', 'output_action', label='action_mean [N, 12]', style='bold', color='#D32F2F', fontsize='10', fontcolor='#666666')

    return dot

# ---- Render ----
if __name__ == '__main__':
    graph = build_graph()
    graph.render(format='pdf', cleanup=True)
    graph.render(format='png', cleanup=True)
    print(f"Generated: {graph.filename}.pdf, {graph.filename}.png")

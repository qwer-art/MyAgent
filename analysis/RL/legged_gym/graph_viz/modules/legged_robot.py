"""
LeggedRobot Environment Detail Diagram (Cassie Config)
Uses Python graphviz library to generate

Dependency: pip install graphviz
Run: python legged_robot.py
Output: legged_robot.pdf, legged_robot.png

Classic Config: CassieRoughCfg
- num_envs = 4096, num_observations = 169, num_actions = 12
- control_type = 'P', action_scale = 0.5, decimation = 4
- sim.dt = 0.005s, policy_dt = 0.02s (50Hz)
- PD stiffness: hip_abduction=100, hip_rotation=100, hip_flexion=200, thigh=200, ankle=200, toe=40
- PD damping: hip_abduction=3, hip_rotation=3, hip_flexion=6, thigh=6, ankle=6, toe=1
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
        name='LeggedRobot_Detail',
        graph_attr=GRAPH_ATTR,
    )

    # ========== step() ==========
    with dot.subgraph(name='cluster_step') as c:
        c.attr(label='LeggedRobot.step()\nSource: legged_robot.py:79-103', **CLUSTER_ATTR)

        c.node('clip_actions', 'Clip Actions\nclip(actions, -100, 100)', **OP_NODE)
        c.node('decimation_loop', 'Decimation Loop (4x)\nfor _ in range(4):', **OP_NODE)
        c.node('compute_torques', '_compute_torques()\nPD Controller', **MODULE_NODE)
        c.node('set_forces', 'gym.set_dof_actuation\n_force_tensor()', **OP_NODE)
        c.node('simulate', 'gym.simulate()\nPhysX step dt=0.005s', **OP_NODE)
        c.node('refresh_dof', 'gym.refresh_dof\n_state_tensor()', **OP_NODE)
        c.node('post_physics', 'post_physics_step()', **MODULE_NODE)
        c.node('clip_obs', 'Clip Observations\nclip(obs, -100, 100)', **OP_NODE)

        c.edge('clip_actions', 'decimation_loop', label='actions [N,12]', **DATA_EDGE)
        c.edge('decimation_loop', 'compute_torques', label='actions', **DATA_EDGE)
        c.edge('compute_torques', 'set_forces', label='torques [N,12]', **DATA_EDGE)
        c.edge('set_forces', 'simulate', label='', **DATA_EDGE)
        c.edge('simulate', 'refresh_dof', label='', **DATA_EDGE)
        c.edge('decimation_loop', 'post_physics', label='(after 4 substeps)', **DATA_EDGE)
        c.edge('post_physics', 'clip_obs', label='', **DATA_EDGE)

    # ========== _compute_torques (PD Controller) ==========
    with dot.subgraph(name='cluster_pd') as c:
        c.attr(label='PD Controller\n_compute_torques()\nSource: legged_robot.py:353-375\ncontrol_type = P', **CLUSTER_ATTR)

        c.node('scale_action', 'actions * action_scale\n(0.5)', **OP_NODE)
        c.node('target_pos', 'target_pos = scaled_action\n+ default_dof_pos', **OP_NODE)
        c.node('pos_error', 'pos_error = target_pos\n- dof_pos', **OP_NODE)
        c.node('kp_term', 'Kp * pos_error\n[100,100,200,200,\n 200,40] x2 (L/R)', **OP_NODE)
        c.node('kd_term', 'Kd * dof_vel\n[3,3,6,6,6,1]\nx2 (L/R)', **OP_NODE)
        c.node('torque_sum', 'torque = Kp*error\n- Kd*vel', **OP_NODE)
        c.node('clip_torque', 'clip(torque,\n-torque_limits,\n+torque_limits)', **OP_NODE)

        c.edge('scale_action', 'target_pos', label='[N,12]', **DATA_EDGE)
        c.edge('target_pos', 'pos_error', label='[N,12]', **DATA_EDGE)
        c.edge('pos_error', 'kp_term', label='[N,12]', **DATA_EDGE)
        c.edge('kp_term', 'torque_sum', label='[N,12]', **DATA_EDGE)
        c.edge('kd_term', 'torque_sum', label='[N,12]', **DATA_EDGE)
        c.edge('torque_sum', 'clip_torque', label='[N,12]', **DATA_EDGE)

    # ========== post_physics_step ==========
    with dot.subgraph(name='cluster_post') as c:
        c.attr(label='post_physics_step()\nSource: legged_robot.py:105-136', **CLUSTER_ATTR)

        c.node('refresh_root', 'Refresh root states\n& contact forces', **OP_NODE)
        c.node('compute_vel', 'Compute Velocities\nbase_lin_vel = quat_inv(root[7:10])\nbase_ang_vel = quat_inv(root[10:13])\nprojected_gravity = quat_inv(gravity)', **OP_NODE)
        c.node('callback', '_post_physics_step_callback()\n- resample commands\n- heading command\n- measure heights\n- push robots', **OP_NODE)
        c.node('check_term', 'check_termination()\npelvis contact > 1N\nOR time_out', **OP_NODE)
        c.node('compute_rew', 'compute_reward()\nweighted sum of reward fn', **OP_NODE)
        c.node('reset_idx', 'reset_idx()\nreset terminated envs', **OP_NODE)
        c.node('compute_obs', 'compute_observations()\nconcat + noise + clip', **OP_NODE)
        c.node('update_last', 'Update last_* buffers\nlast_actions, last_dof_vel\nlast_root_vel', **OP_NODE)

        c.edge('refresh_root', 'compute_vel', label='', **DATA_EDGE)
        c.edge('compute_vel', 'callback', label='', **DATA_EDGE)
        c.edge('callback', 'check_term', label='', **DATA_EDGE)
        c.edge('check_term', 'compute_rew', label='', **DATA_EDGE)
        c.edge('compute_rew', 'reset_idx', label='', **DATA_EDGE)
        c.edge('reset_idx', 'compute_obs', label='', **DATA_EDGE)
        c.edge('compute_obs', 'update_last', label='', **DATA_EDGE)

    # ========== Observation Construction ==========
    with dot.subgraph(name='cluster_obs') as c:
        c.attr(label='compute_observations()\nSource: legged_robot.py:209-226\nCassie: 169 = 3+3+3+3+12+12+12+121', **CLUSTER_ATTR)

        c.node('obs_lin_vel', 'base_lin_vel * 2.0\n[N, 3]', **OP_NODE)
        c.node('obs_ang_vel', 'base_ang_vel * 0.25\n[N, 3]', **OP_NODE)
        c.node('obs_gravity', 'projected_gravity\n[N, 3]', **OP_NODE)
        c.node('obs_commands', 'commands[:3] * scale\n[N, 3]', **OP_NODE)
        c.node('obs_dof_pos', '(dof_pos - default) * 1.0\n[N, 12]', **OP_NODE)
        c.node('obs_dof_vel', 'dof_vel * 0.05\n[N, 12]', **OP_NODE)
        c.node('obs_actions', 'prev_actions\n[N, 12]', **OP_NODE)
        c.node('obs_heights', 'height_meas * 5.0\nclip(root_z-0.5-heights, -1, 1)\n[N, 121]\n(11x11 grid)', **OP_NODE)

        c.node('concat_core', 'Concat\n[N, 48]', **OP_NODE)
        c.node('concat_full', 'Concat\n[N, 169]', **OP_NODE)
        c.node('add_noise', '+ Uniform Noise\nnoise_level=1.0', **OP_NODE)

        c.edge('obs_lin_vel', 'concat_core', label='[3]', **DATA_EDGE)
        c.edge('obs_ang_vel', 'concat_core', label='[3]', **DATA_EDGE)
        c.edge('obs_gravity', 'concat_core', label='[3]', **DATA_EDGE)
        c.edge('obs_commands', 'concat_core', label='[3]', **DATA_EDGE)
        c.edge('obs_dof_pos', 'concat_core', label='[12]', **DATA_EDGE)
        c.edge('obs_dof_vel', 'concat_core', label='[12]', **DATA_EDGE)
        c.edge('obs_actions', 'concat_core', label='[12]', **DATA_EDGE)

        c.edge('concat_core', 'concat_full', label='[48]', **DATA_EDGE)
        c.edge('obs_heights', 'concat_full', label='[121]', **DATA_EDGE)
        c.edge('concat_full', 'add_noise', label='[169]', **DATA_EDGE)

    # ========== Reward Functions (Cassie) ==========
    with dot.subgraph(name='cluster_rewards') as c:
        c.attr(label='Reward Functions (Cassie)\nSource: cassie_config.py:83-100 + legged_robot.py:816-906', **CLUSTER_ATTR)

        c.node('rew_tracking_lin', '_reward_tracking_lin_vel\nexp(-lin_vel_error^2/0.25)\nscale: 1.0\n(inherited from base)', **OP_NODE)
        c.node('rew_tracking_ang', '_reward_tracking_ang_vel\nexp(-ang_vel_error^2/0.25)\nscale: 1.0', **OP_NODE)
        c.node('rew_lin_vel_z', '_reward_lin_vel_z\nbase_lin_vel[:,2]^2\nscale: -0.5', **OP_NODE)
        c.node('rew_feet_air', '_reward_feet_air_time\n(air_time - 0.5) * first_contact\nscale: 5.0', **OP_NODE)
        c.node('rew_no_fly', '_reward_no_fly (Cassie)\nsum(contacts)==1\nscale: 0.25', **OP_NODE)
        c.node('rew_torques', '_reward_torques\nsum(torques^2)\nscale: -5e-6', **OP_NODE)
        c.node('rew_dof_acc', '_reward_dof_acc\nsum((last_vel-vel)^2/dt)\nscale: -2e-7', **OP_NODE)
        c.node('rew_dof_limits', '_reward_dof_pos_limits\nsum(out_of_limits)\nscale: -1.0', **OP_NODE)
        c.node('rew_termination', '_reward_termination\nreset_buf * ~time_out_buf\nscale: -200.0', **OP_NODE)

        c.node('sum_rewards', 'Sum All Rewards\nrew_buf = sum(reward_fn * scale)\n(only_positive_rewards=False)', **OP_NODE)

        c.edge('rew_tracking_lin', 'sum_rewards', label='1.0', **DATA_EDGE)
        c.edge('rew_tracking_ang', 'sum_rewards', label='1.0', **DATA_EDGE)
        c.edge('rew_lin_vel_z', 'sum_rewards', label='-0.5', **DATA_EDGE)
        c.edge('rew_feet_air', 'sum_rewards', label='5.0', **DATA_EDGE)
        c.edge('rew_no_fly', 'sum_rewards', label='0.25', **DATA_EDGE)
        c.edge('rew_torques', 'sum_rewards', label='-5e-6', **DATA_EDGE)
        c.edge('rew_dof_acc', 'sum_rewards', label='-2e-7', **DATA_EDGE)
        c.edge('rew_dof_limits', 'sum_rewards', label='-1.0', **DATA_EDGE)
        c.edge('rew_termination', 'sum_rewards', label='-200.0', **DATA_EDGE)

    # ========== Output ==========
    dot.node('env_output', 'Environment Output\n(obs_buf, privileged_obs_buf,\nrew_buf, reset_buf, extras)\nobs: [N,169], rew: [N]', **OUTPUT_NODE)
    dot.edge('clip_obs', 'env_output', label='', **DATA_EDGE)

    return dot

# ---- Render ----
if __name__ == '__main__':
    graph = build_graph()
    graph.render(format='pdf', cleanup=True)
    graph.render(format='png', cleanup=True)
    print(f"Generated: {graph.filename}.pdf, {graph.filename}.png")
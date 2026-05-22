# LeggedGym (Cassie) Architecture Diagrams

Generated from source code at `raw_data/RL/legged_gym` using the `graph_viz` skill.

## Classic Configuration: CassieRoughCfg

| Parameter | Value |
|-----------|-------|
| Robot | Cassie (bipedal, 12 DOF) |
| num_envs | 4096 |
| num_observations | 169 |
| num_actions | 12 |
| Actor MLP | [512, 256, 128] + ELU |
| Critic MLP | [512, 256, 128] + ELU |
| PPO clip_param | 0.2 |
| gamma / lambda | 0.99 / 0.95 |
| learning_rate | 1e-3 (adaptive) |
| num_steps_per_env | 24 |
| num_mini_batches | 4 |
| num_learning_epochs | 5 |

## Diagram Index

### Top-Level Diagrams

| Diagram | File | Description |
|---------|------|-------------|
| **Inference** | `inference.py` -> `LeggedGym_Cassie_Inference.gv.{pdf,png}` | Full inference pipeline: obs construction -> Actor MLP -> PD controller -> IsaacGym sim |
| **Training** | `training.py` -> `LeggedGym_Cassie_Training.gv.{pdf,png}` | Full training loop: rollout collection -> GAE -> PPO update -> logging |

### Sub-Module Detail Diagrams

| Diagram | File | Description |
|---------|------|-------------|
| **ActorCritic** | `modules/actor_critic.py` -> `ActorCritic_Detail.gv.{pdf,png}` | Actor MLP + Critic MLP + learnable std + Normal distribution (layer-by-layer) |
| **PPO** | `modules/ppo.py` -> `PPO_Detail.gv.{pdf,png}` | PPO update: ratio, clipping, value loss, KL divergence, adaptive LR, gradient step |
| **RolloutStorage** | `modules/rollout_storage.py` -> `RolloutStorage_Detail.gv.{pdf,png}` | Transition storage, GAE computation, mini-batch generator |
| **LeggedRobot** | `modules/legged_robot.py` -> `LeggedRobot_Detail.gv.{pdf,png}` | Environment: step(), PD controller, obs construction, reward functions |

## Observation Breakdown (Cassie, 169-dim)

| Component | Dim | Scale | Source |
|-----------|-----|-------|--------|
| base_lin_vel | 3 | 2.0 | body-frame linear velocity |
| base_ang_vel | 3 | 0.25 | body-frame angular velocity |
| projected_gravity | 3 | 1.0 | gravity in body frame |
| commands[:3] | 3 | (2.0, 2.0, 0.25) | (vx, vy, vyaw) commands |
| dof_pos - default | 12 | 1.0 | joint positions relative to default |
| dof_vel | 12 | 0.05 | joint velocities |
| prev_actions | 12 | 1.0 | previous step actions |
| height_measurements | 121 | 5.0 | 11x11 terrain height scan |
| **Total** | **169** | | |

## Cassie Reward Functions

| Reward | Scale | Formula |
|--------|-------|---------|
| tracking_lin_vel | 1.0 | exp(-lin_vel_error^2 / sigma) |
| tracking_ang_vel | 1.0 | exp(-ang_vel_error^2 / sigma) |
| lin_vel_z | -0.5 | base_lin_vel[:,2]^2 |
| feet_air_time | 5.0 | (air_time - 0.5) * first_contact |
| no_fly | 0.25 | sum(contacts)==1 (Cassie-specific) |
| torques | -5e-6 | sum(torques^2) |
| dof_acc | -2e-7 | sum((last_vel - vel)^2 / dt) |
| dof_pos_limits | -1.0 | sum(out_of_limits) |
| termination | -200.0 | reset_buf * ~time_out_buf |

## PD Controller (Cassie)

```
torque = Kp * (action * 0.5 + default_pos - dof_pos) - Kd * dof_vel
```

| Joint (per side) | Kp [N*m/rad] | Kd [N*m*s/rad] |
|------------------|-------------|----------------|
| hip_abduction | 100 | 3 |
| hip_rotation | 100 | 3 |
| hip_flexion | 200 | 6 |
| thigh_joint | 200 | 6 |
| ankle_joint | 200 | 6 |
| toe_joint | 40 | 1 |

## Verification Status

All module names, tensor shapes, and data flow connections verified against source code.
Key discrepancy found and fixed: `tracking_lin_vel = 1.0` (inherited from base config) was initially omitted from reward functions.

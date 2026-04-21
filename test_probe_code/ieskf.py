#!/usr/bin/env python3
"""
2D IESKF 最简实现 - 理解核心概念

IESKF的核心思想:
1. 名义状态 (nominal state): 直接积分更新，不考虑误差
2. 误差状态 (error state): 用卡尔曼滤波估计的小量
3. 迭代更新: 测量时多次迭代，逐步收敛到最优估计

状态向量 (5维):
    [x, y, vx, vy, θ]^T
    - x, y: 位置
    - vx, vy: 速度
    - θ: 偏航角

误差状态 (5维):
    [δx, δy, δvx, δvy, δθ]^T
"""

import numpy as np
import matplotlib.pyplot as plt


class IESKF_2D:
    """2维迭代误差状态卡尔曼滤波器"""

    def __init__(self):
        # ===== 名义状态 =====
        self.x = 0.0      # x位置
        self.y = 0.0      # y位置
        self.vx = 1.0     # x速度 (初始速度与真实状态一致)
        self.vy = 0.5     # y速度
        self.theta = 0.0  # 偏航角

        # ===== 误差状态协方差 =====
        # P表示误差状态的不确定性
        self.P = np.diag([0.001, 0.001, 0.001, 0.001, 0.0001])  # [x, y, vx, vy, θ]

        # ===== 噪声参数 =====
        self.Q_acc = 0.001    # 加速度过程噪声 (高质量IMU)
        self.Q_gyro = 0.0001  # 角速度过程噪声 (高质量IMU)

        # ===== 迭代参数 =====
        self.max_iter = 5
        self.tol = 1e-4

    def predict(self, ax, ay, omega, dt):
        """
        预测步骤: IMU积分

        输入:
            ax, ay: 加速度 (世界坐标系)
            omega: 角速度
            dt: 时间间隔
        """
        # ===== 1. 名义状态积分 =====
        # 位置: p = p + v*dt + 0.5*a*dt^2
        self.x += self.vx * dt + 0.5 * ax * dt**2
        self.y += self.vy * dt + 0.5 * ay * dt**2

        # 速度: v = v + a*dt
        self.vx += ax * dt
        self.vy += ay * dt

        # 姿态: θ = θ + ω*dt
        self.theta += omega * dt

        # ===== 2. 误差状态传播 =====
        # 状态转移矩阵 F: δx_k+1 = F * δx_k + w
        F = np.eye(5)
        F[0, 2] = dt  # ∂x/∂vx
        F[1, 3] = dt  # ∂y/∂vy

        # 过程噪声 Q (3维: ax, ay, omega)
        Q = np.diag([self.Q_acc * dt**2,
                     self.Q_acc * dt**2,
                     self.Q_gyro * dt**2])

        # 噪声映射矩阵 G: 将噪声映射到状态空间
        G = np.zeros((5, 3))
        G[2, 0] = dt  # ax -> vx
        G[3, 1] = dt  # ay -> vy
        G[4, 2] = dt  # omega -> theta

        # 协方差预测: P = F*P*F^T + G*Q*G^T
        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def update(self, z_x, z_y, R_meas):
        """
        迭代更新步骤: 位置测量

        这是IESKF的核心创新!

        输入:
            z_x, z_y: 位置测量
            R_meas: 测量噪声方差
        """
        # 保存先验状态 (迭代前的状态)
        state_prior = [self.x, self.y, self.vx, self.vy, self.theta]

        # 测量矩阵 H (观测位置)
        H = np.array([[1, 0, 0, 0, 0],   # 观测x
                      [0, 1, 0, 0, 0]])  # 观测y

        R = np.eye(2) * R_meas

        print("  迭代更新过程:")
        for i in range(self.max_iter):
            # ===== 计算残差 =====
            # r = 测量值 - 预测值
            r = np.array([z_x - self.x, z_y - self.y])

            # ===== 计算Kalman增益 =====
            # K = P * H^T * (H*P*H^T + R)^{-1}
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)

            # ===== 计算误差状态 =====
            dx = K @ r

            # ===== 更新名义状态 =====
            self.x = state_prior[0] + dx[0]
            self.y = state_prior[1] + dx[1]
            self.vx = state_prior[2] + dx[2]
            self.vy = state_prior[3] + dx[3]
            self.theta = state_prior[4] + dx[4]

            # ===== 检查收敛 =====
            dx_norm = np.linalg.norm(dx)
            print(f"    迭代{i+1}: 残差||r||={np.linalg.norm(r):.4f}, 更新||dx||={dx_norm:.6f}")

            if dx_norm < self.tol:
                print(f"    ✓ 收敛!")
                break

        # ===== 更新协方差 (Joseph形式) =====
        # 保证数值稳定性: P = (I-KH)*P*(I-KH)^T + K*R*K^T
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


def test_ieskf():
    """
    单一测试案例: 一般运动模式

    运动过程:
    1. 0-3秒: 匀速直线
    2. 3-6秒: 加速
    3. 6-10秒: 转弯
    """
    print("=" * 80)
    print("2D IESKF 测试: 一般运动模式")
    print("=" * 80)

    # 参数
    dt = 0.01  # 100Hz
    T = 10.0   # 10秒
    N = int(T / dt)

    # 初始化
    kf = IESKF_2D()

    # 真实状态
    true_state = {'x': 0.0, 'y': 0.0, 'vx': 1.0, 'vy': 0.5, 'theta': 0.0}

    # 记录数据
    traj_est = []
    traj_true = []
    times = []

    # 测量间隔
    meas_interval = 10  # 10Hz (增加测量频率)

    print(f"\n参数: IMU={1/dt:.0f}Hz, 测量={1/(dt*meas_interval):.0f}Hz, 时长={T:.0f}s\n")

    # 主循环
    for i in range(N):
        t = i * dt

        # ===== 生成运动指令 =====
        # 运动模式：匀速直线 -> 加速 -> 转弯
        if t < 3.0:
            # 匀速直线
            ax, ay, omega = 0.0, 0.0, 0.0
        elif t < 6.0:
            # 加速
            ax, ay, omega = 0.3, 0.2, 0.0
        else:
            # 转弯：通过向心加速度实现
            # 向心加速度 = v^2 / r，方向垂直于速度方向
            speed = np.sqrt(true_state['vx']**2 + true_state['vy']**2)
            # 计算当前速度方向
            vel_angle = np.arctan2(true_state['vy'], true_state['vx'])
            # 向心加速度方向垂直于速度方向（向左转）
            centripetal_acc = speed * 0.3  # 角速度约0.3 rad/s
            ax = -centripetal_acc * np.sin(vel_angle)
            ay = centripetal_acc * np.cos(vel_angle)
            omega = 0.3  # 角速度

        # ===== 更新真实状态 =====
        # 位置更新
        true_state['x'] += true_state['vx'] * dt + 0.5 * ax * dt**2
        true_state['y'] += true_state['vy'] * dt + 0.5 * ay * dt**2

        # 速度更新
        true_state['vx'] += ax * dt
        true_state['vy'] += ay * dt

        # 姿态更新
        true_state['theta'] += omega * dt

        # ===== IMU预测 =====
        # 添加噪声模拟真实IMU (高质量IMU)
        ax_meas = ax + np.random.randn() * 0.01  # 噪声标准差 0.01 m/s^2
        ay_meas = ay + np.random.randn() * 0.01
        omega_meas = omega + np.random.randn() * 0.001  # 噪声标准差 0.001 rad/s

        kf.predict(ax_meas, ay_meas, omega_meas, dt)

        # ===== 测量更新 =====
        if (i + 1) % meas_interval == 0:
            # 生成位置测量 (添加噪声)
            z_x = true_state['x'] + np.random.randn() * 0.05  # 测量噪声 0.05m
            z_y = true_state['y'] + np.random.randn() * 0.05

            print(f"t={t:.2f}s: 测量更新")
            print(f"  真实: ({true_state['x']:.2f}, {true_state['y']:.2f})")
            print(f"  测量: ({z_x:.2f}, {z_y:.2f})")
            print(f"  预测: ({kf.x:.2f}, {kf.y:.2f})")

            kf.update(z_x, z_y, R_meas=0.0025)  # 测量噪声方差 0.05^2 = 0.0025

            print(f"  更新后: ({kf.x:.2f}, {kf.y:.2f})\n")

        # 记录
        traj_est.append([kf.x, kf.y])
        traj_true.append([true_state['x'], true_state['y']])
        times.append(t)

    # 统计
    traj_est = np.array(traj_est)
    traj_true = np.array(traj_true)
    errors = np.linalg.norm(traj_est - traj_true, axis=1)

    print(f"\n统计: 误差均值={np.mean(errors):.4f}m, 最大={np.max(errors):.4f}m")

    # 可视化
    plot_results(times, traj_est, traj_true, errors)


def plot_results(times, traj_est, traj_true, errors):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 轨迹
    axes[0].plot(traj_true[:, 0], traj_true[:, 1], 'b-', lw=2, label='True')
    axes[0].plot(traj_est[:, 0], traj_est[:, 1], 'r--', lw=1.5, label='Estimated')
    axes[0].scatter(traj_true[0, 0], traj_true[0, 1], c='g', s=100, marker='o', label='Start')
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Y [m]')
    axes[0].set_title('Trajectory Comparison')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')

    # 误差曲线
    axes[1].plot(times, errors, 'g-', lw=1)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Error [m]')
    axes[1].set_title('Position Error')
    axes[1].grid(True)

    # 误差分布
    axes[2].hist(errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[2].axvline(np.mean(errors), color='r', ls='--', label=f'Mean={np.mean(errors):.3f}m')
    axes[2].set_xlabel('Error [m]')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    output_path = '/home/jerett/OpenProject/MyAgent/test_probe_code/workdirs/ieskf_2d.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n结果图: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IESKF 核心概念:")
    print("1. 名义状态: 直接积分 (预测)")
    print("2. 误差状态: 卡尔曼滤波估计 (更新)")
    print("3. 迭代更新: 多次迭代直到收敛")
    print("=" * 80 + "\n")

    test_ieskf()

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

测量模型: 2D激光雷达点云配准 (参考FAST-LIO2)
"""

import numpy as np
import matplotlib.pyplot as plt


class Lidar2D:
    """2D激光雷达模拟器"""

    def __init__(self, map_points, num_rays=180, max_range=20.0):
        """
        Args:
            map_points: 地图点云 (N x 2)
            num_rays: 射线数量
            max_range: 最大探测距离
        """
        self.map_points = map_points
        self.num_rays = num_rays
        self.max_range = max_range

    def scan(self, x, y, theta):
        """
        模拟激光雷达扫描

        Args:
            x, y: 位置
            theta: 偏航角

        Returns:
            scan_points: 扫描点云 (机体坐标系)
        """
        scan_points = []

        # 简化：直接返回可见的地图点（转换到机体坐标系）
        for map_point in self.map_points:
            # 地图点在世界坐标系
            px_world, py_world = map_point

            # 转换到机体坐标系
            dx = px_world - x
            dy = py_world - y

            # 旋转到机体坐标系
            cos_theta = np.cos(-theta)
            sin_theta = np.sin(-theta)
            px_body = cos_theta * dx - sin_theta * dy
            py_body = sin_theta * dx + cos_theta * dy

            # 只保留前方的点
            dist = np.sqrt(px_body**2 + py_body**2)
            if dist < self.max_range and dist > 0.5:  # 忽略太近的点
                scan_points.append([px_body, py_body])

        # 限制点数，避免太多
        if len(scan_points) > self.num_rays:
            indices = np.random.choice(len(scan_points), self.num_rays, replace=False)
            scan_points = [scan_points[i] for i in indices]

        return np.array(scan_points) if len(scan_points) > 0 else np.array([[0, 0]])


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

    def update_with_lidar(self, scan_points, lidar):
        """
        迭代更新步骤: 2D激光雷达点云配准

        这是IESKF的核心创新! 参考FAST-LIO2

        Args:
            scan_points: 当前扫描点云 (机体坐标系)
            lidar: Lidar2D对象
        """
        # 保存先验状态 (迭代前的状态)
        state_prior = [self.x, self.y, self.vx, self.vy, self.theta]

        print("  迭代更新过程:")
        for iteration in range(self.max_iter):
            # ===== 计算残差和雅可比 =====
            residuals = []
            jacobians = []

            # 将扫描点转换到世界坐标系
            cos_theta = np.cos(self.theta)
            sin_theta = np.sin(self.theta)

            for p_body in scan_points:
                # 转换到世界坐标系
                p_world = np.array([
                    cos_theta * p_body[0] - sin_theta * p_body[1] + self.x,
                    sin_theta * p_body[0] + cos_theta * p_body[1] + self.y
                ])

                # 找最近的地图点
                distances = np.linalg.norm(lidar.map_points - p_world, axis=1)
                nearest_idx = np.argmin(distances)
                q_nearest = lidar.map_points[nearest_idx]

                # 残差: 点到最近地图点的向量
                diff = p_world - q_nearest
                dist = np.linalg.norm(diff)

                if dist > 1e-6 and dist < 5.0:  # 限制最大距离，避免异常值
                    # 法向量 (指向扫描点)
                    normal = diff / dist
                    # 残差: 点到面的有符号距离 (可正可负)
                    r = normal @ diff  # 有符号距离
                    residuals.append(r)

                    # 雅可比矩阵
                    # dr/dx = n_x, dr/dy = n_y
                    # dr/dθ = n^T * [-sin(θ)*p_x - cos(θ)*p_y, cos(θ)*p_x - sin(θ)*p_y]
                    J = np.zeros(5)
                    J[0] = normal[0]  # dr/dx
                    J[1] = normal[1]  # dr/dy
                    J[4] = normal[0] * (-sin_theta * p_body[0] - cos_theta * p_body[1]) + \
                           normal[1] * (cos_theta * p_body[0] - sin_theta * p_body[1])  # dr/dθ
                    jacobians.append(J)

            if len(residuals) == 0:
                print("    警告: 没有有效的扫描点")
                return

            # 转换为数组
            r = np.array(residuals)
            H = np.array(jacobians)

            # ===== 计算Kalman增益 =====
            # 测量噪声
            R = np.eye(len(r)) * 0.1  # 点云测量噪声

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
            r_norm = np.linalg.norm(r)
            print(f"    迭代{iteration+1}: 残差||r||={r_norm:.4f}, 更新||dx||={dx_norm:.6f}")

            if dx_norm < self.tol:
                print(f"    ✓ 收敛!")
                break

        # ===== 更新协方差 (Joseph形式) =====
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


def test_ieskf():
    """
    单一测试案例: 2D激光雷达SLAM

    运动过程:
    1. 0-3秒: 匀速直线
    2. 3-6秒: 加速
    3. 6-10秒: 转弯
    """
    print("=" * 80)
    print("2D IESKF 测试: 激光雷达SLAM")
    print("=" * 80)

    # 参数
    dt = 0.01  # 100Hz
    T = 10.0   # 10秒
    N = int(T / dt)

    # ===== 生成地图 =====
    # 创建一个简单的室内环境 (墙壁)
    map_points = []

    # 外墙 (矩形房间)
    room_size = 30
    wall_density = 100
    for i in range(wall_density):
        # 四面墙
        x = -room_size/2 + i * room_size / wall_density
        map_points.append([x, -room_size/2])  # 下墙
        map_points.append([x, room_size/2])   # 上墙

        y = -room_size/2 + i * room_size / wall_density
        map_points.append([-room_size/2, y])  # 左墙
        map_points.append([room_size/2, y])   # 右墙

    map_points = np.array(map_points)

    # 初始化激光雷达
    lidar = Lidar2D(map_points, num_rays=180, max_range=20.0)

    # 初始化滤波器
    kf = IESKF_2D()
    # 从房间中心开始
    kf.x = 0.0
    kf.y = 0.0

    # 真实状态 (从房间中心开始)
    true_state = {'x': 0.0, 'y': 0.0, 'vx': 1.0, 'vy': 0.5, 'theta': 0.0}

    # 记录数据
    traj_est = []
    traj_true = []
    times = []

    # 测量间隔
    meas_interval = 10  # 10Hz

    print(f"\n参数: IMU={1/dt:.0f}Hz, LiDAR={1/(dt*meas_interval):.0f}Hz, 时长={T:.0f}s")
    print(f"地图点数: {len(map_points)}\n")

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
            # 生成激光雷达扫描 (从真实位置)
            scan_points = lidar.scan(true_state['x'], true_state['y'], true_state['theta'])

            # 添加测量噪声
            scan_points += np.random.randn(*scan_points.shape) * 0.02

            print(f"t={t:.2f}s: LiDAR扫描更新")
            print(f"  真实: ({true_state['x']:.2f}, {true_state['y']:.2f})")
            print(f"  预测: ({kf.x:.2f}, {kf.y:.2f})")
            print(f"  扫描点数: {len(scan_points)}")

            kf.update_with_lidar(scan_points, lidar)

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
    plot_results(times, traj_est, traj_true, errors, map_points)


def plot_results(times, traj_est, traj_true, errors, map_points):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 轨迹 + 地图
    axes[0].scatter(map_points[:, 0], map_points[:, 1], c='gray', s=1, alpha=0.3, label='Map')
    axes[0].plot(traj_true[:, 0], traj_true[:, 1], 'b-', lw=2, label='True')
    axes[0].plot(traj_est[:, 0], traj_est[:, 1], 'r--', lw=1.5, label='Estimated')
    axes[0].scatter(traj_true[0, 0], traj_true[0, 1], c='g', s=100, marker='o', label='Start')
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Y [m]')
    axes[0].set_title('Trajectory with Map')
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

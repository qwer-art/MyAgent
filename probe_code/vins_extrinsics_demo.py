"""
VINS外参与预积分 - 完整示例
单文件版本，严格参照非线性优化标准模式
- Problem类：严格的4个函数（residual, jacobian, cost, gradient）
- Optimizer类：Gauss-Newton优化器

环境: my_agent (conda)
依赖: numpy, scipy
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Pose:
    """相机位姿（旋转+平移）"""
    R: np.ndarray  # 3x3 旋转矩阵
    t: np.ndarray  # [3,] 平移向量

    def __post_init__(self):
        assert self.R.shape == (3, 3), "旋转矩阵必须是3x3"
        assert self.t.shape == (3,), "平移向量必须是3D"


# ============================================================================
# 优化问题定义（标准模式）
# ============================================================================

class MultiViewReprojectionErrorProblem:
    """
    多视图重投影误差优化问题

    目标: 最小化多视图重投影误差
      min Σ ||π(Tcw_i * Pj) - obs_ij||²

    参数 theta 的组成:
      - poses: 3个位姿，每个6维 [rx, ry, rz, tx, ty, tz]
      - points: 10个3D点，每个3维 [x, y, z]
      总维度: 3*6 + 10*3 = 48

    残差: 每个观测2维像素
      [u1 - obs1, v1 - obs1, u2 - obs2, v2 - obs2, ...]

    严格遵循4函数标准:
      1. residual(theta) -> np.ndarray
      2. jacobian(theta) -> np.ndarray
      3. cost(theta) -> float
      4. gradient(theta) -> np.ndarray
    """

    def __init__(self,
                 observations: List[Tuple[int, int, np.ndarray]],
                 n_poses: int,
                 n_points: int,
                 focal_length: float = 500.0,
                 cx: float = 320.0,
                 cy: float = 240.0) -> None:
        """
        输入:
          observations: [(pose_idx, point_idx, [u, v]), ...] 稀疏观测列表
          n_poses: 位姿数量
          n_points: 3D点数量
          focal_length, cx, cy: 相机内参
        输出: 无
        参数: 存储观测数据和相机内参
        """
        self.observations = observations  # [(pose_idx, point_idx, [u, v]), ...]
        self.n_poses = n_poses
        self.n_points = n_points
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.n_obs = len(observations)

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """
        投影函数：3D相机坐标 -> 2D像素坐标

        输入: p_cam = [X, Y, Z] (3D点在相机系中)
        输出: [u, v] (2D像素坐标)
        参数: 相机内参
        """
        if p_cam[2] < 1e-6:
            p_cam[2] = 1e-6

        u = self.focal_length * p_cam[0] / p_cam[2] + self.cx
        v = self.focal_length * p_cam[1] / p_cam[2] + self.cy

        return np.array([u, v])

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        计算残差向量

        theta = [pose1(6), pose2(6), pose3(6), point1(3), ..., point10(3)]

        输入: theta (48,)
        输出: residual (2*n_obs,) - 重投影误差
        参数: 无
        """
        residual = np.zeros(2 * self.n_obs)

        for idx, (pose_idx, point_idx, obs) in enumerate(self.observations):
            # 提取pose
            pose_start = pose_idx * 6
            pose_vec = theta[pose_start:pose_start + 6]
            R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
            t = pose_vec[3:]

            # 提取point
            point_start = self.n_poses * 6 + point_idx * 3
            point = theta[point_start:point_start + 3]

            # 变换到相机系
            p_cam = R @ point + t

            # 投影
            pred = self.project(p_cam)

            # 残差
            residual[2*idx] = pred[0] - obs[0]
            residual[2*idx + 1] = pred[1] - obs[1]

        return residual

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        计算Jacobian矩阵

        J = [
            [∂r1_u/∂θ],
            [∂r1_v/∂θ],
            ...
        ]  (2*n_obs, 48)

        使用数值微分计算Jacobian

        输入: theta (48,)
        输出: Jacobian矩阵 (2*n_obs, 48)
        参数: 无
        """
        # 数值微分计算Jacobian
        epsilon = 1e-6
        n_params = self.n_poses * 6 + self.n_points * 3

        # 计算当前残差
        residual_0 = self.residual(theta)

        # 计算Jacobian
        J = np.zeros((2 * self.n_obs, n_params))

        for i in range(n_params):
            theta_perturbed = theta.copy()
            theta_perturbed[i] += epsilon

            residual_perturbed = self.residual(theta_perturbed)

            J[:, i] = (residual_perturbed - residual_0) / epsilon

        return J

    def cost(self, theta: np.ndarray) -> float:
        """
        计算代价函数值

        J(theta) = 0.5 * sum(r²)

        输入: theta (48,)
        输出: 标量代价值
        参数: 无
        """
        r = self.residual(theta)
        return 0.5 * np.sum(r**2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        计算梯度: grad = J^T @ r

        输入: theta (48,)
        输出: 梯度向量 (48,)
        参数: 无
        """
        J = self.jacobian(theta)
        r = self.residual(theta)
        return J.T @ r


class ReprojectionErrorProblem:
    """
    重投影误差优化问题（标准模式）

    目标: 最小化重投影误差
      min Σ ||π(θ) - obs||²

    参数 theta 的组成:
      - 位姿: 6维 [rx, ry, rz, tx, ty, tz] (旋转3维+平移3维)
      - 3D点: 每个点3维 [x, y, z]
      总维度: 6 + 3*N

    残差: 每个观测2维像素，总共 2*N 维
      [u1_pred - u1_obs, v1_pred - v1_obs, u2_pred - u2_obs, v2_pred - v2_obs, ...]

    严格遵循4函数标准:
      1. residual(theta) -> np.ndarray
      2. jacobian(theta) -> np.ndarray
      3. cost(theta) -> float
      4. gradient(theta) -> np.ndarray
    """

    def __init__(self, observations: np.ndarray,
                 focal_length: float = 500.0,
                 cx: float = 320.0,
                 cy: float = 240.0) -> None:
        """
        输入: observations (N, 2) - N个观测的像素坐标 [u, v]
        输出: 无
        参数: 存储观测数据和相机内参
        """
        self.observations = observations
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.N = len(observations)  # 观测数量

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """
        投影函数：3D相机坐标 -> 2D像素坐标

        输入: p_cam = [X, Y, Z] (3D点在相机系中)
        输出: [u, v] (2D像素坐标)
        参数: 相机内参
        """
        if p_cam[2] < 1e-6:
            p_cam[2] = 1e-6

        u = self.focal_length * p_cam[0] / p_cam[2] + self.cx
        v = self.focal_length * p_cam[1] / p_cam[2] + self.cy

        return np.array([u, v])

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        计算残差向量

        theta = [rx, ry, rz, tx, ty, tz, x1, y1, z1, x2, y2, z2, ...]
              ^^^^^^^^^^^^----------- 位姿(6维)  ^^^^^^^^^^^^^--- 3D点(3*N维)

        输入: theta (6 + 3*N,)
        输出: residual (2*N,) - 重投影误差
        参数: 无
        """
        # 解析theta
        pose_vec = theta[:6]  # [rx, ry, rz, tx, ty, tz]
        points = theta[6:].reshape(self.N, 3)  # (N, 3)

        # 构造相机位姿
        R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
        t = pose_vec[3:]

        # 计算重投影误差
        residual = np.zeros(2 * self.N)

        for i in range(self.N):
            # 变换到相机系
            p_world = points[i]
            p_cam = R @ p_world + t

            # 投影
            pred = self.project(p_cam)

            # 残差
            obs = self.observations[i]
            residual[2*i] = pred[0] - obs[0]
            residual[2*i + 1] = pred[1] - obs[1]

        return residual

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        计算Jacobian矩阵

        J = [
            [∂r1_u/∂θ],
            [∂r1_v/∂θ],
            ...
        ]  (2*N, 6+3*N)

        使用数值微分计算Jacobian，保证通用性

        输入: theta (6 + 3*N,)
        输出: Jacobian矩阵 (2*N, 6+3*N)
        参数: 无
        """
        # 数值微分计算Jacobian
        epsilon = 1e-6

        # 计算当前残差
        residual_0 = self.residual(theta)

        # 计算Jacobian
        J = np.zeros((2 * self.N, 6 + 3 * self.N))

        for i in range(6 + 3 * self.N):
            theta_perturbed = theta.copy()
            theta_perturbed[i] += epsilon

            residual_perturbed = self.residual(theta_perturbed)

            J[:, i] = (residual_perturbed - residual_0) / epsilon

        return J

    def cost(self, theta: np.ndarray) -> float:
        """
        计算代价函数值

        J(theta) = 0.5 * sum(r²)

        输入: theta (6 + 3*N,)
        输出: 标量代价值
        参数: 无
        """
        r = self.residual(theta)
        return 0.5 * np.sum(r**2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        计算梯度: grad = J^T @ r

        输入: theta (6 + 3*N,)
        输出: 梯度向量 (6+3*N,)
        参数: 无
        """
        J = self.jacobian(theta)
        r = self.residual(theta)
        return J.T @ r


# ============================================================================
# Gauss-Newton优化器（标准模式）
# ============================================================================

class GaussNewtonOptimizer:
    """
    阻尼Gauss-Newton优化器（Levenberg-Marquardt风格）

    迭代公式:
      H * delta = -grad
      theta_new = theta_old + delta

    其中:
      H = J^T @ J + λ*I (阻尼Hessian，保证正定)
      grad = J^T @ r (梯度)
      λ = 阻尼因子 (damping)
    """

    def __init__(self,
                 max_iterations: int = 100,
                 gradient_threshold: float = 1e-6,
                 step_threshold: float = 1e-6,
                 cost_change_threshold: float = 1e-10,
                 damping: float = 1e-3) -> None:
        """
        配置终止条件

        输入: 无
        输出: 无
        参数:
          - max_iterations: 最大迭代次数
          - gradient_threshold: 梯度范数阈值
          - step_threshold: 步长范数阈值
          - cost_change_threshold: 代价变化阈值
          - damping: 阻尼因子（Levenberg-Marquardt风格）
        """
        self.max_iterations = max_iterations
        self.gradient_threshold = gradient_threshold
        self.step_threshold = step_threshold
        self.cost_change_threshold = cost_change_threshold
        self.damping = damping

    def solve(self, problem: Union['ReprojectionErrorProblem', 'MultiViewReprojectionErrorProblem'], theta_init: np.ndarray) -> np.ndarray:
        """
        执行优化

        输入: problem (优化问题), theta_init (初始参数)
        输出: theta_opt (最优参数)
        参数: 无
        """
        theta = theta_init.copy()
        cost_prev = problem.cost(theta)
        grad = problem.gradient(theta)

        print(f"\n初始参数维度: {len(theta)}")
        print(f"初始代价: {cost_prev:.6e}")
        print(f"初始梯度范数: {np.linalg.norm(grad):.6e}")
        print("-" * 60)

        for iter in range(self.max_iterations):
            # �骤1: 计算Jacobian和残差
            J = problem.jacobian(theta)
            r = problem.residual(theta)

            # 步骤2: 构造正规方程: H * delta = -grad
            # H = J^T @ J + λ*I (阻尼Gauss-Newton/Levenberg-Marquardt)
            # grad = J^T @ r (n_params,)
            n_params = J.shape[1]
            H = J.T @ J + self.damping * np.eye(n_params)
            grad = J.T @ r

            # 步骤3: 求解增量: delta = -H^(-1) @ grad
            try:
                delta = -np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                print("警告: Hessian矩阵奇异，无法求解!")
                break

            # 步骤4: 更新参数
            theta_new = theta + delta

            # 步骤5: 计算新的代价和梯度
            cost_new = problem.cost(theta_new)
            grad_new = problem.gradient(theta_new)

            # 打印迭代信息
            print(f"迭代 {iter+1:3d} | "
                  f"cost={cost_new:.6e} | "
                  f"||grad||={np.linalg.norm(grad_new):.6e} | "
                  f"||delta||={np.linalg.norm(delta):.6e}")

            # 步骤6: 检查终止条件
            converged = False
            stop_reason = None

            # 终止条件1: 梯度阈值
            if np.linalg.norm(grad_new) < self.gradient_threshold:
                converged = True
                stop_reason = "梯度阈值满足"

            # 终止条件2: 步长阈值
            elif np.linalg.norm(delta) < self.step_threshold:
                converged = True
                stop_reason = "步长阈值满足"

            # 终止条件3: 代价变化阈值
            elif abs(cost_new - cost_prev) < self.cost_change_threshold:
                converged = True
                stop_reason = "代价变化阈值满足"

            if converged:
                print("-" * 60)
                print(f"收敛! 原因: {stop_reason}")
                print(f"总迭代次数: {iter + 1}")
                theta = theta_new
                break

            # 步骤7: 更新状态
            theta = theta_new
            cost_prev = cost_new

            # 步骤8: 检查发散
            if not np.isfinite(cost_new):
                print("-" * 60)
                print("警告: 优化发散! cost = Inf/NaN")
                break

        return theta


# ============================================================================
# 测试函数
# ============================================================================

def test_quaternion_interpolation():
    """测试1: 四元数插值"""
    print("=" * 60)
    print("测试1: 四元数插值")
    print("=" * 60)
    print("待实现...")
    print("=" * 60)


def test_pose_feature_optimization():
    """测试2: 多Pose/多特征点优化（真实SLAM场景）"""
    print("\n" + "=" * 80)
    print("测试2: 多Pose/多特征点优化问题（真实SLAM场景）")
    print("=" * 80)

    # 相机内参
    focal_length = 500.0
    cx = 320.0
    cy = 240.0
    image_width = 640
    image_height = 480

    # 创建真实数据
    # 3个相机位姿：相机沿x轴前进，每次移动1m
    poses_true = []
    for i in range(3):
        angle = np.radians(i * 5)  # 每次绕z轴旋转5度
        R = Rotation.from_euler('z', angle).as_matrix()
        t = np.array([float(i), 0.0, 0.0])  # 沿x轴移动
        poses_true.append(Pose(R, t))

    # 10个3D特征点（随机分布在相机前方）
    np.random.seed(42)
    n_points = 10
    points_true = []
    for i in range(n_points):
        # x: 0~5m, y: -2~2m, z: 3~8m (相机前方)
        x = np.random.uniform(0, 5)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(3, 8)
        points_true.append(np.array([x, y, z]))

    # 创建临时problem用于投影
    temp_problem = ReprojectionErrorProblem(
        observations=np.zeros((1, 2)),
        focal_length=focal_length,
        cx=cx, cy=cy
    )

    # 构建观测数据（考虑可见性）
    # observations_list: [(pose_idx, point_idx, [u, v]), ...]
    observations_list = []

    print("\n>>> 构建观测数据（考虑视野和遮挡）:")
    for pose_idx, pose in enumerate(poses_true):
        print(f"  位姿{pose_idx+1}: t={pose.t}")
        for point_idx, point in enumerate(points_true):
            # 变换到相机系
            p_cam = pose.R @ point + pose.t

            # 检查深度（必须在相机前方）
            if p_cam[2] < 0.5:
                continue  # 太近或背后

            # 投影
            obs = temp_problem.project(p_cam)

            # 检查是否在图像范围内
            if not (0 <= obs[0] < image_width and 0 <= obs[1] < image_height):
                continue  # 超出视野

            # 随机遮挡（20%概率被遮挡）
            if np.random.random() < 0.2:
                continue

            observations_list.append((pose_idx, point_idx, obs))

    print(f"\n  总观测数: {len(observations_list)}")
    print(f"  理论最大: {3 * 10} (3个位姿 × 10个点)")
    print(f"  可见率: {len(observations_list) / 30 * 100:.1f}%")

    # 显示观测分布
    obs_count_per_pose = [0] * 3
    obs_count_per_point = [0] * n_points
    for pose_idx, point_idx, _ in observations_list:
        obs_count_per_pose[pose_idx] += 1
        obs_count_per_point[point_idx] += 1

    print(f"\n  每个位姿的观测数: {obs_count_per_pose}")
    print(f"  每个点的观测次数: {obs_count_per_point}")

    # 构造theta结构
    # theta = [pose1(6), pose2(6), pose3(6), point1(3), ..., point10(3)]
    #        ^^^^^^^^^^^^^^^^^^^ poses ^^^^^^^^^^^^^^^^^^^^^^^^^ points
    n_poses = 3
    theta_size = n_poses * 6 + n_points * 3  # 48维

    # 创建带噪声的初始估计
    theta_init = np.zeros(theta_size)

    for i, pose in enumerate(poses_true):
        # 位姿噪声
        angle_noise = np.random.uniform(-2, 2)  # ±2度
        R_noisy = Rotation.from_euler('z', np.radians(angle_noise)).as_matrix() @ pose.R
        t_noisy = pose.t + np.random.normal(0, 0.1, 3)

        rotvec = Rotation.from_matrix(R_noisy).as_rotvec()
        theta_init[i*6:(i+1)*6] = np.concatenate([rotvec, t_noisy])

    for i, point in enumerate(points_true):
        # 3D点噪声
        point_noisy = point + np.random.normal(0, 0.2, 3)
        theta_init[n_poses*6 + i*3:n_poses*6 + (i+1)*3] = point_noisy

    print(f"\n>>> 初始theta维度: {len(theta_init)}")
    print(f"  结构: [3个pose(18维), 10个point(30维)]")

    # 创建多视图优化问题
    problem = MultiViewReprojectionErrorProblem(
        observations=observations_list,
        n_poses=n_poses,
        n_points=n_points,
        focal_length=focal_length,
        cx=cx, cy=cy
    )

    # 创建优化器
    optimizer = GaussNewtonOptimizer(
        max_iterations=100,
        gradient_threshold=1e-6,
        step_threshold=1e-6,
        cost_change_threshold=1e-10,
        damping=1e-3
    )

    # 求解
    print(f"\n>>> 开始优化")
    theta_opt = optimizer.solve(problem, theta_init)

    # 解析优化结果
    print("\n>>> 优化结果:")
    for i in range(n_poses):
        rotvec = theta_opt[i*6:i*6+3]
        t = theta_opt[i*6+3:(i+1)*6]
        R = Rotation.from_rotvec(rotvec).as_matrix()
        print(f"  位姿{i+1}: t={t}")

    print(f"\n  3D点:")
    for i in range(n_points):
        point = theta_opt[n_poses*6 + i*3:n_poses*6 + (i+1)*3]
        print(f"    点{i+1}: {point}")

    # 计算最终误差
    cost_final = problem.cost(theta_opt)
    print(f"\n  最终代价: {cost_final:.6e}")

    # 与真实值对比
    print(f"\n>>> 与真实值对比:")
    for i in range(n_poses):
        t_true = poses_true[i].t
        t_opt = theta_opt[i*6+3:(i+1)*6]
        error_t = np.linalg.norm(t_opt - t_true)
        print(f"  位姿{i+1}平移误差: {error_t:.6f}")

    total_point_error = 0.0
    for i in range(n_points):
        p_true = points_true[i]
        p_opt = theta_opt[n_poses*6 + i*3:n_poses*6 + (i+1)*3]
        error = np.linalg.norm(p_opt - p_true)
        total_point_error += error
        print(f"  点{i+1}误差: {error:.6f}")

    print(f"\n  平均点误差: {total_point_error / n_points:.6f}")

    print("=" * 80)


# ============================================================================
# 其他测试函数（占位符）
# ============================================================================

def test_marginalization():
    """测试3: 边缘化"""
    print("\n" + "=" * 80)
    print("测试3: 边缘化问题")
    print("=" * 80)
    print("待实现...")
    print("=" * 80)


def test_hessian_analysis():
    """测试4: Hessian矩阵分析"""
    print("\n" + "=" * 80)
    print("测试4: Hessian矩阵分析")
    print("=" * 80)
    print("待实现...")
    print("=" * 80)


def run_all_tests():
    """运行所有测试"""
    test_quaternion_interpolation()
    test_pose_feature_optimization()
    test_marginalization()
    test_hessian_analysis()


if __name__ == "__main__":
    # 只运行测试2（Pose/特征点优化）
    test_pose_feature_optimization()

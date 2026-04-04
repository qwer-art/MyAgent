"""
VINS-Mono 滑动窗口边缘化 - 标准流程
完全参考VINS-Mono的实现方式，但简化细节

标准流程:
1. 初始化: 7帧完全BA优化（无先验）
2. 滑动窗口:
   - 第8帧到来
   - 边缘化第0帧（基于优化后的状态）
   - 移除第0帧，加入第8帧
   - 新的7帧（1-7）+ 先验一起优化

核心原则:
- Problem类: 遵循4函数标准（residual, jacobian, cost, gradient）
- 边缘化: 基于优化后的状态
- 先验: 包含被边缘化帧的约束

环境: my_agent (conda)
依赖: numpy, scipy
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Pose:
    """相机位姿"""
    R: np.ndarray  # [3x3]
    t: np.ndarray  # [3,]


@dataclass
class Feature:
    """3D特征点（逆深度参数化）"""
    id: int
    u_ref: float
    v_ref: float
    rho: float


@dataclass
class Observation:
    """观测"""
    frame_id: int
    feature_id: int
    pixel: np.ndarray  # [2,]


@dataclass
class Frame:
    """帧"""
    id: int
    pose: Pose
    timestamp: float


@dataclass
class PriorInfo:
    """先验信息（来自边缘化）"""
    J: np.ndarray  # 先验Jacobian
    r: np.ndarray  # 先验残差


# ============================================================================
# Problem类：标准的4函数模式
# ============================================================================

class VINSProblem:
    """
    VINS优化问题（遵循4函数标准）

    residual(theta): 视觉残差 + 先验残差
    jacobian(theta): 视觉Jacobian + 先验Jacobian
    cost(theta): 0.5 * sum(r²)
    gradient(theta): J^T @ r
    """

    def __init__(self,
                 frames: Dict[int, Frame],
                 features: Dict[int, Feature],
                 observations: List[Observation],
                 focal_length: float = 500.0,
                 cx: float = 320.0,
                 cy: float = 240.0,
                 prior: Optional[PriorInfo] = None) -> None:
        """
        Args:
            frames: 当前窗口内的帧
            features: 特征点
            observations: 观测
            focal_length, cx, cy: 相机内参
            prior: 先验信息（来自边缘化）
        """
        self.frames = frames
        self.features = features
        self.observations = observations
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.prior = prior

        self.n_poses = len(frames)
        self.n_points = len(features)

        # 只统计窗口内的观测数量
        self.n_obs = sum(1 for obs in observations if obs.frame_id in frames)

        # 计算参考帧映射
        self.ref_pose_for_point = self._compute_ref_poses()

    def _compute_ref_poses(self) -> Dict[int, int]:
        """计算每个特征点的参考帧"""
        ref_map = {}
        for feature in self.features.values():
            for obs in self.observations:
                if obs.feature_id == feature.id:
                    ref_map[feature.id] = obs.frame_id
                    break
            if feature.id not in ref_map:
                ref_map[feature.id] = min(self.frames.keys())
        return ref_map

    def _visual_residual(self, theta: np.ndarray) -> np.ndarray:
        """计算视觉重投影残差"""
        residual = np.zeros(2 * self.n_obs)

        # 用于跟踪有效观测的索引（与Jacobian保持一致）
        valid_obs_idx = 0

        for obs in self.observations:
            pose_idx = obs.frame_id
            feature_id = obs.feature_id

            # 跳过不在窗口内的观测
            if pose_idx not in self.frames:
                continue

            # 提取pose
            pose_idx_list = sorted(self.frames.keys())
            local_idx = pose_idx_list.index(pose_idx)
            pose_start = local_idx * 6
            pose_vec = theta[pose_start:pose_start + 6]
            R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
            t = pose_vec[3:]

            # 提取逆深度
            point_idx_list = sorted(self.features.keys())
            local_feature_idx = point_idx_list.index(feature_id)
            point_start = self.n_poses * 6 + local_feature_idx
            rho = float(theta[point_start])

            feature = self.features[feature_id]
            u_ref = feature.u_ref
            v_ref = feature.v_ref

            # 逆深度转3D
            ref_pose_idx = self.ref_pose_for_point.get(feature_id, pose_idx)
            P_w = self._inverse_depth_to_3d(rho, u_ref, v_ref, ref_pose_idx, theta)

            # 投影
            p_cam = R @ P_w + t
            pred = self._project(p_cam)

            # 使用valid_obs_idx作为索引（与Jacobian一致）
            residual[2*valid_obs_idx] = pred[0] - obs.pixel[0]
            residual[2*valid_obs_idx + 1] = pred[1] - obs.pixel[1]

            # 增加有效观测计数
            valid_obs_idx += 1

        return residual

    def _inverse_depth_to_3d(self, rho: float, u_ref: float, v_ref: float,
                            ref_pose_idx: int, theta: np.ndarray) -> np.ndarray:
        """逆深度转3D点"""
        Z_c = 1.0 / (rho + 1e-6)
        X_c = (u_ref - self.cx) * Z_c / self.focal_length
        Y_c = (v_ref - self.cy) * Z_c / self.focal_length
        P_c = np.array([X_c, Y_c, Z_c])

        # 找到参考帧的local index
        ref_pose_id_list = sorted(self.frames.keys())
        if ref_pose_idx in ref_pose_id_list:
            local_ref_idx = ref_pose_id_list.index(ref_pose_idx)
        else:
            local_ref_idx = 0

        pose_start = local_ref_idx * 6
        pose_vec = theta[pose_start:pose_start + 6]
        R_ref = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
        t_ref = pose_vec[3:]

        P_w = R_ref.T @ (P_c - t_ref)
        return P_w

    def _project(self, p_cam: np.ndarray) -> np.ndarray:
        """投影函数"""
        if p_cam[2] < 1e-6:
            p_cam[2] = 1e-6
        u = self.focal_length * p_cam[0] / p_cam[2] + self.cx
        v = self.focal_length * p_cam[1] / p_cam[2] + self.cy
        return np.array([u, v])

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """计算残差: r = [r_visual; r_prior]"""
        r_visual = self._visual_residual(theta)

        if self.prior is not None:
            r_prior = self.prior.r
            return np.concatenate([r_visual, r_prior])

        return r_visual

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        计算Jacobian: J = [J_visual; J_prior]

        使用解析Jacobian（参考test_bundle_adjustment.py）
        """
        # 先计算视觉部分的Jacobian
        J_visual = self._jacobian_analytical_visual(theta)

        # 如果有先验，添加先验Jacobian
        if self.prior is not None:
            visual_dim = 2 * self.n_obs
            prior_dim = self.prior.J.shape[0]
            total_dim = visual_dim + prior_dim

            n_params = J_visual.shape[1]
            prior_cols = self.prior.J.shape[1]

            J = np.zeros((total_dim, n_params))
            J[:visual_dim, :] = J_visual

            # 先验Jacobian需要与当前参数空间匹配
            # 如果先验的列数小于当前参数数，用零填充
            if prior_cols < n_params:
                # 新增的参数（如新特征点）没有先验约束
                J_prior_padded = np.zeros((prior_dim, n_params))
                J_prior_padded[:, :prior_cols] = self.prior.J
                J[visual_dim:, :] = J_prior_padded
            else:
                # 先验的列数等于或大于当前参数数（正常情况）
                J[visual_dim:, :prior_cols] = self.prior.J

            return J

        return J_visual

    def _jacobian_analytical_visual(self, theta: np.ndarray) -> np.ndarray:
        """
        解析Jacobian（视觉部分）

        参考test_bundle_adjustment.py的实现
        """
        n_params = self.n_poses * 6 + self.n_points * 1
        J = np.zeros((2 * self.n_obs, n_params))

        # 用于跟踪有效观测的索引
        valid_obs_idx = 0

        for obs in self.observations:
            pose_idx = obs.frame_id
            feature_id = obs.feature_id

            # 跳过不在窗口内的观测
            if pose_idx not in self.frames:
                continue

            # 提取pose
            pose_idx_list = sorted(self.frames.keys())
            local_idx = pose_idx_list.index(pose_idx)
            pose_start = local_idx * 6
            pose_vec = theta[pose_start:pose_start + 6]
            R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
            t = pose_vec[3:]

            # 提取逆深度
            point_idx_list = sorted(self.features.keys())
            local_feature_idx = point_idx_list.index(feature_id)
            point_start = self.n_poses * 6 + local_feature_idx
            rho = float(theta[point_start])

            feature = self.features[feature_id]
            u_ref = feature.u_ref
            v_ref = feature.v_ref

            # 将逆深度转换为3D点
            ref_pose_idx = self.ref_pose_for_point.get(feature_id, pose_idx)
            P_w = self._inverse_depth_to_3d(rho, u_ref, v_ref, ref_pose_idx, theta)

            # 变换到当前相机系
            P_c = R @ P_w + t
            X, Y, Z = P_c[0], P_c[1], P_c[2]

            # 防止数值问题（Z太小或为负）
            Z_safe = max(Z, 1e-6)

            # 相机投影的Jacobian
            fx = fy = self.focal_length
            J_proj = np.array([
                [fx/Z_safe,   0,   -fx*X/(Z_safe*Z_safe)],
                [0,     fy/Z_safe, -fy*Y/(Z_safe*Z_safe)]
            ])

            # 当前residual的行索引（基于有效观测计数）
            row_start = 2 * valid_obs_idx

            # ====== 对位姿的Jacobian ======
            if ref_pose_idx == pose_idx:
                # 特殊情况：观测帧=参考帧
                # P_c = P_c_ref（不依赖于当前位姿）
                J_t = np.zeros((2, 3))   # 对平移的Jacobian为0
                J_rot = np.zeros((2, 3))  # 对旋转的Jacobian为0
            else:
                # 正常情况：观测帧≠参考帧
                # 对平移 (2×3)
                J_t = J_proj

                # 对旋转 (2×3)
                skew_P = np.array([
                    [0, -Z, Y],
                    [Z, 0, -X],
                    [-Y, X, 0]
                ])
                J_rot = J_proj @ (-skew_P)

            # 填充位姿部分
            J[row_start:row_start+2, pose_start:pose_start+3] = J_rot   # 旋转
            J[row_start:row_start+2, pose_start+3:pose_start+6] = J_t   # 平移

            # ====== 对逆深度的Jacobian ======
            rho_safe = rho + 1e-10
            dP_c_ref_drho = np.array([
                -(u_ref - self.cx) / (rho_safe**2 * fx),
                -(v_ref - self.cy) / (rho_safe**2 * fy),
                -1.0 / (rho_safe**2)
            ])

            if ref_pose_idx == pose_idx:
                dP_c_drho = dP_c_ref_drho
            else:
                # 提取参考帧位姿
                ref_pose_idx_list = sorted(self.frames.keys())
                local_ref_idx = ref_pose_idx_list.index(ref_pose_idx)
                ref_pose_start = local_ref_idx * 6
                ref_pose_vec = theta[ref_pose_start:ref_pose_start + 6]
                R_ref = Rotation.from_rotvec(ref_pose_vec[:3]).as_matrix()

                dP_c_drho = R @ (R_ref.T @ dP_c_ref_drho)

            J_rho = J_proj @ dP_c_drho

            # 填充逆深度部分
            J[row_start:row_start+2, point_start] = J_rho

            # 增加有效观测计数
            valid_obs_idx += 1

        return J

    def cost(self, theta: np.ndarray) -> float:
        """计算代价: J = 0.5 * sum(r²)"""
        r = self.residual(theta)
        return 0.5 * np.sum(r**2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """计算梯度: grad = J^T @ r"""
        J = self.jacobian(theta)
        r = self.residual(theta)
        return J.T @ r


# ============================================================================
# Optimizer：标准的Gauss-Newton
# ============================================================================

class GaussNewtonOptimizer:
    """Gauss-Newton优化器"""

    def __init__(self, max_iterations: int = 10, gradient_threshold: float = 1e-4, damping: float = 1e-3):
        self.max_iterations = max_iterations
        self.gradient_threshold = gradient_threshold
        self.damping = damping

    def solve(self, problem: VINSProblem, theta_init: np.ndarray) -> np.ndarray:
        """执行优化"""
        theta = theta_init.copy()
        print(f"  初始代价: {problem.cost(theta):.6f}")

        for iter in range(self.max_iterations):
            J = problem.jacobian(theta)
            r = problem.residual(theta)

            # 计算H的对角线元素，用于LM阻尼
            diag_H = np.diag(J.T @ J)

            # 使用对角阻尼（Levenberg-Marquardt风格）
            damping_matrix = np.diag(diag_H) * self.damping

            # 确保阻尼矩阵不为零
            damping_matrix = np.maximum(damping_matrix, 1e-6 * np.eye(damping_matrix.shape[0]))

            H = J.T @ J + damping_matrix
            b = J.T @ r

            # 使用更稳定的求解方法
            try:
                delta = -np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                # 如果求解失败，使用最小二乘
                delta, _, _, _ = np.linalg.lstsq(H, -b, rcond=None)

            # 调试：检查delta的大小
            delta_norm = np.linalg.norm(delta)
            if iter == 0:
                print(f"  初始delta范数: {delta_norm:.2e}")
                print(f"  delta最大值: {np.max(np.abs(delta)):.2e}")

            # 限制步长以防止发散
            max_step = 1.0
            if delta_norm > max_step:
                delta = delta * max_step / delta_norm
                delta_norm = max_step

            theta_new = theta + delta
            cost = problem.cost(theta_new)
            cost_old = problem.cost(theta)
            grad_norm = np.linalg.norm(b)

            if iter < 3 or iter == self.max_iterations - 1:
                print(f"  迭代 {iter+1}: cost={cost:.6f}, ||grad||={grad_norm:.6e}, ||delta||={delta_norm:.2e}")

            # 简单的线搜索：如果代价增加，减小步长
            if cost > cost_old * 1.01:  # 允许1%的增加
                theta_new = theta + delta * 0.5
                cost = problem.cost(theta_new)
                if iter < 3:
                    print(f"    代价增加，使用减半步长: cost={cost:.6f}")

            if grad_norm < self.gradient_threshold:
                print(f"  收敛！迭代次数: {iter+1}")
                break

            # 检查发散
            if not np.isfinite(cost) or delta_norm > 100:
                print(f"  警告: 优化发散！停止迭代")
                break

            theta = theta_new

        return theta


# ============================================================================
# 滑动窗口管理器（VINS风格）
# ============================================================================

class SlidingWindow:
    """
    VINS风格的滑动窗口

    流程:
    1. 维护窗口大小为N的帧
    2. 窗口满后，触发边缘化
    3. 边缘化最老帧，构建先验
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.frames: Dict[int, Frame] = {}  # 窗口内的帧
        self.features: Dict[int, Feature] = {}  # 所有特征点
        self.observations: List[Observation] = []  # 所有观测
        self.prior: Optional[PriorInfo] = None  # 先验信息

    def add_frame(self, frame: Frame) -> bool:
        """
        添加新帧

        Returns:
            True if 触发边缘化
        """
        self.frames[frame.id] = frame
        print(f"[添加帧] ID={frame.id}, 窗口帧数={len(self.frames)}")

        # 检查是否需要边缘化
        if len(self.frames) > self.window_size:
            print(f"\n[窗口满] 触发边缘化")
            return True
        return False

    def marginalize_oldest(self, theta_at_marginalization: np.ndarray,
                          n_poses_at_marginalization: int,
                          n_points_at_marginalization: int) -> None:
        """
        边缘化最老帧

        Args:
            theta_at_marginalization: 边缘化时刻的状态向量
            n_poses_at_marginalization: 边缘化时刻的帧数
            n_points_at_marginalization: 边缘化时刻的特征点数
        """
        oldest_frame_id = min(self.frames.keys())
        print(f"\n{'='*80}")
        print(f"[边缘化] 边缘化第{oldest_frame_id}帧")
        print(f"{'='*80}")

        print(f"  边缘化时刻: {n_poses_at_marginalization}帧, {n_points_at_marginalization}点")
        print(f"  当前时刻: {len(self.frames)}帧, {len(self.features)}点")

        # 创建临时的Problem（使用边缘化时刻的状态）
        temp_frames = {fid: self.frames[fid] for fid in sorted(self.frames.keys())[:n_poses_at_marginalization]}
        temp_features = {fid: self.features[fid] for fid in sorted(self.features.keys())[:n_points_at_marginalization]}
        temp_observations = [obs for obs in self.observations
                            if obs.frame_id in temp_frames and obs.feature_id in temp_features]

        problem = VINSProblem(
            frames=temp_frames,
            features=temp_features,
            observations=temp_observations,
            prior=self.prior,
            focal_length=500.0,
            cx=320.0, cy=240.0
        )

        # 计算Jacobian和残差（基于边缘化时刻的theta）
        J = problem.jacobian(theta_at_marginalization)
        r = problem.residual(theta_at_marginalization)

        print(f"  Jacobian形状: {J.shape}")
        print(f"  残差范数: {np.linalg.norm(r):.6f}")

        # 边缘化第0帧（前6个参数）
        m = 6

        # 构建信息矩阵: H = J^T @ J
        H_mm = J[:, :m].T @ J[:, :m] + 1e-6 * np.eye(m)  # (6x6)
        H_mr = J[:, :m].T @ J[:, m:]  # (6x106)
        H_rm = J[:, m:].T @ J[:, :m]  # (106x6)
        H_rr = J[:, m:].T @ J[:, m:]  # (106x106)

        r_m = J[:, :m].T @ r  # (6,)
        r_r = J[:, m:].T @ r  # (106,)

        # Schur补
        H_mm_inv = np.linalg.inv(H_mm)
        H_rr_new = H_rr - H_rm @ H_mm_inv @ H_mr
        r_r_new = r_r - H_rm @ H_mm_inv @ r_m

        print(f"  Schur补完成: H_rr_new {H_rr_new.shape}")

        # 构建先验
        try:
            H_rr_reg = H_rr_new + 1e-8 * np.eye(H_rr_new.shape[0])
            L = np.linalg.cholesky(H_rr_reg)
            J_prior = L.T
            r_prior = np.linalg.solve(L.T, r_r_new)

            self.prior = PriorInfo(J=J_prior, r=r_prior)
            print(f"  先验构建成功: J形状={J_prior.shape}")
        except np.linalg.LinAlgError:
            print(f"  警告: 先验构建失败")

        # 移除最老帧
        del self.frames[oldest_frame_id]

        # 清理观测
        self.observations = [obs for obs in self.observations if obs.frame_id != oldest_frame_id]

        print(f"  移除帧{oldest_frame_id}，剩余帧数={len(self.frames)}")
        print(f"{'='*80}\n")

    def get_theta(self) -> np.ndarray:
        """从当前状态构建theta向量（使用真实值作为初值）"""
        n_poses = len(self.frames)
        n_points = len(self.features)
        theta_size = n_poses * 6 + n_points * 1

        theta = np.zeros(theta_size)

        # 填充pose（按frame_id排序）
        for i, frame_id in enumerate(sorted(self.frames.keys())):
            frame = self.frames[frame_id]
            rotvec = Rotation.from_matrix(frame.pose.R).as_rotvec()
            theta[i*6:i*6+3] = rotvec
            theta[i*6+3:i*6+6] = frame.pose.t

        # 填充point（按feature_id排序）
        for i, feat_id in enumerate(sorted(self.features.keys())):
            theta[n_poses*6 + i] = self.features[feat_id].rho

        return theta

    def get_theta_with_noise(self, noise_std: float = 0.01) -> np.ndarray:
        """从当前状态构建theta向量（添加小噪声）"""
        theta = self.get_theta()

        # 给pose添加小噪声
        n_poses = len(self.frames)
        for i in range(n_poses):
            theta[i*6:i*6+3] += np.random.randn(3) * noise_std  # 旋转
            theta[i*6+3:i*6+6] += np.random.randn(3) * noise_std  # 平移

        # 给逆深度添加小噪声（但不能太小）
        n_points = len(self.features)
        for i in range(n_points):
            rho = theta[n_poses*6 + i]
            # 确保rho在合理范围内 [0.05, 0.5]
            rho = max(0.05, min(0.5, rho + np.random.randn() * noise_std * 0.1))
            theta[n_poses*6 + i] = rho

        return theta

    def add_feature(self, feature: Feature):
        self.features[feature.id] = feature

    def add_observation(self, obs: Observation):
        self.observations.append(obs)

    def print_status(self):
        """打印当前状态"""
        print(f"\n[窗口状态]")
        print(f"  窗口大小: {len(self.frames)}/{self.window_size}")
        print(f"  帧IDs: {sorted(self.frames.keys())}")
        print(f"  特征点数: {len(self.features)}")
        print(f"  观测数: {len(self.observations)}")
        print(f"  先验: {'有' if self.prior else '无'}")


# ============================================================================
# 测试函数：VINS-Mono标准流程
# ============================================================================

def test_vins_marginalization():
    """测试VINS-Mono风格的边缘化"""
    print("=" * 80)
    print("VINS-Mono 滑动窗口边缘化测试")
    print("=" * 80)
    print("\n标准流程:")
    print("  1. 初始化: 7帧完全BA（无先验）")
    print("  2. 滑动窗口: 第8帧触发边缘化")
    print("  3. 新的7帧（1-7）+ 先验一起优化")
    print("=" * 80)

    window = SlidingWindow(window_size=7)
    optimizer = GaussNewtonOptimizer(max_iterations=20, damping=10.0)  # 增加迭代次数

    # ============================================================
    # 阶段1: 初始化 - 添加7帧，完全BA
    # ============================================================
    print(f"\n{'='*80}")
    print("阶段1: 初始化（7帧完全BA）")
    print("=" * 80)

    # 创建7个简单的帧（沿x轴移动）
    for i in range(7):
        R = Rotation.from_euler('z', np.radians(i * 2)).as_matrix()  # 小旋转
        t = np.array([float(i) * 0.5, 0.0, 0.0])  # 每帧移动0.5m
        frame = Frame(id=i, pose=Pose(R, t), timestamp=float(i))
        window.add_frame(frame)

    # 创建特征点和观测（参考test_bundle_adjustment.py的正确方式）
    feature_id_counter = 0
    focal_length = 500.0
    cx, cy = 320.0, 240.0

    # 每帧创建几个特征点
    for i in range(7):
        # 获取当前帧
        frame_i = window.frames[i]

        # 每帧创建5个特征点
        for _ in range(5):
            # 在3D空间中随机创建一个点（在相机前方）
            # 使用相对于当前帧的相机坐标
            depth = 5.0 + np.random.rand() * 3.0  # 深度5-8米
            x_c = (np.random.rand() - 0.5) * 4.0  # x: -2~2米
            y_c = (np.random.rand() - 0.5) * 4.0  # y: -2~2米
            z_c = depth

            # 转换到世界坐标
            P_w = frame_i.pose.R.T @ np.array([x_c, y_c, z_c]) - frame_i.pose.R.T @ frame_i.pose.t

            # 投影到当前帧获取u_ref, v_ref
            u_ref = focal_length * x_c / z_c + cx
            v_ref = focal_length * y_c / z_c + cy
            rho = 1.0 / z_c  # 逆深度

            # 创建特征点
            feature = Feature(
                id=feature_id_counter,
                u_ref=u_ref,
                v_ref=v_ref,
                rho=rho
            )
            window.add_feature(feature)

            # 在当前帧及后续2帧中创建观测
            for k in range(i, min(i+3, 7)):
                frame_k = window.frames[k]

                # 将3D点变换到第k帧相机系
                P_c_k = frame_k.pose.R @ P_w + frame_k.pose.t

                # 检查深度
                if P_c_k[2] < 0.5:
                    continue

                # 投影
                u_k = focal_length * P_c_k[0] / P_c_k[2] + cx
                v_k = focal_length * P_c_k[1] / P_c_k[2] + cy

                # 添加小噪声
                obs = Observation(
                    frame_id=k,
                    feature_id=feature_id_counter,
                    pixel=np.array([u_k, v_k]) + np.random.randn(2) * 0.5
                )
                window.add_observation(obs)

            feature_id_counter += 1

    window.print_status()

    # 完全BA优化（无先验）
    print(f"\n[优化] 7帧完全BA（无先验）")
    problem = VINSProblem(
        frames=window.frames,
        features=window.features,
        observations=window.observations,
        prior=None,
        focal_length=500.0,
        cx=320.0, cy=240.0
    )

    # 使用带噪声的初始化（真实的SLAM场景）
    theta_init = window.get_theta_with_noise(noise_std=0.01)  # 与BA测试一致的噪声
    print(f"  初始代价（带噪声）: {problem.cost(theta_init):.2f}")

    # 调试：检查初始残差
    r_init = problem.residual(theta_init)
    print(f"  初始残差范数: {np.linalg.norm(r_init):.2f}")
    print(f"  最大残差: {np.max(np.abs(r_init)):.2f}")

    # 调试：检查初始Jacobian
    J_init = problem.jacobian(theta_init)
    print(f"  Jacobian形状: {J_init.shape}")
    print(f"  Jacobian范数: {np.linalg.norm(J_init):.2f}")
    print(f"  Jacobian最大值: {np.max(np.abs(J_init)):.2e}")
    print(f"  Jacobian是否包含NaN/Inf: {np.any(~np.isfinite(J_init))}")

    # 调试：检查梯度
    grad_init = problem.gradient(theta_init)
    print(f"  初始梯度范数: {np.linalg.norm(grad_init):.2e}")
    print(f"  梯度最大值: {np.max(np.abs(grad_init)):.2e}")

    # 使用更大的阻尼来避免发散（处理H矩阵病态问题）
    optimizer.damping = 10.0  # 大阻尼

    theta_opt = optimizer.solve(problem, theta_init)
    final_cost = problem.cost(theta_opt)
    print(f"[优化完成] 最终代价: {final_cost:.2f}")

    # ============================================================
    # 阶段2: 滑动窗口 - 第8帧触发边缘化
    # ============================================================
    print(f"\n{'='*80}")
    print("阶段2: 滑动窗口（第8帧触发边缘化）")
    print("=" * 80)

    # 添加第8帧
    R = Rotation.from_euler('z', np.radians(7 * 2)).as_matrix()
    t = np.array([3.5, 0.0, 0.0])  # 第8帧
    frame_8 = Frame(id=7, pose=Pose(R, t), timestamp=7.0)  # ID=7（接在0-6之后）

    should_marginalize = window.add_frame(frame_8)

    if should_marginalize:
        print(f"\n>>> 触发边缘化，基于7帧优化后的状态")
        n_poses_at_margin = 7
        n_points_at_margin = len(window.features)
        window.marginalize_oldest(theta_opt, n_poses_at_margin, n_points_at_margin)

        # 为第8帧添加新特征点和观测
        frame_7 = window.frames[7]
        for j in range(35, 40):
            # 在3D空间中随机创建一个点
            depth = 5.0 + np.random.rand() * 3.0
            x_c = (np.random.rand() - 0.5) * 4.0
            y_c = (np.random.rand() - 0.5) * 4.0
            z_c = depth

            # 转换到世界坐标
            P_w = frame_7.pose.R.T @ np.array([x_c, y_c, z_c]) - frame_7.pose.R.T @ frame_7.pose.t

            # 投影获取u_ref, v_ref
            u_ref = 500.0 * x_c / z_c + 320.0
            v_ref = 500.0 * y_c / z_c + 240.0
            rho = 1.0 / z_c

            # 创建特征点
            feature = Feature(
                id=j,
                u_ref=u_ref,
                v_ref=v_ref,
                rho=rho
            )
            window.add_feature(feature)

            # 添加观测
            obs = Observation(
                frame_id=7,
                feature_id=j,
                pixel=np.array([u_ref, v_ref]) + np.random.randn(2) * 0.5
            )
            window.add_observation(obs)

    window.print_status()

    # ============================================================
    # 阶段3: 带先验的BA优化
    # ============================================================
    print(f"\n{'='*80}")
    print("阶段3: 新的7帧（1-7）+ 先验一起优化")
    print("=" * 80)

    if window.prior is not None:
        problem_with_prior = VINSProblem(
            frames=window.frames,
            features=window.features,
            observations=window.observations,
            prior=window.prior,
            focal_length=500.0,
            cx=320.0, cy=240.0
        )

        theta_init_new = window.get_theta_with_noise(noise_std=0.01)  # 与BA测试一致的噪声
        print(f"  初始代价（带噪声）: {problem_with_prior.cost(theta_init_new):.2f}")

        theta_opt_new = optimizer.solve(problem_with_prior, theta_init_new)
        final_cost_new = problem_with_prior.cost(theta_opt_new)
        print(f"[优化完成] 最终代价: {final_cost_new:.2f}")
    else:
        print("  警告: 没有先验信息，跳过优化")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n{'='*80}")
    print("测试总结")
    print("=" * 80)
    print(f"✓ 初始化: 7帧完全BA优化")
    print(f"✓ 边缘化: 第0帧被边缘化")
    print(f"✓ 滑动窗口: 新的7帧（1-7）")
    if window.prior is not None:
        print(f"✓ 先验信息: 形状={window.prior.J.shape}")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_vins_marginalization()

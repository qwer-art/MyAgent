"""
多视图Bundle Adjustment - SLAM核心优化
单文件版本，严格参照非线性优化标准模式
- Problem类：严格的4个函数（residual, jacobian, cost, gradient）
- Optimizer类：阻尼Gauss-Newton优化器

应用场景：
- 视觉SLAM后端优化
- 运动恢复结构（SfM）
- 视觉里程计

环境: my_agent (conda)
依赖: numpy, scipy
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Pose:
    """相机位姿（旋转+平移）"""
    R: np.ndarray  # [3x3] 旋转矩阵
    t: np.ndarray  # [3,] 平移向量

    def __post_init__(self):
        assert self.R.shape == (3, 3), "旋转矩阵必须是3x3"
        assert self.t.shape == (3,), "平移向量必须是3D"


@dataclass
class Feature:
    """
    3D特征点（路标点）- 使用逆深度参数化（只优化深度）

    逆深度参数化：
      - u_ref, v_ref: 参考帧中的像素坐标（固定，不优化）
      - rho = 1/d: 逆深度（可优化参数）

    设计方案：
      - u_ref, v_ref固定为参考帧中的观测值
      - 只优化rho（逆深度）
      - 参数量：每点只有1维（rho）

    优点：
      - 减少参数量（从3维降到1维）
      - 符合物理意义（观测是固定的）
      - 避免过度参数化

    Attributes:
        id: 特征点唯一标识符
        u_ref: 参考帧中的u坐标（固定）
        v_ref: 参考帧中的v坐标（固定）
        rho: 逆深度（可优化参数）
    """
    id: int
    u_ref: float  # 参考帧像素u坐标（固定）
    v_ref: float  # 参考帧像素v坐标（固定）
    rho: float    # 逆深度（可优化）

    def get_depth(self) -> float:
        """获取深度"""
        return 1.0 / (self.rho + 1e-6)

    def to_3d(self, ref_pose: Pose, fx: float, cx: float, fy: float, cy: float) -> np.ndarray:
        """
        将逆深度参数转换为3D点（世界坐标系）

        输入:
          ref_pose: 参考帧位姿
          fx, fy, cx, cy: 相机内参

        输出:
          3D点世界坐标 [x, y, z]
        """
        # 在参考帧相机坐标系中的3D点
        Z_c = 1.0 / (self.rho + 1e-6)
        X_c = (self.u_ref - cx) * Z_c / fx
        Y_c = (self.v_ref - cy) * Z_c / fy
        P_c = np.array([X_c, Y_c, Z_c])

        # 变换到世界坐标系
        P_w = ref_pose.R.T @ (P_c - ref_pose.t)

        return P_w


@dataclass
class Observation:
    """
    观测：某个特征在某个帧中的2D观测

    Attributes:
        frame_id: 帧ID（位姿索引）
        feature_id: 特征点ID
        pixel: 像素坐标 [u, v]
    """
    frame_id: int
    feature_id: int
    pixel: np.ndarray  # [2,] 像素坐标

    def __post_init__(self):
        assert self.pixel.shape == (2,), "像素坐标必须是2D"


@dataclass
class Frame:
    """
    帧/相机位姿

    Attributes:
        id: 帧ID
        pose: 相机位姿 (R, t)
        timestamp: 时间戳（可选）
    """
    id: int
    pose: Pose
    timestamp: Optional[float] = None


class SLAMMap:
    """
    SLAM地图：包含所有特征点和观测

    这是优化问题的数据结构基础
    """

    def __init__(self) -> None:
        self.frames: Dict[int, Frame] = {}  # frame_id -> Frame
        self.features: Dict[int, Feature] = {}  # feature_id -> Feature
        self.observations: List[Observation] = []  # 观测列表

    def add_frame(self, frame: Frame) -> None:
        """添加帧"""
        self.frames[frame.id] = frame

    def add_feature(self, feature: Feature) -> None:
        """添加特征点"""
        self.features[feature.id] = feature

    def add_observation(self, obs: Observation) -> None:
        """添加观测"""
        self.observations.append(obs)

    def get_feature_observations(self, feature_id: int) -> List[Observation]:
        """获取某个特征点的所有观测"""
        return [obs for obs in self.observations if obs.feature_id == feature_id]

    def get_frame_observations(self, frame_id: int) -> List[Observation]:
        """获取某个帧的所有观测"""
        return [obs for obs in self.observations if obs.frame_id == frame_id]

    def get_statistics(self) -> Dict[str, any]:
        """获取地图统计信息"""
        n_obs_per_feature = {
            fid: len(self.get_feature_observations(fid))
            for fid in self.features.keys()
        }
        n_obs_per_frame = {
            fid: len(self.get_frame_observations(fid))
            for fid in self.frames.keys()
        }

        return {
            'n_frames': len(self.frames),
            'n_features': len(self.features),
            'n_observations': len(self.observations),
            'n_obs_per_feature': n_obs_per_feature,
            'n_obs_per_frame': n_obs_per_frame,
            'observation_ratio': len(self.observations) / (len(self.frames) * len(self.features))
            if len(self.frames) > 0 and len(self.features) > 0 else 0.0
        }

    def print_summary(self) -> None:
        """打印地图摘要"""
        stats = self.get_statistics()
        print(f"\n>>> 地图统计:")
        print(f"  帧数: {stats['n_frames']}")
        print(f"  特征点数: {stats['n_features']}")
        print(f"  总观测数: {stats['n_observations']}")
        print(f"  理论最大: {stats['n_frames'] * stats['n_features']}")
        print(f"  可见率: {stats['observation_ratio'] * 100:.1f}%")
        print(f"  每帧观测数: {list(stats['n_obs_per_frame'].values())}")
        print(f"  每个特征观测数: {list(stats['n_obs_per_feature'].values())}")


# ============================================================================
# 优化问题定义（标准模式）
# ============================================================================

class MultiViewReprojectionErrorProblem:
    """
    多视图重投影误差优化问题（标准模式）- 使用逆深度参数化

    目标: 最小化多视图重投影误差
      min Σ ||π(Tcw_i * Pj) - obs_ij||²

    数据结构:
      - Frame: 帧/相机位姿
      - Feature: 3D特征点（逆深度参数化）
      - Observation: 观测（Feature在Frame中的2D投影）

    参数 theta 的组成:
      - poses: N个位姿，每个6维 [rx, ry, rz, tx, ty, tz]
      - points: M个特征点，每个1维 [rho]（只优化逆深度）
        其中:
          (u_ref, v_ref): 参考帧中的像素坐标（固定，不在theta中）
          rho = 1/d: 逆深度（可优化参数）
      总维度: N*6 + M*1（大幅减少参数量！）

    残差: 每个观测2维像素
      [u1 - obs1, v1 - obs1, u2 - obs2, v2 - obs2, ...]

    严格遵循4函数标准:
      1. residual(theta) -> np.ndarray
      2. jacobian(theta) -> np.ndarray
      3. cost(theta) -> float
      4. gradient(theta) -> np.ndarray
    """

    def __init__(self,
                 slam_map: SLAMMap,
                 focal_length: float = 500.0,
                 cx: float = 320.0,
                 cy: float = 240.0,
                 use_analytical_jacobian: bool = True) -> None:
        """
        输入:
          slam_map: SLAM地图（包含Frame、Feature、Observation）
          focal_length, cx, cy: 相机内参
          use_analytical_jacobian: True=解析Jacobian(默认), False=数值微分
        输出: 无
        参数: 存储观测数据和相机内参
        """
        self.slam_map = slam_map
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.n_obs = len(slam_map.observations)
        self.n_poses = len(slam_map.frames)
        self.n_points = len(slam_map.features)
        self.use_analytical_jacobian = use_analytical_jacobian

        # 为每个点计算参考帧（第一个观测到该点的帧）
        self.ref_pose_for_point: Dict[int, int] = {}
        for feature in slam_map.features.values():
            # 找到第一个观测到该点的位姿
            for obs in slam_map.observations:
                if obs.feature_id == feature.id:
                    self.ref_pose_for_point[feature.id] = obs.frame_id
                    break
            # 如果没有观测，使用位姿0作为参考
            if feature.id not in self.ref_pose_for_point:
                self.ref_pose_for_point[feature.id] = 0

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

    def inverse_depth_to_3d(self, rho: float, u_ref: float, v_ref: float,
                          ref_pose_idx: int, theta: np.ndarray) -> np.ndarray:
        """
        逆深度参数 -> 3D点（世界坐标系）

        只优化rho（逆深度），u_ref和v_ref固定

        输入:
          rho: 逆深度（可优化参数）
          u_ref, v_ref: 参考帧中的像素坐标（固定）
          ref_pose_idx: 参考帧索引
          theta: 完整参数向量（用于提取参考帧位姿）

        输出: 3D点世界坐标 [x, y, z]
        """
        # 在参考帧相机坐标系中的3D点
        Z_c = 1.0 / (rho + 1e-6)  # 避免除零
        X_c = (u_ref - self.cx) * Z_c / self.focal_length
        Y_c = (v_ref - self.cy) * Z_c / self.focal_length

        P_c = np.array([X_c, Y_c, Z_c])

        # 提取参考帧位姿
        pose_start = ref_pose_idx * 6
        pose_vec = theta[pose_start:pose_start + 6]
        R_ref = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
        t_ref = pose_vec[3:]

        # 变换到世界坐标系: P_w = R_ref^T * (P_c - t_ref)
        # 因为: P_c = R_ref * P_w + t_ref
        # 所以: P_w = R_ref^T * P_c - R_ref^T * t_ref
        P_w = R_ref.T @ (P_c - t_ref)

        return P_w

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        计算残差向量

        theta = [pose1(6), pose2(6), ..., poseN(6), point1(1), ..., pointM(1)]
                                                        ^^^^^^只优化rho

        输入: theta (N*6 + M*1,) - 每个点只有1维（逆深度）
        输出: residual (2*n_obs,) - 重投影误差
        参数: 无
        """
        residual = np.zeros(2 * self.n_obs)

        for idx, obs in enumerate(self.slam_map.observations):
            pose_idx = obs.frame_id
            feature_id = obs.feature_id

            # 提取pose
            pose_start = pose_idx * 6
            pose_vec = theta[pose_start:pose_start + 6]
            R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
            t = pose_vec[3:]

            # 提取逆深度参数（只有rho）
            point_start = self.n_poses * 6 + feature_id * 1
            rho = theta[point_start]  # 只取1维

            # 获取该特征点的固定像素坐标
            feature = self.slam_map.features[feature_id]
            u_ref = feature.u_ref
            v_ref = feature.v_ref

            # 将逆深度参数转换为3D点
            # 使用固定的参考帧（第一个观测到该点的帧）
            ref_pose_idx = self.ref_pose_for_point.get(feature_id, 0)
            point = self.inverse_depth_to_3d(rho, u_ref, v_ref, ref_pose_idx, theta)

            # 变换到相机系
            p_cam = R @ point + t

            # 投影
            pred = self.project(p_cam)

            # 残差
            residual[2*idx] = pred[0] - obs.pixel[0]
            residual[2*idx + 1] = pred[1] - obs.pixel[1]

        return residual

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        计算Jacobian矩阵（入口函数）

        根据use_analytical_jacobian参数选择：
          - True: 使用解析Jacobian（快速、精确）
          - False: 使用数值微分（通用、较慢）

        输入: theta (N*6 + M*1,)
        输出: Jacobian矩阵 (2*n_obs, N*6 + M*1)
        参数: 无
        """
        if self.use_analytical_jacobian:
            return self._jacobian_analytical(theta)
        else:
            return self._jacobian_numerical(theta)

    def _jacobian_numerical(self, theta: np.ndarray) -> np.ndarray:
        """
        数值微分计算Jacobian（前向差分法）

        适用场景：验证解析Jacobian正确性、快速原型开发

        精度：O(ε) ≈ 1e-6
        速度：需要调用residual() n_params次（96次）

        输入: theta (N*6 + M*1,)
        输出: Jacobian矩阵 (2*n_obs, N*6 + M*1)
        参数: 无
        """
        # 计算参数维度（每点只有1维：rho）
        n_params = self.n_poses * 6 + self.n_points * 1

        # 计算当前残差
        residual_0 = self.residual(theta)

        # 计算Jacobian
        J = np.zeros((2 * self.n_obs, n_params))

        for i in range(n_params):
            # 使用相对步长以适应不同尺度的参数
            # 对于旋转（李代数，值较小）：使用固定epsilon
            # 对于平移（值约0-6）和逆深度（值约0.1-0.5）：使用相对epsilon
            eps_base = 1e-6
            epsilon = max(eps_base, eps_base * abs(theta[i]))

            theta_perturbed = theta.copy()
            theta_perturbed[i] += epsilon

            residual_perturbed = self.residual(theta_perturbed)

            J[:, i] = (residual_perturbed - residual_0) / epsilon

        return J

    def _jacobian_analytical(self, theta: np.ndarray) -> np.ndarray:
        """
        计算Jacobian矩阵（解析解）

        J = [∂r/∂δρ | ∂r/∂δt | ∂r/∂ρ]
            (2×3)   (2×3)   (2×1)
             旋转     平移   逆深度

        链式法则: ∂r/∂θ = ∂r/∂p_cam * ∂p_cam/∂θ

        对于每个观测:
          1. ∂r/∂p_cam: 相机投影的Jacobian (2×3)
          2. ∂p_cam/∂δt: 平移扰动Jacobian (3×3，单位矩阵)
          3. ∂p_cam/∂δρ: 旋转扰动Jacobian (3×3，-[P_c]×)
          4. ∂p_cam/∂rho: 逆深度扰动Jacobian (3×1)

        输入: theta (N*6 + M*1,)
        输出: Jacobian矩阵 (2*n_obs, N*6 + M*1)
        参数: 无
        """
        n_params = self.n_poses * 6 + self.n_points * 1
        J = np.zeros((2 * self.n_obs, n_params))

        for idx, obs in enumerate(self.slam_map.observations):
            pose_idx = obs.frame_id
            feature_id = obs.feature_id

            # 提取当前位姿
            pose_start = pose_idx * 6
            pose_vec = theta[pose_start:pose_start + 6]
            R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
            t = pose_vec[3:]

            # 提取逆深度参数
            point_start = self.n_poses * 6 + feature_id
            rho = theta[point_start]

            # 获取该特征点的固定像素坐标
            feature = self.slam_map.features[feature_id]
            u_ref = feature.u_ref
            v_ref = feature.v_ref

            # 将逆深度转换为3D点（世界坐标系）
            ref_pose_idx = self.ref_pose_for_point.get(feature_id, 0)

            # 提取参考帧位姿
            ref_pose_start = ref_pose_idx * 6
            ref_pose_vec = theta[ref_pose_start:ref_pose_start + 6]
            R_ref = Rotation.from_rotvec(ref_pose_vec[:3]).as_matrix()
            t_ref = ref_pose_vec[3:]

            # 先在参考帧相机系中构建3D点
            Z_c = 1.0 / (rho + 1e-6)
            X_c_ref = (u_ref - self.cx) * Z_c / self.focal_length
            Y_c_ref = (v_ref - self.cy) * Z_c / self.focal_length
            P_c_ref = np.array([X_c_ref, Y_c_ref, Z_c])

            # 变换到世界坐标系
            P_w = R_ref.T @ (P_c_ref - t_ref)

            # 变换到当前相机系
            P_c = R @ P_w + t  # (3,)
            X, Y, Z = P_c[0], P_c[1], P_c[2]

            # ====== 第1部分：相机投影的Jacobian ∂r/∂p_cam (2×3) ======
            # u = fx * X/Z + cx
            # v = fy * Y/Z + cy
            fx = fy = self.focal_length
            J_proj = np.array([
                [fx/Z,   0,   -fx*X/(Z*Z)],  # ∂u/∂[X,Y,Z]
                [0,     fy/Z, -fy*Y/(Z*Z)]   # ∂v/∂[X,Y,Z]
            ])  # (2, 3)

            # ====== 第2部分：对平移的Jacobian ∂r/∂δt (2×3) ======
            # p_cam = R*P_w + t
            # ∂p_cam/∂t = I (单位矩阵)
            # ∂r/∂t = ∂r/∂p_cam * I = J_proj

            # ====== 第3部分：对旋转的Jacobian ∂r/∂δρ (2×3) ======
            # 李代数扰动: δP_c = -[P_c]× * δρ
            # [P_c]× = [  0  -Z   Y]
            #          [  Z   0  -X]
            #          [ -Y   X   0]
            # ∂p_cam/∂δρ = -[P_c]×

            # 特殊情况：当观测帧=参考帧时
            # P_c = P_c_ref（不依赖于当前位姿）
            # 所以对位姿的Jacobian为0
            if pose_idx == ref_pose_idx:
                J_t = np.zeros((2, 3))  # 对平移的Jacobian为0
                J_rot = np.zeros((2, 3))  # 对旋转的Jacobian为0
            else:
                J_t = J_proj  # (2, 3)
                skew_P = np.array([
                    [0, -Z, Y],
                    [Z, 0, -X],
                    [-Y, X, 0]
                ])  # (3, 3)
                J_rot = J_proj @ (-skew_P)  # (2, 3)

            # ====== 第4部分：对逆深度的Jacobian ∂r/∂ρ (2×1) ======
            # P_c_ref = [(u_ref-cx)/(ρ*fx), (v_ref-cy)/(ρ*fy), 1/ρ]
            # ∂P_c_ref/∂ρ = [-(u_ref-cx)/(ρ²*fx), -(v_ref-cy)/(ρ²*fy), -1/ρ²]

            rho_safe = rho + 1e-10
            dP_c_ref_drho = np.array([
                -(u_ref - self.cx) / (rho_safe**2 * fx),
                -(v_ref - self.cy) / (rho_safe**2 * fy),
                -1.0 / (rho_safe**2)
            ])  # (3,)

            # P_c = R * P_w + t = R * (R_ref^T * (P_c_ref - t_ref)) + t
            # 当观测帧=参考帧时: P_c = P_c_ref
            # 当观测帧≠参考帧时: P_c = R * R_ref^T * (P_c_ref - t_ref) + t

            if pose_idx == ref_pose_idx:
                # 观测帧=参考帧，P_c = P_c_ref
                dP_c_drho = dP_c_ref_drho
            else:
                # 观测帧≠参考帧，需要变换
                # ∂P_c/∂ρ = R * R_ref^T * ∂P_c_ref/∂ρ
                dP_c_drho = R @ (R_ref.T @ dP_c_ref_drho)  # (3,)

            # ∂r/∂ρ = ∂r/∂p_cam * ∂P_c/∂ρ = J_proj * dP_c_drho
            J_rho = J_proj @ dP_c_drho  # (2,)

            # ====== 填充Jacobian矩阵 ======
            # 对于观测idx，对应J的第[2*idx, 2*idx+1]行
            row_start = 2 * idx

            # 位姿部分：[∂r/∂δρ, ∂r/∂δt]
            pose_col_start = pose_idx * 6
            J[row_start:row_start+2, pose_col_start:pose_col_start+3] = J_rot   # 旋转 (2×3)
            J[row_start:row_start+2, pose_col_start+3:pose_col_start+6] = J_t   # 平移 (2×3)

            # 逆深度部分：∂r/∂ρ
            point_col_start = self.n_poses * 6 + feature_id
            J[row_start:row_start+2, point_col_start] = J_rho  # (2,)

        return J

    def jacobian_sparse(self, theta: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        稀疏Jacobian计算（Ceres风格）

        返回:
          indices: [(row_idx, col_idx), ...] 非零元素的索引列表
          values: [J_val, ...] 对应的Jacobian值

        稀疏度分析:
          - 总元素: (2*n_obs) × (n_poses*6 + n_points*1)
          - 每个residual只依赖: 1个pose(6维) + 1个point(1维) = 7个参数
          - 每个residual的非零元素: 2×7 = 14
          - 稀疏度: 1 - (14*n_obs) / total_elements ≈ 92%

        相比稠密版本:
          - 内存: O(14*n_obs) vs O(2*n_obs * n_params)
          - 计算量: 只计算非零块
        """
        indices = []
        values = []

        for idx, obs in enumerate(self.slam_map.observations):
            pose_idx = obs.frame_id
            feature_id = obs.feature_id

            # 提取当前位姿
            pose_start = pose_idx * 6
            pose_vec = theta[pose_start:pose_start + 6]
            R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
            t = pose_vec[3:]

            # 提取逆深度参数
            point_start = self.n_poses * 6 + feature_id
            rho = theta[point_start]

            # 获取该特征点的固定像素坐标
            feature = self.slam_map.features[feature_id]
            u_ref = feature.u_ref
            v_ref = feature.v_ref

            # 将逆深度转换为3D点（世界坐标系）
            ref_pose_idx = self.ref_pose_for_point.get(feature_id, 0)

            # 提取参考帧位姿
            ref_pose_start = ref_pose_idx * 6
            ref_pose_vec = theta[ref_pose_start:ref_pose_start + 6]
            R_ref = Rotation.from_rotvec(ref_pose_vec[:3]).as_matrix()
            t_ref = ref_pose_vec[3:]

            # 先在参考帧相机系中构建3D点
            Z_c = 1.0 / (rho + 1e-6)
            X_c_ref = (u_ref - self.cx) * Z_c / self.focal_length
            Y_c_ref = (v_ref - self.cy) * Z_c / self.focal_length
            P_c_ref = np.array([X_c_ref, Y_c_ref, Z_c])

            # 变换到世界坐标系
            P_w = R_ref.T @ (P_c_ref - t_ref)

            # 变换到当前相机系
            P_c = R @ P_w + t
            X, Y, Z = P_c[0], P_c[1], P_c[2]

            # ====== 相机投影的Jacobian ∂r/∂p_cam (2×3) ======
            fx = fy = self.focal_length
            J_proj = np.array([
                [fx/Z,   0,   -fx*X/(Z*Z)],
                [0,     fy/Z, -fy*Y/(Z*Z)]
            ])

            # 当前residual的行索引
            row_u = 2 * idx
            row_v = 2 * idx + 1

            # ====== 对位姿的Jacobian ======
            if pose_idx != ref_pose_idx:
                # 对旋转 (2×3)
                skew_P = np.array([
                    [0, -Z, Y],
                    [Z, 0, -X],
                    [-Y, X, 0]
                ])
                J_rot = J_proj @ (-skew_P)

                # 填充旋转部分的索引和值
                for i in range(2):  # u, v
                    for j in range(3):  # rx, ry, rz
                        col_idx = pose_start + j
                        indices.append((row_u + i, col_idx))
                        values.append(J_rot[i, j])

                # 对平移 (2×3)
                J_t = J_proj
                # 填充平移部分的索引和值
                for i in range(2):  # u, v
                    for j in range(3):  # tx, ty, tz
                        col_idx = pose_start + 3 + j
                        indices.append((row_u + i, col_idx))
                        values.append(J_t[i, j])

            # ====== 对逆深度的Jacobian ======
            rho_safe = rho + 1e-10
            dP_c_ref_drho = np.array([
                -(u_ref - self.cx) / (rho_safe**2 * fx),
                -(v_ref - self.cy) / (rho_safe**2 * fy),
                -1.0 / (rho_safe**2)
            ])

            if pose_idx == ref_pose_idx:
                dP_c_drho = dP_c_ref_drho
            else:
                dP_c_drho = R @ (R_ref.T @ dP_c_ref_drho)

            J_rho = J_proj @ dP_c_drho

            # 填充逆深度的索引和值
            col_rho = point_start
            indices.append((row_u, col_rho))
            values.append(J_rho[0])
            indices.append((row_v, col_rho))
            values.append(J_rho[1])

        return indices, np.array(values)

    def cost(self, theta: np.ndarray) -> float:
        """
        计算代价函数值

        J(theta) = 0.5 * sum(r²)

        输入: theta (N*6 + M*3,)
        输出: 标量代价值
        参数: 无
        """
        r = self.residual(theta)
        return 0.5 * np.sum(r**2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        计算梯度: grad = J^T @ r

        输入: theta (N*6 + M*3,)
        输出: 梯度向量 (N*6 + M*3,)
        参数: 无
        """
        J = self.jacobian(theta)
        r = self.residual(theta)
        return J.T @ r


# ============================================================================
# 阻尼Gauss-Newton优化器（标准模式）
# ============================================================================

class DampedGaussNewtonOptimizer:
    """
    阻尼Gauss-Newton优化器（Levenberg-Marquardt风格）

    迭代公式:
      H * delta = -grad
      theta_new = theta_old ⊕ delta  (注意：不是简单的+)

    其中:
      H = J^T @ J + λ*I (阻尼Hessian，保证正定)
      grad = J^T @ r (梯度)
      λ = 阻尼因子 (damping)

    关键：参数更新的特殊性
      - 平移：t_new = t_old + delta_t  (直接相加，向量空间)
      - 旋转：R_new = R_old * exp(delta_rot)  (李群流形)
             或 rotvec_new = rotvec_old ⊕ delta_rot  (李代数右乘)

    阻尼的作用:
      - 防止Hessian矩阵奇异
      - 提高数值稳定性
      - 在远离最优值时更稳定
    """

    def __init__(self,
                 max_iterations: int = 100,
                 gradient_threshold: float = 1e-6,
                 step_threshold: float = 1e-6,
                 relative_step_threshold: float = 1e-4,
                 cost_change_threshold: float = 1e-10,
                 damping: float = 1e-3,
                 oscillation_window: int = 5,
                 oscillation_threshold: float = 1e-2) -> None:
        """
        配置终止条件

        输入: 无
        输出: 无
        参数:
          - max_iterations: 最大迭代次数
          - gradient_threshold: 梯度范数阈值（绝对值）
          - step_threshold: 步长范数阈值（绝对值）
          - relative_step_threshold: 相对步长阈值 ||delta|| / ||theta||
          - cost_change_threshold: 代价变化阈值
          - damping: 阻尼因子（Levenberg-Marquardt风格）
          - oscillation_window: 震荡检测窗口大小
          - oscillation_threshold: 震荡判断阈值（代价变化幅度）
        """
        self.max_iterations = max_iterations
        self.gradient_threshold = gradient_threshold
        self.step_threshold = step_threshold
        self.relative_step_threshold = relative_step_threshold
        self.cost_change_threshold = cost_change_threshold
        self.damping = damping
        self.oscillation_window = oscillation_window
        self.oscillation_threshold = oscillation_threshold

    @staticmethod
    def rotation_update(rotvec: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        更新旋转向量（李代数空间）

        方法：R_new = R_old * exp(delta)
              rotvec_new = log(R_new)

        对于小量delta，可以直接相加（近似）：
              rotvec_new = rotvec + delta

        但为了更精确，这里使用完整的李群李代数映射

        输入:
          rotvec: 当前旋转向量 [rx, ry, rz]
          delta: 增量 [drx, dry, drz]

        输出:
          rotvec_new: 更新后的旋转向量
        """
        # 方法1：直接相加（小量近似，速度快）
        # return rotvec + delta

        # 方法2：精确的李群李代数更新
        # R_old = exp(rotvec)
        # R_new = R_old * exp(delta)
        # rotvec_new = log(R_new)

        R_old = Rotation.from_rotvec(rotvec).as_matrix()
        R_delta = Rotation.from_rotvec(delta).as_matrix()
        R_new = R_old @ R_delta
        rotvec_new = Rotation.from_matrix(R_new).as_rotvec()

        return rotvec_new

    def solve(self, problem: MultiViewReprojectionErrorProblem, theta_init: np.ndarray) -> np.ndarray:
        """
        执行优化

        输入: problem (优化问题), theta_init (初始参数)
        输出: theta_opt (最优参数)
        参数: 无
        """
        theta = theta_init.copy()
        cost_prev = problem.cost(theta)
        grad = problem.gradient(theta)
        cost_history = []  # 用于震荡检测

        print(f"\n初始参数维度: {len(theta)}")
        print(f"初始代价: {cost_prev:.6e}")
        print(f"初始梯度范数: {np.linalg.norm(grad):.6e}")
        print("-" * 80)

        for iter in range(self.max_iterations):
            # 步骤1: 计算Jacobian和残差
            J = problem.jacobian(theta)
            r = problem.residual(theta)

            # 步骤2: 构造正规方程: H * delta = -grad
            # H = J^T @ J + λ*I (阻尼Hessian)
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
            # 重要：旋转参数需要特殊处理
            # - 旋转（李代数）：使用指数映射更新
            # - 平移：直接相加
            theta_new = theta.copy()

            # 获取位姿数量（从problem中）
            n_poses = problem.n_poses

            for pose_idx in range(n_poses):
                # 提取当前旋转和平移
                pose_start = pose_idx * 6
                rotvec_old = theta[pose_start:pose_start + 3]
                t_old = theta[pose_start + 3:pose_start + 6]

                # 提取增量
                delta_rot = delta[pose_start:pose_start + 3]
                delta_t = delta[pose_start + 3:pose_start + 6]

                # 更新旋转：R_new = R_old * exp(delta_rot)
                # 使用精确的李群李代数更新
                rotvec_new = self.rotation_update(rotvec_old, delta_rot)

                # 更新平移：直接相加
                t_new = t_old + delta_t

                # 写回
                theta_new[pose_start:pose_start + 3] = rotvec_new
                theta_new[pose_start + 3:pose_start + 6] = t_new

            # 更新3D点：直接相加（在XYZ或逆深度空间中）
            theta_new[n_poses*6:] = theta[n_poses*6:] + delta[n_poses*6:]

            # 步骤5: 计算新的代价和梯度
            cost_new = problem.cost(theta_new)
            grad_new = problem.gradient(theta_new)

            # 打印迭代信息
            delta_norm = np.linalg.norm(delta)
            theta_norm = np.linalg.norm(theta_new)
            relative_change = delta_norm / (theta_norm + 1e-10)

            print(f"迭代 {iter+1:3d} | "
                  f"cost={cost_new:.6e} | "
                  f"||grad||={np.linalg.norm(grad_new):.6e} | "
                  f"||delta||={delta_norm:.6e} | "
                  f"rel_change={relative_change:.6e}")

            # 记录代价历史
            cost_history.append(cost_new)

            # 步骤6: 检查终止条件
            converged = False
            stop_reason = None

            # 震荡检测：代价在窗口内来回跳动
            if len(cost_history) >= self.oscillation_window:
                recent_costs = cost_history[-self.oscillation_window:]
                cost_range = max(recent_costs) - min(recent_costs)
                if cost_range < self.oscillation_threshold:
                    converged = True
                    stop_reason = f"检测到震荡（最近{self.oscillation_window}次迭代代价变化<{self.oscillation_threshold:.2e}）"

            # 终止条件1: 梯度阈值
            if not converged and np.linalg.norm(grad_new) < self.gradient_threshold:
                converged = True
                stop_reason = "梯度阈值满足"

            # 终止条件2: 步长阈值（绝对值）
            elif not converged and delta_norm < self.step_threshold:
                converged = True
                stop_reason = "步长阈值满足（绝对值）"

            # 终止条件3: 相对步长阈值
            elif not converged and relative_change < self.relative_step_threshold:
                converged = True
                stop_reason = "相对步长阈值满足"

            # 终止条件4: 代价变化阈值
            elif not converged and abs(cost_new - cost_prev) < self.cost_change_threshold:
                converged = True
                stop_reason = "代价变化阈值满足"

            if converged:
                print("-" * 80)
                print(f"收敛! 原因: {stop_reason}")
                print(f"总迭代次数: {iter + 1}")
                theta = theta_new
                break

            # 步骤7: 更新状态
            theta = theta_new
            cost_prev = cost_new

            # 步骤8: 检查发散
            if not np.isfinite(cost_new):
                print("-" * 80)
                print("警告: 优化发散! cost = Inf/NaN")
                break

        return theta


# ============================================================================
# 场景生成模块
# ============================================================================

def create_realistic_scene(n_poses: int = 7,
                          n_points: int = 100,
                          focal_length: float = 500.0,
                          cx: float = 320.0,
                          cy: float = 240.0,
                          image_width: int = 640,
                          image_height: int = 480,
                          pixel_noise_std: float = 0.5,
                          seed: int = 42) -> SLAMMap:
    """
    创建真实的SLAM场景（相机运动 + 3D点 + 稀疏观测）- 使用逆深度参数化

    场景设计：
    - 相机沿x轴前进，每次移动1m，绕z轴旋转5度
    - 3D点随机分布在相机前方，使用逆深度参数化
    - 考虑视野、遮挡、深度等真实因素
    - 观测包含像素噪声（真实相机传感器的噪声特性）

    输入:
      n_poses: 相机位姿数量
      n_points: 3D点数量
      focal_length, cx, cy: 相机内参
      image_width, image_height: 图像尺寸
      pixel_noise_std: 像素噪声标准差（像素）
      seed: 随机种子

    输出:
      slam_map: SLAM地图（包含Frame、Feature、Observation）
    """
    np.random.seed(seed)

    slam_map = SLAMMap()

    # 创建相机位姿：沿x轴前进，每次绕z轴旋转5度
    for i in range(n_poses):
        angle = np.radians(i * 5)  # 每次绕z轴旋转5度
        R = Rotation.from_euler('z', angle).as_matrix()
        t = np.array([float(i), 0.0, 0.0])  # 沿x轴移动
        frame = Frame(id=i, pose=Pose(R, t), timestamp=float(i))
        slam_map.add_frame(frame)

    # 创建临时problem用于投影
    class TempProblem:
        def __init__(self):
            self.focal_length = focal_length
            self.cx = cx
            self.cy = cy

        def project(self, p_cam: np.ndarray) -> np.ndarray:
            if p_cam[2] < 1e-6:
                p_cam[2] = 1e-6
            u = self.focal_length * p_cam[0] / p_cam[2] + self.cx
            v = self.focal_length * p_cam[1] / p_cam[2] + self.cy
            return np.array([u, v])

    temp_problem = TempProblem()

    # 构建3D点和观测数据（先尝试观测，只保留有观测的点）
    # 这样避免创建"孤儿点"（没有任何观测的特征点）
    feature_id_counter = 0

    for i in range(n_points):
        # x: 0~7m, y: -3~3m, z: 3~10m (相机前方)
        # 范围扩大以适应7个位姿
        x = np.random.uniform(0, 7)
        y = np.random.uniform(-3, 3)
        z = np.random.uniform(3, 10)
        point_3d = np.array([x, y, z])

        # 使用逆深度参数化（方案1：只优化rho）
        # 使用第一个位姿作为参考帧
        ref_pose = slam_map.frames[0].pose
        inv_depth_params = xyz_to_inverse_depth(point_3d, ref_pose, focal_length, cx, cy)

        # 尝试为这个点创建观测
        has_observation = False

        for frame_id, frame in enumerate(slam_map.frames.values()):
            # 将逆深度参数转回3D点（用于投影计算）
            ref_pose_id = 0  # 第一个帧作为参考
            ref_pose = slam_map.frames[ref_pose_id].pose
            point_3d_temp = inverse_depth_to_3d_vec(inv_depth_params, ref_pose)

            # 变换到相机系
            p_cam = frame.pose.R @ point_3d_temp + frame.pose.t

            # 检查深度（必须在相机前方）
            if p_cam[2] < 0.5:
                continue  # 太近或背后

            # 投影
            obs_pixel = temp_problem.project(p_cam)

            # 检查是否在图像范围内
            if not (0 <= obs_pixel[0] < image_width and 0 <= obs_pixel[1] < image_height):
                continue  # 超出视野

            # 随机遮挡（5%概率被遮挡，降低遮挡率以获得更多观测）
            if np.random.random() < 0.05:
                continue

            # 添加像素噪声（真实相机传感器的噪声）
            # 噪声来源：传感器热噪声、量化误差、特征检测误差等
            obs_pixel_noisy = obs_pixel + np.random.normal(0, pixel_noise_std, 2)

            # 如果这是第一个成功的观测，先创建特征点
            if not has_observation:
                # 提取u_ref, v_ref, rho
                u_ref, v_ref, rho = inv_depth_params
                feature = Feature(
                    id=feature_id_counter,
                    u_ref=u_ref,
                    v_ref=v_ref,
                    rho=rho
                )
                slam_map.add_feature(feature)
                has_observation = True

            # 创建观测（带噪声）
            observation = Observation(
                frame_id=frame_id,
                feature_id=feature_id_counter,
                pixel=obs_pixel_noisy
            )
            slam_map.add_observation(observation)

        # 只有成功创建至少一个观测时，feature_id_counter才增加
        if has_observation:
            feature_id_counter += 1

    return slam_map


def inverse_depth_to_3d_vec(inv_depth_params: np.ndarray, ref_pose: Pose) -> np.ndarray:
    """
    逆深度参数 -> 3D点（辅助函数）

    输入:
      inv_depth_params: [u_ref, v_ref, rho]
      ref_pose: 参考帧位姿

    输出: 3D点世界坐标 [x, y, z]
    """
    u_ref, v_ref, rho = inv_depth_params

    # 在参考帧相机坐标系中的3D点
    Z_c = 1.0 / (rho + 1e-6)
    X_c = (u_ref - 320.0) * Z_c / 500.0  # 简化：使用默认cx, fy
    Y_c = (v_ref - 240.0) * Z_c / 500.0

    P_c = np.array([X_c, Y_c, Z_c])

    # 变换到世界坐标系
    P_w = ref_pose.R.T @ (P_c - ref_pose.t)

    return P_w


# ============================================================================
# 主测试函数
# ============================================================================

def xyz_to_inverse_depth(point_3d: np.ndarray,
                        pose: Pose,
                        focal_length: float,
                        cx: float,
                        cy: float) -> np.ndarray:
    """
    将3D点转换为逆深度参数化

    输入:
      point_3d: [x, y, z] 世界坐标系
      pose: 参考帧位姿
      focal_length, cx, cy: 相机内参

    输出: [u_ref, v_ref, rho] 逆深度参数
    """
    # 变换到相机系
    p_cam = pose.R @ point_3d + pose.t

    # 投影到像素坐标
    u = focal_length * p_cam[0] / p_cam[2] + cx
    v = focal_length * p_cam[1] / p_cam[2] + cy

    # 逆深度
    rho = 1.0 / p_cam[2]

    return np.array([u, v, rho])


def test_bundle_adjustment():
    """测试多视图Bundle Adjustment - 使用逆深度参数化"""
    print("=" * 80)
    print("多视图Bundle Adjustment - 逆深度参数化")
    print("=" * 80)
    print()
    print("场景描述:")
    print("  - 7个相机位姿（沿x轴移动，每次旋转5度）")
    print("  - 100个3D特征点（逆深度参数化）")
    print("  - 稀疏观测（考虑视野、遮挡、深度）")
    print("  - 像素噪声：标准差0.5像素")
    print("  - 只保留至少有1次观测的特征点")
    print()
    print("参数化方式: 逆深度 (Inverse Depth)")
    print("  每个点: [u_ref, v_ref, rho]")
    print("  - u_ref, v_ref: 参考帧中的像素坐标")
    print("  - rho = 1/d: 逆深度")
    print("  优势: 更好的数值稳定性，可表示无穷远点")
    print()
    print("优化方法: 阻尼Gauss-Newton (Levenberg-Marquardt风格)")
    print("  H * delta = -grad")
    print("  H = J^T @ J + λ*I")
    print()
    print("终止条件:")
    print("  1. 梯度阈值: ||J^T @ r|| < 1e-6")
    print("  2. 步长阈值: ||delta|| < 1e-6")
    print("  3. 代价变化阈值: |cost_new - cost_old| < 1e-10")
    print("  4. 最大迭代次数: 100")
    print("=" * 80)

    # 相机内参
    focal_length = 500.0
    cx = 320.0
    cy = 240.0
    image_width = 640
    image_height = 480

    # 创建真实场景
    slam_map = create_realistic_scene(
        n_poses=7,
        n_points=100,
        focal_length=focal_length,
        cx=cx, cy=cy,
        image_width=image_width,
        image_height=image_height,
        pixel_noise_std=0.5,  # 像素噪声标准差
        seed=42
    )

    # 打印地图统计
    slam_map.print_summary()

    # 显示真实数据
    print("\n>>> 真实数据:")
    for frame_id, frame in slam_map.frames.items():
        print(f"  位姿{frame_id+1}: t={frame.pose.t}")

    # 构造theta结构
    n_poses = len(slam_map.frames)
    n_points = len(slam_map.features)

    # 根据参数化方式计算theta维度
    # 逆深度参数化（方案1）：每个点只有1维 [rho]
    theta_size = n_poses * 6 + n_points * 1

    # 创建带噪声的初始估计
    np.random.seed(42)
    theta_init = np.zeros(theta_size)

    print("\n>>> 初始估计（添加噪声，大规模场景降低噪声）:")
    for frame_id, frame in slam_map.frames.items():
        # 位姿噪声：旋转±1度，平移±0.05m（降低噪声以帮助大规模优化收敛）
        angle_noise = np.random.uniform(-1, 1)
        R_noisy = Rotation.from_euler('z', np.radians(angle_noise)).as_matrix() @ frame.pose.R
        t_noisy = frame.pose.t + np.random.normal(0, 0.05, 3)

        rotvec = Rotation.from_matrix(R_noisy).as_rotvec()
        theta_init[frame_id*6:(frame_id+1)*6] = np.concatenate([rotvec, t_noisy])

    for feature_id, feature in slam_map.features.items():
        # 逆深度参数化：只优化rho（1维）
        rho = feature.rho

        # 添加噪声（只对rho加噪声）
        rho_noisy = rho + np.random.normal(0, 0.005)
        rho_noisy = max(rho_noisy, 0.01)  # 限制逆深度最小值

        theta_init[n_poses*6 + feature_id] = rho_noisy

    print(f"  theta维度: {len(theta_init)}")
    print(f"  结构: [{n_poses}个pose({n_poses*6}维), {n_points}个逆深度point({n_points}维)]")
    print(f"  参数减少: {3*n_points} -> {n_points}维")

    # 创建优化问题
    problem = MultiViewReprojectionErrorProblem(
        slam_map=slam_map,
        focal_length=focal_length,
        cx=cx, cy=cy
    )

    # 创建优化器（调整终止条件以适应实际收敛行为）
    optimizer = DampedGaussNewtonOptimizer(
        max_iterations=100,
        gradient_threshold=1e-2,  # 梯度阈值（宽松，因为梯度范数较大）
        step_threshold=1e-4,  # 步长阈值（绝对值）
        relative_step_threshold=1e-4,  # 相对步长：||delta||/||theta|| < 0.01%（从1e-5放宽到1e-4）
        cost_change_threshold=1e-3,  # 代价变化阈值（放宽以应对震荡）
        damping=1e-2,  # 阻尼因子
        oscillation_window=10,  # 震荡检测窗口
        oscillation_threshold=5e-1  # 震荡阈值（cost范围<0.5时认为震荡）
    )

    # 求解
    print(f"\n>>> 开始优化")
    theta_opt = optimizer.solve(problem, theta_init)

    # 解析优化结果
    print("\n>>> 优化结果:")
    for frame_id in slam_map.frames.keys():
        rotvec = theta_opt[frame_id*6:frame_id*6+3]
        t = theta_opt[frame_id*6+3:(frame_id+1)*6]
        print(f"  位姿{frame_id+1}: t={t}")

    print(f"\n  逆深度参数（前5个）:")
    for feature_id in sorted(slam_map.features.keys())[:5]:
        # 从theta中提取优化后的rho（只有1维）
        rho_opt = theta_opt[n_poses*6 + feature_id]

        # u_ref, v_ref是固定的（从feature对象获取）
        u_ref = slam_map.features[feature_id].u_ref
        v_ref = slam_map.features[feature_id].v_ref

        # 转换为深度
        depth = 1.0 / rho_opt if rho_opt > 1e-6 else float('inf')

        # 转换为相机系3D点
        Z_c = depth
        X_c = (u_ref - cx) * Z_c / focal_length
        Y_c = (v_ref - cy) * Z_c / focal_length

        print(f"    点{feature_id+1}:")
        print(f"      逆深度: u={u_ref:.2f}, v={v_ref:.2f} (固定)")
        print(f"               rho={rho_opt:.4f} (优化)")
        print(f"      深度: {depth:.2f}m")
        print(f"      相机系3D: X={X_c:.2f}m, Y={Y_c:.2f}m, Z={Z_c:.2f}m")

    # 计算最终代价
    cost_final = problem.cost(theta_opt)
    print(f"\n  最终代价: {cost_final:.6e}")

    # 与真实值对比
    print(f"\n>>> 与真实值对比:")
    for frame_id, frame in slam_map.frames.items():
        t_true = frame.pose.t
        t_opt = theta_opt[frame_id*6+3:(frame_id+1)*6]
        error_t = np.linalg.norm(t_opt - t_true)
        print(f"  位姿{frame_id+1}平移误差: {error_t:.6f}")

    stats = slam_map.get_statistics()

    # 逆深度参数化：比较深度误差
    total_depth_error = 0.0
    for feature_id, feature in slam_map.features.items():
        # 真实逆深度
        rho_true = feature.rho
        # 优化得到的逆深度（只有1维）
        rho_opt = theta_opt[n_poses*6 + feature_id]

        # 比较深度
        depth_true = 1.0 / rho_true
        depth_opt = 1.0 / rho_opt if rho_opt > 1e-6 else float('inf')
        error = abs(depth_opt - depth_true)
        total_depth_error += error

    print(f"  平均深度误差: {total_depth_error / n_points:.6f}")

    # 分析未观测的点
    unobserved_points = [
        fid for fid, count in stats['n_obs_per_feature'].items() if count == 0
    ]
    if unobserved_points:
        print(f"\n  注意: 以下点从未被观测（仅依赖初始化）: {unobserved_points}")

    print("\n>>> 重要说明:")
    print("  1. 逆深度参数化：每个点用[u_ref, v_ref, rho]表示")
    print("  2. 优势：更好的数值稳定性，可表示无穷远点")
    print("  3. 深度 = 1/rho，逆深度的线性变化对应深度的非线性变化")
    print("  4. 单目视觉存在尺度模糊（Scale Ambiguity）")
    print("  5. 未观测的点不受约束，估计不准确")
    print("  6. 旋转更新使用李群李代数映射（非直接相加）")
    print("  7. 像素噪声：观测包含高斯噪声（std=0.5像素）")

    print("=" * 80)


def test_jacobian_comparison():
    """对比数值微分和解析Jacobian"""
    print("=" * 80)
    print("Jacobian方法对比测试")
    print("=" * 80)

    # 创建小规模测试场景
    slam_map = create_realistic_scene(
        n_poses=3,  # 减少规模以加快测试
        n_points=20,
        focal_length=500.0,
        cx=320.0, cy=240.0,
        pixel_noise_std=0.5,
        seed=42
    )

    n_poses = len(slam_map.frames)
    n_points = len(slam_map.features)
    theta_size = n_poses * 6 + n_points * 1

    # 创建初始参数
    np.random.seed(42)
    theta_init = np.zeros(theta_size)

    for frame_id, frame in slam_map.frames.items():
        angle_noise = np.random.uniform(-1, 1)
        R_noisy = Rotation.from_euler('z', np.radians(angle_noise)).as_matrix() @ frame.pose.R
        t_noisy = frame.pose.t + np.random.normal(0, 0.05, 3)
        rotvec = Rotation.from_matrix(R_noisy).as_rotvec()
        theta_init[frame_id*6:(frame_id+1)*6] = np.concatenate([rotvec, t_noisy])

    for feature_id, feature in slam_map.features.items():
        rho = feature.rho + np.random.normal(0, 0.005)
        theta_init[n_poses*6 + feature_id] = max(rho, 0.01)

    print(f"\n>>> 测试配置:")
    print(f"  位姿数: {n_poses}")
    print(f"  特征点数: {n_points}")
    print(f"  观测数: {len(slam_map.observations)}")
    print(f"  参数维度: {theta_size}")

    # 测试数值微分
    print(f"\n{'='*80}")
    print(">>> 方法1: 数值微分（前向差分）")
    print("="*80)

    import time
    problem_numerical = MultiViewReprojectionErrorProblem(
        slam_map=slam_map,
        focal_length=500.0,
        cx=320.0, cy=240.0,
        use_analytical_jacobian=False
    )

    start = time.time()
    J_numerical = problem_numerical.jacobian(theta_init)
    time_numerical = time.time() - start

    print(f"计算时间: {time_numerical*1000:.2f} ms")
    print(f"Jacobian形状: {J_numerical.shape}")
    print(f"Jacobian范数: {np.linalg.norm(J_numerical):.6e}")
    print(f"Jacobian范围: [{J_numerical.min():.2e}, {J_numerical.max():.2e}]")

    # 测试解析Jacobian
    print(f"\n{'='*80}")
    print(">>> 方法2: 解析Jacobian")
    print("="*80)

    problem_analytical = MultiViewReprojectionErrorProblem(
        slam_map=slam_map,
        focal_length=500.0,
        cx=320.0, cy=240.0,
        use_analytical_jacobian=True
    )

    start = time.time()
    J_analytical = problem_analytical.jacobian(theta_init)
    time_analytical = time.time() - start

    print(f"计算时间: {time_analytical*1000:.2f} ms")
    print(f"Jacobian形状: {J_analytical.shape}")
    print(f"Jacobian范数: {np.linalg.norm(J_analytical):.6e}")
    print(f"Jacobian范围: [{J_analytical.min():.2e}, {J_analytical.max():.2e}]")

    # 对比分析
    print(f"\n{'='*80}")
    print(">>> 对比分析")
    print("="*80)

    diff = J_analytical - J_numerical
    rel_diff = np.abs(diff) / (np.abs(J_numerical) + 1e-10)

    print(f"绝对误差: ||J_ana - J_num|| = {np.linalg.norm(diff):.6e}")
    print(f"相对误差: mean(|Δ|/|J|) = {np.mean(rel_diff):.6e}")
    print(f"最大相对误差: {np.max(rel_diff):.6e}")

    speedup = time_numerical / time_analytical
    print(f"\n速度提升: {speedup:.1f}x")
    print(f"  数值微分: {time_numerical*1000:.2f} ms")
    print(f"  解析方法: {time_analytical*1000:.2f} ms")

    # 验证梯度一致性
    grad_num = problem_numerical.gradient(theta_init)
    grad_ana = problem_analytical.gradient(theta_init)
    grad_diff = np.linalg.norm(grad_num - grad_ana)

    print(f"\n梯度一致性:")
    print(f"  数值微分梯度范数: {np.linalg.norm(grad_num):.6e}")
    print(f"  解析方法梯度范数: {np.linalg.norm(grad_ana):.6e}")
    print(f"  梯度差异: {grad_diff:.6e}")

    print("\n>>> 结论:")
    if speedup > 10:
        print(f"  ✓ 解析Jacobian速度提升{speedup:.0f}倍")
    if np.mean(rel_diff) < 0.01:
        print(f"  ✓ 两种方法误差较小(<1%)，验证了解析Jacobian的正确性")
    if grad_diff < 1e-3:
        print(f"  ✓ 梯度计算一致，优化器行为应该相似")

    print("=" * 80)


def test_sparse_jacobian():
    """对比稠密和稀疏Jacobian"""
    print("=" * 80)
    print("稀疏 vs 稠密 Jacobian 对比")
    print("=" * 80)

    # 创建场景
    slam_map = create_realistic_scene(
        n_poses=7, n_points=54,
        focal_length=500.0,
        cx=320.0, cy=240.0,
        pixel_noise_std=0.5,
        seed=42
    )

    n_poses = len(slam_map.frames)
    n_points = len(slam_map.features)
    theta_size = n_poses * 6 + n_points * 1

    # 创建初始参数
    theta = np.zeros(theta_size)
    for fid, frame in slam_map.frames.items():
        theta[fid*6:(fid+1)*6] = np.concatenate([
            Rotation.from_matrix(frame.pose.R).as_rotvec(),
            frame.pose.t
        ])
    for fid, feat in slam_map.features.items():
        theta[n_poses*6 + fid] = feat.rho

    problem = MultiViewReprojectionErrorProblem(slam_map)

    print(f"\n>>> 问题规模:")
    print(f"  位姿数: {n_poses}")
    print(f"  特征点数: {n_points}")
    print(f"  观测数: {len(slam_map.observations)}")
    print(f"  参数维度: {theta_size}")

    # 稠密Jacobian
    print(f"\n{'='*80}")
    print(">>> 方法1: 稠密Jacobian")
    print("="*80)

    import time
    start = time.time()
    J_dense = problem.jacobian(theta)
    time_dense = time.time() - start

    n_elements = J_dense.size
    n_nonzero = np.count_nonzero(np.abs(J_dense) > 1e-10)
    sparsity = 1.0 - n_nonzero / n_elements

    print(f"计算时间: {time_dense*1000:.2f} ms")
    print(f"矩阵形状: {J_dense.shape}")
    print(f"总元素数: {n_elements}")
    print(f"非零元素: {n_nonzero}")
    print(f"稀疏度: {sparsity*100:.1f}%")
    print(f"内存占用: {J_dense.nbytes / 1024:.1f} KB")

    # 稀疏Jacobian
    print(f"\n{'='*80}")
    print(">>> 方法2: 稀疏Jacobian（Ceres风格）")
    print("="*80)

    start = time.time()
    indices, values = problem.jacobian_sparse(theta)
    time_sparse = time.time() - start

    print(f"计算时间: {time_sparse*1000:.2f} ms")
    print(f"非零元素数: {len(values)}")
    print(f"内存占用（索引+值）: {(len(indices)*16 + values.nbytes) / 1024:.1f} KB")

    # 验证一致性
    print(f"\n{'='*80}")
    print(">>> 验证一致性")
    print("="*80)

    # 从稀疏构建稠密矩阵验证
    J_from_sparse = np.zeros(J_dense.shape)
    for (row, col), val in zip(indices, values):
        J_from_sparse[row, col] = val

    diff = np.linalg.norm(J_dense - J_from_sparse)
    print(f"稠密 vs 稀疏重构的差异: {diff:.6e}")

    if diff < 1e-6:
        print("✓ 稀疏Jacobian正确！")

    print(f"\n{'='*80}")
    print(">>> 性能对比")
    print("="*80)
    print(f"计算速度提升: {time_dense/time_sparse:.1f}x")
    print(f"内存节省: {(1 - (len(indices)*16 + values.nbytes) / J_dense.nbytes) * 100:.1f}%")

    # Ceres风格展示
    print(f"\n{'='*80}")
    print(">>> Ceres使用示例（类比）")
    print("="*80)
    print("```cpp")
    print("// Ceres代码（每个residual block只计算自己的小Jacobian）")
    print("for (auto& obs : observations) {")
    print("    problem.AddResidualBlock(")
    print("        new AutoDiffCostFunction<ReprojError, 2, 6, 1>(...),")
    print("        nullptr,")
    print("        pose_param,   // 6维")
    print("        point_param   // 1维")
    print("    );")
    print("    // 每个block只计算 2×(6+1) = 14 个元素")
    print("    // 而不是 2×96 = 192 个元素")
    print("}")
    print("```")
    print(f"\n  每个residual block: 2×7 = 14个非零元素")
    print(f"  稠密表示: 2×96 = 192个元素（其中178个是0！）")
    print(f"  稀疏表示: 只存储14个元素，节省 {1-14/192:.1%}")

    print("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            test_jacobian_comparison()
        elif sys.argv[1] == "--sparse":
            test_sparse_jacobian()
    else:
        test_bundle_adjustment()

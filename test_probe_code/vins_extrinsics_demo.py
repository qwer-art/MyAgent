"""
VINS外参与预积分 - 符合VINS-Mono实际的实现

核心思想：
1. IMU预积分在IMU系中进行
2. 优化变量在相机系中
3. 通过外参R_ci, t_ci进行坐标转换
4. 残差在相机系中构建
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple

class VINSExtrinsics:
    """
    VINS外参管理
    
    标准符号：
    - R_ci: 从IMU到相机的旋转
    - t_ci: 从IMU到相机的平移（杆臂）
    """
    
    def __init__(self, R_ci: np.ndarray, t_ci: np.ndarray):
        """
        Args:
            R_ci: 3x3旋转矩阵，IMU到相机
            t_ci: 3x1平移向量，IMU到相机（杆臂）
        """
        self.R_ci = R_ci
        self.t_ci = t_ci
        
        print("VINS外参配置:")
        print(f"  R_ci (IMU->相机): \n{R_ci}")
        print(f"  t_ci (杆臂): {t_ci} m")
    
    def imu_to_camera(self, p_i: np.ndarray, v_i: np.ndarray, omega: np.ndarray) -> Tuple:
        """
        IMU状态转换到相机系
        
        公式（考虑杆臂的速度补偿）：
        - p_c = R_ci * p_i + t_ci
        - v_c = R_ci * v_i + ω × t_ci  # 注意这里的叉乘！
        
        Args:
            p_i: IMU系位置
            v_i: IMU系速度
            omega: IMU系角速度
            
        Returns:
            (p_c, v_c): 相机系位置和速度
        """
        # 位置转换
        p_c = self.R_ci @ p_i + self.t_ci
        
        # 速度转换（带杆臂补偿）
        v_c = self.R_ci @ v_i + np.cross(omega, self.t_ci)
        
        return p_c, v_c
    
    def camera_to_imu(self, p_c: np.ndarray, v_c: np.ndarray, omega: np.ndarray) -> Tuple:
        """
        相机状态转换到IMU系（逆变换）
        
        公式：
        - p_i = R_ci^T * (p_c - t_ci)
        - v_i = R_ci^T * (v_c - ω × t_ci)
        
        Args:
            p_c: 相机系位置
            v_c: 相机系速度
            omega: IMU系角速度
            
        Returns:
            (p_i, v_i): IMU系位置和速度
        """
        # 位置逆转换
        p_i = self.R_ci.T @ (p_c - self.t_ci)
        
        # 速度逆转换（带杆臂补偿）
        v_i = self.R_ci.T @ (v_c - np.cross(omega, self.t_ci))
        
        return p_i, v_i
    
    def transform_preintegration(self, delta_p_i: np.ndarray, delta_v_i: np.ndarray, 
                                R_c1: np.ndarray, R_c2: np.ndarray) -> Tuple:
        """
        转换预积分结果到相机系，用于残差构建
        
        在VINS中，残差是这样构建的：
        
        r_p = R_ci * Δp_i - (p_c2 - R_c2 @ R_c1.T @ p_c1)
        r_v = R_ci * Δv_i - (v_c2 - R_c2 @ R_c1.T @ v_c1)
        
        Args:
            delta_p_i: IMU系预积分位置增量
            delta_v_i: IMU系预积分速度增量
            R_c1: 时刻1的相机姿态
            R_c2: 时刻2的相机姿态
            
        Returns:
            (delta_p_c, delta_v_c): 相机系中的预积分增量
        """
        # 将IMU预积分结果转换到相机系
        delta_p_c = self.R_ci @ delta_p_i
        delta_v_c = self.R_ci @ delta_v_i
        
        return delta_p_c, delta_v_c


def demonstrate_vins_workflow():
    """演示VINS中的典型工作流程"""
    print("=" * 80)
    print("VINS外参与预积分演示")
    print("=" * 80)
    
    # 1. 定义外参（相机在IMU前方0.1m，上方0.2m）
    t_ci = np.array([0.1, 0.0, 0.2])
    R_ci = Rotation.from_euler('y', np.radians(30)).as_matrix()  # 相机相对IMU偏航30°
    
    extrinsics = VINSExtrinsics(R_ci, t_ci)
    
    # 2. 模拟IMU测量
    print("\n>>> IMU测量（在IMU系中）")
    p_i = np.array([1.0, 2.0, 0.0])  # IMU位置
    v_i = np.array([5.0, 0.0, 0.0])  # IMU速度
    omega = np.array([0.0, 0.0, 0.5])  # 绕z轴旋转
    
    print(f"  p_i = {p_i}")
    print(f"  v_i = {v_i}")
    print(f"  ω   = {omega} (rad/s)")
    
    # 3. 转换到相机系
    print("\n>>> 转换到相机系")
    p_c, v_c = extrinsics.imu_to_camera(p_i, v_i, omega)
    
    print(f"  p_c = {p_c}")
    print(f"  v_c = {v_c}")
    print(f"\n  杆臂效应导致速度偏差:")
    print(f"    不考虑杆臂: v_c_simple = R_ci @ v_i = {R_ci @ v_i}")
    print(f"    考虑杆臂:   v_c_correct = R_ci @ v_i + ω×t_ci = {v_c}")
    print(f"    差异: {v_c - R_ci @ v_i}")
    
    # 4. 预积分残差构建
    print("\n>>> 预积分残差构建")
    delta_p_i = np.array([0.5, 0.1, 0.0])  # IMU系预积分位置增量
    delta_v_i = np.array([0.2, 0.0, 0.0])  # IMU系预积分速度增量
    
    # 假设两个关键帧的相机姿态
    R_c1 = Rotation.from_euler('xyz', [0, 0, 0]).as_matrix()
    R_c2 = Rotation.from_euler('z', np.radians(10)).as_matrix()
    
    # 转换预积分到相机系
    delta_p_c, delta_v_c = extrinsics.transform_preintegration(
        delta_p_i, delta_v_i, R_c1, R_c2
    )
    
    print(f"  Δp_c (相机系) = {delta_p_c}")
    print(f"  Δv_c (相机系) = {delta_v_c}")
    
    # 假设优化变量（相机位姿和速度）
    p_c1 = np.array([0.0, 0.0, 0.0])
    p_c2 = np.array([0.6, 0.1, 0.0])
    v_c1 = np.array([5.0, 0.0, 0.0])
    v_c2 = np.array([5.2, 0.1, 0.0])
    
    # 构建残差（在相机系中）
    r_p = delta_p_c - (p_c2 - R_c2 @ R_c1.T @ p_c1)
    r_v = delta_v_c - (v_c2 - R_c2 @ R_c1.T @ v_c1)
    
    print(f"\n  预积分残差:")
    print(f"    r_p = {r_p}")
    print(f"    r_v = {r_v}")
    
    print("\n" + "=" * 80)
    print("关键点:")
    print("  1. 预积分在IMU系中进行，但通过外参转换到相机系")
    print("  2. 优化变量(p, v, R)都在相机系中")
    print("  3. 杆臂补偿主要影响速度转换: v_c = R_ci@v_i + ω×t_ci")
    print("  4. 残差在相机系中构建，与视觉残差统一")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_vins_workflow()

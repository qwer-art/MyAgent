"""
Jacobian与Hessian分析 - SLAM优化测试探针代码
单文件版本，包含Jacobian分析、Gauss-Newton Hessian分析、条件数分析等功能

环境: my_agent (conda)
依赖: numpy, matplotlib

知识点（SLAM相关）：
1. Gauss-Newton近似：H_GN = J^T @ J（不需要二阶导数）
2. 梯度：g = J^T @ r
3. 条件数：cond(H) = |λ_max| / |λ_min|，衡量优化困难度
4. 特征值分解：H = QΛQ^T，用于分析矩阵性质
5. 牛顿方向：d = -H^(-1) @ g
6. 阻尼方法：H_LM = H + μI，改善条件数
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class HessianAnalysisResult:
    """Hessian分析结果"""
    hessian: np.ndarray  # Hessian矩阵
    eigenvalues: np.ndarray  # 特征值
    eigenvectors: np.ndarray  # 特征向量
    condition_number: float  # 条件数
    is_positive_definite: bool  # 是否正定
    rank: int  # 矩阵秩


class JacobianAnalyzer:
    """Jacobian与Hessian分析器（SLAM优化专用）"""

    def __init__(self) -> None:
        """初始化分析器"""
        pass

    def analyze_jacobian_structure(self, J: np.ndarray) -> None:
        """
        分析Jacobian矩阵的结构

        Args:
            J: Jacobian矩阵，形状(m, n)，m是残差维度，n是参数维度
        """
        print(f"\n{'='*80}")
        print(f"Jacobian矩阵结构分析")
        print(f"{'='*80}")
        print(f"形状: {J.shape}")
        print(f"  残差维度（行）: {J.shape[0]}")
        print(f"  参数维度（列）: {J.shape[1]}")
        print(f"  过约束比: {J.shape[0] / J.shape[1]:.2f}")

        # 稀疏性分析
        n_zeros = np.sum(np.abs(J) < 1e-10)
        sparsity = n_zeros / J.size * 100
        print(f"\n稀疏性:")
        print(f"  零元素数: {n_zeros}/{J.size} ({sparsity:.1f}%)")
        print(f"  非零元素数: {J.size - n_zeros}")

        # 范数分析
        J_F = np.linalg.norm(J)
        J_2 = np.linalg.norm(J, 2)
        max_elem = np.max(np.abs(J))

        print(f"\n范数:")
        print(f"  ||J||_F (Frobenius): {J_F:.6e}")
        print(f"  ||J||_2 (谱范数): {J_2:.6e}")
        print(f"  最大元素: {max_elem:.6e}")

        # 范数诊断
        print(f"\n  诊断:")
        if J_F > 20 or J_2 > 10:
            print(f"    ⚠️ 范数偏大，优化过于敏感")
            print(f"    建议：使用Levenberg-Marquardt阻尼 (H + μI)")
        elif J_F < 0.1 or J_2 < 0.01:
            print(f"    ⚠️ 范数偏小，梯度微弱，优化可能不动")
            print(f"    建议：检查参数化或调整优化器阈值")
        else:
            print(f"    ✅ 范数合适，优化稳定性良好")

    def analyze_gn_hessian(self, J: np.ndarray, print_details: bool = True) -> np.ndarray:
        """
        分析Gauss-Newton Hessian矩阵：H = J^T @ J

        Args:
            J: Jacobian矩阵
            print_details: 是否打印详细信息

        Returns:
            H_GN: Gauss-Newton Hessian矩阵
        """
        H_GN = J.T @ J

        if print_details:
            print(f"\n{'='*80}")
            print(f"Gauss-Newton Hessian分析: H = J^T @ J")
            print(f"{'='*80}")
            print(f"Hessian形状: {H_GN.shape}")
            print(f"  参数维度: {H_GN.shape[0]}")
            print(f"  元素总数: {H_GN.shape[0] * H_GN.shape[1]}")

            # 对称性验证
            is_symmetric = np.allclose(H_GN, H_GN.T, atol=1e-10)
            print(f"\n对称性: {'✓ 是' if is_symmetric else '✗ 否'}")
            if not is_symmetric:
                diff = np.linalg.norm(H_GN - H_GN.T)
                print(f"  ||H - H^T|| = {diff:.2e}")

        return H_GN

    def analyze_hessian(self, H: np.ndarray) -> HessianAnalysisResult:
        """
        分析Hessian矩阵的性质

        分析目的：
        1. 特征值分解 → 理解地形曲率，判断优化难度
           - 大特征值 → 陡峭方向，收敛快
           - 小特征值 → 平坦方向，收敛慢

        2. 条件数 → 衡量病态程度，决定是否需要阻尼
           - 条件数 = |λ_max| / |λ_min|（地形崎岖程度）
           - 条件数越大 → 优化越难、越不稳定、收敛越慢
           - κ<100: 优化健康 ✓
           - κ>1000: 需要阻尼 ⚠️

        3. 正定性 → 判断是否在最小值附近，优化方向是否正确
           - 正定（λ>0）：在局部最小值附近 ✓
           - 不定（有λ<0）：在鞍点附近 ✗
           - 负定（λ<0）：在局部最大值附近 ✗✗

        4. 矩阵秩 → 检测参数冗余，是否存在不可观测参数
           - 满秩：所有参数都可被观测 ✓
           - 秩亏：存在不可观测参数（如尺度模糊）✗

        Args:
            H: Hessian矩阵

        Returns:
            Hessian分析结果
        """
        print(f"\n{'='*80}")
        print(f"Hessian矩阵性质分析")
        print(f"{'='*80}")

        # 特征值分解
        # 说明：特征值表示各方向的曲率大小
        #   - 大特征值 → 陡峭方向，收敛快
        #   - 小特征值 → 平坦方向，收敛慢
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        print(f"\n特征值分解: H = QΛQ^T")
        print(f"特征值个数: {len(eigenvalues)}")
        print(f"特征值范围: [{np.min(eigenvalues):.6e}, {np.max(eigenvalues):.6e}]")

        # 打印前10个和后10个特征值
        if len(eigenvalues) > 20:
            print(f"\n最小10个特征值:")
            for i in range(10):
                print(f"  λ_{i} = {eigenvalues[i]:.6e}")
            print(f"\n最大10个特征值:")
            for i in range(-10, 0):
                print(f"  λ_{len(eigenvalues)+i} = {eigenvalues[i]:.6e}")
        else:
            print(f"\n所有特征值:")
            for i, val in enumerate(eigenvalues):
                print(f"  λ_{i} = {val:.6e}")

        # 条件数
        # 说明：条件数 = 地形崎岖程度 = |λ_max| / |λ_min|
        #   - 条件数越大 → 优化越难、越不稳定、收敛越慢
        #   - 条件数越小越好，理想值κ=1（所有方向曲率相同）
        #   - 条件数每增大10倍，收敛速度慢约10倍
        #   - κ<100: 优化健康 ✓
        #   - κ>1000: 需要阻尼 ⚠️
        abs_eigenvalues = np.abs(eigenvalues)
        max_eig = np.max(abs_eigenvalues)
        min_eig = np.min(abs_eigenvalues[abs_eigenvalues > 1e-10])  # 忽略接近零的特征值

        if min_eig < 1e-10 or max_eig / min_eig > 1e10:
            condition_number = np.inf
        else:
            condition_number = max_eig / min_eig

        print(f"\n条件数: κ(H) = {condition_number:.2e}")
        print(f"  |λ_max| / |λ_min| = {max_eig:.6e} / {min_eig:.6e}")

        if condition_number < 100:
            print(f"  矩阵状态: ✓ 良态（well-conditioned）")
            print(f"  影响：优化稳定，收敛快，无需特殊处理")
        elif condition_number < 1e6:
            print(f"  矩阵状态: ⚠️  中等病态（moderately ill-conditioned）")
            print(f"  影响：优化变慢，可能震荡，建议使用LM阻尼")
        elif condition_number < 1e10:
            print(f"  矩阵状态: ✗ 严重病态（severely ill-conditioned）")
            print(f"  影响：优化困难，数值不稳定，需要强阻尼")
        else:
            print(f"  矩阵状态: ✗✗ 极度病态（extremely ill-conditioned）")
            print(f"  影响：优化可能发散，必须阻尼+参数归一化")

        # 正定性判断
        # 说明：正定性判断当前位置的性质
        #   - 正定（所有λ>0）：在局部最小值附近 ✓
        #     → 优化方向正确，可以继续优化
        #     → 地形像碗状，稳定平衡点
        #   - 不定（有λ<0）：在鞍点附近 ✗
        #     → 牛顿方向可能错误
        #     → 地形像马鞍，不稳定平衡点
        #     → 需要：检查初始值或使用信赖域
        #   - 负定（所有λ<0）：在局部最大值附近 ✗✗
        #     → 优化方向反了，需要反转
        #     → 地形像山峰，不稳定平衡点
        is_positive_definite = np.all(eigenvalues > 0)
        is_positive_semidefinite = np.all(eigenvalues >= -1e-10)

        print(f"\n正定性分析:")
        print(f"  正定（positive definite）: {'✓ 是' if is_positive_definite else '✗ 否'}")
        print(f"  半正定（positive semidefinite）: {'✓ 是' if is_positive_semidefinite else '✗ 否'}")

        if not is_positive_definite:
            neg_count = np.sum(eigenvalues < -1e-10)
            zero_count = np.sum(np.abs(eigenvalues) < 1e-10)
            print(f"  负特征值数: {neg_count}")
            print(f"  零特征值数: {zero_count}")

        # 矩阵秩
        # 说明：矩阵秩 = 可观测参数数
        #   - 满秩（rank=n）：所有参数都可被观测 ✓
        #   - 秩亏（rank<n）：存在冗余或不可观测参数 ✗
        #     → 可能原因：尺度模糊（单目SLAM）
        #     → 可能原因：观测不足
        #     → 可能原因：特征点共线
        #     → 建议：添加先验约束
        rank = np.sum(np.abs(eigenvalues) > 1e-10)
        rank_deficiency = H.shape[0] - rank

        print(f"\n矩阵秩: rank(H) = {rank}")
        print(f"  秩亏: {rank_deficiency}")

        return HessianAnalysisResult(
            hessian=H,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            condition_number=condition_number,
            is_positive_definite=is_positive_definite,
            rank=rank
        )

    def analyze_damping(self, H: np.ndarray, damping_values: list = None) -> None:
        """
        分析阻尼对条件数的影响

        Args:
            H: Hessian矩阵
            damping_values: 阻尼系数列表
        """
        if damping_values is None:
            damping_values = [0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]

        print(f"\n{'='*80}")
        print(f"阻尼方法分析: H_damped = H + μI")
        print(f"{'='*80}")

        # 原始条件数
        eigenvalues = np.linalg.eigvalsh(H)
        abs_eigenvalues = np.abs(eigenvalues)
        max_eig = np.max(abs_eigenvalues)
        min_eig = np.min(abs_eigenvalues[abs_eigenvalues > 1e-10])

        if min_eig > 1e-10:
            cond_original = max_eig / min_eig
        else:
            cond_original = np.inf

        print(f"\n原始Hessian:")
        print(f"  条件数: {cond_original:.2e}")
        print(f"  特征值范围: [{np.min(eigenvalues):.6e}, {np.max(eigenvalues):.6e}]")

        print(f"\n{'μ':<12} {'κ(H+μI)':<15} {'改善倍数':<12} {'λ_min':<12}")
        print(f"{'-'*60}")

        for mu in damping_values:
            H_damped = H + mu * np.eye(H.shape[0])
            eigenvalues_damped = np.linalg.eigvalsh(H_damped)

            abs_eig = np.abs(eigenvalues_damped)
            max_eig_d = np.max(abs_eig)
            min_eig_d = np.min(eigenvalues_damped)

            if min_eig_d > 1e-10:
                cond_damped = max_eig_d / min_eig_d
            else:
                cond_damped = np.inf

            if cond_original < np.inf:
                improvement = cond_original / cond_damped if cond_damped > 0 else 0
            else:
                improvement = np.inf if cond_damped < np.inf else 1.0

            print(f"{mu:<12.2e} {cond_damped:<15.2e} {improvement:<12.1f} {min_eig_d:<12.6e}")

    def newton_direction(self, H: np.ndarray, g: np.ndarray,
                        method: str = "solve") -> Tuple[np.ndarray, dict]:
        """
        计算牛顿方向：d = -H^(-1) @ g

        Args:
            H: Hessian矩阵
            g: 梯度
            method: 求解方法 ("solve" 或 "inv")

        Returns:
            d: 牛顿方向
            info: 额外信息字典
        """
        info = {
            "success": False,
            "method_used": method,
            "condition_number": np.linalg.cond(H)
        }

        print(f"\n{'='*80}")
        print(f"牛顿方向计算: H * d = -g")
        print(f"{'='*80}")
        print(f"梯度范数: ||g|| = {np.linalg.norm(g):.6e}")
        print(f"Hessian条件数: κ(H) = {info['condition_number']:.2e}")

        try:
            if method == "solve":
                d = np.linalg.solve(H, -g)
                info["success"] = True
                info["method_used"] = "solve"
            else:
                H_inv = np.linalg.inv(H)
                d = -H_inv @ g
                info["success"] = True
                info["method_used"] = "inv"

            print(f"✓ 求解成功")
            print(f"  方法: {info['method_used']}")
            print(f"  牛顿方向范数: ||d|| = {np.linalg.norm(d):.6e}")

            # 验证残差
            residual = H @ d + g
            print(f"  残差: ||Hd + g|| = {np.linalg.norm(residual):.2e}")

        except np.linalg.LinAlgError:
            print(f"✗ 求解失败：Hessian矩阵奇异")
            print(f"  尝试阻尼方法...")

            # 阻尼方法
            mu = 1e-3
            H_damped = H + mu * np.eye(len(g))
            cond_damped = np.linalg.cond(H_damped)

            print(f"  阻尼系数 μ = {mu:.2e}")
            print(f"  阻尼后条件数: {cond_damped:.2e}")

            try:
                d = np.linalg.solve(H_damped, -g)
                info["success"] = True
                info["method_used"] = f"damped_solve(mu={mu})"
                print(f"  ✓ 阻尼求解成功")
                print(f"  牛顿方向范数: ||d|| = {np.linalg.norm(d):.6e}")
            except np.linalg.LinAlgError:
                print(f"  ✗ 阻尼求解也失败")
                d = np.zeros_like(g)

        return d, info


# ============================================================================
# 测试函数：使用真实SLAM场景
# ============================================================================

def run_test_bundle_adjustment():
    """测试1：Bundle Adjustment的Jacobian分析"""
    print(f"\n{'#'*80}")
    print(f"测试1: Bundle Adjustment - Jacobian与Hessian分析")
    print(f"{'#'*80}")

    analyzer = JacobianAnalyzer()

    # 模拟BA问题的Jacobian
    # 10个相机pose（6DoF），50个3D点（逆深度）
    # 每个点平均被3个相机观测
    n_cameras = 10
    n_points = 50
    n_observations = n_points * 3

    n_pose_params = n_cameras * 6  # 每个相机6个参数
    n_point_params = n_points * 1  # 每个点1个参数（逆深度）
    n_params = n_pose_params + n_point_params

    # 随机生成Jacobian（模拟）
    np.random.seed(42)
    J_visual = np.random.randn(n_observations * 2, n_params) * 0.1

    # 添加稀疏性（每个残差只依赖1个pose和1个point）
    J_visual_sparse = np.zeros_like(J_visual)
    for i in range(n_observations):
        cam_id = i % n_cameras
        point_id = i % n_points

        # pose部分
        row_start = i * 2
        col_start = cam_id * 6
        J_visual_sparse[row_start:row_start+2, col_start:col_start+6] = \
            J_visual[row_start:row_start+2, col_start:col_start+6]

        # point部分
        col_point = n_pose_params + point_id
        J_visual_sparse[row_start:row_start+2, col_point] = \
            J_visual[row_start:row_start+2, col_point]

    print(f"\n>>> BA问题设置")
    print(f"  相机数: {n_cameras}")
    print(f"  3D点数: {n_points}")
    print(f"  观测数: {n_observations}")
    print(f"  参数总数: {n_params} (pose: {n_pose_params}, point: {n_point_params})")

    # 分析Jacobian结构
    analyzer.analyze_jacobian_structure(J_visual_sparse)

    # 分析Gauss-Newton Hessian
    H_gn = analyzer.analyze_gn_hessian(J_visual_sparse)

    # 分析Hessian性质
    result = analyzer.analyze_hessian(H_gn)

    # 分析阻尼效果
    analyzer.analyze_damping(H_gn)

    return J_visual_sparse, H_gn, result


def run_test_marginalization_prior():
    """测试2：边缘化先验的Hessian分析"""
    print(f"\n{'#'*80}")
    print(f"测试2: 边缘化先验 - Jacobian与Hessian分析")
    print(f"{'#'*80}")

    analyzer = JacobianAnalyzer()

    # 模拟先验Jacobian（正定矩阵的Cholesky分解转置）
    n_retained = 69  # 保留的参数数
    np.random.seed(42)

    # 生成正定矩阵
    A = np.random.randn(n_retained, n_retained)
    H_rr = A.T @ A + np.eye(n_retained)  # 确保正定

    # 分解得到Jacobian: J_prior = L.T，其中 H_rr = L @ L.T
    L = np.linalg.cholesky(H_rr)
    J_prior = L.T  # (69, 69)

    print(f"\n>>> 先验信息")
    print(f"  保留参数数: {n_retained}")
    print(f"  先验Jacobian形状: {J_prior.shape}")

    # 分析先验Jacobian
    analyzer.analyze_jacobian_structure(J_prior)

    # 分析先验Hessian（应该等于H_rr）
    H_prior = analyzer.analyze_gn_hessian(J_prior)

    # 验证：H_prior ≈ H_rr
    diff = np.linalg.norm(H_prior - H_rr)
    print(f"\n验证: ||J_prior^T @ J_prior - H_rr|| = {diff:.2e}")

    # 分析Hessian性质
    result = analyzer.analyze_hessian(H_prior)

    return J_prior, H_prior, result


def run_test_combined_optimization():
    """测试3：视觉+先验联合优化"""
    print(f"\n{'#'*80}")
    print(f"测试3: 视觉+先验联合优化 - Hessian分析")
    print(f"{'#'*80}")

    analyzer = JacobianAnalyzer()

    # 小规模示例
    n_pose = 2 * 6  # 2个pose
    n_point = 10 * 1  # 10个点
    n_params = n_pose + n_point
    n_obs = 20

    # 视觉Jacobian
    np.random.seed(42)
    J_visual = np.random.randn(n_obs * 2, n_params) * 0.1

    # 先验Jacobian（只约束前n_pose-6个参数，模拟边缘化）
    n_prior_params = n_pose - 6
    A_prior = np.random.randn(n_prior_params, n_prior_params)
    H_prior = A_prior.T @ A_prior + np.eye(n_prior_params)
    L = np.linalg.cholesky(H_prior)
    J_prior = np.zeros((n_prior_params, n_params))
    J_prior[:, :n_prior_params] = L.T

    print(f"\n>>> 联合优化问题")
    print(f"  总参数数: {n_params}")
    print(f"  视觉Jacobian: {J_visual.shape}")
    print(f"  先验Jacobian: {J_prior.shape}")

    # 组合Jacobian
    J_combined = np.vstack([J_visual, J_prior])
    print(f"  组合Jacobian: {J_combined.shape}")

    # 分析组合Hessian
    H_combined = analyzer.analyze_gn_hessian(J_combined)
    result = analyzer.analyze_hessian(H_combined)

    # 对比：视觉Hessian vs 组合Hessian
    H_visual = J_visual.T @ J_visual
    cond_visual = np.linalg.cond(H_visual)
    cond_combined = result.condition_number

    print(f"\n>>> 对比分析")
    print(f"  视觉Hessian条件数: {cond_visual:.2e}")
    print(f"  组合Hessian条件数: {cond_combined:.2e}")
    print(f"  改善倍数: {cond_visual / cond_combined:.1f}x")

    return J_combined, H_combined, result


def main():
    """主测试函数"""
    print(f"\n{'='*80}")
    print(f"Jacobian与Hessian分析 - SLAM优化测试探针代码")
    print(f"{'='*80}")
    print(f"\n本测试专注于SLAM中的实际优化问题：")
    print(f"1. Gauss-Newton近似：H = J^T @ J（不需要二阶导数）")
    print(f"2. Jacobian结构分析（稀疏性、维度）")
    print(f"3. Hessian性质分析（特征值、条件数、正定性）")
    print(f"4. 阻尼方法对条件数的影响")
    print(f"5. 边缘化先验的Hessian结构")
    print(f"6. 视觉+先验联合优化")

    # 运行测试
    print(f"\n\n运行测试...")
    run_test_bundle_adjustment()
    run_test_marginalization_prior()
    run_test_combined_optimization()

    print(f"\n{'='*80}")
    print(f"所有测试完成！")
    print(f"{'='*80}")
    print(f"\n关键知识点总结：")
    print(f"1. SLAM中使用Gauss-Newton近似：H = J^T @ J")
    print(f"2. 只需计算Jacobian（一阶导数），不需要Hessian（二阶导数）")
    print(f"3. 条件数大 → 优化困难，需要阻尼方法")
    print(f"4. 特征值分解用于分析矩阵性质（正定性、秩）")
    print(f"5. 先验通过Jacobian编码：H_prior = J_prior^T @ J_prior")
    print(f"6. 视觉+先验：H_total = H_visual + H_prior")


if __name__ == "__main__":
    main()

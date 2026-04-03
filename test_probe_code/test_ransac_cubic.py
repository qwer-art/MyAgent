"""
RANSAC (Random Sample Consensus) - 测试探针代码
使用三次多项式模型: y = ax^3 + bx^2 + c

知识点：
1. RANSAC算法原理：迭代式鲁棒估计
2. 三次多项式拟合：最小二乘法求解超定方程组
3. 鲁棒性处理：通过随机采样抵抗离群点

数学公式：
- 模型：y = ax^3 + bx^2 + c
- 矩阵形式：Y = XB，其中 X = [x^3, x^2, 1], B = [a, b, c]^T
- 最小二乘解：B = (X^T X)^(-1) X^T Y

RANSAC步骤：
1. 随机选择最小样本集（3个点拟合a,b,c）
2. 拟合模型并计算所有点的误差
3. 统计内点（误差小于阈值的点）
4. 重复迭代，保留内点最多的模型
5. 用所有内点重新拟合最佳模型

环境: my_agent (conda)
依赖: numpy, matplotlib
"""

import numpy as np
import matplotlib
# 配置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class RANSACConfig:
    """RANSAC配置参数"""
    n_iterations: int = 100        # 迭代次数
    threshold: float = 0.5         # 内点判定阈值
    min_inliers: int = 10          # 最小内点数量
    random_seed: int = 42          # 随机种子


@dataclass
class RANSACResult:
    """RANSAC结果"""
    best_params: np.ndarray        # 最佳模型参数 [a, b, c]
    best_inliers: np.ndarray       # 最佳内点索引
    best_inlier_count: int         # 最佳内点数量
    best_error: float              # 最佳模型误差
    all_errors: List[float]        # 所有迭代的误差历史


class CubicPolynomialModel:
    """
    三次多项式模型: y = ax^3 + bx^2 + c

    属性:
        params: 模型参数 [a, b, c]
    """

    def __init__(self, params: np.ndarray) -> None:
        """
        初始化模型

        Args:
            params: 模型参数 [a, b, c]
        """
        self.params = params

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测给定x的y值

        数学公式:
        y = a*x^3 + b*x^2 + c

        Args:
            x: 输入x值，shape (N,)

        Returns:
            预测的y值，shape (N,)
        """
        a, b, c = self.params
        return a * x**3 + b * x**2 + c

    def get_design_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        构建设计矩阵

        数学公式:
        X = [[x1^3, x1^2, 1],
             [x2^3, x2^2, 1],
             ...
             [xn^3, xn^2, 1]]

        Args:
            x: 输入x值，shape (N,)

        Returns:
            设计矩阵，shape (N, 3)
        """
        n = len(x)
        X = np.zeros((n, 3))
        X[:, 0] = x**3  # x^3 列
        X[:, 1] = x**2  # x^2 列
        X[:, 2] = 1     # 常数列
        return X

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray) -> 'CubicPolynomialModel':
        """
        使用最小二乘法拟合三次多项式

        数学推导:
        目标：最小化 ||Y - XB||^2
        解：B = (X^T X)^(-1) X^T Y

        其中:
        - X是设计矩阵 [x^3, x^2, 1]
        - Y是观测值向量
        - B是参数向量 [a, b, c]^T

        Args:
            x: 输入x值，shape (N,)
            y: 观测y值，shape (N,)

        Returns:
            拟合的模型对象
        """
        # 构建设计矩阵
        X = np.column_stack([x**3, x**2, np.ones_like(x)])

        print(f"    设计矩阵形状: {X.shape}")
        print(f"    条件数: {np.linalg.cond(X.T @ X):.2e}")

        # 最小二乘求解: B = (X^T X)^(-1) X^T Y
        # 使用np.linalg.solve而不是求逆，更稳定
        XT_X = X.T @ X
        XT_Y = X.T @ y

        try:
            params = np.linalg.solve(XT_X, XT_Y)
        except np.linalg.LinAlgError:
            # 矩阵奇异，使用最小二乘
            params = np.linalg.lstsq(X, y, rcond=None)[0]

        return CubicPolynomialModel(params)

    def compute_errors(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算预测误差

        Args:
            x: 输入x值
            y: 真实y值

        Returns:
            绝对误差数组
        """
        y_pred = self.predict(x)
        return np.abs(y - y_pred)


class RANSAC:
    """
    RANSAC算法实现

    核心思想：
    1. 随机选择最小样本集拟合模型
    2. 计算模型对所有数据的误差
    3. 统计内点（误差小于阈值）
    4. 迭代寻找内点最多的模型
    """

    def __init__(self, config: RANSACConfig) -> None:
        """
        初始化RANSAC

        Args:
            config: RANSAC配置参数
        """
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

    def run(self, x: np.ndarray, y: np.ndarray,
            verbose: bool = True) -> RANSACResult:
        """
        执行RANSAC算法

        Args:
            x: 输入x值，shape (N,)
            y: 观测y值，shape (N,)
            verbose: 是否打印详细信息

        Returns:
            RANSAC结果对象
        """
        n_points = len(x)
        min_samples = 3  # 三次多项式需要至少3个点

        if verbose:
            print("=" * 80)
            print("RANSAC算法 - 三次多项式拟合")
            print("=" * 80)
            print(f"数据点数: {n_points}")
            print(f"模型: y = ax^3 + bx^2 + c")
            print(f"最小样本数: {min_samples}")
            print(f"迭代次数: {self.config.n_iterations}")
            print(f"内点阈值: {self.config.threshold}")
            print("=" * 80)

        # 初始化最佳结果
        best_params = np.zeros(3)  # 三次多项式参数 [a, b, c]
        best_inliers = np.array([], dtype=int)
        best_inlier_count = 0
        best_error: float = float('inf')
        all_errors = []

        # RANSAC迭代
        for iteration in range(self.config.n_iterations):
            # 步骤1: 随机采样
            sample_indices = self.rng.choice(
                n_points, min_samples, replace=False
            )
            sample_indices.sort()

            # 步骤2: 拟合模型
            x_sample = x[sample_indices]
            y_sample = y[sample_indices]

            model = CubicPolynomialModel.fit(x_sample, y_sample)

            # 步骤3: 计算所有点的误差
            errors = model.compute_errors(x, y)

            # 步骤4: 统计内点
            inliers = np.where(errors < self.config.threshold)[0]
            inlier_count = len(inliers)

            # 步骤5: 计算模型质量（内点的平均误差）
            if inlier_count > 0:
                model_error = np.mean(errors[inliers])
            else:
                model_error = float('inf')

            all_errors.append(model_error)

            # 打印迭代信息
            if verbose and (iteration < 5 or iteration % 20 == 0):
                print(f"\n迭代 {iteration + 1}/{self.config.n_iterations}:")
                print(f"  采样点索引: {sample_indices}")
                print(f"  拟合参数: a={model.params[0]:.4f}, "
                      f"b={model.params[1]:.4f}, c={model.params[2]:.4f}")
                print(f"  内点数量: {inlier_count}/{n_points} "
                      f"({inlier_count/n_points*100:.1f}%)")
                print(f"  内点平均误差: {model_error:.4f}")

            # 更新最佳模型
            if inlier_count > best_inlier_count or \
               (inlier_count == best_inlier_count and model_error < best_error):
                best_params = model.params.copy()
                best_inliers = inliers.copy()
                best_inlier_count = inlier_count
                best_error = model_error

                if verbose:
                    print(f"  ⭐ 找到更好的模型！")

        # 步骤6: 用所有内点重新拟合最佳模型
        if len(best_inliers) >= self.config.min_inliers:
            if verbose:
                print("\n" + "=" * 80)
                print("用所有内点重新拟合最佳模型...")
                print("=" * 80)

            final_model = CubicPolynomialModel.fit(
                x[best_inliers], y[best_inliers]
            )
            best_params = final_model.params

            if verbose:
                print(f"最终参数: a={best_params[0]:.6f}, "
                      f"b={best_params[1]:.6f}, c={best_params[2]:.6f}")
                print(f"内点数量: {best_inlier_count}/{n_points}")
        else:
            print("警告: 未找到足够的内点！")

        return RANSACResult(
            best_params=best_params,
            best_inliers=best_inliers,
            best_inlier_count=best_inlier_count,
            best_error=best_error,
            all_errors=all_errors
        )


def generate_test_data(n_points: int = 100,
                       outlier_ratio: float = 0.3,
                       noise_std: float = 0.1,
                       random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成测试数据

    真实模型: y = 0.5*x^3 - 0.2*x^2 + 1.0

    Args:
        n_points: 总点数
        outlier_ratio: 离群点比例
        noise_std: 高斯噪声标准差
        random_seed: 随机种子

    Returns:
        x: x值
        y: 观测y值
        is_inlier: 是否为内点的标记
    """
    rng = np.random.RandomState(random_seed)

    # 生成x值（在-2到2之间均匀分布）
    x = rng.uniform(-2, 2, n_points)

    # 计算真实y值（使用真实参数）
    true_params = np.array([0.5, -0.2, 1.0])
    y_true = true_params[0] * x**3 + true_params[1] * x**2 + true_params[2]

    # 确定内点和离群点
    n_outliers = int(n_points * outlier_ratio)
    is_inlier = np.ones(n_points, dtype=bool)
    outlier_indices = rng.choice(n_points, n_outliers, replace=False)
    is_inlier[outlier_indices] = False

    # 添加噪声和离群点
    y = y_true.copy()

    # 内点添加高斯噪声
    y[is_inlier] += rng.normal(0, noise_std, np.sum(is_inlier))

    # 离群点添加大噪声
    y[~is_inlier] += rng.uniform(-5, 5, n_outliers)

    return x, y, is_inlier


def visualize_results(x: np.ndarray, y: np.ndarray,
                     is_inlier: np.ndarray,
                     result: RANSACResult,
                     true_params: np.ndarray) -> None:
    """
    可视化RANSAC结果

    Args:
        x: x值
        y: 观测y值
        is_inlier: 真实内点标记
        result: RANSAC结果
        true_params: 真实模型参数
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 数据和拟合结果
    ax1 = axes[0, 0]
    ax1.scatter(x[is_inlier], y[is_inlier], c='green',
                alpha=0.6, label='内点（真实）', s=30)
    ax1.scatter(x[~is_inlier], y[~is_inlier], c='red',
                alpha=0.6, label='离群点（真实）', s=30)

    # 绘制拟合曲线
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_fit = (result.best_params[0] * x_smooth**3 +
             result.best_params[1] * x_smooth**2 +
             result.best_params[2])
    ax1.plot(x_smooth, y_fit, 'b-', linewidth=2,
             label=f'RANSAC拟合\na={result.best_params[0]:.3f}, '
                   f'b={result.best_params[1]:.3f}, c={result.best_params[2]:.3f}')

    # 绘制真实曲线
    y_true = (true_params[0] * x_smooth**3 +
              true_params[1] * x_smooth**2 +
              true_params[2])
    ax1.plot(x_smooth, y_true, 'g--', linewidth=2,
             label=f'真实模型\na={true_params[0]:.3f}, '
                   f'b={true_params[1]:.3f}, c={true_params[2]:.3f}')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('RANSAC三次多项式拟合结果', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: RANSAC识别的内点
    ax2 = axes[0, 1]
    inlier_mask = np.zeros(len(x), dtype=bool)
    inlier_mask[result.best_inliers] = True

    ax2.scatter(x[inlier_mask], y[inlier_mask], c='blue',
                alpha=0.6, label='RANSAC识别的内点', s=30)
    ax2.scatter(x[~inlier_mask], y[~inlier_mask], c='orange',
                alpha=0.6, label='RANSAC识别的离群点', s=30)
    ax2.plot(x_smooth, y_fit, 'b-', linewidth=2, label='RANSAC拟合')
    ax2.plot(x_smooth, y_true, 'g--', linewidth=2, label='真实模型')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('RANSAC内点识别结果', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 误差历史
    ax3 = axes[1, 0]
    ax3.plot(result.all_errors, 'b-', linewidth=1)
    ax3.axhline(y=result.best_error, color='r',
                linestyle='--', label=f'最佳误差: {result.best_error:.4f}')
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('内点平均误差', fontsize=12)
    ax3.set_title('RANSAC误差历史', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 残差分布
    ax4 = axes[1, 1]
    model = CubicPolynomialModel(result.best_params)
    residuals = y - model.predict(x)
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.axvline(x=0.5, color='orange', linestyle=':',
                linewidth=2, label='内点阈值')
    ax4.axvline(x=-0.5, color='orange', linestyle=':', linewidth=2)

    ax4.set_xlabel('残差', fontsize=12)
    ax4.set_ylabel('频数', fontsize=12)
    ax4.set_title('残差分布', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/jerett/OpenProject/MyAgent/test_probe_code/ransac_result.png',
                dpi=150, bbox_inches='tight')
    print("\n图表已保存到: ransac_result.png")
    plt.show()


def run_test():
    """运行完整的RANSAC测试"""
    print("\n" + "=" * 80)
    print("RANSAC三次多项式拟合 - 测试探针代码")
    print("=" * 80)

    # 步骤1: 生成测试数据
    print("\n步骤1: 生成测试数据")
    print("-" * 80)
    n_points = 100
    outlier_ratio = 0.3
    print(f"数据点数: {n_points}")
    print(f"离群点比例: {outlier_ratio * 100:.1f}%")
    print(f"真实模型: y = 0.5*x^3 - 0.2*x^2 + 1.0")

    x, y, is_inlier = generate_test_data(
        n_points=n_points,
        outlier_ratio=outlier_ratio,
        noise_std=0.1,
        random_seed=42
    )

    print(f"生成数据完成:")
    print(f"  x范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  y范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  实际内点数: {np.sum(is_inlier)}/{n_points}")

    # 步骤2: 运行RANSAC
    print("\n步骤2: 运行RANSAC算法")
    print("-" * 80)

    config = RANSACConfig(
        n_iterations=100,
        threshold=0.5,
        min_inliers=10,
        random_seed=42
    )

    ransac = RANSAC(config)
    result = ransac.run(x, y, verbose=True)

    # 步骤3: 分析结果
    print("\n步骤3: 结果分析")
    print("-" * 80)
    true_params = np.array([0.5, -0.2, 1.0])
    print(f"真实参数: a={true_params[0]:.6f}, "
          f"b={true_params[1]:.6f}, c={true_params[2]:.6f}")
    print(f"估计参数: a={result.best_params[0]:.6f}, "
          f"b={result.best_params[1]:.6f}, c={result.best_params[2]:.6f}")
    print(f"参数误差: Δa={abs(result.best_params[0] - true_params[0]):.6f}, "
          f"Δb={abs(result.best_params[1] - true_params[1]):.6f}, "
          f"Δc={abs(result.best_params[2] - true_params[2]):.6f}")

    # 计算准确率和召回率
    inlier_mask = np.zeros(len(x), dtype=bool)
    inlier_mask[result.best_inliers] = True

    true_positives = np.sum(is_inlier & inlier_mask)
    false_positives = np.sum(~is_inlier & inlier_mask)
    false_negatives = np.sum(is_inlier & ~inlier_mask)

    precision = true_positives / (true_positives + false_positives) \
        if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) \
        if (true_positives + false_negatives) > 0 else 0

    print(f"\n内点检测性能:")
    print(f"  准确率 (Precision): {precision*100:.1f}%")
    print(f"  召回率 (Recall): {recall*100:.1f}%")

    # 步骤4: 可视化
    print("\n步骤4: 可视化结果")
    print("-" * 80)
    visualize_results(x, y, is_inlier, result, true_params)

    # 步骤5: 对比普通最小二乘法
    print("\n步骤5: 对比普通最小二乘法（不含RANSAC）")
    print("-" * 80)
    model_ls = CubicPolynomialModel.fit(x, y)
    print(f"普通最小二乘参数: a={model_ls.params[0]:.6f}, "
          f"b={model_ls.params[1]:.6f}, c={model_ls.params[2]:.6f}")
    print(f"参数误差: Δa={abs(model_ls.params[0] - true_params[0]):.6f}, "
          f"Δb={abs(model_ls.params[1] - true_params[1]):.6f}, "
          f"Δc={abs(model_ls.params[2] - true_params[2]):.6f}")

    # 对比误差
    ransac_error = np.mean((y - result.best_params[0] * x**3
                            - result.best_params[1] * x**2
                            - result.best_params[2])**2)
    ls_error = np.mean((y - model_ls.params[0] * x**3
                       - model_ls.params[1] * x**2
                       - model_ls.params[2])**2)
    true_error = np.mean((y - true_params[0] * x**3
                         - true_params[1] * x**2
                         - true_params[2])**2)

    print(f"\n均方误差 (MSE):")
    print(f"  真实模型: {true_error:.4f}")
    print(f"  RANSAC: {ransac_error:.4f}")
    print(f"  普通最小二乘: {ls_error:.4f}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    run_test()

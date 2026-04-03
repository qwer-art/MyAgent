"""
多项式非线性优化求解 - SLAM探针代码
单文件版本，包含所有功能和测试

环境: my_agent (conda)
依赖: numpy
"""

import numpy as np

# ============================================================
# 数据生成模块
# ============================================================
def generate_data(n_points=50, noise_std=0.1, seed=42):
    """
    生成观测数据

    真实解: x_true = 2.0, y_true = 3.0

    3个多项式方程（对每个观测点）:
      方程1: x² + y² - r1 = 0  (圆约束)
      方程2: x + y - r2 = 0     (线性约束)
      方程3: x*y - r3 = 0       (双曲线约束)

    其中 r1, r2, r3 是含噪声的观测值

    输入: None
    输出: observations (n, 3) - 50组观测值 [r1, r2, r3]
          true_solution - 真实解 (x, y)
    参数: 无（生成数据）
    """
    np.random.seed(seed)

    # 真实解
    x_true, y_true = 2.0, 3.0

    # 计算真实观测值
    r1_true = x_true**2 + y_true**2  # = 4 + 9 = 13
    r2_true = x_true + y_true         # = 5
    r3_true = x_true * y_true         # = 6

    # 生成50组含噪声的观测值
    observations = np.zeros((n_points, 3))
    observations[:, 0] = r1_true + np.random.normal(0, noise_std, n_points)  # r1
    observations[:, 1] = r2_true + np.random.normal(0, noise_std, n_points)  # r2
    observations[:, 2] = r3_true + np.random.normal(0, noise_std, n_points)  # r3

    return observations, (x_true, y_true)


# ============================================================
# 优化问题定义模块
# ============================================================
class PolynomialFittingProblem:
    """
    多项式拟合问题（3个多项式方程）

    目标: 从50组观测值求解 x, y

    3个方程（每组观测）:
      方程1: x² + y² - r1 = 0  (圆约束)
      方程2: x + y - r2 = 0     (线性约束)
      方程3: x*y - r3 = 0       (双曲线约束)

    参数: theta = [x, y] (2个参数)

    残差: 每组观测有3个残差
      r1_i = x² + y² - obs_r1_i
      r2_i = x + y - obs_r2_i
      r3_i = x*y - obs_r3_i

    总残差向量: [r1_1, r2_1, r3_1, r1_2, r2_2, r3_2, ...] (150维)

    代价: J(theta) = 0.5 * sum(r²)
    """

    def __init__(self, observations: np.ndarray) -> None:
        """
        输入: observations (n, 3) - n组观测值 [r1, r2, r3]
        输出: 无
        参数: 存储观测数据
        """
        self.observations = observations
        self.n = len(observations)  # 50组观测

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        计算残差向量

        每组观测有3个残差，总共 n*3 个残差
        顺序: [r1_1, r2_1, r3_1, r1_2, r2_2, r3_2, ...]

        输入: theta = [x, y]
        输出: residual (n*3,)
        参数: 2个参数 (x, y)
        """
        x, y = theta

        # 预计算常用项
        x_sq = x**2
        y_sq = y**2
        xy = x * y

        # 方程1的残差: x² + y² - r1
        r1 = x_sq + y_sq - self.observations[:, 0]

        # 方程2的残差: x + y - r2
        r2 = x + y - self.observations[:, 1]

        # 方程3的残差: x*y - r3
        r3 = xy - self.observations[:, 2]

        # 交错排列: [r1_1, r2_1, r3_1, r1_2, r2_2, r3_2, ...]
        residual = np.zeros(self.n * 3)
        residual[0::3] = r1  # 索引 0, 3, 6, ...
        residual[1::3] = r2  # 索引 1, 4, 7, ...
        residual[2::3] = r3  # 索引 2, 5, 8, ...

        return residual

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        计算Jacobian矩阵

        J = [
            [∂r1_1/∂x, ∂r1_1/∂y],
            [∂r2_1/∂x, ∂r2_1/∂y],
            [∂r3_1/∂x, ∂r3_1/∂y],
            [∂r1_2/∂x, ∂r1_2/∂y],
            [∂r2_2/∂x, ∂r2_2/∂y],
            [∂r3_2/∂x, ∂r3_2/∂y],
            ...
        ]  (n*3, 2)

        其中:
          ∂r1_i/∂x = 2x,  ∂r1_i/∂y = 2y
          ∂r2_i/∂x = 1,   ∂r2_i/∂y = 1
          ∂r3_i/∂x = y,   ∂r3_i/∂y = x

        输入: theta = [x, y]
        输出: Jacobian矩阵 (n*3, 2)
        参数: 无
        """
        x, y = theta

        J = np.zeros((self.n * 3, 2))

        # 对于每组观测 i（3个方程的Jacobian相同）
        for i in range(self.n):
            # 第3*i行: r1_i 对参数的偏导
            J[3*i, 0] = 2 * x   # ∂r1_i/∂x = 2x
            J[3*i, 1] = 2 * y   # ∂r1_i/∂y = 2y

            # 第3*i+1行: r2_i 对参数的偏导
            J[3*i+1, 0] = 1.0   # ∂r2_i/∂x = 1
            J[3*i+1, 1] = 1.0   # ∂r2_i/∂y = 1

            # 第3*i+2行: r3_i 对参数的偏导
            J[3*i+2, 0] = y     # ∂r3_i/∂x = y
            J[3*i+2, 1] = x     # ∂r3_i/∂y = x

        return J

    def cost(self, theta: np.ndarray) -> float:
        """
        计算代价函数值

        J(theta) = 0.5 * sum(r_i²)

        输入: theta = [x, y]
        输出: 标量代价值
        参数: 无
        """
        r = self.residual(theta)
        return 0.5 * np.sum(r**2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        计算梯度: grad = J^T @ r

        输入: theta = [x, y]
        输出: 梯度向量 (3,)
        参数: 无
        """
        J = self.jacobian(theta)
        r = self.residual(theta)
        return J.T @ r


# ============================================================
# Gauss-Newton优化器模块
# ============================================================
class GaussNewtonOptimizer:
    """
    Gauss-非线性优化器

    迭代公式:
      H * delta = -grad
      theta_new = theta_old + delta

    其中:
      H = J^T @ J (近似Hessian)
      grad = J^T @ r (梯度)
    """

    def __init__(self,
                 max_iterations: int = 100,
                 gradient_threshold: float = 1e-6,
                 step_threshold: float = 1e-6,
                 cost_change_threshold: float = 1e-10) -> None:
        """
        配置终止条件

        输入: 无
        输出: 无
        参数:
          - max_iterations: 最大迭代次数
          - gradient_threshold: 梯度范数阈值
          - step_threshold: 步长范数阈值
          - cost_change_threshold: 代价变化阈值
        """
        self.max_iterations = max_iterations
        self.gradient_threshold = gradient_threshold
        self.step_threshold = step_threshold
        self.cost_change_threshold = cost_change_threshold

    def solve(self, problem: PolynomialFittingProblem, theta_init: np.ndarray) -> np.ndarray:
        """
        执行优化

        输入: problem (优化问题), theta_init (初始参数)
        输出: theta_opt (最优参数)
        参数: 无
        """
        theta = theta_init.copy()
        cost_prev = problem.cost(theta)
        grad = problem.gradient(theta)

        print(f"\n初始参数: x={theta[0]:.6f}, y={theta[1]:.6f}")
        print(f"初始代价: {cost_prev:.6e}")
        print(f"初始梯度范数: {np.linalg.norm(grad):.6e}")
        print("-" * 80)

        for iter in range(self.max_iterations):
            # 步骤1: 计算Jacobian和残差
            J = problem.jacobian(theta)
            r = problem.residual(theta)

            # 步骤2: 构造正规方程: H * delta = -grad
            # H = J^T @ J (n_params x n_params)
            # grad = J^T @ r (n_params,)
            H = J.T @ J
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
                  f"x={theta_new[0]:10.6f}, y={theta_new[1]:10.6f} | "
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


# ============================================================
# 主测试函数
# ============================================================
def run_test():
    """
    运行完整的测试流程
    """
    print("=" * 80)
    print("多项式非线性优化求解 - SLAM探针代码")
    print("=" * 80)
    print()
    print("问题描述: 从50组观测值求解 (x, y)")
    print("  方程1: x² + y² - r1 = 0  (圆约束)")
    print("  方程2: x + y - r2 = 0     (线性约束)")
    print("  方程3: x*y - r3 = 0       (双曲线约束)")
    print()
    print("参数: theta = [x, y] (2个参数)")
    print("观测: 50组 [r1, r2, r3]，每组含3个观测值")
    print()
    print("优化方法: Gauss-Newton迭代")
    print("  H * delta = -grad")
    print("  theta_new = theta_old + delta")
    print()
    print("终止条件:")
    print("  1. 梯度阈值: ||J^T @ r|| < 1e-6")
    print("  2. 步长阈值: ||delta|| < 1e-6")
    print("  3. 代价变化阈值: |cost_new - cost_old| < 1e-10")
    print("  4. 最大迭代次数: 100")
    print("=" * 80)

    # 生成观测数据
    observations, true_solution = generate_data(n_points=50, noise_std=0.1, seed=42)
    x_true, y_true = true_solution

    print(f"\n>>> 数据生成")
    print(f"  观测组数: {len(observations)}")
    print(f"  真实解: x={x_true}, y={y_true}")
    print(f"  观测示例 (前3组):")
    for i in range(3):
        print(f"    组{i}: r1={observations[i,0]:.2f}, "
              f"r2={observations[i,1]:.2f}, r3={observations[i,2]:.2f}")

    # 创建优化问题
    problem = PolynomialFittingProblem(observations)

    # 测试不同的初始值
    test_cases = [
        ("随机初始化", np.random.randn(2)),
        ("零初始化", np.zeros(2)),
        ("远离最优值", np.array([10.0, 10.0])),
    ]

    for test_name, theta_init in test_cases:
        print(f"\n{'=' * 80}")
        print(f">>> 测试: {test_name}")
        print(f"初始值: x={theta_init[0]:.6f}, y={theta_init[1]:.6f}")
        print("=" * 80)

        # 创建优化器
        optimizer = GaussNewtonOptimizer(
            max_iterations=100,
            gradient_threshold=1e-6,
            step_threshold=1e-6,
            cost_change_threshold=1e-10
        )

        # 求解
        theta_opt = optimizer.solve(problem, theta_init)

        # 输出结果
        print("=" * 80)
        print("优化结果")
        print("=" * 80)
        x_opt, y_opt = theta_opt
        print(f"最优解: x={x_opt:.6f}, y={y_opt:.6f}")
        print(f"真实解: x={x_true:.6f}, y={y_true:.6f}")
        print(f"参数误差: x={abs(x_opt-x_true):.6e}, y={abs(y_opt-y_true):.6e}")
        print(f"最终代价: {problem.cost(theta_opt):.6e}")
        print(f"最终残差范数: {np.linalg.norm(problem.residual(theta_opt)):.6e}")

        # 验证前3组观测
        print("-" * 80)
        print("验证（前3组观测的残差）:")
        for i in range(3):
            r1_pred = x_opt**2 + y_opt**2
            r2_pred = x_opt + y_opt
            r3_pred = x_opt * y_opt
            r1_obs, r2_obs, r3_obs = observations[i]
            print(f"  组{i}: r1={r1_pred:7.4f}(观测{r1_obs:6.2f}) | "
                  f"r2={r2_pred:7.4f}(观测{r2_obs:6.2f}) | "
                  f"r3={r3_pred:7.4f}(观测{r3_obs:6.2f})")

        print("=" * 80)


if __name__ == "__main__":
    run_test()

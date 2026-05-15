# LIO-SAM 技术详解报告

> **把书读厚** - LIO-SAM框架深度技术解析
>
> 作者: Claude SLAM研究组
> 日期: 2026-03-28
>
> **论文信息**:
> - 标题: LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping
> - 作者: Tixiao Shan, Brendan Englot, Drew Meyers, Wei Wang, Carlo Ratti, Daniela Rus
> - 会议: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2020)
> - arXiv: https://arxiv.org/abs/2007.00258
> - GitHub: https://github.com/TixiaoShan/LIO-SAM

---

## 目录

- [1. 系统概述](#1-系统概述)
- [2. 整体Pipeline与数据流](#2-整体pipeline与数据流)
- [3. 数学建模与理论基础](#3-数学建模与理论基础)
- [4. 核心模块详解](#4-核心模块详解)
  - [4.1 ImageProjection模块](#41-imageprojection模块)
  - [4.2 FeatureExtraction模块](#42-featureextraction模块)
  - [4.3 IMUPreintegration模块](#43-imupreintegration模块)
  - [4.4 MapOptimization模块](#44-mapoptimization模块)
- [5. 因子图优化](#5-因子图优化)
- [6. 代码架构与工程实现](#6-代码架构与工程实现)
- [7. 性能分析](#7-性能分析)
- [8. 参考资源](#8-参考资源)

---

## 1. 系统概述

### 1.1 核心思想

LIO-SAM (**L**i**d**ar **I**nertial **O**dometry via **S**moothing and **M**apping) 是一个**紧耦合**的激光雷达-惯性里程计系统,其核心创新在于:

1. **因子图框架**: 将激光-惯性里程计问题建模为因子图优化问题
2. **多传感器融合**: 可以灵活融合IMU预积分、激光里程计、GPS、闭环检测等多种约束
3. **滑动窗口优化**: 使用局部子关键帧进行scan-to-map匹配,而非全局地图,保证实时性
4. **运动畸变校正**: 利用IMU预积分对点云进行去畸变

### 1.2 系统特性

| 特性 | 描述 |
|------|------|
| **传感器配置** | 3D激光雷达 + IMU + GPS(可选) |
| **耦合方式** | 紧耦合 (Tightly-coupled) |
| **优化框架** | GTSAM因子图 + iSAM2增量优化 |
| **实时性能** | 可达13×实时速度处理 |
| **定位精度** | 平移误差 < 0.2m (在公园数据集) |
| **应用场景** | 室内外、UGV、无人机、手持设备 |

### 1.3 坐标系定义

```
W: 世界坐标系 (World Frame)
B: 机器人本体坐标系 (Body Frame, 与IMU重合)
L: 激光雷达坐标系 (Lidar Frame)
```

**假设**: IMU坐标系与机器人本体坐标系重合

### 1.4 状态向量定义

机器人在时刻 $t$ 的状态向量定义为:

$$
\mathbf{x}_t = \begin{bmatrix}
\mathbf{R}_t \\
\mathbf{p}_t \\
\mathbf{v}_t \\
\mathbf{b}_t
\end{bmatrix} \in \mathbb{R}^{3 \times 3} \times \mathbb{R}^3 \times \mathbb{R}^3 \times \mathbb{R}^6
$$

其中:
- $\mathbf{R}_t \in SO(3)$: 旋转矩阵 (3×3)
- $\mathbf{p}_t \in \mathbb{R}^3$: 位置向量
- $\mathbf{v}_t \in \mathbb{R}^3$: 速度向量
- $\mathbf{b}_t = \begin{bmatrix} \mathbf{b}_t^a \\ \mathbf{b}_t^g \end{bmatrix} \in \mathbb{R}^6$: IMU偏差
  - $\mathbf{b}_t^a \in \mathbb{R}^3$: 加速度计偏差
  - $\mathbf{b}_t^g \in \mathbb{R}^3$: 陀螺仪偏差

**状态总维度**: 9+6 = 15维 (考虑旋转矩阵参数化时)

从坐标系 $B$ 到 $W$ 的变换 $\mathbf{T} \in SE(3)$ 表示为:

$$
\mathbf{T} = \begin{bmatrix}
\mathbf{R} & \mathbf{p} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

---

## 2. 整体Pipeline与数据流

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LIO-SAM 系统架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │  3D LiDAR│───▶│ ImageProjection  │───▶│ FeatureExtraction   │   │
│  │ (10 Hz) │    │  (畸变校正)       │    │  (特征提取)          │   │
│  └─────────┘    └──────────────────┘    └──────────┬──────────┘   │
│                                                    │                │
│                                                    ▼                │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │   IMU   │───▶│ IMUPreintegration│───▶│  MapOptimization     │   │
│  │(400 Hz) │    │  (预积分)         │    │  (因子图优化)         │   │
│  └─────────┘    └──────────────────┘    └──────────┬──────────┘   │
│                                                    │                │
│  ┌─────────┐                                      │                │
│  │   GPS   │──────────────────────────────────────┘                │
│  │ (1 Hz)  │                                                       │
│  └─────────┘                                                       │
│                                                                     │
│  输出: 优化后位姿、速度、IMU偏差、全局地图                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流详细说明

#### 阶段1: 数据采集与预处理

**输入数据**:
- **LiDAR点云**: 频率 ~10 Hz, 每帧 ~10^5-10^6 点
- **IMU测量**: 频率 ~400 Hz
  - 角速度: $\hat{\omega}_t = \omega_t + \mathbf{b}_t^g + \mathbf{n}_t^g$
  - 加速度: $\hat{\mathbf{a}}_t = \mathbf{R}_{BW}^t(\mathbf{a}_t - \mathbf{g}) + \mathbf{b}_t^a + \mathbf{n}_t^a$
- **GPS测量** (可选): 频率 ~1 Hz

#### 阶段2: 运动畸变校正 (ImageProjection)

**目的**: 消除激光雷达扫描期间机器人运动造成的点云畸变

**方法**: 利用IMU预积分得到的位姿变换,将每个点投影到扫描起始时刻坐标系

**输入**:
- 原始点云: $\mathcal{P}_{raw} = \{\mathbf{p}_i^{raw}\}_{i=1}^{N}$
- IMU预积分位姿: $\{\mathbf{T}_t\}_{t=t_0}^{t_{end}}$

**输出**:
- 去畸变点云: $\mathcal{P}_{undistorted}$

**数学模型** (详见4.1节)

#### 阶段3: 特征提取 (FeatureExtraction)

**目的**: 从点云中提取边缘特征和平面特征,减少计算量

**方法**: 基于LOAM的特征提取算法

**输入**:
- 去畸变点云: $\mathcal{P}_{undistorted}$

**输出**:
- 边缘特征集: $\mathcal{F}^e = \{\mathbf{p}^e_k\}_{k=1}^{N_e}$
- 平面特征集: $\mathcal{F}^p = \{\mathbf{p}^p_k\}_{k=1}^{N_p}$

**数学模型** (详见4.2节)

#### 阶段4: IMU预积分 (IMUPreintegration)

**目的**: 计算两个关键帧之间的相对运动,作为因子图的先验

**方法**: 流形上的IMU预积分 (基于Forster et al. 2016)

**输入**:
- 连续IMU测量: $\{\hat{\omega}_t, \hat{\mathbf{a}}_t\}_{t=t_i}^{t_j}$

**输出**:
- 预积分状态: $\{\Delta\mathbf{R}_{ij}, \Delta\mathbf{p}_{ij}, \Delta\mathbf{v}_{ij}\}$
- 雅可比矩阵: $\{\partial\Delta/\partial\mathbf{b}\}$

**数学模型** (详见4.3节)

#### 阶段5: 因子图优化 (MapOptimization)

**目的**: 融合所有约束,优化机器人轨迹

**方法**: GTSAM因子图 + iSAM2增量优化

**输入**:
- 当前帧特征: $\mathcal{F}_{i+1} = \{\mathcal{F}^e_{i+1}, \mathcal{F}^p_{i+1}\}$
- 子关键帧地图: $\mathcal{M}_i = \{\mathcal{M}^e_i, \mathcal{M}^p_i\}$
- IMU预积分因子
- GPS测量 (可选)
- 闭环约束 (可选)

**输出**:
- 优化后的状态: $\{\mathbf{x}_j\}_{j=0}^{i+1}$
- 更新的IMU偏差: $\mathbf{b}_{i+1}$

**数学模型** (详见第5节)

### 2.3 时间同步策略

```
时刻轴:
t0 ────── t1 ────── t2 ────── t3 ────── t4 ──────
│        │        │        │        │
LiDAR:   ●                 ●                 ●  (10 Hz)
IMU:     ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  (400 Hz)
GPS:     ●        ●        ●        ●        ●     (1 Hz)

关键帧选择策略:
当位姿变化超过阈值 (Δp > 1m 或 ΔR > 10°) 时插入关键帧
```

---

## 3. 数学建模与理论基础

### 3.1 IMU测量模型

#### 3.1.1 原始测量方程

IMU在时刻 $t$ 的角速度和加速度测量为:

$$
\hat{\omega}_t = \omega_t + \mathbf{b}_t^g + \mathbf{n}_t^g \quad (2)
$$

$$
\hat{\mathbf{a}}_t = \mathbf{R}_{BW}^t(\mathbf{a}_t - \mathbf{g}) + \mathbf{b}_t^a + \mathbf{n}_t^a \quad (3)
$$

其中:
- $\hat{\omega}_t \in \mathbb{R}^3$, $\hat{\mathbf{a}}_t \in \mathbb{R}^3$: 原始测量值
- $\omega_t$, $\mathbf{a}_t$: 真实角速度和加速度
- $\mathbf{b}_t^g$, $\mathbf{b}_t^a$: 陀螺仪和加速度计偏差 (随机游走)
- $\mathbf{n}_t^g$, $\mathbf{n}_t^a$: 测量噪声 (高斯白噪声)
- $\mathbf{R}_{BW}^t \in SO(3)$: 从世界系到本体系的旋转矩阵
- $\mathbf{g} \in \mathbb{R}^3$: 世界坐标系下的重力向量

**噪声模型**:
$$
\mathbf{n}_t^g \sim \mathcal{N}(0, \mathbf{\Sigma}_g), \quad \mathbf{n}_t^a \sim \mathcal{N}(0, \mathbf{\Sigma}_a)
$$

**偏差演化** (随机游走):
$$
\dot{\mathbf{b}}_t^g = \mathbf{n}_t^{b_g}, \quad \dot{\mathbf{b}}_t^a = \mathbf{n}_t^{b_a}
$$

#### 3.1.2 状态传播方程

给定时刻 $t$ 的状态 $\{\mathbf{R}_t, \mathbf{p}_t, \mathbf{v}_t, \mathbf{b}_t\}$,时刻 $t+\Delta t$ 的状态为:

$$
\mathbf{v}_{t+\Delta t} = \mathbf{v}_t + \mathbf{g}\Delta t + \mathbf{R}_t(\hat{\mathbf{a}}_t - \mathbf{b}_t^a - \mathbf{n}_t^a)\Delta t \quad (4)
$$

$$
\mathbf{p}_{t+\Delta t} = \mathbf{p}_t + \mathbf{v}_t\Delta t + \frac{1}{2}\mathbf{g}\Delta t^2 + \frac{1}{2}\mathbf{R}_t(\hat{\mathbf{a}}_t - \mathbf{b}_t^a - \mathbf{n}_t^a)\Delta t^2 \quad (5)
$$

$$
\mathbf{R}_{t+\Delta t} = \mathbf{R}_t \exp\left((\hat{\omega}_t - \mathbf{b}_t^g - \mathbf{n}_t^g)\Delta t\right) \quad (6)
$$

其中 $\exp: \mathbb{R}^3 \rightarrow SO(3)$ 是指数映射 (指数映射将李代数 $\mathfrak{so}(3)$ 映射到李群 $SO(3)$)

**假设**: 在 $\Delta t$ 内,角速度和加速度保持不变

### 3.2 IMU预积分理论

#### 3.2.1 预积分状态定义

时刻 $i$ 和 $j$ 之间的预积分状态定义为:

$$
\Delta\mathbf{R}_{ij} = \mathbf{R}_i^T\mathbf{R}_j \in SO(3) \quad (7)
$$

$$
\Delta\mathbf{p}_{ij} = \mathbf{R}_i^T(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) \in \mathbb{R}^3 \quad (8)
$$

$$
\Delta\mathbf{v}_{ij} = \mathbf{R}_i^T(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) \in \mathbb{R}^3 \quad (9)
$$

其中 $\Delta t_{ij} = t_j - t_i$

**物理意义**:
- $\Delta\mathbf{R}_{ij}$: 相对旋转 (本体系)
- $\Delta\mathbf{p}_{ij}$: 相对位移 (本体系,去除重力和初速度)
- $\Delta\mathbf{v}_{ij}$: 相对速度变化 (本体系,去除重力)

#### 3.2.2 预积分增量更新

在 $[t_k, t_{k+1}]$ 时间间隔内 ($\Delta t = t_{k+1} - t_k$):

$$
\Delta\tilde{\mathbf{R}}_{k+1} = \Delta\tilde{\mathbf{R}}_k \exp\left((\hat{\omega}_k - \mathbf{b}_i^g)\Delta t\right)
$$

$$
\Delta\tilde{\mathbf{v}}_{k+1} = \Delta\tilde{\mathbf{v}}_k + \Delta\tilde{\mathbf{R}}_k^T(\hat{\mathbf{a}}_k - \mathbf{b}_i^a)\Delta t
$$

$$
\Delta\tilde{\mathbf{p}}_{k+1} = \Delta\tilde{\mathbf{p}}_k + \Delta\tilde{\mathbf{v}}_k\Delta t + \frac{1}{2}\Delta\tilde{\mathbf{R}}_k^T(\hat{\mathbf{a}}_k - \mathbf{b}_i^a)\Delta t^2
$$

**关键优势**: 预积分仅依赖初始偏差 $\mathbf{b}_i$,不依赖旋转 $\mathbf{R}_i$,因此可以在优化迭代中复用

#### 3.2.3 偏差更新的一阶近似

当IMU偏差从 $\mathbf{b}_i$ 更新到 $\mathbf{b}_i^{\text{new}}$ 时,使用一阶近似修正预积分值:

$$
\Delta\mathbf{R}_{ij}(\mathbf{b}_i^{\text{new}}) \approx \Delta\mathbf{R}_{ij}(\mathbf{b}_i) \exp\left(\mathbf{J}_R^g \delta\mathbf{b}^g\right)
$$

$$
\Delta\mathbf{v}_{ij}(\mathbf{b}_i^{\text{new}}) \approx \Delta\mathbf{v}_{ij}(\mathbf{b}_i) + \mathbf{J}_v^g \delta\mathbf{b}^g + \mathbf{J}_v^a \delta\mathbf{b}^a
$$

$$
\Delta\mathbf{p}_{ij}(\mathbf{b}_i^{\text{new}}) \approx \Delta\mathbf{p}_{ij}(\mathbf{b}_i) + \mathbf{J}_p^g \delta\mathbf{b}^g + \mathbf{J}_p^a \delta\mathbf{b}^a
$$

其中 $\delta\mathbf{b} = \mathbf{b}_i^{\text{new}} - \mathbf{b}_i$,雅可比矩阵 $\mathbf{J}$ 在预积分过程中同时计算

### 3.3 点云畸变校正模型

#### 3.3.1 问题描述

3D激光雷达通过旋转扫描获取点云,扫描周期通常为100ms (10Hz)。在此期间,机器人的运动会导致点云畸变。

#### 3.3.2 畸变校正公式

对于扫描时刻为 $t \in [t_0, t_0 + T_{scan}]$ 的点 $\mathbf{p}_t^{raw}$,其去畸变后的坐标为:

$$
\mathbf{p}_{t_0}^{undistorted} = \mathbf{T}_{t_0}^B \mathbf{T}_t^{B^{-1}} \mathbf{p}_t^{raw}
$$

其中:
- $\mathbf{T}_t^B$: 时刻 $t$ 机器人本体到世界坐标系的变换 (来自IMU预积分)
- $\mathbf{T}_{t_0}^B$: 扫描起始时刻的变换

**简化计算**:
使用相对变换:
$$
\mathbf{p}_{t_0}^{undistorted} = \mathbf{T}_{t_0 \leftarrow t} \cdot \mathbf{p}_t^{raw}
$$

其中 $\mathbf{T}_{t_0 \leftarrow t}$ 由IMU预积分在 $[t_0, t]$ 的结果给出

### 3.4 特征提取数学模型

#### 3.4.1 点的曲率计算

对于点云中的点 $\mathbf{p}_i$,计算其局部曲率:

$$
c_i = \frac{\|\sum_{j=i-l}^{i+l} (\mathbf{p}_j - \mathbf{p}_i) \cdot \mathbf{n}_i\|}{\|\sum_{j=i-l}^{i+l} \mathbf{p}_j - \mathbf{p}_i\|}
$$

其中:
- $l$: 邻域半径 (通常取5)
- $\mathbf{n}_i$: 点 $\mathbf{p}_i$ 的法向量

**工程实现**: LIO-SAM使用简化的曲率计算 (基于距离图像投影):

$$
c_i = \frac{\|\mathbf{p}_i - \bar{\mathbf{p}}_i\|}{\|\mathbf{p}_i\|}, \quad \bar{\mathbf{p}}_i = \frac{1}{2l+1}\sum_{j=i-l}^{i+l} \mathbf{p}_j
$$

#### 3.4.2 特征分类

- **边缘特征 (Edge Features)**:
  - 条件: $c_i > c_{edge\_thresh}$ (曲率大)
  - 提取: 每个区域提取曲率最大的 $N_e$ 个点 (通常 $N_e=2$)
  - 符号: $\mathcal{F}^e = \{\mathbf{p}_k^e\}_{k=1}^{N_e}$

- **平面特征 (Planar Features)**:
  - 条件: $c_i < c_{planar\_thresh}$ (曲率小)
  - 提取: 每个区域提取曲率最小的 $N_p$ 个点 (通常 $N_p=4$)
  - 符号: $\mathcal{F}^p = \{\mathbf{p}_k^p\}_{k=1}^{N_p}$

**参数设置** (典型值):
- $c_{edge\_thresh} = 0.1$
- $c_{planar\_thresh} = 0.01$
- 邻域半径 $l = 5$

### 3.5 Scan-to-Map匹配数学模型

#### 3.5.1 点到边缘距离

对于边缘特征点 $\mathbf{p}_{i+1,k}^e$ 和边缘线上的两点 $\mathbf{p}_{i,u}^e, \mathbf{p}_{i,v}^e$:

$$
d_k^e = \frac{\|(\mathbf{p}_{i+1,k}^e - \mathbf{p}_{i,u}^e) \times (\mathbf{p}_{i+1,k}^e - \mathbf{p}_{i,v}^e)\|}{\|\mathbf{p}_{i,u}^e - \mathbf{p}_{i,v}^e\|} \quad (10)
$$

**物理意义**: 点到直线的垂直距离

#### 3.5.2 点到平面距离

对于平面特征点 $\mathbf{p}_{i+1,k}^p$ 和平面上的三点 $\mathbf{p}_{i,u}^p, \mathbf{p}_{i,v}^p, \mathbf{p}_{i,w}^p$:

$$
d_k^p = \frac{|(\mathbf{p}_{i+1,k}^p - \mathbf{p}_{i,u}^p) \cdot [(\mathbf{p}_{i,v}^p - \mathbf{p}_{i,u}^p) \times (\mathbf{p}_{i,w}^p - \mathbf{p}_{i,u}^p)]|}{\|(\mathbf{p}_{i,v}^p - \mathbf{p}_{i,u}^p) \times (\mathbf{p}_{i,w}^p - \mathbf{p}_{i,u}^p)\|} \quad (11)
$$

**物理意义**: 点到平面的垂直距离

#### 3.5.3 优化目标函数

寻找最优变换 $\mathbf{T}_{i \leftarrow i+1}$ 最小化:

$$
\min_{\mathbf{T}_{i \leftarrow i+1}} \left( \sum_{k=1}^{N_e} d_k^e + \sum_{k=1}^{N_p} d_k^p \right)
$$

**求解方法**: Gauss-Newton 或 Levenberg-Marquardt

**变量**: 6自由度变换 $\mathbf{T} \in SE(3)$

---

## 4. 核心模块详解

### 4.1 ImageProjection模块

#### 4.1.1 模块功能

**主要任务**:
1. 接收原始激光点云
2. 使用IMU数据进行运动畸变校正
3. 将点云投影到距离图像 (Range Image)
4. 发布去畸变点云到FeatureExtraction节点

#### 4.1.2 输入输出

**输入 (Input)**:
| 数据类型 | Topic名称 | 频率 | 维度 |
|---------|----------|------|------|
| 原始点云 | `/points_raw` | 10 Hz | ~10^5-10^6 点 |
| IMU测量 | `/imu_raw` | 400 Hz | 6维 (ω, a) |
| 里程计位姿 | `/odometry` | 10 Hz | 16维 (T矩阵) |

**输出 (Output)**:
| 数据类型 | Topic名称 | 频率 | 维度 |
|---------|----------|------|------|
| 去畸变点云 | `/velodyne_cloud_2` | 10 Hz | ~10^5-10^6 点 |
| IMU里程计 | `/odom` | 400 Hz | 16维 (T矩阵) |

#### 4.1.3 核心算法流程

```cpp
// 伪代码: ImageProjection主循环
void cloudHandler(const PointCloudMsg::ConstPtr& msg) {
    // 1. 获取点云时间戳
    double timeScanCur = msg->header.stamp.toSec();

    // 2. 提取一帧点云期间的IMU数据
    std::vector<IMUData> imuQueue = extractIMUData(timeScanCur - 0.1, timeScanCur + 0.1);

    // 3. 对每个点进行畸变校正
    for (int i = 0; i < msg->points.size(); ++i) {
        // 3.1 获取当前点的时间戳
        double pointTime = msg->points[i].time;

        // 3.2 查找该时刻的IMU位姿
        Pose3D pose = interpolateIMUPose(imuQueue, pointTime);

        // 3.3 将点变换到扫描起始时刻
        PointType pointUndistorted = transformPoint(msg->points[i], pose);

        cloudOut.points.push_back(pointUndistorted);
    }

    // 4. 投影到距离图像
    projectToRangeImage(cloudOut);

    // 5. 发布去畸变点云
    pubCloud.publish(cloudOut);
}
```

#### 4.1.4 运动畸变校正详细实现

**步骤1**: 插值计算每个点的位姿

对于扫描时刻为 $t$ 的点,使用IMU预积分计算其相对于扫描起始时刻 $t_0$ 的位姿:

$$
\mathbf{T}_{t_0 \leftarrow t} = \mathbf{T}_{t_0}^W \cdot (\mathbf{T}_t^W)^{-1}
$$

**步骤2**: 点坐标变换

$$
\mathbf{p}_{t_0} = \mathbf{R}_{t_0 \leftarrow t} \cdot \mathbf{p}_t + \mathbf{t}_{t_0 \leftarrow t}
$$

**代码实现细节**:

```cpp
// 点云畸变校正核心代码
void undistortPoint(PointType& point, double pointTime,
                    const IMUPreintegration& preintegrated) {
    // 计算相对时间
    double dt = pointTime - timeScanCur;

    // 获取该时刻的相对变换 (来自IMU预积分)
    Eigen::Affine3f transform = preintegrated.getTransform(dt);

    // 应用变换
    Eigen::Vector3f pt(point.x, point.y, point.z);
    pt = transform * pt;

    point.x = pt.x();
    point.y = pt.y();
    point.z = pt.z();
}
```

#### 4.1.5 距离图像投影

**目的**: 将3D点云转换为2.距离图像,便于邻域搜索

**Velodyne VLP-16 参数**:
- 垂直通道数: 16
- 水平分辨率: 1800 (每圈360°)
- 图像尺寸: 16 × 1800

**投影公式**:

$$
\text{row} = \text{findRow}(vertical\_angle)
$$

$$
\text{col} = \lfloor \frac{\text{horizontal\_angle}}{2\pi} \times 1800 \rfloor \mod 1800
$$

**数据结构**:

```cpp
struct RangeImage {
    int rows = 16;
    int cols = 1800;
    std::vector<std::vector<PointType>> image;
    std::vector<std::vector<float>> range;

    PointType& at(int row, int col) {
        return image[row][col];
    }

    bool isEmpty(int row, int col) {
        return range[row][col] == 0;
    }
};
```

### 4.2 FeatureExtraction模块

#### 4.2.1 模块功能

**主要任务**:
1. 接收去畸变点云
2. 计算每个点的曲率
3. 提取边缘特征和平面特征
4. 发布特征点云到MapOptimization节点

#### 4.2.2 输入输出

**输入**:
| 数据类型 | Topic名称 | 频率 |
|---------|----------|------|
| 去畸变点云 | `/velodyne_cloud_2` | 10 Hz |

**输出**:
| 数据类型 | Topic名称 | 频率 |
|---------|----------|------|
| 边缘特征点云 | `/laser_cloud_corner` | 10 Hz |
| 平面特征点云 | `/laser_cloud_surf` | 10 Hz |
| 完整特征点云 | `/velodyne_cloud_less_sharp` | 10 Hz |

#### 4.2.3 特征提取算法

**步骤1**: 计算点曲率

基于距离图像的局部曲率计算:

```cpp
float computeCurvature(int row, int col, const RangeImage& rangeImage) {
    // 取前后各5个点 (共10个邻域点)
    std::vector<PointType> neighbors;
    for (int i = -5; i <= 5; ++i) {
        if (i != 0 && !rangeImage.isEmpty(row, col + i)) {
            neighbors.push_back(rangeImage.at(row, col + i));
        }
    }

    // 计算曲率
    PointType center = rangeImage.at(row, col);
    PointType mean = computeMean(neighbors);

    float curvature = distance(center, mean) / distance(center, origin);

    return curvature;
}
```

**步骤2**: 特征分类与提取

```cpp
void extractFeatures(const RangeImage& rangeImage,
                    PointCloud& edgeFeatures,
                    PointCloud& planarFeatures) {
    // 将图像划分为多个区域
    int imageRows = 16;
    int imageCols = 1800;
    int regionRows = 16;
    int regionCols = 60; // 每行6个区域

    for (int r = 0; r < imageRows; ++r) {
        for (int c = 0; c < imageCols; c += regionCols) {
            // 提取当前区域的点
            std::vector<PointWithCurvature> pointsInRegion;

            for (int i = 0; i < regionCols; ++i) {
                if (!rangeImage.isEmpty(r, c + i)) {
                    PointType pt = rangeImage.at(r, c + i);
                    float curv = computeCurvature(r, c + i, rangeImage);
                    pointsInRegion.push_back({pt, curv});
                }
            }

            // 按曲率排序
            std::sort(pointsInRegion.begin(), pointsInRegion.end(),
                     [](const auto& a, const auto& b) {
                         return a.curvature > b.curvature;
                     });

            // 提取边缘特征 (曲率最大)
            for (int i = 0; i < 2 && i < pointsInRegion.size(); ++i) {
                if (pointsInRegion[i].curvature > 0.1) {
                    edgeFeatures.push_back(pointsInRegion[i].point);
                }
            }

            // 提取平面特征 (曲率最小)
            for (int i = pointsInRegion.size() - 1;
                 i >= pointsInRegion.size() - 4 && i >= 0; --i) {
                if (pointsInRegion[i].curvature < 0.01) {
                    planarFeatures.push_back(pointsInRegion[i].point);
                }
            }
        }
    }
}
```

#### 4.2.4 输出数据格式

**边缘特征点云** (`/laser_cloud_corner`):
- 点数: ~2000-5000 点/帧
- 包含字段: x, y, z, intensity, time, ring

**平面特征点云** (`/laser_cloud_surf`):
- 点数: ~10000-20000 点/帧
- 包含字段: x, y, z, intensity, time, ring

### 4.3 IMUPreintegration模块

#### 4.3.1 模块功能

**主要任务**:
1. 接收原始IMU数据和激光里程计
2. 执行IMU预积分
3. 发布高频IMU里程计 (用于点云去畸变)
4. 为因子图提供IMU预积分因子

#### 4.3.2 类结构

```cpp
class IMUPreintegration {
public:
    // 构造函数
    IMUPreintegration(ros::NodeHandle& nh);

    // 回调函数
    void odomHandler(const nav_msgs::Odometry::ConstPtr& msg);
    void imuHandler(const sensor_msgs::Imu::ConstPtr& msg);

    // 预积分
    void integrateIMU(double startTime, double endTime);

    // 获取结果
    Pose3D getPose(double time);
    Velocity3D getVelocity(double time);

private:
    // GTSAM预积分器
    gtsam::PreintegratedImuMeasurements preintegrated_;

    // 状态队列
    std::deque<State> stateQueue_;
    std::deque<IMUData> imuQueue_;

    // 当前状态
    Pose3D currentPose_;
    Velocity3D currentVelocity_;
    gtsam::imuBias::ConstantBias currentBias_;
};
```

#### 4.3.3 输入输出

**输入**:
| 数据类型 | Topic名称 | 频率 | 维度 |
|---------|----------|------|------|
| 激光里程计 | `/odometry` | 10 Hz | 16维 (T) |
| IMU原始数据 | `/imu_raw` | 400 Hz | 6维 (ω, a) |

**输出**:
| 数据类型 | Topic名称 | 频率 | 维度 |
|---------|----------|------|------|
| IMU里程计 | `/odom` | 400 Hz | 16维 (T) |

#### 4.3.4 预积分算法实现

**步骤1**: 初始化GTSAM预积分器

```cpp
void initializePreintegrator() {
    // 噪声参数
    gtsam::Vector6 noiseParams;
    noiseParams << 0.001,  // 加速度计噪声
                   0.001, 0.001,
                   0.0001, // 陀螺仪噪声
                   0.0001, 0.0001;

    gtsam::Vector6 biasParams;
    biasParams << 0.0001,  // 加速度计偏差随机游走
                   0.0001, 0.0001,
                   0.0001, // 陀螺仪偏差随机游走
                   0.0001, 0.0001;

    auto noise = gtsam::noiseModel::Diagonal::Sigmas(noiseParams);
    auto biasNoise = gtsam::noiseModel::Diagonal::Sigmas(biasParams);

    // 创建预积分器
    gtsam::PreintegrationParams params;
    params.accelerometerCovariance = noise->covariance();
    params.gyroscopeCovariance = noise->covariance();
    params.integrationCovariance = 0.0001 * I_3x3;

    preintegrated_ = gtsam::PreintegratedImuMeasurements(params, bias);
}
```

**步骤2**: 增量预积分

```cpp
void integrateIMU(double startTime, double endTime) {
    // 重置预积分器
    preintegrated_.resetIntegration();

    // 遍历IMU数据
    for (const auto& imu : imuQueue_) {
        if (imu.time < startTime) continue;
        if (imu.time > endTime) break;

        // 添加IMU测量
        preintegrated_.integrateMeasurement(
            imu.acceleration,   // a: 加速度 (m/s²)
            imu.angularRate,    // ω: 角速度 (rad/s)
            imu.dt              // Δt: 时间间隔 (s)
        );
    }

    // 获取预积分结果
    gtsam::NavState result = preintegrated_.predict(
        currentState_.navState,
        currentBias_
    );

    return result;
}
```

**步骤3**: 位姿插值 (用于点云去畸变)

```cpp
Pose3D interpolatePose(double queryTime) {
    // 找到queryTime前后的两个状态
    auto it = std::lower_bound(stateQueue_.begin(), stateQueue_.end(),
                               queryTime,
                               [](const State& s, double t) {
                                   return s.time < t;
                               });

    if (it == stateQueue_.begin()) return stateQueue_.front().pose;
    if (it == stateQueue_.end()) return stateQueue_.back().pose;

    // 线性插值
    const State& s1 = *(it - 1);
    const State& s2 = *it;

    double alpha = (queryTime - s1.time) / (s2.time - s1.time);

    // 位姿插值 (SO(3)上的球面线性插值)
    Pose3D pose;
    pose.rotation = s1.rotation.slerp(alpha, s2.rotation);
    pose.translation = (1 - alpha) * s1.translation + alpha * s2.translation;

    return pose;
}
```

#### 4.3.5 双队列机制

LIO-SAM使用两个IMU队列:

1. **预积分队列**: 用于因子图优化的预积分计算
2. **状态更新队列**: 用于高频IMU里程计发布

```cpp
// 双队列实现
class IMUPreintegration {
private:
    std::deque<IMUData> imuQueueForOptimization_;  // 预积分队列
    std::deque<IMUData> imuQueueForPublishing_;    // 发布队列

    void imuHandler(const IMUMsg::ConstPtr& msg) {
        IMUData data = extractIMUData(msg);

        // 添加到两个队列
        imuQueueForOptimization_.push_back(data);
        imuQueueForPublishing_.push_back(data);

        // 限制队列大小
        if (imuQueueForOptimization_.size() > 1000)
            imuQueueForOptimization_.pop_front();
        if (imuQueueForPublishing_.size() > 1000)
            imuQueueForPublishing_.pop_front();

        // 发布高频IMU里程计
        publishIMUOdom(data);
    }
};
```

### 4.4 MapOptimization模块

#### 4.4.1 模块功能

**主要任务**:
1. 接收特征点云
2. 构建局部子关键帧地图
3. 执行scan-to-map匹配
4. 构建并优化因子图
5. 闭环检测与优化
6. GPS融合 (可选)

#### 4.4.2 输入输出

**输入**:
| 数据类型 | Topic名称 | 频率 |
|---------|----------|------|
| 边缘特征 | `/laser_cloud_corner` | 10 Hz |
| 平面特征 | `/laser_cloud_surf` | 10 Hz |
| 完整点云 | `/velodyne_cloud_less_sharp` | 10 Hz |
| GPS数据 | `/gps` | 1 Hz (可选) |

**输出**:
| 数据类型 | Topic名称 | 频率 |
|---------|----------|------|
| 优化位姿 | `/odom` | 10 Hz |
| 全局地图 | `/cloud_map` | 1 Hz |
| 路径轨迹 | `/path` | 10 Hz |

#### 4.4.3 核心算法流程

```cpp
// MapOptimization主循环
void laserCloudInfoHandler(const FeatureCloudMsg::ConstPtr& msg) {
    // 1. 接收特征点云
    extractFeatures(msg);

    // 2. 判断是否插入关键帧
    if (!isKeyframe(currentPose_, lastKeyframePose_)) {
        return;
    }

    // 3. 构建子关键帧地图
    buildSubKeyframeMap();

    // 4. Scan-to-map匹配
    TransformT relPose = scanToMapMatching();

    // 5. 添加激光里程计因子到因子图
    addLidarOdometryFactor(relPose);

    // 6. 优化因子图
    optimizeFactorGraph();

    // 7. 闭环检测
    detectLoopClosure();

    // 8. GPS融合 (如果可用)
    if (gpsAvailable_) {
        addGPSFactor();
    }

    // 9. 发布结果
    publishResults();
}
```

#### 4.4.4 关键帧选择策略

```cpp
bool isKeyframe(const Pose& current, const Pose& lastKeyframe) {
    // 计算位姿变化
    double deltaDist = (current.position - lastKeyframe.position).norm();
    double deltaAngle = angularDistance(current.rotation, lastKeyframe.rotation);

    // 阈值判断
    const double DIST_THRESH = 1.0;    // 米
    const double ANGLE_THRESH = 10.0;  // 度

    return (deltaDist > DIST_THRESH) || (deltaAngle > ANGLE_THRESH);
}
```

#### 4.4.5 子关键帧地图构建

```cpp
void buildSubKeyframeMap() {
    // 提取最近N个关键帧 (N=25)
    int numSubKeyframes = 25;
    auto subKeyframes = recentKeyframes_.getLastN(numSubKeyframes);

    // 创建体素地图
    VoxelMap edgeVoxelMap(0.2);  // 边缘特征体素大小: 0.2m
    VoxelMap planarVoxelMap(0.4); // 平面特征体素大小: 0.4m

    // 将子关键帧变换到世界坐标系并合并
    for (const auto& kf : subKeyframes) {
        // 获取该关键帧优化后的位姿
        Pose3D pose = getOptimizedPose(kf.id);

        // 变换特征到世界系
        PointCloud edgeTransformed = transform(kf.edgeFeatures, pose);
        PointCloud planarTransformed = transform(kf.planarFeatures, pose);

        // 添加到体素地图 (自动降采样)
        edgeVoxelMap.insert(edgeTransformed);
        planarVoxelMap.insert(planarTransformed);
    }

    // 保存局部地图
    currentEdgeMap_ = edgeVoxelMap;
    currentPlanarMap_ = planarVoxelMap;
}
```

**体素地图数据结构**:

```cpp
template<typename PointType>
class VoxelMap {
private:
    float voxelSize_;
    std::unordered_map<VoxelKey, std::vector<PointType>> voxels_;

public:
    void insert(const PointCloud& cloud) {
        for (const auto& pt : cloud) {
            VoxelKey key = computeVoxelKey(pt, voxelSize_);

            // 检查体素是否已存在
            if (voxels_.find(key) == voxels_.end()) {
                voxels_[key] = {pt};
            } else {
                // 体素已存在,取平均值降采样
                voxels_[key].push_back(pt);
            }
        }
    }

    std::vector<PointType> getPointsInVoxel(const VoxelKey& key) {
        if (voxels_.find(key) != voxels_.end()) {
            return voxels_[key];
        }
        return {};
    }

    VoxelKey computeVoxelKey(const PointType& pt, float size) {
        return {
            static_cast<int>(std::floor(pt.x / size)),
            static_cast<int>(std::floor(pt.y / size)),
            static_cast<int>(std::floor(pt.z / size))
        };
    }
};
```

#### 4.4.6 Scan-to-Map匹配

```cpp
TransformT scanToMapMatching() {
    // 将当前帧特征变换到世界坐标系
    PointCloud edgeWorld = transform(currentEdgeFeatures_, currentPose_);
    PointCloud planarWorld = transform(currentPlanarFeatures_, currentPose_);

    // 构建优化问题
    ceres::Problem problem;
    double pose_array[7]; // [x, y, z, qx, qy, qz, qw]

    // 添加边缘特征约束
    for (const auto& pt : edgeWorld) {
        // 在边缘地图中搜索对应点
        auto [pt1, pt2] = findEdgeCorrespondence(pt, currentEdgeMap_);

        if (pt1 && pt2) {
            // 添加点到直线距离残差
            ceres::CostFunction* cost_function =
                EdgeFactor::Create(pt, *pt1, *pt2);
            problem.AddResidualBlock(cost_function, nullptr, pose_array);
        }
    }

    // 添加平面特征约束
    for (const auto& pt : planarWorld) {
        // 在平面地图中搜索对应点
        auto [pt1, pt2, pt3] = findPlanarCorrespondence(pt, currentPlanarMap_);

        if (pt1 && pt2 && pt3) {
            // 添加点到平面距离残差
            ceres::CostFunction* cost_function =
                PlanarFactor::Create(pt, *pt1, *pt2, *pt3);
            problem.AddResidualBlock(cost_function, nullptr, pose_array);
        }
    }

    // 求解
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 返回相对变换
    return arrayToTransform(pose_array);
}
```

**对应点搜索** (KD-Tree):

```cpp
std::tuple<PointType*, PointType*> findEdgeCorrespondence(
    const PointType& query,
    const VoxelMap& edgeMap) {

    // 1. 在KD-Tree中搜索最近邻
    std::vector<int> indices;
    std::vector<float> distances;
    edgeMap.knnSearch(query, 1, indices, distances);

    if (indices.empty()) return {nullptr, nullptr};

    // 2. 获取最近点及其邻域
    PointType& pt1 = edgeMap.points[indices[0]];
    auto neighbors = edgeMap.getNeighbors(pt1, 10);

    if (neighbors.size() < 2) return {nullptr, nullptr};

    // 3. 寻找与pt1形成直线的点
    PointType* pt2 = nullptr;
    float maxLineScore = 0;

    for (const auto& candidate : neighbors) {
        float score = evaluateLineQuality(query, pt1, candidate);
        if (score > maxLineScore) {
            maxLineScore = score;
            pt2 = &candidate;
        }
    }

    return {&pt1, pt2};
}
```

---

## 5. 因子图优化

### 5.1 因子图结构

LIO-SAM使用GTSAM库构建因子图,系统包含以下变量和因子:

#### 5.1.1 图节点 (变量)

每个关键帧对应一个状态节点:

$$
\mathbf{x}_i = \{\mathbf{R}_i, \mathbf{p}_i, \mathbf{v}_i, \mathbf{b}_i\}
$$

在GTSAM中表示为 `gtsam::NavState`:

```cpp
gtsam::NavState state(
    gtsam::Pose3(R, p),  // 位姿: Pose3(R ∈ SO(3), p ∈ R³)
    gtsam::Vector3(v)    // 速度: Vector3 ∈ R³
);

// IMU偏差单独存储
gtsam::imuBias::ConstantBias bias(b_a, b_g);  // 6维
```

#### 5.1.2 因子类型

**1. IMU预积分因子**

连接两个连续状态节点 $\mathbf{x}_i$ 和 $\mathbf{x}_j$:

$$
\mathcal{F}_{IMU}(\mathbf{x}_i, \mathbf{x}_j, \mathbf{b}_i) =
\left\|
\begin{bmatrix}
\mathbf{R}_i^T\mathbf{R}_j \odot \Delta\tilde{\mathbf{R}}_{ij}^{-1} \\
\mathbf{R}_i^T(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) - \Delta\tilde{\mathbf{p}}_{ij} \\
\mathbf{R}_i^T(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \Delta\tilde{\mathbf{v}}_{ij}
\end{bmatrix}
\right\|_{\Sigma_{IMU}}^2
$$

其中 $\odot$ 表示李群上的流形距离,$\|\cdot\|_{\Sigma}^2$ 表示马氏距离

**GTSAM实现**:

```cpp
gtsam::PreintegratedImuMeasurements preintegrated_;

// 添加IMU因子
gtsam::ImuFactor imuFactor(
    key_i,                           // 前一个状态键
    key_j,                           // 当前状态键
    key_bias,                        // IMU偏差键
    preintegrated_,                  // 预积分结果
    imuParams                        // IMU参数
);

graph.add(imuFactor);
```

**2. 激光里程计因子**

提供两个状态之间的相对位姿约束:

$$
\mathcal{F}_{LIO}(\mathbf{x}_i, \mathbf{x}_{i+1}) =
\left\|
\mathbf{T}_{i \leftarrow i+1}^{measured} -
\mathbf{T}_i^{-1} \mathbf{T}_{i+1}
\right\|_{\Sigma_{LIO}}^2
$$

其中 $\mathbf{T}_{i \leftarrow i+1}^{measured}$ 是scan-to-map匹配得到的相对变换

**GTSAM实现**:

```cpp
// 创建相对变换因子
gtsam::BetweenFactor<gtsam::Pose3> lidarFactor(
    key_i,
    key_j,
    measuredTransform,  // gtsam::Pose3
    noiseModel          // gtsam::noiseModel::Robust
);

graph.add(lidarFactor);
```

**3. GPS因子**

提供绝对位置约束 (仅位置,无姿态):

$$
\mathcal{F}_{GPS}(\mathbf{x}_i) =
\left\|
\mathbf{p}_i^{measured} - \mathbf{p}_i
\right\|_{\Sigma_{GPS}}^2
$$

**GTSAM实现**:

```cpp
// GPS先验因子 (仅约束位置)
gtsam::GPSFactor gpsFactor(
    key_i,
    gtsam::Point3(gps.x, gps.y, gps.z),  // GPS测量位置
    gpsNoiseModel
);

graph.add(gpsFactor);
```

**4. 闭环因子**

提供非连续状态之间的相对位姿约束:

$$
\mathcal{F}_{Loop}(\mathbf{x}_i, \mathbf{x}_j) =
\left\|
\mathbf{T}_{i \leftarrow j}^{loop} -
\mathbf{T}_i^{-1} \mathbf{T}_j
\right\|_{\Sigma_{Loop}}^2
$$

**GTSAM实现**:

```cpp
// 闭环检测成功后添加闭环因子
gtsam::BetweenFactor<gtsam::Pose3> loopFactor(
    key_current,    // 当前状态
    key_loop,       // 闭环匹配的历史状态
    loopTransform,  // 闭环相对变换
    loopNoiseModel  // 通常使用鲁棒核函数
);

graph.add(loopFactor);
```

### 5.2 因子图示例

```
时间: t0 ───── t1 ───── t2 ───── t3 ───── t4

状态节点:
x0 ─────── x1 ─────── x2 ─────── x3 ─────── x4

因子:
├─ IMU因子 (连续节点间)
│  x0 ──[IMU]── x1 ──[IMU]── x2 ──[IMU]── x3 ──[IMU]── x4
│
├─ 激光里程计因子 (关键帧间)
│  x0 ──[LIO]─── x2 ──[LIO]─── x4
│
├─ GPS因子 (绝对位置)
│  x1 ──[GPS]
│  x3 ──[GPS]
│
└─ 闭环因子 (非连续)
   x0 ──────────────────[Loop]─────────────── x4
```

### 5.3 优化目标函数

整个因子图的优化目标为:

$$
\min_{\{\mathbf{x}_i\}} \sum_{k} \rho\left(\mathcal{F}_k(\mathbf{x}_{i}, \mathbf{x}_{j})\right)
$$

其中:
- $\mathcal{F}_k$: 第 $k$ 个因子
- $\rho(\cdot)$: 鲁棒核函数 (如Huber核)

$$
\rho(r) = \begin{cases}
\frac{1}{2}r^2 & \text{if } |r| \leq \delta \\
\delta(|r| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

### 5.4 iSAM2增量优化

LIO-SAM使用iSAM2 (incremental Smoothing and Mapping)进行增量优化:

**优势**:
1. **增量更新**: 仅更新受新因子影响的变量
2. **Bayes树**: 高效的因子分解和变量消元
3. **实时性能**: 避免每次全图优化

**GTSAM实现**:

```cpp
// 创建iSAM2优化器
gtsam::ISAM2 isam2;

// 每次添加新因子后
void updateFactorGraph() {
    // 1. 添加新因子到图
    graph.add(imuFactor);
    graph.add(lidarFactor);

    // 2. 添加初始估计
    initialEstimate.insert(key_new, newState);

    // 3. 使用iSAM2更新 (增量优化)
    gtsam::ISAM2Result result = isam2.update(graph, initialEstimate);

    // 4. 获取优化结果
    gtsam::Values result = isam2.calculateEstimate();

    // 5. 清空图和初始估计,为下一轮准备
    graph.resize(0);
    initialEstimate.clear();
}
```

**iSAM2参数**:

```cpp
gtsam::ISAM2Params parameters;
parameters.relinearizeThreshold = 0.01;  // 重线性化阈值
parameters.relinearizeSkip = 1;          // 每次都检查重线性化
parameters.enablePartialRelinearization = true;
parameters.evaluateNonlinearError = true;  // 计算非线性误差

gtsam::ISAM2 isam2(parameters);
```

### 5.5 IMU偏差优化

IMU偏差在因子图中与状态联合优化:

```cpp
// 创建偏差变量
gtsam::imuBias::ConstantBias prevBias = ...;
gtsam::imuBias::ConstantBias currBias = ...;

// 添加偏差随机游走因子
gtsam::BetweenFactor<gtsam::imuBias::ConstantBias> biasEvolution(
    key_prev_bias,
    key_curr_bias,
    gtsam::imuBias::ConstantBias(),  // 均值: 0 (期望偏差不变)
    biasWalkNoiseModel               // 随机游走噪声
);

graph.add(biasEvolution);

// IMU因子会自动优化偏差
gtsam::ImuFactor imuFactor(
    key_prev_state,
    key_curr_state,
    key_prev_bias,  // IMU因子依赖偏差
    preintegrated_
);

// 优化后获取更新的偏差
gtsam::imuBias::ConstantBias optimizedBias =
    isam2.calculateEstimate<gtsam::imuBias::ConstantBias>(key_curr_bias);
```

---

## 6. 代码架构与工程实现

### 6.1 ROS节点架构

LIO-SAM由以下ROS节点组成:

| 节点名称 | 功能 | 输入Topic | 输出Topic |
|---------|------|-----------|-----------|
| `imageProjection` | 点云畸变校正 | `/points_raw`, `/imu_raw` | `/velodyne_cloud_2` |
| `featureExtraction` | 特征提取 | `/velodyne_cloud_2` | `/laser_cloud_corner`, `/laser_cloud_surf` |
| `imuPreintegration` | IMU预积分 | `/imu_raw`, `/odometry` | `/odom` |
| `mapOptimization` | 后端优化 | `/laser_cloud_corner`, `/laser_cloud_surf`, `/gps` | `/odom`, `/cloud_map` |

### 6.2 文件结构

```
LIO-SAM/
├── config/                                 # 配置文件
│   ├── config.yaml                         # 系统参数
│   └── lov.yaml                           # IMU外参
│
├── launch/                                 # 启动文件
│   └── run.launch                         # 系统启动
│
├── src/                                    # 源代码
│   ├── imageProjection.cpp                 # 畸变校正节点
│   ├── featureExtraction.cpp               # 特征提取节点
│   ├── imuPreintegration.cpp               # IMU预积分节点
│   ├── mapOptmization.cpp                  # 后端优化节点
│   └── transformFusion.cpp                 # 位姿融合节点
│
├── CMakeLists.txt                          # 编译配置
├── package.xml                             # ROS包配置
└── README.md                               # 说明文档
```

### 6.3 关键参数配置

**config.yaml 示例**:

```yaml
# IMU参数
imu:
  accelerometer_noise: 0.001
  gyroscope_noise: 0.0001
  integration_noise: 0.0001
  accelerometer_bias_random_walk: 0.0001
  gyroscope_bias_random_walk: 0.0001

# 激光雷达参数
lidar:
  scan_period: 0.1          # 扫描周期 (10 Hz)
  distance_threshold: 2.0   # 距离阈值

# 特征提取参数
featureExtraction:
  edge_threshold: 0.1       # 边缘特征曲率阈值
  planar_threshold: 0.01    # 平面特征曲率阈值
  neighbor_radius: 5        # 邻域半径

# 关键帧选择
keyframe:
  distance_threshold: 1.0   # 位置变化阈值 (米)
  angle_threshold: 10.0     # 角度变化阈值 (度)

# 地图参数
map:
  submap_size: 25          # 子关键帧数量
  edge_voxel_size: 0.2     # 边缘特征体素大小
  planar_voxel_size: 0.4   # 平面特征体素大小

# 闭环检测
loopClosure:
  search_radius: 15.0      # 搜索半径 (米)
  submap_radius: 12        # 子地图半径

# GPS参数
gps:
  covariance: [1.0, 0, 0,
               0, 1.0, 0,
               0, 0, 1.0]   # GPS协方差
```

### 6.4 性能优化技巧

**1. 点云降采样**

```cpp
// 使用体素网格滤波器降采样
pcl::VoxelGrid<PointType> voxelFilter;
voxelFilter.setLeafSize(0.2, 0.2, 0.2);  // 20cm体素
voxelFilter.setInputCloud(cloud);
voxelFilter.filter(downsampledCloud);
```

**2. 并行化特征提取**

```cpp
// 使用OpenMP并行计算曲率
#pragma omp parallel for num_threads(8)
for (int i = 0; i < cloudSize; ++i) {
    float curvature = computeCurvature(i, cloud);
    curvatures[i] = curvature;
}
```

**3. KD-Tree加速搜索**

```cpp
// 创建KD-Tree索引
pcl::KdTreeFLANN<PointType> kdtree;
kdtree.setInputCloud(cloud);

// K近邻搜索
std::vector<int> indices(K);
std::vector<float> distances(K);
kdtree.nearestKSearch(queryPoint, K, indices, distances);
```

**4. 内存管理**

```cpp
// 使用对象池避免频繁内存分配
class PointCloudPool {
private:
    std::vector<PointCloud::Ptr> pool_;

public:
    PointCloud::Ptr acquire() {
        if (pool_.empty()) {
            return boost::make_shared<PointCloud>();
        }
        auto cloud = pool_.back();
        pool_.pop_back();
        cloud->clear();
        return cloud;
    }

    void release(PointCloud::Ptr cloud) {
        pool_.push_back(cloud);
    }
};
```

### 6.5 调试与可视化

**可视化工具**:
1. **RViz**: 实时可视化点云、轨迹、地图
2. **PGO可视化**: 使用python-matplotlib绘制因子图
3. **日志输出**: ROS info日志记录关键信息

**RViz配置**:

```xml
<launch>
  <node pkg="rviz" type="rviz" name="rviz" args="-d $(find lio-sam)/rviz/lio-sam.rviz"/>
</launch>
```

---

## 7. 性能分析

### 7.1 实验数据集

LIO-SAM在以下数据集上进行了评估:

| 数据集 | 平台 | 轨迹长度 | 持续时间 | 场景 |
|--------|------|---------|---------|------|
| Rotation | 手持设备 | 213.9 m | - | 快速旋转 |
| Walking | 手持设备 | 801 m | - | 校园行走 |
| Campus | 手持设备 | 1437 m | - | MIT校园 |
| Park | UGV (Jackal) | 2898 m | 40 min | 森林步道 |
| Amsterdam | 船 (Duffy 21) | 19065 m | 3 hr | 阿姆斯特丹运河 |

### 7.2 精度对比

**相对平移误差 (返回起点)**:

| 数据集 | LOAM | LIOM | LIO-odom | LIO-GPS | LIO-SAM |
|--------|------|------|----------|---------|---------|
| Campus | 192.43 m | Fail | 9.44 m | 6.87 m | **0.12 m** |
| Park | 121.74 m | 34.60 m | 36.36 m | 2.93 m | **0.04 m** |
| Amsterdam | Fail | Fail | Fail | 1.21 m | **0.17 m** |

**RMSE (相对于GPS, Park数据集)**:

| 方法 | LOAM | LIOM | LIO-odom | LIO-GPS | LIO-SAM |
|------|------|------|----------|---------|---------|
| RMSE (m) | 47.31 | 28.96 | 23.96 | 1.09 | **0.96** |

### 7.3 实时性能

**单帧处理时间 (毫秒)**:

| 数据集 | LOAM | LIOM | LIO-SAM | 加速比 |
|--------|------|------|---------|--------|
| Rotation | 83.6 | Fail | **41.9** | 13× |
| Walking | 253.6 | 339.8 | **58.4** | 13× |
| Campus | 244.9 | Fail | **97.8** | 10× |
| Park | 266.4 | 245.2 | **100.5** | 9× |
| Amsterdam | Fail | Fail | **79.3** | 11× |

**压力测试** (更快播放数据):

| 数据集 | 最大播放速度 |
|--------|------------|
| Rotation | 13× |
| Walking | 13× |
| Campus | 10× |
| Park | 9× |
| Amsterdam | 11× |

**关键发现**:
- LIO-SAM在所有测试中均能实时运行 (10 Hz点云)
- 运行时间主要受特征地图密度影响,而非因子图节点数量
- Park数据集: 4,573节点,9,365因子
- Amsterdam数据集: 23,304节点,49,617因子 (但运行更快)

### 7.4 优势与局限

**优势**:
1. ✅ **紧耦合融合**: IMU预积分与激光里程计联合优化
2. ✅ **多传感器扩展**: 易于添加GPS、闭环等因子
3. ✅ **实时性能**: 滑动窗口策略保证实时性
4. ✅ **高精度**: 相对误差 < 0.2 m
5. ✅ **鲁棒性**: 在快速旋转、特征稀疏场景下表现良好

**局限**:
1. ❌ **依赖IMU质量**: 低端IMU会影响预积分精度
2. ❌ **初始化敏感**: 需要静止初始化
3. ❌ **计算资源**: CPU负载较高
4. ❌ **动态环境**: 对移动物体敏感
5. ❌ **退化场景**: 隧道、长走廊等特征单一场景

---

## 8. 参考资源

### 8.1 论文与文献

**核心论文**:
- [LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping](https://arxiv.org/abs/2007.00258) (IROS 2020)

**相关论文**:
- LOAM: [Low-drift and Real-time Lidar Odometry and Mapping](https://ieeexplore.ieee.org/document/7983050) (J. Zhang and S. Singh, Autonomous Robots 2017)
- LeGO-LOAM: [Lightweight and Ground-Optimized Lidar Odometry and Mapping](https://ieeexplore.ieee.org/document/8759369) (Shan and Englot, IROS 2018)
- IMU预积分: [On-Manifold Preintegration for Real-Time Visual-Inertial Odometry](https://ieeexplore.ieee.org/document/7459622) (Forster et al., TRO 2017)
- iSAM2: [Incremental Smoothing and Mapping Using the Bayes Tree](https://journals.sagepub.com/doi/10.1177/0278364911430419) (Kaess et al., IJRR 2012)

### 8.2 代码与资源

**开源实现**:
- [官方LIO-SAM仓库](https://github.com/TixiaoShan/LIO-SAM) (推荐)
- [LIO-SAM详细注释版](https://github.com/JokerJohn/opensource_slam_noted/tree/master/LIO-SAM-noted)
- [LIO-SAM-DetailedNote](https://github.com/lausen001/LIO-SAM-DetailedNote)

**依赖库**:
- [GTSAM: Georgia Tech Smoothing and Mapping library](https://gtsam.org/)
- [PCL: Point Cloud Library](https://pointclouds.org/)
- [Ceres Solver](http://ceres-solver.org/)
- [ROS: Robot Operating System](https://www.ros.org/)

### 8.3 技术博客 (中文)

**知乎专栏**:
- [LIO-SAM源码解析系列](https://zhuanlan.zhihu.com/p/352039509)
- [LIO-SAM算法详解](https://zhuanlan.zhihu.com/p/381739765)

**CSDN博客**:
- [LIO-SAM代码解析系列](https://blog.csdn.net/iwanderu/article/details/123167888)
- [LIO-SAM论文与代码](https://blog.csdn.net/weixin_41331879/article/details/145082086)

### 8.4 视频资源

- [LIO-SAM演示视频](https://youtu.be/A0H8CoORZJU)
- [MIT课程: SLAM与状态估计](https://www.youtube.com/watch?v=Ca7HhRnQf6I)

---

## 附录: 常见问题 (FAQ)

### Q1: LIO-SAM与LOAM的区别?

**答**:
- **LOAM**: 松耦合,IMU仅用于去畸变,无全局优化
- **LIO-SAM**: 紧耦合,IMU与激光里程计联合优化,支持多传感器融合

### Q2: 如何选择IMU?

**答**:
- 推荐使用工业级MEMS IMU (如MicroStrain 3DM-GX5-25)
- 陀螺仪零偏不稳定性: < 5 °/hr
- 加速度计零偏不稳定性: < 0.05 m/s²

### Q3: 如何调参?

**答**:
1. 首先调整IMU参数 (使用IMU厂商数据)
2. 调整关键帧选择阈值 (根据场景尺度)
3. 调整特征提取阈值 (根据点云密度)
4. 最后调整闭环检测参数

### Q4: 实际部署建议?

**答**:
1. 传感器时间同步: 硬件同步最佳
2. IMU-LiDAR外参标定: 精确标定
3. 初始化: 系统启动时保持静止5-10秒
4. 计算资源: 建议使用Intel i7或更高CPU

---

**报告结束**

本文档持续更新中,欢迎补充和指正!

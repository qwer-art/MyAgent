# Gradio 点云可视化演示

一个基于 Gradio 的交互式可视化应用，支持图像、3D点云和点云投影的可视化。

## 功能特性

1. **图像可视化** - 生成带网格和标记点的仿真图像
2. **3D点云可视化** - 使用 Plotly 实现交互式 3D 点云显示
3. **点云投影** - 将 3D 点云投影到 2D 图像平面

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

```bash
python gradio_visualization.py
```

应用将在 `http://localhost:7860` 启动。

## 使用方法

1. 调整参数：
   - 点云数量 (100-5000)
   - 图像宽度 (320-1280)
   - 图像高度 (240-960)

2. 点击"生成数据"按钮

3. 查看可视化结果：
   - 仿真图像：带渐变背景和网格的图像
   - 3D点云：可旋转、缩放的交互式点云
   - 投影结果：点云投影到图像上的效果

## 点云类型

- 🔴 **红色点**: 平面点云
- 🟢 **绿色点**: 球形点云
- 🔵 **蓝色点**: 随机散点

## 核心类说明

### DataGenerator
仿真数据生成器，提供：
- `generate_image()`: 生成仿真图像
- `generate_point_cloud()`: 生成 3D 点云
- `project_points_to_image()`: 点云投影

### Visualizer
可视化工具类，提供：
- `create_point_cloud_plot()`: 创建 Plotly 3D 点云图

## 自定义扩展

### 修改相机参数

在 `project_points_to_image()` 方法中可以自定义相机内参：

```python
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)
```

### 加载真实数据

替换 `generate_and_visualize()` 函数中的数据生成部分：

```python
# 加载真实图像
image = cv2.imread("your_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 加载真实点云
points = np.load("your_points.npy")  # (N, 3)
colors = np.load("your_colors.npy")  # (N, 3)
```

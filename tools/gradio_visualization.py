"""
Gradio 可视化演示应用
支持：图像、点云、点云投影到图像的可视化
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from typing import Tuple, Dict, Any
import sys

# 添加调试信息
print("=" * 60)
print("Gradio 点云可视化应用启动")
print(f"Python: {sys.version}")
print(f"Gradio: {gr.__version__}")
print(f"NumPy: {np.__version__}")
print("=" * 60)


class DataGenerator:
    """仿真数据生成器"""

    @staticmethod
    def generate_image(width: int = 640, height: int = 480) -> np.ndarray:
        """生成仿真图像（带网格和标记点）"""
        # 创建渐变背景
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # 添加渐变背景
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    int(255 * x / width),
                    int(255 * y / height),
                    int(128 + 127 * np.sin(x * 0.01) * np.cos(y * 0.01))
                ]

        # 添加网格线
        grid_color = (255, 255, 255)
        for i in range(0, width, 50):
            cv2.line(image, (i, 0), (i, height), grid_color, 1)
        for i in range(0, height, 50):
            cv2.line(image, (0, i), (width, i), grid_color, 1)

        # 添加一些标记点
        for _ in range(10):
            x, y = np.random.randint(50, width-50), np.random.randint(50, height-50)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    @staticmethod
    def generate_point_cloud(num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成仿真点云数据
        返回: (points, colors)
        - points: (N, 3) xyz坐标
        - colors: (N, 3) RGB颜色
        """
        # 生成多种几何形状的点云

        # 1. 平面点云
        plane_points = np.random.randn(num_points // 3, 3) * 0.5
        plane_points[:, 2] = plane_points[:, 2] * 0.1 + 2  # z轴偏移
        plane_colors = np.zeros((num_points // 3, 3))
        plane_colors[:, 0] = 255  # 红色

        # 2. 球形点云
        theta = np.random.rand(num_points // 3) * 2 * np.pi
        phi = np.random.rand(num_points // 3) * np.pi
        r = 0.5
        sphere_points = np.zeros((num_points // 3, 3))
        sphere_points[:, 0] = r * np.sin(phi) * np.cos(theta)
        sphere_points[:, 1] = r * np.sin(phi) * np.sin(theta)
        sphere_points[:, 2] = r * np.cos(phi) + 1
        sphere_colors = np.zeros((num_points // 3, 3))
        sphere_colors[:, 1] = 255  # 绿色

        # 3. 随机散点
        random_points = np.random.randn(num_points // 3, 3) * 0.8
        random_points[:, 2] += 1.5
        random_colors = np.zeros((num_points // 3, 3))
        random_colors[:, 2] = 255  # 蓝色

        # 合并所有点
        points = np.vstack([plane_points, sphere_points, random_points])
        colors = np.vstack([plane_colors, sphere_colors, random_colors])

        return points, colors

    @staticmethod
    def project_points_to_image(
        points: np.ndarray,
        colors: np.ndarray,
        image: np.ndarray,
        camera_matrix: np.ndarray = None,
        dist_coeffs: np.ndarray = None
    ) -> np.ndarray:
        """
        将点云投影到图像上

        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 点云颜色
            image: 背景图像
            camera_matrix: 相机内参矩阵 (3x3)
            dist_coeffs: 畸变系数

        Returns:
            投影后的图像
        """
        height, width = image.shape[:2]

        # 默认相机内参
        if camera_matrix is None:
            fx = fy = 500  # 焦距
            cx, cy = width // 2, height // 2  # 主点
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)

        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # 投影点到图像平面
        points_2d, _ = cv2.projectPoints(
            points.astype(np.float64),
            np.zeros(3, dtype=np.float64),  # 旋转向量
            np.zeros(3, dtype=np.float64),  # 平移向量
            camera_matrix,
            dist_coeffs
        )
        points_2d = points_2d.reshape(-1, 2)

        # 在图像上绘制投影点
        result = image.copy()
        for i, (pt, color) in enumerate(zip(points_2d, colors)):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(result, (x, y), 2, tuple(color.astype(int).tolist()), -1)

        return result


class Visualizer:
    """可视化工具类"""

    @staticmethod
    def create_point_cloud_plot(
        points: np.ndarray,
        colors: np.ndarray,
        title: str = "3D Point Cloud"
    ) -> go.Figure:
        """
        使用 Plotly 创建交互式 3D 点云图

        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) RGB颜色 (0-255)
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        # 将 RGB 颜色转换为字符串格式
        color_strings = [
            f'rgb({int(c[0])}, {int(c[1])}, {int(c[2])})'
            for c in colors
        ]

        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_strings,
                opacity=0.8
            ),
            hovertemplate='<b>Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig


def create_gradio_app():
    """创建 Gradio 应用"""

    # 初始化数据生成器
    data_gen = DataGenerator()

    def generate_and_visualize(
        num_points: int,
        image_width: int,
        image_height: int
    ) -> Tuple[np.ndarray, go.Figure, np.ndarray, str]:
        """
        生成数据并可视化

        Returns:
            (image, point_cloud_plot, projection_image, info_text)
        """
        try:
            print(f"\n{'='*60}")
            print(f"开始生成数据: 点数={num_points}, 宽={image_width}, 高={image_height}")

            # 生成仿真数据
            print("1. 生成图像...")
            image = data_gen.generate_image(image_width, image_height)
            print(f"   图像生成完成: shape={image.shape}, dtype={image.dtype}")

            print("2. 生成点云...")
            points, colors = data_gen.generate_point_cloud(num_points)
            print(f"   点云生成完成: {len(points)} 个点")

            # 创建点云可视化
            print("3. 创建3D可视化...")
            fig = Visualizer.create_point_cloud_plot(points, colors)
            print(f"   3D图创建完成")

            # 投影点云到图像
            print("4. 投影点云到图像...")
            projection = data_gen.project_points_to_image(points, colors, image)
            print(f"   投影完成")

            # 统计信息
            info = f"""
            ### 数据统计
            - **图像尺寸**: {image_width} x {image_height}
            - **点云数量**: {num_points}
            - **点云范围**:
              - X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]
              - Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]
              - Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]
            """

            print("数据生成成功！")
            print(f"{'='*60}\n")

            return image, fig, projection, info

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回空数据而不是抛出异常
            empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
            empty_fig = go.Figure()
            return empty_img, empty_fig, empty_img, f"错误: {str(e)}"

    # 创建 Gradio 界面
    with gr.Blocks(title="点云可视化演示") as app:
        gr.Markdown("# 点云可视化演示系统")
        gr.Markdown("支持图像、3D点云、点云投影到图像的可视化")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 参数设置")
                num_points = gr.Slider(
                    minimum=100,
                    maximum=5000,
                    value=1000,
                    step=100,
                    label="点云数量"
                )
                image_width = gr.Slider(
                    minimum=320,
                    maximum=1280,
                    value=640,
                    step=32,
                    label="图像宽度"
                )
                image_height = gr.Slider(
                    minimum=240,
                    maximum=960,
                    value=480,
                    step=32,
                    label="图像高度"
                )
                generate_btn = gr.Button("生成数据", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. 仿真图像")
                image_output = gr.Image(label="生成的图像")

            with gr.Column(scale=1):
                gr.Markdown("## 2. 3D 点云")
                point_cloud_output = gr.Plot(label="点云可视化")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 3. 点云投影到图像")
                projection_output = gr.Image(label="投影结果")

            with gr.Column(scale=1):
                gr.Markdown("## 数据信息")
                info_output = gr.Markdown()

        # 绑定事件
        generate_btn.click(
            fn=generate_and_visualize,
            inputs=[num_points, image_width, image_height],
            outputs=[image_output, point_cloud_output, projection_output, info_output]
        )

        # 页面加载时自动生成数据
        app.load(
            fn=lambda: generate_and_visualize(1000, 640, 480),
            outputs=[image_output, point_cloud_output, projection_output, info_output]
        )

        # 示例说明
        gr.Markdown("""
        ---
        ### 使用说明
        1. 调整左侧参数（点云数量、图像尺寸）
        2. 点击"生成数据"按钮
        3. 查看三个可视化结果：
           - **仿真图像**: 带网格和标记点的渐变图像
           - **3D点云**: 可旋转、缩放的交互式点云
           - **投影结果**: 点云投影到图像上的效果

        ### 点云说明
        - 🔴 红色点: 平面点云
        - 🟢 绿色点: 球形点云
        - 🔵 蓝色点: 随机散点
        """)

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=None,  # 自动寻找可用端口
        share=False,
        show_error=True
    )

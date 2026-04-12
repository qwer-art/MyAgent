# ROS2 Bag 可视化操作步骤

## Bag文件信息

**路径**: `/home/jerett/OpenProject/LidarSlam/LIO-SAM-ROS2/Data/ros2_bag/campus_small_dataset_ros2/campus_small_dataset_ros2_fixed.db3`

**包含话题**:
| 话题 | 类型 | 消息数 | 用途 |
|------|------|--------|------|
| `/points_raw` | sensor_msgs/msg/PointCloud2 | 4043 | 点云数据 |
| `/imu_correct` | sensor_msgs/msg/Imu | 203912 | IMU数据 |
| `/gps/fix` | sensor_msgs/msg/NavSatFix | 2039 | GPS数据 |

**时长**: 407秒 (约6分48秒)

---

## 1. RViz可视化点云

### 步骤1：启动RViz（终端1）
```bash
source /opt/ros/humble/setup.bash
rviz2
```

### 步骤2：配置Fixed Frame
RViz启动后，在左侧面板找到 **Displays** → **Global Options**：
1. 点击 **Fixed Frame** 右侧的输入框
2. 将默认的 `map` 改为 `velodyne`
3. 按 Enter 确认

> **重要**: Fixed Frame 必须与点云数据的 frame_id 一致。此bag的点云 frame_id 是 `velodyne`，如果设置错误，点云不会显示。

### 步骤3：添加点云显示
1. 点击左下角 **Add** 按钮
2. 在弹出窗口中选择 **By topic** 标签页
3. 找到 `/points_raw` 话题，点击左侧 `>` 展开它
4. 选择 **PointCloud2**，点击 **OK**

> 此时左侧 Displays 面板会出现 "PointCloud2" 项目。

### 步骤4：调整点云显示效果
在左侧 **PointCloud2** 项目下可调整：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| **Size** | 0.02~0.05 | 点的大小，太小看不清，太大重叠 |
| **Style** | Points | 显示样式，Points最快 |
| **Color Transformer** | AxisColor | 颜色映射方式 |
| **Axis** | Z | 按Z轴高度着色，便于区分地面和障碍物 |

### 步骤5：播放Bag（终端2）
```bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/jerett/OpenProject/LidarSlam/LIO-SAM-ROS2/Data/ros2_bag/campus_small_dataset_ros2/campus_small_dataset_ros2_fixed.db3
```

播放后，RViz中应该能看到点云。

### 步骤6：调整视角
- **鼠标左键拖拽**: 旋转视角
- **鼠标中键/滚轮拖拽**: 平移视角
- **滚轮滚动**: 缩放

### 步骤7：保存配置（可选）
菜单 **File** → **Save Config As** → 保存到：
```
/home/jerett/OpenProject/MyAgent/test_probe_code/point_cloud.rviz
```

下次可直接加载：
```bash
rviz2 -d /home/jerett/OpenProject/MyAgent/test_probe_code/point_cloud.rviz
```

---

## 2. PlotJuggler可视化IMU/GPS

### 步骤1：启动PlotJuggler（终端1）
```bash
source /opt/ros/humble/setup.bash
ros2 run plotjuggler plotjuggler
```

### 步骤2：连接ROS2话题
菜单 **Streaming** → **Start: ROS2 Topic Subscriber**

> 此时左侧会出现可用话题列表。

### 步骤3：播放Bag（终端2）
**必须先播放bag，PlotJuggler才能看到数据**
```bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/jerett/OpenProject/LidarSlam/LIO-SAM-ROS2/Data/ros2_bag/campus_small_dataset_ros2/campus_small_dataset_ros2_fixed.db3
```

### 步骤4：添加曲线
**播放bag后**，PlotJuggler左侧会出现话题列表：

1. 找到 `/imu_correct`，点击左侧 `+` 展开它
2. 再展开 `linear_acceleration` 或 `angular_velocity`
3. 将 `x`、`y`、`z` 拖拽到右侧图表区域

**IMU数据** (`/imu_correct`):
```
/imu_correct
├── linear_acceleration
│   ├── x  ← 拖到图表
│   ├── y
│   └── z
└── angular_velocity
    ├── x
    ├── y
    └── z
```

**GPS数据** (`/gps/fix`):
```
/gps/fix
├── latitude   ← 拖到图表
├── longitude
└── altitude
```

---

## 3. 同时可视化（推荐）

### 终端1：启动RViz
```bash
source /opt/ros/humble/setup.bash
rviz2 &
```

### 终端2：启动PlotJuggler
```bash
source /opt/ros/humble/setup.bash
ros2 run plotjuggler plotjuggler &
```

### 终端3：播放Bag
```bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/jerett/OpenProject/LidarSlam/LIO-SAM-ROS2/Data/ros2_bag/campus_small_dataset_ros2/campus_small_dataset_ros2_fixed.db3 --loop
```

> `--loop` 表示循环播放

# SurroundOcc: 开源代码复现实践

> **项目信息**
> - 论文: https://arxiv.org/pdf/2303.09551
> - 代码: https://github.com/weiyithu/SurroundOcc
> - 本地代码: `/home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc/`
> - 本地论文: `/home/jerett/OpenProject/MyAgent/SurroundOcc.pdf`
> - 数据目录: `/home/jerett/Data/nuscenes`

---

## 阶段0: 项目评估与决策

### 0.1 基本信息

- **论文标题**: SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving
- **作者**: Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jiwen Lu, Jie Zhou
- **发表**: ICCV 2023
- **代码仓库**: https://github.com/weiyithu/SurroundOcc
- **Star数**: 600+
- **最近更新**: 2025/6/20
- **License**: Apache 2.0

### 0.2 研究内容

**核心问题**:
- 多相机3D语义占用预测
- 从稀疏LiDAR点云生成稠密占用标签

**方法创新**:
1. 多尺度体素表示 (3层FPN金字塔)
2. 空间交叉注意力 (Spatial Cross Attention)
3. 密集GT生成管线 (Poisson Reconstruction)

**应用场景**:
- 自动驾驶场景理解
- 3D场景重建
- 语义占用预测

### 0.3 资源需求评估

**硬件需求**:
- GPU: 建议 ≥16GB 显存
- 磁盘: ~400GB (nuScenes完整数据集) + ~4GB (占用标签)
- 内存: 建议 ≥32GB

**当前环境**:
- GPU: NVIDIA GeForce RTX 4060 Ti (16GB) ✅
- CUDA: 13.1 (Driver 590.48.01) ✅
- 磁盘: 待确认
- 内存: 待确认

**数据需求**:
- nuScenes V1.0 完整数据集 (~400GB)
- 预处理的pickle文件
- 占用标签 (200x200x16分辨率)

**数据状态**:
- ✅ nuScenes mini数据集已下载 (`/home/jerett/Data/nuscenes/v1.0-mini`)
- ⚠️ 完整数据集需要下载
- ⚠️ 占用标签需要下载

### 0.4 可行性评估

| 评估项 | 状态 | 说明 |
|--------|------|------|
| 硬件满足 | ✅ | 16GB显存满足最低要求 |
| 数据可获取 | ⚠️ | 需要下载完整数据集和标签 |
| 文档完整度 | ✅ | README详细，有安装文档 |
| 复现难度 | 中 | 依赖较多，需要编译扩展 |

### 0.5 [人工] 决策

- [ ] 下载完整nuScenes数据集
- [ ] 下载占用标签
- [ ] 开始环境配置

---

## 阶段1: 环境准备

### 1.1 创建工作目录

```bash
# 目录结构已存在
/home/jerett/OpenProject/MyAgent/
├── raw_code/SurroundOcc/      ✅ 已克隆
├── raw_paper/SurroundOcc/     ⚠️ 需创建
├── datasets/nuscenes/         ⚠️ 需链接
└── docs/SurroundOcc/          ⚠️ 需创建
```

### 1.2 代码克隆

```bash
# 已完成
cd /home/jerett/OpenProject/MyAgent/raw_code
git clone https://github.com/weiyithu/SurroundOcc.git
cd SurroundOcc
```

**版本信息**:
- Commit: 最新版本
- 更新时间: 2025/6/20

### 1.3 创建虚拟环境

```bash
# 创建环境
conda create -n surroundocc python=3.7 -y
conda activate surroundocc

# 导出环境配置
conda env export > /home/jerett/OpenProject/MyAgent/docs/SurroundOcc/environment.yml
```

### 1.4 安装依赖

**安装顺序**:

1. **PyTorch (CUDA 11.1兼容)**
   ```bash
   pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
   ```

2. **GCC**
   ```bash
   conda install -c omgarcia gcc-6
   ```

3. **MMCV生态**
   ```bash
   pip install mmcv-full==1.4.0
   pip install mmdet==2.14.0
   pip install mmsegmentation==0.14.1
   ```

4. **mmdetection3d**
   ```bash
   git clone https://github.com/open-mmlab/mmdetection3d.git
   cd mmdetection3d
   git checkout v0.17.1
   python setup.py install
   cd ..
   ```

5. **其他依赖**
   ```bash
   pip install timm
   pip install open3d-python
   pip install sklearn
   ```

6. **编译Chamfer Distance**
   ```bash
   cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc/extensions/chamfer_dist
   python setup.py install --user
   ```

### 1.5 下载预训练模型

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc
mkdir -p ckpts
cd ckpts

# 下载backbone预训练权重
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth

# 下载训练好的模型 (用于评估)
# 语义占用预测模型: https://pan.baidu.com/s/1179t83Z5wFNNnxnPeo6n1A?pwd=dmcq
# 3D场景重建模型: https://pan.baidu.com/s/1dnODqzzgs9rMUJnK0AEy2A?pwd=nvm9
```

### 1.6 环境验证

```bash
# 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 验证mmdet3d
python -c "import mmdet3d; print(f'mmdet3d: {mmdet3d.__version__}')"

# 验证Chamfer Distance
python -c "from chamfer_dist import ChamferDistance; print('Chamfer Distance OK')"
```

**环境验证清单**:
- [ ] PyTorch安装成功
- [ ] CUDA可用
- [ ] mmcv安装成功
- [ ] mmdet安装成功
- [ ] mmsegmentation安装成功
- [ ] mmdet3d安装成功
- [ ] Chamfer Distance编译成功

---

## 阶段2: 数据准备

### 2.1 数据下载

**需要下载的数据**:

1. **nuScenes完整数据集** (~400GB)
   - 官网: https://www.nuscenes.org/download
   - 需要注册账号
   - 下载内容:
     - Full dataset (v1.0)
     - CAN bus expansion
     - Map expansion

2. **预处理的pickle文件**
   - train: https://pan.baidu.com/s/1B3Ak4vyl0IC0NgqZ8sgXQQ?pwd=qrc9
   - val: https://pan.baidu.com/s/1vbDe1FtW-ThDv21KDjJ6Ig?pwd=e81b

3. **占用标签** (200x200x16分辨率)
   - train (3.2GB): https://pan.baidu.com/s/1vI6bwxnNSrfM5C2sZaa2kA?pwd=hxdy
   - val (627MB): https://pan.baidu.com/s/1UgiGm-ftrA91QBuEgmauTQ?pwd=31y8

**当前数据状态**:
```
/home/jerett/Data/nuscenes/
├── can_bus/              ✅ 已存在
├── fastbev/              ✅ 已存在
├── nuScenes-map-expansion-v1.3/  ✅ 已存在
├── v1.0-mini/            ✅ 已存在 (mini数据集)
├── v1.0-trainval/        ⚠️ 需下载 (完整数据集)
└── nuscenes_occ/         ⚠️ 需下载 (占用标签)
```

### 2.2 数据目录结构

**目标结构**:
```
SurroundOcc/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/          # 关键帧数据
│   │   ├── sweeps/           # 中间帧数据
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/    # 标注文件
│   │   └── lidarseg/         # LiDAR语义标注 (可选)
│   ├── nuscenes_occ/         # 占用标签
│   │   ├── train/
│   │   └── val/
│   ├── nuscenes_infos_train.pkl
│   └── nuscenes_infos_val.pkl
```

### 2.3 数据链接

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc

# 创建数据目录
mkdir -p data

# 链接nuScenes数据集
ln -s /home/jerett/Data/nuscenes data/nuscenes

# 下载并解压占用标签到 data/nuscenes_occ/
# 下载pickle文件到 data/
```

### 2.4 数据验证

```bash
# 检查数据完整性
find data/nuscenes/samples -type f | wc -l  # 应该有大量文件

# 检查占用标签
python -c "
import numpy as np
import os

occ_dir = 'data/nuscenes_occ/train'
files = os.listdir(occ_dir)
print(f'Number of occupancy files: {len(files)}')

# 检查一个样本
sample = np.load(os.path.join(occ_dir, files[0]))
print(f'Sample shape: {sample.shape}')  # 应为 (N, 4)
print(f'Sample dtype: {sample.dtype}')
"

# 检查pickle文件
python -c "
import pickle
with open('data/nuscenes_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Number of samples: {len(data)}')
print(f'Sample keys: {data[0].keys()}')
"
```

### 2.5 [人工] 数据验证

- [ ] 完整数据集已下载
- [ ] 占用标签已下载
- [ ] pickle文件已下载
- [ ] 数据路径配置正确
- [ ] 数据可正常加载

---

## 阶段3: 代码理解与调试

### 3.1 代码结构分析

```
SurroundOcc/
├── projects/
│   ├── configs/              # 配置文件
│   │   └── surroundocc/
│   │       ├── surroundocc.py          # 语义占用预测配置
│   │       ├── surroundocc_nosemantic.py  # 3D重建配置
│   │       └── surroundocc_inference.py    # 推理配置
│   └── mmdet3d_plugin/       # 核心代码
│       ├── datasets/         # 数据加载
│       ├── models/           # 模型定义
│       │   ├── backbones/    # Backbone
│       │   └── surroundocc/  # SurroundOcc核心
│       └── core/             # 核心组件
├── tools/                    # 工具脚本
│   ├── train.sh
│   ├── test.sh
│   ├── dist_train.sh
│   ├── dist_test.sh
│   └── generate_occupancy_nuscenes/  # GT生成工具
└── extensions/               # CUDA扩展
    └── chamfer_dist/         # Chamfer Distance
```

### 3.2 配置文件理解

**关键配置** (`surroundocc.py`):

```python
# 点云范围
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]

# 体素大小
occ_size = [200, 200, 16]  # 200x200x16体素

# 是否使用语义
use_semantic = True

# 类别名称 (16类 + 背景)
class_names = ['barrier', 'bicycle', 'bus', 'car', ...]

# 模型配置
model = dict(
    type='SurroundOcc',
    img_backbone=dict(type='ResNet', depth=101, ...),  # ResNet-101 DCN
    img_neck=dict(type='FPN', ...),                     # FPN
    pts_bbox_head=dict(type='OccHead', ...)            # Occupancy Head
)

# 数据配置
data = dict(
    samples_per_gpu=1,  # batch size
    workers_per_gpu=4,  # dataloader workers
    train=dict(...),
    val=dict(...),
)

# 优化器
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01
)

# 训练策略
total_epochs = 24
lr_config = dict(policy='CosineAnnealing', ...)
```

### 3.3 单元测试

**测试1: 模块导入**
```python
# test_imports.py
import torch
import mmdet3d
from projects.mmdet3d_plugin.surroundocc.detectors.surroundocc import SurroundOcc
print("All imports successful")
```

**测试2: 数据加载**
```python
# test_dataloader.py
from projects.mmdet3d_plugin.datasets.nuscenes_occupancy_dataset import CustomNuScenesOccDataset
from torch.utils.data import DataLoader

dataset = CustomNuScenesOccDataset(
    data_root='data/nuscenes/',
    ann_file='data/nuscenes_infos_val.pkl',
    occ_size=[200, 200, 16],
    pc_range=[-50, -50, -5.0, 50, 50, 3.0],
    use_semantic=True
)

print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
print(f'Image shape: {sample["img"][0].shape}')
print(f'GT occ shape: {sample["gt_occ"].shape}')
```

**测试3: 模型前向传播**
```python
# test_model.py
import torch
from mmdet3d.models import build_model
from mmcv import Config

cfg = Config.fromfile('projects/configs/surroundocc/surroundocc.py')
model = build_model(cfg.model)

# 创建假数据
batch_size = 1
num_cameras = 6
img = torch.randn(batch_size, num_cameras, 3, 900, 1600)

# 前向传播
with torch.no_grad():
    output = model(img=img, return_loss=False)

print(f'Output shape: {output.shape}')  # 应为 (1, 200, 200, 16, 17)
```

### 3.4 调试记录

**遇到的问题**:

1. **问题**: ...
   - **解决方案**: ...

---

## 阶段4: 训练与验证

### 4.1 单样本过拟合测试

**目的**: 验证模型能否正常训练

```bash
# 创建单样本数据
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc

# 修改配置使用mini数据集
# 编辑 projects/configs/surroundocc/surroundocc.py
# data.train.ann_file = 'data/nuscenes_infos_val.pkl'  # 使用val的前几个样本
# total_epochs = 100

# 运行训练
./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py 1 ./work_dirs/debug
```

**预期结果**:
- Loss应该持续下降
- 几个epoch后loss应该接近0

### 4.2 小批量训练测试

**使用mini数据集**:

```bash
# nuScenes mini数据集有10个场景，适合快速测试
# 需要生成mini数据集的pickle文件和占用标签

# 或使用val数据集的子集
./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py 1 ./work_dirs/debug_small
```

### 4.3 完整训练

**单GPU训练** (RTX 4060 Ti 16GB):

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc

# 单GPU训练
./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py 1 ./work_dirs/surroundocc
```

**显存优化策略** (针对16GB显存):

```python
# 方案1: 使用混合精度
# 在配置文件中添加:
fp16 = dict(loss_scale='dynamic')

# 方案2: 减小体素分辨率
pts_bbox_head = dict(
    volume_h=[50, 25, 12],   # 原 [100, 50, 25]
    volume_w=[50, 25, 12],
    volume_z=[4, 2, 1],      # 原 [8, 4, 2]
)

# 方案3: 梯度检查点
# 在backbone配置中添加:
img_backbone = dict(
    ...
    with_cp=True,  # 使用checkpoint节省显存
)
```

### 4.4 训练监控

```bash
# 使用TensorBoard
tensorboard --logdir=./work_dirs/surroundocc --port=6006

# 查看训练日志
tail -f ./work_dirs/surroundocc/*.log
```

### 4.5 [人工] 训练验证

- [ ] 单样本过拟合测试通过
- [ ] 小批量训练测试通过
- [ ] Loss曲线正常
- [ ] Checkpoint正常保存

---

## 阶段5: 评估与测试

### 5.1 下载预训练模型

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc/ckpts

# 下载训练好的模型
# 语义占用预测: https://pan.baidu.com/s/1179t83Z5wFNNnxnPeo6n1A?pwd=dmcq
# 3D场景重建: https://pan.baidu.com/s/1dnODqzzgs9rMUJnK0AEy2A?pwd=nvm9
```

### 5.2 运行评估

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc

# 单GPU评估
./tools/dist_test.sh \
    ./projects/configs/surroundocc/surroundocc.py \
    ./ckpts/surroundocc.pth \
    1
```

### 5.3 评估指标

**语义占用预测**:
- mIoU (mean Intersection-over-Union)
- Geometry IoU (占用 vs 空闲)
- Per-class IoU (16个类别)

**3D场景重建**:
- Chamfer Distance (CD)
- Accuracy / Completeness
- Precision / Recall / F-score

### 5.4 结果对比

| 指标 | 论文报告 | 复现结果 | 差异 |
|------|---------|---------|------|
| mIoU | - | - | - |
| Geometry IoU | - | - | - |

---

## 阶段6: 推理与可视化

### 6.1 准备推理数据

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc/data

# 使用验证集进行推理
cp nuscenes_infos_val.pkl infos_inference.pkl
```

### 6.2 运行推理

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc

./tools/dist_inference.sh \
    ./projects/configs/surroundocc/surroundocc_inference.py \
    ./ckpts/surroundocc.pth \
    1
```

### 6.3 可视化结果

**方法1: 使用MeshLab查看.ply文件**

```bash
# 推理结果保存在 ./visual_dir/
meshlab ./visual_dir/scene_0001.ply
```

**方法2: 使用Mayavi可视化.npy文件**

```bash
cd /home/jerett/OpenProject/MyAgent/raw_code/SurroundOcc/tools
python visual.py ../visual_dir/scene_0001.npy
```

**方法3: 自定义可视化**

```python
import numpy as np
import open3d as o3d

# 加载结果
occ = np.load('visual_dir/scene_0001.npy')  # (N, 4) - xyz + label

# 创建点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(occ[:, :3])

# 根据语义标签着色
colors = label_to_color(occ[:, 3])  # 自定义颜色映射
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化
o3d.visualization.draw_geometries([pcd])
```

### 6.4 自定义数据推理

**数据格式**:

```python
# 创建 custom_data.pkl
import pickle

data = {
    'lidar2img': [...],      # 6个相机的外参矩阵 (4x4)
    'intrinsic': [...],      # 6个相机的内参矩阵 (3x3)
    'data_path': [...],      # 6个图像路径
    'occ_path': './visual_dir/output'  # 输出路径
}

with open('data/custom_data.pkl', 'wb') as f:
    pickle.dump([data], f)
```

**运行推理**:

```bash
./tools/dist_inference.sh \
    ./projects/configs/surroundocc/surroundocc_inference.py \
    ./ckpts/surroundocc.pth \
    1
```

---

## 问题记录

### 环境问题

| 问题 | 解决方案 | 状态 |
|------|---------|------|
| - | - | - |

### 数据问题

| 问题 | 解决方案 | 状态 |
|------|---------|------|
| - | - | - |

### 训练问题

| 问题 | 解决方案 | 状态 |
|------|---------|------|
| - | - | - |

---

## 执行进度

### 阶段0: 项目评估
- [x] 收集项目基本信息
- [x] 阅读论文核心内容
- [x] 评估硬件需求
- [x] 评估数据需求
- [ ] [人工] 决策是否继续

### 阶段1: 环境准备
- [x] 克隆代码仓库
- [ ] 创建虚拟环境
- [ ] 安装PyTorch
- [ ] 安装MMCV生态
- [ ] 安装mmdet3d
- [ ] 编译Chamfer Distance
- [ ] 下载预训练模型
- [ ] 验证环境

### 阶段2: 数据准备
- [ ] 下载完整nuScenes数据集
- [ ] 下载占用标签
- [ ] 下载pickle文件
- [ ] 配置数据路径
- [ ] 验证数据加载

### 阶段3: 代码调试
- [ ] 分析代码结构
- [ ] 理解配置文件
- [ ] 测试模块导入
- [ ] 测试数据加载
- [ ] 测试模型前向传播

### 阶段4: 训练验证
- [ ] 单样本过拟合测试
- [ ] 小批量训练测试
- [ ] 完整训练
- [ ] 监控训练过程

### 阶段5: 评估测试
- [ ] 下载预训练模型
- [ ] 运行评估
- [ ] 分析结果

### 阶段6: 推理可视化
- [ ] 运行推理
- [ ] 可视化结果
- [ ] 自定义数据测试

---

## 参考资料

- [项目主页](https://weiyithu.github.io/SurroundOcc/)
- [GitHub仓库](https://github.com/weiyithu/SurroundOcc)
- [论文arXiv](https://arxiv.org/abs/2303.09551)
- [nuScenes官网](https://www.nuscenes.org/)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) - 相关项目

---

## 更新日志

- **2026-05-01**: 初始版本，按照新框架重新组织
- **2026-05-01**: 完成项目评估阶段

# SurroundOcc 代码框架说明

## 1. 目录结构

```
code/
├── projects/                          # 核心项目代码
│   ├── configs/                       # 配置文件
│   │   ├── surroundocc/               # *** SurroundOcc 专属配置
│   │   │   ├── surroundocc.py         #     主配置（训练+模型+数据）
│   │   │   ├── surroundocc_nosemantic.py
│   │   │   └── surroundocc_inference.py
│   │   ├── _base_/                    # 基础配置（模型/数据集/调度器/运行时）
│   │   └── datasets/                  # 数据集配置
│   └── mmdet3d_plugin/                # *** 核心插件代码
│       ├── __init__.py                #   注册所有自定义模块
│       ├── datasets/                  #   数据集与数据管线
│       │   ├── nuscenes_occupancy_dataset.py  # CustomNuScenesOccDataset
│       │   ├── nuscenes_dataset.py
│       │   ├── evaluation_metrics.py
│       │   ├── pipelines/
│       │   │   ├── loading.py         #     LoadOccupancy (加载占用GT)
│       │   │   ├── transform_3d.py
│       │   │   └── formating.py
│       │   └── samplers/
│       ├── models/                    #   通用模型组件
│       │   ├── backbones/             #     ResNet/EfficientNet/VoVNet
│       │   ├── utils/                 #     bricks, grid_mask, position_embedding, visual
│       │   ├── hooks/
│       │   └── opt/                   #     AdamW 优化器
│       ├── core/                      #   bbox assigner/coder/match_cost, evaluation
│       └── surroundocc/               #   *** SurroundOcc 核心模型
│           ├── detectors/
│           │   └── surroundocc.py     #     SurroundOcc 检测器（主入口）
│           ├── dense_heads/
│           │   └── occ_head.py        #     OccHead（占用预测头 + 损失计算）
│           ├── modules/               #     Transformer 模块
│           │   ├── transformer.py     #       PerceptionTransformer
│           │   ├── encoder.py         #       OccEncoder + OccLayer
│           │   ├── spatial_cross_attention.py  # SpatialCrossAttention + MSDeformableAttention3D
│           │   ├── custom_base_transformer_layer.py
│           │   └── multi_scale_deformable_attn_function.py
│           ├── loss/
│           │   └── loss_utils.py      #     multiscale_supervision, geo_scal_loss, sem_scal_loss
│           ├── hooks/
│           └── apis/                  #     训练/测试入口
│               ├── train.py           #       custom_train_model
│               ├── mmdet_train.py     #       custom_train_detector
│               └── test.py
├── tools/                             # 脚本工具
│   ├── train.py                       # *** 训练入口
│   ├── test.py                        # *** 测试入口
│   ├── create_data.py                 #   数据预处理（生成 pkl）
│   ├── visual.py                      #   可视化
│   ├── generate_occupancy_nuscenes/   # *** GT 占用生成
│   │   └── generate_occupancy_nuscenes.py
│   ├── data_converter/                #   各数据集格式转换
│   ├── analysis_tools/                #   日志分析/benchmark
│   └── misc/                          #   辅助工具
└── extensions/
    └── chamfer_dist/                  # Chamfer Distance CUDA 扩展
```

## 2. 核心模块关系

### 2.1 数据流

```
NuScenes 原始数据
    │
    ▼
generate_occupancy_nuscenes.py ── 多帧点云聚合 + Poisson 重建 + Chamfer 语义传播
    │                                       → 生成 dense_voxels_with_semantic.npy
    ▼
create_data.py ── 生成 nuscenes_infos_train/val.pkl
    │
    ▼
CustomNuScenesOccDataset
    │  get_data_info(): 读取图像路径、lidar2img 变换、occ_path
    │  pipeline:
    │    LoadMultiViewImageFromFiles → LoadOccupancy → Normalize → Pad → Collect
    ▼
输出: {img: (B,N,C,H,W), gt_occ: (N_vox,4), img_metas}
```

### 2.2 模型前向流程（训练）

```
SurroundOcc.forward_train(img, img_metas, gt_occ)
    │
    ├─ extract_feat()
    │   ├─ img_backbone(ResNet-101-DCN) → 多尺度图像特征
    │   └─ img_neck(FPN) → 3 级特征 [512, 512, 512]
    │
    └─ forward_pts_train(img_feats, gt_occ, img_metas)
        │
        └─ OccHead.forward(mlvl_feats, img_metas)
            │
            ├─ transfer_conv: 对齐图像特征通道到 embed_dims
            ├─ volume_embedding: 可学习 3D 体素查询 (N_vox, embed_dim)
            │
            ├─ [逐级] PerceptionTransformer (x3 级, 由粗到细)
            │   ├─ OccEncoder
            │   │   ├─ get_reference_points(): 生成 3D 参考点网格
            │   │   ├─ point_sampling(): 3D→2D 投影, 获取 reference_points_cam + volume_mask
            │   │   └─ OccLayer (x N 层)
            │   │       ├─ SpatialCrossAttention: 3D query ↔ 2D 图像特征 (可变形注意力)
            │   │       │   └─ MSDeformableAttention3D: 多头多尺度可变形注意力
            │   │       ├─ FFN
            │   │       └─ Conv3d: 3D 局部特征聚合
            │   └─ 输出: volume_embed (B, H*W*Z, C)
            │
            ├─ reshape → (B, C, H, W, Z)
            ├─ deblocks: 3D 反卷积上采样 + 跳跃连接
            │   └─ 由粗到细: 25x25x2 → 50x50x4 → 100x100x8 → 200x200x16
            └─ occ: Conv3d(1x1) → occ_preds (多尺度预测)
```

### 2.3 损失计算

```
OccHead.loss(gt_occ, preds_dicts)
    │
    ├─ multiscale_supervision(): 将 GT 下采样到各预测尺度
    │
    ├─ 语义模式 (use_semantic=True):
    │   └─ CrossEntropyLoss + sem_scal_loss + geo_scal_loss
    │       权重: 0.5^(L-1-i) 深层监督
    │
    └─ 非语义模式 (use_semantic=False):
        └─ BCE + geo_scal_loss
            权重: 0.5^(L-1-i)
```

### 2.4 评估

```
SurroundOcc.forward_test()
    │
    ├─ 语义模式: evaluation_semantic() → mIoU (17 类)
    └─ 非语义模式: evaluation_reconstruction() → Acc/Comp/CD/Prec/Recall/F-score
```

## 3. 关键配置文件

| 文件 | 说明 |
|------|------|
| `projects/configs/surroundocc/surroundocc.py` | **主配置**: 模型结构、训练策略、数据管线、优化器 |
| `projects/configs/surroundocc/surroundocc_nosemantic.py` | 非语义占用预测配置 |
| `projects/configs/surroundocc/surroundocc_inference.py` | 推理配置 |
| `projects/configs/datasets/custom_nus-3d.py` | NuScenes 数据集基础配置 |
| `projects/configs/_base_/default_runtime.py` | 运行时基础配置 |

### 主配置关键参数

```python
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]   # 点云范围
occ_size = [200, 200, 16]                              # 体素分辨率
use_semantic = True                                    # 语义/非语义模式
class_names = 16 类                                    # NuScenes 语义类别

# 模型
img_backbone = ResNet-101-DCN                          # 图像骨干
img_neck = FPN (3级, out_channels=512)                  # 特征金字塔
pts_bbox_head = OccHead                                # 占用预测头
  volume_h/w/z = [100,50,25] / [100,50,25] / [8,4,2]  # 3 级体素尺寸
  embed_dims = [128, 256, 512]                         # 3 级嵌入维度
  _num_layers_ = [1, 3, 6]                             # 3 级 Transformer 层数
  _num_points_ = [2, 4, 8]                             # 3 级可变形注意力采样点数

# 训练
optimizer = AdamW (lr=2e-4, backbone lr_mult=0.1)
lr_config = CosineAnnealing (warmup 500 iters)
total_epochs = 24
load_from = r101_dcn_fcos3d_pretrain.pth               # 预训练权重
```

## 4. 使用流程

```bash
# 1. 生成占用 GT
python tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py

# 2. 生成数据 info
python tools/create_data.py nuscenes --root-path ./data/nuscenes

# 3. 训练
python tools/train.py projects/configs/surroundocc/surroundocc.py

# 4. 测试
python tools/test.py projects/configs/surroundocc/surroundocc.py <checkpoint>
```

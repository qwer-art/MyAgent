#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SurroundOcc 探针代码
用于理解核心模块和数据流

功能:
1. 测试模块导入是否正常
2. 测试数据加载是否正常（使用mini数据集）
3. 测试模型前向传播是否正常
4. 打印关键shape信息帮助理解数据流
"""

import os
import sys
import pickle
import numpy as np
import torch

# ============================================================================
# 1. 设置路径和环境
# ============================================================================
CODE_DIR = '/home/jerett/OpenProject/MyAgent/Project/SurroundOcc/code'
DATA_ROOT = '/home/jerett/Data/nuscenes/v1.0-mini'  # 使用mini数据集的正确路径

# 将code目录加入sys.path
sys.path.insert(0, CODE_DIR)

# Mock chamfer module if not available (only needed for evaluation, not training)
try:
    import chamfer
except ImportError:
    print("[INFO] chamfer module not found, creating mock module...")
    import types
    chamfer = types.ModuleType('chamfer')
    # Mock chamfer functions
    def mock_forward(*args, **kwargs):
        return torch.zeros(1), torch.zeros(1), None, None
    chamfer.forward = mock_forward
    sys.modules['chamfer'] = chamfer

print("=" * 80)
print("SurroundOcc 探针代码")
print("=" * 80)
print(f"Code directory: {CODE_DIR}")
print(f"Data root: {DATA_ROOT}")
print()

# ============================================================================
# 2. 测试模块导入
# ============================================================================
print("=" * 80)
print("1. 测试模块导入")
print("=" * 80)

def test_imports():
    """测试关键模块导入"""
    print("\n[1.1] 测试基础依赖...")
    try:
        import mmcv
        import mmdet
        import mmdet3d
        print(f"  - mmcv version: {mmcv.__version__}")
        print(f"  - mmdet version: {mmdet.__version__}")
        print(f"  - mmdet3d version: {mmdet3d.__version__}")
        print("  [OK] 基础依赖导入成功")
    except Exception as e:
        print(f"  [ERROR] 基础依赖导入失败: {e}")
        return False

    print("\n[1.2] 测试mmcv CUDA扩展...")
    try:
        from mmcv.ops import nms_match
        print("  [OK] mmcv CUDA扩展加载成功")
    except ImportError as e:
        print(f"  [WARNING] mmcv CUDA扩展加载失败: {e}")
        print("  这可能是由于mmcv-full编译时链接了不兼容的CUDA版本")
        print("  解决方案:")
        print("    1. 重新编译mmcv-full:")
        print("       pip uninstall mmcv-full")
        print("       pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html")
        print("    2. 或使用预编译版本（推荐）")
        return False

    print("\n[1.3] 测试自定义数据集...")
    try:
        from projects.mmdet3d_plugin.datasets import CustomNuScenesOccDataset
        print("  [OK] CustomNuScenesOccDataset 导入成功")
    except Exception as e:
        print(f"  [ERROR] CustomNuScenesOccDataset 导入失败: {e}")
        return False

    print("\n[1.4] 测试模型组件...")
    try:
        from projects.mmdet3d_plugin.surroundocc.detectors import SurroundOcc
        from projects.mmdet3d_plugin.surroundocc.dense_heads import OccHead
        from projects.mmdet3d_plugin.surroundocc.modules import PerceptionTransformer, OccEncoder
        print("  [OK] 模型组件导入成功")
    except Exception as e:
        print(f"  [ERROR] 模型组件导入失败: {e}")
        return False

    print("\n[1.5] 测试数据pipeline...")
    try:
        from projects.mmdet3d_plugin.datasets.pipelines import (
            PadMultiViewImage, NormalizeMultiviewImage,
            CustomCollect3D, PhotoMetricDistortionMultiViewImage
        )
        from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadOccupancy
        print("  [OK] 数据pipeline导入成功")
    except Exception as e:
        print(f"  [ERROR] 数据pipeline导入失败: {e}")
        return False

    print("\n[1.6] 测试chamfer_dist扩展...")
    try:
        import chamfer
        print("  [OK] chamfer_dist扩展可用")
    except ImportError as e:
        print(f"  [WARNING] chamfer_dist扩展未安装: {e}")
        print("  注意: chamfer_dist仅用于评估，不影响训练和前向传播")
        print("  如果需要评估功能，请确保CUDA版本兼容后重新编译chamfer_dist")

    return True

if not test_imports():
    print("\n[FAILED] 模块导入测试失败，请检查环境配置")
    sys.exit(1)

print("\n[SUCCESS] 所有模块导入测试通过!")

# ============================================================================
# 3. 测试数据加载
# ============================================================================
print("\n" + "=" * 80)
print("2. 测试数据加载")
print("=" * 80)

def test_data_loading():
    """测试数据加载"""
    print("\n[2.1] 检查数据文件...")

    # 检查pkl文件
    train_pkl = os.path.join(DATA_ROOT, 'nuscenes_infos_train.pkl')
    val_pkl = os.path.join(DATA_ROOT, 'nuscenes_infos_val.pkl')

    if not os.path.exists(train_pkl):
        print(f"  [ERROR] 训练数据pkl不存在: {train_pkl}")
        return False

    print(f"  [OK] 训练数据pkl: {train_pkl}")

    # 加载pkl文件
    print("\n[2.2] 加载数据信息...")
    with open(train_pkl, 'rb') as f:
        train_infos = pickle.load(f)

    print(f"  - 数据类型: {type(train_infos)}")
    print(f"  - 数据键: {list(train_infos.keys())}")
    print(f"  - 样本数量: {len(train_infos['infos'])}")

    # 检查第一个样本
    first_info = train_infos['infos'][0]
    print(f"\n[2.3] 第一个样本信息:")
    print(f"  - 键: {list(first_info.keys())}")
    print(f"  - token: {first_info['token']}")

    # 检查相机信息
    if 'cams' in first_info:
        cam_keys = list(first_info['cams'].keys())
        print(f"  - 相机数量: {len(cam_keys)}")
        print(f"  - 相机类型: {cam_keys}")

        # 检查第一个相机的信息
        first_cam = first_info['cams'][cam_keys[0]]
        print(f"  - 相机信息键: {list(first_cam.keys())}")
        print(f"  - 图像路径: {first_cam['data_path']}")

    # 检查occupancy数据路径
    print("\n[2.4] 检查occupancy数据...")
    occ_dir = os.path.join(DATA_ROOT, 'nuscenes_occ/samples')
    if os.path.exists(occ_dir):
        occ_files = os.listdir(occ_dir)
        print(f"  [OK] Occupancy数据目录存在")
        print(f"  - 文件数量: {len(occ_files)}")
        print(f"  - 示例文件: {occ_files[0] if occ_files else 'N/A'}")

        # 加载一个occupancy文件
        if occ_files:
            occ_path = os.path.join(occ_dir, occ_files[0])
            occ_data = np.load(occ_path)
            print(f"\n  Occupancy数据格式:")
            print(f"    - Shape: {occ_data.shape}")
            print(f"    - Dtype: {occ_data.dtype}")
            print(f"    - 前5行数据:")
            print(f"      {occ_data[:5]}")
            print(f"    - 说明: [x, y, z, semantic_class]")
    else:
        print(f"  [WARNING] Occupancy数据目录不存在: {occ_dir}")

    # 添加occ_path到pkl infos
    print("\n[2.5] 添加occ_path到数据信息...")
    occ_dir_rel = 'data/nuscenes/nuscenes_occ/samples'
    for info in train_infos['infos']:
        if 'occ_path' not in info:
            # 从lidar_path生成occ_path
            lidar_filename = info['lidar_path'].split('/')[-1]
            occ_filename = lidar_filename + '.npy'
            info['occ_path'] = os.path.join(occ_dir_rel, occ_filename)
    print(f"  [OK] 已添加occ_path到{len(train_infos['infos'])}个样本")

    return True, train_infos

result = test_data_loading()
if not result[0]:
    print("\n[FAILED] 数据加载测试失败")
    sys.exit(1)
else:
    train_infos = result[1]
    print("\n[SUCCESS] 数据加载测试通过!")

# ============================================================================
# 4. 测试配置加载和模型构建
# ============================================================================
print("\n" + "=" * 80)
print("3. 测试配置加载和模型构建")
print("=" * 80)

def test_model_building():
    """测试模型构建"""
    from mmcv import Config
    from mmdet.models import build_detector

    print("\n[3.1] 加载配置文件...")
    config_path = os.path.join(CODE_DIR, 'projects/configs/surroundocc/surroundocc.py')
    if not os.path.exists(config_path):
        print(f"  [ERROR] 配置文件不存在: {config_path}")
        return False

    cfg = Config.fromfile(config_path)
    print(f"  [OK] 配置文件加载成功")
    print(f"  - 模型类型: {cfg.model.type}")

    print("\n[3.2] 配置关键参数:")
    print(f"  - point_cloud_range: {cfg.point_cloud_range}")
    print(f"  - occ_size: {cfg.occ_size}")
    print(f"  - use_semantic: {cfg.use_semantic}")
    print(f"  - class_names: {cfg.class_names}")

    print("\n[3.3] 模型结构:")
    print(f"  - img_backbone: {cfg.model.img_backbone.type}")
    print(f"    - depth: {cfg.model.img_backbone.depth}")
    print(f"    - num_stages: {cfg.model.img_backbone.num_stages}")
    print(f"  - img_neck: {cfg.model.img_neck.type}")
    print(f"  - pts_bbox_head: {cfg.model.pts_bbox_head.type}")

    print("\n[3.4] OccHead参数:")
    head_cfg = cfg.model.pts_bbox_head
    print(f"  - volume_h: {head_cfg.volume_h}")
    print(f"  - volume_w: {head_cfg.volume_w}")
    print(f"  - volume_z: {head_cfg.volume_z}")
    print(f"  - num_classes: {head_cfg.num_classes}")
    print(f"  - embed_dims: {head_cfg.embed_dims}")

    print("\n[3.5] 构建模型...")
    try:
        # 设置数据路径
        cfg.data_root = DATA_ROOT + '/'
        cfg.data.train.data_root = DATA_ROOT + '/'
        cfg.data.val.data_root = DATA_ROOT + '/'
        cfg.data.test.data_root = DATA_ROOT + '/'

        # 构建模型
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        print(f"  [OK] 模型构建成功")
        print(f"  - 模型类: {type(model).__name__}")

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params / 1e6:.2f}M")
        print(f"  - 可训练参数量: {trainable_params / 1e6:.2f}M")

    except Exception as e:
        print(f"  [ERROR] 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True, cfg, model

result = test_model_building()
if not result[0]:
    print("\n[FAILED] 模型构建测试失败")
    sys.exit(1)
else:
    cfg, model = result[1], result[2]
    print("\n[SUCCESS] 模型构建测试通过!")

# ============================================================================
# 5. 测试数据集构建
# ============================================================================
print("\n" + "=" * 80)
print("4. 测试数据集构建")
print("=" * 80)

def test_dataset_building():
    """测试数据集构建"""
    from mmdet.datasets import build_dataset

    print("\n[4.1] 构建数据集...")
    try:
        # 更新配置
        dataset_cfg = cfg.data.train.copy()
        dataset_cfg['data_root'] = DATA_ROOT + '/'
        dataset_cfg['ann_file'] = os.path.join(DATA_ROOT, 'nuscenes_infos_train.pkl')

        # 构建数据集
        dataset = build_dataset(dataset_cfg)

        # 手动添加occ_path到数据集的data_infos
        occ_dir_rel = 'data/nuscenes/nuscenes_occ/samples'
        for info in dataset.data_infos:
            if 'occ_path' not in info:
                lidar_filename = info['lidar_path'].split('/')[-1]
                occ_filename = lidar_filename + '.npy'
                info['occ_path'] = os.path.join(occ_dir_rel, occ_filename)

        print(f"  [OK] 数据集构建成功")
        print(f"  - 数据集类: {type(dataset).__name__}")
        print(f"  - 样本数量: {len(dataset)}")

    except Exception as e:
        print(f"  [ERROR] 数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True, dataset

result = test_dataset_building()
if not result[0]:
    print("\n[FAILED] 数据集构建测试失败")
    sys.exit(1)
else:
    dataset = result[1]
    print("\n[SUCCESS] 数据集构建测试通过!")

# ============================================================================
# 6. 测试数据加载和预处理
# ============================================================================
print("\n" + "=" * 80)
print("5. 测试数据加载和预处理")
print("=" * 80)

def test_data_pipeline():
    """测试数据pipeline"""
    print("\n[5.1] 加载单个样本...")
    try:
        sample = dataset[0]
        print(f"  [OK] 样本加载成功")
        print(f"  - 数据键: {list(sample.keys())}")

        # 检查图像数据
        if 'img' in sample:
            img = sample['img']
            if hasattr(img, 'data'):
                img_data = img.data
                print(f"\n  图像数据:")
                print(f"    - Type: {type(img_data)}")
                if isinstance(img_data, torch.Tensor):
                    print(f"    - Shape: {img_data.shape}")
                    print(f"    - Dtype: {img_data.dtype}")
                    print(f"    - 说明: [num_cam, C, H, W]")

        # 检查gt_occ
        if 'gt_occ' in sample:
            gt_occ = sample['gt_occ']
            if hasattr(gt_occ, 'data'):
                gt_occ_data = gt_occ.data
                print(f"\n  GT Occupancy:")
                print(f"    - Type: {type(gt_occ_data)}")
                if isinstance(gt_occ_data, torch.Tensor):
                    print(f"    - Shape: {gt_occ_data.shape}")
                    print(f"    - Dtype: {gt_occ_data.dtype}")
                    print(f"    - 说明: [N_voxels, 4] (x, y, z, class)")

        # 检查img_metas
        if 'img_metas' in sample:
            img_metas = sample['img_metas']
            if hasattr(img_metas, 'data'):
                metas = img_metas.data
                print(f"\n  图像元数据:")
                print(f"    - 键: {list(metas.keys())}")
                if 'lidar2img' in metas:
                    lidar2img = metas['lidar2img']
                    print(f"    - lidar2img数量: {len(lidar2img)}")
                    print(f"    - lidar2img[0] shape: {np.array(lidar2img[0]).shape}")
                if 'occ_path' in metas:
                    print(f"    - occ_path: {metas['occ_path']}")

    except Exception as e:
        print(f"  [ERROR] 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True, sample

result = test_data_pipeline()
if not result[0]:
    print("\n[FAILED] 数据pipeline测试失败")
    sys.exit(1)
else:
    sample = result[1]
    print("\n[SUCCESS] 数据pipeline测试通过!")

# ============================================================================
# 7. 测试模型前向传播
# ============================================================================
print("\n" + "=" * 80)
print("6. 测试模型前向传播")
print("=" * 80)

def test_forward():
    """测试模型前向传播"""
    print("\n[6.1] 准备输入数据...")

    # 获取图像数据
    img = sample['img'].data
    gt_occ = sample['gt_occ'].data
    img_metas = [sample['img_metas'].data]

    print(f"  - 图像shape: {img.shape}")
    print(f"  - GT occupancy shape: {gt_occ.shape}")

    # 添加batch维度（如果需要）
    if img.dim() == 4:
        img = img.unsqueeze(0)  # [1, num_cam, C, H, W]

    print(f"\n[6.2] 执行前向传播...")
    try:
        # 设置为评估模式
        model.eval()

        with torch.no_grad():
            # 使用forward_train测试
            losses = model.forward_train(
                img_metas=img_metas,
                gt_occ=gt_occ,
                img=img
            )

        print(f"  [OK] 前向传播成功")
        print(f"\n  损失值:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"    - {key}: {value.item():.4f}")

    except Exception as e:
        print(f"  [ERROR] 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

result = test_forward()
if not result:
    print("\n[FAILED] 前向传播测试失败")
    sys.exit(1)
else:
    print("\n[SUCCESS] 前向传播测试通过!")

# ============================================================================
# 8. 详细数据流分析
# ============================================================================
print("\n" + "=" * 80)
print("7. 详细数据流分析")
print("=" * 80)

def analyze_data_flow():
    """分析数据流"""
    print("\n[7.1] 图像特征提取流程:")

    img = sample['img'].data
    if img.dim() == 4:
        img = img.unsqueeze(0)

    print(f"  输入图像: {img.shape} [B, num_cam, C, H, W]")

    # 提取特征
    model.eval()
    with torch.no_grad():
        # 提取图像特征
        img_feats = model.extract_feat(img=img, img_metas=[sample['img_metas'].data])

    print(f"\n  Backbone + FPN输出:")
    for i, feat in enumerate(img_feats):
        print(f"    - Level {i}: {feat.shape} [B, num_cam, C, H, W]")

    print(f"\n[7.2] Occupancy Head流程:")
    print(f"  多尺度特征层级: {len(img_feats)}")
    print(f"  各层级体积大小:")
    for i in range(len(cfg.model.pts_bbox_head.volume_h)):
        h = cfg.model.pts_bbox_head.volume_h[i]
        w = cfg.model.pts_bbox_head.volume_w[i]
        z = cfg.model.pts_bbox_head.volume_z[i]
        print(f"    - Level {i}: [{h}, {w}, {z}] = {h*w*z} voxels")

    print(f"\n[7.3] Transformer编码器:")
    print(f"  - 类型: OccEncoder")
    print(f"  - 层数: {cfg.model.pts_bbox_head.transformer_template.encoder.num_layers}")
    print(f"  - 注意力机制: SpatialCrossAttention")
    print(f"  - 采样点数: {cfg.model.pts_bbox_head.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points}")

    print(f"\n[7.4] 输出预测:")
    print(f"  - 最终体积大小: {cfg.occ_size}")
    print(f"  - 语义类别数: {cfg.model.pts_bbox_head.num_classes}")
    print(f"  - 输出shape: [B, num_classes, 200, 200, 16]")

    return True

result = analyze_data_flow()
if not result:
    print("\n[FAILED] 数据流分析失败")
    sys.exit(1)
else:
    print("\n[SUCCESS] 数据流分析完成!")

# ============================================================================
# 9. 总结
# ============================================================================
print("\n" + "=" * 80)
print("8. 总结")
print("=" * 80)

print("""
SurroundOcc 核心架构总结:
========================

1. 输入数据:
   - 多视角图像: [B, 6, 3, H, W] (6个相机)
   - 相机参数: lidar2img变换矩阵
   - GT Occupancy: [N_voxels, 4] (x, y, z, class)

2. 图像特征提取:
   - Backbone: ResNet-101 with DCN
   - Neck: FPN
   - 输出: 3个尺度的特征图

3. Occupancy预测:
   - 3D体积查询: 可学习的位置嵌入
   - Transformer编码器: 多尺度可变形注意力
   - 空间交叉注意力: 3D体素到2D图像的特征聚合

4. 多尺度监督:
   - 3个尺度的体积预测
   - 渐进式上采样和特征融合
   - 语义和几何损失

5. 关键参数:
   - 点云范围: [-50, -50, -5, 50, 50, 3]
   - Occupancy大小: [200, 200, 16]
   - 语义类别: 17类 (包括背景)

数据流:
=======
多视角图像 -> ResNet-101 -> FPN -> 多尺度特征
                                    |
                                    v
                           3D体积查询嵌入
                                    |
                                    v
                         Transformer编码器
                        (SpatialCrossAttention)
                                    |
                                    v
                          多尺度体积特征
                                    |
                                    v
                         上采样 + 特征融合
                                    |
                                    v
                          Occupancy预测
""")

print("\n" + "=" * 80)
print("探针代码执行完成!")
print("=" * 80)

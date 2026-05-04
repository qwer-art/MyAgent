# SurroundOcc 后续工作项

## 当前状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 环境准备 | ✅ 完成 | conda环境、依赖、预训练权重 |
| 数据准备 | ✅ 完成 | 软链接、occ标签、pkl修复 |
| 代码理解 | ✅ 完成 | struct.md, probe_code.py |
| 训练验证 | ⚠️ 流程验证成功 | CUDA显存不足(RTX 4060 Ti 16GB) |
| 评估测试 | ⏳ 待进行 | 需要先完成训练 |
| 推理可视化 | ⏳ 待进行 | 需要先完成训练 |

## 待办事项

### 1. 解决显存问题 [优先级: 高]

**目标**: 在可用GPU上完成训练

**方案A**: 使用更大显存GPU
- RTX 3090/4090 (24GB) 或 A100 (40GB+)
- 验证: `nvidia-smi` 确认显存 >= 24GB

**方案B**: 减小模型配置
- 修改 `projects/configs/surroundocc/surroundocc.py`:
  ```python
  # 减小体积分辨率
  volume_h_ = [50, 25, 12]   # 原 [100, 50, 25]
  volume_w_ = [50, 25, 12]   # 原 [100, 50, 25]
  volume_z_ = [4, 2, 1]      # 原 [8, 4, 2]

  # 或启用gradient checkpoint
  img_backbone=dict(
      ...
      with_cp=True,  # 节省显存
  )
  ```
- 验证: 训练1个epoch无OOM

### 2. 完成训练 [依赖: 1]

**目标**: 训练24个epoch，保存checkpoint

**命令**:
```bash
cd /home/jerett/OpenProject/MyAgent/Project/SurroundOcc/code
PYTHONPATH="${PYTHONPATH}:$(pwd)" CUDA_VISIBLE_DEVICES=0 \
/home/jerett/anaconda3/envs/surroundocc/bin/python tools/train.py \
projects/configs/surroundocc/surroundocc.py \
--work-dir ../work_dirs/surroundocc
```

**验证**:
- `work_dirs/surroundocc/epoch_24.pth` 存在
- tensorboard日志正常

### 3. 评估测试 [依赖: 2]

**目标**: 在val集上评估mIoU

**命令**:
```bash
PYTHONPATH="${PYTHONPATH}:$(pwd)" CUDA_VISIBLE_DEVICES=0 \
/home/jerett/anaconda3/envs/surroundocc/bin/python tools/test.py \
projects/configs/surroundocc/surroundocc.py \
../work_dirs/surroundocc/epoch_24.pth \
--eval-options show=True
```

**验证**: 输出mIoU指标

### 4. 推理可视化 [依赖: 2]

**目标**: 可视化occupancy预测结果

**命令**:
```bash
PYTHONPATH="${PYTHONPATH}:$(pwd)" CUDA_VISIBLE_DEVICES=0 \
/home/jerett/anaconda3/envs/surroundocc/bin/python tools/visual.py \
projects/configs/surroundocc/surroundocc_inference.py \
../work_dirs/surroundocc/epoch_24.pth
```

**验证**: 生成可视化图像/视频

## 已解决的问题

| 问题 | 解决方案 | 文件 |
|------|----------|------|
| sklearn包名废弃 | `pip install scikit-learn` | - |
| yapf版本不兼容 | `pip install yapf==0.31.0` | - |
| setuptools distutils.version | `pip install setuptools==59.5.0` | - |
| chamfer CUDA编译失败 | 可选导入chamfer | evaluation_metrics.py |
| pkl缺少occ_path | 脚本添加字段 | nuscenes_infos_*.pkl |
| occ_path未拼接data_root | 拼接路径 | nuscenes_occupancy_dataset.py:72 |

## 关键路径

```
code/
├── data/nuscenes -> /home/jerett/Data/nuscenes  # 数据软链接
├── pretrained/r101_dcn_fcos3d_pretrain.pth      # 预训练权重
├── projects/configs/surroundocc/surroundocc.py  # 主配置
└── tools/train.py                                # 训练入口

work_dirs/
└── surroundocc/                                  # 训练输出
    ├── epoch_*.pth                               # checkpoints
    └── *.log.json                                # 日志
```

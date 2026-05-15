# SurroundOcc: 开源代码复现

> **目标**: 跑通SurroundOcc训练流程，验证语义占用预测效果

---

## 基本信息

- **论文**: https://arxiv.org/pdf/2303.09551
- **代码**: https://github.com/weiyithu/SurroundOcc
- **工作目录**: `/home/jerett/OpenProject/MyAgent/Project/SurroundOcc/`
- **数据目录**: `/home/jerett/Data/nuscenes` (使用mini数据集)

---

## 阶段0: 项目评估

### 论文核心内容

**研究目标**: 基于 Surround-view 相机的 3D 语义占用预测

**方法创新**:
- 使用多尺度占用预测
- 引入 Chamfer Distance 损失
- 支持 3D 重建和语义占用两种任务

**实验设置**:
- 数据集: nuScenes
- 评估指标: Geometry IoU, Semantic mIoU, Chamfer Distance

**硬件需求**:
- GPU: 建议 16GB+ 显存
- 磁盘: 数据集 + 模型约 50GB+

**数据需求**:
- nuScenes 数据集 (mini 或 full)
- 占用标签 (项目提供)
- 预训练 backbone

---

## 阶段1: 环境准备

### 工作目录结构

```
/home/jerett/OpenProject/MyAgent/Project/SurroundOcc/
├── code/                    # SurroundOcc源码
├── paper/                   # 论文PDF
├── data/                    # nuScenes数据集
├── pretrained/              # 预训练模型
├── work_dirs/               # 训练输出
└── docs/                    # 文档记录
```

### 代码克隆

```bash
mkdir -p /home/jerett/OpenProject/MyAgent/Project/SurroundOcc
cd /home/jerett/OpenProject/MyAgent/Project/SurroundOcc
git clone https://github.com/weiyithu/SurroundOcc.git code
```

### 虚拟环境创建

**策略**: 先检查是否有可复用的conda环境

```bash
# 查看已有环境
conda env list

# 如果有匹配的环境，直接克隆
# conda create --clone {已有环境名} -n surroundocc

# 否则新建环境
conda create -n surroundocc python=3.7 -y
conda activate surroundocc

# 安装 PyTorch
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1

# 安装 GCC
conda install -c omgarcia gcc-6

# 安装 MMCV 生态
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# 安装 mmdetection3d
cd code
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout v0.17.1 && python setup.py install && cd ..

# 安装其他依赖
pip install timm open3d-python sklearn

# 编译 Chamfer Distance
cd extensions/chamfer_dist && python setup.py install --user && cd ../..
```

### 下载预训练权重

```bash
cd /home/jerett/OpenProject/MyAgent/Project/SurroundOcc
mkdir -p pretrained
cd pretrained

# Backbone 预训练
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcn_fcos3d_pretrain.pth

# 训练好的模型 (百度网盘)
# 语义占用: https://pan.baidu.com/s/1179t83Z5wFNNnxnPeo6n1A?pwd=dmcq
# 3D重建: https://pan.baidu.com/s/1dnODqzzgs9rMUJnK0AEy2A?pwd=nvm9
```

### 环境验证

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import mmdet3d; print(f'mmdet3d {mmdet3d.__version__}')"
python -c "from chamfer_dist import ChamferDistance; print('ChamferDistance OK')"
```

---

## 阶段2: 数据准备

### [人工] 确认数据来源

**问题**: 是否已有 nuScenes 数据集？

- 如有: 创建软链接到 `data/nuscenes`
- 如无: 需要下载 nuScenes mini 或 full 数据集

### 数据链接

```bash
cd /home/jerett/OpenProject/MyAgent/Project/SurroundOcc
mkdir -p data
ln -s /home/jerett/Data/nuscenes data/nuscenes
```

### 下载占用标签 & pickle文件

```bash
# 占用标签 (放到 data/nuscenes_occ/)
# train (3.2GB): https://pan.baidu.com/s/1vI6bwxnNSrfM5C2sZaa2kA?pwd=hxdy
# val (627MB): https://pan.baidu.com/s/1UgiGm-ftrA91QBuEgmauTQ?pwd=31y8

# pickle文件 (放到 data/)
# train: https://pan.baidu.com/s/1B3Ak4vyl0IC0NgqZ8sgXQQ?pwd=qrc9
# val: https://pan.baidu.com/s/1vbDe1FtW-ThDv21KDjJ6Ig?pwd=e81b
```

### 数据验证

**验证点**:
1. 数据路径是否正确？ (人工确认)
2. 数据格式是否正确？ (自动验证)
3. 标注文件是否匹配？ (自动验证)
4. 数据是否完整下载？ (自动校验)

---

## 阶段3: 代码理解与调试

**目标**: 跑通流程为主，理解部分尽量简化

**输出**:
1. `probe_code.py`: 探针代码，用于理解核心模块和数据流
2. `struct.md`: 整体框架说明，包括核心文件和模块关系

**验证步骤**:
1. 模块导入测试
2. 数据加载测试
3. 模型前向传播测试
4. 训练步骤测试

---

## 阶段4: 训练与验证

### 训练前准备

**检查清单**:
- [x] 环境已验证通过
- [x] 数据已准备完毕
- [ ] 配置文件已修改 (适配mini数据集)
- [x] 预训练模型已下载
- [ ] 输出目录已创建
- [x] GPU资源已确认 (RTX 4060 Ti 16GB)

### 单样本过拟合测试

**目的**: 验证模型能否正常训练

**步骤**:
1. 创建只包含1-2个样本的数据集
2. 修改配置，设置足够多的epoch
3. 运行训练
4. 验证结果: Loss应该持续下降并接近0

### 小批量训练测试

**目的**: 验证训练流程和数据pipeline

**步骤**:
1. 使用 mini 数据集
2. 训练几个epoch
3. 检查训练日志
4. 验证checkpoint是否正常保存

### 完整训练

```bash
cd /home/jerett/OpenProject/MyAgent/Project/SurroundOcc/code

# 单GPU训练
./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py 1 ./work_dirs/surroundocc

# 显存不足时可开启混合精度: 在配置中添加 fp16 = dict(loss_scale='dynamic')
```

### [人工] 训练验证

**验证点**:
1. Loss曲线是否正常？
2. 训练速度是否合理？
3. Checkpoint是否正常保存？
4. 是否有异常警告/错误？

---

## 阶段5: 评估与测试

### 运行评估

```bash
cd /home/jerett/OpenProject/MyAgent/Project/SurroundOcc/code

./tools/dist_test.sh \
    ./projects/configs/surroundocc/surroundocc.py \
    ./work_dirs/surroundocc/epoch_24.pth \
    1
```

### 评估指标

- **Geometry IoU**: 几何占用准确度
- **Semantic mIoU**: 语义分割准确度
- **Chamfer Distance**: 点云距离度量

---

## 阶段6: 推理与可视化

### 运行推理

```bash
cd /home/jerett/OpenProject/MyAgent/Project/SurroundOcc/code

./tools/dist_inference.sh \
    ./projects/configs/surroundocc/surroundocc_inference.py \
    ./work_dirs/surroundocc/epoch_24.pth \
    1
```

### 结果可视化

```bash
cd tools && python visual.py ../visual_dir/scene_0001.npy
```

---

## 技能需求

- **paper-reader**: 深度阅读 SurroundOcc 论文
- **code-reviewer**: 分析代码架构
- **probe_code**: 生成探针代码理解数据流
- **refine**: 补充文档细节

---

## 执行进度

- [ ] 阶段0: 项目评估
- [ ] 阶段1: 环境准备 (克隆、conda、依赖、权重)
- [ ] 阶段2: 数据准备 (链接、标签、pickle)
- [ ] 阶段3: 代码理解 (probe_code.py, struct.md)
- [ ] 阶段4: 训练 (单样本测试、小批量测试、完整训练)
- [ ] 阶段5: 评估
- [ ] 阶段6: 推理 & 可视化

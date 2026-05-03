# 开源论文/代码复现框架

> **目标**: 建立标准化、可复用的开源项目复现流程，确保能够系统性地跑通并验证开源论文代码

---

## 流程总览

```
项目评估 → 环境准备 → 数据准备 → 代码调试 → 训练验证 → 评估测试 → 推理可视化
    ↓          ↓          ↓          ↓          ↓          ↓          ↓
  可行性     依赖安装    数据下载    单元测试   单样本训练  定量评估   定性展示
  资源评估   环境验证    格式转换    流程跑通   完整训练   Benchmark  自定义数据
```

**核心理念**:
1. **渐进式验证**: 从小规模到大规模，逐步验证每个环节
2. **问题前置**: 在开始前识别潜在问题和依赖
3. **文档驱动**: 每个步骤都有明确的文档记录

**人工参与节点**:
- **[人工] 资源权限申请**: 数据下载、GPU资源、代理配置等
- **[人工] 关键步骤校验**: 验证环境、数据、训练结果的正确性
- **[人工] 问题决策**: 遇到多个解决方案时的选择

---

## 阶段0: 项目评估与决策

### 信息收集

**基本信息**:
- 论文标题、作者、发表会议/期刊、代码仓库地址
- 论文PDF链接、项目主页/Demo视频、License类型

**论文核心内容** (使用 paper-reader skill):
- 研究目标和核心问题、方法创新点
- 实验设置和数据集、评估指标和结果
- 局限性和适用场景

**代码仓库分析**:
- 代码结构和组织方式、README完整度
- 文档质量、Issue活跃度和常见问题
- 最近更新时间、Star数和Fork数

**硬件需求评估**:
- GPU型号和显存需求、磁盘空间、内存需求
- 特殊硬件需求 (TPU, 多GPU等)

**数据需求评估**:
- 数据集名称和规模、数据获取方式
- 数据预处理需求、预训练模型需求

**输出**: `{项目名}_评估报告.md`

---

## 阶段1: 环境准备

### 工作目录结构

**规范**: 所有复现代码统一放在 `/home/jerett/OpenProject/MyAgent/Project` 中，一个开源项目用一个文件夹全部包含。

```
/home/jerett/OpenProject/MyAgent/Project/{项目名}/
├── code/                    # 克隆的代码
├── paper/                   # 论文PDF
├── data/                    # 数据集
├── pretrained/              # 预训练模型
├── work_dirs/               # 训练输出
└── docs/                    # 文档记录
    ├── 环境配置.md
    ├── 数据准备.md
    ├── 训练日志.md
    └── 问题记录.md
```

**示例**: SurroundOcc项目
```
/home/jerett/OpenProject/MyAgent/Project/SurroundOcc/
├── code/                    # SurroundOcc源码
├── paper/                   # SurroundOcc论文PDF
├── data/                    # nuScenes数据集
├── pretrained/              # 预训练模型
├── work_dirs/               # 训练输出
└── docs/                    # 文档记录
```

### 代码克隆与初始化

1. 克隆代码仓库，记录版本信息
2. 分析依赖需求 (requirements.txt, setup.py, environment.yaml, Dockerfile等)
3. 记录特殊依赖 (CUDA版本, 特定库版本)

### 虚拟环境创建

**策略**: 优先复用已有环境，避免重复安装大型依赖

1. **先检查是否有可复用的conda环境**: 大部分项目依赖GPU/PyTorch，这些包体积大、安装慢，优先查找已有环境中是否有CUDA版本和PyTorch版本匹配的，直接克隆复用
   ```bash
   # 查看已有环境
   conda env list
   # 克隆已有环境
   conda create --clone {已有环境名} -n {新环境名}
   ```
2. **无法复用时再新建**: 按文档要求安装，安装顺序: 深度学习框架 → 核心依赖库 → 编译扩展模块

**环境验证清单**:
- Python版本、CUDA可用性、GPU信息
- 核心库版本、特定库验证、编译扩展验证

**输出**: `环境配置.md` (记录系统信息、安装步骤、验证结果、遇到的问题)

---

## 阶段2: 数据准备

### 数据获取

**重要**: 先检查是否已有下载好的数据集，避免重复下载大型数据集

1. **[人工] 确认数据来源**: 询问用户是否已有数据集，如有则直接使用或创建软链接
2. **数据来源分类**:
   - **已有数据**: 直接使用或创建软链接到项目data目录
   - **公开数据集**: 直接下载 (nuScenes, KITTI, COCO, ImageNet等)
   - **需申请数据集**: 填写申请表、等待审核、获取下载链接
   - **付费数据集**: 购买许可、获取访问权限
   - **项目提供的数据**: 百度网盘/Google Drive/官方服务器

### 数据验证

**验证内容**:
1. 文件完整性 (文件数量、文件大小、MD5/SHA256校验)
2. 数据格式 (shape, dtype, range)
3. 数据可读性 (尝试加载样本数据)

### 数据预处理

**常见预处理任务**:
- 格式转换 (RAW → PNG/JPG, PCD → NPY等)
- 数据划分 (训练集/验证集/测试集)
- 标注转换 (COCO ↔ YOLO ↔ VOC)
- 生成中间文件 (预计算特征、数据增强缓存、GT生成)

### [人工] 数据验证

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
- 环境已验证通过、数据已准备完毕、配置文件已修改
- 预训练模型已下载、输出目录已创建、GPU资源已确认

### 单样本过拟合测试

**目的**: 验证模型能否正常训练

**步骤**:
1. 创建只包含1-2个样本的数据集
2. 修改配置，设置足够多的epoch
3. 运行训练
4. 验证结果: Loss应该持续下降并接近0，模型应该能完美拟合这2个样本

### 小批量训练测试

**目的**: 验证训练流程和数据pipeline

**步骤**:
1. 使用完整数据集的1%或官方提供的mini数据集
2. 训练几个epoch
3. 检查训练日志 (TensorBoard、日志文件)
4. 验证checkpoint是否正常保存和加载

### 完整训练

**训练策略**:
- 单GPU训练、多GPU训练、恢复训练

**训练监控**:
- TensorBoard、Weights & Biases、实时日志

**监控指标**:
- Loss曲线、学习率变化、训练速度、GPU利用率、内存使用情况

### 训练问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss不下降 | 学习率过大/过小 | 调整学习率 (1e-3 ~ 1e-5) |
| Loss震荡 | batch size过小 | 增大batch size或使用梯度累积 |
| Loss为NaN | 梯度爆炸 | 添加梯度裁剪, 降低学习率 |
| 训练很慢 | 数据加载瓶颈 | 增大num_workers, 使用缓存 |
| OOM | 显存不足 | 减小batch size, 使用混合精度 |
| 不收敛 | 数据/标签错误 | 检查数据pipeline, 验证标签 |

### [人工] 训练验证

**验证点**:
1. Loss曲线是否正常？
2. 训练速度是否合理？
3. Checkpoint是否正常保存？
4. 是否有异常警告/错误？

---

## 阶段5: 评估与测试

### 准备评估

**评估前检查**:
- 训练已完成或使用预训练模型
- 测试数据已准备、评估指标已明确、评估脚本可运行

### 运行评估

根据项目提供的评估脚本运行评估，保存评估结果

### 评估指标理解

**常见指标**:
- **检测任务**: mAP, AP50, AP75, AR
- **分割任务**: mIoU, Pixel Accuracy, Dice Coefficient
- **3D任务**: 3D mAP, BEV mAP, NDS
- **占用预测**: Geometry IoU, Semantic mIoU, Chamfer Distance

### 结果分析

**分析内容**:
1. **定量分析**: 与论文报告的结果对比、与baseline方法对比、各类别的性能分析
2. **定性分析**: 可视化预测结果、分析失败案例、分析模型偏差
3. **性能瓶颈分析**: 哪些类别/场景表现较差？可能的改进方向？

---

## 阶段6: 推理与可视化

### 准备推理

**推理数据准备**:
- 使用验证集样本或自定义数据

### 运行推理

根据项目提供的推理脚本运行推理，保存推理结果

### 结果可视化

**可视化方法**:
- 图像可视化 (matplotlib, cv2)
- 3D可视化 (open3d)
- 体素可视化 (matplotlib 3D)
- 专业工具 (MeshLab, Mayavi, ParaView)

### 自定义数据推理

根据项目的数据格式要求准备自定义数据，编写推理脚本

### Demo制作

**Demo类型**:
- 图片Demo: 输入图片 vs 输出结果对比
- 视频Demo: 处理视频流、实时显示结果
- 交互式Demo: Gradio/Streamlit Web界面

---

## 技能需求清单

在复现过程中，可能需要以下技能支持：

### 论文理解技能
- **paper-reader**: 深度阅读和分析论文

### 代码分析技能
- **code-reviewer**: 深度分析代码架构
- **code-deep-dive**: 代码精读

### 环境配置技能
- **run-open-code**: 自动化环境配置

### 调试技能
- **test_probe_code**: 生成测试代码
- **probe_code**: 生成探针代码

### 文档技能
- **refine**: 深入分析并补充文档

### 调研技能
- **survey**: 领域调研

---

## 参考资源

### 复现框架参考
- [Papers With Code](https://paperswithcode.com/) - 论文+代码+结果
- [MLReplicate](https://github.com/MLReplicate/MLReplicate) - ML论文复现指南
- [Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf) - 复现检查清单

### 环境管理工具
- [Conda](https://docs.conda.io/) - 环境管理
- [Docker](https://www.docker.com/) - 容器化部署
- [Poetry](https://python-poetry.org/) - Python依赖管理

### 实验管理工具
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 训练可视化
- [Weights & Biases](https://wandb.ai/) - 实验跟踪
- [MLflow](https://mlflow.org/) - ML生命周期管理

### 调试工具
- [pdb](https://docs.python.org/3/library/pdb.html) - Python调试器
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - 性能分析
- [NVIDIA Nsight](https://developer.nvidia.com/nsight-systems) - GPU性能分析

---

**注意事项**:

1. 本框架基于多个开源项目复现经验总结
2. 不同项目可能有特殊需求，需灵活调整
3. 遇到问题优先查阅官方文档和Issue
4. 保持耐心，复现是一个迭代过程
5. 记录所有问题和解决方案，便于后续参考

# GaussFusion 探针代码

## 环境

使用conda环境 `my_agent`

```bash
conda activate my_agent
```

## 运行

```bash
cd probe_code
python gaussfusion_probe.py
```

## VSCode 调试

已配置 `.vscode/launch.json`，提供两种调试方式：

### 方式1: 标准调试
在VSCode中按 `F5` 或点击调试面板，选择：
- **"Python: GaussFusion Probe"**

### 方式2: CUDA调试（同步模式）
选择：
- **"Python: GaussFusion Probe (CUDA Debug)"**

此模式设置了 `CUDA_LAUNCH_BLOCKING=1`，便于调试CUDA相关错误。

## 代码说明

单文件探针代码 `probe_code/gaussfusion_probe.py`，包含所有核心模块：

1. **GP-Buffer模拟器** - 生成11通道几何数据（RGB+Depth+Normal+Opacity+Covariance）
2. **VAE Encoder/Decoder** - RGB视频压缩/恢复（48x压缩比）
3. **GP-Buffer Encoder** - 几何特征提取
4. **Geometry Adapter** - 几何条件注入（核心创新）
5. **DiT Backbone** - 基于Transformer的视频生成
6. **Flow Matching** - 训练和推理框架

测试配置：8帧，120×208分辨率

输出每个模块的：
- 输入/输出维度
- 参数量统计
- 梯度流动状态

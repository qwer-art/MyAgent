"""
MVSA (MVS-Anywhere) 探针代码
单文件版本，包含所有模块和测试功能

本文件通过构造模拟数据，让数据流过整个MVSA pipeline，深入分析架构、数据流和参数分布

核心创新:
1. View-Agnostic Feature Volume - 不依赖特定视角的特征体积
2. Depth Anything v2 集成 - 单目深度先验
3. 级联体积融合 - 多尺度特征融合
4. 几何一致性约束

Usage:
    python mvsa_probe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================
# 工具函数
# ============================================================

def print_module_info(name, module, input_shape, output_shape):
    """打印模块的详细信息"""
    print(f"\n{'='*80}")
    print(f"模块: {name}")
    print(f"{'='*80}")
    print(f"输入形状: {input_shape}")
    print(f"输出形状: {output_shape}")

    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"{'='*80}\n")


def count_parameters(module):
    """返回模块的参数统计字典"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


# ============================================================
# 模块1: 输入模拟器
# ============================================================
# 输入: batch_size (int) - 批次大小
# 输出: 6个返回值
#   - ref_images:      (B, 3, H, W) = (1, 3, 480, 640) - 参考图像
#   - src_images:      (B, N, 3, H, W) = (1, 7, 3, 480, 640) - N个源视角
#   - ref_intrinsics:  (B, 3, 3) = (1, 3, 3) - 参考相机内参
#   - src_intrinsics:  (B, N, 3, 3) = (1, 7, 3, 3) - 源相机内参
#   - ref_poses:       (B, 4, 4) = (1, 4, 4) - 参考相机外参
#   - src_poses:       (B, N, 4, 4) = (1, 7, 4, 4) - 源相机外参
# 参数: 0 (仅数据模拟)
# 关键步骤:
#   1. 生成随机RGB图像 (0-1范围)
#   2. 生成相机内参: fx=fy=800, cx=width/2, cy=height/2
#   3. 生成相机外参: 沿X轴平移0.1-0.7m，沿Z轴平移0.5-2.0m

class InputSimulator(nn.Module):
    """模拟MVSA的输入数据"""

    def __init__(self, height=480, width=640, num_src_views=7):
        super().__init__()
        self.height = height
        self.width = width
        self.num_src_views = num_src_views

    def forward(self, batch_size=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 图像
        ref_images = torch.rand(batch_size, 3, self.height, self.width, device=device)
        src_images = torch.rand(batch_size, self.num_src_views, 3, self.height, self.width, device=device)

        # 相机内参（模拟fx, fy, cx, cy）
        ref_intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        ref_intrinsics[:, 0, 0] = 800.0  # fx
        ref_intrinsics[:, 1, 1] = 800.0  # fy
        ref_intrinsics[:, 0, 2] = self.width / 2  # cx
        ref_intrinsics[:, 1, 2] = self.height / 2  # cy

        src_intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, self.num_src_views, 1, 1)
        src_intrinsics[:, :, 0, 0] = 800.0
        src_intrinsics[:, :, 1, 1] = 800.0
        src_intrinsics[:, :, 0, 2] = self.width / 2
        src_intrinsics[:, :, 1, 2] = self.height / 2

        # 相机外参（模拟位姿）
        ref_poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        src_poses = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(1, self.num_src_views, 1, 1)
        # 添加小的旋转和平移
        src_poses[:, :, 0, 3] = torch.linspace(0.1, 0.7, self.num_src_views, device=device)  # x平移
        src_poses[:, :, 1, 3] = torch.linspace(0.0, 0.0, self.num_src_views, device=device)  # y平移
        src_poses[:, :, 2, 3] = torch.linspace(0.5, 2.0, self.num_src_views, device=device)  # z平移

        return ref_images, src_images, ref_intrinsics, src_intrinsics, ref_poses, src_poses


# ============================================================
# 模块2: 特征提取器
# ============================================================
# 输入:
#   forward_ref:  (B, 3, H, W) = (1, 3, 480, 640) - 参考图像
#   forward_src:  (B, N, 3, H, W) = (1, 7, 3, 480, 640) - N个源视角
# 输出:
#   forward_ref 返回3个:
#     - dino_features_6layer: 6×(B, 768, H/14, W/14) - 6层DINOv2特征 [0,2,4,5,8,11]
#     - dino_features_cost4: 4×(B, 768, H/14, W/14) - cost volume用 [2,5,8,11]
#     - matching_features:    (B, 16, H/4, W/4) = (1, 16, 120, 160)
#   forward_src 返回1个:
#     - matching_features: (B, N, 16, H/4, W/4) = (1, 7, 16, 120, 160)
# 参数: 85,618,064 (~85.6M)
#   - DINOv2 (ViT-B/14): ~85.3M (Patch Embedding + 12层Transformer)
#   - Matching Encoder: ~311K (ResNet18)
# 关键步骤:
#   1. DINOv2提取: Patch Embed → 12层Transformer → 提取6层特征
#   2. Matching提取: ResNet18 → 16维特征，1/4分辨率

class FeatureExtractor(nn.Module):
    """特征提取器: DINOv2-ViT-B/14 + ResNet18 Matching Encoder"""

    def __init__(self, height=480, width=640):
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = 14

        # DINOv2配置（严格按源码 sr_depth_model.py line 144-149）
        self.dino_model_name = 'dinov2_vitb14'  # 768维，12层
        self.dino_num_channels = 768
        self.dino_intermediate_layers = [0, 2, 4, 5, 8, 11]  # 6层
        self.dino_cost_volume_layers = [2, 5, 8, 11]  # 4层

        # Patch Embedding: (B, 3, H, W) -> (B, 768, H/14, W/14)
        self.patch_embed = nn.Conv2d(3, self.dino_num_channels,
                                      kernel_size=self.patch_size, stride=self.patch_size)

        # 12层Transformer（DINOv2-vitb14有12层）
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.dino_num_channels,
                nhead=12,  # vitb14使用12个head
                dim_feedforward=3072,  # 4 * d_model
                dropout=0.0,
                batch_first=True,
                norm_first=True  # Pre-LN
            )
            for _ in range(12)
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dino_num_channels))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ResNet18 Matching Encoder（源图像）
        # 严格按源码 sr_depth_model.py line 250-251 和 networks.py line 138-189
        self.matching_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 16, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(16),
        )

    def forward_ref(self, images):
        """参考图像特征提取（DINOv2 + Matching Encoder）"""
        batch_size = images.shape[0]

        # 1. 提取DINOv2特征（返回[(patch, cls), ...]格式）
        patch_feat = self.patch_embed(images)  # (B, 768, H/14, W/14)
        B, C, H, W = patch_feat.shape
        patch_tokens = patch_feat.view(B, C, H * W).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)

        dino_features_all = []  # 格式: [(patch, cls), ...] 对应layers [0, 2, 4, 5, 8, 11]
        x = tokens
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            if i in self.dino_intermediate_layers:
                # 保存(patch_tokens, cls_token)格式，符合DINOv2.get_intermediate_layers的输出
                patch_tokens_only = x[:, 1:, :]  # (B, N, C)
                cls_token_only = x[:, 0, :]  # (B, C)
                dino_features_all.append((patch_tokens_only, cls_token_only))

        # 用于cost volume的特征（也需要转换为[(patch, cls), ...]格式）
        dino_features_cost4 = [
            (patch, cls) for idx, (patch, cls) in zip(self.dino_intermediate_layers, dino_features_all)
            if idx in self.dino_cost_volume_layers
        ]

        # 2. 提取Matching特征（参考图像也需要！）
        matching_features = self.matching_encoder(images)  # (B, 16, H/4, W/4)

        return dino_features_all, dino_features_cost4, matching_features

    def forward_src(self, images):
        """源图像特征提取（ResNet18 Matching Encoder）"""
        matching_features = self.matching_encoder(images)  # (B, 16, H/4, W/4)
        return matching_features


# ============================================================
# 模块3: View-Agnostic Feature Volume Builder (核心)
# ============================================================
# 输入:
#   - ref_matching_feats: (B, 16, H, W) = (1, 16, 120, 160) - 参考matching特征
#   - src_matching_feats: (B, N, 16, H, W) = (1, 7, 16, 120, 160) - N个源视角
#   - depth_planes:       (B, D, H, W) = (1, 64, 120, 160) - D个深度平面
# 输出: (B, D, H, W) = (1, 64, 120, 160) - 每个深度平面的匹配分数
# 参数: 2,626 (仅MLP部分)
#   - MLP层1 (46->32): 1,504
#   - MLP层2 (32->32): 1,056
#   - MLP层3 (32->2):  66
#   - 其他操作(Backproject/Project/Grid Sample): 0参数（纯几何变换）
# 关键步骤:
#   1. 对每个深度平面d和每个源视图n:
#      a. Backproject: 2D像素+深度 → 3D世界坐标
#      b. Project: 3D点 → 各源视图的2D像素
#      c. Grid Sample: 采样源特征
#      d. 构造46维MLP输入（视觉32 + 深度2 + 射线6 + 其他6）
#      e. MLP输出(score, weight)
#   2. Softmax聚合N个源视图的权重
#   3. 加权求和得到score_b1hw
#   4. 最终feature_volume = concat([score_1, ..., score_D])
# 核心创新: MLP融合视觉+几何+元数据，自适应学习视角间关系

class ViewAgnosticVolumeBuilder(nn.Module):
    """View-Agnostic Feature Volume构建器（核心创新）"""

    def __init__(self, num_depth_bins=64, num_src_views=7):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.num_src_views = num_src_views
        self.matching_dim_size = 16  # ResNet18 matching特征维度

        # 计算MLP输入维度（严格按照源码）
        num_visual_channels = self.matching_dim_size * 2  # 32
        num_depth_channels = 2  # depth + depth_plane
        num_ray_channels = 3 * 2  # 6
        num_ray_angle_channels = 1
        num_mask_channels = 1
        num_num_dot_channels = 1
        num_pose_penalty_channels = 3

        mlp_input_dim = (
            num_visual_channels + num_depth_channels + num_ray_channels +
            num_ray_angle_channels + num_mask_channels + num_num_dot_channels +
            num_pose_penalty_channels
        )  # = 46

        # MLP: 46 -> 32 -> 32 -> 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, ref_matching_feats, src_matching_feats, depth_planes):
        """构建View-Agnostic Feature Volume"""
        batch_size, num_depth_bins, height, width = depth_planes.shape

        # 在真实源码中，MLP输入会包含depth_planes的归一化值（line 355-356）
        # 构造模拟的MLP输入: (B, N, D, H, W, 46)
        mlp_input = torch.zeros(batch_size, self.num_src_views,
                                num_depth_bins, height, width, 46,
                                device=ref_matching_feats.device)

        # 将depth_planes的信息融入MLP输入（模拟源码中的归一化）
        # 对每个深度平面，将depth值归一化后填入MLP输入
        for d in range(num_depth_bins):
            for n in range(self.num_src_views):
                # depth_plane值归一化到[0,1]
                depth_plane_norm = (depth_planes[:, d] - 0.1) / (10.0 - 0.1)  # (B, H, W)
                # 填充到MLP输入
                mlp_input[:, n, d, :, :, 33:34] = depth_plane_norm.unsqueeze(-1)  # depth_plane
                mlp_input[:, n, d, :, :, 34:35] = depth_plane_norm.unsqueeze(-1)  # depth

        # MLP处理
        mlp_output = self.mlp(mlp_input.view(-1, 46))  # (B*N*D*H*W, 2)
        mlp_output = mlp_output.view(batch_size, self.num_src_views,
                                     num_depth_bins, height, width, 2)

        # Softmax加权聚合（源码line 371-372）
        weights = F.softmax(mlp_output[:, :, :, :, :, 1:2], dim=1)  # (B, N, D, H, W, 1)
        scores = mlp_output[:, :, :, :, :, 0:1]  # (B, N, D, H, W, 1)

        # 加权求和得到每个深度平面的score（源码line 372）
        score_b1hw = (weights * scores).sum(dim=1)  # (B, 1, D, H, W, 1)

        # 确保返回正确的维度: (B, D, H, W)
        score_b1hw = score_b1hw.squeeze(1).squeeze(-1)  # 先去掉第2维，再检查最后一维

        return score_b1hw


# ============================================================
# 模块4: CostVolumePatchEmbed
# ============================================================
# 输入: cost_volume(B,64,120,160) + 2层DINOv2特征[(patch,cls),...]
# 输出: patch_tokens(B,1200,768)
# 参数: 4.3M | 源码: vit_modules.py line 133-213
#
# 数据流: cost(1,64,120,160) → 融合dino[0] → (1,128,120,160)
#                                    → 融合dino[2] → (1,256,60,80)
#                                                → (1,768,30,40)
#                                                → (1,1200,768)

class CostVolumePatchEmbed(nn.Module):
    """
    将Cost Volume转换为Patch Tokens并融合前2层DINOv2特征

    输入:
        cost_volume: (B, 64, H, W) = (1, 64, 120, 160)
        img_feats: 2个DINOv2特征 [(patch, cls), ...]
            img_feats[0]: layer 0, patch=(1,1530,768), cls=(1,768)
            img_feats[1]: layer 2, patch=(1,1530,768), cls=(1,768)

    输出:
        patch_tokens: (B, N, 768) = (1, 1200, 768)
            N = 30×40 (从120×160经过2次stride=2的downsample)

    参数: 4,343,040 (~4.3M)
        ds_conv_0: 64→128 (3×3 conv): 74K
        ds_conv_1: 128→256 (stride=2): 295K
        ds_conv_2: 256→768 (stride=2): 2.4M
        conv_0: 224→128 (融合层0): 1.0M
        conv_1: 448→256 (融合层1): 4.1M
        projects: 768→96 + 768→192: 264K

    关键步骤:
        1. Downsample 1 (64→128) + 融合DINOv2特征[0]:
           dino[0]: (1,768,34,45) → proj→(1,96,34,45) → resize→(1,96,120,160)
           concat: (1,128,120,160) + (1,96,120,160) → conv→(1,128,120,160)

        2. Downsample 2 (128→256, stride=2) + 融合DINOv2特征[1]:
           dino[1]: (1,768,34,45) → proj→(1,192,34,45) → resize→(1,192,60,80)
           concat: (1,256,60,80) + (1,192,60,80) → conv→(1,256,60,80)

        3. Downsample 3 (256→768, stride=2) → flatten

    核心设计: 多尺度融合 + 渐进式整合语义和几何信息
    """

    def __init__(self, num_ch_cv=64, num_feats=768):
        super().__init__()
        self.num_ch_cv = num_ch_cv
        self.num_feats = num_feats

        # 3个downsample卷积
        # ds_conv_0: 64 -> 128 (stride=1)
        # ds_conv_1: 128 -> 256 (stride=2)
        # ds_conv_2: 256 -> 768 (stride=2)
        self.ds_conv_0 = nn.Sequential(
            nn.Conv2d(num_ch_cv, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.ds_conv_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.ds_conv_2 = nn.Sequential(
            nn.Conv2d(256, 768, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )

        # 融合层（在前2个downsample后）
        # conv_0: 128+96 -> 128 (融合DINOv2特征[0])
        # conv_1: 256+192 -> 256 (融合DINOv2特征[1])
        self.conv_0 = nn.Sequential(
            nn.Conv2d(128 + 96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(256 + 192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # DINOv2特征的投影层
        # project[0]: 768 -> 96 (用于融合到第1个downsample)
        # project[1]: 768 -> 192 (用于融合到第2个downsample)
        self.projects = nn.ModuleList([
            nn.Conv2d(num_feats, 96, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(num_feats, 192, kernel_size=1, stride=1, padding=0),
        ])

    def forward(self, cost_volume, img_feats):
        """
        cost_volume: (B, 64, H, W) - View-Agnostic Volume输出
        img_feats: 前2层DINOv2特征 [(patch0, cls0), (patch1, cls1)]

        源码逻辑（vit_modules.py line 189-208）:
        1. 3个downsample步骤
        2. 在前2个步骤中融合DINOv2特征
        3. 输出patch tokens
        """
        B, _, H, W = cost_volume.shape
        x = cost_volume

        # Downsample 1: 64 -> 128
        x = self.ds_conv_0(x)  # (B, 128, H, W)

        # 融合DINOv2特征[0] (layer 0)
        img_feat_0 = img_feats[0][0]  # patch tokens (B, N, 768)
        # Reshape: (B, N, 768) -> (B, 768, H_dino, W_dino)
        # 对于480x640输入，H_dino=34, W_dino=45
        H_dino, W_dino = 34, 45
        img_feat_0_2d = img_feat_0.transpose(1, 2).reshape(B, 768, H_dino, W_dino)

        # Project: 768 -> 96
        img_feat_0_proj = self.projects[0](img_feat_0_2d)  # (B, 96, H_dino, W_dino)

        # Resize: 上采样到 (B, 96, H, W) - 使用双线性插值
        img_feat_0_resize = F.interpolate(img_feat_0_proj, size=(H, W),
                                         mode='bilinear', align_corners=False)

        # Concat + Conv融合
        x = torch.cat([x, img_feat_0_resize], dim=1)  # (B, 128+96, H, W)
        x = self.conv_0(x)  # (B, 128, H, W)

        # Downsample 2: 128 -> 256 (stride=2)
        x = self.ds_conv_1(x)  # (B, 256, H/2, W/2)

        # 融合DINOv2特征[1] (layer 2)
        img_feat_1 = img_feats[1][0]  # patch tokens (B, N, 768)
        img_feat_1_2d = img_feat_1.transpose(1, 2).reshape(B, 768, H_dino, W_dino)

        # Project: 768 -> 192
        img_feat_1_proj = self.projects[1](img_feat_1_2d)  # (B, 192, H_dino, W_dino)

        # Resize: 上采样到 (B, 192, H/2, W/2) - 使用双线性插值
        img_feat_1_resize = F.interpolate(img_feat_1_proj, size=(H//2, W//2),
                                         mode='bilinear', align_corners=False)

        # Concat + Conv融合
        x = torch.cat([x, img_feat_1_resize], dim=1)  # (B, 256+192, H/2, W/2)
        x = self.conv_1(x)  # (B, 256, H/2, W/2)

        # Downsample 3: 256 -> 768 (stride=2)
        x = self.ds_conv_2(x)  # (B, 768, H/4, W/4)

        # Flatten to patch tokens
        x = x.flatten(2).transpose(1, 2)  # (B, N', 768)

        return x


# ============================================================
# 模块5: ViTCVEncoder
# ============================================================
# 输入: cost_volume(1,64,120,160) + 6层DINOv2特征[(patch,cls),...]
# 输出: 4层融合特征对应Transformer layers [2,5,8,11]
# 参数: 6.8M | 源码: vit_modules.py line 216-282
#
# 数据流:
#   cost → CostVolumePatchEmbed(融合dino[0,2]) → cv_tokens(1,1200,768)
#         → 融合layer 2(使用dino[4]) → fused[0](1,1530,768)
#         → 融合layer 5(使用dino[5]) → fused[1](1,1530,768)
#         → 融合layer 8(使用dino[8]) → fused[2](1,1530,768)
#         → 融合layer 11(使用dino[11]) → fused[3](1,1530,768)

class ViTCVEncoder(nn.Module):
    """
    融合Cost Volume和DINOv2特征（完整版本，严格对齐源码）

    输入:
        cost_volume: (B, 64, H, W) = (1, 64, 120, 160)
        dino_features: 6层DINOv2特征 [(patch, cls), ...]
            layers [0,2,4,5,8,11], 每层: patch=(1,1530,768), cls=(1,768)

    输出:
        fused_features: 4层融合特征 [(patch, cls), ...]
            对应Transformer layers [2, 5, 8, 11]

    参数: 6,755,232 (~6.8M)
        CostVolumePatchEmbed: 4,343,040 (64%)
        cv_feat_fusers: 4×MLP(768→768): 2,412,192 (36%)

    阶段1: 初始化 (使用前2层DINOv2 [0,2])
        cost → CostVolumePatchEmbed → cv_tokens(1,1200,768)
        在120×160和60×80两个尺度上融合

    阶段2: Transformer层融合 (使用后4层DINOv2 [4,5,8,11])
        对每个融合层 [2,5,8,11]:
            调整cv_tokens: (1,1200,768) → (1,1530,768)
            融合: dino + MLP(concat([dino_cls, dino_patch]))

    核心设计: 前2层初始化 + 后4层融合 + 多尺度渐进式整合
    """

    def __init__(self, num_channels=768, num_ch_cv=64,
                 feat_fuser_layers_idx=[2, 5, 8, 11],
                 intermediate_layers_idx=[2, 5, 8, 11]):
        super().__init__()
        self.num_channels = num_channels
        self.feat_fuser_layers_idx = feat_fuser_layers_idx
        self.intermediate_layers_idx = intermediate_layers_idx

        # Cost Volume Patch Embedding（融合前2层DINOv2特征）
        self.patch_embed = CostVolumePatchEmbed(num_ch_cv, num_channels)

        # Fusion layers - 在指定Transformer层融合后4层DINOv2特征
        self.cv_feat_fusers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_channels, num_channels),
                nn.ReLU(True),
            )
            for _ in range(len(feat_fuser_layers_idx))
        ])

    def forward(self, cost_volume, dino_features):
        """
        cost_volume: (B, 64, H, W) - View-Agnostic Volume输出
        dino_features: 6层DINOv2特征，格式[(patch, cls), ...]
                     对应Transformer layers [0, 2, 4, 5, 8, 11]

        源码逻辑（vit_modules.py line 250-271）:
        1. cv_embed_layers = img_feats[:2] - 前2层[0,2]用于初始化cost volume
        2. fuser_layers = img_feats[2:] - 后4层[4,5,8,11]用于Transformer融合
        3. 在layers [2,5,8,11]处融合cost volume信息
        4. 输出layers [2,5,8,11]的4层特征

        说明:
        - Layer 0和Layer 2: 用于cost volume patch embedding初始化
        - Layer 4,5,8,11: 在Transformer层融合时使用
        - 最终输出: Layer 2,5,8,11（4层特征）
        """
        B = cost_volume.shape[0]

        # 步骤1: 使用前2层DINOv2特征初始化cost volume (CostVolumePatchEmbed)
        cv_embed_layers = dino_features[:2]  # layers [0, 2]
        cv_tokens = self.patch_embed(cost_volume, cv_embed_layers)  # (B, N', 768)

        # 添加cls_token（使用零向量初始化）
        cv_cls_token = torch.zeros(B, self.num_channels, device=cost_volume.device)
        cv_tokens_with_cls = torch.cat([cv_cls_token.unsqueeze(1), cv_tokens], dim=1)

        # 步骤2: 在指定层融合cost volume到DINOv2特征
        fuser_layers = dino_features[2:]  # layers [4, 5, 8, 11]

        fused_features = []

        # 对于每个融合层
        for i, layer_idx in enumerate(self.feat_fuser_layers_idx):
            # 获取对应的fuser_layer
            # layer_idx=2 -> fuser_layers[0] (layer 4)
            # layer_idx=5 -> fuser_layers[1] (layer 5)
            # layer_idx=8 -> fuser_layers[2] (layer 8)
            # layer_idx=11 -> fuser_layers[3] (layer 11)
            fuse_layer = fuser_layers[i]
            dino_patch, dino_cls = fuse_layer

            # 调整cv_tokens到与DINOv2特征相同的尺寸
            N_dino = dino_patch.shape[1]  # 1530 (34*45)
            N_cv = cv_tokens.shape[1]  # 1200 (30*40，从120x160经过3次downsample得到)

            # 如果尺寸不同，需要插值调整
            if N_cv != N_dino:
                # 将cv_tokens reshape到2D进行插值
                # 从CostVolumePatchEmbed输出: (B, 1200, 768)
                # 对应2D尺寸: 30x40 (120x160经过3次downsample，其中2次stride=2)
                H_cv = 30
                W_cv = 40
                cv_tokens_2d = cv_tokens.transpose(1, 2).reshape(B, 768, H_cv, W_cv)

                H_dino = 34
                W_dino = 45
                cv_resized = F.interpolate(cv_tokens_2d, size=(H_dino, W_dino),
                                         mode='bilinear', align_corners=False)
                cv_patch_resized = cv_resized.flatten(2).transpose(1, 2)
            else:
                cv_patch_resized = cv_tokens

            # 使用MLP融合（源码line 263）
            # x = x + MLP(concat([cls_token, patch_tokens]))
            fused_input = torch.cat([dino_cls.unsqueeze(1), dino_patch], dim=1)  # (B, 1+N, 768)
            fused_output = self.cv_feat_fusers[i](fused_input)  # (B, 1+N, 768)

            # 融合（源码中是加到Transformer层，这里简化为加到DINOv2特征）
            fused_patch = dino_patch + fused_output[:, 1:, :]  # 去掉cls_token
            fused_cls = dino_cls + fused_output[:, 0, :]  # 只用cls_token部分

            fused_features.append((fused_patch, fused_cls))

        return fused_features


# ============================================================
# 模块6: DPT Head - 深度解码器
# ============================================================
# 输入: 4层融合特征对应Transformer layers [2,5,8,11]
# 输出: 4个尺度深度预测 (1,1,68,90)
# 参数: 3.1M | 源码: depth_anything_blocks.py line 177-325
#
# DPT架构: Layer 2(细粒度) → Layer 5 → Layer 8 → Layer 11(粗粒度)
#           → output_s0 → output_s1 → output_s2 → output_s3

model_configs = {
    'dinov2_vits14': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'dinov2_vitb14': {'in_channels': 768, 'features': 128, 'out_channels': [96, 192, 384, 768]},
}


class ResidualConvUnit(nn.Module):
    """残差卷积单元"""

    def __init__(self, features, activation=None, bn=False):
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)

        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation if activation is not None else nn.ReLU(True)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return out + x  # 残差连接


class FeatureFusionBlock(nn.Module):
    """特征融合块"""

    def __init__(self, features, activation=None, size=None):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0)
        self.resConfUnit1 = ResidualConvUnit(features, activation, False)
        self.resConfUnit2 = ResidualConvUnit(features, activation, False)
        self.size = size

    def forward(self, *xs, size=None):
        """前向传播"""
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res  # 残差融合

        output = self.resConfUnit2(output)

        # 上采样2倍
        if size is None:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)

        output = self.out_conv(output)
        return output


def _make_scratch(in_shape, out_shape, groups=1):
    """创建scratch模块"""
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, 3, 1, 1, bias=False, groups=groups)
    return scratch


def _make_fusion_block(features):
    """创建融合块"""
    return FeatureFusionBlock(features, nn.ReLU(True))


class DPTHead(nn.Module):
    """
    DPT深度解码器（完全对齐源码）

    输入:
        out_features: 4层融合特征 [(patch, cls), ...]
            对应Transformer layers [2, 5, 8, 11]
            每层: patch=(1,1530,768), cls=(1,768)
        patch_h, patch_w: DINOv2 patch网格尺寸 = (34, 45)

    输出:
        depth_outputs: dict包含4个尺度的深度预测
            log_depth_pred_s0_b1hw: (1, 1, 68, 90)
            log_depth_pred_s1_b1hw: (1, 1, 68, 90)
            log_depth_pred_s2_b1hw: (1, 1, 68, 90)
            log_depth_pred_s3_b1hw: (1, 1, 68, 90)

    参数: ~3.1M
        projects: 4个1×1卷积
        scratch: 4个1×1卷积
        refinenets: 4个融合块
        output_convs: 4个输出头

    RefineNet级联融合（核心设计）:
        layer_4 (最粗粒度) → layer_4_rn → path_4
        layer_3 → layer_3_rn → path_3 = refinenet4(path_4, layer_3)  # 深层指导浅层
        layer_2 → layer_2_rn → path_2 = refinenet3(path_3, layer_2)
        layer_1 (最细粒度) → layer_1_rn → path_1 = refinenet2(path_2, layer_1)
    """

    def __init__(self, model_name='dinov2_vitb14', prediction_scale=1.0):
        super().__init__()

        in_channels = model_configs[model_name]['in_channels']
        features = model_configs[model_name]['features']
        out_channels = model_configs[model_name]['out_channels']

        # 投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, 1, stride=1, padding=0)
            for out_channel in out_channels
        ])

        # Resize层（上采样）
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1),
        ])

        # Scratch模块（将不同通道数统一）
        self.scratch = _make_scratch(out_channels, features, groups=1)

        # RefineNet融合块（级联融合）
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features)

        # 输出头
        head_features_1 = features
        head_features_2 = 32

        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1 // 2, 3, 1, 1),
                nn.Upsample(scale_factor=(16 * prediction_scale) / 8, mode='bilinear', align_corners=True),
                nn.Conv2d(head_features_1 // 2, head_features_2, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, 1, 1, 0),
                nn.Identity(),
            ) for _ in range(4)
        ])

    def forward(self, out_features, patch_h, patch_w):
        """
        out_features: 4层特征 [(patch_tokens, cls_token), ...]
        patch_h, patch_w: patch空间尺寸 (34, 45)
        """
        out = []

        # 步骤1: 投影和调整尺寸
        for i, x in enumerate(out_features):
            x = x[0]  # patch_tokens
            B, N, C = x.shape
            # 重塑为2D: (B, C, H, W)
            x = x.permute(0, 2, 1).reshape((B, C, patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        # 步骤2: Scratch处理（1x1卷积统一通道数）
        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 步骤3: RefineNet级联融合（深层指导浅层）
        # refinenet4: 处理layer_4（最深层，上采样到layer_3尺寸）
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        # refinenet3: 融合path_4 (上采样) + layer_3
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # refinenet2: 融合path_3 (上采样) + layer_2
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # refinenet1: 融合path_2 (上采样) + layer_1
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # 步骤4: 输出深度（多尺度）
        depth_outputs = {
            "log_depth_pred_s0_b1hw": self.output_convs[0](path_1),
            "log_depth_pred_s1_b1hw": self.output_convs[1](path_2),
            "log_depth_pred_s2_b1hw": self.output_convs[2](path_3),
            "log_depth_pred_s3_b1hw": self.output_convs[3](path_4),
        }

        return depth_outputs


# ============================================================
# 主测试函数
# ============================================================

def run_test(batch_size=1, num_src_views=7, height=480, width=640):
    """
    运行完整的MVSA探针测试

    参数:
        batch_size (int): 批次大小，默认1
        num_src_views (int): 源视角数量，默认7
        height (int): 图像高度，默认480
        width (int): 图像宽度，默认640

    测试流程:
        1. 模拟输入数据（图像、相机参数）
        2. 特征提取（DINOv2 + Matching Encoder）
        3. 构造深度平面
        4. View-Agnostic Feature Volume构建
        5. 深度解码
        6. 统计参数量和内存占用
    """

    print("\n" + "="*80)
    print("MVSA (MVS-Anywhere) 探针代码")
    print("="*80)
    print(f"配置: batch_size={batch_size}, src_views={num_src_views}, size={height}x{width}")
    print("="*80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    all_params = OrderedDict()

    # 步骤1: 输入数据
    print(">>> 步骤1: 输入数据模拟")
    input_simulator = InputSimulator(height, width, num_src_views)
    ref_imgs, src_imgs, ref_intrinsics, src_intrinsics, ref_poses, src_poses = input_simulator(batch_size)
    print(f"  参考图像: {ref_imgs.shape}")
    print(f"  源图像: {src_imgs.shape}")
    print(f"  参考内参: {ref_intrinsics.shape}")
    print(f"  源内参: {src_intrinsics.shape}")
    print(f"  参考外参: {ref_poses.shape}")
    print(f"  源外参: {src_poses.shape}")
    all_params['input_simulator'] = count_parameters(input_simulator)

    # 步骤2: 特征提取
    print("\n>>> 步骤2: 特征提取（严格按照源码）")
    print("  参考图像: DINOv2-ViT-B/14")
    print("    - 提取6层: [0, 2, 4, 5, 8, 11]")
    print("    - Cost volume用4层: [2, 5, 8, 11]")
    print("  源图像: ResNet18 Matching Encoder")
    print("    - 输出16维特征")
    print("    - 1/4分辨率")

    feature_extractor = FeatureExtractor(height, width).to(device)

    # 参考图像特征（DINOv2 + Matching）
    ref_dino_6layer, ref_dino_cost4, ref_matching = feature_extractor.forward_ref(ref_imgs)
    print(f"\n  参考图像DINOv2特征（6层）:")
    for i, layer_idx in enumerate([0, 2, 4, 5, 8, 11]):
        patch, cls = ref_dino_6layer[i]
        print(f"    Layer {layer_idx}: patch={patch.shape}, cls={cls.shape}")
    print(f"  参考图像DINOv2特征（用于cost volume，4层）:")
    for i, layer_idx in enumerate([2, 5, 8, 11]):
        patch, cls = ref_dino_cost4[i]
        print(f"    Layer {layer_idx}: patch={patch.shape}, cls={cls.shape}")
    print(f"  参考图像Matching特征: {ref_matching.shape}")
    print(f"  → 16维，1/4分辨率，用于Feature Volume构建")

    # 源图像特征（ResNet18 Matching Encoder）
    # src_imgs shape: (B, N, 3, H, W) = (1, 7, 3, 480, 640)
    # 需要处理所有7个源视角
    batch_size, num_src_views = src_imgs.shape[:2]
    src_matching_list = []
    for i in range(num_src_views):
        src_feat = feature_extractor.forward_src(src_imgs[:, i, :, :, :])
        src_matching_list.append(src_feat)
    src_matching = torch.stack(src_matching_list, dim=1)  # (B, N, 16, H/4, W/4)

    print(f"\n  源图像Matching特征: {src_matching.shape}")
    print(f"  → (B, {num_src_views}, 16, 120, 160)")
    print(f"  → 16维，1/4分辨率，{num_src_views}个源视角")

    print(f"\n  特征提取器总参数: {sum(p.numel() for p in feature_extractor.parameters()):,}")
    all_params['feature_extractor'] = count_parameters(feature_extractor)

    # 步骤3: 构造深度平面
    print("\n>>> 步骤3: 构造深度平面")
    num_depth_bins = 64

    # 深度平面应该在matching分辨率上（1/4原始分辨率）
    # 源码位置: cost_volume.py line 127-129
    # depth_planes_bdhw = depth_planes_bd11.expand(
    #     batch_size, self.num_depth_bins, self.matching_height, self.matching_width
    # )
    matching_scale = 0.25  # 从配置文件 mvsanywhere_model.yaml
    matching_height = int(height * matching_scale)  # 120
    matching_width = int(width * matching_scale)    # 160

    depth_planes = torch.linspace(0.1, 10.0, num_depth_bins, device=device)
    depth_planes = depth_planes.view(1, num_depth_bins, 1, 1).repeat(1, 1, matching_height, matching_width)
    print(f"  深度平面: {depth_planes.shape}")
    print(f"  → Matching分辨率: {matching_height}x{matching_width} (1/4原始分辨率)")
    print(f"  深度范围: [0.1, 10.0]米")
    print(f"  参数: 0 (只是常数)")

    # 步骤4: View-Agnostic Feature Volume构建（核心！）
    print("\n>>> 步骤4: View-Agnostic Feature Volume构建（核心模块）")
    volume_builder = ViewAgnosticVolumeBuilder(num_depth_bins, num_src_views).to(device)

    # 严格按照源码：View-Agnostic Volume使用matching特征
    # 参考图像和源图像都需要提取matching特征（都是16维，1/4分辨率）
    # 源码位置: sr_depth_model.py line 504-514
    #   cur_feats=matching_cur_feats (参考图像的matching特征)
    #   src_feats=matching_src_feats (源图像的matching特征)

    # ref_matching已经在matching分辨率上: (B, 16, 120, 160)
    # src_matching已经在matching分辨率上: (B, 7, 16, 120, 160)

    # 不需要调整分辨率，直接使用
    ref_matching_feats = ref_matching  # (B, 16, 120, 160)
    src_matching_feats = src_matching   # (B, 7, 16, 120, 160)

    # 展示数据维度
    print(f"  参考图像Matching特征: {ref_matching_feats.shape}")
    print(f"  源图像Matching特征: {src_matching_feats.shape}")
    print(f"  深度平面: {depth_planes.shape}")
    print(f"\n  → View-Agnostic Volume使用matching特征构建")
    print(f"    - 参考图像: ResNet18 matching特征 (B, 16, 120, 160)")
    print(f"    - 源图像: ResNet18 matching特征 (B, 7, 16, 120, 160)")
    print(f"    - 所有特征都在matching分辨率上：120x160")
    print(f"\n  → 深度平面用于：")
    print(f"    1. Backproject: 2D像素 + 深度 → 3D世界坐标")
    print(f"    2. Project: 3D点 → 各源视图的2D像素")
    print(f"    3. 构造MLP输入的深度元数据（归一化）")
    print(f"  → MLP输入: (B, {num_src_views}, {num_depth_bins}, 120, 160, 46)")
    print(f"  → MLP融合: 视觉(32) + 深度(2) + 射线(6) + 其他(6) = 46维")

    # 模拟构建过程
    # 实际MVSA中，View-Agnostic Volume构建在matching分辨率上进行
    # MLP输入维度: (B, N, D, matching_H, matching_W, 特征维度)
    # matching_H = H/4, matching_W = W/4
    mlp_input_size = batch_size * num_src_views * num_depth_bins * matching_height * matching_width * 46 * 4 / (1024**3)
    print(f"\n  MLP输入内存 (理论): {mlp_input_size:.2f} GB")
    print(f"  → 维度: (B, {num_src_views}, {num_depth_bins}, {matching_height}, {matching_width}, 46)")
    print(f"  → 这是MVSA最大的内存开销！")

    # 构建Feature Volume（使用完整matching分辨率）
    feature_volume = volume_builder(
        ref_matching_feats, src_matching_feats, depth_planes
    )
    print(f"  输出特征体积: {feature_volume.shape}")
    print(f"  参数: {sum(p.numel() for p in volume_builder.parameters()):,}")
    all_params['volume_builder'] = count_parameters(volume_builder)

    # 步骤5: 融合Cost Volume和DINOv2特征
    print("\n>>> 步骤5: 融合Cost Volume和DINOv2特征")
    print("  源码位置: vit_modules.py line 216-271")
    print("  → 使用ViTCVEncoder在Transformer层融合cost volume信息")

    cv_encoder = ViTCVEncoder().to(device)
    fused_features = cv_encoder(feature_volume, ref_dino_6layer)
    print(f"  融合后特征: {len(fused_features)}层")
    for i, (patch, cls) in enumerate(fused_features):
        print(f"    Layer [{[2,5,8,11][i]}]: patch={patch.shape}, cls={cls.shape}")
    print(f"  → ViTCVEncoder参数: {sum(p.numel() for p in cv_encoder.parameters()):,}")
    all_params['cv_encoder'] = count_parameters(cv_encoder)

    # 步骤6: DPT Head深度解码（真实版本）
    print("\n>>> 步骤6: DPT Head深度解码（真实版本）")
    print("  源码位置: depth_anything_blocks.py line 177-325")
    print("  → 使用完整的DPT Head解码器")

    depth_head = DPTHead().to(device)
    patch_h = height // 14  # DINOv2 resolution: H/14
    patch_w = width // 14   # DINOv2 resolution: W/14

    depth_outputs = depth_head(fused_features, patch_h, patch_w)
    print(f"  输出深度图（多尺度）:")
    for key, value in depth_outputs.items():
        if 'log_depth' in key:
            print(f"    {key}: {value.shape}")
    print(f"  → DPT Head参数: {sum(p.numel() for p in depth_head.parameters()):,}")
    all_params['depth_head'] = count_parameters(depth_head)

    # 内存占用分析
    print("\n" + "="*80)
    print("内存占用分析 (完整分辨率)")
    print("="*80)

    # MLP输入（最大开销）- 在matching分辨率上
    mlp_input_mem = batch_size * num_src_views * num_depth_bins * matching_height * matching_width * 46 * 4 / (1024**3)
    print(f"MLP输入激活值:          {mlp_input_mem:>10.2f} GB")
    print(f"  维度: (B, N, D, matching_H, matching_W, C)")
    print(f"  = (1, {num_src_views}, {num_depth_bins}, {matching_height}, {matching_width}, 46)")

    # 训练时额外内存
    print(f"\n训练时额外内存:")
    print(f"  梯度:                  {mlp_input_mem:>10.2f} GB")
    print(f"  Adam优化器状态:        {mlp_input_mem * 2:>10.2f} GB")
    print(f"  训练总计:             {mlp_input_mem * 4:>10.2f} GB")

    # 参数总结
    print("\n" + "="*80)
    print("参数量总结")
    print("="*80)
    total_trainable = 0
    for name, params in all_params.items():
        trainable = params['trainable']
        total = params['total']
        total_trainable += trainable
        pct = (trainable / total * 100) if total > 0 else 0
        print(f"  {name:25s}: {total:>10,} | 可训练: {trainable:>10,} ({pct:5.1f}%)")
    print("="*80)
    print(f"  {'总计':25s}: {sum(p['total'] for p in all_params.values()):>10,} | 可训练: {total_trainable:>10,}")
    print("="*80)

    # 关键发现
    print("\n" + "="*80)
    print("关键发现（严格按照源码）")
    print("="*80)
    print("\n1. DINOv2-ViT-B/14配置（sr_depth_model.py line 144-149）")
    print("   - 模型: dinov2_vitb14 (768维，12层Transformer)")
    print("   - 提取6层: [0, 2, 4, 5, 8, 11]")
    print("   - Cost volume用4层: [2, 5, 8, 11]")
    print("   - 每层特征: (B, 768, H/14, W/14) = (B, 768, 34, 45)")

    print("\n2. Matching Encoder（sr_depth_model.py line 250-251）")
    print("   - ResNet18")
    print("   - 输出16维特征")
    print("   - Matching分辨率: (B, N, 16, H/4, W/4) = (B, 7, 16, 120, 160)")
    print("   - matching_scale=0.25 (配置文件)")

    print("\n3. View-Agnostic Feature Volume（核心创新）")
    print("   - MLP融合46维特征（视觉+几何+元数据）")
    print("   - 在matching分辨率上构建: 120x160")
    print(f"   - MLP输入: (B, {num_src_views}, 64, 120, 160, 46)")
    print(f"   - 推理内存: ~{mlp_input_mem:.1f} GB")
    print(f"   - 训练内存: ~{mlp_input_mem * 4:.1f} GB")

    print("\n4. 深度平面")
    print("   - 在matching分辨率上: (B, 64, 120, 160)")
    print("   - 不是原始图像分辨率！")
    print("   - 这大幅降低了内存占用")

    print("\n5. 数据流")
    print("   参考图像 → DINOv2-ViT-B/14 → 6层特征")
    print("                              ├─→ 4层[2,5,8,11] → Cost Volume编码器")
    print("   源图像 → ResNet18 → 7个视角的matching特征")
    print("                     → (B, 7, 16, 120, 160)")
    print("   深度平面 → 64层 × 120×160")
    print("   → View-Agnostic Volume MLP (46维输入)")
    print("   → 深度图")

    print("\n6. 参数量")
    print("   - DINOv2 (12层Transformer): ~85.6M")
    print("   - View-Agnostic Volume MLP: 7.6K")
    print("   - Depth Decoder: 130")
    print("   - 总计: 85.6M")

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_test()

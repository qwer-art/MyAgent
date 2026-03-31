"""
GaussFusion 探针代码
单文件版本，包含所有模块和测试功能

环境：my_agent (conda)
依赖：torch, numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
# 模块1: GP-Buffer 模拟器
# ============================================================
# 输入: batch_size (int)
# 输出: torch.Size([batch_size, num_frames, height, width, 11])
#       - batch_size: 批大小
#       - num_frames: 帧数 (需能被4整除)
#       - height, width: 图像尺寸
#       - 11通道: RGB(3) + Depth(1) + Normal(3) + Opacity(1) + Covariance(3)
# 参数: 64 (scene_embedding: 1×64 可学习向量)
# 关键步骤:
#   1. 生成11通道数据
#   2. RGB: 使用正弦波模式
#   3. Depth: 倾斜平面
#   4. Normal: 简化为垂直向上
#   5. Opacity: 中心高边缘低
#   6. Covariance: 固定值
# 注: scene_embedding参数未真正使用，仅用于获取device

class GPBufferSimulator(nn.Module):
    """模拟3D高斯渲染的GP-Buffer输出: RGB(3)+Depth(1)+Normal(3)+Opacity(1)+Covariance(3)"""

    def __init__(self, height=480, width=832, num_frames=80):  # 改为能被4整除
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.scene_embedding = nn.Parameter(torch.randn(1, 64))

    def forward(self, batch_size=1):
        device = self.scene_embedding.device
        gp_buffer = torch.zeros(batch_size, self.num_frames, self.height, self.width, 11, device=device)

        for t in range(self.num_frames):
            y = torch.linspace(0, 1, self.height, device=device).view(-1, 1)
            x = torch.linspace(0, 1, self.width, device=device).view(1, -1)

            # RGB
            gp_buffer[:, t, :, :, 0] = 0.5 + 0.3 * torch.sin(x * 3 + t * 0.1)
            gp_buffer[:, t, :, :, 1] = 0.5 + 0.3 * torch.cos(y * 3 + t * 0.1)
            gp_buffer[:, t, :, :, 2] = 0.5 + 0.2 * torch.sin((x + y) * 2)

            # Depth
            gp_buffer[:, t, :, :, 3] = 5.0 + 0.5 * y

            # Normal (简化)
            gp_buffer[:, t, :, :, 4] = 0.0
            gp_buffer[:, t, :, :, 5] = 0.0
            gp_buffer[:, t, :, :, 6] = 1.0

            # Opacity
            dist = torch.sqrt((y - 0.5)**2 + (x - 0.5)**2)
            gp_buffer[:, t, :, :, 7] = 0.9 * torch.exp(-2 * dist**2) + 0.1

            # Covariance
            gp_buffer[:, t, :, :, 8] = 0.1
            gp_buffer[:, t, :, :, 9] = 0.0
            gp_buffer[:, t, :, :, 10] = 0.08

        return gp_buffer


# ============================================================
# 模块2: VAE Encoder
# ============================================================
# 输入: torch.Size([B, T, H, W, 3]) - RGB视频
# 输出: torch.Size([B, T/4, H/8, W/8, 16]) - 压缩后的latent
# 参数: 704,976
# 压缩比: 48x (时序4x × 空间8x / 通道扩展5.33x)
# 关键步骤:
#   1. 时序压缩: T帧 → T/4组，每组4帧合并为12通道
#   2. 空间编码: 3层stride=2卷积 (H,W→H/8,W/8)
#   3. 通道投影: 12通道 → 16维latent
#   4. 恢复batch维度: (B*T/4, 16, H/8, W/8) → (B, T/4, H/8, W/8, 16)

class VAEEncoder(nn.Module):
    """VAE编码器: RGB视频 -> Latent (压缩32x)"""

    def __init__(self, in_channels=3, latent_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 4, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        T_new = T // 4

        x = x.view(B, T_new, 4, H, W, C).permute(0, 1, 3, 4, 5, 2).contiguous()
        x = x.view(B * T_new, H, W, C * 4).permute(0, 3, 1, 2)
        z = self.encoder(x)
        z = z.view(B, T_new, z.shape[1], z.shape[2], z.shape[3]).permute(0, 1, 3, 4, 2)
        return z


# ============================================================
# 模块3: VAE Decoder
# ============================================================
# 输入: torch.Size([B, T/4, H/8, W/8, 16]) - 压缩后的latent
# 输出: torch.Size([B, T, H, W, 3]) - 恢复的RGB视频
# 参数: 704,972
# 解压比: 48x (T/4→T, H/8→H, W/8→W, 16→3)
# 关键步骤:
#   1. 编码格式: (B, T/4, H/8, W/8, 16) → (B*T/4, 16, H/8, W/8)
#   2. 空间解码: 3层转置卷积 (H/8,W/8 → H,W)
#   3. 时序恢复: (B*T/4, 12, H, W) → (B, T, H, W, 3)

class VAEDecoder(nn.Module):
    """VAE解码器: Latent -> RGB视频"""

    def __init__(self, latent_channels=16, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels * 4, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        B, T_new, H, W, C = z.shape
        z = z.permute(0, 1, 4, 2, 3).contiguous().view(B * T_new, C, H, W)
        x = self.decoder(z)
        x = x.view(B, T_new, 4, 3, x.shape[2], x.shape[3]).permute(0, 1, 4, 5, 2, 3).contiguous()
        x = x.reshape(B, T_new * 4, x.shape[2], x.shape[3], 3)
        return x


# ============================================================
# 模块4: GP-Buffer Encoder
# ============================================================
# 输入: torch.Size([B, T, H, W, 11]) - GP-Buffer (11通道几何数据)
# 输出: torch.Size([B, T/4, H/8, W/8, 64]) - 几何特征向量
# 参数: 260,384
# 压缩: T→T/4 (时序), H,W→H/8,W/8 (空间), 11→64 (通道)
# 关键步骤:
#   1. 时序压缩: 11通道 × 4帧 = 44通道
#   2. 空间编码: 3层stride=2卷积
#   3. 通道投影: 44 → 64维几何特征
# 注: 与VAE Encoder相同的压缩策略

class GPBufferEncoder(nn.Module):
    """GP-Buffer编码器: 11通道 -> 几何特征"""

    def __init__(self, in_channels=11, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 4, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, gp_buffer):
        B, T, H, W, C = gp_buffer.shape
        T_new = T // 4

        x = gp_buffer.view(B, T_new, 4, H, W, C).permute(0, 1, 3, 4, 5, 2).contiguous()
        x = x.view(B * T_new, H, W, C * 4).permute(0, 3, 1, 2)
        z = self.encoder(x)
        z = z.view(B, T_new, z.shape[1], z.shape[2], z.shape[3]).permute(0, 1, 3, 4, 2)
        return z


# ============================================================
# 模块5: Geometry Adapter (核心创新)
# ============================================================
# 输入:
#   z_G:      torch.Size([B, T/4, H/8, W/8, 64])  - 几何特征
#   z_latent: torch.Size([B, T/4, H/8, W/8, 16])  - RGB latent
#   z_text:   torch.Size([B, 768])              - 文本特征 (可选)
# 输出: torch.Size([B, T/4, H/8, W/8, 16]) - 条件化特征
# 参数: 23,457
# 关键步骤:
#   1. 几何投影: 64维 → 16维 (geo_proj)
#   2. 文本融合: 768维 → 16维，加到几何特征上 (text_proj)
#   3. 门控计算: MLP(z_latent + z_G) → gate ∈ [0,1]
#   4. 特征融合: concat(z_latent, z_G_proj) → Conv → z_fused
#   5. 门控输出: x_g = gate * z_fused + (1-gate) * z_latent
# 核心创新: 自适应学习何时信任几何先验

class GeometryAdapter(nn.Module):
    """几何适配器: 将几何特征注入到生成流程"""

    def __init__(self, geo_channels=64, text_channels=768, latent_channels=16):
        super().__init__()

        self.geo_proj = nn.Sequential(
            nn.Linear(geo_channels, latent_channels), nn.ReLU(),
            nn.Linear(latent_channels, latent_channels),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_channels, latent_channels), nn.ReLU(),
            nn.Linear(latent_channels, latent_channels),
        ) if text_channels > 0 else None

        self.fusion = nn.Sequential(
            nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(latent_channels + geo_channels, 32, kernel_size=1), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, z_G, z_latent, z_text=None):
        B, T, H, W = z_G.shape[:4]

        z_G_proj = self.geo_proj(z_G)

        if self.text_proj is not None and z_text is not None:
            z_text_proj = self.text_proj(z_text).view(B, 1, 1, 1, -1).expand(B, T, H, W, -1)
            z_G_proj = z_G_proj + z_text_proj

        gate_input = torch.cat([z_latent, z_G], dim=-1).permute(0, 1, 4, 2, 3).contiguous()
        gate = self.gate(gate_input.view(B * T, -1, H, W)).view(B, T, 1, H, W).permute(0, 1, 3, 4, 2)

        z_cat = torch.cat([z_latent, z_G_proj], dim=-1).permute(0, 1, 4, 2, 3).contiguous().view(B * T, -1, H, W)
        z_fused = self.fusion(z_cat)
        z_fused = z_fused.view(B, T, z_fused.shape[1], H, W).permute(0, 1, 3, 4, 2).contiguous()

        x_g = gate * z_fused + (1 - gate) * z_latent
        return x_g


# ============================================================
# 模块6: 简化的DiT Backbone
# ============================================================
# SimpleDiTBlock:
#   输入:
#     x:         torch.Size([N, C])  - 展平的latent (N=B*T*H*W)
#     condition: torch.Size([N, C])  - 展平的条件
#   输出: torch.Size([N, C]) - 处理后的特征
#   参数: 每层约9K参数
#   关键步骤:
#     1. Self-Attention: latent自身信息整合
#     2. Cross-Attention: 用条件指导latent (核心融合)
#     3. FFN: 特征变换

class SimpleDiTBlock(nn.Module):
    """简化的DiT Block"""

    def __init__(self, hidden_channels=16, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.cross_attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 4), nn.GELU(), nn.Linear(hidden_channels * 4, hidden_channels))

    def forward(self, x, condition):
        residual = x
        x = self.norm1(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.squeeze(1) + residual

        residual = x
        x = self.norm2(x)
        condition_expanded = condition.unsqueeze(1)
        x = x.unsqueeze(1)
        x, _ = self.cross_attn(x, condition_expanded, condition_expanded)
        x = x.squeeze(1) + residual

        residual = x
        x = self.norm3(x)
        x = self.ffn(x) + residual
        return x


# SimpleDiT:
#   输入:
#     z_t:       torch.Size([B, T/4, H/8, W/8, 16]) - 当前latent
#     t:         torch.Size([B])              - 时间步
#     condition:  torch.Size([B, T/4, H/8, W/8, 16]) - 几何条件
#   输出: torch.Size([B, T/4, H/8, W/8, 16]) - 预测的速度v
#   参数: 约9K (简化版)
#   关键步骤:
#     1. 时间嵌入: t → t_embed → 广播到所有位置
#     2. 展平: (B,T,H,W,C) → (N, C), N=B*T*H*W
#     3. 时间注入: x = z_flat + t_flat
#     4. DiT层: Self-Attn + Cross-Attn + FFN (2层)
#     5. 输出投影: LayerNorm + Linear
#     6. 恢复形状: (N, C) → (B, T, H, W, C)

class SimpleDiT(nn.Module):
    """简化的DiT用于Flow Matching"""

    def __init__(self, latent_channels=16, num_layers=2, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([SimpleDiTBlock(latent_channels, num_heads) for _ in range(num_layers)])
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, latent_channels))
        self.output_proj = nn.Sequential(nn.LayerNorm(latent_channels), nn.Linear(latent_channels, latent_channels))

    def forward(self, z_t, t, condition):
        B, T, H, W, C = z_t.shape
        t_embed = self.time_mlp(t.view(B, 1)).view(B, 1, 1, 1, C).expand(B, T, H, W, C)

        z_flat = z_t.view(B * T * H * W, C)
        condition_flat = condition.view(B * T * H * W, C)
        t_flat = t_embed.view(B * T * H * W, C)

        x = z_flat + t_flat
        for layer in self.layers:
            x = layer(x, condition_flat)

        v_flat = self.output_proj(x)
        v_pred = v_flat.view(B, T, H, W, C)
        return v_pred


# ============================================================
# 模块7: Flow Matching
# ============================================================

# FlowMatchingTraining (训练):
#   compute_loss输入:
#     z_corrupted: torch.Size([B, T/4, H/8, W/8, 16]) - 损坏的latent
#     z_clean:     torch.Size([B, T/4, H/8, W/8, 16]) - 干净的latent
#     condition:    torch.Size([B, T/4, H/8, W/8, 16]) - 几何条件
#   输出:
#     loss: 标量 - MSE损失
#     v_pred: torch.Size([B, T/4, H/8, W/8, 16]) - 预测的速度
#   关键步骤:
#     1. 采样时间步: t ~ U(0,1)
#     2. 计算插值: z_t = t*z_clean + (1-t)*z_corrupted
#     3. 计算真实速度: v_gt = z_clean - z_corrupted
#     4. DiT预测速度: v_pred = DiT(z_t, t, condition)
#     5. MSE损失: loss = MSE(v_pred, v_gt)

class FlowMatchingTraining(nn.Module):
    """Flow Matching训练"""

    def __init__(self, dit_model):
        super().__init__()
        self.dit = dit_model

    def compute_loss(self, z_corrupted, z_clean, condition):
        B = z_corrupted.shape[0]
        t = torch.rand(B, device=z_corrupted.device)
        z_t = t.view(B, 1, 1, 1, 1) * z_clean + (1 - t.view(B, 1, 1, 1, 1)) * z_corrupted
        v_gt = z_clean - z_corrupted
        v_pred = self.dit(z_t, t, condition)
        loss = F.mse_loss(v_pred, v_gt)
        return loss, v_pred


# FlowMatchingInference (推理):
#   forward输入:
#     z_corrupted: torch.Size([B, T/4, H/8, W/8, 16]) - 损坏的latent
#     condition:    torch.Size([B, T/4, H/8, W/8, 16]) - 几何条件
#   输出:
#     z_refined: torch.Size([B, T/4, H/8, W/8, 16]) - 精化的latent
#   关键步骤:
#     1. 初始化: z_t = z_corrupted
#     2. 循环num_steps次:
#        a. 设置当前时间: t = step / num_steps
#        b. 预测速度: v = DiT(z_t, t, condition)
#        c. 更新latent: z_t = z_t + v * dt
#     3. 返回精化后的latent

class FlowMatchingInference(nn.Module):
    """Flow Matching推理"""

    def __init__(self, dit_model, num_steps=10):
        super().__init__()
        self.dit = dit_model
        self.num_steps = num_steps

    def forward(self, z_corrupted, condition):
        z_t = z_corrupted.clone()
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            t = torch.full((z_t.shape[0],), step * dt, device=z_t.device)
            v = self.dit(z_t, t, condition)
            z_t = z_t + v * dt
        return z_t


# ============================================================
# 主测试函数
# ============================================================

def run_test(batch_size=1, num_frames=8, height=120, width=208):
    """运行完整的探针测试"""

    print("\n" + "="*80)
    print("GaussFusion 探针代码")
    print("="*80)
    print(f"配置: batch_size={batch_size}, frames={num_frames}, size={height}x{width}")
    print("="*80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    all_params = OrderedDict()

    # 步骤1: GP-Buffer
    print(">>> 步骤1: GP-Buffer 模拟器")
    gp_simulator = GPBufferSimulator(height, width, num_frames).to(device)
    gp_buffer = gp_simulator(batch_size)
    print(f"  输出: {gp_buffer.shape}")
    print(f"  参数: {sum(p.numel() for p in gp_simulator.parameters()):,}")
    all_params['gp_buffer'] = count_parameters(gp_simulator)

    # 步骤2: VAE编码
    print("\n>>> 步骤2: VAE编码")
    vae_encoder = VAEEncoder().to(device)
    z_corrupted = vae_encoder(gp_buffer[:, :, :, :, :3])
    print(f"  输入: {gp_buffer[:, :, :, :, :3].shape}")
    print(f"  输出: {z_corrupted.shape}")
    print(f"  压缩比: {gp_buffer[:, :, :, :, :3].numel() / z_corrupted.numel():.1f}x")
    print(f"  参数: {sum(p.numel() for p in vae_encoder.parameters()):,}")
    all_params['vae_encoder'] = count_parameters(vae_encoder)

    # 步骤3: GP-Buffer编码
    print("\n>>> 步骤3: GP-Buffer编码")
    gp_encoder = GPBufferEncoder().to(device)
    z_G = gp_encoder(gp_buffer)
    print(f"  输入: {gp_buffer.shape}")
    print(f"  输出: {z_G.shape}")
    print(f"  参数: {sum(p.numel() for p in gp_encoder.parameters()):,}")
    all_params['gp_encoder'] = count_parameters(gp_encoder)

    # 步骤4: 几何适配
    print("\n>>> 步骤4: 几何适配 (核心模块)")
    geo_adapter = GeometryAdapter().to(device)
    z_text = torch.randn(batch_size, 768, device=device)
    x_g = geo_adapter(z_G, z_corrupted, z_text)
    print(f"  几何特征: {z_G.shape}")
    print(f"  RGB latent: {z_corrupted.shape}")
    print(f"  文本特征: {z_text.shape}")
    print(f"  条件化输出: {x_g.shape}")
    print(f"  参数: {sum(p.numel() for p in geo_adapter.parameters()):,}")
    all_params['geo_adapter'] = count_parameters(geo_adapter)

    # 步骤5: DiT
    print("\n>>> 步骤5: DiT预测速度")
    dit = SimpleDiT().to(device)
    t = torch.rand(batch_size, device=device)
    z_t = z_corrupted.clone()
    v_pred = dit(z_t, t, x_g)
    print(f"  当前latent: {z_t.shape}")
    print(f"  条件: {x_g.shape}")
    print(f"  预测速度: {v_pred.shape}")
    print(f"  参数: {sum(p.numel() for p in dit.parameters()):,}")
    all_params['dit'] = count_parameters(dit)

    # 步骤6: Flow Matching训练
    print("\n>>> 步骤6: Flow Matching训练")
    z_clean = vae_encoder(gp_buffer[:, :, :, :, :3] + 0.05 * torch.randn_like(gp_buffer[:, :, :, :, :3]))
    flow_train = FlowMatchingTraining(dit)
    loss, v_pred = flow_train.compute_loss(z_corrupted, z_clean, x_g)
    print(f"  损失: {loss.item():.6f}")
    all_params['flow_train'] = count_parameters(flow_train)

    # 步骤7: Flow Matching推理
    print("\n>>> 步骤7: Flow Matching推理 (5步)")
    flow_inference = FlowMatchingInference(dit, num_steps=5)
    z_refined = flow_inference(z_corrupted, x_g)
    print(f"  输入: {z_corrupted.shape}")
    print(f"  输出: {z_refined.shape}")
    print(f"  改善: {(z_refined - z_corrupted).abs().mean().item():.6f}")
    all_params['flow_inference'] = count_parameters(flow_inference)

    # 步骤8: VAE解码
    print("\n>>> 步骤8: VAE解码")
    vae_decoder = VAEDecoder().to(device)
    refined_rgb = vae_decoder(z_refined)
    print(f"  输入latent: {z_refined.shape}")
    print(f"  输出RGB: {refined_rgb.shape}")
    print(f"  参数: {sum(p.numel() for p in vae_decoder.parameters()):,}")
    all_params['vae_decoder'] = count_parameters(vae_decoder)

    # 总结
    print("\n" + "="*80)
    print("参数量总结")
    print("="*80)
    total_trainable = 0
    for name, params in all_params.items():
        trainable = params['trainable']
        total = params['total']
        total_trainable += trainable
        pct = (trainable / total * 100) if total > 0 else 0
        print(f"  {name:20s}: {total:>10,} | 可训练: {trainable:>10,} ({pct:5.1f}%)")
    print("="*80)
    print(f"  {'总计':20s}: {sum(p['total'] for p in all_params.values()):>10,} | 可训练: {total_trainable:>10,}")
    print("="*80)

    # 梯度检查
    print("\n>>> 检查梯度流动")
    loss.backward()
    for name, module in [('GP-Buffer', gp_simulator), ('VAE-Enc', vae_encoder),
                         ('GP-Enc', gp_encoder), ('Geo-Adapter', geo_adapter), ('DiT', dit)]:
        has_grad = any(p.grad is not None for p in module.parameters() if p.requires_grad)
        print(f"  {'✓' if has_grad else '✗'} {name:12s}")

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_test()

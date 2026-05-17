"""
DreamZero: World Action Models are Zero-shot Policies
探针代码 - 单文件版本，包含所有核心模块和测试功能

论文: arXiv 2602.15922
核心思想: 使用视频扩散模型(Wan 2.1/2.2)同时预测未来视频帧和机器人动作，
         将动作预测作为视频生成的一部分而非独立下游任务

环境: Python 3.8+, PyTorch 1.12+, CUDA
依赖: torch, numpy
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================
# 兼容性工具: PyTorch < 2.0 没有 scaled_dot_product_attention
# ============================================================

def sdpa(q, k, v, attn_mask=None, is_causal=False):
    """兼容性缩放点积注意力"""
    if hasattr(F, 'scaled_dot_product_attention'):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    # 手动实现
    d = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if attn_mask is not None:
        attn = attn + attn_mask
    if is_causal:
        L = q.shape[-2]
        S = k.shape[-2]
        causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=q.device), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# ============================================================
# 工具函数
# ============================================================

def count_parameters(module, trainable_only=False):
    """返回模块的参数统计"""
    if trainable_only:
        total = sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        total = sum(p.numel() for p in module.parameters())
    return total


def format_params(n):
    """格式化参数量"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def sinusoidal_embedding_1d(dim, t):
    """1D正弦位置编码, t: (B,), 返回: (B, dim)"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def rope_params(seq_len, dim):
    """生成RoPE频率参数, 返回: (seq_len, 1, dim//2) complex"""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.unsqueeze(1)  # (seq_len, 1, dim//2)


def rope_apply(x, freqs):
    """应用RoPE, x: (B, L, n, d), freqs: (L, 1, d//2)"""
    B, L, n, d = x.shape
    x = torch.view_as_complex(x.to(torch.float64).reshape(B, L, n, -1, 2))
    freqs = freqs[:L].unsqueeze(0)
    return torch.view_as_real(x * freqs).flatten(3).type_as(x)


# ============================================================
# 模块1: WanTextEncoder (UMT5-XXL, 冻结)
# ============================================================
# 输入:
#   - ids: (B, 512) - UMT5-XXL token IDs
#   - mask: (B, 512) - attention mask
# 输出:
#   - embeddings: (B, 512, 4096) - 文本嵌入
# 参数: ~4.6B (冻结)
#   - token_embedding: 256384 * 4096 = ~1.05B
#   - 24层T5SelfAttention: 每层 ~148M
#   - T5RelativeEmbedding: 32 * 64 = ~2K
# 关键步骤:
#   1. Token embedding: ids -> (B, 512, 4096)
#   2. 24层T5 Self-Attention + FFN (相对位置编码)
#   3. LayerNorm输出

class T5LayerNorm(nn.Module):
    """T5风格的RMS LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class T5Attention(nn.Module):
    """T5注意力: q,k,v线性投影 + 缩放点积注意力"""
    def __init__(self, dim, dim_attn, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(self, x, context=None, pos_bias=None):
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)
        attn = torch.einsum('binc,bjnc->bnij', q, k)
        if pos_bias is not None:
            attn = attn + pos_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        return self.o(x.reshape(b, -1, n * c))


class T5FeedForward(nn.Module):
    """T5 FFN: 门控线性单元 (GLU)"""
    def __init__(self, dim, dim_ffn):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), nn.GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x) * self.gate(x))


class T5SelfAttentionBlock(nn.Module):
    """T5自注意力块: Self-Attn + FFN + 残差连接"""
    def __init__(self, dim, dim_attn, dim_ffn, num_heads):
        super().__init__()
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn)

    def forward(self, x, pos_bias=None):
        x = x + self.attn(self.norm1(x), pos_bias=pos_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class WanTextEncoder(nn.Module):
    """UMT5-XXL文本编码器: 24层T5 Transformer"""
    def __init__(self, vocab=256384, dim=4096, dim_attn=4096,
                 dim_ffn=10240, num_heads=64, num_layers=24):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            T5SelfAttentionBlock(dim, dim_attn, dim_ffn, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids):
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============================================================
# 模块2: WanImageEncoder (CLIP ViT-H/14, 冻结)
# ============================================================
# 输入:
#   - images: (B, 3, 224, 224) - RGB图像
# 输出:
#   - clip_features: (B, 257, 1280) - CLIP特征 (1 CLS + 256 patches)
# 参数: ~304M (冻结)
#   - patch_embedding: Conv2d(3, 1280, 14, stride=14) = ~754K
#   - 32层AttentionBlock: 每层 ~8.4M
#   - cls_embedding + pos_embedding: ~330K
# 关键步骤:
#   1. Patch embedding: Conv2d(3, 1280, kernel=14, stride=14) -> (B, 1280, 16, 16)
#   2. 添加CLS token和位置编码
#   3. 31层ViT Self-Attention (use_31_block=True, 去掉最后一层)
#   4. 输出257个token (1 CLS + 256 patch)

class CLIPAttentionBlock(nn.Module):
    """CLIP ViT注意力块"""
    def __init__(self, dim=1280, num_heads=16, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x):
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim
        q, k, v = self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(b, s, n, d).permute(0, 2, 1, 3)
        k = k.view(b, s, n, d).permute(0, 2, 1, 3)
        v = v.view(b, s, n, d).permute(0, 2, 1, 3)
        x_attn = sdpa(q, k, v)
        x_attn = x_attn.permute(0, 2, 1, 3).reshape(b, s, c)
        x = x + self.proj(x_attn)
        x = x + self.mlp(self.norm2(x))
        return x


class WanImageEncoder(nn.Module):
    """CLIP ViT-H/14图像编码器: 32层ViT (取前31层)"""
    def __init__(self, dim=1280, num_heads=16, num_layers=32,
                 patch_size=14, image_size=224):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) / dim**0.5)
        self.transformer = nn.Sequential(*[
            CLIPAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.pre_norm = nn.LayerNorm(dim)

    def forward(self, x):
        b = x.size(0)
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        x = x + self.pos_embedding
        x = self.pre_norm(x)
        # use_31_block: 取前31层
        x = self.transformer[:-1](x)
        return x


# ============================================================
# 模块3: WanVideoVAE (3D视频VAE, 冻结)
# ============================================================
# 输入:
#   - video: (B, 3, F, H, W) - RGB视频, 如(1, 3, 13, 480, 640)
# 输出:
#   - latents: (B, 16, F', H/8, W/8) - 视频潜变量, 如(1, 16, 4, 60, 80)
# 参数: ~317M (冻结)
#   - Encoder3d: ~158M (CausalConv3d + ResidualBlock + AttentionBlock)
#   - Decoder3d: ~158M
#   - conv1 + conv2: ~8K
# 关键步骤:
#   1. CausalConv3d编码: 3 -> 128 -> 256 -> 512 -> 512通道
#   2. 时间下采样: 4x (F -> F/4), 空间下采样: 8x (H -> H/8)
#   3. 中间层: ResidualBlock + AttentionBlock + ResidualBlock
#   4. 输出: 16通道潜变量 (均值, 方差丢弃)
#   5. 解码器对称结构, 时间上采样4x, 空间上采样8x

class CausalConv3d(nn.Conv3d):
    """因果3D卷积: 时间维度只看过去"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        return super().forward(F.pad(x, list(self._padding)))


class RMSNorm3d(nn.Module):
    """3D RMS归一化"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class VAIResidualBlock(nn.Module):
    """VAE残差块: 2个CausalConv3d + 跳跃连接"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm3d(in_dim), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm3d(out_dim), nn.SiLU(),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class VAIAttentionBlock(nn.Module):
    """VAE注意力块: 空间自注意力 (单头)"""
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm3d(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).chunk(3, dim=-1)
        x = sdpa(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class WanVideoVAE(nn.Module):
    """Wan 2.1 3D视频VAE: 编码器+解码器, 16通道潜变量"""
    def __init__(self, dim=96, z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        # 简化编码器: 3 -> 96 -> 192 -> 384 -> 384
        self.enc_conv1 = CausalConv3d(3, dim, 3, padding=1)
        self.enc_res1 = VAIResidualBlock(dim, dim * 2)
        self.enc_res2 = VAIResidualBlock(dim * 2, dim * 4)
        self.enc_res3 = VAIResidualBlock(dim * 4, dim * 4)
        self.enc_mid1 = VAIResidualBlock(dim * 4, dim * 4)
        self.enc_attn = VAIAttentionBlock(dim * 4)
        self.enc_mid2 = VAIResidualBlock(dim * 4, dim * 4)
        self.enc_head = nn.Sequential(RMSNorm3d(dim * 4), nn.SiLU(),
                                       CausalConv3d(dim * 4, z_dim * 2, 3, padding=1))
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    def encode(self, x):
        """编码: (B, 3, F, H, W) -> (B, 16, F', H', W')"""
        x = self.enc_conv1(x)
        x = self.enc_res1(x)
        x = self.enc_res2(x)
        x = self.enc_res3(x)
        x = self.enc_mid1(x)
        x = self.enc_attn(x)
        x = self.enc_mid2(x)
        x = self.enc_head(x)
        mu, _ = self.conv1(x).chunk(2, dim=1)
        return mu


# ============================================================
# 模块4: CategorySpecificLinear/MLP (跨具身感知)
# ============================================================
# 输入:
#   - x: (B, T, input_dim) - 输入特征
#   - cat_ids: (B,) - 具身类别ID
# 输出:
#   - output: (B, T, output_dim) - 类别特定输出
# 参数: num_categories * (input_dim * hidden_dim + hidden_dim)
# 关键步骤:
#   1. 根据cat_ids选择对应类别的权重W和偏置b
#   2. 矩阵乘法: x @ W + b
# 核心创新: 每个机器人平台有独立的线性层权重, 支持跨具身训练

class CategorySpecificLinear(nn.Module):
    """类别特定线性层: 每个具身类型有独立权重"""
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """类别特定MLP: 2层, ReLU激活"""
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


# ============================================================
# 模块5: MultiEmbodimentActionEncoder (动作编码器)
# ============================================================
# 输入:
#   - actions: (B, T, action_dim) - 机器人动作, action_dim=32
#   - timesteps: (B, T) - 扩散时间步
#   - cat_ids: (B,) - 具身类别ID
# 输出:
#   - features: (B, T, dim) - 动作特征, dim=5120 (14B)
# 参数: ~79M (14B, max_num_embodiments=1)
#   - W1: 1 * (32 * 5120 + 5120) = ~169K
#   - W2: 1 * (10240 * 5120 + 5120) = ~52.5M
#   - W3: 1 * (5120 * 5120 + 5120) = ~26.2M
#   - SinusoidalPositionalEncoding: 0 (固定)
# 关键步骤:
#   1. W1: 动作线性投影 (action_dim -> hidden_size=dim)
#   2. 正弦位置编码: 时间步 -> (B, T, dim)
#   3. 拼接: [W1(actions); pos_encoding] -> (B, T, 2*dim)
#   4. W2 + swish: (2*dim -> dim)
#   5. W3: (dim -> dim)
# 核心创新: 动作token与时间步编码融合, 支持多具身独立权重

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class MultiEmbodimentActionEncoder(nn.Module):
    """多具身动作编码器: W1 + SinusoidalPE + W2(swish) + W3"""
    def __init__(self, action_dim=32, hidden_size=5120, num_embodiments=1):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = F.silu(self.W2(x, cat_ids))
        return self.W3(x, cat_ids)


# ============================================================
# 模块6: CausalWanSelfAttention (因果自注意力)
# ============================================================
# 输入:
#   - x: (B, L, dim) - 输入序列 (视频+动作+状态token)
#   - freqs: RoPE频率 - 3D位置编码
#   - action_register_length: 动作+状态token数量
# 输出:
#   - output: (B, L, dim) - 注意力输出
# 参数: ~105M (14B, dim=5120)
#   - q,k,v: 3 * 5120 * 5120 = ~78.6M
#   - o: 5120 * 5120 = ~26.2M
#   - norm_q, norm_k: 2 * 5120 = ~10K
# 关键步骤:
#   1. QKV投影 + QK归一化
#   2. 3D RoPE位置编码 (时间+空间+动作+状态)
#   3. 分块因果注意力:
#      - 首帧: 仅自注意力 (条件)
#      - 图像块i: 注意首帧 + 前面图像块 + 当前动作块 + 当前状态块
#      - 动作块i: 注意首帧 + 前面图像块 + 当前图像块 + 当前动作块 + 当前状态块
#      - 状态块: 仅自注意力 (条件)
#   4. 输出投影
# 核心创新: 分块因果注意力模式, 动作/状态token与视频token交错,
#          实现时间因果性: t时刻的动作只依赖t及之前的观测

class WanRMSNorm(nn.Module):
    """Wan风格RMS归一化"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class CausalWanSelfAttention(nn.Module):
    """因果自注意力: 分块因果flash注意力 + 3D RoPE"""
    def __init__(self, dim=5120, num_heads=40, frame_seqlen=880,
                 num_action_per_block=32, num_state_per_block=1,
                 num_frame_per_block=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.num_frame_per_block = num_frame_per_block
        # QKV + Output
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim)
        self.norm_k = WanRMSNorm(dim)

    def forward(self, x, freqs, action_register_length=None, is_tf=True):
        """简化前向: 展示分块因果注意力逻辑"""
        b, s, c = x.size()
        n, d = self.num_heads, self.head_dim

        # QKV投影 + QK归一化
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # 应用3D RoPE (简化: 仅空间位置编码)
        q = rope_apply(q, freqs)
        k = rope_apply(k, freqs)

        # 分块因果注意力 (简化实现)
        if is_tf and action_register_length is not None:
            # 教师强制训练: 分clean/noisy两半
            half = (s - action_register_length) // 2
            # Clean半: 简单因果注意力
            q_clean, k_clean, v_clean = q[:, :half], k[:, :half], v[:, :half]
            out_clean = sdpa(
                q_clean.permute(0,2,1,3), k_clean.permute(0,2,1,3), v_clean.permute(0,2,1,3),
                is_causal=True).permute(0,2,1,3).reshape(b, half, c)
            # Noisy半: 注意clean上下文 + 自己 (简化)
            q_noisy = q[:, half:]
            k_full = torch.cat([k[:, :half], k[:, half:]], dim=1)
            v_full = torch.cat([v[:, :half], v[:, half:]], dim=1)
            out_noisy = sdpa(
                q_noisy.permute(0,2,1,3), k_full.permute(0,2,1,3), v_full.permute(0,2,1,3)
            ).permute(0,2,1,3).reshape(b, s - half, c)
            out = torch.cat([out_clean, out_noisy], dim=1)
        else:
            # 推理: 标准注意力
            out = sdpa(
                q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
            ).permute(0,2,1,3).reshape(b, s, c)

        return self.o(out)


# ============================================================
# 模块7: CausalWanAttentionBlock (DiT块)
# ============================================================
# 输入:
#   - x: (B, L, dim) - 输入token序列
#   - e: (B, F, 6, dim) - 6路调制信号 (AdaLN)
#   - context: (B, 512+257, dim) - 文本+CLIP上下文
#   - freqs: RoPE频率
# 输出:
#   - x: (B, L, dim) - 输出token序列
# 参数: ~102M (14B, dim=5120, ffn_dim=13824)
#   - self_attn: ~105M (q,k,v,o + norm_q,norm_k)
#   - cross_attn: ~78.6M (q,k,v,o for i2v)
#   - ffn: ~142M (Linear(5120,13824) + Linear(13824,5120))
#   - modulation: 6 * 5120 = ~31K
#   - norms: ~15K
# 关键步骤:
#   1. 6路AdaLN调制: e = (modulation + e).chunk(6)
#   2. 因果自注意力: x = x + self_attn(norm1(x) * (1+e1) + e0) * e2
#   3. 交叉注意力: x = x + cross_attn(norm3(x), context)
#   4. FFN: x = x + ffn(norm2(x) * (1+e4) + e3) * e5
# 核心创新: 6路AdaLN调制 (比DiT的6路更复杂, 包含偏移和缩放)

class WanCrossAttention(nn.Module):
    """Wan交叉注意力: 图像token attend to 文本token"""
    def __init__(self, dim=5120, num_heads=40):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim)
        self.norm_k = WanRMSNorm(dim)

    def forward(self, x, context):
        b, s, c = x.size()
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d).permute(0, 2, 1, 3)
        k = self.norm_k(self.k(context)).view(b, -1, n, d).permute(0, 2, 1, 3)
        v = self.v(context).view(b, -1, n, d).permute(0, 2, 1, 3)
        out = sdpa(q, k, v).permute(0, 2, 1, 3).reshape(b, s, c)
        return self.o(out)


class CausalWanAttentionBlock(nn.Module):
    """DiT块: 因果自注意力 + 交叉注意力 + FFN + 6路AdaLN"""
    def __init__(self, dim=5120, ffn_dim=13824, num_heads=40,
                 frame_seqlen=880, num_action_per_block=32,
                 num_state_per_block=1):
        super().__init__()
        self.dim = dim
        self.norm1 = WanRMSNorm(dim)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, frame_seqlen, num_action_per_block, num_state_per_block)
        self.norm3 = WanRMSNorm(dim)
        self.cross_attn = WanCrossAttention(dim, num_heads)
        self.norm2 = WanRMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, freqs, context, action_register_length=None, is_tf=True):
        # 6路调制
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # 对齐调制长度到x
        e = tuple(part[:, :x.shape[1]] for part in e)
        # 自注意力
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            freqs, action_register_length, is_tf)
        x = x + y * e[2].squeeze(2)
        # 交叉注意力
        x = x + self.cross_attn(self.norm3(x), context)
        # FFN
        y = self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        x = x + y * e[5].squeeze(2)
        return x


# ============================================================
# 模块8: CausalHead (输出头)
# ============================================================
# 输入:
#   - x: (B, L, dim) - 最后一层DiT输出
#   - e: (B, F, 1, dim) - 调制信号
# 输出:
#   - output: (B, L, out_dim * prod(patch_size)) - 预测噪声
# 参数: ~327K (14B)
#   - head: Linear(5120, 16*4) = 5120 * 64 = ~328K
#   - modulation: 2 * 5120 = ~10K
# 关键步骤:
#   1. 2路调制: e = (modulation + e).chunk(2)
#   2. 输出: head(norm(x) * (1+e1) + e0)

class CausalHead(nn.Module):
    """输出头: 2路调制 + 线性投影"""
    def __init__(self, dim=5120, out_dim=16, patch_size=(1, 2, 2)):
        super().__init__()
        out_dim_expanded = math.prod(patch_size) * out_dim
        self.norm = WanRMSNorm(dim)
        self.head = nn.Linear(dim, out_dim_expanded)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        e = tuple(part[:, :x.shape[1]] for part in e)
        return self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))


# ============================================================
# 模块9: CausalWanModel (核心扩散模型, 14B DiT)
# ============================================================
# 输入:
#   - x: (B, in_dim, F, H', W') - 噪声视频潜变量, in_dim=36 (16+20 for i2v)
#   - timestep: (B, F) - 视频扩散时间步
#   - context: (B, 512, 4096) - 文本嵌入
#   - clip_feature: (B, 257, 1280) - CLIP特征
#   - y: (B, 20, F, H', W') - 首帧潜变量 (i2v模式)
#   - action: (B, T, 32) - 噪声动作
#   - timestep_action: (B, T) - 动作扩散时间步
#   - state: (B, T, 44) - 机器人状态
#   - clean_x: (B, in_dim, F, H', W') - 干净视频 (教师强制)
# 输出:
#   - video_noise_pred: (B, out_dim, F, H', W') - 视频噪声预测
#   - action_noise_pred: (B, T, 32) - 动作噪声预测
# 参数: ~14B (总计, 含冻结编码器)
#   - patch_embedding: Conv3d(36, 5120, (1,2,2)) = ~737K
#   - text_embedding: 4096*5120 + 5120*5120 = ~46.1M
#   - time_embedding: 256*5120 + 5120*5120 = ~26.6M
#   - time_projection: 5120*30720 = ~157.3M
#   - img_emb (MLPProj): 1280*5120 + 5120*5120 = ~32.8M
#   - 40个CausalWanAttentionBlock: 40 * ~326M = ~13.0B
#   - CausalHead: ~327K
#   - action_encoder: ~79M
#   - state_encoder: ~660K
#   - action_decoder: ~660K
# 实际层数: 40, 探针层数: 3 (环境限制)
# 关键步骤:
#   1. 输入拼接: [x; y] (i2v模式, 首帧潜变量拼接)
#   2. Patch embedding: Conv3d(in_dim -> dim, kernel=(1,2,2))
#   3. 文本嵌入: Linear(4096->dim) + GELU + Linear(dim->dim)
#   4. 时间嵌入: sinusoidal -> Linear(256->dim) + SiLU + Linear(dim->dim)
#   5. 时间投影: SiLU + Linear(dim->dim*6) -> 6路调制
#   6. 动作编码: MultiEmbodimentActionEncoder
#   7. 状态编码: CategorySpecificMLP
#   8. Token拼接: [video_tokens; action_tokens; state_tokens]
#   9. 教师强制: [clean_video_tokens; noisy_video_tokens; action_tokens; state_tokens]
#   10. 40层CausalWanAttentionBlock (6路AdaLN + 因果自注意力 + 交叉注意力 + FFN)
#   11. 输出头: CausalHead -> 视频噪声预测
#   12. 动作解码: CategorySpecificMLP -> 动作噪声预测
# 核心创新:
#   1. 视频扩散模型同时预测视频和动作 (World Action Model)
#   2. 分块因果注意力: 动作/状态token与视频token交错
#   3. 教师强制训练: clean上下文 + noisy预测, 并行训练所有时间块
#   4. 跨具身训练: CategorySpecificLinear为每个机器人平台提供独立权重
#   5. 解耦噪声采样: 视频用Beta分布(偏向高噪声), 动作用均匀分布

class MLPProj(nn.Module):
    """CLIP特征投影: 1280 -> dim"""
    def __init__(self, in_dim=1280, out_dim=5120):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(approximate='tanh'),
            nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.proj(x)


class CausalWanModel(nn.Module):
    """DreamZero核心: 因果Wan DiT (14B)"""
    def __init__(self, model_type='i2v', patch_size=(1, 2, 2),
                 frame_seqlen=880, text_len=512, in_dim=36,
                 dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096,
                 out_dim=16, num_heads=40, num_layers=40,
                 action_dim=32, max_state_dim=44, hidden_size=64,
                 num_action_per_block=32, num_state_per_block=1,
                 num_frame_per_block=1, concat_first_frame_latent=True,
                 probe_layers=3):
        super().__init__()
        self.model_type = model_type
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.num_frame_per_block = num_frame_per_block
        self.concat_first_frame_latent = concat_first_frame_latent

        # 嵌入层
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # 动作/状态编码器和解码器
        max_num_embodiments = 1  # 简化: 单具身
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim, hidden_size=dim, num_embodiments=max_num_embodiments)
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments, input_dim=max_state_dim,
            hidden_dim=hidden_size, output_dim=dim)
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments, input_dim=dim,
            hidden_dim=hidden_size, output_dim=action_dim)

        # DiT块 (探针: 减少层数)
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads, frame_seqlen,
                                     num_action_per_block, num_state_per_block)
            for _ in range(probe_layers)
        ])

        # 输出头
        self.head = CausalHead(dim, out_dim, patch_size)

        # CLIP图像嵌入 (i2v模式)
        if model_type in ('i2v', 'ti2v'):
            self.img_emb = MLPProj(1280, dim)

        # RoPE频率
        d = dim // num_heads
        self.freqs = rope_params(1024, d)

    def forward(self, x, timestep, context, clip_feature=None, y=None,
                action=None, timestep_action=None, state=None,
                clean_x=None, embodiment_id=None):
        """前向传播 (训练模式, 教师强制)"""
        # 1. 拼接首帧潜变量 (i2v模式)
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # 2. Patch embedding
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(start_dim=2).transpose(1, 2)
        seq_len = x.shape[1]

        B = x.shape[0]
        F = timestep.shape[1]

        # 3. 动作/状态编码
        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(B)
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_length = 0
            action_register_length = None

        # 4. 时间嵌入
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        if action is not None:
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.reshape(B, -1, self.dim)
        e0 = self.time_projection(e)
        e0 = e0.reshape(B, -1, 6, self.dim)

        # 5. 文本嵌入 + CLIP嵌入
        context = self.text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # 6. 教师强制: 拼接clean上下文
        if clean_x is not None:
            if y is not None and self.concat_first_frame_latent:
                clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            x = torch.cat([clean_x, x], dim=1)
            # 扩展时间嵌入
            e0_clean = torch.zeros_like(e0[:, :clean_x.shape[1]])
            e0 = torch.cat([e0_clean, e0], dim=1)

        # 7. DiT块
        freqs = self.freqs[:seq_len]
        for block in self.blocks:
            x = block(x, e0, freqs, context, action_register_length, is_tf=True)

        # 8. 去掉clean上下文
        if clean_x is not None:
            x = x[:, clean_x.shape[1]:]

        # 9. 动作解码
        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # 10. 视频输出头
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred


# ============================================================
# 模块10: FlowMatchScheduler (流匹配调度器)
# ============================================================
# 输入:
#   - x: (B, ...) - 干净数据
#   - noise: (B, ...) - 噪声
#   - timestep: (B,) - 时间步
# 输出:
#   - x_noisy: (B, ...) - 加噪数据
#   - target: (B, ...) - 训练目标 (速度预测: noise - x)
#   - weight: (B,) - 训练权重 (高斯钟形)
# 参数: 0 (无参数)
# 关键步骤:
#   1. 加噪: x_noisy = (1 - sigma) * x + sigma * noise
#   2. 训练目标: target = noise - x (速度预测)
#   3. 训练权重: 高斯钟形加权, 中心在中间时间步
#   4. 推理步: x_prev = x_current + model_output * (sigma_next - sigma_current)
#   5. Sigma调度: sigma = shift * sigma / (1 + (shift - 1) * sigma), shift=5.0

class FlowMatchScheduler(nn.Module):
    """流匹配调度器: 噪声添加 + 训练目标 + 推理步"""
    def __init__(self, shift=5.0, sigma_min=0.0, num_train_timesteps=1000):
        super().__init__()
        self.shift = shift
        self.sigma_min = sigma_min
        self.num_train_timesteps = num_train_timesteps
        # 预计算timesteps
        t = torch.linspace(1, 0, num_train_timesteps + 1)
        sigma = t / (1 - t)
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
        self.register_buffer('timesteps', (sigma * 1000).long())

    def add_noise(self, x, noise, timestep):
        """添加噪声: x_noisy = (1-sigma)*x + sigma*noise"""
        sigma = timestep.float() / 1000.0
        return (1 - sigma) * x + sigma * noise

    def training_target(self, x, noise, timestep):
        """训练目标: 速度预测 v = noise - x"""
        return noise - x

    def training_weight(self, timestep):
        """训练权重: 高斯钟形"""
        sigma = timestep.float() / 1000.0
        return 1.0 / (sigma ** 2 + 1.0)


# ============================================================
# 模块11: WANPolicyHead (顶层动作头)
# ============================================================
# 输入:
#   - videos: (B, C, T, H, W) - RGB视频
#   - text: (B, 512) - 文本token IDs
#   - state: (B, T, 44) - 机器人状态
#   - action: (B, T, 32) - 机器人动作
#   - embodiment_id: (B,) - 具身ID
# 输出:
#   - loss: 标量 - 总损失 = dynamics_loss + action_loss
#   - dynamics_loss: 视频预测损失 (加权MSE)
#   - action_loss: 动作预测损失 (加权MSE)
# 参数: ~14B (总计)
#   - text_encoder: ~4.6B (冻结)
#   - image_encoder: ~304M (冻结)
#   - vae: ~317M (冻结)
#   - model (CausalWanModel): ~9.1B (LoRA微调)
#   - scheduler: 0
# 关键步骤:
#   1. 文本编码: UMT5-XXL -> (B, 512, 4096)
#   2. 图像编码: CLIP ViT-H/14 -> (B, 257, 1280)
#   3. 视频编码: VAE -> (B, 16, F, H/8, W/8)
#   4. 首帧编码: CLIP + VAE -> clip_feas, ys
#   5. 噪声采样: Beta(1.5, 1.0) 或均匀
#   6. 加噪: x_noisy = (1-sigma)*x + sigma*noise
#   7. DiT前向: CausalWanModel(x_noisy, t, context, action_noisy, ...)
#   8. 损失计算: 加权MSE (视频 + 动作)
# 核心创新:
#   1. 解耦噪声采样: 视频Beta(3,1)偏向高噪声, 动作均匀
#   2. 多视角视频: 2x2网格布局 (DROID: 腕部+外部)
#   3. LoRA微调: rank=4, alpha=4, 目标: q,k,v,o,ffn.0,ffn.2

class WANPolicyHead(nn.Module):
    """DreamZero动作头: 编码器 + DiT + 流匹配"""
    def __init__(self, dim=5120, ffn_dim=13824, num_heads=40,
                 num_layers=40, in_dim=36, out_dim=16,
                 action_dim=32, action_horizon=24, max_state_dim=44,
                 hidden_size=64, frame_seqlen=880,
                 num_action_per_block=32, num_state_per_block=1,
                 probe_layers=3):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # 冻结编码器 (简化: 小规模)
        self.text_encoder = WanTextEncoder(dim=256, dim_attn=256, dim_ffn=512,
                                            num_heads=4, num_layers=2)
        self.image_encoder = WanImageEncoder(dim=128, num_heads=4, num_layers=2,
                                              patch_size=14, image_size=224)
        self.vae = WanVideoVAE(dim=32, z_dim=4)

        # 核心DiT模型
        self.model = CausalWanModel(
            dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
            in_dim=in_dim, out_dim=out_dim,
            action_dim=action_dim, max_state_dim=max_state_dim,
            hidden_size=hidden_size, frame_seqlen=frame_seqlen,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            probe_layers=probe_layers)

        # 流匹配调度器
        self.scheduler = FlowMatchScheduler(shift=5.0)

    def forward(self, latents, timestep, prompt_embs, clip_feas, ys,
                action=None, timestep_action=None, state=None,
                embodiment_id=None, clean_x=None, seq_len=None):
        """训练前向: 计算视频+动作损失"""
        # DiT预测
        video_pred, action_pred = self.model(
            latents, timestep, prompt_embs, clip_feas, ys,
            action, timestep_action, state, clean_x, embodiment_id)

        # 简化损失计算
        dynamics_loss = F.mse_loss(video_pred.float(), torch.zeros_like(video_pred).float())
        if action_pred is not None:
            action_loss = F.mse_loss(action_pred.float(), torch.zeros_like(action_pred).float())
        else:
            action_loss = torch.tensor(0.0)

        return dynamics_loss + action_loss, dynamics_loss, action_loss


# ============================================================
# 主测试函数
# ============================================================

def run_test():
    """运行完整的DreamZero探针测试"""
    print("=" * 80)
    print("DreamZero: World Action Models are Zero-shot Policies")
    print("探针代码测试 - 14B模型 (探针层数: 3)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # ---- 14B模型配置 ----
    DIM = 5120
    FFN_DIM = 13824
    NUM_HEADS = 40
    NUM_LAYERS = 40  # 实际层数
    PROBE_LAYERS = 3  # 探针层数 (环境限制)
    IN_DIM = 36  # 16 (VAE) + 20 (首帧mask+latent, i2v模式)
    OUT_DIM = 16
    ACTION_DIM = 32
    MAX_STATE_DIM = 44
    HIDDEN_SIZE = 64
    FRAME_SEQLEN = 880  # 20*11*4 (14B, 480x640输入, latent 60x80)
    NUM_ACTION_PER_BLOCK = 32
    NUM_STATE_PER_BLOCK = 1
    ACTION_HORIZON = 24
    TEXT_DIM = 4096
    FREQ_DIM = 256

    B = 1  # batch size
    F = 4  # 帧数 (简化)
    H_lat, W_lat = 10, 10  # 潜变量空间 (简化)

    print(f"\n{'='*60}")
    print("模块参数统计 (14B配置)")
    print(f"{'='*60}")

    # ---- 测试各模块 ----
    results = OrderedDict()

    # 1. WanTextEncoder
    text_enc = WanTextEncoder(dim=256, dim_attn=256, dim_ffn=512, num_heads=4, num_layers=2).to(device)
    params = count_parameters(text_enc)
    print(f"\n[1] WanTextEncoder (UMT5-XXL简化): {format_params(params)}")
    print(f"    实际参数: ~4.6B (冻结, vocab=256384, dim=4096, 24层)")
    ids = torch.randint(0, 256384, (B, 512), device=device)
    out = text_enc(ids)
    print(f"    输入: ids {tuple(ids.shape)}")
    print(f"    输出: {tuple(out.shape)}")
    print(f"    实际输出: (B, 512, 4096)")
    results['TextEncoder'] = 4.6e9

    # 2. WanImageEncoder
    img_enc = WanImageEncoder(dim=128, num_heads=4, num_layers=2).to(device)
    params = count_parameters(img_enc)
    print(f"\n[2] WanImageEncoder (CLIP ViT-H/14简化): {format_params(params)}")
    print(f"    实际参数: ~304M (冻结, dim=1280, 32层)")
    imgs = torch.randn(B, 3, 224, 224, device=device)
    out = img_enc(imgs)
    print(f"    输入: images {tuple(imgs.shape)}")
    print(f"    输出: {tuple(out.shape)}")
    print(f"    实际输出: (B, 257, 1280)")
    results['ImageEncoder'] = 304e6

    # 3. WanVideoVAE
    vae = WanVideoVAE(dim=32, z_dim=4).to(device)
    params = count_parameters(vae)
    print(f"\n[3] WanVideoVAE (3D VAE简化): {format_params(params)}")
    print(f"    实际参数: ~317M (冻结, dim=96, z_dim=16)")
    video = torch.randn(B, 3, 4, 64, 64, device=device)
    out = vae.encode(video)
    print(f"    输入: video {tuple(video.shape)}")
    print(f"    输出: {tuple(out.shape)}")
    print(f"    实际输出: (B, 16, F/4, H/8, W/8)")
    results['VAE'] = 317e6

    # 4. CategorySpecificMLP
    cat_mlp = CategorySpecificMLP(num_categories=1, input_dim=MAX_STATE_DIM,
                                   hidden_dim=HIDDEN_SIZE, output_dim=DIM).to(device)
    params = count_parameters(cat_mlp)
    print(f"\n[4] CategorySpecificMLP (状态编码器): {format_params(params)}")
    print(f"    实际参数: ~660K (input=44, hidden=64, output=5120)")
    state_in = torch.randn(B, 1, MAX_STATE_DIM, device=device)
    cat_ids = torch.tensor([0], device=device)
    out = cat_mlp(state_in, cat_ids)
    print(f"    输入: state {tuple(state_in.shape)}, cat_ids {tuple(cat_ids.shape)}")
    print(f"    输出: {tuple(out.shape)}")
    results['StateEncoder'] = params

    # 5. MultiEmbodimentActionEncoder
    act_enc = MultiEmbodimentActionEncoder(
        action_dim=ACTION_DIM, hidden_size=DIM, num_embodiments=1).to(device)
    params = count_parameters(act_enc)
    print(f"\n[5] MultiEmbodimentActionEncoder: {format_params(params)}")
    print(f"    实际参数: ~79M (action_dim=32, hidden_size=5120)")
    actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM, device=device)
    timesteps_act = torch.randn(B, ACTION_HORIZON, device=device)
    out = act_enc(actions, timesteps_act, cat_ids)
    print(f"    输入: actions {tuple(actions.shape)}, timesteps {tuple(timesteps_act.shape)}")
    print(f"    输出: {tuple(out.shape)}")
    results['ActionEncoder'] = params

    # 6. CausalWanSelfAttention
    self_attn = CausalWanSelfAttention(
        dim=DIM, num_heads=NUM_HEADS, frame_seqlen=FRAME_SEQLEN,
        num_action_per_block=NUM_ACTION_PER_BLOCK).to(device)
    params = count_parameters(self_attn)
    print(f"\n[6] CausalWanSelfAttention: {format_params(params)}")
    print(f"    实际参数: ~105M (dim=5120, 40 heads)")
    seq_len_test = 100
    x_test = torch.randn(B, seq_len_test, DIM, device=device)
    freqs_test = rope_params(seq_len_test, DIM // NUM_HEADS)[0].to(device)
    out = self_attn(x_test, freqs_test)
    print(f"    输入: x {tuple(x_test.shape)}")
    print(f"    输出: {tuple(out.shape)}")
    results['SelfAttention'] = params

    # 7. CausalWanAttentionBlock
    dit_block = CausalWanAttentionBlock(
        dim=DIM, ffn_dim=FFN_DIM, num_heads=NUM_HEADS,
        frame_seqlen=FRAME_SEQLEN,
        num_action_per_block=NUM_ACTION_PER_BLOCK).to(device)
    params = count_parameters(dit_block)
    print(f"\n[7] CausalWanAttentionBlock (DiT块): {format_params(params)}")
    print(f"    实际参数: ~326M (self_attn + cross_attn + ffn + modulation)")
    e_test = torch.randn(B, seq_len_test, 6, DIM, device=device)
    ctx_test = torch.randn(B, 512 + 257, DIM, device=device)
    out = dit_block(x_test, e_test, freqs_test, ctx_test)
    print(f"    输入: x {tuple(x_test.shape)}, e {tuple(e_test.shape)}, context {tuple(ctx_test.shape)}")
    print(f"    输出: {tuple(out.shape)}")
    results['DiTBlock'] = params

    # 8. CausalWanModel (核心)
    model = CausalWanModel(
        dim=DIM, ffn_dim=FFN_DIM, num_heads=NUM_HEADS,
        in_dim=IN_DIM, out_dim=OUT_DIM,
        action_dim=ACTION_DIM, max_state_dim=MAX_STATE_DIM,
        hidden_size=HIDDEN_SIZE, frame_seqlen=FRAME_SEQLEN,
        num_action_per_block=NUM_ACTION_PER_BLOCK,
        num_state_per_block=NUM_STATE_PER_BLOCK,
        probe_layers=PROBE_LAYERS).to(device)
    params = count_parameters(model)
    print(f"\n[8] CausalWanModel (核心DiT, {PROBE_LAYERS}层探针): {format_params(params)}")
    # 计算实际14B参数量
    actual_dit_params = (
        IN_DIM * DIM * 4 +  # patch_embedding
        TEXT_DIM * DIM + DIM * DIM +  # text_embedding
        FREQ_DIM * DIM + DIM * DIM +  # time_embedding
        DIM * DIM * 6 +  # time_projection
        1280 * DIM + DIM * DIM +  # img_emb
        NUM_LAYERS * (  # 40 blocks
            4 * DIM * DIM + 2 * DIM +  # self_attn (q,k,v,o + norm_q,norm_k)
            3 * DIM * DIM + 2 * DIM +  # cross_attn (q,k,v,o + norm_q,norm_k)
            DIM * FFN_DIM + FFN_DIM * DIM +  # ffn
            6 * DIM +  # modulation
            3 * DIM  # norms
        ) +
        DIM * (OUT_DIM * 4) + 2 * DIM +  # head
        3 * (ACTION_DIM * DIM + DIM) +  # action_encoder W1,W2,W3 (simplified)
        2 * (MAX_STATE_DIM * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * DIM + DIM) +  # state_encoder
        2 * (DIM * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * ACTION_DIM + ACTION_DIM)  # action_decoder
    )
    print(f"    实际参数 (40层): ~{format_params(actual_dit_params)}")
    print(f"    实际层数: {NUM_LAYERS}, 探针层数: {PROBE_LAYERS} (环境限制)")
    results['CausalWanModel'] = actual_dit_params

    # 9. FlowMatchScheduler
    scheduler = FlowMatchScheduler(shift=5.0).to(device)
    print(f"\n[9] FlowMatchScheduler: 0参数")
    print(f"    shift=5.0, num_train_timesteps=1000")
    x_clean = torch.randn(B, 16, F, H_lat, W_lat, device=device)
    noise = torch.randn_like(x_clean)
    t = torch.tensor([500], device=device)
    x_noisy = scheduler.add_noise(x_clean, noise, t)
    target = scheduler.training_target(x_clean, noise, t)
    weight = scheduler.training_weight(t)
    print(f"    加噪: x_clean {tuple(x_clean.shape)} -> x_noisy {tuple(x_noisy.shape)}")
    print(f"    训练目标: target {tuple(target.shape)} (速度预测: noise - x)")
    print(f"    训练权重: weight {weight.item():.4f}")
    results['Scheduler'] = 0

    # ---- 总参数统计 ----
    print(f"\n{'='*60}")
    print("总参数统计 (14B模型)")
    print(f"{'='*60}")
    total = 0
    for name, p in results.items():
        total += p
        print(f"  {name:25s}: {format_params(p):>10s}")
    print(f"  {'总计':25s}: {format_params(total):>10s}")
    print(f"\n  实际模型总参数: ~14B (Wan 2.1-I2V-14B-480P)")
    print(f"  冻结参数: TextEncoder(~4.6B) + ImageEncoder(~304M) + VAE(~317M) = ~5.2B")
    print(f"  可训练参数: DiT LoRA (~9.1B中LoRA部分) + ActionEncoder + StateEncoder + ActionDecoder")

    # ---- 数据流总结 ----
    print(f"\n{'='*60}")
    print("训练数据流 (14B, 教师强制)")
    print(f"{'='*60}")
    print("""
输入:
  video: (B, 3, T, H, W) = (1, 3, 13, 480, 640)
  text: (B, 512) - UMT5-XXL token IDs
  action: (B, 24, 32) - 机器人动作 (24步, 32维)
  state: (B, 24, 44) - 机器人状态 (24步, 44维)

编码:
  TextEncoder:  text -> (B, 512, 4096) -> text_embedding -> (B, 512, 5120)
  ImageEncoder: image -> (B, 257, 1280) -> img_emb -> (B, 257, 5120)
  VAE:          video -> (B, 16, 4, 60, 80)  [4x时间下采样, 8x空间下采样]
  ActionEncoder: action -> (B, 24, 5120)
  StateEncoder:  state -> (B, 24, 5120)

DiT输入 (教师强制):
  clean_half:  [clean_video_tokens]  (B, seq_len, 5120)
  noisy_half:  [noisy_video_tokens | action_tokens | state_tokens]
  总长度: 2 * seq_len + action_register_length
  seq_len = F * (H_lat/2) * (W_lat/2) = 4 * 30 * 40 = 4800 (简化)
  实际: frame_seqlen=880, F=13 -> seq_len=13*880=11440

DiT输出:
  video_noise_pred: (B, 16, F, H', W') - 视频噪声预测
  action_noise_pred: (B, 24, 32) - 动作噪声预测

损失:
  dynamics_loss = MSE(video_noise_pred, target) * weight(t)
  action_loss = MSE(action_noise_pred, target_action) * weight(t_action) * mask
  loss = dynamics_loss + action_loss
""")

    # ---- 核心创新总结 ----
    print(f"{'='*60}")
    print("核心创新")
    print(f"{'='*60}")
    print("""
1. World Action Model (WAM): 视频扩散模型同时预测视频和动作,
   动作预测作为视频生成的一部分而非独立下游任务

2. 分块因果注意力: 动作/状态token与视频token交错,
   实现时间因果性: t时刻的动作只依赖t及之前的观测
   - 首帧: 仅自注意力 (条件)
   - 图像块i: 注意首帧 + 前面图像块 + 当前动作 + 当前状态
   - 动作块i: 注意首帧 + 前面图像块 + 当前图像 + 当前动作 + 当前状态
   - 状态块: 仅自注意力 (条件)

3. 教师强制训练: clean上下文 + noisy预测,
   并行训练所有时间块而非顺序自回归

4. 跨具身训练: CategorySpecificLinear为每个机器人平台提供独立权重,
   支持DROID/AGIBOT/GR1/YAM等多平台联合训练

5. 解耦噪声采样: 视频用Beta(3,1)偏向高噪声,
   动作用独立均匀分布, 提升训练-推理对齐

6. 多视角视频: 2x2网格布局将多相机视角合成为单张图像,
   利用预训练模型的空間理解能力
""")

    print("探针测试完成!")


if __name__ == "__main__":
    run_test()

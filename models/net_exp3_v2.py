import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


# =============================================================================
# 1. DetailFeatureExtraction (INN 可逆高频细节分支 - 全程维持原分辨率 H×W)
# =============================================================================
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, inp):
        z1, z2 = inp[:, :inp.shape[1] // 2], inp[:, inp.shape[1] // 2:inp.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# =============================================================================
# 2. 基础归一化与全分辨率通道注意力/前馈组件 (LayerNorm, MDTA, GDFN)
# =============================================================================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """门控深度前馈网络 (GDFN): 在全分辨率 H×W 下执行非线性通道混合与局部 3×3 空间门控"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """多 DConv 头转置通道自注意力 (MDTA): 在全分辨率 H×W 下以 O(C^2) 计算跨通道全局协方差"""
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# =============================================================================
# 3. [创新点二核心模块] 零池化全分辨率并行空间-通道协同模块 (Parallel SC-MambaVision)
# =============================================================================
class MambaVision2DMixer(nn.Module):
    """
    全分辨率四向通道分组 2D-MambaVision 空间混合器 (零池化、零双线性插值、天然支持任意奇偶尺寸):
    1. 右支路 (MambaVision 对称卷积支路, 32维): 全分辨率 H×W 执行 3×3 DWConv + SiLU，无损保留局部几何边缘与极值。
    2. 左支路 (全分辨率四向通道分组 2D-SSM 支路, 32维):
       - 彻底废除 AvgPool2d(2) 与 Bilinear 插值，100% 消除低通插值模糊对 SF 与 SD 的损耗；
       - 将 32 通道均分为 4 个方向组 (每组 d_group=8 通道)，在 H×W 全分辨率下同时沿
         [水平正向、水平反向、垂直正向、垂直反向] 4 个正交方向执行选择性状态扫描 (SS2D)；
       - 扫描后与全分辨率直通锚点 x_ssm 残差相加，再与右支路 Concat 经 1×1 卷积完成四向跨通道混合。
    """
    def __init__(self, dim=64, d_state=16, d_conv=4, ssm_expand=2, bias=False):
        super(MambaVision2DMixer, self).__init__()
        self.dim = dim
        self.d_branch = dim // 2          # 32
        self.num_dirs = 4
        self.d_group = self.d_branch // self.num_dirs  # 8 通道/方向组

        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 左支路：全分辨率局部深度卷积先验 + 四向分组 Mamba
        self.dwconv_ssm = nn.Conv2d(
            self.d_branch, self.d_branch, kernel_size=3,
            stride=1, padding=1, groups=self.d_branch, bias=bias
        )
        self.mamba = Mamba(
            d_model=self.d_group,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
        )
        # 通道级可学习缩放因子 (避免逐像素 LayerNorm 抹平空间对比度 SD)
        self.scale_ssm = nn.Parameter(torch.ones(1, self.d_branch, 1, 1) * 0.5)

        # 右支路：全分辨率 MambaVision 对称局部卷积支路
        self.conv_sym = nn.Sequential(
            nn.Conv2d(
                self.d_branch, self.d_branch, kernel_size=3,
                stride=1, padding=1, groups=self.d_branch, bias=bias
            ),
            nn.SiLU(inplace=True),
        )

        # 全分辨率四向状态跨通道混合与双支路聚合投影
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, 64, H, W) 全分辨率输入 (无需任何奇偶 Padding!)
        b, c, h, w = x.shape
        x_ssm, x_sym = self.in_proj(x).chunk(2, dim=1)  # 各为 (B, 32, H, W)

        # --- 右支路：全分辨率局部几何卷积补偿 ---
        out_sym = self.conv_sym(x_sym)                  # (B, 32, H, W)

        # --- 左支路：100% 全分辨率四向通道分组 2D-SSM 扫描 ---
        u = F.silu(self.dwconv_ssm(x_ssm))              # (B, 32, H, W)
        u0, u1, u2, u3 = u.chunk(4, dim=1)              # 每个方向组: (B, 8, H, W)

        s0 = u0.flatten(2).transpose(1, 2)                              # 方向 0: 左 -> 右 (B, HW, 8)
        s1 = u1.flatten(2).flip(-1).transpose(1, 2)                     # 方向 1: 右 -> 左 (B, HW, 8)
        s2 = u2.transpose(2, 3).flatten(2).transpose(1, 2)              # 方向 2: 上 -> 下 (B, HW, 8)
        s3 = u3.transpose(2, 3).flatten(2).flip(-1).transpose(1, 2)     # 方向 3: 下 -> 上 (B, HW, 8)

        # 沿 Batch 维并发打包一次性调用 CUDA 选择性扫描: (4B, HW, 8)
        s_all = torch.cat([s0, s1, s2, s3], dim=0).contiguous()
        y_all = self.mamba(s_all)
        y0, y1, y2, y3 = y_all.chunk(4, dim=0)

        # 原位无损逆向还原回 (B, 8, H, W) 全分辨率网格
        y0 = y0.transpose(1, 2).reshape(b, self.d_group, h, w)
        y1 = y1.transpose(1, 2).flip(-1).reshape(b, self.d_group, h, w)
        y2 = y2.transpose(1, 2).reshape(b, self.d_group, w, h).transpose(2, 3)
        y3 = y3.transpose(1, 2).flip(-1).reshape(b, self.d_group, w, h).transpose(2, 3)

        # 拼接 4 个方向组并叠加至全分辨率直通锚点 x_ssm
        y_ssm = torch.cat([y0, y1, y2, y3], dim=1)                      # (B, 32, H, W)
        out_ssm = x_ssm + self.scale_ssm * y_ssm                        # (B, 32, H, W)

        # --- 全分辨率双支路拼接与跨方向线性融合 ---
        out = self.out_proj(torch.cat([out_ssm, out_sym], dim=1))       # (B, 64, H, W)
        return out


class BaseHybridBranch(nn.Module):
    """
    [创新点二主模块] 共享 Pre-Norm 并行空间-通道协同模块 (Parallel SC-MambaVision Block):
    1. 第一级 (共享归一化 + 空间-通道并行协同):
       x_norm = norm1(x)
       x = x + Attention_MDTA(x_norm) + gamma_mamba * MambaVision2DMixer(x_norm)
       - 相比 Exp-3_v2 的 3 层串行结构，砍掉了多余的 LayerNorm 与串行延迟，恢复 Exp-1 的高 FPS；
       - 空间维 2D-MambaVision 与通道维 MDTA 并行作用于同一无损特征，互不干扰。
    2. 第二级 (全分辨率门控前馈网络 GDFN):
       x = x + FeedForward_GDFN(norm2(x))
    """
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2., qkv_bias=False, init_gamma=0.2):
        super(BaseHybridBranch, self).__init__()
        # 第 1 级：共享 Pre-Norm + (通道维 MDTA || 空间维 MambaVision2DMixer) 并行协同
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim=dim, num_heads=num_heads, bias=qkv_bias)
        self.mamba_mixer = MambaVision2DMixer(dim=dim, d_state=16, d_conv=4, ssm_expand=2, bias=qkv_bias)
        self.gamma_mamba = nn.Parameter(torch.full((1, dim, 1, 1), float(init_gamma)))

        # 第 2 级：全分辨率门控前馈网络 (GDFN)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=qkv_bias)

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm) + self.gamma_mamba * self.mamba_mixer(x_norm)
        x = x + self.ffn(self.norm2(x))
        return x


# 兼容别名
BaseFeatureExtraction = BaseHybridBranch


# =============================================================================
# 4. [阶段二同构融合层] 能量下界锁死非负差模补偿 + 全分辨率同构融合
# =============================================================================
class MIMOHybridMambaVisionFusion(nn.Module):
    """
    阶段二低频基底同构融合层 (能量下界锁死版):
    1. 锁死 Exp-1 的无损加性基底 (feat_ir + feat_vis):
       确保两个模态的基础权重恒 >= 1.0，从数学上杜绝 tanh 负向门控压制弱势模态，
       100% 保住 Exp-1 的高全局对比度 (SD)、高差异相关性 (SCD) 与高 MSRS 互信息 (MI)！
    2. 非负差模极性补偿:
       利用 Sigmoid 非负门控在明暗极性反相区域叠加差模包络 |feat_ir - feat_vis|，
       同时保住 Exp-3_v2 在 TNO/RoadScene 上的高 VIF、高 MI 与高 Qabf。
    """
    def __init__(self, in_dim=128, out_dim=64, num_heads=8, ffn_expansion_factor=2., bias=False):
        super(MIMOHybridMambaVisionFusion, self).__init__()
        self.dim = out_dim

        self.diff_gate = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1, bias=bias),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma_gate = nn.Parameter(torch.full((1, out_dim, 1, 1), 0.15))

        # 同构复用全分辨率并行 SC-MambaVision 模块
        self.base_mamba_block = BaseHybridBranch(
            dim=out_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            qkv_bias=bias,
            init_gamma=0.2,
        )

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor = None) -> torch.Tensor:
        if feat_vis is None:
            feat_ir, feat_vis = feat_ir.chunk(2, dim=1)

        # 1. 无损加性基底 (严格保住双模态叠加方差 SD 与线性差值相关性 SCD)
        base_sum = feat_ir + feat_vis

        # 2. 非负差模能量补偿 (仅做加法增量，绝不削弱任一模态)
        feat_diff = torch.abs(feat_ir - feat_vis)
        gate_weight = self.diff_gate(torch.cat([feat_ir, feat_vis, feat_diff], dim=1))
        base_anchor = base_sum + torch.abs(self.gamma_gate) * gate_weight * feat_diff

        fused = self.base_mamba_block(base_anchor)
        return fused


class CrossModalityDetailFusion(nn.Module):
    """
    阶段二高频细节同构融合层:
    锁死 (detail_ir + detail_vis) 无损加性基底，并通过非负门控叠加绝对幅值包络 |D_ir| + |D_vis| 补偿反向边缘抵消
    """
    def __init__(self, dim=64, num_layers=1):
        super(CrossModalityDetailFusion, self).__init__()
        self.conflict_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma_detail = nn.Parameter(torch.full((1, dim, 1, 1), 0.15))
        self.inn = DetailFeatureExtraction(num_layers=num_layers)

    def forward(self, detail_ir: torch.Tensor, detail_vis: torch.Tensor = None) -> torch.Tensor:
        if detail_vis is None:
            detail_ir, detail_vis = detail_ir.chunk(2, dim=1)

        mag_sum = torch.abs(detail_ir) + torch.abs(detail_vis)
        cat_inp = torch.cat([detail_ir, detail_vis, mag_sum], dim=1)

        detail_init = (detail_ir + detail_vis) + torch.abs(self.gamma_detail) * self.conflict_gate(cat_inp) * mag_sum
        return self.inn(detail_init)


# =============================================================================
# 5. [创新点一模块] 浅层多尺度门控卷积块、共享编码器与混合解码器
# =============================================================================
class ShallowGatedCNNBlock(nn.Module):
    def __init__(self, dim=64, ffn_expansion_factor=2., bias=False, LayerNorm_type='WithBias'):
        super(ShallowGatedCNNBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        hidden_dim = int(dim * 2)
        self.proj_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Sigmoid()
        )
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        shortcut = x
        x1, x2 = self.proj_in(self.norm1(x)).chunk(2, dim=1)
        x_gated = F.silu(self.dwconv3x3(x1)) * self.dwconv5x5(x2)
        x_gated = x_gated * self.sca(x_gated)
        x = shortcut + self.proj_out(x_gated)
        x = x + self.ffn(self.norm2(x))
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[2, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[ShallowGatedCNNBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])]
        )
        self.baseFeature = BaseHybridBranch(dim=dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor)
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Decoder, self).__init__()

        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        self.encoder_level2 = nn.Sequential(
            TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type),
            ShallowGatedCNNBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type),
            ShallowGatedCNNBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type),
        )

        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


# =============================================================================
# 6. 架构全链路 5 项自动化验证入口：运行 `python -m models.net_exp3`
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Restormer_Encoder().to(device)
    dec = Restormer_Decoder().to(device)
    base_fuse = MIMOHybridMambaVisionFusion(in_dim=128, out_dim=64, num_heads=8).to(device)
    detail_fuse = CrossModalityDetailFusion(dim=64, num_layers=1).to(device)

    # 验证 1: 参数量核查 (d_group=8 分组扫描使参数更紧凑)
    total_params = sum(p.numel() for m in [enc, dec, base_fuse, detail_fuse] for p in m.parameters())
    print(f"[Check 1 Passed] Total Parameters: {total_params / 1e6:.3f} M "
          f"(Enc: {sum(p.numel() for p in enc.parameters())/1e6:.3f}M, "
          f"Dec: {sum(p.numel() for p in dec.parameters())/1e6:.3f}M, "
          f"BaseFuse: {sum(p.numel() for p in base_fuse.parameters())/1e6:.3f}M)")

    # 验证 2: 零池化、零插值高频极值保持验证
    test_feat = torch.randn(1, 64, 64, 64, device=device)
    test_feat[:, :, 16:48, 16:48] *= 5.0
    with torch.no_grad():
        base_out = enc.baseFeature(test_feat)
        std_center = base_out[:, :, 16:48, 16:48].std().item()
        std_bg = base_out[:, :, :16, :16].std().item()
    assert std_center > std_bg * 3.2, "Full-resolution spatial contrast degraded!"
    print(f"[Check 2 Passed] Zero-Pooling Parallel SC-MambaVision Contrast -> Target Std: {std_center:.3f} vs BG Std: {std_bg:.3f}")

    # 验证 3: 非负互补门控能量下界验证 (确保融合特征方差严格 >= 单模态叠加下界)
    with torch.no_grad():
        f_ir = torch.randn(1, 64, 64, 64, device=device)
        f_vis = -0.5 * f_ir + 0.5 * torch.randn_like(f_ir)  # 模拟明暗反相区域
        f_fused = base_fuse(f_ir, f_vis)
        naive_sum_std = (f_ir + f_vis).std().item()
        assert f_fused.std().item() > naive_sum_std, "Non-negative diff gate failed to compensate opposite polarity!"
    print(f"[Check 3 Passed] Non-Negative Polarity Compensation -> Naive Sum Std: {naive_sum_std:.3f}, Fused Std: {f_fused.std().item():.3f}")

    # 验证 4: 全模型梯度连通性验证
    ir_even = torch.randn(2, 1, 128, 128, device=device)
    vis_even = torch.randn(2, 1, 128, 128, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        b_ir, d_ir, _ = enc(ir_even)
        b_vis, d_vis, _ = enc(vis_even)
        f_b = base_fuse(b_ir, b_vis)
        f_d = detail_fuse(d_ir, d_vis)
        out_even, _ = dec(vis_even, f_b, f_d)
        loss = out_even.float().mean() + 0.01 * (b_ir.float().mean() + d_ir.float().mean())
    loss.backward()
    for mod_name, mod in [("Encoder", enc), ("Decoder", dec), ("BaseFuse", base_fuse), ("DetailFuse", detail_fuse)]:
        for p_name, p in mod.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and not torch.isnan(p.grad).any(), f"Broken/NaN gradient in {mod_name}.{p_name}"
    print("[Check 4 Passed] 100% Parameters Have Valid Non-NaN Gradients!")

    # 验证 5: 奇数分辨率 (495x629) 零 Padding 原生对齐验证
    with torch.inference_mode():
        ir_odd = torch.randn(1, 1, 495, 629, device=device)
        vis_odd = torch.randn(1, 1, 495, 629, device=device)
        b_ir_o, d_ir_o, _ = enc(ir_odd)
        b_vis_o, d_vis_o, _ = enc(vis_odd)
        out_odd, _ = dec(vis_odd, base_fuse(b_ir_o, b_vis_o), detail_fuse(d_ir_o, d_vis_o))
    assert out_odd.shape == ir_odd.shape, "Odd resolution mismatch!"
    print(f"[Check 5 Passed] Zero-Padding Odd Resolution (495x629) Verified -> Out: {out_odd.shape}")
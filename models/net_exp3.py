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
# 3. [创新点二核心模块] 全分辨率空间-通道协同 SC-MambaVision 模块 (BaseHybridBranch)
# =============================================================================
class MambaVision2DMixer(nn.Module):
    """
    双路对称全向 2D-MambaVision 空间混合器 (输入 H×W -> 输出 H×W):
    1. 右支路 (MambaVision 对称卷积支路, 32维): 全分辨率 H×W 执行 3×3 DWConv + SiLU，零损耗保留局部几何对比度。
    2. 左支路 (VMamba 四向状态空间支路, 32维):
       - 保留全分辨率直通锚点 X_ssm (H×W)，杜绝下采样信息瓶颈；
       - 内部通过低通池化在 H/2 × W/2 尺度执行全部 32 通道的四向全向选择性扫描 (SS2D)，
         使单次扫描序列长度缩短 75%，兼顾 2× 大感受野与高速推理；
       - 采用通道级可学习缩放 scale_ssm (废除抹平像素对比度的逐像素 LayerNorm)，
         上采样后叠加回全分辨率锚点 X_ssm。
    3. 双支路在全分辨率 H×W 下 Concat 拼接并经 1×1 线性层融合输出。
    """
    def __init__(self, dim=64, d_state=16, d_conv=4, ssm_expand=2, bias=False):
        super(MambaVision2DMixer, self).__init__()
        self.dim = dim
        self.d_branch = dim // 2  # 32

        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 左支路：多尺度四向全向 2D-SSM 状态空间扫描
        self.dwconv_ssm = nn.Conv2d(
            self.d_branch, self.d_branch, kernel_size=3,
            stride=1, padding=1, groups=self.d_branch, bias=bias
        )
        self.mamba = Mamba(
            d_model=self.d_branch,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
        )
        # 通道级可学习缩放因子 (不改变空间像素间的相对能量比值，100% 保护 SD)
        self.scale_ssm = nn.Parameter(torch.ones(1, self.d_branch, 1, 1))

        # 右支路：全分辨率 MambaVision 对称局部卷积支路
        self.conv_sym = nn.Sequential(
            nn.Conv2d(
                self.d_branch, self.d_branch, kernel_size=3,
                stride=1, padding=1, groups=self.d_branch, bias=bias
            ),
            nn.SiLU(inplace=True),
        )

        # 全分辨率双路聚合线性投影
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, 64, H, W) 全分辨率输入
        b, c, h, w = x.shape
        x_ssm, x_sym = self.in_proj(x).chunk(2, dim=1)  # 各为 (B, 32, H, W)

        # --- 右支路：全分辨率局部几何卷积补偿 ---
        out_sym = self.conv_sym(x_sym)                  # (B, 32, H, W)

        # --- 左支路：内嵌多尺度四方向 2D-SSM 全局扫描 + 全分辨率残差直通 ---
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x_ssm_pad = F.pad(x_ssm, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_ssm_pad = x_ssm

        # 降采样至 H_d × W_d 提取大尺度平滑热辐射与背景光照场
        x_low = F.avg_pool2d(x_ssm_pad, kernel_size=2, stride=2)
        h_d, w_d = x_low.shape[-2:]

        u_low = F.silu(self.dwconv_ssm(x_low))
        s0 = u_low.flatten(2).transpose(1, 2)                           # 方向 0: 水平正向
        s1 = s0.flip(1)                                                 # 方向 1: 水平反向
        s2 = u_low.transpose(2, 3).flatten(2).transpose(1, 2)           # 方向 2: 垂直正向
        s3 = s2.flip(1)                                                 # 方向 3: 垂直反向

        s_all = torch.cat([s0, s1, s2, s3], dim=0).contiguous()         # (4B, h_d*w_d, 32)
        y_all = self.mamba(s_all)
        y0, y1, y2, y3 = y_all.chunk(4, dim=0)

        y0 = y0.transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y1 = y1.flip(1).transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y2 = y2.transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)
        y3 = y3.flip(1).transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)

        # 四方向对称求均值并经通道缩放
        y_low = ((y0 + y1 + y2 + y3) * 0.25) * self.scale_ssm

        # 平滑上采样回原分辨率并叠加至全分辨率直通锚点 x_ssm
        y_up = F.interpolate(y_low, size=x_ssm_pad.shape[-2:], mode='bilinear', align_corners=False)
        if pad_h > 0 or pad_w > 0:
            y_up = y_up[:, :, :h, :w]

        out_ssm = x_ssm + y_up                                          # (B, 32, H, W)

        # --- 全分辨率双支路拼接投影 ---
        out = self.out_proj(torch.cat([out_ssm, out_sym], dim=1))       # (B, 64, H, W)
        return out


class BaseHybridBranch(nn.Module):
    """
    [创新点二主模块] 空间-通道协同全分辨率低频基底模块 (SC-MambaVision Block):
    全程維持 (B, 64, H, W) 原生分辨率，严格包含 Exp-1 的全分辨率 MDTA + GDFN 能力下界：
    1. 空间维 O(N) 长程-局部混合: Pre-Norm + MambaVision2DMixer (带可学习残差缩放 gamma_mamba)
    2. 通道维 O(C^2) 全局协方差交互: Pre-Norm + Attention (全分辨率 MDTA，弥补 SSM 通道独立局限)
    3. 局部非线性门控变换: Pre-Norm + FeedForward (全分辨率 GDFN)
    """
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2., qkv_bias=False, init_gamma=0.2):
        super(BaseHybridBranch, self).__init__()
        # 第 1 子层：空间维 2D-MambaVision 混合器
        self.norm_mamba = LayerNorm(dim, 'WithBias')
        self.mamba_mixer = MambaVision2DMixer(dim=dim, d_state=16, d_conv=4, ssm_expand=2, bias=qkv_bias)
        self.gamma_mamba = nn.Parameter(torch.full((1, dim, 1, 1), float(init_gamma)))

        # 第 2 子层：全分辨率通道维自注意力 (MDTA)
        self.norm_attn = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim=dim, num_heads=num_heads, bias=qkv_bias)

        # 第 3 子层：全分辨率门控前馈网络 (GDFN)
        self.norm_ffn = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=qkv_bias)

    def forward(self, x):
        # 全程在 (B, 64, H, W) 原分辨率下残差递进，零空间池化瓶颈
        x = x + self.gamma_mamba * self.mamba_mixer(self.norm_mamba(x))
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


# 兼容别名
BaseFeatureExtraction = BaseHybridBranch


# =============================================================================
# 4. [阶段二同构融合层] 差模极性消解 + 全分辨率同构复用 (彻底移除有害的 std_ortho 压缩)
# =============================================================================
class MIMOHybridMambaVisionFusion(nn.Module):
    """
    阶段二低频基底同构融合层:
    1. 差模互补残差门控 (diff_gate): 在原版 (feat_ir + feat_vis) 基础上，利用 |feat_ir - feat_vis|
       自适应补偿 TNO/RoadScene 明暗极性相反区域的能量抵消；
    2. 直接同构调用全分辨率 BaseHybridBranch (SC-MambaVision Block) 执行跨模态空间-通道协同融合；
    3. [重要修复]: 彻底移除此前因正相关方差不等式导致 scale < 1.0 的 std_ortho 缩放块，100% 保护 SD 与 Qabf。
    """
    def __init__(self, in_dim=128, out_dim=64, num_heads=8, ffn_expansion_factor=2., bias=False):
        super(MIMOHybridMambaVisionFusion, self).__init__()
        self.dim = out_dim

        self.diff_gate = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1, bias=bias),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim * 2, kernel_size=1, bias=True),
        )
        self.gamma_gate = nn.Parameter(torch.full((1, out_dim, 1, 1), 0.1))

        # 同构复用全分辨率 SC-MambaVision 模块
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

        feat_diff = torch.abs(feat_ir - feat_vis)
        gate_inp = torch.cat([feat_ir, feat_vis, feat_diff], dim=1)
        w_ir, w_vis = self.diff_gate(gate_inp).chunk(2, dim=1)

        w_ir = 1.0 + self.gamma_gate * torch.tanh(w_ir)
        w_vis = 1.0 + self.gamma_gate * torch.tanh(w_vis)
        base_anchor = w_ir * feat_ir + w_vis * feat_vis

        fused = self.base_mamba_block(base_anchor)
        return fused


class CrossModalityDetailFusion(nn.Module):
    """
    阶段二高频细节同构融合层:
    利用梯度绝对幅值包络 |D_ir| + |D_vis| 补偿反向边缘抵消，并同构复用单层可逆 INN (DetailFeatureExtraction)
    """
    def __init__(self, dim=64, num_layers=1):
        super(CrossModalityDetailFusion, self).__init__()
        self.conflict_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
        self.gamma_detail = nn.Parameter(torch.full((1, dim, 1, 1), 0.1))
        self.inn = DetailFeatureExtraction(num_layers=num_layers)

    def forward(self, detail_ir: torch.Tensor, detail_vis: torch.Tensor = None) -> torch.Tensor:
        if detail_vis is None:
            detail_ir, detail_vis = detail_ir.chunk(2, dim=1)

        mag_sum = torch.abs(detail_ir) + torch.abs(detail_vis)
        cat_inp = torch.cat([detail_ir, detail_vis, mag_sum], dim=1)

        detail_init = (detail_ir + detail_vis) + self.gamma_detail * self.conflict_gate(cat_inp)
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

    # 验证 1: 参数量核查
    total_params = sum(p.numel() for m in [enc, dec, base_fuse, detail_fuse] for p in m.parameters())
    print(f"[Check 1 Passed] Total Parameters: {total_params / 1e6:.3f} M "
          f"(Enc: {sum(p.numel() for p in enc.parameters())/1e6:.3f}M, "
          f"Dec: {sum(p.numel() for p in dec.parameters())/1e6:.3f}M, "
          f"BaseFuse: {sum(p.numel() for p in base_fuse.parameters())/1e6:.3f}M)")

    # 验证 2: 全分辨率通路与空间对比度保持验证 (确保无 LayerNorm 抹平且无下采样瓶颈)
    test_feat = torch.randn(1, 64, 64, 64, device=device)
    test_feat[:, :, 16:48, 16:48] *= 5.0
    with torch.no_grad():
        base_out = enc.baseFeature(test_feat)
        std_center = base_out[:, :, 16:48, 16:48].std().item()
        std_bg = base_out[:, :, :16, :16].std().item()
    assert std_center > std_bg * 3.0, "Full-resolution spatial contrast degraded!"
    print(f"[Check 2 Passed] Full-Res BaseHybridBranch Contrast Preserved -> Target Std: {std_center:.3f} vs BG Std: {std_bg:.3f}")

    # 验证 3: 融合层正相关特征能量不被压缩验证 (排查此前 std_ortho / std_f < 1.0 的 Bug)
    with torch.no_grad():
        f_ir = torch.randn(1, 64, 64, 64, device=device)
        f_vis = 0.9 * f_ir + 0.1 * torch.randn_like(f_ir)  # 模拟 Phase I 训练后的强正相关低频特征
        f_fused = base_fuse(f_ir, f_vis)
        assert f_fused.std().item() > f_ir.std().item() * 1.4, "Fused base feature contrast was erroneously shrunk!"
    print(f"[Check 3 Passed] Positive-Correlation Fusion Energy Preserved -> Input Std: {f_ir.std().item():.3f}, Fused Std: {f_fused.std().item():.3f}")

    # 验证 4: 全模型每个可学习参数的梯度连通性验证 (确保无断路或孤立参数)
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

    # 验证 5: 奇数尺寸 (495x629) 端到端推理无损对齐
    with torch.inference_mode():
        ir_odd = torch.randn(1, 1, 495, 629, device=device)
        vis_odd = torch.randn(1, 1, 495, 629, device=device)
        b_ir_o, d_ir_o, _ = enc(ir_odd)
        b_vis_o, d_vis_o, _ = enc(vis_odd)
        out_odd, _ = dec(vis_odd, base_fuse(b_ir_o, b_vis_o), detail_fuse(d_ir_o, d_vis_o))
    assert out_odd.shape == ir_odd.shape, "Odd resolution mismatch!"
    print(f"[Check 5 Passed] Odd Resolution (495x629) End-to-End Inference Verified -> Out: {out_odd.shape}")
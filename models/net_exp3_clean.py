import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


# =============================================================================
# 1. DetailFeatureExtraction (INN 可逆高频细节分支 - 维持原分辨率 H×W 不变)
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
# 2. 基础归一化与 Restormer 组件 (供解码器全局整合块使用)
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
    """门控前馈网络 (GDFN)"""
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
    """多 DConv 头转置自注意力 (MDTA): 仅保留在解码器首层做高低频通道整合"""
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
# 3. [创新点二核心模块] 双路对称全向 2D-MambaVision 混合器与精简低频基底分支
# =============================================================================
class MambaVision2DMixer(nn.Module):
    """
    双路对称全向 2D-MambaVision Mixer:
    1. 左支路 (d_branch=32): 3×3 DWConv + 四方向全向二维扫描 (SS2D)
    2. 右支路 (d_branch=32): 3×3 DWConv + SiLU 对称局部卷积补偿支路
    3. [关键修复]: 移除会抹平像素间对比度能量的 self.norm_ssm (逐像素 LayerNorm)，
       改用通道级可学习缩放因子 self.scale_ssm，100% 保留空间热目标高对比度 (SD)。
    """
    def __init__(self, dim=64, d_state=16, d_conv=4, ssm_expand=2, bias=False):
        super(MambaVision2DMixer, self).__init__()
        self.dim = dim
        self.d_branch = dim // 2

        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 左支路：全向 2D-SSM 状态空间扫描
        self.dwconv_ssm = nn.Conv2d(self.d_branch, self.d_branch, kernel_size=3,
                                    stride=1, padding=1, groups=self.d_branch, bias=bias)
        self.mamba = Mamba(
            d_model=self.d_branch,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
        )
        # 通道级可学习缩放，绝不抹平不同像素坐标 (h, w) 之间的能量方差
        self.scale_ssm = nn.Parameter(torch.ones(1, self.d_branch, 1, 1))

        # 右支路：对称局部卷积补偿支路
        self.conv_sym = nn.Sequential(
            nn.Conv2d(self.d_branch, self.d_branch, kernel_size=3,
                      stride=1, padding=1, groups=self.d_branch, bias=bias),
            nn.SiLU(inplace=True),
        )

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h_d, w_d = x.shape
        x_ssm, x_sym = self.in_proj(x).chunk(2, dim=1)

        x_ssm = F.silu(self.dwconv_ssm(x_ssm))
        s0 = x_ssm.flatten(2).transpose(1, 2)                           # 水平正向
        s1 = s0.flip(1)                                                 # 水平反向
        s2 = x_ssm.transpose(2, 3).flatten(2).transpose(1, 2)           # 垂直正向
        s3 = s2.flip(1)                                                 # 垂直反向

        s_all = torch.cat([s0, s1, s2, s3], dim=0).contiguous()
        y_all = self.mamba(s_all)
        y0, y1, y2, y3 = y_all.chunk(4, dim=0)

        y0 = y0.transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y1 = y1.flip(1).transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y2 = y2.transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)
        y3 = y3.flip(1).transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)

        out_ssm = ((y0 + y1 + y2 + y3) * 0.25) * self.scale_ssm
        out_sym = self.conv_sym(x_sym)
        out = self.out_proj(torch.cat([out_ssm, out_sym], dim=1))
        return out


class BaseHybridBranch(nn.Module):
    """
    [精简定稿版] 多尺度 2D-MambaVision 低频基底模块:
    严格对齐原版 BaseFeatureExtraction 的双子层结构 (MambaVision2DMixer + GDFN)，
    移除冗余堆砌的 MDTA 与 down_refine，在 H/2 × W/2 尺度高效提取全局光照与热辐射场。
    """
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2., qkv_bias=False, init_gamma=1.0):
        super(BaseHybridBranch, self).__init__()
        # 子层 1: Pre-Norm + 2D-MambaVision Mixer (一对一替换原版 AttentionBase)
        self.norm_mamba = LayerNorm(dim, 'WithBias')
        self.mamba_mixer = MambaVision2DMixer(dim=dim, d_state=16, d_conv=4, ssm_expand=2, bias=qkv_bias)

        # 子层 2: Pre-Norm + GDFN (对应原版 Mlp)
        self.norm_ffn = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=qkv_bias)

        # 上采样抗混叠平滑映射层
        self.up_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
        )
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(init_gamma)))

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        # 1. 均值池化低通下采样至 H/2 × W/2
        x_low = F.avg_pool2d(x_pad, kernel_size=2, stride=2)

        # 2. 2D-MambaVision 全局空间扫描 + GDFN 通道非线性变换
        feat = x_low + self.mamba_mixer(self.norm_mamba(x_low))
        feat = feat + self.ffn(self.norm_ffn(feat))

        # 3. 提取低频全局上下文增量并平滑上采样回原分辨率
        delta = feat - x_low
        out = F.interpolate(delta, size=x_pad.shape[-2:], mode='bilinear', align_corners=False)
        out = self.up_refine(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return x + self.gamma * out


# =============================================================================
# 4. [阶段二同构融合层] 消除跨模态极性抵消 + 直接同构复用编码器算子
# =============================================================================
class MIMOHybridMambaVisionFusion(nn.Module):
    """
    低频基底同构融合层:
    1. 差模互补门控 (diff_gate): 消除红外与可见光在明暗反转区域的直接相加抵消
    2. 同构复用 BaseHybridBranch (2D-MambaVision 模块): 建立跨模态全局低频交互
    3. 正交能量对比度校准: 将融合特征空间标准差对齐至双模态正交合成能量 sqrt(var_ir + var_vis)
    """
    def __init__(self, in_dim=128, out_dim=64, num_heads=8, ffn_expansion_factor=2., bias=False):
        super(MIMOHybridMambaVisionFusion, self).__init__()
        self.dim = out_dim

        # 1. 差模互补动态门控 (替代盲目的 IR + VIS 硬相加)
        self.diff_gate = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1, bias=bias),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim * 2, kernel_size=1, bias=True),
        )

        # 2. 直接同构复用创新点二的 BaseHybridBranch (init_gamma=0.2 保证二阶段平滑切入)
        self.base_mamba_block = BaseHybridBranch(
            dim=out_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            qkv_bias=bias,
            init_gamma=0.2,
        )

        # 3. 正交对比度补偿系数 (初始 sigmoid(0.0) = 0.5，自适应恢复反相抵消损失的对比度 SD)
        self.contrast_logit = nn.Parameter(torch.zeros(1, out_dim, 1, 1))

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor = None) -> torch.Tensor:
        if feat_vis is None:
            feat_ir, feat_vis = feat_ir.chunk(2, dim=1)

        # Step 1: 差模互补动态加权
        feat_diff = torch.abs(feat_ir - feat_vis)
        gate_inp = torch.cat([feat_ir, feat_vis, feat_diff], dim=1)
        w_ir, w_vis = self.diff_gate(gate_inp).chunk(2, dim=1)
        w_ir = 1.0 + 0.25 * torch.tanh(w_ir)
        w_vis = 1.0 + 0.25 * torch.tanh(w_vis)
        base_anchor = w_ir * feat_ir + w_vis * feat_vis

        # Step 2: 同构 2D-MambaVision 全局交互
        fused = self.base_mamba_block(base_anchor)

        # Step 3: 正交能量对比度校准 (保护并提升 SD 与 EN)
        std_ir = torch.sqrt(feat_ir.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-5)
        std_vis = torch.sqrt(feat_vis.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-5)
        std_ortho = torch.sqrt(std_ir ** 2 + std_vis ** 2)

        mean_f = fused.mean(dim=(-2, -1), keepdim=True)
        std_f = torch.sqrt(fused.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-5)

        ratio = torch.clamp(std_ortho / std_f, min=0.95, max=1.35)
        alpha_c = torch.sigmoid(self.contrast_logit)
        scale = 1.0 + alpha_c * (ratio - 1.0)
        fused = mean_f + scale * (fused - mean_f)

        return fused


class CrossModalityDetailFusion(nn.Module):
    """
    高频细节同构融合层:
    通过梯度幅值包络 |D_ir| + |D_vis| 消解正负极性边缘相消，并同构复用单层可逆 INN (DetailFeatureExtraction)
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
# 6. 第一性原理独立自检入口：运行 `python -m models.net_exp3`
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Restormer_Encoder().to(device)
    dec = Restormer_Decoder().to(device)
    base_fuse = MIMOHybridMambaVisionFusion(in_dim=128, out_dim=64, num_heads=8).to(device)
    detail_fuse = CrossModalityDetailFusion(dim=64, num_layers=1).to(device)

    # Check 1: 统计精简后的各模块参数量
    total_params = sum(p.numel() for m in [enc, dec, base_fuse, detail_fuse] for p in m.parameters())
    print(f"[Check 1 Passed] Total Parameters: {total_params / 1e6:.3f} M "
          f"(Enc: {sum(p.numel() for p in enc.parameters())/1e6:.3f}M, "
          f"Dec: {sum(p.numel() for p in dec.parameters())/1e6:.3f}M, "
          f"BaseFuse: {sum(p.numel() for p in base_fuse.parameters())/1e6:.3f}M)")

    # Check 2: 验证空间对比度能量保持 (确保 MambaVision2DMixer 不会把高对比度热目标抹平为单位方差)
    mixer = enc.baseFeature.mamba_mixer
    test_feat = torch.randn(1, 64, 64, 64, device=device)
    test_feat[:, :, 16:48, 16:48] *= 5.0  # 模拟中心高亮热目标区域 (5倍振幅)
    with torch.no_grad():
        mixer_out = mixer(test_feat)
        std_center = mixer_out[:, :, 16:48, 16:48].std().item()
        std_bg = mixer_out[:, :, :16, :16].std().item()
    assert std_center > std_bg * 2.5, "Contrast flattened inside MambaVision2DMixer!"
    print(f"[Check 2 Passed] Spatial Contrast Ratio Preserved -> Target Std: {std_center:.3f} vs BG Std: {std_bg:.3f}")

    # Check 3: 验证偶数尺寸 BF16 训练梯度流与奇数尺寸 (495x629) 推理对齐
    ir_even = torch.randn(2, 1, 128, 128, device=device)
    vis_even = torch.randn(2, 1, 128, 128, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        b_ir, d_ir, _ = enc(ir_even)
        b_vis, d_vis, _ = enc(vis_even)
        f_b = base_fuse(b_ir, b_vis)
        f_d = detail_fuse(d_ir, d_vis)
        out_even, _ = dec(vis_even, f_b, f_d)
        loss = out_even.float().mean()
    loss.backward()

    with torch.inference_mode():
        ir_odd = torch.randn(1, 1, 495, 629, device=device)
        vis_odd = torch.randn(1, 1, 495, 629, device=device)
        b_ir_o, d_ir_o, _ = enc(ir_odd)
        b_vis_o, d_vis_o, _ = enc(vis_odd)
        out_odd, _ = dec(vis_odd, base_fuse(b_ir_o, b_vis_o), detail_fuse(d_ir_o, d_vis_o))
    assert out_odd.shape == ir_odd.shape, "Odd resolution mismatch!"
    print(f"[Check 3 Passed] Even (128x128) & Odd (495x629) Forward/Backward Verified!")
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
# 2. 基础归一化与 Restormer 组件 (LayerNorm, MDTA Attention, GDFN FeedForward)
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
    """多 DConv 头转置自注意力 (MDTA): O(C^2) 复杂度计算全局通道互协方差"""
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
# 3. [保持与 Exp-2_v2 严格一致以无缝加载 Epoch 40 权重] 编码器低频分支组件
# =============================================================================
class MambaVision2DMixer(nn.Module):
    def __init__(self, dim=64, d_state=16, d_conv=4, ssm_expand=2, bias=False):
        super(MambaVision2DMixer, self).__init__()
        self.dim = dim
        self.d_branch = dim // 2

        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dwconv_ssm = nn.Conv2d(self.d_branch, self.d_branch, kernel_size=3,
                                    stride=1, padding=1, groups=self.d_branch, bias=bias)
        self.mamba = Mamba(
            d_model=self.d_branch,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
        )
        self.norm_ssm = LayerNorm(self.d_branch, 'WithBias')

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
        s0 = x_ssm.flatten(2).transpose(1, 2)
        s1 = s0.flip(1)
        s2 = x_ssm.transpose(2, 3).flatten(2).transpose(1, 2)
        s3 = s2.flip(1)

        s_all = torch.cat([s0, s1, s2, s3], dim=0).contiguous()
        y_all = self.mamba(s_all)
        y0, y1, y2, y3 = y_all.chunk(4, dim=0)

        y0 = y0.transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y1 = y1.flip(1).transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y2 = y2.transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)
        y3 = y3.flip(1).transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)

        out_ssm = self.norm_ssm((y0 + y1 + y2 + y3) * 0.25)
        out_sym = self.conv_sym(x_sym)
        out = self.out_proj(torch.cat([out_ssm, out_sym], dim=1))
        return out


class BaseHybridBranch(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2., qkv_bias=False):
        super(BaseHybridBranch, self).__init__()
        self.down_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
        )
        self.norm_mamba = LayerNorm(dim, 'WithBias')
        self.mamba_mixer = MambaVision2DMixer(dim=dim, d_state=16, d_conv=4, ssm_expand=2, bias=qkv_bias)

        self.norm_attn = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim=dim, num_heads=num_heads, bias=qkv_bias)

        self.norm_ffn = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=qkv_bias)

        self.up_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
        )
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        x_low = F.avg_pool2d(x_pad, kernel_size=2, stride=2)
        feat = x_low + self.down_refine(x_low)

        feat = feat + self.mamba_mixer(self.norm_mamba(feat))
        feat = feat + self.attn(self.norm_attn(feat))
        feat = feat + self.ffn(self.norm_ffn(feat))

        delta = feat - x_low
        out = F.interpolate(delta, size=x_pad.shape[-2:], mode='bilinear', align_corners=False)
        out = self.up_refine(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return x + self.gamma * out


# =============================================================================
# 4. [Exp-3 核心创新] 阶段二：跨模态复数同构 MIMO MambaVision 融合层与高频调制融合层
# =============================================================================
class ComplexMIMO2DMambaBlock(nn.Module):
    """
    复数域同构 MIMO MambaVision 块 (Complex-Isomorphic MIMO 2D-MambaVision Block):
    将红外特征视为实部 (Real)，可见光特征视为虚部 (Imaginary): Z = X_ir + i * X_vis。
    1. 复数反对称输入耦合: [U_re; U_im] = [W_re, -W_im; W_im, W_re] [X_ir; X_vis]
    2. 二维空间连续扫描: 在不拆散空间坐标的前提下，同步扫描实部流与虚部流
    3. 复数状态旋转解调: [H'_re; H'_im] = [C_re, -C_im; C_im, C_re] [H_re; H_im]
    4. 移除会压缩对比度的逐像素 LayerNorm，改用通道能量缩放因子，100% 保护全局对比度 SD
    """
    def __init__(self, dim=64, d_state=16, d_conv=4, ssm_expand=2, bias=False):
        super(ComplexMIMO2DMambaBlock, self).__init__()
        self.dim = dim
        self.d_complex = dim // 2  # 32 维实部 + 32 维虚部 = 64 维复数状态流

        # 1. 复数同构输入投影算子 W_c = W_re + i * W_im
        self.W_re = nn.Conv2d(dim, self.d_complex, kernel_size=1, bias=bias)
        self.W_im = nn.Conv2d(dim, self.d_complex, kernel_size=1, bias=bias)

        # 局部二维空間先验深度卷积
        self.dwconv_re = nn.Conv2d(self.d_complex, self.d_complex, kernel_size=3,
                                   stride=1, padding=1, groups=self.d_complex, bias=bias)
        self.dwconv_im = nn.Conv2d(self.d_complex, self.d_complex, kernel_size=3,
                                   stride=1, padding=1, groups=self.d_complex, bias=bias)

        # 共享复数状态转移核 (实部与虚部通过反对称耦合共享动态系统参数)
        self.mamba_ssm = Mamba(
            d_model=self.d_complex,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
        )

        # 2. 复数状态旋转解调矩阵 C_c = C_re + i * C_im (交织实部与虚部隐状态)
        self.C_re = nn.Conv2d(self.d_complex, self.d_complex, kernel_size=1, bias=bias)
        self.C_im = nn.Conv2d(self.d_complex, self.d_complex, kernel_size=1, bias=bias)

        # 通道级能量缩放 (替代会抹平空间对比度的逐像素 LayerNorm)
        self.scale_ssm = nn.Parameter(torch.ones(1, dim, 1, 1))

        # 3. MambaVision 对称局部卷积分支 (直接从 128 维拼接特征提取局部跨模态补偿)
        self.sym_branch = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
            nn.SiLU(inplace=True),
        )

        # 最终输出融合映射
        self.out_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def _scan_2d(self, x_2d: torch.Tensor) -> torch.Tensor:
        """在不破坏二维空间连续性的前提下执行四方向全向扫描"""
        b, c, h, w = x_2d.shape
        s0 = x_2d.flatten(2).transpose(1, 2)                        # 水平正向
        s1 = s0.flip(1)                                             # 水平反向
        s2 = x_2d.transpose(2, 3).flatten(2).transpose(1, 2)        # 垂直正向
        s3 = s2.flip(1)                                             # 垂直反向
        return torch.cat([s0, s1, s2, s3], dim=0)                   # (4*B, L, C)

    def _unscan_2d(self, y_seq: torch.Tensor, b: int, c: int, h: int, w: int) -> torch.Tensor:
        """将四方向序列还原回二维空间特征图并求均值"""
        y0, y1, y2, y3 = y_seq.chunk(4, dim=0)
        y0 = y0.transpose(1, 2).reshape(b, c, h, w)
        y1 = y1.flip(1).transpose(1, 2).reshape(b, c, h, w)
        y2 = y2.transpose(1, 2).reshape(b, c, w, h).transpose(2, 3)
        y3 = y3.flip(1).transpose(1, 2).reshape(b, c, w, h).transpose(2, 3)
        return (y0 + y1 + y2 + y3) * 0.25

    def forward(self, feat_ir_low: torch.Tensor, feat_vis_low: torch.Tensor) -> torch.Tensor:
        b, c, h_d, w_d = feat_ir_low.shape

        # Step 1: 通道拼接 (Dim=1) 用于对称卷积支路
        feat_cat_low = torch.cat([feat_ir_low, feat_vis_low], dim=1)  # (B, 128, H_d, W_d)
        out_sym = self.sym_branch(feat_cat_low)                       # (B, 64, H_d, W_d)

        # Step 2: 复数域反对称耦合映射 (W_re + i*W_im)(X_ir + i*X_vis)
        u_re = F.silu(self.dwconv_re(self.W_re(feat_ir_low) - self.W_im(feat_vis_low)))
        u_im = F.silu(self.dwconv_im(self.W_im(feat_ir_low) + self.W_re(feat_vis_low)))

        # Step 3: 实部与虚部合并批次单次调用 CUDA Mamba 扫描 (8*B, L, 32)
        seq_re = self._scan_2d(u_re)
        seq_im = self._scan_2d(u_im)
        seq_both = torch.cat([seq_re, seq_im], dim=0).contiguous()
        out_both = self.mamba_ssm(seq_both)
        out_seq_re, out_seq_im = out_both.chunk(2, dim=0)

        h_re = self._unscan_2d(out_seq_re, b, self.d_complex, h_d, w_d)
        h_im = self._unscan_2d(out_seq_im, b, self.d_complex, h_d, w_d)

        # Step 4: 复数隐状态旋转交织 (C_re + i*C_im)(H_re + i*H_im)
        z_re = self.C_re(h_re) - self.C_im(h_im)
        z_im = self.C_im(h_re) + self.C_re(h_im)

        out_ssm = torch.cat([z_re, z_im], dim=1) * self.scale_ssm     # (B, 64, H_d, W_d)

        # Step 5: 聚合复数 SSM 分支与对称局部卷积分支
        return self.out_proj(torch.cat([out_ssm, out_sym], dim=1))


class MIMOHybridMambaVisionFusion(nn.Module):
    """
    [Exp-3 阶段二低频融合层] 跨模态 MIMO MambaVision 融合层:
    1. 特征通道拼接 (Concat, Dim=1) + 差模互补调制 (解决 IR+VIS 反相抵消与 SCD 下降问题)
    2. ComplexMIMO2DMambaBlock + MDTA + GDFN 全局跨模态状态交织
    3. 正交能量对比度校准 (Orthogonal Contrast Calibration): 确保融合图像空间标准差 SD 与信息熵 EN 饱满
    """
    def __init__(self, in_dim=128, out_dim=64, num_heads=8, ffn_expansion_factor=2., bias=False):
        super(MIMOHybridMambaVisionFusion, self).__init__()
        self.dim = out_dim

        # 1. 全分辨率差模互补门控分支 (基于拼接的 128 维特征 + 64 维差模特征)
        self.diff_gate = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1, bias=bias),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim * 2, kernel_size=1, bias=True),
        )

        # 2. H/2 × W/2 尺度 Complex-MIMO MambaVision + MDTA + GDFN
        self.mimo_mamba = ComplexMIMO2DMambaBlock(dim=out_dim, d_state=16, d_conv=4, ssm_expand=2, bias=bias)
        self.norm_attn = LayerNorm(out_dim, 'WithBias')
        self.attn = Attention(dim=out_dim, num_heads=num_heads, bias=bias)
        self.norm_ffn = LayerNorm(out_dim, 'WithBias')
        self.ffn = FeedForward(dim=out_dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.up_refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=bias),
        )

        # 渐进式热启动缩放因子 (初始为 0.1，防止 Epoch 41 切换瞬间冲击预训练解码器)
        self.gamma_mimo = nn.Parameter(torch.full((1, out_dim, 1, 1), 0.1))
        # 正交对比度补偿系数 (初始 sigmoid(-1.5) ≈ 0.18，自适应补偿反相抵消损失的对比度 SD)
        self.contrast_logit = nn.Parameter(torch.full((1, out_dim, 1, 1), -1.5))

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor = None) -> torch.Tensor:
        # 兼容单输入或双输入签名
        if feat_vis is None:
            feat_ir, feat_vis = feat_ir.chunk(2, dim=1)

        b, c, h, w = feat_ir.shape

        # --- A. 差模互补动态加权基底 (保护双模态差异相关性 SCD 与结构相似度 SSIM) ---
        feat_diff = torch.abs(feat_ir - feat_vis)
        gate_inp = torch.cat([feat_ir, feat_vis, feat_diff], dim=1)  # (B, 192, H, W)
        w_ir, w_vis = self.diff_gate(gate_inp).chunk(2, dim=1)
        # 在原版 (1, 1) 加法基底上叠加零均值动态互补调制，确保 Epoch 41 分布无缝衔接
        w_ir = 1.0 + 0.2 * torch.tanh(w_ir)
        w_vis = 1.0 + 0.2 * torch.tanh(w_vis)
        base_anchor = w_ir * feat_ir + w_vis * feat_vis

        # --- B. H/2 × W/2 复数同构 MIMO MambaVision 全局交互 ---
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            ir_pad = F.pad(feat_ir, (0, pad_w, 0, pad_h), mode='reflect')
            vis_pad = F.pad(feat_vis, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            ir_pad, vis_pad = feat_ir, feat_vis

        ir_low = F.avg_pool2d(ir_pad, kernel_size=2, stride=2)
        vis_low = F.avg_pool2d(vis_pad, kernel_size=2, stride=2)

        # 复数 MIMO 扫描 + MDTA 通道提纯 + GDFN
        mimo_feat = self.mimo_mamba(ir_low, vis_low)
        mimo_feat = mimo_feat + self.attn(self.norm_attn(mimo_feat))
        mimo_feat = mimo_feat + self.ffn(self.norm_ffn(mimo_feat))

        mimo_up = F.interpolate(mimo_feat, size=ir_pad.shape[-2:], mode='bilinear', align_corners=False)
        mimo_up = self.up_refine(mimo_up)
        if pad_h > 0 or pad_w > 0:
            mimo_up = mimo_up[:, :, :h, :w]

        fused = base_anchor + self.gamma_mimo * mimo_up

        # --- C. 复数正交能量对比度校准 (Orthogonal Contrast Calibration, 恢复并拉升 SD 与 EN) ---
        # 计算单模态空间标准差与正交合成目标标准差 sqrt(var_ir + var_vis)
        std_ir = torch.sqrt(feat_ir.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-5)
        std_vis = torch.sqrt(feat_vis.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-5)
        std_ortho = torch.sqrt(std_ir ** 2 + std_vis ** 2)

        mean_f = fused.mean(dim=(-2, -1), keepdim=True)
        std_f = torch.sqrt(fused.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-5)

        # 限制缩放比在 [0.9, 1.35] 安全区间内，杜绝像素过饱和伪影
        ratio = torch.clamp(std_ortho / std_f, min=0.9, max=1.35)
        alpha_c = torch.sigmoid(self.contrast_logit)
        scale = 1.0 + alpha_c * (ratio - 1.0)
        fused = mean_f + scale * (fused - mean_f)

        return fused


class CrossModalityDetailFusion(nn.Module):
    """
    [Exp-3 阶段二高频融合层] 跨模态极性感知可逆细节融合层:
    1. 将红外与可见光高频细节按通道维拼接 (Concat, Dim=1, 128维)
    2. 极性冲突消解门控: 解决红外与可见光边缘梯度一正一负相加抵消的问题
    3. INN 可逆仿射耦合 (DetailNode): 深度交织双模态高频纹理 (拉升 SF 与 Qabf)
    """
    def __init__(self, dim=64, num_layers=1):
        super(CrossModalityDetailFusion, self).__init__()
        # 跨模态局部梯度能量与极性门控
        self.conflict_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
        self.gamma_detail = nn.Parameter(torch.full((1, dim, 1, 1), 0.1))

        # INN 可逆耦合网络
        self.inn = DetailFeatureExtraction(num_layers=num_layers)

    def forward(self, detail_ir: torch.Tensor, detail_vis: torch.Tensor = None) -> torch.Tensor:
        if detail_vis is None:
            detail_ir, detail_vis = detail_ir.chunk(2, dim=1)

        # 提取梯度幅值包络 |D_ir| + |D_vis|，补偿因正负极性相反被 (D_ir + D_vis) 抵消的边缘
        mag_sum = torch.abs(detail_ir) + torch.abs(detail_vis)
        cat_inp = torch.cat([detail_ir, detail_vis, mag_sum], dim=1)  # (B, 192, H, W)

        detail_init = (detail_ir + detail_vis) + self.gamma_detail * self.conflict_gate(cat_inp)
        return self.inn(detail_init)


# =============================================================================
# 5. [保持与 Exp-2_v2 严格一致] 浅层门控卷积块、共享编码器与混合解码器
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
        self.baseFeature = BaseHybridBranch(dim=dim, num_heads=heads[2])
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
# 6. 独立自检入口：运行 `python -m models.net_exp3` 验证双模态 MIMO 融合与奇偶尺寸
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Restormer_Encoder().to(device)
    dec = Restormer_Decoder().to(device)
    base_fuse = MIMOHybridMambaVisionFusion(in_dim=128, out_dim=64, num_heads=8).to(device)
    detail_fuse = CrossModalityDetailFusion(dim=64, num_layers=1).to(device)

    # 1. 验证训练阶段 (128x128) + BF16 下 Phase II 完整前向与反向梯度流
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
    print(f"[Check 1 Passed] Even Size (128x128) -> Fused Base: {f_b.shape}, Fused Detail: {f_d.shape}, Out: {out_even.shape}")

    # 2. 验证测试集奇数尺寸 (495x629) 无报错对齐
    with torch.inference_mode():
        ir_odd = torch.randn(1, 1, 495, 629, device=device)
        vis_odd = torch.randn(1, 1, 495, 629, device=device)
        b_ir_o, d_ir_o, _ = enc(ir_odd)
        b_vis_o, d_vis_o, _ = enc(vis_odd)
        f_b_o = base_fuse(b_ir_o, b_vis_o)
        f_d_o = detail_fuse(d_ir_o, d_vis_o)
        out_odd, _ = dec(vis_odd, f_b_o, f_d_o)
    assert out_odd.shape == ir_odd.shape, "Odd resolution mismatch!"
    print(f"[Check 2 Passed] Odd Size (495x629)  -> Fused Base: {f_b_o.shape}, Fused Detail: {f_d_o.shape}, Out: {out_odd.shape}")
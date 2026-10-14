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
# 2. 基础归一化与原版 BaseFeatureExtraction (供 Phase II BaseFuseLayer 保持控制变量)
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


class AttentionBase(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    """
    保留给 Phase II 的 self.BaseFuseLayer 使用（直到 Exp-3 再替换为 MIMO 融合层），
    避免 Phase II 发生二次重复下采样模糊，并严格保证 Exp-2 单一控制变量。
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor=1., qkv_bias=False):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


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
# 3. [Exp-2 终极审查修正版] 全向 2D-MambaVision 混合器与低频混合基底流 (BaseHybridBranch)
# =============================================================================
class MambaVision2DMixer(nn.Module):
    """
    [理论与工程双重对齐版] 2D-MambaVision Mixer:
    1. 双路对称分流: 输入映射为 d_branch (32维) 的 2D-SSM 全局扫描支路 + d_branch (32维) 的对称卷积支路。
    2. 全通道四方向全向扫描 (Omnidirectional SS2D):
       对全部 32 个 SSM 通道同时构建 4 个方向的扫描序列，沿 Batch 维并发调用 Mamba(d_model=32)，
       再将 4 个方向的结果相加平均，使每个通道、每个像素均具备全向二维感受野！
    3. Post-SSM LayerNorm: 锚定不同测试集分辨率下的状态累积方差，消除跨尺度分布漂移。
    """
    def __init__(self, dim=64, d_state=16, d_conv=4, ssm_expand=2, bias=False):
        super(MambaVision2DMixer, self).__init__()
        self.dim = dim
        self.d_branch = dim // 2  # 32 channels for SSM branch, 32 for Symmetric Conv branch

        # 输入映射并拆分为 SSM 支路 (32维) 与对称局部卷积支路 (32维)
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 支路 1: 全向 2D-SSM 全局扫描
        self.dwconv_ssm = nn.Conv2d(self.d_branch, self.d_branch, kernel_size=3,
                                    stride=1, padding=1, groups=self.d_branch, bias=bias)
        self.mamba = Mamba(
            d_model=self.d_branch,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
        )
        # 关键防漂移归一化层：稳定不同图像分辨率 (L=H*W/4) 下的 Mamba 递归输出方差
        self.norm_ssm = LayerNorm(self.d_branch, 'WithBias')

        # 支路 2: MambaVision 对称局部卷积补偿支路
        self.conv_sym = nn.Sequential(
            nn.Conv2d(self.d_branch, self.d_branch, kernel_size=3,
                      stride=1, padding=1, groups=self.d_branch, bias=bias),
            nn.SiLU(inplace=True),
        )

        # 双支路融合输出投影
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, 64, H_d, W_d)
        b, c, h_d, w_d = x.shape
        x_ssm, x_sym = self.in_proj(x).chunk(2, dim=1)  # 各自为 (B, 32, H_d, W_d)

        # --- 支路 1: 全通道四方向 2D-SSM 扫描 ---
        x_ssm = F.silu(self.dwconv_ssm(x_ssm))

        # 全部 32 通道同时按 4 个方向展平为 (B, L, 32)
        s0 = x_ssm.flatten(2).transpose(1, 2)                           # 方向 0: 水平正向
        s1 = s0.flip(1)                                                 # 方向 1: 水平反向
        s2 = x_ssm.transpose(2, 3).flatten(2).transpose(1, 2)           # 方向 2: 垂直正向
        s3 = s2.flip(1)                                                 # 方向 3: 垂直反向

        # 4 个方向共享完全相同的 32 通道语义空间，沿 Batch 维合并为 (4*B, L, 32) 单次扫描
        s_all = torch.cat([s0, s1, s2, s3], dim=0).contiguous()
        y_all = self.mamba(s_all)
        y0, y1, y2, y3 = y_all.chunk(4, dim=0)

        # 逆向还原回二维空间坐标 (B, 32, H_d, W_d)
        y0 = y0.transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y1 = y1.flip(1).transpose(1, 2).reshape(b, self.d_branch, h_d, w_d)
        y2 = y2.transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)
        y3 = y3.flip(1).transpose(1, 2).reshape(b, self.d_branch, w_d, h_d).transpose(2, 3)

        # 四方向全向聚合 + 方差归一化
        out_ssm = self.norm_ssm((y0 + y1 + y2 + y3) * 0.25)

        # --- 支路 2: 对称局部卷积分支 ---
        out_sym = self.conv_sym(x_sym)

        # --- 双支路拼接并映射回 64 通道 ---
        out = self.out_proj(torch.cat([out_ssm, out_sym], dim=1))
        return out


class BaseHybridBranch(nn.Module):
    """
    [Exp-2 阶段一专属] 低频基底流 (Base Hybrid Branch):
    严格对应论文架构图:
    1. 降采样层 (AvgPool 低通滤波 + 残差卷积微调): 分辨率降至 H/2 × W/2，滤除高频噪点并扩大相对感受野
    2. MambaVision2DMixer: O(N) 复杂度全局二维空间四向扫描
    3. MDTA (Attention): O(C^2) 复杂度计算通道互协方差提纯基底特征
    4. GDFN (FeedForward): 门控前馈网络
    5. 平滑增量上采样: 仅对低频上下文增量 (feat - x_low) 进行双线性上采样与卷积平滑，彻底杜绝信号翻倍与棋盘格伪影
    """
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2., qkv_bias=False):
        super(BaseHybridBranch, self).__init__()

        # 1. 残差式低通下采样微调层 (在 AvgPool2d 基础上做平滑特征调整，保证初始第 1 轮即有纯净低频信号)
        self.down_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
        )

        # 2. 全向 2D-MambaVision Mixer
        self.norm_mamba = LayerNorm(dim, 'WithBias')
        self.mamba_mixer = MambaVision2DMixer(dim=dim, d_state=16, d_conv=4, ssm_expand=2, bias=qkv_bias)

        # 3. MDTA (多 DConv 头转置自注意力)
        self.norm_attn = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim=dim, num_heads=num_heads, bias=qkv_bias)

        # 4. GDFN (门控前馈网络)
        self.norm_ffn = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=qkv_bias)

        # 5. 平滑上采样映射层
        self.up_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
        )
        # 可学习的残差缩放系数
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape

        # 动态奇偶 Padding，兼容任意奇数尺寸测试集图像
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        # 1. 物理低通下采样至 H_p/2 × W_p/2
        x_low = F.avg_pool2d(x_pad, kernel_size=2, stride=2)
        feat = x_low + self.down_refine(x_low)

        # 2. MambaVision 全局二维空间扫描 + MDTA 通道提纯 + GDFN
        feat = feat + self.mamba_mixer(self.norm_mamba(feat))
        feat = feat + self.attn(self.norm_attn(feat))
        feat = feat + self.ffn(self.norm_ffn(feat))

        # 3. 仅提取低频上下文修正增量 delta，双线性平滑上采样回原尺寸 (避免 x_low 被重复叠加两次)
        delta = feat - x_low
        out = F.interpolate(delta, size=x_pad.shape[-2:], mode='bilinear', align_corners=False)
        out = self.up_refine(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return x + self.gamma * out


# =============================================================================
# 4. [沿用 Exp-1 验证成果] 浅层门控卷积块、共享编码器与混合解码器
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
        # [Exp-2 严格单变量控制] 编码器低频基底流升级为 BaseHybridBranch (H/2 MambaVision + MDTA + GDFN)
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Restormer_Encoder().to(device)
    dec = Restormer_Decoder().to(device)
    base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)

    x_even = torch.randn(2, 1, 128, 128, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        b_feat, d_feat, _ = enc(x_even)
        out_even, _ = dec(x_even, b_feat, d_feat)
        loss = out_even.float().mean()
    loss.backward()
    print(f"[Check 1 Passed] Even Size (128x128) -> Base: {b_feat.shape}, Detail: {d_feat.shape}, Out: {out_even.shape}")

    with torch.inference_mode():
        x_odd = torch.randn(1, 1, 495, 629, device=device)
        b_odd, d_odd, _ = enc(x_odd)
        f_b = base_fuse(b_odd)
        f_d = detail_fuse(d_odd)
        out_odd, _ = dec(x_odd, f_b, f_d)
    assert out_odd.shape == x_odd.shape, "Odd resolution mismatch!"
    print(f"[Check 2 Passed] Odd Size (495x629)  -> Base: {b_odd.shape}, Detail: {d_odd.shape}, Out: {out_odd.shape}")
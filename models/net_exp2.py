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
    """多 DConv 头转置自注意力 (MDTA): O(C^2) 复杂度计算通道互协方差"""
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
# 3. [Exp-2 核心创新组件] 2D-MambaVision 混合器与多尺度低频基底流 (BaseFeatureExtraction)
# =============================================================================
class MambaVision2DMixer(nn.Module):
    """
    2D-MambaVision Mixer:
    结合 NVIDIA MambaVision 的双路对称结构（SSM 分支 + 对称局部卷积补偿分支）
    与 VMamba/RestorMixer 的四方向二维空间交叉扫描（CS-SS2D）。
    通过 Batch 维并发打包 4 个扫描方向，仅需单次 mamba_ssm CUDA 内核调用即可完成 O(N) 全局二维扫描。
    """
    def __init__(self, dim=64, d_state=16, d_conv=4, ssm_expand=2, bias=False):
        super(MambaVision2DMixer, self).__init__()
        assert dim % 4 == 0, f"dim ({dim}) must be divisible by 4 for 4-way 2D scan."
        self.dim = dim
        self.c_group = dim // 4  # 当 dim=64 时，c_group=16 (对齐 CUDA Tensor Core)

        # 输入双路投影：拆分为 2D-SSM 全局扫描分支 (dim) 与对称局部卷积分支 (dim)
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        # 分支 1 (2D-SSM 分支): 2D 深度卷积提取局部空间先验 + 四方向 Mamba 状态空间扫描
        self.dwconv_ssm = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mamba = Mamba(
            d_model=self.c_group,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
        )

        # 分支 2 (MambaVision 对称补偿分支): 局部深度可分离卷积，补偿序列展平丢失的二维邻域连续性
        self.conv_sym = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
            nn.SiLU(inplace=True),
        )

        # 双分支聚合投影
        self.out_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, dim, H_d, W_d)
        b, c, h_d, w_d = x.shape
        x_ssm, x_sym = self.in_proj(x).chunk(2, dim=1)

        # --- 1. 2D-SSM 四方向全局扫描分支 ---
        x_ssm = F.silu(self.dwconv_ssm(x_ssm))
        # 按通道均分为 4 组，每组形状 (B, c_group, H_d, W_d)，保证每个序列位置 l 的通道属于同一像素
        x0, x1, x2, x3 = x_ssm.chunk(4, dim=1)

        # 构造 4 个二维扫描方向的序列，形状均为 (B, H_d * W_d, c_group)
        s0 = x0.flatten(2).transpose(1, 2)                              # 方向 0: 水平正向 (Row-major Fwd)
        s1 = x1.flatten(2).flip(-1).transpose(1, 2)                     # 方向 1: 水平反向 (Row-major Bwd)
        s2 = x2.transpose(2, 3).flatten(2).transpose(1, 2)              # 方向 2: 垂直正向 (Col-major Fwd)
        s3 = x3.transpose(2, 3).flatten(2).flip(-1).transpose(1, 2)     # 方向 3: 垂直反向 (Col-major Bwd)

        # 沿 Batch 维合并为 (4*B, L, c_group) 执行单次高速 CUDA 扫描
        s_all = torch.cat([s0, s1, s2, s3], dim=0).contiguous()
        y_all = self.mamba(s_all)
        y0, y1, y2, y3 = y_all.chunk(4, dim=0)

        # 逆变换精确还原回二维空间坐标 (B, c_group, H_d, W_d)
        y0 = y0.transpose(1, 2).reshape(b, self.c_group, h_d, w_d)
        y1 = y1.transpose(1, 2).flip(-1).reshape(b, self.c_group, h_d, w_d)
        y2 = y2.transpose(1, 2).reshape(b, self.c_group, w_d, h_d).transpose(2, 3)
        y3 = y3.transpose(1, 2).flip(-1).reshape(b, self.c_group, w_d, h_d).transpose(2, 3)

        out_ssm = torch.cat([y0, y1, y2, y3], dim=1).contiguous()

        # --- 2. 对称局部卷积分支 ---
        out_sym = self.conv_sym(x_sym)

        # --- 3. 双路拼接与输出投影 ---
        out = self.out_proj(torch.cat([out_ssm, out_sym], dim=1))
        return out


class BaseFeatureExtraction(nn.Module):
    """
    [Exp-2 升级] 低频混合基底流 (Base Hybrid Branch):
    1. 动态奇偶反射填充 + PixelUnshuffle(2) 无损下采样至 H/2 × W/2
    2. MambaVision2DMixer: O(N) 复杂度全局二维空间扫描
    3. MDTA (Attention): O(C^2) 复杂度计算通道互协方差提纯基底特征
    4. GDFN (FeedForward): 门控前馈网络非线性特征变换
    5. PixelShuffle(2) 上采样 + 动态裁剪恢复至原始分辨率 H × W
    """
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2., qkv_bias=False):
        super(BaseFeatureExtraction, self).__init__()

        # 1. 下采样层: (B, dim, H_p, W_p) -> (B, dim*4, H_p/2, W_p/2) -> (B, dim, H_p/2, W_p/2)
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=qkv_bias),
        )

        # 2. 2D-MambaVision Mixer (全局二维空间状态扫描)
        self.norm_mamba = LayerNorm(dim, 'WithBias')
        self.mamba_mixer = MambaVision2DMixer(dim=dim, d_state=16, d_conv=4, ssm_expand=2, bias=qkv_bias)

        # 3. MDTA (多 DConv 头转置自注意力)
        self.norm_attn = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim=dim, num_heads=num_heads, bias=qkv_bias)

        # 4. GDFN (门控前馈网络)
        self.norm_ffn = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=qkv_bias)

        # 5. 上采样层: (B, dim, H_p/2, W_p/2) -> (B, dim*4, H_p/2, W_p/2) -> (B, dim, H_p, W_p)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, bias=qkv_bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        # 输入 x: (B, 64, H, W)，兼容任意奇偶分辨率
        b, c, h, w = x.shape

        # 动态奇偶 Padding，防止测试集奇数高宽在下采样时丢失 1 像素
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        # 1. 下采样至 H_p/2 × W_p/2
        feat = self.downsample(x_pad)

        # 2. 2D-MambaVision 全局空间扫描
        feat = feat + self.mamba_mixer(self.norm_mamba(feat))

        # 3. MDTA 通道互协方差提纯
        feat = feat + self.attn(self.norm_attn(feat))

        # 4. GDFN 门控前馈变换
        feat = feat + self.ffn(self.norm_ffn(feat))

        # 5. PixelShuffle 上采样并裁剪回原始 (H, W)
        out = self.upsample(feat)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return x + out


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
        # Exp-2: 调用升级后的多尺度 2D-MambaVision 低频混合分支
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
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

        # 严格保持 Sigmoid 在 + inp_img 之后，杜绝恒等残差短路
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
# 5. 独立自检入口：可在训练前运行 `python -m models.net_exp2` 验证维度与梯度流
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Restormer_Encoder().to(device)
    dec = Restormer_Decoder().to(device)
    base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)

    # 1. 验证训练阶段偶数尺寸 (128x128) + BF16 前向与反向梯度流
    x_even = torch.randn(2, 1, 128, 128, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        b_feat, d_feat, _ = enc(x_even)
        out_even, _ = dec(x_even, b_feat, d_feat)
        loss = out_even.float().mean()
    loss.backward()
    print(f"[Check 1 Passed] Even Size (128x128) -> Base: {b_feat.shape}, Detail: {d_feat.shape}, Out: {out_even.shape}")

    # 2. 验证测试集奇数尺寸 (495x629) 无报错对齐
    with torch.inference_mode():
        x_odd = torch.randn(1, 1, 495, 629, device=device)
        b_odd, d_odd, _ = enc(x_odd)
        f_b = base_fuse(b_odd)
        f_d = detail_fuse(d_odd)
        out_odd, _ = dec(x_odd, f_b, f_d)
    assert out_odd.shape == x_odd.shape, "Odd resolution mismatch!"
    print(f"[Check 2 Passed] Odd Size (495x629)  -> Base: {b_odd.shape}, Detail: {d_odd.shape}, Out: {out_odd.shape}")
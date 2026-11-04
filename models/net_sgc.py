"""B混合版：共享编码器浅层采用3个SGC块，再接1个原版Transformer块。

保留原版patch embedding、基础/细节分支、GDFN、解码器与融合层。
训练入口须从本模块导入；原4-SGC版检查点与本结构不兼容，须另起实验。
这是模块替换的受控候选，不代表质量/速度或新颖性已经验证。
"""
from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.net import (
    Attention,
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    FeedForward,
    # LayerNorm,
    OverlapPatchEmbed,
    # Restormer_Decoder,
    # TransformerBlock,
)

__all__ = [
    "ShallowGatedCNNBlock",
    "ShallowGatedCNNBlockV2",
    "ShallowGatedCNNBlockv3",
    "Restormer_Encoder",
    "Restormer_Decoder",
    "BaseFeatureExtraction",
    "DetailFeatureExtraction",
]

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) 模块
    论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (CVPR 2020)
    """
    def __init__(self, channels: int = None, k_size: int = None, gamma: int = 2, b: int = 1):
        super(ECABlock, self).__init__()
        if k_size is None:
            assert channels is not None, "当 k_size 为 None 时必须提供 channels 维度"
            t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
            k_size = t if t % 2 == 1 else t + 1
        
        self.k_size = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=k_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class ShallowGatedCNNBlock(nn.Module):

    def __init__(
        self,
        dim: int = 64,
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.dwconv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=bias)
        self.eca = ECABlock(channels=dim)
        self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.proj_in(self.norm1(x)).chunk(2, dim=1)
        gated = F.silu(self.dwconv3x3(u)) * self.dwconv5x5(v)
        return self.norm2(self.eca(gated))


class ShallowGatedCNNBlockV2(nn.Module):

    def __init__(
        self,
        dim: int = 64,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.dwconv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.proj_in(self.norm1(x)).chunk(2, dim=1)
        gated = F.silu(self.dwconv3x3(u)) * self.dwconv5x5(v)
        return self.norm2(x + self.proj_out(gated))


class Prominent(nn.Module):
    """DCEvo prominent 空间调制；低精度计算提升至FP32并保护零分母。"""
    def __init__(self, channels: int = 0):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = torch.sigmoid(work)
        mean = y.mean(dim=(2, 3), keepdim=True).clamp_min(torch.finfo(y.dtype).tiny)
        return F.silu(work * torch.sigmoid(y.square() / mean)).to(dtype=x.dtype)


class ShallowGatedCNNBlockv3(nn.Module):

    def __init__(
        self,
        dim: int = 64,
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        layer_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.dwconv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=bias)
        self.eca = ECABlock(channels=dim)
        self.proj_out = nn.Conv2d(3 * dim, dim, 1, bias=bias)
        self.prominent = Prominent()
        self.scale1 = nn.Parameter(torch.full((1, dim, 1, 1), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.proj_in(self.norm1(x)).chunk(2, dim=1)
        gated = F.silu(self.dwconv3x3(u)) * self.dwconv5x5(v)
        output = F.silu(self.eca(gated)) + gated
        merged = torch.cat((u, v, output), dim=1)
        refined = F.silu(self.proj_out(merged))
        return x + self.scale1 * self.prominent(refined)


class Restormer_Encoder(nn.Module):
    def __init__(
        self,
        inp_channels: int = 1,
        out_channels: int = 1,
        dim: int = 64,
        num_blocks: Sequence[int] = (4, 4),
        heads: Sequence[int] = (8, 8, 8),
        ffn_expansion_factor: float = 2,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[ShallowGatedCNNBlockv3(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])],
        )
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shallow = self.encoder_level1(self.patch_embed(inp_img))
        return self.baseFeature(shallow), self.detailFeature(shallow), shallow
    

class LayerNorm(nn.Module):
    """DCEvo输出头的逐位置通道归一化，不使用可学习缩放。"""

    def __init__(self, dim: int, norm_type: str) -> None:
        super().__init__()
        self.bias_free = norm_type == "BiasFree"
        self.bias = None if self.bias_free else nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        if self.bias_free:
            return x / torch.sqrt(variance + 1e-5)
        mean = x.mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + 1e-5) + self.bias.view(1, -1, 1, 1)


class TransformerBlock(nn.Module):
    """参考DCEvo DE_Decoder输出头：两个Prominent残差分支。"""

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float,
                 bias: bool, LayerNorm_type: str = "WithBias") -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.att1 = Prominent(channels=dim)
        self.att2 = Prominent(channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.att1(self.attn(self.norm1(x)))
        return x + self.att2(self.ffn(self.norm2(x)))


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
            *[ShallowGatedCNNBlockv3(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])],
        )
        self.output = nn.Sequential(
            TransformerBlock(dim, heads[1], ffn_expansion_factor, bias, LayerNorm_type),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, stride=1, padding=0, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        embedded_feature = torch.cat((base_feature, detail_feature), dim=1)
        embedded_feature = self.reduce_channel(embedded_feature)
        base_feature = self.encoder_level2(embedded_feature)
        if inp_img is not None:
            if len(inp_img) == 1:
                out = self.output(base_feature) + inp_img[0]
            elif len(inp_img) == 2:
                out = self.output(base_feature) + (inp_img[0] + inp_img[1]) * 0.5
            else:
                raise ValueError(f"inp_img must be a list of length 1 or 2, but got {len(inp_img)}")
        else:
            out = self.output(base_feature)
        return self.sigmoid(out), base_feature
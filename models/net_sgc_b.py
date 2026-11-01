"""B混合版：共享编码器浅层采用3个SGC块，再接1个原版Transformer块。

保留原版patch embedding、基础/细节分支、GDFN、解码器与融合层。
训练入口须从本模块导入；原4-SGC版检查点与本结构不兼容，须另起实验。
这是模块替换的受控候选，不代表质量/速度或新颖性已经验证。
"""
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    FeedForward,
    LayerNorm,
    OverlapPatchEmbed,
    Restormer_Decoder,
    TransformerBlock,
)

__all__ = [
    "ShallowGatedCNNBlock",
    "ShallowGatedCNNBlockV2",
    "Restormer_Encoder",
    "Restormer_Decoder",
    "BaseFeatureExtraction",
    "DetailFeatureExtraction",
]


class ShallowGatedCNNBlock(nn.Module):
    """3x3/5x5逐通道卷积门控，原分辨率，双残差及原版GDFN。"""

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
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.proj_in(self.norm1(x)).chunk(2, dim=1)
        gated = F.silu(self.dwconv3x3(u)) * self.dwconv5x5(v)
        x = x + self.proj_out(gated * self.sca(gated))
        return x + self.ffn(self.norm2(x))


class ShallowGatedCNNBlockV2(nn.Module):
    """无通道注意力及GDFN：多尺度门控、投影、残差相加后做输出LN。

    保留输入LN，输入输出均为(B, dim, H, W)。需显式实例化本版本。
    """

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
        if not num_blocks or num_blocks[0] != 4:
            raise ValueError("B混合版固定4个浅层块（3 SGC + 1 Transformer）；减少层数应另设实验。")
        if len(heads) < 3:
            raise ValueError("heads须保留原版的三个配置项。")
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            ShallowGatedCNNBlockV2(dim=dim, bias=bias, LayerNorm_type=LayerNorm_type),
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias, LayerNorm_type=LayerNorm_type),
            ShallowGatedCNNBlockV2(dim=dim, bias=bias, LayerNorm_type=LayerNorm_type),
            TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias, LayerNorm_type=LayerNorm_type),
        )
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shallow = self.encoder_level1(self.patch_embed(inp_img))
        return self.baseFeature(shallow), self.detailFeature(shallow), shallow
    

class Restormer_Decoder_bak(nn.Module):
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
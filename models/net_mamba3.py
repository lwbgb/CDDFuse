import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mamba_ssm import Mamba3


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm_Optimized(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm_Optimized, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, 1e-5)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm_Optimized(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


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
        return x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        self.net = nn.ModuleList([DetailNode() for _ in range(num_layers)])

    def forward(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


############################ 优化后的 Mamba 模块 ############################
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, headdim=64):
        super().__init__()
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 构造 4 向序列
        x_hw = x.flatten(2).transpose(1, 2)  # (B, L, C)
        x_wh = x.transpose(2, 3).contiguous().flatten(2).transpose(1, 2)  # (B, L, C)
        x_hw_rev = torch.flip(x_hw, dims=[1])
        x_wh_rev = torch.flip(x_wh, dims=[1])

        # 🚀 显存与速度优化：在 Batch 维度拼接，一次性调用 Mamba3
        x_merged = torch.cat([x_hw, x_wh, x_hw_rev, x_wh_rev], dim=0)  # (4B, L, C)
        out_merged = self.mamba(x_merged)

        out_hw, out_wh, out_hw_rev, out_wh_rev = torch.chunk(out_merged, 4, dim=0)

        out_hw_rev = torch.flip(out_hw_rev, dims=[1])
        out_wh_rev = torch.flip(out_wh_rev, dims=[1])
        out_wh = out_wh.transpose(1, 2).view(B, C, W, H).transpose(2, 3).contiguous().flatten(2).transpose(1, 2)

        out = (out_hw + out_wh + out_hw_rev + out_wh_rev) * 0.25
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class BaseMambaEncoder(nn.Module):
    def __init__(self, dim, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim)
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.local_dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def _forward_impl(self, x):
        shortcut = x
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.ss2d(x_norm) + self.local_dwc(x)
        out = self.conv_out(out)
        return out + shortcut

    def forward(self, x):
        # 🚀 修改为 use_reentrant=True
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=True)
        return self._forward_impl(x)


class MIMOMambaFusion(nn.Module):
    def __init__(self, in_dim=128, out_dim=64, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.input_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.ss2d = SS2D(d_model=out_dim)
        self.out_proj = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def _forward_impl(self, x_ir, x_vis):
        x_concat = torch.cat([x_ir, x_vis], dim=1)
        x = self.act(self.input_proj(x_concat))
        shortcut = x
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.ss2d(x_norm)
        out = self.out_proj(out)
        return out + shortcut

    def forward(self, x_ir, x_vis):
        # 🚀 修改为 use_reentrant=True
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward_impl, x_ir, x_vis, use_reentrant=True)
        return self._forward_impl(x_ir, x_vis)


class Restormer_Encoder(nn.Module):
    def __init__(self, inp_channels=1, dim=64, num_blocks=[4, 4]):
        super(Restormer_Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[BaseMambaEncoder(dim=dim) for _ in range(num_blocks[0])])
        self.baseFeature = BaseMambaEncoder(dim=dim)
        self.detailFeature = DetailFeatureExtraction(num_layers=3)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, num_blocks=[4, 4], bias=False):
        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[BaseMambaEncoder(dim=dim) for _ in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(inplace=True),
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
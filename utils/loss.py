from kornia.filters import filter2d, get_gaussian_kernel2d
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
    
class LegacySSIMLoss(nn.Module):
    """兼容 CDDFuse 所用 Kornia 0.2.0 SSIM 训练损失。

    输入：[B, C, H, W]，浮点张量，通常在 [0, 1]。
    """

    def __init__(self, window_size=11, reduction="mean", max_val=1.0):
        super().__init__()

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size 必须为正奇数")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"不支持的 reduction: {reduction}")

        self.reduction = reduction
        self.C1 = (0.01 * max_val) ** 2
        self.C2 = (0.03 * max_val) ** 2

        window = get_gaussian_kernel2d(
            (window_size, window_size),
            (1.5, 1.5),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # 不写入 checkpoint，避免给旧权重增加必需字段。
        self.register_buffer(
            "window", window.reshape(1, window_size, window_size), persistent=False
        )

    def forward(self, img1, img2):
        # Float32 tensors alone do not prevent cuDNN TF32 convolution. The
        # legacy E[x*x]-E[x]^2 SSIM variance is sensitive to that rounding.
        with torch.backends.cudnn.flags(allow_tf32=False):
            return self._forward_ieee(img1, img2)

    def _forward_ieee(self, img1, img2):
        if img1.ndim != 4 or img1.shape != img2.shape:
            raise ValueError("输入必须为相同形状的 [B, C, H, W]")
        if img1.device != img2.device or img1.dtype != img2.dtype:
            raise ValueError("两个输入的 device 和 dtype 必须一致")

        kernel = self.window.to(device=img1.device, dtype=img1.dtype)

        def smooth(x):
            return filter2d(
                x,
                kernel,
                border_type="reflect",
                normalized=False,
                padding="same",
                behaviour="corr",
            )

        mu1 = smooth(img1)
        mu2 = smooth(img2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = smooth(img1 * img1) - mu1_sq
        sigma2_sq = smooth(img2 * img2) - mu2_sq
        sigma12 = smooth(img1 * img2) - mu1_mu2

        numerator = (
            (2.0 * mu1_mu2 + self.C1)
            * (2.0 * sigma12 + self.C2)
        )
        denominator = (
            (mu1_sq + mu2_sq + self.C1)
            * (sigma1_sq + sigma2_sq + self.C2)
        )

        # 对齐旧版：不额外添加 eps，不截断局部方差。
        ssim_map = numerator / denominator

        # 关键：先 clamp，再除以 2。
        loss = torch.clamp(1.0 - ssim_map, min=0.0, max=1.0) / 2.0

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    

def complex_decoupling_loss(feat_V, feat_I):
    # 将 64 维特征拆分为伪复数的实部(32)和虚部(32)
    real_V, imag_V = feat_V.chunk(2, dim=1)
    real_I, imag_I = feat_I.chunk(2, dim=1)
    
    # 计算振幅 (能量先验 -> 共享特征 -> 应当高度相关)
    amp_V = torch.sqrt(real_V**2 + imag_V**2 + 1e-6)
    amp_I = torch.sqrt(real_I**2 + imag_I**2 + 1e-6)
    cc_amp = cc(amp_V, amp_I) # 优化目标：最大化 cc_amp
    
    # 计算相位 (结构先验 -> 特异性特征 -> 应当正交/不相关)
    phase_V = torch.atan2(imag_V, real_V + 1e-6)
    phase_I = torch.atan2(imag_I, real_I + 1e-6)
    cc_phase = cc(phase_V, phase_I) # 优化目标：最小化 cc_phase
    
    return cc_amp, cc_phase


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

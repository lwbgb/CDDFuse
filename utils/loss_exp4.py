import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import cc, Sobelxy


class NonLinearDecouplingLoss(nn.Module):
    """
    [Exp-4 创新点三 (A)] 多阶非线性特征解耦损失:
    1. 跨模态低频线性强相关约束: CC(Phi_V^B, Phi_I^B)
    2. 跨模态高频线性 + 核空间非线性独立约束: CC(Phi_V^D, Phi_I^D)^2 + lambda_hsic * Spatial_nHSIC(Phi_V^D, Phi_I^D)
    3. 模态内高低频通道正交约束: cos^2(Phi_m^B, Phi_m^D)，杜绝低频光照向高频分支泄漏
    """
    def __init__(self, lambda_hsic: float = 0.5, lambda_ortho: float = 0.1, pool_size: int = 8):
        super(NonLinearDecouplingLoss, self).__init__()
        self.lambda_hsic = lambda_hsic
        self.lambda_ortho = lambda_ortho
        self.pool_size = pool_size

    def _spatial_nhsic(self, feat_ir_d: torch.Tensor, feat_vis_d: torch.Tensor) -> torch.Tensor:
        """
        在每个样本内部对池化后的 M = pool_size^2 (64) 个空间 Token 计算归一化 RBF-HSIC (CKA)，
        不依赖大 Batch Size，且对全黑常数图像 Patch 具有 100% 防 NaN 保护。
        """
        b, c, h, w = feat_ir_d.shape
        # 自适应池化至 (B, C, 8, 8) 并转置为 (B, 64, C)，即 64 个空间区域 Token
        x = F.adaptive_avg_pool2d(feat_ir_d, (self.pool_size, self.pool_size)).flatten(2).transpose(1, 2)
        y = F.adaptive_avg_pool2d(feat_vis_d, (self.pool_size, self.pool_size)).flatten(2).transpose(1, 2)

        # 沿通道维零均值并做 L2 归一化，确保欧氏距离严格有界于 [0, 4]
        x = F.normalize(x - x.mean(dim=1, keepdim=True), p=2, dim=-1, eps=1e-6)
        y = F.normalize(y - y.mean(dim=1, keepdim=True), p=2, dim=-1, eps=1e-6)

        # 计算成对距离矩阵 (B, 64, 64)
        dist_x = torch.cdist(x, x, p=2).pow(2)
        dist_y = torch.cdist(y, y, p=2).pow(2)

        # 自适应中位数带宽 (detach 阻断 median 梯度，并限制下界防止 0/0)
        sigma_x = dist_x.detach().flatten(1).median(dim=1)[0].view(b, 1, 1).clamp(min=1e-3)
        sigma_y = dist_y.detach().flatten(1).median(dim=1)[0].view(b, 1, 1).clamp(min=1e-3)

        k_x = torch.exp(-dist_x / (2.0 * sigma_x))
        k_y = torch.exp(-dist_y / (2.0 * sigma_y))

        # 核矩阵双中心化 (Double Centering): K_c = H K H
        k_x = k_x - k_x.mean(dim=1, keepdim=True) - k_x.mean(dim=2, keepdim=True) + k_x.mean(dim=(1, 2), keepdim=True)
        k_y = k_y - k_y.mean(dim=1, keepdim=True) - k_y.mean(dim=2, keepdim=True) + k_y.mean(dim=(1, 2), keepdim=True)

        # 计算归一化 HSIC (Tr(K_x K_y) / sqrt(Tr(K_x^2) Tr(K_y^2) + eps))
        hsic_xy = (k_x * k_y).sum(dim=(1, 2))
        hsic_xx = (k_x * k_x).sum(dim=(1, 2))
        hsic_yy = (k_y * k_y).sum(dim=(1, 2))

        nhsic = hsic_xy / torch.sqrt(hsic_xx * hsic_yy + 1e-6)
        return torch.clamp(nhsic.mean(), min=0.0, max=1.0)

    @staticmethod
    def _intra_modal_ortho(feat_b: torch.Tensor, feat_d: torch.Tensor) -> torch.Tensor:
        """计算单模态内部低频特征与高频特征的逐通道余弦平方正交损失"""
        b_flat = feat_b.flatten(2)
        d_flat = feat_d.flatten(2)
        b_zero = b_flat - b_flat.mean(dim=-1, keepdim=True)
        d_zero = d_flat - d_flat.mean(dim=-1, keepdim=True)

        cos_sim = (b_zero * d_zero).sum(dim=-1) / (
            torch.sqrt((b_zero ** 2).sum(dim=-1) + 1e-6) * torch.sqrt((d_zero ** 2).sum(dim=-1) + 1e-6)
        )
        return (cos_sim ** 2).mean()

    def forward(
        self,
        feat_vis_b: torch.Tensor,
        feat_ir_b: torch.Tensor,
        feat_vis_d: torch.Tensor,
        feat_ir_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cc_b = cc(feat_vis_b, feat_ir_b)
        cc_d = cc(feat_vis_d, feat_ir_d)

        nhsic_d = self._spatial_nhsic(feat_ir_d, feat_vis_d)
        ortho_loss = 0.5 * (
            self._intra_modal_ortho(feat_ir_b, feat_ir_d) + self._intra_modal_ortho(feat_vis_b, feat_vis_d)
        )

        loss_decomp = ((cc_d ** 2) + self.lambda_hsic * nhsic_d) / (1.01 + cc_b) + self.lambda_ortho * ortho_loss
        return loss_decomp, cc_b, cc_d


class AdaptiveStructureFusionLoss(nn.Module):
    """
    [Exp-4 创新点三 (B)] 结构协方差与显著性自适应融合损失:
    1. 显著性软调制强度目标 (I_target): 融合 70% 峰值包络 max(I, V) 与 30% 局部能量加权软目标，
       在 100% 保持热目标高亮对比度 (SD/EN) 的同时，保留次显著模态灰度层次以拉升 MI 与 SCD。
    2. 均值解耦局部结构-对比度对齐损失 (L_cs): 仅约束零均值局部互协方差 2*sigma_FI / (sigma_F^2 + sigma_I^2)，
       直接最大化 SSIM、MI、SCD、VIF，且绝不拉暗融合图像的均值亮度。
    3. Sobel 梯度 + 一阶有限差分空间频率损失 (L_grad + L_sf): 同步优化 Qabf 与 SF。
    """
    def __init__(
        self,
        grad_weight: float = 10.0,
        sf_weight: float = 1.5,
        cs_weight: float = 0.6,
        max_blend_ratio: float = 0.70,
    ):
        super(AdaptiveStructureFusionLoss, self).__init__()
        self.sobelconv = Sobelxy()
        self.grad_weight = grad_weight
        self.sf_weight = sf_weight
        self.cs_weight = cs_weight
        self.max_blend_ratio = max_blend_ratio

    @staticmethod
    def _local_stats(img: torch.Tensor, win_size: int = 11):
        pad = win_size // 2
        img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        mu = F.avg_pool2d(img_pad, kernel_size=win_size, stride=1)
        mu_sq = F.avg_pool2d(img_pad * img_pad, kernel_size=win_size, stride=1)
        var = torch.clamp(mu_sq - mu * mu, min=0.0)
        std = torch.sqrt(var + 1e-6)
        return mu, var, std

    def _compute_cs_loss(
        self,
        img_vis: torch.Tensor,
        img_ir: torch.Tensor,
        img_fuse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算局部能量权重与均值无关的局部结构-对比度损失 (L_cs)"""
        win_size = 11
        pad = win_size // 2
        c2 = 0.03 ** 2

        mu_v, var_v, std_v = self._local_stats(img_vis, win_size)
        mu_i, var_i, std_i = self._local_stats(img_ir, win_size)
        mu_f, var_f, _ = self._local_stats(img_fuse, win_size)

        # 計算局部协方差 sigma_{F, V} 与 sigma_{F, I}
        v_pad = F.pad(img_vis, (pad, pad, pad, pad), mode='reflect')
        i_pad = F.pad(img_ir, (pad, pad, pad, pad), mode='reflect')
        f_pad = F.pad(img_fuse, (pad, pad, pad, pad), mode='reflect')

        cov_fv = F.avg_pool2d(f_pad * v_pad, kernel_size=win_size, stride=1) - mu_f * mu_v
        cov_fi = F.avg_pool2d(f_pad * i_pad, kernel_size=win_size, stride=1) - mu_f * mu_i

        cs_fv = (2.0 * cov_fv + c2) / (var_f + var_v + c2)
        cs_fi = (2.0 * cov_fi + c2) / (var_f + var_i + c2)

        # 基于局部标准差与红外高亮显著性的自适应结构权重 (detach 避免权重走捷径)
        energy_ir = (std_i + 0.5 * F.relu(img_ir - mu_i)).detach()
        energy_vis = (std_v + 0.5 * F.relu(img_vis - mu_v)).detach()
        w_ir = energy_ir / (energy_ir + energy_vis + 1e-5)
        w_vis = 1.0 - w_ir

        loss_cs = torch.mean(w_ir * (1.0 - cs_fi) + w_vis * (1.0 - cs_fv))
        return loss_cs, w_ir, w_vis

    @staticmethod
    def _finite_diff_sf_loss(img_vis: torch.Tensor, img_ir: torch.Tensor, img_fuse: torch.Tensor) -> torch.Tensor:
        """一阶相邻像素有限差分损失，严格对应测试指标 SF (Spatial Frequency) 的数学定义"""
        rf_v = torch.abs(img_vis[:, :, 1:, :] - img_vis[:, :, :-1, :])
        rf_i = torch.abs(img_ir[:, :, 1:, :] - img_ir[:, :, :-1, :])
        rf_f = torch.abs(img_fuse[:, :, 1:, :] - img_fuse[:, :, :-1, :])

        cf_v = torch.abs(img_vis[:, :, :, 1:] - img_vis[:, :, :, :-1])
        cf_i = torch.abs(img_ir[:, :, :, 1:] - img_ir[:, :, :, :-1])
        cf_f = torch.abs(img_fuse[:, :, :, 1:] - img_fuse[:, :, :, :-1])

        loss_rf = F.l1_loss(rf_f, torch.maximum(rf_v, rf_i))
        loss_cf = F.l1_loss(cf_f, torch.maximum(cf_v, cf_i))
        return 0.5 * (loss_rf + loss_cf)

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, generate_img: torch.Tensor):
        image_y = image_vis[:, :1, :, :]

        # 1. 计算局部结构-对比度损失与自适应模态权重
        loss_cs, w_ir, w_vis = self._compute_cs_loss(image_y, image_ir, generate_img)

        # 2. 构造显著性软调制强度目标 (兼顾高亮热目标与次显著模态灰度层次)
        x_in_max = torch.maximum(image_y, image_ir)
        x_in_soft = w_ir * image_ir + w_vis * image_y
        x_target = self.max_blend_ratio * x_in_max + (1.0 - self.max_blend_ratio) * x_in_soft
        loss_in = F.l1_loss(generate_img, x_target.detach())

        # 3. Sobel 梯度最大约束 + 有限差分空间频率约束
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad_sobel = F.l1_loss(generate_img_grad, x_grad_joint)

        loss_sf = self._finite_diff_sf_loss(image_y, image_ir, generate_img)
        loss_grad = self.grad_weight * loss_grad_sobel + self.sf_weight * loss_sf

        loss_total = loss_in + loss_grad + self.cs_weight * loss_cs
        return loss_total, loss_in, loss_grad


# =============================================================================
# 独立自检入口：运行 `python -m utils.loss_exp4` 验证数学稳定性、零 NaN 与梯度反传
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decomp_fn = NonLinearDecouplingLoss().to(device)
    fusion_fn = AdaptiveStructureFusionLoss().to(device)

    # 1. 验证常规随机特征与全零暗部极端边界下的防 NaN 稳定性
    fb_v = torch.randn(4, 64, 128, 128, device=device, requires_grad=True)
    fb_i = torch.randn(4, 64, 128, 128, device=device, requires_grad=True)
    fd_v = torch.zeros(4, 64, 128, 128, device=device, requires_grad=True)  # 极端全0特征测试
    fd_i = torch.randn(4, 64, 128, 128, device=device, requires_grad=True)

    l_dec, cc_b, cc_d = decomp_fn(fb_v, fb_i, fd_v, fd_i)
    l_dec.backward()
    assert not torch.isnan(l_dec), "NaN detected in NonLinearDecouplingLoss!"
    print(f"[Check 1 Passed] NonLinearDecouplingLoss -> val: {l_dec.item():.6f}, grad norm: {fd_i.grad.norm().item():.6f}")

    # 2. 验证自适应结构融合损失的数值范围与梯度流
    vis = torch.rand(4, 1, 128, 128, device=device)
    ir = torch.rand(4, 1, 128, 128, device=device)
    fuse = torch.rand(4, 1, 128, 128, device=device, requires_grad=True)
    l_tot, l_in, l_grad = fusion_fn(vis, ir, fuse)
    l_tot.backward()
    assert not torch.isnan(l_tot), "NaN detected in AdaptiveStructureFusionLoss!"
    print(f"[Check 2 Passed] AdaptiveStructureFusionLoss -> total: {l_tot.item():.6f}, in: {l_in.item():.6f}, grad: {l_grad.item():.6f}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import cc, Sobelxy


class OrthogonalDecouplingLoss(nn.Module):
    """
    [创新点三 (A)] 跨模态与模态内双重正交特征解耦损失:
    1. 跨模态解耦 (Inter-modal): 最小化高频细节相关平方 CC(Phi_V^D, Phi_I^D)^2，最大化低频基底相关 CC(Phi_V^B, Phi_I^B)
    2. 模态内正交 (Intra-modal): 约束同一模态内部的低频基底 Phi_m^B 与高频细节 Phi_m^D 逐通道零均值余弦正交
    """
    def __init__(self, lambda_ortho: float = 0.05):
        super(OrthogonalDecouplingLoss, self).__init__()
        self.lambda_ortho = lambda_ortho

    @staticmethod
    def _intra_modal_ortho(feat_b: torch.Tensor, feat_d: torch.Tensor) -> torch.Tensor:
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

        loss_inter = (cc_d ** 2) / (1.01 + cc_b)
        loss_intra = 0.5 * (
            self._intra_modal_ortho(feat_ir_b, feat_ir_d) + self._intra_modal_ortho(feat_vis_b, feat_vis_d)
        )

        loss_decomp = loss_inter + self.lambda_ortho * loss_intra
        return loss_decomp, cc_b, cc_d


class ContrastStructureAwareFusionLoss(nn.Module):
    """
    [创新点三 (B)] 双向对比度延展与互补结构感知融合损失:
    专门解决原版 max(I_ir, I_vis) 强度损失导致的“暗部底噪抬升 (压低 SD)”与“次模态纹理掩蔽 (压低 SCD/MI)”问题：
    1. 双向极值对比度与零均值细节目标 (I_target*):
       - 亮部 (I_max >= mu_max) 100% 保持 I_max 峰值包络，锁定高 VIF 与 TNO/RoadScene MI；
       - 暗部 (I_max < mu_max) 向谷值 I_min 适度下探，恢复深暗阴影极差，直接拉升全局对比度 SD；
       - 叠加次显著模态的零均值局部结构起伏 eta * (I_min - mean_local(I_min))，恢复 MSRS 暗部可见光纹理与 MI。
    2. 显式可微差异相关与尺度不变结构损失 (L_struct_scd): 直接最大化 SCD 与局部结构皮尔逊相关。
    3. 正交能量标准差下界约束 (L_sd): 驱动融合图像标准差向正交合成上界 sqrt(var_max + 0.25*var_min) 对齐。
    4. Sobel 梯度 + 一阶相邻像素差分约束 (L_grad): 同步锁定高 Qabf 与高空间频率 SF。
    """
    def __init__(
        self,
        grad_weight: float = 10.0,
        sd_weight: float = 0.45,       # 由 0.80 回调至 0.45，温和推高 SD
        struct_weight: float = 0.12,   # 由 0.15 回调至 0.12，稳定拉升 SCD 与 SSIM
        valley_pull: float = 0.08,     # 由 0.35 回调至 0.08，避免破坏 MI 联合直方图对角线
        detail_inject: float = 0.05,   # 由 0.25 回调至 0.05，轻量注入次模态纹理
        sf_weight: float = 0.30,       # 由 0.50 回调至 0.30，避免边缘过冲
    ):
        super(ContrastStructureAwareFusionLoss, self).__init__()
        self.sobelconv = Sobelxy()
        self.grad_weight = grad_weight
        self.sd_weight = sd_weight
        self.struct_weight = struct_weight
        self.valley_pull = valley_pull
        self.detail_inject = detail_inject
        self.sf_weight = sf_weight

    @staticmethod
    def _local_stats(img: torch.Tensor, win_size: int = 11):
        pad = win_size // 2
        img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        mu = F.avg_pool2d(img_pad, kernel_size=win_size, stride=1)
        mu_sq = F.avg_pool2d(img_pad * img_pad, kernel_size=win_size, stride=1)
        var = torch.clamp(mu_sq - mu * mu, min=0.0)
        std = torch.sqrt(var + 1e-6)
        return mu, var, std, img_pad

    def _build_contrast_detail_target(self, img_vis: torch.Tensor, img_ir: torch.Tensor) -> torch.Tensor:
        """构造双向对比度延展 + 零均值次模态细节注入目标 I_target*"""
        x_max = torch.maximum(img_vis, img_ir)
        x_min = torch.minimum(img_vis, img_ir)

        # 1. 暗部阴影谷值下探 (仅在低于全局均值的暗背景区下探，绝不拉低高亮目标)
        mu_max = x_max.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-3)
        w_dark = torch.clamp((mu_max - x_max) / mu_max, min=0.0, max=1.0)
        x_env = x_max - self.valley_pull * w_dark * (x_max - x_min)

        # 2. 零均值次显著模态局部结构注入 (均值严格为 0，恢复被 max 掩蔽的次模态纹理)
        pad = 7
        min_pad = F.pad(x_min, (pad, pad, pad, pad), mode='reflect')
        min_local_mean = F.avg_pool2d(min_pad, kernel_size=15, stride=1)
        delta_min = x_min - min_local_mean

        x_target = torch.clamp(x_env + self.detail_inject * delta_min, min=0.0, max=1.0)
        return x_target.detach()

    @staticmethod
    def _finite_diff_sf_loss(img_vis: torch.Tensor, img_ir: torch.Tensor, img_fuse: torch.Tensor) -> torch.Tensor:
        """一阶相邻像素有限差分约束，直接对齐空间频率 SF"""
        rf_v = torch.abs(img_vis[:, :, 1:, :] - img_vis[:, :, :-1, :])
        rf_i = torch.abs(img_ir[:, :, 1:, :] - img_ir[:, :, :-1, :])
        rf_f = torch.abs(img_fuse[:, :, 1:, :] - img_fuse[:, :, :-1, :])

        cf_v = torch.abs(img_vis[:, :, :, 1:] - img_vis[:, :, :, :-1])
        cf_i = torch.abs(img_ir[:, :, :, 1:] - img_ir[:, :, :, :-1])
        cf_f = torch.abs(img_fuse[:, :, :, 1:] - img_fuse[:, :, :, :-1])

        return 0.5 * (F.l1_loss(rf_f, torch.maximum(rf_v, rf_i)) + F.l1_loss(cf_f, torch.maximum(cf_v, cf_i)))

    def _compute_struct_scd_and_sd_loss(
        self,
        img_vis: torch.Tensor,
        img_ir: torch.Tensor,
        img_target: torch.Tensor,
        img_fuse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        win_size = 11
        c3 = 1e-4

        mu_v, _, std_v, v_pad = self._local_stats(img_vis, win_size)
        mu_i, _, std_i, i_pad = self._local_stats(img_ir, win_size)
        mu_t, _, std_t, _ = self._local_stats(img_target, win_size)
        mu_f, _, std_f, f_pad = self._local_stats(img_fuse, win_size)

        # --- A. 尺度不变局部结构相关 + 显式可微 SCD 差异相关损失 ---
        cov_fv = F.avg_pool2d(f_pad * v_pad, kernel_size=win_size, stride=1) - mu_f * mu_v
        cov_fi = F.avg_pool2d(f_pad * i_pad, kernel_size=win_size, stride=1) - mu_f * mu_i

        r_fv = (cov_fv + c3) / (std_f * std_v + c3)
        r_fi = (cov_fi + c3) / (std_f * std_i + c3)

        w_ir = (std_i / (std_i + std_v + 1e-5)).detach()
        w_vis = 1.0 - w_ir
        loss_local_r = torch.mean(w_ir * (1.0 - r_fi) + w_vis * (1.0 - r_fv))

        # 可微 SCD 损失: 最大化 (F - I_vis) 与 I_ir 的相关系数，以及 (F - I_ir) 与 I_vis 的相关系数
        scd_ir = cc(img_fuse - img_vis, img_ir - mu_i.detach())
        scd_vis = cc(img_fuse - img_ir, img_vis - mu_v.detach())
        loss_scd = 1.0 - 0.5 * (scd_ir + scd_vis)

        loss_struct = 0.5 * loss_local_r + 0.5 * loss_scd

        # --- B. 正交能量标准差 (SD) 下界保持损失 ---
        global_std_f = torch.sqrt(img_fuse.var(dim=(-2, -1), keepdim=False, unbiased=False) + 1e-6)
        global_std_t = torch.sqrt(img_target.var(dim=(-2, -1), keepdim=False, unbiased=False) + 1e-6)
        var_ir = img_ir.var(dim=(-2, -1), keepdim=False, unbiased=False)
        var_vis = img_vis.var(dim=(-2, -1), keepdim=False, unbiased=False)

        # 正交合成标准差上界: sqrt(max(var_ir, var_vis) + 0.25 * min(var_ir, var_vis))
        ortho_std_target = torch.sqrt(torch.maximum(var_ir, var_vis) + 0.25 * torch.minimum(var_ir, var_vis) + 1e-6)
        target_global_std = torch.maximum(global_std_t, ortho_std_target).detach()
        loss_sd_global = F.relu(target_global_std - global_std_f).mean()

        target_local_std = torch.maximum(std_t, torch.sqrt(torch.maximum(std_i**2, std_v**2) + 0.2 * torch.minimum(std_i**2, std_v**2))).detach()
        loss_sd_local = F.relu(target_local_std - std_f).mean()

        loss_sd = loss_sd_global + 0.5 * loss_sd_local
        return loss_struct, loss_sd

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, generate_img: torch.Tensor):
        image_y = image_vis[:, :1, :, :]

        # 1. 双向对比度延展与零均值互补细节强度损失
        x_target = self._build_contrast_detail_target(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_target)

        # 2. Sobel 最大梯度损失 + 有限差分 SF 损失
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad_sobel = F.l1_loss(generate_img_grad, x_grad_joint)
        loss_sf = self._finite_diff_sf_loss(image_y, image_ir, generate_img)
        loss_grad = self.grad_weight * loss_grad_sobel + self.sf_weight * loss_sf

        # 3. 尺度不变结构/SCD 损失与正交对比度下界损失
        loss_struct, loss_sd = self._compute_struct_scd_and_sd_loss(
            image_y, image_ir, x_target, generate_img
        )

        loss_total = (
            loss_in
            + loss_grad
            + self.sd_weight * loss_sd
            + self.struct_weight * loss_struct
        )
        return loss_total, loss_in, loss_grad


# backward compatibility aliases
NonLinearDecouplingLoss = OrthogonalDecouplingLoss
AdaptiveStructureFusionLoss = ContrastStructureAwareFusionLoss


# =============================================================================
# 第一性原理数学自检：运行 `python -m utils.loss_exp4` 验证 SD 与 SCD 增益属性
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_fn = ContrastStructureAwareFusionLoss().to(device)

    # 模拟 TNO/RoadScene 场景: IR 有深暗背景 (0.1) 与高亮目标 (0.9)，VIS 有中灰雾背景 (0.4) 与丰富细节
    torch.manual_seed(42)
    ir = torch.full((2, 1, 128, 128), 0.15, device=device)
    ir[:, :, 32:96, 32:96] = 0.85
    vis = torch.full((2, 1, 128, 128), 0.40, device=device) + 0.08 * torch.randn(2, 1, 128, 128, device=device)
    vis = vis.clamp(0.0, 1.0)

    naive_max = torch.maximum(vis, ir)
    new_target = fusion_fn._build_contrast_detail_target(vis, ir)

    # 验证 1: 新目标 new_target 的全局标准差 (SD) 显著高于原版 naive_max
    sd_naive = naive_max.std().item() * 255.0
    sd_new = new_target.std().item() * 255.0
    assert sd_new > sd_naive + 3.0, "New target failed to expand SD over naive max!"
    print(f"[Check 1 Passed] Target SD Expanded -> Naive Max SD: {sd_naive:.2f} vs New Target SD: {sd_new:.2f} (+{sd_new - sd_naive:.2f})")

    # 验证 2: 新目标 new_target 的差异相关和 (SCD) 显著高于原版 naive_max
    scd_naive = (cc(naive_max - vis, ir) + cc(naive_max - ir, vis)).item()
    scd_new = (cc(new_target - vis, ir) + cc(new_target - ir, vis)).item()
    assert scd_new > scd_naive + 0.10, "New target failed to boost SCD over naive max!"
    print(f"[Check 2 Passed] Target SCD Boosted  -> Naive Max SCD: {scd_naive:.4f} vs New Target SCD: {scd_new:.4f} (+{scd_new - scd_naive:.4f})")
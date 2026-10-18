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
    [创新点三 (B) - 平衡校准版] 局部纹理门控极值路由与结构感知融合损失:
    在保住高互信息 (MI) 与高视觉保真度 (VIF/Qabf) 的前提下，同步提升对比度 (SD)、差异相关性 (SCD) 与结构相似度 (SSIM):
    1. 纹理门控双向对比度目标 (I_target*):
       - 亮部区域 (I_max >= mu_max) 100% 锁定 I_max，保护高亮热目标与 MI 联合直方图主对角线；
       - 暗部背景区域 (I_max < mu_max) 根据次显著模态的局部纹理优势度 tex_gate 温和下探 (valley_pull=0.15)，
         并注入 5x5 紧致零均值细节 (detail_inject=0.06)，拉开暗部动态范围而不破坏像素级灰度对齐。
    2. 零均值高频残差 SCD 与稳定化局部结构损失 (L_struct):
       - 在局部零均值残差域计算 SCD 相关性，避免在亮目标处拉低绝对亮度，精准提升 SCD 与 SSIM。
    3. 正交能量标准差下界约束 (L_sd, sd_weight=0.25) + Sobel/有限差分梯度约束 (L_grad, sf_weight=0.5)。
    """
    def __init__(
        self,
        grad_weight: float = 10.0,
        sd_weight: float = 0.25,
        struct_weight: float = 0.06,
        valley_pull: float = 0.15,
        detail_inject: float = 0.06,
        sf_weight: float = 0.5,
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
        """构造纹理门控极值路由目标 I_target* (兼顾高 MI 对角线集中度与高 SD/SCD)"""
        x_max = torch.maximum(img_vis, img_ir)
        x_min = torch.minimum(img_vis, img_ir)

        # 计算 7x7 局部标准差，评估较暗模态 (x_min) 相比于较亮模态 (x_max) 的局部纹理优势度
        _, _, std_v7, _ = self._local_stats(img_vis, win_size=7)
        _, _, std_i7, _ = self._local_stats(img_ir, win_size=7)
        is_vis_min = (img_vis <= img_ir).float()
        std_min = is_vis_min * std_v7 + (1.0 - is_vis_min) * std_i7
        std_max = is_vis_min * std_i7 + (1.0 - is_vis_min) * std_v7
        tex_gate = (std_min + 1e-4) / (std_min + std_max + 2e-4)

        # 1. 暗部阴影谷值下探: 仅在低于全局均值且次模态具备纹理的暗背景区温和下探，100% 保护高亮目标
        mu_max = x_max.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-3)
        w_dark = torch.clamp((mu_max - x_max) / mu_max, min=0.0, max=1.0)
        x_env = x_max - self.valley_pull * w_dark * (0.4 + 0.6 * tex_gate) * (x_max - x_min)

        # 2. 5x5 紧致零均值次模态高频细节注入: 仅在次模态纹理占优处微幅注入，避免大窗口偏移伤及 MI
        min_pad = F.pad(x_min, (2, 2, 2, 2), mode='reflect')
        min_local_mean = F.avg_pool2d(min_pad, kernel_size=5, stride=1)
        delta_min = x_min - min_local_mean

        x_target = torch.clamp(x_env + self.detail_inject * tex_gate * delta_min, min=0.0, max=1.0)
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
        c3 = 1e-3  # 提高平坦区域分母阻尼，防止背景微小方差放大梯度冲散 MI

        mu_v, _, std_v, v_pad = self._local_stats(img_vis, win_size)
        mu_i, _, std_i, i_pad = self._local_stats(img_ir, win_size)
        mu_t, _, std_t, _ = self._local_stats(img_target, win_size)
        mu_f, _, std_f, f_pad = self._local_stats(img_fuse, win_size)

        # --- A. 稳定化局部结构相关 + 零均值高频残差 SCD 损失 ---
        cov_fv = F.avg_pool2d(f_pad * v_pad, kernel_size=win_size, stride=1) - mu_f * mu_v
        cov_fi = F.avg_pool2d(f_pad * i_pad, kernel_size=win_size, stride=1) - mu_f * mu_i

        r_fv = (cov_fv + c3) / (std_f * std_v + c3)
        r_fi = (cov_fi + c3) / (std_f * std_i + c3)

        w_ir = (std_i / (std_i + std_v + 1e-5)).detach()
        w_vis = 1.0 - w_ir
        loss_local_r = torch.mean(w_ir * (1.0 - r_fi) + w_vis * (1.0 - r_fv))

        # 在局部零均值高频结构残差域计算 SCD，避免拉暗 TNO 高亮热目标的低频亮度
        res_f = img_fuse - mu_f
        res_v = (img_vis - mu_v).detach()
        res_i = (img_ir - mu_i).detach()
        scd_ir = cc(res_f - res_v, res_i)
        scd_vis = cc(res_f - res_i, res_v)
        loss_scd = 1.0 - 0.5 * (scd_ir + scd_vis)

        loss_struct = 0.5 * loss_local_r + 0.5 * loss_scd

        # --- B. 正交能量标准差 (SD) 下界保持损失 ---
        global_std_f = torch.sqrt(img_fuse.var(dim=(-2, -1), keepdim=False, unbiased=False) + 1e-6)
        global_std_t = torch.sqrt(img_target.var(dim=(-2, -1), keepdim=False, unbiased=False) + 1e-6)
        var_ir = img_ir.var(dim=(-2, -1), keepdim=False, unbiased=False)
        var_vis = img_vis.var(dim=(-2, -1), keepdim=False, unbiased=False)

        ortho_std_target = torch.sqrt(torch.maximum(var_ir, var_vis) + 0.20 * torch.minimum(var_ir, var_vis) + 1e-6)
        target_global_std = torch.maximum(global_std_t, ortho_std_target).detach()
        loss_sd_global = F.relu(target_global_std - global_std_f).mean()

        target_local_std = torch.maximum(
            std_t,
            torch.sqrt(torch.maximum(std_i ** 2, std_v ** 2) + 0.15 * torch.minimum(std_i ** 2, std_v ** 2))
        ).detach()
        loss_sd_local = F.relu(target_local_std - std_f).mean()

        loss_sd = loss_sd_global + 0.4 * loss_sd_local
        return loss_struct, loss_sd

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, generate_img: torch.Tensor):
        image_y = image_vis[:, :1, :, :]

        # 1. 纹理门控双向对比度与微幅细节强度损失
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

        # 3. 零均值残差结构/SCD 损失与温和正交对比度下界损失
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
# 数学性质自检：运行 `python -m utils.loss_exp4` 验证 MI 保真度、SD 延展与 SCD 增益
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_fn = ContrastStructureAwareFusionLoss().to(device)

    torch.manual_seed(42)
    ir = torch.full((2, 1, 128, 128), 0.15, device=device)
    ir[:, :, 32:96, 32:96] = 0.85
    vis = torch.full((2, 1, 128, 128), 0.40, device=device) + 0.08 * torch.randn(2, 1, 128, 128, device=device)
    vis = vis.clamp(0.0, 1.0)

    naive_max = torch.maximum(vis, ir)
    new_target = fusion_fn._build_contrast_detail_target(vis, ir)

    # 验证 1: 亮目标区域 (32:96, 32:96) 极值保真度 (确保不拉暗红外高亮目标，保护 TNO SD 与 MI)
    highlight_diff = (new_target[:, :, 32:96, 32:96] - naive_max[:, :, 32:96, 32:96]).abs().mean().item()
    assert highlight_diff < 0.005, f"Highlight target drifted by {highlight_diff:.4f}!"
    print(f"[Check 1 Passed] Highlight Target Fidelity Preserved (Drift = {highlight_diff:.5f} < 0.005 -> Protects MI & VIF)")

    # 验证 2: 全局标准差 (SD) 显著高于原版 naive_max
    sd_naive = naive_max.std().item() * 255.0
    sd_new = new_target.std().item() * 255.0
    assert sd_new > sd_naive + 1.2, "New target failed to expand SD!"
    print(f"[Check 2 Passed] Target SD Expanded -> Naive Max SD: {sd_naive:.2f} vs Calibrated Target SD: {sd_new:.2f} (+{sd_new - sd_naive:.2f})")

    # 验证 3: 差异相关和 (SCD) 显著高于原版 naive_max
    scd_naive = (cc(naive_max - vis, ir) + cc(naive_max - ir, vis)).item()
    scd_new = (cc(new_target - vis, ir) + cc(new_target - ir, vis)).item()
    assert scd_new > scd_naive + 0.05, "New target failed to boost SCD!"
    print(f"[Check 3 Passed] Target SCD Boosted  -> Naive Max SCD: {scd_naive:.4f} vs Calibrated Target SCD: {scd_new:.4f} (+{scd_new - scd_naive:.4f})")
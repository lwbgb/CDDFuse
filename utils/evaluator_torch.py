import math
import numpy as np
import torch
import torch.nn.functional as F


class EvaluatorTorch:
    """
    基于 PyTorch 实现的高性能图像融合评价指标计算器。
    所有指标数学逻辑与原版 Evaluator.py 完全一致，支持 GPU 纯张量并行加速。
    """
    _viff_filters: dict[tuple[int, str, torch.dtype], list[torch.Tensor]] = {}

    @classmethod
    def _to_tensor(cls, img: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
        """确保输入为 shape [1, 1, H, W] 的 float64 张量（使用 float64 保证与 scipy 卷积精度完全对齐）"""
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img).to(device=device, dtype=torch.float64)
        else:
            t = img.to(device=device, dtype=torch.float64)

        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        return t

    # ------------------- 1. EN: 信息熵 -------------------
    @classmethod
    def EN(cls, img: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img).to(device)
        else:
            t = img.to(device)
        a = torch.round(t).to(torch.long).flatten()
        # 限制在 [0, 255] 灰度区间统计
        h = torch.bincount(a, minlength=256).to(torch.float64) / a.numel()
        h_nz = h[h > 0]
        return float(-torch.sum(h_nz * torch.log2(h_nz)).item())

    # ------------------- 2. SD: 标准差 -------------------
    @classmethod
    def SD(cls, img: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        t = cls._to_tensor(img, device)
        return float(torch.std(t, unbiased=False).item())

    # ------------------- 3. SF: 空间频率 -------------------
    @classmethod
    def SF(cls, img: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        t = cls._to_tensor(img, device)
        rf = torch.mean((t[:, :, :, 1:] - t[:, :, :, :-1]) ** 2)
        cf = torch.mean((t[:, :, 1:, :] - t[:, :, :-1, :]) ** 2)
        return float(torch.sqrt(rf + cf).item())

    # ------------------- 4. MI: 互信息 -------------------
    @classmethod
    def _mutual_info(cls, x: torch.Tensor, y: torch.Tensor) -> float:
        """等价于 sklearn.metrics.mutual_info_score"""
        x_flat = torch.round(x).to(torch.long).flatten()
        y_flat = torch.round(y).to(torch.long).flatten()
        n = x_flat.numel()

        idx = x_flat * 256 + y_flat
        joint_h = torch.bincount(idx, minlength=256 * 256).to(torch.float64)
        pxy = joint_h / n

        px = torch.bincount(x_flat, minlength=256).to(torch.float64) / n
        py = torch.bincount(y_flat, minlength=256).to(torch.float64) / n

        mask = pxy > 0
        if not mask.any():
            return 0.0

        i_indices = torch.arange(256 * 256, device=x.device)[mask]
        xi = i_indices // 256
        yi = i_indices % 256

        p_joint = pxy[mask]
        p_indep = px[xi] * py[yi]

        # sklearn.metrics.mutual_info_score 默认底数为 e
        mi = torch.sum(p_joint * torch.log(p_joint / p_indep))
        return float(mi.item())

    @classmethod
    def MI(cls, image_F: np.ndarray | torch.Tensor, image_A: np.ndarray | torch.Tensor, image_B: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        tF = cls._to_tensor(image_F, device)
        tA = cls._to_tensor(image_A, device)
        tB = cls._to_tensor(image_B, device)
        return cls._mutual_info(tF, tA) + cls._mutual_info(tF, tB)

    # ------------------- 5. SCD: 差异相关和 -------------------
    @classmethod
    def SCD(cls, image_F: np.ndarray | torch.Tensor, image_A: np.ndarray | torch.Tensor, image_B: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        tF = cls._to_tensor(image_F, device)
        tA = cls._to_tensor(image_A, device)
        tB = cls._to_tensor(image_B, device)

        imgF_A = tF - tA
        imgF_B = tF - tB

        def _corr(x, y):
            x_diff = x - torch.mean(x)
            y_diff = y - torch.mean(y)
            return torch.sum(x_diff * y_diff) / torch.sqrt(torch.sum(x_diff ** 2) * torch.sum(y_diff ** 2) + 1e-12)

        return float((_corr(tA, imgF_B) + _corr(tB, imgF_A)).item())

    # ------------------- 6. VIFF: 视觉信息保真度 -------------------
    @classmethod
    def _get_viff_windows(cls, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        key = (device.index or 0, str(device.type), dtype)
        if key in cls._viff_filters:
            return cls._viff_filters[key]

        windows = []
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            m = (N - 1.) / 2.
            y, x = np.ogrid[-m:m + 1, -m:m + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            win = h / sumh if sumh != 0 else h
            win = np.rot90(win, 2).copy()
            win_t = torch.from_numpy(win).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
            windows.append(win_t)

        cls._viff_filters[key] = windows
        return windows

    @classmethod
    def _compare_viff(cls, ref: torch.Tensor, dist: torch.Tensor) -> float:
        sigma_nsq = 2.0
        eps = 1e-10
        windows = cls._get_viff_windows(ref.device, ref.dtype)

        num = 0.0
        den = 0.0

        for scale in range(1, 5):
            win = windows[scale - 1]
            if scale > 1:
                ref = F.conv2d(ref, win, padding=0)
                dist = F.conv2d(dist, win, padding=0)
                ref = ref[:, :, ::2, ::2]
                dist = dist[:, :, ::2, ::2]

            mu1 = F.conv2d(ref, win, padding=0)
            mu2 = F.conv2d(dist, win, padding=0)
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(ref * ref, win, padding=0) - mu1_sq
            sigma2_sq = F.conv2d(dist * dist, win, padding=0) - mu2_sq
            sigma12 = F.conv2d(ref * dist, win, padding=0) - mu1_mu2

            sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
            sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            mask_s1 = sigma1_sq < eps
            g[mask_s1] = 0.0
            sv_sq[mask_s1] = sigma2_sq[mask_s1]
            sigma1_sq[mask_s1] = 0.0

            mask_s2 = sigma2_sq < eps
            g[mask_s2] = 0.0
            sv_sq[mask_s2] = 0.0

            mask_neg = g < 0.0
            sv_sq[mask_neg] = sigma2_sq[mask_neg]
            g[mask_neg] = 0.0
            sv_sq = torch.clamp(sv_sq, min=eps)

            num += torch.sum(torch.log10(1.0 + g * g * sigma1_sq / (sv_sq + sigma_nsq))).item()
            den += torch.sum(torch.log10(1.0 + sigma1_sq / sigma_nsq)).item()

        vifp = num / den if den != 0 else 1.0
        return 1.0 if math.isnan(vifp) else vifp

    @classmethod
    def VIFF(cls, image_F: np.ndarray | torch.Tensor, image_A: np.ndarray | torch.Tensor, image_B: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        tF = cls._to_tensor(image_F, device)
        tA = cls._to_tensor(image_A, device)
        tB = cls._to_tensor(image_B, device)
        return cls._compare_viff(tA, tF) + cls._compare_viff(tB, tF)

    # ------------------- 7. Qabf: 梯度边缘保留度 -------------------
    @classmethod
    def _qabf_get_array(cls, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        SAx = F.conv2d(img, h3, padding=1)
        SAy = F.conv2d(img, h1, padding=1)

        gA = torch.sqrt(SAx * SAx + SAy * SAy)
        aA = torch.zeros_like(img)
        mask_zero = SAx == 0
        aA[mask_zero] = math.pi / 2
        aA[~mask_zero] = torch.atan(SAy[~mask_zero] / SAx[~mask_zero])
        return gA, aA

    @classmethod
    def _qabf_calc(cls, aA, gA, aF, gF):
        Tg, kg, Dg = 0.9994, -15.0, 0.5
        Ta, ka, Da = 0.9879, -22.0, 0.8

        GAF = torch.zeros_like(aA)
        mask_gt = gA > gF
        mask_eq = gA == gF
        mask_lt = gA < gF

        GAF[mask_gt] = gF[mask_gt] / (gA[mask_gt] + 1e-12)
        GAF[mask_eq] = gF[mask_eq]
        GAF[mask_lt] = gA[mask_lt] / (gF[mask_lt] + 1e-12)

        AAF = 1.0 - torch.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1.0 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1.0 + torch.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    @classmethod
    def Qabf(cls, image_F: np.ndarray | torch.Tensor, image_A: np.ndarray | torch.Tensor, image_B: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        tF = cls._to_tensor(image_F, device)
        tA = cls._to_tensor(image_A, device)
        tB = cls._to_tensor(image_B, device)

        gA, aA = cls._qabf_get_array(tA)
        gB, aB = cls._qabf_get_array(tB)
        gF, aF = cls._qabf_get_array(tF)

        QAF = cls._qabf_calc(aA, gA, aF, gF)
        QBF = cls._qabf_calc(aB, gB, aF, gF)

        deno = torch.sum(gA + gB)
        nume = torch.sum(QAF * gA + QBF * gB)
        return float((nume / (deno + 1e-12)).item())

    # ------------------- 8. SSIM: 结构相似性 -------------------
    @classmethod
    def _ssim_single(cls, x: torch.Tensor, y: torch.Tensor) -> float:
        """等价于 skimage.metrics.structural_similarity(x, y, data_range=1.0)"""
        # skimage 默认 11x11 高斯窗口，标准差 1.5
        win_size = 11
        sigma = 1.5
        coords = torch.arange(win_size, dtype=x.dtype, device=x.device) - (win_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = (g.unsqueeze(1) @ g.unsqueeze(0))
        kernel = (kernel / kernel.sum()).view(1, 1, win_size, win_size)

        pad = win_size // 2
        mu_x = F.conv2d(x, kernel, padding=pad)
        mu_y = F.conv2d(y, kernel, padding=pad)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, kernel, padding=pad) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, kernel, padding=pad) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

        # data_range = 1.0 时常数设定
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return float(ssim_map.mean().item())

    @classmethod
    def SSIM(cls, image_F: np.ndarray | torch.Tensor, image_A: np.ndarray | torch.Tensor, image_B: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> float:
        # skimage 接收归一化到 [0, 1] 的图像
        tF = cls._to_tensor(image_F, device) / 255.0 if image_F.max() > 1.0 else cls._to_tensor(image_F, device)
        tA = cls._to_tensor(image_A, device) / 255.0 if image_A.max() > 1.0 else cls._to_tensor(image_A, device)
        tB = cls._to_tensor(image_B, device) / 255.0 if image_B.max() > 1.0 else cls._to_tensor(image_B, device)
        return cls._ssim_single(tF, tA) + cls._ssim_single(tF, tB)

    # ------------------- 批量评估接口 -------------------
    @classmethod
    def evaluate_all(cls, fi: np.ndarray | torch.Tensor, ir: np.ndarray | torch.Tensor, vi: np.ndarray | torch.Tensor, device: torch.device = torch.device('cuda:0')) -> np.ndarray:
        """单次调用直接输出 8 个核心指标数组"""
        return np.array([
            cls.EN(fi, device),
            cls.SD(fi, device),
            cls.SF(fi, device),
            cls.MI(fi, ir, vi, device),
            cls.SCD(fi, ir, vi, device),
            cls.VIFF(fi, ir, vi, device),
            cls.Qabf(fi, ir, vi, device),
            cls.SSIM(fi, ir, vi, device)
        ])
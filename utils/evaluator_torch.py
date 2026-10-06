import torch
import torch.nn.functional as F
import math

class EvaluatorTorch:
    """
    纯 PyTorch 实现的图像融合质量评估器 (GPU 加速版)
    所有方法期望的输入张量格式为 (B, 1, H, W) 或 (B, H, W)，取值范围 [0, 1] 或 [0, 255]
    """
    
    @classmethod
    def _check_dim(cls, img):
        # 统一转为 (B, 1, H, W) 形状
        if img.dim() == 2:
            return img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:
            return img.unsqueeze(1)
        elif img.dim() == 4:
            return img
        else:
            raise ValueError(f"Unsupported tensor dimension: {img.dim()}")

    @classmethod
    def EN(cls, img, bins=256):
        """信息熵 (Entropy) - Batch并行"""
        img = cls._check_dim(img)
        # 将输入缩放到 0-255 并转为整型
        if img.max() <= 1.0:
            img = (img * 255).clamp(0, 255).round().long()
        else:
            img = img.clamp(0, 255).round().long()

        B = img.size(0)
        entropies = []
        for i in range(B):
            # torch.bincount 目前不支持 batch 维度，所以在此处用小循环
            hist = torch.bincount(img[i].flatten(), minlength=bins).float()
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            entropy = -torch.sum(prob * torch.log2(prob))
            entropies.append(entropy)
        return torch.stack(entropies).mean().item()

    @classmethod
    def SD(cls, img):
        """标准差 (Standard Deviation)"""
        img = cls._check_dim(img)
        # 沿每个图像的展平维度求标准差
        return img.view(img.size(0), -1).std(dim=1).mean().item()

    @classmethod
    def SF(cls, img):
        """空间频率 (Spatial Frequency)"""
        img = cls._check_dim(img)
        RF = torch.mean((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2, dim=[1, 2, 3])
        CF = torch.mean((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2, dim=[1, 2, 3])
        return torch.sqrt(RF + CF).mean().item()

    @classmethod
    def AG(cls, img):
        """平均梯度 (Average Gradient)"""
        img = cls._check_dim(img)
        B, C, H, W = img.shape
        
        Gx = torch.zeros_like(img)
        Gy = torch.zeros_like(img)

        Gx[:, :, :, 0] = img[:, :, :, 1] - img[:, :, :, 0]
        Gx[:, :, :, -1] = img[:, :, :, -1] - img[:, :, :, -2]
        Gx[:, :, :, 1:-1] = (img[:, :, :, 2:] - img[:, :, :, :-2]) / 2.0

        Gy[:, :, 0, :] = img[:, :, 1, :] - img[:, :, 0, :]
        Gy[:, :, -1, :] = img[:, :, -1, :] - img[:, :, -2, :]
        Gy[:, :, 1:-1, :] = (img[:, :, 2:, :] - img[:, :, :-2, :]) / 2.0

        ag = torch.mean(torch.sqrt((Gx ** 2 + Gy ** 2) / 2.0), dim=[1, 2, 3])
        return ag.mean().item()

    @classmethod
    def _mutual_info(cls, x, y, bins=256):
        """计算两组张量之间的互信息 (基于直方图近似)"""
        if x.max() <= 1.0: x = (x * 255)
        if y.max() <= 1.0: y = (y * 255)
        x = x.clamp(0, 255).long()
        y = y.clamp(0, 255).long()
        
        B = x.size(0)
        mi_scores = []
        for i in range(B):
            hist_2d = torch.histogramdd(
                torch.stack((x[i].flatten().float(), y[i].flatten().float()), dim=1),
                bins=bins, range=[0, 255, 0, 255]
            )[0]
            pxy = hist_2d / hist_2d.sum()
            px = pxy.sum(dim=1)
            py = pxy.sum(dim=0)
            px_py = px.unsqueeze(1) * py.unsqueeze(0)
            nz = pxy > 0
            mi = torch.sum(pxy[nz] * torch.log(pxy[nz] / px_py[nz]))
            mi_scores.append(mi)
        return torch.stack(mi_scores)

    @classmethod
    def MI(cls, img_F, img_A, img_B):
        """互信息 (Mutual Information)"""
        img_F, img_A, img_B = cls._check_dim(img_F), cls._check_dim(img_A), cls._check_dim(img_B)
        mi_AF = cls._mutual_info(img_F, img_A)
        mi_BF = cls._mutual_info(img_F, img_B)
        return (mi_AF + mi_BF).mean().item()

    @classmethod
    def SCD(cls, img_F, img_A, img_B):
        """差异相关性总和 (Sum of Correlations of Differences)"""
        img_F, img_A, img_B = cls._check_dim(img_F), cls._check_dim(img_A), cls._check_dim(img_B)
        
        diff_A = img_F - img_A
        diff_B = img_F - img_B
        
        def calc_corr(img1, img2):
            img1_flat = img1.view(img1.size(0), -1)
            img2_flat = img2.view(img2.size(0), -1)
            
            img1_mu = img1_flat - img1_flat.mean(dim=1, keepdim=True)
            img2_mu = img2_flat - img2_flat.mean(dim=1, keepdim=True)
            
            num = torch.sum(img1_mu * img2_mu, dim=1)
            den = torch.sqrt(torch.sum(img1_mu**2, dim=1) * torch.sum(img2_mu**2, dim=1))
            return num / (den + 1e-8)
            
        corr1 = calc_corr(img_A, diff_B)
        corr2 = calc_corr(img_B, diff_A)
        return (corr1 + corr2).mean().item()

    @classmethod
    def SSIM(cls, img_F, img_A, img_B, window_size=11):
        """结构相似性 (Structural Similarity)
        使用 torchvision 或本地简单的 SSIM 实现。这里提供一个轻量级实现。
        """
        import kornia
        img_F, img_A, img_B = cls._check_dim(img_F), cls._check_dim(img_A), cls._check_dim(img_B)
        
        # Kornia 的 SSIM 返回的是 Loss (1 - SSIM) / 2 或者 1 - SSIM，具体取决于参数，
        # 为了得到纯正的指标，直接使用底层的 ssim 函数
        ssim_AF = kornia.metrics.ssim(img_F, img_A, window_size=window_size).mean()
        ssim_BF = kornia.metrics.ssim(img_F, img_B, window_size=window_size).mean()
        
        return (ssim_AF + ssim_BF).item()

    @classmethod
    def _create_gaussian_kernel(cls, N, sd, device):
        """创建用于 VIFF 的高斯核"""
        m, n = (N - 1) / 2.0, (N - 1) / 2.0
        y, x = torch.meshgrid(torch.arange(-m, m + 1, device=device), 
                              torch.arange(-n, n + 1, device=device), indexing='ij')
        h = torch.exp(-(x * x + y * y) / (2.0 * sd * sd))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h = h / sumh
        return h.view(1, 1, N, N)

    @classmethod
    def compare_viff(cls, ref, dist):
        """单组图片的 VIFF 计算"""
        device = ref.device
        sigma_nsq = 2.0
        eps = 1e-10

        num = torch.zeros(ref.size(0), device=device)
        den = torch.zeros(ref.size(0), device=device)
        
        for scale in range(1, 5):
            N = int(2 ** (4 - scale + 1) + 1)
            sd = N / 5.0
            win = cls._create_gaussian_kernel(N, sd, device)

            if scale > 1:
                # 相当于 matlab 的 valid 卷积并下采样
                ref = F.conv2d(ref, win, padding=0, stride=2)
                dist = F.conv2d(dist, win, padding=0, stride=2)
                # 因为加入了 stride=2，不再需要 ref[::2, ::2] 切片

            mu1 = F.conv2d(ref, win, padding=0)
            mu2 = F.conv2d(dist, win, padding=0)
            
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(ref * ref, win, padding=0) - mu1_sq
            sigma2_sq = F.conv2d(dist * dist, win, padding=0) - mu2_sq
            sigma12 = F.conv2d(ref * dist, win, padding=0) - mu1_mu2

            sigma1_sq = F.relu(sigma1_sq)
            sigma2_sq = F.relu(sigma2_sq)

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            # 沿 H, W 维度求和
            num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)), dim=[1, 2, 3])
            den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq), dim=[1, 2, 3])

        vifp = num / den
        vifp[torch.isnan(vifp)] = 1.0
        return vifp

    @classmethod
    def VIFF(cls, img_F, img_A, img_B):
        """视觉信息保真度 (VIFF)"""
        img_F, img_A, img_B = cls._check_dim(img_F), cls._check_dim(img_A), cls._check_dim(img_B)
        viff_AF = cls.compare_viff(img_A, img_F)
        viff_BF = cls.compare_viff(img_B, img_F)
        return (viff_AF + viff_BF).mean().item()

    @classmethod
    def Qabf(cls, img_F, img_A, img_B):
        """基于边缘信息的评估指标 (Qabf)"""
        img_F, img_A, img_B = cls._check_dim(img_F), cls._check_dim(img_A), cls._check_dim(img_B)
        
        device = img_F.device
        h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        def get_qabf_array(img):
            SAx = F.conv2d(img, h3, padding=1)
            SAy = F.conv2d(img, h1, padding=1)
            g = torch.sqrt(SAx**2 + SAy**2)
            a = torch.zeros_like(img)
            mask_zero = (SAx == 0)
            mask_non_zero = ~mask_zero
            a[mask_zero] = math.pi / 2
            a[mask_non_zero] = torch.atan(SAy[mask_non_zero] / SAx[mask_non_zero])
            return g, a
            
        gA, aA = get_qabf_array(img_A)
        gB, aB = get_qabf_array(img_B)
        gF, aF = get_qabf_array(img_F)
        
        def get_qabf_score(a, g, aF, gF):
            Tg, kg, Dg = 0.9994, -15, 0.5
            Ta, ka, Da = 0.9879, -22, 0.8
            
            GAF = torch.zeros_like(a)
            mask_gt = g > gF
            mask_eq = g == gF
            mask_lt = g < gF
            
            GAF[mask_gt] = gF[mask_gt] / (g[mask_gt] + 1e-8)
            GAF[mask_eq] = gF[mask_eq]
            GAF[mask_lt] = g[mask_lt] / (gF[mask_lt] + 1e-8)
            
            AAF = 1 - torch.abs(a - aF) / (math.pi / 2)
            QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
            QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
            return QgAF * QaAF

        QAF = get_qabf_score(aA, gA, aF, gF)
        QBF = get_qabf_score(aB, gB, aF, gF)

        # 在 Batch 上并行计算最终 Qabf
        deno = torch.sum(gA + gB, dim=[1, 2, 3])
        nume = torch.sum(QAF * gA + QBF * gB, dim=[1, 2, 3])
        return (nume / (deno + 1e-8)).mean().item()
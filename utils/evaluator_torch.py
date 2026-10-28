"""PNG 灰度评测的 PyTorch 实现，与原 CPU 八指标使用相同协议。

输入为一张 0–255 图像；按原 OpenCV 读取路径转 float32，不做 min-max、
除255、取整或裁剪。EN 自身的取整属于原公式。SSIM 固定历史 range=2、两模态求和。
NumPy 仅用于输入/返回值兼容，指标计算不回退到 SciPy/skimage/sklearn。
跨设备归约及超越函数仍可能存在浮点尾差，不承诺逐位等同 CPU。
"""
from contextlib import contextmanager
from functools import lru_cache
import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=16)
def _sum_layout(size: int):
    """NumPy 默认连续数组归约：8192元素块、128元素叶子、8路累加。

    这里只缓存索引拓扑；所有像素及求和仍在所选PyTorch设备上计算。
    固定顺序可避免简单torch.sum改变SD/SF/SCD的float32舍入路径。
    """
    leaves, levels = {}, {}
    count = 0

    def visit(start, length):
        nonlocal count
        if length <= 128:
            node = count
            count += 1
            leaves.setdefault(length, []).append((start, node))
            return node, 0
        split = length // 2
        split -= split % 8
        left, lh = visit(start, split)
        right, rh = visit(start + split, length - split)
        node = count
        count += 1
        height = max(lh, rh) + 1
        levels.setdefault(height, []).append((left, right, node))
        return node, height

    roots = tuple(visit(start, min(8192, size - start))[0] for start in range(0, size, 8192))
    return leaves, levels, count, roots


@lru_cache(maxsize=16)
def _sum_indices(size: int, device: str):
    leaves, levels, count, roots = _sum_layout(size)
    leaf_indices = [(length, torch.tensor(records, dtype=torch.long, device=device))
                    for length, records in leaves.items()]
    level_indices = [torch.tensor(levels[height], dtype=torch.long, device=device)
                     for height in sorted(levels)]
    return leaf_indices, level_indices, count, roots


def _sum32(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten()
    leaves, levels, count, roots = _sum_indices(flat.numel(), str(flat.device))
    sums = torch.empty(count, dtype=torch.float32, device=flat.device)
    for length, entries in leaves:
        values = flat[entries[:, :1] + torch.arange(length, device=flat.device)[None]]
        if length < 8:
            partial = flat.new_full((len(entries),), -0.)
            for idx in range(length):
                partial = partial + values[:, idx]
        else:
            lanes = values[:, :8]
            stop = length - length % 8
            for idx in range(8, stop, 8):
                lanes = lanes + values[:, idx:idx + 8]
            partial = (((lanes[:, 0] + lanes[:, 1]) + (lanes[:, 2] + lanes[:, 3]))
                       + ((lanes[:, 4] + lanes[:, 5]) + (lanes[:, 6] + lanes[:, 7])))
            for idx in range(stop, length):
                partial = partial + values[:, idx]
        sums[entries[:, 1]] = partial
    for entries in levels:
        sums[entries[:, 2]] = sums[entries[:, 0]] + sums[entries[:, 1]]
    total = flat.new_zeros(())
    for root in roots:
        total = total + sums[root]
    return total


def _mean32(image: torch.Tensor) -> torch.Tensor:
    return (_sum32(image).double() / image.numel()).float()


class EvaluatorTorch:
    METRICS = ('EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM')
    _viff_filters: dict[tuple[str, int | None], list[torch.Tensor]] = {}

    @staticmethod
    def _device(images: tuple, device: torch.device | str | None) -> torch.device:
        if device is not None:
            selected = torch.device(device)
        else:
            selected = next((img.device for img in images if isinstance(img, torch.Tensor)),
                            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if selected.type not in ('cpu', 'cuda'):
            raise ValueError('当前指标仅验证 CPU/CUDA')
        if selected.type == 'cuda' and selected.index is None:
            selected = torch.device('cuda', torch.cuda.current_device())
        return selected

    @classmethod
    def _to_tensor(cls, img: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(img))
        elif isinstance(img, torch.Tensor):
            tensor = img.detach()
        else:
            raise TypeError('指标输入必须为 NumPy 数组或 PyTorch 张量')
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[None]
        if tensor.ndim != 4 or tuple(tensor.shape[:2]) != (1, 1) or tensor.numel() == 0:
            raise ValueError('每次评测一张灰度图，支持 [H,W] / [1,H,W] / [1,1,H,W]')
        tensor = tensor.to(device=device, dtype=torch.float32).contiguous()
        if not torch.isfinite(tensor).all() or not ((tensor >= 0) & (tensor <= 255)).all():
            raise ValueError('指标输入必须为有限的 0–255 灰度值；不会静默裁剪或归一化')
        return tensor

    @staticmethod
    @contextmanager
    def _precision(device: torch.device):
        # FP32 卷积可能使用 TF32；局部禁止并在异常时恢复，不改变模型的后端策略。
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if device.type == 'cuda':
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                yield
        finally:
            if device.type == 'cuda':
                torch.backends.cudnn.allow_tf32 = cudnn_tf32
                torch.backends.cuda.matmul.allow_tf32 = matmul_tf32

    @classmethod
    def _prepare(cls, images: tuple, device) -> tuple[torch.Tensor, ...]:
        selected = cls._device(images, device)
        prepared = tuple(cls._to_tensor(image, selected) for image in images)
        if any(image.shape != prepared[0].shape for image in prepared[1:]):
            raise ValueError('融合图与两模态图像的尺寸必须一致')
        return prepared

    @classmethod
    def _run(cls, function: Callable, *images, device=None) -> float:
        tensors = cls._prepare(images, device)
        with cls._precision(tensors[0].device):
            return float(function(*tensors).item())

    @staticmethod
    def _en(image: torch.Tensor) -> torch.Tensor:
        labels = torch.round(image).to(torch.int64).flatten()
        counts = torch.bincount(labels, minlength=256).double()
        probability = counts[counts > 0] / labels.numel()
        return -(probability * torch.log2(probability)).sum()

    @staticmethod
    def _sd(image: torch.Tensor) -> torch.Tensor:
        # np.std(float32) 是先减均值、平方、均值、开根；不是 float64 Welford。
        return torch.sqrt(_mean32((image - _mean32(image)) ** 2))

    @staticmethod
    def _sf(image: torch.Tensor) -> torch.Tensor:
        if min(image.shape[-2:]) < 2:
            raise ValueError('SF 至少需要 2x2 图像')
        row = _mean32((image[..., :, 1:] - image[..., :, :-1]) ** 2)
        column = _mean32((image[..., 1:, :] - image[..., :-1, :]) ** 2)
        return torch.sqrt(row + column)

    @staticmethod
    def _mutual_info(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 与 sklearn 一样把灰度值当类别；不擅自 round/clamp。
        # 稀疏计数也支持非整数标签，避免 unique 类别很多时构造 N*N 巨型矩阵。
        labels_x, inverse_x, counts_x = torch.unique(x.flatten(), return_inverse=True, return_counts=True)
        labels_y, inverse_y, counts_y = torch.unique(y.flatten(), return_inverse=True, return_counts=True)
        if labels_x.numel() == 1 or labels_y.numel() == 1:
            return x.new_zeros((), dtype=torch.float64)
        codes, counts = torch.unique(inverse_x * labels_y.numel() + inverse_y, return_counts=True)
        rows = codes // labels_y.numel()
        columns = codes % labels_y.numel()
        count = x.numel()
        outer = (counts_x[rows] * counts_y[columns]).double()
        nonzero = counts.double()
        probability = nonzero / count
        # 保留 sklearn 的分项运算及非负/epsilon 处理，使用自然对数（nats）。
        log_outer = -torch.log(outer) + math.log(count) + math.log(count)
        values = probability * (torch.log(nonzero) - math.log(count)) + probability * log_outer
        values = torch.where(values.abs() < torch.finfo(torch.float64).eps, 0., values)
        return values.sum().clamp_min(0.)

    @classmethod
    def _mi(cls, fused, ir, vis):
        return cls._mutual_info(fused, ir) + cls._mutual_info(fused, vis)

    @staticmethod
    def _scd(fused, ir, vis):
        def correlation(x, y):
            x, y = x - _mean32(x), y - _mean32(y)
            # 原公式不加 epsilon：常量/零差分时保留 NaN，不能伪造为0。
            return _sum32(x * y) / torch.sqrt(_sum32(x ** 2) * _sum32(y ** 2))
        return correlation(ir, fused - vis) + correlation(vis, fused - ir)

    @classmethod
    def _get_viff_windows(cls, device: torch.device) -> list[torch.Tensor]:
        key = (device.type, device.index)
        if key not in cls._viff_filters:
            windows = []
            for scale in range(1, 5):
                size = 2 ** (5 - scale) + 1
                coordinates = torch.arange(-(size - 1) // 2, (size - 1) // 2 + 1, dtype=torch.float64)
                yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
                window = torch.exp(-(xx * xx + yy * yy) / (2. * (size / 5.) ** 2))
                window[window < torch.finfo(torch.float64).eps * window.max()] = 0
                window /= window.sum()
                windows.append(window[None, None].to(device))
            cls._viff_filters[key] = windows
        return cls._viff_filters[key]

    @classmethod
    def _compare_viff(cls, ref: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        if min(ref.shape[-2:]) < 41:
            raise ValueError('四尺度 VIFF 要求图像至少 41x41；不对小图做隐式填充或缩放')
        numerator = ref.new_zeros((), dtype=torch.float64)
        denominator = numerator.clone()
        eps = 1e-10
        for scale, window in enumerate(cls._get_viff_windows(ref.device)):
            convolution = lambda image: F.conv2d(image.double(), window)
            if scale:
                ref, dist = convolution(ref)[..., ::2, ::2], convolution(dist)[..., ::2, ::2]
            mu1, mu2 = convolution(ref), convolution(dist)
            # 第一尺度 ref/dist 为float32，乘积先float32后经SciPy双精度卷积；后续尺度为float64。
            variance1 = (convolution(ref * ref) - mu1 * mu1).clamp_min(0)
            variance2 = (convolution(dist * dist) - mu2 * mu2).clamp_min(0)
            covariance = convolution(ref * dist) - mu1 * mu2
            gain = covariance / (variance1 + eps)
            residual = variance2 - gain * covariance
            small1 = variance1 < eps
            gain[small1] = 0
            residual[small1] = variance2[small1]
            variance1[small1] = 0
            small2 = variance2 < eps
            gain[small2] = 0
            residual[small2] = 0
            negative = gain < 0
            residual[negative] = variance2[negative]
            gain[negative] = 0
            residual[residual <= eps] = eps
            numerator += torch.log10(1. + gain * gain * variance1 / (residual + 2.)).sum()
            denominator += torch.log10(1. + variance1 / 2.).sum()
        ratio = numerator / denominator
        # 原实现仅在NaN时回退1；不把所有分母为0的情形改写成1。
        return torch.where(torch.isnan(ratio), 1., ratio)

    @classmethod
    def _viff(cls, fused, ir, vis):
        return cls._compare_viff(ir, fused) + cls._compare_viff(vis, fused)

    @staticmethod
    def _qabf_get_array(image):
        # PyTorch是互相关；翻转原Sobel核，等价scipy.convolve2d(..., mode='same')零填充。
        horizontal = image.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None]
        vertical = image.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])[None, None]
        sx, sy = F.conv2d(image, horizontal, padding=1), F.conv2d(image, vertical, padding=1)
        magnitude = torch.sqrt(sx * sx + sy * sy)
        angle = torch.empty_like(image)
        zero = sx == 0
        angle[zero] = math.pi / 2
        angle[~zero] = torch.atan(sy[~zero] / sx[~zero])
        return magnitude, angle

    @staticmethod
    def _qabf_calc(a, g, af, gf):
        strength = torch.zeros_like(a)
        greater, equal, less = g > gf, g == gf, g < gf
        strength[greater] = gf[greater] / g[greater]
        strength[equal] = gf[equal]  # 保留原源码的相等分支，不改成常见变体1。
        strength[less] = g[less] / gf[less]
        angle = 1. - torch.abs(a - af) / (math.pi / 2)
        quality_g = .9994 / (1. + torch.exp(-15. * (strength - .5)))
        quality_a = .9879 / (1. + torch.exp(-22. * (angle - .8)))
        return quality_g * quality_a

    @classmethod
    def _qabf(cls, fused, ir, vis):
        gi, ai = cls._qabf_get_array(ir)
        gv, av = cls._qabf_get_array(vis)
        gf, af = cls._qabf_get_array(fused)
        qi, qv = cls._qabf_calc(ai, gi, af, gf), cls._qabf_calc(av, gv, af, gf)
        return _sum32(qi * gi + qv * gv) / _sum32(gi + gv)

    @staticmethod
    def _box7(image: torch.Tensor) -> torch.Tensor:
        """等价uniform_filter的有效内区：axis0后axis1，每轴双精度累加、float32写回。

        只计算最终会保留的内区，故无需模拟被裁掉的reflect边界。先除7并写回，
        再沿另一轴除7，不能替换为一次49点float64均值。
        """
        height, width = image.shape[-2:]
        if min(height, width) < 7:
            raise ValueError('SSIM 至少需要 7x7 图像')
        vertical = torch.zeros((*image.shape[:-2], height - 6, width), dtype=torch.float64, device=image.device)
        for offset in range(7):
            vertical += image[..., offset:offset + height - 6, :].double()
        vertical = (vertical / 7.).float()
        horizontal = torch.zeros((*image.shape[:-2], height - 6, width - 6), dtype=torch.float64, device=image.device)
        for offset in range(7):
            horizontal += vertical[..., :, offset:offset + width - 6].double()
        return (horizontal / 7.).float()

    @classmethod
    def _ssim_single(cls, x, y):
        ux, uy = cls._box7(x), cls._box7(y)
        uxx, uyy, uxy = cls._box7(x * x), cls._box7(y * y), cls._box7(x * y)
        vx = (49. / 48.) * (uxx - ux * ux)
        vy = (49. / 48.) * (uyy - uy * uy)
        vxy = (49. / 48.) * (uxy - ux * uy)
        c1, c2 = (.01 * 2.) ** 2, (.03 * 2.) ** 2
        a1, a2 = 2 * ux * uy + c1, 2 * vxy + c2
        b1, b2 = ux ** 2 + uy ** 2 + c1, vx + vy + c2
        # skimage在float32中生成SSIM图，最后才用float64求均值。
        return ((a1 * a2) / (b1 * b2)).double().mean()

    @classmethod
    def _ssim(cls, fused, ir, vis):
        return cls._ssim_single(fused, ir) + cls._ssim_single(fused, vis)

    @classmethod
    def EN(cls, img, device=None) -> float:
        return cls._run(cls._en, img, device=device)

    @classmethod
    def SD(cls, img, device=None) -> float:
        return cls._run(cls._sd, img, device=device)

    @classmethod
    def SF(cls, img, device=None) -> float:
        return cls._run(cls._sf, img, device=device)

    @classmethod
    def MI(cls, image_F, image_A, image_B, device=None) -> float:
        return cls._run(cls._mi, image_F, image_A, image_B, device=device)

    @classmethod
    def SCD(cls, image_F, image_A, image_B, device=None) -> float:
        return cls._run(cls._scd, image_F, image_A, image_B, device=device)

    @classmethod
    def VIFF(cls, image_F, image_A, image_B, device=None) -> float:
        return cls._run(cls._viff, image_F, image_A, image_B, device=device)

    @classmethod
    def Qabf(cls, image_F, image_A, image_B, device=None) -> float:
        return cls._run(cls._qabf, image_F, image_A, image_B, device=device)

    @classmethod
    def SSIM(cls, image_F, image_A, image_B, device=None) -> float:
        return cls._run(cls._ssim, image_F, image_A, image_B, device=device)

    @classmethod
    def evaluate_all(cls, fi, ir, vi, device=None) -> np.ndarray:
        fused, infrared, visible = cls._prepare((fi, ir, vi), device)
        with cls._precision(fused.device):
            functions = (cls._en, cls._sd, cls._sf, cls._mi, cls._scd, cls._viff, cls._qabf, cls._ssim)
            values = [function(fused) if idx < 3 else function(fused, infrared, visible)
                      for idx, function in enumerate(functions)]
            return torch.stack([value.double() for value in values]).cpu().numpy()

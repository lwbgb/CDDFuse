#!/usr/bin/env python3
"""独立的红外/可见光融合评价：NumPy/SciPy float64，无模型或 MATLAB 依赖。

用法（不会自动修改旧评价器）：
    from test_metrics import FusionEvaluator
    result = FusionEvaluator().evaluate(ir, vis, fused, metrics="core")
    print(result.values, result.issues)
    python test_metrics.py --ir DIR/ir --vis DIR/vi --fused RESULT --metrics all --output OUT
    python test_metrics.py --describe

数组输入为二维灰度，或单张 [1,1,H,W] 张量；浮点范围必须通过 data_range 显式声明。
默认0..255，所有内部物理尺度统一到0..255；不逐图拉伸（Qcb/Qcv原始定义内部除外）。
MI默认bit；SSIM/VIF/MS-SSIM/MI默认两源求和，其他指标采用各自定义的融合聚合。
无效指标返回NaN并记录原因，strict=True可逐项抛错；PSNR的+Inf表示零均方误差。
本文件是有声明约定的参考实现，不声称与所有同名历史评测器逐值兼容。

参考：
* Wang et al., TIP 2004, SSIM；Sheikh/Bovik, TIP 2006, LIVE VIFp。
* Xydeas/Petrovic, Electronics Letters 2000, Qabf（修正等梯度强度比）。
* Kumar, modified fusion artifacts, 2011；Haghighat et al., FMI 2011/Fast-FMI 2014。
* Ma/Zeng/Wang, TIP 2015, MEF-SSIM（与普通MS-SSIM分别命名）。
* VIFB: github.com/xingchenzhang/VIFB @22b2ea9d91499cc5962d9bdcae9c290270f0fade。
* github.com/Linfeng-Tang/Evaluation-for-Image-Fusion @f5f055bcadb49c22fb734c3498aef6c56fc71f2a。
* github.com/zhengliu6699/imageFusionMetrics (Piella, Yang, Cvejic, NCIE, Tsallis)。
* Ordinary MS-SSIM odd-size pooling: github.com/VainF/pytorch-msssim。
* dmey coefficients: PyWavelets v1.8.0 (permission notice at end of file)。

MEF算法的研究用途许可、LIVE及PyWavelets的许可与出处见文件末尾。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft, ndimage
from scipy.special import xlogy


PROTOCOL_VERSION = "fusion-reference-1.0"
CORE_METRICS = ("EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM")
_DESCRIPTION = {
    "EN": ("信息熵", "↑", "256-bin discrete entropy, log2"),
    "SD": ("总体标准差", "↑", "population std, ddof=0"),
    "SF": ("空间频率", "↑", "adjacent differences; valid or HW denominators"),
    "MI": ("互信息", "↑", "256-bin discrete MI; configured log base and pair aggregation"),
    "SCD": ("差分相关性之和", "↑", "corr(IR,F-VIS)+corr(VIS,F-IR)"),
    "VIF": ("视觉信息保真度", "↑", "LIVE four-scale pixel-domain VIFp, noise variance2; no NaN-to-one"),
    "Qabf": ("边缘信息保留", "↑", "Xydeas/Petrovic; equal positive strength ratio1; declared power/angle"),
    "SSIM": ("结构相似性", "↑", "Wang: Gaussian11 sigma1.5, population covariance, L255, valid"),
    "CC": ("相关系数", "↑", "mean of two Pearson correlations"),
    "AG": ("平均梯度", "↑", "central interior / one-sided boundary, sum/((H-1)(W-1)), VIFB"),
    "MSE": ("均方误差", "↓", "mean of two source MSEs, 0..255 squared units"),
    "EI": ("边缘强度", "↑", "mean Sobel magnitude, replicate boundary, VIFB"),
    "NMI": ("归一化互信息", "↑", "Hossny: sum of 2I/(Hsource+Hfused), range 0..2"),
    "CE": ("交叉熵类散度", "↓", "mean KL(source||fused), log2; missing support => +Inf"),
    "CE_overlap": ("历史重叠分箱散度", "↓", "VIFB CE: skip bins absent in either histogram; can be negative"),
    "RMSE": ("均方根误差", "↓", "mean of two source RMSEs; not sqrt(mean source MSE)"),
    "PSNR": ("峰值信噪比", "↑", "10log10(255^2/mean source MSE); exact equality => +Inf"),
    "MS_SSIM": ("多尺度结构相似性", "↑", "ordinary five-scale pairwise MS-SSIM; ReLU CS; zero-padded odd pooling"),
    "Nabf": ("新增融合伪影", "↓", "Kumar modified artifact measure, Sobel/8 mirror, threshold2, power1.5"),
    "Labf": ("融合信息损失", "↓", "Kumar/Petrovic loss complement; shares Nabf local preservation"),
    "FMI_pixel": ("像素特征互信息", "↑", "Fast-FMI, explicitly raw pixels (not default edge), window3"),
    "FMI_dct": ("DCT特征互信息", "↑", "Fast-FMI, whole-image orthonormal DCT-II, window3"),
    "FMI_w": ("小波特征互信息", "↑", "Fast-FMI, dmey62, half-sample symmetric DWT, window3"),
    "Qcb": ("Chen–Blum感知指标", "↑", "Chen/Blum 2009, VIFB parameters, DoG, 31x31 contrast filters"),
    "Qcv": ("Chen–Varshney感知失真", "↓", "Chen/Varshney 2007, Mannos CSF, padded16x16 blocks, alpha5"),
    "MEF_SSIM": ("多曝光融合结构相似性", "↑", "Ma/Zeng/Wang reference single-scale, Gaussian11 sigma1.5"),
    "MEF_MS_SSIM": ("多尺度多曝光融合结构相似性", "↑", "three MEF-SSIM scales, weights .0448,.2856,.3001 normalized"),
    "Qw": ("Piella加权融合质量", "↑", "Piella case2; variance saliency with Wang SSIM maps"),
    "Qy": ("Yang融合质量", "↑", "Gaussian7, C1=C2=2e-16, threshold .75, population covariance"),
    "Qc": ("Cvejic融合质量", "↑", "Cvejic case2, covariance weights, Gaussian7"),
    "Qncie": ("非线性相关信息熵", "↑", "Wang NCIE, pair MI_bits/8, eigenvalue entropy normalized by8"),
    "TE": ("Tsallis互信息", "↑", "Cvejic q=1.85, aligned marginal products, two-source sum"),
    "UQI": ("通用图像质量指数", "↑", "Wang/Bovik, uniform8 valid, mean over sources"),
}
ALL_METRICS = tuple(_DESCRIPTION)
_ALIASES = {name.upper().replace("-", "_"): name for name in ALL_METRICS}
_ALIASES.update({"VIFF": "VIF", "VIFP": "VIF", "QAB/F": "Qabf", "QAB_F": "Qabf",
                 "NAB/F": "Nabf", "MS-SSIM": "MS_SSIM", "QNCIE": "Qncie"})
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class UndefinedMetric(ValueError):
    """输入合法，但该指标在当前图像或尺寸上没有定义。"""


@dataclass(frozen=True)
class EvaluationConfig:
    data_range: float = 255.0
    pair_aggregate: str = "sum"
    mi_log_base: float = 2.0
    sf_denominator: str = "valid"
    qabf_power: float = 1.0
    qabf_angle: str = "legacy"
    fmi_chunk_size: int = 2048
    mef_chunk_size: int = 512
    strict: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.data_range) or self.data_range <= 0:
            raise ValueError("data_range 必须是有限正数")
        if self.pair_aggregate not in {"sum", "mean"}:
            raise ValueError("pair_aggregate 应为 sum 或 mean")
        if not math.isfinite(self.mi_log_base) or self.mi_log_base <= 1:
            raise ValueError("mi_log_base 必须为有限数且大于1")
        if self.sf_denominator not in {"valid", "hw"}:
            raise ValueError("sf_denominator 应为 valid 或 hw")
        if not math.isfinite(self.qabf_power) or self.qabf_power <= 0:
            raise ValueError("qabf_power 必须为有限正数")
        if self.qabf_angle not in {"legacy", "periodic"}:
            raise ValueError("qabf_angle 应为 legacy 或 periodic")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (self.fmi_chunk_size, self.mef_chunk_size)):
            raise ValueError("分块大小必须为正整数")


@dataclass
class EvaluationResult:
    values: dict[str, float]
    issues: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    diagnostics: dict[str, dict] = field(default_factory=dict)

    def serializable(self) -> dict:
        return {"values": {name: _json_number(value) for name, value in self.values.items()},
                "issues": self.issues, "elapsed_seconds": self.elapsed_seconds,
                "diagnostics": self.diagnostics}


def _json_number(value: float) -> float | str | None:
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return float(value)


def resolve_metrics(metrics: str | Sequence[str] = "all") -> tuple[str, ...]:
    if isinstance(metrics, str):
        if metrics.lower() == "all":
            return ALL_METRICS
        if metrics.lower() == "core":
            return CORE_METRICS
        metrics = [part.strip() for part in metrics.split(",") if part.strip()]
    result = []
    for name in metrics:
        key = str(name).upper().replace("-", "_")
        if key not in _ALIASES:
            raise ValueError(f"未知指标：{name}；可选 {', '.join(ALL_METRICS)}")
        canonical = _ALIASES[key]
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("至少选择一个指标")
    return tuple(result)


def _gray_array(value: object, data_range: float) -> np.ndarray:
    # Avoid importing torch just to evaluate NumPy images; CUDA transfer is explicit.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().double().numpy()
    x = np.asarray(value)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError(f"输入须为至少2x2的单张灰度图，实际 {x.shape}")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("输入须为实数图像")
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("输入包含NaN/Inf")
    if np.min(x) < 0 or np.max(x) > data_range:
        raise ValueError(f"输入像素须位于[0,{data_range}]，不会静默裁剪或拉伸")
    return np.ascontiguousarray(x * (255.0 / data_range) if data_range != 255 else x)


@lru_cache(maxsize=32)
def _gaussian(size: int, sigma: float) -> np.ndarray:
    t = np.arange(size, dtype=np.float64) - (size - 1) / 2
    w = np.exp(-t * t / (2 * sigma * sigma))
    w /= w.sum()
    w.flags.writeable = False
    return w


def _separable(x: np.ndarray, w: np.ndarray, *, valid: bool = True,
               mode: str = "constant") -> np.ndarray:
    size = len(w)
    if valid and min(x.shape) < size:
        raise UndefinedMetric(f"图像最短边{min(x.shape)}小于窗口{size}")
    out = ndimage.correlate1d(x, w, axis=0, mode=mode, cval=0.0)
    out = ndimage.correlate1d(out, w, axis=1, mode=mode, cval=0.0)
    if valid:
        left, right = size // 2, size - 1 - size // 2
        return out[left:out.shape[0] - right, left:out.shape[1] - right]
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    a, b = x.ravel() - x.mean(), y.ravel() - y.mean()
    da, db = np.sum(a * a), np.sum(b * b)
    if da == 0 or db == 0:
        raise UndefinedMetric("Pearson相关系数遇到常量图或常量差分")
    return float(np.clip(np.sum(a * b) / math.sqrt(da * db), -1, 1))


def _hist_entropy(p: np.ndarray, base: float = 2.0) -> float:
    return float(-np.sum(xlogy(p, p)) / math.log(base))


def _sobel(x: np.ndarray, mode: str = "constant") -> tuple[np.ndarray, np.ndarray]:
    gx = ndimage.sobel(x, axis=1, mode=mode, cval=0.0)
    gy = ndimage.sobel(x, axis=0, mode=mode, cval=0.0)
    return gx, gy


def _angle(gx: np.ndarray, gy: np.ndarray, *, zero_angle: float = math.pi / 2) -> np.ndarray:
    # atan(gy/gx), including the original direction interval, with explicit zero convention.
    theta = np.full(gx.shape, zero_angle, dtype=np.float64)
    np.divide(gy, gx, out=theta, where=gx != 0)
    mask = gx != 0
    theta[mask] = np.arctan(theta[mask])
    return theta


def _downsample_ms(x: np.ndarray) -> np.ndarray:
    """Matches pytorch-msssim avg_pool2d(k2,padding=size%2,count_include_pad=True)."""
    h, w = x.shape
    ph, pw = h % 2, w % 2
    z = np.pad(x, ((ph, ph), (pw, pw)), mode="constant")
    hh, ww = z.shape[0] // 2 * 2, z.shape[1] // 2 * 2
    return z[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).mean(axis=(1, 3))


def _downsample_mef(x: np.ndarray) -> np.ndarray:
    # MATLAB imfilter(ones(2)/4,'symmetric','same'), followed by 1:2:end.
    z = np.pad(x, ((0, 1), (0, 1)), mode="symmetric")
    return ((z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:]) / 4)[::2, ::2]


def _ssim_maps(a: np.ndarray, b: np.ndarray, size: int = 11,
               c1: float = 6.5025, c2: float = 58.5225) -> tuple[np.ndarray, ...]:
    w = _gaussian(size, 1.5)
    ma, mb = _separable(a, w), _separable(b, w)
    va = np.maximum(_separable(a * a, w) - ma * ma, 0.0)
    vb = np.maximum(_separable(b * b, w) - mb * mb, 0.0)
    cov = _separable(a * b, w) - ma * mb
    bound = np.sqrt(va * vb)
    cov = np.clip(cov, -bound, bound)
    cs = (2 * cov + c2) / (va + vb + c2)
    score = (2 * ma * mb + c1) / (ma * ma + mb * mb + c1) * cs
    return score, cs, va, vb, cov


def _ms_ssim_pair(a: np.ndarray, b: np.ndarray,
                  first_maps: tuple[np.ndarray, ...] | None = None) -> float:
    weights = np.array([.0448, .2856, .3001, .2363, .1333])
    if min(a.shape) <= 160:
        raise UndefinedMetric("五尺度MS-SSIM需要最短边至少161；不会静默减少尺度")
    factors = []
    for level in range(5):
        score, cs, *_ = first_maps if level == 0 and first_maps is not None else _ssim_maps(a, b)
        factors.append(max(float(score.mean() if level == 4 else cs.mean()), 0.0))
        if level < 4:
            a, b = _downsample_ms(a), _downsample_ms(b)
    return float(np.prod(np.asarray(factors) ** weights))


def _vif_pair(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0:
        raise UndefinedMetric("VIFp参考图为常量，没有可归一化的参考信息")
    num, den, eps = 0.0, 0.0, 1e-10
    for scale, size in enumerate((17, 9, 5, 3)):
        w = _gaussian(size, size / 5.0)
        if scale:
            a, b = _separable(a, w)[::2, ::2], _separable(b, w)[::2, ::2]
        ma, mb = _separable(a, w), _separable(b, w)
        va = np.maximum(_separable(a * a, w) - ma * ma, 0.0)
        vb = np.maximum(_separable(b * b, w) - mb * mb, 0.0)
        cov = _separable(a * b, w) - ma * mb
        gain = cov / (va + eps)
        noise = vb - gain * cov
        mask = va < eps
        gain[mask], noise[mask], va[mask] = 0.0, vb[mask], 0.0
        mask = vb < eps
        gain[mask], noise[mask] = 0.0, 0.0
        mask = gain < 0
        noise[mask], gain[mask] = vb[mask], 0.0
        noise = np.maximum(noise, eps)
        num += float(np.log1p(gain * gain * va / (noise + 2)).sum())
        den += float(np.log1p(va / 2).sum())
    if den == 0:
        raise UndefinedMetric("VIFp参考信息为零，分母为0")
    return num / den


# dmey reconstruction-lowpass coefficients, PyWavelets v1.8.0.  See license below.
_DMEY_REC_LO = np.array([
-1.0099999569414229e-012, 8.519459636796214e-009, -1.111944952595278e-008, -1.0798819539621958e-008, 6.0669757413511352e-008, -1.0866516536735883e-007, 8.2006806503864813e-008, 1.1783004497663934e-007, -5.5063405652522782e-007, 1.1307947017916706e-006, -1.4895492164971559e-006, 7.367572885903746e-007, 3.2054419133447798e-006, -1.6312699734552807e-005, 6.5543059305751491e-005, -0.00060115023435160925, -0.002704672124643725, 0.0022025341009110021, 0.006045814097323304, -0.0063877183184971563, -0.011061496392513451, 0.015270015130934803, 0.017423434103729693, -0.032130793990211758, -0.024348745906078023, 0.063739024322801596, 0.030655091960824263, -0.13284520043622938, -0.035087555656258346, 0.44459300275757724, 0.74458559231880628, 0.44459300275757724, -0.035087555656258346, -0.13284520043622938, 0.030655091960824263, 0.063739024322801596, -0.024348745906078023, -0.032130793990211758, 0.017423434103729693, 0.015270015130934803, -0.011061496392513451, -0.0063877183184971563, 0.006045814097323304, 0.0022025341009110021, -0.002704672124643725, -0.00060115023435160925, 6.5543059305751491e-005, -1.6312699734552807e-005, 3.2054419133447798e-006, 7.367572885903746e-007, -1.4895492164971559e-006, 1.1307947017916706e-006, -5.5063405652522782e-007, 1.1783004497663934e-007, 8.2006806503864813e-008, -1.0866516536735883e-007, 6.0669757413511352e-008, -1.0798819539621958e-008, -1.111944952595278e-008, 8.519459636796214e-009, -1.0099999569414229e-012, 0.0
], dtype=np.float64)


def _dwt_axis(x: np.ndarray, coefficients: np.ndarray, axis: int) -> np.ndarray:
    """Half-sample symmetric extension; y[k]=sum_j h[j]x[2k+1-j]."""
    n, length = x.shape[axis], coefficients.size
    count = (n + length - 1) // 2
    # correlate1d is faster and bounded in memory; pad reproduces the same folded indices.
    pads = [(0, 0)] * x.ndim
    pads[axis] = (length - 2, length - 1)
    extended = np.pad(x, pads, mode="symmetric")
    # Valid convolution and odd phase in original coordinates.
    filtered = ndimage.correlate1d(extended, coefficients[::-1], axis=axis, mode="constant")
    take = [slice(None)] * x.ndim
    take[axis] = slice(length // 2, length // 2 + 2 * count, 2)
    return filtered[tuple(take)]


def _dmey_feature(x: np.ndarray) -> np.ndarray:
    lo = _DMEY_REC_LO[::-1]
    hi = _DMEY_REC_LO * np.where(np.arange(62) % 2 == 0, -1.0, 1.0)
    low0, high0 = _dwt_axis(x, lo, 0), _dwt_axis(x, hi, 0)
    aa, ad = _dwt_axis(low0, lo, 1), _dwt_axis(low0, hi, 1)
    da, dd = _dwt_axis(high0, lo, 1), _dwt_axis(high0, hi, 1)
    # MATLAB/PyWavelets cH is highpass along axis0, cV along axis1.
    out = np.block([[aa, da], [ad, dd]])
    span = float(out.max() - out.min())
    return (out - out.min()) / span if span else np.ones_like(out)


def _patch_batches(arrays: Sequence[np.ndarray], window: int,
                   chunk: int) -> Iterable[list[np.ndarray]]:
    h, w = arrays[0].shape
    if min(h, w) < window:
        raise UndefinedMetric(f"图像小于{window}x{window}窗口")
    views = [sliding_window_view(a, (window, window)) for a in arrays]
    rows, cols = h - window + 1, w - window + 1
    for start in range(0, rows * cols, chunk):
        linear = np.arange(start, min(start + chunk, rows * cols))
        r, c = linear // cols, linear % cols
        # MATLAB vec(:) is column-major; use the same ordering for Fast-FMI's copula.
        yield [v[r, c].transpose(0, 2, 1).reshape(len(linear), -1) for v in views]


def _local_fmi(a: np.ndarray, b: np.ndarray, diagnostics: dict | None = None) -> np.ndarray:
    """Vectorized Fast-FMI for a batch of window vectors (not global histogram MI)."""
    identical = np.all(a == b, axis=1)
    def pdf(z: np.ndarray) -> np.ndarray:
        lower, span = z.min(axis=1, keepdims=True), np.ptp(z, axis=1, keepdims=True)
        norm = np.divide(z - lower, span, out=np.ones_like(z), where=span != 0)
        return norm / norm.sum(axis=1, keepdims=True)
    pa, pb = pdf(a), pdf(b)
    ca, cb = np.cumsum(pa, axis=1), np.cumsum(pb, axis=1)
    xa, xb = pa - pa.mean(axis=1, keepdims=True), pb - pb.mean(axis=1, keepdims=True)
    dot = np.sum(xa * xb, axis=1)
    norm = np.sqrt(np.sum(xa * xa, axis=1) * np.sum(xb * xb, axis=1))
    correlation = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
    index = np.arange(1, pa.shape[1] + 1, dtype=float)
    ea, eb = np.sum(pa * index, axis=1), np.sum(pb * index, axis=1)
    variance = np.maximum(np.sum(pa * index ** 2, axis=1) - ea * ea, 0) * np.maximum(np.sum(pb * index ** 2, axis=1) - eb * eb, 0)
    sd = np.sqrt(variance)
    independent = pa[:, :, None] * pb[:, None, :]
    product_cdf = ca[:, :, None] * cb[:, None, :]
    upper = np.minimum(ca[:, :, None], cb[:, None, :])
    lower = np.maximum(ca[:, :, None] + cb[:, None, :] - 1, 0)
    bound = np.where((correlation >= 0)[:, None, None], upper, lower)
    denominator = (bound - product_cdf).sum(axis=(1, 2))
    scale = np.divide(correlation * sd, denominator, out=np.zeros_like(sd),
                      where=(sd > 0) & (denominator != 0) & (correlation != 0))
    padded = np.pad(bound, ((0, 0), (1, 0), (1, 0)), mode="constant")
    joint_bound = padded[:, 1:, 1:] - padded[:, :-1, 1:] - padded[:, 1:, :-1] + padded[:, :-1, :-1]
    joint = scale[:, None, None] * joint_bound + (1 - scale[:, None, None]) * independent
    if diagnostics is not None:
        diagnostics['windows'] = diagnostics.get('windows', 0) + len(joint)
        diagnostics['negative_joint_windows'] = diagnostics.get('negative_joint_windows', 0) + int(np.any(joint < -1e-12, axis=(1, 2)).sum())
        diagnostics['minimum_joint_term'] = min(diagnostics.get('minimum_joint_term', 0.0), float(joint.min()))
    # The author's Fast-FMI estimator can exceed Frechet mixture bounds because
    # it estimates dependence from correlation of the marginal PDF vectors.
    # Preserve its explicit real(-p*log2(p)) convention, including negative p,
    # rather than silently clipping the mixture and defining a different metric.
    hab = -np.sum(xlogy(joint, np.abs(joint)), axis=(1, 2)) / np.log(2)
    ha = -np.sum(xlogy(pa, pa), axis=1) / np.log(2)
    hb = -np.sum(xlogy(pb, pb), axis=1) / np.log(2)
    mi = ha + hb - hab
    out = np.divide(2 * mi, ha + hb, out=np.zeros_like(mi), where=ha + hb > 0)
    out[identical] = 1
    return out


def _fmi(a: np.ndarray, b: np.ndarray, f: np.ndarray, chunk: int,
         diagnostics: dict | None = None) -> float:
    total, count = 0.0, 0
    for aa, bb, ff in _patch_batches((a, b, f), 3, chunk):
        scores = (_local_fmi(aa, ff, diagnostics) + _local_fmi(bb, ff, diagnostics)) / 2
        total += float(scores.sum())
        count += len(scores)
    return total / count


def _mef_ssim(a: np.ndarray, b: np.ndarray, f: np.ndarray, chunk: int) -> float:
    window = np.outer(_gaussian(11, 1.5), _gaussian(11, 1.5)).T.ravel()
    total, count, eps = 0.0, 0, np.finfo(float).eps
    for aa, bb, ff in _patch_batches((a, b, f), 11, chunk):
        xa, xb = aa - aa.mean(axis=1, keepdims=True), bb - bb.mean(axis=1, keepdims=True)
        na, nb = np.linalg.norm(xa, axis=1), np.linalg.norm(xb, axis=1)
        consistency = (np.linalg.norm(xa + xb, axis=1) + eps) / (na + nb + eps)
        consistency = np.clip(consistency, 0, 1)
        power = np.minimum(np.tan(np.pi / 2 * consistency), 10)
        eda, edb = na + .001, nb + .001
        wa, wb = (eda / 11) ** power + eps, (edb / 11) ** power + eps
        r = wa[:, None] / (wa + wb)[:, None] * xa / eda[:, None]
        r += wb[:, None] / (wa + wb)[:, None] * xb / edb[:, None]
        norm = np.linalg.norm(r, axis=1)
        r *= np.divide(np.maximum(eda, edb), norm, out=np.zeros_like(norm), where=norm > 0)[:, None]
        mr, mf = np.sum(r * window, axis=1), np.sum(ff * window, axis=1)
        rc, fc = r - mr[:, None], ff - mf[:, None]
        vr, vf = np.sum(rc * rc * window, axis=1), np.sum(fc * fc * window, axis=1)
        covariance = np.sum(rc * fc * window, axis=1)
        scores = (2 * covariance + 58.5225) / (vr + vf + 58.5225)
        total += float(scores.sum())
        count += len(scores)
    return total / count


def _normalized255(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min())
    if span == 0:
        raise UndefinedMetric("Qcb/Qcv内部逐图归一化遇到常量图")
    return np.floor((x - x.min()) / span * 255 + .5)


def _frequency_radius(shape: tuple[int, int], divisor: float) -> np.ndarray:
    h, w = shape
    y = np.fft.fftshift(np.fft.fftfreq(h)) * 2 * h / divisor
    x = np.fft.fftshift(np.fft.fftfreq(w)) * 2 * w / divisor
    return np.hypot(y[:, None], x[None, :])


def _block16_sum(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    z = np.pad(x, ((0, (-h) % 16), (0, (-w) % 16)), mode="constant")
    return z.reshape(z.shape[0] // 16, 16, z.shape[1] // 16, 16).sum(axis=(1, 3))


class _Context:
    def __init__(self, a: np.ndarray, b: np.ndarray, f: np.ndarray, config: EvaluationConfig):
        self.images = (a, b, f)
        self.cfg = config
        self.cache: dict[object, object] = {}
        self.diagnostics: dict[str, dict] = {}

    def cached(self, key: object, producer: Callable[[], object]):
        if key not in self.cache:
            self.cache[key] = producer()
        return self.cache[key]

    def aggregate(self, values: Sequence[float]) -> float:
        return float(sum(values) / (len(values) if self.cfg.pair_aggregate == "mean" else 1))

    def labels(self, i: int) -> np.ndarray:
        return self.cached(("labels", i), lambda: np.floor(self.images[i] + .5).astype(np.int64))

    def hist(self, i: int) -> np.ndarray:
        return self.cached(("hist", i), lambda: np.bincount(self.labels(i).ravel(), minlength=256) / self.images[i].size)

    def joint(self, i: int, j: int) -> np.ndarray:
        if i > j:
            return self.joint(j, i).T
        return self.cached(("joint", i, j), lambda: np.bincount(
            (self.labels(i) * 256 + self.labels(j)).ravel(), minlength=65536).reshape(256, 256) / self.images[i].size)

    def mutual_information(self, i: int, j: int) -> float:
        def compute():
            p = self.joint(i, j)
            pi, pj = self.hist(i), self.hist(j)
            r, c = np.nonzero(p)
            values = p[r, c]
            return float(np.sum(values * (np.log(values) - np.log(pi[r]) - np.log(pj[c]))))
        return self.cached(("mi_nats", min(i, j), max(i, j)), compute)

    def gradients(self, i: int, mode: str = "constant") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def compute():
            gx, gy = _sobel(self.images[i], mode)
            return gx, gy, np.hypot(gx, gy)
        return self.cached(("sobel", i, mode), compute)

    def stats(self, i: int, size: int = 11) -> tuple[np.ndarray, np.ndarray]:
        def compute():
            x = self.images[i]
            w = _gaussian(size, 1.5)
            mu = _separable(x, w)
            return mu, np.maximum(_separable(x * x, w) - mu * mu, 0)
        return self.cached(("moments", i, size), compute)

    def ssim_map(self, i: int, j: int, size: int = 11,
                 c1: float = 6.5025, c2: float = 58.5225) -> tuple[np.ndarray, ...]:
        def compute():
            mi, vi = self.stats(i, size)
            mj, vj = self.stats(j, size)
            covariance = _separable(self.images[i] * self.images[j], _gaussian(size, 1.5)) - mi * mj
            bound = np.sqrt(vi * vj)
            covariance = np.clip(covariance, -bound, bound)
            cs = (2 * covariance + c2) / (vi + vj + c2)
            s = (2 * mi * mj + c1) / (mi * mi + mj * mj + c1) * cs
            return s, cs, vi, vj, covariance
        return self.cached(("ssim", min(i, j), max(i, j), size, c1, c2), compute)

    def mse_pairs(self) -> tuple[float, float]:
        return self.cached("mse", lambda: tuple(float(np.mean((x - self.images[2]) ** 2)) for x in self.images[:2]))

    def en(self) -> float:
        return _hist_entropy(self.hist(2))

    def sd(self) -> float:
        return float(np.std(self.images[2]))

    def sf(self) -> float:
        f = self.images[2]
        dx, dy = np.diff(f, axis=1), np.diff(f, axis=0)
        if self.cfg.sf_denominator == "hw":
            return float(np.sqrt((np.sum(dx * dx) + np.sum(dy * dy)) / f.size))
        return float(np.sqrt(np.mean(dx * dx) + np.mean(dy * dy)))

    def ag(self) -> float:
        dy, dx = np.gradient(self.images[2])
        h, w = dx.shape
        return float(np.sqrt((dx * dx + dy * dy) / 2).sum() / ((h - 1) * (w - 1)))

    def ei(self) -> float:
        return float(self.gradients(2, "nearest")[2].mean())

    def mi(self) -> float:
        return self.aggregate([self.mutual_information(i, 2) / math.log(self.cfg.mi_log_base) for i in (0, 1)])

    def nmi(self) -> float:
        hf = _hist_entropy(self.hist(2), math.e)
        values = []
        for i in (0, 1):
            denominator = hf + _hist_entropy(self.hist(i), math.e)
            if denominator == 0:
                raise UndefinedMetric("NMI的两项边际熵均为零")
            values.append(2 * self.mutual_information(i, 2) / denominator)
        return float(sum(values))

    def ce(self, *, overlap: bool = False) -> float:
        q, values = self.hist(2), []
        for i in (0, 1):
            p = self.hist(i)
            if not overlap and np.any((p > 0) & (q == 0)):
                return math.inf
            mask = (p > 0) & (q > 0)
            values.append(float(np.sum(p[mask] * np.log2(p[mask] / q[mask]))))
        return float(np.mean(values))

    def mse(self) -> float:
        return float(np.mean(self.mse_pairs()))

    def rmse(self) -> float:
        return float(np.mean(np.sqrt(self.mse_pairs())))

    def psnr(self) -> float:
        error = self.mse()
        return math.inf if error == 0 else float(10 * math.log10(255 * 255 / error))

    def cc(self) -> float:
        a, b, f = self.images
        return (_pearson(a, f) + _pearson(b, f)) / 2

    def scd(self) -> float:
        a, b, f = self.images
        return _pearson(a, f - b) + _pearson(b, f - a)

    def ssim(self) -> float:
        return self.aggregate([float(self.ssim_map(i, 2)[0].mean()) for i in (0, 1)])

    def ms_ssim(self) -> float:
        return self.aggregate([_ms_ssim_pair(self.images[i], self.images[2], self.ssim_map(i, 2)) for i in (0, 1)])

    def vif(self) -> float:
        return self.aggregate([_vif_pair(x, self.images[2]) for x in self.images[:2]])

    def qabf(self) -> float:
        gxf, gyf, gf = self.gradients(2)
        af = _angle(gxf, -gyf)
        num, den = 0.0, 0.0
        for i in (0, 1):
            gx, gy, g = self.gradients(i)
            angle = _angle(gx, -gy)
            maximum = np.maximum(g, gf)
            ratio = np.divide(np.minimum(g, gf), maximum, out=np.ones_like(g), where=maximum > 0)
            delta = np.abs(angle - af)
            if self.cfg.qabf_angle == "periodic":
                delta = np.minimum(delta, np.pi - delta)
            orientation = 1 - delta / (np.pi / 2)
            quality = .9994 / (1 + np.exp(-15 * (ratio - .5)))
            quality *= .9879 / (1 + np.exp(-22 * (orientation - .8)))
            weights = g ** self.cfg.qabf_power
            num += float(np.sum(quality * weights))
            den += float(weights.sum())
        if den == 0:
            raise UndefinedMetric("Qabf两源梯度权重总和为0")
        return num / den

    def artifacts(self) -> tuple[float, float]:
        def compute():
            gx, gy, gf = self.gradients(2, "mirror")
            gf = gf / 8
            def orientation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                ratio = np.divide(x, y, out=np.copysign(np.full_like(x, np.inf), x), where=y != 0)
                ratio[(x == 0) & (y == 0)] = 0
                return np.arctan(ratio)
            af = orientation(gx, gy)
            qualities, weights, strengths = [], [], []
            for i in (0, 1):
                sx, sy, g = self.gradients(i, "mirror")
                g = g / 8
                angle = orientation(sx, sy)
                maximum = np.maximum(g, gf)
                ratio = np.divide(np.minimum(g, gf), maximum, out=np.zeros_like(g), where=maximum > 0)
                orient = np.abs(np.abs(angle - af) - np.pi / 2) * 2 / np.pi
                qg = .9999 / (1 + np.exp(-19 * (ratio - .5)))
                qa = .9995 / (1 + np.exp(-22 * (orient - .5)))
                qualities.append(np.sqrt(qg * qa))
                weights.append(np.where(g >= 2, g ** 1.5, .001))
                strengths.append(g)
            w0, w1 = weights
            error = (1 - qualities[0]) * w0 + (1 - qualities[1]) * w1
            new_edge = (gf > strengths[0]) & (gf > strengths[1])
            denominator = float((w0 + w1).sum())
            return float(error[new_edge].sum() / denominator), float(error[~new_edge].sum() / denominator)
        return self.cached("artifacts", compute)

    def fmi(self, feature: str) -> float:
        def transform(x: np.ndarray) -> np.ndarray:
            if feature == "pixel":
                return x
            if feature == "dct":
                return fft.dctn(x, type=2, norm="ortho", workers=1)
            return _dmey_feature(x)
        transformed = [transform(x) for x in self.images]
        diagnostics: dict = {}
        result = _fmi(*transformed, self.cfg.fmi_chunk_size, diagnostics)
        self.diagnostics['FMI_' + ('w' if feature == 'wavelet' else feature)] = diagnostics
        return result

    def normalized_images(self) -> tuple[np.ndarray, ...]:
        return self.cached("normalized255", lambda: tuple(_normalized255(x) for x in self.images))

    def qcb(self) -> float:
        images = self.normalized_images()
        radius = _frequency_radius(images[0].shape, 30)
        transfer = np.exp(-(radius / 15.3870) ** 2) - .7622 * np.exp(-(radius / 1.3456) ** 2)
        contrasts = []
        for x in images:
            filtered = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(x)) * transfer)).real
            # The original 31x31 Gaussians are NOT renormalized after truncation.
            t = np.arange(-15, 16, dtype=float)
            gaussian2 = np.exp(-t * t / 8) / math.sqrt(8 * math.pi)
            gaussian4 = np.exp(-t * t / 32) / math.sqrt(32 * math.pi)
            inner = _separable(filtered, gaussian2, valid=False)
            outer = _separable(filtered, gaussian4, valid=False)
            if np.any(outer == 0):
                raise UndefinedMetric("Qcb局部对比度归一化分母为0")
            contrast = np.abs(inner / outer - 1)
            contrasts.append(contrast ** 3 / (contrast ** 2 + .0001))
        a, b, f = contrasts
        def ratio(x):
            den = np.maximum(x, f)
            return np.divide(np.minimum(x, f), den, out=np.ones_like(x), where=den > 0)
        den = a * a + b * b
        weight = np.divide(a * a, den, out=np.full_like(a, .5), where=den > 0)
        return float(np.mean(weight * ratio(a) + (1 - weight) * ratio(b)))

    def qcv(self) -> float:
        a, b, f = self.normalized_images()
        r = _frequency_radius(a.shape, 8)
        transfer = 2.6 * (.0192 + .144 * r) * np.exp(-(.144 * r) ** 1.1)
        numerator, denominator = 0.0, 0.0
        for source in (a, b):
            gx, gy = _sobel(source)
            weight = _block16_sum(np.hypot(gx, gy) ** 5)
            difference = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(source - f)) * transfer)).real
            distortion = _block16_sum(difference * difference) / 256
            numerator += float(np.sum(weight * distortion))
            denominator += float(weight.sum())
        if denominator == 0:
            raise UndefinedMetric("Qcv显著性权重为零")
        return numerator / denominator

    def mef(self, multiscale: bool) -> float:
        a, b, f = self.images
        if not multiscale:
            return self.cached('mef_scale0', lambda: _mef_ssim(a, b, f, self.cfg.mef_chunk_size))
        if min(a.shape) < 41:
            raise UndefinedMetric("三尺度MEF-MS-SSIM需要最短边至少41")
        scores = []
        for level in range(3):
            value = self.mef(False) if level == 0 else _mef_ssim(a, b, f, self.cfg.mef_chunk_size)
            if value < 0:
                raise UndefinedMetric("MEF-MS-SSIM负的单尺度分数无法取实数分数幂")
            scores.append(value)
            if level < 2:
                a, b, f = (_downsample_mef(x) for x in (a, b, f))
        weights = np.array([.0448, .2856, .3001])
        return float(np.prod(np.asarray(scores) ** (weights / weights.sum())))

    def qw(self) -> float:
        _, va = self.stats(0)
        _, vb = self.stats(1)
        sa, sb = self.ssim_map(0, 2)[0], self.ssim_map(1, 2)[0]
        flat = va + vb == 0
        va, vb = va + flat * .5, vb + flat * .5
        lam = va / (va + vb)
        saliency = np.maximum(va, vb)
        return float(np.sum(saliency * (lam * sa + (1 - lam) * sb)) / saliency.sum())

    def qy(self) -> float:
        sab, _, va, vb, _ = self.ssim_map(0, 1, 7, 2e-16, 2e-16)
        sa, sb = self.ssim_map(0, 2, 7, 2e-16, 2e-16)[0], self.ssim_map(1, 2, 7, 2e-16, 2e-16)[0]
        lam = np.divide(va, va + vb, out=np.full_like(va, .5), where=va + vb > 0)
        return float(np.mean(np.where(sab >= .75, lam * sa + (1 - lam) * sb, np.maximum(sa, sb))))

    def qc(self) -> float:
        sa, _, _, _, ca = self.ssim_map(0, 2, 7, 2e-16, 2e-16)
        sb, _, _, _, cb = self.ssim_map(1, 2, 7, 2e-16, 2e-16)
        denominator = ca + cb
        if np.any(denominator == 0):
            raise UndefinedMetric("Cvejic权重协方差之和为零")
        lam = ca / denominator
        return float(np.mean(np.where(lam < 0, 0, np.where(lam > 1, 1, lam * sa + (1 - lam) * sb))))

    def qncie(self) -> float:
        matrix = np.eye(3)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            matrix[i, j] = matrix[j, i] = self.mutual_information(i, j) / (8 * math.log(2))
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min() < -1e-12:
            raise UndefinedMetric("NCIE相关矩阵存在负特征值")
        p = np.maximum(eigenvalues, 0) / 3
        return 1 - _hist_entropy(p) / 8

    def te(self) -> float:
        q, scores = 1.85, []
        for i in (0, 1):
            joint = self.joint(i, 2)
            rows, cols = np.nonzero(joint)
            probability = joint[rows, cols]
            independent = self.hist(i)[rows] * self.hist(2)[cols]
            scores.append(float((np.sum(probability ** q / independent ** (q - 1)) - 1) / (q - 1)))
        return float(sum(scores))

    def uqi(self) -> float:
        values = []
        a, b, f = self.images
        for source in (a, b):
            w = np.ones(8) / 8
            ms, mf = _separable(source, w), _separable(f, w)
            vs = np.maximum(_separable(source * source, w) - ms * ms, 0)
            vf = np.maximum(_separable(f * f, w) - mf * mf, 0)
            cov = _separable(source * f, w) - ms * mf
            lden, cden = ms * ms + mf * mf, vs + vf
            luminance = np.divide(2 * ms * mf, lden, out=np.ones_like(ms), where=lden > 0)
            contrast = np.divide(2 * cov, cden, out=np.ones_like(ms), where=cden > 0)
            values.append(float((luminance * contrast).mean()))
        return float(np.mean(values))


_FUNCTIONS: dict[str, Callable[[_Context], float]] = {
    "EN": _Context.en, "SD": _Context.sd, "SF": _Context.sf, "AG": _Context.ag,
    "EI": _Context.ei, "MI": _Context.mi, "NMI": _Context.nmi, "CE": _Context.ce,
    "CE_overlap": lambda c: c.ce(overlap=True), "MSE": _Context.mse,
    "RMSE": _Context.rmse, "PSNR": _Context.psnr, "CC": _Context.cc,
    "SCD": _Context.scd, "SSIM": _Context.ssim, "MS_SSIM": _Context.ms_ssim,
    "VIF": _Context.vif, "Qabf": _Context.qabf,
    "Nabf": lambda c: c.artifacts()[0], "Labf": lambda c: c.artifacts()[1],
    "FMI_pixel": lambda c: c.fmi("pixel"), "FMI_dct": lambda c: c.fmi("dct"),
    "FMI_w": lambda c: c.fmi("wavelet"), "Qcb": _Context.qcb, "Qcv": _Context.qcv,
    "MEF_SSIM": lambda c: c.mef(False), "MEF_MS_SSIM": lambda c: c.mef(True),
    "Qw": _Context.qw, "Qy": _Context.qy, "Qc": _Context.qc,
    "Qncie": _Context.qncie, "TE": _Context.te, "UQI": _Context.uqi,
}


class FusionEvaluator:
    """单次调用共享计算；实例无图像级可变状态，可在线程之间共享。"""
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(self, ir: object, vis: object, fused: object,
                 metrics: str | Sequence[str] = "all") -> EvaluationResult:
        names = resolve_metrics(metrics)
        started = time.perf_counter()
        arrays = tuple(_gray_array(x, self.config.data_range) for x in (ir, vis, fused))
        if not arrays[0].shape == arrays[1].shape == arrays[2].shape:
            raise ValueError("IR、VIS、融合图尺寸必须相同；不会自动缩放或裁剪")
        ctx = _Context(*arrays, self.config)
        values, issues = {}, {}
        for name in names:
            try:
                with np.errstate(divide="raise", invalid="raise", over="raise"):
                    value = float(_FUNCTIONS[name](ctx))
                if math.isnan(value):
                    raise UndefinedMetric("计算得到NaN")
                values[name] = value
                if math.isinf(value):
                    issues[name] = ("正无穷：两源均方误差为0" if name == "PSNR"
                                    else "正无穷：源图直方图包含融合图中不存在的灰度，KL散度发散")
            except (UndefinedMetric, FloatingPointError) as exc:
                if self.config.strict:
                    raise UndefinedMetric(f"{name}: {exc}") from exc
                values[name] = math.nan
                issues[name] = str(exc)
        return EvaluationResult(values, issues, time.perf_counter() - started, ctx.diagnostics)

    def describe(self) -> dict:
        return {"protocol": PROTOCOL_VERSION, "config": asdict(self.config),
                "metrics": {name: {"中文名称": desc[0], "方向": desc[1], "definition": desc[2]}
                            for name, desc in _DESCRIPTION.items()},
                "conventions": {"gray": "OpenCV float32 BGR2GRAY followed by numpy.rint, matching repository reader",
                    "histogram": "round half-up to 256 integer labels; no automatic contrast stretching",
                    "undefined": "NaN with reason; JSON null; no NaN-to-perfect fallback",
                    "dataset_mean": "per-image arithmetic mean only when no undefined sample; finite_mean separately",
                    "boundaries": "SSIM valid; Qabf zero; EI replicate; Nabf mirror; FMI_w symmetric",
                    "FMI_w": "bundled dmey62 coefficients; NumPy/SciPy implementation, no PyWavelets dependency",
                    "FMI_estimator": "author Fast-FMI real(-p*log2(p)) convention; negative joint terms counted in diagnostics; not silently clipped",
                    "numerics": "float64; nonnegative variances and covariance Cauchy bounds suppress roundoff in flat windows",
                    "scope": "classical fusion metrics; learned IQA/NIQE/BRISQUE and downstream tasks are separate"}}


def read_gray(path: str | Path) -> tuple[np.ndarray, str]:
    raw = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f"图像无法解码或不是8位图像：{path}")
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ValueError(f"仅接受8位灰度或BGR三通道图，拒绝隐式处理alpha：{path}")
        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        image = np.rint(image)
    return np.asarray(image, dtype=np.float64), hashlib.sha256(raw).hexdigest()


def _image_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"图像目录不存在：{directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f"去扩展名或忽略大小写后重名：{result[key]} 与 {path}")
            result[key] = path
    if not result:
        raise ValueError(f"目录内没有图像：{directory}")
    return result


def _pairs(ir: Path, vis: Path, fused: Path, allow_subset: bool) -> list[tuple[Path, Path, Path]]:
    if all(p.is_file() for p in (ir, vis, fused)):
        return [(ir, vis, fused)]
    maps = [_image_map(p) for p in (ir, vis, fused)]
    keys = set(maps[2])
    if not keys <= set(maps[0]) or not keys <= set(maps[1]):
        raise ValueError("融合图缺少对应源图；必须按去扩展名后同名配对")
    if not allow_subset and (keys != set(maps[0]) or keys != set(maps[1])):
        raise ValueError("三目录图像名单不同；仅有意评价子集时使用 --allow-subset")
    return [tuple(m[key] for m in maps) for key in sorted(keys)]


def _summary(results: Sequence[EvaluationResult], names: Sequence[str]) -> dict:
    summary = {}
    for name in names:
        array = np.array([result.values[name] for result in results])
        finite = np.isfinite(array)
        undefined = np.isnan(array)
        mean = math.nan if undefined.any() else float(array.mean())
        summary[name] = {"mean": _json_number(mean),
                         "finite_mean": float(array[finite].mean()) if finite.any() else None,
                         "finite_std": float(array[finite].std()) if finite.any() else None,
                         "total_count": len(array), "finite_count": int(finite.sum()),
                         "undefined_count": int(undefined.sum()), "infinite_count": int(np.isinf(array).sum())}
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ir", type=Path, help="红外图像文件或目录")
    parser.add_argument("--vis", type=Path, help="可见光图像文件或目录")
    parser.add_argument("--fused", type=Path, help="已保存融合图文件或目录")
    parser.add_argument("--metrics", default="all", help="all、core，或逗号分隔指标名")
    parser.add_argument("--output", type=Path, help="可选输出目录；不指定时仅打印JSON，不写文件")
    parser.add_argument("--workers", type=int, default=1, help="按图像并行的CPU线程数，默认1")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="明确允许覆盖本工具已有输出")
    parser.add_argument("--pair-aggregate", choices=("sum", "mean"), default="sum")
    parser.add_argument("--mi-base", type=float, default=2.0)
    parser.add_argument("--sf-denominator", choices=("valid", "hw"), default="valid")
    parser.add_argument("--qabf-power", type=float, default=1.0)
    parser.add_argument("--qabf-angle", choices=("legacy", "periodic"), default="legacy")
    parser.add_argument("--strict", action="store_true", help="遇到未定义指标立即报错，不输出不完整结果")
    parser.add_argument("--describe", action="store_true", help="列出每项定义与协议后退出")
    args = parser.parse_args(argv)
    config = EvaluationConfig(pair_aggregate=args.pair_aggregate, mi_log_base=args.mi_base,
                              sf_denominator=args.sf_denominator, qabf_power=args.qabf_power,
                              qabf_angle=args.qabf_angle, strict=args.strict)
    evaluator = FusionEvaluator(config)
    if args.describe:
        print(json.dumps(evaluator.describe(), ensure_ascii=False, indent=2))
        return 0
    if any(p is None for p in (args.ir, args.vis, args.fused)):
        parser.error("评测需要 --ir、--vis、--fused")
    if args.workers < 1:
        parser.error("--workers 必须为正整数")
    names = resolve_metrics(args.metrics)
    triples = _pairs(args.ir, args.vis, args.fused, args.allow_subset)
    if args.output and not args.overwrite:
        for name in ("metrics.csv", "summary.json"):
            if (args.output / name).exists():
                raise FileExistsError(f"输出已存在；请另选目录或显式传入--overwrite：{args.output / name}")
    started = time.perf_counter()
    def process(paths: tuple[Path, Path, Path]):
        decoded = [read_gray(path) for path in paths]
        result = evaluator.evaluate(*(item[0] for item in decoded), metrics=names)
        manifest = [{"path": str(p.resolve()), "sha256": item[1]} for p, item in zip(paths, decoded)]
        return paths[2].name, result, manifest
    if args.workers == 1:
        records = [process(paths) for paths in triples]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(process, triples))
    elapsed = time.perf_counter() - started
    package_versions = {}
    for package in ("numpy", "scipy", "opencv-python"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unknown"
    report = {"protocol": evaluator.describe(), "metrics_requested": names,
              "summary": _summary([record[1] for record in records], names),
              "image_count": len(records), "elapsed_seconds": elapsed,
              "images_per_second": len(records) / elapsed,
              "runtime": {"python": sys.version, "platform": platform.platform(),
                          "executable": sys.executable, "packages": package_versions,
                          "workers": args.workers,
                          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "images": [{"image": name, "files": manifest, **result.serializable()}
                         for name, result, manifest in records]}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        with (args.output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", *names, "elapsed_seconds", "issues"])
            writer.writeheader()
            for name, result, _ in records:
                writer.writerow({"image": name, **{k: _json_number(v) for k, v in result.values.items()},
                                 "elapsed_seconds": result.elapsed_seconds,
                                 "issues": json.dumps(result.issues, ensure_ascii=False)})
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


# THIRD-PARTY NOTICES (research algorithms and dmey filter coefficients)
__THIRD_PARTY_NOTICES__ = r"""
LIVE VIFp reference notice:

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2005 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
at the University of Texas at Austin (UT Austin, 
http://www.utexas.edu), is acknowledged in any publication that reports research using this code. The research
is to be cited in the bibliography as:

H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality", IEEE Transactions on 
Image Processing, (to appear).

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------

This software release consists of a MULTISCALE PIXEL DOMAIN, SCALAR GSM implementation of the algorithm described in the paper:

MEF-SSIM (research/educational use):
Copyright(c) 2015 Kede Ma, Kai Zeng and Zhou Wang
All Rights Reserved.

----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is hereby
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
----------------------------------------------------------------------

PyWavelets dmey filter coefficients:
Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
Copyright (c) 2012-2020 The PyWavelets Developers <https://github.com/PyWavelets/pywt>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


if __name__ == "__main__":
    raise SystemExit(main())

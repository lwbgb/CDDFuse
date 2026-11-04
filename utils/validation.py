"""独立 RoadScene 验证集：只推理、固定指标、按预定指标保存 best。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from schemas.custom_dataset import CustomImageDataset
from utils import test_metrics
from utils.img_read_save import image_read_cv2
from utils.logger_initializer import logger

if TYPE_CHECKING:
    from models.cddfuse_model import CDDFuseModel

METRICS = ("EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM", "CC", "AG", "MSE")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# 仅包含每次前向重新生成的输入/特征；不触碰参数、梯度、优化器或损失记录。
_FORWARD_TENSORS = (
    "data_IR", "data_VIS", "data_Fuse",
    "feature_VIS_Base", "feature_VIS_Detail", "feature_IR_Base", "feature_IR_Detail",
    "feature_Fuse_Base", "feature_Fuse_Detail",
)


def _clear_forward_tensors(model: CDDFuseModel) -> None:
    for name in _FORWARD_TENSORS:
        if hasattr(model, name):
            setattr(model, name, None)


def _normalize_to_uint8(fused: torch.Tensor) -> np.ndarray:
    """与 test.py 的 normalize_to_uint8 保持相同的图像处理口径。"""
    if fused.ndim != 4 or tuple(fused.shape[:2]) != (1, 1):
        raise ValueError(f'融合输出应为 [1,1,H,W]，实际 {tuple(fused.shape)}')
    if not torch.isfinite(fused).all():
        raise FloatingPointError('融合输出包含 NaN/Inf，停止保存与评测')
    fused = fused.float()
    lower, upper = fused.amin(), fused.amax()
    normalized = (fused - lower) / (upper - lower).clamp_min(1e-8)
    return (normalized * 255.0).to(torch.uint8)[0, 0].cpu().numpy()


def image_names(root: Path, folder: str) -> list[str]:
    directory = root / folder
    if not directory.is_dir():
        raise FileNotFoundError(f"验证所需图像目录不存在：{directory}")
    return sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def image_signature(path: Path) -> str:
    image = image_read_cv2(str(path), mode="GRAY").astype(np.uint8)
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


class RoadSceneValidator:
    def __init__(self, opt: DictConfig, output_dir: Path):
        self.root = Path(opt.validation_dir).resolve()
        self.output_dir = Path(output_dir)
        self.interval = int(opt.get("validation_every", 5))
        self.start_epoch = opt.get("validation_start_epoch")
        if self.start_epoch is not None:
            if isinstance(self.start_epoch, bool) or not isinstance(self.start_epoch, int) or self.start_epoch < 1:
                raise ValueError("validation_start_epoch 必须为正整数或 null")
        self.metric = str(opt.get("validation_metric", "Qabf"))
        if self.interval < 1 or self.metric not in METRICS:
            raise ValueError(f"validation_every 必须为正整数，validation_metric 应为 {METRICS}")
        self.evaluator = test_metrics.FusionEvaluator()
        description = self.evaluator.describe()
        self.selection_mode = "min" if self.metric == "MSE" else "max"
        self.metric_protocol = {
            "implementation": "utils.test_metrics.FusionEvaluator",
            "source_sha256": hashlib.sha256(Path(test_metrics.__file__).read_bytes()).hexdigest(),
            "protocol": description["protocol"], "config": description["config"],
            "metrics": list(METRICS),
            "postprocess": "test_float32_minmax_clamp1e-8_truncate_uint8_v1",
        }
        names = image_names(self.root, "ir")
        if not names or names != image_names(self.root, "vi"):
            raise ValueError("验证集 ir/vi 必须按相同文件名逐对匹配，且不能为空")
        expected = int(opt.get("validation_expected_pairs", 50))
        if expected < 1 or len(names) != expected:
            raise ValueError(f"验证集应有 {expected} 对，实际 {len(names)} 对")
        test_root = Path(opt.get("validation_test_dir", "test_img/RoadScene")).resolve()
        test_names = image_names(test_root, "ir")
        if not test_names or test_names != image_names(test_root, "vi"):
            raise ValueError("隔离检查需要有效且配对完整的 RoadScene 测试目录")
        if self.root == test_root or set(names) & set(test_names):
            raise ValueError("验证集与测试集目录或文件名重叠，请提供独立验证集")
        signatures = {folder: [image_signature(self.root / folder / name) for name in names]
                      for folder in ("ir", "vi")}
        for folder in ("ir", "vi"):
            test_signatures = {image_signature(test_root / folder / name) for name in test_names}
            if set(signatures[folder]) & test_signatures:
                raise ValueError("验证集与测试集图像内容重叠（即使文件已重命名）")
        identity = {"names": names, "image_signatures": signatures}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        dataset = CustomImageDataset(self.root, "ir", "vi")
        # 仅使用已核验的图像名单；排除 .gitkeep、说明文件等非图像。
        dataset._items = [(name, name) for name in names]
        self.loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                 generator=torch.Generator().manual_seed(0))
        self.best: dict | None = None
        self.best_path = self.output_dir / "best_validation.json"
        if opt.get("continue_train", False) and self.best_path.is_file():
            self.best = json.loads(self.best_path.read_text(encoding="utf-8"))
            if self.best["fingerprint"] != self.fingerprint or self.best["selection_metric"] != self.metric:
                raise ValueError("已有 best 记录的验证集或选择指标不同，请使用新的实验目录")
            if (self.best.get("metric_protocol") != self.metric_protocol
                    or self.best.get("selection_mode") != self.selection_mode):
                raise ValueError("已有 best 记录的指标口径不同或缺少协议，不能与新指标混比，请使用新的实验目录")
        self.manifest = {"root": str(self.root), "fingerprint": self.fingerprint,
                         "count": len(names), "names": names,
                         "selection_metric": self.metric, "selection_mode": self.selection_mode,
                         "metric_protocol": self.metric_protocol,
                         "validation_start_epoch": self.start_epoch, "validation_every": self.interval}

    def should_validate(self, model: CDDFuseModel) -> bool:
        epoch = model.get_epoch()
        if model.get_phase() != 2:
            return False
        if self.start_epoch is not None:
            first_epoch = max(self.start_epoch, int(model.opt.epoch_gap) + 1)
            return epoch >= first_epoch and ((epoch - first_epoch) % self.interval == 0
                                            or epoch == int(model.opt.n_epochs))
        # 未指定起始轮数时，保留原有阶段首轮、阶段内间隔和末轮验证逻辑。
        phase_epoch = epoch - int(model.opt.epoch_gap)
        return phase_epoch == 1 or phase_epoch % self.interval == 0 or epoch == int(model.opt.n_epochs)

    def _evaluate_image(self, model: CDDFuseModel, data: tuple | list) -> np.ndarray:
        """GPU张量仅在单张前向期间存在；返回的指标数组位于CPU。"""
        try:
            model.set_input(data)
            model.forward()
            # 尽早转换为CPU数组，后续指标计算不持有GPU输出局部引用。
            fused = _normalize_to_uint8(model.data_Fuse)
            ir = torch.round(model.data_IR.squeeze() * 255).cpu().numpy()
            vi = torch.round(model.data_VIS.squeeze() * 255).cpu().numpy()
            result = self.evaluator.evaluate(ir, vi, fused, metrics=METRICS)
            values = np.asarray([result.values[name] for name in METRICS], dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"验证指标未定义或非有限：{data[2]}；{result.issues}，不保存 best checkpoint")
            return values
        finally:
            _clear_forward_tensors(model)

    def evaluate(self, model: CDDFuseModel) -> dict[str, float]:
        if model.get_phase() != 2:
            raise ValueError("融合验证仅在第二阶段执行")
        was_training = model.isTrain
        states = [(module, module.training) for net in model.models.values() for module in net.modules()]
        total = np.zeros(len(METRICS), dtype=np.float64)
        try:
            # 前一训练batch的前向结果不再参与优化；释放后再开始整图验证。
            _clear_forward_tensors(model)
            model.isTrain = False  # FP32推理；不启用训练BF16 autocast。
            model.eval()
            with torch.inference_mode():
                for data in self.loader:
                    total += self._evaluate_image(model, data)
        finally:
            _clear_forward_tensors(model)
            model.isTrain = was_training
            for module, training in states:
                module.training = training
            # 每轮验证边界仅调用一次，避免逐图清缓存反复触发CUDA分配。
            if model.device.type == "cuda":
                with torch.cuda.device(model.device):
                    torch.cuda.empty_cache()
        values = total / len(self.loader.dataset)
        if not np.isfinite(values).all():
            raise FloatingPointError("验证指标出现 NaN/Inf，不保存 best checkpoint")
        return dict(zip(METRICS, values.tolist()))

    def run(self, model: CDDFuseModel) -> dict:
        metrics = self.evaluate(model)
        score = metrics[self.metric]
        improved = self.best is None or (score < self.best["score"] if self.selection_mode == "min"
                                         else score > self.best["score"])
        record = {"epoch": model.get_epoch(), "pairs": len(self.loader.dataset),
                  "metrics": metrics, "selection_metric": self.metric, "score": score,
                  "fingerprint": self.fingerprint, "improved": improved,
                  "selection_mode": self.selection_mode, "metric_protocol": self.metric_protocol}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "validation_manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if improved:
            checkpoint_name = f"{model.opt.model}_best_RoadScene.pth"
            model.save_model(checkpoint_name)
            self.best = {**record, "checkpoint": checkpoint_name}
            temporary = self.best_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.best, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.best_path)
        with (self.output_dir / "validation_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        logger.info(f"RoadScene validation epoch={model.get_epoch()}, {self.metric}={score:.6f}, "
                    f"mode={self.selection_mode}, best={improved}; "
                    f"evaluator={self.metric_protocol['implementation']}; "
                    + ", ".join(f"{name}={metrics[name]:.6f}" for name in METRICS))
        return record

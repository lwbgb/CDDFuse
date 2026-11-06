"""通用独立验证：按数据集验证，并汇总保存结果。"""
from __future__ import annotations

import hashlib
import json
import re
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


class Validator:
    def __init__(self, opt: DictConfig, dataset_name: str, metric: str):
        if not isinstance(dataset_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", dataset_name):
            raise ValueError(f"非法验证数据集名称：{dataset_name!r}")
        self.dataset_name = dataset_name
        self.root = (Path("datasets") / f"{dataset_name}_valid").resolve()
        self.interval = int(opt.get("validation_every", 5))
        self.start_epoch = opt.get("validation_start_epoch")
        if self.start_epoch is not None:
            if isinstance(self.start_epoch, bool) or not isinstance(self.start_epoch, int) or self.start_epoch < 1:
                raise ValueError("validation_start_epoch 必须为正整数或 null")
        self.metric = str(metric)
        if self.interval < 1 or self.metric not in METRICS:
            raise ValueError(f"validation_every 必须为正整数，metric 应为 {METRICS}")
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
        test_root = (Path("test_img") / dataset_name).resolve()
        test_names = image_names(test_root, "ir")
        if not test_names or test_names != image_names(test_root, "vi"):
            raise ValueError("隔离检查需要有效且配对完整的测试目录")
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
            model.isTrain = False  # 切换验证模式，前向精度由 validation_use_bf16 控制。
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

    def run(self, model: CDDFuseModel, previous_best: dict | None) -> tuple[dict, dict | None]:
        metrics = self.evaluate(model)
        score = metrics[self.metric]
        improved = previous_best is None or (score < previous_best["score"] if self.selection_mode == "min"
                                            else score > previous_best["score"])
        record = {"epoch": model.get_epoch(), "pairs": len(self.loader.dataset),
                  "metrics": metrics, "selection_metric": self.metric, "score": score,
                  "fingerprint": self.fingerprint, "improved": improved,
                  "selection_mode": self.selection_mode, "metric_protocol": self.metric_protocol}
        best = previous_best
        if improved:
            checkpoint_name = f"{model.opt.model}_best_{self.dataset_name}.pth"
            best = {**record, "checkpoint": checkpoint_name}
        logger.info(f"{self.dataset_name} validation epoch={model.get_epoch()}, {self.metric}={score:.6f}, "
                    f"mode={self.selection_mode}, best={improved}; "
                    f"evaluator={self.metric_protocol['implementation']}; "
                    + ", ".join(f"{name}={metrics[name]:.6f}" for name in METRICS))
        return record, best


class ValidationManager:
    """统一写入各数据集结果，兼容旧版单 RoadScene 的 best 与 manifest。"""
    def __init__(self, opt: DictConfig, output_dir: Path, validators: list[Validator]):
        self.output_dir = Path(output_dir)
        self.validators = validators
        names = [v.dataset_name for v in validators]
        if len(set(names)) != len(names):
            raise ValueError("验证数据集不能重复")
        self.best_path = self.output_dir / "best_validation.json"
        self.manifest_path = self.output_dir / "validation_manifest.json"
        continuing = bool(opt.get("continue_train", False))
        if not continuing and (self.best_path.exists() or self.manifest_path.exists() or
                               (self.output_dir / "validation_history.jsonl").exists()):
            raise ValueError("实验目录已有验证记录，请使用新的实验目录，避免覆盖")
        self.best = self._read_records(self.best_path) if continuing else {}
        self.manifests = self._read_records(self.manifest_path) if continuing else {}
        for validator in validators:
            old = self.best.get(validator.dataset_name)
            if old is not None:
                if old.get("fingerprint") != validator.fingerprint or old.get("selection_metric") != validator.metric:
                    raise ValueError(f"{validator.dataset_name} 已有 best 的验证集或选择指标不同，请使用新的实验目录")
                if old.get("metric_protocol") != validator.metric_protocol or old.get("selection_mode") != validator.selection_mode:
                    raise ValueError(f"{validator.dataset_name} 已有 best 指标口径不同，不能混比，请使用新的实验目录")

    @staticmethod
    def _read_records(path: Path) -> dict:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "datasets" in data:
            if not isinstance(data["datasets"], dict):
                raise ValueError(f"验证记录格式错误：{path}")
            return data["datasets"]
        if isinstance(data, dict) and "fingerprint" in data:
            return {"RoadScene": data}
        raise ValueError(f"未知验证记录格式：{path}")

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def run(self, model: CDDFuseModel) -> dict:
        records = {}
        pending_best = dict(self.best)
        pending_manifests = dict(self.manifests)
        for validator in self.validators:
            name = validator.dataset_name
            record, best = validator.run(model, self.best.get(name))
            records[name] = record
            pending_best[name] = best
            pending_manifests[name] = validator.manifest
        # 所有数据集成功评估后，再写入任何 best，避免后一个数据集失败时留下单边结果。
        for name, record in records.items():
            if record["improved"]:
                model.save_model(pending_best[name]["checkpoint"])
        self.best = pending_best
        self.manifests = pending_manifests
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.manifest_path, {"datasets": self.manifests})
        self._write_json(self.best_path, {"datasets": self.best})
        record = {"epoch": model.get_epoch(), "datasets": records}
        # 暂时不在 history 中记录校验元数据；best 和 manifest 仍保留以检查续训兼容性。
        history_record = {"epoch": record["epoch"], "datasets": {
            name: {key: value for key, value in data.items() if key not in ("fingerprint", "metric_protocol")}
            for name, data in records.items()
        }}
        with (self.output_dir / "validation_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history_record, allow_nan=False) + "\n")
        return record

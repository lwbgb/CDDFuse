import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
                

def check_file_path(path: str | Path, create: bool = False) -> bool:
    path = Path(path)
    
    # 1. 如果路径已存在，但它是一个目录而非文件，说明路径无效（被同名文件夹占用）
    if path.exists() and not path.is_file():
        raise ValueError(f"目标路径已被目录占用，无法作为文件使用：{path}")
        
    parent = path.parent
    
    # 2. 检查父目录状态
    if parent.exists():
        if not parent.is_dir():
            raise NotADirectoryError(f"路径的父级已存在，但它是一个文件而非目录：{parent}")
        return True
    else:
        # 3. 父目录不存在时，根据 create 参数决定行为
        if create:
            parent.mkdir(parents=True, exist_ok=True)
            return True
        else:
            raise FileNotFoundError(f"父目录不存在且未授权创建：{parent}")
    


def save_epoch_checkpoint(
    checkpoint_path: str | Path,
    phase: int,
    epoch: int,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    config: dict | None = None,
) -> None:
    
    checkpoint_path: Path = Path(checkpoint_path)
    check_file_path(checkpoint_path, create=True)
        
    checkpoint = {
        "phase": phase,
        "epoch": epoch,
        "models": {name: unwrap_model(model).state_dict() for name, model in models.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
        "config": config or {},
    }

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_epoch_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    expected_phase: int | None = None,
    strict: bool = True,
) -> dict:
    checkpoint_file = Path(checkpoint_path)
    check_file_path(checkpoint_file, create=False)

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)

    required_keys = ["epoch", "phase", "models", "optimizers", "schedulers"]
    if not all(k in checkpoint for k in required_keys):
        raise KeyError("Checkpoint 文件结构不完整，缺少必要的键，无法恢复训练状态。")

    checkpoint_phase = int(checkpoint["phase"])
    if checkpoint_phase not in {1, 2}:
        raise ValueError(f"Checkpoint 中的 phase 无效：{checkpoint_phase}")

    if expected_phase is not None and checkpoint_phase != expected_phase:
        raise ValueError(f"Checkpoint phase 不匹配，期望 {expected_phase}，" f"实际为 {checkpoint_phase}")

    required_names = {
        1: {
            "models": {
                "DIDF_Encoder",
                "DIDF_Decoder",
            },
            "optimizers": {
                "optimizer_encoder",
                "optimizer_decoder",
            },
            "schedulers": {
                "scheduler_encoder",
                "scheduler_decoder",
            },
        },
        2: {
            "models": {
                "DIDF_Encoder",
                "DIDF_Decoder",
                "BaseFuseLayer",
                "DetailFuseLayer",
            },
            "optimizers": {
                "optimizer_encoder",
                "optimizer_decoder",
                "optimizer_base_fusion",
                "optimizer_detail_fusion",
            },
            "schedulers": {
                "scheduler_encoder",
                "scheduler_decoder",
                "scheduler_base_fusion",
                "scheduler_detail_fusion",
            },
        },
    }

    phase_requirements = required_names[checkpoint_phase]

    for category in ("models", "optimizers", "schedulers"):
        missing_in_checkpoint = phase_requirements[category] - checkpoint[category].keys()
        missing_in_runtime = phase_requirements[category] - locals()[category].keys()

        if missing_in_checkpoint:
            raise KeyError(f"Checkpoint 的 {category} 缺少：" f"{sorted(missing_in_checkpoint)}")

        if missing_in_runtime:
            raise KeyError(f"当前程序的 {category} 缺少：" f"{sorted(missing_in_runtime)}")

    for name in phase_requirements["models"]:
        unwrap_model(models[name]).load_state_dict(
            checkpoint["models"][name],
            strict=strict,
        )

    for name in phase_requirements["optimizers"]:
        optimizers[name].load_state_dict(checkpoint["optimizers"][name])
        move_optimizer_to_device(optimizers[name], device)

    for name in phase_requirements["schedulers"]:
        schedulers[name].load_state_dict(checkpoint["schedulers"][name])

    return checkpoint


def load_models(checkpoint_path: str, device: torch.device, models: dict[str, nn.Module]) -> bool:
    checkpoint_file = Path(checkpoint_path)
    check_file_path(checkpoint_file, create=False)

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)

    if "models" not in checkpoint:
        raise KeyError("Checkpoint 文件中缺少 'models' 键，无法加载模型参数。")

    for name, model in models.items():
        if name in checkpoint["models"]:
            unwrap_model(model).load_state_dict(checkpoint["models"][name], strict=True)
        else:
            raise KeyError(f"Checkpoint 中缺少模型 '{name}' 的参数。")

    return True
    
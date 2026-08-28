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


def save_epoch_checkpoint(
    checkpoint_path: str,
    phase: int,
    epoch: int,
    global_step: int,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, Any],
    config: dict | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "phase": phase,
        "epoch": epoch,
        "global_step": global_step,
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

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint 文件不存在：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    required_fields = {
        "phase",
        "epoch",
        "global_step",
        "models",
        "optimizers",
        "schedulers",
    }

    missing_fields = required_fields.difference(checkpoint.keys())

    if missing_fields:
        raise KeyError(f"Checkpoint 缺少必要字段：{sorted(missing_fields)}")

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

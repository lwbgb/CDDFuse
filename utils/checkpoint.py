import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def get_grad_state(module: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    grad_state = {}

    for name, parameter in unwrap_model(module).named_parameters():
        grad_state[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()

    return grad_state


def load_grad_state(module: nn.Module, grad_state: Optional[Dict[str, Optional[torch.Tensor]]]) -> None:
    if not grad_state:
        return

    parameters = dict(unwrap_model(module).named_parameters())

    for name, gradient in grad_state.items():
        if name not in parameters:
            continue

        parameter = parameters[name]

        if gradient is None:
            parameter.grad = None
        else:
            parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype).clone()


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    checkpoint_path: str,
    phase: int,
    epoch: int,
    batch_index: int,
    global_step: int,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    criteria: Dict[str, nn.Module],
    latest_losses: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    target_path = Path(checkpoint_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "phase": phase,
        "epoch": epoch,
        "batch_index": batch_index,
        "global_step": global_step,
        "models": {
            name: unwrap_model(model).state_dict()
            for name, model in models.items()
        },
        "model_gradients": {
            name: get_grad_state(model)
            for name, model in models.items()
        },
        "optimizers": {
            name: optimizer.state_dict()
            for name, optimizer in optimizers.items()
        },
        "schedulers": {
            name: scheduler.state_dict()
            for name, scheduler in schedulers.items()
        },
        "criteria": {
            name: criterion.state_dict()
            for name, criterion in criteria.items()
        },
        "criterion_gradients": {
            name: get_grad_state(criterion)
            for name, criterion in criteria.items()
        },
        "latest_losses": latest_losses or {},
        "config": config or {},
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, target_path)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    criteria: Dict[str, nn.Module],
    restore_gradients: bool = True,
    restore_random_state: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    for name, model in models.items():
        if name not in checkpoint["models"]:
            raise KeyError(f"Checkpoint 中不存在模型状态：{name}")

        unwrap_model(model).load_state_dict(checkpoint["models"][name], strict=strict)

    for name, optimizer in optimizers.items():
        if name not in checkpoint["optimizers"]:
            raise KeyError(f"Checkpoint 中不存在优化器状态：{name}")

        optimizer.load_state_dict(checkpoint["optimizers"][name])
        move_optimizer_to_device(optimizer, device)

    for name, scheduler in schedulers.items():
        if name not in checkpoint["schedulers"]:
            raise KeyError(f"Checkpoint 中不存在调度器状态：{name}")

        scheduler.load_state_dict(checkpoint["schedulers"][name])

    for name, criterion in criteria.items():
        criterion_state = checkpoint.get("criteria", {}).get(name)

        if criterion_state is not None:
            criterion.load_state_dict(criterion_state, strict=strict)

    if restore_gradients:
        for name, model in models.items():
            load_grad_state(model, checkpoint.get("model_gradients", {}).get(name))

        for name, criterion in criteria.items():
            load_grad_state(criterion, checkpoint.get("criterion_gradients", {}).get(name))

    if restore_random_state:
        random_state = checkpoint.get("random_state", {})

        if random_state.get("python") is not None:
            random.setstate(random_state["python"])

        if random_state.get("numpy") is not None:
            np.random.set_state(random_state["numpy"])

        if random_state.get("torch_cpu") is not None:
            torch.set_rng_state(random_state["torch_cpu"].cpu())

        if torch.cuda.is_available() and random_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(random_state["torch_cuda"])

    return checkpoint

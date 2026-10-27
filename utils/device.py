import os
import random

import numpy as np
from omegaconf import DictConfig
import torch
import torch.distributed as dist


def init_ddp():
    # Initialize DDP if LOCAL_RANK is set
    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Initialized with device {device}")
    return device


def init_cuda(opt: DictConfig):
    """Initialize CUDA settings based on the provided configuration."""
    torch.backends.cudnn.allow_tf32 = bool(opt.get("cudnn_allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(opt.get("matmul_allow_tf32", False))
    torch.backends.cudnn.benchmark = bool(opt.get("cudnn_benchmark", True))
    

def set_seed(seed: int = 1696) -> None:
    """Set random seed for reproducibility across various libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
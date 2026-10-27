from omegaconf import DictConfig
import torch
from torch import nn
import os
from torch.optim import lr_scheduler
from torch.nn import init
import torch.distributed as dist

class PhaseStepLR(lr_scheduler.StepLR):
    """阶段内独立计数的 StepLR，阶段参数和当前阶段随 state_dict 保存。"""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 phase_settings: dict[int, dict[str, float | int]], phase: int):
        self.phase_settings = phase_settings
        self.training_phase = phase
        settings = phase_settings[phase]
        super().__init__(optimizer, step_size=settings["step"], gamma=settings["gamma"])

    def set_phase(self, phase: int) -> None:
        if phase == self.training_phase:
            return
        if phase not in self.phase_settings:
            raise ValueError(f"未知训练阶段：{phase}")
        self.training_phase = phase
        # 保留当前 lr；新阶段以当前值为基准，从 0 次调度开始。
        for group in self.optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        settings = self.phase_settings[phase]
        super().__init__(self.optimizer, step_size=settings["step"], gamma=settings["gamma"])


def phase_step_settings(opt: DictConfig) -> dict[int, dict[str, float | int]] | None:
    fields = ("phase1_optim_step", "phase1_optim_gamma", "phase2_optim_step", "phase2_optim_gamma")
    if not any(opt.get(key) is not None for key in fields):
        return None
    settings = {}
    for phase in (1, 2):
        step = opt.get(f"phase{phase}_optim_step")
        gamma = opt.get(f"phase{phase}_optim_gamma")
        step = opt.optim_step if step is None else step
        gamma = opt.optim_gamma if gamma is None else gamma
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"phase{phase}_optim_step 必须为正整数")
        if not 0 < float(gamma) <= 1:
            raise ValueError(f"phase{phase}_optim_gamma 必须在 (0, 1] 内")
        settings[phase] = {"step": step, "gamma": float(gamma)}
    return settings


def get_scheduler(optimizer, opt: DictConfig, phase: int | None = None):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.get("epoch_count", 1) - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        settings = phase_step_settings(opt)
        if settings is None:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.optim_step, gamma=opt.optim_gamma)
        else:
            if phase is None:
                phase = 1 if opt.get("start_epoch", 1) <= opt.get("epoch_gap", 40) else 2
            scheduler = PhaseStepLR(optimizer, settings, phase)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_net(net: nn.Module, opt, init_type: str = "normal", init_gain: float = 0.02) -> nn.Module:
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        opt (option class) -- stores all the experiment flags
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # net = net.to(device)
    # 多显卡训练
    if dist.is_available() and dist.is_initialized():
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device.index])
        # Sync all processes after DDP wrapping
        dist.barrier()
    # init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

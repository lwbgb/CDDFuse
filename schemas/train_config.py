from dataclasses import dataclass


@dataclass
class TrainConfig:
    model: str = "CDDFuse"
    n_epochs: int = 120
    epoch_gap: int = 40
    start_epoch: int = 1
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0
    print_epoch_freq: int = 1
    print_iter_freq: int = 10000
    display_freq: int = 100
    save_latest_freq: int = 1
    save_epoch_freq: int = 5

    coeff_mse_loss_VF: float = 1.0
    coeff_mse_loss_IF: float = 1.0
    coeff_decomp: float = 2.0
    coeff_tv: float = 5.0
    SSIM_window_size: int = 11

    clip_grad_norm_value: float = 0.01
    optim_step: int = 20
    optim_gamma: float = 0.5

    device: str = "cpu"

    batch_size: int = 4
    shuffle: bool = True
    num_threads: int = 4
    drop_last: bool = False

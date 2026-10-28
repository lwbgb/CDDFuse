from dataclasses import dataclass, field

from schemas.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    n_epochs: int = 120
    epoch_gap: int = 40
    start_epoch: int = 1
    n_epochs_decay: int = 20
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0
    print_epoch_freq: int = 1
    print_iter_freq: int = 1
    display_freq: int = 100
    save_latest_freq: int = 1
    save_epoch_freq: int = 5
    ckp_name: str = "CDDFuse"
    load_dir: str = ""
    continue_train: bool = False
    reset_lr_on_resume: bool = False
    resume_lrs: dict[str, float] = field(default_factory=dict)
    norm: str = "layer"

    coeff_mse_loss_VF: float = 1.0
    coeff_mse_loss_IF: float = 1.0
    coeff_decomp: float = 2.0
    coeff_tv: float = 5.0
    SSIM_window_size: int = 11

    clip_grad_norm_value: float = 0.01
    lr_policy: str = "step"
    optim_step: int = 20
    optim_gamma: float = 0.5
    phase1_optim_step: int | None = None
    phase1_optim_gamma: float | None = None
    phase2_optim_step: int | None = None
    phase2_optim_gamma: float | None = None
    reset_lr_on_resume: bool = False
    resume_lrs: dict[str, float] = field(default_factory=dict)

    device: str = "cpu"
    use_bf16: bool = False
    cudnn_allow_tf32: bool = False
    matmul_allow_tf32: bool = False
    cudnn_benchmark: bool = False
    seed: int = 1696

    batch_size: int = 8
    shuffle: bool = True
    num_threads: int = 8
    drop_last: bool = False
    dataset_root: str = "data/"
    dataset_mode: str = "h5"
    h5_file_name: str = "MSRS_train_imgsize_128_stride_200.h5"
    dataset_dir: str = "datasets/MSRS_train"
    crop_size: int = 128
    samples_per_epoch: int = 6487
    augment: bool = False
    filter_low_contrast: bool = True
    max_crop_attempts: int = 20

    validation_enabled: bool = False
    validation_dir: str = "datasets/RoadScene"
    validation_test_dir: str = "test_img/RoadScene"
    validation_expected_pairs: int = 50
    validation_start_epoch: int | None = None
    validation_every: int = 5
    validation_metric: str = "Qabf"

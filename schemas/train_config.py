
from dataclasses import dataclass


@dataclass
class TrainConfig:
	model: str = "CDDFuse"
	n_epochs: int = 120
	epoch_gap: int = 40
	lr: float = 1e-4
	weight_decay: float = 0

	coeff_mse_loss_VF: float = 1.0
	coeff_mse_loss_IF: float = 1.0
	coeff_decomp: float = 2.0
	coeff_tv: float = 5.0

	clip_grad_norm_value: float = 0.01
	optim_step: int = 20
	optim_gamma: float = 0.5

	device: str = "cpu"

	batch_size: int = 4
	shuffle: bool = True
	num_threads: int = 4
	drop_last: bool = False
  
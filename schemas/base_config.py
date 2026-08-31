from dataclasses import dataclass

from omegaconf import DictConfig

@dataclass
class BaseConfig(DictConfig):

    isTrain: bool = True
    continue_train: bool = False
    verbose: bool = True
    checkpoint_root: str = r"checkpoints/"
    ...

from dataclasses import dataclass

from omegaconf import DictConfig

@dataclass
class BaseConfig(DictConfig):

    model: str = "CDDFuse"
    verbose: bool = True
    checkpoint_root: str = r"checkpoints/"
    ...

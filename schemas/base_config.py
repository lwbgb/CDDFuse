from dataclasses import dataclass

from omegaconf import DictConfig

@dataclass
class BaseConfig(DictConfig):
    
    ...
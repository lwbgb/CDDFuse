from dataclasses import dataclass

from schemas.base_config import BaseConfig

@dataclass
class DatasetConfig(BaseConfig):
    root: str = "datasets/"
    name: str = "MSRS_train"
    crop_size: int = 128
    stride: int = 200
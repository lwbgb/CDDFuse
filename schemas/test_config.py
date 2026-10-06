from dataclasses import dataclass

from schemas.base_config import BaseConfig


@dataclass
class TestConfig(BaseConfig):
    device: str = "cpu"
    isTrain: bool = False
    num_threads: int = 4
    batch_size: int = 1
    shuffle: bool = False
    no_flip: bool = True
    drop_last: bool = False
    num_test: int = 500

    root: str = "test_img/"
    output_dir: str = "test_results/"
    load_dir: str = "base_model"
    ckp_name: str = "CDDFuse_phase2_epoch120.pth"
    ...
from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.dataset import H5Dataset, PairedRandomCropDataset, get_loader
from utils.logger_initializer import init_custom_resolvers, init_logger
from utils.logger_initializer import logger

__all__ = ["init_logger", "init_custom_resolvers", "create_dataset"]

init_logger()
init_custom_resolvers()

def create_dataset(opt: DictConfig, h5_file_name: str) -> DataLoader:
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    mode = opt.get("dataset_mode", "h5")
    if mode == "h5":
        h5_file_path = Path(opt.dataset_root) / h5_file_name
        dataset = H5Dataset(h5_file_path)
    elif mode == "random_crop":
        dataset = PairedRandomCropDataset(
        image_root=opt.dataset_dir,
        ir_dir=opt.get("ir_dir", "ir"),
        vis_dir=opt.get("vis_dir", "vi"),
        crop_size=opt.get("crop_size", 128),
        samples_per_epoch=opt.get("samples_per_epoch", 6000),
        augment=opt.get("augment", False),
        filter_low_contrast=opt.get("filter_low_contrast", True),
        max_crop_attempts=opt.get("max_crop_attempts", 20),
        seed=opt.get("seed", 3407),
    )
    else:
        raise ValueError(f"未知 dataset_mode={mode!r}，应为 h5 或 random_crop")
    if len(dataset) == 0:
        raise ValueError("训练数据集为空")
    loader = get_loader(opt, dataset)
    if len(loader) == 0:
        raise ValueError("DataLoader 没有可用 batch，请检查 batch_size 和 drop_last")
    logger.info(f"Dataset mode={mode}, samples={len(dataset)}, batches={len(loader)}")
    return loader

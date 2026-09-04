from pathlib import Path

from omegaconf import DictConfig

from utils.dataset import H5Dataset, get_loader
from utils.logger_initializer import init_custom_resolvers, init_logger
from utils.logger_initializer import logger

__all__ = ["init_logger", "init_custom_resolvers", "create_dataset"]

init_logger()
init_custom_resolvers()

def create_dataset(opt: DictConfig, h5_file_name: str):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    h5_file_path = Path(opt.dataset_root) / h5_file_name
    try:
        dataset = H5Dataset(h5_file_path)
    except Exception as e:
        logger.error(f"Failed to create dataset from {h5_file_path}: {e}")
        raise RuntimeError(f"Failed to create dataset from {h5_file_path}: {e}")
    data_loader = get_loader(opt, dataset)
    return data_loader

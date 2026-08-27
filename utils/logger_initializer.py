from pathlib import Path
import sys
from time import localtime, strftime

import hydra
from omegaconf import DictConfig
from loguru import logger
from hydra import initialize, compose


def init_logger(file_name: str | None = None):
    with initialize(version_base=None, config_path="../configs"):
        opt: DictConfig = compose(config_name="loguru")
    logger.remove()
    # 控制台输出
    logger.add(sys.stderr, level='DEBUG', format=opt.format, colorize=True, backtrace=opt.backtrace, 
               diagnose=opt.diagnose, enqueue=True)
    # 输出到文件
    try:
        root = Path(opt.root)
        parent = root.parent
        if not parent.exists():
            raise OSError(f"path:{parent} not exist!")
        else:
            if not parent.is_dir():
                raise NotADirectoryError(f"path:{parent} is not a dictionary!")
        file_name = file_name if file_name else strftime("%Y%m%d", localtime()) + ".log"
        file_path = root / opt.prefix / file_name if opt.prefix else root / file_name
    except:
        raise IOError("Logger 文件路径配置出错！")
    logger.add(file_path, level='INFO', format=opt.format, colorize=False, backtrace=opt.backtrace,
               diagnose=opt.diagnose, rotation=opt.rotation, retention=opt.retention, encoding=opt.encoding, enqueue=True)
from pathlib import Path
import sys
from time import localtime, strftime
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from hydra import initialize, compose


def init_logger(root: Path | str | None = None, file_name: str | None = None) -> int:
    """初始化日志记录器，包括控制台输出和文件输出

    Args:
        root (Path | str | None, optional): 日志文件根目录. Defaults to None.
        file_name (str | None, optional): 日志文件名. Defaults to None.、
        
    Returns:
        int: 返回日志记录器的句柄，用于后续操作，如关闭或删除日志记录器
    """
    with initialize(version_base=None, config_path="../configs"):
        opt: DictConfig = compose(config_name="loguru")
    logger.remove()
    # 控制台输出
    logger.add(sys.stderr, level='DEBUG', format=opt.format, colorize=True, backtrace=opt.backtrace, 
               diagnose=opt.diagnose, enqueue=True)
    # 输出到文件
    try:
        root = Path(root) if root else Path(opt.root)
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
    return logger.add(file_path, level='INFO', format=opt.format, colorize=False, backtrace=opt.backtrace,
               diagnose=opt.diagnose, rotation=opt.rotation, retention=opt.retention, encoding=opt.encoding, enqueue=True)
    

def init_custom_resolvers() -> list[str]:
    
    resolvers: list[str] = []
    resolvers.append("current_date")
    OmegaConf.register_new_resolver("current_date", lambda fmt="%Y%m%d": time.strftime(fmt, time.localtime()))
    
    return resolvers
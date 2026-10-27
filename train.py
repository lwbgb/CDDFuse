import json
import math
from pathlib import Path
import random
import sys
import time

from hydra import compose, initialize
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.cddfuse_model import CDDFuseModel
from schemas.base_config import BaseConfig
from schemas.train_config import TrainConfig
from utils import create_dataset, path_util
from utils.validation import RoadSceneValidator
from utils.device import cleanup_ddp, set_seed
from utils.logger_initializer import init_logger, logger


def train(opt: TrainConfig) -> Path:
    set_seed(int(opt.get('seed', 1696)))
    
    loader: DataLoader = create_dataset(opt, str(opt.h5_file_name))
    dataset_size = len(loader.dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")
    
    model = CDDFuseModel(opt)
    model.setup()
    validator = RoadSceneValidator(opt, model.save_dir) if opt.get('validation_enabled', False) else None
    
    path_util.check_file_path(model.save_dir / 'config.yaml', create=True)
    OmegaConf.save(opt, model.save_dir / 'config.yaml')
    sink = init_logger(model.save_dir, file_name='train.log')
    try:
        mode = 'BF16' if model.use_bf16 else 'FP32'
        logger.info(f'Starting {opt.model}: epoch {opt.start_epoch}/{opt.n_epochs}, batch={opt.batch_size} pairs, '
                    f'seed={opt.get("seed", 3407)}, precision={mode}, device={model.device}')
        logger.info(f'TF32 cuDNN/matmul={torch.backends.cudnn.allow_tf32}/{torch.backends.cuda.matmul.allow_tf32}, '
                    f'benchmark={torch.backends.cudnn.benchmark}, deterministic={torch.are_deterministic_algorithms_enabled()}; '
                    f'dataset={getattr(loader.dataset, "h5file_path", opt.get("dataset_dir"))}, output={model.save_dir}')
        
        
        epoch_iters = (dataset_size // opt.batch_size) if opt.drop_last else math.ceil(dataset_size / opt.batch_size)
        total_iters = (opt.start_epoch - 1) * epoch_iters  # the total number of training iterations
        total_epochs = opt.start_epoch - 1  # the total number of training epochs
        
        for epoch in range(int(opt.start_epoch), int(opt.n_epochs) + 1):
            if hasattr(loader.dataset, 'set_epoch'):
                loader.dataset.set_epoch(epoch)
            if hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)
            
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iters = 0  # the number of training iterations in current epoch, reset to 0
            epoch_samples = 0
            epoch_loss = 0.0
            components = {}
            
            model.update_state(epoch)
            learning_rates = {name: optimizer.param_groups[0]['lr'] for name, optimizer in model.optimizers.items()}
            
            with tqdm(loader, desc=f'[Phase {model.get_phase()}] [Epoch {epoch}/{opt.n_epochs}]',
                      dynamic_ncols=True, leave=False) as pbar:
                for idx, data in enumerate(pbar):
                    iter_start_time = time.time()  # timer for computation per iteration
                    current_bs = data[0].size(0)  # the actual batch size of current iteration
                    if total_iters % opt.print_epoch_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    
                    model.set_input(data)
                    model.optimize_parameters()
                    
                    epoch_iters += 1
                    epoch_samples += current_bs
                    loss = model.get_current_loss()
                    epoch_loss += loss
                    for key, value in model.get_current_losses().items():
                        components[key] = components.get(key, 0.0) + float(value)
                    pbar.set_postfix(loss=f'{loss:.6f}', avg=f'{epoch_loss / epoch_iters:.6f}', refresh=False)

            model.update_learning_rate()
            # 完成该轮优化和 scheduler 推进后验证，best checkpoint 可正常续训。
            if validator is not None and validator.should_validate(model):
                validator.run(model)
            
            total_epochs += 1
            total_iters += epoch_iters
            avg_loss = epoch_loss / epoch_iters
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"Epoch {epoch}/{opt.n_epochs} completed. Average loss: {avg_loss:.6f}. Time taken: {epoch_time:.2f} sec"
            )
            record = {'epoch': epoch, 'phase': model.get_phase(), 'loss': avg_loss, 'epoch_iters': epoch_iters,
                      'epoch_samples': epoch_samples, 'epoch_time': epoch_time, 'loss_components': {k: v / epoch_iters for k, v in components.items()},
                      'learning_rates_used': learning_rates,
                      'learning_rates': {name: optimizer.param_groups[0]['lr'] for name, optimizer in model.optimizers.items()}}
            
            with (model.save_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                
            if epoch % opt.save_latest_freq == 0:
                model.save_model(f'{opt.model}_latest.pth')
                
            if epoch % opt.save_epoch_freq == 0 or epoch == opt.epoch_gap or epoch == opt.n_epochs:
                model.save_model(f"{opt.model}_epoch_{epoch}.pth")
                
        logger.info(f'Training completed: {model.save_dir}')
        return model.save_dir
    
    finally:
        logger.remove(sink)
        if hasattr(loader.dataset, 'close'):
            loader.dataset.close()
        cleanup_ddp()


if __name__ == '__main__':
    with initialize(version_base=None, config_path='./configs'):
        opt: BaseConfig | TrainConfig = compose(config_name='config', overrides=['+mode@_global_=train', *sys.argv[1:]])
    train(opt)

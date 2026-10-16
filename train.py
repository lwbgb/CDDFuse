import random
import time

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from models.cddfuse_model import CDDFuseModel
from schemas.base_config import BaseConfig
from schemas.train_config import TrainConfig
from utils import create_dataset
from utils.device import cleanup_ddp
from utils.logger_initializer import logger
from hydra import initialize, compose

def seed_everything(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    with initialize(version_base=None, config_path="./configs"):
        opt: BaseConfig | TrainConfig = compose(config_name="config", 
                                                overrides=["+mode@_global_=train"])

    seed_everything(opt.get("seed", 3407))

    dataLoader: DataLoader = create_dataset(opt, "MSRS_train_imgsize_128_stride_200.h5")
    dataset_size = len(dataLoader.dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")

    model: CDDFuseModel = CDDFuseModel(opt)  # create a model given opt.model and other options
    model.setup()  # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

    total_iters = (opt.start_epoch - 1) * dataset_size  # the total number of training iterations
    total_epochs = opt.start_epoch - 1  # the total number of training epochs
    epoch_batches: int = dataset_size // opt.batch_size

    logger.info(
        f"Starting [{opt.model}] training from epoch {opt.start_epoch} to {opt.n_epochs}, "
        f"batch_size={opt.batch_size}, lr={opt.lr}, use_bf16={opt.get('use_bf16', True)}"
    )
    for epoch in range(opt.start_epoch, opt.n_epochs + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iters = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_loss = 0.0

        model.update_state(epoch)
        # visualizer.reset()

        pbar = tqdm(
            dataLoader,
            desc=f"[Phase {model.get_phase()}] [Epoch {epoch}/{opt.n_epochs}]",
            dynamic_ncols=True,  # 进度条宽度自适应
            leave=False,  # 进度条完成后不保留
        )

        for batch_idx, data in enumerate(pbar):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            current_bs = data[0].size(0)
            if total_iters % opt.print_epoch_freq == 0:
                t_data = iter_start_time - iter_data_time

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            epoch_iters += current_bs
            loss = model.get_current_loss()
            epoch_loss += loss * current_bs

            # if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if (batch_idx + 1) % max(1, opt.print_iter_freq // opt.batch_size) == 0:  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                decomp_val = model.losses.get("loss_decomp", 0.0)
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.8f}",
                        "decomp": f"{decomp_val:.4f}", 
                    }
                )
                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # visualizer.plot_current_losses(total_iters, losses)

            # if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            #     logger.info(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
            #     save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        total_epochs += 1
        total_iters += epoch_iters
        logger.info(
            f"Epoch {epoch}/{opt.n_epochs} completed. Average loss: {(epoch_loss / dataset_size):.6f}. Time taken: {time.time() - epoch_start_time:.2f} sec"
        )
        
        if epoch % opt.save_latest_freq == 0:
            model.save_model(f"{opt.model}_latest.pth")
            print(f"The latest model saved at the end of epoch: {epoch}, iters: {total_iters}")

        if epoch % opt.save_epoch_freq == 0 or epoch == opt.epoch_gap or epoch == opt.n_epochs:
            model.save_model(f"{opt.model}_epoch_{epoch}.pth")
            print(f"The model saved at the end of epoch: {epoch}, iters: {total_iters}")

    cleanup_ddp()

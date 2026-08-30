# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
"""

from pathlib import Path

from cv2 import phase
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from net import BaseMambaEncoder, Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from schemas.train_config import TrainConfig
from utils.checkpoint import load_epoch_checkpoint, save_epoch_checkpoint
from utils.dataset import H5Dataset, get_loader
import os
import sys
import time
from time import localtime, strftime
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.device import init_ddp
from utils.loss import Fusionloss, cc
import kornia
from hydra import initialize, compose
from utils.logger_initializer import logger

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"\
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    """
    ------------------------------------------------------------------------------
    Configure our network
    ------------------------------------------------------------------------------
    """
    device = init_ddp()
    criteria_fusion = Fusionloss().to(device)
    with initialize(version_base=None, config_path="./configs"):
        opt: TrainConfig = compose(config_name="train")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Model
    DIDF_Encoder = Restormer_Encoder().to(device)
    DIDF_Decoder = Restormer_Decoder().to(device)
    BaseFuseLayer = BaseMambaEncoder(dim=64).to(device)
    # BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)

    # optimizer, scheduler and loss function
    optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=opt.optim_step, gamma=opt.optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opt.optim_step, gamma=opt.optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=opt.optim_step, gamma=opt.optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=opt.optim_step, gamma=opt.optim_gamma)

    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    Loss_ssim = kornia.losses.SSIMLoss(window_size=opt.SSIM_window_size, reduction="mean")


    # data loader
    trainloader = get_loader(opt, H5Dataset(Path(opt.dataset_root) / "MSRS_train_imgsize_128_stride_200.h5"))
    # trainloader = DataLoader(
    #     H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
    #     batch_size=opt.batch_size,
    #     shuffle=opt.shuffle,
    #     num_workers=opt.num_threads,
    # )
    dataset_size = len(trainloader.dataset)

    models: dict[str, nn.Module] = {
        "DIDF_Encoder": DIDF_Encoder,
        "DIDF_Decoder": DIDF_Decoder,
        "BaseFuseLayer": BaseFuseLayer,
        "DetailFuseLayer": DetailFuseLayer,
    }

    optimizers: dict[str, torch.optim.Optimizer] = {
        "optimizer_encoder": optimizer1,
        "optimizer_decoder": optimizer2,
        "optimizer_base_fusion": optimizer3,
        "optimizer_detail_fusion": optimizer4,
    }

    schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] = {
        "scheduler_encoder": scheduler1,
        "scheduler_decoder": scheduler2,
        "scheduler_base_fusion": scheduler3,
        "scheduler_detail_fusion": scheduler4,
    }
    
    train_state = {"models": models, "optimizers": optimizers, "schedules": schedulers}
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    train_date = strftime("%Y%m%d-%H%M%S", localtime())

    """
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    """
    
    current_phase = 1 if opt.start_epoch <= opt.epoch_gap else 2
    total_iters: int = (opt.start_epoch - 1) * dataset_size
    total_epochs: int = opt.start_epoch - 1
    epoch_batches: int = dataset_size // opt.batch_size
    epoch_losses: list[float] = []

    if opt.continue_train:
        date = "20260830-063544"
        checkpoint_path = Path(opt.checkpoint_root) / date / f"CDDFuse_phase2_latest.pth"
        load_epoch_checkpoint(checkpoint_path, device, models, optimizers, schedulers, expected_phase=2)

    for epoch in range(opt.start_epoch, opt.n_epochs + 1):
        
        current_phase = 1 if epoch <= opt.epoch_gap else 2
        pbar = tqdm(
            trainloader,
            desc=f"[Phase {current_phase}] [Epoch {epoch}/{opt.n_epochs}]",
            dynamic_ncols=True,
            leave=False  # 每个 epoch 结束后保留该行记录
        )
        epoch_start_time = time.time()  
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iters = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_loss = 0.0
        
        """ train """
        for batch_idx, (data_VIS, data_IR) in enumerate(pbar):

            iter_start_time = time.time()  # timer for computation per iteration
            data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
            
            for model in models.values():
                model.train()
                model.zero_grad()

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            if current_phase == 1:  # Phase I
                feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
                feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
                data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
                data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

                Gradient_loss = L1Loss(
                    kornia.filters.SpatialGradient()(data_VIS), kornia.filters.SpatialGradient()(data_VIS_hat)
                )

                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                loss = (
                    opt.coeff_mse_loss_VF * mse_loss_V
                    + opt.coeff_mse_loss_IF * mse_loss_I
                    + opt.coeff_decomp * loss_decomp
                    + opt.coeff_tv * Gradient_loss
                )
                loss.backward()

                for model in list(models.values())[:2]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                
                for optimizer in list(optimizers.values())[:2]:
                    optimizer.step()

            else:  # Phase II
                feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
                feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
                feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
                data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)

                mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

                loss = fusionloss + opt.coeff_decomp * loss_decomp
                loss.backward()
                
                for model in models.values():
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                
                for optimizer in optimizers.values():
                    optimizer.step()
                    

            epoch_iters += opt.batch_size
            epoch_loss += loss.item() * opt.batch_size

            # Determine approximate time left
            iters_left = dataset_size * opt.n_epochs - total_iters - epoch_iters
            time_left = datetime.timedelta(seconds=int((iters_left / opt.batch_size) * (time.time() - iter_start_time)))
            
            # statement = f"[Phase {current_phase}] [Epoch {epoch}/{opt.n_epochs}] [Batch {batch_idx + 1}/{epoch_batches}] [loss: {loss.item():.8f}] ETA: {time_left}"
            # sys.stdout.write("\r" + statement)
            # 2. 动态更新进度条右侧的指标信息（替代手动打印 loss 和 ETA）
            pbar.set_postfix({
                "loss": f"{loss.item():.8f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        

        # adjust the learning rate
        if current_phase == 1:
            for scheduler in list(schedulers.values())[:2]:
                scheduler.step()
        else:
            for scheduler in schedulers.values():
                scheduler.step()

        for optimizer in optimizers.values():
            lr = optimizer.param_groups[0]["lr"]
            lr = max(lr, opt.min_lr)
    
            
        total_epochs += 1
        total_iters += epoch_iters
        epoch_losses.append(epoch_loss / dataset_size)
        logger.info(f"Epoch {epoch}/{opt.n_epochs} completed. Average loss: {epoch_losses[-1]:.6f}. Time taken: {time.time() - epoch_start_time:.2f}s")
        

        # save latest checkpoint
        if epoch % opt.print_epoch_freq == 0 or epoch == opt.epoch_gap or epoch == opt.n_epochs:
            
            latest_checkpoint_path = Path(opt.checkpoint_root) / train_date / f"CDDFuse_phase{current_phase}_latest.pth"

            save_epoch_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                phase=current_phase,
                epoch=epoch,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                config=OmegaConf.to_container(opt, resolve=True),
            )

            logger.info(f"Checkpoint saved: phase={current_phase}, " f"global_epoch={epoch}, " f"path={latest_checkpoint_path}")

        # save checkpoint by save_epoch_freq
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.epoch_gap or epoch == opt.n_epochs:

            checkpoint_path = Path(opt.checkpoint_root) / train_date / f"CDDFuse_phase{current_phase}_" f"epoch{epoch:03d}.pth"

            save_epoch_checkpoint(
                checkpoint_path=checkpoint_path,
                phase=current_phase,
                epoch=epoch,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                config=OmegaConf.to_container(opt, resolve=True),
            )

            logger.info(f"Checkpoint saved: phase={current_phase}, " f"global_epoch={epoch}, " f"path={checkpoint_path}")


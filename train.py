# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
"""

from omegaconf import DictConfig, OmegaConf

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from schemas.train_config import TrainConfig
from utils.checkpoint import save_epoch_checkpoint
from utils.dataset import H5Dataset
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

if __name__ == "__main__":
    """
    ------------------------------------------------------------------------------
    Configure our network
    ------------------------------------------------------------------------------
    """

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = init_ddp()
    criteria_fusion = Fusionloss().to(device)
    with initialize(version_base=None, config_path="./configs"):
        opt: TrainConfig = compose(config_name="train")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Model
    modules = []
    DIDF_Encoder = Restormer_Encoder().to(device)
    DIDF_Decoder = Restormer_Decoder().to(device)
    BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)
    modules.extend([DIDF_Encoder, DIDF_Decoder, BaseFuseLayer, DetailFuseLayer])

    # optimizer, scheduler and loss function
    optimizers = []
    optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizers.extend([optimizer1, optimizer2, optimizer3, optimizer4])

    schedulers = []
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=opt.optim_step, gamma=opt.optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opt.optim_step, gamma=opt.optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=opt.optim_step, gamma=opt.optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=opt.optim_step, gamma=opt.optim_gamma)
    schedulers.extend([scheduler1, scheduler2, scheduler3, scheduler4])

    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    Loss_ssim = kornia.losses.SSIMLoss(window_size=opt.SSIM_window_size, reduction="mean")


    # data loader
    trainloader = DataLoader(
        H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.num_threads,
    )

    loader = {"train": trainloader}
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

    models = {
        "DIDF_Encoder": DIDF_Encoder,
        "DIDF_Decoder": DIDF_Decoder,
        "BaseFuseLayer": BaseFuseLayer,
        "DetailFuseLayer": DetailFuseLayer,
    }

    optimizers = {
        "optimizer_encoder": optimizer1,
        "optimizer_decoder": optimizer2,
        "optimizer_base_fusion": optimizer3,
        "optimizer_detail_fusion": optimizer4,
    }

    schedulers = {
        "scheduler_encoder": scheduler1,
        "scheduler_decoder": scheduler2,
        "scheduler_base_fusion": scheduler3,
        "scheduler_detail_fusion": scheduler4,
    }


    """
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    """

    torch.backends.cudnn.benchmark = True
    prev_time = time.time()
    save_dir = "models/" + strftime("%Y%m%d", localtime())
    global_step = 0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(opt.start_epoch, opt.n_epochs + 1):
        current_phase = 1 if epoch <= opt.epoch_gap else 2
        """ train """
        for batch, (data_VIS, data_IR) in enumerate(loader["train"]):
            data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
            
            for model in modules:
                model.train()
                model.zero_grad()

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            if epoch <= opt.epoch_gap:  # Phase I
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
                nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                
                for optimizer in optimizers.values():
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
                
                nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
                
                for optimizer in optimizers.values():
                    optimizer.step()

            # Determine approximate time left
            batches_done = (epoch - 1) * len(loader["train"]) + batch
            batches_left = opt.n_epochs * len(loader["train"]) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            statement = f"[Phase {current_phase}] [Epoch {epoch}/{opt.n_epochs}] [Batch {batch + 1}/{len(loader['train'])}] [loss: {loss.item()}] ETA: {time_left}"
            logger.info(statement)

            # sys.stdout.write("\r" + statement)
            # sys.stdout.write(
            #     "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            #     % (
            #         epoch,
            #         opt.n_epochs,
            #         batch + 1,
            #         len(loader["train"]),
            #         loss.item(),
            #         time_left,
            #     )
            # )

            global_step += 1

        # adjust the learning rate
        scheduler1.step()
        scheduler2.step()
        if not epoch <= opt.epoch_gap:
            scheduler3.step()
            scheduler4.step()

        for optimizer in optimizers.values():
            lr = optimizer.param_groups[0]["lr"]
            lr = max(lr, opt.min_lr)

        # save latest checkpoint
        if epoch % opt.print_epoch_freq == 0 or epoch == opt.epoch_gap or epoch == opt.n_epochs:

            latest_checkpoint_path = os.path.join(
                save_dir,
                f"CDDFuse_phase{current_phase}_latest.pth",
            )

            save_epoch_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                phase=current_phase,
                epoch=epoch,
                global_step=global_step,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                config=OmegaConf.to_container(opt, resolve=True),
            )

            logger.info(f"Checkpoint saved: phase={current_phase}, " f"global_epoch={epoch}, " f"path={latest_checkpoint_path}")

        # save checkpoint by save_epoch_freq
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.epoch_gap or epoch == opt.n_epochs:

            checkpoint_name = f"CDDFuse_phase{current_phase}_" f"epoch_{epoch:03d}.pth"
            checkpoint_path = os.path.join(save_dir, checkpoint_name)

            save_epoch_checkpoint(
                checkpoint_path=checkpoint_path,
                phase=current_phase,
                epoch=epoch,
                global_step=global_step,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                config=OmegaConf.to_container(opt, resolve=True),
            )

            logger.info(f"Checkpoint saved: phase={current_phase}, " f"global_epoch={epoch}, " f"path={checkpoint_path}")

    if False:
        checkpoint = {
            "DIDF_Encoder": DIDF_Encoder.state_dict(),
            "DIDF_Decoder": DIDF_Decoder.state_dict(),
            "BaseFuseLayer": BaseFuseLayer.state_dict(),
            "DetailFuseLayer": DetailFuseLayer.state_dict(),
        }
        torch.save(checkpoint, os.path.join("models/CDDFuse_" + timestamp + ".pth"))

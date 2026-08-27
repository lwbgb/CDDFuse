# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from omegaconf import DictConfig, OmegaConf

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from schemas.train_config import TrainConfig
from utils.checkpoint import save_checkpoint
from utils.dataset import H5Dataset
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.device import init_ddp
from utils.loss import Fusionloss, cc
import kornia
from hydra import initialize, compose



'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = init_ddp()
criteria_fusion = Fusionloss().to(device)
with initialize(version_base=None, config_path="./configs"):
        opt: TrainConfig = compose(config_name="train")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 120 # total epoch
epoch_gap = 40  # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5




# Model
DIDF_Encoder = Restormer_Encoder().to(device)
DIDF_Decoder = Restormer_Decoder().to(device)
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=opt.optim_step, gamma=opt.optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opt.optim_step, gamma=opt.optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=opt.optim_step, gamma=opt.optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=opt.optim_step, gamma=opt.optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(window_size=11, reduction='mean')


# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=opt.batch_size,
                         shuffle=opt.shuffle,
                         num_workers=opt.num_threads)

loader = {'train': trainloader, }
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

criteria = {
    "fusion_loss": criteria_fusion.to(device),
    "mse_loss": MSELoss.to(device),
    "l1_loss": L1Loss.to(device),
    "ssim_loss": Loss_ssim.to(device),
}


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
save_interval = 5
torch.backends.cudnn.benchmark = True
prev_time = time.time()
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(opt.n_epochs):
    current_phase = 1 if epoch < opt.epoch_gap else 2
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < opt.epoch_gap: #Phase I
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            loss = opt.coeff_mse_loss_VF * mse_loss_V + opt.coeff_mse_loss_IF * \
                   mse_loss_I + opt.coeff_decomp * loss_decomp + opt.coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()

            latest_losses = {
                "total_loss": float(loss.detach().cpu()),
                # "loss_ir": float(loss_ir.detach().cpu()),
                # "loss_vis": float(loss_vis.detach().cpu()),
                "loss_decomp": float(loss_decomp.detach().cpu()),
            }

        else:  #Phase II
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)  

            
            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            loss = fusionloss + opt.coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=opt.clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            
            latest_losses = {
                "total_loss": float(loss.detach().cpu()),
                "fusion_loss": float(fusionloss.detach().cpu()),
                "loss_decomp": float(loss_decomp.detach().cpu()),
            }


        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = opt.n_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1.step()  
    scheduler2.step()
    if not epoch < opt.epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/CDDFuse_"+timestamp+'.pth'))



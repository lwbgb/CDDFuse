import datetime
from email.headerregistry import DateHeader
import os

import kornia
import torch
from torch import nn

from net import BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder, Restormer_Encoder
from utils.dataset import H5Dataset
from utils.loss import Fusionloss


class TrainModel:

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        self.criteria_fusion = Fusionloss()
        self.model_str = "CDDFuse"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = DateHeader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=self.batch_size,
                         shuffle=True,
                         num_workers=0)
        self.timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

        # . Set the hyper-parameters for training
        self.num_epochs = 120  # total epoch
        self.epoch_gap = 40  # epoches of Phase I
        self.lr = 1e-4
        self.weight_decay = 0
        self.batch_size = 4
        self.GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]

        # Coefficients of the loss function
        self.coeff_mse_loss_VF = 1.0  # alpha1
        self.coeff_mse_loss_IF = 1.0
        self.coeff_decomp = 2.0  # alpha2 and alpha4
        self.coeff_tv = 5.0
        self.clip_grad_norm_value = 0.01
        self.optim_step = 20
        self.optim_gamma = 0.5

        # 网络结构
        self.DIDF_Encoder = Restormer_Encoder().to(self.device)
        self.DIDF_Decoder = Restormer_Decoder().to(self.device)
        self.BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(self.device)
        self.DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(self.device)

        # optimizer, scheduler and loss function
        self.optimizer1 = torch.optim.Adam(
            self.DIDF_Encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer2 = torch.optim.Adam(
            self.DIDF_Decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer3 = torch.optim.Adam(
            self.BaseFuseLayer.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer4 = torch.optim.Adam(
            self.DetailFuseLayer.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=self.optim_step, gamma=self.optim_gamma)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer2, step_size=self.optim_step, gamma=self.optim_gamma)
        self.scheduler3 = torch.optim.lr_scheduler.StepLR(self.optimizer3, step_size=self.optim_step, gamma=self.optim_gamma)
        self.scheduler4 = torch.optim.lr_scheduler.StepLR(self.optimizer4, step_size=self.optim_step, gamma=self.optim_gamma)

        self.MSELoss = nn.MSELoss()  
        self.L1Loss = nn.L1Loss()
        self.Loss_ssim = kornia.losses.SSIMLoss(window_size=11, reduction='mean')

    def save_model(self, epoch: int, path: str):
        checkpoint = {
            'epoch': epoch,
            'DIDF_Encoder': self.DIDF_Encoder.state_dict(),
            'DIDF_Decoder': self.DIDF_Decoder.state_dict(),
            'BaseFuseLayer': self.BaseFuseLayer.state_dict(),
            'DetailFuseLayer': self.DetailFuseLayer.state_dict(),
            
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict(),
            'optimizer3': self.optimizer3.state_dict(),
            'optimizer4': self.optimizer4.state_dict(),
            
            'scheduler1': self.scheduler1.state_dict(),
            'scheduler2': self.scheduler2.state_dict(),
            'scheduler3': self.scheduler3.state_dict(),
            'scheduler4': self.scheduler4.state_dict(),
        }

        save_path = os.path.join(path, f"CDDFuse_epoch_{epoch}.pth")
        torch.save(checkpoint, save_path)
        print(f"\nCheckpoint saved to {save_path}")

    def load_model(self, mode_path: str):
        if mode_path is not None and os.path.exists(mode_path):
            print(f"Loading checkpoint from {mode_path} ...")
            ckpt = torch.load(mode_path, map_location=self.device)

            self.DIDF_Encoder.load_state_dict(ckpt['DIDF_Encoder'])
            self.DIDF_Decoder.load_state_dict(ckpt['DIDF_Decoder'])
            self.BaseFuseLayer.load_state_dict(ckpt['BaseFuseLayer'])
            self.DetailFuseLayer.load_state_dict(ckpt['DetailFuseLayer'])

            self.optimizer1.load_state_dict(ckpt['optimizer1'])
            self.optimizer2.load_state_dict(ckpt['optimizer2'])
            self.optimizer3.load_state_dict(ckpt['optimizer3'])
            self.optimizer4.load_state_dict(ckpt['optimizer4'])

            self.scheduler1.load_state_dict(ckpt['scheduler1'])
            self.scheduler2.load_state_dict(ckpt['scheduler2'])
            self.scheduler3.load_state_dict(ckpt['scheduler3'])
            self.scheduler4.load_state_dict(ckpt['scheduler4'])

            start_epoch = ckpt['epoch'] + 1
            print(f"Successfully loaded! Resuming training from epoch {start_epoch}")
        else:
            raise FileNotFoundError("mode path Error!")

    def train_phase1(self, start_epoch: int):
        if start_epoch > 0:
            model_path = "./models/base/phase1" + f"CDDFuse_epoch_{start_epoch}.pth"
            self.load_model(model_path)
        
        pass

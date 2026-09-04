from pathlib import Path
import unittest

import kornia
from torch import nn, optim
import torch

from net import BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder, Restormer_Encoder
from schemas.model_checkpoint import ModelCkp
from schemas.train_config import TrainConfig
from utils.checkpoint import save_epoch_checkpoint, load_epoch_checkpoint
from utils.device import init_ddp
from hydra import initialize, compose


class TestModel(unittest.TestCase):

    device = init_ddp()
    with initialize(version_base=None, config_path="../configs"):
        opt: TrainConfig = compose(config_name="train")

    # Model
    DIDF_Encoder = Restormer_Encoder().to(device)
    DIDF_Decoder = Restormer_Decoder().to(device)
    BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
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

    def test_load_model(self):
        checkpoint_path = "checkpoints/base_model/CDDFuse_phase1_latest.pth"
        checkpoint = load_epoch_checkpoint(checkpoint_path, self.device, self.models, self.optimizers, self.schedulers, expected_phase=1, strict=True)
        model_ckp = ModelCkp.from_dict(checkpoint)
        print(type(model_ckp))
        ...

    def test_01(self):
        prefix = ''
        path = Path("checkpoints") / prefix / "CDDFuse_phase1_latest.pth"
        print(f"Checkpoint path: {path}")
        
    
    def test_02(self):
        a = 2
        
        try:
            if a == 2:
                raise ValueError("a should not be 2")
            if a == 3:
                raise ValueError("a should not be 3")
        except ValueError as e:
            print(f"Error: {e}")
            raise
        
        print("Test 02 completed successfully.")

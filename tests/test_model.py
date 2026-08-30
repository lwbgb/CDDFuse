import unittest

from torch import nn, optim
import torch

from net import BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder, Restormer_Encoder
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
        checkpoint_path = "checkpoints/20260829/CDDFuse_phase1_latest.pth"
        checkpoint = load_epoch_checkpoint(checkpoint_path, self.device, self.models, self.optimizers, self.schedulers, expected_phase=1, strict=True)
        ...
        
    
    def test_01(self):
        d = {
            "encoder1": nn.Linear(10, 10),
            "encoder2": nn.Linear(10, 10),
            "decoder1": nn.Linear(10, 10),
            "decoder2": nn.Linear(10, 10),
        }
        
        print(list(d.values())[:2])
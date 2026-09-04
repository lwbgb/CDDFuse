import os
from time import localtime, strftime
from typing import Any
from omegaconf import DictConfig
import torch
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod

from torch import nn
import torch.distributed as dist
from schemas.base_config import BaseConfig
from schemas.model_checkpoint import ModelCkp
from utils import networks, path_util
from utils.device import init_ddp
from utils.logger_initializer import logger


class BaseModel(ABC):
    """
    针对 CDDFuse 和 Hydra 架构定制的 BaseModel 抽象类。
    必须实现的函数:
        -- <__init__>:           初始化类; 首先调用 BaseModel.__init__(self, opt)。
        -- <set_input>:          从 dataset 解包数据并预处理。
        -- <forward>:            执行前向传播计算。
        -- <optimize_parameters>:计算损失、梯度，并更新网络权重。
    """

    def __init__(self, opt: DictConfig):
        torch.backends.cudnn.benchmark = True
        self.opt = opt
        self.init_time = strftime("%Y%m%d_%H%M%S", localtime())
        self.device = init_ddp()
        self.isTrain = opt.get("isTrain", False)
        self.ckp_name = opt.get("ckp_name")
        self.save_dir = Path(opt.checkpoint_root) / self.init_time

        self.models: dict[str, nn.Module] = dict()
        self.losses: dict[str, Any] = dict()
        self.schedulers: dict[str, Any] = dict()
        self.optimizers: dict[str, Any] = dict()
        self.visual_names = []
        self.image_paths = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """

        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        pass

    @abstractmethod
    def load_model(self, ckp_name: str):
        pass

    @abstractmethod
    def save_model(self, ckp_name: str):
        pass

    def setup(self):
        # 将模型挂载到设备并初始化权重
        for name, net in self.models.items():
            net = networks.init_net(net, self.opt)

        # 测试或是继续训练时加载模型
        if not self.isTrain or self.opt.continue_train:
            model_ckp: ModelCkp = self.load_model(self.ckp_name)

        self.print_networks(self.opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""

        for net in self.models.values():
            net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """

        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        pass

    def get_current_losses(self):
        """返回当前的 losses 字典用于打印"""
        return self.losses

    def get_current_visuals(self):
        """返回当前的图像数据字典用于可视化"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def print_networks(self, verbose):
        print("---------- Networks initialized -------------")
        for name, net in self.models.items():
            num_params = sum(p.numel() for p in net.parameters())
            if verbose:
                print(net)
            print(f"[Network {name}] Total number of parameters : {num_params / 1e6:.3f} M")
        print("-----------------------------------------------")

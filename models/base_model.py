import os
import torch
import logging
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod
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

    def __init__(self, opt):
        self.opt = opt
        
        # Hydra 配置读取（DictConfig 既可以通过 . 访问属性，也可用 .get() 防止缺少键报错）
        self.isTrain = opt.get("isTrain", True)
        
        # 1. 采用 Hydra yaml 中的 checkpoint_root 替代原本的 checkpoints_dir
        save_dir_str = opt.get("checkpoint_root", "checkpoints/")
        self.save_dir = Path(save_dir_str)
        if self.isTrain:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # 2. 匹配 train.yaml 中的设备配置，或自动检测
        if str(opt.get("device", "")).lower() == "gpu" and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        torch.backends.cudnn.benchmark = True
        
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        
        # 3. 将原先的列表改为字典，方便统一检查点系统进行键值(Key)匹配
        self.optimizers = {}
        self.schedulers = {}

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self):
        """挂载模型到设备，并根据 Hydra 配置初始化学习率调度器"""
        # 抛弃 "net" 前缀，直接利用 self.model_names 挂载到设备
        for name in self.model_names:
            if hasattr(self, name):
                model = getattr(self, name)
                model.to(self.device)
            else:
                logger.warning(f"在 model_names 中定义了 '{name}'，但类中未找到同名实例。")

        self.print_networks(verbose=False)

        # 动态绑定调度器 (Schedulers)
        if self.isTrain:
            for opt_name, optimizer in self.optimizers.items():
                # 按照 train.py 规则，动态生成调度器。如：optimizer_encoder -> scheduler_encoder
                sch_name = opt_name.replace("optimizer", "scheduler")
                self.schedulers[sch_name] = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=self.opt.optim_step, 
                    gamma=self.opt.optim_gamma
                )

    def eval(self):
        """测试时开启评估模式 (脱离 "net" 前缀限制)"""
        for name in self.model_names:
            if hasattr(self, name):
                getattr(self, name).eval()

    def train(self):
        """训练时开启训练模式"""
        for name in self.model_names:
            if hasattr(self, name):
                getattr(self, name).train()

    def update_learning_rate(self, current_phase=1):
        """根据当前阶段 (Phase) 更新学习率"""
        active_sch_keys = ["scheduler_encoder", "scheduler_decoder"]
        if current_phase == 2:
            active_sch_keys.extend(["scheduler_base_fusion", "scheduler_detail_fusion"])
            
        for key in active_sch_keys:
            if key in self.schedulers:
                self.schedulers[key].step()
        
        # 保证学习率不低于 train.yaml 设置的 min_lr
        min_lr = self.opt.get("min_lr", 1e-6)
        for opt_name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"], min_lr)

    def save_checkpoint(self, epoch, phase, filename=None):
        """将模型、优化器、调度器打包存入单个 .pth 文件 (替代旧版的 save_networks)"""
        if filename is None:
            filename = f"CDDFuse_phase{phase}_epoch_{epoch:03d}.pth"
        checkpoint_path = self.save_dir / filename
        
        # 动态获取模型实例字典
        models_dict = {name: getattr(self, name) for name in self.model_names if hasattr(self, name)}
        
        checkpoint = {
            "epoch": epoch,
            "phase": phase,
            "models_state_dict": {name: model.state_dict() for name, model in models_dict.items()},
            "optimizers_state_dict": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "schedulers_state_dict": {name: sch.state_dict() for name, sch in self.schedulers.items()},
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint 已保存: Phase={phase}, Epoch={epoch}, Path={checkpoint_path}")

    def load_checkpoint(self, filename):
        """从单个 .pth 恢复训练状态 (替代旧版的 load_networks)"""
        checkpoint_path = self.save_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 Checkpoint 文件: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        saved_epoch = checkpoint["epoch"]
        saved_phase = checkpoint["phase"]
        
        models_dict = {name: getattr(self, name) for name in self.model_names if hasattr(self, name)}
        
        # 1. 恢复模型参数
        for name, model in models_dict.items():
            if name in checkpoint["models_state_dict"]:
                model.load_state_dict(checkpoint["models_state_dict"][name])
            else:
                logger.warning(f"Checkpoint 中缺少模型 '{name}' 的参数。")
                
        # 2. 区分阶段恢复优化器和调度器
        active_opt_keys = ["optimizer_encoder", "optimizer_decoder"]
        active_sch_keys = ["scheduler_encoder", "scheduler_decoder"]
        
        if saved_phase == 2:
            active_opt_keys.extend(["optimizer_base_fusion", "optimizer_detail_fusion"])
            active_sch_keys.extend(["scheduler_base_fusion", "scheduler_detail_fusion"])
            
        for key in active_opt_keys:
            if key in checkpoint["optimizers_state_dict"] and key in self.optimizers:
                self.optimizers[key].load_state_dict(checkpoint["optimizers_state_dict"][key])
                
        for key in active_sch_keys:
            if key in checkpoint["schedulers_state_dict"] and key in self.schedulers:
                self.schedulers[key].load_state_dict(checkpoint["schedulers_state_dict"][key])
                
        logger.info(f"Checkpoint 恢复成功: Epoch {saved_epoch}, Phase {saved_phase}")
        return saved_epoch, saved_phase

    def get_current_losses(self):
        """返回当前的 losses 字典用于打印"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if hasattr(self, "loss_" + name):
                errors_ret[name] = float(getattr(self, "loss_" + name))
        return errors_ret

    def get_current_visuals(self):
        """返回当前的图像数据字典用于可视化"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def print_networks(self, verbose):
        """打印参数量（剔除了 "net" 前缀）"""
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if hasattr(self, name):
                net = getattr(self, name)
                num_params = sum(p.numel() for p in net.parameters())
                if verbose:
                    print(net)
                print(f"[Network {name}] Total number of parameters : {num_params / 1e6:.3f} M")
        print("-----------------------------------------------")
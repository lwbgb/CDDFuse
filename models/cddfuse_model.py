from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import kornia
from models.base_model import BaseModel
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils import path_util
from utils.loss import Fusionloss, cc
from utils.logger_initializer import logger
from schemas.model_checkpoint import ModelCkp


class CDDFuseModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.name = opt.model

        # 定义网络结构
        self.DIDF_Encoder = Restormer_Encoder()
        self.DIDF_Decoder = Restormer_Decoder()
        self.BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8)
        self.DetailFuseLayer = DetailFeatureExtraction(num_layers=1)
        self.models: dict[str, nn.Module] = {
            "DIDF_Encoder": self.DIDF_Encoder,
            "DIDF_Decoder": self.DIDF_Decoder,
            "BaseFuseLayer": self.BaseFuseLayer,
            "DetailFuseLayer": self.DetailFuseLayer,
        }

        if self.isTrain:
            # 定义损失函数
            self.MSELoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()
            self.Loss_ssim = kornia.losses.SSIMLoss(window_size=opt.SSIM_window_size, reduction="mean")
            self.losses: dict[str, nn.Module] = {
                "MSELoss": self.MSELoss,
                "L1Loss": self.L1Loss,
                "SSIMLoss": self.Loss_ssim,
            }

            # 定义优化器。按照 base_model 的规范，需要将所有优化器存入 self.optimizers 列表
            self.optimizer_encoder = torch.optim.Adam(
                self.DIDF_Encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            )
            self.optimizer_decoder = torch.optim.Adam(
                self.DIDF_Decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            )
            self.optimizer_base = torch.optim.Adam(
                self.BaseFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            )
            self.optimizer_detail = torch.optim.Adam(
                self.DetailFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            )
            self.optimizers: dict[str, torch.optim.Optimizer] = {
                "optimizer_encoder": self.optimizer_encoder,
                "optimizer_decoder": self.optimizer_decoder,
                "optimizer_base_fusion": self.optimizer_base,
                "optimizer_detail_fusion": self.optimizer_detail,
            }

            self.scheduler1 = torch.optim.lr_scheduler.StepLR(
                self.optimizer_encoder, step_size=opt.optim_step, gamma=opt.optim_gamma
            )
            self.scheduler2 = torch.optim.lr_scheduler.StepLR(
                self.optimizer_decoder, step_size=opt.optim_step, gamma=opt.optim_gamma
            )
            self.scheduler3 = torch.optim.lr_scheduler.StepLR(
                self.optimizer_base, step_size=opt.optim_step, gamma=opt.optim_gamma
            )
            self.scheduler4 = torch.optim.lr_scheduler.StepLR(
                self.optimizer_detail, step_size=opt.optim_step, gamma=opt.optim_gamma
            )
            self.schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {
                "scheduler_encoder": self.scheduler1,
                "scheduler_decoder": self.scheduler2,
                "scheduler_base_fusion": self.scheduler3,
                "scheduler_detail_fusion": self.scheduler4,
            }

            # 初始化日志中可能未被计算的损失值（防止 base_model 打印时报错）
            self._init_losses()

    def _init_losses(self):
        """将所有的 loss 初始化为 0，防止跨阶段打印日志时找不到变量"""
        for name in self.loss_names:
            setattr(self, "loss_" + name, 0.0)

    def set_phase(self, epoch):
        """
        供外部调用的方法，用于控制训练阶段。
        例如在 train.py 中：model.set_phase(epoch)
        """
        self.current_phase = 1 if epoch <= self.opt.epoch_gap else 2

    def set_input(self, input):
        """从 DataLoader 解包数据。"""
        # 假设 dataloader 字典输出的键名为 'VIS' 和 'IR'
        self.data_VIS = input["VIS"].to(self.device)
        self.data_IR = input["IR"].to(self.device)

    def forward(self):
        """执行前向传播计算。会根据 current_phase 采取不同的网络拓扑。"""
        if self.current_phase == 1:
            # Phase I: 自编码重构
            self.feature_V_B, self.feature_V_D, _ = self.netEncoder(self.data_VIS)
            self.feature_I_B, self.feature_I_D, _ = self.netEncoder(self.data_IR)

            self.data_VIS_hat, _ = self.netDecoder(self.data_VIS, self.feature_V_B, self.feature_V_D)
            self.data_IR_hat, _ = self.netDecoder(self.data_IR, self.feature_I_B, self.feature_I_D)

            # Phase I 没有最终的融合图像，置为 None
            self.data_Fuse = None

        elif self.current_phase == 2:
            # Phase II: 融合任务
            self.feature_V_B, self.feature_V_D, _ = self.DIDF_Encoder(self.data_VIS)
            self.feature_I_B, self.feature_I_D, _ = self.DIDF_Encoder(self.data_IR)

            self.feature_F_B = self.BaseFuseLayer(self.feature_I_B + self.feature_V_B)
            self.feature_F_D = self.DetailFuseLayer(self.feature_I_D + self.feature_V_D)

            self.data_Fuse, _ = self.DIDF_Decoder(self.data_VIS, self.feature_F_B, self.feature_F_D)

            # Phase II 不产出独立的可见光/红外重构图，保留 None
            self.data_VIS_hat = None
            self.data_IR_hat = None

    def backward(self):
        """计算损失，并进行反向传播。"""
        # 公共特征分解损失
        self.loss_cc_B = cc(self.feature_V_B, self.feature_I_B)
        self.loss_cc_D = cc(self.feature_V_D, self.feature_I_D)
        self.loss_decomp = (self.loss_cc_D**2) / (1.01 + self.loss_cc_B)

        if self.current_phase == 1:
            # 第一阶段重构损失计算
            self.loss_mse_V = 5 * self.Loss_ssim(self.data_VIS, self.data_VIS_hat) + self.MSELoss(
                self.data_VIS, self.data_VIS_hat
            )
            self.loss_mse_I = 5 * self.Loss_ssim(self.data_IR, self.data_IR_hat) + self.MSELoss(
                self.data_IR, self.data_IR_hat
            )

            spatial_grad = kornia.filters.SpatialGradient()
            self.loss_gradient = self.L1Loss(spatial_grad(self.data_VIS), spatial_grad(self.data_VIS_hat))

            self.loss_total = (
                self.opt.coeff_mse_loss_VF * self.loss_mse_V
                + self.opt.coeff_mse_loss_IF * self.loss_mse_I
                + self.opt.coeff_decomp * self.loss_decomp
                + self.opt.coeff_gradient * self.loss_gradient
            )
            self.loss_total.backward()

        else:
            # 第二阶段融合损失计算
            self.loss_fusion, _, _ = self.criteria_fusion(self.data_VIS, self.data_IR, self.data_Fuse)

            self.loss_total = self.loss_fusion + self.opt.coeff_decomp * self.loss_decomp
            self.loss_total.backward()

    def optimize_parameters(self):
        """更新网络权重（每个 iteration 调用一次）。"""
        self.forward()  # 前向传播

        # 根据阶段清空梯度
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        if self.current_phase == 2:
            self.optimizer_base.zero_grad()
            self.optimizer_detail.zero_grad()

        self.backward()  # 反向传播

        # 梯度裁剪及权重更新
        nn.utils.clip_grad_norm_(self.DIDF_Encoder.parameters(), max_norm=self.opt.clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(self.DIDF_Decoder.parameters(), max_norm=self.opt.clip_grad_norm_value, norm_type=2)

        self.optimizer_encoder.step()
        self.optimizer_decoder.step()

        if self.current_phase == 2:
            nn.utils.clip_grad_norm_(
                self.BaseFuseLayer.parameters(), max_norm=self.opt.clip_grad_norm_value, norm_type=2
            )
            nn.utils.clip_grad_norm_(
                self.DetailFuseLayer.parameters(), max_norm=self.opt.clip_grad_norm_value, norm_type=2
            )

            self.optimizer_base.step()
            self.optimizer_detail.step()

    def load_model(self, ckp_name: str, prefix: str | Path = "") -> ModelCkp:
        try:
            load_path = self.opt.checkpoint_root / prefix / ckp_name
            path_util.check_file_path(load_path)

            checkpoint: dict = torch.load(load_path, weights_only=True)
            model_ckp = ModelCkp.from_dict(checkpoint)

            logger.info(f"Loading {self.name} checkpoint from {load_path}...")
            for name, model in self.models.items():
                model_state = model_ckp.models.get(name)
                if model_state is None:
                    raise KeyError(f"Checkpoint 中缺少模型 '{name}' 的参数")
                model.load_state_dict(model_state)

            for name, optimizer in self.optimizers.items():
                optimizer_state = model_ckp.optimizers.get(name)
                if optimizer_state is None:
                    raise KeyError(f"Checkpoint 中缺少优化器 '{name}' 的参数")
                optimizer.load_state_dict(optimizer_state)

            for name, scheduler in self.schedulers.items():
                scheduler_state = model_ckp.schedulers.get(name)
                if scheduler_state is None:
                    raise KeyError(f"Checkpoint 中缺少调度器 '{name}' 的参数")
                scheduler.load_state_dict(scheduler_state)
        except Exception as e:
            logger.error(f"Error occurred while loading checkpoint: {e}")
            raise

        logger.info(f"{self.name}_phase{model_ckp.phase}_epoch{model_ckp.epoch} loaded successfully from {load_path}")
        return model_ckp

    def save_model(self, ckp_name: str) -> None:
        try:
            save_path = self.save_dir / ckp_name
            path_util.check_file_path(save_path, create=True)

            model_ckp = ModelCkp(
                phase=self.phase,
                epoch=self.epoch,
                models={name: model.state_dict() for name, model in self.models.items()},
                optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
                schedulers={name: scheduler.state_dict() for name, scheduler in self.schedulers.items()},
                config=OmegaConf.to_container(self.opt, resolve=True),
            )

            torch.save(model_ckp.to_dict(), save_path)
            logger.info(
                f"{self.name}_phase{model_ckp.phase}_epoch{model_ckp.epoch} checkpoint saved successfully at {save_path}"
            )
        except Exception as e:
            logger.error(f"Error occurred while saving checkpoint: {e}")
            raise

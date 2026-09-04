from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import kornia
from models.base_model import BaseModel
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils import networks, path_util
from utils.loss import Fusionloss, cc
from utils.logger_initializer import logger
from schemas.model_checkpoint import ModelCkp
from torch.optim import lr_scheduler


class CDDFuseModel(BaseModel):

    def __init__(self, opt: DictConfig):
        BaseModel.__init__(self, opt)
        self.name = opt.model
        self._epoch = opt.start_epoch
        self._phase = 1 if opt.start_epoch <= opt.epoch_gap else 2

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
            # 损失函数
            self.MSELoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()
            self.Loss_ssim = kornia.losses.SSIMLoss(window_size=opt.SSIM_window_size, reduction="mean")
            self.criteria_fusion = Fusionloss().to(self.device)

            # 优化器
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

            self.scheduler_encoder = networks.get_scheduler(self.optimizer_encoder, opt)
            self.scheduler_decoder = networks.get_scheduler(self.optimizer_decoder, opt)
            self.scheduler_base = networks.get_scheduler(self.optimizer_base, opt)
            self.scheduler_detail = networks.get_scheduler(self.optimizer_detail, opt)
            self.schedulers: dict[str, lr_scheduler.LRScheduler] = {
                "scheduler_encoder": self.scheduler_encoder,
                "scheduler_decoder": self.scheduler_decoder,
                "scheduler_base_fusion": self.scheduler_base,
                "scheduler_detail_fusion": self.scheduler_detail,
            }

    def get_phase(self):
        return self._phase

    def get_epoch(self):
        return self._epoch

    def get_current_loss(self) -> torch.Tensor | None:
        return self.losses.get("loss_total", None)

    def set_input(self, input):
        """从 DataLoader 解包数据。"""
        self.data_VIS, self.data_IR = input
        self.data_VIS, self.data_IR = self.data_VIS.to(self.device), self.data_IR.to(self.device)

    def update_learning_rate(self):
        """根据当前阶段 (Phase) 更新学习率"""
        active_sch_keys = ["scheduler_encoder", "scheduler_decoder"]
        if self._phase == 2:
            active_sch_keys.extend(["scheduler_base_fusion", "scheduler_detail_fusion"])

        for key in active_sch_keys:
            if key in self.schedulers.keys():
                self.schedulers[key].step()

        # 保证学习率不低于 train.yaml 设置的 min_lr
        min_lr = self.opt.get("min_lr", 1e-6)
        for opt_name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"], min_lr)

    def update_state(self, epoch: int):
        """更新模型的训练状态，包括 epoch 和 phase"""
        self._epoch = epoch
        self._phase = 1 if epoch <= self.opt.epoch_gap else 2

    def forward(self):
        """执行前向传播计算。会根据 phase 采取不同的网络拓扑。"""

        if self._phase == 1:
            # Phase I: 自编码重构
            self.feature_VIS_Base, self.feature_VIS_Detail, _ = self.DIDF_Encoder(self.data_VIS)
            self.feature_IR_Base, self.feature_IR_Detail, _ = self.DIDF_Encoder(self.data_IR)

            self.data_VIS_hat, _ = self.DIDF_Decoder(self.data_VIS, self.feature_VIS_Base, self.feature_VIS_Detail)
            self.data_IR_hat, _ = self.DIDF_Decoder(self.data_IR, self.feature_IR_Base, self.feature_IR_Detail)

        elif self._phase == 2:
            # Phase II: 融合任务
            self.feature_VIS_Base, self.feature_VIS_Detail, _ = self.DIDF_Encoder(self.data_VIS)
            self.feature_IR_Base, self.feature_IR_Detail, _ = self.DIDF_Encoder(self.data_IR)

            self.feature_Fuse_Base = self.BaseFuseLayer(self.feature_IR_Base + self.feature_VIS_Base)
            self.feature_Fuse_Detail = self.DetailFuseLayer(self.feature_IR_Detail + self.feature_VIS_Detail)

            self.data_Fuse, _ = self.DIDF_Decoder(self.data_VIS, self.feature_Fuse_Base, self.feature_Fuse_Detail)

    def backward(self):
        """计算损失，并进行反向传播。"""

        # 公共特征分解损失
        self.loss_cc_Base = cc(self.feature_VIS_Base, self.feature_IR_Base)
        self.loss_cc_Detail = cc(self.feature_VIS_Detail, self.feature_IR_Detail)
        self.loss_decomp = (self.loss_cc_Detail**2) / (1.01 + self.loss_cc_Base)

        if self._phase == 1:
            self.loss_MSE_VIS = 5 * self.Loss_ssim(self.data_VIS, self.data_VIS_hat) + self.MSELoss(
                self.data_VIS, self.data_VIS_hat
            )
            self.loss_MSE_IR = 5 * self.Loss_ssim(self.data_IR, self.data_IR_hat) + self.MSELoss(
                self.data_IR, self.data_IR_hat
            )

            spatial_grad = kornia.filters.SpatialGradient()
            self.loss_gradient = self.L1Loss(spatial_grad(self.data_VIS), spatial_grad(self.data_VIS_hat))

            self.loss_total = (
                self.opt.coeff_mse_loss_VF * self.loss_MSE_VIS
                + self.opt.coeff_mse_loss_IF * self.loss_MSE_IR
                + self.opt.coeff_decomp * self.loss_decomp
                + self.opt.coeff_tv * self.loss_gradient
            )
            self.loss_total.backward()

            self.losses |= {
                "loss_MSE_VIS": self.loss_MSE_VIS,
                "loss_MSE_IR": self.loss_MSE_IR,
                "loss_gradient": self.loss_gradient,
            }

        elif self._phase == 2:
            self.loss_fusion, _, _ = self.criteria_fusion(self.data_VIS, self.data_IR, self.data_Fuse)

            self.loss_total = self.loss_fusion + self.opt.coeff_decomp * self.loss_decomp
            self.loss_total.backward()

            self.losses |= {"loss_fusion": self.loss_fusion}

        self.losses |= {
            "loss_total": self.loss_total,
            "loss_cc_Base": self.loss_cc_Base,
            "loss_cc_Detail": self.loss_cc_Detail,
            "loss_decomp": self.loss_decomp,
        }

    def optimize_parameters(self):
        """更新网络权重（每个 iteration 调用一次）。"""
        self.forward()  # 前向传播

        # 根据阶段清空梯度
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        self.backward()  # 反向传播

        # 梯度裁剪及权重更新
        nn.utils.clip_grad_norm_(self.DIDF_Encoder.parameters(), max_norm=self.opt.clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(self.DIDF_Decoder.parameters(), max_norm=self.opt.clip_grad_norm_value, norm_type=2)

        self.optimizer_encoder.step()
        self.optimizer_decoder.step()

        if self._phase == 2:
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
            logger.info(f"Models loaded successfully from checkpoint.")

            if self.isTrain:
                for name, optimizer in self.optimizers.items():
                    optimizer_state = model_ckp.optimizers.get(name)
                    if optimizer_state is None:
                        raise KeyError(f"Checkpoint 中缺少优化器 '{name}' 的参数")
                    optimizer.load_state_dict(optimizer_state)
                logger.info(f"Optimizers loaded successfully from checkpoint.")

                for name, scheduler in self.schedulers.items():
                    scheduler_state = model_ckp.schedulers.get(name)
                    if scheduler_state is None:
                        raise KeyError(f"Checkpoint 中缺少调度器 '{name}' 的参数")
                    scheduler.load_state_dict(scheduler_state)
                logger.info(f"Schedulers loaded successfully from checkpoint.")
        except Exception as e:
            logger.error(f"Error occurred while loading checkpoint: {e}")
            raise

        logger.info(f"{self.name}_phase{model_ckp.phase}_epoch{model_ckp.epoch} loaded successfully from {load_path}.")
        return model_ckp

    def save_model(self, ckp_name: str) -> ModelCkp:
        try:
            save_path = self.save_dir / ckp_name
            path_util.check_file_path(save_path, create=True)

            model_ckp = ModelCkp(
                phase=self._phase,
                epoch=self._epoch,
                models={name: model.state_dict() for name, model in self.models.items()},
                optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
                schedulers={name: scheduler.state_dict() for name, scheduler in self.schedulers.items()},
                config=OmegaConf.to_container(self.opt, resolve=True),
            )

            torch.save(model_ckp.to_dict(), save_path)
            logger.info(
                f"{self.name}_phase{model_ckp.phase}_epoch{model_ckp.epoch} checkpoint saved successfully at {save_path}."
            )
            return model_ckp
        except Exception as e:
            logger.error(f"Error occurred while saving checkpoint: {e}")
            raise

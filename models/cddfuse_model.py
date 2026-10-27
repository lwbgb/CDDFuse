import os
import math
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import kornia
from models.base_model import BaseModel
from models.net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

from utils import networks, path_util
from utils.loss import Fusionloss, cc, LegacySSIMLoss

from utils.logger_initializer import logger
from schemas.model_checkpoint import ModelCkp
from torch.optim import lr_scheduler


class CDDFuseModel(BaseModel):

    def __init__(self, opt: DictConfig):
        BaseModel.__init__(self, opt)
        self.name = opt.model
        self.use_bf16: bool = bool(opt.get("use_bf16", False))
        self.epoch_completed = False
        if opt.isTrain:
            self._epoch = opt.start_epoch
            self._phase = 1 if opt.start_epoch <= opt.epoch_gap else 2
        else:
            self._phase = 2

        self.DIDF_Encoder = Restormer_Encoder().to(self.device)
        self.DIDF_Decoder = Restormer_Decoder().to(self.device)
        self.BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(self.device)
        self.DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(self.device)

        self.models: dict[str, nn.Module] = {
            "DIDF_Encoder": self.DIDF_Encoder,
            "DIDF_Decoder": self.DIDF_Decoder,
            "BaseFuseLayer": self.BaseFuseLayer,
            "DetailFuseLayer": self.DetailFuseLayer,
        }

        if self.isTrain:
            self.MSELoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()
            self.Loss_ssim = LegacySSIMLoss(opt.SSIM_window_size, reduction="mean").to(self.device)
            self.spatial_grad = kornia.filters.SpatialGradient().to(self.device)

            self.criteria_decomp = None
            self.criteria_fusion = Fusionloss().to(self.device)
            
            self.optimizer_encoder = torch.optim.Adam(self.DIDF_Encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizer_decoder = torch.optim.Adam(self.DIDF_Decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizer_base = torch.optim.Adam(self.BaseFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizer_detail = torch.optim.Adam(self.DetailFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
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

    def get_current_loss(self) -> float | None:
        return self.losses.get("loss_total", None)

    def set_input(self, input):
        """解包输入图像"""
        if self.isTrain:
            self.data_IR, self.data_VIS = input
        else:
            self.data_IR, self.data_VIS, self.ir_img_name, self.vis_img_name = input
            self.ir_img_name = self.ir_img_name[0]
            self.vis_img_name = self.vis_img_name[0]
        self.data_IR: torch.Tensor = self.data_IR.to(self.device, non_blocking=True)
        self.data_VIS: torch.Tensor = self.data_VIS.to(self.device, non_blocking=True)

    def update_learning_rate(self):
        """更新学习率"""
        active_sch_keys = ["scheduler_encoder", "scheduler_decoder"]
        if self._phase == 2:
            active_sch_keys.extend(["scheduler_base_fusion", "scheduler_detail_fusion"])

        for key in active_sch_keys:
            if key in self.schedulers.keys():
                self.schedulers[key].step()

        min_lr = self.opt.get("min_lr", 1e-6)
        for opt_name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"], min_lr)
        self.epoch_completed = True

    def update_state(self, epoch: int):
        """更新当前训练状态"""
        self._epoch = epoch
        self.epoch_completed = False
        self._phase = 1 if epoch <= self.opt.epoch_gap else 2
        for scheduler in self.schedulers.values():
            if isinstance(scheduler, networks.PhaseStepLR):
                scheduler.set_phase(self._phase)


    def forward(self):
        """执行前向传播计算"""
        use_amp = self.isTrain and self.use_bf16 and self.device.type == "cuda"
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            if self._phase == 1:
                # Phase I: 自编码重构
                self.feature_VIS_Base, self.feature_VIS_Detail, _ = self.DIDF_Encoder(self.data_VIS)
                self.feature_IR_Base, self.feature_IR_Detail, _ = self.DIDF_Encoder(self.data_IR)

                self.data_VIS_hat, _ = self.DIDF_Decoder(self.data_VIS, self.feature_VIS_Base, self.feature_VIS_Detail)
                self.data_IR_hat, _ = self.DIDF_Decoder(self.data_IR, self.feature_IR_Base, self.feature_IR_Detail)

            elif self._phase == 2:
                # Phase II: 跨模态 MIMO 融合任务
                self.feature_VIS_Base, self.feature_VIS_Detail, _ = self.DIDF_Encoder(self.data_VIS)
                self.feature_IR_Base, self.feature_IR_Detail, _ = self.DIDF_Encoder(self.data_IR)
                
                self.feature_Fuse_Base = self.BaseFuseLayer(self.feature_IR_Base + self.feature_VIS_Base)
                self.feature_Fuse_Detail = self.DetailFuseLayer(self.feature_IR_Detail + self.feature_VIS_Detail)

                self.data_Fuse, _ = self.DIDF_Decoder(self.data_VIS, self.feature_Fuse_Base, self.feature_Fuse_Detail)

    def backward(self):
        """执行反向传播计算"""
        # BF16只覆盖网络前向，相关性、SSIM及其余损失均在FP32中计算。
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            self.losses = {}
            feat_vis_base = self.feature_VIS_Base.float()
            feat_vis_detail = self.feature_VIS_Detail.float()
            feat_ir_base = self.feature_IR_Base.float()
            feat_ir_detail = self.feature_IR_Detail.float()
    
            self.loss_cc_Base = cc(feat_vis_base, feat_ir_base)
            self.loss_cc_Detail = cc(feat_vis_detail, feat_ir_detail)
            self.loss_decomp = (self.loss_cc_Detail ** 2) / (1.01 + self.loss_cc_Base)
    
            if self._phase == 1:
                vis_hat = self.data_VIS_hat.float()
                ir_hat = self.data_IR_hat.float()
    
                self.loss_MSE_VIS = (
                    5 * self.Loss_ssim(self.data_VIS, vis_hat)
                    + self.MSELoss(self.data_VIS, vis_hat)
                )
    
                self.loss_MSE_IR = (
                    5 * self.Loss_ssim(self.data_IR, ir_hat)
                    + self.MSELoss(self.data_IR, ir_hat)
                )
    
                self.loss_gradient: torch.Tensor = self.L1Loss(
                    self.spatial_grad(self.data_VIS), self.spatial_grad(vis_hat)
                )
    
                self.loss_total: torch.Tensor = (
                    self.opt.coeff_mse_loss_VF * self.loss_MSE_VIS
                    + self.opt.coeff_mse_loss_IF * self.loss_MSE_IR
                    + self.opt.coeff_decomp * self.loss_decomp
                    + self.opt.coeff_tv * self.loss_gradient
                )
                self.loss_total.backward()
    
                self.losses |= {
                    "loss_MSE_VIS": self.loss_MSE_VIS.item(),
                    "loss_MSE_IR": self.loss_MSE_IR.item(),
                    "loss_gradient": self.loss_gradient.item(),
                }
    
            elif self._phase == 2:
                fuse_out = self.data_Fuse.float()
                self.loss_fusion, _, _ = self.criteria_fusion(self.data_VIS, self.data_IR, fuse_out)
    
                self.loss_total: torch.Tensor = self.loss_fusion + self.opt.coeff_decomp * self.loss_decomp
                self.loss_total.backward()
                
                self.losses |= {"loss_fusion": self.loss_fusion.item()}
    
            if not torch.isfinite(self.loss_total):
                raise FloatingPointError(f"Non-finite loss at epoch {self._epoch}")
            
            self.losses |= {
                "loss_total": self.loss_total.item(),
                "loss_cc_Base": self.loss_cc_Base.item(),
                "loss_cc_Detail": self.loss_cc_Detail.item(),
                "loss_decomp": self.loss_decomp.item(),
            }

    def optimize_parameters(self):
        """更新网络权重"""
        self.epoch_completed = False
        for model in self.models.values():
            model.train()

        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        self.forward()
        self.backward()

        active: list[tuple[nn.Module, torch.optim.Optimizer]] = [(self.DIDF_Encoder, self.optimizer_encoder), (self.DIDF_Decoder, self.optimizer_decoder)]
        if self._phase == 2:
            active += [(self.BaseFuseLayer, self.optimizer_base), (self.DetailFuseLayer, self.optimizer_detail)]
        for model, _ in active:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.opt.clip_grad_norm_value,
                                    norm_type=2, error_if_nonfinite=True)
        for _, optimizer in active:
            optimizer.step()

    def reset_learning_rate_and_schedulers(self) -> None:
        """保留 Adam 状态，按当前配置设置学习率，并从零重建调度器。"""
        if not self.isTrain:
            raise ValueError("仅训练模式可以重设学习率和调度器")
        overrides: dict = self.opt.get("resume_lrs", {})
        unknown = set(overrides) - set(self.optimizers)
        if unknown:
            raise ValueError(f"resume_lrs 包含未知优化器：{sorted(unknown)}")
        
        min_lr = float(self.opt.get("min_lr", 1e-6))
        rates = {
            name: [float(overrides[name]) if name in overrides else float(group["lr"])
                   for group in optimizer.param_groups]
            for name, optimizer in self.optimizers.items()
        }
        if not math.isfinite(min_lr) or min_lr < 0:
            raise ValueError("min_lr 必须是有限非负数")
        if any(not math.isfinite(lr) or lr <= 0 or (name in overrides and lr < min_lr)
               for name, group_rates in rates.items() for lr in group_rates):
            raise ValueError("重设学习率必须是有限正数且不小于 min_lr")
        
        # 当前训练循环按 epoch 无指标推进，不支持 ReduceLROnPlateau。
        if self.opt.lr_policy not in ("step", "cosine", "linear"):
            raise ValueError("重设调度器仅支持 step、cosine、linear")
        if self.opt.lr_policy == "step":
            phase_settings = networks.phase_step_settings(self.opt)
            if phase_settings is None and (int(self.opt.optim_step) < 1 or not 0 < float(self.opt.optim_gamma) <= 1):
                raise ValueError("optim_step 必须为正整数，optim_gamma 必须在 (0, 1] 内")
        if self.opt.lr_policy == "linear" and int(self.opt.get("epoch_count", 1)) != 1:
            raise ValueError("重设 linear 调度器要求 epoch_count=1")
        
        for name, optimizer in self.optimizers.items():
            for group, lr in zip(optimizer.param_groups, rates[name]):
                # 未配置的模块沿用 checkpoint 当前学习率，并以它作为新调度基准。
                group["lr"] = lr
                group["initial_lr"] = lr
        for suffix, attr in (("encoder", "encoder"), ("decoder", "decoder"),
                             ("base_fusion", "base"), ("detail_fusion", "detail")):
            scheduler = networks.get_scheduler(self.optimizers[f"optimizer_{suffix}"], self.opt, phase=self._phase)
            self.schedulers[f"scheduler_{suffix}"] = scheduler
            setattr(self, f"scheduler_{attr}", scheduler)
        logger.warning(f"续训已重设学习率和调度器（计数从零开始，保留 Adam 动量）：{rates}; "
                       f"policy={self.opt.lr_policy}")

    def load_model(self, ckp_name: str, prefix: str | Path = "") -> ModelCkp:
        """加载模型权重"""
        path = Path(self.opt.checkpoint_root) / prefix / ckp_name
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        model_ckp = ModelCkp.from_dict(checkpoint)
        
        if set(model_ckp.models) != set(self.models):
            raise ValueError("加载的网络模型架构不一致！")
        
        for name, model in self.models.items():
            model.load_state_dict(model_ckp.models[name], strict=True)
        
        if self.isTrain:
            expected_phase = 1 if model_ckp.epoch <= self.opt.epoch_gap else 2
            if set(model_ckp.optimizers) != set(self.optimizers) or set(model_ckp.schedulers) != set(self.schedulers):
                raise ValueError("续训checkpoint必须包含全部优化器与调度器，不能静默跳过")
            if model_ckp.phase != expected_phase:
                raise ValueError("Checkpoint epoch/phase与当前阶段设置不一致")
            
            diff_configs = {}
            keys = ("epoch_gap", "batch_size", "lr", "weight_decay", "lr_policy", "optim_step", "optim_gamma",
                    "min_lr", "use_bf16", "SSIM_window_size", "clip_grad_norm_value", "coeff_mse_loss_VF",
                    "coeff_mse_loss_IF", "coeff_decomp", "coeff_tv", "cudnn_allow_tf32", "matmul_allow_tf32")
            for key in keys:
                if key in model_ckp.config and model_ckp.config[key] != self.opt.get(key):
                    diff_configs[key] = (model_ckp.config[key], self.opt.get(key))
            logger.warning(f"加载checkpoint时存在配置项与当前设置不一致：{diff_configs} ")
            
            reset_schedule = bool(self.opt.get("reset_lr_on_resume", False))
            if not reset_schedule:
                for state in model_ckp.schedulers.values():
                    if not {"last_epoch", "_last_lr"} <= state.keys():
                        raise ValueError("Checkpoint调度器状态不完整")

            for name, optimizer in self.optimizers.items():
                optimizer.load_state_dict(model_ckp.optimizers[name])
            self._epoch, self._phase = model_ckp.epoch, model_ckp.phase
            if reset_schedule:
                self.reset_learning_rate_and_schedulers()
            else:
                saved_options = OmegaConf.merge(self.opt, model_ckp.config)
                for suffix, attr in (("encoder", "encoder"), ("decoder", "decoder"),
                                     ("base_fusion", "base"), ("detail_fusion", "detail")):
                    key = f"scheduler_{suffix}"
                    optimizer = self.optimizers[f"optimizer_{suffix}"]
                    current_lrs = [group["lr"] for group in optimizer.param_groups]
                    state = model_ckp.schedulers[key]
                    if "phase_settings" in state:
                        scheduler = networks.PhaseStepLR(optimizer, state["phase_settings"],
                                                         state["training_phase"])
                    else:
                        for field in ("phase1_optim_step", "phase1_optim_gamma",
                                      "phase2_optim_step", "phase2_optim_gamma"):
                            saved_options[field] = None
                        scheduler = networks.get_scheduler(optimizer, saved_options, phase=self._phase)
                    scheduler.load_state_dict(state)
                    # LambdaLR 构造可能修改 lr，普通恢复必须保留 checkpoint 实际值。
                    for group, lr in zip(optimizer.param_groups, current_lrs):
                        group["lr"] = lr
                    self.schedulers[key] = scheduler
                    setattr(self, f"scheduler_{attr}", scheduler)
                
            self._epoch, self._phase = model_ckp.epoch, model_ckp.phase
            self.epoch_completed = True
            logger.warning("已恢复权重与Adam，调度器按续训配置处理；现有格式未保存RNG/采样状态，不保证逐位精确续训")
        logger.info(f"Loaded checkpoint: {path} (epoch={model_ckp.epoch}, phase={model_ckp.phase})")
        return model_ckp

    def save_model(self, ckp_name: str) -> ModelCkp:
        if not self.isTrain or not self.epoch_completed:
            raise ValueError("仅在完整epoch结束且调度器推进后保存训练checkpoint")
        
        target = self.save_dir / ckp_name
        path_util.check_file_path(target, create=True)
        model_ckp = ModelCkp(
            phase=self._phase, epoch=self._epoch,
            models={name: model.state_dict() for name, model in self.models.items()},
            optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            schedulers={name: scheduler.state_dict() for name, scheduler in self.schedulers.items()},
            config=OmegaConf.to_container(self.opt, resolve=True),
        )
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            torch.save(model_ckp.to_dict(), temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info(f"Checkpoint saved: {target} (epoch={self._epoch})")
        return model_ckp

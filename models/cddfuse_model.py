from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import kornia
from models.base_model import BaseModel
# from models.net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

# [Exp-1 修改 1] 切换为导入 net_exp1
# from models.net_exp1 import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

# [Exp-2 修改 1] 切换为导入 net_exp2
# from models.net_exp2_v2 import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

# [Exp-3 修改 1] 切换为导入 net_exp3
from models.net_exp3 import Restormer_Encoder, Restormer_Decoder, MIMOHybridMambaVisionFusion, CrossModalityDetailFusion

from utils import networks, path_util
from utils.loss import Fusionloss, cc

# [Exp-4 修改 1] 导入 Exp-4 非线性解耦损失与结构自适应融合损失
from utils.loss_exp4 import NonLinearDecouplingLoss, AdaptiveStructureFusionLoss

from utils.logger_initializer import logger
from schemas.model_checkpoint import ModelCkp
from torch.optim import lr_scheduler


class CDDFuseModel(BaseModel):

    def __init__(self, opt: DictConfig):
        BaseModel.__init__(self, opt)
        self.name = opt.model
        self.use_bf16: bool = opt.get("use_bf16", True)
        if opt.isTrain:
            self._epoch = opt.start_epoch
            self._phase = 1 if opt.start_epoch <= opt.epoch_gap else 2
        else:
            self._phase = 2  # 测试阶段直接进入 Phase II

        # 定义网络结构
        self.DIDF_Encoder = Restormer_Encoder().to(self.device)
        self.DIDF_Decoder = Restormer_Decoder().to(self.device)

        # self.BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(self.device)
        # self.DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(self.device)

        # [Exp-3 修改 2] 实例化 MIMO 低频融合层与跨模态极性感知高频融合层    
        self.BaseFuseLayer = MIMOHybridMambaVisionFusion(in_dim=128, out_dim=64, num_heads=8).to(self.device)
        self.DetailFuseLayer = CrossModalityDetailFusion(dim=64, num_layers=1).to(self.device)

        self.models: dict[str, nn.Module] = {
            "DIDF_Encoder": self.DIDF_Encoder,
            "DIDF_Decoder": self.DIDF_Decoder,
            "BaseFuseLayer": self.BaseFuseLayer,
            "DetailFuseLayer": self.DetailFuseLayer,
        }

        if self.isTrain:
            self.MSELoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()
            self.Loss_ssim = kornia.losses.SSIMLoss(window_size=opt.SSIM_window_size, reduction="mean")
            self.spatial_grad = kornia.filters.SpatialGradient().to(self.device)
            self.criteria_fusion = Fusionloss().to(self.device)
            
            # [Exp-4 修改 2] 实例化多阶非线性特征解耦损失与自适应结构融合损失
            self.criteria_decomp = NonLinearDecouplingLoss(
                lambda_hsic=opt.get("lambda_hsic", 0.5),
                lambda_ortho=opt.get("lambda_ortho", 0.1),
            ).to(self.device)
            self.criteria_fusion = AdaptiveStructureFusionLoss(
                grad_weight=10.0,
                sf_weight=opt.get("sf_weight", 1.5),
                cs_weight=opt.get("cs_weight", 0.6),
                max_blend_ratio=opt.get("max_blend_ratio", 0.70),
            ).to(self.device)

            # 优化器
            # self.optimizer_encoder = torch.optim.Adam(
            #     self.DIDF_Encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            # )
            # self.optimizer_decoder = torch.optim.Adam(
            #     self.DIDF_Decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            # )
            # self.optimizer_base = torch.optim.Adam(
            #     self.BaseFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            # )
            # self.optimizer_detail = torch.optim.Adam(
            #     self.DetailFuseLayer.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
            # )
            
            # [Exp-2 修改 2] 使用 Mamba 安全参数分组构建优化器（自动将 A_log, D, dt_proj.bias, norm 的 weight_decay 设为 0）
            self.optimizer_encoder = self._build_optimizer(self.DIDF_Encoder, opt.lr, opt.weight_decay)
            self.optimizer_decoder = self._build_optimizer(self.DIDF_Decoder, opt.lr, opt.weight_decay)
            self.optimizer_base = self._build_optimizer(self.BaseFuseLayer, opt.lr, opt.weight_decay)
            self.optimizer_detail = self._build_optimizer(self.DetailFuseLayer, opt.lr, opt.weight_decay)
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

    @staticmethod
    def _build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        """为含 Mamba 的混合模型构建安全参数分组"""
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in [
                "A_log", "D", "dt_proj.bias", "norm", "temperature",
                "gamma", "scale_ssm", "contrast_logit"
            ]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.Adam(param_groups, lr=lr)

    def get_phase(self):
        return self._phase

    def get_epoch(self):
        return self._epoch

    def get_current_loss(self) -> float | None:
        return self.losses.get("loss_total", None)

    def set_input(self, input):
        """从 DataLoader 解包数据。"""
        if self.isTrain:
            self.data_IR, self.data_VIS = input
        else:
            self.data_IR, self.data_VIS, self.ir_img_name, self.vis_img_name = input
            self.ir_img_name = self.ir_img_name[0]
            self.vis_img_name = self.vis_img_name[0]
        self.data_IR: torch.Tensor = self.data_IR.to(self.device, non_blocking=True)
        self.data_VIS: torch.Tensor = self.data_VIS.to(self.device, non_blocking=True)

    def update_learning_rate(self):
        """根据当前阶段 (Phase) 更新学习率"""
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

    def update_state(self, epoch: int):
        """更新模型的训练状态，当首次切入 Phase II 时温和降低编码器学习率以保护 Phase I 特征流形"""
        prev_phase = self._phase
        self._epoch = epoch
        self._phase = 1 if epoch <= self.opt.epoch_gap else 2


    def forward(self):
        """执行前向传播计算。训练时启用 bfloat16 混合精度加速。"""
        use_amp = self.isTrain and self.use_bf16 and self.device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
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
                
                # self.feature_Fuse_Base = self.BaseFuseLayer(self.feature_IR_Base + self.feature_VIS_Base)
                # self.feature_Fuse_Detail = self.DetailFuseLayer(self.feature_IR_Detail + self.feature_VIS_Detail)

                # [Exp-3 修改 3] 传入双模态特征张量，执行复数 MIMO 融合与极性感知细节融合
                self.feature_Fuse_Base = self.BaseFuseLayer(self.feature_IR_Base, self.feature_VIS_Base)
                self.feature_Fuse_Detail = self.DetailFuseLayer(self.feature_IR_Detail, self.feature_VIS_Detail)

                self.data_Fuse, _ = self.DIDF_Decoder(self.data_VIS, self.feature_Fuse_Base, self.feature_Fuse_Detail)

    def backward(self):
        """计算损失（强制使用 FP32 保证数值精度），并进行反向传播。"""
        feat_vis_b = self.feature_VIS_Base.float()
        feat_vis_d = self.feature_VIS_Detail.float()
        feat_ir_b = self.feature_IR_Base.float()
        feat_ir_d = self.feature_IR_Detail.float()

        # self.loss_cc_Base = cc(feat_vis_b, feat_ir_b)
        # self.loss_cc_Detail = cc(feat_vis_d, feat_ir_d)
        # self.loss_decomp = (self.loss_cc_Detail ** 2) / (1.01 + self.loss_cc_Base)
        
        # 计算多阶非线性解耦损失 (含 CC + Spatial-nHSIC + 模态内正交约束)
        self.loss_decomp, self.loss_cc_Base, self.loss_cc_Detail = self.criteria_decomp(
            feat_vis_b, feat_ir_b, feat_vis_d, feat_ir_d
        )

        if self._phase == 1:
            vis_hat = self.data_VIS_hat.float()
            ir_hat = self.data_IR_hat.float()

            self.loss_MSE_VIS: torch.Tensor = 5 * self.Loss_ssim(self.data_VIS, vis_hat) + self.MSELoss(
                self.data_VIS, vis_hat
            )
            self.loss_MSE_IR: torch.Tensor = 5 * self.Loss_ssim(self.data_IR, ir_hat) + self.MSELoss(
                self.data_IR, ir_hat
            )

            # self.loss_gradient: torch.Tensor = self.L1Loss(
            #     self.spatial_grad(self.data_VIS), self.spatial_grad(vis_hat)
            # )
            
            # [Exp-4 修复] 补齐原仓库缺失的红外图像梯度重构损失，实现双模态对称边缘监督
            self.loss_gradient: torch.Tensor = self.L1Loss(
                self.spatial_grad(self.data_VIS), self.spatial_grad(vis_hat)
            ) + self.L1Loss(
                self.spatial_grad(self.data_IR), self.spatial_grad(ir_hat)
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

        self.losses |= {
            "loss_total": self.loss_total.item(),
            "loss_cc_Base": self.loss_cc_Base.item(),
            "loss_cc_Detail": self.loss_cc_Detail.item(),
            "loss_decomp": self.loss_decomp.item(),
        }

    def optimize_parameters(self):
        """更新网络权重（每个 iteration 调用一次）。"""
        for model in self.models.values():
            model.train()

        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        self.forward()
        self.backward()

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
        """
        [Exp-3 修改 4] 智能权重加载：
        支持直接加载 Exp-2_v2 的 Phase I (epoch_40.pth) 权重热启动 Phase II 训练，
        自动跳过 Phase I 未训练的旧融合层权重冲突。
        """
        try:
            load_path = Path(self.opt.checkpoint_root) / prefix / ckp_name
            path_util.check_file_path(load_path)

            checkpoint: dict = torch.load(load_path, weights_only=True, map_location=self.device)
            model_ckp = ModelCkp.from_dict(checkpoint)

            logger.info(f"Loading {self.name} checkpoint from {load_path} (ckp phase={model_ckp.phase}, epoch={model_ckp.epoch})...")
            is_phase_switch = self.isTrain and (model_ckp.phase == 1) and (self._phase == 2)

            for name, model in self.models.items():
                if is_phase_switch and name in ["BaseFuseLayer", "DetailFuseLayer"]:
                    logger.info(f"[Hot-Start] Skipping Phase-I untrained '{name}' from checkpoint; keeping new Exp-3 initialization.")
                    continue
                model_state = model_ckp.models.get(name)
                if model_state is None:
                    raise KeyError(f"Checkpoint 中缺少模型 '{name}' 的参数")
                model.load_state_dict(model_state)
            logger.info("Models loaded successfully from checkpoint.")

            if self.isTrain:
                for name, optimizer in self.optimizers.items():
                    if is_phase_switch and name in ["optimizer_base_fusion", "optimizer_detail_fusion"]:
                        continue
                    optimizer_state = model_ckp.optimizers.get(name)
                    if optimizer_state is not None:
                        try:
                            optimizer.load_state_dict(optimizer_state)
                        except Exception as opt_e:
                            logger.warning(f"Optimizer '{name}' state dict skipped due to group change: {opt_e}")

                for name, scheduler in self.schedulers.items():
                    if is_phase_switch and name in ["scheduler_base_fusion", "scheduler_detail_fusion"]:
                        continue
                    scheduler_state = model_ckp.schedulers.get(name)
                    if scheduler_state is not None:
                        try:
                            scheduler.load_state_dict(scheduler_state)
                        except Exception as sch_e:
                            logger.warning(f"Scheduler '{name}' state dict skipped: {sch_e}")

        except Exception as e:
            logger.error(f"Error occurred while loading checkpoint: {e}")
            raise

        logger.info(f"Checkpoint (phase={model_ckp.phase}, epoch={model_ckp.epoch}) loaded successfully from {load_path}.")
        return model_ckp

    def save_model(self, ckp_name: str) -> ModelCkp:
        try:
            save_path = Path(self.save_dir) / ckp_name
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
            print(
                f'{self.name}_phase{model_ckp.phase}_epoch{model_ckp.epoch} checkpoint saved successfully at "{save_path}".'
            )
            return model_ckp
        except Exception as e:
            logger.error(f"Error occurred while saving checkpoint: {e}")
            raise

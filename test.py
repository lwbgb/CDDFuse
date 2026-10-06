import os
import time
import warnings
import logging
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import v2
from hydra import initialize, compose

from models.cddfuse_model import CDDFuseModel
from schemas.base_config import BaseConfig
from schemas.test_config import TestConfig
from schemas.custom_dataset import CustomImageDataset
from utils.dataset import get_loader
from utils.img_read_save import img_save
from utils.loss import Fusionloss
from utils.logger_initializer import logger, init_logger
from utils.evaluator_torch import EvaluatorTorch  # 引入我们优化的纯 GPU 并行评估器

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
init_logger("test.log")

if __name__ == '__main__':
    # 恢复多个数据集的泛化性测试
    dataset_names = ["MSRS", "TNO", "RoadScene"]
    
    with initialize(version_base=None, config_path="./configs"):
        opt: BaseConfig | TestConfig = compose(config_name="config", overrides=["+mode@_global_=test"])

    # 模型初始化
    model: CDDFuseModel = CDDFuseModel(opt)
    model.setup()
    model.eval()
    device = model.device
    
    # 实例化损失函数，用于测试阶段的 Loss 跟踪
    criteria_fusion = Fusionloss().to(device)

    for dataset_name in dataset_names:
        test_folder = Path(opt.root) / dataset_name
        test_out_folder = Path(opt.output_dir) / dataset_name
        
        # 如果数据集文件夹不存在，则跳过
        if not test_folder.exists():
            logger.warning(f"Dataset folder {test_folder} not found, skipping...")
            continue

        transform_PIL_to_tensor = v2.Compose([
            v2.ToImage(), 
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        dataset = CustomImageDataset(root=test_folder, ir_dir="ir", vis_dir="vi", transform=transform_PIL_to_tensor)
        dataloader = get_loader(opt, dataset)
        
        logger.info(f"Start evaluating on {dataset_name} (Size: {len(dataset)})...")

        # 指标与 Loss 累加器（防 OOM 流式设计）
        metric_sums = {k: 0.0 for k in ["EN", "SD", "SF", "MI", "SCD", "VIFF", "Qabf", "SSIM"]}
        total_loss = 0.0
        num_processed = 0

        pbar = tqdm(dataloader, desc=f"Testing {dataset_name}", dynamic_ncols=True, leave=False)
        start_time = time.time()

        for i, data in enumerate(pbar):
            if num_processed >= opt.num_test:
                break
            
            # 安全的输入解析：防止 DataLoader 返回 Dict 导致 set_input 异常
            if isinstance(data, dict):
                model.set_input((data["vi"], data["ir"]))
                img_names = data.get("img_name", [f"img_{num_processed+j}" for j in range(len(data["vi"]))])
            else:
                model.set_input(data)
                img_names = [f"img_{num_processed+j}" for j in range(model.data_VIS.size(0))]
            
            # 前向推理 (内部自动封装了 torch.no_grad())
            model.test()

            # 获取当前 Batch 的张量 (在 GPU 上)
            batch_fused: torch.Tensor = model.data_Fuse
            batch_ir = model.data_IR
            batch_vi = model.data_VIS
            B = batch_fused.size(0)

            # ---------------------------------------------------------
            # 1. 实例级归一化 (Instance-wise Normalization)
            # ---------------------------------------------------------
            fused_flat = batch_fused.view(B, -1)
            mins = fused_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            maxs = fused_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            batch_fused_norm = (batch_fused - mins) / (maxs - mins + 1e-8)

            # ---------------------------------------------------------
            # 2. 计算测试损失 (Monitoring Test Loss)
            # ---------------------------------------------------------
            # 这里调用测试集上的 Fusion Loss 来观察损失情况
            with torch.no_grad():
                loss, _, _ = criteria_fusion(batch_vi, batch_ir, batch_fused_norm)
            
            total_loss += loss.item() * B
            pbar.set_postfix({"Test Loss": f"{loss.item():.4f}"})

            # ---------------------------------------------------------
            # 3. GPU 并行计算评估指标 (Batch-wise 流式累加)
            # ---------------------------------------------------------
            metric_sums["EN"] += EvaluatorTorch.EN(batch_fused_norm) * B
            metric_sums["SD"] += EvaluatorTorch.SD(batch_fused_norm) * B
            metric_sums["SF"] += EvaluatorTorch.SF(batch_fused_norm) * B
            metric_sums["MI"] += EvaluatorTorch.MI(batch_fused_norm, batch_ir, batch_vi) * B
            metric_sums["SCD"] += EvaluatorTorch.SCD(batch_fused_norm, batch_ir, batch_vi) * B
            metric_sums["VIFF"] += EvaluatorTorch.VIFF(batch_fused_norm, batch_ir, batch_vi) * B
            metric_sums["Qabf"] += EvaluatorTorch.Qabf(batch_fused_norm, batch_ir, batch_vi) * B
            metric_sums["SSIM"] += EvaluatorTorch.SSIM(batch_fused_norm, batch_ir, batch_vi) * B

            # ---------------------------------------------------------
            # 4. 异步保存融合图像 (非阻塞)
            # ---------------------------------------------------------
            for b in range(B):
                # 仅将单张需要保存的图片转到 CPU
                fi_np = (batch_fused_norm[b].squeeze().cpu().numpy() * 255).astype(np.uint8)
                current_name = img_names[b].split('.')[0]
                img_save(fi_np, current_name, test_out_folder)

            num_processed += B

        # ---------------------------------------------------------
        # 5. 汇总与优化可视化输出 (Logging)
        # ---------------------------------------------------------
        avg_loss = total_loss / max(num_processed, 1)
        avg_metrics = {k: v / max(num_processed, 1) for k, v in metric_sums.items()}
        time_cost = time.time() - start_time

        logger.info("=" * 95)
        logger.info(f"Test Model : {opt.model}")
        logger.info(f"Dataset    : {dataset_name} ({num_processed} imgs) | Time: {time_cost:.1f}s | Avg Loss: {avg_loss:.5f}")
        logger.info("-" * 95)
        logger.info(f"{'Model Name':<15} {'EN':>8} {'SD':>8} {'SF':>8} {'MI':>8} {'SCD':>8} {'VIFF':>8} {'Qabf':>8} {'SSIM':>8}")
        logger.info(f"{opt.model:<15} "
                    f"{avg_metrics['EN']:>8.4f} "
                    f"{avg_metrics['SD']:>8.4f} "
                    f"{avg_metrics['SF']:>8.4f} "
                    f"{avg_metrics['MI']:>8.4f} "
                    f"{avg_metrics['SCD']:>8.4f} "
                    f"{avg_metrics['VIFF']:>8.4f} "
                    f"{avg_metrics['Qabf']:>8.4f} "
                    f"{avg_metrics['SSIM']:>8.4f}")
        logger.info("=" * 95 + "\n")
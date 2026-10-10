import os
import time
import warnings
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from hydra import initialize, compose
from tqdm import tqdm

from models.cddfuse_model import CDDFuseModel
from schemas.base_config import BaseConfig
from schemas.custom_dataset import CustomImageDataset
from schemas.test_config import TestConfig
from utils.evaluator_torch import EvaluatorTorch
from utils.img_read_save import img_save, image_read_cv2
from utils.logger_initializer import logger, init_logger
from utils.loss import Fusionloss

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

init_logger("test.log")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def print_metrics_table(dataset_name: str, model_name: str, metrics: np.ndarray, 
                        ckpt_name: str, total_imgs: int, avg_loss: float, elapsed_time: float):
    """可视化输出终端与日志报告面板"""
    fps = total_imgs / elapsed_time if elapsed_time > 0 else 0.0
    headers = ["EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM"]
    line_sep = "+" + "+".join(["-" * 10 for _ in range(8)]) + "+"
    header_str = "|" + "|".join([f"{h:^10}" for h in headers]) + "|"
    val_str = "|" + "|".join([f"{val:^10.2f}" for val in metrics]) + "|"

    border = "=" * 89
    sub_border = "-" * 89

    report = (
        f"\n{border}\n"
        f"                       TEST & EVALUATION SUMMARY REPORT\n"
        f"{border}\n"
        f"  Model Name       : {model_name.strip()}\n"
        f"  Checkpoint       : {ckpt_name}\n"
        f"  Target Dataset   : {dataset_name}\n"
        f"  Dataset Size     : {total_imgs} images\n"
        f"  Average Loss     : {avg_loss:.4f}\n"
        f"  Total Duration   : {elapsed_time:.2f} s  ({fps:.2f} img/s)\n"
        f"{sub_border}\n"
        f"{line_sep}\n"
        f"{header_str}\n"
        f"{line_sep}\n"
        f"{val_str}\n"
        f"{line_sep}\n"
        f"{border}"
    )
    logger.info(report)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    with initialize(version_base=None, config_path="./configs"):
        opt: BaseConfig | TestConfig = compose(config_name="config", overrides=["+mode@_global_=test"])

    model_name = opt.model
    model: CDDFuseModel = CDDFuseModel(opt)
    model.setup()
    model.eval()
    
    criterion_fusion = Fusionloss().to(model.device)

    datasets = ["MSRS"]

    for dataset_name in datasets:
        test_folder = Path(opt.root) / dataset_name
        test_out_folder = Path(opt.output_dir) / dataset_name
        test_out_folder.mkdir(parents=True, exist_ok=True)

        dataset = CustomImageDataset(root=test_folder, ir_dir="ir", vis_dir="vi")
        total_imgs = len(dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        logger.info(f"==> [{dataset_name}] Starting Inference & Evaluation (Total Dataset Size: {total_imgs} samples)...")

        metric_accum = np.zeros(8)
        loss_accum = 0.0
        start_time = time.time()

        # 2. 结合 tqdm 进度条执行高效推理与评测
        pbar = tqdm(
            dataloader, 
            desc=f"[{dataset_name} | Size: {total_imgs}]", 
            unit="img", 
            dynamic_ncols=True, 
            leave=True
        )

        with torch.inference_mode():
            for i, data in enumerate(pbar):
                model.set_input(data)
                model.test()

                # 计算当前样本的融合损失
                loss_val = criterion_fusion(model.data_VIS, model.data_IR, model.data_Fuse)
                curr_loss = loss_val[0].item() if isinstance(loss_val, (tuple, list)) else loss_val.item()
                loss_accum += curr_loss
                running_avg_loss = loss_accum / (i + 1)

                # GPU 快速 Min-Max 归一化并缩放到 [0, 255]
                fuse_t = model.data_Fuse
                f_min = torch.min(fuse_t)
                f_max = torch.max(fuse_t)
                fuse_norm = (fuse_t - f_min) / (f_max - f_min + 1e-12)
                fuse_255: torch.Tensor = fuse_norm * 255.0

                # 获取图像名称并保存融合结果
                img_name: str = model.ir_img_name
                fi_np = np.squeeze(fuse_255.round().cpu().numpy()).astype(np.uint8)
                img_save(fi_np, img_name.split('.')[0], str(test_out_folder))

                # 读取原图并转入 GPU 计算融合指标
                ir_path = os.path.join(test_folder, "ir", img_name)
                vi_path = os.path.join(test_folder, "vi", img_name)
                ir_np = image_read_cv2(ir_path, 'GRAY')
                vi_np = image_read_cv2(vi_path, 'GRAY')

                ir_t = torch.from_numpy(ir_np).to(model.device)
                vi_t = torch.from_numpy(vi_np).to(model.device)

                curr_metric = EvaluatorTorch.evaluate_all(fuse_255.squeeze(), ir_t, vi_t, device=model.device)
                metric_accum += curr_metric

                # 实时更新进度条后缀显示
                pbar.set_postfix({
                    "Loss": f"{curr_loss:.4f}",
                    "Avg_Loss": f"{running_avg_loss:.4f}"
                })

        pbar.close()
        elapsed_total = time.time() - start_time

        # 3. 计算全局平均指标并格式化输出可视化面板
        final_metrics = metric_accum / total_imgs
        final_avg_loss = loss_accum / total_imgs
        print_metrics_table(
            dataset_name=dataset_name,
            model_name=model_name,
            metrics=final_metrics,
            ckpt_name=opt.ckp_name,
            total_imgs=total_imgs,
            avg_loss=final_avg_loss,
            elapsed_time=elapsed_total
        )
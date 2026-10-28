from __future__ import annotations

import csv
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import hashlib
from pathlib import Path
import re
import sys
import time

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.cddfuse_model import CDDFuseModel
from schemas.custom_dataset import CustomImageDataset
from schemas.test_config import TestConfig
from utils.Evaluator import Evaluator
from utils.img_read_save import image_read_cv2, img_save
from utils.logger_initializer import init_logger, logger
from utils.loss import Fusionloss

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS = ('EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM')
PAPER_METRICS = {
    'MSRS': (6.70, 43.38, 11.56, 3.47, 1.62, 1.05, .69, 1.00),
    'TNO': (7.12, 46.00, 13.15, 2.19, 1.76, .77, .54, 1.03),
    'RoadScene': (7.44, 54.67, 16.36, 2.30, 1.81, .69, .52, .98),
}


def evaluate(opt: TestConfig) -> Path:
    backend = opt.evaluator_backend
    result_dir = Path(opt.output_dir) / opt.model.strip() / opt.load_dir / Path(opt.ckp_name).stem
    result_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(opt, result_dir / 'config.yaml')
    sink = init_logger(result_dir, 'test.log')
    if backend == 'torch':
        from utils.evaluator_torch import EvaluatorTorch
        def evaluate_metrics(fi, ir, vis):
            return EvaluatorTorch.evaluate_all(fi, ir, vis, device=model.device)
    else:
        evaluate_metrics = evaluate_cpu

    try:
        model = CDDFuseModel(opt)
        model.setup()
        model.eval()
        criterion = Fusionloss().to(model.device)
        
        logger.info(f'FP32;TF32 cuDNN/matmul={torch.backends.cudnn.allow_tf32}/{torch.backends.cuda.matmul.allow_tf32};'
                    f'benchmark={torch.backends.cudnn.benchmark}; result_dir={result_dir}')
        logger.info(f'指标后端={backend}; PNG 回读八指标; device={model.device}')
        
        completed_pairs = 0
        datasets = ["MSRS", "TNO", "RoadScene"]
        for dataset_name in datasets[-1:]:
            test_folder = Path(opt.root) / dataset_name
            test_out_folder = result_dir / dataset_name
            test_out_folder.mkdir(parents=True, exist_ok=True)
            
            dataset = CustomImageDataset(root=test_folder, ir_dir="ir", vis_dir="vi")
            total_imgs = len(dataset)
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=opt.num_threads,
                                pin_memory=model.device.type == 'cuda',
                                generator=torch.Generator().manual_seed(0))
            
            total = np.zeros(8, dtype=np.float64)
            loss_total, infer_total = 0.0, 0.0
            started = time.perf_counter()
            
            columns = ['image', 'png', 'png_sha256', 'fusion_loss', 'inference_sec', *METRICS]
            with (test_out_folder / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                
                # 按第一张真实图像预热；不计入计时或指标，避免固定256尺寸干扰benchmark。
                with torch.inference_mode():
                    first = dataset[0]
                    model.set_input((first[0][None], first[1][None], [first[2]], [first[3]]))
                    model.test()
                    if model.device.type == 'cuda':
                        torch.cuda.synchronize(model.device)
        
                with torch.inference_mode(), tqdm(loader, desc=f'[{dataset_name} | Size: {total_imgs}]', unit='img',
                                                  dynamic_ncols=True, leave=True) as pbar:
                    for idx, data in enumerate(pbar):
                        model.set_input(data)
                        if model.device.type == 'cuda':
                            torch.cuda.synchronize(model.device)
                        infer_started = time.perf_counter()
                        model.test()
                        if model.device.type == 'cuda':
                            torch.cuda.synchronize(model.device)
                        
                        seconds = time.perf_counter() - infer_started
                        image = normalize_to_uint8(model.data_Fuse)
                        losses = criterion(model.data_VIS, model.data_IR, model.data_Fuse)
                        loss = float((losses[0] if isinstance(losses, (tuple, list)) else losses).item())
                        
                        if not np.isfinite(loss):
                            raise FloatingPointError(f'{dataset_name}/{model.ir_img_name} 融合 loss 非有限')
                        
                        stem = Path(model.ir_img_name).stem
                        img_save(image, stem, str(test_out_folder))
                        png_path = test_out_folder / f'{stem}.png'
                        # 正式指标基于实际落盘 PNG；原图也按原版 OpenCV 路径读取。
                        fi = image_read_cv2(str(png_path), mode='GRAY')
                        if not np.array_equal(fi, image):
                            raise ValueError(f'PNG 回读像素与保存前不一致：{png_path}')
                        ir, vis = dataset.read_pair(idx)
                        values = np.asarray(evaluate_metrics(fi, ir, vis), dtype=np.float64)
                        if values.shape != (8,) or not np.isfinite(values).all():
                            raise FloatingPointError(f'{dataset_name}/{model.ir_img_name} {backend} 指标无效或非有限')
                        row = {'image': model.ir_img_name, 'png': png_path.name, 'fusion_loss': loss,
                               'inference_sec': seconds, **dict(zip(METRICS, values.tolist()))}
                        # 仅在 CSV 写出时四舍五入到四位小数，汇总仍使用原始精度。
                        for key in ('fusion_loss', 'inference_sec', *METRICS):
                            row[key] = format(Decimal(str(row[key])).quantize(
                                Decimal('0.0001'), rounding=ROUND_HALF_UP), '.4f')
                        writer.writerow(row)
                        stream.flush()
                        
                        total += values
                        loss_total += loss
                        infer_total += seconds
                        pbar.set_postfix(Loss=f'{loss:.4f}', Avg_Loss=f'{loss_total / (idx + 1):.4f}', refresh=False)
            
            elapsed = time.perf_counter() - started
            average = total / total_imgs
            print_metrics_table(dataset_name, str(opt.model), average, Path(opt.ckp_name).stem, total_imgs,
                                loss_total / total_imgs, elapsed, infer_total)
            completed_pairs += total_imgs
            
        logger.info(f'评测完成，共 {completed_pairs} 对，结果：{result_dir}')
    except BaseException as e:
        logger.exception(f'评测异常：{e}')
        raise
    finally:
        logger.remove(sink)
    return result_dir

def normalize_to_uint8(fused: torch.Tensor) -> np.ndarray:
    """逐图 min-max，分母 clamp；截断而非四舍五入，保持已校准口径。"""
    if fused.ndim != 4 or tuple(fused.shape[:2]) != (1, 1):
        raise ValueError(f'融合输出应为 [1,1,H,W]，实际 {tuple(fused.shape)}')
    if not torch.isfinite(fused).all():
        raise FloatingPointError('融合输出包含 NaN/Inf，停止保存与评测')
    fused = fused.float()
    lower, upper = fused.amin(), fused.amax()
    normalized = (fused - lower) / (upper - lower).clamp_min(1e-8)
    return (normalized * 255.0).to(torch.uint8)[0, 0].cpu().numpy()


def evaluate_cpu(fused: np.ndarray, ir: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """输入为 PNG 回读的 0–255 float32；避免 uint8 差分/平方溢出。"""
    arrays = [np.asarray(image, dtype=np.float32) for image in (fused, ir, vis)]
    if any(image.ndim != 2 or not np.isfinite(image).all() for image in arrays):
        raise ValueError('指标输入必须为有限二维图像')
    if not arrays[0].shape == arrays[1].shape == arrays[2].shape:
        raise ValueError('融合图与两模态的尺寸必须一致')
    fi, ir, vis = arrays
    values = np.array([Evaluator.EN(fi), Evaluator.SD(fi), Evaluator.SF(fi),
                       Evaluator.MI(fi, ir, vis), Evaluator.SCD(fi, ir, vis),
                       Evaluator.VIFF(fi, ir, vis), Evaluator.Qabf(fi, ir, vis),
                       Evaluator.SSIM(fi, ir, vis)], dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError('CPU 指标出现 NaN/Inf，停止评测')
    return values


def print_metrics_table(dataset_name: str, model_name: str, metrics: np.ndarray,
                        ckpt_name: str, total_imgs: int, avg_loss: float,
                        elapsed_time: float, pure_infer_time: float):
    """日志中展示实测、论文参考与差值；短程检查只展示实测。"""
    eval_fps = total_imgs / elapsed_time if elapsed_time > 0 else 0.0
    infer_fps = total_imgs / pure_infer_time if pure_infer_time > 0 else 0.0
    infer_ms = pure_infer_time / total_imgs * 1000.0
    line_sep = '+' + '-' * 14 + '+' + '+'.join(['-' * 12 for _ in METRICS]) + '+'
    header = '|' + f'{"Metric":^14}' + '|' + '|'.join(f'{name:^12}' for name in METRICS) + '|'

    def table_row(label: str, values: np.ndarray, signed: bool = False) -> str:
        cells = [f'{value:+12.4f}' if signed else f'{value:12.4f}' for value in values]
        return '|' + f'{label:^14}' + '|' + '|'.join(cells) + '|'

    rows = [table_row('Measured', metrics)]
    paper = np.asarray(PAPER_METRICS[dataset_name], dtype=np.float64)
    rows += [table_row('Paper', paper), table_row('Delta', metrics - paper, signed=True)]
    table = '\n'.join([line_sep, header, line_sep, *rows, line_sep])
    logger.info(
        f'\n{"=" * len(line_sep)}\n'
        f'  Model Name       : {model_name.strip()}\n'
        f'  Checkpoint       : {ckpt_name}\n'
        f'  Target Dataset   : {dataset_name}\n'
        f'  Dataset Size     : {total_imgs} images\n'
        f'  Average Loss     : {avg_loss:.6f}\n'
        f'  Pure Inference   : {pure_infer_time:.2f} s  ({infer_ms:.2f} ms/img | {infer_fps:.2f} FPS)\n'
        f'  Total Eval Time  : {elapsed_time:.2f} s  ({eval_fps:.2f} img/s)\n'
        f'{table}')


if __name__ == '__main__':
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / 'configs')):
            opt = compose(config_name='config', overrides=['+mode@_global_=test', *sys.argv[1:]])
    evaluate(opt)

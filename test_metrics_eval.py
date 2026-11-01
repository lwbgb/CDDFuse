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
from utils.test_metrics import FusionEvaluator
from utils.img_read_save import image_read_cv2, img_save
from utils.logger_initializer import init_logger, logger
from utils.loss import Fusionloss

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS = ('EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM', 'CC', 'AG', 'MSE')



# CDDFuse_IVF.pth 官方权重在相同 test_metrics 默认协议下的统一重测均值，非论文表格值。
# 指标源码 SHA-256: f0dc77f38dc53e9adf3665520be010d7210b3b498ce783cf0bc6ad863dc9e40f
OFFICIAL_METRICS = {
    'MSRS': (6.7006647372749795, 43.37423451512681, 11.556104178745574, 5.0084418764972085, 1.6209795567504304, 1.050955265590018, 0.6928106907550281, 1.4208598840792626, 0.6007133575202376, 3.7472391843182336, 2415.805394412223),
    'TNO': (7.116913319875514, 46.002095644439436, 13.149122084907471, 3.1534889030536717, 1.760473645409709, 0.7682372647099158, 0.5391141983477212, 1.3790759770969268, 0.512536610560724, 4.915306685594809, 3535.2470815223314),
    'RoadScene': (7.430135935525422, 54.660544738303216, 16.357299707566565, 3.316664825858883, 1.805581171446576, 0.694712846698949, 0.5238954616455953, 1.3925549774587649, 0.5927060735535394, 5.836164109737954, 3337.0038794179627),
}
OFFICIAL_COUNTS = {'MSRS': 361, 'TNO': 25, 'RoadScene': 50}

def evaluate(opt: TestConfig) -> Path:
    backend = opt.evaluator_backend
    result_dir = Path(opt.output_dir) / opt.model.strip() / opt.load_dir / Path(opt.ckp_name).stem
    result_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(opt, result_dir / 'config.yaml')
    sink = init_logger(result_dir, 'test.log')
    metric_evaluator = FusionEvaluator()

    def evaluate_metrics(fi: np.ndarray, ir: np.ndarray, vis: np.ndarray) -> np.ndarray:
        return evaluate_cpu(fi, ir, vis, evaluator=metric_evaluator)

    try:
        model = CDDFuseModel(opt)
        model.setup()
        model.eval()
        criterion = Fusionloss().to(model.device)
        
        logger.info(f'FP32;TF32 cuDNN/matmul={torch.backends.cudnn.allow_tf32}/{torch.backends.cuda.matmul.allow_tf32};'
                    f'benchmark={torch.backends.cudnn.benchmark}; result_dir={result_dir}')
        protocol = metric_evaluator.describe()
        metric_path = PROJECT_ROOT / 'utils' / 'test_metrics.py'
        metric_sha256 = hashlib.sha256(metric_path.read_bytes()).hexdigest()
        logger.info(f'指标实现=utils.test_metrics.FusionEvaluator; CPU float64; '
                    f'PNG 回读 {len(METRICS)} 项; device={model.device}; '
                    f'配置 evaluator_backend={backend}，本入口统一使用 test_metrics，不切换旧后端')
        logger.info(f'指标协议={protocol["protocol"]}; 配置={protocol["config"]}; '
                    f'指标源码 SHA-256={metric_sha256}')
        logger.info('双源 MI/VIF/SSIM 按协议求和，CC/MSE 取均值；MI 单位 bit，'
                    'SSIM data_range=255；不与旧论文表格直接计算差值。')
        
        completed_pairs = 0
        datasets = ["MSRS", "TNO", "RoadScene"]
        for dataset_name in datasets[:]:
            test_folder = Path(opt.root) / dataset_name
            test_out_folder = result_dir / dataset_name
            test_out_folder.mkdir(parents=True, exist_ok=True)
            
            dataset = CustomImageDataset(root=test_folder, ir_dir="ir", vis_dir="vi")
            total_imgs = len(dataset)
            logger.info(f'开始评价 {dataset_name}：{total_imgs} 对；图像目录={test_folder}；'
                        f'输出目录={test_out_folder}；指标顺序={", ".join(METRICS)}')
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=opt.num_threads,
                                pin_memory=model.device.type == 'cuda',
                                generator=torch.Generator().manual_seed(0))
            
            total = np.zeros(len(METRICS), dtype=np.float64)
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
                        if values.shape != (len(METRICS),) or not np.isfinite(values).all():
                            raise FloatingPointError(f'{dataset_name}/{model.ir_img_name} test_metrics 指标无效或非有限')
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


def evaluate_cpu(fused: np.ndarray, ir: np.ndarray, vis: np.ndarray,
                 *, evaluator: FusionEvaluator | None = None) -> np.ndarray:
    """沿用原图/PNG读取路径，直接调用 test_metrics 的 float64 十一指标。"""
    evaluator = evaluator if evaluator is not None else FusionEvaluator()
    # 新 API 的参数顺序为 IR、VIS、融合图，与旧 Evaluator 不同。
    result = evaluator.evaluate(ir, vis, fused, metrics=METRICS)
    values = np.asarray([result.values[name] for name in METRICS], dtype=np.float64)
    if not np.isfinite(values).all():
        invalid = {name: result.issues.get(name, '非有限值')
                   for name, value in zip(METRICS, values) if not np.isfinite(value)}
        raise FloatingPointError(f'test_metrics 指标未定义或非有限：{invalid}')
    return values


def print_metrics_table(dataset_name: str, model_name: str, metrics: np.ndarray,
                        ckpt_name: str, total_imgs: int, avg_loss: float,
                        elapsed_time: float, pure_infer_time: float):
    """横向展示均值、相对官方统一重测的差值及官方均值。"""
    eval_fps = total_imgs / elapsed_time if elapsed_time > 0 else 0.0
    infer_fps = total_imgs / pure_infer_time if pure_infer_time > 0 else 0.0
    infer_ms = pure_infer_time / total_imgs * 1000.0
    line_sep = '+' + '-' * 12 + '+' + '+'.join('-' * 12 for _ in METRICS) + '+'
    header = '|' + f'{"Metric":^12}' + '|' + '|'.join(
        f'{name + (" ↓" if name == "MSE" else " ↑"):^12}' for name in METRICS) + '|'

    def table_row(label: str, values: np.ndarray, signed: bool = False) -> str:
        cells = [f'{value:+12.4f}' if signed else f'{value:12.4f}' for value in values]
        return '|' + f'{label:^12}' + '|' + '|'.join(cells) + '|'

    rows = [table_row('Mean', metrics)]
    official = OFFICIAL_METRICS.get(dataset_name)
    comparable = official is not None and total_imgs == OFFICIAL_COUNTS[dataset_name]
    if comparable:
        baseline = np.asarray(official, dtype=np.float64)
        rows += [table_row('Delta', metrics - baseline, signed=True),
                 table_row('Official', baseline)]
    else:
        rows.append('|' + f'{"Delta":^12}' + '|' + '|'.join(f'{"N/A":>12}' for _ in METRICS) + '|')
        logger.warning(f'{dataset_name} 当前配对数 {total_imgs} 与官方完整测试集'
                       f' {OFFICIAL_COUNTS.get(dataset_name, "未知")} 不一致或缺少参考值，差值显示 N/A。')
    table = '\n'.join([line_sep, header, line_sep, *rows, line_sep])
    logger.info(
        f'\n{"=" * len(line_sep)}\n'
        f'  模型             : {model_name.strip()}\n'
        f'  权重             : {ckpt_name}\n'
        f'  测试集           : {dataset_name}\n'
        f'  有效配对         : {total_imgs}/{total_imgs}（逐图算分后取均值）\n'
        f'  平均融合损失     : {avg_loss:.6f}\n'
        f'  纯推理耗时       : {pure_infer_time:.2f} s  ({infer_ms:.2f} ms/img | {infer_fps:.2f} FPS)\n'
        f'  总评价耗时       : {elapsed_time:.2f} s  ({eval_fps:.2f} img/s)\n'
        f'  官方参考         : CDDFuse_IVF.pth 同口径统一重测（非论文表格值）\n'
        f'  差值定义         : Delta = Mean - Official；MSE 负值更好，其余指标正值更好\n'
        f'  参考样本数       : {OFFICIAL_COUNTS.get(dataset_name, "未知")} 对；同数量仍需确保使用相同测试集\n'
        f'{table}')


if __name__ == '__main__':
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / 'configs')):
            opt = compose(config_name='config', overrides=['+mode@_global_=test', *sys.argv[1:]])
    evaluate(opt)

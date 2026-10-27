import os
import random
from pathlib import Path

# 保留旧 H5 读取配置；必须在导入 h5py 前设置。
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


class H5Dataset(Dataset):
    """读取配对 float32 数组；IR/VIS 返回顺序与现有训练入口保持一致。"""

    def __init__(self, h5file_path):
        self.h5file_path = str(h5file_path)
        with h5py.File(self.h5file_path, "r") as f:
            ir, vis = f["ir_patchs"], f["vis_patchs"]
            if not isinstance(ir, h5py.Dataset) or not isinstance(vis, h5py.Dataset):
                raise ValueError("H5 必须使用配对数组布局 [N,1,H,W]")
            if ir.shape != vis.shape or ir.ndim != 4 or ir.shape[1] != 1 or min(ir.shape) <= 0:
                raise ValueError(f"H5 配对维度不正确：IR={ir.shape}，VIS={vis.shape}")
            if ir.dtype != np.dtype("float32") or vis.dtype != np.dtype("float32"):
                raise ValueError("H5 patch 必须为 float32")
            self.dataset_len = ir.shape[0]
        self.h5f = None
        self._owner_pid = None

    def __len__(self):
        return self.dataset_len

    def close(self) -> None:
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None
        self._owner_pid = None

    def __getstate__(self):
        # spawn worker 不能序列化打开的 HDF5 句柄；原进程句柄仍可继续用。
        state = self.__dict__.copy()
        state["h5f"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self):
        handle = getattr(self, "h5f", None)
        if handle is not None:
            try:
                handle.close()
            except (AttributeError, RuntimeError):
                pass  # 解释器退出时 HDF5 资源可能已经释放。

    def __getitem__(self, index):
        if self.h5f is None or self._owner_pid != os.getpid():
            # fork 后也重新打开，避免父子进程共用同一个 HDF5 句柄。
            self.close()
            self.h5f = h5py.File(self.h5file_path, "r")
            self._owner_pid = os.getpid()
        ir = self.h5f["ir_patchs"][index]
        vis = self.h5f["vis_patchs"][index]
        for name, values in (("IR", ir), ("VIS", vis)):
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"H5 patch {index} 的 {name} 不为 [0,1] 有限值")
        return torch.from_numpy(ir), torch.from_numpy(vis)


def is_low_contrast(image, fraction_threshold=0.1):
    lower, upper = np.percentile(image, [10, 90])
    # 原预处理对全黑区域会产生 0/0；在此明确判为低对比度。
    return upper <= 0 or (upper - lower) / upper < fraction_threshold


class PairedRandomCropDataset(Dataset):
    """从配对原图采样；每次返回两个 [1, S, S] float32 张量。

    每个 epoch 共采样 samples_per_epoch 次，原图采用均匀有放回抽样。
    启用对比度筛选时，不合格样本重新抽取原图和坐标；达到上限则报错。
    """

    def __init__(self, image_root, ir_dir="ir", vis_dir="vi", crop_size=128,
                 samples_per_epoch=6000, augment=False,
                 filter_low_contrast=True, max_crop_attempts=20, seed=3407):
        self.crop_size = int(crop_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_crop_attempts = int(max_crop_attempts)
        self.augment = bool(augment)
        self.filter_low_contrast = bool(filter_low_contrast)
        self.seed = int(seed)
        if min(self.crop_size, self.samples_per_epoch, self.max_crop_attempts) <= 0:
            raise ValueError("crop_size、samples_per_epoch、max_crop_attempts 必须为正整数")

        root = Path(image_root)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        def image_files(folder):
            if not folder.is_dir():
                raise FileNotFoundError(f"图像目录不存在：{folder}")
            return {p.name: p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions}

        ir_files = image_files(root / ir_dir)
        vis_files = image_files(root / vis_dir)
        if not ir_files:
            raise ValueError("红外图像目录中没有可用图像")
        if ir_files.keys() != vis_files.keys():
            missing_vis = sorted(ir_files.keys() - vis_files.keys())[:5]
            missing_ir = sorted(vis_files.keys() - ir_files.keys())[:5]
            raise ValueError(f"配对文件名不一致：缺少 VIS={missing_vis}；缺少 IR={missing_ir}")
        self.pairs = [(ir_files[name], vis_files[name]) for name in sorted(ir_files)]

        # 与 persistent_workers 中的副本共享 epoch，避免只更新主进程。
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        self._epoch[0] = int(epoch)

    @staticmethod
    def _read_pair(ir_path, vis_path):
        with Image.open(ir_path) as image:
            ir = np.array(image)
        if ir.ndim != 2 or ir.dtype != np.uint8:
            raise ValueError(f"红外输入必须是 8-bit 单通道原图：{ir_path}")
        with Image.open(vis_path) as image:
            vis_raw = np.array(image)
            if vis_raw.dtype != np.uint8:
                raise ValueError(f"可见光输入必须是 8-bit 原图：{vis_path}")
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        ir = ir.astype(np.float32) / 255.0
        # 与 data_process.py 一致，不采用量化后的灰度转换。
        vis = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if ir.shape != vis.shape:
            raise ValueError(f"配对图像尺寸不一致：{ir_path.name}")
        return ir, vis

    def __getitem__(self, index):
        # 独立于 worker 调度的随机流；同 seed/epoch/index 可复现。
        sample_seed = (self.seed + int(self._epoch[0]) * len(self) + int(index)) % (2**63 - 1)
        rng = torch.Generator().manual_seed(sample_seed)

        def randint(high):
            return torch.randint(high, (), generator=rng).item()

        size = self.crop_size
        attempts = self.max_crop_attempts if self.filter_low_contrast else 1
        for _ in range(attempts):
            ir_path, vis_path = self.pairs[randint(len(self.pairs))]
            ir, vis = self._read_pair(ir_path, vis_path)
            height, width = ir.shape
            if min(height, width) < size:
                raise ValueError(f"{ir_path.name} 尺寸 {ir.shape} 小于 crop_size={size}；不自动缩放")
            top = randint(height - size + 1)
            left = randint(width - size + 1)
            ir = ir[top:top + size, left:left + size]
            vis = vis[top:top + size, left:left + size]
            if self.filter_low_contrast and (is_low_contrast(ir) or is_low_contrast(vis)):
                continue
            if self.augment:
                if randint(2):
                    ir, vis = np.flip(ir, 1), np.flip(vis, 1)
                if randint(2):
                    ir, vis = np.flip(ir, 0), np.flip(vis, 0)
            return (torch.from_numpy(np.ascontiguousarray(ir)).unsqueeze(0),
                    torch.from_numpy(np.ascontiguousarray(vis)).unsqueeze(0))
        raise RuntimeError(
            f"连续 {attempts} 次未采到满足对比度条件的图像对；"
            "请检查数据，或显式调整 max_crop_attempts / filter_low_contrast。"
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(opt: DictConfig, dataset: Dataset) -> DataLoader:
    num_workers = int(opt.get("num_threads", 4))
    seed = opt.seed
    # worker 建立/重建会消耗 RNG；分开随机流，防止它改变下一 epoch 的 shuffle。
    sampler_generator = torch.Generator().manual_seed(seed)
    worker_generator = torch.Generator().manual_seed(seed + 1)
    sampler = RandomSampler(dataset, generator=sampler_generator) if opt.shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=opt.drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=worker_generator,
    )

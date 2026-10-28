"""测试图像严格同名配对，保持原版 OpenCV float32 灰度转换。"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.img_read_save import image_read_cv2

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def image_names(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f'测试图像目录不存在：{directory}')
    names = sorted(p.name for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    # Windows 文件名大小写不敏感；跨平台运行也必须保持同一配对结果。
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(f'图像文件名大小写冲突：{directory}')
    return names


class CustomImageDataset(Dataset):
    def __init__(self, root: str | Path, ir_dir: str, vis_dir: str, transform=None):
        self._root = Path(root)
        self._ir_path, self._vis_path = self._root / ir_dir, self._root / vis_dir
        self._ir_images, self._vis_images = image_names(self._ir_path), image_names(self._vis_path)
        ir_names, vis_names = set(self._ir_images), set(self._vis_images)
        if ir_names != vis_names:
            raise ValueError(f'IR/VIS 必须严格同名配对；缺少 VIS：{sorted(ir_names - vis_names)[:5]}；'
                             f'缺少 IR：{sorted(vis_names - ir_names)[:5]}')
        if not self._ir_images:
            raise ValueError(f'测试集不能为空：{self._root}')
        # 保存为 PNG 后去掉原扩展名；防止 a.jpg/a.png 覆盖同一个结果。
        stems = [Path(name).stem.casefold() for name in self._ir_images]
        if len(set(stems)) != len(stems):
            raise ValueError(f'图像去扩展名后重名，无法安全保存 PNG：{self._root}')
        self._items = [(name, name) for name in self._ir_images]
        self.transform = transform  # 保留既有接口；正式评测固定灰度归一化，不引入随机变换。
        for name in self._ir_images:
            ir, vis = (cv2.imread(str(folder / name), cv2.IMREAD_UNCHANGED)
                       for folder in (self._ir_path, self._vis_path))
            if ir is None or vis is None:
                raise ValueError(f'图像读取失败：{self._root}/{name}')
            if ir.dtype != np.uint8 or vis.dtype != np.uint8:
                raise ValueError(f'评测输入必须为 8bit 图像：{self._root}/{name}')
            if ir.shape[:2] != vis.shape[:2]:
                raise ValueError(f'IR/VIS 空间尺寸不一致：{self._root}/{name}')

    def __len__(self) -> int:
        return len(self._items)

    def read_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """直接读取 0–255 float32，供 PNG 回读后的 CPU 指标使用。"""
        ir_name, vis_name = self._items[idx]
        ir = image_read_cv2(str(self._ir_path / ir_name), mode='GRAY')
        vis = image_read_cv2(str(self._vis_path / vis_name), mode='GRAY')
        if ir.shape != vis.shape or ir.ndim != 2:
            raise ValueError(f'IR/VIS 灰度尺寸不一致：{ir_name}/{vis_name}')
        return ir, vis

    def __getitem__(self, idx: int):
        ir, vis = self.read_pair(idx)
        ir_name, vis_name = self._items[idx]
        return (torch.from_numpy(ir / 255.0).unsqueeze(0),
                torch.from_numpy(vis / 255.0).unsqueeze(0), ir_name, vis_name)

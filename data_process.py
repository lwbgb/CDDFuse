"""生成原 CDDFuse 规则的配对 patch；写入成功前不触碰已有数据。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig
from skimage.io import imread
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'}


def get_img_file(file_name: str | Path) -> list[str]:
    """只收集图片；排序保持原代码的路径字典序，不混入附属文件。"""
    folder = Path(file_name)
    if not folder.is_dir():
        raise FileNotFoundError(f'图像目录不存在：{folder}')
    return sorted(str(p) for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def paired_files(image_root: Path) -> list[tuple[str, Path, Path]]:
    """按相对文件名配对；子目录不同也不能因 basename 相同而错误匹配。"""
    ir_root, vis_root = image_root / 'ir', image_root / 'vi'
    ir = {Path(p).relative_to(ir_root).as_posix(): Path(p) for p in get_img_file(ir_root)}
    vis = {Path(p).relative_to(vis_root).as_posix(): Path(p) for p in get_img_file(vis_root)}
    if not ir or not vis:
        raise ValueError(f'训练图像不能为空：{image_root}')
    if ir.keys() != vis.keys():
        raise ValueError(f'配对文件名不一致：缺少 VIS={sorted(ir.keys() - vis.keys())[:5]}；'
                         f'缺少 IR={sorted(vis.keys() - ir.keys())[:5]}')
    return [(name, ir[name], vis[name]) for name in sorted(ir)]


def rgb2y(img: np.ndarray) -> np.ndarray:
    # 保持原版 float32 RGB→Y 运算顺序，不能换成量化灰度转换。
    return img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} 必须为正整数')
    return int(value)


def Im2Patch(img: np.ndarray, win: int, stride: int = 1) -> np.ndarray:
    """返回 [C,win,win,N]；先遍历行位置，再遍历列位置，与原版一致。"""
    win, stride = positive_int(win, 'crop_size'), positive_int(stride, 'stride')
    if img.ndim != 3 or img.shape[0] == 0 or min(img.shape[1:]) < win:
        raise ValueError(f'裁块输入必须为 [C,H,W]，且 H/W 不小于 {win}，实际为 {img.shape}')
    # 滑动窗口只是视图；仅复制最终 patch，省去原先 win*win 次 Python 循环。
    windows = np.lib.stride_tricks.sliding_window_view(img, (win, win), axis=(1, 2))
    windows = windows[:, ::stride, ::stride, :, :]
    patches = windows.transpose(0, 3, 4, 1, 2).reshape(img.shape[0], win, win, -1)
    return np.ascontiguousarray(patches, dtype=np.float32)


def is_low_contrast(image: np.ndarray, fraction_threshold: float = 0.1,
                    lower_percentile: int = 10, upper_percentile: int = 90) -> bool:
    """维持原百分位筛选；全黑块明确剔除，避免 0/0。"""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    if limits[1] <= 0:
        return True
    return bool((limits[1] - limits[0]) / limits[1] < fraction_threshold)


def read_pair(ir_path: Path, vis_path: Path, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    ir, vis = imread(ir_path), imread(vis_path)
    if ir.dtype != np.uint8 or ir.ndim != 2:
        raise ValueError(f'红外必须为 8-bit 单通道原图：{ir_path}，实际 {ir.dtype}/{ir.shape}')
    if vis.dtype != np.uint8 or vis.ndim != 3 or vis.shape[2] != 3:
        raise ValueError(f'可见光必须为 8-bit RGB 原图：{vis_path}，实际 {vis.dtype}/{vis.shape}')
    if ir.shape != vis.shape[:2]:
        raise ValueError(f'配对尺寸不一致：{ir_path.name}，IR={ir.shape}，VIS={vis.shape}')
    if min(ir.shape) < crop_size:
        raise ValueError(f'图像尺寸 {ir.shape} 小于 crop_size={crop_size}：{ir_path}')
    ir_float = ir.astype(np.float32)[None, :, :] / 255.0
    vis_float = vis.astype(np.float32).transpose(2, 0, 1) / 255.0
    return ir_float, rgb2y(vis_float)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_h5(path: Path, expected_count: int) -> None:
    """发布前重新打开临时文件，检查配对形状、数据范围和清单计数。"""
    with h5py.File(path, 'r') as f:
        ir, vis = f['ir_patchs'], f['vis_patchs']
        if ir.shape != vis.shape or ir.ndim != 4 or ir.shape[1] != 1 or len(ir) != expected_count or not len(ir):
            raise ValueError('H5 配对维度或有效 patch 数量不正确')
        for array in (ir, vis):
            if array.dtype != np.dtype('float32'):
                raise ValueError('H5 必须保存 float32 patch')
            for offset in range(0, len(array), 128):
                values = array[offset:offset + 128]
                if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                    raise ValueError('H5 patch 必须为 [0,1] 内的有限值')
        manifest = json.loads(f['preprocess_manifest'].asstr()[()])
        if manifest['patch_count'] != expected_count or sum(s['accepted'] for s in manifest['sources']) != expected_count:
            raise ValueError('预处理清单与 patch 数量不一致')


def build_dataset(opt: DictConfig) -> Path:
    """生成数组 H5。overwrite=true 才允许原子替换同名文件。"""
    size, stride = positive_int(opt.crop_size, 'crop_size'), positive_int(opt.stride, 'stride')
    name = str(opt.name)
    if not name or name in {'.', '..'} or Path(name).name != name:
        raise ValueError('name 必须为单个数据集目录名')
    overwrite = opt.get('overwrite', False)
    if not isinstance(overwrite, bool):
        raise ValueError('overwrite 必须为布尔值')
    # 相对路径以项目目录为基准，从其他工作目录调用也不会误读/误写。
    image_root = PROJECT_ROOT / Path(opt.root) / name
    output_dir = PROJECT_ROOT / Path(opt.get('output_dir', 'data'))
    output = output_dir / f'{name}_imgsize_{size}_stride_{stride}.h5'
    if not overwrite and (output.exists() or output.is_symlink()):
        raise FileExistsError(f'数据文件已存在：{output}。需要重建时显式设置 overwrite=true。')
    if output.is_dir():
        raise IsADirectoryError(output)
    pairs = paired_files(image_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output_dir)
    os.close(fd)
    temporary = Path(temp_name)
    count, rejected, sources = 0, 0, []
    try:
        with h5py.File(temporary, 'w') as f:
            # 一个 chunk 保存一对中的单张 patch，适合后续随机读取；不改变像素。
            arrays = [f.create_dataset(key, shape=(0, 1, size, size), maxshape=(None, 1, size, size),
                                       dtype=np.float32, chunks=(1, 1, size, size))
                      for key in ('ir_patchs', 'vis_patchs')]
            for relative, ir_path, vis_path in tqdm(pairs, desc=f'裁剪{name}'):
                ir, vis = read_pair(ir_path, vis_path, size)
                ir_patches, vis_patches = Im2Patch(ir, size, stride), Im2Patch(vis, size, stride)
                indices = [i for i in range(ir_patches.shape[-1])
                           if not (is_low_contrast(ir_patches[0, :, :, i]) or is_low_contrast(vis_patches[0, :, :, i]))]
                accepted = len(indices)
                removed = ir_patches.shape[-1] - accepted
                sources.append({'name': relative, 'ir_sha256': file_sha256(ir_path), 'vis_sha256': file_sha256(vis_path),
                                'shape': list(ir.shape[1:]), 'accepted': accepted, 'rejected': removed})
                if accepted:
                    for array, patches in zip(arrays, (ir_patches, vis_patches)):
                        array.resize(count + accepted, axis=0)
                        array[count:count + accepted] = patches[..., indices].transpose(3, 0, 1, 2)
                    count += accepted
                rejected += removed
            if not count:
                raise ValueError('没有通过低对比度筛选的 patch，未生成训练文件')
            # 元数据与 patch 同文件提交，不产生 H5/外部清单两份文件不同步的问题。
            manifest = {'format_version': 1, 'dataset': name, 'crop_size': size, 'stride': stride,
                        'normalization': 'float32 / 255', 'rgb2y': [0.299, 0.587, 0.114],
                        'contrast': {'fraction_threshold': 0.1, 'percentiles': [10, 90]},
                        'order': 'relative_filename_then_row_major', 'source_count': len(pairs),
                        'patch_count': count, 'rejected_count': rejected, 'sources': sources}
            f.create_dataset('preprocess_manifest', data=json.dumps(manifest, ensure_ascii=False),
                             dtype=h5py.string_dtype('utf-8'))
        validate_h5(temporary, count)
        if overwrite:
            os.replace(temporary, output)
        else:
            # 同目录硬链接原子发布且不覆盖：即使检查后出现同名文件，也会报错。
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'预处理完成：{len(pairs)} 对原图，保留 {count} 对 patch，剔除 {rejected} 对。输出：{output}')
    return output


def main() -> None:
    with initialize(version_base=None, config_path='./configs'):
        opt = compose(config_name='dataset', overrides=sys.argv[1:])
    build_dataset(opt)


if __name__ == '__main__':
    main()

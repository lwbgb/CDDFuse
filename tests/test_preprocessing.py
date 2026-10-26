"""数据阶段回归检查：像素/顺序、输入约束、原子发布和 worker 随机流。"""
import gc
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

import data_process as preprocessing

ROOT = Path(__file__).resolve().parents[1]
# 直接加载文件，不触发 utils/__init__.py 中与本次数据检查无关的日志初始化。
spec = importlib.util.spec_from_file_location('preprocessing_h5_reader', ROOT / 'utils/dataset.py')
reader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reader
spec.loader.exec_module(reader)
torch.set_num_threads(2)


def config(root: Path, **overrides):
    return OmegaConf.create({'root': str(root / 'raw'), 'name': 'demo', 'crop_size': 8, 'stride': 6,
                             'output_dir': str(root / 'out'), 'overwrite': False, 'progress': False, **overrides})


def pair(root: Path, name: str = 'a.png', seed: int = 7):
    rng = np.random.default_rng(seed)
    ir = rng.integers(0, 256, (20, 23), dtype=np.uint8)
    vis = rng.integers(0, 256, (20, 23, 3), dtype=np.uint8)
    save_pair(root, name, ir, vis)
    return ir, vis


def save_pair(root, name, ir, vis):
    for folder, values in [('ir', ir), ('vi', vis)]:
        path = root / 'raw/demo' / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(values).save(path)


def legacy_patches(image, size, stride):
    # 独立逐位置切片作为顺序参考，不使用被测的滑动窗口实现。
    blocks = [image[:, top:top + size, left:left + size]
              for top in range(0, image.shape[1] - size + 1, stride)
              for left in range(0, image.shape[2] - size + 1, stride)]
    return np.stack(blocks, axis=-1).astype(np.float32)


def write_h5(path, count=10):
    values = np.broadcast_to(np.arange(count, dtype=np.float32)[:, None, None, None] / count,
                             (count, 1, 8, 8)).copy()
    with h5py.File(path, 'w') as f:
        f['ir_patchs'] = values
        f['vis_patchs'] = values + np.float32(.01)


class PreprocessingTests(unittest.TestCase):
    def test_window_pixels_and_row_major_order_match_legacy(self):
        rng = np.random.default_rng(3407)
        for channels, height, width, size, stride in [(1, 20, 23, 8, 6), (3, 33, 29, 7, 4),
                                                     (1, 128, 128, 128, 200), (1, 529, 641, 128, 200)]:
            image = rng.random((channels, height, width), dtype=np.float32)
            np.testing.assert_array_equal(preprocessing.Im2Patch(image, size, stride),
                                          legacy_patches(image, size, stride))

    def test_full_output_values_filtering_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = {}
            for name, seed in [('nested/b.png', 1), ('a.png', 2), ('black.png', 3)]:
                ir, vis = pair(root, name, seed)
                if name == 'a.png':
                    ir[:8, :8] = 0
                if name == 'black.png':
                    ir[:] = 0
                save_pair(root, name, ir, vis)
                images[name] = (ir, vis)
            for folder in ['ir', 'vi']:
                (root / 'raw/demo' / folder / 'a.png:Zone.Identifier').write_text('metadata')
            path = preprocessing.build_dataset(config(root))
            expected_ir, expected_vis = [], []
            rejected = 0
            for name in sorted(images):
                raw_ir, raw_vis = images[name]
                ir = raw_ir.astype(np.float32)[None] / 255.0
                rgb = raw_vis.astype(np.float32).transpose(2, 0, 1) / 255.0
                vis = rgb[:1] * .299000 + rgb[1:2] * .587000 + rgb[2:3] * .114000
                ir_blocks, vis_blocks = legacy_patches(ir, 8, 6), legacy_patches(vis, 8, 6)
                for i in range(ir_blocks.shape[-1]):
                    accepted = True
                    for block in [ir_blocks[0, :, :, i], vis_blocks[0, :, :, i]]:
                        lower, upper = np.percentile(block, [10, 90])
                        if upper <= 0 or (upper - lower) / upper < .1:
                            accepted = False
                    if accepted:
                        expected_ir.append(ir_blocks[..., i])
                        expected_vis.append(vis_blocks[..., i])
                    else:
                        rejected += 1
            with h5py.File(path, 'r') as f:
                np.testing.assert_array_equal(f['ir_patchs'][:], np.stack(expected_ir))
                np.testing.assert_array_equal(f['vis_patchs'][:], np.stack(expected_vis))
                self.assertEqual(f['ir_patchs'].chunks, (1, 1, 8, 8))
                manifest = json.loads(f['preprocess_manifest'].asstr()[()])
                self.assertEqual(manifest['patch_count'], len(expected_ir))
                self.assertEqual(manifest['rejected_count'], rejected)
                self.assertEqual([s['name'] for s in manifest['sources']], sorted(images))
                self.assertEqual(manifest['sources'][1]['accepted'], 0)
                self.assertTrue(all(len(s['ir_sha256']) == 64 for s in manifest['sources']))

    def test_pair_names_empty_input_and_nonimage_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair(root)
            (root / 'raw/demo/vi/a.png').rename(root / 'raw/demo/vi/b.png')
            with self.assertRaisesRegex(ValueError, '配对文件名'):
                preprocessing.build_dataset(config(root))
            for p in (root / 'raw/demo').rglob('*.png'):
                p.unlink()
            (root / 'raw/demo/ir/readme.txt').write_text('not an image')
            with self.assertRaisesRegex(ValueError, '不能为空'):
                preprocessing.build_dataset(config(root))
            self.assertFalse((root / 'out').exists())

    def test_reject_channels_dtype_shape_and_small_images(self):
        cases = [(np.zeros((20, 23, 3), np.uint8), np.zeros((20, 23, 3), np.uint8)),
                 (np.zeros((20, 23), np.uint8), np.zeros((20, 23), np.uint8)),
                 (np.zeros((20, 23), np.uint16), np.zeros((20, 23, 3), np.uint8)),
                 (np.zeros((19, 23), np.uint8), np.zeros((20, 23, 3), np.uint8)),
                 (np.zeros((7, 23), np.uint8), np.zeros((7, 23, 3), np.uint8))]
        for ir, vis in cases:
            with self.subTest(shape=(ir.shape, vis.shape), dtype=str(ir.dtype)), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                save_pair(root, 'a.tiff', ir, vis)
                with self.assertRaises(ValueError):
                    preprocessing.build_dataset(config(root))
                self.assertFalse(list((root / 'out').glob('*.h5')))
                self.assertFalse(list((root / 'out').glob('*.tmp')))

    def test_all_low_contrast_does_not_publish_empty_h5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_pair(root, 'a.png', np.zeros((20, 23), np.uint8), np.zeros((20, 23, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, '没有通过'):
                preprocessing.build_dataset(config(root))
            self.assertEqual(list((root / 'out').iterdir()), [])

    def test_existing_file_protection_and_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair(root)
            path = preprocessing.build_dataset(config(root))
            before = path.read_bytes()
            pair(root, seed=99)
            with self.assertRaises(FileExistsError):
                preprocessing.build_dataset(config(root))
            self.assertEqual(before, path.read_bytes())
            preprocessing.build_dataset(config(root, overwrite=True))
            self.assertNotEqual(before, path.read_bytes())

    def test_failure_after_partial_write_keeps_existing_h5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair(root)
            path = preprocessing.build_dataset(config(root))
            before = path.read_bytes()
            # 前一对已写入临时H5，后一对解码后校验失败。
            save_pair(root, 'z.png', np.zeros((20, 23, 3), np.uint8), np.zeros((20, 23, 3), np.uint8))
            with self.assertRaises(ValueError):
                preprocessing.build_dataset(config(root, overwrite=True))
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(list(path.parent.glob('*.tmp')))

    def test_validation_or_publish_failure_keeps_existing_h5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair(root)
            path = preprocessing.build_dataset(config(root))
            before = path.read_bytes()
            for target in ['data_process.validate_h5', 'data_process.os.replace']:
                with patch(target, side_effect=OSError('injected write failure')):
                    with self.assertRaises(OSError):
                        preprocessing.build_dataset(config(root, overwrite=True))
                self.assertEqual(before, path.read_bytes())
                self.assertFalse(list(path.parent.glob('*.tmp')))

    def test_no_overwrite_even_if_target_appears_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair(root)
            real_link = preprocessing.os.link

            def concurrent_writer(source, target):
                Path(target).write_bytes(b'concurrent output')
                real_link(source, target)

            with patch('data_process.os.link', side_effect=concurrent_writer):
                with self.assertRaises(FileExistsError):
                    preprocessing.build_dataset(config(root))
            target = root / 'out/demo_imgsize_8_stride_6.h5'
            self.assertEqual(target.read_bytes(), b'concurrent output')
            self.assertFalse(list(target.parent.glob('*.tmp')))

    def test_cli_overrides_work_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair(root)
            command = [sys.executable, str(ROOT / 'data_process.py'), f'root={root / "raw"}', 'name=demo',
                       'crop_size=8', 'stride=6', f'output_dir={root / "out"}', 'progress=false']
            completed = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True)
            self.assertIn('预处理完成', completed.stdout)
            self.assertTrue((root / 'out/demo_imgsize_8_stride_6.h5').is_file())


class H5ReaderTests(unittest.TestCase):
    def test_array_read_order_validation_and_pickle_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'patches.h5'
            write_h5(path)
            dataset = reader.H5Dataset(path)
            ir, vis = dataset[2]
            self.assertEqual(ir.dtype, torch.float32)
            self.assertEqual(float(ir[0, 0, 0]), float(np.float32(.2)))
            self.assertEqual(float(vis[0, 0, 0]), float(np.float32(.2) + np.float32(.01)))
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertIsNone(restored.h5f)
            self.assertIsNotNone(dataset.h5f)
            torch.testing.assert_close(restored[2][0], ir, atol=0, rtol=0)
            restored.close()
            dataset.close()
            with h5py.File(path, 'r+') as f:
                f['ir_patchs'][2, 0, 0, 0] = np.nan
            invalid = reader.H5Dataset(path)
            with self.assertRaisesRegex(ValueError, '有限值'):
                invalid[2]
            invalid.close()

    def test_reject_invalid_h5_layout_and_dtype(self):
        for shape, dtype in [((0, 1, 8, 8), np.float32), ((3, 2, 8, 8), np.float32),
                             ((3, 1, 8, 8), np.float64)]:
            with self.subTest(shape=shape, dtype=str(dtype)), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'invalid.h5'
                with h5py.File(path, 'w') as f:
                    for name in ['ir_patchs', 'vis_patchs']:
                        f.create_dataset(name, shape=shape, dtype=dtype)
                with self.assertRaises(ValueError):
                    reader.H5Dataset(path)

    def test_worker_count_does_not_change_two_epoch_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'patches.h5'
            write_h5(path)
            orders = []
            for workers in [0, 2]:
                dataset = reader.H5Dataset(path)
                dataset[0]  # 在父进程打开句柄，再检查fork子进程仍能正确读取。
                options = OmegaConf.create({'num_threads': workers, 'seed': 3407, 'shuffle': True,
                                            'batch_size': 3, 'drop_last': False})
                with patch.object(torch.cuda, 'is_available', return_value=False):
                    loader = reader.get_loader(options, dataset)
                run = []
                for _ in range(2):
                    indices = [int(round(float(v) * 10)) for ir, _ in loader for v in ir[:, 0, 0, 0]]
                    self.assertEqual(sorted(indices), list(range(10)))
                    run.append(indices)
                orders.append(run)
                del loader
                gc.collect()
                torch.testing.assert_close(dataset[0][0], torch.zeros(1, 8, 8), atol=0, rtol=0)
                dataset.close()
            self.assertEqual(orders[0], orders[1])
            self.assertNotEqual(orders[0][0], orders[0][1])


if __name__ == '__main__':
    unittest.main()

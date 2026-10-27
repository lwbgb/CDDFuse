"""训练阶段回归：原损失/Adam步骤、精度、阶段、当前checkpoint格式和入口统计。"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from models.cddfuse_model import CDDFuseModel
from models.net import LayerNorm, WithBias_LayerNorm, BiasFree_LayerNorm
from schemas.train_config import TrainConfig
from train import train
from utils.loss import Fusionloss, LegacySSIMLoss, cc

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def options(root, **overrides):
    return OmegaConf.merge(OmegaConf.load(ROOT / 'configs/base.yaml'),
                           OmegaConf.load(ROOT / 'configs/mode/train.yaml'),
                           {'device': 'cpu', 'batch_size': 2, 'num_threads': 0, 'cpu_threads': 1,
                            'checkpoint_root': str(root), 'run_dir': str(root / 'run'), 'verbose': False,
                            'dataset_root': str(root), 'h5_file_name': 'custom.h5',
                            'n_epochs': 2, 'epoch_gap': 1, 'optim_step': 1,
                            'save_latest_freq': 1, 'save_epoch_freq': 1,
                            'deterministic': True}, overrides)


def write_data(path, count=5):
    rng = np.random.default_rng(17)
    with h5py.File(path, 'w') as f:
        for key in ['ir_patchs', 'vis_patchs']:
            f[key] = rng.random((count, 1, 16, 16), dtype=np.float32)


def snapshots(model):
    return {name: {key: value.detach().clone() for key, value in net.state_dict().items()}
            for name, net in model.models.items()}


def assert_same(test, actual, expected):
    for name in expected:
        for key in expected[name]:
            torch.testing.assert_close(actual[name][key], expected[name][key], atol=0, rtol=0)


class TrainingTests(unittest.TestCase):
    def test_default_fp32_and_manual_layernorm(self):
        self.assertFalse(TrainConfig.use_bf16)
        with tempfile.TemporaryDirectory() as tmp:
            opt = options(Path(tmp))
            self.assertFalse(opt.use_bf16)
            self.assertNotIn('progress', opt)
            with patch('models.base_model.torch.cuda.set_device') as select:
                model = CDDFuseModel(opt)
                model.setup()
                select.assert_not_called()
            self.assertEqual(model.device.type, 'cpu')
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            for net in model.models.values():
                self.assertTrue(all(p.dtype == torch.float32 for p in net.parameters()))
                for layer in net.modules():
                    if isinstance(layer, LayerNorm):
                        self.assertIsInstance(layer.body, (WithBias_LayerNorm, BiasFree_LayerNorm))

    def test_bare_cuda_device_selects_index_zero(self):
        from types import SimpleNamespace
        from models.base_model import BaseModel
        with tempfile.TemporaryDirectory() as tmp, patch('models.base_model.torch.cuda.set_device') as select:
            stub = SimpleNamespace()
            BaseModel.__init__(stub, options(Path(tmp), device='cuda'))
            self.assertEqual(stub.device, torch.device('cuda:0'))
            select.assert_called_once_with(torch.device('cuda:0'))

    def test_nonempty_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / 'custom.h5')
            (root / 'run').mkdir()
            marker = root / 'run/old.pth'
            marker.write_bytes(b'old checkpoint')
            with self.assertRaises(FileExistsError):
                train(options(root))
            self.assertEqual(marker.read_bytes(), b'old checkpoint')

    def test_bf16_cpu_request_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'BF16训练需要'):
                CDDFuseModel(options(Path(tmp), use_bf16=True))

    def test_fp32_mode_disables_outer_autocast_and_ssim_restores_tf32(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = CDDFuseModel(options(Path(tmp)))
            ir, vi = torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
            model.set_input((ir, vi))
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                model.optimize_parameters()
            self.assertEqual(model.loss_total.dtype, torch.float32)
            self.assertEqual(model.data_VIS_hat.dtype, torch.float32)
            with torch.backends.cudnn.flags(allow_tf32=True):
                value = LegacySSIMLoss(11)(vi, vi)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertEqual(value.item(), 0.)

    def test_both_phases_match_reference_loss_and_adam_updates(self):
        import kornia
        for phase in [1, 2]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(3407)
                model = CDDFuseModel(options(Path(tmp)))
                nets = [copy.deepcopy(net) for net in model.models.values()]
                encoder, decoder, base, detail = nets
                optimizers = [torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0) for net in nets]
                ir, vi = torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
                model.update_state(phase)
                model.set_input((ir, vi))
                model.optimize_parameters()
                vb, vd, _ = encoder(vi)
                ib, id_, _ = encoder(ir)
                decomp = cc(vd, id_) ** 2 / (1.01 + cc(vb, ib))
                if phase == 1:
                    v_hat, _ = decoder(vi, vb, vd)
                    i_hat, _ = decoder(ir, ib, id_)
                    ssim = LegacySSIMLoss(11)
                    loss = (5 * ssim(vi, v_hat) + torch.nn.functional.mse_loss(vi, v_hat)
                            + 5 * ssim(ir, i_hat) + torch.nn.functional.mse_loss(ir, i_hat)
                            + 2 * decomp
                            + 5 * torch.nn.functional.l1_loss(kornia.filters.SpatialGradient()(vi),
                                                             kornia.filters.SpatialGradient()(v_hat)))
                else:
                    fused, _ = decoder(vi, base(ib + vb), detail(id_ + vd))
                    loss = Fusionloss()(vi, ir, fused)[0] + 2 * decomp
                loss.backward()
                active = 2 if phase == 1 else 4
                for net in nets[:active]:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), .01, error_if_nonfinite=True)
                for optimizer in optimizers[:active]:
                    optimizer.step()
                self.assertAlmostEqual(model.get_current_loss(), loss.item(), places=6)
                for actual, expected in zip(model.models.values(), nets):
                    for a, b in zip(actual.parameters(), expected.parameters()):
                        torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-6)

    def test_phase_boundary_and_fusion_scheduler_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = CDDFuseModel(options(Path(tmp), epoch_gap=40))
            data = (torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16))
            model.update_state(40)
            model.set_input(data)
            before = snapshots(model)
            model.optimize_parameters()
            self.assertEqual(model.get_phase(), 1)
            for name in ['BaseFuseLayer', 'DetailFuseLayer']:
                assert_same(self, snapshots(model), {name: before[name]})
            model.update_learning_rate()
            self.assertEqual(model.optimizer_encoder.param_groups[0]['lr'], 5e-5)
            self.assertEqual(model.optimizer_base.param_groups[0]['lr'], 1e-4)
            model.update_state(41)
            model.optimize_parameters()
            model.update_learning_rate()
            self.assertEqual(model.get_phase(), 2)
            self.assertEqual(model.optimizer_encoder.param_groups[0]['lr'], 2.5e-5)
            self.assertEqual(model.optimizer_base.param_groups[0]['lr'], 5e-5)
            self.assertNotIn('loss_MSE_VIS', model.losses)

    def test_nan_loss_stops_before_any_parameter_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = CDDFuseModel(options(Path(tmp)))
            before = snapshots(model)
            ir = torch.full((2, 1, 16, 16), float('nan'))
            model.set_input((ir, torch.rand_like(ir)))
            with self.assertRaises(FloatingPointError):
                model.optimize_parameters()
            assert_same(self, snapshots(model), before)
            self.assertTrue(all(not optimizer.state for optimizer in model.optimizers.values()))

    def test_nan_fusion_gradient_does_not_partially_step_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = CDDFuseModel(options(Path(tmp)))
            model.update_state(2)
            model.set_input((torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)))
            before = snapshots(model)
            handle = next(model.DetailFuseLayer.parameters()).register_hook(lambda grad: torch.full_like(grad, float('nan')))
            with self.assertRaises(RuntimeError):
                model.optimize_parameters()
            handle.remove()
            assert_same(self, snapshots(model), before)
            self.assertTrue(all(not optimizer.state for optimizer in model.optimizers.values()))

    def test_current_checkpoint_format_resume_next_epoch_and_atomic_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = CDDFuseModel(options(root))
            model.update_state(1)
            model.set_input((torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)))
            model.optimize_parameters()
            with self.assertRaises(ValueError):
                model.save_model('model.pth')
            model.update_learning_rate()
            model.save_model('model.pth')
            path = root / 'run/model.pth'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(set(payload), {'epoch', 'phase', 'models', 'optimizers', 'schedulers', 'config'})
            before = path.read_bytes()
            with patch('models.cddfuse_model.torch.save', side_effect=OSError('injected failure')):
                with self.assertRaises(OSError):
                    model.save_model('model.pth')
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(list(path.parent.glob('*.tmp')))
            resumed = CDDFuseModel(options(root, continue_train=True, load_dir='run', ckp_name='model.pth',
                                          run_dir=str(root / 'next')))
            resumed.setup()
            self.assertEqual(resumed.opt.start_epoch, 2)
            assert_same(self, snapshots(resumed), snapshots(model))
            self.assertEqual(resumed.optimizer_encoder.param_groups[0]['lr'], 5e-5)
            self.assertEqual(resumed.schedulers['scheduler_encoder'].state_dict(), model.schedulers['scheduler_encoder'].state_dict())
            resumed.update_state(2)
            self.assertEqual(resumed.get_phase(), 2)
            payload['optimizers'].pop('optimizer_detail_fusion')
            torch.save(payload, root / 'broken.pth')
            with self.assertRaisesRegex(ValueError, '全部优化器'):
                resumed.load_model('broken.pth')

    def test_real_entrypoint_uses_config_filename_and_actual_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_data(root / 'custom.h5', count=5)
            observed = []
            original = CDDFuseModel.get_current_loss

            def observe(model):
                value = original(model)
                observed.append((value, model.data_IR.shape[0]))
                return value

            with patch.object(CDDFuseModel, 'get_current_loss', observe):
                run = train(options(root, n_epochs=1, drop_last=True))
            record = json.loads((run / 'history.jsonl').read_text())
            expected = sum(v * n for v, n in observed) / sum(n for _, n in observed)
            self.assertEqual(record['samples'], 4)
            self.assertEqual(record['steps'], 2)
            self.assertEqual(record['loss'], expected)
            self.assertTrue((run / 'training.log').is_file())

    def test_random_crop_entrypoint_advances_epoch(self):
        from utils.dataset import PairedRandomCropDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(42)
            for folder, shape in [('ir', (23, 25)), ('vi', (23, 25, 3))]:
                (root / 'images' / folder).mkdir(parents=True)
                Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(root / 'images' / folder / 'a.png')
            epochs = []
            original = PairedRandomCropDataset.set_epoch

            def record_epoch(dataset, epoch):
                epochs.append(epoch)
                original(dataset, epoch)

            with patch.object(PairedRandomCropDataset, 'set_epoch', record_epoch):
                run = train(options(root, dataset_mode='random_crop', dataset_dir=str(root / 'images'),
                                    crop_size=16, samples_per_epoch=2, filter_low_contrast=False))
            self.assertEqual(epochs, [1, 2])
            records = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in records], [1, 2])

    def test_unverified_multirank_training_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'WORLD_SIZE': '2'}):
            with self.assertRaises(NotImplementedError):
                CDDFuseModel(options(Path(tmp)))


if __name__ == '__main__':
    unittest.main()

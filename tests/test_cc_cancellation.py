"""相关性抵消实验：真实 cc()/backward() 构造验证，以及可选的权重特征诊断。

从项目根目录运行（不修改权重或训练目标）：
    python -B -m unittest tests.test_cc_cancellation -v
    python -B -m tests.test_cc_cancellation
    python -B -m tests.test_cc_cancellation --checkpoint checkpoints/...pth \
        --h5 data/MSRS_train_imgsize_128_stride_200.h5 --samples 32 --device cpu

默认仅向终端输出JSON。--output 可保存JSON，但必须位于项目外。
真实特征的相关性差异不证明修正损失能改善融合质量。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from utils.loss import cc


def per_channel_cc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """保留[N,C]结果，使用项目cc相同的中心化、稳定项和裁剪。"""
    x = x.float().flatten(2)
    y = y.float().flatten(2)
    x = x - x.mean(-1, keepdim=True)
    y = y - y.mean(-1, keepdim=True)
    denominator = (torch.finfo(torch.float32).eps
                   + x.square().sum(-1).sqrt() * y.square().sum(-1).sqrt())
    return ((x * y).sum(-1) / denominator).clamp(-1, 1)


def make_case(batch: bool = False, same_sign: bool = False):
    # 两个零均值、等范数且互相正交的空间向量，产生约±0.8的相关性。
    x = torch.tensor([-1., -1., 1., 1.]).reshape(1, 1, 2, 2)
    z = torch.tensor([-1., 1., -1., 1.]).reshape_as(x)
    dim = 0 if batch else 1
    first = .8 * x + .6 * z
    second = (.8 if same_sign else -.8) * x + .6 * z
    return torch.cat([x, x], dim), torch.cat([first, second], dim)


def actual_backward(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """直接执行生产模型backward；只以零融合损失隔离分解项，不构建网络。"""
    from models.cddfuse_model import CDDFuseModel
    probe = CDDFuseModel.__new__(CDDFuseModel)
    probe.device = x.device
    probe.opt = SimpleNamespace(coeff_decomp=1.)
    probe._phase = 2
    probe.feature_VIS_Base = x
    probe.feature_IR_Base = x
    probe.feature_VIS_Detail = x
    probe.feature_IR_Detail = y
    probe.data_VIS = probe.data_IR = torch.zeros(1, 1, 2, 2, device=x.device)
    probe.data_Fuse = torch.zeros_like(probe.data_VIS, requires_grad=True)
    probe.criteria_fusion = lambda vis, ir, out: (out.sum() * 0., None, None)
    probe.backward()
    return {name: float(getattr(probe, name).detach())
            for name in ('loss_cc_Base', 'loss_cc_Detail', 'loss_decomp')}


def describe(x: torch.Tensor, y: torch.Tensor) -> dict:
    """gap等于rho的总体方差，包含异质性，不是纯粹的符号抵消量。"""
    rho = per_channel_cc(x, y)
    torch.testing.assert_close(rho.mean(), cc(x, y))
    mean_square = rho.square().mean()
    square_mean = cc(x, y).square()
    return dict(mean_rho=float(rho.mean()), mean_abs_rho=float(rho.abs().mean()),
                square_of_mean=float(square_mean), mean_of_squares=float(mean_square),
                cancellation_gap=float(mean_square - square_mean),
                positive_channels=int((rho > 0).sum()),
                negative_channels=int((rho < 0).sum()),
                rho=rho.cpu().tolist())


class CorrelationCancellationTests(unittest.TestCase):
    def test_channel_cancellation_in_actual_model_backward(self):
        x, y = make_case()
        result = actual_backward(x, y)
        self.assertLess(abs(result['loss_decomp']), 1e-10)
        self.assertGreater(float(per_channel_cc(x, y).square().mean()), .63)

    def test_batch_cancellation_in_actual_model_backward(self):
        x, y = make_case(batch=True)
        self.assertLess(abs(actual_backward(x, y)['loss_decomp']), 1e-10)
        self.assertGreater(float(per_channel_cc(x, y).abs().mean()), .79)

    def test_same_sign_control(self):
        x, y = make_case(same_sign=True)
        stats = describe(x, y)
        self.assertAlmostEqual(stats['square_of_mean'], stats['mean_of_squares'], places=6)
        self.assertGreater(actual_backward(x, y)['loss_decomp'], .3)

    def test_uncorrelated_control(self):
        x, _ = make_case()
        z = torch.tensor([-1., 1., -1., 1.]).reshape(1, 1, 2, 2).expand_as(x)
        self.assertLess(float(per_channel_cc(x, z).abs().max()), 1e-7)

    def test_gradient_is_hidden_by_cancellation(self):
        x, y = make_case()
        y = y.clone().requires_grad_()
        actual_backward(x, y)
        self.assertIsNotNone(y.grad)
        self.assertLess(float(y.grad.abs().max()), 1e-7)
        corrected = per_channel_cc(x, y).square().mean() / (1.01 + cc(x, x))
        gradient, = torch.autograd.grad(corrected, y)
        self.assertGreater(float(gradient.abs().max()), .01)


def checkpoint_diagnostic(checkpoint: Path, h5_path: Path, samples: int,
                          batch_size: int, device: str) -> dict:
    from models.net import Restormer_Encoder
    from utils.dataset import H5Dataset
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请使用 --device cpu')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = state['models']['DIDF_Encoder']
    if weights and all(k.startswith('module.') for k in weights):
        weights = {k.removeprefix('module.'): v for k, v in weights.items()}
    encoder = Restormer_Encoder().to(target)
    encoder.load_state_dict(weights, strict=True)
    encoder.eval()
    dataset = H5Dataset(h5_path)
    if samples > len(dataset):
        dataset.close()
        raise ValueError(f'samples不能超过数据集长度 {len(dataset)}')
    indices = torch.linspace(0, len(dataset) - 1, samples).round().long().tolist()
    collected = {'base': [], 'detail': []}
    batch_stats = []
    try:
        with torch.inference_mode():
            for offset in range(0, samples, batch_size):
                selected = indices[offset:offset + batch_size]
                pairs = [dataset[i] for i in selected]
                ir = torch.stack([p[0] for p in pairs]).to(target)
                vis = torch.stack([p[1] for p in pairs]).to(target)
                vb, vd, _ = encoder(vis)
                ib, id_, _ = encoder(ir)
                for name, a, b in [('base', vb, ib), ('detail', vd, id_)]:
                    rho = per_channel_cc(a, b)
                    torch.testing.assert_close(rho.mean(), cc(a.float(), b.float()))
                    collected[name].append(rho.cpu())
                batch_stats.append(dict(indices=selected, **describe(vd.float(), id_.float())))
    finally:
        dataset.close()
    summary = {}
    for name, pieces in collected.items():
        rho = torch.cat(pieces)
        # pooled统计只描述全体样本；真实训练项以各batch统计为准。
        summary[name] = dict(mean_rho=float(rho.mean()), mean_abs_rho=float(rho.abs().mean()),
                             mean_of_squares=float(rho.square().mean()),
                             square_of_pooled_mean=float(rho.mean().square()),
                             positive_fraction=float((rho > 0).float().mean()),
                             negative_fraction=float((rho < 0).float().mean()),
                             per_sample_mean=rho.mean(1).tolist(), rho=rho.tolist())
    return dict(checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                epoch=state.get('epoch'), training_config=state.get('config'),
                h5=str(h5_path.resolve()), indices=indices, device=str(target),
                precision='FP32, TF32 disabled', batch_size=batch_size,
                summaries=summary, batches=batch_stats,
                limitation='固定训练patch上的描述性诊断；不证明修正后融合质量改善或RoadScene差距的原因。')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--h5', type=Path)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, help='可选JSON路径，必须在项目外且尚不存在')
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.h5):
        parser.error('--checkpoint 与 --h5 必须同时指定')
    if args.samples < 1 or args.batch_size < 1:
        parser.error('samples和batch-size必须为正数')
    root = Path(__file__).resolve().parents[1]
    if args.output and (args.output.resolve().is_relative_to(root) or args.output.exists()):
        parser.error('--output 必须在项目外且尚不存在，避免覆盖结果')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorrelationCancellationTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        return 1
    report = {'torch': torch.__version__, 'synthetic': {}, 'scope': '仅诊断，未改变训练目标或权重',
              'gap_definition': 'mean(rho^2)-mean(rho)^2是rho的总体方差，差异也可能来自同号相关的异质性。',
              'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (root / 'utils/loss.py', root / 'models/cddfuse_model.py',
                                          Path(__file__).resolve())}}
    for name, kwargs in [('channel_opposition', {}), ('batch_opposition', {'batch': True}),
                         ('same_sign_control', {'same_sign': True})]:
        x, y = make_case(**kwargs)
        report['synthetic'][name] = dict(**describe(x, y), actual_model=actual_backward(x, y))
    if args.checkpoint:
        report['checkpoint_diagnostic'] = checkpoint_diagnostic(
            args.checkpoint, args.h5, args.samples, args.batch_size, args.device)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
        print(f'结果已保存：{args.output.resolve()}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

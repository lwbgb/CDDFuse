"""独立验证器的合成数据测试；不代表真实数据集结果。"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from utils.validation import Validator, ValidationManager, METRICS, _normalize_to_uint8


def pairs(root, prefix, seed, count=2):
    rng = np.random.default_rng(seed)
    for folder in ("ir", "vi"):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (96, 96), dtype=np.uint8)).save(root / folder / f"{prefix}{i}.png")


class DemoModel:
    def __init__(self):
        self.isTrain = True
        self.device = torch.device("cpu")
        self.opt = OmegaConf.create(dict(model="Demo", epoch_gap=40, n_epochs=120))
        self.epoch = 41
        self.phase = 2
        net = torch.nn.Conv2d(2, 1, 1, bias=False)
        torch.nn.init.constant_(net.weight, 0.5)
        self.models = {"fusion": net}
        self.optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        self.saved = []

    def get_epoch(self): return self.epoch
    def get_phase(self): return self.phase
    def eval(self):
        for net in self.models.values(): net.eval()
    def set_input(self, data): self.data_IR, self.data_VIS = data[:2]
    def forward(self):
        assert not self.isTrain and not torch.is_grad_enabled() and not self.models["fusion"].training
        self.data_Fuse = self.models["fusion"](torch.cat((self.data_IR, self.data_VIS), dim=1))
    def save_model(self, name):
        assert self.isTrain
        self.saved.append(name)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_cwd = os.getcwd()
        os.chdir(self.root)
        for name, split, prefix, seed in (("RoadScene", "valid", "rv", 1), ("RoadScene", "test", "rt", 2),
                                         ("M3FD", "valid", "mv", 3), ("M3FD", "test", "mt", 4)):
            path = self.root / ("datasets" if split == "valid" else "test_img") / (name+"_valid" if split == "valid" else name)
            pairs(path, prefix, seed)
        self.opt = OmegaConf.create(dict(validation_every=5, validation_start_epoch=None, continue_train=False))
        self.road = Validator(self.opt, "RoadScene", "SD")
        self.m3fd = Validator(self.opt, "M3FD", "Qabf")
        self.output = self.root / "run"

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp.cleanup()

    def manager(self): return ValidationManager(self.opt, self.output, [self.road, self.m3fd])
    def scores(self, x): return dict.fromkeys(METRICS, x)

    def test_default_dataset_paths_and_counts(self):
        self.assertEqual(self.road.root, (self.root/"datasets/RoadScene_valid").resolve())
        self.assertEqual(len(self.road.loader.dataset), 2)
        self.assertEqual(len(self.m3fd.loader.dataset), 2)

    def test_validation_does_not_require_a_fixed_count(self):
        pairs(self.root/"datasets/M3FD_valid", "extra", 9, count=1)
        self.assertEqual(len(Validator(self.opt, "M3FD", "Qabf").loader.dataset), 3)

    def test_real_metrics_preserve_training_state(self):
        model = DemoModel()
        for p in model.models["fusion"].parameters(): p.grad = torch.ones_like(p)
        model.optimizer.step(); weights = copy.deepcopy(model.models["fusion"].state_dict())
        optimizer = copy.deepcopy(model.optimizer.state_dict()); gradient = next(model.models["fusion"].parameters()).grad.clone()
        rng = torch.random.get_rng_state().clone()
        values = self.m3fd.evaluate(model)
        self.assertEqual(set(values), set(METRICS));self.assertTrue(np.isfinite(list(values.values())).all())
        self.assertTrue(model.isTrain and model.models["fusion"].training)
        torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
        torch.testing.assert_close(gradient, next(model.models["fusion"].parameters()).grad, rtol=0, atol=0)
        for k, p in model.models["fusion"].state_dict().items(): torch.testing.assert_close(p, weights[k], rtol=0, atol=0)
        self.assertEqual(optimizer['param_groups'], model.optimizer.state_dict()['param_groups'])

    def test_both_datasets_share_records_and_separate_best(self):
        manager=self.manager();model=DemoModel()
        with patch.object(self.road, 'evaluate', return_value=self.scores(0.5)), patch.object(self.m3fd, 'evaluate', return_value=self.scores(0.7)):
            record=manager.run(model)
        self.assertEqual(set(record['datasets']), {'RoadScene','M3FD'})
        self.assertEqual(model.saved, ['Demo_best_RoadScene.pth','Demo_best_M3FD.pth'])
        for name in ('best_validation.json','validation_manifest.json'):
            self.assertEqual(set(json.loads((self.output/name).read_text())['datasets']), {'RoadScene','M3FD'})
        rows=(self.output/'validation_history.jsonl').read_text().splitlines()
        self.assertEqual(len(rows),1);self.assertEqual(set(json.loads(rows[0])['datasets']),{'RoadScene','M3FD'})
        self.assertEqual(record['datasets']['RoadScene']['selection_metric'],'SD')
        self.assertEqual(record['datasets']['M3FD']['selection_metric'],'Qabf')

    def test_independent_best_and_resume(self):
        manager=self.manager();model=DemoModel()
        for epoch, a, b in ((41,0.3,0.5),(45,0.4,0.4),(50,0.35,0.6)):
            model.epoch=epoch
            with patch.object(self.road,'evaluate',return_value=self.scores(a)),patch.object(self.m3fd,'evaluate',return_value=self.scores(b)):
                manager.run(model)
        best=json.loads((self.output/'best_validation.json').read_text())['datasets']
        self.assertEqual(best['RoadScene']['epoch'],45);self.assertEqual(best['M3FD']['epoch'],50)
        self.opt.continue_train=True;restored=self.manager()
        self.assertEqual(restored.best['RoadScene']['score'],0.4);self.assertEqual(restored.best['M3FD']['score'],0.6)

    def test_legacy_roadscene_resume_and_history_preserved(self):
        model=DemoModel();m=self.manager()
        with patch.object(self.road,'evaluate',return_value=self.scores(0.5)),patch.object(self.m3fd,'evaluate',return_value=self.scores(0.4)):m.run(model)
        for fn in ('best_validation.json','validation_manifest.json'):
            path=self.output/fn;d=json.loads(path.read_text());path.write_text(json.dumps(d['datasets']['RoadScene']))
        history=self.output/'validation_history.jsonl';history.write_text(json.dumps({'epoch':40,'metrics':{'SD':0.4}})+'\n')
        self.opt.continue_train=True;m=self.manager();self.assertEqual(m.best['RoadScene']['score'],0.5)
        with patch.object(self.road,'evaluate',return_value=self.scores(0.4)),patch.object(self.m3fd,'evaluate',return_value=self.scores(0.6)):m.run(model)
        self.assertEqual(len(history.read_text().splitlines()),2)
        self.assertEqual(set(json.loads((self.output/'best_validation.json').read_text())['datasets']), {'RoadScene','M3FD'})

    def test_wrong_metric_or_protocol_fails_resume(self):
        m=self.manager();model=DemoModel()
        with patch.object(self.road,'evaluate',return_value=self.scores(0.5)),patch.object(self.m3fd,'evaluate',return_value=self.scores(0.4)):m.run(model)
        self.opt.continue_train=True
        with self.assertRaisesRegex(ValueError,'选择指标不同'):
            ValidationManager(self.opt,self.output,[Validator(self.opt,'RoadScene','VIF')])
        path=self.output/'best_validation.json';record=json.loads(path.read_text());record['datasets']['RoadScene']['metric_protocol'].pop('postprocess');path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'指标口径不同'):self.manager()

    def test_fresh_experiment_refuses_existing_history(self):
        self.output.mkdir();(self.output/'validation_history.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'已有验证记录'):self.manager()

    def test_duplicate_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError,'不能重复'):ValidationManager(self.opt,self.output,[self.road,self.road])

    def test_filename_overlap_rejected(self):
        test=self.root/'test_img/M3FD'
        for mod in ('ir','vi'):(test/mod/'mv0.png').write_bytes((test/mod/'mt0.png').read_bytes())
        with self.assertRaisesRegex(ValueError,'重叠'):Validator(self.opt,'M3FD','Qabf')

    def test_renamed_pixel_overlap_rejected(self):
        (self.root/'datasets/M3FD_valid/ir/mv0.png').write_bytes((self.root/'test_img/M3FD/ir/mt0.png').read_bytes())
        with self.assertRaisesRegex(ValueError,'内容重叠'):Validator(self.opt,'M3FD','Qabf')

    def test_unpaired_or_empty_data_rejected(self):
        (self.root/'datasets/M3FD_valid/vi/mv0.png').unlink()
        with self.assertRaisesRegex(ValueError,'逐对匹配'):Validator(self.opt,'M3FD','Qabf')

    def test_invalid_name_metric_interval_and_start(self):
        for name in ('../M3FD','M3FD/test','','a.b'):
            with self.assertRaisesRegex(ValueError,'非法'):Validator(self.opt,name,'SD')
        with self.assertRaisesRegex(ValueError,'metric'):Validator(self.opt,'M3FD','bad')
        self.opt.validation_every=0
        with self.assertRaisesRegex(ValueError,'validation_every'):Validator(self.opt,'M3FD','SD')
        self.opt.validation_every=5
        for v in (0,-1,1.5,True,'60'):
            self.opt.validation_start_epoch=v
            with self.assertRaisesRegex(ValueError,'validation_start_epoch'):Validator(self.opt,'M3FD','SD')

    def test_schedule_with_and_without_explicit_start(self):
        model=DemoModel()
        for epoch,phase,want in ((40,1,False),(41,2,True),(42,2,False),(45,2,True),(120,2,True)):
            model.epoch,model.phase=epoch,phase;self.assertEqual(self.road.should_validate(model),want)
        self.opt.validation_start_epoch=63;val=Validator(self.opt,'RoadScene','SD')
        for epoch,want in ((41,False),(62,False),(63,True),(68,True),(120,True)):
            model.epoch=epoch;model.phase=2;self.assertEqual(val.should_validate(model),want)

    def test_failed_second_dataset_does_not_save_first_best(self):
        manager=self.manager();model=DemoModel()
        with patch.object(self.road,'evaluate',return_value=self.scores(0.4)),patch.object(self.m3fd,'evaluate',side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError,'failure'):manager.run(model)
        self.assertFalse(model.saved);self.assertFalse(self.output.exists())

    def test_nonfinite_output_fails_and_restores_state(self):
        model=DemoModel()
        def bad():model.data_Fuse=torch.full((1,1,96,96),float('nan'))
        with patch.object(model,'forward',side_effect=bad):
            with self.assertRaises(FloatingPointError):self.road.evaluate(model)
        self.assertTrue(model.isTrain and model.models['fusion'].training)

    def test_nonfinite_metrics_are_rejected(self):
        from utils.test_metrics import EvaluationResult
        model=DemoModel()
        with patch.object(self.road.evaluator,'evaluate',return_value=EvaluationResult(self.scores(float('nan')), {'VIF':'bad'})):
            with self.assertRaises(FloatingPointError):self.road.evaluate(model)

    def test_mse_uses_minimum(self):
        val=Validator(self.opt,'RoadScene','MSE');manager=ValidationManager(self.opt,self.output,[val]);model=DemoModel()
        for epoch,score in ((41,10.),(45,11.),(50,9.)):
            model.epoch=epoch
            with patch.object(val,'evaluate',return_value=self.scores(score)):manager.run(model)
        self.assertEqual(manager.best['RoadScene']['score'],9.)
        self.assertEqual(manager.best['RoadScene']['selection_mode'],'min')

    def test_postprocess_matches_existing_semantics(self):
        x=torch.tensor([[[[0.50,0.55],[0.60,0.65]]]])
        np.testing.assert_array_equal(_normalize_to_uint8(x), [[0,85],[170,255]])
        with self.assertRaises(FloatingPointError):_normalize_to_uint8(torch.full((1,1,2,2),float('nan')))

    def test_metric_calculation_matches_evaluator(self):
        from utils.test_metrics import FusionEvaluator
        model=DemoModel(); references=[]
        with torch.inference_mode():
            for data in self.road.loader:
                ir,vi=data[:2];fused=model.models['fusion'](torch.cat((ir,vi),dim=1))
                pixels=_normalize_to_uint8(fused)
                values=FusionEvaluator().evaluate(torch.round(ir.squeeze()*255),torch.round(vi.squeeze()*255),pixels,metrics=METRICS).values
                references.append([values[k] for k in METRICS])
        actual=self.road.evaluate(model)
        np.testing.assert_allclose([actual[k] for k in METRICS],np.mean(references,axis=0),rtol=0,atol=0)

    def test_forward_tensors_released_and_partial_failure_restored(self):
        from utils.validation import _FORWARD_TENSORS
        model=DemoModel();model.models['fusion'].train(False)
        def bad():
            model.feature_IR_Base=torch.ones(1,64,96,96)
            raise RuntimeError('partial')
        with patch.object(model,'forward',side_effect=bad):
            with self.assertRaisesRegex(RuntimeError,'partial'):self.road.evaluate(model)
        self.assertTrue(model.isTrain);self.assertFalse(model.models['fusion'].training)
        self.assertTrue(all(getattr(model,n,None) is None for n in _FORWARD_TENSORS))

    def test_cuda_cache_cleanup_at_validation_boundary(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        model=DemoModel();model.device=SimpleNamespace(type='cuda')
        with patch('utils.validation.torch.cuda.device',return_value=nullcontext()),patch('utils.validation.torch.cuda.empty_cache') as empty:
            self.road.evaluate(model);empty.assert_called_once()

    def test_all_forward_tensors_released_between_images(self):
        import weakref
        from utils.validation import _FORWARD_TENSORS
        model=DemoModel();previous=[];original=model.forward
        def tracked():
            self.assertTrue(all(r() is None for r in previous));original();previous.append(weakref.ref(model.data_Fuse));model.feature_VIS_Base=model.data_Fuse
        for name in _FORWARD_TENSORS:setattr(model,name,torch.ones(1))
        with patch.object(model,'forward',side_effect=tracked):
            for _ in range(2):
                self.road.evaluate(model);self.assertTrue(all(getattr(model,n,None) is None for n in _FORWARD_TENSORS));self.assertTrue(all(r() is None for r in previous))

if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

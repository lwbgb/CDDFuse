"""合成图像验证流程演示，不冒充真实 RoadScene 分数。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from utils.validation import RoadSceneValidator, METRICS


def write_pairs(root: Path, prefix: str, seed: int, count: int = 2) -> None:
    rng = np.random.default_rng(seed)
    for folder in ("ir", "vi"):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (96, 96), dtype=np.uint8)).save(
                root / folder / f"{prefix}{index}.png")


def options(root: Path):
    return OmegaConf.create(dict(validation_dir=str(root / "val"),
                                 validation_test_dir=str(root / "test"),
                                 validation_expected_pairs=2, validation_every=5,
                                 validation_metric="Qabf", continue_train=False))


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

    def get_epoch(self):
        return self.epoch

    def get_phase(self):
        return self.phase

    def eval(self):
        for net in self.models.values():
            net.eval()

    def set_input(self, data):
        self.data_IR, self.data_VIS = data[:2]

    def forward(self):
        assert not self.isTrain
        assert not torch.is_grad_enabled()
        assert not self.models["fusion"].training
        self.data_Fuse = self.models["fusion"](torch.cat((self.data_IR, self.data_VIS), dim=1))

    def save_model(self, name):
        assert self.isTrain
        self.saved.append(name)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_pairs(self.root / "val", "val_", 1)
        write_pairs(self.root / "test", "test_", 2)
        self.opt = options(self.root)
        self.validator = RoadSceneValidator(self.opt, self.root / "run")

    def tearDown(self):
        self.temporary.cleanup()

    def test_real_metrics_preserve_parameters_gradients_optimizer_and_rng(self):
        model = DemoModel()
        for parameter in model.models["fusion"].parameters():
            parameter.grad = torch.ones_like(parameter)
        model.optimizer.step()
        params = copy.deepcopy(model.models["fusion"].state_dict())
        state = copy.deepcopy(model.optimizer.state_dict())
        gradient = next(model.models["fusion"].parameters()).grad.clone()
        rng = torch.random.get_rng_state().clone()
        metrics = self.validator.evaluate(model)
        self.assertEqual(set(metrics), set(METRICS))
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertTrue(model.isTrain)
        self.assertTrue(model.models["fusion"].training)
        torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
        torch.testing.assert_close(gradient, next(model.models["fusion"].parameters()).grad, rtol=0, atol=0)
        for key, parameter in model.models["fusion"].state_dict().items():
            torch.testing.assert_close(parameter, params[key], rtol=0, atol=0)
        current = model.optimizer.state_dict()
        self.assertEqual(current["param_groups"], state["param_groups"])
        for key in state["state"][0]:
            torch.testing.assert_close(current["state"][0][key], state["state"][0][key], rtol=0, atol=0)
        # 验证之后确实仍能继续反向传播。
        model.optimizer.zero_grad()
        model.models["fusion"](torch.ones(1, 2, 8, 8)).sum().backward()
        model.optimizer.step()

    def test_restore_modes_on_failure(self):
        model = DemoModel()
        with patch.object(model, "forward", side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError):
                self.validator.evaluate(model)
        self.assertTrue(model.isTrain)
        self.assertTrue(model.models["fusion"].training)

    def test_best_selection_and_resume_history(self):
        model = DemoModel()
        for epoch, score in ((41, 0.4), (45, 0.39), (50, 0.43)):
            model.epoch = epoch
            with patch.object(self.validator, "evaluate", return_value=dict.fromkeys(METRICS, score)):
                self.validator.run(model)
        self.assertEqual(len(model.saved), 2)
        best = json.loads((self.root / "run/best_validation.json").read_text())
        self.assertEqual(best["epoch"], 50)
        self.assertEqual(best["score"], 0.43)
        records = (self.root / "run/validation_history.jsonl").read_text().splitlines()
        self.assertEqual(len(records), 3)
        self.opt.continue_train = True
        restored = RoadSceneValidator(self.opt, self.root / "run")
        self.assertEqual(restored.best["score"], 0.43)
        self.opt.validation_metric = "VIF"
        with self.assertRaisesRegex(ValueError, "选择指标不同"):
            RoadSceneValidator(self.opt, self.root / "run")

    def test_test_set_rejected_even_if_files_renamed(self):
        self.opt.validation_dir = self.opt.validation_test_dir
        with self.assertRaisesRegex(ValueError, "重叠"):
            RoadSceneValidator(self.opt, self.root / "run")
        self.opt.validation_dir = str(self.root / "val")
        source = self.root / "test/ir/test_0.png"
        (self.root / "val/ir/val_0.png").write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "内容重叠"):
            RoadSceneValidator(self.opt, self.root / "run")

    def test_interval_phase_boundary_and_final_epoch(self):
        model = DemoModel()
        for epoch, phase, expected in ((40, 1, False), (41, 2, True), (42, 2, False),
                                       (45, 2, True), (119, 2, False), (120, 2, True)):
            model.epoch, model.phase = epoch, phase
            self.assertEqual(self.validator.should_validate(model), expected)

    def test_explicit_start_epoch_and_interval(self):
        self.opt.validation_start_epoch = 63
        validator = RoadSceneValidator(self.opt, self.root / "run")
        model = DemoModel()
        for epoch, expected in ((41, False), (60, False), (62, False), (63, True),
                                (65, False), (68, True), (73, True), (119, False), (120, True)):
            model.epoch = epoch
            self.assertEqual(validator.should_validate(model), expected, epoch)

    def test_start_cannot_enable_phase_one_or_force_final_before_start(self):
        model = DemoModel()
        self.opt.validation_start_epoch = 10
        validator = RoadSceneValidator(self.opt, self.root / "run")
        model.epoch, model.phase = 40, 1
        self.assertFalse(validator.should_validate(model))
        model.epoch, model.phase = 41, 2
        self.assertTrue(validator.should_validate(model))
        model.epoch = 46
        self.assertTrue(validator.should_validate(model))
        self.opt.validation_start_epoch = 121
        validator = RoadSceneValidator(self.opt, self.root / "run")
        model.epoch = 120
        self.assertFalse(validator.should_validate(model))

    def test_invalid_start_epoch_rejected(self):
        for value in (0, -1, 1.5, True, "60"):
            with self.subTest(value=value):
                self.opt.validation_start_epoch = value
                with self.assertRaisesRegex(ValueError, "validation_start_epoch"):
                    RoadSceneValidator(self.opt, self.root / "run")

    def test_missing_or_unpaired_data_and_invalid_metrics_rejected(self):
        (self.root / "val/vi/val_0.png").unlink()
        with self.assertRaisesRegex(ValueError, "逐对匹配"):
            RoadSceneValidator(self.opt, self.root / "run")

    def test_nonfinite_model_output_is_not_hidden_by_uint8_conversion(self):
        model = DemoModel()
        def invalid_forward():
            model.data_Fuse = torch.full((1, 1, 96, 96), float("nan"))
        with patch.object(model, "forward", side_effect=invalid_forward):
            with self.assertRaises(FloatingPointError):
                self.validator.run(model)
        self.assertTrue(model.isTrain)
        self.assertFalse(model.saved)

    def test_nonfinite_metrics_do_not_save_best(self):
        model = DemoModel()
        with patch("utils.validation.EvaluatorTorch.evaluate_all", return_value=np.full(8, np.nan)):
            with self.assertRaises(FloatingPointError):
                self.validator.run(model)
        self.assertFalse(model.saved)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

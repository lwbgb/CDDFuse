"""使用小网络验证真实 checkpoint 加载、Adam 保留和续训调度重设。"""
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import torch

from models.cddfuse_model import CDDFuseModel
from utils import networks


def make_model(root: Path, **overrides) -> CDDFuseModel:
    model = CDDFuseModel.__new__(CDDFuseModel)
    model.opt = OmegaConf.create(dict(checkpoint_root=str(root), epoch_gap=0, lr=1e-4,
                                     min_lr=1e-6, lr_policy="step", optim_step=2,
                                     optim_gamma=0.5, n_epochs=10, n_epochs_decay=5,
                                     reset_lr_on_resume=False, resume_lrs={}))
    model.opt.update(overrides)
    model.isTrain = True
    model.device = torch.device("cpu")
    model.save_dir = root
    model._epoch, model._phase = 120, 2
    model.epoch_completed = True
    model.models, model.optimizers, model.schedulers = {}, {}, {}
    for suffix, attr in (("encoder", "encoder"), ("decoder", "decoder"),
                         ("base_fusion", "base"), ("detail_fusion", "detail")):
        net = torch.nn.Linear(1, 1)
        model.models[suffix] = net
        optimizer = torch.optim.Adam(net.parameters(), lr=model.opt.lr)
        model.optimizers[f"optimizer_{suffix}"] = optimizer
        setattr(model, f"optimizer_{attr}", optimizer)
        scheduler = networks.get_scheduler(optimizer, model.opt, phase=model._phase)
        model.schedulers[f"scheduler_{suffix}"] = scheduler
        setattr(model, f"scheduler_{attr}", scheduler)
    return model


def advance(model: CDDFuseModel) -> None:
    for optimizer in model.optimizers.values():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    model.update_learning_rate()


class ResumeScheduleTests(unittest.TestCase):
    def test_default_keeps_checkpoint_schedule_even_with_new_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_model(root)
            advance(source)
            advance(source)
            source.save_model("old.pth")
            resumed = make_model(root, lr=2e-5, optim_step=40)
            resumed.load_model("old.pth")
            self.assertEqual(resumed.optimizer_encoder.param_groups[0]["lr"], 5e-5)
            self.assertEqual(resumed.scheduler_encoder.state_dict(), source.scheduler_encoder.state_dict())

    def test_reset_preserves_adam_and_restarts_decay_with_per_module_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_model(root)
            advance(source)
            advance(source)
            source.save_model("old.pth")
            resumed = make_model(root, reset_lr_on_resume=True, lr=1e-5, optim_step=3,
                                 resume_lrs={"optimizer_base_fusion": 2e-5})
            resumed.load_model("old.pth")
            self.assertEqual(resumed.get_epoch(), 120)
            self.assertEqual(resumed.scheduler_encoder.last_epoch, 0)
            self.assertEqual(resumed.scheduler_encoder.base_lrs, [5e-5])
            self.assertIs(resumed.scheduler_encoder, resumed.schedulers["scheduler_encoder"])
            for name, optimizer in resumed.optimizers.items():
                expected = 2e-5 if name == "optimizer_base_fusion" else 5e-5
                self.assertEqual(optimizer.param_groups[0]["lr"], expected)
                for old, new in zip(source.optimizers[name].state.values(), optimizer.state.values()):
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
            advance(resumed)
            advance(resumed)
            self.assertEqual(resumed.optimizer_encoder.param_groups[0]["lr"], 5e-5)
            advance(resumed)
            self.assertEqual(resumed.optimizer_encoder.param_groups[0]["lr"], 2.5e-5)
            self.assertEqual(resumed.optimizer_base.param_groups[0]["lr"], 1e-5)
            resumed.save_model("reset.pth")
            again = make_model(root)
            again.load_model("reset.pth")
            self.assertEqual(again.scheduler_encoder.state_dict(), resumed.scheduler_encoder.state_dict())
            advance(again)
            advance(resumed)
            self.assertEqual(again.scheduler_encoder.state_dict(), resumed.scheduler_encoder.state_dict())

    def test_empty_overrides_preserve_all_checkpoint_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_model(root)
            for i, optimizer in enumerate(source.optimizers.values(), start=1):
                optimizer.param_groups[0]["lr"] = i * 1e-5
            source.save_model("old.pth")
            resumed = make_model(root, reset_lr_on_resume=True, lr=0.01)
            resumed.load_model("old.pth")
            for name, optimizer in resumed.optimizers.items():
                self.assertEqual(optimizer.param_groups[0]["lr"],
                                 source.optimizers[name].param_groups[0]["lr"])
            self.assertEqual(resumed.scheduler_encoder.last_epoch, 0)
            self.assertEqual(resumed.scheduler_encoder.base_lrs, [1e-5])

    def test_unspecified_optimizer_preserves_each_parameter_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(Path(tmp), lr=0.01)
            optimizer = model.optimizer_encoder
            optimizer.param_groups[0]["lr"] = 2e-5
            optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))], "lr": 3e-5})
            model.reset_learning_rate_and_schedulers()
            self.assertEqual([group["lr"] for group in optimizer.param_groups], [2e-5, 3e-5])
            self.assertEqual(model.scheduler_encoder.base_lrs, [2e-5, 3e-5])

    def test_switch_step_to_cosine_uses_new_initial_lr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_model(root)
            advance(source)
            source.save_model("old.pth")
            resumed = make_model(root, reset_lr_on_resume=True, lr=2e-5, lr_policy="cosine",
                                 resume_lrs={"optimizer_encoder": 2e-5})
            resumed.load_model("old.pth")
            reference = torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reference, T_max=10)
            for _ in range(4):
                advance(resumed)
                reference.step()
                scheduler.step()
                self.assertAlmostEqual(resumed.optimizer_encoder.param_groups[0]["lr"], reference.param_groups[0]["lr"])

    def test_phase_one_keeps_fusion_schedule_frozen_after_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(Path(tmp), lr=2e-6, epoch_gap=40)
            model.reset_learning_rate_and_schedulers()
            model.update_state(40)
            advance(model)
            self.assertEqual(model.scheduler_encoder.last_epoch, 1)
            self.assertEqual(model.scheduler_base.last_epoch, 0)
            model.update_state(41)
            advance(model)
            self.assertEqual(model.scheduler_base.last_epoch, 1)
            self.assertEqual(model.optimizer_encoder.param_groups[0]["lr"], 1e-6)

    def test_invalid_reset_settings_fail(self):
        for overrides in ({"resume_lrs": {"encoder": 1e-5}}, {"resume_lrs": {"optimizer_encoder": float("nan")}},
                          {"resume_lrs": {"optimizer_encoder": 1e-8}}, {"optim_step": 0}, {"lr_policy": "plateau"}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmp:
                model = make_model(Path(tmp))
                model.opt.update(overrides)
                with self.assertRaises(ValueError):
                    model.reset_learning_rate_and_schedulers()

    def test_independent_phase_clocks_and_learning_rate_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(Path(tmp), epoch_gap=4, phase1_optim_step=3,
                               phase1_optim_gamma=0.5, phase2_optim_step=2, phase2_optim_gamma=0.8)
            for epoch in range(1, 5):
                model.update_state(epoch)
                advance(model)
            self.assertEqual(model.optimizer_encoder.param_groups[0]["lr"], 5e-5)
            self.assertEqual(model.optimizer_base.param_groups[0]["lr"], 1e-4)
            model.update_state(5)
            self.assertEqual(model.scheduler_encoder.last_epoch, 0)
            self.assertEqual(model.optimizer_encoder.param_groups[0]["lr"], 5e-5)
            advance(model)
            self.assertEqual(model.optimizer_encoder.param_groups[0]["lr"], 5e-5)
            model.update_state(6)
            advance(model)
            self.assertAlmostEqual(model.optimizer_encoder.param_groups[0]["lr"], 4e-5)
            self.assertAlmostEqual(model.optimizer_base.param_groups[0]["lr"], 8e-5)

    def test_phase_checkpoint_boundary_and_mid_phase_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = dict(epoch_gap=4, phase1_optim_step=3, phase1_optim_gamma=0.5,
                            phase2_optim_step=2, phase2_optim_gamma=0.8)
            source = make_model(root, **settings)
            for epoch in range(1, 5):
                source.update_state(epoch)
                advance(source)
            source.save_model("boundary.pth")
            # 关闭重设时，新配置不应替代 checkpoint 的两阶段参数。
            resumed = make_model(root, epoch_gap=4, start_epoch=5,
                                 phase1_optim_step=1, phase2_optim_step=99)
            resumed.load_model("boundary.pth")
            self.assertIsInstance(resumed.scheduler_encoder, networks.PhaseStepLR)
            for epoch in (5, 6):
                for model in (source, resumed):
                    model.update_state(epoch)
                    advance(model)
                self.assertEqual(source.scheduler_encoder.state_dict(), resumed.scheduler_encoder.state_dict())
                self.assertEqual(source.optimizer_encoder.param_groups[0]["lr"], resumed.optimizer_encoder.param_groups[0]["lr"])
            resumed.save_model("mid.pth")
            again = make_model(root, epoch_gap=4)
            again.load_model("mid.pth")
            again.update_state(7)
            self.assertEqual(again.scheduler_encoder.last_epoch, 2)
            advance(again)
            resumed.update_state(7)
            advance(resumed)
            self.assertEqual(again.scheduler_encoder.state_dict(), resumed.scheduler_encoder.state_dict())

    def test_phase_scheduler_type_survives_resaving_with_legacy_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_model(root, phase2_optim_step=3, phase2_optim_gamma=0.8)
            source.update_state(120)
            advance(source)
            source.save_model("phase.pth")
            resumed = make_model(root)
            resumed.load_model("phase.pth")
            resumed.save_model("resaved.pth")
            again = make_model(root)
            again.load_model("resaved.pth")
            self.assertIsInstance(again.scheduler_encoder, networks.PhaseStepLR)
            self.assertEqual(again.scheduler_encoder.state_dict(), resumed.scheduler_encoder.state_dict())

    def test_legacy_checkpoint_does_not_enable_new_phase_settings_without_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_model(root)
            advance(source)
            source.save_model("legacy.pth")
            resumed = make_model(root, phase2_optim_step=40)
            resumed.load_model("legacy.pth")
            self.assertNotIsInstance(resumed.scheduler_encoder, networks.PhaseStepLR)
            advance(resumed)
            self.assertEqual(resumed.optimizer_encoder.param_groups[0]["lr"], 5e-5)

    def test_phase_reset_uses_checkpoint_phase_and_new_phase_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_model(root, epoch_gap=4)
            source.update_state(4)
            advance(source)
            source.save_model("boundary.pth")
            resumed = make_model(root, epoch_gap=4, start_epoch=5, reset_lr_on_resume=True,
                                 phase1_optim_step=3, phase2_optim_step=2, phase2_optim_gamma=0.8)
            resumed.load_model("boundary.pth")
            self.assertEqual(resumed.scheduler_encoder.training_phase, 1)
            resumed.update_state(5)
            self.assertEqual(resumed.scheduler_encoder.training_phase, 2)
            self.assertEqual(resumed.scheduler_encoder.last_epoch, 0)
            advance(resumed)
            resumed.update_state(6)
            advance(resumed)
            self.assertAlmostEqual(resumed.optimizer_encoder.param_groups[0]["lr"], 8e-5)

    def test_partial_phase_config_falls_back_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(Path(tmp), phase2_optim_step=5)
            settings = model.scheduler_encoder.phase_settings
            self.assertEqual(settings[1], {"step": 2, "gamma": 0.5})
            self.assertEqual(settings[2], {"step": 5, "gamma": 0.5})
            for values in ({"phase1_optim_step": 0}, {"phase2_optim_step": 1.5},
                           {"phase1_optim_gamma": float("nan")}, {"phase2_optim_gamma": 2}):
                with self.subTest(values=values):
                    with self.assertRaises(ValueError):
                        make_model(Path(tmp), **values)

    def test_linear_can_restart_without_epoch_count_in_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(Path(tmp), lr_policy="linear")
            model.reset_learning_rate_and_schedulers()
            advance(model)
            self.assertEqual(model.scheduler_encoder.last_epoch, 1)
            self.assertEqual(model.optimizer_encoder.param_groups[0]["lr"], 1e-4)


if __name__ == "__main__":
    unittest.main()

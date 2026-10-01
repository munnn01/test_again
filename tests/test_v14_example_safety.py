from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config
from src.engine import (
    _task_regret_constraint_settings,
    _task_regret_terms,
    _validate_task_regret_analyzer,
)
from src.losses import classification_regret_terms
from src.tasks.action_recognition import ActionRecognitionAnalyzer
from src.tasks.base import TaskAnalyzer
from src.tasks.multi_teacher import MultiTeacherAnalyzer

ROOT = Path(__file__).resolve().parents[1]


class ToyClassifier(TaskAnalyzer):
    task_name = "action_recognition"
    supports_task_logits = True

    def __init__(self, sign=1.0):
        super().__init__()
        self.net = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.net.weight.copy_(torch.eye(2) * sign)
        self.logit_calls = 0

    def logits(self, x_hat):
        self.logit_calls += 1
        return self.net(x_hat)

    def accuracy_loss(self, x_hat, target):
        logits = self.logits(x_hat)
        return F.cross_entropy(logits, target), {"logits": logits.detach()}

    @torch.no_grad()
    def predict(self, x_hat):
        return self.logits(x_hat)


class ScalarTask(TaskAnalyzer):
    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name

    def accuracy_loss(self, x_hat, target):
        return ((x_hat - target) ** 2).mean(), {}

    @torch.no_grad()
    def predict(self, x_hat):
        return x_hat


def settings(**overrides):
    raw = {"enabled": True, "mode": "positive_per_example",
           "margin_weight": 1.0, "kl_weight": 0.5, **overrides}
    return _task_regret_constraint_settings(
        {"loss": {"task_regret_constraint": raw}}, [30, 35], ("h264",)
    )


def test_positive_regret_cannot_cancel_another_examples_damage():
    anchor = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    treatment = torch.tensor([[1.0, 0.0], [2.0, 0.0]], requires_grad=True)
    target = torch.zeros(2, dtype=torch.long)
    parts = classification_regret_terms(
        treatment, anchor, target, margin_weight=1.0, kl_weight=0.5
    )
    assert parts["signed_regret"] < 0  # V13 would release its task dual.
    expected = (F.cross_entropy(treatment, target, reduction="none") -
                F.cross_entropy(anchor, target, reduction="none")).clamp_min(0).mean()
    assert torch.allclose(parts["positive_ce_regret"], expected)
    assert parts["positive_ce_regret"] > 0
    assert parts["margin_guard"].item() == pytest.approx(0.5)
    assert parts["kl_guard"] > 0
    assert parts["regret"] > parts["positive_ce_regret"]
    parts["regret"].backward()
    assert anchor.grad is None
    assert treatment.grad is not None and treatment.grad[0].abs().sum() > 0
    assert torch.isfinite(treatment.grad).all()
    # The improved example has an incorrect anchor: no CE damage or safeguards.
    assert treatment.grad[1].abs().sum() == 0


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_identity_is_zero_and_input_gradients_finite(temperature):
    anchor = torch.tensor([[3.0, 0.0, -1.0], [0.0, 2.0, 1.0]], requires_grad=True)
    treatment = anchor.detach().clone().requires_grad_(True)
    parts = classification_regret_terms(
        treatment, anchor, torch.tensor([0, 0]), margin_weight=1.0,
        kl_weight=0.5, temperature=temperature,
    )
    for key in ("positive_ce_regret", "margin_guard", "kl_guard", "regret"):
        assert parts[key].item() == 0.0
    parts["regret"].backward()
    assert anchor.grad is None
    assert torch.isfinite(treatment.grad).all()


def test_both_safeguards_ignore_incorrect_anchor():
    anchor = torch.tensor([[0.0, 2.0]], requires_grad=True)
    treatment = torch.tensor([[-1.0, 3.0]], requires_grad=True)
    parts = classification_regret_terms(
        treatment, anchor, torch.tensor([0]), margin_weight=1.0, kl_weight=0.5,
    )
    assert parts["positive_ce_regret"] > 0
    assert parts["margin_guard"].item() == parts["kl_guard"].item() == 0.0


def test_ar_logits_are_differentiable_and_legacy_logging_detaches():
    # Construct a small stand-in network to avoid pretrained-weight downloads.
    ar = ActionRecognitionAnalyzer.__new__(ActionRecognitionAnalyzer)
    TaskAnalyzer.__init__(ar)
    ar.clip_size = 2
    ar.net = nn.Sequential(nn.Flatten(), nn.Linear(24, 3))
    ar.register_buffer("mean", torch.zeros(1, 3, 1, 1, 1))
    ar.register_buffer("std", torch.ones(1, 3, 1, 1, 1))
    ar.freeze()
    x = torch.rand(2, 3, 2, 2, 2, requires_grad=True)
    components = ar.task_logits(x)
    assert components[0][0] == 1.0 and components[0][1].requires_grad
    loss, aux = ar.accuracy_loss(x, torch.tensor([0, 1]))
    assert not aux["logits"].requires_grad
    assert not ar.predict(x).requires_grad
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is None for p in ar.parameters())


def test_sampled_teacher_is_pinned_across_anchor_treatment_and_scalar_loss(monkeypatch):
    panel = MultiTeacherAnalyzer([ToyClassifier(1), ToyClassifier(-1)]).freeze()
    calls = []

    def draw(weights, count):
        calls.append(count)
        return torch.tensor([1])

    monkeypatch.setattr(torch, "multinomial", draw)
    panel.pin_active()
    anchor = torch.tensor([[0.0, 2.0]], requires_grad=True)
    treatment = torch.tensor([[1.0, 1.0]], requires_grad=True)
    target = torch.tensor([0])
    parts = _task_regret_terms(panel, treatment, anchor, target, settings())
    panel.accuracy_loss(treatment, target)
    assert calls == [1]
    assert panel._active == 1
    assert panel.teachers[0].logit_calls == 0
    assert panel.teachers[1].logit_calls == 3
    parts["regret"].backward()
    assert anchor.grad is None
    assert treatment.grad is not None and treatment.grad.abs().sum() > 0
    assert all(p.grad is None for p in panel.parameters())
    panel.unpin_active()
    panel.task_logits(treatment)
    assert calls == [1, 1]


def test_unpinned_sample_panel_fails_before_mismatched_anchor_forward():
    panel = MultiTeacherAnalyzer([ToyClassifier(), ToyClassifier(-1)]).freeze()
    with pytest.raises(ValueError, match="must be pinned"):
        _task_regret_terms(panel, torch.ones(1, 2), torch.ones(1, 2),
                           torch.tensor([0]), settings())
    assert all(t.logit_calls == 0 for t in panel.teachers)


def test_mean_teacher_mode_averages_nonlinear_regrets_and_keeps_input_gradients():
    panel = MultiTeacherAnalyzer(
        [ToyClassifier(1), ToyClassifier(-1)], sampling="mean", weights=[1, 3]
    ).freeze()
    panel.pin_active()
    anchor = torch.tensor([[2.0, 0.0]], requires_grad=True)
    treatment = torch.tensor([[1.0, 0.0]], requires_grad=True)
    target = torch.tensor([0])
    parts = _task_regret_terms(panel, treatment, anchor, target, settings())
    individual = [classification_regret_terms(
        t.logits(treatment), t.logits(anchor), target,
        margin_weight=1.0, kl_weight=0.5,
    ) for t in panel.teachers]
    for key in parts:
        assert torch.allclose(parts[key], 0.25 * individual[0][key] + 0.75 * individual[1][key])
    # One teacher improves, but cannot erase the other's positive regret.
    assert parts["signed_regret"] < 0
    assert parts["positive_ce_regret"] > 0
    parts["regret"].backward()
    assert anchor.grad is None
    assert treatment.grad is not None and treatment.grad.abs().sum() > 0
    assert all(p.grad is None for p in panel.parameters())
    panel.unpin_active()


@pytest.mark.parametrize("task_name", ["object_detection", "tracking"])
def test_default_scalar_regret_preserves_nonclassification_tasks(task_name):
    cfg = {"loss": {"task_regret_constraint": {"enabled": True}}}
    legacy = _task_regret_constraint_settings(cfg, [30], ("h264",))
    assert legacy["mode"] == "mean"
    assert legacy["margin_weight"] == legacy["kl_weight"] == 0.0
    assert legacy["temperature"] == 1.0
    analyzer = ScalarTask(task_name).freeze()
    _validate_task_regret_analyzer(legacy, analyzer)
    treatment = torch.tensor([3.0], requires_grad=True)
    anchor = torch.tensor([2.0], requires_grad=True)
    parts = _task_regret_terms(analyzer, treatment, anchor, 0.0, legacy)
    assert parts["task_loss"].item() == 9.0
    assert parts["anchor_loss"].item() == 4.0
    assert parts["regret"].item() == 5.0
    parts["regret"].backward()
    assert treatment.grad.item() == 6.0 and anchor.grad is None
    with pytest.raises(ValueError, match="differentiable class logits"):
        _validate_task_regret_analyzer(settings(), analyzer)


def test_positive_mode_requires_frozen_training_teachers():
    with pytest.raises(ValueError, match="frozen training teachers"):
        _validate_task_regret_analyzer(settings(), ToyClassifier())
    mixed = MultiTeacherAnalyzer([ToyClassifier().freeze(), ScalarTask("tracking").freeze()])
    with pytest.raises(ValueError, match="differentiable class logits"):
        _validate_task_regret_analyzer(settings(), mixed)


@pytest.mark.parametrize("overrides,match", [
    ({"mode": "unknown"}, "mode must be"),
    ({"margin_weight": -1}, "non-negative"),
    ({"kl_weight": float("nan")}, "finite"),
    ({"temperature": 0}, "positive"),
    ({"temperature": float("inf")}, "finite"),
    ({"mode": "mean", "margin_weight": 1}, "require mode"),
])
def test_bad_safety_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        settings(**overrides)


@pytest.mark.parametrize("raw", [[], "positive_per_example", 2])
def test_constraint_settings_require_mapping(raw):
    with pytest.raises(ValueError, match="must be a mapping"):
        _task_regret_constraint_settings({"loss": {"task_regret_constraint": raw}}, [30])


def test_missing_constraint_and_missing_optional_safeguards_have_safe_defaults():
    empty = _task_regret_constraint_settings({}, [30])
    assert empty["enabled"] is False and empty["mode"] == "mean"
    ce_only = _task_regret_constraint_settings(
        {"loss": {"task_regret_constraint": {"enabled": True,
                                              "mode": "positive_per_example"}}}, [30]
    )
    assert ce_only["margin_weight"] == ce_only["kl_weight"] == 0.0
    assert ce_only["temperature"] == 1.0


@pytest.mark.parametrize("weights", [[-1, 2], [0, 0], [1, float("nan")]])
def test_invalid_teacher_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="teacher weights"):
        MultiTeacherAnalyzer([ToyClassifier(), ToyClassifier()], weights=weights)


def test_n4_config_is_fixed_and_matches_v13_common_architecture_and_regularizers():
    cfg = load_config(str(ROOT / "configs/v14_example_safety_ar.yaml"))
    v13 = load_config(str(ROOT / "configs/v13_task_regret_ar.yaml"))
    st = _task_regret_constraint_settings(cfg, cfg["train"]["qp_list"], ("h264",))
    assert cfg["seed"] == 302002
    assert cfg["train"]["epochs"] == 1 and cfg["train"]["max_steps"] == 1000
    assert cfg["train"]["finetune"] is True and cfg["train"]["resume"] is False
    assert cfg["model"] == v13["model"] and cfg["task"] == v13["task"]
    for key in ("kappa", "kappa_t", "mu", "omega", "tau", "gamma", "delta"):
        assert cfg["loss"][key] == v13["loss"][key]
    assert st["mode"] == "positive_per_example"
    assert st["margin_weight"] == 1.0 and st["kl_weight"] == 0.5
    assert st["temperature"] == 1.0 and st["epsilon"] == 0.0
    assert st["mu_init"] == 1.0 and st["mu_max"] == 100.0
    assert st["per_codec"]["h264"] == {"epsilon": 0.0, "dual_lr": 0.02, "rate_weight": 0.25}
    assert cfg["eval"] == v13["eval"]

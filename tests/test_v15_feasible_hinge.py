from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src import engine
from src.config import load_config
from src.models.ste_codec import STECodec


ROOT = Path(__file__).resolve().parents[1]


class Edit(nn.Module):
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.2))

    def forward(self, clips, cond, mask=None):
        return clips + self.amount


class Codec(STECodec):
    """Finite differentiable stand-in; no external codec or data required."""

    def __init__(self, *, alternate=False):
        nn.Module.__init__(self)
        self.codec = "h264"
        self.alternate = alternate
        self.qualities = [1, 2]
        self.forward_calls = 0

    def forward(self, clips, quality):
        self.forward_calls += 1
        # The edit improves the task in the feasible case but costs more bits.
        return clips, 1.0 + 0.1 * clips.mean()

    def compress_decompress(self, clips, quality):
        return clips.detach(), 1.0


class Analyzer:
    def __init__(self, codec, *, feasible=True, by_codec=False, fail=False):
        self.codec = codec
        self.feasible = feasible
        self.by_codec = by_codec
        self.fail = fail
        self.active = None
        self.pin_count = 0
        self.unpin_count = 0
        self.calls = []
        # A cached leaf returned inside no_grad still requires explicit detach.
        self.anchor = torch.tensor(1.0, requires_grad=True)

    def pin_active(self):
        assert self.active is None
        self.pin_count += 1
        self.active = self.pin_count

    def unpin_active(self):
        assert self.active is not None
        self.unpin_count += 1
        self.active = None

    def accuracy_loss(self, clips, target):
        assert self.active is not None, "both arms require the same pinned teacher"
        self.calls.append((self.active, torch.is_grad_enabled()))
        if not torch.is_grad_enabled():
            return self.anchor, {}
        if self.fail:
            raise RuntimeError("treatment failed")
        feasible = self.codec.codec == "h264" if self.by_codec else self.feasible
        sign = -1.0 if feasible else 1.0
        return 1.0 + sign * clips.mean(), {}

    def features(self, _clips):
        return []


class Tracker:
    def __init__(self):
        self.records = []

    def log_params(self, _params):
        pass

    def log_step(self, step, metrics):
        self.records.append((step, metrics))

    def finish(self, _metrics):
        pass


class Progress:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def set_postfix(self, **_kwargs):
        pass


def run_fit(tmp_path, monkeypatch, *, mode="positive_part", feasible=True,
            epsilon=0.0, alternate=False, two_qps=False, fail=False,
            tag="action_recognition"):
    cfg = copy.deepcopy(load_config(str(ROOT / "configs/v14_dual_safety_ar.yaml")))
    cfg["out_dir"] = str(tmp_path)
    cfg["task"]["name"] = tag
    constraint = cfg["loss"]["task_regret_constraint"]
    if mode is not None:
        constraint["primal_mode"] = mode
    constraint["epsilon"] = epsilon
    for values in constraint["per_codec"].values():
        values["epsilon"] = epsilon
    # Isolate rate and task gradients in the integration fixture.
    cfg["loss"]["kappa"] = cfg["loss"]["mu"] = 0.0
    cfg["train"].update(finetune=False, epochs=1, lr=0.001, clip_grad=0.0,
                         qp_list=[30, 35] if two_qps else [30],
                         qp_to_quality={30: 2, 35: 1} if two_qps else {30: 2})
    n_batches = (2 if alternate else 1) * (2 if two_qps else 1)
    cfg["train"]["max_steps"] = n_batches
    pre = Edit()
    codec = Codec(alternate=alternate)
    analyzer = Analyzer(codec, feasible=feasible, by_codec=alternate, fail=fail)
    tracker = Tracker()
    monkeypatch.setattr(engine, "make_tracker", lambda *_args, **_kwargs: tracker)
    monkeypatch.setattr(engine, "tqdm", lambda batches, **_kwargs: Progress(batches))
    batches = [(torch.zeros(1, 3, 2, 4, 4), torch.tensor([0]))] * n_batches
    engine._fit(cfg, pre, codec, analyzer, batches, None, lambda batch: batch,
                tag=tag, n_train=n_batches)
    state = json.loads((tmp_path / "task_regret_constraint_state.json").read_text())
    return pre, codec, analyzer, tracker, state


@pytest.mark.parametrize("tag", ["action_recognition", "object_detection", "tracking"])
def test_feasible_hinge_removes_confidence_bonus_and_reduces_rate(
    tmp_path, monkeypatch, tag
):
    pre, _, analyzer, tracker, state = run_fit(tmp_path, monkeypatch, tag=tag)
    metrics = tracker.records[0][1]
    assert metrics["rate_excess_ratio"] > 0.0
    assert metrics["task_regret"] == pytest.approx(-0.2)
    assert metrics["task_violation"] == pytest.approx(-0.2)
    assert metrics["loss_task_objective"] == 0.0
    assert pre.amount.grad.item() == pytest.approx(0.25 * 0.1)
    assert pre.amount.item() < 0.2  # descend toward lower rate
    assert state["duals"]["h264:30"] == pytest.approx(10.0 - 0.02 * 0.2)
    assert analyzer.anchor.grad is None
    assert analyzer.calls == [(1, False), (1, True)]
    assert analyzer.pin_count == analyzer.unpin_count == 1
    assert analyzer.active is None


def test_violating_hinge_restores_task_protection(tmp_path, monkeypatch):
    pre, _, analyzer, tracker, state = run_fit(
        tmp_path, monkeypatch, feasible=False, epsilon=0.1
    )
    metrics = tracker.records[0][1]
    assert metrics["task_regret"] == pytest.approx(0.2)
    assert metrics["task_violation"] == pytest.approx(0.1)
    assert metrics["loss_task_objective"] == pytest.approx(0.1)
    dual = 10.0 + 0.02 * 0.1
    assert state["duals"]["h264:30"] == pytest.approx(dual)
    assert pre.amount.grad.item() == pytest.approx(dual + 0.25 * 0.1)
    assert pre.amount.item() < 0.2  # reduces the synthetic task regret
    assert analyzer.anchor.grad is None


def test_hinge_uses_budget_not_zero_regret(tmp_path, monkeypatch):
    pre, _, _, tracker, state = run_fit(
        tmp_path, monkeypatch, feasible=False, epsilon=0.3
    )
    metrics = tracker.records[0][1]
    assert metrics["task_regret"] == pytest.approx(0.2)
    assert metrics["task_violation"] == pytest.approx(-0.1)
    assert metrics["loss_task_objective"] == 0.0
    assert pre.amount.grad.item() == pytest.approx(0.25 * 0.1)
    assert state["duals"]["h264:30"] == pytest.approx(10.0 - 0.02 * 0.1)


def test_omitted_mode_matches_signed_path_exactly(tmp_path, monkeypatch):
    default = run_fit(tmp_path / "default", monkeypatch, mode=None)
    signed = run_fit(tmp_path / "signed", monkeypatch, mode="signed")
    assert torch.equal(default[0].amount, signed[0].amount)
    assert torch.equal(default[0].amount.grad, signed[0].amount.grad)
    for (step_a, values_a), (step_b, values_b) in zip(
        default[3].records, signed[3].records, strict=True
    ):
        assert step_a == step_b
        assert values_a == pytest.approx(values_b, rel=0.0, abs=0.0, nan_ok=True)
    assert default[0].amount.grad.item() == pytest.approx(-9.996 + 0.25 * 0.1)
    assert default[3].records[0][1]["loss_task_objective"] == pytest.approx(-0.2)
    assert default[4] == signed[4]
    assert default[0].amount.item() > 0.2  # signed term buys additional confidence
    assert default[4]["settings"]["primal_mode"] == "signed"


def test_duals_remain_independent_for_each_codec_qp(tmp_path, monkeypatch):
    _, _, analyzer, tracker, state = run_fit(
        tmp_path, monkeypatch, alternate=True, two_qps=True
    )
    assert set(state["duals"]) == {"h264:30", "h264:35", "h265:30", "h265:35"}
    assert all(value < 10.0 for cell, value in state["duals"].items()
               if cell.startswith("h264:"))
    assert all(value > 10.0 for cell, value in state["duals"].items()
               if cell.startswith("h265:"))
    for _, metrics in tracker.records:
        if metrics["codec_is_h265"]:
            assert metrics["task_violation"] > 0.0
            assert metrics["loss_task_objective"] > 0.0
        else:
            assert metrics["task_violation"] < 0.0
            assert metrics["loss_task_objective"] == 0.0
    assert analyzer.pin_count == analyzer.unpin_count == 4
    for i in range(4):
        assert analyzer.calls[2*i:2*i+2] == [(i+1, False), (i+1, True)]
    assert analyzer.anchor.grad is None


@pytest.mark.parametrize("mode", ["absolute", "positive_per_example", "", None])
def test_settings_reject_unknown_explicit_modes(mode):
    cfg = {"loss": {"task_regret_constraint": {"primal_mode": mode}}}
    with pytest.raises(ValueError, match="primal_mode must be signed or positive_part"):
        engine._task_regret_constraint_settings(cfg, [30])


def test_unknown_mode_fails_before_training_forward(tmp_path):
    codec = Codec()
    analyzer = Analyzer(codec)
    cfg = load_config(str(ROOT / "configs/v15_feasible_hinge_ar.yaml"))
    cfg["out_dir"] = str(tmp_path)
    cfg["loss"]["task_regret_constraint"]["primal_mode"] = "absolute"
    cfg["train"].update(finetune=False, max_steps=1, qp_list=[30],
                         qp_to_quality={30: 2})
    batches = [(torch.zeros(1, 3, 2, 4, 4), torch.tensor([0]))]
    with pytest.raises(ValueError, match="primal_mode must be signed or positive_part"):
        engine._fit(cfg, Edit(), codec, analyzer, batches, None, lambda batch: batch,
                    tag="action_recognition", n_train=1)
    assert codec.forward_calls == 0
    assert analyzer.calls == []
    assert analyzer.pin_count == 0
    assert not (tmp_path / "checkpoints").exists()


def test_failed_treatment_releases_pinned_teacher(tmp_path, monkeypatch):
    codec = Codec()
    analyzer = Analyzer(codec, fail=True)
    cfg = load_config(str(ROOT / "configs/v15_feasible_hinge_ar.yaml"))
    cfg["out_dir"] = str(tmp_path)
    cfg["train"].update(finetune=False, max_steps=1, qp_list=[30],
                         qp_to_quality={30: 2})
    monkeypatch.setattr(engine, "make_tracker", lambda *_args, **_kwargs: Tracker())
    monkeypatch.setattr(engine, "tqdm", lambda batches, **_kwargs: Progress(batches))
    batches = [(torch.zeros(1, 3, 2, 4, 4), torch.tensor([0]))]
    with pytest.raises(RuntimeError, match="treatment failed"):
        engine._fit(cfg, Edit(), codec, analyzer, batches, None, lambda batch: batch,
                    tag="action_recognition", n_train=1)
    assert analyzer.active is None
    assert analyzer.pin_count == analyzer.unpin_count == 1


def test_registered_config_changes_only_one_parsed_field():
    source = load_config(str(ROOT / "configs/v14_dual_safety_ar.yaml"))
    candidate = load_config(str(ROOT / "configs/v15_feasible_hinge_ar.yaml"))
    assert candidate["loss"]["task_regret_constraint"].pop("primal_mode") == "positive_part"
    assert candidate == source

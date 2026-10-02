from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src import engine
from src.config import load_config
from src.losses import LossWeights, preprocessing_loss
from src.models.ste_codec import STECodec

ROOT = Path(__file__).resolve().parents[1]


class Features(nn.Module):
    def __init__(self, *, image=False, empty=False, fail=False, require_pin=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.image, self.empty, self.fail = image, empty, fail
        self.require_pin = require_pin
        self.active = None
        self.pins = self.unpins = 0
        self.calls = []

    def pin_active(self):
        assert self.active is None
        self.pins += 1
        self.active = self.pins

    def unpin_active(self):
        assert self.active is not None
        self.unpins += 1
        self.active = None

    def record(self, kind, x):
        if self.require_pin:
            assert self.active is not None
        self.calls.append((kind, self.active, x.mean().item(),
                           torch.is_grad_enabled(), x.data_ptr()))

    def accuracy_loss(self, x, target):
        self.record("ce", x)
        return 1.0 - x.mean(), {}

    def features(self, x):
        self.record("features", x)
        if self.fail:
            raise RuntimeError("features failed")
        if self.empty:
            return []
        return [self.scale * (x[:, :, 0] if self.image else x)]


class Edit(nn.Module):
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, cond, mask=None):
        return x + self.amount


class Codec(STECodec):
    def __init__(self):
        nn.Module.__init__(self)
        self.codec = "h264"
        self.qualities = [2]
        self.anchor_value = nn.Parameter(torch.tensor(0.5))
        self.forward_calls = []
        self.anchor_calls = []

    def forward(self, x, quality):
        self.forward_calls.append((self.codec, quality))
        return x - 0.25, 1.0 + 0.1 * x.mean()

    def compress_decompress(self, x, quality):
        self.anchor_calls.append((self.codec, quality))
        # Deliberately live graph: the shared engine must detach raw targets.
        return self.anchor_value.expand_as(x), 1.0


class Tracker:
    def __init__(self):
        self.steps = []

    def log_params(self, _params):
        pass

    def log_step(self, _step, metrics):
        self.steps.append(metrics)

    def log_epoch(self, *_args):
        pass

    def finish(self, _metrics):
        pass


class Progress:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def set_postfix(self, **_kwargs):
        pass


def weights(**kwargs):
    values = dict(lam_task=0.0, omega=0.5, beta=0.0, tau=0.0,
                  delta=0.0, gamma=0.0, gamma_res=0.0, kappa=0.0,
                  kappa_t=0.0, mu=0.0, rho=0.0)
    values.update(kwargs)
    return LossWeights(**values)


def cfg_for(tmp_path):
    cfg = load_config(str(ROOT / "configs/v16_semantic_anchor_ar.yaml"))
    cfg["out_dir"] = str(tmp_path)
    cfg["loss"]["kappa"] = 0.0  # finite fixture isolates semantic/rate gradients
    cfg["train"].update(finetune=False, max_steps=1, qp_list=[30],
                         qp_to_quality={30: 2}, lr=0.001, clip_grad=0.0)
    return cfg


def run_fit(tmp_path, monkeypatch, *, mode="raw_codec", validation=False,
            empty=False, fail=False, constrained=True):
    cfg = cfg_for(tmp_path)
    cfg["loss"]["distill_anchor"] = mode
    cfg["loss"]["task_regret_constraint"]["enabled"] = constrained
    pre, codec = Edit(), Codec()
    analyzer = Features(empty=empty, fail=fail, require_pin=True)
    tracker = Tracker()
    monkeypatch.setattr(engine, "make_tracker", lambda *_args, **_kwargs: tracker)
    monkeypatch.setattr(engine, "tqdm", lambda batches, **_kwargs: Progress(batches))
    batches = [(torch.full((1, 3, 2, 4, 4), 0.75), torch.tensor([0]))]
    engine._fit(cfg, pre, codec, analyzer, batches, batches if validation else None,
                lambda batch: batch, tag="fixture", n_train=1)
    return pre, codec, analyzer, tracker


@pytest.mark.parametrize("image", [False, True])
def test_decoded_anchor_changes_feature_gradient_and_detaches_targets(image):
    shape = (1, 3, 1 if image else 2, 4, 4)
    source = torch.full(shape, 2.0, requires_grad=True)
    anchor = torch.full(shape, 4.0, requires_grad=True)
    recon = torch.full(shape, 3.0, requires_grad=True)
    analyzer = Features(image=image)
    old = preprocessing_loss(analyzer, source, recon, torch.tensor(0.1), None,
                             weights(), task_loss=recon.new_zeros(()))
    old_grad = torch.autograd.grad(old["loss"], recon)[0]
    raw = preprocessing_loss(analyzer, source, recon, torch.tensor(0.1), None,
                             weights(), task_loss=recon.new_zeros(()),
                             feature_anchor=anchor)
    raw["loss"].backward()
    assert old_grad.sum().item() == pytest.approx(0.25)
    assert recon.grad.sum().item() == pytest.approx(-0.0625)
    assert anchor.grad is None
    # The unchanged zero-weight temporal term can allocate a zero source grad.
    assert source.grad is None or torch.count_nonzero(source.grad).item() == 0
    assert analyzer.scale.grad is None
    assert old_grad.isfinite().all() and recon.grad.isfinite().all()
    # New target features use no_grad; reconstructed features stay differentiable.
    assert analyzer.calls[-2][3] is False
    assert analyzer.calls[-1][3] is True


def test_feature_anchor_does_not_change_pixel_or_temporal_source_meaning():
    source = torch.stack((torch.ones(1, 3, 4, 4),
                          torch.full((1, 3, 4, 4), 3.0)), dim=2)
    recon = torch.full_like(source, 2.0)
    anchor = torch.full_like(source, 0.5)
    analyzer = Features()
    w = weights(mu=2.0, tau=3.0)
    common = dict(analyzer=analyzer, x_source=source, x_hat=recon,
                  bpp=torch.tensor(0.1), target=None, w=w,
                  task_loss=torch.tensor(0.0))
    original = preprocessing_loss(**common)
    altered = preprocessing_loss(**common, feature_anchor=anchor)
    assert altered["loss_d"].item() == original["loss_d"].item() == 1.0
    assert altered["loss_temp"].item() == original["loss_temp"].item() == 4.0
    assert altered["loss_dist"].item() != original["loss_dist"].item()


@pytest.mark.parametrize("empty", [False, True])
def test_omitted_feature_anchor_matches_original_source_formula(empty):
    torch.manual_seed(0)
    source = torch.rand(1, 3, 2, 4, 4, requires_grad=True)
    recon = torch.rand(1, 3, 2, 4, 4, requires_grad=True)
    analyzer = Features(empty=empty)
    parts = preprocessing_loss(analyzer, source, recon, torch.tensor(0.1), None,
                               weights(), task_loss=recon.new_zeros(()))
    if empty:
        assert parts["loss_dist"].item() == parts["loss"].item() == 0.0
    else:
        fs, fh = 2 * source.detach(), 2 * recon
        expected = 0.5 * F.mse_loss(fh, fs) / (fs.pow(2).mean() + 1e-6)
        assert torch.equal(parts["loss"], expected)
        got_grad = torch.autograd.grad(parts["loss"], recon, retain_graph=True)[0]
        old_grad = torch.autograd.grad(expected, recon)[0]
        assert torch.equal(got_grad, old_grad)
    assert source.grad is None and analyzer.scale.grad is None


def test_fit_reuses_task_anchor_and_pins_both_feature_arms(tmp_path, monkeypatch):
    pre, codec, analyzer, tracker = run_fit(tmp_path, monkeypatch)
    assert codec.forward_calls == codec.anchor_calls == [("h264", 2)]
    assert pre.amount.grad.item() > 0.0 and pre.amount.item() < 0.1
    assert codec.anchor_value.grad is None and analyzer.scale.grad is None
    assert analyzer.pins == analyzer.unpins == 1 and analyzer.active is None
    assert [call[0] for call in analyzer.calls] == ["ce", "ce", "features", "features"]
    assert [call[1] for call in analyzer.calls] == [1, 1, 1, 1]
    assert [call[3] for call in analyzer.calls] == [False, True, False, True]
    assert analyzer.calls[0][4] == analyzer.calls[2][4]  # same raw storage
    assert tracker.steps[0]["task_regret"] == pytest.approx(-0.1)
    state = json.loads((tmp_path / "task_regret_constraint_state.json").read_text())
    assert state["duals"]["h264:30"] < 10.0


def test_source_and_raw_fit_have_opposite_edit_gradients(tmp_path, monkeypatch):
    old = run_fit(tmp_path / "source", monkeypatch, mode="source")
    raw = run_fit(tmp_path / "raw", monkeypatch)
    assert old[0].amount.grad.item() < 0.0 < raw[0].amount.grad.item()
    assert old[0].amount.item() > 0.1 > raw[0].amount.item()
    assert old[1].anchor_calls == raw[1].anchor_calls == [("h264", 2)]


def test_fit_keeps_actual_ste_for_validation_and_pins_ce_features(tmp_path, monkeypatch):
    pre, codec, analyzer, _ = run_fit(tmp_path, monkeypatch, validation=True)
    assert codec.forward_calls == codec.anchor_calls == [("h264", 2), ("h264", 2)]
    assert analyzer.pins == analyzer.unpins == 2 and analyzer.active is None
    validation_calls = analyzer.calls[4:]
    assert [call[0] for call in validation_calls] == ["ce", "features", "features"]
    assert [call[1] for call in validation_calls] == [2, 2, 2]
    assert all(call[3] is False for call in validation_calls)
    assert validation_calls[1][2] == pytest.approx(0.5)
    assert pre.training is True
    assert codec.anchor_value.grad is None


def test_raw_anchor_works_without_task_regret_constraint(tmp_path, monkeypatch):
    _, codec, analyzer, _ = run_fit(tmp_path, monkeypatch, constrained=False)
    assert codec.anchor_calls == [("h264", 2)]
    assert analyzer.pins == analyzer.unpins == 1
    assert [call[0] for call in analyzer.calls] == ["ce", "features", "features"]


@pytest.mark.parametrize("mode", ["codec_proxy", "", None])
def test_invalid_anchor_mode_fails_before_training_forward(tmp_path, mode):
    cfg = cfg_for(tmp_path)
    cfg["loss"]["distill_anchor"] = mode
    pre, codec, analyzer = Edit(), Codec(), Features()
    batch = [(torch.full((1, 3, 2, 4, 4), 0.75), torch.tensor([0]))]
    with pytest.raises(ValueError, match="distill_anchor must be source or raw_codec"):
        engine._fit(cfg, pre, codec, analyzer, batch, None, lambda x: x, "fixture", 1)
    assert codec.forward_calls == codec.anchor_calls == []
    assert analyzer.calls == []
    assert not (tmp_path / "checkpoints").exists()


def test_raw_anchor_rejects_non_ste_in_training_and_validation(tmp_path):
    cfg = cfg_for(tmp_path)
    pre, analyzer, proxy = Edit(), Features(), nn.Identity()
    batch = [(torch.full((1, 3, 2, 4, 4), 0.75), torch.tensor([0]))]
    with pytest.raises(ValueError, match="requires codec.kind=ste"):
        engine._fit(cfg, pre, proxy, analyzer, batch, None, lambda x: x, "fixture", 1)
    with pytest.raises(ValueError, match="requires codec.kind=ste"):
        engine._val_loss(pre, proxy, analyzer, batch, weights(), [30], {30: 2},
                         cfg, lambda x: x, 1)
    assert analyzer.calls == [] and pre.training


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_weighted_raw_features_fail_clearly_and_release_teacher(tmp_path, monkeypatch, failure):
    cfg = cfg_for(tmp_path)
    pre, codec = Edit(), Codec()
    analyzer = Features(empty=failure == "empty", fail=failure == "exception",
                        require_pin=True)
    monkeypatch.setattr(engine, "make_tracker", lambda *_args, **_kwargs: Tracker())
    monkeypatch.setattr(engine, "tqdm", lambda batches, **_kwargs: Progress(batches))
    batch = [(torch.full((1, 3, 2, 4, 4), 0.75), torch.tensor([0]))]
    error = ValueError if failure == "empty" else RuntimeError
    match = "requires nonempty analyzer.features" if failure == "empty" else "features failed"
    with pytest.raises(error, match=match):
        engine._fit(cfg, pre, codec, analyzer, batch, None, lambda x: x, "fixture", 1)
    assert analyzer.active is None and analyzer.pins == analyzer.unpins == 1
    assert codec.anchor_calls == [("h264", 2)]
    with pytest.raises(error, match=match):
        engine._val_loss(pre, codec, analyzer, batch, weights(), [30], {30: 2},
                         cfg, lambda x: x, 1)
    assert analyzer.active is None and analyzer.pins == analyzer.unpins == 2


def test_weight_zero_permits_empty_alternate_features():
    source = torch.zeros(1, 3, 2, 4, 4)
    parts = preprocessing_loss(Features(empty=True), source, source, torch.tensor(0.1),
                               None, weights(omega=0.0), task_loss=torch.tensor(0.0),
                               feature_anchor=source)
    assert parts["loss_dist"].item() == 0.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src import engine
from src.config import load_config
from src.models import ste_codec as module

ROOT = Path(__file__).resolve().parents[1]


class Proxy(nn.Module):
    def __init__(self, *, fixed=None, stochastic=False, **kwargs):
        super().__init__()
        self.qualities = list(kwargs.get("qualities", [2]))
        self.fixed, self.stochastic = fixed, stochastic

    def forward(self, x, quality):
        recon = x.sin()
        if self.stochastic:
            recon = recon + torch.rand_like(x)
        rate = 1.0 + x.square().mean()
        if self.fixed is not None:
            # Keep a nonzero derivative even for the exact floor fixture value.
            value = x.new_tensor(self.fixed)
            rate = value + (x.mean() - x.mean().detach())
        return recon, rate


@pytest.fixture
def real(monkeypatch):
    calls = []
    values = {"bpp": 2.0}

    class Real:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def compress_decompress(self, x):
            assert not torch.is_grad_enabled()
            assert not x.requires_grad
            calls.append((dict(self.kwargs), tuple(x.shape)))
            return 0.4 * x + 0.1, values["bpp"]

    monkeypatch.setattr(module, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(module, "StandardCodec", Real)
    return calls, values


def codec(proxy=None, *, calibrate=True):
    return module.STECodec(proxy or Proxy(), codec="h264",
                           quality_to_qp={2: 30}, preset="medium",
                           calibrate_rate=calibrate)


def test_disabled_path_matches_legacy_arithmetic_gradient_rng_and_calls(real):
    x = torch.linspace(0.1, 0.8, 12, dtype=torch.float64).reshape(1, 3, 1, 2, 2)
    x.requires_grad_(True)
    proxy = Proxy(stochastic=True)
    ste = module.STECodec(proxy, codec="h264", quality_to_qp={2: 30})
    torch.manual_seed(123)
    recon, bpp = ste(x, 2)
    rng_after = torch.get_rng_state().clone()
    torch.manual_seed(123)
    xp, bp = proxy(x, 2)
    xr = 0.4 * x.detach() + 0.1
    expected_recon = xp + (xr - xp).detach()
    expected_bpp = bp + (xp.new_tensor(2.0) - bp).detach()
    assert torch.equal(torch.get_rng_state(), rng_after)
    assert torch.equal(recon, expected_recon)
    assert torch.equal(bpp, expected_bpp)
    actual = torch.autograd.grad(recon.sum() + bpp, x, retain_graph=True)[0]
    expected = torch.autograd.grad(expected_recon.sum() + expected_bpp, x)[0]
    assert torch.equal(actual, expected)
    assert len(real[0]) == 1
    assert real[0][0][0] == {"codec": "h264", "qp": 30, "preset": "medium"}
    assert ste.calibrate_rate is False and ste.rate_calibration_diagnostics == {}


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_batch_mean_calibration_has_real_value_and_detached_analytic_derivative(real, batch):
    x = torch.linspace(0.1, 0.8, batch * 24, dtype=torch.float64).reshape(batch, 3, 2, 2, 2)
    x.requires_grad_(True)
    ste = codec()
    recon, bpp = ste(x, 2)
    proxy_bpp = 1.0 + x.detach().square().mean()
    scale = 2.0 / proxy_bpp
    assert bpp.item() == 2.0  # exact honest forward, not a surrogate rate
    rate_grad = torch.autograd.grad(bpp, x, retain_graph=True)[0]
    assert torch.allclose(rate_grad, scale * 2.0 * x.detach() / x.numel(), atol=1e-14)
    assert torch.count_nonzero(rate_grad) == x.numel()
    recon_grad = torch.autograd.grad(recon.sum(), x)[0]
    assert torch.equal(recon_grad, x.detach().cos())
    expected_recon = x.sin() + (0.4 * x.detach() + 0.1 - x.sin()).detach()
    assert torch.equal(recon, expected_recon)
    assert len(real[0]) == 1 and real[0][0][1][0] == batch
    diag = ste.rate_calibration_diagnostics
    assert diag["ste_rate_proxy_bpp"] == pytest.approx(proxy_bpp.item())
    assert diag["ste_rate_real_bpp"] == 2.0
    assert diag["ste_rate_calibration_scale"] == pytest.approx(scale.item())
    assert diag["ste_rate_floor_active"] == 0.0
    assert all(isinstance(value, float) for value in diag.values())
    # The tempting live ratio cancels the rate derivative, unlike this STE.
    bad_proxy = 1.0 + x.square().mean()
    bad_rate = bad_proxy * (2.0 / bad_proxy)
    bad_grad = torch.autograd.grad(bad_rate, x)[0]
    assert bad_grad.abs().max().item() < 1e-14


def test_reconstruction_value_and_gradient_match_unscaled_ste_exactly(real):
    x = torch.rand(2, 3, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    unscaled = codec(calibrate=False)
    scaled = codec(calibrate=True)
    x0, _ = unscaled(x, 2)
    x1, _ = scaled(x, 2)
    assert torch.equal(x0, x1)
    assert torch.equal(torch.autograd.grad(x0.sum(), x)[0],
                       torch.autograd.grad(x1.sum(), x)[0])
    assert len(real[0]) == 2  # one real encode per separate forward


@pytest.mark.parametrize("proxy_value,active", [(0.0, 1.0), (1e-10, 1.0),
                                               (1e-8, 0.0), (1e-6, 0.0)])
def test_fixed_floor_is_reported_and_does_not_cap_calibrated_derivative(real, proxy_value, active):
    real[1]["bpp"] = 0.2
    x = torch.zeros(1, 3, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    ste = codec(Proxy(fixed=proxy_value))
    _, bpp = ste(x, 2)
    bpp.backward()
    expected_scale = 0.2 / max(proxy_value, 1e-8)
    assert bpp.item() == 0.2
    assert torch.allclose(x.grad, torch.full_like(x, expected_scale / x.numel()))
    diag = ste.rate_calibration_diagnostics
    assert diag["ste_rate_floor_active"] == active
    assert diag["ste_rate_calibration_scale"] == expected_scale
    assert len(real[0]) == 1


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_bad_proxy_rate_fails_before_real_encode(real, value):
    ste = codec(Proxy(fixed=value))
    with pytest.raises(ValueError, match="proxy_bpp must be finite and non-negative"):
        ste(torch.zeros(1, 3, 1, 2, 2), 2)
    assert real[0] == [] and ste.rate_calibration_diagnostics == {}


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_bad_real_rate_fails_after_single_encode(real, value):
    real[1]["bpp"] = value
    ste = codec()
    with pytest.raises(ValueError, match="real_bpp must be finite and non-negative"):
        ste(torch.ones(1, 3, 1, 2, 2), 2)
    assert len(real[0]) == 1 and ste.rate_calibration_diagnostics == {}


def test_nonfinite_scale_is_rejected_without_caps(real):
    real[1]["bpp"] = 1e308
    ste = codec(Proxy(fixed=0.0))
    with pytest.raises(ValueError, match="scale must be finite and non-negative"):
        ste(torch.zeros(1, 3, 1, 2, 2, dtype=torch.float64), 2)
    assert len(real[0]) == 1


def test_scale_overflow_in_proxy_dtype_is_rejected(real):
    real[1]["bpp"] = 1e38
    ste = codec(Proxy(fixed=0.0))
    with pytest.raises(ValueError, match="scale is not finite in proxy dtype"):
        ste(torch.zeros(1, 3, 1, 2, 2, dtype=torch.float32), 2)
    assert len(real[0]) == 1


def test_real_rate_overflow_in_proxy_dtype_is_rejected(real):
    real[1]["bpp"] = 1e40
    with pytest.raises(ValueError, match="real_bpp is not finite in proxy dtype"):
        codec()(torch.zeros(1, 3, 1, 2, 2, dtype=torch.float32), 2)
    assert len(real[0]) == 1


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_non_floating_proxy_rate_fails_before_encode(real, dtype):
    class BadDtypeProxy(Proxy):
        def forward(self, x, quality):
            return x, torch.tensor(1, dtype=dtype)

    with pytest.raises(ValueError, match="requires floating-point proxy_bpp"):
        codec(BadDtypeProxy())(torch.zeros(1, 3, 1, 2, 2), 2)
    assert real[0] == []


def test_non_scalar_proxy_rate_is_rejected(real):
    class VectorProxy(Proxy):
        def forward(self, x, quality):
            return x, x.new_ones(2)

    with pytest.raises(ValueError, match="requires scalar proxy_bpp"):
        codec(VectorProxy())(torch.zeros(2, 3, 1, 2, 2), 2)
    assert real[0] == []


@pytest.mark.parametrize("value", [0, 1, 1.0, "true", None])
def test_constructor_requires_strict_bool_before_ffmpeg_check(monkeypatch, value):
    monkeypatch.setattr(module, "ffmpeg_available", lambda: False)
    with pytest.raises(ValueError, match="calibrate_rate must be a bool"):
        module.STECodec(Proxy(), calibrate_rate=value)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_factory_rejects_non_bool_before_model_construction(value):
    cfg = {"codec": {"kind": "ste", "ste_rate_calibration": value}}
    with pytest.raises(ValueError, match="ste_rate_calibration must be a bool"):
        engine._build_models(cfg, torch.device("cpu"))


@pytest.mark.parametrize("kind", ["virtual", "entropy", "compressai"])
def test_factory_rejects_non_ste_opt_in_before_model_construction(kind):
    cfg = {"codec": {"kind": kind, "ste_rate_calibration": True}}
    with pytest.raises(ValueError, match="requires codec.kind=ste"):
        engine._build_models(cfg, torch.device("cpu"))


def test_factory_passes_calibration_to_common_ste_wrapper(real, monkeypatch):
    # The independent log-proxy contribution is combined by the coordinator.
    cfg = load_config(str(ROOT / "configs/v15_feasible_hinge_ar.yaml"))
    cfg["codec"]["ste_rate_calibration"] = True
    monkeypatch.setattr(engine, "VirtualCodec", Proxy)
    monkeypatch.setattr(engine, "build_analyzer", lambda *_args, **_kwargs: nn.Identity())
    _, ste, _ = engine._build_models(cfg, torch.device("cpu"))
    assert isinstance(ste, module.STECodec) and ste.calibrate_rate is True
    assert ste.quality_to_qp == {8: 30, 5: 35, 3: 40, 2: 45, 1: 50}
    _, bpp = ste(torch.ones(1, 3, 1, 2, 2), 3)
    assert bpp.item() == 2.0 and len(real[0]) == 1


class Edit(nn.Module):
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, cond, mask=None):
        return x + self.amount


class Analyzer:
    def __init__(self):
        self.active = False
        self.pins = self.unpins = 0
        self.calls = []

    def pin_active(self):
        assert not self.active
        self.active = True
        self.pins += 1

    def unpin_active(self):
        assert self.active
        self.active = False
        self.unpins += 1

    def accuracy_loss(self, x, target):
        assert self.active
        self.calls.append(torch.is_grad_enabled())
        return 1.0 - x.mean(), {}

    def features(self, x):
        return []


class Tracker:
    def __init__(self):
        self.steps = []

    def log_params(self, _params):
        pass

    def log_step(self, _step, values):
        self.steps.append(values)

    def finish(self, _metrics):
        pass


class Progress:
    def __init__(self, batches, postfix):
        self.batches, self.postfix = batches, postfix

    def __iter__(self):
        return iter(self.batches)

    def set_postfix(self, **kwargs):
        self.postfix.append(kwargs)


def fit_fixture(tmp_path, monkeypatch, *, calibrate=True, bad_proxy=False):
    cfg = load_config(str(ROOT / "configs/v15_feasible_hinge_ar.yaml"))
    cfg["out_dir"] = str(tmp_path)
    cfg["codec"]["ste_rate_calibration"] = calibrate
    cfg["loss"]["kappa"] = cfg["loss"]["mu"] = 0.0
    cfg["train"].update(finetune=False, max_steps=1, qp_list=[30], qp_to_quality={30: 2},
                         clip_grad=0.0, lr=0.001)
    pre, analyzer = Edit(), Analyzer()
    ste = codec(Proxy(fixed=-1.0) if bad_proxy else Proxy(), calibrate=calibrate)
    tracker, postfix = Tracker(), []
    monkeypatch.setattr(engine, "make_tracker", lambda *_args, **_kwargs: tracker)
    monkeypatch.setattr(engine, "tqdm", lambda batches, **_kwargs: Progress(batches, postfix))
    batch = [(torch.full((1, 3, 2, 2, 2), 0.1), torch.tensor([0]))]
    engine._fit(cfg, pre, ste, analyzer, batch, None, lambda x: x, "fixture", 1)
    state = json.loads((tmp_path / "task_regret_constraint_state.json").read_text())
    return pre, analyzer, ste, tracker, postfix, state


def test_common_fit_logs_scale_and_retains_teacher_task_and_dual_behavior(real, tmp_path, monkeypatch):
    real[1]["bpp"] = 1.5
    plain = fit_fixture(tmp_path / "plain", monkeypatch, calibrate=False)
    scaled = fit_fixture(tmp_path / "scaled", monkeypatch, calibrate=True)
    assert plain[5] == scaled[5]
    for index in (1,):
        assert plain[index].pins == plain[index].unpins == 1
        assert scaled[index].pins == scaled[index].unpins == 1
        assert plain[index].calls == scaled[index].calls == [False, True]
        assert not plain[index].active and not scaled[index].active
    old_values = plain[3].steps[0]
    values = scaled[3].steps[0]
    for key in ("task_regret", "task_violation", "task_dual", "loss_task_objective",
                "rate_anchor_bpp", "loss_rate", "rate_excess_ratio"):
        assert values[key] == old_values[key]
    assert values["ste_rate_calibration_scale"] == pytest.approx(1.5 / 1.09)
    assert values["ste_rate_proxy_bpp"] == pytest.approx(1.09)
    assert values["ste_rate_real_bpp"] == 1.5
    assert values["ste_rate_floor_active"] == 0.0
    assert scaled[0].amount.grad.item() == pytest.approx(
        plain[0].amount.grad.item() * values["ste_rate_calibration_scale"]
    )
    assert set(scaled[4][0]) - set(plain[4][0]) == {"rscale", "rproxy", "rreal", "rfloor"}
    assert scaled[4][0]["rfloor"] == 0
    assert len(real[0]) == 4  # two fits, each existing treatment+raw anchor encode


def test_calibration_failure_releases_training_teacher(real, tmp_path, monkeypatch):
    cfg = load_config(str(ROOT / "configs/v15_feasible_hinge_ar.yaml"))
    cfg["out_dir"] = str(tmp_path)
    cfg["train"].update(finetune=False, max_steps=1, qp_list=[30], qp_to_quality={30: 2})
    analyzer = Analyzer()
    monkeypatch.setattr(engine, "make_tracker", lambda *_args, **_kwargs: Tracker())
    monkeypatch.setattr(engine, "tqdm", lambda batches, **_kwargs: Progress(batches, []))
    batch = [(torch.zeros(1, 3, 2, 2, 2), torch.tensor([0]))]
    with pytest.raises(ValueError, match="proxy_bpp must be finite and non-negative"):
        engine._fit(cfg, Edit(), codec(Proxy(fixed=-1.0)), analyzer,
                    batch, None, lambda x: x, "fixture", 1)
    assert analyzer.pins == analyzer.unpins == 1 and not analyzer.active
    assert real[0] == []

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src import engine
from src.models.additive_cond import AdditiveCondPreprocessor


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _live_model(sigma=0.0, *, temporal_frames=3, cond_dim=3):
    model = AdditiveCondPreprocessor(temporal_frames=temporal_frames, cond_dim=cond_dim,
                                     residual_filter_sigma=sigma)
    with torch.no_grad():
        model.to_rgb.weight.normal_(0, 0.003)
        model.to_rgb.bias.normal_(0, 0.003)
        model.film[2].weight.normal_(0, 0.04)
        model.film[2].bias.normal_(0, 0.04)
    return model


def _historical_forward(model, x, cond):
    """Frozen pre-option arithmetic, evaluated on live weights (not zero-init).

    This reference intentionally has no residual-filter call. It protects exact
    legacy operator order, the causal window and condition expansion.
    """
    b, c, t, h, w = x.shape
    frames = x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    sp = F.relu(model.spatial_stem(frames))
    body = model.spatial_residual.body
    sp = sp + body[2](F.relu(body[0](sp)))
    pad = x[:, :, :1].expand(b, c, model.tframes-1, h, w)
    padded = torch.cat([pad, x], dim=2)
    win = padded.unfold(2, model.tframes, 1)
    stack = win.permute(0, 2, 5, 1, 3, 4).reshape(b*t, model.tframes*c, h, w)
    tp = F.relu(model.temporal_stem(stack))
    gate = torch.sigmoid(model.fusion.gate[2](F.relu(model.fusion.gate[0](torch.cat([sp, tp], dim=1)))))
    fused = gate*sp + (1.0-gate)*tp
    if cond is None:
        cond = frames.new_zeros(b, model.cond_dim)
    cond_f = cond.repeat_interleave(t, dim=0).to(fused.dtype)
    gamma, beta = model.film(cond_f).chunk(2, dim=1)
    fused = fused*(1.0+gamma[:, :, None, None]) + beta[:, :, None, None]
    out = frames + model.strength*model.to_rgb(fused)
    return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


@pytest.mark.parametrize("conditioned", [False, True])
def test_default_live_weights_match_historical_output_gradient_and_rng(conditioned):
    torch.manual_seed(310)
    model = _live_model()
    x = (0.3+0.4*torch.rand(2, 3, 4, 11, 17)).requires_grad_()
    old_x = x.detach().clone().requires_grad_()
    cond = torch.rand(2, 3, requires_grad=True) if conditioned else None
    old_cond = cond.detach().clone().requires_grad_() if conditioned else None
    rng = torch.get_rng_state().clone()
    actual = model(x, cond)
    assert torch.equal(rng, torch.get_rng_state())
    expected = _historical_forward(model, old_x, old_cond)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, x), "live residual is essential to this parity check"
    params = list(model.parameters())
    actual_inputs = params + [x] + ([cond] if conditioned else [])
    old_inputs = params + [old_x] + ([old_cond] if conditioned else [])
    actual_grads = torch.autograd.grad(actual.square().mean(), actual_inputs)
    old_grads = torch.autograd.grad(expected.square().mean(), old_inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, old_grads, strict=True))


@pytest.mark.parametrize("sigma", [0.5, 1.0])
def test_enabled_filter_attenuates_only_learned_rgb_hf_preserving_source_and_time(sigma):
    model = AdditiveCondPreprocessor(temporal_frames=3, cond_dim=3, residual_filter_sigma=sigma)
    height, width = 12, 16
    checker = (torch.arange(height)[:, None]+torch.arange(width)[None, :]).remainder(2)*2-1
    checker = checker.float()
    source = (0.5+0.2*checker).expand(1, 3, 3, height, width).clone()
    residual = torch.zeros(3, 3, height, width)
    residual[0, 0] = 0.04*checker  # Red-only HF edit on first frame.
    residual[2] = torch.tensor([0.03, -0.02, 0.01])[:, None, None]
    handle = model.to_rgb.register_forward_hook(lambda _module, _args, _out: residual)
    try:
        edited = model(source, torch.zeros(1, 3))
    finally:
        handle.remove()
    change = edited-source
    assert torch.equal(edited[:, :, 1], source[:, :, 1]), "filter must not mix frames"
    assert torch.equal(edited[:, 1:, 0], source[:, 1:, 0]), "filter must not mix RGB channels"
    assert torch.allclose(change[0, :, 2], residual[2], atol=1e-7), "replicate borders preserve constant edits"
    assert change.abs().max() <= residual.abs().max()+1e-7, "source must not be blurred"
    assert edited[0, 0, 0].std() > 0.8*source[0, 0, 0].std(), "source HF details remain present"
    radius = math.ceil(3*sigma)
    interior = change[0, 0, 0, radius:-radius, radius:-radius]
    assert interior.square().mean().sqrt() < 0.5*0.04


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 1e-50])
@pytest.mark.parametrize("shape", [(1, 3, 1, 5), (2, 3, 5, 1), (6, 3, 9, 13)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replicate_filter_preserves_per_channel_constant_residual_at_borders(sigma, shape, dtype):
    model = AdditiveCondPreprocessor(residual_filter_sigma=sigma)
    values = torch.arange(shape[0]*shape[1], dtype=dtype).reshape(shape[0], shape[1], 1, 1)*0.01
    residual = values.expand(*shape).clone()
    filtered = model._filter_residual(residual)
    assert filtered.shape == residual.shape
    tolerance = 1e-13 if dtype == torch.float64 else 1e-6
    assert torch.allclose(filtered, residual, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("shape", [(1, 3, 1, 1, 5), (2, 3, 1, 13, 7), (1, 3, 3, 11, 19), (2, 3, 16, 16, 12)])
def test_rectangular_single_frame_and_temporal_clip_shapes(shape):
    model = _live_model(0.8, temporal_frames=8)
    clip = 0.3+0.4*torch.rand(*shape)
    result = model(clip, torch.rand(shape[0], 3))
    assert result.shape == clip.shape and torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1


def test_enabled_filter_gradients_reach_input_weights_and_condition():
    torch.manual_seed(311)
    model = _live_model(0.8)
    clip = (0.3+0.4*torch.rand(2, 3, 3, 12, 17)).requires_grad_()
    cond = torch.rand(2, 3, requires_grad=True)
    model(clip, cond).square().mean().backward()
    assert torch.isfinite(clip.grad).all() and clip.grad.abs().sum() > 0
    assert torch.isfinite(cond.grad).all() and cond.grad.abs().sum() > 0
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
    assert model.to_rgb.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("sigma", [0.5, 1.0])
def test_checkpoint_keys_parameter_count_and_strict_load_unchanged(sigma):
    torch.manual_seed(312)
    old = _live_model()
    old_state = copy.deepcopy(old.state_dict())
    filtered = AdditiveCondPreprocessor(temporal_frames=3, cond_dim=3, residual_filter_sigma=sigma)
    report = filtered.load_state_dict(old_state, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    assert filtered.state_dict().keys() == old_state.keys()
    assert all(torch.equal(filtered.state_dict()[key], value) for key, value in old_state.items())
    assert sum(p.numel() for p in filtered.parameters()) == sum(p.numel() for p in old.parameters())
    assert not list(filtered.buffers())  # The fixed kernel adds no checkpoint state.
    assert not old.load_state_dict(filtered.state_dict(), strict=True).missing_keys


@pytest.mark.parametrize("sigma", [-1.0, float('nan'), float('inf'), float('-inf'), True, False, '0.5', None, torch.tensor(0.5), 10**1000])
def test_invalid_or_boolean_sigma_rejects(sigma):
    with pytest.raises(ValueError, match="residual_filter_sigma"):
        AdditiveCondPreprocessor(residual_filter_sigma=sigma)


def _factory_cfg(task, sigma):
    return {"model": {"arch": "additive_cond", "temporal_frames": 3, "cond_dim": 3,
                      "residual_filter_sigma": sigma},
            "codec": {"kind": "virtual", "qualities": [1], "block": 4},
            "train": {}, "task": {"name": task}, "device": "cpu"}


@pytest.mark.parametrize("task", ["action_recognition", "object_detection"])
@pytest.mark.parametrize("role", ["train", "eval"])
@pytest.mark.parametrize("sigma", [0.0, 0.5])
def test_actual_common_factory_honors_option_for_od_ar_train_eval(monkeypatch, task, role, sigma):
    calls = []
    def analyzer(cfg, *, role):
        calls.append((cfg['task']['name'], role))
        return nn.Identity()
    monkeypatch.setattr(engine, "build_analyzer", analyzer)
    model, _, _ = engine._build_models(_factory_cfg(task, sigma), torch.device('cpu'), role=role)
    assert model.residual_filter_sigma == sigma and calls == [(task, role)]


def test_factory_does_not_coerce_boolean_sigma(monkeypatch):
    monkeypatch.setattr(engine, "build_analyzer", lambda *_args, **_kwargs: nn.Identity())
    with pytest.raises(ValueError, match="residual_filter_sigma"):
        engine._build_models(_factory_cfg('action_recognition', True), torch.device('cpu'))


@pytest.mark.parametrize("task", ["action_recognition", "object_detection"])
@pytest.mark.parametrize("recorded_sigma", [0.0, 0.8])
def test_checkpoint_loaded_evaluate_restores_recorded_filter_before_shared_factory(tmp_path, monkeypatch, task, recorded_sigma):
    torch.manual_seed(313)
    source = _live_model(recorded_sigma)
    saved_cfg = _factory_cfg(task, recorded_sigma)
    checkpoint = tmp_path/'source.pth'
    torch.save({'model': source.state_dict(), 'cfg': saved_cfg}, checkpoint)
    clip, cond = 0.3+0.4*torch.rand(1, 3, 3, 11, 17), torch.rand(1, 3)
    expected = source(clip, cond)
    seen = []
    def mocked_evaluation(cfg, pre, _codec, _analyzer, _out):
        seen.append((cfg['model']['residual_filter_sigma'], pre.residual_filter_sigma))
        assert torch.equal(pre(clip, cond), expected)
        return {'checked': True}
    monkeypatch.setattr(engine, 'build_analyzer', lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(engine, '_evaluate_classification', mocked_evaluation)
    monkeypatch.setattr(engine, '_evaluate_detection', mocked_evaluation)
    requested = _factory_cfg(task, 0.8 if recorded_sigma == 0 else 0.0)
    assert engine.evaluate(requested, str(checkpoint), str(tmp_path/'eval')) == {'checked': True}
    assert seen == [(recorded_sigma, recorded_sigma)]


def test_old_checkpoint_without_option_evaluates_with_exact_default(tmp_path, monkeypatch):
    source = _live_model()
    cfg = _factory_cfg('action_recognition', 0.0)
    del cfg['model']['residual_filter_sigma']
    checkpoint = tmp_path/'old.pth'
    torch.save({'model': source.state_dict(), 'cfg': cfg}, checkpoint)
    clip, cond = torch.rand(1, 3, 2, 9, 13), torch.rand(1, 3)
    expected = source(clip, cond)
    def mocked_evaluation(_cfg, pre, _codec, _analyzer, _out):
        assert pre.residual_filter_sigma == 0
        assert torch.equal(pre(clip, cond), expected)
        return {'checked': True}
    monkeypatch.setattr(engine, 'build_analyzer', lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(engine, '_evaluate_classification', mocked_evaluation)
    assert engine.evaluate(copy.deepcopy(cfg), str(checkpoint), str(tmp_path/'eval')) == {'checked': True}

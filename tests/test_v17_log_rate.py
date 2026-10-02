from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src import engine
from src.models import ste_codec as ste_module
from src.models.virtual_codec import VirtualCodec


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_log_coefficient_value_and_analytical_gradient():
    codec = VirtualCodec(rate_model="log_coeff")
    coeff = torch.tensor([0.0, 0.5, -1.0, 1.5], dtype=torch.float64).reshape(1, 1, 1, 4).requires_grad_()
    quantized, bits = codec._quant_rate(coeff, step=0.5, training=False)
    assert torch.equal(quantized, torch.tensor([0, 1, -2, 3], dtype=torch.float64).reshape_as(coeff))
    assert bits.item() == pytest.approx(3 + math.log2(3), rel=1e-14)
    bits.backward()
    expected = torch.tensor([0, 1/(math.log(2)), -1/(1.5*math.log(2)), 1/(2*math.log(2))], dtype=torch.float64)
    assert torch.allclose(coeff.grad.flatten(), expected, rtol=1e-14, atol=0)


def test_default_gaussian_value_and_gradient_match_original_power_contract():
    # Two positions of one frequency: scaled coefficients 1,3 have power5.
    codec = VirtualCodec()
    assert codec.rate_model == "gaussian_power"
    coeff = torch.tensor([0.5, 1.5], dtype=torch.float64).reshape(2, 1, 1, 1).requires_grad_()
    _, bits = codec._quant_rate(coeff, step=0.5, training=False)
    assert bits.item() == pytest.approx(math.log2(61), rel=1e-14)
    bits.backward()
    expected = torch.tensor([12, 36], dtype=torch.float64) / (0.5*61*math.log(2))
    assert torch.allclose(coeff.grad.flatten(), expected, rtol=1e-14, atol=0)


def test_log_rate_signal_scaling_differs_from_replica_scaling():
    codec = VirtualCodec(rate_model="log_coeff")
    coeff = torch.tensor([1.0, 3.0], dtype=torch.float64).reshape(2, 1, 1, 1)
    _, bits = codec._quant_rate(coeff, 1.0, False)
    _, repeated = codec._quant_rate(coeff.repeat(3, 1, 1, 1), 1.0, False)
    _, amplified = codec._quant_rate(2*coeff, 1.0, False)
    assert bits.item() == pytest.approx(3)
    assert repeated.item() == pytest.approx(9)
    assert amplified.item() == pytest.approx(math.log2(3) + math.log2(7))
    assert bits < amplified < 2*bits  # Signal amplitude is not a sample-count multiplier.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("training", [False, True])
def test_zero_coefficient_cost_and_subgradient_are_finite(dtype, training):
    codec = VirtualCodec(rate_model="log_coeff")
    coeff = torch.zeros(2, 4, 3, 2, dtype=dtype, requires_grad=True)
    _, bits = codec._quant_rate(coeff, 0.125, training)
    assert bits.item() == 0
    bits.backward()
    assert torch.equal(coeff.grad, torch.zeros_like(coeff))
    assert torch.isfinite(coeff.grad).all()


@pytest.mark.parametrize("colorspace,height,width", [("rgb", 4, 6), ("rgb", 5, 7), ("yuv420", 4, 6), ("yuv420", 5, 7)])
@pytest.mark.parametrize("chroma_scale", [1.0, 2.0])
def test_bpp_uses_full_resolution_pixels_and_actual_chroma_samples(colorspace, height, width, chroma_scale):
    # Block1 makes each coefficient exactly a plane sample. Achromatic 0.5
    # gives luma and both chroma planes0.5, including odd-size edge padding.
    codec = VirtualCodec(qualities=(1,), block=1, q_steps={1: 0.5}, inter=False,
                         colorspace=colorspace, chroma_step_scale=chroma_scale, rate_model="log_coeff").double()
    clip = torch.full((2, 3, 3, height, width), 0.5, dtype=torch.float64)
    reconstructed, bpp = codec.compress_decompress(clip, 1)
    chroma_samples = 2*math.ceil(height/2)*math.ceil(width/2)
    expected = 3 if colorspace == "rgb" else 1 + chroma_samples/(height*width)*math.log2(1 + 1/chroma_scale)
    assert reconstructed.shape == clip.shape
    assert bpp == pytest.approx(expected, rel=1e-14, abs=1e-14)


@pytest.mark.parametrize("colorspace", ["rgb", "yuv420"])
@pytest.mark.parametrize("replication", ["batch", "time", "spatial"])
def test_bpp_invariant_to_replicating_samples_with_intra_prediction(colorspace, replication):
    torch.manual_seed(101)
    clip = torch.rand(1, 3, 1, 16, 16, dtype=torch.float64)
    codec = VirtualCodec(qualities=(1,), block=4, q_steps={1: 0.2}, inter=False,
                         colorspace=colorspace, rate_model="log_coeff").double()
    factors = {"batch": (3, 1, 1, 1, 1), "time": (1, 1, 4, 1, 1), "spatial": (1, 1, 1, 2, 3)}
    _, rate = codec.compress_decompress(clip, 1)
    _, replicated_rate = codec.compress_decompress(clip.repeat(*factors[replication]), 1)
    assert replicated_rate == pytest.approx(rate, rel=1e-13)


@pytest.mark.parametrize("colorspace", ["rgb", "yuv420"])
def test_log_rate_full_closed_loop_input_gradient_is_finite(colorspace):
    torch.manual_seed(102)
    clip = torch.rand(1, 3, 3, 16, 16, requires_grad=True)
    codec = VirtualCodec(qualities=(1,), block=4, q_steps={1: 0.2}, inter=True,
                         colorspace=colorspace, rate_model="log_coeff")
    codec.set_anneal(1.0)
    reconstructed, bpp = codec(clip, 1)
    bpp.backward()
    assert reconstructed.shape == clip.shape and torch.isfinite(reconstructed).all()
    assert torch.isfinite(bpp) and torch.isfinite(clip.grad).all()
    assert clip.grad.abs().sum() > 0


@pytest.mark.parametrize("colorspace", ["rgb", "yuv420"])
@pytest.mark.parametrize("training,anneal", [(True, 0.0), (True, 0.6), (True, 1.0), (False, 0.0)])
@pytest.mark.parametrize("motion", [False, True])
def test_rate_modes_preserve_reconstruction_quantization_rng_and_closed_loop(colorspace, training, anneal, motion):
    torch.manual_seed(103)
    clip = torch.rand(2, 3, 3, 16, 16)
    common = dict(qualities=(1,), block=4, q_steps={1: 0.2}, inter=True,
                  colorspace=colorspace, motion=motion, motion_range=1, motion_block=4)
    gaussian, logarithmic = VirtualCodec(**common), VirtualCodec(**common, rate_model="log_coeff")
    gaussian.set_anneal(anneal)
    logarithmic.set_anneal(anneal)
    torch.manual_seed(104)
    old_reconstruction, old_bpp = gaussian._code(clip, 1, training)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(104)
    new_reconstruction, new_bpp = logarithmic._code(clip, 1, training)
    assert torch.equal(old_reconstruction, new_reconstruction)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert torch.isfinite(new_bpp) and not torch.equal(old_bpp, new_bpp)


@pytest.mark.parametrize("colorspace", ["rgb", "yuv420"])
@pytest.mark.parametrize("training,anneal", [(True, 0.0), (True, 0.6), (True, 1.0), (False, 0.0)])
def test_omitted_rate_model_matches_explicit_gaussian_numerically(colorspace, training, anneal):
    torch.manual_seed(105)
    clip = torch.rand(2, 3, 3, 16, 16)
    common = dict(qualities=(1,), block=4, q_steps={1: 0.2}, colorspace=colorspace)
    default = VirtualCodec(**common)
    explicit = VirtualCodec(**common, rate_model="gaussian_power")
    default.set_anneal(anneal)
    explicit.set_anneal(anneal)
    torch.manual_seed(106)
    reconstruction, bits = default._code(clip, 1, training)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(106)
    explicit_reconstruction, explicit_bits = explicit._code(clip, 1, training)
    assert torch.equal(reconstruction, explicit_reconstruction)
    assert torch.equal(bits, explicit_bits)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["unknown", "log", None, 0])
def test_unsupported_rate_model_rejects(mode):
    with pytest.raises(ValueError, match="rate_model must"):
        VirtualCodec(rate_model=mode)


@pytest.mark.parametrize("task", ["action_recognition", "object_detection", "tracking"])
@pytest.mark.parametrize("kind", ["virtual", "ste"])
@pytest.mark.parametrize("mode", [None, "log_coeff"])
def test_shared_factory_passes_rate_model_without_task_branch(monkeypatch, task, kind, mode):
    monkeypatch.setattr(engine, "build_analyzer", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(ste_module, "ffmpeg_available", lambda: True)
    codec_config = {"kind": kind, "qualities": [1], "block": 4, "ste_codec": "h264"}
    if mode is not None:
        codec_config["rate_model"] = mode
    cfg = {"model": {"arch": "additive_cond", "temporal_frames": 2, "cond_dim": 3},
           "codec": codec_config, "train": {"qp_to_quality": {30: 1}}, "task": {"name": task}}
    _, codec, _ = engine._build_models(cfg, torch.device("cpu"))
    proxy = codec.proxy if kind == "ste" else codec
    assert isinstance(proxy, VirtualCodec) and proxy.rate_model == (mode or "gaussian_power")
    if kind == "ste":
        assert codec.codec == "h264" and codec.quality_to_qp == {1: 30}


@pytest.mark.parametrize("kind", ["virtual", "ste"])
def test_shared_factory_rejects_unsupported_rate_model(monkeypatch, kind):
    monkeypatch.setattr(engine, "build_analyzer", lambda *_args, **_kwargs: nn.Identity())
    cfg = {"model": {"arch": "additive_cond"}, "codec": {"kind": kind, "rate_model": "unknown"},
           "train": {}, "task": {"name": "action_recognition"}}
    with pytest.raises(ValueError, match="rate_model must"):
        engine._build_models(cfg, torch.device("cpu"))

"""Real-codec-in-the-loop straight-through wrapper (upgrade3, contribution A3).

upgrade2 trained *entirely* through a differentiable proxy (CompressAI or the
block-DCT virtual codec) and hoped the edit transferred to x264/x265. Its own
report shows the transfer stalling near break-even -- the proxy's rate/quant
geometry simply is not the deployment codec's.

The most reliable fix in the literature is **run the real codec in the forward
pass and borrow the proxy's gradient in the backward pass** (straight-through):
Lu et al. (arXiv:2206.05650) measured forward-real-codec at BD-Rate -20.3% vs
-14.6% for a proxy used in *both* directions. This wrapper implements exactly
that so the loss the optimiser sees is computed on the TRUE x264/x265
reconstruction and TRUE coded bpp, while gradients still flow to the
preprocessor through the differentiable proxy:

    x_hat = x_proxy + (x_real  - x_proxy).detach()      # value=real, grad=proxy
    bpp   = bpp_proxy + (bpp_real - bpp_proxy).detach()

Because ffmpeg runs per step this is ~10-50x slower than proxy-only training and
is meant as a **short calibration fine-tune** on top of a proxy-pretrained
checkpoint (see configs/universal_action_recognition.yaml -> stage 2), not for
training from scratch. ``quality -> qp`` inversion lets it slot into the engine's
existing ``codec(x, quality)`` call unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

from ..codecs import StandardCodec, ffmpeg_available


class STECodec(nn.Module):
    """Straight-through: real x264/x265 value, differentiable proxy gradient.

    Args:
        proxy: a differentiable codec (``VirtualCodec`` / ``CompressAICodec``)
            exposing ``forward(x, quality) -> (x_hat, bpp)``.
        codec: ``"h264"`` or ``"h265"`` -- the real deployment codec.
        quality_to_qp: maps each proxy quality id to the real-codec QP to encode
            at (the inverse of the config's ``train.qp_to_quality``).
        preset: x264/x265 speed preset.
        calibrate_rate: optionally scale only the rate gradient by detached
            real/proxy bpp, using one batch-mean ratio and a fixed 1e-8 floor.
    """

    def __init__(
        self,
        proxy: nn.Module,
        codec: str = "h265",
        quality_to_qp: Dict[int, int] | None = None,
        preset: str = "medium",
        calibrate_rate: bool = False,
    ):
        super().__init__()
        if not isinstance(calibrate_rate, bool):
            raise ValueError("STECodec.calibrate_rate must be a bool")
        self.calibrate_rate = calibrate_rate
        self.rate_calibration_diagnostics: Dict[str, float] = {}
        if not ffmpeg_available():
            raise RuntimeError("STECodec needs ffmpeg (with libx264/libx265) on PATH")
        self.proxy = proxy
        self.codec = codec
        self.preset = preset
        self.quality_to_qp = {int(k): int(v) for k, v in (quality_to_qp or {}).items()}
        self.qualities: List[int] = list(getattr(proxy, "qualities", []))

    def _qp(self, quality: int) -> int:
        if quality in self.quality_to_qp:
            return self.quality_to_qp[quality]
        # linear fallback: higher quality id -> lower QP, clamped to the x26x range
        return int(max(18, min(51, 51 - 3 * int(quality))))

    def forward(self, x: torch.Tensor, quality: int):
        x_p, bpp_p = self.proxy(x, quality)                       # differentiable
        if self.calibrate_rate:
            self.rate_calibration_diagnostics = {}
            if not torch.is_tensor(bpp_p) or bpp_p.numel() != 1:
                raise ValueError("STE rate calibration requires scalar proxy_bpp")
            if not bpp_p.is_floating_point():
                raise ValueError("STE rate calibration requires floating-point proxy_bpp")
            proxy_value = float(bpp_p.detach())
            if not math.isfinite(proxy_value) or proxy_value < 0:
                raise ValueError(
                    "STE rate calibration proxy_bpp must be finite and non-negative"
                )
        sc = StandardCodec(codec=self.codec, qp=self._qp(quality), preset=self.preset)
        with torch.no_grad():
            x_r, bpp_r = sc.compress_decompress(x.detach())       # real ffmpeg
            x_r = x_r.to(x_p.device, x_p.dtype)
        # straight-through: forward value = real codec, backward grad = proxy
        x_hat = x_p + (x_r - x_p).detach()
        if self.calibrate_rate:
            real_value = float(bpp_r)
            if not math.isfinite(real_value) or real_value < 0:
                raise ValueError(
                    "STE rate calibration real_bpp must be finite and non-negative"
                )
            real_bpp = bpp_p.new_tensor(real_value)
            if not bool(torch.isfinite(real_bpp).all()):
                raise ValueError("STE rate calibration real_bpp is not finite in proxy dtype")
            # Eq40-inspired adaptation: one ratio for the batch's mean rate.
            # The scalar is detached. Differentiating real/proxy would cancel
            # the desired rate gradient. No fitted coefficient or cap is used.
            scale_value = real_value / max(proxy_value, 1e-8)
            if not math.isfinite(scale_value) or scale_value < 0:
                raise ValueError(
                    "STE rate calibration scale must be finite and non-negative"
                )
            scale = bpp_p.new_tensor(scale_value)
            if not bool(torch.isfinite(scale).all()):
                raise ValueError("STE rate calibration scale is not finite in proxy dtype")
            bpp = real_bpp + scale * (bpp_p - bpp_p.detach())
            self.rate_calibration_diagnostics = {
                "ste_rate_proxy_bpp": proxy_value,
                "ste_rate_real_bpp": real_value,
                "ste_rate_calibration_scale": float(scale),
                "ste_rate_floor_active": float(proxy_value < 1e-8),
            }
        else:
            # Keep the legacy expression exact when calibration is disabled.
            bpp = bpp_p + (x_p.new_tensor(float(bpp_r)) - bpp_p).detach()
        return x_hat, bpp

    @torch.no_grad()
    def compress_decompress(self, x: torch.Tensor, quality: int):
        """Eval path -> honest real-codec reconstruction + bpp (no proxy)."""
        sc = StandardCodec(codec=self.codec, qp=self._qp(quality), preset=self.preset)
        x_r, bpp_r = sc.compress_decompress(x)
        return x_r.to(x.device), float(bpp_r)


def _demo() -> None:
    from .virtual_codec import VirtualCodec

    if not ffmpeg_available():
        print("ste_codec self-check SKIPPED (ffmpeg not found)")
        return
    torch.manual_seed(0)
    proxy = VirtualCodec(qualities=(1, 3, 5), block=8)
    ste = STECodec(proxy, codec="h264", quality_to_qp={1: 42, 3: 32, 5: 24})
    x = torch.rand(1, 3, 4, 64, 64, requires_grad=True)
    x_hat, bpp = ste(x, 3)
    # forward VALUE must equal the real-codec reconstruction (STE identity)
    sc_val, _ = ste.compress_decompress(x.detach(), 3)
    assert torch.allclose(x_hat.detach(), sc_val, atol=1e-4)
    # but gradient must exist (borrowed from the proxy)
    (x_hat.mean() + bpp).backward()
    assert x.grad is not None and torch.isfinite(bpp)
    print(f"ste_codec self-check passed (real-value/proxy-grad, bpp={float(bpp.detach()):.3f})")


if __name__ == "__main__":
    _demo()

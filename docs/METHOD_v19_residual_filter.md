# V19 residual-only spatial filter: method and CPU readiness

Status: implementation readiness only. No GPU training, real-codec BD metric,
held-out task result, or threshold pass is reported by this change.

## Evidence and bounded hypothesis

The public MPEG document w24254, *Optimization of encoders and receiving systems
for machine analysis of coded video content (draft 6)*, is published as
*Updated text of ISO/IEC CDTR 23888-3 Optimization of encoders and receiving
systems for machine analysis of coded video content*. Its section 7.5
"Noise filtering" motivates removing
noise that can consume bits or impair machine inference while retaining
task-relevant information. It discusses configurable strength and content-aware
application; it does not mandate Gaussian filtering or filtering learned edits.

Primary source: [public MPEG 147 Sapporo archive w24254](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/147_Sapporo/w24254.zip).
The source reviewed was WG5_N0307_JVET-AI2030-v1.docx from that archive, converted
locally to text. Archive SHA256:
cf8b1e5c17030d32c10f8a7a65fb1c6610d22ca24bade92d5f9a0cae6fd97bbe;
DOCX SHA256:
b3426c94851662fc991c8c2f9c89859d24d5a1ae7d1ee33b2777d29133de1563.
This is a public draft source, not evidence of standards compliance.

Our adaptation is a fixed Gaussian spatial filter on the RGB edit produced by
the existing additive_cond model. The hypothesis is that attenuating rapid
spatial variations in learned edits may reduce codec cost while preserving
task guards. This source does not establish that the residual is noise, that
this particular filter helps AR, or that the real bitrate will decrease.

The preceding V18 rate-weight screens produced approximately -3.122846% BD-rate
with passing task guards for weight 0.5, and +2.448% with failing task guards for
weight 1. Neither met the -10% acceptance threshold. Those observations motivate
testing a different fixed mechanism; they do not establish its benefit.

## Shared model behavior

Let r be the learned RGB residual after the existing spatial/temporal fusion,
FiLM conditioning and to_rgb. The output is

    sigma = 0:     clamp(x + strength * r, 0, 1)
    sigma > 0:     clamp(x + strength * G_sigma(r), 0, 1)
    radius = ceil(3 * sigma)
    k[i] = exp(-0.5 * (i / sigma)^2) / sum_j exp(-0.5 * (j / sigma)^2)
    G_sigma(r) = vertical_k(horizontal_k(r))

model.residual_filter_sigma is a nonnegative finite real scalar, measured in
pixels. It defaults to zero; booleans, negative/nonfinite values and nonscalar
types are rejected. The zero path returns the original residual directly,
preserving the historical addition order and RNG behavior.

For positive sigma, each RGB channel of each output frame is filtered
independently with normalized separable grouped 2D convolutions and replicate
padding. The source pixels are added directly. The filter introduces no temporal
mixing, learnable parameters or persistent buffers. The existing temporal model,
conditioning, strength and final clamp are retained. The small fixed kernel is
built in float64, then converted to the residual dtype.

The common engine._build_models factory passes the scalar without coercing
booleans into numbers. Both OD and AR train/eval use this same path.
engine.evaluate restores a recorded residual_filter_sigma from checkpoint
model configuration before building the model. Legacy configurations with the
option omitted default to zero. The weights have the same keys and tensor shapes,
so either option can strict-load the original model weights.

GPU recipe registration and experiment execution belong to the coordinator.
This implementation does not change codec, source checkpoint, loss, task
evaluator, scorer, experiment registry, or confirmation gates. A dev screen
must meet BD-rate <= -10%, BD-accuracy >= 0 and every QP Top1 gap >= -0.5 pp
before an immutable checkpoint can proceed to the disjoint 802 and full 1010
evaluations. Historical validation exposure still limits the interpretation.

## Verification

On 2026-10-02, the related CPU suite passed **200 tests** in 17.82 seconds:
59 new residual-filter checks and 141 existing additive, calibration, log-rate,
task-regret, feasible-hinge and CRC regressions. The sole warning was the existing
invalid escape sequence in src/models/cast_ar.py.

The new checks cover live nonzero-weight historical output/gradient/RNG parity,
spatial high-frequency attenuation of the residual while retaining source
details, independent frames/RGB channels, constant edits including borders,
float32/float64 and very small sigma, rectangular/T=1/temporal clips, finite
backpropagation, invalid scalar rejection, unchanged keys/parameter counts and
strict loading in both directions. Factory checks exercise OD/AR train/eval;
checkpoint-loaded engine.evaluate checks restore recorded values including
zero. These are synthetic CPU checks. An actual pretrained-source checkpoint
probe is a separate coordinator integration check.

Exact test invocation from this worktree (Python 3.12.14, PyTorch 2.14.1+cpu):

    & 'C:\Users\Admin\AppData\Local\Temp\vcm-v13-tools\Scripts\python.exe' -B -c "import torch,pytest; torch.set_num_threads(2); raise SystemExit(pytest.main(['tests/test_v19_residual_filter.py','tests/test_additive_cond.py','tests/test_v17_rate_calibration.py','tests/test_v17_log_rate.py','tests/test_v15_feasible_hinge.py','tests/test_v13_task_regret.py','tests/test_crc_v5.py','-q','-p','no:cacheprovider','--basetemp','C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/pytest-v19-residual-filter-20261002-r2']))"

## Bounded local CPU forward timing

One sequential run used float32 [2,3,16,128,128], two CPU threads and one interop
thread, temporal_frames=8, cond_dim=3, strength 1, eval and
torch.inference_mode(). Each sigma used five warmup calls followed by ten calls
timed with time.perf_counter. The same clip, condition and model state were
used across all arms. Synthetic live weights were initialized with seed 19019;
to_rgb weight/bias standard deviation was 0.003 and FiLM output weight/bias
standard deviation was 0.04. This is not timing on a trained checkpoint.

Host processor identification was
Intel64 Family 6 Model 140 Stepping 1, GenuineIntel.
All arms strict-loaded the same 18 state tensors, had 10,403 parameters and zero
buffers. The option adds no learned model capacity.

| Sigma (pixels) | Radius | Median ms/call | Mean ms/call | Sample SD ms | Min-max ms | Median ratio to sigma 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 631.58 | 720.33 | 178.74 | 572.60-1018.79 | 1.000 |
| 0.5 | 2 | 752.33 | 751.28 | 50.95 | 665.25-811.45 | 1.191 |
| 1 | 3 | 648.54 | 652.46 | 59.69 | 566.38-769.56 | 1.027 |

These ten local calls are timing repetitions, not independent scientific runs.
The spread and sequential arm order preclude a reliable relative-runtime ranking.
This measures local preprocessing forward only: no codec encoding/decoding,
training/backward, task model, GPU timing or end-to-end CTC runtime was measured.

The nominal added convolution work per frame pixel is
2 * RGB_channels * kernel_length: 30 MACs for sigma 0.5 (kernel length 5), or
42 MACs for sigma 1 (kernel length 7). This excludes padding, kernel construction
and all existing network operations; it is not total network or CTC complexity.

# V17 n13: coefficient-wise logarithmic rate proxy screen

Design registered on 2026-10-02 before any V17 GPU result. One fixed
hypothesis is tested: replacing only the Gaussian-power rate estimate by
a sum of per-coefficient log costs improves the true H.264 rate/task
tradeoff. No outcome-dependent tuning is allowed inside this node.

## Literature source and adaptation

Primary source: Guleryuz et al., [Sandwiched Compression: Repurposing
Standard Codecs with Neural Network Wrappers](D:/STUDY/LAB/bao_1/papers/Sandwiched_Compression_Neural_Network_Wrappers_Repurposing_Standard_Codecs.pdf),
author-hosted 2024 version,20-page local PDF, physical p17, supplementary
Section B, Eq.39--40. These equation numbers refer specifically to that
inspected version; the associated [arXiv preprint2402.05887](https://arxiv.org/abs/2402.05887)
has different equation numbering in v1. The rendered local page was
inspected. Source SHA-256:
`a7f57a9f319ebe77b30eb1baa7b7e2e91f7acb287b55bfdb72e41ee2a874e4e1`.

Eq.39 approximates a count of nonzero transform coefficients with a
sum of log(1+abs(coefficient)/step), multiplied by a scale a and offset b.
Eq.40 sets b=0 and chooses a per-image scale to match the actual standard
JPEG codelength. This experiment adapts the coefficient cost only to the
existing video's intra/prediction-residual DCT blocks. It does not copy
the paper's pre/post architecture, JPEG representation or perceptual
results, and no paper gain is asserted to transfer to AR.

Our uncalibrated proxy is explicitly:

```text
y_i = DCT(residual)_i / step
bits_proxy = sum_i log2(1 + abs(y_i))
bpp_proxy = bits_proxy / (B*T*H*W)
```

Using base2 instead of the paper's unspecified log base fixes an explicit
bit-cost scale. Relative to a natural log this is multiplication by
1/ln(2). A subsequent detached Eq.40-style multiplicative calibration
would absorb that constant: a is divided by the same change in scale,
so both calibrated value and local gradient scale agree. Calibration is
OFF by default for this screen, so scale is part of this registered
intervention. The coordinator may integrate separate optional calibration
code into the common runtime, but must not enable it for this arm.

The existing Gaussian-power estimate is:

```text
power_f = mean_{batch,block_position}(y_f^2)
bits_proxy = sum_f 0.5*log2(1 + 12*power_f) * positions_per_frequency
```

The new cost has a coefficient-local magnitude derivative, with PyTorch's
zero subgradient at zero. It can change gradient shape and scale; that is
not proof that either estimate predicts entropy-coded video accurately.
It omits syntax, motion-vector and entropy-context costs. A true-codec
screen is required to decide whether the adaptation transfers.

The audited n9 H.264 parent produced BD-rate +2.6945%, BD-accuracy
-1.0680 percentage points and same-QP bit inflation +3.18--5.89%.
That failure motivates an explicit proxy-rate ablation, without proving
that the Gaussian-power model caused the measured outcome.

## Minimal implementation and controls

- Parent n9 code/artifact: `60168b918bc0eab76ebe54ad4589ab6dd18e8e43`.
- `VirtualCodec(rate_model="gaussian_power")` is the exact legacy default;
  `rate_model="log_coeff"` changes only estimated bits in `_quant_rate`.
  Unsupported modes fail on construction. Quantisation, annealing,
  noise RNG, reconstruction, closed-loop references, motion decisions,
  colour conversion and chroma subsampling are unchanged.
- Count each luma/chroma coefficient at its own quant step. Chroma still
  uses `step*chroma_step_scale`. Both RGB and yuv420 keep the original
  full-resolution normalization B*T*H*W, including the actual chroma
  sample counts and legacy transform padding geometry.
- Wire the option through the single shared virtual/STE factory path in
  `engine._build_models`. No task-specific branch or task API is added.
  This executor does not change STECodec or introduce calibration.
- `configs/v17_log_rate_ar.yaml` copies the n9 config with exactly one
  parsed addition: `codec.rate_model: log_coeff`. All other parsed leaves
  stay identical, including output path `outputs/v14_dual_safety_ar`;
  the central wrapper assigns a unique actual job directory.
- Retain additive_cond, cond_dim3, strength1, sampled frozen r3d_18/mc3_18
  teachers, same teacher pinned for raw/treatment, real H.264 STE medium,
  yuv420, all quantizer/proxy settings, and the existing common OD/AR
  pipeline. Forward STE rate and reconstruction remain true-codec values;
  this change affects the borrowed rate derivative.
- Retain positive_part primal, signed per-codec/QP dual update, epsilon0,
  task mu_init10/cap100/LR0.02, rate_weight0.25 globally and in both codec
  overrides, DCT kappa10, pixel MSE mu10, and omega0. No V16 penalty
  removal or rate-weight change is combined with this hypothesis.

## Immutable source, budget and evaluator

Audited H.264 source checkpoint at step500, SHA-256:
`ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`.
Load only model weights, resetting optimizer/scheduler and dual state.
Seed302001; one epoch capped at exactly 1,000 fresh steps; batch2;
LR1e-5; balanced QPs30/35/40/45/50. Evaluate only the final-last
checkpoint at global_step1000; diagnostic validation cannot select it.
The coordinator restricts training validation to the fixed dev208 index.

Development clips remain the exact existing208 IDs, fingerprint
`f321b810af2c507c`, shard0/5, salt `v13-task-regret-screen-v1`.
Evaluate frozen held-out `r2plus1d_18`, real H.264 medium at
QP30/35/40/45/50 with same-codec paired raw anchors. Primary BD-rate uses
aggregate Top-1 rate-quality curves. Non-finite metrics fail.

All development gates must pass:

1. BD-rate <=-10%;
2. BD-accuracy >=0; and
3. every QP treatment-minus-anchor Top-1 gap >=-0.5 percentage point.

Only then may the coordinator evaluate the same immutable checkpoint in
a fresh process on the disjoint802 confirmation complement and full1010
validation set using unchanged gates. No held-out task output is inspected
by this executor. The complement is excluded from current search but has
known historical validation use; it is not a new untouched population.
No development result alone can promote a best artifact.

## Local readiness

The focused CPU suite passed104 tests:65 new coefficient-rate/factory
checks,28 existing hinge/task-regret/CRC regressions and11 existing
motion/learned-entropy checks. It verifies analytical value and coefficient
derivative; zero cost/subgradient; finite full input gradients; signal
amplification versus sample replication; full-resolution bpp normalization;
RGB and luma/chroma counting, including odd geometry and chroma step scale;
unchanged quantisation, closed-loop reconstruction and RNG for noise,
annealed STE and evaluation; unsupported-mode failure; and option plumbing
through both virtual and STE factories for AR, OD and tracking. No actual
codec, teacher weights, dataset download or network access is required by
these CPU fixtures.

An independent ad hoc comparison loaded the original VirtualCodec source
from the immutable n9 commit using `git show`, rather than reconstructing
its formula from the modified file. Across32 combinations of RGB/yuv420,
intra/inter, motion off/on and noise/annealed/hard/evaluation quantisation,
the omitted/default rate option was exactly `torch.equal` to n9 for
reconstruction, estimated bpp, combined reconstruction/rate input gradient
and post-forward RNG state. This verifies the promised numerical legacy
path. The parsed-YAML comparison also confirms the only candidate addition
is `codec.rate_model=log_coeff`; all existing leaves are unchanged.

Validation command (two CPU threads avoid tiny-fixture thread overhead):

```powershell
& 'C:\Users\Admin\AppData\Local\Temp\vcm-v13-tools\Scripts\python.exe' -B -c "import torch,pytest; torch.set_num_threads(2); raise SystemExit(pytest.main(['tests/test_v17_log_rate.py','tests/test_v15_feasible_hinge.py','tests/test_v13_task_regret.py','tests/test_crc_v5.py','tests/test_virtual_codec_motion.py','tests/test_entropy_codec.py','-q','-p','no:cacheprovider','--basetemp','C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/pytest-v17-log-rate-20261002-r1']))"
```

The existing cast_ar.py invalid-escape warning is unrelated. These checks
establish local implementation readiness, not codec-proxy transfer, AR
accuracy or measured BD-rate gains. No GPU job is launched or uploaded by
this executor, dev_score is null, and recurring automations remain off.

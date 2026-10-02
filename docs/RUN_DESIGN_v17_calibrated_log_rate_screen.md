# V17 n14: detached real/proxy rate-gradient calibration

Registered on 2026-10-02 before this candidate has GPU results. Fixed
hypothesis: scaling the same log-coefficient proxy rate derivative by a
detached real/proxy bpp ratio improves its scale and compression/task
tradeoff relative to the identical unscaled log proxy. There is no
outcome-based tuning within this node. The original V13 and closed V14/V15
screens retain their registered interpretations.

## Source and adaptation

The supplied [Sandwiched Compression: Neural Network Wrappers for
Repurposing Standard Codecs](https://arxiv.org/abs/2402.05887), physical
PDF page17, supplemental printed page3, Eqs39--40, defines a smooth
log-coefficient rate proxy and calibrates its coefficient to the actual
JPEG rate on each bottleneck image. The local PDF and rendered page were
inspected under D:/STUDY/LAB/bao_1/papers and
D:/STUDY/LAB/bao_1/output/research_v17_20261002/figures/sandwich_p17.png.

This experiment adapts that idea to a batch-mean video bpp and H.264/H.265
STE. It is not a literal per-image JPEG implementation or a claim that the
paper established machine-task improvements for this pipeline. Our ratio
is explicitly detached: differentiating the ratio would cancel the proxy
rate derivative and defeat the intended calibrated rate pressure.

## Fixed treatment and equations

Both n13 (arm A) and n14 (arm B) use identical merged common code and the
same log-coefficient rate proxy. The independent executor supplies
VirtualCodec rate_model=log_coeff and its factory plumbing; this executor
does not modify VirtualCodec. Arm B alone enables STE rate calibration.

For batch-mean proxy rate P and measured real rate R:

```text
a = stopgrad(R / max(stopgrad(P), 1e-8))
bpp_forward = R + a * (P - stopgrad(P))
d(bpp_forward)/d(input) = a * d(P)/d(input)
```

The forward rate is exactly the real codec rate in the returned dtype.
Reconstruction STE keeps its exact existing value and gradient:

```text
x_hat = x_proxy + stopgrad(x_real - x_proxy)
```

There is one real encode in each forward, with no extra real measurement
for calibration. The raw anchor encode required by the task constraint is
unchanged. Disabled calibration retains the exact legacy rate arithmetic,
RNG use and real-call count. STECodec calibrate_rate and shared factory
codec.ste_rate_calibration accept only booleans; opt-in with non-STE fails
before model construction. Non-floating, non-scalar, negative or non-finite proxy rate,
negative/non-finite real rate, non-finite calibration ratio and non-finite
representation in the proxy dtype fail clearly. No scale cap or fitted
coefficient is introduced.

The preregistered denominator floor is exactly 1e-8. A finite but tiny
proxy rate can produce a large scale; this remains visible rather than
being clipped by a new cap. Existing train.clip_grad=1 remains unchanged.
The batch-mean ratio does not establish correct per-clip allocation of
rate gradients, particularly when clips differ substantially in rate.

Opt-in detached diagnostics are ste_rate_proxy_bpp, ste_rate_real_bpp,
ste_rate_calibration_scale (actual represented multiplier), and
ste_rate_floor_active (1 when P<1e-8). The common tracker receives all
four metrics and progress output reports rproxy/rreal/rscale/rfloor, so
the GPU log exposes actual strength even without optional online tracking.
No task-specific branch, alternate training loop, loss or gate is added.

## Controlled configuration and budget

configs/v17_calibrated_log_rate_ar.yaml is copied from n9
configs/v15_feasible_hinge_ar.yaml, base commit
60168b918bc0eab76ebe54ad4589ab6dd18e8e43, with only two added parsed leaves:

- codec.rate_model: log_coeff;
- codec.ste_rate_calibration: true.

All other parsed fields, including inherited out_dir, remain unchanged.
The coordinator will combine both implementations before packaging either
arm and supply a unique job output path. This isolated branch does not yet
implement the independent log mode; no integrated log-mode result is
claimed before the coordinator's combined-code tests.

One additive_cond preprocessor, cond_dim=3, frozen sampled teachers
r3d_18/mc3_18, positive-part task primal, epsilon=0, dual mu_init=10,
mu_max=100, dual_lr=0.02, rate_weight=0.25, source pixel MSE mu=10,
DCT kappa=10, omega=0 and all codec/reconstruction settings remain fixed.
Use the same audited H.264 source at step500, SHA-256
ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a.
Reset optimizer/scheduler/duals and fine-tune only model weights for
exactly 1000 fresh steps, seed302001, LR1e-5, batch2, one epoch and balanced
QP30/35/40/45/50. Evaluate final-last global_step1000 rather than the
proxy-validation-selected checkpoint. No architecture or seed changes.

## Locked screen and acceptance

Development208 clips only, fingerprint f321b810af2c507c, original salt
v13-task-regret-screen-v1, shard0/5; training diagnostic validation uses
the coordinator's development index. Held-out r2plus1d_18, real H.264
medium, same-codec raw anchors and aggregate Top1 rate-quality curves at
QP30/35/40/45/50 remain fixed.

Advance only if BD-rate <= -10%, BD-accuracy >= 0, and every QP Top1 gap
>= -0.5 percentage point; non-finite metrics fail. No reserved802 or
full1010 task outputs tune this node. A passing development checkpoint
must still pass the unchanged coordinator-owned reserved802 and full1010
evaluations. The validation set has historical V12 use; the reserved
complement is excluded from this current search, not newly collected data.

## Status

Implementation is complete. The focused CPU suite passed 67 tests: 39 new
calibration cases and all 28 parent hinge/regret/CRC regressions. Mocked real
codecs verify exact disabled legacy arithmetic, RNG and call count; exact
calibrated real forward rate; the detached analytic multiplier for batch
sizes 1/2/3; unchanged reconstruction value/gradient; fixed floor activation
and uncapped derivatives; invalid type/shape/numeric failures; and strict
factory opt-in. The common training integration exposes all four diagnostic
metrics in tracker and progress output while CE, signed task duals and pinned
teacher behavior remain unchanged. A calibration failure releases the teacher.
Git whitespace checks pass. The existing cast_ar.py invalid-escape warning
is unrelated.

```powershell
python -B -m pytest tests/test_v17_rate_calibration.py tests/test_v15_feasible_hinge.py tests/test_v13_task_regret.py tests/test_crc_v5.py -q -p no:cacheprovider --basetemp C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/pytest-v17-calibration-20261002-r2
```

This executor did not edit VirtualCodec, train a new entropy prior, submit
a GPU job, alter the evaluator or inspect reserved task outputs. The existing
D8 entropy codec remains a separate trained Laplacian approach. Calibration
unit fixtures use a finite differentiable proxy; the coordinator must combine
and verify the independent log-coefficient implementation before either A/B
job. dev_score remains null. Batch-mean calibration, the 1e-8 denominator floor,
and potentially large finite scales are explicit limitations; these checks
establish gradient wiring, not better codec transfer, AR accuracy or BD-rate.

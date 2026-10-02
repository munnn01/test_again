# V18 n15: calibrated log-rate weight0.5 screen

Design registered on 2026-10-02 before this candidate has GPU results.
Arbor node n15 tests one fixed hypothesis: increasing only the fixed rate
weight from0.25 to0.5 on the V17 calibrated log-rate/positive-part recipe
improves H.264 bitrate efficiency while the unchanged initial10 task duals
and fast signed updates preserve task guards. No outcome-based tuning or
second training pipeline is introduced.

## Evidence and bounded hypothesis

The [audited V17 results](D:/STUDY/LAB/111/docs/RESULTS_autonomous_ar_v17_2026-10-02.md)
show that historical arm B, calibrated log rate with weight0.25, had
BD-rate -1.77382249%, BD-accuracy +0.77028 percentage point, worst-QP
Top-1 gap +0.48077 percentage point, and same-QP bit inflation7.27--13.17%.
It passes the task gates but fails the unchanged -10% rate target. It is
a fixed-source, seed302001,1000-step historical control; it is not a
new concurrent multi-seed replication.

Recorded calibration display scales were0.09542--0.6099, all below1,
with no denominator-floor activation. This attenuates the recorded local
proxy-rate multiplier and is consistent with remaining bit expenditure,
but does not establish a gradient-norm measurement or a universally
correct gradient direction. The2000 progress entries contain repeated
displays, not2000 independent steps or samples.

V15 higher-rate experiments used the signed Gaussian-power recipe and
failed its acceptance gates. Their outcomes are not evidence of monotonic
transfer to the present log-coefficient, detached-calibration and
positive-part recipe. Higher weight can sacrifice task accuracy, alter
the dual trajectory or still fail to reduce real codec bits. The screen
will falsify or bound this specific coefficient change without assigning
prior failure to an unmeasured cause.

## Fixed intervention and common objective

Copy `configs/v17_calibrated_log_rate_ar.yaml` to
`configs/v18_calibrated_rate05_ar.yaml`. Exactly three parsed leaves
change from0.25 to0.5, representing one conceptual intervention:

```text
loss.task_regret_constraint.rate_weight
loss.task_regret_constraint.per_codec.h264.rate_weight
loss.task_regret_constraint.per_codec.h265.rate_weight
```

Both codec overrides must change because they take precedence over the
global value. The common engine remains unchanged. The nominal initial
task/rate coefficient ratio falls from40 to20; this is not a measured
gradient ratio.

```text
P = sum log2(1 + abs(DCT(residual))/step) / (B*T*H*W)
a = stopgrad(R_real / max(stopgrad(P), 1e-8))
R_STE = R_real + a*(P-stopgrad(P))
G[c,q] = mean CE(pre+same_codec) - mean CE(raw+same_codec)
mu[c,q] <- clamp(mu[c,q] + 0.02*(G[c,q]-epsilon[c,q]), 0, 100)
L = 0.5*(R_STE/R_raw - 1) + mu[c,q]*relu(G[c,q]-epsilon[c,q])
    + unchanged pixel MSE and DCT penalties
```

Quantization, reconstruction, RNG, closed-loop prediction, rate calibration,
codec calls, teacher selection and scoring code are unchanged. There is
no change to STE implementation or addition of a trained entropy prior.
The same shared additive_cond preprocessor and common OD/AR pipeline
remain in use.

## Immutable source, controls and budget

- Common parent code commit: `0e35c0d2be0fdf377ff6e7cb92582833f4542e58`.
  The coordinator combines both V18 configs in the same candidate commit
  and payload; this executor adds only this config and registered design.
- Audited H.264 source checkpoint at step500, SHA-256:
  `ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`.
- Load only model weights; reset optimizer, scheduler and dual state.
  Seed302001, exactly1000 fresh steps, one epoch, batch2, LR1e-5,
  balanced QPs30/35/40/45/50 and200 visits per H.264/QP cell. Evaluate
  final-LAST global_step1000, never a proxy-validation-selected checkpoint.
- Retain additive_cond, cond_dim3, strength1, sampled frozen r3d_18/mc3_18,
  pinned same teacher for raw/treatment, real H.264 medium STE, yuv420,
  log_coeff, detached calibration=true, floor1e-8 and every proxy setting.
- Retain positive_part primal, signed dual update, epsilon0, task
  mu_init10, mu_max100, dual_lr0.02; retain pixel MSE loss.mu10,
  DCT kappa10, omega0 and every other parent loss coefficient. The old
  rate constraint stays disabled.
- Every other parsed leaf is exactly identical, including out_dir
  `outputs/v14_dual_safety_ar`; the central wrapper assigns a unique
  actual output directory. Diagnostic validation uses only dev208 IDs.

## Fixed evaluator and acceptance

Use the exact development208 clips, fingerprint `f321b810af2c507c`,
shard0/5 and salt `v13-task-regret-screen-v1`. Evaluate frozen held-out
`r2plus1d_18`, real H.264 medium at QPs30/35/40/45/50, with same-codec/QP
paired raw anchors. Primary BD-rate is based on the aggregate Top-1
rate-quality curve; no new evaluator, interpolation or gate is introduced.
Non-finite metrics fail.

All development gates must pass:

1. BD-rate <=-10%;
2. BD-accuracy >=0; and
3. every QP treatment-minus-anchor Top-1 gap >=-0.5 percentage point.

Only a passing immutable development checkpoint may advance to fresh,
coordinator-owned evaluation on the disjoint802 confirmation complement
and full1010 validation set with unchanged gates. No confirmation output
is inspected or used to tune this node. The802 complement was excluded
from current search but has known historical validation exposure; it is
not a newly collected untouched population. A development score alone
cannot establish success or promote a best artifact.

## Local readiness

YAML syntax and exact parsed comparison passed: only the three registered
rate_weight leaves differ, each from0.25 to0.5. Every other parsed leaf,
including the inherited output directory, matches the V17 calibrated config.

Actual `src.engine._task_regret_constraint_settings` and
`_codec_task_regret_constraint` resolve `rate_weight=0.5` for both H.264
and H.265, retaining epsilon0, dual_lr0.02, initial task mu10, cap100,
positive_part and ten independent codec/QP cell keys. Log rate and detached
calibration remain enabled; pixel MSE10, DCT10, omega0, disabled old rate
constraint, seed302001 and1000-step budget were also verified unchanged.
The smoke ran with the existing CPU Python and `-B`, without data, weights,
models, codec execution or network access. The existing cast_ar.py
invalid-escape import warning is unrelated.

No source/test files changed, no config-mirroring test was added, and the
unchanged full suite was not repeated. Git staged whitespace checks pass.
These checks establish recipe readiness only. No GPU job, network transfer
or GitHub push is performed by this executor. dev_score is null.

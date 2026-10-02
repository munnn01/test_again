# V18 n16: stronger rate pressure on the calibrated log proxy

Registered on 2026-10-02 before this candidate has GPU results. One fixed
hypothesis: increase the rate coefficient from 0.25 to 1 on the V17
calibrated log-coefficient proxy and positive-part task regret, retaining
strong initial codec/QP task duals. This may reduce bit overhead enough
to meet the requested quality-matched rate gate while preserving task
safety. This is a registered screen, not a guarantee of improvement.
Original V13 and closed screens keep their recorded interpretations.

## Evidence and bounded motivation

The coordinator's audited
[2026-10-02 V17 results](RESULTS_autonomous_ar_v17_2026-10-02.md)
show that the calibrated B arm has BD-rate -1.77382249%, BD-accuracy
+0.77028 percentage point and worst QP Top1 gap +0.48077 point. It passes
task gates but fails the -10% rate gate. Every same-QP bitrate is higher
than the raw anchor: overhead ranges from 7.2671% to 13.1692%.

Its recorded calibration scales range from 0.09542 to 0.6099, with no
floor activation. The 2000 rounded progress records contain duplicate
displays; they are not 2000 independent training steps. The observed
scales attenuate the proxy rate derivative. Raising the rate coefficient
fourfold is therefore a plausible way to reduce bit spending within the
same proxy/dual formulation. At identical inputs, its rate derivative
coefficient is fourfold higher; this does not predict later learned
inputs, calibration ratios, clipped gradients or held-out task quality.
Earlier higher-rate signed Gaussian attempts failed, so they do not
establish that this adaptation will succeed.

## Fixed treatment and common code

Configuration configs/v18_calibrated_rate1_ar.yaml is copied from
configs/v17_calibrated_log_rate_ar.yaml, parent common-code commit
0e35c0d2be0fdf377ff6e7cb92582833f4542e58. Only three parsed leaves change:

- loss.task_regret_constraint.rate_weight: 0.25 -> 1;
- loss.task_regret_constraint.per_codec.h264.rate_weight: 0.25 -> 1;
- loss.task_regret_constraint.per_codec.h265.rate_weight: 0.25 -> 1.

The explicit overrides must change with the global value because the
shared _codec_task_regret_constraint returns the resolved per-codec
settings. Changing only the global leaf would leave the active codec at
0.25. No engine, codec, architecture, scorer, task gate or loss formula
is edited by this candidate. The coordinator will integrate the two V18
configs on the same immutable code commit and payload.

The task objective and update remain:

```text
L = 1 * (R_pre / R_raw - 1)
    + mu_task[c,q] * relu(meanCE_pre - meanCE_raw - epsilon[c,q])
    + 10 * existing_DCT + 10 * source_pixel_MSE
mu_task[c,q] <- clamp(mu_task[c,q] + 0.02 * signed_violation[c,q], 0, 100)
```

The log-coefficient rate model and detached real/proxy calibration
(including fixed denominator floor 1e-8) remain enabled and unchanged.
Reconstruction forward and gradient stay at their registered STE forms.
Initial task dual 10, cap 100 and fast dual LR 0.02 remain fixed; they are
surrogate CE controls rather than proof that every held-out Top1 guard
will hold. Batch-mean regret still permits per-example cancellation.

## Source, training and checkpoint controls

- One additive_cond preprocessor, cond_dim=3; no separate OD/AR pipeline.
- Frozen training teachers r3d_18 and mc3_18, sampled and pinned across
  raw/treatment measurements in each step; held-out r2plus1d_18 at eval.
- Same audited H.264 source checkpoint at step500, SHA-256
  ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a.
- Model-only load; reset optimizer/scheduler/duals; seed302001, one epoch
  capped at exactly 1000 fresh steps, LR1e-5, batch2, balanced QPs
  30/35/40/45/50. Evaluate final-last global_step1000 only.
- Real H.264 medium STE, log_coeff rate model, detached calibration;
  source pixel MSE mu=10, DCT kappa=10, omega=0, positive-part task primal,
  epsilon=0, task mu_init=10/mu_max=100/dual_lr=0.02 all remain fixed.
- All other parsed fields, including inherited out_dir, stay identical.
  The coordinator supplies a unique submitted job output directory.
- H.265 uses the same declared coefficient to avoid hidden stale
  overrides; this registration does not claim H.265 transfer or schedule
  a separate H.265 job.

## Locked development screen and acceptance

Use only the existing development208 population, fingerprint
f321b810af2c507c, salt v13-task-regret-screen-v1, shard0/5. Restrict training
diagnostic validation to the coordinator's development index. Keep
same-codec raw anchors, real H.264 medium, frozen held-out r2plus1d_18 and
aggregate Top1 rate-quality curves at QP30/35/40/45/50. No proxy monitor
selects the evaluated checkpoint and no denominator, QP or scorer changes.

Advance only when finite BD-rate <= -10%, BD-accuracy >= 0, and every
same-QP Top1 gap >= -0.5 percentage point. Only a passing immutable final
checkpoint may be independently confirmed on the reserved802 complement
and then full1010 with the identical scorer/gates. Neither task output
is available as an optimization oracle here. The validation set has
historical V12 exposure; the complement is excluded from the current
search rather than a new untouched data population.

## Readiness and limits

Local YAML syntax and parsed-delta checks passed: exactly the three
registered rate_weight leaves change from 0.25 to 1; every other parsed
leaf matches the parent, including source/model/out_dir. The existing
_task_regret_constraint_settings and _codec_task_regret_constraint resolve
rate_weight=1 for both H.264 and H.265. Git whitespace checks pass. No
source or test changes, GPU execution, network transfer, new data,
checkpoint upload or GitHub push are made by this executor. The unchanged
143-test common-code suite is not repeated for this reversible recipe
edit; coordinator preflight owns the combined candidate checks.
dev_score remains null until the coordinator executes and audits the
registered development screen.

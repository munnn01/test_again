# V15 n8: fixed rate-pressure screen

Design registered on 2026-10-02 before this candidate has GPU results.
Arbor node `n8` tests one fixed hypothesis; no local hyperparameter pivot
is permitted. The coordinator owns GPU execution and independent auditing.

## Hypothesis and prior evidence

Increasing the fixed rate-objective coefficient from 0.25 to 1 while
retaining `mu_init=10`, `mu_max=100`, `dual_lr=0.02`, and `epsilon[c,q]=0`
may reduce the excess coded bits observed with V14 n3 without sacrificing
the task safeguards. This is a falsifiable tradeoff hypothesis, not a
predicted BD-rate improvement.

The independently audited V14 n3 H.264 development screen achieved
BD-rate -1.7594%, BD-accuracy +0.3187 percentage point, and a QP30
Top-1 gap of -1.4423 percentage points. Same-QP rates increased by
17.04--32.85% over raw anchors. V14 n7, which changed only initial task
mu from 10 to 3 with the same source, seed and step budget, failed with
BD-rate +2.1869%, BD-accuracy -0.4226 percentage point, and worst-QP
gap -0.4808 percentage point. Lower initial mu alone did not improve
BD-rate in that screen. These observations motivate testing the rate
coefficient directly; they do not identify a causal optimum.

The existing common engine computes:

```text
mu[c,q] <- clamp(mu[c,q] + 0.02 * task_regret, 0, 100)
L <- mu[c,q] * task_regret + rate_weight * (R(pre+codec)/R(raw+codec)-1)
     + unchanged DCT and distortion regularizers
```

Changing `rate_weight` from 0.25 to 1 multiplies its fixed coefficient
by four. The nominal initial task/rate coefficient ratio changes from
40 to 10. This is not a gradient-norm ratio. The same independent dual
cells continue to react to regret; the stronger rate coefficient can
still worsen task accuracy or fail to reduce true encoded bits.

## Artifact and controlled settings

- Base code: `d39927e92298c111d498f0e1a9f72069469ed0b9`; parent artifact:
  `4e8f8b5ef68b7952d02bead532c27fd54bf8fa0f` (V14 n3).
- Candidate config: `configs/v15_rate_pressure_ar.yaml`.
- Exactly three parsed leaves change relative to
  `configs/v14_dual_safety_ar.yaml`: global
  `loss.task_regret_constraint.rate_weight`, and its `per_codec.h264`
  and `per_codec.h265` overrides, all from 0.25 to 1.0. They implement
  one conceptual intervention. Keeping the per-codec values explicit
  prevents an inherited override from silently shadowing the treatment.
- Every other parsed leaf is identical, including seed `302001`,
  `out_dir: outputs/v14_dual_safety_ar`, model, common OD/AR engine,
  teachers, real-codec STE, proxy settings, non-dual regularizers,
  optimizer LR 1e-5, balanced sampling and the QP mapping. The central
  runner sets a unique actual run directory without changing the recipe.
- Audited H.264 source checkpoint SHA-256:
  `ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`.
  Load only its model weights at source global step 500; reset optimizer
  and all dual state. Train one fresh epoch capped at exactly 1,000
  steps, batch size 2, seed `302001`; each of five balanced H.264
  codec/QP cells is visited 200 times.
- Evaluate the final last checkpoint after all 1,000 fresh steps.
  Proxy validation cannot select the checkpoint; the runner restricts
  training validation to the same development 208 clips.
- Initial codec is real H.264. Both codec overrides implement the same
  treatment; this executor does not upload or launch an H.265 job.

## Evaluator and fixed gates

Use the existing 208 development IDs: shard 0/5, salt
`v13-task-regret-screen-v1`, fingerprint `f321b810af2c507c`. Evaluate
frozen held-out `r2plus1d_18` with real H.264 at QPs 30/35/40/45/50,
preset medium, same-codec paired raw anchors. Primary BD-rate is
calculated from the aggregate Top-1 rate-quality curve, not the average
of per-clip BD-rate.

Advance only if all three development gates pass:

1. H.264 BD-rate <= -10%;
2. BD-accuracy >= 0; and
3. each QP treatment-minus-anchor Top-1 gap >= -0.5 percentage point.

Non-finite BD metrics fail. Same-QP Top-1 improvements or decreasing
training loss alone cannot satisfy the objective. Do not evaluate the
802-clip confirmation complement or full 1,010-clip validation set
unless development passes; the coordinator then evaluates the same
immutable checkpoint independently. The 802 clips are excluded from
this search but were used in historical validation and are not a new
untouched publication holdout. A development pass alone is not success.

## Local readiness

The local parsed-YAML comparison passed: exactly these three leaves changed,
and each changed from 0.25 to 1.0:

```text
loss.task_regret_constraint.rate_weight
loss.task_regret_constraint.per_codec.h264.rate_weight
loss.task_regret_constraint.per_codec.h265.rate_weight
```

The smoke called the actual existing
`src.engine._task_regret_constraint_settings` and
`src.engine._codec_task_regret_constraint` functions. Both codecs resolved
`epsilon=0`, `dual_lr=0.02`, and `rate_weight=1`; initial mu 10, cap 100,
and ten distinct codec/QP cell keys were verified. The old rate constraint
remains disabled, and seed, output directory, epoch and step budget match
the parent. No implementation change or extra YAML-mirroring test was
introduced. The import emitted the existing `cast_ar.py` invalid-escape
warning; the smoke assertions passed.

No GPU score exists for n8. User-resumed improvement does not re-enable
recurring automations; the coordinator owns all execution decisions.

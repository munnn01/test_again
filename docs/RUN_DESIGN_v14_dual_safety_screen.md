# V14 n3: task dual safety screen

Design frozen on 2026-10-01 before this candidate has GPU screen results.
This is a new exploratory experiment following V13, not a reinterpretation
of V13's registered failure. Arbor node `n3` tests one fixed hypothesis.

## Hypothesis and evidence

V13's task duals start too weak relative to the rate objective; preserving
semantics from the first update should let rate optimization remain in a
useful region. The fixed treatment is `mu_init=10`, `mu_max=100`,
`dual_lr=0.02`, `rate_weight=0.25`, still with `epsilon[c,q]=0`.

The audited V13 H.264 screen reduced same-QP bitrate by 3.17--8.41% but
lost 2.40--5.77 Top-1 percentage points; quality-matched BD-rate became
+3.7335%. Its final task duals were 6.8859, 7.2068, 6.8529, 5.4503,
2.8540 for QPs 30/35/40/45/50 after 12,948 steps. Starting at 0.1 makes
semantic correction weak before substantial regret has accumulated.
Those observations motivate this hypothesis but do not prove causation.

`src/engine.py` updates each visited codec/QP dual before its primal step:

```text
mu[c,q] <- clamp(mu[c,q] + dual_lr * (CE(pre+codec)-CE(raw+codec)), 0, mu_max)
L <- mu[c,q] * task_regret + rate_weight * (R(pre+codec)/R(raw+codec)-1)
     + existing DCT and distortion penalties
```

The nominal initial task/rate coefficient ratio rises from 0.1 to 40,
a factor of 400; the dual LR rises by 20 and the cap by 10. This ratio
is not a gradient-norm measurement. Increasing task safety can reduce
available bitrate savings, and training-teacher CE may not transfer to
held-out Top-1. Those are measured failure modes, not reasons to alter
the hypothesis after seeing its result.

## Artifact and controlled settings

- Base code: `d39927e92298c111d498f0e1a9f72069469ed0b9`.
- Config: `configs/v14_dual_safety_ar.yaml`, copied from V13. Model,
  teachers, real-codec STE, optimizer LR, all non-dual regularizers,
  QP mapping and balanced sampling remain unchanged.
- Audited private H.264 500-step source checkpoint SHA-256:
  `ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`.
- Fresh model-only finetune; reset optimizer and both dual schedules.
  Training seed `302001`, one epoch capped at exactly 1,000 steps,
  output `outputs/v14_dual_safety_ar`. With five balanced H.264 QP
  cells this visits each cell 200 times.
- Initial codec is H.264. H.265 fields use the same treatment values
  to prevent an inherited per-codec override from restoring V13's
  `dual_lr=0.001` or `rate_weight=1`. No H.265 job is launched by this
  executor.
- Diagnostic-only change: static `loss.lam_task=1` instead of 0.
  `_fit` replaces it with the current task dual whenever the constraint
  is active, so the optimized training objective is unchanged by this
  field. `_val_loss` uses the static weights, so this adds CE to its
  proxy validation monitor. This monitor is neither constrained regret
  nor real-codec acceptance evidence and cannot select the evaluated
  checkpoint. Static `beta=0` remains unchanged.
- Use the final 1,000-step checkpoint, never a proxy-validation-selected
  checkpoint. The coordinator confirmed that its runner generates a
  training index with val restricted to the exact development 208 IDs.
  This prevents the original full-val proxy monitor from touching the
  untouched validation complement.

## Evaluator and decision rule

Evaluate only the exact existing V13 development IDs: shard 0/5, salt
`v13-task-regret-screen-v1`, 208 clips, fingerprint `f321b810af2c507c`.
Use frozen `r2plus1d_18` and real H.264 QPs 30/35/40/45/50 with
same-codec paired raw anchors. Primary BD-rate uses the aggregate
Top-1 rate-quality curve, not the mean of per-clip BD-rate.

Advance only if the development screen simultaneously has:

1. H.264 BD-rate <= -10%;
2. BD-accuracy >= 0; and
3. every QP Top-1 treatment-minus-anchor gap >= -0.5 percentage point.

Non-finite BD metrics fail this gate. This development result remains
exploratory. The untouched 802-clip validation complement is excluded
from this executor, source inspection and optimization feedback; the
coordinator owns the independent confirmation gate and later test.
No claim of success is made from a training-loss decrease or same-QP
rate decrease alone. No hyperparameter pivot occurs inside node n3.

## Status

CPU configuration smoke passed: exactly 13 intended field changes from
V13, correct overrides for either codec, and independent codec/QP dual
cells. All 30 relevant existing tests passed (V13 regret/loss, CRC-V5,
QP sampling, sequence reporting and additive conditioning).
The first pytest attempt had 29 passes and one setup error because the
explicit temporary directory parent did not exist; creating that parent
resolved it. The existing cast_ar.py invalid-escape warning is unrelated.

Validation command:

```powershell
python -B -m pytest tests/test_v13_task_regret.py tests/test_crc_v5.py tests/test_qp_sampling.py tests/test_sequence_bd_report.py tests/test_additive_cond.py -q -p no:cacheprovider --basetemp outputs/pytest-n3-20261001-r2
```

GPU training and screen evaluation are coordinator-owned; this file
contains no measured V14 result yet.

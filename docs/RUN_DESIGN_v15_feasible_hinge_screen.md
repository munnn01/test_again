# V15 n9: primal task hinge with signed codec/QP duals

Design registered on 2026-10-02 before this candidate has GPU results.
Arbor node n9 tests one fixed hypothesis. Original V13 code, configuration,
registered interpretation and archived outcomes remain unchanged. This
candidate extends the shared OD/AR/tracking engine through an optional mode;
no second task-specific training loop or new dependency is introduced.

## Hypothesis

The n3 screen obtained H.264 BD-rate -1.7594%, BD-accuracy +0.3187
percentage point, but QP30 Top1 fell 1.4423 points. Same-QP rate rose
17.04--32.85%. Its signed batch-mean CE primal can buy greater confidence
even after treatment CE is better than its raw-codec anchor. Removing that
confidence reward once the task budget is met may leave more gradient
pressure to reduce rate. This is a mechanistic hypothesis; the screen
alone cannot prove that confidence reward caused the excess bits.

## Fixed treatment

Let G[c,q] be mean CE(pre+real_codec[c,q]) minus mean
CE(raw+real_codec[c,q]) measured using the same pinned frozen teacher.
The anchor loss is detached. The only new parsed configuration field
relative to configs/v14_dual_safety_ar.yaml is:

```yaml
loss:
  task_regret_constraint:
    primal_mode: positive_part
```

The optional primal mode changes n3's signed objective:

```text
L_signed = lambda_R * (R_pre / R_raw - 1) + mu[c,q] * (G[c,q] - epsilon[c,q])
           + unchanged DCT and distortion penalties
L_hinge  = lambda_R * (R_pre / R_raw - 1) + mu[c,q] * relu(G[c,q] - epsilon[c,q])
           + unchanged DCT and distortion penalties
```

Both modes update only the visited codec/QP cell using SIGNED violation:

```text
mu[c,q] <- clamp(mu[c,q] + dual_lr * (G[c,q] - epsilon[c,q]), 0, mu_max)
```

The signed regret and violation are logged before rectification. The
primal hinge is applied after the dual update, so feasible regret still
lowers mu. Omitting primal_mode is exactly the existing signed path;
unsupported names fail before the training iteration. The hinge changes
neither CE averaging nor teacher selection. It still permits positive
per-example regret to cancel against negative regret elsewhere in the
batch. It also has zero task derivative at the budget boundary under
PyTorch ReLU. It is a heuristic feasible-region primal update rather than
a claim of the same signed Lagrangian or convergence guarantee.

## Controlled artifact and budget

- Candidate configuration: configs/v15_feasible_hinge_ar.yaml.
- Code lineage: n3 branch commit 4e8f8b5ef68b7952d02bead532c27fd54bf8fa0f
  over base d39927e92298c111d498f0e1a9f72069469ed0b9.
- Model additive_cond, cond_dim=3; one common preprocessor pipeline.
- Frozen sampled training teachers r3d_18 and mc3_18; pin the same
  active teacher across raw and treatment arms in each step.
- Audited H.264 source checkpoint at step500, SHA-256
  ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a.
- Model-only fresh finetune; reset optimizer, scheduler and duals.
- Seed302001, exactly 1,000 fresh steps, one epoch, batch2, LR1e-5,
  balanced QPs30/35/40/45/50. Evaluate final-last global_step1000;
  proxy validation cannot choose the evaluated checkpoint.
- mu_init=10, mu_max=100, dual_lr=0.02, epsilon=0,
  lambda_R/rate_weight=0.25 for both H.264 and H.265 overrides.
- Retain every n3 codec/proxy setting and non-dual coefficient,
  including DCT kappa=10 and distortion mu=10. The initial experiment
  uses real H.264 medium; no H.265 execution is implied by this design.
- Preserve the parsed out_dir outputs/v14_dual_safety_ar; the
  coordinator gives every submitted job its own output directory.

## Locked evaluation and acceptance

Use only the existing development208 clips, fingerprint
f321b810af2c507c, fixed salt v13-task-regret-screen-v1, shard0/5.
Train and diagnostic validation must use the coordinator's development
index. Evaluate frozen held-out r2plus1d_18, real H.264 medium at
QP30/35/40/45/50, paired to the same-codec raw anchors. Primary BD-rate
comes from the aggregate Top1 rate-quality curve. Non-finite metrics fail.

Advance only when all three development conditions hold:

1. BD-rate <= -10%;
2. BD-accuracy >= 0; and
3. every QP treatment-minus-anchor Top1 gap >= -0.5 percentage point.

No inspection or tuning feedback from the reserved802 complement or
full1010 task outputs is permitted here. A passing development result
would still require a fresh coordinator-owned run evaluating the exact
same checkpoint on both reserved802 and full1010, with these gates. The
validation set had historical V12 usage, so the complement is reserved
from the current optimization search, not a newly collected population.

## Status

Implementation is complete. The focused CPU suite passed 28 tests,
covering the shared-loop hinge integration, original V13 regret tests and
CRC-V5 regressions. In a controlled scalar edit fixture, feasible signed
regret=-0.2 with excess rate=0.02 gives hinge gradient +0.025 toward lower
rate, while the unchanged signed path gives -9.971 toward more confidence.
The same signed feasible violation lowers its dual from10 to9.996. With
regret=+0.2 and epsilon=0.1, the hinge remains active and increases its
visited dual to10.002. Other tests verify nonzero epsilon budgets,
independent codec/QP cells, anchor detachment, one pinned teacher for both
arms, release on exceptions and unknown-mode failure before any forward.
The candidate YAML has exactly one added parsed field relative to n3.

The first test run exposed ordinary NaN equality in a diagnostic assertion;
the assertion now compares numeric diagnostics exactly with matching NaNs
accepted. There was no implementation failure. The existing cast_ar.py
invalid-escape warning is unrelated.

```powershell
python -B -m pytest tests/test_v15_feasible_hinge.py tests/test_v13_task_regret.py tests/test_crc_v5.py -q -p no:cacheprovider --basetemp C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/pytest-v15-hinge-20261002-r3
```

No GPU job was submitted by this executor and dev_score is null. These
finite CPU fixtures verify objective wiring and gradient behavior; they do
not establish codec-proxy transfer, AR accuracy or measured BD-rate gains.

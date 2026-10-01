# V14 n7: matched moderate task-dual initialization

Registered 2026-10-02 before any n7 GPU result. The candidate is pending
coordinator allocation/dispatch. This executor changes configuration
only and does not submit jobs, inspect confirmation results, or update
the persistent search tree or active registry.

## Fixed hypothesis and preceding evidence

n3's initial semantic coefficient is excessive for this source and
1000-step calibration: reducing only task-dual initialization from10
to3 will reduce added bitrate while retaining useful task performance.
This tests excessive semantic weighting, not a mathematical necessity
for stronger or weaker duals in general.

The coordinator's audited n3 H.264 dev208 report is BD-rate -1.7594%,
BD-accuracy +0.3187 percentage point, and worst QP30 Top-1 gap -1.4423
percentage points. At every QP, same-QP rate increased17.0--32.8%; final
task duals fell from10 into6.60--9.63. n3 therefore failed the -10%
BD-rate threshold and the per-QP task guardrail despite positive
BD-accuracy. The positive same-QP rates suggest that strong semantic
optimization can spend additional bits; the falling duals suggest the
training-teacher mean regret often allowed relaxation. Those facts
motivate this intervention but do not prove a causal explanation.

The nominal initial task/rate coefficient ratio changes from40 to12.
Gradient clipping, Adam and surrogate geometry prevent interpreting
that ratio as a corresponding ratio of parameter updates. Lowering
mu_init could revive V13's semantic collapse or fail held-out transfer;
these possibilities are evaluated by the unchanged acceptance gate.
No hypothesis pivot follows this candidate's result.

## Artifact and exact matching

- Parent commit `4e8f8b5ef68b7952d02bead532c27fd54bf8fa0f`;
  isolated branch `arbor/v14-moderate-dual`.
- `configs/v14_moderate_dual_ar.yaml` is parsed-identical to n3's
  `configs/v14_dual_safety_ar.yaml` except the single leaf
  `loss.task_regret_constraint.mu_init=3.0`.
- All other parsed fields remain fixed: epsilon0, mu_max100,
  dual_lr0.02, rate_weight0.25 (including both per-codec overrides),
  DCT kappa10, distortion mu10, teachers, architecture, strength,
  real-codec STE/preset, proxy, QPs and balanced sampling.
- Training seed302001 deliberately matches n3. Both start from the
  same private H.264 seed282001 audited500-step last checkpoint,
  SHA-256 `ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`.
  Reload model weights only, reset optimizer and duals. Never initialize
  from n3's trained final checkpoint or from V13 final weights.
- One epoch capped at exactly1000 steps, giving200 visits per QP.
  LR1e-5, gradient clipping1, and total-step anneal are unchanged.
  n3 and n7 both use the1000-step anneal schedule.
- Static loss.lam_task1 is unchanged: constrained training replaces it
  with the cell dual; proxy validation includes ordinary CE and is not
  the real-codec acceptance objective. Evaluate the final1000-step
  checkpoint, not a validation-selected best checkpoint.
- The inherited out_dir is deliberately unchanged to preserve the
  one-leaf comparison. The coordinator must override it to a unique
  job directory, e.g. `outputs/v14_moderate_dual_ar`, before execution.
  Do not run this config directly against n3's output directory.
- Training diagnostic validation uses only the exact dev208 membership
  through the coordinator-generated index. No source/model/engine/loss
  change or additional training loop is introduced.

## Evaluator and admission

Use exactly V13 development shard0/5, salt
`v13-task-regret-screen-v1`:208 clips, fingerprint `f321b810af2c507c`.
Evaluate frozen `r2plus1d_18` with real H.264 QPs30/35/40/45/50 and
same-codec paired raw anchors. Primary BD-rate uses aggregate Top-1
rate-quality curves, not mean per-clip BD-rate.

Advance only if all development conditions hold simultaneously:

1. BD-rate <= -10%;
2. BD-accuracy >=0; and
3. every QP Top-1 treatment-minus-anchor gap >= -0.5 percentage point.

Non-finite BD metrics fail the gate. A dev pass is exploratory. Only
the coordinator may evaluate unchanged checkpoint bytes/code/settings
on the disjoint802 validation complement and full1010; automatic
success requires all three criteria on both confirmation802 and
full1010 for at least one codec. No tuning follows individual
confirmation clip/QP output, and this executor does not inspect it.

Historical-use disclosure: the1010 validation clips were used in prior
research, and the historical1159 test clips were inspected in V1/V2.
Confirmation802 is excluded from the present autonomous search's
training diagnostics and development feedback; it is not a new
population or pristine publication holdout. Confirmation and historical
test clips do not enter training. OD remains frozen; this AR comparison
does not validate a shared OD/AR checkpoint.

## Local validation and status

Parsed-config smoke passed: exactly one changed leaf, mu_init10 ->3;
all other fields matched n3, including seed302001 and1000 steps.
Constraint validation passed for H.264, H.265 and both together, with
five independent QP cells per codec and the unchanged per-codec
epsilon/dual-LR/rate-weight overrides. No training or new test loop
was introduced. GPU allocation, immutable packaging, dispatch and
result auditing remain coordinator-owned. No n7 dev score exists yet.

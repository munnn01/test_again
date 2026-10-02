# V16 n11: pixel reconstruction release screen

Design registered on 2026-10-02 before any V16 GPU result. This executor
tests one fixed hypothesis and makes no outcome-dependent recipe change.
The existing common OD/AR/tracking engine and learned preprocessor remain
in use. GPU execution, auditing and confirmation are coordinator-owned.

## Hypothesis and prior evidence

Pixel reconstruction MSE may preserve details that the machine task does
not require and oppose rate reduction even after batch-mean task regret
is feasible. Removing just that term could let the unchanged rate and
DCT objectives reduce coded bits while the positive-part task primal and
per-codec/QP signed dual updates continue to guard task regret. This is a
mechanistic hypothesis, not proof that MSE caused prior bitrate inflation.

The audited parent n9 H.264 screen (same source, seed302001 and 1,000
fresh steps) had BD-rate +2.6945%, BD-accuracy -1.0680 percentage points,
worst-QP Top-1 gap -0.9615 percentage point, and same-QP bit inflation
+3.18--5.89%. It failed development acceptance. These measurements bound
the positive-part recipe's outcome; they do not isolate the MSE term.
Removing source fidelity could also damage recognition or expose a
codec/proxy gradient mismatch. Neither decreasing train loss nor a
favorable toy derivative can establish the machine-at-equal-bitrate claim.

## Fixed intervention

Copy `configs/v15_feasible_hinge_ar.yaml` to
`configs/v16_pixel_release_ar.yaml` and change exactly one parsed leaf:

```yaml
loss:
  mu: 0.0  # parent 10.0; source pixel MSE coefficient
```

`loss.mu` is the pixel MSE coefficient, distinct from the task dual
`mu[c,q]`. The latter retains its initial value10, cap100 and LR0.02.
The existing `src.losses.preprocessing_loss` uses the pixel coefficient
for `MSE(x_hat, x_source)`, where x_hat is the codec reconstruction.
The candidate objective is the existing composition with that coefficient
set to zero:

```text
G[c,q] = mean CE(pre+codec) - mean CE(raw+same_codec)
mu_task[c,q] <- clamp(mu_task[c,q] + 0.02 * (G[c,q] - epsilon[c,q]), 0, 100)
L = 0.25 * (R_pre/R_raw - 1) + mu_task[c,q] * relu(G[c,q] - epsilon[c,q])
    + 10 * L_DCT + 0 * MSE(x_hat, x_source)
```

All other coefficients remain identical to the parent, including
DCT kappa10. Signed regret still drives the dual update; only the primal
task term uses `positive_part`. The batch-mean constraint still allows
per-example regret cancellation and does not directly guarantee held-out
Top-1. No source code, architecture, codec or scoring helper changes.

## Controlled artifact and budget

- Parent code/artifact: `60168b918bc0eab76ebe54ad4589ab6dd18e8e43`
  (V15 positive-part), over the existing registered code lineage.
- Audited H.264 source checkpoint at step500, SHA-256:
  `ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`.
- Model-only finetune from that immutable source; reset optimizer,
  scheduler and task/rate dual state. Seed302001, one fresh epoch capped
  at exactly 1,000 steps, batch2, LR1e-5, balanced QPs30/35/40/45/50,
  giving200 visits per H.264/QP cell. Evaluate the final-last checkpoint
  at global_step1000; proxy validation cannot select another checkpoint.
- Keep additive_cond with cond_dim3, sampled frozen r3d_18/mc3_18
  training teachers, STE real H.264 medium and every parent proxy setting.
  Raw and treatment use the same pinned teacher in each training step.
- Keep `primal_mode=positive_part`, epsilon0, mu_init10, mu_max100,
  dual_lr0.02, rate_weight0.25 globally and in both codec overrides;
  keep the old rate constraint disabled.
- Every parsed value apart from `loss.mu` is unchanged, including
  out_dir `outputs/v14_dual_safety_ar`. The central wrapper sets a unique
  actual job directory. No separate OD/AR training pipeline is created.
- Diagnostic validation is restricted to the exact development208 IDs
  by the coordinator-generated training index. No confirmation clip
  output or historical test output enters optimization feedback.

## Locked evaluator and acceptance

Use only the fixed existing development208 clips, fingerprint
`f321b810af2c507c`, shard0/5 and salt `v13-task-regret-screen-v1`.
Evaluate frozen held-out `r2plus1d_18` with real H.264 medium at
QP30/35/40/45/50, paired to raw anchors encoded with the same codec/QP.
Primary BD-rate comes from aggregate Top-1 rate-quality curves, not a
mean of per-clip BD-rate. Non-finite metrics fail acceptance.

Advance only if all development conditions pass:

1. BD-rate <= -10%;
2. BD-accuracy >=0; and
3. every QP treatment-minus-anchor Top-1 gap >=-0.5 percentage point.

A development pass alone is insufficient. Only the coordinator may then
run a fresh independent evaluation of the same immutable checkpoint on
the reserved802 complement and full1010 validation set, with unchanged
gates. The complement is excluded from current search but has known
historical validation use; it is not a newly collected untouched holdout.
No best artifact is promoted without that confirmation.

## Local readiness

The parsed-YAML comparison passed with exactly one difference:

```text
loss.mu: 10.0 -> 0.0
```

Actual `src.engine._loss_weights` resolves pixel mu0 and DCT kappa10.
`_task_regret_constraint_settings` and `_codec_task_regret_constraint`
resolve positive_part, initial task mu10, cap100, LR0.02, epsilon0 and
rate_weight0.25 for both codecs, with ten distinct codec/QP cells. Seed,
step budget, disabled old rate constraint and parsed output directory
were checked unchanged.

All28 existing focused CPU tests passed, covering the shared-engine
positive-part integration, original task-regret behavior and CRC-V5.
No extra test that merely mirrors the new YAML was added. The existing
cast_ar.py invalid-escape warning is unrelated.

```powershell
& 'C:\Users\Admin\AppData\Local\Temp\vcm-v13-tools\Scripts\python.exe' -B -m pytest tests/test_v15_feasible_hinge.py tests/test_v13_task_regret.py tests/test_crc_v5.py -q -p no:cacheprovider --basetemp 'C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/pytest-v16-pixel-release-20261002-r1'
```

An ad hoc float64 CPU fixture called the actual composite loss and the
existing primal transform. It used a constant all-one source clip with
shape[1,3,2,16,16], reconstruction/preprocessor output z*source at z=0.5,
supplied task CE0.8 versus detached anchor CE1.0 (regret=-0.2, inactive
hinge), and a deliberately linear synthetic rate objective z-1. Both
runs retained DCT kappa10 and the parent coefficient set, with the initial
task coefficient10 and rate coefficient0.25. The DCT high-frequency term
is numerically negligible for this constant clip. With pixel mu10, the
loss was2.375 and dL/dz=-9.75; with pixel mu0, the loss was-0.125 and
dL/dz=+0.25. The MSE derivative opposes the linear rate derivative in
this constructed feasible-regret case. This establishes objective wiring
and an illustrative possible conflict only: the scalar rate is not a
real codec, and the fixture measures neither AR nor BD-rate nor the cause
of n9's outcome.

No V16 development score has been measured. Recurring automations remain
off; this executor does not upload or launch Kaggle.

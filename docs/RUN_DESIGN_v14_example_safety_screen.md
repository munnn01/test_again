# V14 n4: per-example task-regret safety objective

Registered before GPU results on 2026-10-01. No metric has been evaluated for
this intervention. Its hypothesis is fixed: V13 mean cross-entropy regret
allows accuracy improvements on some examples to cancel damage to other
examples; preventing that cancellation and preserving correct-anchor decisions
will protect machine accuracy while the same rate primal minimizes bitrate.

## Equations

The raw anchor `a_i` is the unedited training clip passed through the same real
codec `c` and QP `q` as treatment `t_i`. All logits are from frozen training
teachers `r3d_18` and `mc3_18`; `r2plus1d_18` is used only for evaluation. Anchor
logits are detached. Treatment logits backpropagate through the frozen teacher
and real-codec STE to the existing preprocessor.

For teacher `j`, label `y_i`, and temperature `T`, define:

```text
ce_j,i(z) = CE(z_j,i, y_i)
b_j,i = 1[argmax(z^a_j,i) == y_i]
m_j,i(z) = z_j,i[y_i] - max_{k != y_i} z_j,i[k]
p_j,i(z) = softmax(z_j,i / T)
P_j = mean_i max(0, ce_j,i(t) - ce_j,i(a))
M_j = mean_i b_j,i * max(0, m_j,i(a) - m_j,i(t))
K_j = T^2 * mean_i b_j,i * max(0, KL(p_j,i(a) || p_j,i(t)))
G_j = P_j + margin_weight * M_j + kl_weight * K_j
G = sum_j w_j * G_j
J = rate_weight * (R(t)/R(a) - 1) + mu[c,q] * (G - epsilon[c,q])
    + existing V13 DCT and source-distortion regularizers
mu[c,q] <- clip(mu[c,q] + dual_lr[c] * (G - epsilon[c,q]), 0, mu_max)
```

Weights `w_j` sum to one. Sample mode uses one teacher pinned for the full
anchor/treatment/saliency/distillation step. Mean mode computes positive regret
and guards separately for each teacher before weighting, so neither examples
nor teachers cancel positive regret. Incorrect anchors still contribute to CE
regret, while both margin and KL guards contribute zero for those examples.
Guards divide by the full batch size, not the count of correct anchors. KL is
clamped only to remove floating-point negative roundoff. Equal anchor and
treatment logits produce zero constraint value and finite input gradients.

At `epsilon=0`, improvements on one example cannot pay for CE damage to
another. This is a training surrogate; held-out Top-1 or BD-rate is not
mathematically guaranteed. All three acceptance metrics must be measured.

## Compatibility and configuration

`loss.task_regret_constraint.mode` defaults to `mean`, which preserves V13's
scalar `CE(t)-CE(a)` and the existing OD/tracking loss fallback. Optional mode
`positive_per_example` requires differentiable logits from frozen training
teachers and is rejected before training for unsupported analyzers. Safeguard
weights default to zero and temperature defaults to one. Weights must be
finite/nonnegative; temperature must be finite/positive. Nonzero safeguard
weights with mean mode are rejected rather than silently ignored.

`configs/v14_example_safety_ar.yaml` freezes the n4 screen:

- Audited H.264 seed-282001 **500-step last** source checkpoint used in V13;
  coordinator must validate its SHA-256 and copy only those source weights to
  the ordinary fine-tune checkpoint slot. Never begin from failed V13 terminal
  weights. Fresh optimizer, step counter, and codec/QP task duals.
- Seed `302002`, H.264, `epochs=1`, `max_steps=1000`, LR `1e-5`, batch size 2;
  no cosine, balanced randomized QPs 30/35/40/45/50, real codec preset medium.
- `epsilon=0`, `mu_init=1`, `mu_max=100`, `dual_lr=0.02`, `rate_weight=0.25`,
  `margin_weight=1`, `kl_weight=0.5`, `T=1`.
- Existing additive conditional architecture and all V13 non-dual regularizers
  retained, including `kappa=10` and source-distortion `mu=10`.
- `loss.lam_task=1.0` supplies scalar CE in diagnostic validation only. Active
  task-dual training replaces it with `mu[c,q]`; rate remains the primal and
  the per-example safety objective remains the constrained task quantity.
  The scalar validation monitor is not the constrained objective or an
  acceptance metric. This screen evaluates the final 1,000-step checkpoint;
  it does not use diagnostic scalar CE to select a checkpoint across epochs.
- One existing common engine and analyzer API, with no new task training branch.
  Codec-specific dual cells remain independent and the common API supports
  either standard codec. This development candidate uses H.264 only.
- Evaluate the final checkpoint on the exact V13 **208 IDs**, fingerprint
  `f321b810af2c507c`, held-out `r2plus1d_18`, QPs 30/35/40/45/50. Coordinator
  restricts validation index to these IDs before training and evaluation.
- The other **802** locked validation IDs are withheld from this search.
  The historical test split is also excluded from this search; README records
  that V1/V2 already evaluated it, so there is no fresh publication holdout
  claim. No inference against either split is performed by this executor.

Advance only if H.264 BD-rate <= -10%, BD-accuracy >= 0, and every QP's
Top-1 treatment-anchor gap is >= -0.5 percentage point. A development pass
must subsequently pass the coordinator's withheld confirmation gate. The
executor's development score remains null until audited GPU results exist.

## Verification

CPU tests cover positive-regret cancellation, differentiable AR logits and
reconstruction gradients, frozen teacher/anchor detachment, correct-anchor
gating, zero identity, pinned sample teachers, nonlinear mean aggregation,
invalid/missing settings, and default scalar OD/tracking compatibility. The
existing V13 and loss tests are also retained. GPU execution belongs to the
coordinator; this design makes no empirical BD-rate claim.

Verified CPU command (2026-10-01):

```powershell
& 'C:\Users\Admin\AppData\Local\Temp\vcm-v13-tools\Scripts\python.exe' -m pytest tests/test_v14_example_safety.py tests/test_v13_task_regret.py tests/test_losses.py tests/test_task_roi.py tests/test_crc_v5.py tests/test_qp_sampling.py tests/test_eval_sharding.py tests/test_config_overrides.py --basetemp 'C:\Users\Admin\AppData\Local\Temp\vcm-n4-example-safety-03' -q
```

Result: **57 passed** in 4.68 seconds. `git diff --check` also passed.
These are CPU implementation checks, not GPU development-score evidence.

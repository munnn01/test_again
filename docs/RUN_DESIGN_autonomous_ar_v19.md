# V19: fixed spatial smoothing of the learned RGB edit

Registered on 2026-10-02 before these candidates have GPU results. Arbor
direction n17 compares n18/A and n19/B under one fixed hypothesis: smoothing
only the learned additive RGB residual may lower codec bit overhead while
retaining sufficient machine-task improvements to reach the registered
quality-matched rate gate. The baseline recipe is historical audited V18A.
The learnable backbone and weights layout are preserved; one fixed optional
residual filter is added. Original V13 and closed lineages stay recorded.

## Audited parent and evidence boundary

The audited V18A calibrated rate_weight=0.5 screen obtained BD-rate
-3.1228461294294574%, BD-accuracy +1.5453310796991189 percentage points,
and worst same-QP Top1 gap +0.961538 point. It passes task guards but
fails the requested -10% rate gate. V18B rate_weight=1 obtained BD-rate
+2.4481624990030815%, BD-accuracy -0.82111897 point and worst gap
-1.4423 points; it fails both rate and task acceptance. These are fixed
208-clip development screens rather than confirmed population effects.
At QPs30/35/40/45/50, V18A's same-QP bit overheads are
+10.1026/+10.7201/+9.7627/+8.1960/+6.1514%. V18B's are
+8.0518/+8.5864/+7.6864/+6.5535/+5.1913%: lower same-QP overhead still
coincides with a worse quality-matched curve, so bit-only reduction is
insufficient acceptance evidence.

The improved V18A recipe motivates a structural edit filter instead of
another rate-coefficient sweep. It does not prove that its residual is
noise or that Gaussian attenuation will preserve useful recognition cues.
Historical control results do not replace a newly matched control, and
small contrasts on this repeatedly used development population remain
exploratory. No closed outcome is overwritten or relabeled as a pass.

## Two registered treatments on identical common code

| Arbor node / arm | Configuration | Only parsed change from V18A |
|---|---|---|
| n18 / A | configs/v19_residual_filter05_ar.yaml | model.residual_filter_sigma=0.5 |
| n19 / B | configs/v19_residual_filter1_ar.yaml | model.residual_filter_sigma=1.0 |

Both configs copy configs/v18_calibrated_rate05_ar.yaml from common-code
parent 64f6dc9495f08ad82863b7ab28e6bb75465284ef. The implementation executor
adds one optional filter to AdditiveCondPreprocessor, and the coordinator
integrates both configs/designs and that implementation into the same
immutable common commit and payload before either job. This design/config
executor does not modify algorithm source or tests.

Let r_theta(x,c) be the existing signed learned RGB edit before scaling,
and let G_sigma be a fixed normalized separable spatial Gaussian:

```text
x_pre = clamp(x + strength * G_sigma(r_theta(x,c)), 0, 1)
```

Apply it per RGB channel and per frame, with no temporal averaging.
Radius=ceil(3*sigma); replicate border padding; sigma0.5 gives radius2,
kernel5; sigma1 gives radius3, kernel7. No learned filter weights or
persistent checkpoint keys are added. The sigma=0 default preserves the
old forward and RNG path exactly. Both arms filter during training and
evaluation, before the existing output clamp; no evaluation-only transform,
strength sweep, QP selector or outcome-based sigma adjustment is allowed.

The source enters through the original additive path; only the edit is
filtered. That algebraic structure does not imply identity, unchanged
pixels or task safety: the signed edit and final clamp still modify output.
Residuals can be positive or negative, so they can also remove source
content. There is no general theorem that additive edits must add bits.

## Connection to existing work and limitations

The older local CDTR w24254, section7.5, motivates suppressing unwanted
content while retaining machine-useful details and cautions about uniform
filtering of fine cues. It does not prescribe Gaussian smoothing of learned
residuals. Our adaptation and the protocol differences are recorded in
[MPEG alignment](MPEG_VCM_alignment_20261002.md). The final published
TR's full text was not read, and this experiment is neither a full MPEG CTC
run nor evidence of standard conformance.

Residual spectrum targeting already exists in this repository.
[RUN_DESIGN_additive.md](RUN_DESIGN_additive.md), sections5.3/5.4, documented
costly added structure, a near-null source-TV sweep, and gamma_res
TV(x_pre-x) with registered strengths10/30/100. Those older analyses are
historical lineage evidence, not universal impossibility proofs. V19 uses
an explicit fixed structural filter on the calibrated log-rate0.5 recipe;
it does not repeat that TV coefficient sweep or claim spectrum targeting
as new. The frozen OD background/decoder Gaussian method and its failed
whole-frame Gaussian POST transfer to AR in
[RESULTS_generalization_matrix.md](RESULTS_generalization_matrix.md) are
also distinct, closed comparisons; their numbers do not validate V19.

Smoothing can erase useful high-frequency semantic corrections, spread a
signed correction spatially, alter borders, or lead the network to increase
edit amplitude to compensate. It cannot directly denoise the retained raw
source path. Lower residual high-frequency energy need not reduce actual
H.264 bits, especially under inter prediction and the imperfect STE rate
derivative. Pixel MSE/DCT penalties, clipping, and teacher CE tradeoffs remain
active. Spatial-only filtering also cannot establish temporal coherence,
H.265 transfer or a jointly validated OD/AR recipe. All five QPs and final
held-out real-codec task curves remain necessary acceptance evidence.

## Immutable source, recipe and evaluation

- Load the original private H.264 source at step500, model-only, SHA-256
  ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a.
  Do not resume or reuse the trained V18A/B weights. Reset optimizer,
  scheduler and codec/QP dual state for each candidate.
- Existing additive_cond model, cond_dim3, unchanged strength1, frozen
  sampled r3d_18/mc3_18 training teachers, pinned across each step;
  frozen held-out r2plus1d_18 at evaluation.
- Seed302001, exactly1000 fresh steps, final-last global_step1000 only;
  LR1e-5, batch2, one epoch, balanced QPs30/35/40/45/50, no cosine or
  proxy-validation checkpoint selection.
- Real H.264 medium STE; log_coeff proxy plus detached rate calibration;
  global/per-codec rate_weight0.5, source pixel MSE mu10, DCT kappa10,
  omega0, positive_part task primal, epsilon0, dual init10/cap100/LR0.02.
  Every other parsed parent field, including out_dir, stays unchanged;
  wrapper overrides each job to its unique output path.
- Development208 fingerprint f321b810af2c507c, salt
  v13-task-regret-screen-v1, shard0/5; diagnostic validation is confined
  to those IDs. Real same-codec raw/treatment pairing at all five QPs,
  aggregate Top1 curves and the existing scorer remain unchanged.
- Advance only if finite BD-rate <= -10%, BD-accuracy >=0, and every
  same-QP Top1 gap >= -0.5 percentage point. Only a passing immutable
  checkpoint may be independently confirmed on reserved802 and full1010
  with identical gates. The validation set has historical V12 exposure;
  the reserved complement is excluded from this current search rather
  than a newly collected untouched population. No reserved task outputs
  are inspected or used to tune these arms.

## Execution scope and readiness

Authorized execution is limited to private qktttttttttt notebooks and free
GPU for these two candidates under the existing user approval. Other
accounts have no authorization here. Do not upload source MPEG documents,
PDFs, data, checkpoints or tokens. Existing Kaggle data/checkpoints remain
mounted sources. No new goal, recurring check or GitHub push is created.
The executor makes only local config/design commits; the coordinator owns
CPU implementation/integration checks, packaging, submission and audit.

Both YAML files parse and each has exactly one added model filter-sigma
leaf relative to V18A; every other parsed field is identical. Whitespace
checks pass. No config-mirror tests or unchanged common-code suite are
added or repeated by this executor. Combined implementation, checkpoint
compatibility and source/API tests remain coordinator-owned before GPU.
Both dev_score values remain null; no measured V19 outcome exists yet.

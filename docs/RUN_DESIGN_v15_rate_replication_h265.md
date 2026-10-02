# V15 paired rate-weight replication on H265

Registered 2026-10-02 before preparing/submitting these six GPU jobs, following
human instruction "có thể dùng nhiều nick kaggle để kiểm chứng trọng số rate".
Private free-GPU notebooks use the three existing allocated accounts. Tokens
stay local; no datasets or checkpoints are uploaded. There is no new goal,
recurring automation or scheduled result check. Existing H264 jobs continue.

## Hypothesis and locked paired blocks

n10 tests whether increasing only rate_weight0.25 to1 in the n3 recipe improves
H265 rate-task efficiency, across three fixed-source fine-tuning seeds. H265
is available to all three accounts; the private H264 source is account-local.
This study therefore estimates the H265 effect and does not imply replication
on H264. Accounts are execution blocks, not extra biological/statistical units.

| Account | Seed | Control | Treatment |
|---|---:|---:|---:|
| qktttttttttt | 302011 | rate_weight0.25 | rate_weight1 |
| baoancut | 302012 | rate_weight0.25 | rate_weight1 |
| trnhlng | 302013 | rate_weight0.25 | rate_weight1 |

Both arms within each block start from exactly the same source bytes, seed,
optimizer reset and independent codec/QP dual initialization;1000 fresh steps,
LR1e-5, dual mu_init10/mu_max100/lr0.02, epsilon0, signed primal remain fixed.
Same source: baooo25r/crc-v5-h265-minus24-candidate-v1/preprocessor.pth,
step500, SHA256 a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e.
The historical44-clip score is not a score for this208-clip experiment.

Control: commit4e8f8b5ef68b7952d02bead532c27fd54bf8fa0f,
configs/v14_dual_safety_ar.yaml. Treatment: commit4460a9f56243d1485c195d884e9c0e03414d90c1,
configs/v15_rate_pressure_ar.yaml. Exactly three parsed rate_weight leaves
(global and both codec overrides) differ0.25->1; these implement one intervention.
The central sealed wrapper overrides actual codec toH265, block seed and output
path. Architecture additive_cond/cond_dim3, shared train.py/_fit, loss.mu10
(reconstruction MSE), DCT kappa10, codec-native proxy normalization, stationary
STE, teacher sampling, QP exposure, data order policy and final-LAST selection
are unchanged. It remains the existing common OD/AR pipeline; OD stays frozen.

## Evaluator and analysis

All six final checkpoints use the exact immutable autonomous-learned-ar-v14
scorer, real H265 medium, QPs30/35/40/45/50, held-out r2plus1d_18 on dev208
fingerprint f321b810af2c507c. Independent local audit checks source, steps,
configuration, effective H265 rate weight and all paired codec score rows.

Primary contrast: paired delta = BD-rate(w1)-BD-rate(w025), negative favoring1.
Report all three deltas, their mean, sample standard deviation, and two-sided
95% Student-t CI with df2 (t=4.302652729911275). Report per-arm mean/sampleSD,
BD-accuracy and worst QP Top1 gaps; same-QP bitrate ratios remain descriptive.
This CI describes fine-tuning seed variation conditional on one pretrained
source and one fixed development set; it does not quantify population/split
uncertainty. With only three pairs, interval estimation is imprecise and assumes
approximately normal seed contrasts. Clip/QP rows are not extra seed replicates.

Missing, running, failed or unaudited runs remain visible. No aggregate/effect
claim until all three pairs are audited. Evidence for lower rate_weight1
BD-rate requires the paired CI upper bound<0 and task guards satisfied by all
three treatment runs. Keep negative results and every seed; no lucky-seed pick.

## Acceptance and confirmation

The representative seed is fixed to302011 before results. A recipe can advance
using that checkpoint only if BD-rate<=-10%, BD-accuracy>=0, everyQP Top1 gap
>=-0.5pp on development; existing gates are unchanged. A cohort effect or a
private notebook completing is not success. Disjoint802 and full1010 must then
pass with identical checkpoint/code bytes in fresh confirmation processes.
Historical use of the validation population is disclosed; current802 task
outputs remain excluded from tuning. Never change the split/QPs or acceptance
threshold to obtain a better rate. The complete manifest is
ops/autonomous_ar/cohorts/v15_rate_h265_20261002.json.

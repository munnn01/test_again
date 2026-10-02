# V16 audited development results

Recorded 2026-10-02T16:49:49+07:00 (Asia/Bangkok). Both V16 jobs completed, with independently recomputed metrics and source/config/checkpoint audits. All20 original evidence files were verified byte-for-byte against downloaded outputs and their SHA256 manifests. Neither candidate passes any of the three development gates. No confirmation802/full1010 run was launched.

Both use the same immutable H.264 source step500, seed302001 and exactly1000 fresh steps. The evaluator is fixed dev208, fingerprint f321b810af2c507c, with2080 paired rows, frozen held-out r2plus1d_18, real H.264 medium and QPs30/35/40/45/50. The registered additive_cond architecture, codec, source, training overrides and loss leaves match each final checkpoint. The final checkpoint bytes, optimizer LR, absent scheduler, reset old rate duals and task-dual cells were verified.

| Candidate | BD-rate (%) | BD-accuracy (pp) | Worst QP Top-1 gap (pp) | Development pass |
|---|---:|---:|---:|---|
| Historical n9 parent: pixel MSE10, feature weight0 | +2.6945 | -1.0680 | -0.9615 | no |
| n11 / A: pixel MSE0, feature weight0 | +6.3309 | -2.6381 | -2.8846 | no |
| n12 / B: pixel MSE0, raw-codec feature weight0.5 | +4.8004 | -1.8241 | -2.8846 | no |

Acceptance remains BD-rate<=-10%, BD-accuracy>=0, and every QP Top-1 gap>=-0.5pp. Positive BD-rate indicates more bitrate at comparable recognition performance in this curve comparison. The parent is the already audited matched V15 screen, not an additional V16 replicate.

| QP | n11 Top-1 gap (pp) | n12 Top-1 gap (pp) | n11 rate inflation (%) | n12 rate inflation (%) |
|---:|---:|---:|---:|---:|
| 30 | -1.9231 | -2.8846 | +5.8417 | +6.0737 |
| 35 | -0.9615 | +1.4423 | +5.7449 | +5.9756 |
| 40 | -0.4808 | -0.4808 | +4.2912 | +4.6826 |
| 45 | +0.0000 | +0.4808 | +3.3384 | +3.7601 |
| 50 | -2.8846 | -1.4423 | +3.1462 | +3.5722 |

## Interpretation bounded to this screen

Removing only pixel MSE increases observed BD-rate by +3.6364 percentage points versus the matched parent. Its same-QP bit inflation remains +3.15--5.84%, almost unchanged from the parent +3.18--5.89%, while held-out recognition worsens. This result does not support source-pixel MSE as the principal cause of inflated bitrate in this recipe. A constructed opposing gradient fixture was insufficient to predict real codec/task transfer.

Adding raw-codec feature fidelity partly improves over pixel release alone: BD-rate contrast -1.5305pp and BD-accuracy contrast +0.8141pp. It nevertheless increases every same-QP bitrate, remains +2.1059pp worse in BD-rate than the parent, and fails the QP30/50 task guard. Neither treatment establishes cheaper machine recognition.

This is one fixed-source/seed1000-step comparison on a repeatedly used development population. No seed confidence interval or universal loss-weight conclusion is claimed. Training protects batch-mean CE regret under sampled r3d_18/mc3_18 teachers; that surrogate does not guarantee the reported held-out backbone Top-1. Feature target reuse and gradients were tested in the common pipeline; actual OD features remain empty and this round supplies no OD performance evidence. The reserved802 complement has historical validation exposure but remains excluded from this search.

## Immutable evidence

- `v16-ar-pixel-release-h264-s302001`: candidate `49c1dd2b9c17d5555b758fac0083f581c720e73f`; final-last global_step1000 SHA `bb0351b3c2aa1ab6b0836a6e9272ee410e63bdb35fa2ae4dc58d466945d92da5`; local audit true. Exact decisions/curves/paired rows and audit manifests are in `results/autonomous_ar/v16-ar-pixel-release-h264-s302001/`.
- `v16-ar-semantic-anchor-h264-s302001`: candidate `fa84dbd0e7782dfaf384ac2f872b2915405d1caa`; final-last global_step1000 SHA `865387ec3d7c526f73d88ffe122660350162ef70563052f4f4c9734574094e48`; local audit true. Exact decisions/curves/paired rows and audit manifests are in `results/autonomous_ar/v16-ar-semantic-anchor-h264-s302001/`.

Preflight and launch hashes remain unchanged in ops/autonomous_ar/cohorts/v16_pixel_semantic_h264_20261002_preflight.json. Registered design: docs/RUN_DESIGN_autonomous_ar_v16.md. Source/config/checkpoint artifacts are preserved; no candidate was promoted and the -10% objective remains unmet. This manual check created no GPU job, goal or recurring automation.

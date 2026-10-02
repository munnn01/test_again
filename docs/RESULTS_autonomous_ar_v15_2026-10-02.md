# V15 audited development results

Recorded 2026-10-02T14:13:51.437659+07:00 (Asia/Bangkok). All eight V15 jobs completed and independently audited.
Every job fails development acceptance. H265 paired replication is complete:
all six registered runs are counted; no missing run or seed replacement.
No candidate advances to802/full1010 confirmation.

Each final checkpoint has1000 fresh steps, source/cfg/steps/SHA verified,
with2080 paired rows on dev208 (fingerprint f321b810af2c507c), held-out
r2plus1d_18, real codec medium, QPs30/35/40/45/50.

| Codec | Candidate | Seed | BD-rate (%) | BD-accuracy (pp) | Worst QP Top1 gap (pp) | Dev pass |
|---|---|---:|---:|---:|---:|---|
| h264 | rate_weight1 | 302001 | +2.2677 | -0.7641 | -0.4808 | no |
| h264 | positive-part primal, rate_weight0.25 | 302001 | +2.6945 | -1.0680 | -0.9615 | no |
| h265 | rate_weight0.25 | 302011 | +3.1113 | -1.0083 | -0.4808 | no |
| h265 | rate_weight1 | 302011 | +2.8697 | -0.6064 | -1.4423 | no |
| h265 | rate_weight0.25 | 302012 | +7.4565 | -3.5988 | -0.4808 | no |
| h265 | rate_weight1 | 302012 | +3.6851 | -2.5520 | -0.9615 | no |
| h265 | rate_weight0.25 | 302013 | +4.0560 | -1.1715 | +0.0000 | no |
| h265 | rate_weight1 | 302013 | +1.4139 | -1.1445 | -0.9615 | no |

H264 matched n3 control (seed302001, same source and1000-step protocol) had
BD-rate -1.7594%, BD-accuracy +0.3187pp, worstQP gap -1.4423pp. Weight1 improves
the worstQP guard but worsens both BD metrics. The positive-part primal reduces
bit inflation substantially but loses0.9615pp at QP45/50. These observations
are bounded to this fixed H264 recipe; no H264 candidate establishes -10%.

H265 locked paired contrasts, weight1 minus weight0.25:

| Seed/account | BD-rate delta (pp) | w1 QP30 gap (pp) |
|---|---:|---:|
| 302011/qktttttttttt | -0.2415 | -1.4423 |
| 302012/baoancut | -3.7714 | -0.9615 |
| 302013/trnhlng | -2.6420 | -0.9615 |

| H265 arm | Mean BD-rate (%) | Sample SD | Mean BD-accuracy (pp) |
|---|---:|---:|---:|
| w025 | +4.8746 | 2.2854 | -1.9262 |
| w1 | +2.6563 | 1.1506 | -1.4343 |

| Paired metric | Mean delta | Sample SD | 95% Student-t CI, df2 |
|---|---:|---:|---|
| bd_rate_pct | -2.2183 | 1.8027 | [-6.6965, +2.2598] |
| bd_accuracy_pp | +0.4919 | 0.5158 | [-0.7895, +1.7733] |
| worst_top1_gap_pp | -0.8013 | 0.2776 | [-1.4908, -0.1118] |

All three observed BD-rate contrasts favor weight1, but its paired95% CI
includes zero. All three treatment checkpoints also fail BD-accuracy and the
QP30 guard. The registered support criterion (BD CI upper<0 AND all treatment
task guards) is false. Its representative seed302011 treatment is +2.8697%
BD-rate, -0.6064pp BD-accuracy, -1.4423pp worstQP gap; it fails advancement.
A lower positive BD-rate is not the required10% reduction relative to raw codec.

The interval estimates variation across three fine-tuning seed/account blocks,
conditional on one fixed pretrained H265 source and the fixed dev208 population.
Accounts, clips and QPs are not extra seed replicates. With n3 the interval is
imprecise and assumes approximately normal seed contrasts. Validation population
has known historical use; this is development evidence. The df2 rate interval
was independently checked using the analytic t-distribution quantile.

Same-QP bitrate inflation remains present:

| Candidate | Range across QPs (%) |
|---|---:|
| v15-ar-rate-pressure-h264-s302001 | +14.05 to +26.28 |
| v15-ar-feasible-hinge-h264-s302001 | +3.18 to +5.89 |
| v15-rate-pair-h265-w025-s302011 | +6.19 to +34.21 |
| v15-rate-pair-h265-w1-s302011 | +5.13 to +29.24 |
| v15-rate-pair-h265-w025-s302012 | +6.11 to +34.17 |
| v15-rate-pair-h265-w1-s302012 | +4.94 to +29.00 |
| v15-rate-pair-h265-w025-s302013 | +6.11 to +34.52 |
| v15-rate-pair-h265-w1-s302013 | +5.04 to +29.17 |

Archive: results/autonomous_ar/<job>/ retains exact original remote decisions,
checkpoint/cfg/source manifest, point rows, curves and local recomputed audit.
evidence_manifest.json pins every curated byte. Scoring helpers and acceptance
are unchanged: BD-rate<=-10%, BD-accuracy>=0, everyQP Top1 gap>=-0.5pp, then
same checkpoint/code must pass disjoint802 and full1010. No current confirmation
task output was evaluated. No candidate is promoted; OD stays frozen in the
existing shared pipeline. No new goal, automation or scheduled check was created.

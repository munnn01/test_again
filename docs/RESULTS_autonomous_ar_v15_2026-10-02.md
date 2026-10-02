# V15 audited development results

Recorded 2026-10-02T13:09:07.866317+07:00 (Asia/Bangkok). Six completed jobs independently audited.
All six fail the preregistered development gates; none advances to confirmation.
BD-rate is measured against raw input plus the same real codec on dev208,
fingerprint f321b810af2c507c, r2plus1d_18, QPs30/35/40/45/50. Each final checkpoint
has1000 fresh steps. Every run has2080 paired per-clip score rows.

| Codec | Candidate | Seed | BD-rate (%) | BD-accuracy (pp) | Worst QP Top1 gap (pp) | Dev pass |
|---|---|---:|---:|---:|---:|---|
| h264 | rate_weight1 | 302001 | +2.2677 | -0.7641 | -0.4808 | no |
| h264 | positive-part primal, rate_weight0.25 | 302001 | +2.6945 | -1.0680 | -0.9615 | no |
| h265 | rate_weight0.25 | 302012 | +7.4565 | -3.5988 | -0.4808 | no |
| h265 | rate_weight1 | 302012 | +3.6851 | -2.5520 | -0.9615 | no |
| h265 | rate_weight0.25 | 302013 | +4.0560 | -1.1715 | +0.0000 | no |
| h265 | rate_weight1 | 302013 | +1.4139 | -1.1445 | -0.9615 | no |

H264 matched control n3 (seed302001, same source and1000-step protocol) had
BD-rate -1.7594%, BD-accuracy +0.3187pp, worstQP gap -1.4423pp. Increasing only
rate_weight0.25->1 improves its worstQP guard but worsens BD-rate and BD-accuracy.
The positive-part primal also fails both BD metrics and the high-QP task guard
(QP45 and50 gaps -0.9615pp). No H264 candidate establishes the -10% objective.

H265 per-block contrasts (descriptive; no three-pair aggregate yet):

| Seed/account | BD-rate delta w1-w025 (pp) | w1 task guards |
|---|---:|---|
| 302012/baoancut | -3.7714 | BD-accuracy<0 and QP30 gap -0.9615pp |
| 302013/trnhlng | -2.6420 | BD-accuracy<0 and QP30 gap -0.9615pp |

Both observed H265 contrasts favor the higher weight within their paired
blocks, but all four BD-rates remain positive. These are two observed pairs,
not evidence that the preregistered three-pair support criterion is met. The
representative seed302011 pair is now API-verified running on qktttttttttt after
its two H264 jobs completed; exact original payloads, source and seed retained.
All aggregate means, seed-variation estimates and95% CIs remain withheld until
every registered pair passes independent artifact audit. No seed is replaced.

Same-QP bitrate inflation remains present:

| Candidate | Range across QPs (%) |
|---|---:|
| v15-ar-rate-pressure-h264-s302001 | +14.05 to +26.28 |
| v15-ar-feasible-hinge-h264-s302001 | +3.18 to +5.89 |
| v15-rate-pair-h265-w025-s302012 | +6.11 to +34.17 |
| v15-rate-pair-h265-w1-s302012 | +4.94 to +29.00 |
| v15-rate-pair-h265-w025-s302013 | +6.11 to +34.52 |
| v15-rate-pair-h265-w1-s302013 | +5.04 to +29.17 |

Archive: results/autonomous_ar/<job>/ includes original remote decisions,
source/config/checkpoint manifest, paired points and locally recomputed audit.
evidence_manifest.json pins exact bytes of every curated file. Backend
floating-point differences are allowed only at1e-10 without changing gate flags.

Acceptance remains BD-rate<=-10%, BD-accuracy>=0, everyQP Top1 gap>=-0.5pp,
then identical checkpoint/code must pass disjoint802 and full1010 confirmation.
Current confirmation task outputs were not evaluated. Historical validation
population use remains disclosed. All training candidates stay on isolated
branches, with no M_best promotion and frozen OD. Existing shared pipeline is
retained. No new goal, recurring automation or scheduled check was created.

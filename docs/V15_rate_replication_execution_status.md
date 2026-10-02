# H265 rate-weight multi-account execution receipt

Recorded 2026-10-02T14:13:51.437659+07:00 (Asia/Bangkok). All six original private notebooks have
completed and passed independent local artifact audit. None passes development.
Three paired seed/account blocks are complete; no replacement or missing run.

| Account | Seed | w025 | w1 |
|---|---:|---|---|
| qktttttttttt | 302011 | [evaluated](https://www.kaggle.com/code/qktttttttttt/v15-rate-pair-h265-w025-s302011) | [evaluated](https://www.kaggle.com/code/qktttttttttt/v15-rate-pair-h265-w1-s302011) |
| baoancut | 302012 | [evaluated](https://www.kaggle.com/code/baoancut/v15-rate-pair-h265-w025-s302012) | [evaluated](https://www.kaggle.com/code/baoancut/v15-rate-pair-h265-w1-s302012) |
| trnhlng | 302013 | [evaluated](https://www.kaggle.com/code/trnhlng/v15-rate-pair-h265-w025-s302013) | [evaluated](https://www.kaggle.com/code/trnhlng/v15-rate-pair-h265-w1-s302013) |

Original SHA-sealed payloads are retained. Explicit human approval resolved the
earlier additional-account egress hold; its history is preserved. The qk pair
ran after its two H264 jobs completed. No token/data/checkpoint uploads.

H265 means: w025 BD-rate +4.8746%; w1 +2.6563%.
Paired BD-rate delta -2.2183pp,95% CI[-6.6965,+2.2598]pp.
Registered support criterion fails; all w1 QP30 guards fail. Representative
seed302011 does not advance. No confirmation or new GPU run was created in this
result check. Full findings: docs/RESULTS_autonomous_ar_v15_2026-10-02.md.

No training/scoring code changed during collection. The preceding60 CPU checks
remain applicable. Artifacts/source/cfg/steps/SHA and every point-level gate were
independently audited for all eight V15 jobs. No M_best promotion.
No new goal, recurring automation or scheduled check was created.

# V18 audited development results

Recorded 2026-10-02T22:39:33.125269+07:00 (Asia/Bangkok). Both private notebooks completed and passed the existing independent local source/config/checkpoint/metric audit. All20 curated original files were verified against raw downloads and SHA256 manifests. Neither passes all development gates; no802/full1010 confirmation or production promotion occurred.

Matched original source step500, seed302001,1000 fresh steps, final-LAST only, five balanced QPs and fixed208 development clips (f321b810af2c507c). Both have2080 paired rows; all1040 raw-codec rows are identical to each other and historical V17 B. Canonical scorer and real H264 medium settings remain unchanged.

| Candidate | rate_weight | BD-rate (%) | BD-accuracy (pp) | Worst QP Top1 gap (pp) | Gates |
|---|---:|---:|---:|---:|---|
| Historical V17 B |0.25|-1.7738|+0.7703|+0.4808|rate fail; task pass|
|n15 / A|0.5|-3.1228|+1.5453|+0.9615|rate fail; task pass|
|n16 / B|1.0|+2.4482|-0.8211|-1.4423|all three fail|

Acceptance remains finite BD-rate<=-10%, BD-accuracy>=0, every QP Top1 gap>=-0.5pp.

| QP | A Top1 gap (pp) | B Top1 gap (pp) | A same-QP rate increase (%) | B same-QP rate increase (%) |
|---:|---:|---:|---:|---:|
|30|+0.9615|-1.4423|+10.1026|+8.0518|
|35|+4.8077|+2.8846|+10.7201|+8.5864|
|40|+4.8077|+1.4423|+9.7627|+7.6864|
|45|+9.1346|+2.8846|+8.1960|+6.5535|
|50|+2.8846|+2.8846|+6.1514|+5.1913|

## Interpretation

Doubling calibrated rate pressure to0.5 improves observed BD-rate by 1.3490pp relative to the preselected historical control and preserves both task guards. It still falls short of -10% by 6.8772pp. Negative BD-rate does not imply same-QP bit savings; actual same-QP overhead is reported above.

Increasing the coefficient to1 gives positive BD-rate, negative BD-accuracy and a QP30 gap below the allowed -0.5pp. Stronger rate pressure is not a monotonic remedy in this recipe. Keep0.5 as the preselected control for any newly registered intervention, without treating its development result as confirmed performance.

These are single-source/seed screens on a repeatedly used development population. There is no seed interval, H265 transfer or joint OD/AR claim. Training mean CE regret remains a surrogate and can allow example cancellation. The802 complement was excluded from this search but has historical validation exposure. No checkpoint passed the gate or was promoted.

## Evidence

- `v18-ar-calibrated-rate05-h264-s302001`: final-last step1000 SHA256 `0034f4890dfecd09b2427d8a390e94812ae0110ef98eb9e3c978cefb993ab07b`, independent audit passed; evidence in `results/autonomous_ar/v18-ar-calibrated-rate05-h264-s302001/`.
- `v18-ar-calibrated-rate1-h264-s302001`: final-last step1000 SHA256 `ba073648ef417cf3d1ea100cca4b4f54c40a741391902f1f642de237a0399494`, independent audit passed; evidence in `results/autonomous_ar/v18-ar-calibrated-rate1-h264-s302001/`.

Design: `docs/RUN_DESIGN_autonomous_ar_v18.md`. Receipts: `docs/V18_execution_status.md`. Exact contrasts, rate overhead and file integrity: `ops/autonomous_ar/cohorts/v18_calibrated_pressure_h264_20261002_audit.json`. Raw training logs and checkpoints remain in ignored downloads. No GitHub push, goal, heartbeat or scheduled check was created.

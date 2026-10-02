# V17 audited development results

Recorded 2026-10-02T18:37:16.030067+07:00 (Asia/Bangkok). Both V17 notebooks completed. The coordinator re-ran the unchanged local source/config/checkpoint/metric audit and verified all20 original curated files against raw downloads and SHA256 manifests. Direct remote status checks returned COMPLETE for both. Neither reaches the registered BD-rate<=-10% gate; no confirmation802/full1010 was launched.

Both use the same immutable H264 source step500, seed302001 and exactly1000 fresh steps. Final-last checkpoints match the shared code commit0e35c0d2be0fdf377ff6e7cb92582833f4542e58 and each registered config. The paired real H264 medium evaluation uses208 development clips, fingerprint f321b810af2c507c, all five QPs and frozen held-out r2plus1d_18. All1040 raw-codec rows are byte-value identical across A, B and the audited n9 parent; each comparison has2080 paired rows. Optimizer LR, absent scheduler, reset old rate duals, source/config/payload identity, checkpoint bytes and task-dual cells passed the existing checks.

| Candidate | BD-rate (%) | BD-accuracy (pp) | Worst QP Top1 gap (pp) | Rate gate | Task gates |
|---|---:|---:|---:|---|---|
| Historical n9 Gaussian parent | +2.6945 | -1.0680 | -0.9615 | fail | fail |
| n13 / A: log coefficient cost | -0.0600 | +0.3714 | +0.4808 | fail | pass |
| n14 / B: log cost + detached calibration | -1.7738 | +0.7703 | +0.4808 | fail | pass |

Acceptance remains finite BD-rate<=-10%, BD-accuracy>=0 and every QP Top1 gap>=-0.5pp. Both V17 arms pass the two held-out task gates but fail the rate gate. Negative BD-rate is the interpolated aggregate curve comparison; it is not same-QP bit reduction.

| QP | A Top1 gap (pp) | B Top1 gap (pp) | A same-QP rate increase (%) | B same-QP rate increase (%) |
|---:|---:|---:|---:|---:|
| 30 | +0.4808 | +0.4808 | +9.0347 | +12.4024 |
| 35 | +5.2885 | +4.3269 | +9.7509 | +13.1692 |
| 40 | +3.3654 | +6.2500 | +8.7801 | +11.9811 |
| 45 | +4.3269 | +6.2500 | +7.5763 | +9.8553 |
| 50 | +0.4808 | +5.2885 | +5.8138 | +7.2671 |

## Interpretation bounded to this screen

Replacing the Gaussian-power cost by the unscaled log coefficient cost improves observed BD-rate by 2.7545pp versus the matched historical parent, and all QP Top1 gaps become positive. Same-QP bit overhead grows from the parent's3.18-5.89% to A's5.81-9.75%. This is a task-quality improvement with essentially zero measured curve-rate saving (-0.0600%), not the requested10% reduction.

Adding only detached scalar calibration improves BD-rate by 1.7138pp and BD-accuracy by 0.3989pp versus A. B improves Top1 at QPs40/45/50, ties A at QP30 and loses0.9615pp relative to A at QP35; it remains above the raw anchor at every QP. It also increases bitrate relative to A at every QP, with7.27-13.17% overhead over raw. The resulting -1.7738% BD-rate falls short of -10% by8.2262 percentage points.

The rounded training progress diagnostics show calibration scale 0.09542-0.6099, always below1 in the recorded displays, and no denominator-floor activation. Thus B attenuates the local rate derivative in this run, consistent with increased bit expenditure. These2000 progress records include duplicate displays, not2000 independent training steps. Scalar rate-value fitting has not established a universally faithful gradient direction.

This is one fixed-source/seed1000-step screen on a repeatedly used development population; no seed interval, H265 transfer or joint OD/AR performance claim follows. Batch-mean training CE regret is a surrogate and passing held-out Top1 does not prove every training regret was nonpositive. The positive task results support further bounded investigation of the proxy, while the complete gate still rejects both candidates. The802 complement was excluded from the search but has historical validation exposure. No immutable checkpoint advanced and no production code was promoted.

## Reproducible evidence

- `v17-ar-log-rate-h264-s302001`: final-last step1000 SHA256 `116feee2a3b89b7996a2a7a6e0f4a3f1841018df72eca87a35f03c24f99a87df`; canonical audit true; exact curves, paired rows and decisions in `results/autonomous_ar/v17-ar-log-rate-h264-s302001/`.
- `v17-ar-calibrated-log-rate-h264-s302001`: final-last step1000 SHA256 `fb749882ad58c363851747b882e4259c7bd8c62fcdaf2e2a7e725f4f64684aa6`; canonical audit true; exact curves, paired rows and decisions in `results/autonomous_ar/v17-ar-calibrated-log-rate-h264-s302001/`.

Registered design: `docs/RUN_DESIGN_autonomous_ar_v17.md`. Execution receipts: `docs/V17_execution_status.md`. Independent audit/contrasts/diagnostic scope: `ops/autonomous_ar/cohorts/v17_log_calibration_h264_20261002_audit.json`. Original training logs/checkpoints remain in ignored `outputs/autonomous_ar/downloads/`; evidence files preserve their original bytes. No new GPU experiment, scheduled check or goal was created by this result check.

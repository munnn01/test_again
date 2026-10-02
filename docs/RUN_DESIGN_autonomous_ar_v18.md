# Registered V18 calibrated-rate pressure screen

Registered2026-10-02 before GPU submission. The objective remains at least one learned-AR codec with finite BD-rate<=-10%, BD-accuracy>=0 and every QP Top1 gap>=-0.5pp, followed by confirmation of the same immutable checkpoint. No acceptance threshold, evaluation population or codec setting is relaxed.

## Evidence and fixed hypotheses

Audited V17 B, log coefficient cost plus detached real/proxy rate scaling, achieves H264 BD-rate-1.77382249214324%, BD-accuracy+0.7702797011812017pp and worst QP gap+0.48076923076922906pp. It passes both task gates but is8.2262pp short of the requested rate threshold. Same-QP bit overhead is7.27-13.17%; logged calibration scales0.09542-0.6099 attenuate the proxy rate derivative and the fixed denominator floor never activates in recorded displays. These are rounded duplicate progress records, not an independent-sample distribution. Evidence: docs/RESULTS_autonomous_ar_v17_2026-10-02.md.

V18 tests whether stronger primal rate pressure on that task-safe recipe removes costly edits without losing the task guard. V15 rate_weight1 failed in the signed Gaussian recipe, so this is not a guarantee of monotonic gains: positive-part regret, coefficient-local cost and calibrated scale now define a different fixed recipe. Failures must be reported rather than compensated by threshold or evaluator changes.

| Node / arm | Config | Only parsed changes from V17 B | Fixed hypothesis |
|---|---|---|---|
| n15 / A | configs/v18_calibrated_rate05_ar.yaml | all three task-regret rate_weight leaves0.25->0.5 | Twice the rate pressure improves H264 rate/task efficiency while fast init10 task duals retain task guards. |
| n16 / B | configs/v18_calibrated_rate1_ar.yaml | all three task-regret rate_weight leaves0.25->1 | Fourfold rate pressure improves H264 efficiency enough for the -10% gate while retaining task guards. |

The three leaves are loss.task_regret_constraint.rate_weight, per_codec.h264.rate_weight and per_codec.h265.rate_weight. Updating explicit codec overrides is necessary for the effective weight to change. Both configs retain every other parsed value, including out_dir; the wrapper assigns unique actual paths. The unused H265 override is kept consistent, without claiming a H265 result. Both arms use one combined candidate commit/payload, with no Python source changes from V17 common commit0e35c0d2be0fdf377ff6e7cb92582833f4542e58. Historical V17 B is a preselected control, not a new replicate or a resumed checkpoint. No in-run tuning, seed selection or epoch selection is permitted.

## Immutable source, training and evaluation

- Source: qktttttttttt/crc-v5-h264-v12-source-s282001, preprocessor_last.pth at source step500, SHA256 ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a. Start from this original source, not V17's trained output; load model-only and reset optimizer/scheduler/duals.
- Existing additive_cond cond_dim3 and shared OD/AR engine; sampled frozen r3d_18/mc3_18 training teachers and held-out r2plus1d_18 evaluator. No architecture/task adapter/data change. OD stays frozen; this round screens AR only.
- Seed302001, exactly1000 fresh steps, final-last global_step1000 only. Batch2, LR1e-5, balanced QPs30/35/40/45/50,200 visits each, no scheduler, patience or checkpoint selection. Training diagnostic validation is development-only.
- Real H264 medium in the existing STE; log_coeff rate, ste_rate_calibration=true, original quantization, yuv420, inter prediction and no-motion settings. Reconstruction/RNG and one real treatment encode per call are unchanged. The raw task anchor still uses its existing separate encode; calibration adds none.
- Pixel MSE10, DCT kappa10, omega0; task-regret epsilon0/init10/cap100/dual_lr0.02/positive_part; signed mean regret updates each codec/QP dual. This surrogate can permit per-example cancellation and does not guarantee held-out Top1.
- Existing8632 training clips; paired raw/treatment real H264 medium evaluation on dev208 fingerprint f321b810af2c507c, shard0/5 salt v13-task-regret-screen-v1; all five QPs,2080 paired rows. Preserve scorer, curve integration and evaluator. Compare prep against raw using the same codec, never substitute a cross-codec comparison.
- All gates must pass: finite BD-rate<=-10%, BD-accuracy>=0, each QP Top1 gap>=-0.5pp. Only a locally audited, development-passing immutable checkpoint may proceed to reserved802 fingerprint740e9571c679071a and full1010 with the same gates. The802 is excluded from this search but has historical validation exposure; it is not a fresh untouched holdout. No production promotion without confirmation.

## Readiness, execution and integrity

No training source is edited: the143 combined V17 CPU tests and actual-proxy/STE five-QP mocked-real check remain applicable. Do not rerun that suite or add tests mirroring the two configs. Instead verify exact parsed deltas, both resolved per-codec weights through the common parser, unchanged source hashes, payload/notebook/metadata hashes, all Python/embedded-block/Bash syntax, canonical scorer bytes and fixed attachments. Pin both configs on one commit with identical payload before submission. Require a positive API push acknowledgment and directly check remote status. Terminal results must pass the existing local source/config/checkpoint/paired-ID/QP/metric audit. Config readiness supplies no measured dev_score.

Use exactly two private free-GPU qktttttttttt notebooks: v18-ar-calibrated-rate05-h264-s302001 and v18-ar-calibrated-rate1-h264-s302001, under the user's earlier explicit permission for subsequent optimization rounds and continued objective instruction. The four previously approved V15 packages are the full baoancut/trnhlng scope; no new package there. Upload sealed repo code/config/tests/design only, with existing Kaggle dataset/source mounts; no token, PDF, data or new checkpoint upload. No GitHub push is authorized or performed; the configured origin was verified as https://github.com/munnn01/pre_updated_v2.git.

No goal API, heartbeat, schedule or automatic result collection is created. Root production training remains unpromoted; exploration stays on isolated local branches. Scores stay null until audited real evaluation. Receipts/readiness belong in docs/V18_execution_status.md. The existing Arbor cycle budget12, depth2 and branching5 are unchanged; this is cycle6.

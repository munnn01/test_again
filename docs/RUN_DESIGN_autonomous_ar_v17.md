# Registered V17 logarithmic rate / calibrated-rate screen

Registered 2026-10-02 before GPU submission. Context and primary sources: `docs/LITERATURE_REVIEW_v17_rate_proxy.md`, with versioned provenance in `docs/V17_literature_sources.json`.

## Fixed comparison

Parent n9 is the audited positive-part recipe at commit60168b918bc0eab76ebe54ad4589ab6dd18e8e43: H264 development BD-rate+2.694502502825813%, BD-accuracy-1.0680087396741276pp, worst QP gap-0.96153846153846pp. V16 pixel release and semantic anchoring both failed all development gates; pixel MSE stays10 and feature weight stays0.

| Node | Config | Parsed delta from n9 | Hypothesis |
|---|---|---|---|
| n13 / A | configs/v17_log_rate_ar.yaml | add codec.rate_model=log_coeff | An elementwise logarithmic coefficient cost improves actual rate/task efficiency relative to Gaussian-power cost at the matched source/seed/budget. |
| n14 / B | configs/v17_calibrated_log_rate_ar.yaml | add codec.rate_model=log_coeff and codec.ste_rate_calibration=true | On the same log proxy, detached real/proxy scaling improves actual rate/task efficiency compared with A. |

Both packages use the same integrated candidate commit, containing both optional mechanisms and both configs. A/B differs only by the calibration config key. The root production training code is not promoted. No best epoch, best seed, strength sweep or additional parameter adjustment is allowed inside these registered screens.

`VirtualCodec.rate_model` defaults to the exact old `gaussian_power` arithmetic. Opted-in `log_coeff` costs every normalized DCT coefficient `log2(1+abs(coeff/step))`, with original bpp units, motion/prediction/reconstruction/chroma and RNG. Default STE rate arithmetic remains exact. Opted-in calibration scales only its backward rate derivative by fully detached `real_bpp / proxy_bpp.clamp_min(1e-8)`; the real-forward value is preserved, as is reconstruction backward routing. Diagnostics record real/proxy bpp, multiplier and floor activation. There is one real encode per call and no ratio cap. Strict boolean/mode validation and finite/nonnegative checks prevent silent option errors. The calibration adapts a per-image JPEG construction to batch-mean video bpp, not an exact codec-gradient reproduction.

Signed batch-mean regret G=CE(prep decoded)-CE(raw decoded)-epsilon still updates the task dual independently for each codec/QP; only the n9 primal uses max(G,0). epsilon0, task dual init10/cap100/LR0.02 and rate weight0.25 are unchanged. This mean constraint permits per-example cancellation and does not guarantee held-out Top-1.

## Immutable protocol and acceptance

- Existing H264 source qktttttttttt/crc-v5-h264-v12-source-s282001, preprocessor_last.pth at source step500, SHA256 ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a. Load model-only; fresh optimizer, scheduler and dual state.
- Existing additive_cond model, cond_dim3. Frozen sampled r3d_18/mc3_18 training teachers; held-out r2plus1d_18 evaluation. No architecture or task adapter change; shared engine for OD/AR.
- Seed302001; exactly1000 fresh steps; final-last global_step1000 only. LR1e-5, batch2, balanced QPs30/35/40/45/50 with200 visits each, no cosine, patience or checkpoint selection.
- Pixel MSE loss.mu10, omega0, DCT kappa10, task primal positive_part and rate_weight0.25. All other parsed fields equal n9. Real H264 medium STE; original transform steps, colorspace, prediction and no-motion settings. The wrapper assigns unique output paths while parsed out_dir stays the parent value.
- Existing Kinetics data and hash split:8632 training; development208 fingerprint f321b810af2c507c, shard0/5 salt v13-task-regret-screen-v1. Training diagnostic validation is development-only and cannot select a checkpoint.
- Real-codec paired raw/treatment evaluation over all five QPs on those208 IDs. Preserve evaluator, curve integration and scorer exactly; score only locally audited terminal evidence.
- Every gate must pass: finite BD-rate<=-10%, BD-accuracy>=0, and every QP Top-1 gap>=-0.5pp. CPU tests and calibration diagnostics supply readiness/mechanism evidence, not a dev_score.
- Only an immutable passing checkpoint may proceed to reserved802 fingerprint740e9571c679071a, then full1010 reporting under the same gates. The802 is excluded from this search but has historical validation exposure; it is not a fresh untouched holdout. No promotion without confirmation.

H264 is selected to match the audited n9 parent. No general H265 or joint OD claim follows from this round. If development fails, do not run confirmation or replace thresholds.

## Execution and integrity

Run exactly two private free-GPU qktttttttttt notebooks: v17-ar-log-rate-h264-s302001 and v17-ar-calibrated-log-rate-h264-s302001. The user's earlier explicit permission for subsequent qktttttttttt optimization rounds, later development resume and literature-improvement instruction authorize this scoped batch. The four V15-only baoancut/trnhlng packages do not authorize V17 on those accounts. Upload sealed repo code/config/tests/design docs only; no PDF, token, dataset or new checkpoint upload. Mount existing Kaggle sources.

Pin a shared candidate commit, payload/notebook/metadata hashes and source SHA. Before submission, inspect exact parsed config deltas, Python/embedded scripts/Bash syntax, unchanged canonical scorer bytes, payload file hashes and private GPU attachments. Require a positive API push acknowledgment and verify remote status. Results must pass the existing independent source/config/checkpoint/paired-ID/QP/metric audit.

The hypothesis tree retains earlier completed leaves. Its branching limit3->5 is recorded as a structural policy amendment before dispatch; objective, evaluator, depth2 and12-cycle budget stay fixed. Nodes n13/n14 have null scores until real-codec evaluation. No goal, heartbeat, scheduled checks or automatic result collection is created. Receipts and readiness are recorded in docs/V17_execution_status.md.

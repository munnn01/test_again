# V16: release pixels while protecting task semantics

Registered 2026-10-02 before V16 GPU submission or development feedback. This round continues the existing learned-AR experiment tree and shared training pipeline. It does not revise the V13 results or thresholds.

## Evidence and bounded hypotheses

All eight V15 screens completed with local source/config/checkpoint/metric audits and zero development passes. The three H.265 rate-weight pairs gave mean BD-rate +4.8746% at weight0.25 and +2.6563% at weight1. The paired difference was -2.2183 percentage points, with t95 CI [-6.6965,+2.2598]; every weight1 run failed the QP30 task guard. More rate pressure alone has not established the requested tradeoff.

The matched H.264 positive-part screen n9 (60168b918bc0eab76ebe54ad4589ab6dd18e8e43) gave BD-rate +2.6945%, BD-accuracy -1.0680pp and worst QP Top-1 gap -0.9615pp. Its same-QP bit inflation was +3.18--5.89%. This recipe is the common parent for two controlled interventions. Pixel MSE may oppose removal of task-irrelevant detail; its contribution or dominance in the failed run has not been established. Source fidelity removal can also harm recognition, and real-codec forwards do not establish surrogate gradient alignment.

## Fixed interventions

| Node / private qktttttttttt job | Parsed loss changes from n9 | Falsifiable question |
|---|---|---|
| n11 / v16-ar-pixel-release-h264-s302001 | mu10->0 only | Does releasing source pixel MSE improve the fixed rate-task tradeoff? |
| n12 / v16-ar-semantic-anchor-h264-s302001 | mu10->0; omega0->0.5; distill_anchor=raw_codec | Does normalized frozen-teacher feature matching to raw plus the same codec/QP preserve semantics better than pixel release alone? |

Here loss.mu is the pixel MSE coefficient, separate from task dual mu[c,q]. The feature weight0.5 is selected before results and is not adjusted within this round. A versus historical n9 isolates pixel MSE removal; B versus A isolates the registered semantic-anchor treatment. Both use the positive-part primal after the unchanged signed dual update. Training still constrains batch-mean CE regret, which permits per-example cancellation and does not guarantee Top-1 on the held-out task backbone.

For n12, raw-codec feature anchoring is an opt-in setting in preprocessing_loss and the common engine. Training reuses the detached raw reconstruction already produced for task regret; it does not perform a second raw encode. One frozen training teacher is pinned across raw CE, treatment CE and feature matching, then released even on failure. The feature target is detached, the edited reconstruction retains its gradient, and normalized feature MSE uses the existing feature API. Default source anchoring stays compatible. An opted-in raw anchor with positive feature weight fails clearly if the analyzer exposes no features.

OD uses the same generic pipeline, but the current OD adapter returns an empty feature list. This round does not add OD features or claim joint OD/AR validation. OD defaults and frozen results stay unchanged.

## Immutable training and evaluation protocol

- Source: qktttttttttt/crc-v5-h264-v12-source-s282001, preprocessor_last.pth at source step500; SHA256 ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a.
- Model: existing additive_cond, cond_dim3; frozen sampled r3d_18/mc3_18 training teachers, unchanged task and architecture. The old source is loaded model-only with fresh optimizer and dual state.
- Seed302001; exactly1000 fresh steps; LR1e-5; batch2; balanced QPs30/35/40/45/50; no cosine schedule or checkpoint selection. Score only final-last at step1000.
- Task epsilon0, dual initialization10, cap100, LR0.02, positive_part primal; rate weight0.25; DCT kappa10; all other coefficients/proxy/real codec settings identical to n9. Codec is STE with real H.264 medium. Parsed out_dir stays the parent value; wrapper assigns distinct actual output paths.
- Training diagnostic validation is limited to the registered development IDs. Validation loss uses the existing codec routing; it cannot select a checkpoint.
- Development: paired raw/treatment real H.264 medium QPs30/35/40/45/50 on the fixed208 clips, fingerprint f321b810af2c507c, shard0/5 salt v13-task-regret-screen-v1, frozen held-out r2plus1d_18. Use the existing aggregate curve evaluator and scorer without modification.
- All gates must pass: finite BD-rate <=-10%, BD-accuracy>=0, every QP Top-1 gap>=-0.5pp. No best-seed selection or threshold/split changes.
- Only a development-passing immutable checkpoint may be confirmed on the reserved802 complement, fingerprint740e9571c679071a, and then full1010 reporting with the same gates. The802 is excluded from this search but has historical validation exposure; it is not a newly collected untouched holdout. No promotion without confirmation.

## Execution and integrity

The two source/config packages run on private free-GPU qktttttttttt notebooks under the user's earlier explicit authorization for subsequent optimization rounds and latest resume instruction. Prior permissions for four V15 packages on baoancut/trnhlng do not authorize new V16 packages there. Only sealed repo code/configuration and accompanying tests/design docs are uploaded; existing Kaggle data/checkpoints are mounted. Tokens, data and new checkpoints are not uploaded.

Before submission, pin each candidate commit and payload/notebook/metadata hashes, check the exact parsed config differences, Python and embedded script syntax, attachment/privacy/GPU metadata, and the source SHA. Require a positive API push acknowledgment and verify remote status. Terminal results undergo the existing independent local audit before a development score is recorded. CPU tests and toy gradients establish readiness only.

No recurring checks, heartbeat or persistent goal is created. No V16 BD-rate result exists at registration. Detailed readiness and receipts belong in docs/V16_execution_status.md; candidate-specific preregistrations stay on their isolated branches.

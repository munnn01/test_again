# V16 execution status

Latest result audit: 2026-10-02T16:49:49+07:00 (Asia/Bangkok).

Both private free-GPU H.264 notebooks completed and passed independent local integrity/metric audits. Neither passes development acceptance. No candidate advances to802/full1010 confirmation; the -10% objective remains unmet.

| Node | Intervention | BD-rate (%) | BD-accuracy (pp) | Worst QP gap (pp) | Remote state |
|---|---|---:|---:|---:|---|
| n11 | Pixel MSE mu10->0 only | +6.3309 | -2.6381 | -2.8846 | [completed / locally audited](https://www.kaggle.com/code/qktttttttttt/v16-ar-pixel-release-h264-s302001) |
| n12 | Pixel MSE0 plus raw-codec feature weight0.5 | +4.8004 | -1.8241 | -2.8846 | [completed / locally audited](https://www.kaggle.com/code/qktttttttttt/v16-ar-semantic-anchor-h264-s302001) |

Full results: [RESULTS_autonomous_ar_v16_2026-10-02.md](RESULTS_autonomous_ar_v16_2026-10-02.md). Original evidence is archived under results/autonomous_ar for both jobs.
## Readiness and locked controls

n11:28 parent hinge/task-regret/CRC CPU tests passed. n12:44 focused CPU tests and the existing loss self-check passed; coordinator independently reran all16 substantive new semantic-anchor tests. These verify gradients, detached targets, same codec/QP raw reuse, coherent pinned teacher, exception cleanup, source-default compatibility and video/image feature tensor support. The actual OD feature adapter remains empty; no joint OD/AR performance claim is made.

Each payload passed Python AST for every Python file, syntax checks for four embedded Python blocks and Bash, exact parsed-config differences, file/payload/notebook/metadata SHA checks, private GPU metadata and mounted-source checks. The scorer bytes match the committed root and latest V15 runs exactly (including their preserved CRLF bytes). No scoring helper was changed.

Both jobs load the same audited source step500, SHA `ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`, and use seed302001, fresh optimizer/duals and exactly1000 new steps. Balanced QPs30/35/40/45/50, additive_cond cond_dim3, real H.264 medium, rate weight0.25, task epsilon0/init10/cap100/LR0.02/positive-part, and DCT kappa10 are fixed. Score only final-last with the unchanged paired dev208 r2plus1d_18 evaluator.

## Sealed packages

- `v16-ar-pixel-release-h264-s302001`: payload `68cc0a14e9bc2d1b02a1f684b2579e6dfe2961bc519ee031805ecd594cb222f9`; notebook `4792bf0651e83ac48d6eb319b71dc15eced2bb0a035a02513f17afffe754f6dd`; metadata `a8553c5261dd45c03f8230aec9d243465965b82c6c1305a17e8633daa3041239`.
- `v16-ar-semantic-anchor-h264-s302001`: payload `4bc2e59a548b08e91905b0a5d913605acfaee354602055ee212f2842e9ee1a79`; notebook `b21f8481ede99299a321ed5dc308170aefb141156050ae2fb6a43dd3497eee6b`; metadata `56c82dcbd7d04efb82ec016a07e3a069a2efc56b364a36bb03e4b4245510bc4a`.

## Authorization and next evidence

Current jobs use only private qktttttttttt under the user-approved subsequent optimization workflow and current resume. The four previously authorized V15 packages on baoancut/trnhlng remain the complete extra-account scope. Only code/configuration plus accompanying docs/tests were uploaded; no token/data/new checkpoint was uploaded. No goal, heartbeat or recurring check was created.

Both terminal runs passed the existing local audit of source/config/checkpoint/208 paired IDs/QPs/metrics and were entered in the tree as executed. Neither passes development acceptance. Advancement requires finite BD-rate<=-10%, BD-accuracy>=0 and every QP Top-1 gap>=-0.5pp. Only a passing immutable checkpoint proceeds to reserved802 and full1010 under the same gates. Exploration remains on isolated branches; no training change has been promoted.

Preregistration: `docs/RUN_DESIGN_autonomous_ar_v16.md`. Sealed preflight: `ops/autonomous_ar/cohorts/v16_pixel_semantic_h264_20261002_preflight.json`. Candidate-specific source/config/design/tests remain in `outputs/autonomous_ar/worktrees/pixel-release` and `outputs/autonomous_ar/worktrees/semantic-anchor`.

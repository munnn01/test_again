# V18 execution status - 2026-10-02

Updated 2026-10-02T20:42:56.516973+07:00 (Asia/Bangkok). Both private free-GPU notebooks have positive version1 API push acknowledgments and manually verified RUNNING status. No V18 BD-rate result is available. No GitHub push, confirmation or promotion occurred.

| Arm | Notebook | Effective rate_weight | Push acknowledged (UTC) | Remote status |
|---|---|---:|---|---|
| n15 / A | [v18-ar-calibrated-rate05-h264-s302001](https://www.kaggle.com/code/qktttttttttt/v18-ar-calibrated-rate05-h264-s302001) | 0.5 | 2026-10-02T13:41:31.324486+00:00 | running at 2026-10-02T13:42:51.080835+00:00 |
| n16 / B | [v18-ar-calibrated-rate1-h264-s302001](https://www.kaggle.com/code/qktttttttttt/v18-ar-calibrated-rate1-h264-s302001) | 1.0 | 2026-10-02T13:41:50.925331+00:00 | running at 2026-10-02T13:42:56.049628+00:00 |

## Registered change and readiness

Both configs use common commit64f6dc9495f08ad82863b7ab28e6bb75465284ef on local branch arbor/v18-common-rate, worktree `outputs/autonomous_ar/worktrees/v18-common-rate`. They combine config/design contributionsc26342fe41de51990edb62d633a58880e97895c5 and08d328edaf3c9f79aad5ecbd3e3aaa805d5e2091. Python source and tests match V17 common commit0e35c0d2be0fdf377ff6e7cb92582833f4542e58 exactly. The existing143-test V17 suite remains applicable and was not repeated for these config changes.

Each candidate changes only the three global/explicit H264/H265 task-regret rate_weight leaves from0.25 to its registered0.5 or1. The shared engine parser resolves the correct effective coefficient for both codecs. All other parsed leaves, including source/model/out_dir, are unchanged. Both jobs screen H264; updating the unused H265 override is consistency, not a H265 result.

Preflight passed every payload/file/notebook/metadata SHA, identical A/B payload, unchanged source bytes, exact parsed deltas, effective common-engine overrides, all Python and four embedded-block ASTs, Bash syntax, canonical scorer hashes, private GPU metadata and fixed existing source/data mounts. This supplies readiness only, not measured performance.

Historical fixed control V17 B has BD-rate-1.7738%, BD-accuracy+0.7703pp, worst QP gap+0.4808pp and7.27-13.17% same-QP bit overhead. Its logged calibration scales were below1. V18 tests whether stronger rate pressure reduces that overhead while the task dual retains recognition; past signed-Gaussian rate-weight failures do not establish success here.

Both begin from the original H264 source step500 SHA ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a, not V17 output. Seed302001,1000 fresh steps, final-LAST only, fresh optimizer/duals, batch2/LR1e-5, balanced QPs30/35/40/45/50, additive_cond cond_dim3, log_coeff plus detached calibration, real H264medium, MSE10/DCT10/omega0, task epsilon0/init10/cap100/dual_lr0.02/positive-part and fixed dev208 held-out r2plus1d_18 are unchanged.

## Sealed receipts

Identical payload SHA256: `b433b4c467f819cd5b5f85cc9735f4dc89fff6defbdadad6289f2ca66c7883c0`.
- `v18-ar-calibrated-rate05-h264-s302001`: notebook `bcecdd43089b411fb417392801f9cb3d60fce6e3f4566704d5a099cd453d9ad5`; metadata `c08013ce98c0a2558d6bbfc0e4365e9cef4fa4af587841707ab30a7d7669f8f4`; positive push acknowledgment true.
- `v18-ar-calibrated-rate1-h264-s302001`: notebook `7bceb855c34fe21c33e739687607647ed69847437e996189f19294a86ce874dd`; metadata `cd95496db2e529857e8360ab49e23eb983d964feb84966cd903a07585cf0634f`; positive push acknowledgment true.

## Gates and permission scope

Acceptance stays finite BD-rate<=-10%, BD-accuracy>=0 and every QP Top1 gap>=-0.5pp. Terminal source/config/checkpoint/paired-ID/QP/metric evidence must pass independent local audit. Only a passing immutable checkpoint may confirm on reserved802, then full1010 under the same gates. The802 has historical validation exposure but is excluded from this search. No threshold, task backbone, split or encoder setting was changed.

Only the earlier authorized subsequent private qktttttttttt free-GPU code/config workflow is used. The four sealed V15 packages remain the full baoancut/trnhlng scope. No token/PDF/data/new checkpoint is uploaded. No GitHub push is performed; origin remains https://github.com/munnn01/pre_updated_v2.git, and new commits stay local. No app goal, heartbeat, schedule or automatic result collection was created.

Design: `docs/RUN_DESIGN_autonomous_ar_v18.md`. Preflight: `ops/autonomous_ar/cohorts/v18_calibrated_pressure_h264_20261002_preflight.json`. Readiness: `ops/autonomous_ar/cohorts/v18_calibrated_pressure_h264_20261002_readiness.json`. Current audited control: `docs/RESULTS_autonomous_ar_v17_2026-10-02.md`.

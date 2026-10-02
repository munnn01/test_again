# V17 execution status - 2026-10-02

Updated 2026-10-02T18:37:16.030067+07:00 (Asia/Bangkok). Both notebooks completed and passed the independent local audit. A BD-rate-0.0600%; B-1.7738%; both pass task gates and fail the -10% rate gate. No confirmation or promotion occurred. Full report: `docs/RESULTS_autonomous_ar_v17_2026-10-02.md`.

| Arm | Notebook | Parsed config delta from n9 | Push acknowledged at (UTC) | Remote status |
|---|---|---|---|---|
| A / n13 | [v17-ar-log-rate-h264-s302001](https://www.kaggle.com/code/qktttttttttt/v17-ar-log-rate-h264-s302001) | add codec.rate_model=log_coeff | 2026-10-02T10:24:27.019571+00:00 | complete; local audit passed |
| B / n14 | [v17-ar-calibrated-log-rate-h264-s302001](https://www.kaggle.com/code/qktttttttttt/v17-ar-calibrated-log-rate-h264-s302001) | add codec.rate_model=log_coeff; codec.ste_rate_calibration=true | 2026-10-02T10:24:48.335297+00:00 | complete; local audit passed |

## Shared implementation and readiness

Both arms use commit `0e35c0d2be0fdf377ff6e7cb92582833f4542e58` on `arbor/v17-common-rate`, isolated at `outputs/autonomous_ar/worktrees/common-rate`. This combines log-rate contribution44a43b72ab12a83a2aace3e918472bae52b39d09 and calibration contributionbcc056e6a1bc2268e6c8d3819d65e41dc18419fc. Both optional mechanisms are in identical code; the selected config alone enables calibration for B. The production training branch remains unpromoted.

The integrated suite passed143 CPU tests covering coefficient values/derivatives and bpp geometry, Gaussian default parity, quantization/RNG/reconstruction preservation, detached scalar calibration, dtype/numeric/floor failures, common factory/logging, teacher pinning, signed task-dual behavior and parent motion/entropy regressions. The only warning is the existing cast_ar.py invalid escape sequence.

An independent check used the actual common factory, VirtualCodec and STE, with a mocked real codec, for all five QPs. Both arms preserved reconstruction and its derivative/RNG, and B rate gradients equaled A gradients multiplied by the detached diagnostic scalar. Twenty calls performed exactly twenty real-fixture encodes. Omitted/default options matched the actual immutable n9 module source for reconstruction, bpp, both gradients and RNG. Mock rates are wiring evidence only, not empirical codec measurements. Detailed readiness: `ops/autonomous_ar/cohorts/v17_log_calibration_h264_20261002_readiness.json`.

Each package passed file/payload/notebook/metadata hashes, all Python ASTs, four embedded Python blocks, Bash syntax, exact parsed config deltas, pinned canonical scorer bytes, private GPU metadata and existing dataset/source mounts. Both packages contain18 repo code/config/design/test/helper files; no PDF, token, data or checkpoint upload. Preflight: `ops/autonomous_ar/cohorts/v17_log_calibration_h264_20261002_preflight.json`.

## Sealed package receipts

Common payload SHA256: `765475198e8b796091cd607704f2c9be7db42d4330252e9e3a779170648c066f`.
- `v17-ar-log-rate-h264-s302001`: notebook `b41725e322aa04a278511d6847605c77fead3cd4b61cc1f98feef3caec247b50`; metadata `0928c773bb7b2b4443eebf3269453eee7c3b38ea087ac4a5b26fd50d4ea13917`; API acknowledgment true.
- `v17-ar-calibrated-log-rate-h264-s302001`: notebook `a1427aa2b9249186f6c646f03b88e4d1943b42115a77464984d2c88e592f252a`; metadata `d2e6b7c6ffde0ad0969a181908bbdf3fb9afa0e3f61ec71a809a3780fb433c0f`; API acknowledgment true.

## Protocol and evidence required

The source remains step500 SHA ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a. Both use seed302001,1000 fresh steps, batch2/LR1e-5, balanced QPs30/35/40/45/50, additive_cond cond_dim3, real H264 medium, pixel MSE10, DCT10, task-regret positive-part with epsilon0/init10/cap100/LR0.02 and rate_weight0.25. Final-last only; same development208 IDs, held-out r2plus1d_18, evaluator and scorer.

Acceptance still requires finite BD-rate<=-10%, BD-accuracy>=0 and every QP Top1 gap>=-0.5pp. Terminal evidence passed the existing independent source/config/checkpoint/paired-ID/QP/metric audit. Neither arm passed all development gates; no reserved802 or full1010 evaluation was launched. A development-passing immutable checkpoint alone may proceed to reserved802 and full1010 under the same gates. The802 complement has historical validation exposure but remains excluded from this search. Neither CPU tests nor a running job supplies acceptance evidence.

Previous V16 screens are complete and failed: pixel release BD-rate+6.3309%, semantic anchor+4.8004%; neither advanced. Literature triage/review prompted the current proxy interventions while retaining the original MSE. See `docs/RESULTS_autonomous_ar_v16_2026-10-02.md`.

Only the previously authorized private qktttttttttt free-GPU optimization workflow is used. The four V15 packages are the complete baoancut/trnhlng scope; no V17 jobs were submitted there. No goal, heartbeat, scheduled check or automatic result collection was created.

Registered design: `docs/RUN_DESIGN_autonomous_ar_v17.md`. Literature review: `docs/LITERATURE_REVIEW_v17_rate_proxy.md`. Original PDF/version/reading-scope provenance: `docs/V17_literature_sources.json`.

To inspect the candidate, use the common worktree above. To collect terminal results manually through the unchanged runner:

```powershell
& 'C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/Scripts/python.exe' -B ops/autonomous_ar/runner.py poll --job v17-ar-log-rate-h264-s302001 --download
& 'C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/Scripts/python.exe' -B ops/autonomous_ar/runner.py poll --job v17-ar-calibrated-log-rate-h264-s302001 --download
```

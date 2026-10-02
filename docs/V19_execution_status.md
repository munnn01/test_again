# V19 execution status

Updated 2026-10-02T23:05:16.003587+07:00 (Asia/Bangkok). Both private free-GPU notebooks have positive API push acknowledgments. The statuses below were checked manually this turn. V19 performance scores are pending; the current audited control V18A has BD-rate-3.1228% with task guards and fails the -10% target.

| Arm | Notebook | Residual filter sigma | API acknowledgment (UTC) | Remote status |
|---|---|---:|---|---|
|n18/A|[v19-ar-residual-filter05-h264-s302001](https://www.kaggle.com/code/qktttttttttt/v19-ar-residual-filter05-h264-s302001)|0.5|2026-10-02T16:04:22.552501+00:00|running at 2026-10-02T16:05:04.036483+00:00|
|n19/B|[v19-ar-residual-filter1-h264-s302001](https://www.kaggle.com/code/qktttttttttt/v19-ar-residual-filter1-h264-s302001)|1.0|2026-10-02T16:04:43.208185+00:00|running at 2026-10-02T16:05:10.522851+00:00|

## Implemented change and verification

One optional fixed spatial Gaussian filters only the learned RGB residual before source-plus-edit and the existing clamp. Radiusceil(3sigma), replicate borders, per-frame/per-channel separable filters; no temporal averaging or new learned weights. Default sigma0 preserves historical forward/backward/RNG and checkpoint keys. Shared train/evaluation factory and checkpoint config restoration honor the scalar for OD/AR; no joint task performance result follows.

Common immutable source commit `a96e0e93e32d8d6c5e6d5c179086c90287c0b8db` is on `arbor/v19-common-filter` in `outputs/autonomous_ar/worktrees/v19-common-filter`. Both configs add only their sigma field to the preselected V18 rate0.5 config; all other parsed fields remain unchanged. Both have identical payload `4ca64646b1ded6c0b87f0ecfc0d017e597b6cfc88f9ab7e9e13ab5ec04c04272`.

200 relevant CPU tests passed. An independent probe strictly loaded audited V18A trained weights at sigma0/0.5/1, confirmed identical 10403 parameters/state keys and exact historical default outputs on three single-frame/temporal/rectangular geometries with unchanged RNG. These are compatibility checks, not V19 task/rate results. Local CPU timing covers only preprocessing; no local GPU or real-codec runtime was measured.

Payload preflight verified code/config/file/notebook/metadata hashes, one-field config deltas, unchanged codec/loss/task settings, shared source, all Python/embedded blocks and Bash syntax, canonical scoring-helper bytes, exact existing source/data mounts and private GPU metadata. Original MPEG documents, tokens, data and checkpoints are absent from the submitted source package.

Original H264 source step500 SHA ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a; model-only load/reset, seed302001,1000 fresh steps and final-LAST. Both keep real H264medium, log_coeff/calibration, rate0.5, task epsilon0/positive-part/dual init10/cap100/LR0.02, MSE10/DCT10, held-out r2plus1d_18 and fixed dev208 f321b810af2c507c with five QPs.

## Acceptance and scope

Acceptance remains finite BD-rate<=-10%, BD-accuracy>=0 and every QP Top1 gap>=-0.5pp. Terminal evidence must pass the existing independent audit. Only a passing immutable checkpoint may advance to reserved802 and full1010 under the same gates. No scorer/grid/split/encoder change, confirmation or production promotion occurred.

This is a bounded adaptation informed by the older public MPEG CDTR, not a full MPEG CTC run or conformance result. ISO publication metadata was verified separately; paid final full text was not accessed. Protocol differences are in the alignment note.

The existing human permission for subsequent private qktttttttttt free-GPU optimization rounds covers these jobs. The four previously approved V15 packages remain the full baoancut/trnhlng scope. All commits stay local; no GitHub push, goal, heartbeat, schedule or automatic result collection was created.

Design: `docs/RUN_DESIGN_autonomous_ar_v19.md`. Sources/limits: `docs/MPEG_VCM_alignment_20261002.md`. Implementation/tests/CPU timing: `docs/METHOD_v19_residual_filter.md`. Preflight: `ops/autonomous_ar/cohorts/v19_residual_filter_h264_20261002_preflight.json`. Readiness: `ops/autonomous_ar/cohorts/v19_residual_filter_h264_20261002_readiness.json`. Latest audited results: `docs/RESULTS_autonomous_ar_v18_2026-10-02.md`.

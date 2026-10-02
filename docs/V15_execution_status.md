# V15 execution status

2026-10-02: user resumed improvement. No recurring automation or new goal created.

| Node | Candidate | Commit | GPU status |
|---|---|---|---|
| n8 | v15-ar-rate-pressure-h264-s302001 | 4460a9f56243d1485c195d884e9c0e03414d90c1 | running |
| n9 | v15-ar-feasible-hinge-h264-s302001 | 60168b918bc0eab76ebe54ad4589ab6dd18e8e43 | running |

n8: exact three rate_weight leaves0.25->1 implement one intervention; actual
codec overrides and sealed ZIP/Python/Bash/private metadata preflight passed.

n9: positive-part primal is optional in shared _fit. All28 CPU checks passed,
including rate descent under feasible regret, task protection under violation,
independent signed codec/QP duals, raw-anchor detachment, teacher release after
failure and exact default signed behavior. Only one parsed config field differs.

Both private Kaggle submissions succeeded. Their immutable payloads are:

- n8: `0a00104eb0ceda87655cc75773d5bdf5c09bad8d46de42549d5c9fd7c02b273d`; [notebook](https://www.kaggle.com/code/qktttttttttt/v15-ar-rate-pressure-h264-s302001).
- n9: `aea3f6ac6c93eb654a6c2b31142ae3d8a4239987263c4a6ca40eaa3e964f1797`; [notebook](https://www.kaggle.com/code/qktttttttttt/v15-ar-feasible-hinge-h264-s302001).

No GPU BD score is available at this recorded status. Candidate changes remain
in isolated branches; none is promoted to M_best. Existing train.py/evaluate.py
and immutable V14 scoring protocol are reused. Acceptance remains BD-rate<=-10%,
BD-accuracy>=0, everyQP Top1 gap>=-0.5pp, followed by unchanged confirmation802
and full1010. Status updates live in ops/autonomous_ar/state.json.

Historical validation use remains disclosed. The 802 confirmation clips are
excluded from current tuning and no confirmation task output has been evaluated.

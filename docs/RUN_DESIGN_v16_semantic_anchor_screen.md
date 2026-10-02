# V16 n12: semantic fidelity to a same-codec raw anchor

Registered on 2026-10-02 before this candidate has GPU results. This node
extends the common OD/AR/tracking preprocessing pipeline with an optional
feature target. It does not create a new task-specific engine or architecture.
Original V13 code, registered design and archived interpretation remain intact.

## Fixed hypothesis and treatment

Replace reconstruction pixel MSE to the uncompressed source with semantic
feature fidelity to the raw clip decoded by the same real codec and QP. That
anchor represents the cues already available to the baseline machine. The
hypothesis is that frozen normalized semantic features can protect those cues
while permitting cheaper pixel changes than source pixel MSE. Neither local
feature-loss improvement nor a training-teacher CE reduction establishes a
held-out recognition or BD-rate improvement.

The candidate is based on n9 commit
60168b918bc0eab76ebe54ad4589ab6dd18e8e43. Its configuration
configs/v16_semantic_anchor_ar.yaml has exactly three parsed differences
from configs/v15_feasible_hinge_ar.yaml:

- loss.mu: 10 -> 0 (remove source pixel reconstruction MSE);
- loss.omega: 0 -> 0.5 (the existing feature-distillation default weight,
  selected here before this screen); and
- loss.distill_anchor: raw_codec (new optional target selection).

Comparator n11 removes pixel MSE with omega=0 and no added feature
regularizer. Thus n12 versus n11 tests the added semantic target at this
fixed weight. The registered treatment has no weight sweep within this node.

For the pinned frozen training teacher F and raw decoded anchor a[c,q]:

```text
D_feat = mean_over_layers( MSE(F_l(x_hat), stopgrad(F_l(a[c,q])))
                          / (mean(stopgrad(F_l(a[c,q]))^2) + 1e-6) )
L = 0.25 * (R_pre / R_raw - 1)
    + mu_task[c,q] * relu(meanCE_pre - meanCE_raw - epsilon[c,q])
    + 0.5 * D_feat + 10 * existing_DCT
```

The codec/QP dual update still uses the signed mean CE violation before
primal rectification. The features, CE anchor and treatment use the same
pinned teacher, released even on a loss exception. Feature targets are
detached and calculated without a target graph. The existing raw real-codec
anchor is reused; task-regret and feature fidelity do not encode it twice.
Pixel distortion, temporal consistency, edit penalties and every other
source-based loss retain their original source meanings. The optional
feature_anchor argument affects only feature_distillation.

Default loss.distill_anchor=source reproduces the existing source-feature
formula and allows empty features exactly as before. Unsupported modes or
raw_codec with a non-STE codec fail before a training forward. Weighted
raw_codec feature distillation requires matching nonempty feature maps and
fails clearly instead of silently disabling the new treatment. The current
OD analyzer returns an empty feature list, so OD cannot opt into this weighted
recipe until its own feature adapter and validation exist. Generic 4D detector
feature tensors and 5D video feature tensors share the same tested loss API;
that API compatibility is not evidence of a jointly validated OD recipe.

The inherited validation docstring called its path proxy-only, but the n9
_fit actually supplies its actual codec to _val_loss. This candidate preserves
that default routing. Opt-in raw_codec validation requires the actual STE,
uses a same-real-codec/QP raw target and pins one teacher for CE and both
feature arms. Diagnostic validation still cannot choose the screened checkpoint.

## Controls, source and budget

- One additive_cond preprocessor, cond_dim=3; no architecture changes.
- Frozen sampled teachers r3d_18 and mc3_18.
- H.264 source checkpoint step500, SHA-256
  ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a.
- Fresh model-only finetune, optimizer/scheduler/duals reset; seed302001,
  exactly 1000 new steps, one epoch, batch2, LR1e-5.
- QPs30/35/40/45/50, balanced sampling, all codec/proxy settings unchanged.
- Positive-part task primal, epsilon=0, mu_init=10, mu_max=100,
  dual_lr=0.02, rate_weight=0.25 in both codec overrides; DCT kappa=10.
- Preserve every other parsed field, including inherited out_dir.
  The coordinator overrides each submitted job with a unique output path.
- Screen initial real H.264 medium using final-last global_step1000.

## Locked evaluation and gates

Evaluate only development208 clips, fingerprint f321b810af2c507c, existing
salt v13-task-regret-screen-v1, shard0/5. Training diagnostic validation
uses the coordinator's development index. Frozen held-out r2plus1d_18,
real same-codec raw pairing, aggregate Top1 rate-quality curves and
QP30/35/40/45/50 remain unchanged.

Advance only if all hold: BD-rate <= -10%, BD-accuracy >= 0, and every
same-QP Top1 gap >= -0.5 percentage point. Non-finite metrics fail. No
reserved802 or full1010 task outputs are used to optimize this node. A
passing screen still requires independent coordinator evaluation of the
identical checkpoint on reserved802 and full1010 with the same gates.
The validation set has historical V12 use; reserved802 is excluded from
this current search rather than a newly collected untouched population.

## Status

Implementation is complete. The focused CPU suite passed 44 tests: 16
substantive new semantic-anchor cases plus all 28 parent hinge/regret/CRC
regressions. The existing tests/test_losses.py standalone self-check also
passed. In a finite fixture, changing a source feature target from 2 to a
decoded target 4 reverses the summed reconstruction gradient from +0.25 to
-0.0625, while the raw target and frozen teacher receive no gradients.
Both 5D video features and 4D image features use the same loss API. Shared
_fit integration confirms one raw codec/QP encode serves both CE regret
and feature fidelity; validation retains actual STE and a coherent teacher.
Unknown/non-STE modes, empty weighted features and feature exceptions fail
clearly, with pinned teachers released. Source-feature defaults, OD-like
empty defaults, source pixel MSE and temporal meanings remain verified.

The first CPU run's fixture expected no source gradient allocation; the
unchanged zero-weight temporal term allocates an all-zero source gradient
for multi-frame input. The corrected assertion accepts zero source
contribution while retaining the strict absent raw-anchor gradient check.
No implementation change was needed. The cast_ar.py invalid-escape warning
is pre-existing and unrelated. Exact configuration leaves are verified by
coordinator preflight rather than a duplicate reversible-config test.

```powershell
python -B -m pytest tests/test_v16_semantic_anchor.py tests/test_v15_feasible_hinge.py tests/test_v13_task_regret.py tests/test_crc_v5.py -q -p no:cacheprovider --basetemp C:/Users/Admin/AppData/Local/Temp/vcm-v13-tools/pytest-v16-semantic-20261002-r3
python -B tests/test_losses.py
```

This executor submitted no GPU job and dev_score remains null. Finite CPU
fixtures establish implementation wiring, not real codec transfer or new
AR/OD performance. The actual OD adapter still has empty features and has
no jointly validated semantic-fidelity recipe.

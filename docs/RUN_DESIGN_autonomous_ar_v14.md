# Autonomous learned-AR optimization after V13

Registered 2026-10-01 before V14 GPU results. User authorization: continue
self-improvement until at least one of H.264/H.265 reaches BD-rate <= -10%.
Task protection remains BD-accuracy >=0 and every QP Top-1 gap >= -0.5 pp.
No claim of guaranteed attainment; every score must come from real artifacts.

V13 remains closed as a negative result. New interventions reuse the existing
preprocessor, common engine/loss and real codec STE. No new AR training loop.
OD is frozen; this search does not claim a jointly validated OD/AR model.

## Fixed partitions and acceptance

The canonical index remains 8632 train / 1010 validation / 1159 historical test.
Development is the exact 208 V13 validation IDs (salt v13-task-regret-screen-v1,
fingerprint f321b810af2c507c). Confirmation is the disjoint 802 validation IDs;
SHA256(newline-joined sorted keys) =
740e9571c679071af6884574fccea6cab5539ab8d1be8beb245babda9ac5ce24.
Training diagnostic validation is restricted to the 208 development IDs.
Neither confirmation nor historical test clips participate in training.

Search ranks development outcomes only. A candidate advances to confirmation
only if it already satisfies all three criteria on development. Confirmation
uses unchanged checkpoint bytes, algorithm, QP settings and code payload;
no tuning follows from individual confirmation QP/clip outputs. It also reports
the full 1010 validation curve. The automated success gate requires both
confirmation802 and full1010 to meet all three criteria for one codec.

These are validation clips used in prior research, and the historical test has
also been inspected in V1/V2. They are reserved from this autonomous search,
not a new population or untouched publication holdout. Report that limitation.

## Initial hypotheses and allocation

| Node | Intervention | Codec/account | Fresh seed | Steps |
|---|---|---|---:|---:|
| n3 | mu_init10, mu_max100, dual_lr.02, rate_weight.25 | H264/qktttttttttt | 302001 | 1000 |
| n4 | positive per-example CE + correct-anchor margin/KL safeguards | H264/qktttttttttt | 302002 | 1000 |

Both begin from audited private H264 seed282001 last checkpoint at500 steps,
SHA256 ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a.
They restart optimizer and task duals. No V13 final weights are used.
Architectural and non-dual regularizer settings are inherited. loss.lam_task1
only restores scalar CE validation diagnostics: active constrained training
overrides that coefficient with its codec/QP task dual.

Each intervention has an isolated Git branch and preregistration document.
A 1000-step development result is a screening observation; failure or a partial
improvement is recorded and used to form the next hypothesis. No candidate
is promoted to final best based only on development or on a undefined BD fit.

Only qktttttttttt and baoancut accounts are currently allocated. H265's public
500-step source remains available for later fixed hypotheses. No paid GPU,
public publication, deployment, or external messaging is authorized by this run.

## Immutable execution and continuation

ops/autonomous_ar/runner.py packages a named candidate commit as a hashed
source overlay on baseline d39927e92298c111d498f0e1a9f72069469ed0b9, then uses
existing train.py/evaluate.py. Source checkpoint SHA is checked before and after
execution; final checkpoint steps/codec/dual state are checked. Per-clip pairing,
index membership, metric units and all five QP curves are audited.

The durable hypothesis tree is docs/autonomous_ar_20261001/.arbor/.
The job registry is ops/autonomous_ar/state.json. It stores no tokens.
Full raw downloads are under ignored outputs/autonomous_ar/downloads/.
Heartbeat t-t-i-u-learned-ar-t-i-bd-rate-10 resumes this chat every30 minutes;
keep quiet for unchanged running state and notify on meaningful changes only.
Stop that heartbeat after proven goal completion or an explicit user stop.

Do not silently mark the goal complete on budget exhaustion, a development
win, or a failed screen. Extend coordinator cycles when the current batch
budget is used and the authorized search can still make meaningful progress.

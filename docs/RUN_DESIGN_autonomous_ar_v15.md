# V15: separate rate pressure from confidence reward

Registered 2026-10-02 before any V15 GPU result. The user resumed improvement
("tiếp tục cải tiến") after explicitly authorizing private Kaggle packages and
subsequent optimization rounds. Work continues on repo111 under the same scope:
private qktttttttttt notebooks, free GPU, source/configuration upload only, no
new data/checkpoint/token upload. The recurring heartbeat remains deleted; no
new goal or scheduled automatic check is created.

## Evidence and bounded hypotheses

All V14 screens used real H264, dev208 and1000 fresh steps. n3 had BD-rate
-1.7594%, BD-accuracy +0.3187pp and QP30 gap -1.4423pp. n6 motion passed the task
guards but reduced BD-rate only0.6846%. n7 mu_init3 reduced same-QP bit overhead,
but BD-rate +2.1869% and BD-accuracy -0.4226pp. No candidate passed development;
confirmation task outputs were never evaluated.

Signed mean regret G=CE(prep+codec)-CE(raw+codec)-epsilon enters the n3 primal
as mu[c,q]*G. When G<0 and mu>0 this term rewards further confidence gains.
It can trade additional bits for reductions of CE within the feasible budget.
This is an incentive in the implemented loss, not a proven sole cause of all
rate overhead. V15 tests two independent interventions from n3:

| Node | Candidate | Only conceptual change from n3 |
|---|---|---|
| n8 | v15_rate_pressure_ar | rate_weight0.25->1, including both codec overrides |
| n9 | v15_feasible_hinge_ar | primal task penalty signed G -> max(G,0) |

n8 retains the signed primal task penalty. n9 retains rate_weight0.25. The n9
dual still updates by signed G with the original independent codec/QP cells,
clipped to[0,100]; diagnostic task_regret/task_violation remain signed. Batch
mean cancellation between examples remains possible and is disclosed. Defaults
stay signed in the shared _fit engine, so existing OD/AR recipes retain their
semantics. No separate AR loop or architecture is introduced. OD remains frozen;
a jointly validated OD/AR model is not claimed.

## Locked comparison and acceptance

Both candidates keep seed302001, source H264 step500 SHA256
`ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`,
1000 fresh steps, same optimizer/LR1e-5, mu_init10/mu_max100/dual_lr0.02/epsilon0,
model additive_cond with cond_dim3, stationary yuv420 STE proxy, real H264 medium,
QPs30/35/40/45/50, pinned sampled r3d_18/mc3_18 teachers. Adaptive spatial DCT
coefficient kappa10 and reconstruction-to-source MSE coefficient loss.mu10
remain unchanged; loss.mu is distinct from task_duals. The final LAST checkpoint
is evaluated, with no intermediate validation selection.

The scorer remains the immutable autonomous-learned-ar-v14 protocol. V15 names
the new training recipes, not a changed evaluator. Development IDs remain the
208 V13 IDs, fingerprint f321b810af2c507c; held-out analyzer r2plus1d_18.

A candidate advances only if dev BD-rate<=-10%, BD-accuracy>=0 and every QP
Top1 gap>=-0.5pp. Then a fresh process evaluates unchanged checkpoint bytes/code
on the disjoint802 and full1010 validation clips. Both must pass to establish
success. No relaxation of the gates, cherry-picked QP curves or reinterpretation
of same-QP accuracy gain as equal-bitrate gain is permitted.

802 clips are reserved from the current search but belonged to validation used
in prior research. They are not a new untouched population. Report this limit.

Each candidate is isolated, registered before GPU submission, committed and
packaged as a SHA256-sealed overlay on d39927e92298c111d498f0e1a9f72069469ed0b9.
Existing train.py/evaluate.py are reused. Coordinator alone owns registry and
GPU submission; executor CPU tests establish readiness, not an AR/BD score.

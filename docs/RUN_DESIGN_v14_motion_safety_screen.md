# V14 n6: motion-compensated proxy with matched task dual safety

Registered 2026-10-01 before this candidate has GPU/AR/BD results.
Node n3 remains unchanged and runs separately. Node n6 has one fixed
hypothesis and is **pending coordinator GPU allocation**; this executor
has not submitted a job or evaluated dataset clips.

## Fixed hypothesis and rationale

With n3's stronger task duals held fixed, replacing the stationary
P-frame proxy predictor by the existing bounded motion-compensated
predictor will reduce pressure to erase translating action cues and
improve the real-codec rate/recognition tradeoff.

`STECodec` uses actual x264 reconstruction and coded rate in its forward
pass; derivatives come from `VirtualCodec`. The default proxy predicts
P-frames from stationary reconstructed reference planes. Its Gaussian
DCT-power surrogate therefore counts frame motion as residual energy.
The existing motion path selects integer displacements under `no_grad`
and gathers reference pixels with gradients; it changes the surrogate
geometry, while the real encoder remains the same.

V13 H.264 training logs show delayed semantic correction: first500-step
mean regret was -0.3039 with mean rate excess +17.54%; steps1000--1999
had regret +2.5536 and rate excess -24.17%; steps2000--4315 reached
+4.1177 regret and -59.60% rate excess. These are rounded training-log
means, not dev performance. They motivate n3's safety mechanism; n6
asks whether a better temporal surrogate helps once that safety is held
fixed. They do not establish motion compensation as a successful remedy.

## Artifact and matching

- Parent commit: `4e8f8b5ef68b7952d02bead532c27fd54bf8fa0f` on
  `arbor/v14-dual-safety`; isolated branch `arbor/v14-motion-safety`.
- New configuration: `configs/v14_motion_safety_ar.yaml`, inherited
  exactly from `configs/v14_dual_safety_ar.yaml` except three added keys:
  `codec.motion=true`, `codec.motion_range=2`, `codec.motion_block=16`.
  Radius2 gives25 displacement candidates per reference plane rather
  than289 at the existing default radius8, bounding runtime in advance.
- Model, teachers, sampling, STE real codec/preset, proxy quantizer,
  strength, regularizers, optimizer, QP mapping and duals are unchanged.
  In particular: epsilon0, mu_init10, mu_max100, dual_lr0.02,
  rate_weight0.25, DCT kappa10 and source-distortion mu10.
- H.264 only in the initial screen. Source is the identical audited
  private seed282001 500-step last checkpoint, SHA-256
  `ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a`.
- Training seed302001 intentionally matches n3, with fresh optimizer
  and duals, one epoch capped at exactly1000 steps. Balanced five-QP
  sampling gives200 visits per cell. Never reuse V13 final weights.
- Quantizer anneal follows total_steps in the existing engine: both
  matched1000-step candidates ramp toward hard rounding over1000 steps.
  Comparing either to V13's12948-step schedule is exploratory.
- Static loss.lam_task1 retains n3's CE proxy monitor; active constrained
  training replaces it by each cell's dual. Static beta0 is unchanged.
- The YAML's out_dir is literally inherited from n3 to keep only the
  three codec additions. Before execution, the coordinator runner must
  override it to a unique job directory, e.g.
  `outputs/v14_motion_safety_ar`. Running the YAML directly against n3's
  output directory is outside this design. Training diagnostic val is
  restricted by the coordinator's generated index to exact dev208 IDs.
- Evaluate the final1000-step checkpoint, never a proxy-selected best.
  No model, engine, codec source or training loop is changed here.

## Development and confirmation gates

Development is exactly V13 shard0/5 with salt
`v13-task-regret-screen-v1`:208 IDs, fingerprint `f321b810af2c507c`.
Evaluate frozen `r2plus1d_18` using real H.264 QPs30/35/40/45/50 and
same-codec raw anchors. Compute primary BD-rate from the aggregate
Top-1 rate-quality curves, not averaged per-sequence BD-rate.

Advance only if all three development conditions hold:

1. BD-rate <= -10%;
2. BD-accuracy >=0; and
3. every QP Top-1 gap >= -0.5 percentage point.

Non-finite BD metrics fail. A development pass is exploratory. The
coordinator alone may then evaluate unchanged checkpoint bytes/code
on the disjoint802 validation complement and full1010; the autonomous
success gate requires all three conditions on both confirmation802 and
full1010 for one codec. No tuning follows individual confirmation
clip/QP outputs, and this executor must not inspect them.

Historical-use disclosure: these1010 validation clips were used in
prior research. The historical1159 test clips were also inspected in
V1/V2. Confirmation802 is excluded from this autonomous search's
training diagnostics and development feedback; it is not a new
population or a pristine publication holdout. No claim of an untouched
historical evaluation is made. No clips from confirmation or historical
test enter training. OD remains frozen; this AR screen does not validate
a shared OD/AR checkpoint.

## Local synthetic runtime profile

This checks only the existing `VirtualCodec` CPU forward/backward,
without an analyzer, standard encoder, training data or model update.
It is **not AR performance, real bitrate, BD-rate, a candidate score,
or a GPU-time prediction**. Both modes use the identical fixed fixture,
seed302001, quality3, training mode and anneal1. Shape is
[B2,C3,T16,H128,W128], CPU intra-op threads2 and inter-op threads1;
local PyTorch2.14.1+cpu. Each mode is measured3 times; no external warmup.

Fixture definition:

```python
torch.manual_seed(302001)
base = torch.rand(2, 3, 128, 128) * 0.5 + 0.25
base = torch.nn.functional.avg_pool2d(base, 3, stride=1, padding=1)
fixture = torch.stack([
    torch.roll(base, shifts=(t // 2, t), dims=(-2, -1))
    for t in range(16)
], dim=2)
```

Each repetition resets the seed, clones the fixture with input gradients,
then calls the existing proxy at quality3 and backpropagates
`x_hat.square().mean() + bpp`. Proxy settings match the YAML; only
motion is switched off/on. Timing uses `time.perf_counter` around each
forward/backward. The texture wraps at boundaries because of `torch.roll`.
This fixture is a reproducible workload, not a realism claim.

| Motion | Repetition | Forward seconds | Backward seconds | Total seconds |
|---|---:|---:|---:|---:|
| Off | 1 | 0.073788 | 0.088971 | 0.162759 |
| Off | 2 | 0.056950 | 0.101413 | 0.158363 |
| Off | 3 | 0.105101 | 0.131399 | 0.236500 |
| On, radius2/block16 | 1 | 0.355028 | 0.291316 | 0.646344 |
| On, radius2/block16 | 2 | 0.343454 | 0.345501 | 0.688955 |
| On, radius2/block16 | 3 | 0.470205 | 0.258359 | 0.728565 |

All reconstructions, surrogate bpp values and input gradients were
finite in all6 executions. Gradient L2 norms were0.0173500 off and
0.00844574 on. Median total time:0.162759s off,0.688955s on, ratio4.233.
CPU thread scheduling and other local activity can vary these timings;
GPU end-to-end cost includes real encodes and frozen teachers and is
unmeasured. The check supports local executability only.

## Status and hypothesis discipline

Configuration smoke passed: exactly the three added codec fields,
every other parsed n3 field matched, seed302001/max_steps1000, five
valid H.264 task-dual cells, and existing VirtualCodec construction
with motion=true/radius2/block16. No new test or training loop was added.
GPU allocation, dispatch, result auditing and persistent tree updates
belong to the coordinator. No AR/BD dev score exists yet. The candidate
remains pending GPU allocation and its hypothesis stays fixed on failure.

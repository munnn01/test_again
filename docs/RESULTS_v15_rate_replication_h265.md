# Rate-weight development cohort: v15-rate-replication-h265-20261002

H.265; source `a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e`; 1000 fresh steps per run.

Audited jobs: 0/6. Complete: false. Goal complete: false.

| Account | Seed | Arm | Job | Status | Counted | BD-rate (%) | BD-accuracy (pp) | Worst-QP Top-1 gap (pp) |
|---|---:|---|---|---|---|---:|---:|---:|
| qktttttttttt | 302011 | w025 | v15-rate-pair-h265-w025-s302011 | submit_failed | no | -- | -- | -- |
| qktttttttttt | 302011 | w1 | v15-rate-pair-h265-w1-s302011 | submit_failed | no | -- | -- | -- |
| baoancut | 302012 | w025 | v15-rate-pair-h265-w025-s302012 | running | no | -- | -- | -- |
| baoancut | 302012 | w1 | v15-rate-pair-h265-w1-s302012 | running | no | -- | -- | -- |
| trnhlng | 302013 | w025 | v15-rate-pair-h265-w025-s302013 | running | no | -- | -- | -- |
| trnhlng | 302013 | w1 | v15-rate-pair-h265-w1-s302013 | running | no | -- | -- | -- |

Aggregates withheld until all six registered jobs have valid audited development evidence.

- v15-rate-pair-h265-w025-s302011: job is submit_failed; terminal audited evaluation required; local audit has not passed.
- v15-rate-pair-h265-w1-s302011: job is submit_failed; terminal audited evaluation required; local audit has not passed.
- v15-rate-pair-h265-w025-s302012: job is running; terminal audited evaluation required; local audit has not passed.
- v15-rate-pair-h265-w1-s302012: job is running; terminal audited evaluation required; local audit has not passed.
- v15-rate-pair-h265-w025-s302013: job is running; terminal audited evaluation required; local audit has not passed.
- v15-rate-pair-h265-w1-s302013: job is running; terminal audited evaluation required; local audit has not passed.

Uncertainty and scope:

- Three independent fine-tuning seeds, each starting from one fixed pretrained source; n=3 gives limited uncertainty precision.
- Account is a paired blocking condition, not an additional independent replication.
- The shared 208 development clips and five QPs are not independent fine-tuning seed replicates.
- These clips have known historical validation use; the cohort is development evidence, not a new untouched holdout.
- Lower paired BD-rate is supported only with all three audited pairs, the paired 95% CI upper bound below zero, and BD-accuracy >=0 plus worst-QP Top-1 gap >=-0.5 pp for all three w1 runs.
- No seed is selected by its outcome. The preregistered representative must pass the unchanged development gates before the same checkpoint is evaluated on 802 disjoint confirmation clips and full 1010 validation clips. This report cannot establish the -10% goal.

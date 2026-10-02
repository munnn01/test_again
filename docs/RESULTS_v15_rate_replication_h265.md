# Rate-weight development cohort: v15-rate-replication-h265-20261002

H.265; source `a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e`; 1000 fresh steps per run.

Audited jobs: 6/6. Complete: true. Goal complete: false.

| Account | Seed | Arm | Job | Status | Counted | BD-rate (%) | BD-accuracy (pp) | Worst-QP Top-1 gap (pp) |
|---|---:|---|---|---|---|---:|---:|---:|
| qktttttttttt | 302011 | w025 | v15-rate-pair-h265-w025-s302011 | evaluated | yes | +3.1113 | -1.0083 | -0.4808 |
| qktttttttttt | 302011 | w1 | v15-rate-pair-h265-w1-s302011 | evaluated | yes | +2.8697 | -0.6064 | -1.4423 |
| baoancut | 302012 | w025 | v15-rate-pair-h265-w025-s302012 | evaluated | yes | +7.4565 | -3.5988 | -0.4808 |
| baoancut | 302012 | w1 | v15-rate-pair-h265-w1-s302012 | evaluated | yes | +3.6851 | -2.5520 | -0.9615 |
| trnhlng | 302013 | w025 | v15-rate-pair-h265-w025-s302013 | evaluated | yes | +4.0560 | -1.1715 | +0.0000 |
| trnhlng | 302013 | w1 | v15-rate-pair-h265-w1-s302013 | evaluated | yes | +1.4139 | -1.1445 | -0.9615 |

| Arm | Metric | Mean | Sample SD |
|---|---|---:|---:|
| w025 | bd_rate_pct | +4.8746 | 2.2854 |
| w025 | bd_accuracy_pp | -1.9262 | 1.4508 |
| w025 | worst_top1_gap_pp | -0.3205 | 0.2776 |
| w1 | bd_rate_pct | +2.6563 | 1.1506 |
| w1 | bd_accuracy_pp | -1.4343 | 1.0046 |
| w1 | worst_top1_gap_pp | -1.1218 | 0.2776 |

Paired contrast is w1 minus w025 within each seed/account block (n=3, df=2).

| Metric | Mean paired delta | Sample SD | 95% Student t CI |
|---|---:|---:|---|
| bd_rate_pct | -2.2183 | 1.8027 | [-6.6965, +2.2598] |
| bd_accuracy_pp | +0.4919 | 0.5158 | [-0.7895, +1.7733] |
| worst_top1_gap_pp | -0.8013 | 0.2776 | [-1.4908, -0.1118] |

Negative BD-rate delta favors w1; positive accuracy or Top-1-gap delta favors w1.

Uncertainty and scope:

- Three independent fine-tuning seeds, each starting from one fixed pretrained source; n=3 gives limited uncertainty precision.
- Account is a paired blocking condition, not an additional independent replication.
- The shared 208 development clips and five QPs are not independent fine-tuning seed replicates.
- These clips have known historical validation use; the cohort is development evidence, not a new untouched holdout.
- Lower paired BD-rate is supported only with all three audited pairs, the paired 95% CI upper bound below zero, and BD-accuracy >=0 plus worst-QP Top-1 gap >=-0.5 pp for all three w1 runs.
- No seed is selected by its outcome. The preregistered representative must pass the unchanged development gates before the same checkpoint is evaluated on 802 disjoint confirmation clips and full 1010 validation clips. This report cannot establish the -10% goal.

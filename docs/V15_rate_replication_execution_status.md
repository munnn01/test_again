# H265 rate-weight multi-account execution receipt

Recorded at 2026-10-02T04:51:20.286900+00:00. All six SHA-sealed packages passed Python/Bash/config preflight.
Four H265 jobs have positive Kaggle push acknowledgements and API status running.
No audited BD-rate result is available yet; paired aggregates remain withheld.

| Account | Seed | w025 | w1 | State |
|---|---:|---|---|---|
| qktttttttttt | 302011 | submit_failed | submit_failed | Awaiting existing H264 GPU slots; both prior attempts rejected at capacity2 |
| baoancut | 302012 | [running](https://www.kaggle.com/code/baoancut/v15-rate-pair-h265-w025-s302012) | [running](https://www.kaggle.com/code/baoancut/v15-rate-pair-h265-w1-s302012) | Both submissions acknowledged; status running verified through API |
| trnhlng | 302013 | [running](https://www.kaggle.com/code/trnhlng/v15-rate-pair-h265-w025-s302013) | [running](https://www.kaggle.com/code/trnhlng/v15-rate-pair-h265-w1-s302013) | Both submissions acknowledged; status running verified through API |

The human explicitly approved the four additional-account packages with:
"Cho phép gửi và chạy bốn gói trên baoancut/trnhlng". The earlier automatic
egress-review rejection is resolved by that approval, with its original record
retained in external_execution_review_history. Only the four named packages
were sent; no token, dataset or checkpoint was uploaded. Private metadata and
notebook SHA256 were rechecked before updating authorization.

Existing H264 jobs, last API status:

- qktttttttttt/v15-ar-rate-pressure-h264-s302001: running at 2026-10-02T04:49:55.037865+00:00.
- qktttttttttt/v15-ar-feasible-hinge-h264-s302001: running at 2026-10-02T04:49:57.505417+00:00.

Readiness verification:60 CPU runner/reporter tests passed in the preceding
commit. No code changed in this approval/submission update. Candidate training
commits and pinned scoring helpers are unchanged. The qk capacity errors had
CLI0 and remain accurately recorded as submit_failed, never successful runs.

All three paired blocks must pass independent artifact audit before cohort
effect estimates are reported. Acceptance stays BD-rate<=-10%, BD-accuracy>=0,
worstQP gap>=-0.5pp, then disjoint802 and full1010 confirmation using the fixed
representative seed302011. A running notebook is not performance evidence.

No new goal, recurring automation or scheduled result check was created.

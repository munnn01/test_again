# H265 rate-weight multi-account execution receipt

Recorded at 2026-10-02T04:46:05.506465+00:00. All six SHA-sealed packages passed Python/Bash/config preflight.
No new H265 GPU job has started; there is no BD-rate result for this cohort.

| Account | Seed | w025 | w1 | Hold reason |
|---|---:|---|---|---|
| qktttttttttt | 302011 | submit_failed | submit_failed | Kaggle rejected both attempts: maximum batch GPU session count of2 reached |
| baoancut | 302012 | prepared | prepared | Automatic approval review rejected new-account egress; explicit four-package/destination approval pending |
| trnhlng | 302013 | prepared | prepared | Automatic approval review rejected new-account egress; explicit four-package/destination approval pending |

The original qktttttttttt account remains authorized for optimization. Other
accounts are held by runner authorization checks. No additional-account push
was executed. No token, data or checkpoint was uploaded. The failed qk pushes
returned CLI0, which was corrected in registry with original records retained;
runner now requires the positive backend push acknowledgement.

Existing H264 jobs, last API status:

- qktttttttttt/v15-ar-rate-pressure-h264-s302001: running at 2026-10-02T04:45:20.297141+00:00.
- qktttttttttt/v15-ar-feasible-hinge-h264-s302001: running at 2026-10-02T04:45:22.309082+00:00.

Readiness verification:60 CPU runner/reporter tests passed. Candidate training
commits and pinned scoring helpers are unchanged. Three-pair aggregates remain
null until every job has passed independent local artifact audit. The final
acceptance stays BD-rate<=-10%, BD-accuracy>=0, worstQP gap>=-0.5pp, plus802
and full1010 confirmation for the preregistered representative seed302011.

No new goal, recurring automation or scheduled result check was created.

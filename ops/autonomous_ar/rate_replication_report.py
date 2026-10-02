#!/usr/bin/env python
"""Summarize the preregistered three-pair H.265 rate-weight development cohort.

This reads audited runner state; it does not train, audit artifacts, select a
seed, evaluate held-out clips, or declare the optimization goal complete.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SOURCE_SHA256 = "a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e"
DEV_FINGERPRINT = "f321b810af2c507c"
ARMS = ("w025", "w1")
METRICS = ("bd_rate_pct", "bd_accuracy_pp", "worst_top1_gap_pp")
T_975_DF2 = 4.302652729911275
LIMITATIONS = [
    "Three independent fine-tuning seeds, each starting from one fixed pretrained source; n=3 gives limited uncertainty precision.",
    "Account is a paired blocking condition, not an additional independent replication.",
    "The shared 208 development clips and five QPs are not independent fine-tuning seed replicates.",
    "These clips have known historical validation use; the cohort is development evidence, not a new untouched holdout.",
    "Lower paired BD-rate is supported only with all three audited pairs, the paired 95% CI upper bound below zero, and BD-accuracy >=0 plus worst-QP Top-1 gap >=-0.5 pp for all three w1 runs.",
    "No seed is selected by its outcome. The preregistered representative must pass the unchanged development gates before the same checkpoint is evaluated on 802 disjoint confirmation clips and full 1010 validation clips. This report cannot establish the -10% goal.",
]


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _validate_manifest(manifest):
    _mapping(manifest, "cohort manifest")
    _text(manifest.get("cohort_id"), "cohort_id")
    if manifest.get("node") != "n10" or manifest.get("codec") != "h265":
        raise ValueError("this fixed cohort requires node n10 and codec h265")
    if manifest.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("cohort source_sha256 differs from the registered H.265 source")
    if type(manifest.get("steps")) is not int or manifest["steps"] != 1000:
        raise ValueError("cohort requires exactly 1000 fresh steps")
    arms = _mapping(manifest.get("arms"), "arms")
    if set(arms) != set(ARMS):
        raise ValueError("cohort requires exactly arms w025 and w1")
    for arm, weight in (("w025", 0.25), ("w1", 1.0)):
        entry = _mapping(arms[arm], f"arms.{arm}")
        if not _finite(entry.get("rate_weight")) or entry["rate_weight"] != weight:
            raise ValueError(f"arms.{arm}.rate_weight must be {weight}")
        ref = _text(entry.get("candidate_ref"), f"arms.{arm}.candidate_ref")
        if len(ref) != 40 or any(c not in "0123456789abcdef" for c in ref):
            raise ValueError(f"arms.{arm}.candidate_ref must be a full lowercase commit SHA")
        _text(entry.get("config"), f"arms.{arm}.config")
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 3:
        raise ValueError("cohort requires exactly three paired seed/account blocks (df=2)")
    accounts, seeds, slugs = set(), set(), set()
    for i, block in enumerate(blocks):
        _mapping(block, f"blocks[{i}]")
        account = _text(block.get("account"), f"blocks[{i}].account")
        seed = block.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"blocks[{i}].seed must be a nonnegative integer")
        if account in accounts or seed in seeds:
            raise ValueError("duplicate account or seed in paired blocks")
        accounts.add(account)
        seeds.add(seed)
        jobs = _mapping(block.get("jobs"), f"blocks[{i}].jobs")
        if set(jobs) != set(ARMS):
            raise ValueError(f"blocks[{i}] must register both arms")
        for arm in ARMS:
            slug = _text(jobs[arm], f"blocks[{i}].jobs.{arm}")
            if slug in slugs:
                raise ValueError("duplicate job ID in paired blocks")
            slugs.add(slug)
    representative = manifest.get("representative_seed")
    if type(representative) is not int or representative not in seeds:
        raise ValueError("representative_seed must be one preregistered block seed")


def _validate_job(job, manifest, block, arm, slug):
    _mapping(job, f"job {slug}")
    expected = {
        "slug": slug, "account": block["account"], "seed": block["seed"],
        "codec": manifest["codec"], "source_sha256": manifest["source_sha256"],
        "steps": manifest["steps"], "stage": "screen",
        "candidate_ref": manifest["arms"][arm]["candidate_ref"],
        "config": manifest["arms"][arm]["config"],
    }
    for key, value in expected.items():
        if type(job.get(key)) is not type(value) or job.get(key) != value:
            raise ValueError(f"job {slug}: {key} differs from the registered block/arm")


def _checkpoint_issues(job, manifest, arm):
    checkpoint = job.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return ["checkpoint manifest missing"]
    issues = []
    if "global_step" not in checkpoint:
        issues.append("checkpoint global_step missing")
    elif type(checkpoint["global_step"]) is not int or checkpoint["global_step"] != manifest["steps"]:
        raise ValueError(f"job {job['slug']}: checkpoint global_step differs from registered steps")
    cfg = checkpoint.get("cfg", {})
    loss = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    constraint = loss.get("task_regret_constraint", {}) if isinstance(loss, dict) else {}
    if not isinstance(constraint, dict):
        return issues + ["checkpoint task_regret_constraint missing"]
    if "enabled" in constraint and constraint["enabled"] is not True:
        raise ValueError(f"job {job['slug']}: checkpoint task-regret constraint disabled")
    per_codec = constraint.get("per_codec", {})
    override = per_codec.get(manifest["codec"], {}) if isinstance(per_codec, dict) else {}
    for label, settings in (("global", constraint), (manifest["codec"], override)):
        if not isinstance(settings, dict) or "rate_weight" not in settings:
            issues.append(f"checkpoint {label} rate_weight missing")
        elif not _finite(settings["rate_weight"]) or settings["rate_weight"] != manifest["arms"][arm]["rate_weight"]:
            raise ValueError(f"job {job['slug']}: checkpoint {label} rate_weight differs from registered arm")
    return issues


def _row(manifest, block, arm, jobs):
    slug = block["jobs"][arm]
    row = {
        "job": slug, "account": block["account"], "seed": block["seed"],
        "arm": arm, "rate_weight": manifest["arms"][arm]["rate_weight"],
        "status": "missing", "local_audit_passed": False, "counted": False,
        "metrics": None, "development_gates_pass": None, "issues": [],
    }
    if slug not in jobs:
        row["issues"].append("registered job missing from state")
        return row
    job = jobs[slug]
    _validate_job(job, manifest, block, arm, slug)
    row["status"] = str(job.get("status", "unknown"))
    row["local_audit_passed"] = job.get("local_audit_passed") is True
    # Available checkpoint values cannot silently disagree even on a pending job.
    checkpoint_issues = _checkpoint_issues(job, manifest, arm)
    if row["status"] != "evaluated":
        row["issues"].append(f"job is {row['status']}; terminal audited evaluation required")
    if not row["local_audit_passed"]:
        row["issues"].append("local audit has not passed")
    if row["issues"]:
        return row  # Never display stale or unaudited metrics as verified results.
    row["issues"].extend(checkpoint_issues)
    scores = job.get("scores")
    score = scores.get("dev") if isinstance(scores, dict) else None
    if not isinstance(score, dict):
        row["issues"].append("audited development score missing")
        return row
    if type(score.get("n_eval")) is not int or score["n_eval"] != 208:
        row["issues"].append("development score must contain exactly 208 clips")
    if score.get("fingerprint") != DEV_FINGERPRINT:
        row["issues"].append("development fingerprint differs from registered 208 clips")
    if "stage" in score and score["stage"] != "dev":
        row["issues"].append("score is not a development result")
    if "codec" in score and score["codec"] != manifest["codec"]:
        row["issues"].append("development score codec differs from cohort")
    for key in METRICS:
        if not _finite(score.get(key)):
            row["issues"].append(f"development {key} missing or nonfinite")
    if row["issues"]:
        return row
    metrics = {key: float(score[key]) for key in METRICS}
    passes = (metrics["bd_rate_pct"] <= -10 and metrics["bd_accuracy_pp"] >= 0
              and metrics["worst_top1_gap_pp"] >= -0.5 - 1e-12)
    if "passes" in score and (type(score["passes"]) is not bool or score["passes"] != passes):
        raise ValueError(f"job {slug}: development passes flag disagrees with finite metrics")
    row.update(metrics=metrics, counted=True, development_gates_pass=passes)
    return row


def _summary(values, paired=False):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    result = {"n": len(values), "mean": mean, "sample_sd": sd}
    if paired:
        half_width = T_975_DF2 * sd / math.sqrt(len(values))
        result.update(ci95=[mean - half_width, mean + half_width], df=2,
                      method="paired Student t interval", contrast="w1-w025")
    return result


def build_report(manifest, state):
    """Return all registered rows; withhold aggregate inference until all six pass audit.

    Metadata or treatment disagreement raises ValueError. Incomplete execution or
    nonfinite development evidence produces a complete=False report with issues.
    Neither input is modified, and this report can never mark goal_complete=True.
    """
    _validate_manifest(manifest)
    jobs = _mapping(_mapping(state, "state").get("jobs"), "state.jobs")
    rows = [_row(manifest, block, arm, jobs) for block in manifest["blocks"] for arm in ARMS]
    complete = all(row["counted"] for row in rows)
    aggregates = None
    if complete:
        by_arm = {arm: {key: _summary([r["metrics"][key] for r in rows if r["arm"] == arm])
                        for key in METRICS} for arm in ARMS}
        contrasts = [{"account": block["account"], "seed": block["seed"],
                      "metrics": {key: rows[2*i + 1]["metrics"][key] - rows[2*i]["metrics"][key]
                                  for key in METRICS}}
                     for i, block in enumerate(manifest["blocks"])]
        paired = {key: _summary([block["metrics"][key] for block in contrasts], paired=True)
                  for key in METRICS}
        aggregates = {"by_arm": by_arm, "paired_w1_minus_w025": paired,
                      "paired_block_deltas": contrasts}
    return {
        "cohort_id": manifest["cohort_id"], "node": "n10", "codec": "h265",
        "source_sha256": manifest["source_sha256"], "steps": 1000, "stage": "dev",
        "development": {"n_eval": 208, "fingerprint": DEV_FINGERPRINT,
                        "qps": [30, 35, 40, 45, 50]},
        "arms": {arm: {key: manifest["arms"][arm][key]
                       for key in ("rate_weight", "candidate_ref", "config")} for arm in ARMS},
        "blocks": [{"account": block["account"], "seed": block["seed"],
                    "jobs": {arm: block["jobs"][arm] for arm in ARMS}}
                   for block in manifest["blocks"]],
        "representative_seed": manifest["representative_seed"],
        "registered_pairs": 3, "registered_jobs": 6,
        "audited_jobs": sum(row["counted"] for row in rows),
        "complete": complete, "goal_complete": False,
        "rows": rows, "issues": [{"job": row["job"], "issues": list(row["issues"])}
                                 for row in rows if row["issues"]],
        "aggregates": aggregates,
        "contrast_interpretation": {"bd_rate_pct": "negative favors w1; difference in percentage points",
                                    "bd_accuracy_pp": "positive favors w1; accuracy percentage points",
                                    "worst_top1_gap_pp": "positive favors w1; Top-1 percentage points"},
        "limitations": list(LIMITATIONS),
    }


def _md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(report):
    lines = [f"# Rate-weight development cohort: {_md(report['cohort_id'])}", "",
             f"H.265; source `{report['source_sha256']}`; {report['steps']} fresh steps per run.", "",
             f"Audited jobs: {report['audited_jobs']}/6. Complete: {str(report['complete']).lower()}. Goal complete: false.", "",
             "| Account | Seed | Arm | Job | Status | Counted | BD-rate (%) | BD-accuracy (pp) | Worst-QP Top-1 gap (pp) |",
             "|---|---:|---|---|---|---|---:|---:|---:|"]
    for row in report["rows"]:
        values = row["metrics"]
        metrics = [f"{values[key]:+.4f}" if values else "--" for key in METRICS]
        cells = [row["account"], row["seed"], row["arm"], row["job"], row["status"],
                 "yes" if row["counted"] else "no", *metrics]
        lines.append("| " + " | ".join(_md(value) for value in cells) + " |")
    if report["issues"]:
        lines += ["", "Aggregates withheld until all six registered jobs have valid audited development evidence.", ""]
        lines.extend(f"- {_md(row['job'])}: {_md('; '.join(row['issues']))}." for row in report["issues"])
    else:
        aggregates = report["aggregates"]
        lines += ["", "| Arm | Metric | Mean | Sample SD |", "|---|---|---:|---:|"]
        for arm in ARMS:
            for key in METRICS:
                result = aggregates["by_arm"][arm][key]
                lines.append(f"| {arm} | {key} | {result['mean']:+.4f} | {result['sample_sd']:.4f} |")
        lines += ["", "Paired contrast is w1 minus w025 within each seed/account block (n=3, df=2).", "",
                  "| Metric | Mean paired delta | Sample SD | 95% Student t CI |",
                  "|---|---:|---:|---|"]
        for key in METRICS:
            result = aggregates["paired_w1_minus_w025"][key]
            lo, hi = result["ci95"]
            lines.append(f"| {key} | {result['mean']:+.4f} | {result['sample_sd']:.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        lines += ["", "Negative BD-rate delta favors w1; positive accuracy or Top-1-gap delta favors w1."]
    lines += ["", "Uncertainty and scope:", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args(argv)
    inputs = {args.cohort.resolve(), args.state.resolve()}
    outputs = [p.resolve() for p in (args.out_json, args.out_md) if p is not None]
    if any(p in inputs for p in outputs) or len(outputs) != len(set(outputs)):
        parser.error("output paths must be distinct and cannot overwrite cohort or read-only state")
    try:
        report = build_report(json.loads(args.cohort.read_text(encoding="utf-8")),
                              json.loads(args.state.read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown = render_markdown(report)
    if args.out_json:
        args.out_json.write_text(encoded, encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    if not args.out_json:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Fixed development/confirmation partitions and real-codec candidate scoring."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIRM_FINGERPRINT = '740e9571c679071a'
SPEC = importlib.util.spec_from_file_location("v13_shared_audit", ROOT / "ops/report_v13_task_regret.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def partitions(index: dict):
    dev = AUDIT.audit_index(index)
    dev_set = set(dev)
    confirm = sorted(AUDIT.clip_key(row["path"]) for row in index["val"] if AUDIT.clip_key(row["path"]) not in dev_set)
    if len(confirm) != 802 or AUDIT.fingerprint(confirm) != CONFIRM_FINGERPRINT or dev_set.intersection(confirm):
        raise ValueError("confirmation membership/fingerprint mismatch")
    return {"dev": dev, "confirmation": confirm,
            "full": sorted(AUDIT.clip_key(row["path"]) for row in index["val"])}


def make_indices(index: dict, out: Path):
    ids = partitions(index)
    out.mkdir(parents=True, exist_ok=True)
    for stage, keys in ids.items():
        selected = set(keys)
        val = [row for row in index["val"] if AUDIT.clip_key(row["path"]) in selected]
        result = {"meta": {"role": stage, "n_val": len(val), "fingerprint": AUDIT.fingerprint(keys)},
                  "train": index["train"] if stage == "dev" else [], "val": val, "test": []}
        (out / f"{stage}.json").write_text(json.dumps(result), encoding="utf-8")
    manifest = {stage: {"n": len(keys), "fingerprint": AUDIT.fingerprint(keys), "clip_ids": keys} for stage, keys in ids.items()}
    (out / "partitions.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def score(report: dict, points: Path, index: dict, codec: str, stage: str):
    ids = partitions(index)[stage]
    raw = AUDIT.audit_screen(report, points, codec, ids, expected_count=len(ids), expected_fingerprint=AUDIT.fingerprint(ids))
    criteria = {
        "bd_rate_le_minus10_pct": raw["bd_rate_pct"] is not None and raw["bd_rate_pct"] <= -10.0,
        "bd_accuracy_nonnegative": raw["bd_accuracy"] is not None and raw["bd_accuracy"] >= 0.0,
        "all_qp_top1_gap_ge_minus0_5_pp": raw["worst_top1_gap_pp"] >= -0.5 - 1e-12,
    }
    result = {key: raw[key] for key in ("codec", "n_eval", "paired_points", "bd_rate_pct", "bd_accuracy", "bd_accuracy_pp", "top1_gap_pp_by_qp", "worst_top1_gap_pp")}
    result.update({"protocol": "autonomous-learned-ar-v14", "stage": stage,
                   "fingerprint": AUDIT.fingerprint(ids), "criteria": criteria,
                   "passes": all(criteria.values()), "goal_complete": False,
                   "confirmation_scope": "802 validation clips excluded from this search; historical validation use disclosed"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("indices", "score"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--codec", choices=("h264", "h265"))
    parser.add_argument("--stage", choices=("dev", "confirmation", "full"))
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    if args.command == "indices":
        result = make_indices(index, args.out)
        print(json.dumps({stage: {k: v for k, v in value.items() if k != "clip_ids"} for stage, value in result.items()}))
    else:
        if not all((args.results, args.points, args.codec, args.stage)):
            parser.error("score requires results, points, codec and stage")
        result = score(json.loads(args.results.read_text(encoding="utf-8")), args.points, index, args.codec, args.stage)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

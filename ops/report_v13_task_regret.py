#!/usr/bin/env python
"""Audit the locked V13 screen and report both registered and user gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

QPS = (30, 35, 40, 45, 50)
SALT = "v13-task-regret-screen-v1"
FINGERPRINT = "f321b810af2c507c"


def clip_key(path: str) -> str:
    return "/".join(str(path).replace("\\", "/").rstrip("/").split("/")[-2:])


def fingerprint(keys) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:16]


def audit_index(index: dict) -> list[str]:
    for split, expected in (("train", 8632), ("val", 1010), ("test", 1159)):
        keys = [clip_key(row["path"]) for row in index[split]]
        if len(keys) != expected or len(set(keys)) != expected:
            raise ValueError(f"{split}: expected {expected} unique clips")
        remainder = {"train": set(range(2, 10)), "val": {1}, "test": {0}}[split]
        if any(int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10 not in remainder for key in keys):
            raise ValueError(f"{split}: wrong canonical split membership")
    test_keys = sorted(clip_key(row["path"]) for row in index["test"])
    if hashlib.md5(",".join(test_keys).encode()).hexdigest()[:12] != "30f083f8520a":
        raise ValueError("canonical test fingerprint mismatch")
    keys = [clip_key(row["path"]) for row in index["val"]]
    selected = sorted(key for key in keys if int(hashlib.md5(f"{SALT}:{key}".encode()).hexdigest()[:8], 16) % 5 == 0)
    if len(selected) != 208 or fingerprint(selected) != FINGERPRINT:
        raise ValueError("V13 screen count/fingerprint mismatch")
    return selected


def _bd_functions():
    # Load the canonical implementation without importing GPU/task dependencies.
    path = Path(__file__).resolve().parents[1] / "src" / "metrics" / "bd_rate.py"
    spec = importlib.util.spec_from_file_location("v13_bd_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bd_rate, module.bd_metric


def audit_screen(report: dict, points: Path, codec: str, expected_ids: list[str], *,
                 expected_count: int = 208, expected_fingerprint: str | None = None) -> dict:
    expected_fingerprint = FINGERPRINT if expected_fingerprint is None else expected_fingerprint
    if codec not in ("h264", "h265"):
        raise ValueError("unsupported codec")
    if report.get("task") != "action_recognition" or report.get("metric") != "top1" or report.get("n_eval") != expected_count:
        raise ValueError(f"expected the {expected_count}-clip AR Top-1 screen")
    expected = set(expected_ids)
    if len(expected_ids) != expected_count or len(expected) != expected_count or fingerprint(expected) != expected_fingerprint:
        raise ValueError("expected clip IDs do not match the registered screen")
    methods = (codec, f"prep+{codec}")
    cells = {method: {qp: {} for qp in QPS} for method in methods}
    with points.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method, qp, key = row["codec"], int(row["qp"]), clip_key(row["path"])
            if method not in methods or qp not in QPS or key not in expected:
                raise ValueError("unexpected method/QP/clip in per-clip points")
            if row["sequence_id"] != key or key in cells[method][qp]:
                raise ValueError("duplicate or inconsistent clip identity")
            rate, top1 = float(row["bpp"]), int(row["top1"])
            probability = float(row["target_prob"])
            if not math.isfinite(rate) or rate <= 0 or top1 not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid per-clip rate/task value")
            cells[method][qp][key] = (rate, top1)
    curves = {}
    for method in methods:
        curve = report["curves"][method]
        if len(curve["keys"]) != 5 or set(map(int, curve["keys"])) != set(QPS) or len(curve["bpp"]) != 5 or len(curve["accuracy"]) != 5:
            raise ValueError("expected all five QPs in each curve")
        mapping = {int(q): (float(rate), float(acc)) for q, rate, acc in zip(curve["keys"], curve["bpp"], curve["accuracy"])}
        for qp in QPS:
            records = cells[method][qp]
            if set(records) != expected:
                raise ValueError(f"incomplete pairing: {method} QP{qp}")
            rate, acc = mapping[qp]
            observed_rate = sum(value[0] for value in records.values()) / expected_count
            observed_acc = sum(value[1] for value in records.values()) / expected_count
            if not math.isfinite(rate) or rate <= 0 or not math.isfinite(acc) or not 0 <= acc <= 1:
                raise ValueError("invalid aggregate curve value")
            if not math.isclose(rate, observed_rate, abs_tol=6e-9, rel_tol=1e-7) or not math.isclose(acc, observed_acc, abs_tol=1e-12):
                raise ValueError("aggregate curve disagrees with paired clip records")
        curves[method] = mapping
    anchor, treatment = curves[codec], curves[f"prep+{codec}"]
    bd_rate, bd_accuracy = _bd_functions()
    ra, qa = zip(*(anchor[qp] for qp in QPS))
    rt, qt = zip(*(treatment[qp] for qp in QPS))
    rate = bd_rate(ra, qa, rt, qt)
    accuracy = bd_accuracy(ra, qa, rt, qt)
    metric = report["bd_prep_gain"][f"prep+{codec} vs {codec}"]
    for name, recomputed in (("bd_rate_pct", rate), ("bd_accuracy", accuracy)):
        observed = metric.get(name)
        if math.isfinite(recomputed):
            if observed is None or not math.isclose(float(observed), recomputed, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"reported {name} disagrees with the canonical BD computation")
        elif observed is not None and math.isfinite(float(observed)):
            raise ValueError(f"reported finite {name} for an undefined curve")
    rate = rate if math.isfinite(rate) else None
    accuracy = accuracy if math.isfinite(accuracy) else None
    gaps = {str(qp): 100 * (treatment[qp][1] - anchor[qp][1]) for qp in QPS}
    conditions = {
        "bd_rate_at_most_minus5_pct": rate is not None and rate <= -5.0,
        "bd_accuracy_nonnegative": accuracy is not None and accuracy >= 0.0,
        "every_qp_gap_at_least_minus0_5_pp": min(gaps.values()) >= -0.5 - 1e-12,
    }
    passed = all(conditions.values())
    return {
        "codec": codec, "n_eval": expected_count, "shard_salt": SALT,
        "shard_fingerprint": fingerprint(expected), "paired_points": expected_count * len(QPS) * 2,
        "bd_rate_pct": rate, "bd_accuracy": accuracy,
        "bd_accuracy_pp": 100 * accuracy if accuracy is not None else None,
        "top1_gap_pp_by_qp": gaps, "worst_top1_gap_pp": min(gaps.values()),
        "registered_conditions": conditions,
        "registered_decision": "advance" if passed else "futility_stop",
        "user_bd_rate_max_pct": -10.0,
        "user_acceptance": passed and rate is not None and rate <= -10.0,
        "evidence_status": "exploratory screen; replication and final test still required",
    }


def audit_training(checkpoint: Path, codec: str, manifest: dict | None = None) -> dict:
    import torch
    import yaml

    if checkpoint.name != "preprocessor_last.pth":
        raise ValueError("V13 must evaluate the final checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    registered = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/v13_task_regret_ar.yaml").read_text(encoding="utf-8"))
    cfg = state["cfg"]
    if state["epoch"] != 3 or state["global_step"] != 12948:
        raise ValueError("V13 requires three complete epochs / 12948 fresh steps")
    for section in ("task", "model", "loss", "train"):
        if cfg[section] != registered[section]:
            raise ValueError(f"checkpoint {section} differs from the registered V13 configuration")
    expected_codec = dict(registered["codec"], ste_codec=codec)
    if cfg["codec"] != expected_codec or cfg["seed"] != {"h264": 292001, "h265": 292002}[codec]:
        raise ValueError("checkpoint codec/seed does not match the registered arm")
    cells = {f"{codec}:{qp}" for qp in QPS}
    duals = state["task_duals"]
    if set(duals) != cells or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 10 for v in duals.values()):
        raise ValueError("invalid per-codec/QP task duals")
    if set(state["rate_duals"]) != cells or any(float(v) != 0 for v in state["rate_duals"].values()):
        raise ValueError("old rate duals were not reset")
    if not state.get("opt") or state.get("sched") is not None or any(group["lr"] != 1e-5 for group in state["opt"]["param_groups"]):
        raise ValueError("optimizer/scheduler does not match V13")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if manifest is not None:
        expected_source = {
            "h264": "ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a",
            "h265": "a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e",
        }[codec]
        if manifest["final_sha256"] != sha or manifest["source_sha256"] != expected_source or manifest["codec"] != codec or manifest["seed"] != cfg["seed"] or manifest["task_duals"] != duals or manifest["commit"] != "35f0bd1c7844ad6965f5b0e148f8bd7869f48d93":
            raise ValueError("run manifest does not match the audited checkpoint/source")
    return {"final_sha256": sha, "epoch": state["epoch"], "global_step": state["global_step"],
            "registered_config_match": True, "old_rate_duals_reset": True, "task_duals": duals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    ids = audit_index(json.loads(args.index.read_text(encoding="utf-8")))
    result = {"n_screen": len(ids), "shard_fingerprint": fingerprint(ids)}
    if args.results:
        if not args.codec or not args.points or not args.out:
            parser.error("--results requires --codec, --points and --out")
        result = audit_screen(json.loads(args.results.read_text(encoding="utf-8")), args.points, args.codec, ids)
        if args.checkpoint:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
            result["checkpoint_audit"] = audit_training(args.checkpoint, args.codec, manifest)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import csv
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = load_module("v13_report", ROOT / "ops/report_v13_task_regret.py")


@pytest.fixture
def screen(tmp_path, monkeypatch):
    ids = [f"class/clip{i}.mp4" for i in range(208)]
    monkeypatch.setattr(REPORT, "FINGERPRINT", REPORT.fingerprint(ids))

    def build(ratio=0.88, lose_at=None, flat=False):
        points = tmp_path / "points.csv"
        curves = {}
        with points.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sequence_id", "path", "class", "codec", "qp", "bpp", "top1", "target_prob"])
            for method in ("h264", "prep+h264"):
                curve = {"keys": [], "bpp": [], "accuracy": []}
                for j, qp in enumerate(REPORT.QPS):
                    n_correct = 120 if flat else 160 - 20 * j
                    if method.startswith("prep+") and qp == lose_at:
                        n_correct -= 2  # 0.962 pp, beyond the 0.5 pp guardrail
                    rate = (0.5 - 0.09 * j) * (ratio if method.startswith("prep+") else 1)
                    curve["keys"].append(qp)
                    curve["bpp"].append(rate)
                    curve["accuracy"].append(n_correct / 208)
                    for i, key in enumerate(ids):
                        writer.writerow([key, f"/mount/{key}", "class", method, qp, rate, int(i < n_correct), 0.5])
                # Deliberately use different QP ordering between the arms.
                if method == "prep+h264":
                    curve = {key: list(reversed(value)) for key, value in curve.items()}
                curves[method] = curve
        bd_rate, bd_acc = REPORT._bd_functions()
        a, p = curves["h264"], curves["prep+h264"]
        rate = bd_rate(a["bpp"], a["accuracy"], p["bpp"], p["accuracy"])
        accuracy = bd_acc(a["bpp"], a["accuracy"], p["bpp"], p["accuracy"])
        import math
        report = {
            "task": "action_recognition", "metric": "top1", "n_eval": 208,
            "curves": curves,
            "bd_prep_gain": {"prep+h264 vs h264": {
                "bd_rate_pct": rate if math.isfinite(rate) else None,
                "bd_accuracy": accuracy if math.isfinite(accuracy) else None,
            }},
        }
        return report, points, ids
    return build


def test_registered_pass_does_not_imply_user_acceptance(screen):
    report, points, ids = screen(ratio=0.92)
    result = REPORT.audit_screen(report, points, "h264", ids)
    assert result["registered_decision"] == "advance"
    assert result["bd_rate_pct"] == pytest.approx(-8)
    assert result["user_acceptance"] is False


def test_user_acceptance_requires_rate_and_task_guards(screen):
    report, points, ids = screen()
    result = REPORT.audit_screen(report, points, "h264", ids)
    assert result["user_acceptance"] is True
    assert result["bd_rate_pct"] == pytest.approx(-12)
    report, points, ids = screen(lose_at=50)
    result = REPORT.audit_screen(report, points, "h264", ids)
    assert result["registered_decision"] == "futility_stop"
    assert result["user_acceptance"] is False
    assert result["worst_top1_gap_pp"] < -0.5


def test_undefined_bd_is_a_futility_stop(screen):
    report, points, ids = screen(flat=True)
    result = REPORT.audit_screen(report, points, "h264", ids)
    assert result["bd_rate_pct"] is None
    assert result["registered_decision"] == "futility_stop"
    assert result["user_acceptance"] is False


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "wrong_clip", "wrong_top1"])
def test_incomplete_or_corrupt_pairing_is_rejected(screen, corruption):
    report, points, ids = screen()
    lines = points.read_text(encoding="utf-8").splitlines()
    if corruption == "missing":
        lines.pop()
    elif corruption == "duplicate":
        lines.append(lines[-1])
    elif corruption == "wrong_clip":
        lines[-1] = lines[-1].replace("clip207", "wrong")
    else:
        report["curves"]["h264"]["accuracy"][0] += 0.01
    points.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        REPORT.audit_screen(report, points, "h264", ids)


def test_notebook_embeds_the_exact_local_audit_helper():
    module = load_module("v13_push_report", ROOT / "ops/push_v13_task_regret_screen.py")
    cell = module.render_cell("a" * 40, "h264", 292001, "b" * 64)
    payload = re.search(r'base64.b64decode\("([A-Za-z0-9+/=]+)"\)', cell).group(1)
    assert base64.b64decode(payload) == (ROOT / "ops/report_v13_task_regret.py").read_bytes()
    assert "__V13_REPORT_BASE64__" not in cell
    assert '--out "$ROOT_OUT/decision.json"' in cell


@pytest.mark.parametrize("corruption", [None, "duration", "epsilon", "old_dual", "missing_cell", "bad_hash", "best_checkpoint"])
def test_training_audit_rejects_protocol_drift(tmp_path, corruption):
    import copy
    import hashlib
    import torch
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/v13_task_regret_ar.yaml").read_text())
    state = {
        "cfg": copy.deepcopy(cfg), "epoch": 3, "global_step": 12948,
        "task_duals": {f"h264:{qp}": 0.1 for qp in REPORT.QPS},
        "rate_duals": {f"h264:{qp}": 0.0 for qp in REPORT.QPS},
        "opt": {"param_groups": [{"lr": 1e-5}]}, "sched": None,
    }
    if corruption == "duration":
        state["global_step"] -= 1
    elif corruption == "epsilon":
        state["cfg"]["loss"]["task_regret_constraint"]["epsilon"] = 0.01
    elif corruption == "old_dual":
        state["rate_duals"]["h264:30"] = 1.0
    elif corruption == "missing_cell":
        state["task_duals"].pop("h264:30")
    checkpoint = tmp_path / ("preprocessor.pth" if corruption == "best_checkpoint" else "preprocessor_last.pth")
    torch.save(state, checkpoint)
    manifest = {
        "codec": "h264", "seed": 292001,
        "commit": "35f0bd1c7844ad6965f5b0e148f8bd7869f48d93",
        "source_sha256": "ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a",
        "final_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "task_duals": state["task_duals"],
    }
    if corruption == "bad_hash":
        manifest["final_sha256"] = "0" * 64
    if corruption is None:
        assert REPORT.audit_training(checkpoint, "h264", manifest)["registered_config_match"] is True
    else:
        with pytest.raises(ValueError):
            REPORT.audit_training(checkpoint, "h264", manifest)

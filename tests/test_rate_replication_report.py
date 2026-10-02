from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rate_replication_report_tests", ROOT / "ops/autonomous_ar/rate_replication_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


@pytest.fixture
def cohort():
    manifest = {
        "cohort_id": "synthetic-paired-fixture", "node": "n10", "codec": "h265",
        "source_sha256": REPORT.SOURCE_SHA256, "steps": 1000,
        "representative_seed": 302011,
        "arms": {
            "w025": {"rate_weight": 0.25, "candidate_ref": "4e8f8b5ef68b7952d02bead532c27fd54bf8fa0f", "config": "configs/v14_dual_safety_ar.yaml"},
            "w1": {"rate_weight": 1, "candidate_ref": "4460a9f56243d1485c195d884e9c0e03414d90c1", "config": "configs/v15_rate_pressure_ar.yaml"},
        }, "blocks": [],
    }
    jobs = {}
    for i, account in enumerate(("qktttttttttt", "baoancut", "trnhlng")):
        seed = 302011 + i
        block = {"account": account, "seed": seed, "jobs": {}}
        for arm in ("w025", "w1"):
            slug = f"v15-rate-pair-h265-{arm}-s{seed}"
            block["jobs"][arm] = slug
            config = manifest["arms"][arm]
            bd_rate = (-30, 10, 50)[i] + ((-4, -3, -2)[i] if arm == "w1" else 0)
            bd_accuracy = (-1, 0, 1)[i] + ((0.3, 0.2, 0.1)[i] if arm == "w1" else 0)
            worst = (-2, -1, 0)[i] + ((0.1, 0.2, 0.3)[i] if arm == "w1" else 0)
            jobs[slug] = {
                "slug": slug, "account": account, "seed": seed, "codec": "h265",
                "source_sha256": manifest["source_sha256"], "steps": 1000, "stage": "screen",
                "candidate_ref": config["candidate_ref"], "config": config["config"],
                "status": "evaluated", "local_audit_passed": True,
                "scores": {"dev": {"n_eval": 208, "fingerprint": REPORT.DEV_FINGERPRINT,
                                   "bd_rate_pct": bd_rate, "bd_accuracy_pp": bd_accuracy,
                                   "worst_top1_gap_pp": worst,
                                   "passes": bd_rate <= -10 and bd_accuracy >= 0 and worst >= -0.5}},
                "checkpoint": {"global_step": 1000, "cfg": {"loss": {"task_regret_constraint": {
                    "enabled": True, "rate_weight": config["rate_weight"],
                    "per_codec": {"h265": {"rate_weight": config["rate_weight"]}},
                }}}},
            }
        manifest["blocks"].append(block)
    return manifest, {"jobs": jobs}


def _first_job(cohort):
    manifest, state = cohort
    return state["jobs"][manifest["blocks"][0]["jobs"]["w1"]]


def test_complete_cohort_uses_within_block_contrasts_and_df2_ci(cohort):
    report = REPORT.build_report(*cohort)
    assert report["complete"] and report["audited_jobs"] == 6
    assert not report["goal_complete"] and not report["issues"]
    aggregate = report["aggregates"]
    control = aggregate["by_arm"]["w025"]["bd_rate_pct"]
    assert control == {"n": 3, "mean": 10, "sample_sd": 40}
    paired = aggregate["paired_w1_minus_w025"]["bd_rate_pct"]
    assert paired["mean"] == -3 and paired["sample_sd"] == 1 and paired["df"] == 2
    half_width = 4.302652729911275 / math.sqrt(3)
    assert paired["ci95"] == pytest.approx([-3 - half_width, -3 + half_width])
    # Seed/account difficulty dominates between-arm dispersion; an unpaired
    # standard error would destroy the information supplied by the pairing.
    unpaired_se = math.sqrt((40**2 + 41**2) / 3)
    assert paired["sample_sd"] / math.sqrt(3) < unpaired_se / 50
    assert [b["metrics"]["bd_rate_pct"] for b in aggregate["paired_block_deltas"]] == [-4, -3, -2]
    assert aggregate["paired_w1_minus_w025"]["bd_accuracy_pp"]["mean"] == pytest.approx(0.2)
    assert aggregate["paired_w1_minus_w025"]["worst_top1_gap_pp"]["sample_sd"] == pytest.approx(0.1)
    assert report["representative_seed"] == 302011
    assert "not independent" in " ".join(report["limitations"])


@pytest.mark.parametrize("problem", ["missing", "running", "failed", "unaudited", "nonfinite", "fingerprint", "checkpoint"])
def test_incomplete_cohort_keeps_all_rows_and_withholds_aggregates(cohort, problem):
    manifest, state = cohort
    slug = manifest["blocks"][0]["jobs"]["w1"]
    job = state["jobs"][slug]
    if problem == "missing":
        del state["jobs"][slug]
    elif problem in ("running", "failed"):
        job["status"] = problem
    elif problem == "unaudited":
        job["local_audit_passed"] = False
    elif problem == "nonfinite":
        job["scores"]["dev"]["bd_rate_pct"] = float("nan")
    elif problem == "fingerprint":
        job["scores"]["dev"]["fingerprint"] = "wrong"
    else:
        del job["checkpoint"]
    report = REPORT.build_report(manifest, state)
    assert not report["complete"] and report["aggregates"] is None
    assert not report["goal_complete"] and report["audited_jobs"] == 5
    assert len(report["rows"]) == 6 and report["issues"][0]["job"] == slug
    row = next(row for row in report["rows"] if row["job"] == slug)
    assert not row["counted"] and row["metrics"] is None and row["development_gates_pass"] is None
    rendered = REPORT.render_markdown(report)
    assert "Aggregates withheld" in rendered
    assert all(slug in rendered for block in manifest["blocks"] for slug in block["jobs"].values())
    json.dumps(report, allow_nan=False)  # Invalid input metrics cannot leak NaN into the report.


@pytest.mark.parametrize("key,value", [
    ("source_sha256", "b" * 64), ("seed", 302099), ("account", "another-account"),
    ("codec", "h264"), ("steps", 999), ("stage", "confirmation"),
    ("candidate_ref", "b" * 40), ("config", "configs/other.yaml"),
])
def test_available_job_identity_mismatch_rejects_even_when_running(cohort, key, value):
    job = _first_job(cohort)
    job["status"] = "running"
    job[key] = value
    with pytest.raises(ValueError, match=key):
        REPORT.build_report(*cohort)


@pytest.mark.parametrize("override", [False, True])
def test_shadowed_or_wrong_checkpoint_rate_rejects(cohort, override):
    constraint = _first_job(cohort)["checkpoint"]["cfg"]["loss"]["task_regret_constraint"]
    (constraint["per_codec"]["h265"] if override else constraint)["rate_weight"] = 0.25
    with pytest.raises(ValueError, match="rate_weight"):
        REPORT.build_report(*cohort)


@pytest.mark.parametrize("kind", ["account", "seed", "job"])
def test_duplicate_manifest_blocks_reject(cohort, kind):
    manifest, _ = cohort
    a, b = manifest["blocks"][:2]
    if kind == "job":
        b["jobs"]["w1"] = a["jobs"]["w025"]
    else:
        b[kind] = a[kind]
    with pytest.raises(ValueError, match="duplicate"):
        REPORT.build_report(*cohort)


@pytest.mark.parametrize("kind", ["source", "rate", "size", "representative"])
def test_invalid_registered_protocol_rejects(cohort, kind):
    manifest, _ = cohort
    if kind == "source":
        manifest["source_sha256"] = "b" * 64
    elif kind == "rate":
        manifest["arms"]["w1"]["rate_weight"] = 0.5
    elif kind == "size":
        manifest["blocks"].pop()
    else:
        manifest["representative_seed"] = 302099
    with pytest.raises(ValueError):
        REPORT.build_report(*cohort)


def test_development_flags_cannot_override_metrics_or_claim_goal(cohort):
    job = _first_job(cohort)
    job["scores"]["dev"]["passes"] = True
    with pytest.raises(ValueError, match="passes flag"):
        REPORT.build_report(*cohort)
    job["scores"]["dev"].update(bd_rate_pct=-20, bd_accuracy_pp=1, worst_top1_gap_pp=0, passes=True)
    job["goal_complete"] = True
    report = REPORT.build_report(*cohort)
    assert report["rows"][1]["development_gates_pass"] is True
    assert report["goal_complete"] is False


def test_report_and_cli_leave_state_read_only(cohort, tmp_path, capsys):
    original = copy.deepcopy(cohort)
    REPORT.build_report(*cohort)
    assert cohort == original
    manifest, state = cohort
    cohort_path, state_path = tmp_path / "cohort.json", tmp_path / "state.json"
    cohort_path.write_text(json.dumps(manifest), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    original_bytes = state_path.read_bytes()
    out_json, out_md = tmp_path / "report.json", tmp_path / "report.md"
    assert REPORT.main(["--cohort", str(cohort_path), "--state", str(state_path),
                        "--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    assert state_path.read_bytes() == original_bytes
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["complete"] and not saved["goal_complete"]
    assert out_md.read_text(encoding="utf-8") == REPORT.render_markdown(saved)
    assert "Student t CI" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        REPORT.main(["--cohort", str(cohort_path), "--state", str(state_path), "--out-json", str(state_path)])
    assert error.value.code == 2 and state_path.read_bytes() == original_bytes


def test_missing_codec_override_cannot_fall_back_to_global_rate(cohort):
    constraint = _first_job(cohort)["checkpoint"]["cfg"]["loss"]["task_regret_constraint"]
    del constraint["per_codec"]["h265"]["rate_weight"]
    report = REPORT.build_report(*cohort)
    assert not report["complete"] and report["aggregates"] is None
    assert "h265 rate_weight missing" in report["issues"][0]["issues"][0]


def test_checkpoint_step_mismatch_rejects(cohort):
    _first_job(cohort)["checkpoint"]["global_step"] = 999
    with pytest.raises(ValueError, match="global_step"):
        REPORT.build_report(*cohort)

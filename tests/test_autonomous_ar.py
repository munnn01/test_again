from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = load("autonomous_protocol_tests", "ops/autonomous_ar/protocol.py")
RUNNER = load("autonomous_runner_tests", "ops/autonomous_ar/runner.py")


def test_partition_indices_exclude_confirmation_from_training_and_dev(tmp_path, monkeypatch):
    keys = [f"class/clip{i:04}.mp4" for i in range(1010)]
    dev = keys[:208]
    monkeypatch.setattr(PROTOCOL.AUDIT, "audit_index", lambda index: dev)
    monkeypatch.setattr(PROTOCOL, "CONFIRM_FINGERPRINT", PROTOCOL.AUDIT.fingerprint(keys[208:]))
    index = {"train": [{"path": "/dataset/train/clip.mp4"}],
             "val": [{"path": f"/another-mount/{key}", "label": 0} for key in reversed(keys)]}
    manifest = PROTOCOL.make_indices(index, tmp_path)
    assert manifest["dev"]["n"] == 208 and manifest["confirmation"]["n"] == 802
    assert set(manifest["dev"]["clip_ids"]).isdisjoint(manifest["confirmation"]["clip_ids"])
    for stage, n in (("dev", 208), ("confirmation", 802), ("full", 1010)):
        saved = json.loads((tmp_path / f"{stage}.json").read_text())
        assert len(saved["val"]) == n and not saved["test"]
        assert saved["train"] == (index["train"] if stage == "dev" else [])
    index["val"].pop(0)
    with pytest.raises(ValueError, match="fingerprint"):
        PROTOCOL.partitions(index)


def test_generic_scoring_cannot_declare_goal_complete_from_dev(tmp_path, monkeypatch):
    ids = [f"class/clip{i}.mp4" for i in range(12)]
    monkeypatch.setattr(PROTOCOL, "partitions", lambda index: {stage: ids for stage in ("dev", "confirmation", "full")})
    points = tmp_path / "points.csv"
    curves = {}
    with points.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_id", "path", "codec", "qp", "bpp", "top1", "target_prob"])
        for method, ratio in (("h264", 1.0), ("prep+h264", 0.8)):
            curve = {"keys": [], "bpp": [], "accuracy": []}
            for j, qp in enumerate(PROTOCOL.AUDIT.QPS):
                rate, correct = (0.5 - 0.09 * j) * ratio, 10 - 2 * j
                curve["keys"].append(qp); curve["bpp"].append(rate); curve["accuracy"].append(correct / len(ids))
                for i, key in enumerate(ids): writer.writerow([key, f"/mount/{key}", method, qp, rate, int(i < correct), 0.5])
            curves[method] = curve
    rate, accuracy = PROTOCOL.AUDIT._bd_functions()
    a, b = curves["h264"], curves["prep+h264"]
    report = {"task": "action_recognition", "metric": "top1", "n_eval": len(ids), "curves": curves,
              "bd_prep_gain": {"prep+h264 vs h264": {"bd_rate_pct": rate(a["bpp"], a["accuracy"], b["bpp"], b["accuracy"]),
                                                        "bd_accuracy": accuracy(a["bpp"], a["accuracy"], b["bpp"], b["accuracy"])}}}
    for stage in ("dev", "confirmation", "full"):
        result = PROTOCOL.score(report, points, {}, "h264", stage)
        assert result["passes"] and result["bd_rate_pct"] == pytest.approx(-20)
        assert not result["goal_complete"]  # coordinator requires confirmation AND full
    report["n_eval"] += 1
    with pytest.raises(ValueError): PROTOCOL.score(report, points, {}, "h264", "confirmation")


def install_state(monkeypatch, tmp_path, authorized=True):
    monkeypatch.setattr(RUNNER, "STATE", tmp_path / "state.json")
    push = tmp_path / "push"; push.mkdir()
    for name in ("notebook.ipynb", "kernel-metadata.json"): (push / name).write_text("{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    job = {"handle": "qktttttttttt/test", "account": "qktttttttttt", "status": "prepared", "push_dir": str(push),
           "notebook_sha256": sha, "metadata_sha256": sha}
    state = {"jobs": {"test": job}, "external_execution_approval": {"status": "authorized" if authorized else "pending_user", "authorized_jobs": ["test"]}}
    RUNNER.save_state(state)
    return state, push


def test_approval_hold_blocks_network_calls(tmp_path, monkeypatch):
    install_state(monkeypatch, tmp_path, authorized=False)
    monkeypatch.setattr(RUNNER, "kaggle", lambda *a, **k: pytest.fail("network call while approval pending"))
    with pytest.raises(PermissionError): RUNNER.submit(argparse.Namespace(job="test"))


def test_modified_package_blocks_network_calls(tmp_path, monkeypatch):
    _, push = install_state(monkeypatch, tmp_path)
    (push / "notebook.ipynb").write_text("changed")
    monkeypatch.setattr(RUNNER, "kaggle", lambda *a, **k: pytest.fail("network call with modified package"))
    with pytest.raises(ValueError, match="modified"): RUNNER.submit(argparse.Namespace(job="test"))


def test_existing_running_kernel_is_not_pushed_again(tmp_path, monkeypatch):
    install_state(monkeypatch, tmp_path)
    calls = []
    def api(account, args, **kwargs):
        calls.append(args)
        return 0, "KernelWorkerStatus.RUNNING"
    monkeypatch.setattr(RUNNER, "kaggle", api)
    RUNNER.submit(argparse.Namespace(job="test"))
    assert len(calls) == 1 and calls[0][1] == "status"
    assert RUNNER.load_state()["jobs"]["test"]["status"] == "running"


def test_ambiguous_remote_status_does_not_push(tmp_path, monkeypatch):
    install_state(monkeypatch, tmp_path)
    calls = []
    def api(account, args, **kwargs):
        calls.append(args)
        return (1, "timeout") if args[1] == "status" else (1, "unauthorized")
    monkeypatch.setattr(RUNNER, "kaggle", api)
    with pytest.raises(RuntimeError): RUNNER.submit(argparse.Namespace(job="test"))
    assert all(call[1] != "push" for call in calls)


def test_duplicate_prepare_cannot_overwrite_active_package(tmp_path, monkeypatch):
    install_state(monkeypatch, tmp_path)
    state = RUNNER.load_state(); state["jobs"]["test"]["status"] = "running"; RUNNER.save_state(state)
    monkeypatch.setattr(RUNNER, "git", lambda *a: pytest.fail("git invoked before duplicate guard"))
    with pytest.raises(ValueError, match="immutable"):
        RUNNER.prepare(argparse.Namespace(slug="test", replace_prepared=True))


def test_prepared_job_is_not_polled_as_a_missing_remote_kernel(tmp_path, monkeypatch):
    install_state(monkeypatch, tmp_path)
    monkeypatch.setattr(RUNNER, "kaggle", lambda *a, **k: pytest.fail("unsubmitted job must not be polled"))
    RUNNER.poll(argparse.Namespace(job="test", download=True))
    assert RUNNER.load_state()["jobs"]["test"]["status"] == "prepared"


def test_cross_backend_roundoff_does_not_change_acceptance():
    a = {"bd_rate_pct": -1.7593811475790422, "passes": False, "n_eval": 208}
    b = dict(a, bd_rate_pct=-1.7593811475783872)
    assert RUNNER.equivalent_json(a, b)
    assert not RUNNER.equivalent_json(a, dict(b, passes=True))
    assert not RUNNER.equivalent_json(a, dict(b, bd_rate_pct=-1.76))
    assert not RUNNER.equivalent_json(a, dict(b, n_eval=209))
    assert not RUNNER.equivalent_json(a, dict(b, missing_field=None))


def test_stop_request_prevents_new_candidate_preparation(tmp_path, monkeypatch):
    install_state(monkeypatch, tmp_path)
    state = RUNNER.load_state()
    state["stop_request"] = {"create_new_jobs": False, "finish_jobs": ["final"]}
    RUNNER.save_state(state)
    monkeypatch.setattr(RUNNER, "git", lambda *a: pytest.fail("no new candidate after user stop request"))
    with pytest.raises(PermissionError, match="stopping"):
        RUNNER.prepare(argparse.Namespace(slug="next"))


def test_stop_request_prevents_later_submission_even_if_previously_authorized(tmp_path, monkeypatch):
    install_state(monkeypatch, tmp_path)
    state = RUNNER.load_state()
    state["stop_request"] = {"create_new_jobs": False, "finish_jobs": ["final"]}
    RUNNER.save_state(state)
    monkeypatch.setattr(RUNNER, "kaggle", lambda *a, **k: pytest.fail("no later upload after user stop request"))
    with pytest.raises(PermissionError, match="stopping"):
        RUNNER.submit(argparse.Namespace(job="test"))

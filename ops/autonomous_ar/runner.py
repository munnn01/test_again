#!/usr/bin/env python
"""Prepare, dispatch and collect immutable learned-AR experiments on Kaggle."""
from __future__ import annotations
import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "ops/autonomous_ar/state.json"
BASE = "d39927e92298c111d498f0e1a9f72069469ed0b9"
SOURCES = {
    "h264": ("qktttttttttt/crc-v5-h264-v12-source-s282001", "preprocessor_last.pth", "ebee0a43e8a7cb80057972ac34330f1b2f2154b4075786c905d55b07f59dc89a"),
    "h265": ("baooo25r/crc-v5-h265-minus24-candidate-v1", "preprocessor.pth", "a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e"),
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
        "protocol": "autonomous-learned-ar-v14", "goal": "one codec BD-rate <= -10% with task guards on confirmation", "jobs": {}, "heartbeat_id": "t-t-i-u-learned-ar-t-i-bd-rate-10"}


def save_state(state):
    state["updated_at"] = now()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE)


def git(tree: Path, *args):
    cmd = ["git", "-c", f"safe.directory={tree.as_posix()}", "-C", str(tree), *args]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def prepare(args):
    state = load_state()
    if args.slug in state["jobs"]:
        prior = state["jobs"][args.slug]
        if not getattr(args, "replace_prepared", False) or prior["status"] != "prepared":
            raise ValueError("job already registered; active packages are immutable")
        state.setdefault("preparation_history", []).append(dict(prior, retired_reason="local preflight repair before submission"))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        raise ValueError("invalid slug")
    if args.account not in ("qktttttttttt", "baoancut"):
        raise ValueError("account is outside the current authorized allocation")
    if not re.fullmatch(r"configs/[a-z0-9_]+\.yaml", args.config):
        raise ValueError("config must be a simple configs/*.yaml path")
    ref = git(args.tree, "rev-parse", "--verify", f"{args.ref}^{{commit}}").decode().strip()
    changed = git(args.tree, "diff", "--name-only", BASE, ref).decode().splitlines()
    files = {}
    for name in changed:
        if Path(name).parts[0] not in ("src", "configs", "ops", "docs", "tests"):
            raise ValueError(f"unexpected candidate file: {name}")
        files[name] = git(args.tree, "show", f"{ref}:{name}")
    for name in ("ops/report_v13_task_regret.py", "ops/autonomous_ar/protocol.py", "ops/autonomous_ar/__init__.py"):
        files[name] = (ROOT / name).read_bytes()
    if args.config not in files:
        files[args.config] = git(args.tree, "show", f"{ref}:{args.config}")
    source, filename, source_sha = SOURCES[args.codec]
    if args.stage == "confirmation":
        if not args.source_job or not args.source_sha:
            raise ValueError("confirmation requires immutable source_job/final SHA")
        prior = state["jobs"][args.source_job]
        if prior["candidate_ref"] != ref or prior["codec"] != args.codec or not prior.get("development_pass"):
            raise ValueError("confirmation requires a matching development-passing candidate")
        if args.source_sha != prior["checkpoint"]["final_sha256"]:
            raise ValueError("confirmation checkpoint SHA differs from the development artifact")
        if {name: hashlib.sha256(content).hexdigest() for name, content in files.items()} != prior["files"]:
            raise ValueError("confirmation source payload differs from the development artifact")
        source, filename, source_sha = prior["handle"], "preprocessor_last.pth", args.source_sha
    elif args.codec == "h264" and args.account != "qktttttttttt":
        raise ValueError("H264 private source is account-local")
    if not re.fullmatch(r"[a-f0-9]{64}", source_sha):
        raise ValueError("invalid source SHA")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    payload = buffer.getvalue()
    spec = {"slug": args.slug, "handle": f"{args.account}/{args.slug}", "account": args.account,
            "codec": args.codec, "candidate_ref": ref, "config": args.config, "stage": args.stage,
            "seed": args.seed, "steps": args.steps, "source": source, "source_sha256": source_sha,
            "source_job": args.source_job, "base_commit": BASE,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}}
    values = {"__CODEC__": args.codec, "__STAGE__": args.stage, "__CONFIG__": args.config,
              "__SEED__": str(args.seed), "__STEPS__": str(args.steps), "__SLUG__": args.slug,
              "__SOURCE_SLUG__": source.split("/")[-1], "__SOURCE_SHA__": source_sha,
              "__SOURCE_FILENAME__": filename, "__PAYLOAD_SHA__": spec["payload_sha256"],
              "__PAYLOAD_B64__": base64.b64encode(payload).decode(),
              "__SPEC_B64__": base64.b64encode(json.dumps(spec, sort_keys=True).encode()).decode()}
    cell = (ROOT / "kaggle/autonomous_ar_cell.sh").read_text(encoding="utf-8")
    for token, value in values.items():
        cell = cell.replace(token, value)
    if re.search(r"__[A-Z_]+__", cell):
        raise ValueError("unresolved template placeholder")
    push = ROOT / "ops/_push" / args.account / args.slug
    push.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [{"cell_type": "code", "id": args.slug, "execution_count": None,
                           "metadata": {}, "outputs": [], "source": ("%%bash\n" + cell).splitlines(keepends=True)}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    metadata = {"id": spec["handle"], "title": args.slug, "code_file": "notebook.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "dataset_sources": ["qktttttttttt/kineticscleaned"] + ([source] if args.stage == "screen" else []),
                "kernel_sources": [source] if args.stage == "confirmation" else [],
                "competition_sources": [], "model_sources": []}
    (push / "notebook.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (push / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    spec.update({"notebook_sha256": hashlib.sha256((push / "notebook.ipynb").read_bytes()).hexdigest(),
                 "metadata_sha256": hashlib.sha256((push / "kernel-metadata.json").read_bytes()).hexdigest()})
    spec.update({"status": "prepared", "push_dir": str(push), "prepared_at": now(), "node": args.node})
    state["jobs"][args.slug] = spec
    save_state(state)
    print(json.dumps({k: spec[k] for k in ("handle", "codec", "candidate_ref", "payload_sha256", "status")}))


def kaggle(account, arguments, timeout=60):
    pool = json.loads(Path("C:/Users/Admin/Documents/pool.json").read_text(encoding="utf-8"))
    token = pool[account]
    env = dict(os.environ, KAGGLE_API_TOKEN=token, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    result = subprocess.run([sys.executable, "-c", "from kaggle.cli import main; main()", *arguments],
                            env=env, capture_output=True, encoding="utf-8", timeout=timeout)
    output = (result.stdout + result.stderr).replace(token, "[redacted]")
    return result.returncode, output


def status_from(output):
    match = re.search(r"KernelWorkerStatus\.([A-Z]+)", output)
    return match.group(1).lower() if match else "unknown"


def submit(args):
    state = load_state()
    approval = state.get("external_execution_approval", {})
    if approval.get("status") != "authorized" or args.job not in approval.get("authorized_jobs", []):
        raise PermissionError("Kaggle submission awaits explicit user authorization for this package and destination")
    job = state["jobs"][args.job]
    push = Path(job["push_dir"])
    for filename, key in (("notebook.ipynb", "notebook_sha256"), ("kernel-metadata.json", "metadata_sha256")):
        if hashlib.sha256((push / filename).read_bytes()).hexdigest() != job[key]:
            raise ValueError("immutable submission package was modified")
    rc, output = kaggle(job["account"], ["kernels", "status", job["handle"]])
    status = status_from(output)
    if status == "unknown" and not re.search(r"404|not found|does not exist", output, re.I):
        list_rc, listing = kaggle(job["account"], ["kernels", "list", "-m", "--page-size", "200", "--format", "json"])
        try:
            rows = json.loads(listing)
            refs = [row["ref"] for row in rows if row.get("ref")]
            absent = list_rc == 0 and len(rows) < 200 and refs and all(ref.split("/")[0] == job["account"] for ref in refs) and job["handle"] not in refs
        except (ValueError, TypeError, KeyError):
            absent = False
        if not absent:
            raise RuntimeError("cannot establish remote status; refusing an ambiguous duplicate submission: " + output[-400:])
        job["remote_absence_verified_at"] = now()
    if status in ("running", "queued", "pending", "complete"):
        job["status"] = status
        save_state(state)
        print(json.dumps({"handle": job["handle"], "status": status, "duplicate_push_avoided": True}))
        return
    rc, output = kaggle(job["account"], ["kernels", "push", "-p", job["push_dir"], "--accelerator", "NvidiaTeslaT4"], timeout=90)
    job["status"] = "submitted" if rc == 0 else "submit_failed"
    job["submit_message"] = output[-2000:]
    job["submitted_at"] = now()
    save_state(state)
    print(json.dumps({"handle": job["handle"], "rc": rc, "status": job["status"], "message": output[-1200:]}))
    if rc:
        raise SystemExit(rc)


def poll(args):
    state = load_state()
    selected = [state["jobs"][args.job]] if args.job else list(state["jobs"].values())
    for job in selected:
        if job["status"] == "evaluated":
            print(json.dumps({"handle": job["handle"], "status": "evaluated", "development_pass": job.get("development_pass"), "goal_complete": job.get("goal_complete", False)}))
            continue
        rc, output = kaggle(job["account"], ["kernels", "status", job["handle"]])
        status = status_from(output)
        job["status"] = status if rc == 0 else "status_error"
        job["last_poll"] = now()
        if status == "complete" and args.download:
            destination = ROOT / "outputs/autonomous_ar/downloads" / job["slug"]
            rc, download_output = kaggle(job["account"], ["kernels", "output", job["handle"], "-p", str(destination)], timeout=180)
            if rc:
                job["download_error"] = download_output[-1000:]
            else:
                run = destination / "outputs/autonomous_ar" / job["slug"]
                summary_path = run / "summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    spec = json.loads((run / "job_spec.json").read_text())
                    if spec["payload_sha256"] != job["payload_sha256"] or spec["candidate_ref"] != job["candidate_ref"]:
                        raise ValueError("downloaded artifact does not match immutable job")
                    job.update({"status": "evaluated", "development_pass": summary["development_pass"],
                                "goal_complete": summary["goal_complete"], "scores": summary["scores"],
                                "checkpoint": json.loads((run / "checkpoint_manifest.json").read_text()),
                                "download_dir": str(destination)})
                else:
                    job["status"] = "artifact_incomplete"
                    job["download_dir"] = str(destination)
        save_state(state)
        print(json.dumps({"handle": job["handle"], "status": job["status"],
                          "development_pass": job.get("development_pass"), "goal_complete": job.get("goal_complete", False),
                          "scores": job.get("scores", {}), "error": output[-600:] if rc else ""}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--account", choices=("qktttttttttt", "baoancut"), required=True)
    p.add_argument("--codec", choices=("h264", "h265"), required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--node", required=True)
    p.add_argument("--stage", choices=("screen", "confirmation"), default="screen")
    p.add_argument("--source-job")
    p.add_argument("--source-sha")
    p.add_argument("--replace-prepared", action="store_true", help="Rebuild only a never-submitted prepared job after local preflight repair")
    p = sub.add_parser("submit")
    p.add_argument("--job", required=True)
    p = sub.add_parser("poll")
    p.add_argument("--job")
    p.add_argument("--download", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "submit": submit, "poll": poll}[args.command](args)


if __name__ == "__main__":
    main()

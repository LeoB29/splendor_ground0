"""Explicit check/prepare/probe/main stages; never starts from a waiting monitor."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.eval import student_generalization_panel as h
from splendor_ai.alphazero import windowless_workers_160 as windows

ROOT = Path(__file__).resolve().parents[3]
PARENT_BINDING = "b5fa68a0e71fd0632e3d2532fd929cea7de6306cab9881d6c37e8f57bd30a51c"
EXTRAS = ("run_student_generalization_panel.py", "scripts/analysis/verify_student_generalization_panel.py",
          "scripts/analysis/check_teacher_batching.py", "tests/test_student_generalization_panel.py",
          "tests/test_teacher_batching.py", "docs/engineering/parallel_tools_after_162.md")


def identity():
    paths = sorted(Path("src").rglob("*.py")) + [Path(p) for p in EXTRAS]
    # Include replay dependencies outside src, including their transitive imports.
    paths += sorted(Path("scripts/analysis").glob("verify_*.py"))
    return {"dependencies": {p.as_posix(): h.old.file_hash(p) for p in sorted(set(paths))},
            "config": h.CONFIG, "shallow": h.SHALLOW, "python": sys.version,
            "torch": str(torch.__version__), "numpy": np.__version__,
            "platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count()}


def verified(item):
    h.require(h.old.file_hash(Path(item["path"])) == item["sha256"], f"Changed artifact: {item['path']}")
    return h.old.read_json(Path(item["path"]))


def parent_evidence():
    base = h.parent.OUTPUT
    status_path = base / "final_status.json"
    h.require(status_path.exists(), "Experiment162 must finish before preparing or running this panel")
    status = h.old.read_json(status_path)
    h.require(status.get("complete") is True and status.get("all_prescribed_audits_pass") is True,
              "Experiment162 is not complete and audited")
    h.require(status.get("manifest") == PARENT_BINDING, "Unexpected162 binding")
    from splendor_ai.alphazero import run_student_refresh
    h.require(run_student_refresh.validate_freeze() == PARENT_BINDING, "162 frozen dependencies changed")
    result = verified(status["result"])
    verified_report = Path(status["report"]["path"])
    h.require(h.old.file_hash(verified_report) == status["report"]["sha256"], "162 report changed")
    h.require(result.get("manifest") == PARENT_BINDING and result.get("all_prescribed_audits_pass") is True,
              "Invalid162 result")
    selected_path = base / "selected_lock.json"
    selected = h.old.read_json(selected_path)
    h.require(selected["manifest"] == PARENT_BINDING, "Invalid162 model selection")
    expected = {f"{arm}_{seed}" for arm in h.ARMS for seed in h.SEEDS}
    h.require(set(selected["models"]) == expected, "Missing or unexpected fit")
    for item in selected["models"].values():
        h.require(h.old.file_hash(Path(item["path"])) == item["sha256"], "Model changed")
    evidence = [h.parent.artifact(status_path), status["result"], status["report"],
                h.parent.artifact(selected_path)]
    for key in ("fits_lock", "fits_audit_lock"):
        verified(selected[key])
        evidence.append(selected[key])
    for seed in h.SEEDS:
        h.require(selected["models"][f"anchor_{seed}"] == h.parent.initializer(seed), "Wrong161 anchor")
        for arm in ("refresh", "replay"):
            folder = base / "fits" / f"{arm}_{seed}"
            fit_path, audit_path = folder / "result.json", folder / "audit.json"
            fit, audit = h.old.read_json(fit_path), h.old.read_json(audit_path)
            h.require(audit["complete_fit_bitwise_identical"] is True and
                      audit["result_sha256"] == h.old.file_hash(fit_path), "Fit audit changed")
            chosen = min(fit["curves"], key=lambda r: (r["pair_loss"], r["update"]))
            item = selected["models"][f"{arm}_{seed}"]
            h.require(chosen["update"] == fit["selected_update"] == audit["selected"] == item["update"] and
                      chosen["checkpoint_sha256"] == item["sha256"], "Wrong selected checkpoint")
            evidence.extend((h.parent.artifact(fit_path), h.parent.artifact(audit_path)))
    h.require(h.old.file_hash(h.CHECKPOINT_023) == h.SHA_023, "023 changed")
    h.require(h.old.file_hash(h.old.CHECKPOINT) == h.old.CHECKPOINT_SHA, "035 changed")
    return {"models": selected["models"], "files": evidence,
            "opponent_checkpoints": [h.parent.artifact(h.CHECKPOINT_023), h.parent.artifact(h.old.CHECKPOINT)],
            "interpretation162": h.decision_map(result)}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    code = ("import torch,pytest; torch.set_num_threads(1); "
            "raise SystemExit(pytest.main(['-q','tests/test_student_generalization_panel.py',"
            "'tests/test_teacher_batching.py']))")
    start = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as stdout, (folder / "stderr.txt").open("x", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    h.require(before == identity(), "Code changed during checks")
    h.old.save_new(folder / "checks.json", {"identity": before, "exit_code": result.returncode,
                   "scope": "new isolated panel and batching tools", "seconds": time.perf_counter()-start,
                   "logs": [h.parent.artifact(folder / name) for name in ("stdout.txt", "stderr.txt")]})
    return result.returncode


def prepare(checks_path, base=h.OUTPUT):
    checks_path, base = Path(checks_path), Path(base)
    h.require(not (base / "manifest.json").exists(), "Panel already prepared")
    inputs = parent_evidence()  # fail before creating production artifacts if162 still runs
    checks = h.old.read_json(checks_path)
    h.require(checks["identity"] == identity() and checks["exit_code"] == 0, "Current checks required")
    for item in checks["logs"]:
        h.require(h.old.file_hash(Path(item["path"])) == item["sha256"], "Check log changed")
    manifest = {"identity": identity(), "parent": inputs, "checks": h.parent.artifact(checks_path),
                "schedules": {stage: h.schedule(stage) for stage in ("probe", "main")}}
    h.old.save_new(base / "manifest.json", manifest)
    h.old.save_new(base / "freeze.json", {"manifest": h.parent.artifact(base / "manifest.json"),
                                          "binding": h.old.digest(manifest)})
    return h.old.digest(manifest)


def validate(base=h.OUTPUT):
    freeze = h.old.read_json(Path(base) / "freeze.json")
    h.require(Path(freeze["manifest"]["path"]).resolve() == (Path(base) / "manifest.json").resolve(), "Wrong manifest path")
    manifest = verified(freeze["manifest"])
    h.require(h.old.digest(manifest) == freeze["binding"], "Panel manifest changed")
    h.require(manifest["identity"] == identity(), "Panel code/runtime changed after preparation")
    verified(manifest["checks"])
    for item in manifest["parent"]["files"] + manifest["parent"]["opponent_checkpoints"]:
        h.require(h.old.file_hash(Path(item["path"])) == item["sha256"], "Parent evidence changed")
    for item in manifest["parent"]["models"].values():
        h.require(h.old.file_hash(Path(item["path"])) == item["sha256"], "Selected model changed")
    h.require(manifest["schedules"] == {s: h.schedule(s) for s in ("probe", "main")}, "Schedule changed")
    return manifest, h.old.digest(manifest)


def worker_init(models, logs):
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(logs / f"{os.getpid()}.out", logs / f"{os.getpid()}.err")
    h.initialize(models)


def play_and_audit(stage, spec, binding, base):
    from scripts.analysis.verify_student_generalization_panel import replay_game
    game = h.play_game(stage, spec, binding, base)
    return replay_game(stage, spec, game, binding, base)


def verify_audits(items, stage, binding, base):
    specs = h.schedule(stage)
    h.require(len(items) == len(specs) and {r["index"] for r in items} == set(range(len(specs))),
              "Missing or duplicate replay receipts")
    reports = []
    for item in sorted(items, key=lambda r: r["index"]):
        spec = specs[item["index"]]
        report = verified(item)
        h.require(report["manifest"] == binding and report["full_replay"] is True, "Invalid replay")
        h.require(all(report["game"].get(k) == v for k, v in spec.items()), "Replay schedule mismatch")
        h.require(Path(report["game_artifact"]["path"]).resolve() == h.game_path(stage, spec["index"], base).resolve(),
                  "Wrong replayed game path")
        verified(report["game_artifact"])
        reports.append(report)
    return reports


def stage_run(stage, base=h.OUTPUT):
    h.require(stage in ("probe", "main"), "Unknown stage")
    base = Path(base)
    with h.old.process_lock(base / "run.lock"):
        manifest, binding = validate(base)
        h.require(not (base / f"{stage}_launch.json").exists(), "Stage already attempted; no automatic retry")
        if stage == "main":
            preflight = h.old.read_json(base / "probe_complete.json")
            h.require(preflight["manifest"] == binding, "Timing probe required first")
            verify_audits(preflight["audits"], "probe", binding, base)
        windows.configure_spawn()
        h.old.save_new(base / f"{stage}_launch.json", {"manifest": binding, "time": h.old.now(), "pid": os.getpid()})
        started, items = time.perf_counter(), []
        try:
            with ProcessPoolExecutor(max_workers=h.CONFIG["workers"], initializer=worker_init,
                                     initargs=(manifest["parent"]["models"], base / stage / "workers")) as pool:
                futures = [pool.submit(play_and_audit, stage, s, binding, base) for s in h.schedule(stage)]
                try:
                    for future in as_completed(futures):
                        items.append(future.result())
                        print(json.dumps({"stage": stage, "completed": len(items), "total": len(futures)}), flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            items.sort(key=lambda r: r["index"])
            h.require(validate(base)[1] == binding, "Binding changed during stage")
            reports = verify_audits(items, stage, binding, base)
            elapsed = time.perf_counter()-started
            completion = {"manifest": binding, "audits": items, "elapsed_seconds": elapsed}
            if stage == "probe":
                # No probe scores or outcomes affect design, sample size, or candidate choice.
                completion["estimated_main_hours"] = elapsed*h.CONFIG["main_clusters"]/3600
            else:
                stats = h.statistics([r["game"] for r in reports])
                timing = defaultdict(list)
                for report in reports:
                    for row in report["timings"]:
                        timing[row["model"]].append(row)
                completion["statistics"] = stats
                completion["timings"] = {key: {kind: {"mean": float(np.mean([r[kind] for r in rows])),
                        "p95": float(np.percentile([r[kind] for r in rows], 95)), "count": len(rows)}
                        for kind in ("wall", "cpu")} for key, rows in sorted(timing.items())}
            h.old.save_new(base / f"{stage}_complete.json", completion)
            if stage == "main":
                write_report(completion, base)
            return completion
        except BaseException:
            h.old.save_new(base / f"{stage}_failure.json", {"manifest": binding, "error": traceback.format_exc()})
            raise


def write_report(result, base):
    lines = ["# Student generalization panel", "", "All 5,400 games replayed. Local opponents; no promotion gate.", ""]
    for name, comparison in result["statistics"]["comparisons"].items():
        lo, hi = comparison["ci95"]
        lines.append(f"{name}: {100*comparison['delta']:.3f} percentage points; paired95% interval [{100*lo:.3f}, {100*hi:.3f}].")
    lines += ["", "Full opponent, fit, seat and timing breakdowns: main_complete.json.",
              "Intervals condition on these three fixed fits and these three opponents; subgroup intervals are descriptive.",
              "No evidence of human superiority or equal-time search superiority is inferred."]
    with (Path(base) / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "prepare", "probe", "main", "launch-probe", "launch-main"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "plan":
        print(json.dumps({"main_games": len(h.schedule("main")), "probe_games": len(h.schedule("probe")),
                          "parent_complete_file_present": (h.parent.OUTPUT / "final_status.json").exists(),
                          "automatic_launch": False, "config": h.CONFIG}, indent=2))
    elif args.command == "check":
        h.require(args.checks is not None, "Supply a new --checks directory")
        return check(args.checks)
    elif args.command == "prepare":
        h.require(args.checks is not None, "Supply --checks path/to/checks.json")
        print(prepare(args.checks))
    elif args.command.startswith("launch-"):
        _, binding = validate()
        stage = args.command.removeprefix("launch-")
        from splendor_ai.alphazero.run_student_refresh import held_launch
        log = h.OUTPUT / f"launch_{stage}"
        with h.old.process_lock(log / "launcher.lock"):
            h.require(not (h.OUTPUT / f"{stage}_launch.json").exists() and
                      not (log / "process.json").exists(), "Stage already attempted")
            if stage == "main":
                preflight = h.old.read_json(h.OUTPUT / "probe_complete.json")
                h.require(preflight["manifest"] == binding, "Timing probe required first")
                verify_audits(preflight["audits"], "probe", binding, h.OUTPUT)
            # Keep the direct parent alive through the child's process checks.
            print(held_launch([sys.executable, ROOT / "run_student_generalization_panel.py", stage],
                              log, h.OUTPUT / f"{stage}_launch.json", binding, cwd=ROOT))
    else:
        if Path(sys.executable).name.lower() == "pythonw.exe":
            windows.assert_windowless_worker(h.OUTPUT / f"{args.command}.out", h.OUTPUT / f"{args.command}.err")
        stage_run(args.command)
    return 0

"""Prepare a saved-root batching benchmark without loading models or playing games.

Only the explicit equivalence/timing commands import the deferred runtime.
No watcher, queue, automatic start, or production integration is installed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "outputs/learned_proposal_arena"
ACTIVE = "outputs/learned_proposal_attribution"
SOURCE_BINDING = "2154ba0b27284ae6077300376e6f4694254ab84f8826be8ce992d284add59a81"
ACTIVE_BINDING = "0f2772aebd99e2fef939dc1a9464790daa1f6c3b1a8e9fe1326a2361a85e44e2"
OUTPUT = "outputs/batching_benchmark_v2"
CODE = ["tools/batching_benchmark_v2/benchmark.py", "tools/batching_benchmark_v2/runtime.py",
        "tools/batching_benchmark_v2/test_benchmark.py", "docs/engineering/batching_benchmark_v2.md"]
CONFIG = {"selection_tag": "batching-readiness-v1", "worlds": 32,
          "batch_sizes": [1, 4, 16], "primary_batch_size": 16,
          "timing_repetitions": 4, "maximum_stage_hours": 6,
          "device": "cpu", "workers": 1, "torch_threads": 1,
          "automatic_start": False, "production_enabled": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def path(name):
    target = (ROOT / name).resolve()
    require(target.is_relative_to(ROOT), "Path outside repository")
    return target


def file_hash(target):
    with Path(target).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def artifact(target):
    target = Path(target).resolve()
    return {"path": target.relative_to(ROOT).as_posix(), "sha256": file_hash(target)}


def read(target):
    return json.loads(Path(target).read_text(encoding="utf-8"))


def verified(item):
    target = path(item["path"])
    require(file_hash(target) == item["sha256"], f"Changed artifact: {item['path']}")
    return read(target)


def save_new(target, value):
    target = Path(target)
    require(target.resolve().is_relative_to(path(OUTPUT)), "Output outside benchmark directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def code_identity():
    return {name: file_hash(path(name)) for name in CODE}


def active_identity():
    """Read-only verification; no Torch imports or live strength-output reads."""
    freeze = read(path(f"{ACTIVE}/freeze.json"))
    manifest = verified(freeze["manifest"])
    require(digest(manifest) == freeze["binding"] == ACTIVE_BINDING, "Active study binding changed")
    dependencies = manifest["identity"]["dependencies"]
    inventory = {p.relative_to(ROOT).as_posix() for p in path("src").rglob("*.py")}
    for folder in ("scripts/analysis", "scripts/experiments"):
        inventory.update(p.relative_to(ROOT).as_posix() for p in path(folder).glob("*.py"))
    expected = {name for name in dependencies if name.startswith(("src/", "scripts/analysis/", "scripts/experiments/"))}
    require(inventory == expected, "Active study Python inventory changed")
    for name, sha in dependencies.items():
        require(file_hash(path(name)) == sha, f"Active study dependency changed: {name}")
    for item in manifest["parent"]["files"]:
        require(file_hash(path(item["path"])) == item["sha256"], f"Parent dependency changed: {item['path']}")
    return {"binding": freeze["binding"], "manifest": freeze["manifest"],
            "dependencies_verified": len(dependencies), "parent_files_verified": len(manifest["parent"]["files"])}


def root_key(row):
    return f"game_{row['index']:04d}/turn_{row['turn']:03d}"


def metadata(row):
    # Explicit allowlist: choices, outcomes, runtime and branch lengths cannot
    # influence selection even though the completed receipts also contain them.
    value = {k: row[k] for k in ("index", "turn", "seat", "planner", "support", "privacy", "sha256")}
    require(value["seat"] in (0, 1) and value["planner"] in ("candidate", "reference"), "Invalid source actor")
    require(1 <= value["support"] <= 4 and 0 <= value["turn"] < 400, "Invalid source support/turn")
    return value


def cell(row):
    phase = 0 if row["turn"] < 20 else 1 if row["turn"] < 40 else 2
    return (phase, row["seat"], row["planner"], int(any(row["privacy"].values())))


def rank(row):
    return digest([CONFIG["selection_tag"], row["index"], row["turn"]])


def select_roots(rows):
    rows = [metadata(row) for row in rows]
    require(len({root_key(row) for row in rows}) == len(rows), "Duplicate source root")
    ordered = sorted(rows, key=lambda row: (rank(row), root_key(row)))
    selected, missing = [], []
    for group in itertools.product(range(3), (0, 1), ("candidate", "reference"), (0, 1)):
        candidates = [row for row in ordered if cell(row) == group]
        if not candidates:
            missing.append(list(group))
        else:
            selected.append({**candidates[0], "purpose": "primary"})
    # Natural short-support cases are stress checks, never replacements for
    # missing primary cells and never included in the throughput aggregate.
    selected_ids = {root_key(row) for row in selected}
    absent_supports = []
    for support in (1, 2, 3):
        candidates = [row for row in ordered if row["support"] == support]
        if not candidates:
            absent_supports.append(support)
        elif not any(row["support"] == support for row in selected):
            extra = next(row for row in candidates if root_key(row) not in selected_ids)
            selected.append({**extra, "purpose": "short_support"})
            selected_ids.add(root_key(extra))
    # One fixed timing root per phase/actor/planner, independent of outcomes.
    timing = []
    for group in itertools.product(range(3), (0, 1), ("candidate", "reference")):
        candidates = [row for row in selected if row["purpose"] == "primary" and cell(row)[:3] == group]
        if candidates:
            timing.append(root_key(min(candidates, key=rank)))
    return selected, {"missing_primary_cells": missing, "absent_short_supports": absent_supports,
                      "timing_roots": timing}


def timing_schedule(root_ids):
    require(len(set(root_ids)) == len(root_ids), "Duplicate timing root")
    schedule = []
    for repetition in range(CONFIG["timing_repetitions"]):
        for index, key in enumerate(root_ids):
            # ABBA/BAAB over repetitions, and alternating first arm across roots.
            reverse = (repetition in (1, 2)) ^ bool(index % 2)
            modes = ["batch16", "serial"] if reverse else ["serial", "batch16"]
            schedule.append({"root": key, "repetition": repetition, "order": modes})
    return schedule


def check(output):
    output = path(output)
    require(not output.exists() and output.is_relative_to(path(OUTPUT)), "Choose a new checks folder")
    output.mkdir(parents=True)
    identity = code_identity()
    command = [sys.executable, "-B", "-m", "unittest", "tools.batching_benchmark_v2.test_benchmark", "-v"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    require(identity == code_identity(), "Code changed during checks")
    receipt = {"time": now(), "passed": result.returncode == 0, "returncode": result.returncode,
               "code": identity, "command": command, "python": sys.version,
               "logs": [artifact(output / name) for name in ("stdout.txt", "stderr.txt")]}
    save_new(output / "checks.json", receipt)
    require(receipt["passed"], "Preparation checks failed; evidence retained")
    print({"passed": True, "checks": str(output / "checks.json")})


def prepare(checks_file, output):
    output = path(output)
    require(not output.exists() and output.is_relative_to(path(OUTPUT)), "Choose a new preparation folder")
    checks = read(path(checks_file))
    require(checks["passed"] and checks["code"] == code_identity(), "Current passing checks required")
    for item in checks["logs"]:
        require(file_hash(path(item["path"])) == item["sha256"], "Changed check log")
    active = active_identity()
    final = read(path(f"{SOURCE}/final_status.json"))
    require(final["complete"] and final["all_prescribed_audits_pass"] and final["full_inner_replay"]
            and final["manifest"] == SOURCE_BINDING, "Source not completely audited")
    audit = verified(final["audit"])
    require(audit["passed"] and audit["full_inner_replay"] and audit["manifest"] == SOURCE_BINDING, "Invalid source audit")
    complete = verified(audit["main"])
    games, audits = verified(complete["game_lock"]), verified(complete["audit_lock"])
    require(len(games["files"]) == len(audits["files"]) == complete["games"] == 402, "Incomplete source locks")
    game_map = {item["index"]: item for item in games["files"]}
    require(set(game_map) == set(range(402)), "Invalid source game schedule")
    rows, receipts, provenance = [], {}, []
    for item in audits["files"]:
        receipt = verified(item)
        require(receipt["passed"] and receipt["full_inner_replay"] and receipt["manifest"] == SOURCE_BINDING,
                "Invalid source receipt")
        index = item["index"]
        require(index not in receipts and receipt["game_artifact"] == game_map[index], "Source audit/game mismatch")
        receipts[index] = item
        for row in receipt["roots"]:
            require(row["index"] == index and row["continuations"] == row["replayed_branches"] == 32*row["support"],
                    "Incomplete source root replay")
            rows.append(metadata(row))
        provenance.append(item)
    selected, coverage = select_roots(rows)
    require(not coverage["missing_primary_cells"] and len(coverage["timing_roots"]) == 12,
            "Missing primary strata; preserve design and review instead of substituting")
    entries, widths = [], []
    for row in selected:
        key = root_key(row)
        root_item = {"path": f"{SOURCE}/main/roots/{key}.json", "sha256": row["sha256"]}
        root = verified(root_item)
        game_item = game_map[row["index"]]
        game = verified(game_item)
        move = next(move for move in game["trace"] if move["turn"] == row["turn"])
        require(move["root_sha256"] == row["sha256"] and move["state_sha256"] == digest(root["snapshot"])
                and root["manifest"] == game["manifest"] == SOURCE_BINDING and "failure" not in root,
                "Source root/game mismatch")
        require(root["privacy"] == row["privacy"] and root["snapshot"]["current_player"] == row["seat"]
                and root["snapshot"]["turn_index"] == row["turn"]
                and len(root["worlds"]) == len(root["schedule"]["world_seeds"]) == 32
                and len(root["candidates"]["actions"]) == row["support"], "Source metadata mismatch")
        widths.append(root["candidates"]["legal_support"])
        entries.append({"id": key, "metadata": row, "root": root_item, "game": game_item,
                        "audit": receipts[row["index"]], "spec": game["spec"],
                        "legal_support": widths[-1], "jobs": 32*row["support"],
                        "forced_action_types": [a["action_type"] for a in root["candidates"]["actions"]]})
    coverage.update(source_roots=len(rows), selected_roots=len(entries),
                    primary_roots=sum(row["purpose"] == "primary" for row in selected),
                    legal_support_min=min(widths), legal_support_max=max(widths),
                    support_counts=dict(Counter(str(row["support"]) for row in selected)),
                    privacy_counts=dict(Counter(f"{int(row['privacy']['own']>0)}/{int(row['privacy']['opponent']>0)}" for row in selected)))
    plan = {"version": 1, "time": now(), "config": CONFIG, "code": code_identity(),
            "checks": artifact(path(checks_file)), "active": active, "source_binding": SOURCE_BINDING,
            "source_final": artifact(path(f"{SOURCE}/final_status.json")), "source_audit": final["audit"],
            "source_main": audit["main"], "source_locks": [complete["game_lock"], complete["audit_lock"]],
            "source_receipts": provenance, "roots": entries, "coverage": coverage,
            "timing_schedule": timing_schedule(coverage["timing_roots"]),
            "execution_started": False, "performance_measured": False, "automatic_start": False}
    require(active_identity() == active, "Active identity changed during preparation")
    save_new(output / "plan.json", plan)
    save_new(output / "prepared.json", {"time": now(), "plan": artifact(output / "plan.json"),
                                       "binding": digest(plan), "prepared": True, "execution_started": False})
    print({"prepared": True, "output": str(output), "coverage": coverage, "execution_started": False})


def load_plan(folder):
    seal = read(path(folder) / "prepared.json")
    plan = verified(seal["plan"])
    require(digest(plan) == seal["binding"] and plan["code"] == code_identity()
            and plan["config"] == CONFIG, "Benchmark preparation changed")
    require(plan["active"] == active_identity(), "Active dependencies changed")
    for item in [plan["checks"], plan["source_final"], plan["source_audit"], plan["source_main"],
                 *plan["source_locks"], *plan["source_receipts"]]:
        verified(item)
    for entry in plan["roots"]:
        for key in ("root", "game", "audit"):
            verified(entry[key])
    return plan, seal["binding"]


def validate_ready_status(status):
    require(status.get("complete") is True and status.get("all_prescribed_audits_pass") is True
            and status.get("full_inner_replay") is True and status.get("manifest") == ACTIVE_BINDING,
            "Current experiment must finish all prescribed audits before benchmark execution")


def blocking_processes(recorded_pids, live_rows, current_pid):
    """Hold on possible old Python workers; recognize definite PID recycling."""
    blocking = []
    for row in live_rows:
        pid = row.get("ProcessId")
        if pid not in recorded_pids or pid == current_pid:
            continue
        name, executable = row.get("Name"), row.get("ExecutablePath")
        definite_other = (isinstance(name, str) and isinstance(executable, str)
                          and PureWindowsPath(executable).is_absolute()
                          and PureWindowsPath(executable).name.casefold() == name.casefold()
                          and name.casefold().endswith(".exe")
                          and "python" not in name.casefold())
        if not definite_other:
            blocking.append(pid)
    return sorted(blocking)


def require_idle():
    # No model imports or output creation until the completed-study guard passes.
    # A definitely different executable can reuse an old PID; ambiguous identities
    # remain a hold. This check never terminates any process.
    target = path(f"{ACTIVE}/final_status.json")
    require(target.exists(), "Current experiment has not finished; benchmark remains unstarted")
    validate_ready_status(read(target))
    require(os.name == "nt", "This prepared workload requires the recorded Windows host")
    records = [path(f"{ACTIVE}/launch.json"), *path(f"{ACTIVE}/workers").rglob("*_health.json")]
    pids = {read(item)["pid"] for item in records if item.exists()}
    command = "@(Get-CimInstance Win32_Process | Select-Object ProcessId,Name,ExecutablePath) | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    live = json.loads(result.stdout)
    require(isinstance(live, list), "Invalid process inventory; defer execution")
    require(not blocking_processes(pids, live, os.getpid()),
            "Recorded current-study supervisor/workers still alive or ambiguous; defer execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checks = commands.add_parser("check")
    checks.add_argument("--output", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--checks", required=True)
    preparation.add_argument("--output", required=True)
    for name in ("verify", "equivalence", "timing"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        if name == "timing":
            command.add_argument("--equivalence", required=True)
        if name != "verify":
            command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "check":
        check(args.output)
    elif args.command == "prepare":
        prepare(args.checks, args.output)
    elif args.command == "verify":
        plan, binding = load_plan(args.plan)
        print({"verified": True, "binding": binding, "roots": len(plan["roots"]), "execution_started": False})
    else:
        require_idle()
        from tools.batching_benchmark_v2 import runtime
        runtime.execute(args.command, args.plan, args.output, getattr(args, "equivalence", None))


if __name__ == "__main__":
    main()

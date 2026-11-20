"""One frozen arithmetic-only sampling screen on existing Experiment 160 evidence.

No game or model imports. Commands are explicit: check, freeze, run, finalize.
The independent verifier must run between run and finalize. Existing artifacts
are never overwritten; no simulation, training, retry or successor is scheduled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/saved_halving_screen"
SOURCE = ROOT / "outputs/world_budget_160"
SOURCE_BINDING = "5e9e9202f44e7b526fff4be97cdcd3186805fbb5e8a184d21a6702de9d922df3"
SOURCE_RESULT_SHA = "2e3efa2c524f8d8c68284fad0f3400365bcb08f4b598aecf121039a4c45b19e5"
SOURCE_ROOTS_SHA = "3c762d22dfcd412239c1f485826fe6ef72e802a5b7f340d2732fa1e305763755"
FILES = (
    "scripts/analysis/saved_halving_screen.py",
    "scripts/analysis/verify_saved_halving_screen.py",
    "tests/test_saved_halving_screen.py",
    "docs/engineering/saved_halving_screen.md",
    "docs/strategy/next_after_generalization_panel.md",
)
CONFIG = {
    "worlds": 128, "partitions": 64, "permutation_seed_start": 2026091700,
    "bootstrap_seed": 2026091764, "bootstrap_resamples": 50000,
    "pairs": 200, "roots": 600, "initial_worlds": 16,
    "additional_worlds": 32, "reference_worlds": 32, "evaluation_worlds": 80,
    "short_support": "uniform32", "ties": "original_candidate_index",
    "primary": "adaptive_minus_uniform_heldout_terminal_score", "score": "(wdl+1)/2",
}
IDENTITY = ("pair", "phase", "source_index", "actor", "turn", "source_path",
            "source_sha256", "support", "game_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def local(path):
    resolved = (ROOT / path).resolve()
    require(resolved.is_relative_to(ROOT), "Path outside repository")
    return resolved


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def check_hashes(inventory):
    require(bool(inventory), "Empty hash inventory")
    for path, expected in inventory.items():
        require(sha(local(path)) == expected, f"Bound artifact changed: {path}")


def check():
    require(not (OUTPUT / "freeze.json").exists(), "Already frozen")
    OUTPUT.mkdir(exist_ok=True)
    checks = OUTPUT / "checks"
    checks.mkdir()  # Failure records are retained; never silently rerun over them.
    identities = {path: sha(ROOT / path) for path in FILES}
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "-q", "--noconftest",
               "tests/test_saved_halving_screen.py"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    (checks / "stdout.txt").write_bytes(result.stdout)
    (checks / "stderr.txt").write_bytes(result.stderr)
    check_hashes(identities)
    save(checks / "checks.json", {
        "passed": result.returncode == 0, "returncode": result.returncode,
        "files": identities, "command": command, "time": now(),
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": sha(checks / "stdout.txt"),
        "stderr_sha256": sha(checks / "stderr.txt"),
    })
    print(result.stdout.decode(errors="replace"), end="", flush=True)
    require(result.returncode == 0, "Focused checks failed")


def source_inventory():
    """Verify hashes and outcome-blind identities only; compute no comparison."""
    inventory = {}

    def load(path, expected=None):
        actual = sha(path)
        require(expected is None or actual == expected, f"Source hash: {path}")
        inventory[relative(path)] = actual
        return read(path)

    final = load(SOURCE / "final_status.json")
    result = load(SOURCE / "result.json", SOURCE_RESULT_SHA)
    manifest = load(SOURCE / "manifest.json")
    require(digest(manifest) == SOURCE_BINDING == final["manifest"] == result["manifest"],
            "Source binding")
    require(final["complete"] and final["prespecified_audit_pass"]
            and final["result_sha256"] == SOURCE_RESULT_SHA, "Source incomplete")
    for path, expected in manifest["dependencies"].items():
        require(sha(local(path)) == expected, f"Source dependency changed: {path}")
        inventory[path] = expected
    audit_lock = load(SOURCE / "main_audit_lock.json", result["main_audit_lock_sha256"])
    game_lock = load(SOURCE / "main_collect_lock.json", audit_lock["source_lock_sha256"])
    old_summary = load(SOURCE / "review_sampling_20260916/summary.json")
    roots = load(SOURCE / "review_sampling_20260916/roots.json", SOURCE_ROOTS_SHA)
    require(old_summary["roots_sha256"] == SOURCE_ROOTS_SHA
            and old_summary["source_result_sha256"] == SOURCE_RESULT_SHA
            and old_summary["source_game_lock_sha256"] == sha(SOURCE / "main_collect_lock.json"),
            "Old review identity")
    review_script = ROOT / "scripts/analysis/review_sampling_after_160.py"
    require(sha(review_script) == old_summary["script_sha256"], "Original review script changed")
    inventory[relative(review_script)] = sha(review_script)
    require(len(roots) == 600, "Source root count")
    expected_roots = {}
    for lock in (audit_lock, game_lock):
        require(lock["manifest"] == SOURCE_BINDING
                and set(lock["files"]) == {str(i) for i in range(400)}, "Source lock coverage")
    for index in range(400):
        pair, seat = divmod(index, 2)
        game = load(SOURCE / "main/games" / f"game_{index:04d}.json", game_lock["files"][str(index)])
        audit = load(SOURCE / "main/audits" / f"game_{index:04d}.json", audit_lock["files"][str(index)])
        require(game["manifest"] == audit["manifest"] == SOURCE_BINDING
                and audit["game_sha256"] == game_lock["files"][str(index)]
                and "failure" not in game and "failure" not in audit, "Source audit identity")
        spec = game["spec"]
        require(spec == {"index": index, "pair": pair, "candidate_seat": seat,
                         "environment_seed": 1600000000 + pair,
                         "search_stream": digest(["160-root-stream-v1", "main", pair])},
                "Source game spec")
        assigned = int(digest(["post160-review-source-seat-v1", pair]), 16) % 2
        if assigned != seat:
            continue
        found = set()
        for move in game["trace"]:
            phase = min(move["turn"] // 20, 2)
            if move["actor"] != seat or phase in found:
                continue
            found.add(phase)
            path = SOURCE / "main/roots" / f"game_{index:04d}" / f"turn_{move['turn']:03d}.json"
            # Freeze the source bytes without computing anything from outcomes.
            require(sha(path) == move["root_sha256"], "Selected source root hash")
            inventory[relative(path)] = move["root_sha256"]
            expected_roots[(pair, phase)] = {
                "pair": pair, "phase": phase, "source_index": index, "actor": seat,
                "turn": move["turn"], "source_path": relative(path),
                "source_sha256": move["root_sha256"],
                "game_sha256": game_lock["files"][str(index)],
            }
        require(found == {0, 1, 2}, "Missing phase")
    identities = []
    for ordinal, row in enumerate(roots):
        pair, phase = divmod(ordinal, 3)
        expected = expected_roots[(pair, phase)]
        require(all(row[key] == value for key, value in expected.items()), "Root identity changed")
        require(type(row["support"]) is int and 1 <= row["support"] <= 4, "Bad source support")
        identities.append({key: row[key] for key in IDENTITY})
    return identities, inventory


def freeze():
    require(not (OUTPUT / "freeze.json").exists(), "Already frozen")
    checks = read(OUTPUT / "checks/checks.json")
    require(checks["passed"] is True and checks["returncode"] == 0, "Checks did not pass")
    check_hashes(checks["files"])
    require(checks["files"] == {path: sha(ROOT / path) for path in FILES}, "Check inventory mismatch")
    for name in ("stdout", "stderr"):
        require(sha(OUTPUT / f"checks/{name}.txt") == checks[f"{name}_sha256"], "Check output changed")
    identities, inputs = source_inventory()
    save(OUTPUT / "selection_manifest.json", {"roots": identities, "source_binding": SOURCE_BINDING})
    permutations = [np.random.default_rng(CONFIG["permutation_seed_start"] + i).permutation(128).tolist()
                    for i in range(64)]
    save(OUTPUT / "permutations.json", permutations)
    for path in (OUTPUT / "selection_manifest.json", OUTPUT / "permutations.json",
                 OUTPUT / "checks/checks.json", OUTPUT / "checks/stdout.txt", OUTPUT / "checks/stderr.txt"):
        inputs[relative(path)] = sha(path)
    payload = {"format": "saved_halving_screen_v1", "frozen_utc": now(), "config": CONFIG,
               "files": checks["files"], "input_files": inputs,
               "python": sys.version, "numpy": np.__version__}
    payload["binding"] = digest(payload)
    save(OUTPUT / "freeze.json", payload)
    print(json.dumps({"frozen": True, "binding": payload["binding"],
                      "roots": len(identities), "input_files": len(inputs)}), flush=True)


def frozen():
    value = read(OUTPUT / "freeze.json")
    require(value["binding"] == digest({k: v for k, v in value.items() if k != "binding"}), "Freeze identity")
    require(value["config"] == CONFIG and value["numpy"] == np.__version__
            and value["python"] == sys.version, "Analysis environment/config changed")
    check_hashes(value["files"])
    check_hashes(value["input_files"])
    return value


class AllocatedCells:
    """A selector sees only requested selection cells; every read is recorded."""
    def __init__(self, matrix, selection_worlds):
        self._matrix = matrix
        self._allowed = set(selection_worlds)
        self.accesses = []
        self._seen = set()

    def consume(self, worlds, actions):
        totals = {action: 0 for action in actions}
        for world in worlds:
            require(world in self._allowed, "Evaluation world accessed by selector")
            for action in actions:
                cell = (world, action)
                require(cell not in self._seen, "Duplicate decision cell")
                self._seen.add(cell)
                self.accesses.append([world, action])
                totals[action] += int(self._matrix[world][action])
        return totals


def select(cells, support, selection_worlds, adaptive):
    require(support in (1, 2, 3, 4) and len(selection_worlds) == 48, "Invalid selection shape")
    actions = list(range(support))
    if not adaptive or support < 4:
        totals = cells.consume(selection_worlds[:32], actions)
        return min(actions, key=lambda a: (-totals[a], a)), actions
    totals = cells.consume(selection_worlds[:16], actions)
    survivors = sorted(sorted(actions, key=lambda a: (-totals[a], a))[:2])
    extra = cells.consume(selection_worlds[16:48], survivors)
    for action in survivors:
        totals[action] += extra[action]
    return min(survivors, key=lambda a: (-totals[a], a)), survivors


def analyze_partition(matrix, permutation):
    require(len(matrix) == 128 and len(permutation) == 128
            and sorted(permutation) == list(range(128)), "Invalid world layout")
    support = len(matrix[0])
    require(support in (1, 2, 3, 4) and all(len(row) == support for row in matrix), "Invalid matrix")
    require(all(type(v) is int and v in (0, 1, 2) for row in matrix for v in row), "Invalid terminal score")
    uniform_cells = AllocatedCells(matrix, permutation[:48])
    adaptive_cells = AllocatedCells(matrix, permutation[:48])
    uniform, _ = select(uniform_cells, support, permutation[:48], False)
    adaptive, survivors = select(adaptive_cells, support, permutation[:48], True)
    require(len(uniform_cells.accesses) == len(adaptive_cells.accesses) == 32 * support, "Budget mismatch")
    # Only after both decisions are locked is the evaluation subset read.
    return {"uniform_choice": uniform, "adaptive_choice": adaptive, "survivors": survivors,
            "uniform_access": uniform_cells.accesses, "adaptive_access": adaptive_cells.accesses,
            "uniform_heldout_sum2": sum(matrix[w][uniform] for w in permutation[48:]),
            "adaptive_heldout_sum2": sum(matrix[w][adaptive] for w in permutation[48:])}


def source_matrix(identity):
    record = read(local(identity["source_path"]))
    require(record["manifest"] == SOURCE_BINDING and "failure" not in record, "Root binding/failure")
    support = identity["support"]
    candidates = record["candidates"]
    require(len(candidates["actions"]) == len(candidates["keys"]) == support
            and len(set(candidates["keys"])) == support, "Candidate alignment")
    require(record["snapshot"]["current_player"] == identity["actor"]
            and record["snapshot"]["turn_index"] == identity["turn"], "Root actor/turn")
    require(len(record["worlds"]) == 128
            and len(set(record["schedule"]["world_seeds"])) == 128, "World count/uniqueness")
    matrix = []
    for world, seed in zip(record["worlds"], record["schedule"]["world_seeds"]):
        require(world["seed"] == seed and len(world["outcomes"]) == support, "Outcome column alignment")
        row = []
        for item in world["outcomes"]:
            wdl, winner = item["wdl"], item["winner"]
            require(type(wdl) in (int, float) and wdl in (-1, 0, 1), "Bad WDL")
            require(winner is None or type(winner) is int and winner in (0, 1), "Bad winner")
            require(wdl == (0 if winner is None else 1 if winner == identity["actor"] else -1), "Wrong perspective")
            require(item["reason"] in ("completed", "forced_pass_deadlock"), "Invalid terminal reason")
            row.append(int(wdl + 1))
        matrix.append(row)
    return matrix


def aggregate_roots(rows):
    require(len(rows) == 600 and [(r["pair"], r["phase"]) for r in rows]
            == [(p, phase) for p in range(200) for phase in range(3)], "Root aggregation identity")
    pairs = []
    for pair in range(200):
        subset = rows[pair * 3:pair * 3 + 3]
        uniform = sum(row["uniform_score"] for row in subset) / 3
        adaptive = sum(row["adaptive_score"] for row in subset) / 3
        pairs.append({"pair": pair, "uniform_score": uniform, "adaptive_score": adaptive,
                      "delta_score": adaptive - uniform})
    return pairs


def primary_statistics(pairs):
    require(len(pairs) == 200 and [row["pair"] for row in pairs] == list(range(200)), "Pair count/order")
    values = np.array([row["delta_score"] for row in pairs], dtype=float)
    require(np.isfinite(values).all(), "Nonfinite pair value")
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = np.empty(CONFIG["bootstrap_resamples"])
    for start in range(0, len(boot), 1000):
        boot[start:start + 1000] = values[rng.integers(200, size=(1000, 200))].mean(axis=1)
    return {"delta_pp": float(100 * values.mean()),
            "ci95_pp": (100 * np.quantile(boot, [.025, .975])).tolist()}


def write_report(result):
    primary = result["primary"]
    low, high = primary["ci95_pp"]
    text = (
        "# Saved-data equal-budget sampling screen\n\n"
        f"Adaptive minus uniform held-out terminal score: **{primary['delta_pp']:+.4f} percentage points**, "
        f"paired 95% interval **[{low:+.4f}, {high:+.4f}]**.\n\n"
        f"Decision: `{result['decision']}`.\n\n"
        f"Uniform mean held-out score: {result['uniform_score'] * 100:.4f}%; "
        f"adaptive: {result['adaptive_score'] * 100:.4f}%. "
        "These conditional continuation scores are not full-game win rates between planners.\n\n"
        "600 fixed roots from 200 source pairs; all 64 fixed partitions retained. "
        "Each split uses up to 48 selection worlds and 80 separate evaluation worlds. "
        "The interval resamples the 200 complete pairs.\n\n"
        f"Support counts: {result['support_counts']}. "
        f"Choice changes: {100 * result['choice_change_fraction']:.3f}% of root-partitions. "
        f"Uniform choice eliminated at stage one: {100 * result['uniform_choice_eliminated_fraction']:.3f}%.\n\n"
        "Four-action budget: uniform 4 x 32 = 128 continuations; adaptive 4 x 16 + 2 x 32 = 128. "
        "Supports below four use the same uniform rule in both arms. Equal counts do not prove equal wall time.\n\n"
        "Reused-data development evidence only. Evaluation worlds are disjoint for each choice, "
        "but the overall corpus is previously used evidence. No new simulation, training or gameplay occurred. "
        "No claim of whole-game improvement, equivalence, human superiority or a validated learning loop follows.\n\n"
        "Independent reconstruction is required: see verification.json and final_status.json when present. "
        "Source integrity inherits 160's prescribed sampled inner replay; this study does not replay inner trajectories.\n\n"
        "No automatic extension, allocation sweep, gameplay experiment, promotion or monitoring.\n"
    )
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run():
    identity = frozen()
    with (OUTPUT / "run_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"binding": identity["binding"], "time": now()}, handle)
    started = time.perf_counter()
    roots = read(OUTPUT / "selection_manifest.json")["roots"]
    permutations = read(OUTPUT / "permutations.json")
    summaries = []
    supports = Counter()
    changes = eliminated = 0
    with (OUTPUT / "root_results.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("[\n")
        for index, root in enumerate(roots):
            matrix = source_matrix(root)
            partitions = [analyze_partition(matrix, p) for p in permutations]
            record = {**root, "partitions": partitions}
            if index:
                handle.write(",\n")
            json.dump(record, handle, separators=(",", ":"), sort_keys=True, allow_nan=False)
            uniform = sum(r["uniform_heldout_sum2"] for r in partitions) / (64 * 160)
            adaptive = sum(r["adaptive_heldout_sum2"] for r in partitions) / (64 * 160)
            summaries.append({**root, "uniform_score": uniform, "adaptive_score": adaptive,
                              "delta_score": adaptive - uniform})
            supports[root["support"]] += 1
            changes += sum(r["uniform_choice"] != r["adaptive_choice"] for r in partitions)
            eliminated += sum(r["uniform_choice"] not in r["survivors"] for r in partitions)
            if (index + 1) % 100 == 0:
                print(f"Reconstructed {index + 1}/600 roots; no interim endpoint reported", flush=True)
        handle.write("\n]\n")
    pairs = aggregate_roots(summaries)
    primary = primary_statistics(pairs)
    result = {
        "binding": identity["binding"], "time": now(), "primary": primary,
        "decision": "positive_development_screen" if primary["ci95_pp"][0] > 0
                    else "no_positive_development_evidence",
        "uniform_score": sum(p["uniform_score"] for p in pairs) / 200,
        "adaptive_score": sum(p["adaptive_score"] for p in pairs) / 200,
        "support_counts": {str(k): supports[k] for k in range(1, 5)},
        "choice_change_fraction": changes / (600 * 64),
        "uniform_choice_eliminated_fraction": eliminated / (600 * 64),
        "roots": 600, "source_pairs": 200, "partitions_per_root": 64,
        "new_simulations": 0, "training": False, "fresh_gameplay": False,
        "descriptive_delta_pp": {
            label: {str(value): 100 * float(np.mean([r["delta_score"] for r in summaries if r[label] == value]))
                    for value in sorted({r[label] for r in summaries})}
            for label in ("phase", "actor", "support")
        },
        "limitation": "Reused-data conditional development screen; not fresh gameplay evidence.",
    }
    save(OUTPUT / "root_summary.json", summaries)
    save(OUTPUT / "pair_values.json", pairs)
    save(OUTPUT / "result.json", result)
    write_report(result)
    check_hashes(identity["files"])
    check_hashes(identity["input_files"])
    outputs = {relative(OUTPUT / name): sha(OUTPUT / name) for name in (
        "root_results.json", "root_summary.json", "pair_values.json", "result.json", "report.md")}
    save(OUTPUT / "analysis_complete.json", {
        "binding": identity["binding"], "complete": True, "outputs": outputs,
        "elapsed_seconds": time.perf_counter() - started, "roots": 600,
        "partitions": 38400, "time": now(), "independent_verification_pending": True,
    })
    print("Analysis complete. Independent reconstruction required before final status.", flush=True)


def finalize():
    identity = frozen()
    completed = read(OUTPUT / "analysis_complete.json")
    verification = read(OUTPUT / "verification.json")
    check_hashes(completed["outputs"])
    require(completed["complete"] is True and verification["passed"] is True
            and verification["independent"] is True
            and completed["binding"] == verification["binding"] == identity["binding"], "Incomplete verification")
    require(verification["analysis_complete_sha256"] == sha(OUTPUT / "analysis_complete.json"), "Verification link")
    result = read(OUTPUT / "result.json")
    save(OUTPUT / "final_status.json", {
        "complete": True, "independent_verification_pass": True, "binding": identity["binding"],
        "decision": result["decision"], "primary": result["primary"], "time": now(),
        "result_sha256": sha(OUTPUT / "result.json"), "report_sha256": sha(OUTPUT / "report.md"),
        "analysis_complete_sha256": sha(OUTPUT / "analysis_complete.json"),
        "verification_sha256": sha(OUTPUT / "verification.json"), "automatic_successor": False,
        "new_simulations": 0, "training": False, "scheduled_monitoring": False,
    })
    print(json.dumps({"complete": True, "primary": result["primary"], "decision": result["decision"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "freeze", "run", "finalize"))
    args = parser.parse_args()
    try:
        {"check": check, "freeze": freeze, "run": run, "finalize": finalize}[args.command]()
    except Exception as exc:
        if OUTPUT.exists():
            path = OUTPUT / f"failure_{args.command}_{time.time_ns()}.json"
            save(path, {"command": args.command, "time": now(), "type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()

"""Independently reconstruct the frozen saved-data halving screen.

This file imports no game, inference, simulation, or production-analysis code.
It reads original 160 evidence only after verifying the screen's frozen identity.
The original 160 sampled inner replay is inherited, never represented as a new
full trajectory replay. The verification receipt is written once, after success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY / "outputs/saved_halving_screen"
SOURCE = REPOSITORY / "outputs/world_budget_160"
SOURCE_BINDING = "5e9e9202f44e7b526fff4be97cdcd3186805fbb5e8a184d21a6702de9d922df3"
SOURCE_RESULT_SHA = "2e3efa2c524f8d8c68284fad0f3400365bcb08f4b598aecf121039a4c45b19e5"
SOURCE_ROOT_LIST_SHA = "3c762d22dfcd412239c1f485826fe6ef72e802a5b7f340d2732fa1e305763755"
IDENTITY_FIELDS = ("pair", "phase", "source_index", "actor", "turn", "source_path",
                   "source_sha256", "support")
REQUIRED_CONFIG = {
    "worlds": 128, "partitions": 64, "permutation_seed_start": 2026091700,
    "bootstrap_seed": 2026091764, "bootstrap_resamples": 50000,
    "pairs": 200, "roots": 600, "initial_worlds": 16,
    "additional_worlds": 32, "reference_worlds": 32, "evaluation_worlds": 80,
    "short_support": "uniform32",
    "primary": "adaptive_minus_uniform_heldout_terminal_score",
    "score": "(wdl+1)/2", "ties": "original_candidate_index",
}


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def repository_path(relative):
    demand(isinstance(relative, str), "Evidence path must be a string")
    value = (REPOSITORY / relative).resolve()
    demand(value.is_relative_to(REPOSITORY), "Evidence path leaves the repository")
    return value


def validate_hash_map(mapping):
    demand(isinstance(mapping, dict) and mapping, "Empty evidence hash inventory")
    for relative, expected in mapping.items():
        demand(file_hash(repository_path(relative)) == expected,
               f"Bound file changed: {relative}")


def expected_partition(sum2, permutation):
    """Independent scalar integer arithmetic, retaining original-index ties."""
    demand(len(sum2) == 128, "Expected 128 worlds")
    support = len(sum2[0])
    demand(support in (1, 2, 3, 4), "Invalid action support")
    demand(all(len(row) == support for row in sum2), "Ragged outcome matrix")
    demand(all(type(value) is int and value in (0, 1, 2)
               for row in sum2 for value in row), "Invalid twice-score")
    demand(len(permutation) == 128 and sorted(permutation) == list(range(128)),
           "Invalid world permutation")
    uniform_access = [[world, action] for world in permutation[:32]
                      for action in range(support)]
    uniform_totals = [sum(sum2[world][action] for world in permutation[:32])
                      for action in range(support)]
    uniform = min(range(support), key=lambda action: (-uniform_totals[action], action))
    if support < 4:
        survivors = list(range(support))
        adaptive_access = list(uniform_access)
        adaptive = uniform
    else:
        first_totals = [sum(sum2[world][action] for world in permutation[:16])
                        for action in range(4)]
        survivors = sorted(sorted(range(4),
                                  key=lambda action: (-first_totals[action], action))[:2])
        adaptive_access = [[world, action] for world in permutation[:16]
                           for action in range(4)]
        adaptive_access += [[world, action] for world in permutation[16:48]
                            for action in survivors]
        final_totals = {action: sum(sum2[world][action] for world in permutation[:48])
                        for action in survivors}
        adaptive = min(survivors, key=lambda action: (-final_totals[action], action))
    demand(len(uniform_access) == len(adaptive_access) == support * 32,
           "Unequal continuation counts")
    for accesses in (uniform_access, adaptive_access):
        demand(len(set(map(tuple, accesses))) == len(accesses), "Duplicate cell access")
        demand(not ({world for world, _ in accesses} & set(permutation[48:])),
               "Evaluation leakage")
    return {
        "uniform_choice": uniform, "adaptive_choice": adaptive, "survivors": survivors,
        "uniform_access": uniform_access, "adaptive_access": adaptive_access,
        "uniform_heldout_sum2": sum(sum2[world][uniform] for world in permutation[48:]),
        "adaptive_heldout_sum2": sum(sum2[world][adaptive] for world in permutation[48:]),
    }


def bootstrap_primary(pair_deltas):
    demand(len(pair_deltas) == 200 and all(math.isfinite(v) for v in pair_deltas),
           "Invalid independent-pair values")
    values = np.asarray(pair_deltas, dtype=float)
    generator = np.random.default_rng(2026091764)
    bootstraps = np.empty(50000, dtype=float)
    for start in range(0, 50000, 1000):
        indices = generator.integers(0, 200, size=(1000, 200))
        bootstraps[start:start + 1000] = values[indices].mean(axis=1)
    return {"delta_pp": float(100 * values.mean()),
            "ci95_pp": (100 * np.quantile(bootstraps, [0.025, 0.975])).tolist()}


def close(actual, expected, label):
    demand(type(actual) in (int, float) and math.isfinite(actual)
           and abs(actual - expected) <= 1e-11, f"Numerical mismatch: {label}")


def source_games():
    """Hash all source games and reproduce outcome-blind selected root identities."""
    final = read_json(SOURCE / "final_status.json")
    demand(final.get("complete") is True and final.get("prespecified_audit_pass") is True,
           "Source 160 did not complete its prescribed audit")
    demand(final["manifest"] == SOURCE_BINDING, "Wrong source binding")
    demand(final["result_sha256"] == SOURCE_RESULT_SHA
           and file_hash(SOURCE / "result.json") == SOURCE_RESULT_SHA, "Source result changed")
    source_manifest = read_json(SOURCE / "manifest.json")
    demand(digest(source_manifest) == SOURCE_BINDING, "Source manifest changed")
    validate_hash_map(source_manifest["dependencies"])
    source_result = read_json(SOURCE / "result.json")
    audit_lock = read_json(SOURCE / "main_audit_lock.json")
    demand(file_hash(SOURCE / "main_audit_lock.json") == source_result["main_audit_lock_sha256"],
           "Source result-to-audit-lock link")
    lock = read_json(SOURCE / "main_collect_lock.json")
    demand(lock["manifest"] == SOURCE_BINDING, "Wrong game lock binding")
    demand(audit_lock["manifest"] == SOURCE_BINDING
           and audit_lock["source_lock_sha256"] == file_hash(SOURCE / "main_collect_lock.json"),
           "Source audit-to-collection link")
    demand(set(lock["files"]) == {str(index) for index in range(400)}, "Source game count")
    demand(set(audit_lock["files"]) == set(lock["files"]), "Source audit count")
    selected = {}
    for index in range(400):
        path = SOURCE / "main/games" / f"game_{index:04d}.json"
        demand(file_hash(path) == lock["files"][str(index)], f"Source game hash: {index}")
        game = read_json(path)
        pair, seat = divmod(index, 2)
        expected_spec = {
            "index": index, "pair": pair, "candidate_seat": seat,
            "environment_seed": 1600000000 + pair,
            "search_stream": digest(["160-root-stream-v1", "main", pair]),
        }
        demand(game["manifest"] == SOURCE_BINDING and game["spec"] == expected_spec
               and "failure" not in game, f"Source game identity: {index}")
        audit_path = SOURCE / "main/audits" / f"game_{index:04d}.json"
        demand(file_hash(audit_path) == audit_lock["files"][str(index)], f"Source audit hash: {index}")
        audit = read_json(audit_path)
        demand(audit["manifest"] == SOURCE_BINDING and "failure" not in audit
               and audit["game_sha256"] == lock["files"][str(index)]
               and audit["audit"]["game"]["index"] == index,
               f"Source audit-to-game link: {index}")
        assigned_seat = int(digest(["post160-review-source-seat-v1", pair]), 16) % 2
        if seat != assigned_seat:
            continue
        phases = set()
        for move in game["trace"]:
            phase = min(move["turn"] // 20, 2)
            if move["actor"] != seat or phase in phases:
                continue
            phases.add(phase)
            relative = (Path("outputs/world_budget_160/main/roots") /
                        f"game_{index:04d}" / f"turn_{move['turn']:03d}.json").as_posix()
            selected[(pair, phase)] = {
                "pair": pair, "phase": phase, "source_index": index,
                "actor": seat, "turn": move["turn"], "source_path": relative,
                "source_sha256": move["root_sha256"], "game_sha256": lock["files"][str(index)],
                "move": move, "spec": expected_spec,
            }
        demand(phases == {0, 1, 2}, f"Missing selected phase: {pair}")
    demand(len(selected) == 600, "Selected root count")
    return selected


def source_matrix(identity, source):
    path = repository_path(identity["source_path"])
    demand(file_hash(path) == identity["source_sha256"], "Source root changed")
    root = read_json(path)
    demand(root["manifest"] == SOURCE_BINDING and "failure" not in root,
           "Source root identity/failure")
    demand("continuation_model" not in root, "Wrong continuation policy")
    demand(digest(root["snapshot"]) == source["move"]["state_sha256"], "Source snapshot identity")
    actions = root["candidates"]["actions"]
    support = len(actions)
    demand(support == identity["support"] and support in (1, 2, 3, 4), "Source support mismatch")
    demand(len({digest(action) for action in actions}) == support, "Duplicate offered action")
    schedule = root["schedule"]
    belief = schedule["belief_sha256"]
    stream = source["spec"]["search_stream"]
    demand(schedule["candidate_nonce"] == digest(["155-candidate-v1", stream, belief]),
           "Candidate nonce identity")
    seeds = [f"155-world-v1|{stream}|{belief}|{index:02d}" for index in range(128)]
    demand(schedule["world_seeds"] == seeds and len(root["worlds"]) == 128,
           "Source world schedule")
    matrix = []
    for index, world in enumerate(root["worlds"]):
        demand(world["seed"] == seeds[index] and len(world["outcomes"]) == support,
               "World identity/candidate alignment")
        row = []
        for outcome in world["outcomes"]:
            wdl, winner = outcome["wdl"], outcome["winner"]
            demand(type(wdl) in (int, float) and wdl in (-1, 0, 1), "Invalid WDL")
            demand(winner is None or type(winner) is int and winner in (0, 1), "Invalid winner")
            demand(wdl == (0 if winner is None else 1 if winner == identity["actor"] else -1),
                   "Wrong outcome perspective")
            demand(outcome["reason"] in ("completed", "forced_pass_deadlock"),
                   "Unaccepted source terminal reason")
            row.append(int(wdl + 1))
        matrix.append(row)
    totals = [sum(row[action] - 1 for row in matrix) for action in range(support)]
    choice = min(range(support), key=lambda action: (-totals[action], action))
    demand(root["choice"] == choice and source["move"]["action"] == actions[choice],
           "Source full128 choice integrity")
    demand(len(root["means"]) == support, "Source mean count")
    for actual, total in zip(root["means"], totals):
        close(actual, total / 128, "Source mean")
    return matrix


def verify(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    demand(output.is_relative_to(REPOSITORY), "Output must remain in repository")
    demand(not (output / "verification.json").exists(), "Verification receipt already exists")
    freeze = read_json(output / "freeze.json")
    demand(freeze["binding"] == digest({key: value for key, value in freeze.items() if key != "binding"}),
           "Frozen binding mismatch")
    for key, expected in REQUIRED_CONFIG.items():
        demand(freeze["config"].get(key) == expected, f"Frozen setting mismatch: {key}")
    demand(freeze["numpy"] == np.__version__ and freeze["python"] == sys.version,
           "Frozen analysis runtime changed")
    validate_hash_map(freeze["files"])
    validate_hash_map(freeze["input_files"])
    completed = read_json(output / "analysis_complete.json")
    demand(completed.get("complete") is True and completed["binding"] == freeze["binding"]
           and completed["roots"] == 600 and completed["partitions"] == 38400,
           "Incomplete/mismatched analysis")
    validate_hash_map(completed["outputs"])
    for name in ("root_results.json", "root_summary.json", "pair_values.json", "result.json", "report.md"):
        relative = (output / name).relative_to(REPOSITORY).as_posix()
        demand(relative in completed["outputs"], f"Unsealed analysis output: {name}")
    for name in ("selection_manifest.json", "permutations.json"):
        relative = (output / name).relative_to(REPOSITORY).as_posix()
        demand(relative in freeze["input_files"], f"Unfrozen selection input: {name}")
    old_roots_path = SOURCE / "review_sampling_20260916/roots.json"
    demand(file_hash(old_roots_path) == SOURCE_ROOT_LIST_SHA, "Original root list changed")
    old_roots = read_json(old_roots_path)
    manifest = read_json(output / "selection_manifest.json")
    demand(manifest["source_binding"] == SOURCE_BINDING and len(manifest["roots"]) == 600,
           "Selection manifest identity/count")
    expected_keys = [(pair, phase) for pair in range(200) for phase in range(3)]
    demand([(row["pair"], row["phase"]) for row in manifest["roots"]] == expected_keys,
           "Unexpected root selection/order")
    demand(len(old_roots) == 600, "Original root selection count")
    source = source_games()
    permutations = read_json(output / "permutations.json")
    demand(len(permutations) == 64, "Partition count")
    for index, permutation in enumerate(permutations):
        demand(permutation == np.random.default_rng(2026091700 + index).permutation(128).tolist(),
               f"World permutation changed: {index}")
    rows = read_json(output / "root_results.json")
    demand(len(rows) == 600, "Root result count")
    root_values = []
    supports = Counter()
    cell_accesses = 0
    changes = eliminated = 0
    for identity, original, recorded, key in zip(manifest["roots"], old_roots, rows, expected_keys):
        for field in IDENTITY_FIELDS:
            demand(identity[field] == original[field] == recorded[field],
                   f"Selected/result root identity mismatch: {key}, {field}")
            if field != "support":
                demand(identity[field] == source[key][field], f"Outcome-blind root selection: {key}")
        if "game_sha256" in identity:
            demand(identity["game_sha256"] == source[key]["game_sha256"], "Source game link")
        matrix = source_matrix(identity, source[key])
        supports[identity["support"]] += 1
        demand(len(recorded["partitions"]) == 64, "Per-root partition count")
        uniform_sums, adaptive_sums = [], []
        for permutation, observed in zip(permutations, recorded["partitions"]):
            expected = expected_partition(matrix, permutation)
            for field, value in expected.items():
                demand(observed[field] == value, f"Independent partition mismatch: {key}, {field}")
            cell_accesses += len(expected["uniform_access"]) + len(expected["adaptive_access"])
            changes += expected["uniform_choice"] != expected["adaptive_choice"]
            eliminated += expected["uniform_choice"] not in expected["survivors"]
            uniform_sums.append(expected["uniform_heldout_sum2"])
            adaptive_sums.append(expected["adaptive_heldout_sum2"])
        root_values.append((sum(uniform_sums) / (64 * 160),
                            sum(adaptive_sums) / (64 * 160)))
    root_summaries = read_json(output / "root_summary.json")
    demand(len(root_summaries) == 600, "Root summary count")
    for identity, summary, (uniform, adaptive) in zip(manifest["roots"], root_summaries, root_values):
        for field in IDENTITY_FIELDS:
            demand(summary[field] == identity[field], "Root summary identity")
        close(summary["uniform_score"], uniform, "Root uniform summary")
        close(summary["adaptive_score"], adaptive, "Root adaptive summary")
        close(summary["delta_score"], adaptive - uniform, "Root delta summary")
    pairs = read_json(output / "pair_values.json")
    demand(len(pairs) == 200, "Pair endpoint count")
    deltas = []
    for pair, recorded in enumerate(pairs):
        demand(recorded["pair"] == pair, "Pair order changed")
        uniform = sum(root_values[3 * pair + phase][0] for phase in range(3)) / 3
        adaptive = sum(root_values[3 * pair + phase][1] for phase in range(3)) / 3
        delta = adaptive - uniform
        close(recorded["uniform_score"], uniform, "Pair uniform score")
        close(recorded["adaptive_score"], adaptive, "Pair adaptive score")
        close(recorded["delta_score"], delta, "Pair delta")
        deltas.append(delta)
    primary = bootstrap_primary(deltas)
    result = read_json(output / "result.json")
    demand(result["binding"] == freeze["binding"] and result["roots"] == 600
           and result["source_pairs"] == 200 and result["partitions_per_root"] == 64,
           "Result identity/counts")
    close(result["primary"]["delta_pp"], primary["delta_pp"], "Primary effect")
    demand(len(result["primary"]["ci95_pp"]) == 2, "CI shape")
    for index in (0, 1):
        close(result["primary"]["ci95_pp"][index], primary["ci95_pp"][index], "Primary CI")
    demand({int(key): value for key, value in result["support_counts"].items() if value}
           == dict(supports), "Support counts mismatch")
    close(result["uniform_score"], sum(row[0] for row in root_values) / 600, "Overall uniform score")
    close(result["adaptive_score"], sum(row[1] for row in root_values) / 600, "Overall adaptive score")
    close(result["choice_change_fraction"], changes / 38400, "Choice change fraction")
    close(result["uniform_choice_eliminated_fraction"], eliminated / 38400, "Elimination fraction")
    for field in ("phase", "actor", "support"):
        expected_groups = sorted({row[field] for row in manifest["roots"]})
        demand(set(result["descriptive_delta_pp"][field]) == {str(group) for group in expected_groups},
               "Descriptive group membership")
        for group in expected_groups:
            deltas_group = [values[1] - values[0] for identity, values
                            in zip(manifest["roots"], root_values) if identity[field] == group]
            close(result["descriptive_delta_pp"][field][str(group)],
                  100 * sum(deltas_group) / len(deltas_group), "Descriptive group delta")
    expected_decision = ("positive_development_screen" if primary["ci95_pp"][0] > 0
                         else "no_positive_development_evidence")
    demand(result["decision"] == expected_decision, "Incorrect frozen decision rule")
    # Recheck inputs/code after reading all matrices; the receipt binds both ends.
    validate_hash_map(freeze["files"])
    validate_hash_map(freeze["input_files"])
    validate_hash_map(completed["outputs"])
    return {
        "binding": freeze["binding"], "independent": True, "passed": True,
        "time": datetime.now(timezone.utc).isoformat(),
        "analysis_complete_sha256": file_hash(output / "analysis_complete.json"),
        "all_bound_hashes_verified": True, "source_games_hashed": 400,
        "source_audits_hashed": 400,
        "selected_source_roots_hashed": 600, "root_selections_reconstructed": 600,
        "partitions_reconstructed": 38400, "decision_cell_accesses_checked": cell_accesses,
        "independent_pairs": 200, "bootstrap_resamples": 50000,
        "primary": primary, "decision": expected_decision,
        "source_full_inner_replay_performed": False,
        "limitations": "Reused-data conditional development screen; no gameplay or new inner replay.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = verify(args.output)
    with (args.output / "verification.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": True, "roots": 600, "partitions": 38400,
                      "primary": receipt["primary"], "decision": receipt["decision"]}))


if __name__ == "__main__":
    main()

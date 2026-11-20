"""Reserve corrected starting states for a proposed study; never play or load models.

This is preparation only, not an executable player, qualification or launch plan.
All writes are exclusive and restricted to the new preparation evidence directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/splendorust_onepass_strength_preparation_20261106_01"
DOMAIN = "splendorust_onepass_strength_20261106_v1"
BOOTSTRAP_SEEDS = {"primary_stratified": 20261106128, "r0": 20261106129, "r1": 20261106130}
SECONDS = 600
MAX_SCAN_BYTES = 8 * 1024**2
EXPLICIT_PRIOR = (
    ("outputs/splendorust_policy_head_20261104_01/assessment/schedule.json", 406, 203),
    ("outputs/splendorust_selfplay_pilot_resume2_20261103_01/assessment/schedule.json", 406, 203),
    ("outputs/splendorust_selfplay_pilot_resume2_20261103_01/game_schedule.json", 256, 256),
)


def require(value, message):
    if not value:
        raise ValueError("ONEPASS_STRENGTH_PREPARATION: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def artifact(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(name, value):
    path = OUT / name
    require(path.parent == OUT and path.name in {
        "runtime_sources.json", "schedule.json", "initial_states.json",
        "seed_inventory.json", "preparation_receipt.json"}, "fixed write allowlist")
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(path)


def setup_seed(stage, repetition, pair):
    require(stage in ("operational", "main") and repetition in (0, 1), "seed identity")
    count = 3 if stage == "operational" else 100
    require(type(pair) is int and 0 <= pair < count, "local pair range")
    return str(int.from_bytes(hashlib.sha256(canonical([DOMAIN, stage, repetition, pair]).encode()).digest(), "big"))


def spec(stage, repetition, pair, order):
    require(type(order) is int and order in (0, 1), "pair order")
    seed = setup_seed(stage, repetition, pair)
    stage_pair = repetition * 3 + pair if stage == "operational" else (pair // 5) * 10 + repetition * 5 + pair % 5
    global_pair = stage_pair if stage == "operational" else 6 + stage_pair
    return {
        "stage": stage, "candidate": f"r{repetition}_broad_joint", "reference": "original_ef8a",
        "repetition": repetition, "local_pair_index": pair, "stage_pair_index": stage_pair,
        "pair_index": global_pair, "game_index": 2 * global_pair + order,
        "stage_game_index": 2 * stage_pair + order, "order_in_pair": order,
        "seed_decimal": seed, "candidate_seat": (repetition + pair + order) % 2,
        "candidate_simulations": 128, "reference_simulations": 128,
        "worlds_each": 3, "cpuct_each": .4, "depth_each": 64,
    }


def schedule():
    rows = [spec("operational", repetition, pair, order)
            for repetition in (0, 1) for pair in range(3) for order in (0, 1)]
    rows += [spec("main", repetition, pair, order)
             for block in range(20) for repetition in (0, 1)
             for pair in range(block * 5, block * 5 + 5) for order in (0, 1)]
    return rows


def validate_schedule(rows):
    require(rows == schedule(), "exact predetermined complete schedule")
    require(len(rows) == 412 and [x["game_index"] for x in rows] == list(range(412)), "all global game identities")
    require({x["pair_index"] for x in rows} == set(range(206)), "all global pair identities")
    seeds = {x["seed_decimal"] for x in rows}
    require(len(seeds) == 206, "independent setup per pair")
    require(not {int(x) for x in seeds} & set(BOOTSTRAP_SEEDS.values()), "game and analysis seeds disjoint")
    require(len(set(BOOTSTRAP_SEEDS.values())) == 3, "three fixed distinct analysis seeds")
    for stage, count in (("operational", 6), ("main", 200)):
        for repetition in (0, 1):
            part = [x for x in rows if x["stage"] == stage and x["repetition"] == repetition]
            require(len(part) == count and {x["local_pair_index"] for x in part} == set(range(count // 2)), "equal complete allocation")
            require(all(sum(x["candidate_seat"] == seat for x in part) == count // 2 for seat in (0, 1)), "balanced seats")
    for i in range(0, len(rows), 2):
        pair = rows[i:i + 2]
        require(pair[0]["seed_decimal"] == pair[1]["seed_decimal"] and
                pair[0]["pair_index"] == pair[1]["pair_index"] and
                {x["candidate_seat"] for x in pair} == {0, 1}, "swapped seat pair")
    main = rows[12:]
    for start in range(0, 400, 10):
        require({x["repetition"] for x in main[start:start + 10]} == {(start // 10) % 2}, "alternating ten game blocks")
    return {"games": 412, "main_games": 400, "main_pairs": 200, "main_games_each_repetition": 200,
            "operational_games": 12, "operational_pairs": 6, "unique_setups": 206,
            "balanced_seats": True, "disjoint_repetitions_and_stages": True,
            "deterministic_schedule_reconstructed": True, "alternating_main_blocks": True}


def registered_seed_inventory(rows, boundary):
    targets = {int(x["seed_decimal"]) for x in rows} | set(BOOTSTRAP_SEEDS.values())
    explicit = []
    for relative, count, seeds in EXPLICIT_PRIOR:
        boundary()
        path = ROOT / relative
        prior = read(path)
        previous_seeds = {int(x["seed_decimal"]) for x in prior}
        require(len(prior) == count and len(previous_seeds) == seeds, "complete explicit prior schedule")
        require(not targets & previous_seeds, "explicit prior seed collision")
        explicit.append({"artifact": artifact(path), "rows": count, "unique_seeds": seeds})
    found = subprocess.run([
        "rg", "--files", "--no-ignore", "outputs", "-g", "*manifest*.json",
        "-g", "*schedule*.json", "-g", "*seed_registry*.json", "-g", "*reserved_panel*.json",
        "-g", "*freeze*.json"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=120)
    checked, skipped, collisions = [], [], []

    def visit(value, filename):
        if type(value) is int or type(value) is str and value.isdecimal():
            if int(value) in targets:
                collisions.append(filename)
        elif type(value) is list:
            for child in value:
                visit(child, filename)
        elif type(value) is dict:
            for child in value.values():
                visit(child, filename)

    for relative in sorted(found.stdout.splitlines()):
        boundary()
        path = (ROOT / relative).resolve()
        if path.is_relative_to(OUT):
            continue
        if path.stat().st_size > MAX_SCAN_BYTES:
            skipped.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size})
            continue
        visit(read(path), relative)
        checked.append(artifact(path))
    require(not collisions, "registered seed collision; no replacement seeds authorized")
    return {"status": "passed", "domain": DOMAIN, "new_setup_count": 206,
            "bootstrap_seeds": BOOTSTRAP_SEEDS, "explicit_prior_schedules": explicit,
            "checked": checked, "checked_count": len(checked), "skipped_oversized": skipped,
            "collisions": collisions,
            "scope": "Registered manifests, schedules, seed registries, reserved panels and freezes up to 8 MiB; excludes own output, arbitrary logs and disclosed oversized files.",
            "freshness_claim": "Disjoint from checked registered seeds and the three explicit complete schedules; not an exhaustive scan of every historical raw request.",
            "search_randomness": "Unchanged canonical observer derived seed; never give game seed or hidden deck order to a player."}


def main():
    origin = time.monotonic()
    wall_origin = time.time()

    def boundary():
        require(time.monotonic() - origin < SECONDS and time.time() - wall_origin < SECONDS,
                "actual ten minute preparation bound")

    names = ("runtime_sources.json", "schedule.json", "initial_states.json", "seed_inventory.json", "preparation_receipt.json")
    require(not any((OUT / name).exists() for name in names), "never overwrite or rerun an existing preparation")
    from tools.future_training_rules_20261021 import (
        verify_runtime_sources, decode_training_snapshot, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT)
    from tools.external_strength_screen_20261018 import rules
    runtime_sources = verify_runtime_sources()  # Exactly once before any prepared environment.
    source_item = save("runtime_sources.json", {"sources": runtime_sources,
        "ruleset_id": RULESET_ID, "catalogue_sha256": CATALOGUE_SHA256, "turn_contract": TURN_CONTRACT,
        "startup_calls": 1, "purpose": "model_free_starting_state_reservation"})
    boundary()
    rows = schedule()
    checks = validate_schedule(rows)
    print("Validated 412 scheduled games; checking registered seeds.", flush=True)
    inventory = registered_seed_inventory(rows, boundary)
    print(f"Checked {inventory['checked_count']} registered files; reserving corrected initial states.", flush=True)
    initials = []
    for row in rows:
        boundary()
        initial = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(row["seed_decimal"])).initial_state())
        decoded = decode_training_snapshot(initial)
        require(not decoded.terminal and decoded.turn_index == 0, "fresh nonterminal initial state")
        require(rules.serialize_snapshot(decoded) == initial, "strict corrected snapshot round trip")
        initials.append({"spec": row, "initial": initial})
    for index in range(0, len(initials), 2):
        require(initials[index]["initial"] == initials[index + 1]["initial"], "identical full state in swapped pair")
    checks.update(corrected_initial_snapshots=412, strict_decode_roundtrips=412,
                  identical_full_initial_state_within_pairs=True)
    schedule_item = save("schedule.json", rows)
    initials_item = save("initial_states.json", initials)
    inventory_item = save("seed_inventory.json", inventory)
    require(read(OUT / "schedule.json") == schedule(), "persisted schedule exact")
    require(read(OUT / "initial_states.json") == initials, "persisted complete initial states exact")
    boundary()
    require("torch" not in sys.modules, "preparation did not import Torch")
    receipt = {
        "schema": "onepass_strength_preparation_v1", "status": "model_free_preparation_passed",
        "preparation_only": True, "execution_frozen": False, "native_player_qualified": False,
        "source": artifact(__file__), "runtime_sources": source_item, "schedule": schedule_item,
        "initial_states": initials_item, "seed_inventory": inventory_item, "checks": checks,
        "domain": DOMAIN, "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "ruleset_id": RULESET_ID, "catalogue_sha256": CATALOGUE_SHA256, "turn_contract": TURN_CONTRACT,
        "model_calls": 0, "search_decisions": 0, "games_played": 0, "optimizer_calls": 0,
        "torch_imported": False, "elapsed_seconds": time.monotonic() - origin,
        "maximum_seconds": SECONDS, "utc_start_unix": wall_origin,
    }
    item = save("preparation_receipt.json", receipt)
    print(canonical({"status": receipt["status"], "receipt": item, "checks": checks,
                     "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

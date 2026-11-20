"""Model-free, deterministic saved-state selection for the timing investigation.

Only this version's manifest is new. Historical receipts are read, never changed.
The saved validation panel is held out from optimization; it is not fresh data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit, rules

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "outputs/splendorust_search_scale_20261024_01"
DOMAIN = "splendorust_timing_saved_selection_20261024_v1"
PHASES = ("0..19", "20..39", "40+")
CELLS = tuple((phase, seat) for phase in PHASES for seat in (0, 1))


def require(value, message):
    if not value:
        raise ValueError("Timing fixtures: " + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def phase_for(ply):
    require(type(ply) is int and ply >= 0, "invalid ply")
    return "0..19" if ply < 20 else "20..39" if ply < 40 else "40+"


def selection_key(row, domain):
    """Exclude timing, results, actions and traces from the ranking key."""
    return digest([domain, row["pair_index"], row["game_index"], row["ply"],
                   row["seat"], row["observer_sha256"]])


def select_stratified(rows, *, domain, per_cell, excluded_observers=(), excluded_pairs=(),
                      unique_pairs=True):
    """Fixed hash selection with strict phase/seat counts and disclosed exclusions."""
    require(type(per_cell) is int and per_cell > 0, "positive per-cell count required")
    used_observers, used_pairs = set(excluded_observers), set(excluded_pairs)
    selected = []
    for phase, seat in CELLS:
        candidates = sorted((row for row in rows if row["phase"] == phase and row["seat"] == seat),
                            key=lambda row: (selection_key(row, domain), row["fixture_id"]))
        count = 0
        for row in candidates:
            if row["observer_sha256"] in used_observers or row["pair_index"] in used_pairs:
                continue
            selected.append({**row, "selection_sha256": selection_key(row, domain)})
            used_observers.add(row["observer_sha256"])
            if unique_pairs:
                used_pairs.add(row["pair_index"])
            count += 1
            if count == per_cell:
                break
        require(count == per_cell, f"insufficient independent saved roots for {phase}, seat {seat}")
    return selected


def repeated_schedule(fixtures, modes=("baseline", "profiled"), passes=2):
    """Interleave resident modes; reverse roots and mode order on pass two.

    Include exact preserved hidden-state variants, even when the lawful root
    duplicates the original. Only one worker may compute at a time.
    """
    require(type(passes) is int and passes > 0, "positive pass count required")
    require(bool(modes) and len(set(modes)) == len(modes), "distinct modes required")
    rows = []
    for pass_index in range(passes):
        order = list(range(len(fixtures)))
        if pass_index % 2:
            order.reverse()
        for fixture_index in order:
            fixture = fixtures[fixture_index]
            variants = ["original"] + [f"preserved/{i}" for i in range(len(fixture.get("preserved_variants", [])))]
            mode_order = list(modes)
            if (fixture_index + pass_index) % 2:
                mode_order.reverse()
            if pass_index % 2:
                variants.reverse()
            for variant in variants:
                for mode in mode_order:
                    rows.append({"index": len(rows), "fixture_index": fixture_index,
                                 "fixture_id": fixture["fixture_id"], "variant": variant,
                                 "pass_index": pass_index, "mode": mode})
    return rows


def snapshot_for_request(fixtures, spec):
    fixture = fixtures[spec["fixture_index"]]
    require(fixture["fixture_id"] == spec["fixture_id"], "request fixture identity")
    if spec["variant"] == "original":
        return fixture["snapshot"]
    require(spec["variant"].startswith("preserved/"), "unknown variant")
    index = int(spec["variant"].split("/")[1])
    require(index >= 0, "negative preserved-variant index")
    return fixture["preserved_variants"][index]["snapshot"]


def _read(path):
    data = Path(path).read_bytes()
    return json.loads(data), {"path": str(Path(path).resolve()), "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()}


def _metadata(row, path, artifact):
    require(row["completed"] is True and row["challenger_simulations"] == 32 and
            row["arm"] == "search" and row["stage"] == "main", "unexpected source decision")
    require(row["before"]["ruleset_id"] == guard.RULESET_ID and
            row["before"]["catalogue_sha256"] == guard.CATALOGUE_SHA256, "uncorrected source")
    return {"fixture_id": f"main/{row['game_index']:03d}/{row['ply']:04d}",
            "pair_index": row["pair_index"], "game_index": row["game_index"],
            "ply": row["ply"], "seat": row["seat"], "phase": phase_for(row["ply"]),
            "observer_sha256": digest(row["observer"]), "source": artifact}


def _materialize(row, *, reasons, historical32=True):
    raw, artifact = _read(row["source"]["path"])
    require(artifact == row["source"], "source changed during selection")
    snapshot = raw["before"]
    audit.validate_ledger(snapshot)
    observer = rules.observation_payload(guard.decode_training_snapshot(snapshot))
    require(observer == raw["observer"] and digest(observer) == row["observer_sha256"],
            "saved observation does not match corrected snapshot")
    result = {**row, "snapshot": snapshot, "observer": observer,
              "reasons": list(reasons), "provenance": [artifact]}
    if historical32:
        result["historical32"] = {"action": raw["action"], "trace": raw["trace"],
                                  "source": artifact,
                                  "receipt_elapsed_seconds": raw["clock"]["receipt_elapsed_seconds"]}
    return result


def build_manifest(attempt=PRIOR, *, controls_per_cell=2, validation_per_cell=4,
                   expected_slow_count=17):
    """Read existing evidence, freeze identities, and return a JSON-safe manifest.

    This function authenticates corrected runtime sources once, but neither
    imports a neural model nor launches a worker. The calling evaluator must
    separately authenticate its own startup and qualify its real entry point.
    """
    runtime_sources = guard.verify_runtime_sources()
    attempt = Path(attempt).resolve()
    plan, plan_artifact = _read(attempt / "qualification_plan_32.json")
    summary64, summary_artifact = _read(attempt / "qualification_summary_64.json")
    replay, replay_artifact = _read(attempt / "replay_sample_manifest.json")
    completion, completion_artifact = _read(attempt / "completion.json")
    require(completion["status"] == "completed_verified", "prior attempt is not verified")
    require(len(plan["fixtures"]) == 23, "prior panel count changed")
    require(plan["ruleset_id"] == guard.RULESET_ID and
            plan["catalogue_sha256"] == guard.CATALOGUE_SHA256, "prior panel identity")

    # Timings are inspected only to recover the required exhaustive slow set.
    # The metadata pool passed to hash selectors contains no timings or outcomes.
    candidates, slow, inventory = [], [], []
    for path in sorted((attempt / "games/main").glob("*/moves/*.json")):
        raw, artifact = _read(path)
        inventory.append(artifact)
        if raw["arm"] != "search":
            continue
        row = _metadata(raw, path, artifact)
        candidates.append(row)
        clock = raw["clock"]
        require(clock["status"] == "ok" and clock["accepted"] is True and
                type(clock["receipt_elapsed_seconds"]) in (int, float), "invalid saved timing")
        if clock["receipt_elapsed_seconds"] > .7:
            slow.append(row)
    require(len(slow) == expected_slow_count, "slow-decision count differs from completed study")
    require(bool(candidates), "empty saved decision pool")

    # Panel originals carry historical32 outputs from the same lawful roots.
    panel = []
    for item in plan["fixtures"]:
        spec = next(spec for spec in plan["timed_requests"]
                    if spec["fixture_index"] == item["fixture_index"] and
                    spec["variant"] == "original" and spec["arm"] == "search" and spec["pass"] == 0)
        raw, artifact = _read(attempt / "qualification_32" / f"{spec['index']:03d}.json")
        observer = rules.observation_payload(guard.decode_training_snapshot(item["original"]))
        require(observer == raw["observer"] and digest(observer) == item["observer_sha256"],
                "panel observation changed")
        require(raw["before"] == item["original"] and raw["challenger_simulations"] == 32,
                "panel provenance changed")
        panel.append({"fixture_id": "panel/" + item["fixture_id"], "snapshot": item["original"],
                      "observer": observer, "observer_sha256": digest(observer),
                      "reasons": ["prior_23_root_panel"], "provenance": [plan_artifact, artifact],
                      "historical32": {"action": raw["action"], "trace": raw["trace"], "source": artifact,
                                       "receipt_elapsed_seconds": raw["clock"]["receipt_elapsed_seconds"]}})
    slow_roots = [_materialize(row, reasons=["all_prior_32_margin_exceedances"]) for row in slow]
    blocked_observers = {item["observer_sha256"] for item in panel + slow_roots}

    # Include one first-search decision per seat in addition to ordinary controls.
    first_controls = []
    for seat in (0, 1):
        pool = [row for row in candidates if row["seat"] == seat and row["ply"] in (0, 1)
                and row["observer_sha256"] not in blocked_observers]
        require(bool(pool), "missing first-decision control")
        chosen = min(pool, key=lambda row: (selection_key(row, DOMAIN + "/first"), row["fixture_id"]))
        first_controls.append(chosen)
        blocked_observers.add(chosen["observer_sha256"])
    controls = select_stratified(candidates, domain=DOMAIN + "/controls", per_cell=controls_per_cell,
                                excluded_observers=blocked_observers,
                                excluded_pairs={row["pair_index"] for row in first_controls})
    control_roots = [_materialize(row, reasons=["first_search_decision_control"]) for row in first_controls]
    control_roots += [_materialize(row, reasons=["hash_selected_ordinary_control"]) for row in controls]
    # First-decision controls lead pass one; all original roots remain.
    development = control_roots[:2] + panel + slow_roots + control_roots[2:]

    # Preserve the exact failed64 hidden-state fixture and receipt identity. Its
    # lawful root already exists in the23-root panel, so attach it as a variant.
    failures64 = []
    for failure in summary64["timing_failures"]:
        raw, artifact = _read(attempt / "qualification_64" / f"{failure['index']:03d}.json")
        require(raw["index"] == failure["index"] and raw["arm"] == failure["arm"] and
                raw["clock"]["receipt_elapsed_seconds"] == failure["elapsed_seconds"],
                "failed64 provenance mismatch")
        found = next((item for item in development if item["observer_sha256"] == digest(raw["observer"])), None)
        require(found is not None, "failed64 root absent from existing panel")
        found["reasons"].append("prior_64_margin_failure")
        found.setdefault("preserved_variants", []).append({"snapshot": raw["before"],
                                                           "reason": "exact_prior_64_failure_state", "source": artifact})
        found["provenance"].append(artifact)
        failures64.append({**failure, "fixture_id": found["fixture_id"], "source": artifact,
                           "variant": raw["variant"], "snapshot_sha256": digest(raw["before"])})
    require(len(failures64) == 1, "expected one preserved64 failure")

    development_pairs = {item["pair_index"] for item in development if "pair_index" in item}
    replay_pairs = {item["pair_index"] for item in replay["decisions"]}
    excluded_observers = {item["observer_sha256"] for item in development}
    excluded_observers.update(item["observer_sha256"] for item in replay["decisions"])
    held = select_stratified(candidates, domain=DOMAIN + "/validation", per_cell=validation_per_cell,
                            excluded_observers=excluded_observers,
                            excluded_pairs=development_pairs | replay_pairs)
    validation = [_materialize(row, reasons=["saved_state_optimization_holdout"], historical32=False) for row in held]
    require(len({item["observer_sha256"] for item in development}) == len(development),
            "duplicate development lawful root")
    return {"schema": "splendorust_timing_saved_fixtures_v1", "domain": DOMAIN,
            "ruleset_id": guard.RULESET_ID, "catalogue_sha256": guard.CATALOGUE_SHA256,
            "turn_contract": guard.TURN_CONTRACT, "runtime_sources": runtime_sources,
            "source_attempt": str(attempt), "source_artifacts": [plan_artifact, summary_artifact,
                                                                      replay_artifact, completion_artifact],
            "selection_inputs": inventory, "selection_inputs_sha256": digest(inventory),
            "development": development, "validation": validation,
            "development_schedule": repeated_schedule(development), "prior_64_failures": failures64,
            "counts": {"prior_panel": len(panel), "slow32": len(slow_roots),
                       "first_decision_controls": len(first_controls), "ordinary_controls": len(controls),
                       "development": len(development), "validation": len(validation),
                       "main32_candidate_decisions": len(candidates)},
            "selection": {"controls_per_phase_seat": controls_per_cell,
                          "validation_per_phase_seat": validation_per_cell,
                          "excluded_development_pairs": sorted(development_pairs),
                          "excluded_prior_replay_pairs": sorted(replay_pairs),
                          "validation_unique_pairs": True, "ranking_uses_timing": False,
                          "ranking_uses_outcomes": False,
                          "timing_used_only_to_identify_all_prior_slow_decisions": True},
            "caveats": ["Saved validation states are held out from this optimization, not fresh independent game data.",
                        "All main timing bytes are scanned to identify the exhaustive slow set; validation timing values are not emitted.",
                        "Validation excludes slow/development pairs, so it cannot estimate an unconditional tail-failure rate.",
                        "Earlier sampled-replay pairs and prior panel observations are excluded; all historical game states were replay-verified.",
                        "First-search controls are first decisions from saved games; true worker-cold behavior depends on the execution schedule.",
                        "No validation model call is authorized merely by preparing this manifest."]}

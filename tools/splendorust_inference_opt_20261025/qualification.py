"""Model-free, prespecified checks for one exact-output inference candidate.

The earlier reader candidate is not installed. Its pure saved-state validator is
reused because the scientific players, paired schedule and timing gates match.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from tools import future_training_rules_20261021 as guard
from tools.splendorust_buffered_reader_20261025 import qualification as prior
from tools.splendorust_timing_20261024 import fixtures as saved

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / "outputs/splendorust_component_timing_20261025_01"
BUFFERED = ROOT / "outputs/splendorust_buffered_reader_20261025_01"
MAIN = ROOT / "outputs/splendorust_search_scale_20261024_01"
DIAGNOSTIC = ROOT / "outputs/splendorust_timing_20261024_01"
DOMAIN = "splendorust_inference_graph_reserved_20261025_v1"
OPS_DOMAIN = "splendorust_inference_graph_ops_20261025_v1"
SCHEMA = "splendorust_inference_opt_qualification_v1"
PARITY_SCHEMA = "splendorust_inference_opt_graph_parity_plan_v1"
PARITY_SOURCE = COMPONENT / "workers/overhead_64_baseline/entrypoint_forensics.jsonl"
MODES = ("baseline", "candidate")
QUOTAS = (64, 32)
MARGIN_SECONDS = .7
MINIMUM_MEAN_REDUCTION = .05


def require(value, message):
    if not value:
        raise ValueError("Inference qualification: " + message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(path):
    return prior._artifact(path)


def _same(left, right):
    return saved.digest(left) == saved.digest(right)


def make_schedule(panels):
    return prior.make_schedule(panels)


def snapshot_for_request(plan, spec):
    return prior.snapshot_for_request(plan, spec)


def timing_gate(baseline, candidate):
    return prior.timing_gate(baseline, candidate)


def verify_sources(plan, boundary=lambda: None):
    for source in plan["source_artifacts"]:
        boundary()
        require(_artifact(ROOT / source["path"]) == source, "frozen historical evidence changed")


def validate_panel(rows, plan, *, boundary=lambda: None):
    # The inherited validator requires each raw trusted receipt, full callback
    # trace and independently checked successor, then every split's own gate.
    return prior.validate_panel(rows, plan, boundary=boundary)


def select_holdout_metadata(candidates, *, excluded_pairs, excluded_observers, read_fixture):
    """Hash identifiers only, then materialize the winning saved candidate.

    Neither the stored timing nor the action/trace/outcome enters its rank.
    Previously observed roots and whole previously selected pairs are excluded.
    """
    used_pairs, used_observers = set(excluded_pairs), set(excluded_observers)
    result = []
    def key(row):
        return saved.digest([DOMAIN, row["pair_index"], row["game_index"], row["ply"], row["seat"]])
    for phase, seat in saved.CELLS:
        pool = sorted((row for row in candidates if row["phase"] == phase and row["seat"] == seat),
                      key=lambda row: (key(row), row["fixture_id"]))
        count = 0
        for row in pool:
            if row["pair_index"] in used_pairs:
                continue
            fixture = read_fixture(row)
            if fixture["observer_sha256"] in used_observers:
                continue
            result.append({**fixture, "selection_sha256": key(row), "pair_index": row["pair_index"],
                           "game_index": row["game_index"], "ply": row["ply"], "seat": seat, "phase": phase})
            used_pairs.add(row["pair_index"])
            used_observers.add(fixture["observer_sha256"])
            count += 1
            if count == 2:
                break
        require(count == 2, f"insufficient fresh reserved roots for {phase}, seat{seat}")
    return result


def prior_panel_exclusions(documents):
    """Read identities from fixed previous manifests, never their timings."""
    pairs, observers = set(), set()
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("observer_sha256"), str):
                observers.add(value["observer_sha256"])
                if type(value.get("pair_index")) is int:
                    pairs.add(value["pair_index"])
            if "excluded_main_pairs" in value:
                pairs.update(value["excluded_main_pairs"])
            if "excluded_observers" in value:
                observers.update(value["excluded_observers"])
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for document in documents:
        visit(document)
    require(all(type(pair) is int and 0 <= pair < 100 for pair in pairs), "prior main pair identity")
    require(all(re.fullmatch(r"[0-9a-f]{64}", item) for item in observers), "prior observer identity")
    return pairs, observers


def build_plan():
    runtime_sources = guard.verify_runtime_sources()
    component_path = COMPONENT / "qualification_plan.json"
    previous = _read(component_path)
    require(previous["ruleset_id"] == guard.RULESET_ID and
            previous["catalogue_sha256"] == guard.CATALOGUE_SHA256, "corrected development source")
    require(_read(COMPONENT / "completion.json")["status"] == "completed_verified" and
            _read(COMPONENT / "execution.json")["status"] == "completed", "verified component diagnostic required")
    require(_read(MAIN / "completion.json")["status"] == "completed_verified", "verified main saved corpus required")
    development = deepcopy(previous["panels"]["development"])
    require(len(development) == len({item["observer_sha256"] for item in development}) == 24,
            "fixed24 development roots")
    require([item["fixture_id"].rsplit("/", 1)[-1] for item in development[:3]] ==
            ["0042.json", "0048.json", "0050.json"], "original three slow roots must lead")

    exclusion_paths = [component_path, BUFFERED / "qualification_plan.json",
                       ROOT / "outputs/splendorust_timing_opt_20261024_01/qualification_plan.json",
                       DIAGNOSTIC / "fixtures.json", MAIN / "qualification_plan_32.json",
                       MAIN / "qualification_plan_64.json", MAIN / "replay_sample_manifest.json"]
    exclusion_paths += [ROOT / f"outputs/splendorust_one_second_20261024_0{i}/qualification_plan.json"
                        for i in (1, 2, 3)]
    documents = [_read(path) for path in exclusion_paths]
    excluded_pairs, excluded_observers = prior_panel_exclusions(documents)
    manifest = _read(DIAGNOSTIC / "fixtures.json")
    require(manifest["ruleset_id"] == guard.RULESET_ID and
            manifest["catalogue_sha256"] == guard.CATALOGUE_SHA256, "corrected saved source index")
    require(saved.digest(manifest["selection_inputs"]) == manifest["selection_inputs_sha256"],
            "historical source index digest changed")
    schedule_path = MAIN / "schedule.json"
    schedule = {row["game_index"]: row for row in _read(schedule_path) if row["stage"] == "main"}
    candidates = []
    for source in manifest["selection_inputs"]:
        path = Path(source["path"])
        match = re.fullmatch(r"p(\d+)_g(\d+)", path.parent.parent.name)
        require(match is not None and path.parent.name == "moves" and path.is_relative_to(MAIN / "games/main"),
                "bounded historical main index shape")
        pair_index, game_index = map(int, match.groups())
        ply, spec = int(path.stem), schedule[game_index]
        require(spec["pair_index"] == pair_index, "saved schedule/index binding")
        if ply % 2 != spec["search_seat"]:
            continue
        candidates.append({"pair_index": pair_index, "game_index": game_index, "ply": ply,
                           "seat": spec["search_seat"], "phase": saved.phase_for(ply),
                           "fixture_id": path.relative_to(ROOT).as_posix(), "source": source})
    def materialize(metadata):
        return prior._fixture(metadata["source"]["path"], "fresh_reserved_identifier_hash_selected_saved_root",
                              expected_quota=32, expected_artifact=metadata["source"])
    holdout = select_holdout_metadata(candidates, excluded_pairs=excluded_pairs,
                                     excluded_observers=excluded_observers, read_fixture=materialize)
    panels = {"development": development, "holdout": holdout}
    require(len(holdout) == 12 and len({item["observer_sha256"] for rows in panels.values() for item in rows}) == 36,
            "fresh reserved roots overlap")
    # Baseline component receipts are additional uninstrumented exact-history
    # targets. The profiled arm is excluded, and history cannot select roots.
    history_paths = []
    for stage in ("overhead", "diagnostic"):
        for spec in previous[stage + "_schedule"]:
            if spec["mode"] == "baseline":
                history_paths.append(COMPONENT / stage / f"{spec['index']:04d}.json")
    require(len(history_paths) == 120, "fixed120 baseline history rows")
    history_inventory = prior.attach_matching_history(panels, history_paths)
    sources = {}
    source_paths = exclusion_paths + [schedule_path, COMPONENT / "completion.json", COMPONENT / "execution.json",
                                      MAIN / "completion.json"]
    for path in source_paths:
        artifact = _artifact(path)
        sources[artifact["path"]] = artifact
    for item in history_inventory:
        sources[item["path"]] = item
    for rows in panels.values():
        for fixture in rows:
            for item in fixture["provenance"]:
                sources[item["path"]] = item
    result = {"schema": SCHEMA, "domain": DOMAIN, "ruleset_id": guard.RULESET_ID,
              "catalogue_sha256": guard.CATALOGUE_SHA256, "turn_contract": guard.TURN_CONTRACT,
              "runtime_sources": runtime_sources, "source_artifacts": [sources[key] for key in sorted(sources)],
              "panels": panels, "timed_requests": make_schedule(panels),
              "historical_matching_scope": {"additional_component_baseline_rows": 120,
                                            "inventory_sha256": saved.digest(history_inventory),
                                            "matching_records": sum(len(item["historical_records"])
                                                                    for rows in panels.values() for item in rows)},
              "selection": {"ranking_uses_timing": False, "ranking_uses_actions_or_outcomes": False,
                            "holdout_unique_pairs": True, "holdout_per_phase_seat": 2,
                            "excluded_main_pairs": sorted(excluded_pairs),
                            "excluded_observers": sorted(excluded_observers),
                            "main_candidate_metadata_count": len(candidates),
                            "prior_panel_manifest_paths": [path.relative_to(ROOT).as_posix() for path in exclusion_paths]},
              "gates": {"candidate_margin_seconds": .7, "hard_deadline_seconds": 1.,
                        "minimum_mean_reduction_per_quota_panel": .05,
                        "candidate_p95_no_worse_per_quota_panel": True,
                        "all_four_quota_panel_gates_required": True,
                        "baseline_margin_exceedances_preserved_not_disqualifying": True},
              "caveats": ["The24 development roots are selected controls, not an unbiased tail-latency sample.",
                          "The12 fresh reserved roots exclude every prior timing panel and diagnosed observer but are saved corrected-game data, not newly generated games.",
                          "Historical raw rows contain times/actions; these are not used for selection. They are inspected only after identifier-based rank and exclusions are fixed.",
                          "Mean/p95 thresholds are engineering point estimates, not uncertainty bounds or a universal deadline guarantee.",
                          "Failure of any reserved gate stops this candidate; no tuning against that panel. Fresh operational admission remains mandatory."]}
    require(len(result["timed_requests"]) == 288, "exact288 timed requests")
    verify_sources(result)
    return result


def select_parity_roots(lines):
    """Validate the complete forensic chain; select first six lawful root inputs.

    Lines are raw bytes, retaining exact serialized line hashes. Only first
    callbacks of completed ordinary search requests can supply neural inputs.
    """
    roots, seen, previous = [], set(), None
    for sequence, raw in enumerate(lines):
        require(isinstance(raw, bytes), "forensic source lines must retain raw bytes")
        row = json.loads(raw)
        require(row.get("sequence") == sequence and row.get("previous_sha256") == previous,
                "forensic sequence/hash chain changed")
        previous = saved.digest(row)
        event = row["event"]
        if len(roots) == 6 or not (event.get("arm") == "search" and event.get("event") == "result" and
                                  event.get("operation") == "ordinary_worker_request"):
            continue
        phases = event["result"]["diagnostics"]["phases"]
        if not phases or not phases[0].get("inference"):
            continue
        callback = phases[0]["inference"][0]
        tokens = callback["tokens"]
        require(callback["batch"] == 1 and len(tokens) == 31*48 and
                all(type(value) in (int, float) and math.isfinite(value) for value in tokens),
                "recorded first callback must be finite batch1 shape31x48")
        input_hash = saved.digest(tokens)
        require(input_hash == callback["input_sha256"] and
                saved.digest(callback["outputs"]) == callback["output_sha256"], "recorded callback hash mismatch")
        if not any(value != 0 for value in tokens) or input_hash in seen:
            continue
        observer = event["request"]["observer"]
        require(observer["ruleset_id"] == guard.RULESET_ID and
                observer["catalogue_sha256"] == guard.CATALOGUE_SHA256, "corrected parity source observer")
        roots.append({"tokens": tokens, "source_line_1based": sequence+1, "source_sequence": sequence,
                      "source_line_sha256": hashlib.sha256(raw).hexdigest(), "source_record_sha256": previous,
                      "case_id": event["case_id"], "observer_sha256": saved.digest(observer),
                      "phase_index": 0, "callback_index": 0, "shape": [1, 31, 48],
                      "input_sha256": input_hash, "historical_output_sha256": callback["output_sha256"],
                      "historical_outputs": callback["outputs"]})
        seen.add(input_hash)
    require(len(roots) == 6, "six distinct nonzero first-callback roots required")
    return roots


def build_parity_plan():
    require(_read(COMPONENT / "completion.json")["status"] == "completed_verified", "verified parity source required")
    roots = select_parity_roots(PARITY_SOURCE.read_bytes().splitlines(keepends=True))
    fixtures, provenance = [], {}
    for ordinal, root in enumerate(roots, 1):
        fixture_id = f"recorded_root_{ordinal:02d}"
        fixtures.append({"fixture_id": fixture_id, "batch": 1, "tokens": root["tokens"]})
        provenance[fixture_id] = {key: value for key, value in root.items() if key != "tokens"}
        provenance[fixture_id]["source_path"] = PARITY_SOURCE.relative_to(ROOT).as_posix()
    for batch, indices, suffix in ((2, [0, 1], "first2"), (8, [0, 1, 2, 3, 4, 5, 0, 1], "cycle6")):
        fixture_id = f"composite_batch{batch}_{suffix}"
        tokens = [value for index in indices for value in roots[index]["tokens"]]
        fixtures.append({"fixture_id": fixture_id, "batch": batch, "tokens": tokens})
        provenance[fixture_id] = {"shape": [batch, 31, 48], "input_sha256": saved.digest(tokens),
                                  "source_fixture_ids": [f"recorded_root_{index+1:02d}" for index in indices]}
    for batch in (1, 2, 8):
        fixture_id = f"architecture_zero_batch{batch}"
        tokens = [0.] * (batch*31*48)
        fixtures.append({"fixture_id": fixture_id, "batch": batch, "tokens": tokens})
        provenance[fixture_id] = {"shape": [batch, 31, 48], "input_sha256": saved.digest(tokens),
                                  "source": "architecture-shaped deterministic float32 zeros"}
    return {"schema": PARITY_SCHEMA, "operation": "graph_parity", "fixtures": fixtures,
            "provenance_by_fixture": provenance,
            "source_artifacts": [_artifact(PARITY_SOURCE), _artifact(COMPONENT / "completion.json")],
            "selection": "Source-order first six distinct nonzero batch1 first callbacks of ordinary search result records; composites first2 and cyclic6; zeros1/2/8.",
            "reserved_roots_used": False, "exact_float32_bitwise_outputs_required": True,
            "relative_or_absolute_tolerance_permitted": False}


def operational_schedule():
    result = []
    for pair in range(3):
        seed = str(int(saved.digest([OPS_DOMAIN, "operational", pair]), 16))
        for game in range(2):
            result.append({"stage": "operational", "pair_index": pair, "game_index": 2*pair+game,
                           "challenger_seat": (pair+game) % 2, "seed_decimal": seed,
                           "challenger_quota": 64, "opponent_quota": 32,
                           "challenger_mode": "candidate", "opponent_mode": "candidate"})
    return result

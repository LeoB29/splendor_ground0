"""Frozen saved-position checks for one bounded stdout-reader optimization.

Selection, replay and timing gates are model-free. Historical source files are
read only. Both modes retain the same corrected scientific worker and search.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import statistics

from tools import future_training_rules_20261021 as guard
from tools.splendorust_search_scale_20261024 import qualification as archive
from tools.splendorust_timing_20261024 import fixtures as saved
from tools.splendorust_timing_opt_20261024 import qualification as preceding

ROOT = Path(__file__).resolve().parents[2]
FAILED = ROOT / "outputs/splendorust_search_scale_opt_20261024_01"
PRIOR = ROOT / "outputs/splendorust_timing_opt_20261024_01"
MAIN = ROOT / "outputs/splendorust_search_scale_20261024_01"
DIAGNOSTIC = ROOT / "outputs/splendorust_timing_20261024_01"
POSTMORTEM = ROOT / "outputs/splendorust_slow_moves_postmortem_20261025_01"
DOMAIN = "splendorust_buffered_response_saved_20261025_v1"
SCHEMA = "splendorust_buffered_reader_qualification_v1"
MODES = ("baseline", "candidate")
QUOTAS = (64, 32)
MARGIN_SECONDS = .7
MINIMUM_MEAN_REDUCTION = .05
FAILURE_PLIES = (42, 48, 50)
SAME_GAME_CONTROLS = (0, 8, 12, 26, 36, 38, 40, 44, 46, 52, 54)
PRIOR_CONTROLS = (
    ("p000_g0", 0), ("p000_g0", 8), ("p000_g0", 20),
    ("p001_g2", 9), ("p001_g3", 2), ("p001_g3", 24),
    ("p002_g4", 8), ("p002_g4", 16), ("p002_g5", 15), ("p002_g5", 21),
)


def require(value, message):
    if not value:
        raise ValueError("Buffered-reader qualification: " + message)


def _same(left, right):
    return saved.digest(left) == saved.digest(right)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(path):
    return archive.artifact(Path(path))


def _completed(row, snapshot, *, quota):
    require(row.get("arm") == "search", "both quotas must use the scientific search arm")
    elapsed, science = preceding._completed(row, snapshot, quota=quota)
    return elapsed, {**science, "after": row["after"]}


def _fixture(path, reason, *, expected_quota, expected_artifact=None):
    """Bind the full historical decision, its raw clock and its exact input."""
    path = Path(path).resolve()
    source = _artifact(path)
    if expected_artifact is not None:
        require(source["sha256"] == expected_artifact["sha256"] and
                source["bytes"] == expected_artifact["bytes"], "saved source index changed")
    row = _read(path)
    quota = row.get("quota", row.get("challenger_simulations"))
    require(type(quota) is int and quota == expected_quota, "historical search quota")
    _, science = _completed(row, row["before"], quota=quota)
    clock_path = ROOT / row["worker_clock_file"]
    input_path = clock_path.parent / "inputs" / clock_path.name
    require(_same(_read(clock_path), row["clock"]), "historical clock differs from move record")
    require(_same(_read(input_path), {"player": "search", "observer": row["observer"],
                                    "before": row["before"]}), "historical saved input differs")
    return {"fixture_id": path.relative_to(ROOT).as_posix(), "snapshot": row["before"],
            "observer_sha256": saved.digest(row["observer"]), "reason": reason,
            "source": source, "provenance": [source, _artifact(clock_path), _artifact(input_path)],
            "historical": {str(quota): science},
            "historical_records": [{"quota": quota, "source": source, "science": science}]}


def attach_matching_history(panels, paths):
    """Include every matching record in the explicitly bounded prior-game pool.

    A swapped-seat game may already provide the other quota's answer at the
    identical saved root. Its repeated-decision receipt is also independently
    bound. This scan never changes root selection or reads game outcomes.
    """
    fixtures = {saved.digest(item["snapshot"]): item for items in panels.values() for item in items}
    inventory = []
    for path in sorted(paths):
        artifact = _artifact(path)
        inventory.append(artifact)
        row = _read(path)
        fixture = fixtures.get(saved.digest(row["before"]))
        if fixture is None or saved.digest(row["observer"]) != fixture["observer_sha256"]:
            continue
        if any(record["source"]["path"] == artifact["path"] for record in fixture["historical_records"]):
            continue
        quota = row.get("quota", row.get("challenger_simulations"))
        require(type(quota) is int and quota in QUOTAS, "matching historical quota")
        extra = _fixture(path, "matching_archived_root", expected_quota=quota)
        science = extra["historical"][str(quota)]
        require(_same(fixture["historical"].setdefault(str(quota), science), science),
                "available historical outputs disagree")
        fixture["historical_records"].extend(extra["historical_records"])
        fixture["provenance"].extend(extra["provenance"])
    return inventory


def make_schedule(panels):
    """At most two resident trees: all64 first, then32; reverse each second pass.

    Alternating per-root mode order is reversed in pass two. The three preserved
    slow roots lead the first64 pass and are never moved behind ordinary roots.
    """
    result = []
    for quota in QUOTAS:
        for panel in ("development", "holdout"):
            for pass_index in range(2):
                order = list(range(len(panels[panel])))
                if pass_index:
                    order.reverse()
                for fixture_index in order:
                    modes = list(MODES)
                    if (fixture_index + pass_index) % 2:
                        modes.reverse()
                    for mode in modes:
                        fixture = panels[panel][fixture_index]
                        result.append({"index": len(result), "panel": panel, "fixture_index": fixture_index,
                                       "fixture_id": fixture["fixture_id"], "variant": "original",
                                       "pass_index": pass_index, "mode": mode, "quota": quota})
    return result


def select_holdout_metadata(candidates, *, excluded_pairs, excluded_observers, read_fixture):
    """Two roots per phase/seat, all from distinct unused historical game pairs.

    The hash uses identifiers only. read_fixture is called only after a candidate
    wins the metadata ranking; its timing/action/outcome cannot affect that rank.
    """
    used_pairs, used_observers = set(excluded_pairs), set(excluded_observers)
    selected = []
    for phase, seat in saved.CELLS:
        pool = [row for row in candidates if row["phase"] == phase and row["seat"] == seat]
        pool.sort(key=lambda row: (saved.digest([DOMAIN, row["pair_index"], row["game_index"],
                                               row["ply"], row["seat"]]), row["fixture_id"]))
        count = 0
        for row in pool:
            if row["pair_index"] in used_pairs:
                continue
            fixture = read_fixture(row)
            if fixture["observer_sha256"] in used_observers:
                continue
            key = saved.digest([DOMAIN, row["pair_index"], row["game_index"], row["ply"], row["seat"]])
            selected.append({**fixture, "selection_sha256": key, "pair_index": row["pair_index"],
                             "game_index": row["game_index"], "ply": row["ply"], "seat": seat, "phase": phase})
            used_pairs.add(row["pair_index"])
            used_observers.add(fixture["observer_sha256"])
            count += 1
            if count == 2:
                break
        require(count == 2, f"insufficient reserved roots for {phase}, seat{seat}")
    return selected


def build_plan():
    runtime_sources = guard.verify_runtime_sources()
    manifest_path = DIAGNOSTIC / "fixtures.json"
    manifest = _read(manifest_path)
    require(manifest["ruleset_id"] == guard.RULESET_ID and
            manifest["catalogue_sha256"] == guard.CATALOGUE_SHA256, "corrected saved corpus identity")
    require(_read(PRIOR / "completion.json")["status"] == "completed_verified" and
            _read(PRIOR / "execution.json")["status"] == "completed", "preceding optimization not verified")
    require(_read(MAIN / "completion.json")["status"] == "completed_verified", "historical main corpus not verified")
    development = [_fixture(FAILED / "games/operational/p000_g0/moves" / f"{ply:04d}.json",
                            "preserved_slow64" if ply in FAILURE_PLIES else "same_game_control", expected_quota=64)
                   for ply in FAILURE_PLIES + SAME_GAME_CONTROLS]
    development += [_fixture(PRIOR / "games/ops64" / game / "moves" / f"{ply:04d}.json",
                             "previous_successful_ops64_control", expected_quota=64)
                    for game, ply in PRIOR_CONTROLS]
    require(len(development) == 24, "fixed development size")

    replay_path = MAIN / "replay_sample_manifest.json"
    replay = _read(replay_path)
    prior_roots = manifest["development"] + manifest["validation"]
    excluded_pairs = {item["pair_index"] for item in prior_roots if "pair_index" in item}
    excluded_pairs.update(item["pair_index"] for item in replay["decisions"])
    excluded_observers = {item["observer_sha256"] for item in prior_roots + development}
    excluded_observers.update(item["observer_sha256"] for item in replay["decisions"])
    # Diagnosis inspected all earlier ops64 decisions and the failed game; both
    # are different corpora. Exclude every inspected observer, not merely the24
    # selected development roots. The analysis carries exact source identities.
    postmortem_path = POSTMORTEM / "analysis.json"
    postmortem = _read(postmortem_path)
    diagnosis_paths = []
    for source in postmortem["sources"]:
        path = ROOT / source["path"]
        if "/moves/" in path.as_posix() and path.suffix == ".json":
            require(_artifact(path)["sha256"] == source["sha256"], "diagnosis source changed")
            excluded_observers.add(saved.digest(_read(path)["observer"]))
            diagnosis_paths.append(_artifact(path))
    schedule_path = MAIN / "schedule.json"
    schedule = {spec["game_index"]: spec for spec in _read(schedule_path) if spec["stage"] == "main"}
    candidates = []
    for source in manifest["selection_inputs"]:
        path = Path(source["path"])
        match = re.fullmatch(r"p(\d+)_g(\d+)", path.parent.parent.name)
        require(match is not None and path.parent.name == "moves", "historical index shape")
        pair_index, game_index = map(int, match.groups())
        ply = int(path.stem)
        spec = schedule[game_index]
        require(spec["pair_index"] == pair_index and spec["stage"] == "main", "historical schedule/index binding")
        if ply % 2 != spec["search_seat"]:
            continue
        candidates.append({"pair_index": pair_index, "game_index": game_index, "ply": ply,
                           "seat": spec["search_seat"], "phase": saved.phase_for(ply),
                           "fixture_id": path.relative_to(ROOT).as_posix(), "source": source})
    def materialize(metadata):
        return _fixture(metadata["source"]["path"], "reserved_hash_selected_saved_root", expected_quota=32,
                        expected_artifact=metadata["source"])
    holdout = select_holdout_metadata(candidates, excluded_pairs=excluded_pairs,
                                     excluded_observers=excluded_observers, read_fixture=materialize)
    panels = {"development": development, "holdout": holdout}
    require(len({item["observer_sha256"] for items in panels.values() for item in items}) == 36,
            "development/holdout roots overlap")
    historical_paths = (list((PRIOR / "games").glob("*/*/moves/*.json")) +
                        list((PRIOR / "repeated_decisions").glob("*.json")) +
                        list((FAILED / "games/operational/p000_g0/moves").glob("*.json")))
    require(len(historical_paths) == 774, "bounded historical game/reproduction inventory changed")
    matching_inventory = attach_matching_history(panels, historical_paths)
    source_paths = [manifest_path, replay_path, schedule_path, postmortem_path,
                    PRIOR / "completion.json", PRIOR / "execution.json", MAIN / "completion.json"]
    sources = {_artifact(path)["path"]: _artifact(path) for path in source_paths}
    sources.update({item["path"]: item for item in diagnosis_paths})
    sources.update({item["path"]: item for item in matching_inventory})
    for items in panels.values():
        for item in items:
            sources.update({source["path"]: source for source in item["provenance"]})
    requests = make_schedule(panels)
    require(len(requests) == 288 and len(holdout) == 12, "frozen request/panel counts")
    return {"schema": SCHEMA, "domain": DOMAIN, "ruleset_id": guard.RULESET_ID,
            "catalogue_sha256": guard.CATALOGUE_SHA256, "turn_contract": guard.TURN_CONTRACT,
            "runtime_sources": runtime_sources, "source_artifacts": [sources[key] for key in sorted(sources)],
            "panels": panels, "timed_requests": requests,
            "historical_matching_scope": {"game_and_reproduction_records": len(matching_inventory),
                                          "inventory_sha256": saved.digest(matching_inventory),
                                          "matching_records": sum(len(item["historical_records"])
                                                                  for items in panels.values() for item in items)},
            "selection": {"ranking_uses_timing": False, "ranking_uses_actions_or_outcomes": False,
                          "holdout_unique_pairs": True, "holdout_per_phase_seat": 2,
                          "excluded_main_pairs": sorted(excluded_pairs),
                          "excluded_observers": sorted(excluded_observers),
                          "main_candidate_metadata_count": len(candidates)},
            "gates": {"candidate_margin_seconds": MARGIN_SECONDS, "hard_deadline_seconds": 1.,
                      "minimum_mean_reduction_per_quota_panel": MINIMUM_MEAN_REDUCTION,
                      "candidate_p95_no_worse_per_quota_panel": True,
                      "all_four_quota_panel_gates_required": True,
                      "baseline_margin_exceedances_preserved_not_disqualifying": True},
            "caveats": ["These selected saved controls are not an unbiased tail-latency sample.",
                        "Reserved roots are held out from this optimization, not fresh independent game data.",
                        "Every historical main state was previously replay-verified; its raw record contains timing and action fields, but neither enters selection.",
                        "Historical comparison includes each selected source and every exact snapshot/observer match in the774 preceding operational/reproduction and failed-game records; both quotas also require exact parent-mode/repeat agreement.",
                        "Engineering mean/p95 gates are point estimates, not uncertainty bounds or a universal latency guarantee.",
                        "A failed holdout gate stops this candidate; do not tune against it. Six fresh operational games remain mandatory."]}


def snapshot_for_request(plan, spec):
    require(spec["variant"] == "original", "only exact saved original states are scheduled")
    fixture = plan["panels"][spec["panel"]][spec["fixture_index"]]
    require(fixture["fixture_id"] == spec["fixture_id"], "fixture identity changed")
    return deepcopy(fixture["snapshot"])


def _distribution(values):
    require(bool(values) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "finite nonnegative timing samples")
    ordered = sorted(values)
    position = .95 * (len(values)-1)
    lo, hi = math.floor(position), math.ceil(position)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo),
            "minimum": ordered[0], "maximum": ordered[-1],
            "above_margin": sum(v > MARGIN_SECONDS for v in values)}


def timing_gate(baseline, candidate):
    require(len(baseline) == len(candidate), "paired timing count")
    old, new = _distribution(baseline), _distribution(candidate)
    require(old["mean"] > 0 and old["maximum"] <= 1 and new["maximum"] <= 1,
            "one-second failure or invalid baseline timing")
    checks = {"mean_reduction_at_least_five_percent": new["mean"] <= .95*old["mean"],
              "candidate_p95_no_worse": new["p95"] <= old["p95"],
              "all_candidate_within_margin": new["maximum"] <= MARGIN_SECONDS}
    passed = all(checks.values())
    return {"status": "passed" if passed else "timing_not_qualified", "qualified": passed,
            "checks": checks, "baseline": old, "candidate": new,
            "mean_reduction_fraction": 1-new["mean"]/old["mean"],
            "baseline_margin_exceedances_preserved": old["above_margin"]}


def verify_sources(plan, boundary=lambda: None):
    for source in plan["source_artifacts"]:
        boundary()
        require(_artifact(ROOT/source["path"]) == source, "frozen historical evidence changed")


def validate_panel(rows, plan, *, boundary=lambda: None):
    """All exact science/replay checks precede four separately enforced gates."""
    require(len(plan["panels"]["development"]) == 24 and len(plan["panels"]["holdout"]) == 12,
            "frozen panel sizes")
    schedule = make_schedule(plan["panels"])
    require(_same(schedule, plan["timed_requests"]) and len(schedule) == len(rows) == 288,
            "complete unchanged schedule")
    verify_sources(plan, boundary)
    timings = {panel: {str(q): {mode: [] for mode in MODES} for q in QUOTAS}
               for panel in plan["panels"]}
    groups, historical = {}, {str(q): 0 for q in QUOTAS}
    for row, spec in zip(rows, schedule, strict=True):
        boundary()
        require(all(key in row and _same(row[key], value) for key, value in spec.items()),
                "request identity/order changed")
        fixture = plan["panels"][spec["panel"]][spec["fixture_index"]]
        elapsed, science = _completed(row, snapshot_for_request(plan, spec), quota=spec["quota"])
        key = (spec["panel"], spec["fixture_id"], spec["quota"])
        require(_same(groups.setdefault(key, science), science), "mode/repeat action/full trace/transition changed")
        records = fixture["historical_records"]
        require(type(records) is list and bool(records), "bound historical records required")
        for record in records:
            require(type(record["quota"]) is int and record["quota"] in QUOTAS, "historical quota identity")
            if record["quota"] == spec["quota"]:
                require(_same(record["science"], science), "matching historical action/full trace/transition changed")
                historical[str(spec["quota"])] += 1
        timings[spec["panel"]][str(spec["quota"])][spec["mode"]].append(elapsed)
    gates = {panel: {q: timing_gate(values["baseline"], values["candidate"])
                     for q, values in quotas.items()} for panel, quotas in timings.items()}
    passed = all(result["qualified"] for quotas in gates.values() for result in quotas.values())
    return {"status": "passed" if passed else "timing_not_qualified", "qualified": passed,
            "requests": len(rows), "exact_scientific_groups": len(groups), "panels": gates,
            "historical_comparisons": historical, "permits_fresh_operational_games": passed,
            "failed_holdout_may_not_be_used_to_retune": True}

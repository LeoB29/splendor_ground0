"""Fixed64-only admission with an unchanged eager32 operational reference.

Model-free saved-state selection and exact historical/transition checks. The
preceding joint attempt stays negative; its graph and parity fixtures stay fixed.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from tools import future_training_rules_20261021 as guard
from tools.splendorust_buffered_reader_20261025 import qualification as archive
from tools.splendorust_inference_opt_20261025 import qualification as preceding
from tools.splendorust_timing_20261024 import fixtures as saved

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "outputs/splendorust_inference_opt_20261025_01"
MAIN = ROOT / "outputs/splendorust_search_scale_20261024_01"
SAMPLER = ROOT / "outputs/splendorust_timing_opt_20261024_01"
INDEX = ROOT / "outputs/splendorust_timing_20261024_01/fixtures.json"
DOMAIN = "splendorust_inference64_reserved_20261025_v1"
OPS_DOMAIN = "splendorust_inference64_ops_20261025_v1"
SCHEMA = "splendorust_inference64_qualification_v1"
MODES = ("baseline", "candidate")
QUOTAS = (64,)
MARGIN_SECONDS = .7
MINIMUM_MEAN_REDUCTION = .05


def require(value, message):
    if not value:
        raise ValueError("Inference64 qualification: " + message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(path):
    return archive._artifact(path)


def canonical_artifact_reference(source):
    """Normalize imported index identities without changing their bytes/hash.

    The old timing index uses absolute paths. The current freezer also collects
    artifact triples nested inside selection metadata, so those must use the
    same workspace-relative convention as ordinary source_artifacts.
    """
    path = (ROOT / source["path"]).resolve()
    require(path.is_relative_to(ROOT), "historical artifact escaped workspace")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": source["bytes"], "sha256": source["sha256"]}


def _same(left, right):
    return saved.digest(left) == saved.digest(right)


def _completed(row, snapshot):
    require(type(row.get("quota")) is int and row["quota"] == 64, "timed scientific quota must be64")
    return archive._completed(row, snapshot, quota=64)


def make_schedule(panels):
    rows = [row for row in archive.make_schedule(panels) if row["quota"] == 64]
    return [{**row, "index": index} for index, row in enumerate(rows)]


def snapshot_for_request(plan, spec):
    require(spec["quota"] == 64, "saved timing is64 only")
    return archive.snapshot_for_request(plan, spec)


def timing_gate(baseline, candidate):
    return archive.timing_gate(baseline, candidate)


def verify_sources(plan, boundary=lambda: None):
    for item in plan["source_artifacts"]:
        boundary()
        require(_artifact(ROOT / item["path"]) == item, "frozen source changed")


def select_holdout_metadata(candidates, *, excluded_observers, read_fixture):
    """Two per phase/seat, one per namespaced pair, ranked on identities only."""
    observers, pairs, selected = set(excluded_observers), set(), []
    def key(row):
        return saved.digest([DOMAIN, row["corpus"], row["source_stage"], row["pair_index"],
                             row["game_index"], row["ply"], row["seat"]])
    for phase, seat in saved.CELLS:
        pool = sorted((row for row in candidates if row["phase"] == phase and row["seat"] == seat),
                      key=lambda row: (key(row), row["fixture_id"]))
        count = 0
        for row in pool:
            if row["pair_key"] in pairs:
                continue
            fixture = read_fixture(row)
            if fixture["observer_sha256"] in observers:
                continue
            selected.append({**fixture, **{name: row[name] for name in
                             ("corpus", "source_stage", "pair_key", "pair_index", "game_index", "ply", "seat", "phase")},
                             "selection_sha256": key(row)})
            pairs.add(row["pair_key"])
            observers.add(fixture["observer_sha256"])
            count += 1
            if count == 2:
                break
        require(count == 2, f"insufficient fresh distinct-pair roots for {phase}, seat{seat}; no fallback")
    return selected


def _verified_corpus(folder, sources):
    """Bind successful prior ledger/log/cleanup verification, never game scores."""
    for name in ("completion.json", "execution.json", "all_game_replay.json", "all_worker_log_integrity.json"):
        path = folder / name
        value = _read(path)
        expected = "completed_verified" if name == "completion.json" else "completed" if name == "execution.json" else "passed"
        require(value["status"] == expected, "unverified archived source corpus")
        if name == "execution.json":
            require(value["exit_code"] == 0 and not value["cleanup_errors"], "archived cleanup did not pass")
        sources.append(_artifact(path))


def candidate_metadata(*, excluded_main_pairs):
    """A fixed14-pair pool: five main pairs and three prior operational groups.

    Schedules and filenames establish seats/quotas. Whole stored move files are
    hashed for provenance, but their timings/actions/outcomes do not rank roots.
    """
    manifest = _read(INDEX)
    require(manifest["ruleset_id"] == guard.RULESET_ID and
            manifest["catalogue_sha256"] == guard.CATALOGUE_SHA256 and
            saved.digest(manifest["selection_inputs"]) == manifest["selection_inputs_sha256"], "corrected source index")
    sources = [_artifact(INDEX)]
    for folder in (MAIN, SAMPLER):
        _verified_corpus(folder, sources)
    main_schedule_path, sampler_schedule_path = MAIN / "schedule.json", SAMPLER / "operational_schedule.json"
    main_schedule, sampler_schedule = _read(main_schedule_path), _read(sampler_schedule_path)
    sources += [_artifact(main_schedule_path), _artifact(sampler_schedule_path)]
    schedule = {row["game_index"]: row for row in main_schedule if row["stage"] == "main"}
    candidates = []
    def metadata(path, source, *, corpus, stage, spec, quota):
        match = re.fullmatch(r"p(\d+)_g(\d+)", path.parent.parent.name)
        require(match is not None and path.parent.name == "moves", "archived move path shape")
        pair, game = map(int, match.groups())
        require((pair, game) == (spec["pair_index"], spec["game_index"]), "archive schedule/path coupling")
        ply = int(path.stem)
        return {"corpus": corpus, "source_stage": stage,
                "pair_key": f"{corpus}/{stage}/pair{pair:03d}", "pair_index": pair,
                "game_index": game, "ply": ply, "seat": ply % 2, "phase": saved.phase_for(ply),
                "fixture_id": path.relative_to(ROOT).as_posix(), "historical_quota": quota,
                "source": canonical_artifact_reference(source)}
    for source in manifest["selection_inputs"]:
        path = Path(source["path"])
        require(path.is_relative_to(MAIN / "games/main"), "main source index escaped corpus")
        match = re.fullmatch(r"p(\d+)_g(\d+)", path.parent.parent.name)
        require(match is not None, "indexed pair path shape")
        pair, game = map(int, match.groups())
        spec, ply = schedule[game], int(path.stem)
        if pair in excluded_main_pairs or ply % 2 != spec["search_seat"]:
            continue
        candidates.append(metadata(path, source, corpus=MAIN.name, stage="main", spec=spec, quota=32))
    remaining = {row["pair_index"] for row in candidates}
    require(remaining == {17, 31, 52, 56, 68}, "five unused main pair identities changed")
    pools = [(MAIN, "operational", main_schedule, 344),
             (SAMPLER, "ops32", sampler_schedule, 340),
             (SAMPLER, "ops64", sampler_schedule, 350)]
    inventory = []
    for folder, stage, specs, expected_count in pools:
        specs = {row["game_index"]: row for row in specs if row["stage"] == stage}
        require(len(specs) == 6 and len({row["pair_index"] for row in specs.values()}) == 3, "fixed operational source pairs")
        paths = sorted((folder / "games" / stage).glob("*/moves/*.json"))
        require(len(paths) == expected_count, "bounded operational source move inventory changed")
        for path in paths:
            source = _artifact(path)
            inventory.append(source)
            game = int(path.parent.parent.name.split("_g")[1]); ply = int(path.stem); spec = specs[game]
            if folder == MAIN:
                if ply % 2 != spec["search_seat"]:
                    continue
                quota = 32
            else:
                quota = spec["challenger_quota"] if ply % 2 == spec["challenger_seat"] else spec["opponent_quota"]
            require(quota in (32, 64), "unsupported archived scientific quota")
            candidates.append(metadata(path, source, corpus=folder.name, stage=stage, spec=spec, quota=quota))
    sources.extend(inventory)
    require(len({row["pair_key"] for row in candidates}) == 14, "fixed14 namespaced source pairs")
    return candidates, sources


def build_plan():
    runtime_sources = guard.verify_runtime_sources()
    previous_path = PRIOR / "qualification_plan.json"
    previous = _read(previous_path)
    require(previous["ruleset_id"] == guard.RULESET_ID and previous["catalogue_sha256"] == guard.CATALOGUE_SHA256,
            "corrected predecessor required")
    require(_read(PRIOR / "completion.json")["status"] == "completed_verified" and
            _read(PRIOR / "execution.json")["status"] == "completed" and
            _read(PRIOR / "qualification_outcome.json")["inference_graph_qualified"] is False,
            "preceding negative joint attempt must remain verified negative")
    development = deepcopy(previous["panels"]["development"])
    require(len(development) == 24, "fixed24 development roots")
    require([row["fixture_id"].rsplit("/", 1)[-1] for row in development[:3]] ==
            ["0042.json", "0048.json", "0050.json"], "original slow roots must lead")
    excluded_main_pairs, excluded_observers = preceding.prior_panel_exclusions([previous])
    require(len(excluded_main_pairs) == 95, "all95 earlier selected main pairs must remain excluded")
    extra_exclusion_paths = [MAIN / "replay_sample_manifest.json"]
    replay_paths = sorted((SAMPLER / "repeated_decisions").glob("*.json"))
    require(len(replay_paths) == 24, "fixed24 preceding operational repetitions")
    for path in replay_paths:
        excluded_observers.add(saved.digest(_read(path)["observer"]))
    extra_exclusion_paths += replay_paths
    for row in _read(MAIN / "replay_sample_manifest.json")["decisions"]:
        excluded_observers.add(row["observer_sha256"])
    parity = _read(PRIOR / "graph_parity_plan.json")
    excluded_observers.update(row["observer_sha256"] for row in parity["provenance_by_fixture"].values()
                              if "observer_sha256" in row)
    candidates, sources = candidate_metadata(excluded_main_pairs=excluded_main_pairs)
    def materialize(row):
        return archive._fixture(ROOT / row["fixture_id"], "fresh_reserved_namespaced_pair_hash",
                                expected_quota=row["historical_quota"], expected_artifact=row["source"])
    holdout = select_holdout_metadata(candidates, excluded_observers=excluded_observers, read_fixture=materialize)
    panels = {"development": development, "holdout": holdout}
    require(len(holdout) == len({row["pair_key"] for row in holdout}) == 12 and
            len({row["observer_sha256"] for rows in panels.values() for row in rows}) == 36,
            "fresh12 distinct-pair roots and no observer overlap required")
    history_paths = [PRIOR / "paired_timing" / f"{spec['index']:04d}.json" for spec in previous["timed_requests"]]
    require(len(history_paths) == 288, "fixed preceding288 timing records")
    history_inventory = archive.attach_matching_history(panels, history_paths)
    sources += history_inventory
    paths = [previous_path, PRIOR / "completion.json", PRIOR / "execution.json", PRIOR / "qualification_outcome.json",
             PRIOR / "graph_parity_plan.json"] + extra_exclusion_paths
    sources += [_artifact(path) for path in paths]
    for rows in panels.values():
        for fixture in rows:
            sources += fixture["provenance"]
    unique_sources = {item["path"]: item for item in sources}
    result = {"schema": SCHEMA, "domain": DOMAIN, "ruleset_id": guard.RULESET_ID,
              "catalogue_sha256": guard.CATALOGUE_SHA256, "turn_contract": guard.TURN_CONTRACT,
              "runtime_sources": runtime_sources, "source_artifacts": [unique_sources[key] for key in sorted(unique_sources)],
              "panels": panels, "timed_requests": make_schedule(panels),
              "selection": {"ranking_uses_timing": False, "ranking_uses_actions_or_outcomes": False,
                            "excluded_main_pairs": sorted(excluded_main_pairs),
                            "excluded_observers": sorted(excluded_observers),
                            "holdout_per_phase_seat": 2, "holdout_unique_namespaced_pairs": True,
                            "maximum_roots_per_namespaced_pair": 1, "wholly_untouched_source_games_claimed": False,
                            "candidate_metadata": candidates, "candidate_metadata_sha256": saved.digest(candidates),
                            "candidate_pairs": sorted({row["pair_key"] for row in candidates}),
                            "selected_pairs": sorted(row["pair_key"] for row in holdout)},
              "historical_matching_scope": {"additional_preceding_timing_rows": 288,
                                            "inventory_sha256": saved.digest(history_inventory),
                                            "matching_records": sum(len(row["historical_records"])
                                                                    for rows in panels.values() for row in rows)},
              "gates": {"candidate_margin_seconds": .7, "hard_deadline_seconds": 1.,
                        "minimum_mean_reduction_per_panel": .05, "candidate_p95_no_worse_per_panel": True,
                        "both64_panels_required": True, "baseline_margin_exceedances_preserved_not_disqualifying": True},
              "scope_selected_after_preceding_joint_result": True, "unchanged_eager32_reference": True,
              "caveats": ["The original joint32/64 attempt remains negative. This quota-specific scope was chosen after that result.",
                          "The24 development roots are selected controls, not an unbiased tail-latency sample.",
                          "The12 reserved positions and observer hashes are new to prior selected panels, from12 distinct namespaced pairs. Some operational source pairs previously supplied other development/replay roots; these are not wholly untouched or newly generated games.",
                          "Metadata ranking uses only corpus/stage/pair/game/ply/seat identities. Source timing, actions and outcomes never enter selection.",
                          "Historical32 answers prove provenance only; no32 answer is relabelled as a64 target. All current64 modes and repeats must agree exactly.",
                          "Mean/p95 thresholds are engineering point estimates. They do not establish a universal tail guarantee or stronger play.",
                          "A failed reserved gate stops this candidate; no tuning against it. Fresh operational admission remains mandatory."]}
    require(len(result["timed_requests"]) == 144, "exact144 saved timings")
    verify_sources(result)
    return result


def validate_panel(rows, plan, *, boundary=lambda: None):
    require(set(plan["panels"]) == {"development", "holdout"} and
            len(plan["panels"]["development"]) == 24 and len(plan["panels"]["holdout"]) == 12, "fixed panel sizes")
    schedule = make_schedule(plan["panels"])
    require(_same(schedule, plan["timed_requests"]) and len(schedule) == len(rows) == 144, "complete unchanged144 schedule")
    verify_sources(plan, boundary)
    timings = {panel: {mode: [] for mode in MODES} for panel in plan["panels"]}
    groups, historical = {}, 0
    for row, spec in zip(rows, schedule, strict=True):
        boundary()
        require(all(key in row and _same(row[key], value) for key, value in spec.items()), "request identity/order changed")
        fixture = plan["panels"][spec["panel"]][spec["fixture_index"]]
        elapsed, science = _completed(row, snapshot_for_request(plan, spec))
        key = (spec["panel"], spec["fixture_id"])
        require(_same(groups.setdefault(key, science), science), "mode/repeat action/full trace/transition changed")
        require(isinstance(fixture["historical_records"], list) and fixture["historical_records"], "bound historical evidence required")
        for record in fixture["historical_records"]:
            require(type(record["quota"]) is int and record["quota"] in (32, 64), "historical quota identity")
            if record["quota"] == 64:
                require(_same(record["science"], science), "historical64 scientific output changed")
                historical += 1
        timings[spec["panel"]][spec["mode"]].append(elapsed)
    panels = {panel: {"64": timing_gate(values["baseline"], values["candidate"])} for panel, values in timings.items()}
    passed = all(group["64"]["qualified"] for group in panels.values())
    return {"status": "passed" if passed else "timing_not_qualified", "qualified": passed,
            "requests": len(rows), "exact_scientific_groups": len(groups), "panels": panels,
            "historical_comparisons": {"64": historical}, "permits_fresh_operational_games": passed,
            "unchanged_eager32_reference": True, "failed_holdout_may_not_be_used_to_retune": True}


def build_parity_plan():
    # Exact entire frozen plan, including inputs, ordering and provenance. The
    # parent binds this prior file and equality-checks it before worker launch.
    return _read(PRIOR / "graph_parity_plan.json")


def operational_schedule():
    return [{"stage": "operational", "pair_index": pair, "game_index": 2*pair+game,
             "challenger_seat": (pair+game) % 2,
             "seed_decimal": str(int(saved.digest([OPS_DOMAIN, "operational", pair]), 16)),
             "challenger_quota": 64, "opponent_quota": 32,
             "challenger_mode": "candidate", "opponent_mode": "baseline"}
            for pair in range(3) for game in range(2)]

"""Fixed saved-position qualification of current-backend timing observation.

Only original graph64 and eager32 positions are replayed. Historical deadline
failures remain failures; a late saved response supplies science, never a score.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.splendorust_buffered_reader_20261025 import qualification as prior
from tools.splendorust_component_timing_20261025.qualification import distribution, overhead_gate
from tools.splendorust_search_scale_20261024 import audit

ROOT = Path(__file__).resolve().parents[2]
POSTMORTEM = ROOT / "outputs/splendorust_graph64_tail_postmortem_20261025_01"
BENCHMARK = ROOT / "outputs/splendorust_graph64_strength_20261025_01"
SCHEMA = "splendorust_boundary_timing_qualification_v1"
PREVIOUS_PLAN = ROOT / "outputs/splendorust_graph_timing_20261025_01/qualification_plan.json"
PREVIOUS_PLAN_SHA256 = "0d48af7cefa1e87788e8bfdea4d51af06f6d3b82a11f06cb5bbcf9f9eeef0772"
OVERHEAD_DOMAIN = "newgraph_timing_overhead_v1"
MODES = ("baseline", "profiled")
QUOTAS = (64, 32)
ROLES = {64: "challenger", 32: "opponent"}
BACKENDS = {64: "graph64", 32: "eager32"}
STAGE_COUNTS = {"overhead": 48, "diagnostic": 416}
EXPECTED_BENCHMARK = "556480a2501f4fb4cf829fe518aa4ba1ccbf10f12ee088fccf627408dfd69560"


def require(value, message):
    if not value:
        raise ValueError("Boundary timing qualification: " + message)


def _same(left, right):
    return prior._same(left, right)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(path):
    return prior._artifact(Path(path))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def choose_overhead(fixtures):
    """Six completed noncases per original role; ranking uses identifiers only."""
    chosen = []
    for quota in QUOTAS:
        eligible = [(index, item) for index, item in enumerate(fixtures)
                    if item["quota"] == quota and item["historical_completed"] and not item["case"]]
        require(len(eligible) >= 6, "six noncase overhead controls per role required")
        eligible.sort(key=lambda pair: _digest([OVERHEAD_DOMAIN, pair[1]["panel_id"]]))
        chosen.extend(index for index, _ in eligible[:6])
    return chosen


def make_schedule(fixtures, *, stage, overhead_indices=None):
    require(stage in STAGE_COUNTS, "unknown stage")
    require(len(fixtures) == 104 and len({f["fixture_id"] for f in fixtures}) == 104, "exact104-root inventory")
    require(sum(f["quota"] == 64 for f in fixtures) == 68 and sum(f["quota"] == 32 for f in fixtures) == 36,
            "original role/quota inventory")
    indices = choose_overhead(fixtures) if stage == "overhead" else list(range(104))
    if overhead_indices is not None:
        require(overhead_indices == choose_overhead(fixtures), "overhead selection changed")
    result = []
    for quota in QUOTAS:
        selected = [i for i in indices if fixtures[i]["quota"] == quota]
        for pass_index in range(2):
            order = list(enumerate(selected))
            if pass_index:
                order.reverse()
            for ordinal, index in order:
                fixture = fixtures[index]
                require(fixture["player_role"] == ROLES[quota] and fixture["backend"] == BACKENDS[quota],
                        "saved player identity changed")
                modes = list(MODES)
                if (ordinal + pass_index) % 2:
                    modes.reverse()
                for mode in modes:
                    result.append({"index": len(result), "stage": stage, "panel": "saved",
                                   "fixture_index": index, "fixture_id": fixture["fixture_id"],
                                   "variant": "original", "pass_index": pass_index, "mode": mode,
                                   "quota": quota, "player_role": ROLES[quota]})
    require(len(result) == STAGE_COUNTS[stage], "fixed request count")
    return result


def _scientific_completed(row, snapshot, *, quota):
    elapsed, science = prior._completed(row, snapshot, quota=quota)
    return elapsed, {**science, "result": deepcopy(row["clock"]["result"])}


def _late_history(row, sources, artifact_index):
    require(row["game_index"] == 66 and row["ply"] == 29 and row["quota"] == 64 and
            row["clock"]["status"] == "timeout" and row["completed"] is False, "only disclosed late historical packet")
    path = (ROOT / row["worker_clock_file"]).parent / "stdout.log.gz"
    item = _artifact(path)
    require(item == artifact_index[item["path"]], "late response archive changed")
    sources[item["path"]] = item
    raw = gzip.decompress(path.read_bytes())
    require(len(raw) < 256 * 1024 and raw.endswith(b"\n"), "bounded complete late response archive")
    packets = [json.loads(line) for line in raw.splitlines()]
    matches = [p for p in packets if p.get("request_id") == row["clock"]["request_id"]]
    require(len(matches) == 1 and matches[0].get("type") == "response", "unique complete late historical response")
    result = matches[0]["result"]
    audit.validate_result(result, row["observer"], "search", expected_quota=64)
    audit.audit_decision(row["observer"], result["action"], result["diagnostics"], "search",
                         before=row["before"], after=None, expected_quota=64)
    return {"kind": "late_unaccepted_science_only", "quota": 64, "source": item,
            "request_id": row["clock"]["request_id"], "historical_clock_accepted": False,
            "science": {"action": result["action"], "trace": result["diagnostics"], "result": result},
            "historical_transition_available": False}


def build_plan():
    runtime_sources = guard.verify_runtime_sources()
    paths = [POSTMORTEM / name for name in ("plan.json", "freeze.json", "completion.json", "execution.json",
             "case_control_panel.json", "source_manifest.json", "failures_raw_audit.json")]
    paths += [BENCHMARK / name for name in ("freeze.json", "completion.json", "execution.json")]
    sources = {item["path"]: item for item in map(_artifact, paths)}
    require(_artifact(BENCHMARK/"freeze.json")["sha256"] == EXPECTED_BENCHMARK, "exact completed benchmark")
    require(_read(BENCHMARK/"completion.json")["status"] == "completed_verified", "verified benchmark required")
    for folder in (POSTMORTEM, BENCHMARK):
        execution = _read(folder/"execution.json")
        require(execution["status"] == "completed" and execution["exit_code"] == 0, "source execution not complete")
    completion = _read(POSTMORTEM/"completion.json")
    require(completion["status"] == "completed_saved_evidence_analysis" and
            completion["benchmark_freeze_sha256"] == EXPECTED_BENCHMARK, "verified saved panel required")
    panel = _read(POSTMORTEM/"case_control_panel.json")
    require(len(panel["records"]) == 104 and len(panel["cases"]) == 20 and len(panel["secondary"]) == 1,
            "fixed postmortem selection")
    case_ids = set(panel["cases"] + panel["secondary"])
    manifest = _read(POSTMORTEM/"source_manifest.json")
    indexed = {item["path"]: item for item in manifest["artifacts"]}
    late_index = {item["path"]: {key: item[key] for key in ("path", "bytes", "sha256")}
                  for item in _read(POSTMORTEM/"failures_raw_audit.json")["selected_verified_artifacts"]}
    fixtures = []
    for record in panel["records"]:
        path = ROOT / record["ledger"]
        source = _artifact(path)
        require(source == indexed[source["path"]], "selected sealed ledger changed")
        row = _read(path)
        quota = 64 if record["role"] == "challenger" else 32
        require(row["quota"] == quota and row["player_role"] == ROLES[quota] and
                row["mode"] == ("candidate" if quota == 64 else "baseline"), "historical backend identity")
        require(record["id"] == f"g{row['game_index']}_p{row['ply']}" and
                record["completed"] is row["completed"], "panel/ledger identity")
        snapshot = row["before"]
        observer = rules.observation_payload(guard.decode_training_snapshot(snapshot))
        require(_same(observer, row["observer"]), "strict corrected snapshot/observer binding")
        clock_path = ROOT / row["worker_clock_file"]
        input_path = clock_path.parent / "inputs" / clock_path.name
        provenance = [source, _artifact(clock_path), _artifact(input_path)]
        for item in provenance:
            require(item == indexed[item["path"]], "selected sealed receipt/input changed")
            sources[item["path"]] = item
        require(_same(_read(clock_path), row["clock"]) and
                _same(_read(input_path), {"player": "search", "observer": observer, "before": snapshot}),
                "historical raw receipt/input binding")
        history = []
        if row["completed"]:
            _, science = _scientific_completed(row, snapshot, quota=quota)
            history.append({"kind": "completed_accepted", "quota": quota, "source": source,
                            "science": science, "historical_transition_available": True})
        elif record["id"] == "g66_p29":
            history.append(_late_history(row, sources, late_index))
        else:
            require(record["id"] == "g12_p56" and row["clock"]["status"] == "worker_error",
                    "unrecognized historical failure")
        fixtures.append({"fixture_id": source["path"], "panel_id": record["id"], "quota": quota,
                         "player_role": ROLES[quota], "backend": BACKENDS[quota], "snapshot": snapshot,
                         "observer_sha256": _digest(observer), "source": source, "provenance": provenance,
                         "case": record["id"] in case_ids, "historical_completed": row["completed"],
                         "historical_status": row["clock"]["status"], "historical_records": history})
    indices = choose_overhead(fixtures)
    plan = {"schema": SCHEMA, "ruleset_id": guard.RULESET_ID, "catalogue_sha256": guard.CATALOGUE_SHA256,
            "turn_contract": guard.TURN_CONTRACT, "runtime_sources": runtime_sources,
            "source_artifacts": [sources[key] for key in sorted(sources)], "panels": {"saved": fixtures},
            "overhead_indices": indices, "overhead_domain": OVERHEAD_DOMAIN,
            "overhead_schedule": make_schedule(fixtures, stage="overhead", overhead_indices=indices),
            "diagnostic_schedule": make_schedule(fixtures, stage="diagnostic", overhead_indices=indices),
            "overhead_gate": {"maximum_profiled_mean_ratio": 1.10, "maximum_added_mean_seconds": .025,
                              "maximum_added_p95_seconds": .050, "both_quotas_required": True},
            "deadline_seconds": 1., "internal_seconds": .9, "diagnostic_margin_seconds": .7,
            "margin_is_admission_gate": False, "no_fresh_games": True,
            "caveats": ["This104-position retrospective case/control panel is not a fresh strength assessment.",
                        "Overhead controls exclude preknown cases; their identifier hash ranking does not use measured new timings.",
                        "Each root retains its original graph64 or eager32 player; no graph32 or cross-quota comparison is permitted.",
                        "A late historical packet is science-only evidence; original failed decisions and losses remain unchanged.",
                        "A newly failed decision stops collection and cannot yield a completed-panel claim."]}
    previous = _artifact(PREVIOUS_PLAN)
    require(previous["sha256"] == PREVIOUS_PLAN_SHA256, "preserved previous frozen panel changed")
    verify_frozen_design(plan, _read(PREVIOUS_PLAN))
    plan["source_artifacts"].append(previous)
    plan["source_artifacts"].sort(key=lambda item: item["path"])
    plan["preserved_frozen_panel"] = previous
    plan["previous_measurements_used_for_selection"] = False
    verify_sources(plan)
    return plan


def verify_frozen_design(plan, previous):
    """The lighter observer receives the exact earlier roots, order and gates."""
    for key in ("panels", "overhead_indices", "overhead_domain", "overhead_schedule",
                "diagnostic_schedule", "overhead_gate", "deadline_seconds", "internal_seconds",
                "diagnostic_margin_seconds", "margin_is_admission_gate", "no_fresh_games"):
        require(key in plan and key in previous and _same(plan[key], previous[key]),
                "previous frozen design changed: " + key)


def verify_sources(plan, boundary=lambda: None):
    for artifact in plan["source_artifacts"]:
        boundary()
        require(_artifact(ROOT/artifact["path"]) == artifact, "historical source changed")


def snapshot_for_request(plan, spec):
    require(spec["panel"] == "saved" and spec["variant"] == "original", "unapproved panel/variant")
    fixture = plan["panels"]["saved"][spec["fixture_index"]]
    require(fixture["fixture_id"] == spec["fixture_id"] and fixture["quota"] == spec["quota"] and
            fixture["player_role"] == spec["player_role"] and fixture["backend"] == BACKENDS[spec["quota"]],
            "saved root/player identity changed")
    return deepcopy(fixture["snapshot"])


def validate_panel(rows, plan, *, stage="overhead", boundary=lambda: None):
    from .analyze import validate_profile_row
    require(stage in STAGE_COUNTS and set(plan["panels"]) == {"saved"}, "only frozen saved panel approved")
    fixtures = plan["panels"]["saved"]
    schedule = make_schedule(fixtures, stage=stage, overhead_indices=plan["overhead_indices"])
    require(_same(schedule, plan[stage+"_schedule"]) and len(rows) == len(schedule), "complete unchanged schedule required")
    verify_sources(plan, boundary)
    groups, history = {}, {"completed_accepted": 0, "late_unaccepted_science_only": 0}
    timings = {str(quota): {mode: [] for mode in MODES} for quota in QUOTAS}
    for row, spec in zip(rows, schedule, strict=True):
        boundary()
        require(all(key in row and _same(row[key], value) for key, value in spec.items()), "request identity/order changed")
        elapsed, science = _scientific_completed(row, snapshot_for_request(plan, spec), quota=spec["quota"])
        validate_profile_row(row)
        key = (spec["fixture_id"], spec["quota"])
        require(_same(groups.setdefault(key, science), science), "mode/repeat full scientific output changed")
        fixture = fixtures[spec["fixture_index"]]
        for previous in fixture["historical_records"]:
            require(previous["quota"] == spec["quota"], "historical quota relabelled")
            require(all(_same(science[k], value) for k, value in previous["science"].items()),
                    "historical full result/action/trace/transition changed")
            history[previous["kind"]] += 1
        timings[str(spec["quota"])][spec["mode"]].append(elapsed)
    if stage == "overhead":
        per_quota = {quota: overhead_gate(values["baseline"], values["profiled"]) for quota, values in timings.items()}
        passed = all(value["qualified"] for value in per_quota.values())
    else:
        per_quota = {quota: {mode: distribution(values[mode]) for mode in MODES} for quota, values in timings.items()}
        passed = True
    return {"status": "passed" if passed else "instrumentation_not_qualified", "qualified": passed,
            "stage": stage, "requests": len(rows), "exact_scientific_groups": len(groups),
            "historical_comparisons": history, "per_quota": per_quota,
            "permits_diagnostic": stage == "overhead" and passed,
            "diagnostic_margin_is_not_admission": True, "permits_fresh_games": False,
            "historical_failures_reclassified": False}

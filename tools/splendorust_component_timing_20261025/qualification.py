"""Fixed development-only qualification of component timing instrumentation."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

from tools import future_training_rules_20261021 as guard
from tools.splendorust_buffered_reader_20261025 import qualification as prior

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "outputs/splendorust_buffered_reader_20261025_01"
SCHEMA = "splendorust_component_timing_qualification_v1"
MODES = ("baseline", "profiled")
QUOTAS = (64, 32)
OVERHEAD_INDICES = (0, 1, 2, 23, 15, 3)
STAGE_COUNTS = {"overhead": 48, "diagnostic": 192}


def require(value, message):
    if not value:
        raise ValueError("Component timing qualification: " + message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _same(left, right):
    return prior._same(left, right)


def make_schedule(fixtures, *, stage):
    require(stage in STAGE_COUNTS, "unknown stage")
    indices = list(OVERHEAD_INDICES) if stage == "overhead" else list(range(24))
    require(len(fixtures) == 24, "exact development inventory")
    result = []
    for quota in QUOTAS:
        for pass_index in range(2):
            order = list(enumerate(indices))
            if pass_index:
                order.reverse()
            for ordinal, fixture_index in order:
                modes = list(MODES)
                if (ordinal + pass_index) % 2:
                    modes.reverse()
                for mode in modes:
                    result.append({"index": len(result), "stage": stage, "panel": "development",
                                   "fixture_index": fixture_index, "fixture_id": fixtures[fixture_index]["fixture_id"],
                                   "variant": "original", "pass_index": pass_index, "mode": mode, "quota": quota})
    return result


def build_plan():
    runtime_sources = guard.verify_runtime_sources()
    path = PRIOR / "qualification_plan.json"
    previous = _read(path)
    require(previous["ruleset_id"] == guard.RULESET_ID and
            previous["catalogue_sha256"] == guard.CATALOGUE_SHA256, "corrected source rules identity")
    require(_read(PRIOR / "completion.json")["status"] == "completed_verified" and
            _read(PRIOR / "execution.json")["status"] == "completed", "verified predecessor required")
    outcome = _read(PRIOR / "qualification_outcome.json")
    require(outcome["buffered_reader_qualified"] is False and outcome["strength_games_launched"] == 0,
            "negative predecessor provenance changed")
    # Deliberately never select, summarize, or evaluate its reserved panel.
    fixtures = deepcopy(previous["panels"]["development"])
    require(len(fixtures) == len({item["observer_sha256"] for item in fixtures}) == 24,
            "exact24 distinct development roots")
    suffixes = ("splendorust_search_scale_opt_20261024_01/games/operational/p000_g0/moves/0042.json",
                "splendorust_search_scale_opt_20261024_01/games/operational/p000_g0/moves/0048.json",
                "splendorust_search_scale_opt_20261024_01/games/operational/p000_g0/moves/0050.json",
                "splendorust_timing_opt_20261024_01/games/ops64/p002_g5/moves/0021.json",
                "splendorust_timing_opt_20261024_01/games/ops64/p000_g0/moves/0008.json",
                "splendorust_search_scale_opt_20261024_01/games/operational/p000_g0/moves/0000.json")
    require(all(fixtures[index]["fixture_id"].endswith(suffix) for index, suffix in zip(OVERHEAD_INDICES, suffixes)),
            "prespecified overhead roots changed")
    sources = {}
    for source_path in (path, PRIOR/"completion.json", PRIOR/"execution.json", PRIOR/"qualification_outcome.json"):
        item = prior._artifact(source_path)
        sources[item["path"]] = item
    for fixture in fixtures:
        for item in fixture["provenance"]:
            sources[item["path"]] = item
    plan = {"schema": SCHEMA, "ruleset_id": guard.RULESET_ID, "catalogue_sha256": guard.CATALOGUE_SHA256,
            "turn_contract": guard.TURN_CONTRACT, "runtime_sources": runtime_sources,
            "source_artifacts": [sources[key] for key in sorted(sources)],
            "panels": {"development": fixtures}, "overhead_indices": list(OVERHEAD_INDICES),
            "overhead_schedule": make_schedule(fixtures, stage="overhead"),
            "diagnostic_schedule": make_schedule(fixtures, stage="diagnostic"),
            "overhead_gate": {"maximum_profiled_mean_ratio": 1.10, "maximum_added_mean_seconds": .025,
                              "maximum_added_p95_seconds": .050, "both_quotas_required": True},
            "deadline_seconds": 1., "diagnostic_margin_seconds": .7,
            "margin_is_admission_gate": False, "raw_reader_optimized_sampler_both_modes": True,
            "no_fresh_games": True, "reserved_panel_used": False,
            "caveats": ["Only the frozen24 development roots are used; the earlier reserved assessment is not retuned.",
                        "The six overhead roots are also included in the24-root diagnostic; neither is an independent strength assessment.",
                        "Overhead thresholds are engineering point estimates and do not authorize subtracting profile durations from deadlines.",
                        "Inclusive parent/child spans overlap. Gaps include unattributed work and waiting, not measured raw transfer CPU."]}
    verify_sources(plan)
    return plan


def snapshot_for_request(plan, spec):
    require(spec["panel"] == "development" and spec["variant"] == "original", "unapproved position panel/variant")
    fixture = plan["panels"]["development"][spec["fixture_index"]]
    require(fixture["fixture_id"] == spec["fixture_id"], "saved fixture changed")
    return deepcopy(fixture["snapshot"])


def distribution(values):
    require(bool(values) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "finite nonnegative timing samples")
    ordered = sorted(values)
    index = .95*(len(values)-1)
    lo, hi = math.floor(index), math.ceil(index)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo),
            "minimum": ordered[0], "maximum": ordered[-1], "above_diagnostic_margin": sum(v > .7 for v in values)}


def overhead_gate(baseline, profiled):
    require(len(baseline) == len(profiled), "paired overhead count")
    old, measured = distribution(baseline), distribution(profiled)
    require(old["mean"] > 0 and old["maximum"] <= 1. and measured["maximum"] <= 1., "hard clock failure")
    added_mean, added_p95 = measured["mean"]-old["mean"], measured["p95"]-old["p95"]
    checks = {"mean_ratio_at_most_1_10": measured["mean"] <= 1.10*old["mean"],
              "added_mean_at_most_025s": measured["mean"] <= old["mean"]+.025,
              "added_p95_at_most_050s": measured["p95"] <= old["p95"]+.050}
    passed = all(checks.values())
    return {"qualified": passed, "status": "passed" if passed else "instrumentation_not_qualified",
            "baseline": old, "profiled": measured, "checks": checks,
            "mean_ratio": measured["mean"]/old["mean"], "added_mean_seconds": added_mean,
            "added_p95_seconds": added_p95, "margin_exceedances_are_diagnostic_only": True}


def verify_sources(plan, boundary=lambda: None):
    for artifact in plan["source_artifacts"]:
        boundary()
        require(prior._artifact(ROOT/artifact["path"]) == artifact, "historical source changed")


def _completed(row, snapshot, *, quota):
    from .analyze import validate_profile_row
    validate_profile_row(row)
    normalized = {**row, "clock": {**row["clock"], "result": dict(row["clock"]["result"])}}
    if row["mode"] == "profiled":
        normalized["clock"]["result"].pop("timing_profile")
    return prior._completed(normalized, snapshot, quota=quota)


def validate_panel(rows, plan, *, stage="overhead", boundary=lambda: None):
    require(stage in STAGE_COUNTS and set(plan["panels"]) == {"development"}, "only development stages approved")
    fixtures = plan["panels"]["development"]
    schedule = make_schedule(fixtures, stage=stage)
    require(_same(schedule, plan[stage+"_schedule"]) and len(rows) == len(schedule) == STAGE_COUNTS[stage],
            "complete unchanged schedule required")
    verify_sources(plan, boundary)
    groups, history = {}, {str(q): 0 for q in QUOTAS}
    timings = {str(q): {mode: [] for mode in MODES} for q in QUOTAS}
    for row, spec in zip(rows, schedule, strict=True):
        boundary()
        require(all(key in row and _same(row[key], value) for key, value in spec.items()), "request identity/order changed")
        elapsed, science = _completed(row, snapshot_for_request(plan, spec), quota=spec["quota"])
        key = (spec["fixture_id"], spec["quota"])
        require(_same(groups.setdefault(key, science), science), "mode/repeat scientific output changed")
        fixture = fixtures[spec["fixture_index"]]
        require(bool(fixture["historical_records"]), "historical evidence missing")
        for previous in fixture["historical_records"]:
            if previous["quota"] == spec["quota"]:
                require(_same(previous["science"], science), "historical action/full trace/transition changed")
                history[str(spec["quota"])] += 1
        timings[str(spec["quota"])][spec["mode"]].append(elapsed)
    if stage == "overhead":
        per_quota = {quota: overhead_gate(values["baseline"], values["profiled"]) for quota, values in timings.items()}
        passed = all(value["qualified"] for value in per_quota.values())
    else:
        per_quota = {quota: {mode: distribution(values[mode]) for mode in MODES} for quota, values in timings.items()}
        require(all(value[mode]["maximum"] <= 1. for value in per_quota.values() for mode in MODES), "hard clock failure")
        passed = True
    return {"status": "passed" if passed else "instrumentation_not_qualified", "qualified": passed,
            "stage": stage, "requests": len(rows), "exact_scientific_groups": len(groups),
            "historical_comparisons": history, "per_quota": per_quota,
            "permits_diagnostic": stage == "overhead" and passed,
            "diagnostic_margin_is_not_admission": True, "permits_fresh_games": False}

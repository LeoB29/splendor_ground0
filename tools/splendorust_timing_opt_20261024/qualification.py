"""Prospective, model-free schedules and gates for one parent-only optimization.

The unchanged worker/checkpoint/search remains the scientific player. These
helpers do not launch a model or a process and never modify historical evidence.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import statistics

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.splendorust_search_scale_20261024 import audit, qualification as original
from tools.splendorust_timing_20261024 import fixtures as saved

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / "outputs/splendorust_timing_20261024_01"
PRIOR = ROOT / "outputs/splendorust_search_scale_20261024_01"
MODES = ("baseline", "candidate")
MARGIN_SECONDS = .7
MINIMUM_MEAN_REDUCTION = .10
SCHEMA = "splendorust_parent_timing_optimization_qualification_v1"


def require(value, message):
    if not value:
        raise ValueError("Timing optimization qualification: " + message)


def _schedule(panels, *, quota, modes):
    result = []
    for panel, fixtures in panels.items():
        for spec in saved.repeated_schedule(fixtures, modes=modes, passes=2):
            result.append({**spec, "index": len(result), "panel": panel, "quota": quota})
    return result


def build_plan():
    """Reuse the already frozen development/holdout split, without reselection."""
    sources = guard.verify_runtime_sources()
    manifest = original.read(DIAGNOSTIC / "fixtures.json")
    require(original.read(DIAGNOSTIC / "completion.json")["status"] == "completed_verified" and
            original.read(DIAGNOSTIC / "execution.json")["status"] == "completed", "diagnostic acceptance")
    require(manifest["ruleset_id"] == guard.RULESET_ID and
            manifest["catalogue_sha256"] == guard.CATALOGUE_SHA256, "saved rules identity")
    panels = {"development": deepcopy(manifest["development"]), "holdout": deepcopy(manifest["validation"])}
    require(len(panels["development"]) == 54 and len(panels["holdout"]) == 24, "frozen panel sizes")
    observers = [{item["observer_sha256"] for item in values} for values in panels.values()]
    require(len(observers[0]) == 54 and len(observers[1]) == 24 and not observers[0] & observers[1],
            "frozen lawful roots overlap or repeat")
    require(all("historical32" not in item for item in panels["holdout"]), "holdout outputs exposed")
    schedule32 = _schedule(panels, quota=32, modes=MODES)
    schedule64 = _schedule({"holdout": panels["holdout"]}, quota=64, modes=("candidate",))
    require(len(schedule32) == 316 and len(schedule64) == 48, "frozen request counts")
    plan64 = original.build_plan(64)
    require(plan64 == original.read(PRIOR / "qualification_plan_64.json"), "prior64 panel changed")
    historical64 = [original.artifact(PRIOR / "qualification_64" / f"{index:03d}.json")
                    for index in range(184)]
    return {"schema": SCHEMA, "ruleset_id": guard.RULESET_ID,
            "catalogue_sha256": guard.CATALOGUE_SHA256, "turn_contract": guard.TURN_CONTRACT,
            "runtime_sources": sources, "source_manifest": original.artifact(DIAGNOSTIC / "fixtures.json"),
            "panels": panels, "stage32_schedule": schedule32,
            "stage64_panel": plan64, "stage64_holdout_schedule": schedule64,
            "historical64": historical64,
            "gate32": {"candidate_margin_seconds": MARGIN_SECONDS,
                       "minimum_mean_reduction_per_panel": MINIMUM_MEAN_REDUCTION,
                       "candidate_p95_no_worse_per_panel": True,
                       "baseline_margin_exceedances_preserved_not_disqualifying": True,
                       "any_one_second_failure_stops": True, "both_panels_must_pass": True},
            "entrypoint_contract": {
                "worker_module": "tools.splendorust_search_scale_20261024.worker",
                "operation": {"operation": "entrypoint_checks"},
                "stage32_parent_modes": list(MODES), "stage64_parent_modes": ["candidate"],
                "complete_decisions_per_worker": 52, "native_operations_per_worker": 82,
                "worker_rejections_per_worker": 26,
                "startup_runtime_guard_required": True,
                "actual_corrected_entrypoint_required_before_timed_requests": True,
                "new_sampler_counterexamples_required_before_model_workers": True},
            "caveats": manifest["caveats"] + [
                "Mean and p95 gates are prospective engineering point estimates, not confidence bounds.",
                "A failed holdout gate stops this candidate; it must not trigger tuning against those outcomes.",
                "A passing saved-state gate still requires the separately frozen fresh operational games."]}


def snapshot_for_request(plan, spec):
    return saved.snapshot_for_request(plan["panels"][spec["panel"]], spec)


def _same(left, right):
    return saved.digest(left) == saved.digest(right)


def _distribution(values):
    require(bool(values) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "finite nonnegative timing samples")
    ordered = sorted(values)
    position = .95 * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo),
            "maximum": ordered[-1], "minimum": ordered[0],
            "above_margin": sum(value > MARGIN_SECONDS for value in values)}


def timing_gate(baseline, candidate):
    """No threshold adaptation and no discard of slow baseline measurements."""
    require(len(baseline) == len(candidate), "paired timing count")
    old, new = _distribution(baseline), _distribution(candidate)
    require(old["mean"] > 0 and old["maximum"] <= 1 and new["maximum"] <= 1,
            "one-second failure or invalid baseline timing")
    reduction = 1 - new["mean"] / old["mean"]
    checks = {"all_candidate_within_margin": new["maximum"] <= MARGIN_SECONDS,
              "mean_reduction_at_least_ten_percent": new["mean"] <= (1-MINIMUM_MEAN_REDUCTION)*old["mean"],
              "candidate_p95_no_worse": new["p95"] <= old["p95"]}
    passed = all(checks.values())
    return {"status": "passed" if passed else "timing_not_qualified", "qualified": passed,
            "checks": checks, "baseline": old, "candidate": new, "mean_reduction_fraction": reduction,
            "baseline_margin_exceedances_preserved": old["above_margin"]}


def _completed(row, snapshot, *, quota, arm="search"):
    require(row["completed"] is True and row["clock"]["status"] == "ok", "failed request; preserve and stop")
    require(type(row.get("after")) is dict, "completed next snapshot required for independent transition replay")
    elapsed = audit.check_clock(row["clock"], limit=1.)
    require(elapsed is not None and row["clock"]["deadline_seconds"] == 1., "full one-second qualification allowance")
    observer = rules.observation_payload(guard.decode_training_snapshot(snapshot))
    require(_same(row["before"], snapshot) and _same(row["observer"], observer), "saved input binding")
    result = row["clock"]["result"]
    require("timing_profile" not in result, "scientific worker must be unchanged and unprofiled")
    require(_same(result["action"], row["action"]) and _same(result["diagnostics"], row["trace"]),
            "scientific result differs from trusted receipt")
    audit.validate_result(result, observer, arm, expected_quota=quota)
    replay = audit.audit_decision(observer, row["action"], row["trace"], arm, before=snapshot,
                                  after=row["after"], expected_quota=quota)
    require(replay["transition_checked"] is True, "independent completed transition was not checked")
    return elapsed, {"action": row["action"], "trace": row["trace"]}


def _check_schedule(rows, schedule):
    require(len(rows) == len(schedule), "incomplete schedule")
    require(len({row["index"] for row in rows}) == len(schedule), "duplicate request index")
    for row, spec in zip(rows, schedule, strict=True):
        require(all(key in row and _same(row[key], value) for key, value in spec.items()), "request schedule changed")


def validate_32(rows, plan):
    """Exact historical/repeated decisions precede independent per-panel gates."""
    _check_schedule(rows, plan["stage32_schedule"])
    timings = {panel: {mode: [] for mode in MODES} for panel in ("development", "holdout")}
    groups, historical = {}, 0
    for row, spec in zip(rows, plan["stage32_schedule"], strict=True):
        require(spec["quota"] == 32 and spec["mode"] in MODES, "stage32 mode/quota")
        fixture = plan["panels"][spec["panel"]][spec["fixture_index"]]
        elapsed, science = _completed(row, snapshot_for_request(plan, spec), quota=32)
        key = (spec["panel"], spec["fixture_id"])
        require(_same(groups.setdefault(key, science), science), "parent mode/hidden variant/repeat changed scientific output")
        if spec["panel"] == "development":
            prior = fixture["historical32"]
            require(_same(science, {"action": prior["action"], "trace": prior["trace"]}), "historical32 output changed")
            historical += 1
        timings[spec["panel"]][spec["mode"]].append(elapsed)
    panels = {name: timing_gate(values["baseline"], values["candidate"]) for name, values in timings.items()}
    passed = all(result["qualified"] for result in panels.values())
    return {"status": "passed" if passed else "timing_not_qualified", "qualified": passed,
            "panels": panels, "requests": len(rows), "exact_scientific_groups": len(groups),
            "historical32_comparisons": historical, "permits64_qualification": passed,
            "failed_candidate_may_not_be_retuned_using_holdout": True}


def validate_64_panel(rows, plan):
    """Original184 checks plus exact full action/trace against all saved64 rows."""
    legacy_plan = plan["stage64_panel"]
    require(len(rows) == len(plan["historical64"]) == 184, "64 historical panel size")
    checked = []
    for row, spec, artifact in zip(rows, legacy_plan["timed_requests"], plan["historical64"], strict=True):
        require(all(_same(row[key], value) for key, value in spec.items()), "64 panel schedule changed")
        snapshot = legacy_plan["fixtures"][spec["fixture_index"]][spec["variant"]]
        elapsed, science = _completed(row, snapshot, quota=64 if spec["arm"] == "search" else 16, arm=spec["arm"])
        path = ROOT / artifact["path"]
        require(original.artifact(path) == artifact, "historical64 evidence changed")
        prior = original.read(path)
        require(_same(prior["before"], snapshot) and _same(prior["observer"], row["observer"]), "historical64 input binding")
        require(_same(science, {"action": prior["action"], "trace": prior["trace"]}), "historical64 action/full callback trace changed")
        checked.append({**row, "elapsed_seconds": elapsed})
    result = original.validate_timed_panel(checked, legacy_plan)
    return {**result, "qualified": result["timing_qualified"], "exact_historical64_comparisons": len(rows)}


def validate_64_holdout(rows, plan):
    _check_schedule(rows, plan["stage64_holdout_schedule"])
    groups, elapsed = {}, []
    for row, spec in zip(rows, plan["stage64_holdout_schedule"], strict=True):
        require(spec["quota"] == 64 and spec["mode"] == "candidate" and spec["panel"] == "holdout", "64 holdout identity")
        timing, science = _completed(row, snapshot_for_request(plan, spec), quota=64)
        require(_same(groups.setdefault(spec["fixture_id"], science), science), "64 holdout repeated scientific output changed")
        elapsed.append(timing)
    timings = _distribution(elapsed)
    passed = timings["maximum"] <= MARGIN_SECONDS
    return {"status": "passed" if passed else "timing_not_qualified", "qualified": passed,
            "requests": len(rows), "exact_repeat_groups": len(groups), "timings": timings,
            "strength_games_require_fresh_operational_gate": True}

"""Run preregistered experiment 134: fresh expanded-resolver comparison."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    _sha256_file,
)
from splendor_ai.eval.run_expanded_root_resolver_champion_challenge import (
    ROOT_CONTRACT,
    _atomic_write,
    _bounded_loops,
    _candidate_contract_matches,
    _deterministic_projection,
    _game_integrity,
    _play_pair,
    _resolver_config,
    _seat_breakdown,
    _summarize_candidate_telemetry,
    _summarize_champion_telemetry,
)
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
    PHASES,
    VALID_TERMINATIONS,
    _frozen_contract_matches_049,
    gameplay_belief_root_index,
    gameplay_continuation_seed,
)
from splendor_ai.eval.run_shallow_operational_gate import summarize_pair_rows


PROTOCOL = Path(
    "docs/engineering/expanded_root_resolver_fresh_champion_comparison_134.md"
)
PROTOCOL_SHA256 = "32bf4ae8a98690115470bd73a0f736a3966f12abc3f115b218dcf31f8368b149"
RESULT_084_SHA256 = "1ed98024072b92baf3c30bc896b9dae2484f7d80b727959307dfd83bbfe2cd0f"
RESULT_086_SHA256 = "8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5"
RESULT_133_SHA256 = "7f74736b4557a1b01b3e6febbf6a4f4cec2fa7643a78655c0f4adc529dde872c"
REVIEW_SHA256 = "2ba4e5c60788b080d517ae8f783186efec1ee9a55f348fb8cb047a28efaa80c2"
REUSED_IMPLEMENTATION_SHA256 = (
    "bcc63bcb9b753e0387aa57115facaea8c9eab13f28ade26ae9f7b80b838f2a28"
)


@dataclass(frozen=True, slots=True)
class ExpandedRootFreshComparisonConfig:
    output_dir: Path = Path("outputs/expanded_root_resolver_fresh_champion_comparison_134")
    checkpoint: Path = CHECKPOINT
    result_084: Path = Path("outputs/resolver_research_champion_confirmation_084/result.json")
    result_086: Path = Path("outputs/expanded_root_regret_stability_replication_086/result.json")
    result_133: Path = Path(
        "outputs/progressive_public_belief_planner_feasibility_133/result.json"
    )
    mechanism_review: Path = Path(
        "docs/strategy/information_set_search_mechanism_review_after_133.md"
    )
    reused_implementation: Path = Path(
        "src/splendor_ai/eval/run_expanded_root_resolver_champion_challenge.py"
    )
    smoke_seed: int = 1_340_000
    strength_seed_start: int = 1_340_100
    pairs: int = 200
    determinizations: int = 3
    champion_roots: int = 4
    opponent_actions: int = 4
    discovery_continuations: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_index_start: int = 1_340_000_000
    continuation_seed_start: int = 1_340_000_000_000_000_000
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 1_340_100
    expected_score_share: float = 0.565
    minimum_score_share: float = 0.55
    minimum_seat_score_share: float = 0.50
    minimum_expansion_switch_rate: float = 0.15
    maximum_loop_adjudicated_fraction: float = 0.001
    maximum_strength_wall_seconds: float = 259_200.0
    device: str = "cpu"


def preflight(config: ExpandedRootFreshComparisonConfig | None = None) -> dict[str, Any]:
    cfg = config or ExpandedRootFreshComparisonConfig()
    _validate_config(cfg)
    resolver_cfg = _resolver_config(cfg)
    seeds = {
        gameplay_continuation_seed(
            resolver_cfg,
            engine_seed,
            seat,
            decision,
            determinization,
            continuation,
        )
        for engine_seed in (cfg.smoke_seed, cfg.strength_seed_start + cfg.pairs - 1)
        for seat in (0, 1)
        for decision in (0, cfg.max_turns - 1)
        for determinization in range(cfg.determinizations)
        for continuation in range(cfg.discovery_continuations)
    }
    belief_indices = {
        gameplay_belief_root_index(resolver_cfg, engine_seed, seat, decision)
        for engine_seed in (cfg.smoke_seed, cfg.strength_seed_start + cfg.pairs - 1)
        for seat in (0, 1)
        for decision in (0, cfg.max_turns - 1)
    }
    checks = {
        "frozen_provenance_and_configuration": _provenance_matches(cfg),
        "exact_ten_workers": cfg.workers == 10,
        "exact_200_pairs": cfg.pairs == 200,
        "fresh_seed_block": cfg.strength_seed_start == cfg.smoke_seed + 100,
        "continuation_namespace_unique_and_int64": len(seeds) == 96
        and min(seeds) >= cfg.continuation_seed_start
        and max(seeds) < 2**63,
        "belief_namespace_unique": len(belief_indices) == 8,
        "exact_candidate_contract": _candidate_contract_matches(cfg),
        "exact_champion_contract": _frozen_contract_matches_049(resolver_cfg),
        "no_087_progress_input": all("087" not in str(path) for path in (
            cfg.output_dir,
            cfg.result_084,
            cfg.result_086,
            cfg.result_133,
            cfg.mechanism_review,
        )),
        "no_training_or_corpus_path": True,
    }
    return {
        "experiment": "134_expanded_root_resolver_fresh_champion_comparison",
        "identity": _identity(cfg),
        "checks": checks,
        "passed": all(checks.values()),
        "launch_required": True,
        "training_authorized": False,
        "gameplay_started": False,
    }


def run_gate(config: ExpandedRootFreshComparisonConfig | None = None) -> dict[str, Any]:
    cfg = config or ExpandedRootFreshComparisonConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 134 result does not match the frozen protocol.")
        return result

    started = time.perf_counter()
    smoke = _run_smoke(cfg, identity)
    smoke_passed = bool(smoke["passed"])
    strength = _run_strength(cfg, identity) if smoke_passed else None
    strength_gates = _strength_gates(cfg, strength) if strength is not None else None
    operational_passed = bool(
        smoke_passed
        and strength_gates
        and all(strength_gates[name] for name in _operational_gate_names())
    )
    passed = bool(operational_passed and strength_gates and all(strength_gates.values()))
    valid = operational_passed
    decision = (
        "promote_expanded_root_public_belief_resolver_134_as_research_champion"
        if passed
        else "retain_050_051_research_champion_close_exact_expansion_after_fresh_134"
        if valid
        else "expanded_root_gameplay_134_invalid_no_strength_decision"
    )
    result = {
        "experiment": "134_expanded_root_resolver_fresh_champion_comparison",
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "protocol_sha256": PROTOCOL_SHA256,
        "identity": identity,
        "smoke": smoke,
        "smoke_passed": smoke_passed,
        "strength": strength,
        "strength_gates": strength_gates,
        "valid": valid,
        "passed": passed,
        "promotion_authorized": passed,
        "research_champion": (
            "expanded_root_public_belief_resolver_134"
            if passed
            else "public_belief_resolver_050_051"
        ),
        "fast_champion": "035",
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "automatic_continuation": False,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write(result_path, result)
    print(f"134 result: {'PASS' if passed else 'FAIL'} ({decision})", flush=True)
    return result


def _run_smoke(
    cfg: ExpandedRootFreshComparisonConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    path = cfg.output_dir / "smoke.json"
    smoke_identity = {**identity, "block": "smoke", "seed": cfg.smoke_seed}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != smoke_identity:
            raise ValueError("Existing 134 smoke does not match the frozen protocol.")
        return payload
    print("[resolver134:smoke] running original and from-zero repeat", flush=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_play_pair, cfg, cfg.smoke_seed) for _ in range(2)]
        original, repeated = [future.result() for future in futures]
    rows = [original, repeated]
    integrity = _game_integrity(rows)
    candidate = _summarize_candidate_telemetry(rows)
    champion = _summarize_champion_telemetry(rows)
    repeat_identical = _deterministic_projection(original) == _deterministic_projection(repeated)
    gates = {
        "frozen_provenance_and_configuration": _provenance_matches(cfg),
        "four_clean_games": integrity["games"] == 4 and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "both_resolvers_exercised": candidate["completed_searches"] > 0
        and champion["completed_searches"] > 0,
        "candidate_exact_branch_accounting": bool(candidate["exact_branch_accounting"]),
        "champion_exact_branch_accounting": bool(champion["exact_branch_accounting"]),
        "all_continuations_resolved": bool(candidate["all_continuations_resolved"])
        and bool(champion["all_continuations_resolved"]),
        "zero_hard_caps": candidate["hard_caps"] == 0 and champion["hard_caps"] == 0,
        "bounded_loop_adjudication": _bounded_loops(cfg, candidate, champion),
        "from_zero_repeat_identical": repeat_identical,
    }
    payload = {
        "format": "expanded_root_resolver_fresh_champion_smoke_134_v1",
        "identity": smoke_identity,
        "integrity": integrity,
        "candidate_telemetry": candidate,
        "champion_telemetry": champion,
        "repeat_identical": repeat_identical,
        "gates": gates,
        "passed": all(gates.values()),
        "original": original,
        "repeated": repeated,
    }
    _atomic_write(path, payload)
    return payload


def _run_strength(
    cfg: ExpandedRootFreshComparisonConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    path = cfg.output_dir / "strength_vs_research_champion.json"
    progress_path = cfg.output_dir / "strength_vs_research_champion.progress.json"
    block_identity = {
        **identity,
        "block": "strength_vs_research_champion",
        "seed_start": cfg.strength_seed_start,
        "pairs": cfg.pairs,
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != block_identity:
            raise ValueError("Existing 134 strength result does not match the frozen protocol.")
        return payload

    completed: dict[int, dict[str, Any]] = {}
    prior_active_seconds = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != block_identity:
            raise ValueError("Existing 134 progress does not match the frozen protocol.")
        completed = {int(row["seed"]): row for row in progress.get("pairs", [])}
        prior_active_seconds = float(progress.get("active_wall_seconds", 0.0))

    seeds = list(range(cfg.strength_seed_start, cfg.strength_seed_start + cfg.pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(
        f"[resolver134:strength] pairs={cfg.pairs} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_play_pair, cfg, seed): seed for seed in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                completed[seed] = future.result()
                active_seconds = prior_active_seconds + time.perf_counter() - started
                ordered = [completed[value] for value in sorted(completed)]
                _atomic_write(
                    progress_path,
                    {
                        "identity": block_identity,
                        "completed": False,
                        "active_wall_seconds": active_seconds,
                        "pairs": ordered,
                    },
                )
                rate = index / (time.perf_counter() - started)
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver134:strength] complete={len(completed)}/{cfg.pairs} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )

    active_seconds = prior_active_seconds + time.perf_counter() - started
    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, cfg.bootstrap_samples, cfg.bootstrap_seed)
    summary["seat_breakdown"] = _seat_breakdown(rows)
    payload = {
        "format": "expanded_root_resolver_fresh_champion_strength_134_v1",
        "identity": block_identity,
        "active_wall_seconds": active_seconds,
        "summary": summary,
        "integrity": _game_integrity(rows),
        "candidate_telemetry": _summarize_candidate_telemetry(rows),
        "champion_telemetry": _summarize_champion_telemetry(rows),
        "pairs": rows,
    }
    _atomic_write(path, payload)
    _atomic_write(
        progress_path,
        {
            "identity": block_identity,
            "completed": True,
            "active_wall_seconds": active_seconds,
            "pairs": rows,
        },
    )
    return payload


def _strength_gates(
    cfg: ExpandedRootFreshComparisonConfig,
    strength: dict[str, Any],
) -> dict[str, bool]:
    summary = strength["summary"]
    integrity = strength["integrity"]
    candidate = strength["candidate_telemetry"]
    champion = strength["champion_telemetry"]
    candidate_phases = candidate["by_phase"]
    return {
        "frozen_provenance_and_configuration": _provenance_matches(cfg),
        "exactly_400_clean_games": integrity["games"] == cfg.pairs * 2
        and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "candidate_exact_variable_branch_accounting": bool(
            candidate["exact_branch_accounting"]
        ),
        "champion_exact_192_branch_accounting": bool(champion["exact_branch_accounting"]),
        "all_continuations_resolved": bool(candidate["all_continuations_resolved"])
        and bool(champion["all_continuations_resolved"]),
        "zero_hard_caps": candidate["hard_caps"] == 0 and champion["hard_caps"] == 0,
        "bounded_loop_adjudication": _bounded_loops(cfg, candidate, champion),
        "within_72_active_wall_hours": float(strength["active_wall_seconds"])
        <= cfg.maximum_strength_wall_seconds,
        "expanded_roots_and_switches_every_phase": all(
            int(candidate_phases[phase].get("optional_roots_retained", 0)) > 0
            and int(candidate_phases[phase].get("expansion_switches", 0)) > 0
            for phase in PHASES
        ),
        "expansion_switch_rate_at_least_15pct": float(candidate["expansion_switch_rate"])
        >= cfg.minimum_expansion_switch_rate,
        "score_share_at_least_55pct": float(summary["score_share"]["mean"])
        >= cfg.minimum_score_share,
        "score_share_ci95_lower_above_50pct": float(summary["score_share"]["ci95_low"])
        > 0.50,
        "positive_mean_paired_margin": float(summary["final_score_margin"]["mean"]) > 0.0,
        "no_seat_collapse": all(
            float(item["score_share"]) >= cfg.minimum_seat_score_share
            for item in summary["seat_breakdown"].values()
        ),
    }


def _operational_gate_names() -> tuple[str, ...]:
    return (
        "frozen_provenance_and_configuration",
        "exactly_400_clean_games",
        "zero_timeouts_and_stalls",
        "candidate_exact_variable_branch_accounting",
        "champion_exact_192_branch_accounting",
        "all_continuations_resolved",
        "zero_hard_caps",
        "bounded_loop_adjudication",
        "within_72_active_wall_hours",
    )


def _provenance_matches(cfg: ExpandedRootFreshComparisonConfig) -> bool:
    if cfg != ExpandedRootFreshComparisonConfig():
        return False
    required = (
        PROTOCOL,
        cfg.checkpoint,
        cfg.result_084,
        cfg.result_086,
        cfg.result_133,
        cfg.mechanism_review,
        cfg.reused_implementation,
    )
    if any(not path.exists() for path in required):
        return False
    result_084 = json.loads(cfg.result_084.read_text(encoding="utf-8"))
    result_133 = json.loads(cfg.result_133.read_text(encoding="utf-8"))
    resolver_cfg = _resolver_config(cfg)
    return (
        _sha256_file(PROTOCOL) == PROTOCOL_SHA256
        and _sha256_file(cfg.checkpoint) == CHECKPOINT_SHA256
        and _sha256_file(cfg.result_084) == RESULT_084_SHA256
        and _sha256_file(cfg.result_086) == RESULT_086_SHA256
        and _sha256_file(cfg.result_133) == RESULT_133_SHA256
        and _sha256_file(cfg.mechanism_review) == REVIEW_SHA256
        and _sha256_file(cfg.reused_implementation) == REUSED_IMPLEMENTATION_SHA256
        and result_084.get("passed") is True
        and result_084.get("research_champion") == "public_belief_resolver_050_051"
        and result_133.get("decision")
        == "progressive_public_belief_planner_133_failed_review_search_mechanism"
        and _frozen_contract_matches_049(resolver_cfg)
        and _candidate_contract_matches(cfg)
    )


def _identity(cfg: ExpandedRootFreshComparisonConfig) -> dict[str, Any]:
    resolver_cfg = _resolver_config(cfg)
    return {
        "configuration": _config_payload(cfg),
        "resolver_configuration": _config_payload(resolver_cfg),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "checkpoint_sha256": _sha256_file(cfg.checkpoint),
        "result_084_sha256": _sha256_file(cfg.result_084),
        "result_086_sha256": _sha256_file(cfg.result_086),
        "result_133_sha256": _sha256_file(cfg.result_133),
        "mechanism_review_sha256": _sha256_file(cfg.mechanism_review),
        "reused_implementation_sha256": _sha256_file(cfg.reused_implementation),
        "candidate_root_contract": ROOT_CONTRACT,
        "candidate_continuation_contract": "3x4x4_discovery_only_variable_roots_v1",
        "champion_continuation_contract": "3x4x4_discovery_only_top4_v1",
        "champion_expected_branches_per_search": EXPECTED_BRANCHES_PER_SEARCH,
        "matched_candidate_champion_namespaces": True,
        "valid_terminations": list(VALID_TERMINATIONS),
        "reads_087_strength_progress": False,
    }


def _validate_config(cfg: ExpandedRootFreshComparisonConfig) -> None:
    if cfg != ExpandedRootFreshComparisonConfig():
        raise ValueError("Experiment 134 permits only its preregistered default configuration.")
    for path in (
        PROTOCOL,
        cfg.checkpoint,
        cfg.result_084,
        cfg.result_086,
        cfg.result_133,
        cfg.mechanism_review,
        cfg.reused_implementation,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if cfg.bootstrap_seed != cfg.strength_seed_start:
        raise ValueError("134 bootstrap seed must equal the frozen strength start.")
    if cfg.strength_seed_start != cfg.smoke_seed + 100 or cfg.pairs != 200:
        raise ValueError("134 fresh paired seed block differs from the preregistration.")
    if cfg.workers != 10:
        raise ValueError("134 freezes exactly ten CPU workers.")
    if cfg.continuation_seed_start + 300 * 1_000_000_000 >= 2**63:
        raise ValueError("134 continuation namespace exceeds signed 64-bit range.")
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 134 frozen provenance or resolver contract mismatch.")


def _config_payload(cfg: Any) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }

"""Run preregistered experiment 084: resolver research-champion confirmation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT_SHA256,
    _sha256_file,
)
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
    PHASES,
    ResolverGameplayConfig,
    _deterministic_projection,
    _frozen_contract_matches_049,
    _integrity,
    _play_pair,
    _seat_breakdown,
    _summarize_telemetry,
)
from splendor_ai.eval.run_shallow_operational_gate import summarize_pair_rows


PROTOCOL = Path("docs/experiments/resolver_research_champion_confirmation_084.md")
PROTOCOL_SHA256 = "7f5966f59a2ad5b6dd3e831dac21eaf3a8572c68a93600945868850764a90ce7"
RESULT_050_SHA256 = "68060ffc5270b320a185f618c016f76892ae4dbc972623cd70be34cc63dff195"
RESULT_051_SHA256 = "1a731e5edf3c63cffb2b46d8adaebba9fc50b81e490367bb0c8448ef6a885ed4"


@dataclass(frozen=True, slots=True)
class ResolverResearchChampionConfig:
    output_dir: Path = Path("outputs/resolver_research_champion_confirmation_084")
    result_050: Path = Path("outputs/paired_public_belief_resolver_gameplay_050/result.json")
    result_051: Path = Path("outputs/broad_public_belief_resolver_confirmation_051/result.json")
    smoke_seed: int = 840_000
    strength_seed_start: int = 840_100
    pairs: int = 200
    belief_root_index_start: int = 840_000_000
    continuation_seed_start: int = 8_400_000_000_000_000_000
    workers: int = 10
    bootstrap_samples: int = 20_000
    minimum_score_share: float = 0.55
    minimum_seat_score_share: float = 0.50
    minimum_switch_rate: float = 0.15
    maximum_loop_adjudicated_fraction: float = 0.001


def run_gate(config: ResolverResearchChampionConfig | None = None) -> dict[str, Any]:
    cfg = config or ResolverResearchChampionConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 084 result does not match the frozen protocol.")
        return result

    started = time.perf_counter()
    smoke = _run_smoke(cfg, identity)
    smoke_passed = bool(smoke["passed"])
    strength = _run_strength(cfg, identity) if smoke_passed else None
    strength_gates = _strength_gates(cfg, strength) if strength is not None else None
    passed = bool(smoke_passed and strength_gates and all(strength_gates.values()))
    result = {
        "experiment": "084_resolver_research_champion_confirmation",
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "protocol_sha256": PROTOCOL_SHA256,
        "identity": identity,
        "smoke": smoke,
        "smoke_passed": smoke_passed,
        "strength": strength,
        "strength_gates": strength_gates,
        "passed": passed,
        "fast_champion": "035",
        "research_champion": "public_belief_resolver_050_051" if passed else "unchanged",
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "decision": (
            "promote_frozen_public_belief_resolver_as_research_champion"
            if passed
            else "do_not_promote_frozen_public_belief_resolver"
            if smoke_passed
            else "operational_smoke_failed_strength_block_forbidden"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write(result_path, result)
    print(f"084 result: {'PASS' if passed else 'FAIL'}", flush=True)
    return result


def _resolver_config(cfg: ResolverResearchChampionConfig) -> ResolverGameplayConfig:
    return ResolverGameplayConfig(
        output_dir=cfg.output_dir,
        smoke_seed=cfg.smoke_seed,
        direct_seed_start=cfg.strength_seed_start,
        direct_pairs=cfg.pairs,
        belief_root_index_start=cfg.belief_root_index_start,
        continuation_seed_start=cfg.continuation_seed_start,
        workers=cfg.workers,
        bootstrap_samples=cfg.bootstrap_samples,
        minimum_score_share=cfg.minimum_score_share,
        minimum_seat_score_share=cfg.minimum_seat_score_share,
        maximum_loop_adjudicated_fraction=cfg.maximum_loop_adjudicated_fraction,
    )


def _run_smoke(
    cfg: ResolverResearchChampionConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    path = cfg.output_dir / "smoke.json"
    smoke_identity = {**identity, "block": "smoke", "seed": cfg.smoke_seed}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != smoke_identity:
            raise ValueError("Existing 084 smoke does not match the frozen protocol.")
        return payload

    print("[resolver084:smoke] running original and from-zero repeat", flush=True)
    resolver_cfg = _resolver_config(cfg)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_play_pair, resolver_cfg, cfg.smoke_seed) for _ in range(2)]
        original, repeated = [future.result() for future in futures]
    rows = [original, repeated]
    telemetry = _summarize_telemetry(rows)
    integrity = _integrity(rows)
    repeat_identical = _deterministic_projection(original) == _deterministic_projection(repeated)
    gates = {
        "frozen_provenance_and_configuration": _provenance_matches(cfg),
        "four_clean_games": integrity["games"] == 4 and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "resolver_exercised": telemetry["completed_searches"] > 0,
        "exact_branch_accounting": bool(telemetry["exact_branch_accounting"]),
        "all_continuations_resolved": bool(telemetry["all_continuations_resolved"]),
        "zero_hard_caps": int(telemetry["hard_caps"]) == 0,
        "bounded_loop_adjudication": float(telemetry["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction,
        "from_zero_repeat_identical": repeat_identical,
    }
    payload = {
        "format": "resolver_research_champion_smoke_084_v1",
        "identity": smoke_identity,
        "integrity": integrity,
        "telemetry": telemetry,
        "repeat_identical": repeat_identical,
        "gates": gates,
        "passed": all(gates.values()),
        "original": original,
        "repeated": repeated,
    }
    _atomic_write(path, payload)
    return payload


def _run_strength(
    cfg: ResolverResearchChampionConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    path = cfg.output_dir / "strength_vs_035.json"
    progress_path = cfg.output_dir / "strength_vs_035.progress.json"
    block_identity = {
        **identity,
        "block": "strength_vs_035",
        "seed_start": cfg.strength_seed_start,
        "pairs": cfg.pairs,
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != block_identity:
            raise ValueError("Existing 084 strength result does not match the frozen protocol.")
        return payload

    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != block_identity:
            raise ValueError("Existing 084 progress does not match the frozen protocol.")
        completed = {int(row["seed"]): row for row in progress.get("pairs", [])}

    seeds = list(range(cfg.strength_seed_start, cfg.strength_seed_start + cfg.pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(
        f"[resolver084:strength] pairs={cfg.pairs} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    resolver_cfg = _resolver_config(cfg)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_play_pair, resolver_cfg, seed): seed for seed in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                completed[seed] = future.result()
                ordered = [completed[value] for value in sorted(completed)]
                _atomic_write(
                    progress_path,
                    {"identity": block_identity, "completed": False, "pairs": ordered},
                )
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver084:strength] complete={len(completed)}/{cfg.pairs} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )

    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, cfg.bootstrap_samples, cfg.strength_seed_start)
    summary["seat_breakdown"] = _seat_breakdown(rows)
    payload = {
        "format": "resolver_research_champion_strength_084_v1",
        "identity": block_identity,
        "summary": summary,
        "integrity": _integrity(rows),
        "telemetry": _summarize_telemetry(rows),
        "pairs": rows,
    }
    _atomic_write(path, payload)
    _atomic_write(
        progress_path,
        {"identity": block_identity, "completed": True, "pairs": rows},
    )
    return payload


def _strength_gates(
    cfg: ResolverResearchChampionConfig,
    strength: dict[str, Any],
) -> dict[str, bool]:
    summary = strength["summary"]
    integrity = strength["integrity"]
    telemetry = strength["telemetry"]
    decisions = int(telemetry["decisions"])
    switches = int(telemetry["switches"])
    switch_rate = switches / decisions if decisions else 0.0
    phase_active = all(
        int(telemetry["by_phase"][phase].get("completed_searches", 0)) > 0
        and int(telemetry["by_phase"][phase].get("switches", 0)) > 0
        for phase in PHASES
    )
    return {
        "frozen_provenance_and_configuration": _provenance_matches(cfg),
        "exactly_400_clean_games": integrity["games"] == cfg.pairs * 2
        and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "exact_branch_accounting": bool(telemetry["exact_branch_accounting"]),
        "all_continuations_resolved": bool(telemetry["all_continuations_resolved"]),
        "zero_hard_caps": int(telemetry["hard_caps"]) == 0,
        "bounded_loop_adjudication": float(telemetry["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction,
        "resolver_active_and_switches_all_phases": phase_active,
        "switch_rate_at_least_15pct": switch_rate >= cfg.minimum_switch_rate,
        "score_share_at_least_55pct": float(summary["score_share"]["mean"])
        >= cfg.minimum_score_share,
        "score_share_ci95_lower_above_50pct": float(summary["score_share"]["ci95_low"])
        > 0.50,
        "positive_mean_paired_margin": float(summary["final_score_margin"]["mean"])
        > 0.0,
        "no_seat_collapse": all(
            float(item["score_share"]) >= cfg.minimum_seat_score_share
            for item in summary["seat_breakdown"].values()
        ),
    }


def _provenance_matches(cfg: ResolverResearchChampionConfig) -> bool:
    resolver_cfg = _resolver_config(cfg)
    return (
        _sha256_file(PROTOCOL) == PROTOCOL_SHA256
        and _sha256_file(cfg.result_050) == RESULT_050_SHA256
        and _sha256_file(cfg.result_051) == RESULT_051_SHA256
        and _sha256_file(resolver_cfg.checkpoint) == CHECKPOINT_SHA256
        and _frozen_contract_matches_049(resolver_cfg)
        and json.loads(cfg.result_050.read_text(encoding="utf-8")).get("passed") is True
        and json.loads(cfg.result_051.read_text(encoding="utf-8")).get("passed") is True
    )


def _identity(cfg: ResolverResearchChampionConfig) -> dict[str, Any]:
    resolver_cfg = _resolver_config(cfg)
    return {
        "configuration": _config_payload(cfg),
        "resolver_configuration": _config_payload(resolver_cfg),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "result_050_sha256": _sha256_file(cfg.result_050),
        "result_051_sha256": _sha256_file(cfg.result_051),
        "checkpoint_sha256": _sha256_file(resolver_cfg.checkpoint),
        "expected_branches_per_search": EXPECTED_BRANCHES_PER_SEARCH,
        "valid_terminations": ["completed", "forced_pass_deadlock"],
    }


def _validate_config(cfg: ResolverResearchChampionConfig) -> None:
    if cfg != ResolverResearchChampionConfig():
        raise ValueError("Experiment 084 permits only its preregistered default configuration.")
    for path in (PROTOCOL, cfg.result_050, cfg.result_051):
        if not path.exists():
            raise FileNotFoundError(path)
    if not _resolver_config(cfg).checkpoint.exists():
        raise FileNotFoundError(_resolver_config(cfg).checkpoint)
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 084 frozen provenance or resolver contract mismatch.")


def _config_payload(cfg: Any) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    run_gate()


if __name__ == "__main__":
    main()

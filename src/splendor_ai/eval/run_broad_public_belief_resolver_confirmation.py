"""Run preregistered experiment 051: broad public-belief resolver confirmation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import torch

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
    _frozen_contract_matches_049,
    _integrity,
    _seat_breakdown,
    _summarize_telemetry,
)
from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT_SHA256,
    _sha256_file,
)
from splendor_ai.eval.run_shallow_operational_gate import (
    SHALLOW_CONTRACT,
    summarize_pair_rows,
)


Opponent = Literal["shallow", "016", "023"]
RESULT_050_SHA256 = "68060ffc5270b320a185f618c016f76892ae4dbc972623cd70be34cc63dff195"
CHECKPOINT_016_SHA256 = "876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056"
CHECKPOINT_023_SHA256 = "29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070"
BLOCKS: tuple[tuple[Opponent, int], ...] = (
    ("shallow", 551_000),
    ("016", 561_000),
    ("023", 571_000),
)


@dataclass(frozen=True, slots=True)
class BroadResolverConfirmationConfig:
    output_dir: Path = Path("outputs/broad_public_belief_resolver_confirmation_051")
    result_050: Path = Path("outputs/paired_public_belief_resolver_gameplay_050/result.json")
    checkpoint_016: Path = Path(
        "outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt"
    )
    checkpoint_023: Path = Path(
        "outputs/canonical_entity_coherent_023/"
        "canonical_entity_action_scorer_coherent_width64_best.pt"
    )
    pairs_per_opponent: int = 50
    workers: int = 10
    bootstrap_samples: int = 20_000
    minimum_score_share: float = 0.55
    minimum_seat_score_share: float = 0.50
    minimum_switches: int = 8
    maximum_loop_adjudicated_fraction: float = 0.001
    device: str = "cpu"


def run_gate(config: BroadResolverConfirmationConfig | None = None) -> dict[str, Any]:
    cfg = config or BroadResolverConfirmationConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 051 result does not match the frozen protocol.")
        return result

    rows = _run_all_pairs(cfg, identity)
    blocks: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for opponent, seed_start in BLOCKS:
        block_rows = [row for row in rows if row["opponent"] == opponent]
        block = _summarize_block(cfg, opponent, seed_start, block_rows)
        blocks[opponent] = block
        gates[opponent] = _block_gates(cfg, block)
        _atomic_write(cfg.output_dir / f"vs_{opponent}.json", block)

    passed = all(all(block.values()) for block in gates.values())
    result = {
        "experiment": "051_broad_public_belief_resolver_confirmation",
        "protocol": "docs/experiments/broad_public_belief_resolver_confirmation_051.md",
        "identity": identity,
        "blocks": blocks,
        "gates": gates,
        "passed": passed,
        "training_authorized": False,
        "gpu_rental_authorized": False,
        "decision": (
            "authorize_separately_preregistered_resolver_teacher_corpus"
            if passed
            else "reject_frozen_resolver_broad_deployment"
        ),
    }
    _atomic_write(result_path, result)
    print(f"051 result: {'PASS' if passed else 'FAIL'}", flush=True)
    return result


def _run_all_pairs(
    cfg: BroadResolverConfirmationConfig,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "all_blocks.progress.json"
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("Existing 051 progress does not match the frozen protocol.")
        completed = {
            (str(row["opponent"]), int(row["seed"])): row
            for row in progress.get("pairs", [])
        }

    tasks = [
        (opponent, seed)
        for opponent, seed_start in BLOCKS
        for seed in range(seed_start, seed_start + cfg.pairs_per_opponent)
    ]
    pending = [task for task in tasks if task not in completed]
    print(
        f"[resolver051] pairs={len(tasks)} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {
                executor.submit(_play_pair, cfg, opponent, seed): (opponent, seed)
                for opponent, seed in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                completed[key] = future.result()
                ordered = [completed[task] for task in tasks if task in completed]
                _atomic_write(
                    progress_path,
                    {"identity": identity, "completed": False, "pairs": ordered},
                )
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver051] complete={len(completed)}/{len(tasks)} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )

    rows = [completed[task] for task in tasks]
    _atomic_write(
        progress_path,
        {"identity": identity, "completed": True, "pairs": rows},
    )
    return rows


def _play_pair(
    cfg: BroadResolverConfirmationConfig,
    opponent: Opponent,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    resolver_cfg = ResolverGameplayConfig()
    games = []
    telemetry = []
    for candidate_seat in (0, 1):
        candidate = FrozenPublicBeliefResolverBot(resolver_cfg, seed, candidate_seat)
        baseline = _opponent(cfg, opponent)
        bots = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
        game = play_game(
            bots[0],
            bots[1],
            seed=seed,
            max_turns=resolver_cfg.max_turns,
            repetition_limit=resolver_cfg.repetition_limit,
            no_progress_limit=resolver_cfg.no_progress_limit,
        )
        games.append(asdict(game))
        telemetry.append(candidate.telemetry)
    return {
        "opponent": opponent,
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": games,
        "candidate_telemetry": telemetry,
    }


def _opponent(cfg: BroadResolverConfirmationConfig, opponent: Opponent) -> Any:
    if opponent == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    if opponent == "016":
        return LegalActionScorerBot(cfg.checkpoint_016, device=cfg.device)
    if opponent == "023":
        return CanonicalEntityActionScorerBot(cfg.checkpoint_023, device=cfg.device)
    raise ValueError(opponent)


def _summarize_block(
    cfg: BroadResolverConfirmationConfig,
    opponent: Opponent,
    seed_start: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_seeds = list(range(seed_start, seed_start + cfg.pairs_per_opponent))
    by_seed = {int(row["seed"]): row for row in rows}
    if sorted(by_seed) != expected_seeds or len(rows) != len(by_seed):
        raise ValueError(f"Opponent {opponent} does not have its exact frozen seed block.")
    ordered = [by_seed[seed] for seed in expected_seeds]
    summary = summarize_pair_rows(ordered, cfg.bootstrap_samples, seed_start)
    summary["seat_breakdown"] = _seat_breakdown(ordered)
    return {
        "opponent": opponent,
        "seed_start": seed_start,
        "pairs": cfg.pairs_per_opponent,
        "summary": summary,
        "integrity": _integrity(ordered),
        "telemetry": _summarize_telemetry(ordered),
        "rows": ordered,
    }


def _block_gates(
    cfg: BroadResolverConfirmationConfig,
    block: dict[str, Any],
) -> dict[str, bool]:
    summary = block["summary"]
    integrity = block["integrity"]
    telemetry = block["telemetry"]
    phase_active = all(
        int(telemetry["by_phase"][phase].get("completed_searches", 0)) > 0
        and int(telemetry["by_phase"][phase].get("switches", 0)) > 0
        for phase in ("early", "mid", "late")
    )
    return {
        "frozen_candidate_opponent_and_configuration": _frozen_identity_matches(cfg),
        "exactly_100_clean_games": integrity["games"] == cfg.pairs_per_opponent * 2
        and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "exact_branch_accounting": bool(telemetry["exact_branch_accounting"]),
        "all_continuations_resolved": bool(telemetry["all_continuations_resolved"]),
        "zero_hard_caps": int(telemetry["hard_caps"]) == 0,
        "bounded_loop_adjudication": float(telemetry["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction,
        "resolver_active_and_switches_all_phases": phase_active
        and int(telemetry["switches"]) >= cfg.minimum_switches,
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


def _identity(cfg: BroadResolverConfirmationConfig) -> dict[str, Any]:
    resolver_cfg = ResolverGameplayConfig()
    return {
        "name": "broad_public_belief_resolver_confirmation_051",
        "protocol": "docs/experiments/broad_public_belief_resolver_confirmation_051.md",
        "configuration": _config_payload(cfg),
        "resolver_configuration": _resolver_config_payload(resolver_cfg),
        "expected_branches_per_search": EXPECTED_BRANCHES_PER_SEARCH,
        "blocks": [
            {
                "opponent": opponent,
                "seed_start": seed_start,
                "pairs": cfg.pairs_per_opponent,
            }
            for opponent, seed_start in BLOCKS
        ],
        "shallow_contract": SHALLOW_CONTRACT,
        "hashes": _hashes(cfg),
    }


def _hashes(cfg: BroadResolverConfirmationConfig) -> dict[str, str]:
    resolver_cfg = ResolverGameplayConfig()
    return {
        "result_050": _sha256_file(cfg.result_050),
        "checkpoint_035": _sha256_file(resolver_cfg.checkpoint),
        "checkpoint_016": _sha256_file(cfg.checkpoint_016),
        "checkpoint_023": _sha256_file(cfg.checkpoint_023),
    }


def _frozen_identity_matches(cfg: BroadResolverConfirmationConfig) -> bool:
    hashes = _hashes(cfg)
    result_050 = json.loads(cfg.result_050.read_text(encoding="utf-8"))
    return (
        cfg == BroadResolverConfirmationConfig()
        and _frozen_contract_matches_049(ResolverGameplayConfig())
        and hashes["result_050"] == RESULT_050_SHA256
        and hashes["checkpoint_035"] == CHECKPOINT_SHA256
        and hashes["checkpoint_016"] == CHECKPOINT_016_SHA256
        and hashes["checkpoint_023"] == CHECKPOINT_023_SHA256
        and bool(result_050.get("passed"))
        and result_050.get("decision")
        == "authorize_separately_preregistered_broad_resolver_confirmation"
    )


def _validate_config(cfg: BroadResolverConfirmationConfig) -> None:
    if cfg != BroadResolverConfirmationConfig():
        raise ValueError("Experiment 051 permits only its preregistered default configuration.")
    for path in (cfg.result_050, cfg.checkpoint_016, cfg.checkpoint_023):
        if not path.exists():
            raise FileNotFoundError(path)
    if not _frozen_identity_matches(cfg):
        raise ValueError("Experiment 051 frozen identity or 050 authorization mismatch.")


def _config_payload(cfg: BroadResolverConfirmationConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }


def _resolver_config_payload(cfg: ResolverGameplayConfig) -> dict[str, Any]:
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

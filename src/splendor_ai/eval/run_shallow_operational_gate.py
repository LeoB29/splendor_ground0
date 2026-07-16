"""Run a resumable paired operational gate for the coherent shallow teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    GreedyHeuristicBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.eval.match import GameResult, play_game


SHALLOW_CONTRACT = {
    "depth": 1,
    "max_branching": 6,
    "max_buy_actions": 4,
    "max_reserve_actions": 2,
    "max_take_actions": 2,
    "seed": 23,
    "information_legal": True,
    "determinizations": 1,
    "determinization_seed": 29,
}
BROAD_OPPONENTS = ("greedy", "016", "035")
ABSOLUTE_FLOORS = {"greedy": 0.80, "016": 0.25, "035": 0.25}
CONTROL_DELTAS = {"greedy": -0.05, "016": 0.05, "035": 0.05}


@dataclass(frozen=True, slots=True)
class ShallowOperationalConfig:
    output_dir: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_035: Path
    workers: int = 4
    parent_pairs: int = 50
    broad_pairs: int = 30
    parent_seed_start: int = 340_000
    greedy_seed_start: int = 341_000
    checkpoint_016_seed_start: int = 342_000
    checkpoint_035_seed_start: int = 343_000
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class BlockSpec:
    name: str
    candidate_kind: str
    opponent_kind: str
    pairs: int
    seed_start: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parent-pairs", type=int, default=50)
    parser.add_argument("--broad-pairs", type=int, default=30)
    parser.add_argument("--parent-seed-start", type=int, default=340_000)
    parser.add_argument("--greedy-seed-start", type=int, default=341_000)
    parser.add_argument("--checkpoint-016-seed-start", type=int, default=342_000)
    parser.add_argument("--checkpoint-035-seed-start", type=int, default=343_000)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> ShallowOperationalConfig:
    return ShallowOperationalConfig(
        output_dir=Path(args.output_dir),
        parent_checkpoint=Path(args.parent_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_035=Path(args.checkpoint_035),
        workers=args.workers,
        parent_pairs=args.parent_pairs,
        broad_pairs=args.broad_pairs,
        parent_seed_start=args.parent_seed_start,
        greedy_seed_start=args.greedy_seed_start,
        checkpoint_016_seed_start=args.checkpoint_016_seed_start,
        checkpoint_035_seed_start=args.checkpoint_035_seed_start,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )


def block_specs(config: ShallowOperationalConfig) -> tuple[BlockSpec, ...]:
    return (
        BlockSpec(
            "parent",
            "shallow",
            "parent",
            config.parent_pairs,
            config.parent_seed_start,
        ),
        BlockSpec(
            "candidate_greedy",
            "shallow",
            "greedy",
            config.broad_pairs,
            config.greedy_seed_start,
        ),
        BlockSpec(
            "control_greedy",
            "parent",
            "greedy",
            config.broad_pairs,
            config.greedy_seed_start,
        ),
        BlockSpec(
            "candidate_016",
            "shallow",
            "016",
            config.broad_pairs,
            config.checkpoint_016_seed_start,
        ),
        BlockSpec(
            "control_016",
            "parent",
            "016",
            config.broad_pairs,
            config.checkpoint_016_seed_start,
        ),
        BlockSpec(
            "candidate_035",
            "shallow",
            "035",
            config.broad_pairs,
            config.checkpoint_035_seed_start,
        ),
        BlockSpec(
            "control_035",
            "parent",
            "035",
            config.broad_pairs,
            config.checkpoint_035_seed_start,
        ),
    )


def _checkpoint_for_kind(config: ShallowOperationalConfig, kind: str) -> Path | None:
    if kind == "parent":
        return config.parent_checkpoint
    if kind == "016":
        return config.checkpoint_016
    if kind == "035":
        return config.checkpoint_035
    return None


def _build_bot(config: ShallowOperationalConfig, kind: str):
    if kind == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    if kind == "parent":
        return CanonicalEntityActionScorerBot(
            config.parent_checkpoint,
            device=config.device,
        )
    if kind == "016":
        return LegalActionScorerBot(config.checkpoint_016, device=config.device)
    if kind == "035":
        return LegalActionAttentionScorerBot(
            config.checkpoint_035,
            device=config.device,
        )
    if kind == "greedy":
        return GreedyHeuristicBot(seed=23)
    raise ValueError(f"Unsupported operational-gate bot kind: {kind}")


def _play_pair(config: ShallowOperationalConfig, spec: BlockSpec, seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    first = play_game(
        _build_bot(config, spec.candidate_kind),
        _build_bot(config, spec.opponent_kind),
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    second = play_game(
        _build_bot(config, spec.opponent_kind),
        _build_bot(config, spec.candidate_kind),
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
    }


def _candidate_points(game: dict[str, Any], candidate_seat: int) -> float:
    winner = game["winner"]
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == candidate_seat else 0.0


def _candidate_margin(game: dict[str, Any], candidate_seat: int) -> float:
    scores = game["final_scores"]
    return float(scores[candidate_seat]) - float(scores[1 - candidate_seat])


def _mean_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    mean = statistics.fmean(values)
    if samples <= 0 or len(values) == 1 or len(set(values)) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    count = len(values)
    bootstrap = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": bootstrap[int(0.025 * (samples - 1))],
        "ci95_high": bootstrap[int(0.975 * (samples - 1))],
    }


def summarize_pair_rows(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pair_shares = []
    pair_margins = []
    candidate_wins = 0
    opponent_wins = 0
    draws = 0
    all_games = []
    candidate_fallbacks = 0
    opponent_fallbacks = 0
    for row in rows:
        seats = [int(value) for value in row["candidate_seats"]]
        games = row["games"]
        pair_shares.append(
            statistics.fmean(
                _candidate_points(game, seat)
                for game, seat in zip(games, seats, strict=True)
            )
        )
        pair_margins.append(
            statistics.fmean(
                _candidate_margin(game, seat)
                for game, seat in zip(games, seats, strict=True)
            )
        )
        for game, seat in zip(games, seats, strict=True):
            all_games.append(game)
            points = _candidate_points(game, seat)
            if points == 1.0:
                candidate_wins += 1
            elif points == 0.5:
                draws += 1
            else:
                opponent_wins += 1
            fallbacks = game.get("loop_fallback_triggers_by_seat", [0, 0])
            candidate_fallbacks += int(fallbacks[seat])
            opponent_fallbacks += int(fallbacks[1 - seat])
    termination_reasons: dict[str, int] = {}
    for game in all_games:
        reason = str(game["termination_reason"])
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    return {
        "pairs": len(rows),
        "games": len(all_games),
        "candidate_wins": candidate_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "score_share": _mean_ci(pair_shares, bootstrap_samples, bootstrap_seed),
        "final_score_margin": _mean_ci(
            pair_margins,
            bootstrap_samples,
            bootstrap_seed + 1,
        ),
        "timed_out_games": sum(bool(game["timed_out"]) for game in all_games),
        "stalled_games": sum(bool(game["stalled"]) for game in all_games),
        "termination_reasons": termination_reasons,
        "candidate_loop_fallback_triggers": candidate_fallbacks,
        "opponent_loop_fallback_triggers": opponent_fallbacks,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_identity(config: ShallowOperationalConfig, spec: BlockSpec) -> dict[str, Any]:
    checkpoints = {}
    for kind in (spec.candidate_kind, spec.opponent_kind):
        path = _checkpoint_for_kind(config, kind)
        if path is not None:
            checkpoints[kind] = {"path": str(path), "sha256": _sha256(path)}
    return {
        "spec": asdict(spec),
        "shallow_contract": SHALLOW_CONTRACT,
        "checkpoints": checkpoints,
        "max_turns": config.max_turns,
        "repetition_limit": config.repetition_limit,
        "no_progress_limit": config.no_progress_limit,
        "bootstrap_samples": config.bootstrap_samples,
        "device": config.device,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _load_progress(
    path: Path,
    identity: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != identity:
        raise ValueError(f"Progress identity mismatch: {path}")
    return {int(row["seed"]): row for row in payload.get("completed_pairs", [])}


def run_block(config: ShallowOperationalConfig, spec: BlockSpec) -> dict[str, Any]:
    output_path = config.output_dir / f"{spec.name}.json"
    progress_path = config.output_dir / f"{spec.name}.progress.json"
    identity = _config_identity(config, spec)
    rows = _load_progress(progress_path, identity)
    target_seeds = list(range(spec.seed_start, spec.seed_start + spec.pairs))
    pending = [seed for seed in target_seeds if seed not in rows]
    started = time.perf_counter()
    print(
        f"[shallow-gate:{spec.name}] pairs={spec.pairs}"
        f" complete={len(rows)} pending={len(pending)} workers={config.workers}"
    )
    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_play_pair, config, spec, seed): seed
                for seed in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                rows[seed] = future.result()
                _atomic_write(
                    progress_path,
                    {
                        "identity": identity,
                        "completed": False,
                        "completed_pairs": [rows[value] for value in sorted(rows)],
                    },
                )
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0.0 else 0.0
                remaining = len(pending) - completed
                eta = remaining / rate if rate > 0.0 else math.inf
                eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
                print(
                    f"[shallow-gate:{spec.name}] {len(rows)}/{spec.pairs}"
                    f" seed={seed} rate={rate:.3f} pairs/s eta={eta_text}"
                )
    ordered = [rows[seed] for seed in target_seeds]
    payload = {
        "format": "shallow_operational_paired_block_v1",
        "identity": identity,
        "summary": summarize_pair_rows(
            ordered,
            config.bootstrap_samples,
            spec.seed_start,
        ),
        "pair_rows": ordered,
        "elapsed_seconds_this_run": time.perf_counter() - started,
    }
    _atomic_write(output_path, payload)
    _atomic_write(
        progress_path,
        {
            "identity": identity,
            "completed": True,
            "completed_pairs": ordered,
        },
    )
    return payload


def integrity_passed(payload: dict[str, Any]) -> bool:
    summary = payload["summary"]
    return int(summary["timed_out_games"]) == 0 and int(summary["stalled_games"]) == 0


def evaluate_parent_gate(payload: dict[str, Any]) -> dict[str, bool]:
    share = payload["summary"]["score_share"]
    return {
        "integrity": integrity_passed(payload),
        "score_share_at_least_60pct": float(share["mean"]) >= 0.60,
        "score_share_ci95_lower_above_50pct": float(share["ci95_low"]) > 0.50,
    }


def evaluate_broad_gate(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = {}
    gates = {}
    for opponent in BROAD_OPPONENTS:
        candidate = blocks[f"candidate_{opponent}"]
        control = blocks[f"control_{opponent}"]
        candidate_share = float(candidate["summary"]["score_share"]["mean"])
        control_share = float(control["summary"]["score_share"]["mean"])
        delta = candidate_share - control_share
        gates[f"{opponent}_candidate_integrity"] = integrity_passed(candidate)
        gates[f"{opponent}_control_integrity"] = integrity_passed(control)
        gates[f"{opponent}_absolute_floor"] = candidate_share >= ABSOLUTE_FLOORS[opponent]
        gates[f"{opponent}_matched_control_delta"] = delta >= CONTROL_DELTAS[opponent]
        results[opponent] = {
            "candidate_score_share": candidate_share,
            "control_score_share": control_share,
            "candidate_minus_control": delta,
            "absolute_floor": ABSOLUTE_FLOORS[opponent],
            "required_control_delta": CONTROL_DELTAS[opponent],
        }
    return {"results": results, "gates": gates, "passed": all(gates.values())}


def _validate_config(config: ShallowOperationalConfig) -> None:
    if config.workers <= 0 or config.parent_pairs <= 0 or config.broad_pairs <= 0:
        raise ValueError("workers and pair counts must be positive.")
    if config.bootstrap_samples <= 0 or config.max_turns <= 0:
        raise ValueError("bootstrap_samples and max_turns must be positive.")
    for path in (
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_035,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    seed_ranges = [
        set(range(spec.seed_start, spec.seed_start + spec.pairs))
        for spec in block_specs(config)[::2]
    ]
    if any(left & right for index, left in enumerate(seed_ranges) for right in seed_ranges[index + 1 :]):
        raise ValueError("Operational gate seed blocks overlap.")


def run_operational_gate(config: ShallowOperationalConfig) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    specs = block_specs(config)
    parent = run_block(config, specs[0])
    parent_gates = evaluate_parent_gate(parent)
    broad_payload = None
    blocks = {"parent": parent}
    if all(parent_gates.values()):
        for spec in specs[1:]:
            blocks[spec.name] = run_block(config, spec)
        broad_payload = evaluate_broad_gate(blocks)
    final = {
        "gate": "shallow_search_operational_030",
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "parent_checkpoint": str(config.parent_checkpoint),
            "checkpoint_016": str(config.checkpoint_016),
            "checkpoint_035": str(config.checkpoint_035),
        },
        "shallow_contract": SHALLOW_CONTRACT,
        "parent_summary": parent["summary"],
        "parent_gates": parent_gates,
        "parent_gate_passed": all(parent_gates.values()),
        "broad_gate": broad_payload,
        "broad_gate_run": broad_payload is not None,
        "operational_gate_passed": (
            all(parent_gates.values())
            and broad_payload is not None
            and bool(broad_payload["passed"])
        ),
    }
    _atomic_write(config.output_dir / "gate_final.json", final)
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    run_operational_gate(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()

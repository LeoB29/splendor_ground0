"""Run the preregistered selective-search hybrid promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    SelectiveSearchConfig,
    SelectiveSearchHybridBot,
    ShallowSearchBot,
)
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_shallow_operational_gate import (
    SHALLOW_CONTRACT,
    integrity_passed,
    summarize_pair_rows,
)


ROUTING_CONTRACT = SelectiveSearchConfig(
    uncertainty_logit_gap=0.5,
    tactical_score_threshold=12,
    winning_score=15,
)
BROAD_OPPONENTS = ("016", "035")
BROAD_ABSOLUTE_FLOOR = 0.25
BROAD_NONINFERIORITY_DELTA = -0.05
MAX_SEARCH_FRACTION = 0.60


@dataclass(frozen=True, slots=True)
class SelectiveSearchGateConfig:
    output_dir: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_035: Path
    workers: int = 4
    retention_pairs: int = 50
    fast_parent_pairs: int = 50
    broad_pairs: int = 30
    retention_seed_start: int = 344_000
    fast_parent_seed_start: int = 345_000
    checkpoint_016_seed_start: int = 346_000
    checkpoint_035_seed_start: int = 347_000
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
    parser.add_argument("--retention-pairs", type=int, default=50)
    parser.add_argument("--fast-parent-pairs", type=int, default=50)
    parser.add_argument("--broad-pairs", type=int, default=30)
    parser.add_argument("--retention-seed-start", type=int, default=344_000)
    parser.add_argument("--fast-parent-seed-start", type=int, default=345_000)
    parser.add_argument("--checkpoint-016-seed-start", type=int, default=346_000)
    parser.add_argument("--checkpoint-035-seed-start", type=int, default=347_000)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> SelectiveSearchGateConfig:
    return SelectiveSearchGateConfig(
        output_dir=Path(args.output_dir),
        parent_checkpoint=Path(args.parent_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_035=Path(args.checkpoint_035),
        workers=args.workers,
        retention_pairs=args.retention_pairs,
        fast_parent_pairs=args.fast_parent_pairs,
        broad_pairs=args.broad_pairs,
        retention_seed_start=args.retention_seed_start,
        fast_parent_seed_start=args.fast_parent_seed_start,
        checkpoint_016_seed_start=args.checkpoint_016_seed_start,
        checkpoint_035_seed_start=args.checkpoint_035_seed_start,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )


def block_specs(config: SelectiveSearchGateConfig) -> tuple[BlockSpec, ...]:
    return (
        BlockSpec(
            "retention",
            "hybrid",
            "shallow",
            config.retention_pairs,
            config.retention_seed_start,
        ),
        BlockSpec(
            "fast_parent",
            "hybrid",
            "parent",
            config.fast_parent_pairs,
            config.fast_parent_seed_start,
        ),
        BlockSpec(
            "candidate_016",
            "hybrid",
            "016",
            config.broad_pairs,
            config.checkpoint_016_seed_start,
        ),
        BlockSpec(
            "control_016",
            "shallow",
            "016",
            config.broad_pairs,
            config.checkpoint_016_seed_start,
        ),
        BlockSpec(
            "candidate_035",
            "hybrid",
            "035",
            config.broad_pairs,
            config.checkpoint_035_seed_start,
        ),
        BlockSpec(
            "control_035",
            "shallow",
            "035",
            config.broad_pairs,
            config.checkpoint_035_seed_start,
        ),
    )


def _build_bot(config: SelectiveSearchGateConfig, kind: str):
    if kind == "hybrid":
        return SelectiveSearchHybridBot(
            config.parent_checkpoint,
            device=config.device,
            routing=ROUTING_CONTRACT,
            shallow_kwargs=SHALLOW_CONTRACT,
        )
    if kind == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    if kind == "parent":
        return CanonicalEntityActionScorerBot(config.parent_checkpoint, device=config.device)
    if kind == "016":
        return LegalActionScorerBot(config.checkpoint_016, device=config.device)
    if kind == "035":
        return LegalActionAttentionScorerBot(config.checkpoint_035, device=config.device)
    raise ValueError(f"Unsupported selective-search bot kind: {kind}")


def _play_one(
    config: SelectiveSearchGateConfig,
    spec: BlockSpec,
    seed: int,
    candidate_seat: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidate = _build_bot(config, spec.candidate_kind)
    opponent = _build_bot(config, spec.opponent_kind)
    bots = (candidate, opponent) if candidate_seat == 0 else (opponent, candidate)
    result = play_game(
        bots[0],
        bots[1],
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    telemetry = getattr(candidate, "telemetry", None)
    return asdict(result), dict(telemetry) if telemetry is not None else None


def _play_pair(
    config: SelectiveSearchGateConfig,
    spec: BlockSpec,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    first, first_telemetry = _play_one(config, spec, seed, 0)
    second, second_telemetry = _play_one(config, spec, seed, 1)
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [first, second],
        "candidate_telemetry": [first_telemetry, second_telemetry],
    }


def _summarize(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary = summarize_pair_rows(rows, bootstrap_samples, bootstrap_seed)
    telemetry_rows = [
        telemetry
        for row in rows
        for telemetry in row.get("candidate_telemetry", [])
        if telemetry is not None
    ]
    if telemetry_rows:
        turns = sum(int(row["turns"]) for row in telemetry_rows)
        search_turns = sum(int(row["search_turns"]) for row in telemetry_rows)
        summary["candidate_routing"] = {
            "turns": turns,
            "neural_turns": turns - search_turns,
            "search_turns": search_turns,
            "search_fraction": search_turns / turns if turns else 0.0,
            "uncertainty_trigger_turns": sum(
                int(row["uncertainty_trigger_turns"]) for row in telemetry_rows
            ),
            "tactical_trigger_turns": sum(
                int(row["tactical_trigger_turns"]) for row in telemetry_rows
            ),
            "both_trigger_turns": sum(
                int(row["both_trigger_turns"]) for row in telemetry_rows
            ),
        }
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(config: SelectiveSearchGateConfig, spec: BlockSpec) -> dict[str, Any]:
    return {
        "spec": asdict(spec),
        "routing_contract": asdict(ROUTING_CONTRACT),
        "shallow_contract": SHALLOW_CONTRACT,
        "checkpoints": {
            "parent": {
                "path": str(config.parent_checkpoint),
                "sha256": _sha256(config.parent_checkpoint),
            },
            "016": {
                "path": str(config.checkpoint_016),
                "sha256": _sha256(config.checkpoint_016),
            },
            "035": {
                "path": str(config.checkpoint_035),
                "sha256": _sha256(config.checkpoint_035),
            },
        },
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


def run_block(config: SelectiveSearchGateConfig, spec: BlockSpec) -> dict[str, Any]:
    output_path = config.output_dir / f"{spec.name}.json"
    progress_path = config.output_dir / f"{spec.name}.progress.json"
    identity = _identity(config, spec)
    rows: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError(f"Progress identity mismatch: {progress_path}")
        rows = {int(row["seed"]): row for row in progress.get("completed_pairs", [])}
    target_seeds = list(range(spec.seed_start, spec.seed_start + spec.pairs))
    pending = [seed for seed in target_seeds if seed not in rows]
    started = time.perf_counter()
    print(
        f"[selective-search:{spec.name}] pairs={spec.pairs}"
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
                rate = completed / elapsed if elapsed else 0.0
                remaining = len(pending) - completed
                eta = remaining / rate if rate else math.inf
                print(
                    f"[selective-search:{spec.name}] {len(rows)}/{spec.pairs}"
                    f" seed={seed} eta={eta:.1f}s"
                )
    ordered = [rows[seed] for seed in target_seeds]
    payload = {
        "format": "selective_search_paired_block_v1",
        "identity": identity,
        "summary": _summarize(ordered, config.bootstrap_samples, spec.seed_start),
        "pair_rows": ordered,
        "elapsed_seconds_this_run": time.perf_counter() - started,
    }
    _atomic_write(output_path, payload)
    _atomic_write(
        progress_path,
        {"identity": identity, "completed": True, "completed_pairs": ordered},
    )
    return payload


def _routing_passed(payload: dict[str, Any]) -> bool:
    routing = payload["summary"].get("candidate_routing", {})
    return (
        int(routing.get("search_turns", 0)) > 0
        and int(routing.get("neural_turns", 0)) > 0
        and float(routing.get("search_fraction", 1.0)) <= MAX_SEARCH_FRACTION
    )


def evaluate_retention_gate(payload: dict[str, Any]) -> dict[str, bool]:
    share = payload["summary"]["score_share"]
    return {
        "integrity": integrity_passed(payload),
        "score_share_at_least_45pct": float(share["mean"]) >= 0.45,
        "score_share_ci95_lower_at_least_35pct": float(share["ci95_low"]) >= 0.35,
        "search_fraction_at_most_60pct": _routing_passed(payload),
    }


def evaluate_fast_parent_gate(payload: dict[str, Any]) -> dict[str, bool]:
    share = payload["summary"]["score_share"]
    return {
        "integrity": integrity_passed(payload),
        "score_share_at_least_55pct": float(share["mean"]) >= 0.55,
        "score_share_ci95_lower_above_45pct": float(share["ci95_low"]) > 0.45,
        "search_fraction_at_most_60pct": _routing_passed(payload),
    }


def evaluate_broad_gate(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for opponent in BROAD_OPPONENTS:
        candidate = blocks[f"candidate_{opponent}"]
        control = blocks[f"control_{opponent}"]
        candidate_share = float(candidate["summary"]["score_share"]["mean"])
        control_share = float(control["summary"]["score_share"]["mean"])
        delta = candidate_share - control_share
        gates[f"{opponent}_candidate_integrity"] = integrity_passed(candidate)
        gates[f"{opponent}_control_integrity"] = integrity_passed(control)
        gates[f"{opponent}_absolute_floor"] = candidate_share >= BROAD_ABSOLUTE_FLOOR
        gates[f"{opponent}_shallow_noninferiority"] = (
            delta >= BROAD_NONINFERIORITY_DELTA - 1e-12
        )
        gates[f"{opponent}_search_fraction"] = _routing_passed(candidate)
        results[opponent] = {
            "candidate_score_share": candidate_share,
            "shallow_control_score_share": control_share,
            "candidate_minus_shallow": delta,
        }
    return {"results": results, "gates": gates, "passed": all(gates.values())}


def _validate_config(config: SelectiveSearchGateConfig) -> None:
    if min(config.workers, config.retention_pairs, config.fast_parent_pairs, config.broad_pairs) <= 0:
        raise ValueError("workers and pair counts must be positive.")
    if config.bootstrap_samples <= 0 or config.max_turns <= 0:
        raise ValueError("bootstrap_samples and max_turns must be positive.")
    for path in (config.parent_checkpoint, config.checkpoint_016, config.checkpoint_035):
        if not path.exists():
            raise FileNotFoundError(path)
    ranges = [
        set(range(spec.seed_start, spec.seed_start + spec.pairs))
        for spec in block_specs(config)
        if not spec.name.startswith("control_")
    ]
    if any(left & right for index, left in enumerate(ranges) for right in ranges[index + 1 :]):
        raise ValueError("Selective-search gate seed blocks overlap.")


def run_selective_search_gate(config: SelectiveSearchGateConfig) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    specs = block_specs(config)
    blocks: dict[str, dict[str, Any]] = {}

    blocks["retention"] = run_block(config, specs[0])
    retention_gates = evaluate_retention_gate(blocks["retention"])
    fast_parent_gates = None
    broad = None
    if all(retention_gates.values()):
        blocks["fast_parent"] = run_block(config, specs[1])
        fast_parent_gates = evaluate_fast_parent_gate(blocks["fast_parent"])
    if fast_parent_gates is not None and all(fast_parent_gates.values()):
        for spec in specs[2:]:
            blocks[spec.name] = run_block(config, spec)
        broad = evaluate_broad_gate(blocks)

    passed = (
        all(retention_gates.values())
        and fast_parent_gates is not None
        and all(fast_parent_gates.values())
        and broad is not None
        and bool(broad["passed"])
    )
    final = {
        "gate": "selective_search_hybrid_031",
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "parent_checkpoint": str(config.parent_checkpoint),
            "checkpoint_016": str(config.checkpoint_016),
            "checkpoint_035": str(config.checkpoint_035),
        },
        "routing_contract": asdict(ROUTING_CONTRACT),
        "shallow_contract": SHALLOW_CONTRACT,
        "retention_summary": blocks["retention"]["summary"],
        "retention_gates": retention_gates,
        "fast_parent_summary": (
            blocks["fast_parent"]["summary"] if "fast_parent" in blocks else None
        ),
        "fast_parent_gates": fast_parent_gates,
        "broad_gate": broad,
        "selective_search_gate_passed": passed,
    }
    _atomic_write(config.output_dir / "gate_final.json", final)
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    run_selective_search_gate(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()

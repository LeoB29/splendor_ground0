"""End-to-end engineering orchestration for AlphaZero programme 090."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .arena import (
    alpha_zero_raw_factory,
    alpha_zero_search_factory,
    policy_035_factory,
    run_paired_arena,
)
from .network import create_initial_checkpoint
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play
from .search import SearchConfig
from .training import TrainingConfig, train_successor


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    output_dir: Path
    device: str = "cpu"
    initialization_seed: int = 900090
    seed_start: int = 900_000
    max_turns: int = 200
    replay_generations: int = 3
    arena_pairs: int = 4
    checkpoint_035: Path | None = None
    torch_threads: int | None = None


def run_vertical_slice(config: SandboxConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    generation_zero = output / "generation_000.pt"
    create_initial_checkpoint(
        generation_zero,
        seed=config.initialization_seed,
    )
    debug_search = SearchConfig(
        simulations=8,
        mode="debug",
        seed=config.seed_start,
    )
    replay_zero = output / "generation_000_selfplay.jsonl"
    selfplay_zero = generate_self_play(
        checkpoint_path=generation_zero,
        device=config.device,
        config=SelfPlayConfig(
            generation=0,
            game_count=4,
            seed_start=config.seed_start,
            search=debug_search,
            max_turns=config.max_turns,
        ),
        replay_path=replay_zero,
        progress_path=output / "generation_000_selfplay.progress.json",
    )
    generation_one = output / "generation_001.pt"
    training = train_successor(
        parent_checkpoint=generation_zero,
        replay_paths=[replay_zero],
        output_checkpoint=generation_one,
        metrics_path=output / "generation_001_training.json",
        device=config.device,
        config=TrainingConfig(
            epochs=1,
            batch_size=64,
            learning_rate=3e-4,
            validation_fraction=0.25,
            seed=config.initialization_seed + 1,
        ),
    )
    replay_one = output / "generation_001_selfplay.jsonl"
    selfplay_one = generate_self_play(
        checkpoint_path=generation_one,
        device=config.device,
        config=SelfPlayConfig(
            generation=1,
            game_count=2,
            seed_start=config.seed_start + 100,
            search=replace(debug_search, seed=config.seed_start + 100),
            max_turns=config.max_turns,
        ),
        replay_path=replay_one,
        progress_path=output / "generation_001_selfplay.progress.json",
    )
    result = {
        "kind": "alphazero_090_vertical_slice",
        "status": "complete",
        "strength_claim": False,
        "config": _jsonable_config(config),
        "generation_zero_checkpoint": str(generation_zero),
        "generation_one_checkpoint": str(generation_one),
        "generation_zero_checkpoint_sha256": _sha256(generation_zero),
        "generation_one_checkpoint_sha256": _sha256(generation_one),
        "generation_zero_selfplay": summarize_self_play(selfplay_zero),
        "training": training,
        "generation_one_reload_selfplay": summarize_self_play(selfplay_one),
        "generation_one_actually_used": all(
            row.checkpoint_generation == 1 for row in selfplay_one.rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "result.json", result)
    return result


def run_three_generation_pilot(
    config: SandboxConfig,
    *,
    target_dose: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir / "pilot"
    output.mkdir(parents=True, exist_ok=True)
    generation_zero = output / "generation_000.pt"
    create_initial_checkpoint(
        generation_zero,
        seed=config.initialization_seed,
    )
    if target_dose:
        game_counts = (32, 64, 64)
        simulations = (32, 64, 64)
        label = "target"
    else:
        game_counts = (4, 6, 6)
        simulations = (8, 12, 12)
        label = "resource_limited"
    checkpoints = [generation_zero]
    replay_paths: list[Path] = []
    generations: list[dict[str, Any]] = []
    for generation in range(3):
        search_config = SearchConfig(
            simulations=simulations[generation],
            mode="real",
            seed=config.seed_start + generation * 100_000,
        )
        replay_path = output / f"generation_{generation:03d}_selfplay.jsonl"
        selfplay = generate_self_play(
            checkpoint_path=checkpoints[generation],
            device=config.device,
            config=SelfPlayConfig(
                generation=generation,
                game_count=game_counts[generation],
                seed_start=config.seed_start + generation * 10_000,
                search=search_config,
                max_turns=config.max_turns,
            ),
            replay_path=replay_path,
            progress_path=output / f"generation_{generation:03d}_selfplay.progress.json",
        )
        replay_paths.append(replay_path)
        buffer_paths = replay_paths[-config.replay_generations :]
        child = output / f"generation_{generation + 1:03d}.pt"
        training = train_successor(
            parent_checkpoint=checkpoints[generation],
            replay_paths=buffer_paths,
            output_checkpoint=child,
            metrics_path=output / f"generation_{generation + 1:03d}_training.json",
            device=config.device,
            config=TrainingConfig(seed=config.initialization_seed + generation + 1),
        )
        checkpoints.append(child)
        arenas = _development_arenas(
            config=config,
            generation=generation,
            parent=checkpoints[generation],
            child=child,
            search_config=search_config,
        )
        generation_result = {
            "generation": generation,
            "checkpoint": str(checkpoints[generation]),
            "checkpoint_sha256": _sha256(checkpoints[generation]),
            "selfplay": summarize_self_play(selfplay),
            "replay_path": str(replay_path),
            "replay_sha256": _sha256(replay_path),
            "child_checkpoint": str(child),
            "child_checkpoint_sha256": _sha256(child),
            "training": training,
            "arenas": arenas,
        }
        generations.append(generation_result)
        _write_json(
            output / "progress.json",
            {
                "dose_label": label,
                "completed_generations": len(generations),
                "generations": generations,
            },
        )
    result = {
        "kind": "alphazero_090_three_generation_pilot",
        "status": "complete",
        "strength_claim": False,
        "dose_label": label,
        "target_game_counts": [32, 64, 64],
        "target_simulations": [32, 64, 64],
        "realized_game_counts": list(game_counts),
        "realized_simulations": list(simulations),
        "config": _jsonable_config(config),
        "generations": generations,
        "final_checkpoint": str(checkpoints[-1]),
        "final_checkpoint_sha256": _sha256(checkpoints[-1]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "result.json", result)
    return result


def _development_arenas(
    *,
    config: SandboxConfig,
    generation: int,
    parent: Path,
    child: Path,
    search_config: SearchConfig,
) -> list[dict[str, Any]]:
    seed = config.seed_start + 500_000 + generation * 1_000
    pairs = config.arena_pairs
    results = [
        run_paired_arena(
            name="parent_search_vs_parent_raw",
            candidate_factory=alpha_zero_search_factory(
                parent, config.device, search_config
            ),
            baseline_factory=alpha_zero_raw_factory(parent, config.device),
            seed_start=seed,
            paired_seeds=pairs,
            max_turns=config.max_turns,
        ),
        run_paired_arena(
            name="child_raw_vs_parent_raw",
            candidate_factory=alpha_zero_raw_factory(child, config.device),
            baseline_factory=alpha_zero_raw_factory(parent, config.device),
            seed_start=seed + 100,
            paired_seeds=pairs,
            max_turns=config.max_turns,
        ),
        run_paired_arena(
            name="child_search_vs_parent_search",
            candidate_factory=alpha_zero_search_factory(
                child, config.device, search_config
            ),
            baseline_factory=alpha_zero_search_factory(
                parent, config.device, search_config
            ),
            seed_start=seed + 200,
            paired_seeds=pairs,
            max_turns=config.max_turns,
        ),
    ]
    if config.checkpoint_035 is not None:
        results.append(
            run_paired_arena(
                name="child_raw_vs_035",
                candidate_factory=alpha_zero_raw_factory(child, config.device),
                baseline_factory=policy_035_factory(
                    config.checkpoint_035,
                    config.device,
                ),
                seed_start=seed + 300,
                paired_seeds=pairs,
                max_turns=config.max_turns,
            )
        )
    return [asdict(result) for result in results]


def _jsonable_config(config: SandboxConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "checkpoint_035": (
            str(config.checkpoint_035) if config.checkpoint_035 is not None else None
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

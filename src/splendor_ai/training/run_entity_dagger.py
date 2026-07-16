"""Run one bounded on-policy DAgger iteration for the canonical entity actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.entity_action_supervised import (
    EntityActionReplayDataset,
    EntityActionReplaySample,
    EntityTrainMetrics,
    collate_entity_action_replay_samples,
    create_entity_action_replay_dataloader,
    fit_entity_action_scorer,
)
from splendor_ai.training.replay import (
    collect_game_replay,
    deserialize_state_snapshot,
    export_replay_games_jsonl,
    serialize_action,
)
from splendor_ai.training.run_entity_coherent_supervised import (
    EXPECTED_TEACHER,
    validate_coherent_source_summary,
)
from splendor_ai.training.run_entity_supervised_scale import (
    _load_checkpoint_model,
    _paired_loss_comparison,
    evaluate_detailed,
)


LEAGUE = ("parent", "shallow", "016", "035")
DAGGER_TEACHER = "information_legal_ShallowSearchBot_depth1_d1_dagger_relabel"


@dataclass(frozen=True, slots=True)
class EntityDaggerConfig:
    output_dir: Path
    old_source_dir: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_035: Path
    games: int = 240
    train_games: int = 192
    validation_games: int = 24
    confirmation_games: int = 24
    seed_start: int = 290_000
    split_seed: int = 2_026_0716
    workers: int = 4
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    bootstrap_samples: int = 10_000
    device: str = "cpu"
    minimum_new_relative_loss_improvement: float = 0.05
    minimum_new_top1_improvement: float = 0.03
    minimum_new_game_win_fraction: float = 0.65
    maximum_old_relative_loss_regression: float = 0.02
    maximum_old_top1_regression: float = 0.02
    maximum_old_mrr_regression: float = 0.01


class BalancedSourceDataset(Dataset[EntityActionReplaySample]):
    """Expose two datasets with exact equal source frequency per epoch."""

    def __init__(self, old: Dataset[EntityActionReplaySample], new: Dataset[EntityActionReplaySample]) -> None:
        if len(old) == 0 or len(new) == 0:
            raise ValueError("Balanced sources must both be non-empty.")
        self.old = old
        self.new = new
        self._per_source = max(len(old), len(new))

    def __len__(self) -> int:
        return 2 * self._per_source

    def __getitem__(self, index: int) -> EntityActionReplaySample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = index // 2
        if index % 2 == 0:
            return self.old[source_index % len(self.old)]
        return self.new[source_index % len(self.new)]

    def source_for_index(self, index: int) -> str:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return "old" if index % 2 == 0 else "new"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-source-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--games", type=int, default=240)
    parser.add_argument("--train-games", type=int, default=192)
    parser.add_argument("--validation-games", type=int, default=24)
    parser.add_argument("--confirmation-games", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=290_000)
    parser.add_argument("--split-seed", type=int, default=2_026_0716)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> EntityDaggerConfig:
    return EntityDaggerConfig(
        output_dir=Path(args.output_dir),
        old_source_dir=Path(args.old_source_dir),
        parent_checkpoint=Path(args.parent_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_035=Path(args.checkpoint_035),
        games=args.games,
        train_games=args.train_games,
        validation_games=args.validation_games,
        confirmation_games=args.confirmation_games,
        seed_start=args.seed_start,
        split_seed=args.split_seed,
        workers=args.workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )


def league_assignment(game_index: int) -> tuple[str, int | None]:
    if game_index < 0:
        raise ValueError("game_index must be non-negative.")
    opponent = LEAGUE[game_index % len(LEAGUE)]
    parent_seat = None if opponent == "parent" else (game_index // len(LEAGUE)) % 2
    return opponent, parent_seat


def split_league_seed_values(
    seed_start: int,
    games: int,
    train_games: int,
    validation_games: int,
    confirmation_games: int,
    seed: int,
) -> dict[str, list[int]]:
    counts = (train_games, validation_games, confirmation_games)
    if games != sum(counts) or any(count % len(LEAGUE) != 0 for count in counts):
        raise ValueError("Every split must be divisible by the four-profile league.")
    splits = {"train": [], "validation": [], "confirmation": []}
    per_profile = tuple(count // len(LEAGUE) for count in counts)
    for profile_index in range(len(LEAGUE)):
        profile_seeds = [
            seed_start + game_index
            for game_index in range(games)
            if game_index % len(LEAGUE) == profile_index
        ]
        random.Random(seed + profile_index).shuffle(profile_seeds)
        train_count, validation_count, _ = per_profile
        splits["train"].extend(profile_seeds[:train_count])
        splits["validation"].extend(
            profile_seeds[train_count : train_count + validation_count]
        )
        splits["confirmation"].extend(
            profile_seeds[train_count + validation_count :]
        )
    return {name: sorted(values) for name, values in splits.items()}


def _shallow_teacher() -> ShallowSearchBot:
    return ShallowSearchBot(
        depth=1,
        max_branching=6,
        max_buy_actions=4,
        max_reserve_actions=2,
        max_take_actions=2,
        seed=23,
        information_legal=True,
        determinizations=1,
        determinization_seed=29,
    )


def _parent_bot(config: EntityDaggerConfig) -> CanonicalEntityActionScorerBot:
    return CanonicalEntityActionScorerBot(config.parent_checkpoint, device=config.device)


def _rollout_bots(config: EntityDaggerConfig, game_index: int):
    opponent, parent_seat = league_assignment(game_index)
    if opponent == "parent":
        return _parent_bot(config), _parent_bot(config), opponent, parent_seat
    if opponent == "shallow":
        opponent_bot = _shallow_teacher()
    elif opponent == "016":
        opponent_bot = LegalActionScorerBot(config.checkpoint_016, device=config.device)
    elif opponent == "035":
        opponent_bot = LegalActionAttentionScorerBot(config.checkpoint_035, device=config.device)
    else:
        raise ValueError(f"Unsupported league opponent: {opponent}")
    parent = _parent_bot(config)
    if parent_seat == 0:
        return parent, opponent_bot, opponent, parent_seat
    return opponent_bot, parent, opponent, parent_seat


def _collect_and_label_game(
    config: EntityDaggerConfig,
    seed: int,
    game_index: int,
    game_dir: str,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    seat0, seat1, opponent, parent_seat = _rollout_bots(config, game_index)
    game = collect_game_replay(
        seat0,
        seat1,
        seed=seed,
        max_turns=400,
        repetition_limit=4,
        no_progress_limit=60,
        include_state_snapshots=True,
    )
    directory = Path(game_dir)
    replay_path = directory / f"{seed}.jsonl"
    raw_path = directory / f"{seed}.rollout.tmp.jsonl"
    temp_path = replay_path.with_suffix(".jsonl.tmp")
    export_replay_games_jsonl(raw_path, (game,))

    codec = ActionCodec()
    env = SplendorEnv()
    teachers = (_shallow_teacher(), _shallow_teacher())
    disagreements = 0
    transitions: dict[str, int] = {}
    with raw_path.open("r", encoding="utf-8") as source, temp_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            row = json.loads(line)
            row, disagreed, transition = relabel_snapshot_row(
                row,
                teachers=teachers,
                env=env,
                codec=codec,
                rollout_profile=opponent,
                rollout_parent_seat=parent_seat,
            )
            if disagreed:
                disagreements += 1
            transitions[transition] = transitions.get(transition, 0) + 1
            destination.write(json.dumps(row))
            destination.write("\n")
    os.replace(temp_path, replay_path)
    raw_path.unlink()

    summary = {
        "seed": seed,
        "rollout_profile": opponent,
        "rollout_parent_seat": parent_seat,
        "winner": game.winner,
        "final_scores": list(game.final_scores),
        "turns": game.turns,
        "steps": len(game.steps),
        "teacher_disagreements": disagreements,
        "teacher_disagreement_rate": disagreements / len(game.steps),
        "action_type_transitions": transitions,
        "stalled": game.stalled,
        "timed_out": game.timed_out,
        "termination_reason": game.termination_reason,
        "bot_seats": list(game.bot_seats),
        "loop_fallback_triggers": sum(game.loop_fallback_triggers_by_seat),
    }
    summary_path = directory / f"{seed}.summary.json"
    temp_summary = summary_path.with_suffix(".json.tmp")
    temp_summary.write_text(json.dumps(summary), encoding="utf-8")
    os.replace(temp_summary, summary_path)
    return summary


def relabel_snapshot_row(
    row: dict[str, Any],
    *,
    teachers: tuple[ShallowSearchBot, ShallowSearchBot],
    env: SplendorEnv,
    codec: ActionCodec,
    rollout_profile: str,
    rollout_parent_seat: int | None,
) -> tuple[dict[str, Any], bool, str]:
    snapshot = row.get("state_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("DAGGER relabeling requires state snapshots.")
    state = deserialize_state_snapshot(snapshot)
    legal_actions = env.legal_actions(state)
    teacher_action = teachers[state.current_player].choose_action(env, state, legal_actions)
    if teacher_action is None:
        raise RuntimeError("Shallow teacher returned no legal action.")
    result = dict(row)
    rollout_index = int(result["action_index"])
    rollout_payload = dict(result["action_payload"])
    teacher_index = codec.encode(state, teacher_action)
    teacher_payload = serialize_action(teacher_action)
    transition = (
        f"{rollout_payload['action_type']}->{teacher_payload['action_type']}"
    )
    result["rollout_action_index"] = rollout_index
    result["rollout_action_payload"] = rollout_payload
    result["action_index"] = teacher_index
    result["action_payload"] = teacher_payload
    result["target_source"] = DAGGER_TEACHER
    result["rollout_profile"] = rollout_profile
    result["rollout_parent_seat"] = rollout_parent_seat
    return result, teacher_index != rollout_index, transition


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_contract(config: EntityDaggerConfig) -> dict[str, Any]:
    return {
        "contract": "canonical_public_belief_v1",
        "experiment": "canonical_entity_dagger_v1",
        "rollout_parent": str(config.parent_checkpoint),
        "rollout_parent_sha256": _checkpoint_sha256(config.parent_checkpoint),
        "rollout_league": list(LEAGUE),
        "games_per_profile": config.games // len(LEAGUE),
        "teacher": DAGGER_TEACHER,
        "teacher_applied_to": "every rollout state; rollout action preserved separately",
        "include_state_snapshots": True,
        "max_turns": 400,
        "repetition_limit": 4,
        "no_progress_limit": 60,
        "training_source_weighting": "exact 50pct old022 and 50pct new025 per epoch",
        "value_loss_weight": 0.0,
    }


def _validate_config(config: EntityDaggerConfig) -> None:
    if config.games != config.train_games + config.validation_games + config.confirmation_games:
        raise ValueError("train, validation, and confirmation games must sum to games.")
    if config.games % len(LEAGUE) != 0:
        raise ValueError("games must be divisible by the four-profile league.")
    if min(config.train_games, config.validation_games, config.confirmation_games) <= 0:
        raise ValueError("Every game split must be non-empty.")
    if min(config.workers, config.epochs, config.batch_size) <= 0:
        raise ValueError("workers, epochs, and batch_size must be positive.")
    for path in (
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_035,
        config.old_source_dir / "replays.jsonl",
        config.old_source_dir / "summary.json",
        config.old_source_dir / "game_splits.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    old_summary = json.loads(
        (config.old_source_dir / "summary.json").read_text(encoding="utf-8")
    )
    validate_coherent_source_summary(old_summary)


def _freeze_contract_and_splits(config: EntityDaggerConfig) -> dict[str, list[int]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    contract = _corpus_contract(config)
    contract_path = config.output_dir / "corpus_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Existing DAGGER corpus contract does not match this run.")
    else:
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    splits = split_league_seed_values(
        config.seed_start,
        config.games,
        config.train_games,
        config.validation_games,
        config.confirmation_games,
        config.split_seed,
    )
    split_path = config.output_dir / "game_splits.json"
    if split_path.exists():
        if json.loads(split_path.read_text(encoding="utf-8")) != splits:
            raise ValueError("Existing DAGGER game split does not match this run.")
    else:
        split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def generate_dagger_corpus(config: EntityDaggerConfig) -> tuple[Path, dict[str, Any]]:
    _validate_config(config)
    _freeze_contract_and_splits(config)
    game_dir = config.output_dir / "corpus_games"
    game_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(config.seed_start, config.seed_start + config.games))
    summaries: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        summary_path = game_dir / f"{seed}.summary.json"
        replay_path = game_dir / f"{seed}.jsonl"
        if summary_path.exists() and replay_path.exists():
            summaries[seed] = json.loads(summary_path.read_text(encoding="utf-8"))
    pending = [(seed, seed - config.seed_start) for seed in seeds if seed not in summaries]
    print(
        f"[entity-dagger:corpus] complete={len(summaries)} "
        f"pending={len(pending)} workers={config.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(
                    _collect_and_label_game,
                    config,
                    seed,
                    game_index,
                    str(game_dir),
                ): seed
                for seed, game_index in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                summary = future.result()
                summaries[int(summary["seed"])] = summary
                done = len(summaries)
                if done % 10 == 0 or done == config.games:
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed if elapsed else 0.0
                    eta = (len(pending) - completed) / rate if rate else math.inf
                    print(
                        f"[entity-dagger:corpus] {done}/{config.games} "
                        f"rate={rate:.3f} games/s eta={eta:.1f}s",
                        flush=True,
                    )
    ordered = [summaries[seed] for seed in seeds]
    replay_path = config.output_dir / "replays.jsonl"
    temp_replay = replay_path.with_suffix(".jsonl.tmp")
    with temp_replay.open("wb") as destination:
        for seed in seeds:
            with (game_dir / f"{seed}.jsonl").open("rb") as source:
                shutil.copyfileobj(source, destination)
    os.replace(temp_replay, replay_path)
    summary = _summarize_corpus(config, ordered)
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if summary["stalled_games"] or summary["timed_out_games"]:
        raise RuntimeError("DAGGER rollout integrity gate failed.")
    return replay_path, summary


def _summarize_corpus(
    config: EntityDaggerConfig,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    transitions: dict[str, int] = {}
    for row in rows:
        for key, count in row["action_type_transitions"].items():
            transitions[key] = transitions.get(key, 0) + int(count)
    total_steps = sum(int(row["steps"]) for row in rows)
    disagreements = sum(int(row["teacher_disagreements"]) for row in rows)
    return {
        "games": len(rows),
        "total_steps": total_steps,
        "teacher_disagreements": disagreements,
        "teacher_disagreement_rate": disagreements / total_steps,
        "profiles": {
            profile: sum(row["rollout_profile"] == profile for row in rows)
            for profile in LEAGUE
        },
        "profile_steps": {
            profile: sum(
                int(row["steps"])
                for row in rows
                if row["rollout_profile"] == profile
            )
            for profile in LEAGUE
        },
        "action_type_transitions": dict(sorted(transitions.items())),
        "seat0_wins": sum(row["winner"] == 0 for row in rows),
        "seat1_wins": sum(row["winner"] == 1 for row in rows),
        "draws": sum(row["winner"] is None for row in rows),
        "stalled_games": sum(bool(row["stalled"]) for row in rows),
        "timed_out_games": sum(bool(row["timed_out"]) for row in rows),
        "loop_fallback_triggers": sum(int(row["loop_fallback_triggers"]) for row in rows),
        "average_turns": statistics.fmean(float(row["turns"]) for row in rows),
        "metadata": _corpus_contract(config),
    }


def _indices_for_seeds(dataset: EntityActionReplayDataset, seeds: set[int]) -> list[int]:
    return [
        index
        for index in range(len(dataset))
        if dataset.game_seed_for_index(index) in seeds
    ]


def _subsets_from_frozen_splits(
    dataset: EntityActionReplayDataset,
    splits: dict[str, list[int]],
) -> dict[str, Subset[EntityActionReplaySample]]:
    return {
        name: Subset(dataset, _indices_for_seeds(dataset, set(seeds)))
        for name, seeds in splits.items()
    }


def _balanced_loader(
    old: Dataset[EntityActionReplaySample],
    new: Dataset[EntityActionReplaySample],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        BalancedSourceDataset(old, new),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_entity_action_replay_samples,
    )


def _confirmation_gates(
    config: EntityDaggerConfig,
    evaluations: dict[str, Any],
) -> dict[str, bool]:
    new = evaluations["new_confirmation"]
    old = evaluations["old_confirmation"]
    new_candidate = new["candidate"]["aggregate"]
    new_parent = new["parent"]["aggregate"]
    old_candidate = old["candidate"]["aggregate"]
    old_parent = old["parent"]["aggregate"]
    new_relative = 1.0 - new_candidate["policy_loss"] / new_parent["policy_loss"]
    old_relative_regression = old_candidate["policy_loss"] / old_parent["policy_loss"] - 1.0
    return {
        "new_policy_loss_improves_at_least_5pct": (
            new_relative >= config.minimum_new_relative_loss_improvement
        ),
        "new_top1_improves_at_least_3pp": (
            new_candidate["top1_accuracy"] - new_parent["top1_accuracy"]
            >= config.minimum_new_top1_improvement
        ),
        "new_mean_reciprocal_rank_improves": (
            new_candidate["mean_reciprocal_rank"]
            > new_parent["mean_reciprocal_rank"]
        ),
        "new_paired_loss_ci_below_zero": new["comparison"]["ci95_high"] < 0.0,
        "new_better_on_at_least_65pct_games": (
            new["comparison"]["candidate_better_fraction"]
            >= config.minimum_new_game_win_fraction
        ),
        "old_policy_loss_regression_at_most_2pct": (
            old_relative_regression <= config.maximum_old_relative_loss_regression
        ),
        "old_top1_regression_at_most_2pp": (
            old_candidate["top1_accuracy"]
            >= old_parent["top1_accuracy"] - config.maximum_old_top1_regression
        ),
        "old_mrr_regression_at_most_1pp": (
            old_candidate["mean_reciprocal_rank"]
            >= old_parent["mean_reciprocal_rank"] - config.maximum_old_mrr_regression
        ),
    }


def train_and_gate_dagger(
    config: EntityDaggerConfig,
    replay_path: Path,
    corpus_summary: dict[str, Any],
) -> dict[str, Any]:
    old_dataset = EntityActionReplayDataset(config.old_source_dir / "replays.jsonl")
    new_dataset = EntityActionReplayDataset(replay_path)
    old_splits = json.loads(
        (config.old_source_dir / "game_splits.json").read_text(encoding="utf-8")
    )
    new_splits = _freeze_contract_and_splits(config)
    old = _subsets_from_frozen_splits(old_dataset, old_splits)
    new = _subsets_from_frozen_splits(new_dataset, new_splits)
    if any(len(subset) == 0 for subset in (*old.values(), *new.values())):
        raise ValueError("Every old/new split must contain positions.")

    torch.manual_seed(config.split_seed)
    model = _load_checkpoint_model(config.parent_checkpoint, config.device)
    parent = _load_checkpoint_model(config.parent_checkpoint, config.device)
    started = time.perf_counter()

    def report_epoch(
        epoch: int,
        train_metric: EntityTrainMetrics,
        validation_metric: EntityTrainMetrics | None,
    ) -> None:
        validation_text = (
            f" validation_loss={validation_metric.policy_loss:.4f}"
            if validation_metric is not None
            else ""
        )
        print(
            f"[entity-dagger:train] epoch={epoch}/{config.epochs} "
            f"train_loss={train_metric.policy_loss:.4f}{validation_text}",
            flush=True,
        )

    artifacts = fit_entity_action_scorer(
        _balanced_loader(old["train"], new["train"], config.batch_size, True),
        _balanced_loader(
            old["validation"], new["validation"], config.batch_size, False
        ),
        model=model,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        value_loss_weight=0.0,
        selection_metric="policy_loss",
        progress_callback=report_epoch,
    )
    if artifacts.best_validation_state_dict is None:
        raise RuntimeError("DAGGER training did not select a validation checkpoint.")
    model.load_state_dict(artifacts.best_validation_state_dict)
    model.to(config.device).eval()
    checkpoint_path = config.output_dir / "canonical_entity_action_scorer_dagger_width64_best.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": artifacts.best_validation_state_dict,
            "model_config": asdict(model.config),
            "train_config": _config_payload(config),
        },
        checkpoint_path,
    )

    evaluations: dict[str, Any] = {}
    evaluation_splits = {
        "new_validation": new["validation"],
        "old_validation": old["validation"],
        "new_confirmation": new["confirmation"],
        "old_confirmation": old["confirmation"],
    }
    for offset, (name, subset) in enumerate(evaluation_splits.items()):
        parent_metrics = evaluate_detailed(parent, subset, config.batch_size, config.device)
        candidate_metrics = evaluate_detailed(model, subset, config.batch_size, config.device)
        comparison = _paired_loss_comparison(
            candidate_metrics,
            parent_metrics,
            config.bootstrap_samples,
            config.split_seed + offset,
        )
        evaluations[name] = {
            "parent": parent_metrics,
            "candidate": candidate_metrics,
            "comparison": comparison,
        }
    gates = _confirmation_gates(config, evaluations)
    payload = {
        "experiment": "canonical_entity_dagger_v1",
        "config": _config_payload(config),
        "corpus": corpus_summary,
        "old_splits": old_splits,
        "new_splits": new_splits,
        "split_positions": {
            "old": {name: len(subset) for name, subset in old.items()},
            "new": {name: len(subset) for name, subset in new.items()},
        },
        "balanced_train_positions_per_epoch": 2
        * max(len(old["train"]), len(new["train"])),
        "source_fraction_per_epoch": {"old022": 0.5, "new025": 0.5},
        "best_epoch": artifacts.best_epoch_index,
        "elapsed_seconds": time.perf_counter() - started,
        "train_history": [asdict(metric) for metric in artifacts.train_metrics],
        "validation_history": [
            asdict(metric) for metric in artifacts.validation_metrics
        ],
        "evaluations": evaluations,
        "checkpoint": str(checkpoint_path),
        "confirmation_gates": gates,
        "offline_gate_passed": all(gates.values()),
        "match_gate_eligible": all(gates.values()),
    }
    (config.output_dir / "entity_dagger_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": payload["best_epoch"],
                "confirmation_gates": gates,
                "match_gate_eligible": payload["match_gate_eligible"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def _config_payload(config: EntityDaggerConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "output_dir",
        "old_source_dir",
        "parent_checkpoint",
        "checkpoint_016",
        "checkpoint_035",
    ):
        payload[key] = str(payload[key])
    return payload


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    _validate_config(config)
    _freeze_contract_and_splits(config)
    if args.skip_generation:
        replay_path = config.output_dir / "replays.jsonl"
        summary_path = config.output_dir / "summary.json"
        if not replay_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                "--skip-generation requires existing replays.jsonl and summary.json."
            )
        corpus_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        replay_path, corpus_summary = generate_dagger_corpus(config)
    if args.generate_only:
        print(
            json.dumps(
                {
                    "generated_only": True,
                    "replay_path": str(replay_path),
                    "summary": corpus_summary,
                },
                indent=2,
            )
        )
        return
    train_and_gate_dagger(config, replay_path, corpus_summary)


if __name__ == "__main__":
    main()

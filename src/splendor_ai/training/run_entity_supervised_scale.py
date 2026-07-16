"""Scale the clean entity imitation foundation with untouched game holdouts."""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from torch.utils.data import Subset

from splendor_ai.bots import GreedyHeuristicBot, ShallowSearchBot
from splendor_ai.training.entity_action_supervised import (
    EntityActionReplayDataset,
    create_entity_action_replay_dataloader,
    fit_entity_action_scorer,
)
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    masked_legal_action_scores,
)
from splendor_ai.training.replay import collect_game_replay, export_replay_games_jsonl


@dataclass(frozen=True, slots=True)
class EntityScaleConfig:
    output_dir: Path
    parent_checkpoint: Path
    games: int = 500
    train_games: int = 400
    validation_games: int = 50
    confirmation_games: int = 50
    seed_start: int = 210_000
    split_seed: int = 2_026_0716
    init_seed: int = 2_026_0716
    workers: int = 4
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    bootstrap_samples: int = 10_000
    minimum_relative_loss_improvement: float = 0.05
    minimum_game_win_fraction: float = 0.65
    pairing: str = "greedy_shallow"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--train-games", type=int, default=400)
    parser.add_argument("--validation-games", type=int, default=50)
    parser.add_argument("--confirmation-games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=210_000)
    parser.add_argument("--split-seed", type=int, default=2_026_0716)
    parser.add_argument("--init-seed", type=int, default=2_026_0716)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--pairing",
        choices=("greedy_shallow", "shallow_shallow"),
        default="greedy_shallow",
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> EntityScaleConfig:
    return EntityScaleConfig(
        output_dir=Path(args.output_dir),
        parent_checkpoint=Path(args.parent_checkpoint),
        games=args.games,
        train_games=args.train_games,
        validation_games=args.validation_games,
        confirmation_games=args.confirmation_games,
        seed_start=args.seed_start,
        split_seed=args.split_seed,
        init_seed=args.init_seed,
        workers=args.workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
        pairing=args.pairing,
    )


def _validate_config(config: EntityScaleConfig) -> None:
    if config.games != config.train_games + config.validation_games + config.confirmation_games:
        raise ValueError("train, validation, and confirmation game counts must sum to games.")
    if min(config.train_games, config.validation_games, config.confirmation_games) <= 0:
        raise ValueError("Every game split must be non-empty.")
    if config.workers <= 0 or config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("workers, epochs, and batch_size must be positive.")
    if not config.parent_checkpoint.exists():
        raise FileNotFoundError(config.parent_checkpoint)
    if config.pairing not in {"greedy_shallow", "shallow_shallow"}:
        raise ValueError("pairing must be greedy_shallow or shallow_shallow.")


def _corpus_contract(config: EntityScaleConfig) -> dict[str, Any]:
    teacher = (
        "information_legal_ShallowSearchBot_depth1_d1_selfplay"
        if config.pairing == "shallow_shallow"
        else "GreedyHeuristicBot_vs_information_legal_ShallowSearchBot_depth1_d1"
    )
    return {
        "contract": "canonical_public_belief_v1",
        "teacher": teacher,
        "games": config.games,
        "seed_start": config.seed_start,
        "swap_seats": True,
        "include_state_snapshots": True,
        "max_turns": 400,
        "repetition_limit": 4,
        "no_progress_limit": 60,
    }


def _collect_clean_game(
    seed: int,
    game_index: int,
    game_dir: str,
    pairing: str = "greedy_shallow",
) -> dict[str, Any]:
    greedy = GreedyHeuristicBot(seed=11)
    def shallow_bot() -> ShallowSearchBot:
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
    if pairing == "shallow_shallow":
        seat0, seat1 = shallow_bot(), shallow_bot()
    elif pairing == "greedy_shallow":
        shallow = shallow_bot()
        seat0, seat1 = (greedy, shallow) if game_index % 2 == 0 else (shallow, greedy)
    else:
        raise ValueError(f"Unsupported pairing: {pairing}")
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
    temp_path = replay_path.with_suffix(".jsonl.tmp")
    export_replay_games_jsonl(temp_path, (game,))
    os.replace(temp_path, replay_path)
    summary = {
        "seed": seed,
        "winner": game.winner,
        "final_scores": list(game.final_scores),
        "turns": game.turns,
        "steps": len(game.steps),
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


def generate_clean_scale_corpus(config: EntityScaleConfig) -> tuple[Path, dict[str, Any]]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    game_dir = config.output_dir / "corpus_games"
    game_dir.mkdir(parents=True, exist_ok=True)
    contract_path = config.output_dir / "corpus_contract.json"
    contract = _corpus_contract(config)
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Existing corpus contract does not match this run.")
    else:
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    seeds = list(range(config.seed_start, config.seed_start + config.games))
    summaries: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        summary_path = game_dir / f"{seed}.summary.json"
        replay_path = game_dir / f"{seed}.jsonl"
        if summary_path.exists() and replay_path.exists():
            summaries[seed] = json.loads(summary_path.read_text(encoding="utf-8"))
    pending = [(seed, seed - config.seed_start) for seed in seeds if seed not in summaries]
    print(f"[entity-scale:corpus] complete={len(summaries)} pending={len(pending)} workers={config.workers}", flush=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(
                    _collect_clean_game, seed, index, str(game_dir), config.pairing
                ): seed
                for seed, index in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                summary = future.result()
                summaries[int(summary["seed"])] = summary
                done = len(summaries)
                if done % 10 == 0 or done == config.games:
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed if elapsed else 0.0
                    eta = (len(pending) - completed) / rate if rate else math.inf
                    print(f"[entity-scale:corpus] {done}/{config.games} rate={rate:.3f} games/s eta={eta:.1f}s", flush=True)

    ordered = [summaries[seed] for seed in seeds]
    replay_path = config.output_dir / "replays.jsonl"
    temp_replay = replay_path.with_suffix(".jsonl.tmp")
    with temp_replay.open("wb") as destination:
        for seed in seeds:
            with (game_dir / f"{seed}.jsonl").open("rb") as source:
                shutil.copyfileobj(source, destination)
    os.replace(temp_replay, replay_path)
    summary = _summarize_corpus(ordered, contract)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (config.output_dir / "manifest.json").write_text(
        json.dumps(_build_manifest(config, ordered, summary), indent=2), encoding="utf-8"
    )
    if summary["stalled_games"] or summary["timed_out_games"]:
        raise RuntimeError("Clean corpus gate failed: stalled or timed-out games were generated.")
    return replay_path, summary


def _summarize_corpus(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    count = len(rows)
    return {
        "games": count,
        "total_steps": sum(int(row["steps"]) for row in rows),
        "seat0_wins": sum(row["winner"] == 0 for row in rows),
        "seat1_wins": sum(row["winner"] == 1 for row in rows),
        "draws": sum(row["winner"] is None for row in rows),
        "stalled_games": sum(bool(row["stalled"]) for row in rows),
        "timed_out_games": sum(bool(row["timed_out"]) for row in rows),
        "loop_fallback_triggers": sum(int(row["loop_fallback_triggers"]) for row in rows),
        "average_turns": statistics.fmean(float(row["turns"]) for row in rows),
        "average_final_score_seat0": statistics.fmean(float(row["final_scores"][0]) for row in rows),
        "average_final_score_seat1": statistics.fmean(float(row["final_scores"][1]) for row in rows),
        "metadata": contract,
    }


def _build_manifest(config: EntityScaleConfig, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    by_seats: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        seats = tuple(str(item) for item in row["bot_seats"])
        block = by_seats.setdefault(seats, {"seat0": seats[0], "seat1": seats[1], "games": 0, "seat0_wins": 0, "seat1_wins": 0, "draws": 0})
        block["games"] += 1
        block["seat0_wins"] += row["winner"] == 0
        block["seat1_wins"] += row["winner"] == 1
        block["draws"] += row["winner"] is None
    return {
        "manifest_version": 1,
        "source_type": "clean_rule_based_legal_replay",
        "files": {"replays": "replays.jsonl", "summary": "summary.json"},
        "seed_block": {"seed_start": config.seed_start, "seed_end_inclusive": config.seed_start + config.games - 1, "games": config.games},
        "results": {key: value for key, value in summary.items() if key != "metadata"},
        "matchups": list(by_seats.values()),
        "metadata": _corpus_contract(config),
    }


def split_game_heldout_indices(
    dataset: EntityActionReplayDataset,
    train_games: int,
    validation_games: int,
    confirmation_games: int,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, list[int]]]:
    game_seeds = sorted({dataset.game_seed_for_index(index) for index in range(len(dataset))})
    if len(game_seeds) != train_games + validation_games + confirmation_games:
        raise ValueError("Dataset game count does not match requested split counts.")
    random.Random(seed).shuffle(game_seeds)
    split_seeds = {
        "train": sorted(game_seeds[:train_games]),
        "validation": sorted(game_seeds[train_games : train_games + validation_games]),
        "confirmation": sorted(game_seeds[train_games + validation_games :]),
    }
    sets = {name: set(values) for name, values in split_seeds.items()}
    indices = {
        name: [index for index in range(len(dataset)) if dataset.game_seed_for_index(index) in seeds]
        for name, seeds in sets.items()
    }
    return indices["train"], indices["validation"], indices["confirmation"], split_seeds


def _candidate_configs() -> dict[str, CanonicalEntityActionScorerConfig]:
    return {
        "width64": CanonicalEntityActionScorerConfig(
            entity_hidden_size=64, action_hidden_size=64, scorer_hidden_size=96,
            entity_layers=1, action_layers=1, attention_heads=4,
            attention_feedforward_size=128, id_embedding_size=16,
        ),
        "width128": CanonicalEntityActionScorerConfig(
            entity_hidden_size=128, action_hidden_size=128, scorer_hidden_size=192,
            entity_layers=1, action_layers=1, attention_heads=4,
            attention_feedforward_size=256, id_embedding_size=32,
        ),
    }


def _load_checkpoint_model(path: Path, device: str) -> CanonicalEntityActionScorerModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = CanonicalEntityActionScorerModel(CanonicalEntityActionScorerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def evaluate_detailed(
    model: CanonicalEntityActionScorerModel,
    dataset: Subset,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    dataloader = create_entity_action_replay_dataloader(dataset, batch_size, shuffle=False)
    totals = {"loss": 0.0, "uniform": 0.0, "correct": 0.0, "top3": 0.0, "rr": 0.0, "samples": 0.0}
    games: dict[int, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {name: value.to(device) for name, value in batch.items()}
            scores, _ = model(
                entity_type_ids=batch["entity_type_ids"], entity_card_ids=batch["entity_card_ids"],
                entity_noble_ids=batch["entity_noble_ids"], entity_numeric_features=batch["entity_numeric_features"],
                entity_mask=batch["entity_mask"], action_type_ids=batch["action_type_ids"],
                action_card_ids=batch["action_card_ids"], action_noble_ids=batch["action_noble_ids"],
                action_numeric_features=batch["action_numeric_features"], action_mask=batch["action_mask"],
            )
            masked = masked_legal_action_scores(scores, batch["action_mask"])
            losses = F.cross_entropy(masked, batch["target_action_row"], reduction="none")
            target_scores = masked.gather(1, batch["target_action_row"].unsqueeze(1))
            ranks = 1 + ((masked > target_scores) & batch["action_mask"]).sum(dim=1)
            uniform = torch.log(batch["action_mask"].sum(dim=1).float())
            correct = ranks == 1
            top3 = ranks <= 3
            reciprocal = 1.0 / ranks.float()
            for row in range(len(losses)):
                seed = int(batch["game_seed"][row].item())
                game = games.setdefault(seed, {"loss": 0.0, "uniform": 0.0, "correct": 0.0, "top3": 0.0, "rr": 0.0, "samples": 0.0})
                values = (losses[row], uniform[row], correct[row], top3[row], reciprocal[row])
                for key, value in zip(("loss", "uniform", "correct", "top3", "rr"), values, strict=True):
                    scalar = float(value.item())
                    game[key] += scalar
                    totals[key] += scalar
                game["samples"] += 1
                totals["samples"] += 1
    def finish(row: dict[str, float]) -> dict[str, float]:
        count = row["samples"]
        return {
            "policy_loss": row["loss"] / count,
            "uniform_policy_loss": row["uniform"] / count,
            "top1_accuracy": row["correct"] / count,
            "top3_accuracy": row["top3"] / count,
            "mean_reciprocal_rank": row["rr"] / count,
            "samples": int(count),
        }
    return {"aggregate": finish(totals), "by_game": {str(seed): finish(row) for seed, row in sorted(games.items())}}


def _paired_loss_comparison(candidate: dict[str, Any], parent: dict[str, Any], samples: int, seed: int) -> dict[str, Any]:
    game_seeds = sorted(set(candidate["by_game"]) & set(parent["by_game"]))
    differences = [candidate["by_game"][game]["policy_loss"] - parent["by_game"][game]["policy_loss"] for game in game_seeds]
    rng = random.Random(seed)
    bootstrap = sorted(
        statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    ) if samples > 0 else [statistics.fmean(differences)]
    return {
        "games": len(differences),
        "mean_candidate_minus_parent": statistics.fmean(differences),
        "ci95_low": bootstrap[int(0.025 * (len(bootstrap) - 1))],
        "ci95_high": bootstrap[int(0.975 * (len(bootstrap) - 1))],
        "candidate_better_games": sum(value < 0.0 for value in differences),
        "candidate_better_fraction": sum(value < 0.0 for value in differences) / len(differences),
    }


def _offline_gates(config: EntityScaleConfig, candidate: dict[str, Any], parent: dict[str, Any], comparison: dict[str, Any]) -> dict[str, bool]:
    candidate_aggregate = candidate["aggregate"]
    parent_aggregate = parent["aggregate"]
    relative = 1.0 - candidate_aggregate["policy_loss"] / parent_aggregate["policy_loss"]
    return {
        "policy_loss_improves_at_least_5pct": relative >= config.minimum_relative_loss_improvement,
        "mean_reciprocal_rank_improves": candidate_aggregate["mean_reciprocal_rank"] > parent_aggregate["mean_reciprocal_rank"],
        "paired_loss_ci_below_zero": comparison["ci95_high"] < 0.0,
        "better_on_at_least_65pct_games": comparison["candidate_better_fraction"] >= config.minimum_game_win_fraction,
        "beats_uniform_policy": candidate_aggregate["policy_loss"] < candidate_aggregate["uniform_policy_loss"],
    }


def train_and_gate_entity_scale(config: EntityScaleConfig, replay_path: Path, corpus_summary: dict[str, Any]) -> dict[str, Any]:
    dataset = EntityActionReplayDataset(replay_path)
    train_indices, validation_indices, confirmation_indices, split_seeds = split_game_heldout_indices(
        dataset, config.train_games, config.validation_games, config.confirmation_games, config.split_seed
    )
    subsets = {
        "train": Subset(dataset, train_indices),
        "validation": Subset(dataset, validation_indices),
        "confirmation": Subset(dataset, confirmation_indices),
    }
    (config.output_dir / "game_splits.json").write_text(json.dumps(split_seeds, indent=2), encoding="utf-8")
    parent = _load_checkpoint_model(config.parent_checkpoint, config.device)
    parent_validation = evaluate_detailed(parent, subsets["validation"], config.batch_size, config.device)
    candidates: dict[str, Any] = {}
    trained_models: dict[str, CanonicalEntityActionScorerModel] = {}
    for offset, (name, model_config) in enumerate(_candidate_configs().items()):
        torch.manual_seed(config.init_seed + offset)
        model = CanonicalEntityActionScorerModel(model_config)
        started = time.perf_counter()
        artifacts = fit_entity_action_scorer(
            create_entity_action_replay_dataloader(subsets["train"], config.batch_size, shuffle=True),
            create_entity_action_replay_dataloader(subsets["validation"], config.batch_size, shuffle=False),
            model=model, epochs=config.epochs, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, device=config.device, selection_metric="policy_loss",
        )
        if artifacts.best_validation_state_dict is None:
            raise RuntimeError("Validation training did not produce a best checkpoint.")
        model.load_state_dict(artifacts.best_validation_state_dict)
        model.to(config.device).eval()
        checkpoint_path = config.output_dir / f"canonical_entity_action_scorer_{name}_best.pt"
        torch.save({
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": artifacts.best_validation_state_dict,
            "model_config": asdict(model_config),
            "train_config": {**_training_config_payload(config), "candidate": name},
        }, checkpoint_path)
        validation = evaluate_detailed(model, subsets["validation"], config.batch_size, config.device)
        comparison = _paired_loss_comparison(validation, parent_validation, config.bootstrap_samples, config.split_seed + offset)
        candidates[name] = {
            "checkpoint": str(checkpoint_path), "model_config": asdict(model_config),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": artifacts.best_epoch_index,
            "elapsed_seconds": time.perf_counter() - started,
            "train_history": [asdict(item) for item in artifacts.train_metrics],
            "validation_history": [asdict(item) for item in artifacts.validation_metrics],
            "validation": validation,
            "validation_vs_parent": comparison,
            "validation_gates": _offline_gates(config, validation, parent_validation, comparison),
        }
        trained_models[name] = model
        print(f"[entity-scale:train] {name} best_epoch={artifacts.best_epoch_index} validation_loss={validation['aggregate']['policy_loss']:.4f} elapsed={candidates[name]['elapsed_seconds']:.1f}s", flush=True)

    selected_name = min(candidates, key=lambda name: candidates[name]["validation"]["aggregate"]["policy_loss"])
    parent_confirmation = evaluate_detailed(parent, subsets["confirmation"], config.batch_size, config.device)
    for offset, name in enumerate(candidates):
        confirmation = evaluate_detailed(trained_models[name], subsets["confirmation"], config.batch_size, config.device)
        comparison = _paired_loss_comparison(confirmation, parent_confirmation, config.bootstrap_samples, config.split_seed + 100 + offset)
        candidates[name]["confirmation"] = confirmation
        candidates[name]["confirmation_vs_parent"] = comparison
        candidates[name]["confirmation_gates"] = _offline_gates(config, confirmation, parent_confirmation, comparison)
    selected_gates = candidates[selected_name]["confirmation_gates"]
    payload = {
        "experiment": "canonical_entity_supervised_scale_v1",
        "config": _training_config_payload(config),
        "corpus": corpus_summary,
        "split_seeds": split_seeds,
        "split_positions": {name: len(subset) for name, subset in subsets.items()},
        "parent": {"checkpoint": str(config.parent_checkpoint), "validation": parent_validation, "confirmation": parent_confirmation},
        "candidates": candidates,
        "selected_by_validation": selected_name,
        "selected_checkpoint": candidates[selected_name]["checkpoint"],
        "selected_confirmation_gates": selected_gates,
        "offline_gate_passed": all(selected_gates.values()),
        "match_gate_eligible": all(selected_gates.values()),
    }
    metrics_path = config.output_dir / "entity_supervised_scale_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected_name, "confirmation_gates": selected_gates, "match_gate_eligible": payload["match_gate_eligible"]}, indent=2), flush=True)
    return payload


def _training_config_payload(config: EntityScaleConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["parent_checkpoint"] = str(config.parent_checkpoint)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    _validate_config(config)
    if args.skip_generation:
        replay_path = config.output_dir / "replays.jsonl"
        summary_path = config.output_dir / "summary.json"
        if not replay_path.exists() or not summary_path.exists():
            raise FileNotFoundError("--skip-generation requires existing replays.jsonl and summary.json.")
        corpus_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        replay_path, corpus_summary = generate_clean_scale_corpus(config)
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
    train_and_gate_entity_scale(config, replay_path, corpus_summary)


if __name__ == "__main__":
    main()

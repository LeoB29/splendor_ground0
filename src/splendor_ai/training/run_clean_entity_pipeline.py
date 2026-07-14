"""Generate a clean legal corpus and train the canonical entity scorer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from splendor_ai.bots import GreedyHeuristicBot, ShallowSearchBot
from splendor_ai.training.corpus import (
    generate_replay_corpus,
    write_replay_corpus,
    write_replay_corpus_manifest,
)
from splendor_ai.training.entity_action_supervised import (
    EntityActionReplayDataset,
    create_entity_action_replay_dataloader,
    fit_entity_action_scorer,
    split_entity_action_replay_dataset,
)
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed-start", type=int, default=70_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--entity-hidden-size", type=int, default=96)
    parser.add_argument("--action-hidden-size", type=int, default=96)
    parser.add_argument("--scorer-hidden-size", type=int, default=128)
    parser.add_argument("--entity-layers", type=int, default=2)
    parser.add_argument("--action-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-size", type=int, default=192)
    parser.add_argument("--id-embedding-size", type=int, default=16)
    parser.add_argument(
        "--teacher",
        choices=("greedy", "shallow"),
        default="greedy",
        help="Use the fast rule-based greedy corpus for a learning gate, or the slower legal shallow-search tranche.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.games <= 1:
        raise ValueError("--games must be at least 2 for a held-out game split.")
    output_dir = Path(args.output_dir)
    shallow_factory = lambda: ShallowSearchBot(
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
    opponent_factory = shallow_factory if args.teacher == "shallow" else lambda: GreedyHeuristicBot(seed=23)
    games, summary = generate_replay_corpus(
        games=args.games,
        bot_seat_0_factory=lambda: GreedyHeuristicBot(seed=11),
        bot_seat_1_factory=opponent_factory,
        seed_start=args.seed_start,
        swap_seats=True,
        include_state_snapshots=True,
        progress_callback=lambda completed, total, _game, elapsed: print(
            f"[clean-entity] corpus {completed}/{total} elapsed={elapsed:.1f}s"
        ) if completed == total or completed % 20 == 0 else None,
    )
    replay_path, _summary_path = write_replay_corpus(
        output_dir,
        games,
        summary,
        metadata={
            "contract": "canonical_public_belief_v1",
            "policy_teacher": "rule_based_only",
            "learned_checkpoint_used": False,
            "pairing": (
                "GreedyHeuristicBot_vs_information_legal_ShallowSearchBot"
                if args.teacher == "shallow"
                else "GreedyHeuristicBot_vs_GreedyHeuristicBot"
            ),
        },
    )
    write_replay_corpus_manifest(
        output_dir,
        games,
        summary,
        metadata={
            "contract": "canonical_public_belief_v1",
            "policy_teacher": "rule_based_only",
            "learned_checkpoint_used": False,
        },
        source_type="clean_rule_based_legal_replay",
    )
    if summary.stalled_games or summary.timed_out_games:
        raise RuntimeError("Clean corpus gate failed: stalled or timed-out games were generated.")

    dataset = EntityActionReplayDataset(replay_path)
    train_dataset, validation_dataset = split_entity_action_replay_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        seed=args.validation_seed,
    )
    model = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=args.entity_hidden_size,
            action_hidden_size=args.action_hidden_size,
            scorer_hidden_size=args.scorer_hidden_size,
            entity_layers=args.entity_layers,
            action_layers=args.action_layers,
            attention_heads=args.attention_heads,
            attention_feedforward_size=args.feedforward_size,
            id_embedding_size=args.id_embedding_size,
        )
    )
    artifacts = fit_entity_action_scorer(
        create_entity_action_replay_dataloader(train_dataset, args.batch_size, shuffle=True),
        None if validation_dataset is None else create_entity_action_replay_dataloader(validation_dataset, args.batch_size, shuffle=False),
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    final_checkpoint = output_dir / "canonical_entity_action_scorer.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": artifacts.model.state_dict(),
            "model_config": asdict(artifacts.model.config),
            "train_config": vars(args),
        },
        final_checkpoint,
    )
    if artifacts.best_validation_state_dict is not None:
        torch.save(
            {
                "model_kind": "canonical_entity_action_scorer",
                "model_state_dict": artifacts.best_validation_state_dict,
                "model_config": asdict(artifacts.model.config),
                "train_config": vars(args),
            },
            output_dir / "canonical_entity_action_scorer_best.pt",
        )
    final_validation = artifacts.validation_metrics[-1] if artifacts.validation_metrics else None
    gate_passed = bool(
        final_validation is not None
        and final_validation.policy_loss < final_validation.uniform_policy_loss
    )
    metrics_payload = {
        "train": [asdict(metric) for metric in artifacts.train_metrics],
        "validation": [asdict(metric) for metric in artifacts.validation_metrics],
        "best_epoch_index": artifacts.best_epoch_index,
        "gates": {
            "no_stalls_or_timeouts": True,
            "held_out_policy_beats_uniform": gate_passed,
        },
    }
    (output_dir / "canonical_entity_action_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics_payload["gates"], indent=2))


if __name__ == "__main__":
    main()

"""Preregistered action-preference generation anchored on champion 035."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from splendor_ai.eval.parallel_paired import (
    PairedCheckpointConfig,
    run_paired_checkpoint_benchmark,
)
from splendor_ai.training.run_action_preference_corpus import (
    build_action_preference_corpus,
)
from splendor_ai.training.run_action_preference_training import (
    ActionPreferenceTrainingConfig,
    train_and_gate_action_preferences,
)
from splendor_ai.training.run_actor_movement_audit import (
    ActorMovementAuditConfig,
    _measure_stream,
)
from splendor_ai.training.run_league_generation import (
    LeagueGenerationConfig,
    evaluate_direct_gate,
    run_broad_gate,
)
from splendor_ai.training.run_trajectory_policy_gradient import _load_attention_model


@dataclass(frozen=True, slots=True)
class ActionPreferenceGenerationConfig:
    output_dir: Path
    data_dir: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    broad_replay: Path
    source_games: int = 2_880
    collection_seed_start: int = 361_000
    continuation_seed_start: int = 35_000_000
    determinizations: int = 3
    continuations_per_determinization: int = 8
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_additional_turns: int = 120
    score_margin_scale: float = 20.0
    minimum_effect_magnitude: float = 0.01
    resolved_effect_magnitude: float = 0.025
    maximum_continuation_cutoff_fraction: float = 0.01
    minimum_split_sign_states: int = 20
    minimum_split_sign_agreement: float = 0.65
    minimum_resolved_fraction: float = 0.10
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    kl_weight: float = 1.0
    max_grad_norm: float = 1.0
    target_validation_kl: float = 0.004
    bisection_steps: int = 12
    direct_pairs: int = 200
    direct_seed_start: int = 364_000
    broad_pairs: int = 50
    broad_016_seed_start: int = 365_000
    broad_shallow_seed_start: int = 366_000
    broad_023_seed_start: int = 367_000
    workers: int = 4
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    device: str = "cpu"


def run_action_preference_generation(config: ActionPreferenceGenerationConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = config.output_dir / "protocol.json"
    protocol = {
        "format": "action_preference_generation_protocol_v1",
        "hypothesis": (
            "Matched continuation preferences between 035's distinct canonical top-two "
            "actions can improve local action ranking without materially drifting its policy."
        ),
        "selection_is_outcome_blind": True,
        "one_state_per_source_game": True,
        "split_counts": {"train": 1536, "validation": 384, "confirmation": 384},
        "confirmation_touched_after_selection_only": True,
        "promotion_order": ["corpus", "offline", "movement", "direct", "broad"],
        "config": _config_payload(config),
    }
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing action-preference protocol does not match.")
    else:
        _atomic_write(protocol_path, protocol)

    corpus = build_action_preference_corpus(config)
    if not corpus["passed"]:
        return _finalize(config, "corpus_gate_failed", corpus=corpus)
    paths = {key: Path(value) for key, value in corpus["preference_paths"].items()}
    offline = train_and_gate_action_preferences(
        ActionPreferenceTrainingConfig(
            parent_checkpoint=config.parent_checkpoint,
            train_replay=paths["train"],
            validation_replay=paths["validation"],
            confirmation_replay=paths["confirmation"],
            output_dir=config.output_dir / "candidate",
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            kl_weight=config.kl_weight,
            max_grad_norm=config.max_grad_norm,
            target_validation_kl=config.target_validation_kl,
            bisection_steps=config.bisection_steps,
            bootstrap_samples=config.bootstrap_samples,
            device=config.device,
        )
    )
    if not offline["passed"]:
        return _finalize(config, "offline_gate_failed", corpus=corpus, offline=offline)
    candidate = Path(offline["candidate_checkpoint"])
    movement = audit_preference_movement(config, candidate)
    if not movement["passed"]:
        return _finalize(
            config, "movement_gate_failed", corpus=corpus, offline=offline, movement=movement
        )

    direct = run_paired_checkpoint_benchmark(
        PairedCheckpointConfig(
            candidate_checkpoint=candidate,
            baseline_checkpoint=config.parent_checkpoint,
            output_path=config.output_dir / "direct_vs_035.json",
            checkpoint_path=config.output_dir / "direct_vs_035.progress.json",
            pairs=config.direct_pairs,
            seed_start=config.direct_seed_start,
            workers=config.workers,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
            bootstrap_samples=config.bootstrap_samples,
            device=config.device,
        )
    )
    direct_gates = evaluate_direct_gate(direct)
    broad = None
    if all(direct_gates.values()):
        broad = run_broad_gate(_league_config(config), candidate)
    decision = (
        "promote_action_preference_candidate"
        if all(direct_gates.values()) and broad is not None and broad["passed"]
        else "direct_gate_failed"
        if not all(direct_gates.values())
        else "broad_gate_failed"
    )
    return _finalize(
        config,
        decision,
        corpus=corpus,
        offline=offline,
        movement=movement,
        direct=direct,
        direct_gates=direct_gates,
        broad=broad,
    )


def audit_preference_movement(config, candidate_checkpoint: Path) -> dict[str, Any]:
    path = config.output_dir / "movement_gate.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    parent, _ = _load_attention_model(config.parent_checkpoint, config.device)
    candidate, _ = _load_attention_model(candidate_checkpoint, config.device)
    audit = ActorMovementAuditConfig(
        parent_checkpoint=config.parent_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        training_metrics_path=config.output_dir / "candidate" / "offline_preference_gate.json",
        on_policy_replay_paths=(config.data_dir / "source_replays.jsonl",),
        broad_replay_paths=(config.broad_replay,),
        output_path=path,
        max_rows_per_stream=5_000,
        batch_size=128,
        min_on_policy_change_rate=0.01,
        min_broad_change_rate=0.01,
        max_change_rate=0.05,
        max_mean_exploration_kl=0.008,
        device=config.device,
    )
    on_policy = _measure_stream(audit.on_policy_replay_paths, parent, candidate, audit, 35)
    broad = _measure_stream(audit.broad_replay_paths, parent, candidate, audit, 36)
    gates = {
        "on_policy_movement_floor": float(on_policy["top_action_change_rate"]) >= 0.01,
        "broad_movement_floor": float(broad["top_action_change_rate"]) >= 0.01,
        "on_policy_movement_ceiling": float(on_policy["top_action_change_rate"]) <= 0.05,
        "broad_movement_ceiling": float(broad["top_action_change_rate"]) <= 0.05,
        "on_policy_kl_ceiling": float(on_policy["raw_policy_kl"]["mean"]) <= 0.008,
        "broad_kl_ceiling": float(broad["raw_policy_kl"]["mean"]) <= 0.008,
    }
    payload = {
        "format": "action_preference_movement_gate_v1",
        "on_policy": on_policy,
        "broad": broad,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_write(path, payload)
    return payload


def _league_config(config):
    return LeagueGenerationConfig(
        output_dir=config.output_dir,
        data_dir=config.data_dir,
        parent_checkpoint=config.parent_checkpoint,
        checkpoint_016=config.checkpoint_016,
        checkpoint_023=config.checkpoint_023,
        calibration_replay=config.broad_replay,
        broad_replay=config.broad_replay,
        direct_pairs=config.direct_pairs,
        direct_seed_start=config.direct_seed_start,
        broad_pairs=config.broad_pairs,
        broad_016_seed_start=config.broad_016_seed_start,
        broad_shallow_seed_start=config.broad_shallow_seed_start,
        broad_023_seed_start=config.broad_023_seed_start,
        workers=config.workers,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
        bootstrap_samples=config.bootstrap_samples,
        device=config.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/action_preference_generation_001")
    parser.add_argument("--data-dir", default="outputs/action_preference_generation_001/data")
    parser.add_argument(
        "--parent-checkpoint",
        default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt",
    )
    parser.add_argument(
        "--checkpoint-016",
        default="outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt",
    )
    parser.add_argument(
        "--checkpoint-023",
        default="outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt",
    )
    parser.add_argument(
        "--broad-replay", default="outputs/canonical_entity_supervised_scale_020/replays.jsonl"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_action_preference_generation(
        ActionPreferenceGenerationConfig(
            output_dir=Path(args.output_dir),
            data_dir=Path(args.data_dir),
            parent_checkpoint=Path(args.parent_checkpoint),
            checkpoint_016=Path(args.checkpoint_016),
            checkpoint_023=Path(args.checkpoint_023),
            broad_replay=Path(args.broad_replay),
            workers=args.workers,
            device=args.device,
        )
    )


def _finalize(config, decision: str, **stages):
    payload = {
        "gate": "action_preference_generation_001",
        "decision": decision,
        "promoted": decision == "promote_action_preference_candidate",
        **stages,
    }
    _atomic_write(config.output_dir / "gate_final.json", payload)
    print(json.dumps({"decision": decision, "promoted": payload["promoted"]}, indent=2))
    return payload


def _config_payload(config) -> dict[str, Any]:
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _validate(config):
    if config.source_games != 2_880:
        raise ValueError("The frozen split protocol requires exactly 2,880 source games.")
    if config.source_games % 24 != 0:
        raise ValueError("Source games must balance over opponent, seat, and phase.")
    for path in (
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
        config.broad_replay,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()

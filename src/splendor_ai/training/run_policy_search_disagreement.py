"""CLI entry point for comparing policy checkpoints with shallow search on replay states."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from splendor_ai.bots import CheckpointLegalActionPriorScorer, ShallowSearchBot
from splendor_ai.encoding import (
    ActionCodec,
    encode_legal_action_features,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    LegalActionScorerConfig,
    LegalActionScorerModel,
    LegalActionSetScorerConfig,
    LegalActionSetScorerModel,
    PolicyValueMLP,
    PolicyValueModelConfig,
    masked_legal_action_scores,
    masked_policy_logits,
)
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action


@dataclass(frozen=True, slots=True)
class ModelActionScore:
    action: Action
    action_index: int
    score: float


@dataclass(frozen=True, slots=True)
class DisagreementConfig:
    checkpoint: Path
    model_kind: str
    replay_paths: tuple[Path, ...]
    output_path: Path
    device: str = "cpu"
    max_rows: int | None = None
    log_every: int = 500
    search_depth: int = 2
    search_max_branching: int = 10
    search_buy_branching: int = 6
    search_reserve_branching: int = 3
    search_take_branching: int = 3
    search_policy_prior_checkpoint_path: Path | None = None
    search_policy_prior_device: str = "cpu"
    search_policy_prior_weight: float = 0.0
    seed: int = 0
    player_id_filter: tuple[int, ...] | None = None
    phase_filter: tuple[str, ...] | None = None
    min_turn_index: int | None = None
    examples_limit: int = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze policy-vs-search action disagreement on snapshot-enabled replay rows."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-kind",
        choices=(
            "flat_policy",
            "legal_action_scorer",
            "legal_action_attention_scorer",
            "legal_action_set_scorer",
        ),
        default="legal_action_scorer",
    )
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Snapshot-enabled replay JSONL path. Repeat to concatenate corpora.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--search-policy-prior-checkpoint", default=None)
    parser.add_argument("--search-policy-prior-device", default="cpu")
    parser.add_argument("--search-policy-prior-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--player-id-filter",
        nargs="*",
        type=int,
        choices=(0, 1),
        default=None,
    )
    parser.add_argument(
        "--phase-filter",
        nargs="*",
        choices=("early", "mid", "late"),
        default=None,
    )
    parser.add_argument("--min-turn-index", type=int, default=None)
    parser.add_argument(
        "--examples-limit",
        type=int,
        default=20,
        help="Number of high-search-margin disagreement examples to store.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze_policy_search_disagreement(
        DisagreementConfig(
            checkpoint=Path(args.checkpoint),
            model_kind=args.model_kind,
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            device=args.device,
            max_rows=args.max_rows,
            log_every=args.log_every,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            search_policy_prior_checkpoint_path=(
                Path(args.search_policy_prior_checkpoint)
                if args.search_policy_prior_checkpoint is not None
                else None
            ),
            search_policy_prior_device=args.search_policy_prior_device,
            search_policy_prior_weight=args.search_policy_prior_weight,
            seed=args.seed,
            player_id_filter=(
                tuple(args.player_id_filter)
                if args.player_id_filter is not None and len(args.player_id_filter) > 0
                else None
            ),
            phase_filter=(
                tuple(args.phase_filter)
                if args.phase_filter is not None and len(args.phase_filter) > 0
                else None
            ),
            min_turn_index=args.min_turn_index,
            examples_limit=args.examples_limit,
        )
    )
    print(
        "Policy/search disagreement:"
        f" analyzed={summary['rows_analyzed']}"
        f" agreement_rate={summary['overall']['agreement_rate']:.4f}"
        f" disagreements={summary['overall']['disagreements']}"
        f" missing_model_search_value={summary['overall']['model_action_missing_from_search_candidates']}"
        f" output={args.output_path}"
    )


def analyze_policy_search_disagreement(config: DisagreementConfig) -> dict[str, Any]:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.examples_limit < 0:
        raise ValueError("examples_limit must be non-negative.")
    if config.search_policy_prior_weight < 0.0:
        raise ValueError("search_policy_prior_weight must be non-negative.")
    if (
        config.search_policy_prior_checkpoint_path is None
        and config.search_policy_prior_weight > 0.0
    ):
        raise ValueError(
            "search_policy_prior_checkpoint_path is required when search_policy_prior_weight is positive."
        )
    unknown_phases = set(config.phase_filter or ()) - {"early", "mid", "late"}
    if unknown_phases:
        raise ValueError(f"Unknown phases: {sorted(unknown_phases)}")

    env = SplendorEnv()
    codec = ActionCodec()
    model_scorer = _load_policy_scorer(config)
    search_action_prior = _build_search_action_prior(config)
    counters = _empty_counters()
    examples: list[dict[str, Any]] = []
    rows_seen = 0
    rows_skipped_by_player_id = 0
    rows_skipped_by_phase = 0
    rows_skipped_by_turn = 0
    rows_missing_snapshot = 0
    rows_without_legal_actions = 0
    rows_without_search_candidate = 0

    for replay_path in config.replay_paths:
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if config.max_rows is not None and rows_seen >= config.max_rows:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                rows_seen += 1
                payload = json.loads(stripped)
                player_id = int(payload.get("player_id", -1))
                if config.player_id_filter is not None and player_id not in config.player_id_filter:
                    rows_skipped_by_player_id += 1
                    continue
                turn_index = int(payload.get("turn_index", 0))
                if config.min_turn_index is not None and turn_index < config.min_turn_index:
                    rows_skipped_by_turn += 1
                    continue
                phase = _phase_for_turn(turn_index)
                if config.phase_filter is not None and phase not in config.phase_filter:
                    rows_skipped_by_phase += 1
                    continue
                snapshot = payload.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    rows_missing_snapshot += 1
                    continue
                state = deserialize_state_snapshot(snapshot)
                legal_actions = env.legal_actions(state)
                if not legal_actions:
                    rows_without_legal_actions += 1
                    continue
                search_bot = _build_search_bot(
                    config,
                    row_offset=rows_seen,
                    action_prior=search_action_prior,
                )
                search_scores = search_bot.rank_actions(env, state, legal_actions)
                if not search_scores:
                    rows_without_search_candidate += 1
                    continue

                model_scores = model_scorer.score_actions(state, legal_actions)
                _record_row(
                    counters=counters,
                    examples=examples,
                    examples_limit=config.examples_limit,
                    codec=codec,
                    state=state,
                    payload=payload,
                    replay_path=replay_path,
                    model_scores=model_scores,
                    search_scores=search_scores,
                )

                if config.log_every > 0 and rows_seen % config.log_every == 0:
                    analyzed = counters["rows_analyzed"]
                    agreement_rate = (
                        counters["agreements"] / analyzed
                        if analyzed
                        else math.nan
                    )
                    print(
                        f"[disagreement] rows_seen={rows_seen}"
                        f" analyzed={analyzed}"
                        f" agreement_rate={agreement_rate:.4f}"
                        f" missing_snapshot={rows_missing_snapshot}"
                    )
        if config.max_rows is not None and rows_seen >= config.max_rows:
            break

    summary = _build_summary(
        config=config,
        counters=counters,
        rows_seen=rows_seen,
        rows_skipped_by_player_id=rows_skipped_by_player_id,
        rows_skipped_by_phase=rows_skipped_by_phase,
        rows_skipped_by_turn=rows_skipped_by_turn,
        rows_missing_snapshot=rows_missing_snapshot,
        rows_without_legal_actions=rows_without_legal_actions,
        rows_without_search_candidate=rows_without_search_candidate,
        examples=examples,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class _PolicyScorer:
    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> tuple[ModelActionScore, ...]:
        raise NotImplementedError


class _FlatPolicyScorer(_PolicyScorer):
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        self._device = device
        self._codec = ActionCodec()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self._model = PolicyValueMLP(PolicyValueModelConfig(**checkpoint.get("model_config", {})))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()

    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> tuple[ModelActionScore, ...]:
        observation = encode_public_observation_tensor(state, state.current_player)
        observation_vector = fit_feature_vector(
            observation.vector,
            self._model.config.observation_size,
        )
        observation_tensor = torch.tensor(
            [observation_vector],
            dtype=torch.float32,
            device=self._device,
        )
        mask = torch.zeros(
            (1, self._codec.action_space_size),
            dtype=torch.bool,
            device=self._device,
        )
        action_indices = [self._codec.encode(state, action) for action in legal_actions]
        for action_index in action_indices:
            mask[0, action_index] = True
        with torch.no_grad():
            logits, _value = self._model(observation_tensor)
            masked_logits = masked_policy_logits(logits, mask)[0].detach().cpu()
        scores = [
            ModelActionScore(
                action=action,
                action_index=action_index,
                score=float(masked_logits[action_index].item()),
            )
            for action, action_index in zip(legal_actions, action_indices, strict=True)
        ]
        return tuple(sorted(scores, key=lambda item: item.score, reverse=True))


class _LegalActionScorer(_PolicyScorer):
    def __init__(self, checkpoint_path: Path, device: str, model_kind: str = "legal_action_scorer") -> None:
        self._device = device
        self._codec = ActionCodec()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if model_kind == "legal_action_attention_scorer":
            self._model = LegalActionAttentionScorerModel(
                LegalActionAttentionScorerConfig(**checkpoint.get("model_config", {}))
            )
        elif model_kind == "legal_action_set_scorer":
            self._model = LegalActionSetScorerModel(
                LegalActionSetScorerConfig(**checkpoint.get("model_config", {}))
            )
        else:
            self._model = LegalActionScorerModel(
                LegalActionScorerConfig(**checkpoint.get("model_config", {}))
            )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()

    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> tuple[ModelActionScore, ...]:
        observation = encode_public_observation_tensor(state, state.current_player)
        observation_vector = fit_feature_vector(
            observation.vector,
            self._model.config.observation_size,
        )
        action_features = encode_legal_action_features(
            state=state,
            legal_actions=legal_actions,
            player_id=state.current_player,
        )
        action_feature_vectors = [
            fit_feature_vector(features.vector, self._model.config.action_feature_size)
            for features in action_features
        ]
        observation_tensor = torch.tensor(
            [observation_vector],
            dtype=torch.float32,
            device=self._device,
        )
        action_feature_tensor = torch.tensor(
            [action_feature_vectors],
            dtype=torch.float32,
            device=self._device,
        )
        legal_action_mask = torch.ones(
            (1, len(legal_actions)),
            dtype=torch.bool,
            device=self._device,
        )
        action_indices = [self._codec.encode(state, action) for action in legal_actions]
        with torch.no_grad():
            action_scores, _value = self._model(
                observation_tensor,
                action_feature_tensor,
                legal_action_mask,
            )
            masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)[0].detach().cpu()
        scores = [
            ModelActionScore(
                action=action,
                action_index=action_index,
                score=float(masked_scores[row_index].item()),
            )
            for row_index, (action, action_index) in enumerate(zip(legal_actions, action_indices, strict=True))
        ]
        return tuple(sorted(scores, key=lambda item: item.score, reverse=True))


def _load_policy_scorer(config: DisagreementConfig) -> _PolicyScorer:
    if config.model_kind == "flat_policy":
        return _FlatPolicyScorer(config.checkpoint, config.device)
    if config.model_kind in {
        "legal_action_scorer",
        "legal_action_attention_scorer",
        "legal_action_set_scorer",
    }:
        return _LegalActionScorer(config.checkpoint, config.device, config.model_kind)
    raise ValueError(f"Unsupported model kind: {config.model_kind}")


def _build_search_action_prior(
    config: DisagreementConfig,
) -> CheckpointLegalActionPriorScorer | None:
    if (
        config.search_policy_prior_checkpoint_path is not None
        and config.search_policy_prior_weight > 0.0
    ):
        return CheckpointLegalActionPriorScorer(
            checkpoint_path=config.search_policy_prior_checkpoint_path,
            device=config.search_policy_prior_device,
        )
    return None


def _build_search_bot(
    config: DisagreementConfig,
    row_offset: int,
    action_prior: CheckpointLegalActionPriorScorer | None = None,
) -> ShallowSearchBot:
    return ShallowSearchBot(
        depth=config.search_depth,
        max_branching=config.search_max_branching,
        max_buy_actions=config.search_buy_branching,
        max_reserve_actions=config.search_reserve_branching,
        max_take_actions=config.search_take_branching,
        seed=config.seed + row_offset,
        action_prior=action_prior,
        action_prior_weight=(
            config.search_policy_prior_weight if action_prior is not None else 0.0
        ),
    )


def _empty_counters() -> dict[str, Any]:
    return {
        "rows_analyzed": 0,
        "agreements": 0,
        "disagreements": 0,
        "known_search_margins": [],
        "missing_model_search_value": 0,
        "by_model_action_type": {},
        "by_search_action_type": {},
        "by_phase": {},
        "by_seat": {},
        "by_seat_phase": {},
        "by_score_margin": {},
        "by_seat_model_action_type": {},
        "by_seat_search_action_type": {},
        "action_type_matrix": {},
    }


def _record_row(
    counters: dict[str, Any],
    examples: list[dict[str, Any]],
    examples_limit: int,
    codec: ActionCodec,
    state: SplendorState,
    payload: dict[str, Any],
    replay_path: Path,
    model_scores: tuple[ModelActionScore, ...],
    search_scores: tuple[Any, ...],
) -> None:
    model_top = model_scores[0]
    search_top = search_scores[0]
    search_top_index = codec.encode(state, search_top.action)
    agreement = model_top.action_index == search_top_index
    model_action_type = model_top.action.action_type.name
    search_action_type = search_top.action.action_type.name
    phase = _phase_for_turn(int(payload.get("turn_index", 0)))
    player_id = int(payload.get("player_id", state.current_player))
    seat = f"seat{player_id}"
    score_margin = state.players[player_id].score - state.players[1 - player_id].score
    search_value_by_index = {
        codec.encode(state, score.action): score.value
        for score in search_scores
    }
    model_search_value = search_value_by_index.get(model_top.action_index)
    search_margin = None if model_search_value is None else search_top.value - model_search_value

    counters["rows_analyzed"] += 1
    if agreement:
        counters["agreements"] += 1
    else:
        counters["disagreements"] += 1
        if model_search_value is None:
            counters["missing_model_search_value"] += 1
        else:
            counters["known_search_margins"].append(search_margin)
            _maybe_store_example(
                examples=examples,
                examples_limit=examples_limit,
                search_margin=float(search_margin),
                replay_path=replay_path,
                payload=payload,
                state=state,
                model_top=model_top,
                search_top=search_top,
                search_top_index=search_top_index,
                model_search_value=float(model_search_value),
            )

    _record_bucket(counters["by_model_action_type"], model_action_type, agreement, search_margin)
    _record_bucket(counters["by_search_action_type"], search_action_type, agreement, search_margin)
    _record_bucket(counters["by_phase"], phase, agreement, search_margin)
    _record_bucket(counters["by_seat"], seat, agreement, search_margin)
    _record_bucket(counters["by_seat_phase"], f"{seat}:{phase}", agreement, search_margin)
    _record_bucket(counters["by_score_margin"], _score_margin_bucket(score_margin), agreement, search_margin)
    _record_bucket(
        counters["by_seat_model_action_type"],
        f"{seat}:{model_action_type}",
        agreement,
        search_margin,
    )
    _record_bucket(
        counters["by_seat_search_action_type"],
        f"{seat}:{search_action_type}",
        agreement,
        search_margin,
    )
    matrix = counters["action_type_matrix"]
    matrix.setdefault(model_action_type, {})
    matrix[model_action_type][search_action_type] = matrix[model_action_type].get(search_action_type, 0) + 1


def _record_bucket(
    buckets: dict[str, dict[str, Any]],
    key: str,
    agreement: bool,
    search_margin: float | None,
) -> None:
    bucket = buckets.setdefault(
        key,
        {
            "count": 0,
            "agreements": 0,
            "disagreements": 0,
            "known_search_margins": [],
            "missing_model_search_value": 0,
        },
    )
    bucket["count"] += 1
    if agreement:
        bucket["agreements"] += 1
    else:
        bucket["disagreements"] += 1
        if search_margin is None:
            bucket["missing_model_search_value"] += 1
        else:
            bucket["known_search_margins"].append(float(search_margin))


def _maybe_store_example(
    examples: list[dict[str, Any]],
    examples_limit: int,
    search_margin: float,
    replay_path: Path,
    payload: dict[str, Any],
    state: SplendorState,
    model_top: ModelActionScore,
    search_top: Any,
    search_top_index: int,
    model_search_value: float,
) -> None:
    if examples_limit == 0:
        return
    example = {
        "search_margin": search_margin,
        "replay_path": str(replay_path),
        "game_seed": payload.get("game_seed"),
        "turn_index": payload.get("turn_index"),
        "player_id": payload.get("player_id"),
        "seat": f"seat{payload.get('player_id')}",
        "phase": _phase_for_turn(int(payload.get("turn_index", 0))),
        "scores": [
            state.players[0].score,
            state.players[1].score,
        ],
        "score_margin": state.players[int(payload.get("player_id", state.current_player))].score
        - state.players[1 - int(payload.get("player_id", state.current_player))].score,
        "model_action_index": model_top.action_index,
        "model_action_type": model_top.action.action_type.name,
        "model_score": model_top.score,
        "model_search_value": model_search_value,
        "model_action_payload": serialize_action(model_top.action),
        "search_action_index": search_top_index,
        "search_action_type": search_top.action.action_type.name,
        "search_value": search_top.value,
        "search_heuristic_score": search_top.heuristic_score,
        "search_action_payload": serialize_action(search_top.action),
        "replay_action_index": payload.get("action_index"),
        "replay_action_payload": payload.get("action_payload"),
    }
    examples.append(example)
    examples.sort(key=lambda item: float(item["search_margin"]), reverse=True)
    del examples[examples_limit:]


def _build_summary(
    config: DisagreementConfig,
    counters: dict[str, Any],
    rows_seen: int,
    rows_skipped_by_player_id: int,
    rows_skipped_by_phase: int,
    rows_skipped_by_turn: int,
    rows_missing_snapshot: int,
    rows_without_legal_actions: int,
    rows_without_search_candidate: int,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_analyzed = counters["rows_analyzed"]
    return {
        "checkpoint": str(config.checkpoint),
        "model_kind": config.model_kind,
        "replay_paths": [str(path) for path in config.replay_paths],
        "rows_seen": rows_seen,
        "rows_analyzed": rows_analyzed,
        "rows_skipped_by_player_id": rows_skipped_by_player_id,
        "rows_skipped_by_phase": rows_skipped_by_phase,
        "rows_skipped_by_turn": rows_skipped_by_turn,
        "rows_missing_snapshot": rows_missing_snapshot,
        "rows_without_legal_actions": rows_without_legal_actions,
        "rows_without_search_candidate": rows_without_search_candidate,
        "overall": {
            "agreements": counters["agreements"],
            "disagreements": counters["disagreements"],
            "agreement_rate": _rate(counters["agreements"], rows_analyzed),
            "known_search_margin_mean_on_disagreement": _mean(counters["known_search_margins"]),
            "known_search_margin_max_on_disagreement": _max_or_none(counters["known_search_margins"]),
            "model_action_missing_from_search_candidates": counters["missing_model_search_value"],
        },
        "by_model_action_type": _finalize_buckets(counters["by_model_action_type"]),
        "by_search_action_type": _finalize_buckets(counters["by_search_action_type"]),
        "by_phase": _finalize_buckets(counters["by_phase"]),
        "by_seat": _finalize_buckets(counters["by_seat"]),
        "by_seat_phase": _finalize_buckets(counters["by_seat_phase"]),
        "by_score_margin": _finalize_buckets(counters["by_score_margin"]),
        "by_seat_model_action_type": _finalize_buckets(counters["by_seat_model_action_type"]),
        "by_seat_search_action_type": _finalize_buckets(counters["by_seat_search_action_type"]),
        "action_type_matrix": counters["action_type_matrix"],
        "examples": examples,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
            "search_policy_prior_checkpoint_path": (
                str(config.search_policy_prior_checkpoint_path)
                if config.search_policy_prior_checkpoint_path is not None
                else None
            ),
        },
    }


def _finalize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized = {}
    for key in sorted(buckets):
        bucket = buckets[key]
        finalized[key] = {
            "count": bucket["count"],
            "agreements": bucket["agreements"],
            "disagreements": bucket["disagreements"],
            "agreement_rate": _rate(bucket["agreements"], bucket["count"]),
            "known_search_margin_mean_on_disagreement": _mean(bucket["known_search_margins"]),
            "known_search_margin_max_on_disagreement": _max_or_none(bucket["known_search_margins"]),
            "model_action_missing_from_search_candidates": bucket["missing_model_search_value"],
        }
    return finalized


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _phase_for_turn(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


def _score_margin_bucket(score_margin: int) -> str:
    if score_margin <= -5:
        return "behind_5plus"
    if score_margin < 0:
        return "behind_1_4"
    if score_margin == 0:
        return "tied"
    if score_margin < 5:
        return "ahead_1_4"
    return "ahead_5plus"


if __name__ == "__main__":
    main()

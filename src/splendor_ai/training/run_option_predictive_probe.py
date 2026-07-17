"""Probe whether an explicit take-option improves future-card prediction."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.encoding.entity_encoding import (
    ENTITY_BOARD_CARD,
    ENTITY_SELF_RESERVED,
    collate_entity_action_batch,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.state import Card, SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_plan_persistence_diagnostic import (
    OPPONENTS,
    PlanPersistenceConfig,
    _anchors_for_game,
    _atomic_json,
    _cluster_mean_ci,
    _read_jsonl,
    _replay_path,
    collect_outcome_blind_games,
)


@dataclass(frozen=True, slots=True)
class OptionPredictiveProbeConfig:
    output_dir: Path
    data_dir: Path
    representation_checkpoint: Path
    policy_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    games: int = 512
    train_games: int = 192
    validation_games: int = 64
    seed_start: int = 391_000
    horizon_own_turns: int = 4
    workers: int = 4
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    head_hidden_size: int = 128
    seed: int = 39_0717
    bootstrap_samples: int = 10_000
    minimum_confirmation_samples: int = 800
    minimum_multi_turn_samples: int = 80
    minimum_relative_nll_improvement: float = 0.05
    minimum_top1_improvement: float = 0.05
    minimum_mrr_improvement: float = 0.03
    minimum_multi_relative_nll_improvement: float = 0.03
    minimum_multi_mrr_improvement: float = 0.02
    maximum_stratum_nll_regression: float = 0.02
    minimum_stratum_samples: int = 50
    device: str = "cpu"


class OptionProbeHead(nn.Module):
    """Identical candidate scorer used with a real or zero option vector."""

    def __init__(
        self,
        entity_hidden_size: int,
        option_hidden_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        width = 2 * entity_hidden_size + option_hidden_size
        self.scorer = nn.Sequential(
            nn.Linear(width, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        state_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        option_hidden: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_state = state_hidden.unsqueeze(1).expand(
            -1, candidate_hidden.shape[1], -1
        )
        expanded_option = option_hidden.unsqueeze(1).expand(
            -1, candidate_hidden.shape[1], -1
        )
        logits = self.scorer(
            torch.cat([expanded_state, candidate_hidden, expanded_option], dim=-1)
        ).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -torch.inf)


class ProbeFeatureDataset(Dataset):
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def __len__(self) -> int:
        return int(self.payload["target_index"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: self.payload[key][index]
            for key in (
                "state_hidden",
                "candidate_hidden",
                "option_hidden",
                "candidate_mask",
                "target_index",
            )
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/option_predictive_probe_039"
    )
    parser.add_argument("--data-dir", default="data/option_predictive_probe_039")
    parser.add_argument(
        "--representation-checkpoint",
        default=(
            "outputs/canonical_entity_coherent_023/"
            "canonical_entity_action_scorer_coherent_width64_best.pt"
        ),
    )
    parser.add_argument(
        "--policy-checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-016",
        default=(
            "outputs/legal_action_scorer_016_policy_guided_mix/"
            "legal_action_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-023",
        default=(
            "outputs/canonical_entity_coherent_023/"
            "canonical_entity_action_scorer_coherent_width64_best.pt"
        ),
    )
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> OptionPredictiveProbeConfig:
    return OptionPredictiveProbeConfig(
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        representation_checkpoint=Path(args.representation_checkpoint),
        policy_checkpoint=Path(args.policy_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_023=Path(args.checkpoint_023),
        games=args.games,
        workers=args.workers,
        epochs=args.epochs,
        device=args.device,
    )


def run_option_predictive_probe(
    config: OptionPredictiveProbeConfig,
) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing protocol differs from the frozen contract.")
    else:
        _atomic_json(protocol_path, protocol)

    collection = collect_outcome_blind_games(_collection_config(config))
    if not collection["passed"]:
        return _finalize(
            config,
            decision="collection_integrity_failed_no_option_model",
            collection=collection,
        )

    backbone = _load_backbone(config)
    train_ids, validation_ids, confirmation_ids = split_game_ids(config)
    train_examples = load_probe_examples(config, train_ids)
    validation_examples = load_probe_examples(config, validation_ids)
    train_features = compile_probe_features(backbone, train_examples, config)
    validation_features = compile_probe_features(
        backbone, validation_examples, config
    )
    training = train_paired_probe_heads(
        backbone, train_features, validation_features, config
    )
    _atomic_json(config.output_dir / "selection.json", training["selection"])
    torch.save(
        {
            "diagnostic_only": True,
            "representation_checkpoint": str(config.representation_checkpoint),
            "baseline_head_state_dict": training["baseline_state"],
            "option_head_state_dict": training["option_state"],
            "head_dimensions": training["head_dimensions"],
            "baseline_epoch": training["selection"]["baseline_epoch"],
            "option_epoch": training["selection"]["option_epoch"],
        },
        config.output_dir / "selected_probe_heads.pt",
    )

    # The confirmation games are not read until both heads and epochs are fixed.
    confirmation_examples = load_probe_examples(config, confirmation_ids)
    confirmation_features = compile_probe_features(
        backbone, confirmation_examples, config
    )
    confirmation = evaluate_paired_heads(
        training["baseline_head"],
        training["option_head"],
        confirmation_features,
        confirmation_examples,
        config,
        bootstrap=True,
    )
    gates = option_probe_gates(confirmation, config)
    decision = (
        "authorize_supervised_hierarchical_option_model"
        if all(gates.values())
        else "explicit_option_not_incremental_close_hierarchy"
    )
    _atomic_json(config.output_dir / "confirmation.json", confirmation)
    return _finalize(
        config,
        decision=decision,
        collection=collection,
        selection=training["selection"],
        confirmation=confirmation,
        gates=gates,
    )


def split_game_ids(
    config: OptionPredictiveProbeConfig,
) -> tuple[range, range, range]:
    train = range(0, config.train_games)
    validation = range(
        config.train_games, config.train_games + config.validation_games
    )
    confirmation = range(
        config.train_games + config.validation_games, config.games
    )
    return train, validation, confirmation


def load_probe_examples(
    config: OptionPredictiveProbeConfig, game_ids: Iterable[int]
) -> list[dict[str, Any]]:
    collection_config = _collection_config(config)
    examples = []
    for game_index in game_ids:
        rows = _read_jsonl(_replay_path(collection_config, game_index))
        row_by_turn = {int(row["turn_index"]): row for row in rows}
        for anchor in _anchors_for_game(rows, config.horizon_own_turns):
            source = row_by_turn[int(anchor["anchor_turn"])]
            examples.append(
                {
                    **anchor,
                    "public_state_snapshot": source["public_state_snapshot"],
                    "take_action": source["action_payload"],
                }
            )
    return examples


def compile_probe_features(
    backbone: CanonicalEntityActionScorerModel,
    examples: list[dict[str, Any]],
    config: OptionPredictiveProbeConfig,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot compile zero probe examples.")
    state_rows = []
    candidate_rows = []
    option_rows = []
    target_rows = []
    candidate_counts = []
    backbone.eval()
    for start in range(0, len(examples), config.batch_size):
        batch_examples = examples[start : start + config.batch_size]
        states = [
            deserialize_state_snapshot(row["public_state_snapshot"])
            for row in batch_examples
        ]
        actions = [_action_from_payload(row["take_action"]) for row in batch_examples]
        batch = collate_entity_action_batch(
            [(state, [action]) for state, action in zip(states, actions, strict=True)],
            device=config.device,
        )
        with torch.no_grad():
            state_hidden, entity_hidden, action_hidden = _backbone_features(
                backbone, batch
            )
        for index, (state, example) in enumerate(
            zip(states, batch_examples, strict=True)
        ):
            entity_types = batch["entity_type_ids"][index]
            positions = torch.nonzero(
                (entity_types == ENTITY_BOARD_CARD)
                | (entity_types == ENTITY_SELF_RESERVED),
                as_tuple=False,
            ).flatten()
            cards = candidate_cards(state)
            if len(cards) != len(positions):
                raise RuntimeError("Candidate cards do not align with entity rows.")
            target_index = next(
                (
                    position
                    for position, card in enumerate(cards)
                    if card.card_id == example["target_card_id"]
                ),
                None,
            )
            if target_index is None:
                raise RuntimeError("Future target is absent from candidate entities.")
            state_rows.append(state_hidden[index].detach().cpu())
            candidate_rows.append(
                entity_hidden[index, positions].detach().cpu()
            )
            option_rows.append(action_hidden[index, 0].detach().cpu())
            target_rows.append(target_index)
            candidate_counts.append(len(cards))
    max_candidates = max(candidate_counts)
    entity_width = int(candidate_rows[0].shape[-1])
    candidate_hidden = torch.zeros(
        (len(examples), max_candidates, entity_width), dtype=torch.float32
    )
    candidate_mask = torch.zeros(
        (len(examples), max_candidates), dtype=torch.bool
    )
    for index, row in enumerate(candidate_rows):
        candidate_hidden[index, : row.shape[0]] = row
        candidate_mask[index, : row.shape[0]] = True
    return {
        "state_hidden": torch.stack(state_rows),
        "candidate_hidden": candidate_hidden,
        "option_hidden": torch.stack(option_rows),
        "candidate_mask": candidate_mask,
        "target_index": torch.tensor(target_rows, dtype=torch.long),
    }


def candidate_cards(state: SplendorState) -> list[Card]:
    cards = []
    for tier in (1, 2, 3):
        cards.extend(sorted(state.visible_tier_cards[tier], key=lambda card: card.card_id))
    cards.extend(
        sorted(
            state.players[state.current_player].reserved_cards,
            key=lambda card: card.card_id,
        )
    )
    return cards


def _backbone_features(
    model: CanonicalEntityActionScorerModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entity_input = torch.cat(
        [
            batch["entity_numeric_features"],
            model.entity_type_embedding(batch["entity_type_ids"]),
            model.card_embedding(batch["entity_card_ids"]),
            model.noble_embedding(batch["entity_noble_ids"]),
        ],
        dim=-1,
    )
    entity_hidden = model.entity_encoder(
        model.entity_input(entity_input),
        src_key_padding_mask=~batch["entity_mask"],
    )
    entity_hidden = entity_hidden.masked_fill(
        ~batch["entity_mask"].unsqueeze(-1), 0.0
    )
    entity_count = (
        batch["entity_mask"]
        .sum(dim=1, keepdim=True)
        .clamp_min(1)
        .to(entity_hidden.dtype)
    )
    state_hidden = entity_hidden.sum(dim=1) / entity_count
    action_input = torch.cat(
        [
            batch["action_numeric_features"],
            model.action_type_embedding(batch["action_type_ids"]),
            model.card_embedding(batch["action_card_ids"]),
            model.noble_embedding(batch["action_noble_ids"]),
        ],
        dim=-1,
    )
    action_hidden = model.action_encoder(
        model.action_input(action_input)
        + model.state_to_action_context(state_hidden).unsqueeze(1),
        src_key_padding_mask=~batch["action_mask"],
    )
    action_hidden = model.action_layer_norm(action_hidden)
    action_hidden = action_hidden.masked_fill(
        ~batch["action_mask"].unsqueeze(-1), 0.0
    )
    return state_hidden, entity_hidden, action_hidden


def train_paired_probe_heads(
    backbone: CanonicalEntityActionScorerModel,
    train_features: dict[str, Any],
    validation_features: dict[str, Any],
    config: OptionPredictiveProbeConfig,
) -> dict[str, Any]:
    entity_width = int(train_features["state_hidden"].shape[-1])
    option_width = int(train_features["option_hidden"].shape[-1])
    torch.manual_seed(config.seed)
    baseline = OptionProbeHead(
        entity_width, option_width, config.head_hidden_size
    ).to(config.device)
    option = copy.deepcopy(baseline).to(config.device)
    baseline_optimizer = AdamW(
        baseline.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    option_optimizer = AdamW(
        option.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = DataLoader(
        ProbeFeatureDataset(train_features),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    history = []
    best_baseline = None
    best_option = None
    best_baseline_epoch = None
    best_option_epoch = None
    best_baseline_nll = math.inf
    best_option_nll = math.inf
    for epoch in range(config.epochs):
        baseline.train()
        option.train()
        for batch in loader:
            state = batch["state_hidden"].to(config.device)
            candidates = batch["candidate_hidden"].to(config.device)
            option_hidden = batch["option_hidden"].to(config.device)
            mask = batch["candidate_mask"].to(config.device)
            targets = batch["target_index"].to(config.device)
            baseline_optimizer.zero_grad(set_to_none=True)
            option_optimizer.zero_grad(set_to_none=True)
            baseline_logits = baseline(
                state, candidates, torch.zeros_like(option_hidden), mask
            )
            option_logits = option(state, candidates, option_hidden, mask)
            baseline_loss = F.cross_entropy(baseline_logits, targets)
            option_loss = F.cross_entropy(option_logits, targets)
            (baseline_loss + option_loss).backward()
            baseline_optimizer.step()
            option_optimizer.step()
        validation = evaluate_paired_heads(
            baseline,
            option,
            validation_features,
            None,
            config,
            bootstrap=False,
        )
        baseline_nll = float(validation["baseline"]["nll"])
        option_nll = float(validation["option"]["nll"])
        history.append(
            {
                "epoch": epoch + 1,
                "baseline_validation_nll": baseline_nll,
                "option_validation_nll": option_nll,
                "baseline_top1": validation["baseline"]["top1"],
                "option_top1": validation["option"]["top1"],
            }
        )
        print(
            f"[option-probe] epoch={epoch+1}/{config.epochs}"
            f" baseline_nll={baseline_nll:.5f} option_nll={option_nll:.5f}",
            flush=True,
        )
        if baseline_nll < best_baseline_nll:
            best_baseline_nll = baseline_nll
            best_baseline_epoch = epoch + 1
            best_baseline = copy.deepcopy(baseline.state_dict())
        if option_nll < best_option_nll:
            best_option_nll = option_nll
            best_option_epoch = epoch + 1
            best_option = copy.deepcopy(option.state_dict())
    if (
        best_baseline is None
        or best_option is None
        or best_baseline_epoch is None
        or best_option_epoch is None
    ):
        raise RuntimeError("Probe selection failed.")
    baseline.load_state_dict(best_baseline)
    option.load_state_dict(best_option)
    selected_validation = evaluate_paired_heads(
        baseline,
        option,
        validation_features,
        None,
        config,
        bootstrap=False,
    )
    return {
        "baseline_head": baseline,
        "option_head": option,
        "baseline_state": best_baseline,
        "option_state": best_option,
        "head_dimensions": {
            "entity_hidden_size": entity_width,
            "option_hidden_size": option_width,
            "head_hidden_size": config.head_hidden_size,
        },
        "selection": {
            "train_examples": int(train_features["target_index"].shape[0]),
            "validation_examples": int(validation_features["target_index"].shape[0]),
            "baseline_epoch": best_baseline_epoch,
            "option_epoch": best_option_epoch,
            "history": history,
            "selected_validation": selected_validation,
        },
    }


def evaluate_paired_heads(
    baseline: OptionProbeHead,
    option: OptionProbeHead,
    features: dict[str, Any],
    examples: list[dict[str, Any]] | None,
    config: OptionPredictiveProbeConfig,
    *,
    bootstrap: bool,
) -> dict[str, Any]:
    baseline.eval()
    option.eval()
    with torch.no_grad():
        state = features["state_hidden"].to(config.device)
        candidates = features["candidate_hidden"].to(config.device)
        option_hidden = features["option_hidden"].to(config.device)
        mask = features["candidate_mask"].to(config.device)
        targets = features["target_index"].to(config.device)
        baseline_logits = baseline(state, candidates, torch.zeros_like(option_hidden), mask)
        option_logits = option(state, candidates, option_hidden, mask)
        baseline_log_probs = F.log_softmax(baseline_logits, dim=1)
        option_log_probs = F.log_softmax(option_logits, dim=1)
        rows = []
        for index in range(targets.shape[0]):
            target = int(targets[index])
            baseline_order = torch.argsort(baseline_logits[index], descending=True)
            option_order = torch.argsort(option_logits[index], descending=True)
            baseline_rank = int(torch.nonzero(baseline_order == target)[0]) + 1
            option_rank = int(torch.nonzero(option_order == target)[0]) + 1
            metadata = examples[index] if examples is not None else {}
            rows.append(
                {
                    "game_index": int(metadata.get("game_index", index)),
                    "opponent": metadata.get("opponent", "validation"),
                    "actor_seat": int(metadata.get("actor_seat", 0)),
                    "phase": metadata.get("phase", "validation"),
                    "own_turn_distance": int(metadata.get("own_turn_distance", 1)),
                    "baseline_nll": -float(baseline_log_probs[index, target]),
                    "option_nll": -float(option_log_probs[index, target]),
                    "nll_delta": float(
                        -option_log_probs[index, target]
                        + baseline_log_probs[index, target]
                    ),
                    "baseline_top1": baseline_rank == 1,
                    "option_top1": option_rank == 1,
                    "top1_delta": float(option_rank == 1) - float(baseline_rank == 1),
                    "baseline_mrr": 1.0 / baseline_rank,
                    "option_mrr": 1.0 / option_rank,
                    "mrr_delta": 1.0 / option_rank - 1.0 / baseline_rank,
                    "prediction_changed": int(baseline_order[0]) != int(option_order[0]),
                }
            )
    result = _paired_summary(rows, config, bootstrap)
    if examples is not None:
        result["multi_turn"] = _paired_summary(
            [row for row in rows if row["own_turn_distance"] >= 2],
            config,
            bootstrap,
        )
        result["breakdowns"] = {}
        for name, key_fn in (
            ("opponent_seat", lambda row: f"{row['opponent']}:seat{row['actor_seat']}"),
            ("phase", lambda row: str(row["phase"])),
            ("horizon", lambda row: str(row["own_turn_distance"])),
        ):
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                groups.setdefault(key_fn(row), []).append(row)
            result["breakdowns"][name] = {
                key: _paired_summary(group, config, False)
                for key, group in sorted(groups.items())
            }
    return result


def _paired_summary(
    rows: list[dict[str, Any]],
    config: OptionPredictiveProbeConfig,
    bootstrap: bool,
) -> dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "baseline": {"nll": 0.0, "top1": 0.0, "mrr": 0.0},
            "option": {"nll": 0.0, "top1": 0.0, "mrr": 0.0},
            "relative_nll_improvement": 0.0,
            "prediction_change_rate": 0.0,
        }
    baseline_nll = statistics.fmean(row["baseline_nll"] for row in rows)
    option_nll = statistics.fmean(row["option_nll"] for row in rows)
    result = {
        "samples": len(rows),
        "games": len({row["game_index"] for row in rows}),
        "baseline": {
            "nll": baseline_nll,
            "top1": statistics.fmean(float(row["baseline_top1"]) for row in rows),
            "mrr": statistics.fmean(row["baseline_mrr"] for row in rows),
        },
        "option": {
            "nll": option_nll,
            "top1": statistics.fmean(float(row["option_top1"]) for row in rows),
            "mrr": statistics.fmean(row["option_mrr"] for row in rows),
        },
        "relative_nll_improvement": (baseline_nll - option_nll) / baseline_nll,
        "prediction_change_rate": statistics.fmean(
            float(row["prediction_changed"]) for row in rows
        ),
    }
    if bootstrap:
        result["nll_delta_ci"] = _cluster_mean_ci(
            rows, "nll_delta", config.bootstrap_samples, config.seed + 101
        )
        result["top1_delta_ci"] = _cluster_mean_ci(
            rows, "top1_delta", config.bootstrap_samples, config.seed + 102
        )
        result["mrr_delta_ci"] = _cluster_mean_ci(
            rows, "mrr_delta", config.bootstrap_samples, config.seed + 103
        )
    return result


def option_probe_gates(
    confirmation: dict[str, Any], config: OptionPredictiveProbeConfig
) -> dict[str, bool]:
    multi = confirmation["multi_turn"]
    gates = {
        "confirmation_coverage": int(confirmation["samples"])
        >= config.minimum_confirmation_samples,
        "multi_turn_coverage": int(multi["samples"])
        >= config.minimum_multi_turn_samples,
        "nll_improves_at_least_5pct": float(
            confirmation["relative_nll_improvement"]
        )
        >= config.minimum_relative_nll_improvement,
        "nll_delta_cluster_ci_below_zero": float(
            confirmation["nll_delta_ci"]["ci95_high"]
        )
        < 0.0,
        "top1_improves_at_least_5pt": (
            float(confirmation["option"]["top1"])
            - float(confirmation["baseline"]["top1"])
            >= config.minimum_top1_improvement
        ),
        "top1_delta_cluster_ci_above_zero": float(
            confirmation["top1_delta_ci"]["ci95_low"]
        )
        > 0.0,
        "mrr_improves_at_least_3pt": (
            float(confirmation["option"]["mrr"])
            - float(confirmation["baseline"]["mrr"])
            >= config.minimum_mrr_improvement
        ),
        "mrr_delta_cluster_ci_above_zero": float(
            confirmation["mrr_delta_ci"]["ci95_low"]
        )
        > 0.0,
        "multi_turn_nll_improves_at_least_3pct": float(
            multi["relative_nll_improvement"]
        )
        >= config.minimum_multi_relative_nll_improvement,
        "multi_turn_top1_nonregression": float(multi["option"]["top1"])
        >= float(multi["baseline"]["top1"]),
        "multi_turn_mrr_improves_at_least_2pt": (
            float(multi["option"]["mrr"])
            - float(multi["baseline"]["mrr"])
            >= config.minimum_multi_mrr_improvement
        ),
    }
    eligible = []
    for dimension in ("opponent_seat", "phase"):
        eligible.extend(
            row
            for row in confirmation["breakdowns"][dimension].values()
            if int(row["samples"]) >= config.minimum_stratum_samples
        )
    gates["all_major_strata_covered"] = len(eligible) >= len(OPPONENTS) * 2 + 3
    gates["no_major_stratum_nll_collapse"] = all(
        float(row["option"]["nll"])
        <= float(row["baseline"]["nll"])
        * (1.0 + config.maximum_stratum_nll_regression)
        for row in eligible
    )
    return gates


def _load_backbone(
    config: OptionPredictiveProbeConfig,
) -> CanonicalEntityActionScorerModel:
    checkpoint = torch.load(
        config.representation_checkpoint,
        map_location=config.device,
        weights_only=True,
    )
    if checkpoint.get("model_kind") != "canonical_entity_action_scorer":
        raise ValueError("Probe representation must be a canonical entity scorer.")
    model = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _action_from_payload(payload: dict[str, Any]) -> Action:
    return Action(
        action_type=ActionType[str(payload["action_type"])],
        tier=payload.get("tier"),
        market_index=payload.get("market_index"),
        reserved_index=payload.get("reserved_index"),
        take_tokens=tuple(payload.get("take_tokens", ())),
        return_tokens=tuple(payload.get("return_tokens", ())),
        spend_tokens=tuple(payload.get("spend_tokens", ())),
        noble_id=payload.get("noble_id"),
        metadata=tuple(tuple(item) for item in payload.get("metadata", ())),
    )


def _collection_config(
    config: OptionPredictiveProbeConfig,
) -> PlanPersistenceConfig:
    return PlanPersistenceConfig(
        output_dir=config.output_dir / "collection",
        data_dir=config.data_dir,
        policy_checkpoint=config.policy_checkpoint,
        checkpoint_016=config.checkpoint_016,
        checkpoint_023=config.checkpoint_023,
        games=config.games,
        seed_start=config.seed_start,
        horizon_own_turns=config.horizon_own_turns,
        workers=config.workers,
        bootstrap_samples=config.bootstrap_samples,
        device=config.device,
    )


def _protocol(config: OptionPredictiveProbeConfig) -> dict[str, Any]:
    return {
        "experiment": "option_predictive_probe_039",
        "frozen_at": "before fresh collection, training, or confirmation access",
        "diagnostic_only": True,
        "training_scope": "two probe heads only; frozen 023 representation; no policy update",
        "promotion_allowed": False,
        "data": {
            "games": config.games,
            "seeds": [config.seed_start, config.seed_start + config.games - 1],
            "train_games": [0, config.train_games - 1],
            "validation_games": [
                config.train_games,
                config.train_games + config.validation_games - 1,
            ],
            "confirmation_games": [
                config.train_games + config.validation_games,
                config.games - 1,
            ],
            "balance": "each split is exactly balanced over opponent and actor seat",
            "outcomes": "never stored or read",
        },
        "target": (
            "first already-visible/self-reserved card bought or visibly reserved "
            "within four own turns after a qualifying token take"
        ),
        "comparison": (
            "identically initialized and parameterized candidate heads over frozen "
            "023 contextual state/card features; baseline option vector is exactly "
            "zero, conditioned vector is the frozen 023 embedding of the selected take"
        ),
        "selection": "each head independently selects lowest validation NLL over 30 epochs",
        "confirmation": "not loaded until both head states and epochs are frozen",
        "gates": {
            "coverage": [
                config.minimum_confirmation_samples,
                config.minimum_multi_turn_samples,
            ],
            "overall_nll": (
                f">={config.minimum_relative_nll_improvement:.0%} relative improvement "
                "and game-cluster delta CI95 high <0"
            ),
            "overall_top1": (
                f">={config.minimum_top1_improvement:.0%} absolute improvement and "
                "game-cluster delta CI95 low >0"
            ),
            "overall_mrr": (
                f">={config.minimum_mrr_improvement:.0%} absolute improvement and "
                "game-cluster delta CI95 low >0"
            ),
            "multi_turn": (
                f">={config.minimum_multi_relative_nll_improvement:.0%} NLL improvement, "
                "top1 nonregression, and >=2-point MRR improvement"
            ),
            "strata": "all major opponent/seat and phase strata <=2% NLL regression",
        },
        "decision": (
            "pass authorizes separately preregistered supervised high-level option "
            "and option-conditioned low-level reconstruction; failure closes hierarchy"
        ),
        "checkpoint_hashes": {
            "representation_023": _sha256(config.representation_checkpoint),
            "035": _sha256(config.policy_checkpoint),
            "016": _sha256(config.checkpoint_016),
            "023": _sha256(config.checkpoint_023),
        },
        "config": _config_payload(config),
    }


def _config_payload(config: OptionPredictiveProbeConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(config: OptionPredictiveProbeConfig) -> None:
    if config.games != config.train_games + config.validation_games + (
        config.games - config.train_games - config.validation_games
    ):
        raise ValueError("Invalid split arithmetic.")
    confirmation_games = config.games - config.train_games - config.validation_games
    if min(config.train_games, config.validation_games, confirmation_games) <= 0:
        raise ValueError("All splits must be nonempty.")
    if any(
        count % (len(OPPONENTS) * 2)
        for count in (config.train_games, config.validation_games, confirmation_games)
    ):
        raise ValueError("Every split must balance opponent and seat exactly.")
    if min(config.workers, config.epochs, config.batch_size) <= 0:
        raise ValueError("Training and collection counts must be positive.")
    for path in (
        config.representation_checkpoint,
        config.policy_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def _finalize(
    config: OptionPredictiveProbeConfig, *, decision: str, **stages: Any
) -> dict[str, Any]:
    payload = {
        "gate": "option_predictive_probe_039",
        "decision": decision,
        "diagnostic_only": True,
        "policy_updated": False,
        "promotion_allowed": False,
        **stages,
    }
    _atomic_json(config.output_dir / "gate_final.json", payload)
    print(
        json.dumps(
            {
                "decision": decision,
                "policy_updated": False,
                "promotion_allowed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    run_option_predictive_probe(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()

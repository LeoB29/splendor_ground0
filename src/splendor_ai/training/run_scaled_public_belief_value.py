"""Train and gate the preregistered scaled public-belief value bootstrap."""

from __future__ import annotations

import argparse
import copy
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
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.bots import (
    InformationSetPuctBot,
    LegalActionAttentionScorerBot,
    PuctConfig,
)
from splendor_ai.bots.puct_bot import legal_heuristic_value
from splendor_ai.bots.search_bot import (
    CheckpointLegalActionPriorScorer,
    CheckpointValueEvaluator,
)
from splendor_ai.encoding import encode_canonical_state_entities
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_shallow_operational_gate import summarize_pair_rows
from splendor_ai.training.model import (
    CanonicalEntityDistributionalValueModel,
    CanonicalEntityValueConfig,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_entity_public_value import (
    _basic_metrics,
    _breakdowns,
    _correlation,
    _ordering_summary,
    _paired_game_error,
    phase_for_turn,
)
from splendor_ai.training.run_league_generation import (
    LeagueGenerationConfig,
    SamplingSpec,
    collect_league_groups,
)


OPPONENTS = ("035", "016", "shallow", "023")
PHASES = ("0-14", "15-29", "30-44", "45+")
SOURCE_NAMES = ("032", "033")
MAX_ENTITIES = 64
MARGIN_BINS = 41
MARGIN_CLIP = 20


@dataclass(frozen=True, slots=True)
class ScaledPublicBeliefValueConfig:
    source_032: Path
    source_033: Path
    actor_parent_checkpoint: Path
    policy_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    output_dir: Path
    confirmation_data_dir: Path
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    wdl_loss_weight: float = 0.5
    search_wdl_blend_weight: float = 0.5
    seed: int = 2_036_0717
    workers: int = 4
    bootstrap_samples: int = 20_000
    device: str = "cpu"
    confirmation_groups: int = 128
    confirmation_seed_start: int = 378_000
    confirmation_sampling_seed_start: int = 41_000_000
    puct_pairs: int = 100
    puct_ab_seed_start: int = 379_000
    puct_direct_seed_start: int = 380_000


class TensorValueDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, payload: dict[str, torch.Tensor], indices: Iterable[int] | None = None):
        self.payload = payload
        self.indices = (
            torch.arange(int(payload["margin_class"].shape[0]), dtype=torch.long)
            if indices is None
            else torch.tensor(list(indices), dtype=torch.long)
        )

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "entity_type_ids": self.payload["entity_type_ids"][row].long(),
            "entity_card_ids": self.payload["entity_card_ids"][row].long(),
            "entity_noble_ids": self.payload["entity_noble_ids"][row].long(),
            "entity_numeric_features": self.payload["entity_numeric_features"][row].float(),
            "entity_mask": self.payload["entity_mask"][row],
            "margin_class": self.payload["margin_class"][row].long(),
            "wdl_class": self.payload["wdl_class"][row].long(),
        }


class ExactlyBalancedPhaseDataset(TensorValueDataset):
    """Cycle all generation/opponent/seat/phase strata equally each epoch."""

    def __init__(self, payload: dict[str, torch.Tensor]):
        groups: dict[tuple[int, int, int, int], list[int]] = {}
        count = int(payload["margin_class"].shape[0])
        for index in range(count):
            key = (
                int(payload["source_id"][index]),
                int(payload["opponent_id"][index]),
                int(payload["actor_seat"][index]),
                int(payload["phase_id"][index]),
            )
            groups.setdefault(key, []).append(index)
        expected = {
            (source, opponent, seat, phase)
            for source in range(2)
            for opponent in range(4)
            for seat in range(2)
            for phase in range(4)
        }
        if set(groups) != expected or any(not rows for rows in groups.values()):
            raise ValueError("All 64 generation/opponent/seat/phase strata are required.")
        self.payload = payload
        self.groups = groups
        self.strata = tuple(sorted(groups))
        self.per_stratum = max(len(rows) for rows in groups.values())
        balanced = []
        for offset in range(self.per_stratum):
            for key in self.strata:
                rows = groups[key]
                balanced.append(rows[offset % len(rows)])
        self.indices = torch.tensor(balanced, dtype=torch.long)


def _group_assignment(group: int) -> tuple[str, int]:
    return OPPONENTS[group % 4], (group // 4) % 2


def _phase_id(turn_index: int) -> int:
    return PHASES.index(phase_for_turn(turn_index))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _selected_group_ids(split: str) -> range:
    if split == "train":
        return range(448)
    if split == "validation":
        return range(448, 512)
    if split == "confirmation":
        return range(128)
    raise ValueError(f"Unknown split: {split}")


def _compile_tensor_corpus(
    source_dirs: tuple[Path, ...],
    split: str,
    output_path: Path,
) -> dict[str, torch.Tensor]:
    if output_path.exists():
        return torch.load(output_path, map_location="cpu", weights_only=True, mmap=True)
    groups = _selected_group_ids(split)
    summaries: list[tuple[int, int, dict[str, Any]]] = []
    expected_rows = 0
    for source_id, source_dir in enumerate(source_dirs):
        for group in groups:
            summary_path = source_dir / "raw" / f"group_{group:04d}.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            opponent, seat = _group_assignment(group)
            if summary["opponent"] != opponent or int(summary["actor_seat"]) != seat:
                raise ValueError(f"Group assignment mismatch: {summary_path}")
            expected_rows += int(summary["actor_turns"])
            summaries.append((source_id, group, summary))
    payload = {
        "entity_type_ids": torch.zeros((expected_rows, MAX_ENTITIES), dtype=torch.int16),
        "entity_card_ids": torch.zeros((expected_rows, MAX_ENTITIES), dtype=torch.int16),
        "entity_noble_ids": torch.zeros((expected_rows, MAX_ENTITIES), dtype=torch.int16),
        "entity_numeric_features": torch.zeros(
            (expected_rows, MAX_ENTITIES, 34), dtype=torch.float16
        ),
        "entity_mask": torch.zeros((expected_rows, MAX_ENTITIES), dtype=torch.bool),
        "margin_class": torch.zeros(expected_rows, dtype=torch.int8),
        "wdl_class": torch.zeros(expected_rows, dtype=torch.int8),
        "heuristic": torch.zeros(expected_rows, dtype=torch.float32),
        "source_id": torch.zeros(expected_rows, dtype=torch.int8),
        "opponent_id": torch.zeros(expected_rows, dtype=torch.int8),
        "actor_seat": torch.zeros(expected_rows, dtype=torch.int8),
        "phase_id": torch.zeros(expected_rows, dtype=torch.int8),
        "group": torch.zeros(expected_rows, dtype=torch.int16),
        "rollout": torch.zeros(expected_rows, dtype=torch.int8),
        "environment_seed": torch.zeros(expected_rows, dtype=torch.int32),
    }
    cursor = 0
    for source_id, group, summary in summaries:
        actor_seat = int(summary["actor_seat"])
        opponent_id = OPPONENTS.index(str(summary["opponent"]))
        for rollout in range(4):
            replay_path = source_dirs[source_id] / "raw" / f"group_{group:04d}_rollout_{rollout}.jsonl"
            with replay_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("behavior_probability") is None:
                        continue
                    if int(row["player_id"]) != actor_seat:
                        raise ValueError(f"Sampling actor seat mismatch in {replay_path}")
                    state = deserialize_state_snapshot(row["state_snapshot"])
                    encoding = encode_canonical_state_entities(state, actor_seat)
                    width = len(encoding.entity_type_ids)
                    if width > MAX_ENTITIES:
                        raise ValueError(f"Canonical entity width {width} exceeds {MAX_ENTITIES}.")
                    payload["entity_type_ids"][cursor, :width] = torch.tensor(
                        encoding.entity_type_ids, dtype=torch.int16
                    )
                    payload["entity_card_ids"][cursor, :width] = torch.tensor(
                        encoding.card_ids, dtype=torch.int16
                    )
                    payload["entity_noble_ids"][cursor, :width] = torch.tensor(
                        encoding.noble_ids, dtype=torch.int16
                    )
                    payload["entity_numeric_features"][cursor, :width] = torch.tensor(
                        encoding.numeric_features, dtype=torch.float16
                    )
                    payload["entity_mask"][cursor, :width] = True
                    scores = row["game_final_scores"]
                    raw_margin = int(scores[actor_seat]) - int(scores[1 - actor_seat])
                    clipped_margin = max(-MARGIN_CLIP, min(MARGIN_CLIP, raw_margin))
                    payload["margin_class"][cursor] = clipped_margin + MARGIN_CLIP
                    payload["wdl_class"][cursor] = 0 if raw_margin < 0 else 2 if raw_margin > 0 else 1
                    payload["heuristic"][cursor] = legal_heuristic_value(
                        state, actor_seat, scale=4_000.0
                    )
                    payload["source_id"][cursor] = source_id
                    payload["opponent_id"][cursor] = opponent_id
                    payload["actor_seat"][cursor] = actor_seat
                    payload["phase_id"][cursor] = _phase_id(int(row["turn_index"]))
                    payload["group"][cursor] = group
                    payload["rollout"][cursor] = rollout
                    payload["environment_seed"][cursor] = int(summary["environment_seed"])
                    cursor += 1
        if len(summaries) <= 32 or (group + 1) % 32 == 0:
            print(f"[value-compile:{split}] source={source_id} group={group} rows={cursor}/{expected_rows}")
    if cursor != expected_rows:
        raise ValueError(f"Compiled {cursor} rows but summaries promised {expected_rows}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return payload


def _model_config_from_parent(parent: dict[str, Any]) -> CanonicalEntityValueConfig:
    config = parent["model_config"]
    return CanonicalEntityValueConfig(
        hidden_size=int(config["entity_hidden_size"]),
        layers=int(config["entity_layers"]),
        attention_heads=int(config["attention_heads"]),
        attention_feedforward_size=int(config["attention_feedforward_size"]),
        id_embedding_size=int(config["id_embedding_size"]),
        dropout=0.0,
    )


def initialize_from_actor(
    model: CanonicalEntityDistributionalValueModel,
    actor_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    prefixes = (
        "entity_type_embedding.", "card_embedding.", "noble_embedding.",
        "entity_input.", "entity_encoder.",
    )
    target = model.state_dict()
    transfer = {
        name: value for name, value in actor_state.items()
        if name.startswith(prefixes) and name in target and value.shape == target[name].shape
    }
    expected = {name for name in target if name.startswith(prefixes)}
    if set(transfer) != expected:
        raise ValueError(f"Incompatible actor encoder; missing {sorted(expected - set(transfer))}")
    model.load_state_dict(transfer, strict=False)
    return tuple(sorted(transfer))


def _model_inputs(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in (
            "entity_type_ids", "entity_card_ids", "entity_noble_ids",
            "entity_numeric_features", "entity_mask",
        )
    }


def _train_epoch(model, loader, optimizer, config) -> dict[str, float | int]:
    model.train()
    margin_loss = 0.0
    wdl_loss = 0.0
    samples = 0
    for batch in loader:
        margin_logits, wdl_logits = model(**_model_inputs(batch, config.device))
        margin_target = batch["margin_class"].to(config.device)
        wdl_target = batch["wdl_class"].to(config.device)
        margin_ce = F.cross_entropy(margin_logits, margin_target)
        wdl_ce = F.cross_entropy(wdl_logits, wdl_target)
        loss = margin_ce + config.wdl_loss_weight * wdl_ce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(margin_target.numel())
        margin_loss += float(margin_ce.detach()) * count
        wdl_loss += float(wdl_ce.detach()) * count
        samples += count
    return {
        "margin_nll": margin_loss / samples,
        "wdl_nll": wdl_loss / samples,
        "samples": samples,
    }


def _ece(probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
    confidence, prediction = probabilities.max(dim=1)
    accuracy = prediction.eq(targets)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index + 1 < bins else confidence <= high)
        if bool(mask.any()):
            result += float(mask.float().mean()) * abs(
                float(accuracy[mask].float().mean()) - float(confidence[mask].mean())
            )
    return result


def _ordering_by_phase(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        phase: _ordering_summary([row for row in records if row["phase"] == phase], key)
        for phase in PHASES
    }


def evaluate_model(model, payload, config, *, bootstrap: bool, seed: int) -> dict[str, Any]:
    model.eval()
    dataset = TensorValueDataset(payload)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    margin_logits_parts, wdl_logits_parts = [], []
    with torch.no_grad():
        for batch in loader:
            margin_logits, wdl_logits = model(**_model_inputs(batch, config.device))
            margin_logits_parts.append(margin_logits.cpu())
            wdl_logits_parts.append(wdl_logits.cpu())
    margin_logits = torch.cat(margin_logits_parts)
    wdl_logits = torch.cat(wdl_logits_parts)
    margin_prob = torch.softmax(margin_logits, dim=1)
    wdl_prob = torch.softmax(wdl_logits, dim=1)
    margin_target = payload["margin_class"].long()
    wdl_target = payload["wdl_class"].long()
    support = torch.linspace(-1.0, 1.0, MARGIN_BINS)
    margin_value = (margin_prob * support).sum(dim=1)
    wdl_value = wdl_prob[:, 2] - wdl_prob[:, 0]
    candidate = (
        (1.0 - config.search_wdl_blend_weight) * margin_value
        + config.search_wdl_blend_weight * wdl_value
    )
    target = (margin_target.float() - MARGIN_CLIP) / MARGIN_CLIP
    records = []
    for index in range(len(dataset)):
        source_id = int(payload["source_id"][index])
        group = int(payload["group"][index])
        rollout = int(payload["rollout"][index])
        records.append({
            "candidate": float(candidate[index]),
            "margin_value": float(margin_value[index]),
            "wdl_value": float(wdl_value[index]),
            "heuristic": float(payload["heuristic"][index]),
            "target": float(target[index]),
            "game_seed": source_id * 1_000_000 + group * 10 + rollout,
            "environment_seed": int(payload["environment_seed"][index]),
            "profile": OPPONENTS[int(payload["opponent_id"][index])],
            "actor_seat": int(payload["actor_seat"][index]),
            "phase": PHASES[int(payload["phase_id"][index])],
        })
    candidate_metrics = _basic_metrics(records, "candidate")
    heuristic_metrics = _basic_metrics(records, "heuristic")
    candidate_breakdowns = _breakdowns(records, "candidate")
    heuristic_breakdowns = _breakdowns(records, "heuristic")
    labels = [f"profile_seat:{profile}:seat{seat}" for profile in OPPONENTS for seat in (0, 1)]
    candidate_metrics["macro_profile_seat_mae"] = statistics.fmean(
        float(candidate_breakdowns[label]["mae"]) for label in labels
    )
    heuristic_metrics["macro_profile_seat_mae"] = statistics.fmean(
        float(heuristic_breakdowns[label]["mae"]) for label in labels
    )
    one_hot = F.one_hot(wdl_target, num_classes=3).float()
    result = {
        "candidate": candidate_metrics,
        "heuristic": heuristic_metrics,
        "candidate_breakdowns": candidate_breakdowns,
        "heuristic_breakdowns": heuristic_breakdowns,
        "candidate_ordering": _ordering_summary(records, "candidate"),
        "heuristic_ordering": _ordering_summary(records, "heuristic"),
        "candidate_ordering_by_phase": _ordering_by_phase(records, "candidate"),
        "heuristic_ordering_by_phase": _ordering_by_phase(records, "heuristic"),
        "calibration": {
            "margin_nll": float(F.cross_entropy(margin_logits, margin_target)),
            "wdl_nll": float(F.cross_entropy(wdl_logits, wdl_target)),
            "wdl_brier": float((wdl_prob - one_hot).square().sum(dim=1).mean()),
            "wdl_ece10": _ece(wdl_prob, wdl_target),
        },
        "records": len(records),
    }
    macro_nlls = []
    for source in range(int(payload["source_id"].max()) + 1):
        for opponent in range(4):
            for seat in range(2):
                for phase in range(4):
                    mask = (
                        (payload["source_id"] == source)
                        & (payload["opponent_id"] == opponent)
                        & (payload["actor_seat"] == seat)
                        & (payload["phase_id"] == phase)
                    )
                    if bool(mask.any()):
                        macro_nlls.append(
                            float(F.cross_entropy(margin_logits[mask], margin_target[mask]))
                            + config.wdl_loss_weight * float(F.cross_entropy(wdl_logits[mask], wdl_target[mask]))
                        )
    result["selection"] = {
        "macro_calibration_nll": statistics.fmean(macro_nlls),
        "ordering_concordance": result["candidate_ordering"]["concordance"],
    }
    result["selection"]["score"] = (
        float(result["selection"]["macro_calibration_nll"])
        - 0.25 * float(result["selection"]["ordering_concordance"])
    )
    if bootstrap:
        result["paired_game_error"] = _paired_game_error(
            records, config.bootstrap_samples, seed
        )
    return result


def _confirmation_collection(config: ScaledPublicBeliefValueConfig) -> dict[str, Any]:
    collection_output = config.output_dir / "confirmation_collection"
    league_config = LeagueGenerationConfig(
        output_dir=collection_output,
        data_dir=config.confirmation_data_dir,
        parent_checkpoint=config.policy_checkpoint,
        checkpoint_016=config.checkpoint_016,
        checkpoint_023=config.checkpoint_023,
        calibration_replay=config.source_032 / "raw" / "group_0000_rollout_0.jsonl",
        broad_replay=config.source_032 / "raw" / "group_0000_rollout_0.jsonl",
        groups=config.confirmation_groups,
        train_groups=config.confirmation_groups,
        trajectories_per_group=4,
        collection_seed_start=config.confirmation_seed_start,
        sampling_seed_start=config.confirmation_sampling_seed_start,
        workers=config.workers,
        device=config.device,
    )
    return collect_league_groups(
        league_config,
        SamplingSpec(
            epsilon=0.5,
            temperature=1.0,
            top_k=3,
            expected_nongreedy_rate=0.15,
        ),
    )


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _reservoir_confirmation_states(config, per_stratum: int = 8):
    rng = random.Random(config.seed + 503)
    reservoirs: dict[tuple[str, int, str], list[tuple[Any, int]]] = {}
    seen: dict[tuple[str, int, str], int] = {}
    for group in range(config.confirmation_groups):
        summary = json.loads(
            (config.confirmation_data_dir / "raw" / f"group_{group:04d}.summary.json").read_text(encoding="utf-8")
        )
        seat = int(summary["actor_seat"])
        opponent = str(summary["opponent"])
        for rollout in range(4):
            path = config.confirmation_data_dir / "raw" / f"group_{group:04d}_rollout_{rollout}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("behavior_probability") is None:
                        continue
                    key = (opponent, seat, phase_for_turn(int(row["turn_index"])))
                    seen[key] = seen.get(key, 0) + 1
                    values = reservoirs.setdefault(key, [])
                    item = (deserialize_state_snapshot(row["state_snapshot"]), seat)
                    if len(values) < per_stratum:
                        values.append(item)
                    else:
                        replacement = rng.randrange(seen[key])
                        if replacement < per_stratum:
                            values[replacement] = item
    expected = {(opponent, seat, phase) for opponent in OPPONENTS for seat in (0, 1) for phase in PHASES}
    if set(reservoirs) != expected or any(len(rows) != per_stratum for rows in reservoirs.values()):
        raise ValueError("Fresh confirmation could not fill all successor-consistency strata.")
    return [(key, state, seat) for key, rows in sorted(reservoirs.items()) for state, seat in rows]


def evaluate_successor_consistency(config, checkpoint_path: Path) -> dict[str, Any]:
    scorer = CheckpointLegalActionPriorScorer(config.policy_checkpoint, device=config.device)
    evaluator = CheckpointValueEvaluator(
        checkpoint_path,
        device=config.device,
        value_scale=1.0,
        prediction_mode="blend",
        outcome_blend_weight=config.search_wdl_blend_weight,
    )
    env = SplendorEnv(seed=config.seed)
    records = []
    for stratum, state, actor in _reservoir_confirmation_states(config):
        actions = env.legal_actions(state)
        if len(actions) < 2:
            continue
        policy_scores = scorer.score_actions(state, actions)
        successors = [env.step(state, action) for action in actions]
        values = evaluator.evaluate_states([(successor, actor) for successor in successors])
        scores = [policy_scores[action] for action in actions]
        policy_top = max(range(len(actions)), key=scores.__getitem__)
        value_top = max(range(len(actions)), key=values.__getitem__)
        top3 = set(sorted(range(len(actions)), key=scores.__getitem__, reverse=True)[:3])
        value_ranks = _rank(values)
        denominator = max(len(actions) - 1, 1)
        records.append({
            "stratum": ":".join(map(str, stratum)),
            "spearman": _correlation(_rank(scores), value_ranks),
            "policy_top_value_percentile": value_ranks[policy_top] / denominator,
            "value_best_in_policy_top3": value_top in top3,
            "legal_actions": len(actions),
        })
    by_stratum = {}
    for label in sorted({row["stratum"] for row in records}):
        rows = [row for row in records if row["stratum"] == label]
        by_stratum[label] = {
            "states": len(rows),
            "mean_spearman": statistics.fmean(row["spearman"] for row in rows),
            "mean_policy_top_value_percentile": statistics.fmean(
                row["policy_top_value_percentile"] for row in rows
            ),
        }
    return {
        "states": len(records),
        "mean_spearman": statistics.fmean(row["spearman"] for row in records),
        "mean_policy_top_value_percentile": statistics.fmean(
            row["policy_top_value_percentile"] for row in records
        ),
        "value_best_in_policy_top3_rate": statistics.fmean(
            float(row["value_best_in_policy_top3"]) for row in records
        ),
        "by_stratum": by_stratum,
    }


def offline_gates(result: dict[str, Any], consistency: dict[str, Any]) -> dict[str, bool]:
    candidate = result["candidate"]
    heuristic = result["heuristic"]
    candidate_ordering = result["candidate_ordering"]
    heuristic_ordering = result["heuristic_ordering"]
    relative_mae = 1.0 - float(candidate["macro_profile_seat_mae"]) / float(
        heuristic["macro_profile_seat_mae"]
    )
    comparison_labels = [
        *(f"profile:{profile}" for profile in OPPONENTS),
        *(f"phase:{phase}" for phase in PHASES),
    ]
    no_collapse = all(
        float(result["candidate_breakdowns"][label]["mae"])
        <= 1.05 * float(result["heuristic_breakdowns"][label]["mae"])
        for label in comparison_labels
    )
    mid_candidate = statistics.fmean(
        float(result["candidate_ordering_by_phase"][phase]["concordance"])
        for phase in ("15-29", "30-44")
    )
    mid_heuristic = statistics.fmean(
        float(result["heuristic_ordering_by_phase"][phase]["concordance"])
        for phase in ("15-29", "30-44")
    )
    return {
        "macro_mae_improves_at_least_15pct": relative_mae >= 0.15,
        "paired_game_mae_ci_below_zero": float(result["paired_game_error"]["ci95_high"]) < 0.0,
        "ordering_concordance_at_least_60pct": float(candidate_ordering["concordance"]) >= 0.60,
        "ordering_improves_at_least_3pt": (
            float(candidate_ordering["concordance"])
            - float(heuristic_ordering["concordance"]) >= 0.03
        ),
        "macro_ordering_correlation_improves": (
            float(candidate_ordering["macro_stratum_correlation"])
            > float(heuristic_ordering["macro_stratum_correlation"])
        ),
        "no_opponent_or_phase_mae_collapse": no_collapse,
        "useful_midgame_ordering": mid_candidate >= 0.58 and mid_candidate - mid_heuristic >= 0.02,
        "wdl_ece_at_most_10pct": float(result["calibration"]["wdl_ece10"]) <= 0.10,
        "successor_policy_value_positive_rank_correlation": float(consistency["mean_spearman"]) > 0.0,
        "successor_policy_top_percentile_at_least_55pct": float(consistency["mean_policy_top_value_percentile"]) >= 0.55,
        "successor_value_best_in_policy_top3_at_least_50pct": float(consistency["value_best_in_policy_top3_rate"]) >= 0.50,
    }


def _puct_config(seed: int) -> PuctConfig:
    return PuctConfig(
        simulations=64,
        c_puct=1.5,
        prior_temperature=1.0,
        fpu_reduction=0.1,
        max_depth=80,
        heuristic_scale=4_000.0,
        seed=seed,
        inference_batch_size=8,
        reuse_tree=False,
    )


def _play_search_pair(policy_path: str, value_path: str, opponent_kind: str, seed: int, device: str):
    torch.set_num_threads(1)
    learned = InformationSetPuctBot(
        policy_path,
        device=device,
        config=_puct_config(seed),
        value_checkpoint_path=value_path,
        value_prediction_mode="blend",
        value_outcome_blend_weight=0.5,
    )
    if opponent_kind == "heuristic_puct":
        opponent = InformationSetPuctBot(
            policy_path, device=device, config=_puct_config(seed + 1_000_000)
        )
    elif opponent_kind == "raw_policy":
        opponent = LegalActionAttentionScorerBot(policy_path, device=device)
    else:
        raise ValueError(opponent_kind)
    first = play_game(learned, opponent, seed=seed)
    learned = InformationSetPuctBot(
        policy_path,
        device=device,
        config=_puct_config(seed + 2_000_000),
        value_checkpoint_path=value_path,
        value_prediction_mode="blend",
        value_outcome_blend_weight=0.5,
    )
    if opponent_kind == "heuristic_puct":
        opponent = InformationSetPuctBot(
            policy_path, device=device, config=_puct_config(seed + 3_000_000)
        )
    else:
        opponent = LegalActionAttentionScorerBot(policy_path, device=device)
    second = play_game(opponent, learned, seed=seed)
    return {"seed": seed, "candidate_seats": [0, 1], "games": [asdict(first), asdict(second)]}


def _run_search_gate(config, checkpoint_path: Path, opponent_kind: str, seed_start: int, name: str):
    output_path = config.output_dir / f"{name}.json"
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    rows = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                _play_search_pair,
                str(config.policy_checkpoint), str(checkpoint_path), opponent_kind,
                seed, config.device,
            ): seed
            for seed in range(seed_start, seed_start + config.puct_pairs)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if completed == 1 or completed % 10 == 0 or completed == config.puct_pairs:
                print(f"[search-gate:{name}] {completed}/{config.puct_pairs}")
    rows.sort(key=lambda row: int(row["seed"]))
    summary = summarize_pair_rows(rows, config.bootstrap_samples, seed_start)
    gates = {
        "zero_stalls": int(summary["stalled_games"]) == 0,
        "zero_timeouts": int(summary["timed_out_games"]) == 0,
        "score_share_at_least_55pct": float(summary["score_share"]["mean"]) >= 0.55,
        "score_share_ci95_low_above_50pct": float(summary["score_share"]["ci95_low"]) > 0.50,
    }
    result = {"opponent": opponent_kind, "summary": summary, "gates": gates, "passed": all(gates.values()), "pair_rows": rows}
    _atomic_json(output_path, result)
    return result


def _protocol(config: ScaledPublicBeliefValueConfig) -> dict[str, Any]:
    return {
        "experiment": "scaled_public_belief_value_036",
        "frozen_at": "before tensor compilation, training, or fresh confirmation evidence",
        "sources": {
            "032": {"path": str(config.source_032), "train_groups": [0, 447], "validation_groups": [448, 511]},
            "033": {"path": str(config.source_033), "train_groups": [0, 447], "validation_groups": [448, 511]},
        },
        "split_contract": "environment-seed groups are disjoint; 56 train and 8 validation groups per generation/opponent/seat cell",
        "sampling_contract": "every training epoch exactly balances generation/opponent/actor-seat/phase (64 strata)",
        "observation_contract": "canonical_public_belief_v1; acting-player-visible entities only",
        "initialization": {
            "path": str(config.actor_parent_checkpoint),
            "sha256": _sha256(config.actor_parent_checkpoint),
            "scope": "all canonical entity embeddings/input/transformer tensors; distribution heads random",
        },
        "policy": {"path": str(config.policy_checkpoint), "sha256": _sha256(config.policy_checkpoint)},
        "targets": {
            "margin": "41-way categorical clipped integer final-score margin [-20,20]",
            "wdl": "three-way categorical loss/draw/win",
            "loss": f"margin CE + {config.wdl_loss_weight} * WDL CE",
            "search_value": f"{1-config.search_wdl_blend_weight:.1f} margin expectation + {config.search_wdl_blend_weight:.1f} WDL expectation",
        },
        "selection": "lowest validation macro-stratum calibration NLL minus 0.25 * within-stratum ordering concordance; confirmation remains unseen",
        "confirmation": {
            "groups": config.confirmation_groups,
            "games": config.confirmation_groups * 4,
            "seed_start": config.confirmation_seed_start,
            "balance": "16 groups per opponent/actor-seat cell, four stochastic-035 trajectories per group",
            "created_only_after_selection": True,
        },
        "offline_gates": {
            "macro_mae_improvement": 0.15,
            "paired_game_mae_bootstrap_ci95_high": "<0",
            "ordering_concordance": ">=0.60 and >= heuristic+0.03",
            "macro_ordering_correlation": ">heuristic",
            "profile_phase_mae": "no cell aggregate worse than heuristic by >5%",
            "midgame_ordering": ">=0.58 and >= heuristic+0.02 over turns 15-44",
            "wdl_ece10": "<=0.10",
            "successor_consistency": "Spearman>0, policy-top value percentile>=0.55, value-best in policy top3>=0.50",
        },
        "conditional_game_gates": {
            "learned_vs_heuristic_puct64": {"pairs": config.puct_pairs, "seed_start": config.puct_ab_seed_start, "share": ">=0.55", "ci95_low": ">0.50"},
            "learned_puct64_vs_raw035": {"pairs": config.puct_pairs, "seed_start": config.puct_direct_seed_start, "share": ">=0.55", "ci95_low": ">0.50"},
        },
        "alpha_loop": "Only a direct-search pass authorizes a separately checkpointed synchronous search-every-move visit-distribution generation/distillation cycle; wins and losses are both retained.",
        "config": {
            **asdict(config),
            **{
                key: str(value)
                for key, value in asdict(config).items()
                if isinstance(value, Path)
            },
        },
    }


def run_scaled_public_belief_value(config: ScaledPublicBeliefValueConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing protocol differs from the preregistered contract.")
    else:
        _atomic_json(protocol_path, protocol)

    train_payload = _compile_tensor_corpus(
        (config.source_032, config.source_033), "train",
        config.output_dir / "train_public_tensors.pt",
    )
    validation_payload = _compile_tensor_corpus(
        (config.source_032, config.source_033), "validation",
        config.output_dir / "validation_public_tensors.pt",
    )
    balanced = ExactlyBalancedPhaseDataset(train_payload)
    parent = torch.load(config.actor_parent_checkpoint, map_location="cpu", weights_only=True)
    if parent.get("model_kind") != "canonical_entity_action_scorer":
        raise ValueError("Representation parent must be a canonical entity action scorer.")
    model_config = _model_config_from_parent(parent)
    torch.manual_seed(config.seed)
    model = CanonicalEntityDistributionalValueModel(model_config, MARGIN_BINS)
    transferred = initialize_from_actor(model, parent["model_state_dict"])
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        balanced,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    history = []
    best_state = None
    best_epoch = None
    best_score = math.inf
    for epoch in range(config.epochs):
        train_metrics = _train_epoch(model, loader, optimizer, config)
        validation = evaluate_model(
            model, validation_payload, config, bootstrap=False, seed=config.seed + epoch
        )
        score = float(validation["selection"]["score"])
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation})
        print(
            f"[scaled-value] epoch={epoch+1}/{config.epochs}"
            f" margin_nll={train_metrics['margin_nll']:.5f}"
            f" wdl_nll={train_metrics['wdl_nll']:.5f}"
            f" validation_selection={score:.5f}"
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None or best_epoch is None:
        raise RuntimeError("No value checkpoint was selected.")
    model.load_state_dict(best_state)
    checkpoint_path = config.output_dir / "canonical_entity_distributional_value_width64_best.pt"
    torch.save({
        "model_kind": "canonical_entity_distributional_value",
        "input_contract": "canonical_public_belief_v1",
        "model_config": asdict(model_config),
        "margin_bins": MARGIN_BINS,
        "margin_clip": MARGIN_CLIP,
        "search_wdl_blend_weight": config.search_wdl_blend_weight,
        "model_state_dict": best_state,
        "representation_parent": str(config.actor_parent_checkpoint),
        "representation_parent_sha256": _sha256(config.actor_parent_checkpoint),
        "selected_epoch": best_epoch,
    }, checkpoint_path)
    selected_validation = evaluate_model(
        model, validation_payload, config, bootstrap=True, seed=config.seed + 100
    )
    selection_payload = {
        "selected_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "history": history,
        "selected_validation": selected_validation,
        "train_rows": int(train_payload["margin_class"].shape[0]),
        "validation_rows": int(validation_payload["margin_class"].shape[0]),
        "balanced_rows_per_epoch": len(balanced),
        "balanced_rows_per_stratum": balanced.per_stratum,
        "transferred_encoder_tensors": list(transferred),
    }
    _atomic_json(config.output_dir / "selection.json", selection_payload)

    collection = _confirmation_collection(config)
    if not collection["passed"]:
        result = {"decision": "fresh_confirmation_collection_failed", "selection": selection_payload, "collection": collection}
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result
    confirmation_payload = _compile_tensor_corpus(
        (config.confirmation_data_dir,), "confirmation",
        config.output_dir / "confirmation_public_tensors.pt",
    )
    confirmation = evaluate_model(
        model, confirmation_payload, config, bootstrap=True, seed=config.seed + 200
    )
    consistency = evaluate_successor_consistency(config, checkpoint_path)
    gates = offline_gates(confirmation, consistency)
    offline = {"confirmation": confirmation, "successor_consistency": consistency, "gates": gates, "passed": all(gates.values())}
    _atomic_json(config.output_dir / "offline_gate.json", offline)
    if not offline["passed"]:
        decision = "reject_offline_keep_035"
        result = {"decision": decision, "promoted": False, "selection": selection_payload, "collection": collection, "offline": offline, "elapsed_seconds": time.perf_counter() - started}
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result

    puct_ab = _run_search_gate(
        config, checkpoint_path, "heuristic_puct", config.puct_ab_seed_start,
        "learned_value_puct64_vs_heuristic_puct64",
    )
    if not puct_ab["passed"]:
        result = {"decision": "reject_search_ab_keep_035", "promoted": False, "selection": selection_payload, "collection": collection, "offline": offline, "puct_ab": puct_ab, "elapsed_seconds": time.perf_counter() - started}
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result
    direct = _run_search_gate(
        config, checkpoint_path, "raw_policy", config.puct_direct_seed_start,
        "learned_value_puct64_vs_raw035",
    )
    decision = (
        "authorize_synchronous_search_visit_distillation"
        if direct["passed"] else "reject_direct_search_keep_035"
    )
    result = {
        "decision": decision,
        "promoted": False,
        "search_teacher_authorized": direct["passed"],
        "selection": selection_payload,
        "collection": collection,
        "offline": offline,
        "puct_ab": puct_ab,
        "direct": direct,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(config.output_dir / "gate_final.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-032", default="data/league_generation_032")
    parser.add_argument("--source-033", default="data/trust_region_generation_033")
    parser.add_argument("--actor-parent-checkpoint", default="outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt")
    parser.add_argument("--policy-checkpoint", default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt")
    parser.add_argument("--checkpoint-016", default="outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt")
    parser.add_argument("--checkpoint-023", default="outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt")
    parser.add_argument("--output-dir", default="outputs/scaled_public_belief_value_036")
    parser.add_argument("--confirmation-data-dir", default="data/scaled_public_belief_value_confirmation_036")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> ScaledPublicBeliefValueConfig:
    return ScaledPublicBeliefValueConfig(
        source_032=Path(args.source_032), source_033=Path(args.source_033),
        actor_parent_checkpoint=Path(args.actor_parent_checkpoint),
        policy_checkpoint=Path(args.policy_checkpoint), checkpoint_016=Path(args.checkpoint_016),
        checkpoint_023=Path(args.checkpoint_023), output_dir=Path(args.output_dir),
        confirmation_data_dir=Path(args.confirmation_data_dir), epochs=args.epochs,
        batch_size=args.batch_size, workers=args.workers, device=args.device,
    )


def main() -> None:
    result = run_scaled_public_belief_value(config_from_args(build_parser().parse_args()))
    print(json.dumps({key: result.get(key) for key in ("decision", "promoted", "search_teacher_authorized")}, indent=2))


if __name__ == "__main__":
    main()

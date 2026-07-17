"""Train and gate the final start-player-aware public-belief value experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from splendor_ai.bots.puct_bot import legal_heuristic_value
from splendor_ai.bots.search_bot import CheckpointLegalActionPriorScorer
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    CanonicalEntityDistributionalValueModel,
    CanonicalEntityDistributionalValueV2Model,
    CanonicalEntityValueConfig,
)
from splendor_ai.training.run_league_generation import (
    LeagueGenerationConfig,
    SamplingSpec,
    collect_league_groups,
)
from splendor_ai.training.run_reliability_blend_value import (
    BlendSpec,
    ReliabilityBlendConfig,
    _blend_metrics,
    _blend_values,
    _predict_payload,
    _reservoir_states,
)
from splendor_ai.training.run_scaled_public_belief_value import (
    ExactlyBalancedPhaseDataset,
    MARGIN_CLIP,
    OPPONENTS,
    PHASES,
    ScaledPublicBeliefValueConfig,
    TensorValueDataset,
    _compile_tensor_corpus,
    _ece,
    _model_inputs,
    _rank,
    _run_search_gate,
    offline_gates,
)
from splendor_ai.training.run_entity_public_value import _correlation, phase_for_turn


DEPLOY_SPEC = BlendSpec("v2_ramp", (0.0, 0.5, 0.75, 1.0), 0.0)


@dataclass(frozen=True, slots=True)
class StartRoleValueV2Config:
    base_value_checkpoint: Path
    train_tensors: Path
    validation_tensors: Path
    source_032: Path
    source_033: Path
    policy_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    output_dir: Path
    confirmation_data_dir: Path
    epochs: int = 6
    batch_size: int = 128
    base_learning_rate: float = 3e-5
    start_role_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 2_037_0717
    workers: int = 4
    bootstrap_samples: int = 20_000
    device: str = "cpu"
    minimum_validation_selection_improvement: float = 0.005
    minimum_seat_mae_improvement: float = 0.01
    confirmation_groups: int = 128
    confirmation_seed_start: int = 384_000
    confirmation_sampling_seed_start: int = 43_000_000
    puct_pairs: int = 100
    puct_ab_seed_start: int = 385_000
    puct_direct_seed_start: int = 386_000


class StartRoleBalancedDataset(ExactlyBalancedPhaseDataset):
    def __getitem__(self, index: int):
        result = super().__getitem__(index)
        row = int(self.indices[index])
        # All engine games start at seat 0; deployment derives this from state.
        result["observer_is_start_player"] = (
            self.payload["actor_seat"][row] == 0
        ).long()
        return result


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


def _load_models(config):
    checkpoint = torch.load(
        config.base_value_checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint.get("model_kind") != "canonical_entity_distributional_value":
        raise ValueError("v2 must warm-start from frozen distributional value 036.")
    model_config = CanonicalEntityValueConfig(**checkpoint["model_config"])
    v1 = CanonicalEntityDistributionalValueModel(
        model_config, margin_bins=int(checkpoint["margin_bins"])
    )
    v1.load_state_dict(checkpoint["model_state_dict"])
    v1.to(config.device).eval()
    v2 = CanonicalEntityDistributionalValueV2Model(
        model_config, margin_bins=int(checkpoint["margin_bins"])
    )
    missing, unexpected = v2.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if set(missing) != {"start_role_embedding.weight"} or unexpected:
        raise ValueError(
            f"Unexpected v2 warm-start mismatch: missing={missing}, unexpected={unexpected}"
        )
    v2.to(config.device)
    return v1, v2, checkpoint


def _v2_inputs(batch, device):
    result = _model_inputs(batch, device)
    result["observer_is_start_player"] = batch[
        "observer_is_start_player"
    ].to(device)
    return result


def _train_epoch(model, loader, optimizer, config):
    model.train()
    margin_total = 0.0
    wdl_total = 0.0
    count = 0
    for batch in loader:
        margin_logits, wdl_logits = model(**_v2_inputs(batch, config.device))
        margin_target = batch["margin_class"].to(config.device)
        wdl_target = batch["wdl_class"].to(config.device)
        margin_loss = F.cross_entropy(margin_logits, margin_target)
        wdl_loss = F.cross_entropy(wdl_logits, wdl_target)
        loss = margin_loss + 0.5 * wdl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        samples = int(margin_target.numel())
        margin_total += float(margin_loss.detach()) * samples
        wdl_total += float(wdl_loss.detach()) * samples
        count += samples
    return {
        "margin_nll": margin_total / count,
        "wdl_nll": wdl_total / count,
        "samples": count,
    }


def _predict_v2(model, payload, config):
    loader = DataLoader(
        TensorValueDataset(payload), batch_size=config.batch_size, shuffle=False
    )
    learned_parts, confidence_parts = [], []
    margin_logits_parts, wdl_logits_parts = [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            size = int(batch["margin_class"].numel())
            start_role = (payload["actor_seat"][offset : offset + size] == 0).long()
            batch["observer_is_start_player"] = start_role
            margin_logits, wdl_logits = model(**_v2_inputs(batch, config.device))
            margin, wdl = model.expected_values(margin_logits, wdl_logits)
            learned_parts.append((0.5 * margin + 0.5 * wdl).cpu())
            confidence_parts.append(
                torch.softmax(wdl_logits, dim=-1).max(dim=-1).values.cpu()
            )
            margin_logits_parts.append(margin_logits.cpu())
            wdl_logits_parts.append(wdl_logits.cpu())
            offset += size
    return (
        torch.cat(learned_parts),
        torch.cat(confidence_parts),
        torch.cat(margin_logits_parts),
        torch.cat(wdl_logits_parts),
    )


def _calibration_and_selection(payload, margin_logits, wdl_logits, ordering):
    margin_target = payload["margin_class"].long()
    wdl_target = payload["wdl_class"].long()
    wdl_probability = torch.softmax(wdl_logits, dim=1)
    one_hot = F.one_hot(wdl_target, num_classes=3).float()
    calibration = {
        "margin_nll": float(F.cross_entropy(margin_logits, margin_target)),
        "wdl_nll": float(F.cross_entropy(wdl_logits, wdl_target)),
        "wdl_brier": float(
            (wdl_probability - one_hot).square().sum(dim=1).mean()
        ),
        "wdl_ece10": _ece(wdl_probability, wdl_target),
    }
    strata_nll = []
    for source in range(2):
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
                        strata_nll.append(
                            float(
                                F.cross_entropy(
                                    margin_logits[mask], margin_target[mask]
                                )
                            )
                            + 0.5
                            * float(
                                F.cross_entropy(wdl_logits[mask], wdl_target[mask])
                            )
                        )
    macro_nll = statistics.fmean(strata_nll)
    return calibration, {
        "macro_calibration_nll": macro_nll,
        "ordering_concordance": ordering,
        "score": macro_nll - 0.25 * ordering,
    }


def evaluate_v2(model, payload, config, *, bootstrap, seed):
    learned, confidence, margin_logits, wdl_logits = _predict_v2(
        model, payload, config
    )
    preliminary_calibration = {
        "margin_nll": 0.0,
        "wdl_nll": 0.0,
        "wdl_brier": 0.0,
        "wdl_ece10": 0.0,
    }
    result = _blend_metrics(
        payload,
        _blend_values(payload, learned, confidence, DEPLOY_SPEC),
        preliminary_calibration,
        config,
        bootstrap=bootstrap,
        seed=seed,
    )
    calibration, selection = _calibration_and_selection(
        payload,
        margin_logits,
        wdl_logits,
        float(result["candidate_ordering"]["concordance"]),
    )
    result["calibration"] = calibration
    result["selection"] = selection
    return result


def _state_components_v2(model, states_and_players, device):
    from splendor_ai.encoding import collate_canonical_state_batch

    batch = collate_canonical_state_batch(states_and_players, device=device)
    batch["observer_is_start_player"] = torch.tensor(
        [state.start_player == player for state, player in states_and_players],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        margin_logits, wdl_logits = model(**batch)
        margin, wdl = model.expected_values(margin_logits, wdl_logits)
        confidence = torch.softmax(wdl_logits, dim=-1).max(dim=-1).values
    return (
        (0.5 * margin + 0.5 * wdl).cpu().tolist(),
        confidence.cpu().tolist(),
    )


def evaluate_v2_successors(config, model, sampled_states):
    scorer = CheckpointLegalActionPriorScorer(
        config.policy_checkpoint, device=config.device
    )
    env = SplendorEnv(seed=config.seed)
    records = []
    for stratum, state, actor in sampled_states:
        actions = env.legal_actions(state)
        if len(actions) < 2:
            continue
        score_map = scorer.score_actions(state, actions)
        policy_scores = [score_map[action] for action in actions]
        successors = [env.step(state, action) for action in actions]
        learned, confidence = _state_components_v2(
            model, [(successor, actor) for successor in successors], config.device
        )
        phase = PHASES.index(phase_for_turn(successors[0].turn_index))
        weight = DEPLOY_SPEC.phase_learned_weights[phase]
        heuristics = [
            legal_heuristic_value(successor, actor, scale=4_000.0)
            for successor in successors
        ]
        values = [
            weight * learned_value + (1.0 - weight) * heuristic
            for learned_value, heuristic, _confidence in zip(
                learned, heuristics, confidence, strict=True
            )
        ]
        policy_top = max(range(len(actions)), key=policy_scores.__getitem__)
        value_top = max(range(len(actions)), key=values.__getitem__)
        policy_top3 = set(
            sorted(
                range(len(actions)), key=policy_scores.__getitem__, reverse=True
            )[:3]
        )
        value_ranks = _rank(values)
        records.append(
            {
                "stratum": ":".join(map(str, stratum)),
                "spearman": _correlation(_rank(policy_scores), value_ranks),
                "policy_top_value_percentile": value_ranks[policy_top]
                / max(len(actions) - 1, 1),
                "value_best_in_policy_top3": value_top in policy_top3,
            }
        )
    by_stratum = {}
    for label in sorted({row["stratum"] for row in records}):
        rows = [row for row in records if row["stratum"] == label]
        by_stratum[label] = {
            "states": len(rows),
            "mean_spearman": statistics.fmean(row["spearman"] for row in rows),
            "mean_policy_top_value_percentile": statistics.fmean(
                row["policy_top_value_percentile"] for row in rows
            ),
            "value_best_in_policy_top3_rate": statistics.fmean(
                float(row["value_best_in_policy_top3"]) for row in rows
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


def _base_v1_metrics(v1_model, payload, calibration, config, *, bootstrap, seed):
    learned, confidence = _predict_payload(v1_model, payload, config)
    return _blend_metrics(
        payload,
        _blend_values(payload, learned, confidence, DEPLOY_SPEC),
        calibration,
        config,
        bootstrap=bootstrap,
        seed=seed,
    )


def _seat_improvement_gates(config, candidate, base):
    gates = {}
    for seat in (0, 1):
        label = f"seat:{seat}"
        candidate_mae = float(candidate["candidate_breakdowns"][label]["mae"])
        base_mae = float(base["candidate_breakdowns"][label]["mae"])
        gates[f"seat{seat}_mae_improves_at_least_1pct_vs_v1"] = (
            candidate_mae <= (1.0 - config.minimum_seat_mae_improvement) * base_mae
        )
        gates[f"seat{seat}_correlation_nonregression_vs_v1"] = (
            float(candidate["candidate_breakdowns"][label]["correlation"])
            >= float(base["candidate_breakdowns"][label]["correlation"])
        )
    return gates


def _confirmation_collection(config):
    league = LeagueGenerationConfig(
        output_dir=config.output_dir / "confirmation_collection",
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
        league,
        SamplingSpec(
            epsilon=0.5,
            temperature=1.0,
            top_k=3,
            expected_nongreedy_rate=0.15,
        ),
    )


def _deployment_checkpoint(config, base_checkpoint, model, selected_epoch, path):
    torch.save(
        {
            "model_kind": "canonical_entity_reliability_blend_v2",
            "input_contract": "canonical_public_belief_v2_start_role",
            "model_config": base_checkpoint["model_config"],
            "margin_bins": base_checkpoint["margin_bins"],
            "margin_clip": base_checkpoint["margin_clip"],
            "search_wdl_blend_weight": 0.5,
            "phase_learned_weights": list(DEPLOY_SPEC.phase_learned_weights),
            "wdl_confidence_floor": 0.0,
            "model_state_dict": model.state_dict(),
            "base_value_checkpoint": str(config.base_value_checkpoint),
            "base_value_checkpoint_sha256": _sha256(config.base_value_checkpoint),
            "selected_epoch": selected_epoch,
        },
        path,
    )


def _seat_search_gate(result):
    seat_points = {0: [], 1: []}
    for row in result["pair_rows"]:
        for game, seat_value in zip(
            row["games"], row["candidate_seats"], strict=True
        ):
            seat = int(seat_value)
            winner = game["winner"]
            point = 0.5 if winner is None else 1.0 if int(winner) == seat else 0.0
            seat_points[seat].append(point)
    shares = {
        f"seat{seat}": statistics.fmean(values)
        for seat, values in seat_points.items()
    }
    result["seat_score_share"] = shares
    result["gates"]["candidate_seat0_share_at_least_50pct"] = (
        shares["seat0"] >= 0.50
    )
    result["gates"]["candidate_seat1_share_at_least_50pct"] = (
        shares["seat1"] >= 0.50
    )
    result["passed"] = all(result["gates"].values())
    return result


def _protocol(config):
    serialized = {
        **asdict(config),
        **{
            key: str(value)
            for key, value in asdict(config).items()
            if isinstance(value, Path)
        },
    }
    return {
        "experiment": "start_role_public_belief_value_v2_037",
        "frozen_before_training_or_fresh_evidence": True,
        "representation": {
            "v1_preserved": True,
            "new_input": "observer_is_start_player",
            "mechanism": "zero-initialized two-value embedding added only to the global entity before the transformer",
            "initial_prediction_identity": "exactly equal to frozen 036 before training",
        },
        "warm_start": {
            "path": str(config.base_value_checkpoint),
            "sha256": _sha256(config.base_value_checkpoint),
        },
        "data": {
            "train_tensors": str(config.train_tensors),
            "validation_tensors": str(config.validation_tensors),
            "splits": "unchanged 032/033 groups 0-447 train and 448-511 validation",
            "start_role_derivation": "engine start_player is always seat0; deployment compares state.start_player to observer",
            "balance": "exact generation/opponent/seat/phase cycling",
        },
        "training": {
            "epochs": config.epochs,
            "base_learning_rate": config.base_learning_rate,
            "start_role_learning_rate": config.start_role_learning_rate,
            "loss": "margin CE + 0.5 WDL CE",
            "epoch0_is_candidate": True,
            "deployment_blend": {
                "name": DEPLOY_SPEC.name,
                "phase_learned_weights": list(DEPLOY_SPEC.phase_learned_weights),
                "wdl_confidence_floor": DEPLOY_SPEC.wdl_confidence_floor,
            },
        },
        "validation": {
            "selection": "lowest macro calibration NLL - 0.25 ordering concordance",
            "minimum_selection_improvement_vs_epoch0": config.minimum_validation_selection_improvement,
            "unchanged_036_gates": True,
            "seat_mae_improvement_vs_v1": config.minimum_seat_mae_improvement,
            "seat_correlation_nonregression_vs_v1": True,
            "successor_sample": "four states per generation/opponent/seat/phase stratum",
        },
        "confirmation": {
            "seed_start": config.confirmation_seed_start,
            "groups": config.confirmation_groups,
            "games": config.confirmation_groups * 4,
            "generated_only_after_validation_pass": True,
            "same_original_and_seat_specific_gates": True,
        },
        "conditional_puct": {
            "ab": {
                "seed_start": config.puct_ab_seed_start,
                "pairs": config.puct_pairs,
                "overall_share": ">=0.55",
                "ci95_low": ">0.50",
                "each_candidate_seat_share": ">=0.50",
            },
            "raw035": {
                "seed_start": config.puct_direct_seed_start,
                "pairs": config.puct_pairs,
                "same_gates": True,
            },
        },
        "terminal_rule": "If this v2 model fails fresh offline or seat-symmetric search gates, close the current value/PUCT bootstrap and move to hierarchical plans/options.",
        "config": serialized,
    }


def run_start_role_value_v2(config: StartRoleValueV2Config):
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing v2 protocol differs from frozen contract.")
    else:
        _atomic_json(protocol_path, protocol)

    train_payload = torch.load(
        config.train_tensors, map_location="cpu", weights_only=True, mmap=True
    )
    validation_payload = torch.load(
        config.validation_tensors, map_location="cpu", weights_only=True, mmap=True
    )
    if not bool((train_payload["actor_seat"] == 0).any()) or not bool(
        (train_payload["actor_seat"] == 1).any()
    ):
        raise ValueError("v2 training requires both start-role classes.")
    v1, model, base_checkpoint = _load_models(config)
    balanced = StartRoleBalancedDataset(train_payload)
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "start_role_embedding.weight"
    ]
    optimizer = AdamW(
        [
            {
                "params": base_parameters,
                "lr": config.base_learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [model.start_role_embedding.weight],
                "lr": config.start_role_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    loader = DataLoader(
        balanced,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    epoch0 = evaluate_v2(
        model,
        validation_payload,
        config,
        bootstrap=True,
        seed=config.seed,
    )
    history = [{"epoch": 0, "train": None, "validation": epoch0}]
    best_epoch = 0
    best_score = float(epoch0["selection"]["score"])
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(model, loader, optimizer, config)
        validation = evaluate_v2(
            model,
            validation_payload,
            config,
            bootstrap=True,
            seed=config.seed + epoch,
        )
        score = float(validation["selection"]["score"])
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation}
        )
        print(
            f"[start-role-v2] epoch={epoch}/{config.epochs}"
            f" margin_nll={train_metrics['margin_nll']:.5f}"
            f" wdl_nll={train_metrics['wdl_nll']:.5f}"
            f" validation_selection={score:.5f}"
        )
        if score < best_score:
            best_epoch = epoch
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected_validation = evaluate_v2(
        model,
        validation_payload,
        config,
        bootstrap=True,
        seed=config.seed + 100,
    )
    base_validation = _base_v1_metrics(
        v1,
        validation_payload,
        selected_validation["calibration"],
        config,
        bootstrap=True,
        seed=config.seed + 101,
    )
    validation_states = _reservoir_states(
        (config.source_032, config.source_033),
        range(448, 512),
        per_stratum=4,
        seed=config.seed + 102,
    )
    validation_consistency = evaluate_v2_successors(
        config, model, validation_states
    )
    validation_gates = offline_gates(
        selected_validation, validation_consistency
    )
    validation_gates.update(
        _seat_improvement_gates(
            config, selected_validation, base_validation
        )
    )
    validation_gates["selection_improves_at_least_0.5pct_vs_epoch0"] = (
        best_score
        <= (1.0 - config.minimum_validation_selection_improvement)
        * float(epoch0["selection"]["score"])
    )
    validation_result = {
        "selected_epoch": best_epoch,
        "epoch0_score": float(epoch0["selection"]["score"]),
        "selected_score": best_score,
        "history": history,
        "selected": selected_validation,
        "v1_base": base_validation,
        "successor_consistency": validation_consistency,
        "gates": validation_gates,
        "passed": all(validation_gates.values()),
    }
    _atomic_json(config.output_dir / "validation_selection.json", validation_result)
    if not validation_result["passed"]:
        result = {
            "decision": "reject_v2_validation_close_value_puct",
            "promoted": False,
            "validation": validation_result,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result

    checkpoint_path = config.output_dir / "canonical_entity_start_role_value_v2.pt"
    _deployment_checkpoint(
        config, base_checkpoint, model, best_epoch, checkpoint_path
    )
    collection = _confirmation_collection(config)
    if not collection["passed"]:
        result = {
            "decision": "fresh_confirmation_collection_failed",
            "promoted": False,
            "validation": validation_result,
            "collection": collection,
        }
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result
    confirmation_payload = _compile_tensor_corpus(
        (config.confirmation_data_dir,),
        "confirmation",
        config.output_dir / "confirmation_public_tensors.pt",
    )
    confirmation = evaluate_v2(
        model,
        confirmation_payload,
        config,
        bootstrap=True,
        seed=config.seed + 200,
    )
    base_confirmation = _base_v1_metrics(
        v1,
        confirmation_payload,
        confirmation["calibration"],
        config,
        bootstrap=True,
        seed=config.seed + 201,
    )
    confirmation_states = _reservoir_states(
        (config.confirmation_data_dir,),
        range(config.confirmation_groups),
        per_stratum=8,
        seed=config.seed + 202,
    )
    consistency = evaluate_v2_successors(config, model, confirmation_states)
    confirmation_gates = offline_gates(confirmation, consistency)
    confirmation_gates.update(
        _seat_improvement_gates(config, confirmation, base_confirmation)
    )
    offline = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "confirmation": confirmation,
        "v1_base": base_confirmation,
        "successor_consistency": consistency,
        "gates": confirmation_gates,
        "passed": all(confirmation_gates.values()),
    }
    _atomic_json(config.output_dir / "offline_gate.json", offline)
    if not offline["passed"]:
        result = {
            "decision": "reject_v2_fresh_offline_close_value_puct",
            "promoted": False,
            "validation": validation_result,
            "collection": collection,
            "offline": offline,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result

    puct_config = ScaledPublicBeliefValueConfig(
        source_032=config.source_032,
        source_033=config.source_033,
        actor_parent_checkpoint=config.checkpoint_023,
        policy_checkpoint=config.policy_checkpoint,
        checkpoint_016=config.checkpoint_016,
        checkpoint_023=config.checkpoint_023,
        output_dir=config.output_dir,
        confirmation_data_dir=config.confirmation_data_dir,
        workers=config.workers,
        bootstrap_samples=config.bootstrap_samples,
        device=config.device,
        puct_pairs=config.puct_pairs,
    )
    puct_ab = _seat_search_gate(
        _run_search_gate(
            puct_config,
            checkpoint_path,
            "heuristic_puct",
            config.puct_ab_seed_start,
            "start_role_v2_puct64_vs_heuristic_puct64",
        )
    )
    _atomic_json(
        config.output_dir / "start_role_v2_puct64_vs_heuristic_puct64.json",
        puct_ab,
    )
    if not puct_ab["passed"]:
        result = {
            "decision": "reject_v2_search_ab_close_value_puct",
            "promoted": False,
            "validation": validation_result,
            "collection": collection,
            "offline": offline,
            "puct_ab": puct_ab,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result
    direct = _seat_search_gate(
        _run_search_gate(
            puct_config,
            checkpoint_path,
            "raw_policy",
            config.puct_direct_seed_start,
            "start_role_v2_puct64_vs_raw035",
        )
    )
    _atomic_json(
        config.output_dir / "start_role_v2_puct64_vs_raw035.json", direct
    )
    result = {
        "decision": (
            "authorize_synchronous_search_visit_distillation"
            if direct["passed"]
            else "reject_v2_direct_close_value_puct"
        ),
        "promoted": False,
        "search_teacher_authorized": direct["passed"],
        "validation": validation_result,
        "collection": collection,
        "offline": offline,
        "puct_ab": puct_ab,
        "direct": direct,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(config.output_dir / "gate_final.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-value-checkpoint", default="outputs/scaled_public_belief_value_036/canonical_entity_distributional_value_width64_best.pt")
    parser.add_argument("--train-tensors", default="outputs/scaled_public_belief_value_036/train_public_tensors.pt")
    parser.add_argument("--validation-tensors", default="outputs/scaled_public_belief_value_036/validation_public_tensors.pt")
    parser.add_argument("--source-032", default="data/league_generation_032")
    parser.add_argument("--source-033", default="data/trust_region_generation_033")
    parser.add_argument("--policy-checkpoint", default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt")
    parser.add_argument("--checkpoint-016", default="outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt")
    parser.add_argument("--checkpoint-023", default="outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt")
    parser.add_argument("--output-dir", default="outputs/start_role_public_belief_value_v2_037")
    parser.add_argument("--confirmation-data-dir", default="data/start_role_public_belief_value_v2_confirmation_037")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args):
    return StartRoleValueV2Config(
        base_value_checkpoint=Path(args.base_value_checkpoint),
        train_tensors=Path(args.train_tensors),
        validation_tensors=Path(args.validation_tensors),
        source_032=Path(args.source_032),
        source_033=Path(args.source_033),
        policy_checkpoint=Path(args.policy_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_023=Path(args.checkpoint_023),
        output_dir=Path(args.output_dir),
        confirmation_data_dir=Path(args.confirmation_data_dir),
        epochs=args.epochs,
        workers=args.workers,
        device=args.device,
    )


def main():
    result = run_start_role_value_v2(config_from_args(build_parser().parse_args()))
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in ("decision", "promoted", "search_teacher_authorized")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

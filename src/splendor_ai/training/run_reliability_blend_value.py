"""Select and gate a phase/confidence reliability blend for value model 036."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from splendor_ai.bots.puct_bot import legal_heuristic_value
from splendor_ai.bots.search_bot import CheckpointLegalActionPriorScorer
from splendor_ai.engine.env import SplendorEnv
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
from splendor_ai.training.run_scaled_public_belief_value import (
    MARGIN_CLIP,
    OPPONENTS,
    PHASES,
    ScaledPublicBeliefValueConfig,
    TensorValueDataset,
    _compile_tensor_corpus,
    _model_inputs,
    _ordering_by_phase,
    _rank,
    _run_search_gate,
    evaluate_model,
    offline_gates,
)


@dataclass(frozen=True, slots=True)
class BlendSpec:
    name: str
    phase_learned_weights: tuple[float, float, float, float]
    wdl_confidence_floor: float


PHASE_PROFILES = (
    ("opening_heuristic", (0.0, 1.0, 1.0, 1.0)),
    ("ramp", (0.0, 0.5, 0.75, 1.0)),
    ("fast_ramp", (0.0, 0.75, 1.0, 1.0)),
    ("soft_open", (0.25, 0.75, 1.0, 1.0)),
)
CONFIDENCE_FLOORS = (0.0, 0.45, 0.55)
CANDIDATE_GRID = tuple(
    BlendSpec(f"{name}_conf{floor:.2f}", weights, floor)
    for name, weights in PHASE_PROFILES
    for floor in CONFIDENCE_FLOORS
)


@dataclass(frozen=True, slots=True)
class ReliabilityBlendConfig:
    base_value_checkpoint: Path
    validation_tensors: Path
    source_032: Path
    source_033: Path
    policy_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    output_dir: Path
    confirmation_data_dir: Path
    seed: int = 2_036_1717
    workers: int = 4
    batch_size: int = 128
    bootstrap_samples: int = 20_000
    device: str = "cpu"
    confirmation_groups: int = 128
    confirmation_seed_start: int = 381_000
    confirmation_sampling_seed_start: int = 42_000_000
    puct_pairs: int = 100
    puct_ab_seed_start: int = 382_000
    puct_direct_seed_start: int = 383_000


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


def _load_base_model(config: ReliabilityBlendConfig):
    checkpoint = torch.load(
        config.base_value_checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint.get("model_kind") != "canonical_entity_distributional_value":
        raise ValueError("036b requires the frozen distributional value checkpoint.")
    model = CanonicalEntityDistributionalValueModel(
        CanonicalEntityValueConfig(**checkpoint["model_config"]),
        margin_bins=int(checkpoint["margin_bins"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()
    return model, checkpoint


def _predict_payload(model, payload, config):
    loader = DataLoader(
        TensorValueDataset(payload), batch_size=config.batch_size, shuffle=False
    )
    learned_parts, confidence_parts = [], []
    with torch.no_grad():
        for batch in loader:
            margin_logits, wdl_logits = model(**_model_inputs(batch, config.device))
            margin, wdl = model.expected_values(margin_logits, wdl_logits)
            learned_parts.append((0.5 * margin + 0.5 * wdl).cpu())
            confidence_parts.append(
                torch.softmax(wdl_logits, dim=-1).max(dim=-1).values.cpu()
            )
    return torch.cat(learned_parts), torch.cat(confidence_parts)


def _blend_values(payload, learned, confidence, spec: BlendSpec):
    phase_weights = torch.tensor(spec.phase_learned_weights, dtype=torch.float32)
    weights = phase_weights[payload["phase_id"].long()].clone()
    if spec.wdl_confidence_floor > 0.0:
        weights[confidence < spec.wdl_confidence_floor] = 0.0
    heuristic = payload["heuristic"].float()
    return weights * learned + (1.0 - weights) * heuristic


def _blend_metrics(
    payload,
    blended,
    calibration,
    config,
    *,
    bootstrap: bool,
    seed: int,
):
    target = (payload["margin_class"].float() - MARGIN_CLIP) / MARGIN_CLIP
    records = []
    for index in range(int(blended.numel())):
        source_id = int(payload["source_id"][index])
        group = int(payload["group"][index])
        rollout = int(payload["rollout"][index])
        records.append({
            "candidate": float(blended[index]),
            "heuristic": float(payload["heuristic"][index]),
            "target": float(target[index]),
            "game_seed": source_id * 1_000_000 + group * 10 + rollout,
            "profile": OPPONENTS[int(payload["opponent_id"][index])],
            "actor_seat": int(payload["actor_seat"][index]),
            "phase": PHASES[int(payload["phase_id"][index])],
        })
    candidate = _basic_metrics(records, "candidate")
    heuristic = _basic_metrics(records, "heuristic")
    candidate_breakdowns = _breakdowns(records, "candidate")
    heuristic_breakdowns = _breakdowns(records, "heuristic")
    labels = [
        f"profile_seat:{profile}:seat{seat}"
        for profile in OPPONENTS
        for seat in (0, 1)
    ]
    candidate["macro_profile_seat_mae"] = statistics.fmean(
        float(candidate_breakdowns[label]["mae"]) for label in labels
    )
    heuristic["macro_profile_seat_mae"] = statistics.fmean(
        float(heuristic_breakdowns[label]["mae"]) for label in labels
    )
    result = {
        "candidate": candidate,
        "heuristic": heuristic,
        "candidate_breakdowns": candidate_breakdowns,
        "heuristic_breakdowns": heuristic_breakdowns,
        "candidate_ordering": _ordering_summary(records, "candidate"),
        "heuristic_ordering": _ordering_summary(records, "heuristic"),
        "candidate_ordering_by_phase": _ordering_by_phase(records, "candidate"),
        "heuristic_ordering_by_phase": _ordering_by_phase(records, "heuristic"),
        "calibration": calibration,
        "records": len(records),
    }
    if bootstrap:
        result["paired_game_error"] = _paired_game_error(
            records, config.bootstrap_samples, seed
        )
    return result


def _reservoir_states(
    sources: tuple[Path, ...],
    groups: range,
    per_stratum: int,
    seed: int,
):
    rng = random.Random(seed)
    reservoirs: dict[tuple[int, str, int, str], list[tuple[Any, int]]] = {}
    seen: dict[tuple[int, str, int, str], int] = {}
    for source_id, source in enumerate(sources):
        for group in groups:
            summary = json.loads(
                (source / "raw" / f"group_{group:04d}.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            seat = int(summary["actor_seat"])
            opponent = str(summary["opponent"])
            for rollout in range(4):
                path = source / "raw" / f"group_{group:04d}_rollout_{rollout}.jsonl"
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row.get("behavior_probability") is None:
                            continue
                        key = (
                            source_id,
                            opponent,
                            seat,
                            phase_for_turn(int(row["turn_index"])),
                        )
                        seen[key] = seen.get(key, 0) + 1
                        values = reservoirs.setdefault(key, [])
                        item = (deserialize_state_snapshot(row["state_snapshot"]), seat)
                        if len(values) < per_stratum:
                            values.append(item)
                        else:
                            replacement = rng.randrange(seen[key])
                            if replacement < per_stratum:
                                values[replacement] = item
    expected = {
        (source, opponent, seat, phase)
        for source in range(len(sources))
        for opponent in OPPONENTS
        for seat in (0, 1)
        for phase in PHASES
    }
    if set(reservoirs) != expected or any(
        len(rows) != per_stratum for rows in reservoirs.values()
    ):
        raise ValueError("Could not fill all frozen successor-consistency strata.")
    return [
        (key, state, seat)
        for key, rows in sorted(reservoirs.items())
        for state, seat in rows
    ]


def _state_components(model, states_and_players, device):
    from splendor_ai.encoding import collate_canonical_state_batch

    batch = collate_canonical_state_batch(states_and_players, device=device)
    with torch.no_grad():
        margin_logits, wdl_logits = model(**batch)
        margin, wdl = model.expected_values(margin_logits, wdl_logits)
        confidence = torch.softmax(wdl_logits, dim=-1).max(dim=-1).values
    return (
        (0.5 * margin + 0.5 * wdl).cpu().tolist(),
        confidence.cpu().tolist(),
    )


def evaluate_successor_grid(config, model, sampled_states, specs):
    scorer = CheckpointLegalActionPriorScorer(
        config.policy_checkpoint, device=config.device
    )
    env = SplendorEnv(seed=config.seed)
    records = {spec.name: [] for spec in specs}
    for stratum, state, actor in sampled_states:
        actions = env.legal_actions(state)
        if len(actions) < 2:
            continue
        scores_by_action = scorer.score_actions(state, actions)
        scores = [scores_by_action[action] for action in actions]
        successors = [env.step(state, action) for action in actions]
        learned, confidence = _state_components(
            model, [(successor, actor) for successor in successors], config.device
        )
        heuristic = [
            legal_heuristic_value(successor, actor, scale=4_000.0)
            for successor in successors
        ]
        policy_top = max(range(len(actions)), key=scores.__getitem__)
        top3 = set(
            sorted(range(len(actions)), key=scores.__getitem__, reverse=True)[:3]
        )
        for spec in specs:
            phase_index = PHASES.index(phase_for_turn(successors[0].turn_index))
            weight = spec.phase_learned_weights[phase_index]
            values = [
                (weight if conf >= spec.wdl_confidence_floor else 0.0) * learned_value
                + (1.0 - (weight if conf >= spec.wdl_confidence_floor else 0.0))
                * heuristic_value
                for learned_value, heuristic_value, conf in zip(
                    learned, heuristic, confidence, strict=True
                )
            ]
            value_ranks = _rank(values)
            value_top = max(range(len(actions)), key=values.__getitem__)
            records[spec.name].append({
                "stratum": ":".join(map(str, stratum)),
                "spearman": _correlation(_rank(scores), value_ranks),
                "policy_top_value_percentile": value_ranks[policy_top]
                / max(len(actions) - 1, 1),
                "value_best_in_policy_top3": value_top in top3,
            })
    return {
        spec.name: _summarize_successor_records(records[spec.name]) for spec in specs
    }


def _summarize_successor_records(records):
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


def _confirmation_collection(config: ReliabilityBlendConfig):
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


def _deployment_checkpoint(config, base_checkpoint, spec, path):
    torch.save(
        {
            "model_kind": "canonical_entity_reliability_blend",
            "input_contract": "canonical_public_belief_v1",
            "model_config": base_checkpoint["model_config"],
            "margin_bins": base_checkpoint["margin_bins"],
            "margin_clip": base_checkpoint["margin_clip"],
            "search_wdl_blend_weight": base_checkpoint[
                "search_wdl_blend_weight"
            ],
            "phase_learned_weights": list(spec.phase_learned_weights),
            "wdl_confidence_floor": spec.wdl_confidence_floor,
            "model_state_dict": base_checkpoint["model_state_dict"],
            "base_value_checkpoint": str(config.base_value_checkpoint),
            "base_value_checkpoint_sha256": _sha256(config.base_value_checkpoint),
            "selection": spec.name,
        },
        path,
    )


def _protocol(config):
    path_fields = {
        key: str(value)
        for key, value in asdict(config).items()
        if isinstance(value, Path)
    }
    return {
        "experiment": "reliability_blend_value_036b",
        "frozen_before_evidence": True,
        "base_checkpoint": {
            "path": str(config.base_value_checkpoint),
            "sha256": _sha256(config.base_value_checkpoint),
        },
        "validation": {
            "tensors": str(config.validation_tensors),
            "successor_sources": [str(config.source_032), str(config.source_033)],
            "groups": [448, 511],
            "successor_sample": "4 deterministic reservoir states per generation/opponent/seat/phase stratum",
        },
        "candidate_grid": [
            {
                "name": spec.name,
                "phase_learned_weights": list(spec.phase_learned_weights),
                "wdl_confidence_floor": spec.wdl_confidence_floor,
            }
            for spec in CANDIDATE_GRID
        ],
        "blend": "alpha*036_value + (1-alpha)*heuristic; alpha is phase weight and becomes zero when max WDL probability is below confidence floor",
        "selection": "require every unchanged 036 offline gate on validation, then minimize macro opponent/seat MAE - 0.1*ordering concordance",
        "confirmation": {
            "groups": config.confirmation_groups,
            "games": config.confirmation_groups * 4,
            "seed_start": config.confirmation_seed_start,
            "generated_only_after_selection": True,
        },
        "offline_gates": "identical thresholds and code path to 036",
        "conditional_puct": {
            "learned_vs_heuristic_puct64": {
                "pairs": config.puct_pairs,
                "seed_start": config.puct_ab_seed_start,
                "share": ">=0.55",
                "ci95_low": ">0.50",
            },
            "learned_vs_raw035": {
                "pairs": config.puct_pairs,
                "seed_start": config.puct_direct_seed_start,
                "share": ">=0.55",
                "ci95_low": ">0.50",
            },
        },
        "config": {**asdict(config), **path_fields},
    }


def run_reliability_blend(config: ReliabilityBlendConfig):
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing 036b protocol differs from frozen contract.")
    else:
        _atomic_json(protocol_path, protocol)

    payload = torch.load(
        config.validation_tensors, map_location="cpu", weights_only=True, mmap=True
    )
    model, base_checkpoint = _load_base_model(config)
    base_validation = evaluate_model(
        model,
        payload,
        ScaledPublicBeliefValueConfig(
            source_032=config.source_032,
            source_033=config.source_033,
            actor_parent_checkpoint=config.checkpoint_023,
            policy_checkpoint=config.policy_checkpoint,
            checkpoint_016=config.checkpoint_016,
            checkpoint_023=config.checkpoint_023,
            output_dir=config.output_dir,
            confirmation_data_dir=config.confirmation_data_dir,
            batch_size=config.batch_size,
            bootstrap_samples=config.bootstrap_samples,
            device=config.device,
        ),
        bootstrap=False,
        seed=config.seed,
    )
    learned, confidence = _predict_payload(model, payload, config)
    scalar_results = {}
    for spec in CANDIDATE_GRID:
        scalar_results[spec.name] = _blend_metrics(
            payload,
            _blend_values(payload, learned, confidence, spec),
            base_validation["calibration"],
            config,
            bootstrap=True,
            seed=config.seed + len(scalar_results),
        )
    sampled = _reservoir_states(
        (config.source_032, config.source_033),
        range(448, 512),
        per_stratum=4,
        seed=config.seed + 100,
    )
    successor_results = evaluate_successor_grid(
        config, model, sampled, CANDIDATE_GRID
    )
    candidates = []
    for spec in CANDIDATE_GRID:
        gates = offline_gates(
            scalar_results[spec.name], successor_results[spec.name]
        )
        score = (
            float(
                scalar_results[spec.name]["candidate"][
                    "macro_profile_seat_mae"
                ]
            )
            - 0.1
            * float(
                scalar_results[spec.name]["candidate_ordering"]["concordance"]
            )
        )
        candidates.append(
            {
                "spec": asdict(spec),
                "metrics": scalar_results[spec.name],
                "successor_consistency": successor_results[spec.name],
                "gates": gates,
                "passed": all(gates.values()),
                "selection_score": score,
            }
        )
    eligible = [row for row in candidates if row["passed"]]
    selection = min(eligible, key=lambda row: float(row["selection_score"]), default=None)
    selection_payload = {
        "candidates": candidates,
        "selected": selection,
        "passed": selection is not None,
    }
    _atomic_json(config.output_dir / "validation_selection.json", selection_payload)
    if selection is None:
        result = {
            "decision": "reject_validation_no_reliable_blend",
            "promoted": False,
            "selection": selection_payload,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result

    selected_spec = BlendSpec(**selection["spec"])
    checkpoint_path = config.output_dir / "canonical_entity_reliability_blend.pt"
    _deployment_checkpoint(config, base_checkpoint, selected_spec, checkpoint_path)

    collection = _confirmation_collection(config)
    if not collection["passed"]:
        result = {
            "decision": "fresh_confirmation_collection_failed",
            "promoted": False,
            "selection": selection,
            "collection": collection,
        }
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result
    confirmation_payload = _compile_tensor_corpus(
        (config.confirmation_data_dir,),
        "confirmation",
        config.output_dir / "confirmation_public_tensors.pt",
    )
    base_confirmation = evaluate_model(
        model,
        confirmation_payload,
        ScaledPublicBeliefValueConfig(
            source_032=config.source_032,
            source_033=config.source_033,
            actor_parent_checkpoint=config.checkpoint_023,
            policy_checkpoint=config.policy_checkpoint,
            checkpoint_016=config.checkpoint_016,
            checkpoint_023=config.checkpoint_023,
            output_dir=config.output_dir,
            confirmation_data_dir=config.confirmation_data_dir,
            batch_size=config.batch_size,
            bootstrap_samples=config.bootstrap_samples,
            device=config.device,
        ),
        bootstrap=False,
        seed=config.seed + 200,
    )
    learned_confirmation, confidence_confirmation = _predict_payload(
        model, confirmation_payload, config
    )
    confirmation = _blend_metrics(
        confirmation_payload,
        _blend_values(
            confirmation_payload,
            learned_confirmation,
            confidence_confirmation,
            selected_spec,
        ),
        base_confirmation["calibration"],
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
    consistency = evaluate_successor_grid(
        config, model, confirmation_states, (selected_spec,)
    )[selected_spec.name]
    gates = offline_gates(confirmation, consistency)
    offline = {
        "selected_spec": asdict(selected_spec),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "confirmation": confirmation,
        "successor_consistency": consistency,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_json(config.output_dir / "offline_gate.json", offline)
    if not offline["passed"]:
        result = {
            "decision": "reject_fresh_offline_keep_035",
            "promoted": False,
            "selection": selection,
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
    puct_ab = _run_search_gate(
        puct_config,
        checkpoint_path,
        "heuristic_puct",
        config.puct_ab_seed_start,
        "reliability_puct64_vs_heuristic_puct64",
    )
    if not puct_ab["passed"]:
        result = {
            "decision": "reject_search_ab_keep_035",
            "promoted": False,
            "selection": selection,
            "collection": collection,
            "offline": offline,
            "puct_ab": puct_ab,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(config.output_dir / "gate_final.json", result)
        return result
    direct = _run_search_gate(
        puct_config,
        checkpoint_path,
        "raw_policy",
        config.puct_direct_seed_start,
        "reliability_puct64_vs_raw035",
    )
    result = {
        "decision": (
            "authorize_synchronous_search_visit_distillation"
            if direct["passed"]
            else "reject_direct_search_keep_035"
        ),
        "promoted": False,
        "search_teacher_authorized": direct["passed"],
        "selection": selection,
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
    parser.add_argument("--validation-tensors", default="outputs/scaled_public_belief_value_036/validation_public_tensors.pt")
    parser.add_argument("--source-032", default="data/league_generation_032")
    parser.add_argument("--source-033", default="data/trust_region_generation_033")
    parser.add_argument("--policy-checkpoint", default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt")
    parser.add_argument("--checkpoint-016", default="outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt")
    parser.add_argument("--checkpoint-023", default="outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt")
    parser.add_argument("--output-dir", default="outputs/reliability_blend_value_036b")
    parser.add_argument("--confirmation-data-dir", default="data/reliability_blend_value_confirmation_036b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args):
    return ReliabilityBlendConfig(
        base_value_checkpoint=Path(args.base_value_checkpoint),
        validation_tensors=Path(args.validation_tensors),
        source_032=Path(args.source_032),
        source_033=Path(args.source_033),
        policy_checkpoint=Path(args.policy_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_023=Path(args.checkpoint_023),
        output_dir=Path(args.output_dir),
        confirmation_data_dir=Path(args.confirmation_data_dir),
        workers=args.workers,
        device=args.device,
    )


def main():
    result = run_reliability_blend(config_from_args(build_parser().parse_args()))
    print(json.dumps({key: result.get(key) for key in ("decision", "promoted", "search_teacher_authorized")}, indent=2))


if __name__ == "__main__":
    main()

"""Run fresh league generation 033 with structural parameter-space KL projection."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from splendor_ai.eval.parallel_paired import (
    PairedCheckpointConfig,
    run_paired_checkpoint_benchmark,
)
from splendor_ai.training.run_actor_movement_audit import (
    ActorMovementAuditConfig,
    _measure_stream,
)
from splendor_ai.training.run_awr_trust_region import (
    AwrTrustRegionConfig,
    project_awr_proposal,
)
from splendor_ai.training.run_group_relative_signal_audit import (
    GroupRelativeSignalAuditConfig,
    audit_group_relative_signal,
)
from splendor_ai.training.run_group_relative_trajectory_corpus import (
    GroupRelativeTrajectoryCorpusConfig,
    build_group_relative_trajectory_corpus,
)
from splendor_ai.training.run_league_generation import (
    LeagueGenerationConfig,
    SamplingSpec,
    _atomic_write,
    _config_payload,
    _sha256,
    _trajectory_path,
    collect_league_groups,
    evaluate_direct_gate,
    run_broad_gate,
)
from splendor_ai.training.run_trajectory_awr import (
    TrajectoryAwrConfig,
    train_trajectory_awr,
)
from splendor_ai.training.run_trajectory_policy_gradient import _load_attention_model


SAMPLING = SamplingSpec(
    epsilon=0.5,
    temperature=1.0,
    top_k=3,
    expected_nongreedy_rate=0.1525708639383316,
)
TARGET_RAW_KL = 0.008
MAX_RAW_KL = 0.01
MIN_VALIDATION_IMPROVEMENT = 0.03


def run_trust_region_generation(config: LeagueGenerationConfig) -> dict[str, Any]:
    _validate_033(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing 033 protocol does not match this run.")
    else:
        _atomic_write(protocol_path, protocol)

    collection = collect_league_groups(config, SAMPLING)
    if not collection["passed"]:
        return _finalize(config, "collection_integrity_failed", collection=collection)

    replay_paths = tuple(
        _trajectory_path(config, group, rollout)
        for group in range(config.groups)
        for rollout in range(config.trajectories_per_group)
    )
    trajectory_path = config.data_dir / "trajectory_advantages.jsonl"
    corpus_summary_path = config.data_dir / "trajectory_summary.json"
    corpus = build_group_relative_trajectory_corpus(
        GroupRelativeTrajectoryCorpusConfig(
            replay_paths=replay_paths,
            output_path=trajectory_path,
            summary_path=corpus_summary_path,
            outcome_weight=0.0,
            score_margin_scale=20.0,
            min_group_trajectories=config.trajectories_per_group,
            baseline_mode="leave_one_out",
        )
    )
    signal = audit_group_relative_signal(
        GroupRelativeSignalAuditConfig(
            trajectory_path=trajectory_path,
            corpus_summary_path=corpus_summary_path,
            parent_checkpoint=config.parent_checkpoint,
            output_path=config.output_dir / "signal_audit.json",
            expected_group_size=config.trajectories_per_group,
            min_groups=config.groups,
            min_informative_group_rate=0.75,
            min_nongreedy_row_rate=0.10,
            sampling_epsilon=SAMPLING.epsilon,
            sampling_temperature=SAMPLING.temperature,
            sampling_top_k=SAMPLING.top_k,
            seed=33,
            device=config.device,
        )
    )
    if not signal["pass"]:
        return _finalize(
            config,
            "signal_gate_failed",
            collection=collection,
            corpus=corpus,
            signal=signal,
        )

    proposal = train_trajectory_awr(
        TrajectoryAwrConfig(
            replay_path=trajectory_path,
            init_checkpoint=config.parent_checkpoint,
            output_dir=config.output_dir / "proposal",
            train_seed_max=config.collection_seed_start + config.train_groups - 1,
            validation_seed_min=config.collection_seed_start + config.train_groups,
            sampling_epsilon=SAMPLING.epsilon,
            sampling_temperature=SAMPLING.temperature,
            sampling_top_k=SAMPLING.top_k,
            seed=33,
            device=config.device,
        )
    )
    projection = project_awr_proposal(
        AwrTrustRegionConfig(
            replay_path=trajectory_path,
            parent_checkpoint=config.parent_checkpoint,
            proposal_checkpoint=Path(proposal["checkpoint_path"]),
            output_dir=config.output_dir / "projected",
            validation_seed_min=config.collection_seed_start + config.train_groups,
            target_raw_kl=TARGET_RAW_KL,
            bisection_steps=12,
            sampling_epsilon=SAMPLING.epsilon,
            sampling_temperature=SAMPLING.temperature,
            sampling_top_k=SAMPLING.top_k,
            device=config.device,
        )
    )
    projected_checkpoint = Path(projection["checkpoint_path"])
    movement = audit_projected_movement(
        config, trajectory_path, projected_checkpoint, proposal, projection
    )
    _atomic_write(config.output_dir / "movement_gate.json", movement)
    if not movement["passed"]:
        return _finalize(
            config,
            "movement_gate_failed",
            collection=collection,
            corpus=corpus,
            signal=signal,
            proposal=proposal,
            projection=projection,
            movement=movement,
        )

    direct = run_paired_checkpoint_benchmark(
        PairedCheckpointConfig(
            candidate_checkpoint=projected_checkpoint,
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
    broad = run_broad_gate(config, projected_checkpoint) if all(direct_gates.values()) else None
    decision = (
        "promote_generation_1"
        if all(direct_gates.values()) and broad is not None and broad["passed"]
        else "direct_gate_failed"
        if not all(direct_gates.values())
        else "broad_gate_failed"
    )
    return _finalize(
        config,
        decision,
        collection=collection,
        corpus=corpus,
        signal=signal,
        proposal=proposal,
        projection=projection,
        movement=movement,
        direct=direct,
        direct_gates=direct_gates,
        broad=broad,
    )


def evaluate_projected_gates(
    projection: dict[str, Any],
    on_policy: dict[str, Any],
    broad: dict[str, Any],
    behavior_mae: float,
) -> dict[str, bool]:
    parent_loss = float(projection["parent"]["policy_loss"])
    projected_loss = float(projection["projected"]["policy_loss"])
    improvement = (parent_loss - projected_loss) / parent_loss
    return {
        "positive_projection_alpha": float(projection["selected_alpha"]) > 0.0,
        "validation_raw_kl_target": float(projection["projected"]["raw_kl"])
        <= TARGET_RAW_KL + 1e-6,
        "validation_policy_loss_improvement": improvement >= MIN_VALIDATION_IMPROVEMENT,
        "on_policy_movement_floor": float(on_policy["top_action_change_rate"]) >= 0.01,
        "broad_movement_floor": float(broad["top_action_change_rate"]) >= 0.01,
        "on_policy_movement_ceiling": float(on_policy["top_action_change_rate"]) <= 0.10,
        "broad_movement_ceiling": float(broad["top_action_change_rate"]) <= 0.10,
        "on_policy_raw_kl_ceiling": float(on_policy["raw_policy_kl"]["mean"])
        <= MAX_RAW_KL,
        "broad_raw_kl_ceiling": float(broad["raw_policy_kl"]["mean"]) <= MAX_RAW_KL,
        "behavior_reconstruction": behavior_mae <= 1e-5,
    }


def audit_projected_movement(config, trajectory_path, checkpoint, proposal, projection):
    parent, _ = _load_attention_model(config.parent_checkpoint, config.device)
    candidate, _ = _load_attention_model(checkpoint, config.device)
    audit_config = ActorMovementAuditConfig(
        parent_checkpoint=config.parent_checkpoint,
        candidate_checkpoint=checkpoint,
        training_metrics_path=config.output_dir / "proposal" / "trajectory_awr_metrics.json",
        on_policy_replay_paths=(trajectory_path,),
        broad_replay_paths=(config.broad_replay,),
        output_path=config.output_dir / "movement_gate.json",
        sampling_epsilon=SAMPLING.epsilon,
        sampling_temperature=SAMPLING.temperature,
        sampling_top_k=SAMPLING.top_k,
        seed=33,
        device=config.device,
    )
    on_policy = _measure_stream((trajectory_path,), parent, candidate, audit_config, seed=33)
    broad = _measure_stream((config.broad_replay,), parent, candidate, audit_config, seed=34)
    behavior_mae = float(proposal["epochs"][-1]["behavior_log_prob_mae"])
    gates = evaluate_projected_gates(projection, on_policy, broad, behavior_mae)
    parent_loss = float(projection["parent"]["policy_loss"])
    projected_loss = float(projection["projected"]["policy_loss"])
    return {
        "format": "trust_region_generation_movement_gate_v1",
        "projection": projection,
        "validation_policy_loss_improvement_fraction": (
            parent_loss - projected_loss
        ) / parent_loss,
        "on_policy": on_policy,
        "broad": broad,
        "behavior_log_prob_mae": behavior_mae,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _protocol(config):
    return {
        "experiment": "trust_region_league_generation_033",
        "config": _config_payload(config),
        "checkpoint_hashes": {
            "035": _sha256(config.parent_checkpoint),
            "016": _sha256(config.checkpoint_016),
            "023": _sha256(config.checkpoint_023),
        },
        "sampling": asdict(SAMPLING),
        "fresh_data_only": True,
        "forbidden_training_source": "data/league_generation_032",
        "proposal_recipe": "exact 032 one-epoch AWR recipe",
        "projection": {
            "method": "linear parameter interpolation from 035 to proposal",
            "selection": "largest alpha satisfying full-validation mean raw KL <= 0.008",
            "bisection_steps": 12,
            "game_results_used_for_alpha": False,
        },
        "offline_gates": {
            "minimum_validation_policy_loss_improvement": MIN_VALIDATION_IMPROVEMENT,
            "movement_range": [0.01, 0.10],
            "validation_target_raw_kl": TARGET_RAW_KL,
            "stream_max_raw_kl": MAX_RAW_KL,
        },
        "stage_order": ["collection", "signal", "proposal", "projection", "movement", "direct_035", "broad"],
    }


def _finalize(config, decision, **stages):
    payload = {
        "gate": "trust_region_league_generation_033",
        "decision": decision,
        "promoted": decision == "promote_generation_1",
        **stages,
    }
    _atomic_write(config.output_dir / "gate_final.json", payload)
    print(json.dumps({"decision": decision, "promoted": payload["promoted"]}, indent=2))
    return payload


def _validate_033(config):
    if config.collection_seed_start != 355_000:
        raise ValueError("033 collection seed block is frozen at 355000.")
    if config.direct_seed_start != 356_000:
        raise ValueError("033 direct seed block is frozen at 356000.")
    if config.groups != 512 or config.train_groups != 448:
        raise ValueError("033 requires the frozen 512/448 group contract.")
    forbidden = Path("data/league_generation_032").resolve()
    if config.data_dir.resolve() == forbidden:
        raise ValueError("033 may not reuse the 032 training directory.")
    for path in (
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
        config.broad_replay,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


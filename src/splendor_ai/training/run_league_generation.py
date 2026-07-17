"""Run one preregistered league self-play generation anchored on policy 035."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionSamplingBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.eval.match import play_game
from splendor_ai.eval.parallel_paired import (
    PairedCheckpointConfig,
    run_paired_checkpoint_benchmark,
)
from splendor_ai.eval.run_shallow_operational_gate import (
    SHALLOW_CONTRACT,
    integrity_passed,
    summarize_pair_rows,
)
from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.replay import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.run_actor_movement_audit import (
    ActorMovementAuditConfig,
    _measure_stream,
)
from splendor_ai.training.run_group_relative_signal_audit import (
    GroupRelativeSignalAuditConfig,
    audit_group_relative_signal,
)
from splendor_ai.training.run_group_relative_trajectory_corpus import (
    GroupRelativeTrajectoryCorpusConfig,
    build_group_relative_trajectory_corpus,
)
from splendor_ai.training.run_trajectory_awr import (
    TrajectoryAwrConfig,
    train_trajectory_awr,
)
from splendor_ai.training.run_trajectory_policy_gradient import (
    _exploration_probabilities,
    _load_attention_model,
)


OPPONENTS = ("035", "016", "shallow", "023")
CALIBRATION_GRID = (
    (0.30, 0.50, 3),
    (0.40, 1.00, 3),
    (0.50, 1.00, 3),
    (0.50, 1.50, 5),
    (0.60, 1.50, 5),
)
TARGET_NONGREEDY_RATE = 0.15
MIN_NONGREEDY_RATE = 0.10
MAX_NONGREEDY_RATE = 0.20


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    epsilon: float
    temperature: float
    top_k: int
    expected_nongreedy_rate: float


@dataclass(frozen=True, slots=True)
class LeagueGenerationConfig:
    output_dir: Path
    data_dir: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    calibration_replay: Path
    broad_replay: Path
    groups: int = 512
    train_groups: int = 448
    trajectories_per_group: int = 4
    collection_seed_start: int = 350_000
    sampling_seed_start: int = 32_000_000
    direct_pairs: int = 200
    direct_seed_start: int = 351_000
    broad_pairs: int = 50
    broad_016_seed_start: int = 352_000
    broad_shallow_seed_start: int = 353_000
    broad_023_seed_start: int = 354_000
    workers: int = 4
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    calibration_rows: int = 5_000
    device: str = "cpu"


def run_league_generation(config: LeagueGenerationConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing league-generation protocol does not match this run.")
    else:
        _atomic_write(protocol_path, protocol)

    calibration = calibrate_exploration(config)
    if not calibration["passed"]:
        return _finalize(config, calibration=calibration, decision="calibration_failed")
    sampling = SamplingSpec(**calibration["selected"])

    collection = collect_league_groups(config, sampling)
    if not collection["passed"]:
        return _finalize(
            config,
            calibration=calibration,
            collection=collection,
            decision="collection_integrity_failed",
        )

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
            min_nongreedy_row_rate=MIN_NONGREEDY_RATE,
            sampling_epsilon=sampling.epsilon,
            sampling_temperature=sampling.temperature,
            sampling_top_k=sampling.top_k,
            seed=32,
            device=config.device,
        )
    )
    if not signal["pass"]:
        return _finalize(
            config,
            calibration=calibration,
            collection=collection,
            corpus=corpus,
            signal=signal,
            decision="signal_gate_failed",
        )

    candidate_dir = config.output_dir / "candidate"
    training = train_trajectory_awr(
        TrajectoryAwrConfig(
            replay_path=trajectory_path,
            init_checkpoint=config.parent_checkpoint,
            output_dir=candidate_dir,
            train_seed_max=config.collection_seed_start + config.train_groups - 1,
            validation_seed_min=config.collection_seed_start + config.train_groups,
            sampling_epsilon=sampling.epsilon,
            sampling_temperature=sampling.temperature,
            sampling_top_k=sampling.top_k,
            device=config.device,
        )
    )
    candidate_checkpoint = Path(training["checkpoint_path"])
    movement = audit_generation_movement(
        config,
        sampling,
        candidate_checkpoint,
        training,
        trajectory_path,
    )
    _atomic_write(config.output_dir / "movement_gate.json", movement)
    if not movement["passed"]:
        return _finalize(
            config,
            calibration=calibration,
            collection=collection,
            corpus=corpus,
            signal=signal,
            training=training,
            movement=movement,
            decision="movement_gate_failed",
        )

    direct = run_paired_checkpoint_benchmark(
        PairedCheckpointConfig(
            candidate_checkpoint=candidate_checkpoint,
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
        broad = run_broad_gate(config, candidate_checkpoint)
    decision = (
        "promote_generation_1"
        if all(direct_gates.values()) and broad is not None and broad["passed"]
        else "direct_gate_failed"
        if not all(direct_gates.values())
        else "broad_gate_failed"
    )
    return _finalize(
        config,
        calibration=calibration,
        collection=collection,
        corpus=corpus,
        signal=signal,
        training=training,
        movement=movement,
        direct=direct,
        direct_gates=direct_gates,
        broad=broad,
        decision=decision,
    )


def calibrate_exploration(config: LeagueGenerationConfig) -> dict[str, Any]:
    output_path = config.output_dir / "exploration_calibration.json"
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    model, _ = _load_attention_model(config.parent_checkpoint, config.device)
    model.eval()
    dataset = LegalActionReplayDataset(
        config.calibration_replay,
        include_stalled_games=False,
        include_timed_out_games=False,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    count = min(config.calibration_rows, len(dataset))
    indices = sorted(random.Random(32).sample(range(len(dataset)), count))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=128,
        shuffle=False,
        collate_fn=collate_legal_action_replay_samples,
    )
    totals = [0.0 for _ in CALIBRATION_GRID]
    observed = 0
    with torch.inference_mode():
        for batch in loader:
            observation = batch["observation"].to(config.device)
            actions = batch["legal_action_features"].to(config.device)
            mask = batch["legal_action_mask"].to(config.device)
            scores, _ = model(observation, actions, mask)
            greedy = torch.argmax(scores.masked_fill(~mask, -torch.inf), dim=1)
            rows = torch.arange(greedy.shape[0], device=config.device)
            for index, (epsilon, temperature, top_k) in enumerate(CALIBRATION_GRID):
                _, probabilities = _exploration_probabilities(
                    scores,
                    scores,
                    mask,
                    epsilon,
                    temperature,
                    top_k,
                )
                totals[index] += float((1.0 - probabilities[rows, greedy]).sum().cpu())
            observed += int(greedy.shape[0])
    candidates = []
    for grid, total in zip(CALIBRATION_GRID, totals, strict=True):
        rate = total / observed
        candidates.append(
            {
                "epsilon": grid[0],
                "temperature": grid[1],
                "top_k": grid[2],
                "expected_nongreedy_rate": rate,
                "inside_target_band": MIN_NONGREEDY_RATE <= rate <= MAX_NONGREEDY_RATE,
            }
        )
    eligible = [row for row in candidates if row["inside_target_band"]]
    selected = min(
        eligible,
        key=lambda row: abs(float(row["expected_nongreedy_rate"]) - TARGET_NONGREEDY_RATE),
        default=None,
    )
    payload = {
        "format": "league_generation_exploration_calibration_v1",
        "rows": observed,
        "target_rate": TARGET_NONGREEDY_RATE,
        "accepted_band": [MIN_NONGREEDY_RATE, MAX_NONGREEDY_RATE],
        "candidates": candidates,
        "selected": selected,
        "passed": selected is not None,
    }
    _atomic_write(output_path, payload)
    return payload


def collect_league_groups(
    config: LeagueGenerationConfig,
    sampling: SamplingSpec,
) -> dict[str, Any]:
    summaries: dict[int, dict[str, Any]] = {}
    pending = []
    for group in range(config.groups):
        summary_path = _group_summary_path(config, group)
        if summary_path.exists() and all(
            _trajectory_path(config, group, rollout).exists()
            for rollout in range(config.trajectories_per_group)
        ):
            summaries[group] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            pending.append(group)
    print(
        f"[league-collect] groups={config.groups} complete={len(summaries)}"
        f" pending={len(pending)} workers={config.workers}"
    )
    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_collect_group, config, sampling, group): group
                for group in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                group = futures[future]
                summaries[group] = future.result()
                if completed == 1 or completed % 16 == 0 or completed == len(pending):
                    print(f"[league-collect] complete={len(summaries)}/{config.groups}")
    ordered = [summaries[index] for index in range(config.groups)]
    games = [game for row in ordered for game in row["games"]]
    by_cell: dict[str, int] = {}
    for row in ordered:
        key = f"{row['opponent']}:seat{row['actor_seat']}"
        by_cell[key] = by_cell.get(key, 0) + 1
    payload = {
        "format": "league_generation_collection_v1",
        "groups": len(ordered),
        "games": len(games),
        "by_opponent_seat_groups": by_cell,
        "stalled_games": sum(bool(game["stalled"]) for game in games),
        "timed_out_games": sum(bool(game["timed_out"]) for game in games),
        "termination_reasons": _termination_counts(games),
        "actor_turns": sum(int(row["actor_turns"]) for row in ordered),
        "actor_nongreedy_turns": sum(int(row["actor_nongreedy_turns"]) for row in ordered),
    }
    payload["actor_nongreedy_rate"] = (
        payload["actor_nongreedy_turns"] / payload["actor_turns"]
    )
    expected_per_cell = config.groups // (len(OPPONENTS) * 2)
    payload["gates"] = {
        "complete": len(ordered) == config.groups,
        "all_games_present": len(games) == config.groups * config.trajectories_per_group,
        "balanced_cells": len(by_cell) == len(OPPONENTS) * 2
        and set(by_cell.values()) == {expected_per_cell},
        "zero_stalls": payload["stalled_games"] == 0,
        "zero_timeouts": payload["timed_out_games"] == 0,
        "nongreedy_rate": MIN_NONGREEDY_RATE
        <= payload["actor_nongreedy_rate"]
        <= MAX_NONGREEDY_RATE,
    }
    payload["passed"] = all(payload["gates"].values())
    _atomic_write(config.output_dir / "collection_summary.json", payload)
    return payload


def _collect_group(
    config: LeagueGenerationConfig,
    sampling: SamplingSpec,
    group: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    opponent, actor_seat = _group_assignment(group)
    games = []
    actor_turns = 0
    actor_nongreedy = 0
    for rollout in range(config.trajectories_per_group):
        actor = LegalActionAttentionSamplingBot(
            config.parent_checkpoint,
            device=config.device,
            seed=config.sampling_seed_start + group * config.trajectories_per_group + rollout,
            epsilon=sampling.epsilon,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
        )
        opponent_bot = _build_opponent(config, opponent)
        bots = (actor, opponent_bot) if actor_seat == 0 else (opponent_bot, actor)
        replay = collect_game_replay(
            bots[0],
            bots[1],
            seed=config.collection_seed_start + group,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
            include_state_snapshots=True,
        )
        path = _trajectory_path(config, group, rollout)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        export_replay_games_jsonl(temporary, [replay])
        os.replace(temporary, path)
        actor_steps = [step for step in replay.steps if step.player_id == actor_seat]
        actor_turns += len(actor_steps)
        actor_nongreedy += sum(step.behavior_greedy_action is False for step in actor_steps)
        games.append(
            {
                "winner": replay.winner,
                "final_scores": list(replay.final_scores),
                "turns": replay.turns,
                "stalled": replay.stalled,
                "timed_out": replay.timed_out,
                "termination_reason": replay.termination_reason,
            }
        )
    payload = {
        "group": group,
        "environment_seed": config.collection_seed_start + group,
        "opponent": opponent,
        "actor_seat": actor_seat,
        "actor_turns": actor_turns,
        "actor_nongreedy_turns": actor_nongreedy,
        "games": games,
    }
    _atomic_write(_group_summary_path(config, group), payload)
    return payload


def audit_generation_movement(
    config: LeagueGenerationConfig,
    sampling: SamplingSpec,
    candidate_checkpoint: Path,
    training: dict[str, Any],
    trajectory_path: Path,
) -> dict[str, Any]:
    parent, _ = _load_attention_model(config.parent_checkpoint, config.device)
    candidate, _ = _load_attention_model(candidate_checkpoint, config.device)
    audit_config = ActorMovementAuditConfig(
        parent_checkpoint=config.parent_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        training_metrics_path=config.output_dir / "candidate" / "trajectory_awr_metrics.json",
        on_policy_replay_paths=(trajectory_path,),
        broad_replay_paths=(config.broad_replay,),
        output_path=config.output_dir / "movement_gate.json",
        sampling_epsilon=sampling.epsilon,
        sampling_temperature=sampling.temperature,
        sampling_top_k=sampling.top_k,
        seed=32,
        device=config.device,
    )
    on_policy = _measure_stream(
        (trajectory_path,), parent, candidate, audit_config, seed=32
    )
    broad = _measure_stream(
        (config.broad_replay,), parent, candidate, audit_config, seed=33
    )
    final_epoch = training["epochs"][-1]
    baseline_validation = training["baseline_validation"]
    validation = final_epoch["validation"]
    gates = {
        "validation_awr_improves": float(validation["policy_loss"])
        < float(baseline_validation["policy_loss"]),
        "on_policy_movement_floor": float(on_policy["top_action_change_rate"]) >= 0.01,
        "broad_movement_floor": float(broad["top_action_change_rate"]) >= 0.01,
        "on_policy_movement_ceiling": float(on_policy["top_action_change_rate"]) <= 0.10,
        "broad_movement_ceiling": float(broad["top_action_change_rate"]) <= 0.10,
        "on_policy_kl_ceiling": float(on_policy["raw_policy_kl"]["mean"]) <= 0.01,
        "broad_kl_ceiling": float(broad["raw_policy_kl"]["mean"]) <= 0.01,
        "behavior_reconstruction": float(final_epoch["behavior_log_prob_mae"]) <= 1e-5,
    }
    return {
        "format": "league_generation_movement_gate_v1",
        "on_policy": on_policy,
        "broad": broad,
        "baseline_validation": baseline_validation,
        "candidate_validation": validation,
        "training": final_epoch,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_direct_gate(payload: dict[str, Any]) -> dict[str, bool]:
    share = payload["paired_statistics"]["score_share"]
    return {
        "integrity": int(payload["timed_out_games"]) == 0
        and int(payload["stalled_games"]) == 0,
        "score_share_at_least_55pct": float(share["mean"]) >= 0.55,
        "score_share_ci95_lower_above_50pct": float(share["ci95_low"]) > 0.50,
    }


def run_broad_gate(config: LeagueGenerationConfig, candidate_checkpoint: Path) -> dict[str, Any]:
    blocks = {}
    for opponent, seed_start in (
        ("016", config.broad_016_seed_start),
        ("shallow", config.broad_shallow_seed_start),
        ("023", config.broad_023_seed_start),
    ):
        for role in ("candidate", "parent"):
            name = f"{role}_{opponent}"
            blocks[name] = _run_broad_block(
                config,
                name,
                candidate_checkpoint if role == "candidate" else config.parent_checkpoint,
                opponent,
                seed_start,
            )
    gates = {}
    results = {}
    for opponent in ("016", "shallow", "023"):
        candidate = blocks[f"candidate_{opponent}"]
        parent = blocks[f"parent_{opponent}"]
        candidate_share = float(candidate["summary"]["score_share"]["mean"])
        parent_share = float(parent["summary"]["score_share"]["mean"])
        gates[f"{opponent}_candidate_integrity"] = integrity_passed(candidate)
        gates[f"{opponent}_parent_integrity"] = integrity_passed(parent)
        gates[f"{opponent}_absolute_floor"] = candidate_share >= 0.25
        gates[f"{opponent}_parent_nonregression"] = candidate_share - parent_share >= -0.05 - 1e-12
        results[opponent] = {
            "candidate_score_share": candidate_share,
            "parent_score_share": parent_share,
            "delta": candidate_share - parent_share,
        }
    payload = {"results": results, "gates": gates, "passed": all(gates.values())}
    _atomic_write(config.output_dir / "broad_gate.json", payload)
    return payload


def _run_broad_block(config, name, checkpoint, opponent, seed_start):
    path = config.output_dir / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                _play_broad_pair, config, checkpoint, opponent, seed
            ): seed
            for seed in range(seed_start, seed_start + config.broad_pairs)
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["seed"]))
    payload = {
        "format": "league_generation_broad_block_v1",
        "summary": summarize_pair_rows(rows, config.bootstrap_samples, seed_start),
        "pair_rows": rows,
    }
    _atomic_write(path, payload)
    return payload


def _play_broad_pair(config, checkpoint, opponent, seed):
    torch.set_num_threads(1)
    first = play_game(
        LegalActionAttentionScorerBot(checkpoint, device=config.device),
        _build_opponent(config, opponent),
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    second = play_game(
        _build_opponent(config, opponent),
        LegalActionAttentionScorerBot(checkpoint, device=config.device),
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
    }


def _build_opponent(config: LeagueGenerationConfig, kind: str):
    if kind == "035":
        return LegalActionAttentionScorerBot(config.parent_checkpoint, device=config.device)
    if kind == "016":
        return LegalActionScorerBot(config.checkpoint_016, device=config.device)
    if kind == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    if kind == "023":
        return CanonicalEntityActionScorerBot(config.checkpoint_023, device=config.device)
    raise ValueError(f"Unsupported league opponent: {kind}")


def _group_assignment(group: int) -> tuple[str, int]:
    return OPPONENTS[group % len(OPPONENTS)], (group // len(OPPONENTS)) % 2


def _trajectory_path(config, group, rollout):
    return config.data_dir / "raw" / f"group_{group:04d}_rollout_{rollout}.jsonl"


def _group_summary_path(config, group):
    return config.data_dir / "raw" / f"group_{group:04d}.summary.json"


def _termination_counts(games):
    counts: dict[str, int] = {}
    for game in games:
        reason = str(game["termination_reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol(config: LeagueGenerationConfig) -> dict[str, Any]:
    return {
        "experiment": "league_generation_032",
        "config": _config_payload(config),
        "checkpoint_hashes": {
            "035": _sha256(config.parent_checkpoint),
            "016": _sha256(config.checkpoint_016),
            "023": _sha256(config.checkpoint_023),
        },
        "opponents": list(OPPONENTS),
        "calibration_grid": [list(row) for row in CALIBRATION_GRID],
        "calibration_target_band": [MIN_NONGREEDY_RATE, MAX_NONGREEDY_RATE],
        "return": "leave-one-out group-ranked tanh(final_score_margin / 20)",
        "update": "one-epoch full-model advantage-weighted regression, lr1e-5, temp0.25, max_weight10, KL1.0",
        "stage_order": ["calibration", "collection", "signal", "training", "movement", "direct_035", "broad"],
    }


def _finalize(config: LeagueGenerationConfig, *, decision: str, **stages):
    payload = {
        "gate": "league_generation_032",
        "decision": decision,
        "promoted": decision == "promote_generation_1",
        **stages,
    }
    _atomic_write(config.output_dir / "gate_final.json", payload)
    print(json.dumps({"decision": decision, "promoted": payload["promoted"]}, indent=2))
    return payload


def _config_payload(config: LeagueGenerationConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _validate(config: LeagueGenerationConfig) -> None:
    if config.groups != config.train_groups + (config.groups - config.train_groups):
        raise ValueError("Invalid group split.")
    if config.groups % (len(OPPONENTS) * 2) != 0:
        raise ValueError("Groups must balance exactly across opponent and seat cells.")
    if not 0 < config.train_groups < config.groups:
        raise ValueError("Train groups must be a strict subset of all groups.")
    if min(config.trajectories_per_group, config.workers, config.direct_pairs) <= 0:
        raise ValueError("Generation counts must be positive.")
    for path in (
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
        config.calibration_replay,
        config.broad_replay,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


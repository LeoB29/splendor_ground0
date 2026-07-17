"""Pre-training signal audit for legal group-relative trajectory corpora."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.run_trajectory_policy_gradient import (
    _exploration_probabilities,
    _load_attention_model,
)


@dataclass(frozen=True, slots=True)
class GroupRelativeSignalAuditConfig:
    trajectory_path: Path
    corpus_summary_path: Path
    parent_checkpoint: Path
    output_path: Path
    expected_group_size: int = 4
    min_groups: int = 50
    min_informative_group_rate: float = 0.75
    min_nongreedy_row_rate: float = 0.02
    max_behavior_log_prob_mae: float = 1e-5
    reconstruction_rows: int = 2_000
    batch_size: int = 128
    sampling_epsilon: float = 0.2
    sampling_temperature: float = 0.5
    sampling_top_k: int = 3
    seed: int = 40
    device: str = "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit legal repeated-rollout signal before a group-relative actor update."
    )
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument("--corpus-summary-path", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--expected-group-size", type=int, default=4)
    parser.add_argument("--min-groups", type=int, default=50)
    parser.add_argument("--min-informative-group-rate", type=float, default=0.75)
    parser.add_argument("--min-nongreedy-row-rate", type=float, default=0.02)
    parser.add_argument("--max-behavior-log-prob-mae", type=float, default=1e-5)
    parser.add_argument("--reconstruction-rows", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-epsilon", type=float, default=0.2)
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
    parser.add_argument("--sampling-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    return parser


def audit_group_relative_signal(config: GroupRelativeSignalAuditConfig) -> dict[str, Any]:
    _validate(config)
    corpus_summary = json.loads(config.corpus_summary_path.read_text(encoding="utf-8"))
    trajectory_returns: dict[tuple[str, int, int], float] = {}
    group_trajectories: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    probabilities: list[float] = []
    logged_log_errors: list[float] = []
    nongreedy_rows = 0
    rows_count = 0
    with config.trajectory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows_count += 1
            row = json.loads(line)
            metadata = row.get("trajectory_metadata", {})
            trajectory_key = (
                str(metadata.get("source_replay_path", "")),
                int(row.get("game_seed", row.get("seed", -1))),
                int(row["player_id"]),
            )
            group_key = json.dumps(metadata.get("group_key", []), sort_keys=True)
            trajectory_returns[trajectory_key] = float(metadata["raw_return"])
            group_trajectories[group_key].add(trajectory_key)
            probability = float(row["behavior_probability"])
            probabilities.append(probability)
            logged_log_errors.append(
                abs(math.log(probability) - float(row["behavior_log_probability"]))
            )
            nongreedy_rows += int(not bool(row.get("behavior_greedy_action", True)))

    group_ranges = []
    observed_group_sizes = []
    for member_keys in group_trajectories.values():
        values = [trajectory_returns[key] for key in member_keys]
        observed_group_sizes.append(len(member_keys))
        group_ranges.append(max(values) - min(values))

    reconstructed = _reconstruct_behavior_log_probs(config)
    groups_total = int(corpus_summary.get("groups_total", 0))
    groups_informative = int(corpus_summary.get("groups_informative", 0))
    informative_rate = groups_informative / groups_total if groups_total else 0.0
    nongreedy_rate = nongreedy_rows / rows_count if rows_count else 0.0
    baseline_mode = str(corpus_summary.get("config", {}).get("baseline_mode", ""))
    gates = {
        "baseline_is_leave_one_out": baseline_mode == "leave_one_out",
        "enough_groups": groups_total >= config.min_groups,
        "all_observed_groups_expected_size": bool(observed_group_sizes)
        and min(observed_group_sizes) == config.expected_group_size
        and max(observed_group_sizes) == config.expected_group_size,
        "informative_group_rate": informative_rate >= config.min_informative_group_rate,
        "nongreedy_row_rate": nongreedy_rate >= config.min_nongreedy_row_rate,
        "logged_probabilities_valid": bool(probabilities)
        and min(probabilities) > 0.0
        and max(probabilities) <= 1.0
        and max(logged_log_errors, default=math.inf) <= 1e-6,
        "behavior_probability_reconstruction": reconstructed["mae"]
        <= config.max_behavior_log_prob_mae,
        "no_stalled_or_timed_out_rows": int(corpus_summary.get("skipped_stalled_game", 0)) == 0
        and int(corpus_summary.get("skipped_timed_out_game", 0)) == 0,
    }
    payload = {
        "format": "group_relative_signal_audit_v1",
        "config": _config_payload(config),
        "corpus": {
            "rows": rows_count,
            "trajectories": len(trajectory_returns),
            "groups_total": groups_total,
            "groups_informative": groups_informative,
            "informative_group_rate": informative_rate,
            "observed_group_size_min": min(observed_group_sizes, default=0),
            "observed_group_size_max": max(observed_group_sizes, default=0),
            "mean_return_range": statistics.fmean(group_ranges) if group_ranges else 0.0,
            "nongreedy_row_rate": nongreedy_rate,
            "baseline_mode": baseline_mode,
        },
        "behavior_probability": {
            "min": min(probabilities, default=0.0),
            "max": max(probabilities, default=0.0),
            "logged_log_probability_max_error": max(logged_log_errors, default=math.inf),
            "reconstruction": reconstructed,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _reconstruct_behavior_log_probs(config: GroupRelativeSignalAuditConfig) -> dict[str, float | int]:
    dataset = LegalActionReplayDataset(config.trajectory_path)
    count = min(config.reconstruction_rows, len(dataset))
    rng = random.Random(config.seed)
    indices = sorted(rng.sample(range(len(dataset)), count)) if count < len(dataset) else list(range(len(dataset)))
    dataloader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_legal_action_replay_samples,
    )
    model, _checkpoint = _load_attention_model(config.parent_checkpoint, config.device)
    model.eval()
    absolute_errors: list[float] = []
    with torch.inference_mode():
        for batch in dataloader:
            observation = batch["observation"].to(config.device)
            action_features = batch["legal_action_features"].to(config.device)
            legal_mask = batch["legal_action_mask"].to(config.device)
            target_rows = batch["target_action_row"].to(config.device)
            scores, _value = model(observation, action_features, legal_mask)
            _current, reference_probs = _exploration_probabilities(
                scores,
                scores,
                legal_mask,
                config.sampling_epsilon,
                config.sampling_temperature,
                config.sampling_top_k,
            )
            reconstructed = reference_probs[
                torch.arange(target_rows.shape[0], device=config.device), target_rows
            ].clamp_min(1e-12).log()
            logged = batch["behavior_log_probability"].to(config.device)
            absolute_errors.extend(torch.abs(reconstructed - logged).cpu().tolist())
    return {
        "rows": len(absolute_errors),
        "mae": statistics.fmean(absolute_errors) if absolute_errors else math.inf,
        "max_error": max(absolute_errors, default=math.inf),
    }


def _config_payload(config: GroupRelativeSignalAuditConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("trajectory_path", "corpus_summary_path", "parent_checkpoint", "output_path"):
        payload[key] = str(payload[key])
    return payload


def _validate(config: GroupRelativeSignalAuditConfig) -> None:
    if config.expected_group_size < 2 or config.min_groups <= 0:
        raise ValueError("Group size and minimum groups are invalid.")
    if config.reconstruction_rows <= 0 or config.batch_size <= 0:
        raise ValueError("Reconstruction rows and batch size must be positive.")
    if not 0.0 <= config.min_informative_group_rate <= 1.0:
        raise ValueError("min_informative_group_rate must be in [0, 1].")
    if not 0.0 <= config.min_nongreedy_row_rate <= 1.0:
        raise ValueError("min_nongreedy_row_rate must be in [0, 1].")


def main() -> None:
    args = build_parser().parse_args()
    payload = audit_group_relative_signal(
        GroupRelativeSignalAuditConfig(
            trajectory_path=Path(args.trajectory_path),
            corpus_summary_path=Path(args.corpus_summary_path),
            parent_checkpoint=Path(args.parent_checkpoint),
            output_path=Path(args.output_path),
            expected_group_size=args.expected_group_size,
            min_groups=args.min_groups,
            min_informative_group_rate=args.min_informative_group_rate,
            min_nongreedy_row_rate=args.min_nongreedy_row_rate,
            max_behavior_log_prob_mae=args.max_behavior_log_prob_mae,
            reconstruction_rows=args.reconstruction_rows,
            batch_size=args.batch_size,
            sampling_epsilon=args.sampling_epsilon,
            sampling_temperature=args.sampling_temperature,
            sampling_top_k=args.sampling_top_k,
            seed=args.seed,
            device=args.device,
        )
    )
    print(
        "Group-relative signal audit:"
        f" pass={payload['pass']}"
        f" groups={payload['corpus']['groups_total']}"
        f" informative_rate={payload['corpus']['informative_group_rate']:.3f}"
        f" behavior_mae={payload['behavior_probability']['reconstruction']['mae']:.3g}"
    )


if __name__ == "__main__":
    main()

"""Compare frozen low- and high-budget PUCT on existing all-legal return labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from typing import Any

from splendor_ai.bots.puct_bot import InformationSetPuctBot, PuctConfig
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_residual_error_audit import _bootstrap_mean_ci
from splendor_ai.training.run_root_prior_control_ablation import (
    _load_replay_rows,
    _state_key,
)


VARIANTS = (
    ("s64_frozen", 64),
    ("s512_frozen", 512),
)


@dataclass(frozen=True, slots=True)
class PuctBudgetAblationConfig:
    audit_path: Path
    replay_paths: tuple[Path, ...]
    policy_checkpoint: Path
    output_path: Path
    c_puct: float = 1.5
    fpu_reduction: float = 0.0
    max_depth: int = 80
    inference_batch_size: int = 8
    n_step_leaf_depth: int = 2
    n_step_leaf_width: int = 1
    n_step_leaf_candidate_limit: int = 4
    n_step_leaf_delta_clip: float = 0.20
    n_step_leaf_min_turn_index: int = 40
    seed: int = 0
    bootstrap_samples: int = 5_000
    device: str = "cpu"
    log_every_states: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen-prior s64 versus s512 clipped PUCT against existing "
            "all-legal matched-return action labels."
        )
    )
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--fpu-reduction", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--n-step-leaf-depth", type=int, default=2)
    parser.add_argument("--n-step-leaf-width", type=int, default=1)
    parser.add_argument("--n-step-leaf-candidate-limit", type=int, default=4)
    parser.add_argument("--n-step-leaf-delta-clip", type=float, default=0.20)
    parser.add_argument("--n-step-leaf-min-turn-index", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--log-every-states", type=int, default=10)
    return parser


def _puct_config(config: PuctBudgetAblationConfig, simulations: int) -> PuctConfig:
    return PuctConfig(
        simulations=simulations,
        c_puct=config.c_puct,
        fpu_reduction=config.fpu_reduction,
        max_depth=config.max_depth,
        seed=config.seed,
        inference_batch_size=config.inference_batch_size,
        n_step_leaf_depth=config.n_step_leaf_depth,
        n_step_leaf_width=config.n_step_leaf_width,
        n_step_leaf_candidate_limit=config.n_step_leaf_candidate_limit,
        n_step_leaf_delta_clip=config.n_step_leaf_delta_clip,
        n_step_leaf_min_turn_index=config.n_step_leaf_min_turn_index,
        root_prior_mix=1.0,
    )


def _variant_summary(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    all_variants = [row["variants"][variant] for row in rows]
    resolved_variants = [row["variants"][variant] for row in rows if row["resolved"]]
    return {
        "states": len(rows),
        "resolved_states": len(resolved_variants),
        "mean_regret": statistics.fmean(row["regret"] for row in resolved_variants) if resolved_variants else 0.0,
        "oracle_match_rate": statistics.fmean(float(row["oracle_match"]) for row in resolved_variants) if resolved_variants else 0.0,
        "mean_elapsed_seconds": statistics.fmean(row["elapsed_seconds"] for row in all_variants) if all_variants else 0.0,
        "mean_simulations_per_second": statistics.fmean(row["simulations_per_second"] for row in all_variants) if all_variants else 0.0,
        "source_replay_choice_match_rate": statistics.fmean(
            float(row["variants"][variant]["selected_action_index"] == row["source_puct_choice"])
            for row in rows
        ) if rows else 0.0,
    }


def _comparison_summary(rows: list[dict[str, Any]], config: PuctBudgetAblationConfig) -> dict[str, Any]:
    resolved = [row for row in rows if row["resolved"]]
    regret_delta = [
        row["variants"]["s512_frozen"]["regret"] - row["variants"]["s64_frozen"]["regret"]
        for row in resolved
    ]
    agreement_delta = [
        float(row["variants"]["s512_frozen"]["oracle_match"])
        - float(row["variants"]["s64_frozen"]["oracle_match"])
        for row in resolved
    ]
    return_delta = [
        row["variants"]["s512_frozen"]["selected_return"]
        - row["variants"]["s64_frozen"]["selected_return"]
        for row in resolved
    ]
    action_change = [
        float(
            row["variants"]["s512_frozen"]["selected_action_index"]
            != row["variants"]["s64_frozen"]["selected_action_index"]
        )
        for row in rows
    ]
    regret = _bootstrap_mean_ci(regret_delta, config.seed + 31, config.bootstrap_samples)
    agreement = _bootstrap_mean_ci(agreement_delta, config.seed + 32, config.bootstrap_samples)
    return_gain = _bootstrap_mean_ci(return_delta, config.seed + 33, config.bootstrap_samples)
    return {
        "resolved_states": len(resolved),
        "regret_delta_s512_minus_s64": regret,
        "oracle_match_delta_s512_minus_s64": agreement,
        "selected_return_delta_s512_minus_s64": return_gain,
        "action_change_rate": _bootstrap_mean_ci(action_change, config.seed + 34, config.bootstrap_samples),
        "gate": {
            "passed": regret["ci95_high"] < 0.0 and agreement["ci95_low"] > 0.0,
            "criteria": {
                "regret_ci95_upper_lt_zero": regret["ci95_high"] < 0.0,
                "oracle_match_ci95_lower_gt_zero": agreement["ci95_low"] > 0.0,
            },
            "reason": (
                "High-budget PUCT improves both resolved-state regret and oracle agreement."
                if regret["ci95_high"] < 0.0 and agreement["ci95_low"] > 0.0
                else "Do not advance; s512 has not cleared both offline action-quality gates."
            ),
        },
    }


def run_puct_budget_ablation(config: PuctBudgetAblationConfig) -> dict[str, Any]:
    _validate_config(config)
    audit = json.loads(config.audit_path.read_text(encoding="utf-8"))
    replay_rows = _load_replay_rows(config.replay_paths)
    env = SplendorEnv()
    codec = ActionCodec()
    bots = {
        name: InformationSetPuctBot(
            config.policy_checkpoint,
            device=config.device,
            config=_puct_config(config, simulations),
        )
        for name, simulations in VARIANTS
    }
    result_rows: list[dict[str, Any]] = []
    missing_states: list[dict[str, Any]] = []
    for position, audit_row in enumerate(audit.get("states", [])):
        replay_row = replay_rows.get(_state_key(audit_row))
        if replay_row is None:
            missing_states.append({key: audit_row[key] for key in ("game_seed", "turn_index", "player_id")})
            continue
        state = deserialize_state_snapshot(replay_row["state_snapshot"])
        legal_actions = env.legal_actions(state)
        returns_by_action = {
            int(action["action_index"]): float(action["rollout_return"])
            for action in audit_row["actions"]
        }
        oracle_action = int(audit_row["oracle_action_index"])
        oracle_return = returns_by_action[oracle_action]
        state_variants: dict[str, dict[str, Any]] = {}
        for name, _simulations in VARIANTS:
            bot = bots[name]
            action = bot.choose_action(env, state, legal_actions)
            if action is None:
                raise RuntimeError("PUCT budget ablation found no legal action.")
            action_index = codec.encode(state, action)
            if action_index not in returns_by_action:
                raise RuntimeError("PUCT selected action missing from all-legal audit targets.")
            stats = bot.last_search_stats
            elapsed = stats.elapsed_seconds
            state_variants[name] = {
                "selected_action_index": action_index,
                "selected_return": returns_by_action[action_index],
                "regret": oracle_return - returns_by_action[action_index],
                "oracle_match": action_index == oracle_action,
                "elapsed_seconds": elapsed,
                "simulations_per_second": stats.simulations / elapsed if elapsed > 0.0 else 0.0,
                "root_effective_actions": stats.root_visit_effective_actions,
                "root_visited_actions": stats.root_visited_actions,
            }
        result_rows.append(
            {
                "game_seed": audit_row["game_seed"],
                "turn_index": audit_row["turn_index"],
                "player_id": audit_row["player_id"],
                "phase": audit_row["phase"],
                "resolved": bool(audit_row["resolved"]),
                "oracle_action_index": oracle_action,
                "oracle_return": oracle_return,
                "source_puct_choice": audit_row.get("puct_choice"),
                "variants": state_variants,
            }
        )
        if config.log_every_states > 0 and (position + 1) % config.log_every_states == 0:
            print(f"[puct-budget-ablation] evaluated={position + 1}/{len(audit['states'])}")
    summary = {
        "states_evaluated": len(result_rows),
        "resolved_states": sum(row["resolved"] for row in result_rows),
        "missing_states": missing_states,
        "variants": {name: _variant_summary(result_rows, name) for name, _simulations in VARIANTS},
        "comparison": _comparison_summary(result_rows, config),
    }
    output = {
        "audit": "puct_budget_action_quality",
        "config": {
            **asdict(config),
            "audit_path": str(config.audit_path),
            "replay_paths": [str(path) for path in config.replay_paths],
            "policy_checkpoint": str(config.policy_checkpoint),
            "output_path": str(config.output_path),
            "variants": [{"name": name, "simulations": simulations} for name, simulations in VARIANTS],
        },
        "source_audit": {
            "path": str(config.audit_path),
            "resolved_states": audit.get("summary", {}).get("resolved_states"),
        },
        "summary": summary,
        "states": result_rows,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _validate_config(config: PuctBudgetAblationConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.device != "cpu":
        raise ValueError("This preregistered budget gate is CPU-only.")
    if (
        config.c_puct <= 0.0
        or config.fpu_reduction < 0.0
        or config.max_depth <= 0
        or config.inference_batch_size <= 0
        or config.n_step_leaf_depth <= 0
        or config.n_step_leaf_width <= 0
        or config.n_step_leaf_candidate_limit <= 0
        or config.n_step_leaf_delta_clip <= 0.0
        or config.n_step_leaf_min_turn_index < 0
        or config.bootstrap_samples < 0
    ):
        raise ValueError("Budget-ablation search settings are invalid.")


def main() -> None:
    args = build_parser().parse_args()
    output = run_puct_budget_ablation(
        PuctBudgetAblationConfig(
            audit_path=Path(args.audit_path),
            replay_paths=tuple(Path(path) for path in args.replay_path),
            policy_checkpoint=Path(args.policy_checkpoint),
            output_path=Path(args.output_path),
            c_puct=args.c_puct,
            fpu_reduction=args.fpu_reduction,
            max_depth=args.max_depth,
            inference_batch_size=args.inference_batch_size,
            n_step_leaf_depth=args.n_step_leaf_depth,
            n_step_leaf_width=args.n_step_leaf_width,
            n_step_leaf_candidate_limit=args.n_step_leaf_candidate_limit,
            n_step_leaf_delta_clip=args.n_step_leaf_delta_clip,
            n_step_leaf_min_turn_index=args.n_step_leaf_min_turn_index,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            device=args.device,
            log_every_states=args.log_every_states,
        )
    )
    print(
        "[puct-budget-ablation]"
        f" evaluated={output['summary']['states_evaluated']}"
        f" resolved={output['summary']['resolved_states']}"
        f" output={args.output_path}"
    )


if __name__ == "__main__":
    main()

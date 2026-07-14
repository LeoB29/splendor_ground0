"""Compare root-only PUCT prior controls on existing counterfactual audit states."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
from typing import Any

from splendor_ai.bots.puct_bot import InformationSetPuctBot, PuctConfig
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.run_residual_error_audit import _bootstrap_mean_ci
from splendor_ai.training.replay import deserialize_state_snapshot


EXPLORATORY_VARIANTS = (
    ("frozen", 1.0),
    ("mixed_050", 0.5),
    ("uniform", 0.0),
)

CONFIRMATION_VARIANTS = (
    ("frozen", 1.0),
    ("uniform", 0.0),
)


@dataclass(frozen=True, slots=True)
class RootPriorControlAblationConfig:
    audit_path: Path
    replay_paths: tuple[Path, ...]
    policy_checkpoint: Path
    output_path: Path
    simulations: int = 64
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
    include_mixed_control: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen, mixed, and uniform root PUCT priors against the "
            "matched-return actions in a residual-error audit."
        )
    )
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--simulations", type=int, default=64)
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-states", type=int, default=10)
    parser.add_argument(
        "--include-mixed-control",
        action="store_true",
        help="Include the already exploratory 50/50 root-prior control.",
    )
    return parser


def _variants(config: RootPriorControlAblationConfig) -> tuple[tuple[str, float], ...]:
    return EXPLORATORY_VARIANTS if config.include_mixed_control else CONFIRMATION_VARIANTS


def _state_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["game_seed"]), int(row["turn_index"]), int(row["player_id"])


def _load_replay_rows(paths: tuple[Path, ...]) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row.get("state_snapshot"), dict):
                    rows[_state_key(row)] = row
    return rows


def _puct_config(config: RootPriorControlAblationConfig, root_prior_mix: float) -> PuctConfig:
    return PuctConfig(
        simulations=config.simulations,
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
        root_prior_mix=root_prior_mix,
    )


def _variant_summary(rows: list[dict[str, Any]], variant: str, config: RootPriorControlAblationConfig) -> dict[str, Any]:
    variant_rows = [row["variants"][variant] for row in rows]
    resolved = [row for row in rows if row["resolved"]]
    resolved_variant_rows = [row["variants"][variant] for row in resolved]
    return {
        "states": len(variant_rows),
        "resolved_states": len(resolved),
        "mean_regret": statistics.fmean(row["regret"] for row in resolved_variant_rows) if resolved_variant_rows else 0.0,
        "oracle_match_rate": statistics.fmean(float(row["oracle_match"]) for row in resolved_variant_rows) if resolved_variant_rows else 0.0,
        "mean_root_effective_actions": statistics.fmean(row["root_effective_actions"] for row in variant_rows) if variant_rows else 0.0,
        "mean_root_visited_actions": statistics.fmean(row["root_visited_actions"] for row in variant_rows) if variant_rows else 0.0,
        "source_replay_choice_match_rate": statistics.fmean(
            float(row["variants"][variant]["selected_action_index"] == row["source_puct_choice"])
            for row in rows
        ) if rows else 0.0,
    }


def _comparison_summary(rows: list[dict[str, Any]], variant: str, config: RootPriorControlAblationConfig) -> dict[str, Any]:
    resolved = [row for row in rows if row["resolved"]]
    regret_deltas = [
        row["variants"][variant]["regret"] - row["variants"]["frozen"]["regret"]
        for row in resolved
    ]
    match_deltas = [
        float(row["variants"][variant]["oracle_match"])
        - float(row["variants"]["frozen"]["oracle_match"])
        for row in resolved
    ]
    return_deltas = [
        row["variants"][variant]["selected_return"] - row["variants"]["frozen"]["selected_return"]
        for row in resolved
    ]
    action_changes = [
        float(
            row["variants"][variant]["selected_action_index"]
            != row["variants"]["frozen"]["selected_action_index"]
        )
        for row in rows
    ]
    regret = _bootstrap_mean_ci(regret_deltas, config.seed + 11, config.bootstrap_samples)
    match = _bootstrap_mean_ci(match_deltas, config.seed + 12, config.bootstrap_samples)
    return_delta = _bootstrap_mean_ci(return_deltas, config.seed + 13, config.bootstrap_samples)
    return {
        "resolved_states": len(resolved),
        "regret_delta_vs_frozen": regret,
        "oracle_match_delta_vs_frozen": match,
        "selected_return_delta_vs_frozen": return_delta,
        "action_change_rate_vs_frozen": _bootstrap_mean_ci(action_changes, config.seed + 14, config.bootstrap_samples),
        "gate": {
            "passed": regret["ci95_high"] < 0.0 and match["ci95_low"] > 0.0,
            "criteria": {
                "regret_ci95_upper_lt_zero": regret["ci95_high"] < 0.0,
                "oracle_match_ci95_lower_gt_zero": match["ci95_low"] > 0.0,
            },
            "reason": (
                "Candidate improves both resolved-state regret and oracle agreement."
                if regret["ci95_high"] < 0.0 and match["ci95_low"] > 0.0
                else "Do not advance; the root-prior control has not cleared both offline gates."
            ),
        },
    }


def run_root_prior_control_ablation(config: RootPriorControlAblationConfig) -> dict[str, Any]:
    _validate_config(config)
    audit = json.loads(config.audit_path.read_text(encoding="utf-8"))
    audit_rows = list(audit.get("states", []))
    replay_rows = _load_replay_rows(config.replay_paths)
    env = SplendorEnv()
    codec = ActionCodec()
    variants = _variants(config)
    bots = {
        name: InformationSetPuctBot(
            config.policy_checkpoint,
            device=config.device,
            config=_puct_config(config, root_prior_mix),
        )
        for name, root_prior_mix in variants
    }
    result_rows: list[dict[str, Any]] = []
    missing_states: list[dict[str, Any]] = []
    for position, audit_row in enumerate(audit_rows):
        replay_row = replay_rows.get(_state_key(audit_row))
        if replay_row is None:
            missing_states.append({key: audit_row[key] for key in ("game_seed", "turn_index", "player_id")})
            continue
        state = deserialize_state_snapshot(replay_row["state_snapshot"])
        legal_actions = env.legal_actions(state)
        action_by_index = {
            codec.encode(state, action): action
            for action in legal_actions
        }
        returns_by_action = {
            int(action["action_index"]): float(action["rollout_return"])
            for action in audit_row["actions"]
        }
        oracle_action = int(audit_row["oracle_action_index"])
        oracle_return = returns_by_action[oracle_action]
        state_variants: dict[str, dict[str, Any]] = {}
        for name, _mix in variants:
            bot = bots[name]
            action = bot.choose_action(env, state, legal_actions)
            if action is None:
                raise RuntimeError("PUCT root-control ablation found no legal action.")
            action_index = codec.encode(state, action)
            if action_index not in returns_by_action:
                raise RuntimeError("PUCT selected action missing from all-legal residual audit targets.")
            stats = bot.last_search_stats
            state_variants[name] = {
                "selected_action_index": action_index,
                "selected_return": returns_by_action[action_index],
                "regret": oracle_return - returns_by_action[action_index],
                "oracle_match": action_index == oracle_action,
                "root_effective_actions": stats.root_visit_effective_actions,
                "root_visited_actions": stats.root_visited_actions,
                "root_prior_effective_actions": stats.root_prior_effective_actions,
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
            print(f"[root-prior-control] evaluated={position + 1}/{len(audit_rows)}")
    summary = {
        "states_evaluated": len(result_rows),
        "resolved_states": sum(row["resolved"] for row in result_rows),
        "missing_states": missing_states,
        "variants": {
            name: _variant_summary(result_rows, name, config)
            for name, _mix in variants
        },
        "comparisons_vs_frozen": {
            name: _comparison_summary(result_rows, name, config)
            for name, _mix in variants
            if name != "frozen"
        },
    }
    output = {
        "audit": "puct_root_prior_control_ablation",
        "config": {
            **asdict(config),
            "audit_path": str(config.audit_path),
            "replay_paths": [str(path) for path in config.replay_paths],
            "policy_checkpoint": str(config.policy_checkpoint),
            "output_path": str(config.output_path),
            "variants": [{"name": name, "root_prior_mix": mix} for name, mix in variants],
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


def _validate_config(config: RootPriorControlAblationConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if (
        config.simulations <= 0
        or config.max_depth <= 0
        or config.inference_batch_size <= 0
        or config.n_step_leaf_depth <= 0
        or config.n_step_leaf_width <= 0
        or config.n_step_leaf_candidate_limit <= 0
    ):
        raise ValueError("Search and n-step settings must be positive.")
    if config.c_puct <= 0.0 or config.fpu_reduction < 0.0 or config.n_step_leaf_delta_clip <= 0.0:
        raise ValueError("PUCT, FPU, and leaf clip settings are invalid.")
    if config.n_step_leaf_min_turn_index < 0 or config.bootstrap_samples < 0:
        raise ValueError("Turn and bootstrap settings are invalid.")


def main() -> None:
    args = build_parser().parse_args()
    output = run_root_prior_control_ablation(
        RootPriorControlAblationConfig(
            audit_path=Path(args.audit_path),
            replay_paths=tuple(Path(path) for path in args.replay_path),
            policy_checkpoint=Path(args.policy_checkpoint),
            output_path=Path(args.output_path),
            simulations=args.simulations,
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
            include_mixed_control=args.include_mixed_control,
        )
    )
    print(
        "[root-prior-control]"
        f" evaluated={output['summary']['states_evaluated']}"
        f" resolved={output['summary']['resolved_states']}"
        f" output={args.output_path}"
    )


if __name__ == "__main__":
    main()

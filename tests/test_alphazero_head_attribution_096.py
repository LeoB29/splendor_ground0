from __future__ import annotations

from pathlib import Path

from splendor_ai.alphazero.head_attribution import (
    attribution_decision,
    paired_arena_difference_bootstrap,
)
from splendor_ai.alphazero.network import (
    AlphaZeroInference,
    HybridAlphaZeroInference,
    create_initial_checkpoint,
)
from splendor_ai.engine.env import SplendorEnv


def _arena(points: list[tuple[float, float]]) -> dict[str, object]:
    rows = []
    for offset, pair in enumerate(points):
        for seat, value in enumerate(pair):
            rows.append({"seed": 100 + offset, "candidate_seat": seat, "candidate_points": value})
    return {"rows": rows}


def test_hybrid_inference_uses_exact_requested_heads(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.pt"
    value_path = tmp_path / "value.pt"
    create_initial_checkpoint(policy_path, seed=1)
    create_initial_checkpoint(value_path, seed=2)
    env = SplendorEnv(seed=96)
    state = env.initial_state()
    actions = env.legal_actions(state)
    policy_prediction = AlphaZeroInference(policy_path).predict_one(state, actions)
    value_prediction = AlphaZeroInference(value_path).predict_one(state, actions)
    hybrid = HybridAlphaZeroInference(policy_path, value_path).predict_one(state, actions)
    assert hybrid.logits == policy_prediction.logits
    assert hybrid.value == value_prediction.value
    hybrid_inference = HybridAlphaZeroInference(policy_path, value_path)
    assert hybrid_inference.checkpoint.path == policy_path
    assert hybrid_inference.value_checkpoint.path == value_path


def test_difference_bootstrap_is_paired_by_seed() -> None:
    left = _arena([(1.0, 1.0), (0.5, 1.0), (0.0, 1.0)])
    right = _arena([(0.0, 0.0), (0.5, 0.0), (0.0, 0.0)])
    result = paired_arena_difference_bootstrap(left, right, samples=1000, seed=96)
    assert result["paired_seeds"] == 3
    assert result["mean_difference"] == (1.0 + 0.5 + 0.5) / 3
    assert result["ci95_low"] >= 0.5


def test_attribution_rubric_is_strict_and_exhaustive() -> None:
    positive = {"ci95_low": 0.001}
    zero = {"ci95_low": 0.0}
    assert attribution_decision(positive, zero) == "child_value_head_primary_blocker"
    assert attribution_decision(zero, positive) == "child_policy_head_primary_blocker"
    assert attribution_decision(positive, positive) == "both_heads_or_joint_incompatibility"
    assert attribution_decision(zero, zero) == "no_clear_gameplay_head_attribution"

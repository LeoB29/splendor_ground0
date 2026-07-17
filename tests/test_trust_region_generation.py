from __future__ import annotations

from splendor_ai.training.run_trust_region_generation import evaluate_projected_gates


def _stream(change=0.04, kl=0.009):
    return {
        "top_action_change_rate": change,
        "raw_policy_kl": {"mean": kl},
    }


def test_projected_gate_requires_loss_movement_and_raw_kl() -> None:
    projection = {
        "selected_alpha": 0.4,
        "parent": {"policy_loss": 1.0},
        "projected": {"policy_loss": 0.95, "raw_kl": 0.008},
    }

    gates = evaluate_projected_gates(projection, _stream(), _stream(), 1e-7)

    assert all(gates.values())
    projection["projected"]["policy_loss"] = 0.98
    assert not all(
        evaluate_projected_gates(projection, _stream(), _stream(), 1e-7).values()
    )
    projection["projected"]["policy_loss"] = 0.95
    assert not all(
        evaluate_projected_gates(projection, _stream(kl=0.011), _stream(), 1e-7).values()
    )

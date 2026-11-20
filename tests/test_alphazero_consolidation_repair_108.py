from __future__ import annotations

import pytest
import torch

from splendor_ai.alphazero.consolidation_repair import (
    INTERPOLATION_LAMBDAS,
    interpolate_state_dicts,
    interpolation_name,
    repair_decision,
    retention_gate,
    select_repair,
)


def _metrics(objective: float, corpus_deltas: dict[int, float] | None = None) -> dict[str, object]:
    deltas = corpus_deltas or {7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0}
    return {
        "aggregate": {"rows": 100, "objective_loss": objective},
        "corpora": {
            str(generation): {"rows": 25, "objective_loss": 2.0 + deltas[generation]}
            for generation in range(7, 11)
        },
    }


def test_interpolate_state_dicts_hits_exact_endpoints_and_midpoint() -> None:
    parent = {"weight": torch.tensor([0.0, 2.0])}
    child = {"weight": torch.tensor([2.0, 6.0])}
    assert torch.equal(interpolate_state_dicts(parent, child, 0.0)["weight"], parent["weight"])
    assert torch.equal(interpolate_state_dicts(parent, child, 1.0)["weight"], child["weight"])
    assert torch.equal(
        interpolate_state_dicts(parent, child, 0.25)["weight"],
        torch.tensor([0.5, 3.0]),
    )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        interpolate_state_dicts(parent, child, 1.1)


def test_108_interpolation_grid_is_fixed_and_uniquely_named() -> None:
    assert INTERPOLATION_LAMBDAS == (0.0625, 0.125, 0.25, 0.5, 0.75)
    assert len({interpolation_name(value) for value in INTERPOLATION_LAMBDAS}) == 5


def test_retention_gate_reuses_107_boundaries() -> None:
    parent = _metrics(2.0)
    passed = _metrics(1.99, {7: -0.01, 8: -0.02, 9: 0.01, 10: 0.005})
    assert retention_gate(passed, parent)["passed"]
    failed = _metrics(1.99, {7: -0.01, 8: -0.02, 9: 0.01, 10: 0.0051})
    assert not retention_gate(failed, parent)["passed"]


def test_selection_prefers_best_coherent_then_hybrid_fallback() -> None:
    variants = {
        "parent": {"kind": "anchor", **_metrics(2.0), "retention_gate": {"passed": False}},
        "full_candidate": {"kind": "anchor", **_metrics(2.1), "retention_gate": {"passed": False}},
        "hybrid_candidate_policy_parent_value": {
            "kind": "hybrid_value_isolation",
            **_metrics(1.98),
            "retention_gate": {"passed": True},
        },
        "interpolation_0625": {
            "kind": "coherent_interpolation",
            "lambda": 0.0625,
            **_metrics(1.99),
            "retention_gate": {"passed": True},
        },
        "interpolation_1250": {
            "kind": "coherent_interpolation",
            "lambda": 0.125,
            **_metrics(1.97),
            "retention_gate": {"passed": True},
        },
    }
    assert select_repair(variants) == {
        "pathway": "coherent_interpolation",
        "variant": "interpolation_1250",
        "lambda": 0.125,
    }
    for name in ("interpolation_0625", "interpolation_1250"):
        variants[name]["retention_gate"]["passed"] = False
    assert select_repair(variants) == {
        "pathway": "hybrid_value_isolation",
        "variant": "hybrid_candidate_policy_parent_value",
    }


def test_repair_decision_separates_coherent_and_value_diagnosis() -> None:
    coherent = {"pathway": "coherent_interpolation"}
    hybrid = {"pathway": "hybrid_value_isolation"}
    assert "parent_safe" in repair_decision(coherent, {"passed": True})
    assert "gameplay_failed" in repair_decision(coherent, {"passed": False})
    assert "value_blocker_confirmed" in repair_decision(hybrid, {"passed": True})
    assert "did_not_transfer" in repair_decision(hybrid, {"passed": False})
    assert "no_repair_variant" in repair_decision(None, None)

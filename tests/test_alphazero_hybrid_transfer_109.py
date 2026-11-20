from __future__ import annotations

from splendor_ai.alphazero.hybrid_transfer import hybrid_transfer_decision


def test_hybrid_transfer_decision_is_controlled_only_by_primary_pass() -> None:
    assert "value_update_implicated" in hybrid_transfer_decision({"passed": True})
    assert "close_consolidation_branch" in hybrid_transfer_decision({"passed": False})

from copy import deepcopy
import pytest
from tools.local_player_qualification_128.verify import expected, compare


def fixture():
    anchor = {"keys": ["a", "b"], "actions": [{"a": 1}, {"b": 2}]}
    candidates = {"keys": ["a", "b", "c"], "actions": [{"a": 1}, {"b": 2}, {"c": 3}]}
    return {"root_identity": ["main", 0, 0], "planner": "candidate", "privacy": {},
            "schedule": {"world_seeds": list(range(128))}, "candidates": candidates,
            "proposal": {"anchor_candidates": anchor, "candidates": candidates,
                         "breadth": True, "added_indices": {"ranked": [2], "random": []}},
            "scores": [1.0], "worlds": [
                {"seed": i, "outcomes": [{"wdl": -1.0, "reason": "completed"},
                    {"wdl": -1.0, "reason": "completed"}, {"wdl": 1.0, "reason": "completed"}]}
                for i in range(128)], "means": [-1.0, -1.0, 1.0], "choice": 2}


def test_independent_prefix_recomputes_first_tied_choice():
    source = fixture()
    result = expected(source, False)
    assert result["record"]["choice"] == 0
    assert result["action"] == {"a": 1}
    assert source["choice"] == 2
    assert len(result["record"]["worlds"][0]["outcomes"]) == 2


def test_changed_hash_or_discrete_action_fails_even_with_same_payoff():
    a = {"outcome": 1.0, "hash": "original", "action": {"slot": 1}}
    b = deepcopy(a)
    b["hash"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        compare(a, b)
    b = deepcopy(a)
    b["action"]["slot"] = True
    with pytest.raises(ValueError, match="slot"):
        compare(a, b)


def test_numeric_tolerance_does_not_accept_nan():
    compare(1.0 + 1e-8, 1.0, "result.record.scores[0]")
    with pytest.raises(ValueError):
        compare(1.0 + 1e-8, 1.0, "result.record.worlds[0].outcomes[0].wdl")
    with pytest.raises(ValueError):
        compare(float("nan"), 1.0)


def test_truncated_original_or_corrupted_saved_choice_fails():
    value = fixture()
    value["worlds"].pop()
    with pytest.raises(ValueError, match="budget"):
        expected(value, True)
    value = fixture()
    value["choice"] = 0
    with pytest.raises(ValueError, match="choice"):
        expected(value, True)

"""Exact JSON-boundary repair and unchanged frozen verifier contracts."""
import ast
import copy
import json
from pathlib import Path

import pytest

from tools.equal_budget_breadth_arena_128_recovery_02 import verify as previous
from tools.equal_budget_breadth_arena_128_recovery_03 import verify as current


@pytest.fixture
def snapshot():
    state = previous.old.SplendorEnv(seed=17).initial_state()
    result = previous.old.serialize_state_snapshot(state)
    # These ordered fields must remain order-sensitive at the resume boundary.
    result["players"][0]["purchased_card_ids"] = ["T1_58", "T1_04"]
    result["players"][0]["reserved_card_ids"] = ["T1_20", "T2_29"]
    result["players"][0]["reserved_card_sources"] = ["visible", "visible"]
    return result


def test_json_integer_keys_round_trip_is_the_only_representation_repair(snapshot):
    saved = json.loads(json.dumps(snapshot))
    assert snapshot != saved
    assert set(snapshot["deck_counts"]) == {1, 2, 3}
    assert set(saved["deck_counts"]) == {"1", "2", "3"}
    current._assert_inherited_snapshot(snapshot, saved, 13, 17)
    current._assert_inherited_snapshot(saved, snapshot, 13, 17)


@pytest.mark.parametrize("mutation", [
    "bank_token", "player_token", "bonus", "score", "turn", "actor", "terminal",
    "winner", "deck_count", "deck_order", "market_order", "purchased_order",
    "reserved_order", "nobles_order", "extra_field", "missing_field",
])
def test_actual_snapshot_changes_fail_with_game_and_turn(snapshot, mutation):
    changed = json.loads(json.dumps(snapshot))
    if mutation == "bank_token": changed["bank_tokens"]["blue"] -= 1
    elif mutation == "player_token": changed["players"][0]["tokens"]["gold"] += 1
    elif mutation == "bonus": changed["players"][0]["bonuses"]["red"] += 1
    elif mutation == "score": changed["players"][1]["score"] += 1
    elif mutation == "turn": changed["turn_index"] += 1
    elif mutation == "actor": changed["current_player"] = 1
    elif mutation == "terminal": changed["terminal"] = True
    elif mutation == "winner": changed["winner"] = 0
    elif mutation == "deck_count": changed["deck_counts"]["1"] -= 1
    elif mutation == "deck_order": changed["hidden_tier_decks"]["1"].reverse()
    elif mutation == "market_order": changed["visible_tier_cards"]["1"].reverse()
    elif mutation == "purchased_order": changed["players"][0]["purchased_card_ids"].reverse()
    elif mutation == "reserved_order": changed["players"][0]["reserved_card_ids"].reverse()
    elif mutation == "nobles_order": changed["nobles"].reverse()
    elif mutation == "extra_field": changed["unexpected"] = 0
    elif mutation == "missing_field": del changed["start_player"]
    else: raise AssertionError(mutation)
    with pytest.raises(AssertionError, match=r"Game 13, turn 17: inherited snapshot canonical mismatch"):
        current._assert_inherited_snapshot(snapshot, changed, 13, 17)


def test_reordered_object_keys_are_not_state_changes(snapshot):
    saved = json.loads(json.dumps(snapshot))
    saved = dict(reversed(list(saved.items())))
    saved["deck_counts"] = dict(reversed(list(saved["deck_counts"].items())))
    current._assert_inherited_snapshot(snapshot, saved, 310, 58)


def test_nonfinite_values_cannot_be_accepted(snapshot):
    changed = copy.deepcopy(snapshot)
    changed["players"][0]["score"] = float("nan")
    with pytest.raises(ValueError):
        current._assert_inherited_snapshot(changed, changed, 13, 17)


@pytest.mark.parametrize("index,prefix_turns", [(13, 17), (317, 0)])
@pytest.mark.parametrize("changed_boundary", [False, True])
def test_actual_audit_game_crosses_saved_json_prefix_without_inner_inference(
        index, prefix_turns, changed_boundary, monkeypatch):
    # Exercise the real outer-game auditor, saved provenance and sample. Inner
    # trajectories are intentionally stubbed; this receipt is never persisted.
    base = current.c.DATA
    lock = current.old.read_json(base / "main_games_lock.json")
    item = lock["files"][index]
    game = current.old.read_json(current.c.resolve_game("main", index, base))
    assert game["inherited_turns"] == prefix_turns
    checked_turns, boundaries = [], []
    real_prefix = current._check_game_prefix
    real_boundary = current._assert_inherited_snapshot
    def prefix(*args, **kwargs):
        inherited = real_prefix(*args, **kwargs)
        assert inherited is not None and len(inherited["trace"]) == prefix_turns
        if changed_boundary:
            inherited = copy.deepcopy(inherited)
            inherited["final_state"]["bank_tokens"]["blue"] += 1
        return inherited
    def boundary(snapshot, inherited_snapshot, game_index, turn):
        boundaries.append((game_index, turn))
        assert set(snapshot["deck_counts"]) == {1, 2, 3}
        assert set(inherited_snapshot["deck_counts"]) == {"1", "2", "3"}
        return real_boundary(snapshot, inherited_snapshot, game_index, turn)
    def root(stage, game_index, state, *args, **kwargs):
        checked_turns.append(state.turn_index)
        return {"test_only_turn": state.turn_index}
    def forbidden(*args, **kwargs):
        raise AssertionError("Outer boundary test must not perform neural inference")
    monkeypatch.setattr(current, "_check_game_prefix", prefix)
    monkeypatch.setattr(current, "_assert_inherited_snapshot", boundary)
    monkeypatch.setattr(current, "audit_root", root)
    monkeypatch.setattr(current, "independent_proposal", forbidden)
    monkeypatch.setattr(current.a149, "replay_branch", forbidden)
    monkeypatch.setattr(current.p, "initialize_models", forbidden)
    if changed_boundary:
        with pytest.raises(AssertionError, match=f"Game {index}, turn {prefix_turns}: inherited snapshot canonical mismatch"):
            current.audit_game("main", index, item["sha256"], current.c.ORIGINAL_BINDING, base)
        assert len(checked_turns) == prefix_turns
    else:
        receipt = current.audit_game("main", index, item["sha256"], current.c.ORIGINAL_BINDING, base)
        assert receipt["game"]["index"] == index
        assert receipt["game"]["turns"] == len(checked_turns) == len(game["trace"])
    assert boundaries == [(index, prefix_turns)]


def _tree(module):
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8-sig"))


def _dump(node):
    return ast.dump(node, include_attributes=False)


def test_every_other_verifier_function_is_ast_identical_to_frozen_r2():
    earlier = {node.name: node for node in _tree(previous).body if isinstance(node, ast.FunctionDef)}
    repaired = {node.name: node for node in _tree(current).body if isinstance(node, ast.FunctionDef)}
    assert set(repaired) == set(earlier) | {"_assert_inherited_snapshot"}
    for name in earlier.keys()-{"audit_game"}:
        assert _dump(repaired[name]) == _dump(earlier[name]), name


def test_complete_module_ast_has_only_declared_import_and_boundary_changes():
    earlier, repaired = _tree(previous), _tree(current)
    expected_assert = ast.parse('assert old.serialize_state_snapshot(state) == inherited["final_state"]').body[0]
    replacement = ast.parse(
        '_assert_inherited_snapshot(old.serialize_state_snapshot(state), inherited["final_state"], index, state.turn_index)'
    ).body[0]
    class DeclaredRepair(ast.NodeTransformer):
        imports = 0
        boundaries = 0
        def visit_ImportFrom(self, node):
            if node.module == "tools.equal_budget_breadth_arena_128_recovery_02" and node.names == []:
                raise AssertionError("Unexpected empty import")
            if (node.module == "tools.equal_budget_breadth_arena_128_recovery_02"
                    and [(alias.name, alias.asname) for alias in node.names] == [("common", "c")]):
                self.imports += 1
                node.module = "tools.equal_budget_breadth_arena_128_recovery_03"
            return node
        def visit_Assert(self, node):
            if _dump(node) == _dump(expected_assert):
                self.boundaries += 1
                return copy.deepcopy(replacement)
            return node
    transform = DeclaredRepair()
    expected = transform.visit(earlier)
    assert (transform.imports, transform.boundaries) == (1, 1)
    repaired.body = [node for node in repaired.body
                     if not (isinstance(node, ast.FunctionDef) and node.name == "_assert_inherited_snapshot")]
    assert _dump(repaired) == _dump(expected)

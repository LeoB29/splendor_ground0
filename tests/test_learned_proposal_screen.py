"""Scientific, replay and launch contracts; synthetic fixtures never become evidence."""
import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from scripts.analysis import learned_proposal_screen as p
from scripts.analysis import verify_learned_proposal_screen as audit


@pytest.fixture(scope="module", autouse=True)
def initialize():
    p.initialize_models()


def state_cases():
    env = p.old.SplendorEnv(seed=2039500000)
    first = env.initial_state()
    hidden = first
    for _ in range(2):
        action = next(a for a in env.legal_actions(hidden) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        hidden = env.step(hidden, action)
    rich = copy.deepcopy(hidden)
    rich.turn_index = 42
    for player in rich.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return first, hidden, rich


def synthetic_scores(keys, winners):
    return {str(seed): [float(key == winner) for key in keys] for seed, winner in zip(p.MODELS, winners)}


def test_fourth_slot_union_and_three_common_candidates():
    keys = list("abcdefghij")
    reference = keys[:4]
    values = synthetic_scores(keys, "fgh")
    raw = list("abcefdghij")
    result = p.assemble_slots(keys, keys, reference, raw, values)
    assert result["union_keys"] == list("abcdefgh")
    assert result["arm_keys"]["reference"] == list("abcd")
    assert result["arm_keys"]["raw_control"] == list("abce")
    assert [result["arm_keys"][f"learned_{s}"][-1] for s in p.MODELS] == list("fgh")
    assert all(v[:3] == list("abc") and len(set(v)) == 4 for v in result["arm_keys"].values())


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_small_support_and_coincidences_are_retained(size):
    keys = list("abcd")[:size]
    result = p.assemble_slots(keys, keys, keys, keys, synthetic_scores(keys, [keys[-1]] * 3))
    assert all(result["arm_keys"][arm] == keys for arm in p.ARMS)
    assert result["union_keys"] == keys


def test_learned_ties_keep_first_canonical_group_not_raw_order():
    keys = list("abcdef")
    values = {str(seed): [0.] * len(keys) for seed in p.MODELS}
    result = p.assemble_slots(keys, keys, list("abce"), list("abcfed"), values)
    assert result["arm_keys"]["raw_control"][-1] == "f"
    assert all(result["arm_keys"][arm][-1] == "d" for arm in p.ARMS[2:])


def test_reject_duplicate_offers_and_truncated_short_support():
    keys = list("abcde")
    values = synthetic_scores(keys, "ddd")
    for reference in (list("aabc"), list("abc")):
        with pytest.raises(ValueError):
            p.assemble_slots(keys, keys, reference, keys, values)


def test_actual_proposals_match_independent_and_frozen155():
    for state in state_cases():
        stream = "learned-proposal-unit-only"
        actual = p.proposals(state, stream)
        expected = audit.independent_proposals(state, stream)
        audit.compare(p.projection(actual), expected)
        assert actual["reference"] == p.baseline.candidate_set(state, actual["schedule"]["candidate_nonce"])
        assert len(actual["learned_logits"]) == 3
        assert all(len(v) == len(actual["legal_actions"]) for v in actual["learned_logits"].values())
        for arm in p.ARMS:
            assert len(actual["arm_indices"][arm]) == min(4, actual["reference"]["canonical_support"])


def test_private_allocations_and_deck_order_cannot_change_proposal():
    state = state_cases()[1]
    worlds = [audit.precedent.independent_world(state, seed) for seed in (2039531000, 2039531001)]
    assert p.old.serialize_state_snapshot(worlds[0]) != p.old.serialize_state_snapshot(worlds[1])
    a = p.projection(p.proposals(worlds[0], "privacy-fixture"))
    b = p.projection(p.proposals(worlds[1], "privacy-fixture"))
    audit.compare(a, b)


def test_selection_is_by_offered_order_and_sees_only_offered_actions():
    offers = {"union_actions": [0, 1, 2, 3, 4], "arm_indices": {arm: [2, 1, 0, 3] for arm in p.ARMS}}
    outcomes = [{"wdl": 0., "reason": "completed"} for _ in range(4)] + [{"wdl": 1., "reason": "completed"}]
    record = {"action_indices": list(range(5)), "worlds": [{"outcomes": outcomes} for _ in range(32)]}
    expected = p.choose_arms(record, offers)
    assert set(expected["choices"].values()) == {2}  # unseen winner4 cannot be chosen
    assert expected == audit.independent_choice(record, offers)
    broken = copy.deepcopy(record)
    broken["worlds"].pop()
    with pytest.raises(ValueError):
        p.choose_arms(broken, offers)
    broken = copy.deepcopy(record)
    broken["worlds"][0]["outcomes"][0]["reason"] = "max_turns"
    with pytest.raises(ValueError):
        p.choose_arms(broken, offers)


def test_selection_evaluation_and_probe_seeds_are_disjoint_and_bounded():
    sets = []
    for stage in ("probe", "main"):
        for kind in ("selection", "evaluation"):
            values = {p.world_seed(stage, kind, i, w) for i in range(p.CONFIG["states"][stage]) for w in range(p.CONFIG["worlds"][kind])}
            assert len(values) == p.CONFIG["states"][stage] * p.CONFIG["worlds"][kind]
            assert not any(values & old for old in sets)
            sets.append(values)
    with pytest.raises(ValueError):
        p.world_seed("main", "evaluation", 288, 0)


def statistic_fixture():
    return [{"index": i, "source_seed": i, "cell": i % 6,
             "primary": ((i % 13) - 4) / 256, "attribution": ((i % 7) - 4) / 256}
            for i in range(288)]


def test_independent_bootstrap_and_required_attribution_contrast():
    rows = statistic_fixture()
    production = p.statistics(rows)
    audit.compare(production, audit.independent_statistics(rows))
    assert production["primary"]["ci95_pp"][0] > 0
    assert production["attribution"]["ci95_pp"][1] < 0
    assert not production["conditional_learning_contribution_established"]
    for row in rows:
        row["primary"] = .125
        row["attribution"] = .0625
    result = p.statistics(rows)
    assert result["conditional_learning_contribution_established"]
    assert result["primary"] == {"delta_pp": 12.5, "ci95_pp": [12.5, 12.5]}
    assert result["attribution"] == {"delta_pp": 6.25, "ci95_pp": [6.25, 6.25]}
    audit.compare(result, audit.independent_statistics(rows))


def test_statistic_rejects_duplicate_sources_and_unbalanced_cells():
    rows = statistic_fixture()
    rows[1]["source_seed"] = rows[0]["source_seed"]
    with pytest.raises(ValueError):
        p.statistics(rows)
    rows = statistic_fixture()
    rows[1]["cell"] = 0
    with pytest.raises(ValueError):
        p.statistics(rows)


def test_full_collection_and_independent_replay_on_synthetic_short_game(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path / "study")
    monkeypatch.setattr(p, "ROOT", tmp_path)
    state = state_cases()[2]
    sample = {"index": 0, "stage": "probe", "source_index": 0, "source_seed": 2039500000,
              "cell": 4, "actor": state.current_player, "turn": state.turn_index,
              "stream": "synthetic-full-replay-only", "snapshot": p.old.serialize_state_snapshot(state)}
    binding = "unit-fixture-not-production"
    offered = p.proposal_task("probe", sample, binding)
    audit.proposal_check("probe", sample, offered, binding)
    collected = p.collect_task("probe", "selection", sample, offered, None, binding)
    receipt = audit.continuation_check("probe", "selection", sample, offered, collected, None, binding)
    assert p.verified(receipt)["full_inner_replay"]
    choice = p.choose_arms(p.verified(collected), p.verified(offered))
    p.old.save_new(p.OUTPUT / "probe/choices.json", {"manifest": binding, "rows": [choice]})
    evaluated = p.collect_task("probe", "evaluation", sample, offered, choice["choices"], binding)
    audit.continuation_check("probe", "evaluation", sample, offered, evaluated, choice["choices"], binding)
    row = p.per_root(sample, p.verified(offered), choice, p.verified(evaluated))
    independent = audit.independent_rows([sample], [p.verified(offered)], [choice], [p.verified(evaluated)])[0]
    audit.compare(row, independent)
    assert collected["continuations"] <= 256 and evaluated["continuations"] <= 640
    assert all(0 <= score <= 1 for score in row["scores"].values())
    with pytest.raises(FileExistsError):
        p.proposal_task("probe", sample, binding)
    path = p.ROOT / evaluated["path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Changed artifact"):
        p.verified(evaluated)


def test_source_replay_uses_synthetic_fixture_and_rejects_wrong_greedy_action(tmp_path, monkeypatch):
    original = p.old.SplendorEnv
    rich = state_cases()[2]
    # Seed is ignored only by this test's synthetic environment; no probe game is sampled.
    class FixtureEnv(original):
        def initial_state(self):
            return copy.deepcopy(rich)
    monkeypatch.setattr(p.old, "SplendorEnv", FixtureEnv)
    monkeypatch.setattr(p, "OUTPUT", tmp_path / "study")
    monkeypatch.setattr(p, "ROOT", tmp_path)
    record = p.source_task("probe", 0, "test")
    result = audit.source_check("probe", record, "test")
    assert p.verified(result)["passed"]
    value = p.verified(record)
    value["trace"][0]["state_hash"] = "corrupt"
    corrupt = p.OUTPUT / "corrupt_source.json"
    p.old.save_new(corrupt, value)
    with pytest.raises(ValueError, match="Source state"):
        audit.source_check("probe", p.artifact(corrupt, index=0), "test")


def test_invalid_identity_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    called = []
    monkeypatch.setattr(p.windows, "detached_launch", lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError):
        p.launch()
    assert not called

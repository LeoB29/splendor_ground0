"""Same-state candidates, disjoint worlds, sealed decisions and score units."""
import copy
from pathlib import Path

import pytest

from tools.proposal_diagnostic_128 import engine as p
from tools.proposal_iteration_128 import verify as parent_verify


@pytest.fixture(scope="module", autouse=True)
def models():
    p.initialize_models()


def private_state():
    env = p.old.SplendorEnv(seed=2089800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(v for v in env.legal_actions(state)
                      if v.action_type.name == "RESERVE_DECK" and v.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


def fixture_spec(index=0):
    snapshot = p.old.serialize_state_snapshot(private_state())
    return {"index": index, "pair": index, "fit": 161001, "role": "treatment", "phase": 0,
            "game_index": 2*index, "turn": snapshot["turn_index"], "source_stage": "test",
            "search_stream": "diagnostic-isolated-source", "snapshot_sha256": p.old.digest(snapshot),
            "fixture_snapshot": snapshot}


def synthetic_union(size, fourths=None):
    keys = list("abcdefg")[:size]
    reps = {key: {"action": key} for key in keys}
    reference = {"keys": keys[:4], "actions": [reps[k] for k in keys[:4]]}
    fourths = fourths or (keys[min(size-1, 3)],)*3
    proposals = {arm: {"candidates": {"keys": keys[:3]+([fourths[i]] if size >= 4 else []),
                "actions": [reps[k] for k in keys[:3]+([fourths[i]] if size >= 4 else [])]}}
                 for i, arm in enumerate(p.PROPOSERS[:3])}
    return reference, proposals, keys, reps


def test_integer_world_domains_disjoint_globally_and_from_parent():
    seen = set()
    for stage, count in (("probe", 12), ("main", 201), ("test", 12)):
        for index in range(count):
            for block in ("selection", "assessment"):
                seeds = p.world_seeds(stage, index, block)
                offset = {"probe": 0, "main": 1000000, "test": 2000000}[stage]
                offset += 512*index + (256 if block == "assessment" else 0)
                assert seeds == [2080000000+offset+i for i in range(128)]
                assert len(set(seeds)) == 128 and set(seeds).isdisjoint(seen)
                seen.update(seeds)
    original = p.parent.world128.root_schedule(private_state(), "diagnostic-isolated-source", 128)["world_seeds"]
    assert set(original).isdisjoint(seen)
    with pytest.raises(ValueError):
        p.world_seeds("main", 201, "selection")


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_short_support_retained_and_union_deduplicated(size):
    assembled = p.assemble_union(*synthetic_union(size), ("test", 0))
    keys = list("abcd")[:size]
    assert assembled["union"]["keys"] == keys
    for arm in p.PROPOSERS:
        assert assembled["candidates"][arm]["keys"] == keys
        assert assembled["proposed_indices"][arm] == (3 if size == 4 else None)


def test_three_different_challengers_and_identity_random_preserve_first_three():
    source = synthetic_union(7, ("d", "e", "f"))
    assembled = p.assemble_union(*source, ("main", 17))
    expected_random = min(list("defg"), key=lambda key: (p.old.digest([
        "proposal-diagnostic-128-random-fourth-v1", "main", 17, key]), key))
    assert assembled["union"]["keys"] == list("abcdef")+(["g"] if expected_random == "g" else [])
    for arm in p.PROPOSERS:
        assert assembled["candidates"][arm]["keys"][:3] == list("abc")
    assert assembled["candidates"]["random"]["keys"] == list("abc")+[expected_random]
    assert assembled["proposed_indices"]["treatment"] == 4
    damaged = copy.deepcopy(source)
    damaged[1]["treatment"]["candidates"]["keys"][0] = "e"
    with pytest.raises(ValueError):
        p.assemble_union(*damaged, ("main", 17))


@pytest.mark.parametrize("fit", [161001, 161002, 161003])
def test_all_model_roles_match_independent_parent_proposals_and_hidden_inputs(fit):
    state = private_state()
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    value = p.proposal_union(state, "lawful-diagnostic", fit, ("test", 0))
    altered = parent_verify.a157.hidden_alternative(state)
    other = p.proposal_union(altered, "lawful-diagnostic", fit, ("test", 0))
    for arm in ("anchor", "treatment", "replay"):
        projection = {k: v for k, v in value["proposals"][arm].items() if k != "timing"}
        parent_verify.compare(projection, parent_verify.independent_proposal(state, "lawful-diagnostic", f"{arm}_{fit}"))
        parent_verify.compare(projection, {k: v for k, v in other["proposals"][arm].items() if k != "timing"})
    assert value["union"] == other["union"] and value["candidates"] == other["candidates"]
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == before


def outcome_worlds(values):
    return [{"outcomes": [{"wdl": value} for value in values]} for _ in range(128)]


def test_missing_outcomes_tie_order_and_wdl_to_score_conversion():
    candidates = {arm: {"indices": [2, 0, 1]} for arm in p.ARMS}
    values, choices = p.choose(outcome_worlds([1., 0., 1.]), candidates, 3)
    assert values == [1., 0., 1.] and all(choice == 2 for choice in choices.values())
    summary = p.summarize(outcome_worlds([-1., 0., 1.]), choices,
                          dict(zip(p.PROPOSERS, [0, 1, 2, None])), 3)
    assert set(summary["arm_scores"].values()) == {1.}
    assert summary["fixed_proposal_scores"] == {"anchor": 0., "treatment": .5, "replay": 1., "random": None}
    assert summary["assessment_means_wdl"] == [-1., 0., 1.]
    missing = outcome_worlds([0., 0., 0.])
    missing[-1]["outcomes"].pop()
    with pytest.raises(ValueError):
        p.choose(missing, candidates, 3)


def test_real_private_root_seals_selection_before_assessment(tmp_path, monkeypatch):
    spec = fixture_spec()
    original = p.old.trajectory
    calls = 0
    union_size = len(p.proposal_union(private_state(), spec["search_stream"], spec["fit"], ("test", 0))["union"]["keys"])
    def observed(world, action):
        nonlocal calls
        if calls == 128*union_size:
            seal = p.old.read_json(p.selection_path("test", 0, tmp_path))
            assert len(seal["selection"]["worlds"]) == 128
            assert set(seal["selection_lock"]["choices"]) == set(p.ARMS)
            assert "assessment" not in seal
        calls += 1
        return original(world, action)
    monkeypatch.setattr(p.old, "trajectory", observed)
    artifact = p.collect("test", spec, "isolated-diagnostic", tmp_path)
    record = p.c.verified(artifact)
    seal = p.c.verified(record["selection_artifact"])
    assert "failure" not in record and calls == artifact["continuations"] == 256*union_size
    assert seal["selection_lock"] == record["selection_lock"]
    assert seal["identity"] == p.proposal_identity(record)
    assert p.old.digest(record["selection"]) == record["selection_lock"]["selection_sha256"]
    assert record["selection_lock"]["created_at_ns"] <= Path(p.selection_path("test", 0, tmp_path)).stat().st_mtime_ns <= record["assessment_started_at_ns"]
    assert record["privacy"] == {"own": 1, "opponent": 1}
    row = p.statistics_row(record)
    assert row["arm_scores"] == record["summary"]["arm_scores"] and row["pair"] == spec["pair"]
    tampered = copy.deepcopy(record)
    tampered["summary"]["arm_scores"]["treatment"] += .125
    with pytest.raises(ValueError):
        p.statistics_row(tampered)
    before = p.old.file_hash(p.root_path("test", 0, tmp_path))
    with pytest.raises(ValueError):
        p.collect("test", spec, "isolated-diagnostic", tmp_path)
    assert p.old.file_hash(p.root_path("test", 0, tmp_path)) == before


def test_fixture_guard_and_failure_preservation(tmp_path, monkeypatch):
    spec = fixture_spec()
    with pytest.raises(ValueError):
        p.collect("test", spec, "test", p.OUTPUT)
    with pytest.raises(ValueError):
        p.collect("main", spec, "test", tmp_path)
    assert "failure" in p.old.read_json(p.root_path("main", 0, tmp_path))
    def invalid(*args):
        return {"reason": "max_turns", "winner": None}
    monkeypatch.setattr(p.old, "trajectory", invalid)
    failed = tmp_path / "failed"
    with pytest.raises(ValueError):
        p.collect("test", spec, "test", failed)
    record = p.old.read_json(p.root_path("test", 0, failed))
    assert record["failure"] and record["selection"]["worlds"][0]["outcomes"][0]["wdl"] is None
    assert not p.selection_path("test", 0, failed).exists()


def test_selection_survives_assessment_failure(tmp_path, monkeypatch):
    spec = fixture_spec()
    original = p.run_block
    def fail_second(state, actions, seeds, block, base):
        if seeds[0] == p.world_seeds("test", 0, "assessment")[0]:
            assert p.selection_path("test", 0, base).is_file()
            raise RuntimeError("isolated assessment failure")
        return original(state, actions, seeds, block, base)
    monkeypatch.setattr(p, "run_block", fail_second)
    with pytest.raises(RuntimeError, match="isolated assessment failure"):
        p.collect("test", spec, "test", tmp_path)
    assert p.old.read_json(p.selection_path("test", 0, tmp_path))["selection_lock"]["choices"]
    assert "failure" in p.old.read_json(p.root_path("test", 0, tmp_path))

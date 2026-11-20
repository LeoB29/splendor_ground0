"""Independent-world, supported-choice and durable-evidence engine checks."""
import copy

import pytest

from . import engine as p
from scripts.analysis import verify_fast_selector_transfer_157 as privacy_audit


def fixture_registry():
    return p.old.read_json(p.ROOT/"outputs/proposal_breadth_learning_128/models_lock.json")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    p.initialize_models(tmp_path_factory.mktemp("signal-capture-models"), fixture_registry())


def outcome(value):
    return {"wdl": value, "winner": None if value in (None, 0) else (0 if value == 1 else 1),
            "reason": "repetition_cutoff" if value is None else "completed"}


def worlds(values):
    return [{"seed": i, "outcomes": [outcome(v) for v in values]} for i in range(128)]


def proposals(size, broad=None, narrow=None):
    values = [float(-i) for i in range(size)]
    return {arm: {"offered_group_scores": list(scores)} for arm, scores in
            (("anchor", values), ("broad", values if broad is None else broad), ("narrow", values if narrow is None else narrow))}


def candidates(size):
    return {"keys": [f"k{i}" for i in range(size)], "actions": [{"fixture": i} for i in range(size)]}


def private_state():
    env = p.old.SplendorEnv(seed=2219800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


def real_fixture_root():
    state = private_state()
    groups = p.old.actions_by_canonical_successor_key(state, p.old.SplendorEnv().legal_actions(state))
    keys, reps = [repr(k) for k in groups], {repr(k): p.old.serialize_action(a) for k, a in groups.items()}
    initial = {"keys": keys[:8], "actions": [reps[k] for k in keys[:8]]}
    scored = p.frozen_proposals(state, initial, 161001)["anchor"]
    scores = dict(zip(scored["group_keys"], scored["learned_group_scores"], strict=True))
    ordered = keys[:3] + sorted(keys[3:], key=lambda k: (-scores[k], keys.index(k)))[:5]
    offered = {"keys": ordered, "actions": [reps[k] for k in ordered]}
    assert len(ordered) == 8
    values = [0, 0, 0, 1, -1, 0, 0, 0]
    return {"snapshot": p.old.serialize_state_snapshot(state), "privacy": p.baseline.privacy(state),
            "candidates": offered, "worlds": worlds(values), "choice": 3, "means": list(map(float, values)),
            "schedule": {"world_seeds": list(range(128))}}


def fixture_spec(root=None, index=0):
    root = real_fixture_root() if root is None else root
    state = p.old.deserialize_state_snapshot(root["snapshot"])
    return {"index": index, "pair": index//4, "game_index": index//2, "fit": 161001,
            "seat": state.current_player, "phase": p.old.phase(state.turn_index), "split": "train",
            "turn": state.turn_index, "snapshot_sha256": p.old.digest(root["snapshot"]), "fixture_root": root}


def test_namespaces_do_not_overlap_or_allow_new_selection_worlds():
    from tools.planner_state_transfer_128 import engine as previous
    seen = set()
    for stage, count in (("probe", 18), ("main", 336), ("test", 18)):
        for index in range(count):
            seeds = p.world_seeds(stage, index)
            assert len(seeds) == len(set(seeds)) == 128 and seen.isdisjoint(seeds)
            seen.update(seeds)
            for old_block in ("selection", "assessment"):
                assert set(seeds).isdisjoint(previous.world_seeds(stage, index, old_block))
    assert min(seen) == 2210000000
    with pytest.raises(ValueError, match="namespace"):
        p.world_seeds("main", 0, "selection")


@pytest.mark.parametrize("size", range(1, 9))
def test_short_supports_and_original_order_ties_are_retained(size):
    source = worlds([0]*size)
    selected = p.rebuild_choices(source, proposals(size), candidates(size))
    assert selected["choices"]["anchor"] == selected["choices"]["teacher"] == 0
    for arm in p.PROPOSERS:
        assert selected["choices"][f"{arm}_direct"] == 0
        assert selected["choices"][f"{arm}_supported"] == 0
        assert selected["choices"][f"{arm}_outsider"] == (3 if size > 3 else None)
    summary = p.summarize(source, selected["choices"])
    assert summary["outsider_available"] == (size > 3)
    assert summary["delta_bounds"] == {name: [0., 0.] for name in p.CONTRASTS}
    if size <= 3:
        assert summary["arm_scores"]["teacher_outsider"] is None
        assert summary["arm_score_bounds"]["original_fourth"] is None


def test_supported_picker_uses_old_rollouts_not_proposer_confidence():
    source = worlds([-1, 0, 0, -1, 1, 0, -1, -1])
    predictions = proposals(8, [0, 0, 0, 0, 99, 0, 0, 0], [0, 0, 0, 0, 0, 99, 0, 0])
    value = p.rebuild_choices(source, predictions, candidates(8))
    assert value["choices"]["anchor"] == 1
    assert value["choices"]["teacher"] == value["choices"]["broad_supported"] == 4
    assert value["choices"]["narrow_direct"] == value["choices"]["narrow_outsider"] == 5
    assert value["choices"]["narrow_supported"] == 1
    assert value["supports"]["broad_supported"] == [0, 1, 2, 4]
    fresh = worlds([1, -1, -1, 1, -1, 1, 1, 1])
    report = p.summarize(fresh, value["choices"])
    assert report["choices"] == value["choices"]
    assert report["fresh_selection"]["teacher_certificate"]["choice"] == 0
    assert not report["fresh_selection"]["teacher_agreement"]
    assert report["delta_bounds"]["teacher_minus_anchor"] == [0., 0.]
    assert "fresh_teacher" not in report["arm_scores"]


def test_restricted_logits_are_finite_and_anchor_fourth_identity_is_required():
    damaged = proposals(5, broad=[0, 1, 2, 3, float("nan")])
    with pytest.raises(ValueError, match="logits"):
        p.rebuild_choices(worlds([0]*5), damaged, candidates(5))
    damaged = proposals(5)
    damaged["anchor"]["offered_group_scores"][4] = 100
    with pytest.raises(ValueError, match="Original fourth"):
        p.rebuild_choices(worlds([0]*5), damaged, candidates(5))


def test_old_null_cells_are_certified_conservatively_and_never_imputed():
    source = worlds([1, -1, -1, -1])
    source[0]["outcomes"][0] = outcome(None)
    selected = p.rebuild_choices(source, proposals(4), candidates(4))
    assert selected["means"][0] is None and selected["means_bounds"][0] == [126/128, 1.]
    assert selected["choices"]["teacher"] == 0
    source = worlds([0, 0, -1, -1])
    source[0]["outcomes"][0] = outcome(None)
    with pytest.raises(p.AmbiguousDecision) as failed:
        p.rebuild_choices(source, proposals(4), candidates(4))
    assert failed.value.selection["choices"]["teacher"] is None


def test_fresh_censoring_preserves_bounds_exact_cancellation():
    source = worlds([0, 0, 0, 1, 0])
    prediction = proposals(5, broad=[0, 0, 0, 0, 10])
    selected = p.rebuild_choices(source, prediction, candidates(5))
    fresh = worlds([0]*5)
    fresh[0]["outcomes"][3] = outcome(None)
    summary = p.summarize(fresh, selected["choices"])
    assert summary["arm_scores"]["anchor"] is None
    assert summary["arm_score_bounds"]["anchor"] == [127/256, 129/256]
    assert summary["delta_bounds"]["teacher_minus_anchor"] == [0., 0.]
    assert summary["delta_bounds"]["broad_outsider_minus_anchor_outsider"] == [-1/256, 1/256]
    assert summary["fresh_selection"]["teacher_certificate"]["choice"] is None


@pytest.mark.parametrize("reason,value", [("turn_limit", None), ("repetition_cutoff", 0.), ("completed", None), ("completed", True)])
def test_invalid_and_imputed_outcomes_fail(reason, value):
    with pytest.raises(ValueError):
        p.outcome_value({"reason": reason, "winner": None, "wdl": value})


def test_registry_is_private_and_transactional(models, tmp_path):
    before = dict(p._MODEL_ARTIFACTS)
    changed = fixture_registry()
    changed["models"]["broad_161001"]["source_fit"] = 161003
    with pytest.raises(ValueError, match="fit"):
        p.initialize_models(tmp_path, changed)
    assert p._MODEL_ARTIFACTS == before
    with pytest.raises(ValueError, match="production"):
        p.initialize_models(p.OUTPUT, fixture_registry())


@pytest.mark.parametrize("fit", [161001, 161002, 161003])
def test_private_information_does_not_affect_frozen_model_choices(models, fit):
    from tools.planner_state_transfer_128 import engine as previous
    before = dict(previous._MODELS), dict(previous._MODEL_ARTIFACTS)
    root, state = real_fixture_root(), private_state()
    snapshot = p.old.digest(p.old.serialize_state_snapshot(state))
    result = p.frozen_proposals(state, root["candidates"], fit)
    changed = p.frozen_proposals(privacy_audit.hidden_alternative(state), root["candidates"], fit)
    for arm in p.PROPOSERS:
        a = {k: v for k, v in result[arm].items() if k != "timing"}
        b = {k: v for k, v in changed[arm].items() if k != "timing"}
        privacy_audit.compare(a, b)
        assert a["model_sha256"] == fixture_registry()["models"][f"{arm}_{fit}"]["sha256"]
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == snapshot
    assert (previous._MODELS, previous._MODEL_ARTIFACTS) == before


def test_real_fixture_saves_lock_before_any_fresh_trajectory(models, tmp_path, monkeypatch):
    spec = fixture_spec()
    original, calls = p.old.trajectory, []
    def observed(world, action, **kwargs):
        seal = p.old.read_json(p.selection_path("test", 0, tmp_path))
        assert "assessment" not in seal and seal["selection_lock"]["choices"]["teacher"] == 3
        calls.append(action)
        return original(world, action, **kwargs)
    monkeypatch.setattr(p.old, "trajectory", observed)
    item = p.collect("test", spec, "fixture", tmp_path)
    record = p.c.verified(item)
    assert len(calls) == item["continuations"] == 1024
    assert record["selection_lock"]["created_at_ns"] <= p.selection_path("test", 0, tmp_path).stat().st_mtime_ns <= record["assessment_started_at_ns"]
    assert p.statistics_row(record)["split"] == "train"
    damaged = copy.deepcopy(record)
    damaged["summary"]["delta_bounds"]["teacher_minus_anchor"][0] += .125
    with pytest.raises(ValueError, match="Summary"):
        p.statistics_row(damaged)
    before = p.old.file_hash(p.root_path("test", 0, tmp_path))
    with pytest.raises(ValueError, match="One-shot"):
        p.collect("test", spec, "fixture", tmp_path)
    assert p.old.file_hash(p.root_path("test", 0, tmp_path)) == before


def test_mid_assessment_failure_keeps_lock_and_partial_outcomes(models, tmp_path, monkeypatch):
    spec = fixture_spec()
    calls = 0
    original = p.old.trajectory
    def fail(world, action, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("isolated fresh outcome failure")
        return original(world, action, **kwargs)
    monkeypatch.setattr(p.old, "trajectory", fail)
    with pytest.raises(RuntimeError, match="isolated fresh"):
        p.collect("test", spec, "fixture", tmp_path)
    assert p.selection_path("test", 0, tmp_path).exists()
    record = p.old.read_json(p.root_path("test", 0, tmp_path))
    assert "failure" in record and len(record["assessment"]["worlds"][0]["outcomes"]) == 2
    assert record["assessment"]["timing"]["wall"] >= 0


def test_corrupt_old_teacher_and_source_proofs_fail_before_any_fresh_work(models, tmp_path, monkeypatch):
    spec = fixture_spec()
    spec["fixture_root"]["choice"] = 2
    monkeypatch.setattr(p.old, "trajectory", lambda *a, **k: pytest.fail("No simulation before a valid lock"))
    with pytest.raises(ValueError, match="Original choice"):
        p.collect("test", spec, "fixture", tmp_path)
    assert not p.selection_path("test", 0, tmp_path).exists()
    spec = fixture_spec(index=1)
    spec["fixture_root"]["equivalence_proofs"] = [{"unsupported": True}]
    with pytest.raises(ValueError, match="Unsupported"):
        p.collect("test", spec, "fixture", tmp_path)


def test_fixture_and_production_boundaries(models, tmp_path):
    spec = fixture_spec()
    with pytest.raises(ValueError, match="production"):
        p.collect("test", spec, "fixture", p.OUTPUT)
    with pytest.raises(ValueError, match="Synthetic"):
        p.collect("main", spec, "fixture", tmp_path)

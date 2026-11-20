"""Bounded real035 prefix, terminal, public-value and timing tests."""
import copy
import hashlib
import random
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from splendor_ai.alphazero import run_value_generalization_scale as value143
from scripts.analysis import verify_fast_selector_transfer_157 as privacy
from . import engine as e, verify as v


@pytest.fixture(scope="module", autouse=True)
def initialized():
    e.old.initialize_worker()
    ref = e.c.artifact(e.c.ROOT/"outputs/value_generalization_scale_143/simple.json")
    e._EVALUATOR = ref
    e._COEFFICIENTS = torch.tensor(e.c.verified(ref)["coefficients"], dtype=torch.float64)


def near_terminal():
    env = e.old.SplendorEnv(seed=2289900000)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


@pytest.fixture(scope="module")
def saved_root():
    state = near_terminal()
    legal = e.old.SplendorEnv().legal_actions(state)
    action = next(a for a in legal if a.action_type.name == "BUY_VISIBLE"
                  and state.visible_tier_cards[a.tier][a.market_index].points > 0)
    spec = {"index": 0, "pair": 0, "game_index": 0, "fit": 161001, "seat": state.current_player,
            "phase": 0, "split": "train"}
    worlds = []
    for wi in range(128):
        seed = 2289900100+wi
        world = e.old.determinize_for_observer(state, state.current_player, random.Random(seed))
        result = e.old.trajectory(world, action)
        result["wdl"] = e.old.outcome_value(result["winner"], state.current_player)
        assert result["reason"] in e.old.VALID_NATURAL_TERMINATIONS
        worlds.append({"seed": seed, "state_sha256": e.old.digest(e.old.serialize_state_snapshot(world)), "outcomes": [result]})
    return {"spec": spec, "snapshot": e.old.serialize_state_snapshot(state),
        "candidates": {"actions": [e.old.serialize_action(action)],
                       "keys": [repr(e.old.canonical_successor_action_key(state, action))]},
        "assessment": {"worlds": worlds}, "old_teacher_choice": 0}


def install_source(tmp_path, monkeypatch, saved_root):
    root = copy.deepcopy(saved_root)
    path = tmp_path/"source.json"
    e.old.save_new(path, root)
    ref = e.c.artifact(path)
    fake = SimpleNamespace(load_root=lambda spec, binding, base, stage: (root, ref),
        evaluator=lambda base: e._EVALUATOR, audit_indices=lambda spec: list(range(8)),
        timing_world_indices=lambda spec: list(range(8)))
    monkeypatch.setattr(sys.modules[e.__package__], "source", fake, raising=False)
    return root, fake


def test_actual_prefixes_equal_mature_full_trajectory_and_independent_replay():
    state = e.old.SplendorEnv(seed=2289900900).initial_state()
    action = next(a for a in e.old.SplendorEnv().legal_actions(state) if a.action_type.name == "TAKE_TOKENS")
    prefix = e.traverse(state, action, state.current_player, (4, 8, 16))
    full = e.old.trajectory(state, action, include_trace=True)
    assert full["turns"] > 16
    v._same(prefix["endpoints"], v.replay_prefix(state, action, state.current_player, (4, 8, 16), e._COEFFICIENTS.tolist()))
    for h in (4, 8, 16):
        hasher = hashlib.sha256()
        for row in full["trace"][:h]: hasher.update(e.old.digest(row["action"]).encode())
        assert prefix["endpoints"][str(h)]["trace_sha256"] == hasher.hexdigest()
        assert prefix["endpoints"][str(h)]["plies"] == h
    assert full["trace"][0]["action"] == e.old.serialize_action(action)


@pytest.mark.parametrize("actor", [0, 1])
def test_143_features_match_frozen_evaluator_and_root_actor_perspective(actor):
    state = near_terminal()
    state.players[actor].tokens["blue"] += 1
    state.current_player = actor
    features = e.public_features(e.public_facts(state), actor)
    assert torch.equal(features, value143.simple_features(state))
    assert features.tolist() == v.independent_features(v.facts_from_state(state), actor)
    assert torch.equal(features, -e.public_features(e.public_facts(state), 1-actor))
    state.current_player = 1-actor
    assert torch.equal(features, e.public_features(e.public_facts(state), actor))


def test_public143_ignores_private_reserve_and_hidden_deck_identity():
    state = near_terminal()
    alternate = privacy.hidden_alternative(state)
    assert e.public_facts(state) == e.public_facts(alternate)
    assert torch.equal(e.public_features(e.public_facts(state), 0), e.public_features(e.public_facts(alternate), 0))


def test_natural_terminal_prefix_uses_exact_score_and_stops(saved_root, monkeypatch):
    root = saved_root
    state = e.old.deserialize_state_snapshot(root["snapshot"])
    saved = root["assessment"]["worlds"][0]
    world = e._world(state, saved)
    action = e.old.decode_action(root["candidates"]["actions"][0])
    monkeypatch.setattr(e, "_COEFFICIENTS", torch.full((15,), 999., dtype=torch.float64))
    result = e.traverse(world, action, state.current_player, (4, 8, 16))
    assert result["outcome"] == saved["outcomes"][0]
    for point in result["endpoints"].values():
        assert point["features"] is point["predicted_wdl"] is None
        assert point["z"] == (1+saved["outcomes"][0]["wdl"])/2
        assert point["plies"] <= 4


def test_invalid_prefix_cutoff_cannot_be_imputed():
    state = e.old.SplendorEnv(seed=2289900999).initial_state()
    action = e.old.SplendorEnv().legal_actions(state)[0]
    state.turn_index = 400
    with pytest.raises(ValueError, match="Non-natural"):
        e.traverse(state, action, state.current_player, (4, 8, 16))


def test_collect_complete128_world_projection_and_all_endpoint_audit(tmp_path, monkeypatch, saved_root):
    root, _ = install_source(tmp_path, monkeypatch, saved_root)
    ref = e.collect(root["spec"], "fixture", tmp_path)
    produced = e.c.verified(ref)
    assert len(produced["y"]) == 128 and set(produced["z"]) == {"4", "8", "16"}
    assert produced["statistics"] == e.statistics_row(produced)
    monkeypatch.setattr(e, "traverse", lambda *a, **k: pytest.fail("Collector traversal entered auditor"))
    receipt = e.c.verified(v.audit(root["spec"], ref, "fixture", tmp_path))
    assert receipt["passed"] and receipt["checked_endpoints"] == 384 and receipt["replayed_candidates"] == 8


def test_seven_timing_arms_match_saved_outcomes_and_independent_prefixes(tmp_path, monkeypatch, saved_root):
    root, _ = install_source(tmp_path, monkeypatch, saved_root)
    ref = e.timing(root["spec"], "fixture", tmp_path)
    produced = e.c.verified(ref)
    assert len(produced["cells"]) == 8
    assert produced["timing"]["full_only"]["cells"] == 8
    assert e.c.verified(v.audit_timing(root["spec"], ref, "fixture", tmp_path))["passed"]


def test_validation_needs_linked_sealed_rule_before_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "_source", lambda *a, **k: pytest.fail("Opened VAL before calibration"))
    with pytest.raises(FileNotFoundError):
        e.collect({"index": 1}, "fixture", tmp_path, stage="validation")
    selected = {"manifest": "fixture", "status": "selected", "config": {"horizon": 8},
                "selected_before_validation": True, "validation_read": False}
    e.old.save_new(tmp_path/"selection_lock.json", selected)
    e.old.save_new(tmp_path/"selection_audit.json", {"manifest": "fixture", "passed": True,
                   "selection": e.c.artifact(tmp_path/"selection_lock.json")})
    assert e._horizons("validation", "fixture", tmp_path) == (8,)
    with pytest.raises(ValueError): e._horizons("validation", "wrong", tmp_path)


def test_fixture_stage_and_dynamic_default_do_not_touch_production(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Fixture cannot"):
        e._horizons("test", "fixture", e.c.OUTPUT)
    monkeypatch.setattr(e.c, "OUTPUT", tmp_path/"different")
    assert e._base(None) == tmp_path/"different"


def test_forecast_is_outcome_blind_and_includes_audit_storage_and_elapsed(tmp_path, monkeypatch, saved_root):
    root, _ = install_source(tmp_path, monkeypatch, saved_root)
    template = e.c.verified(e.timing(root["spec"], "fixture", tmp_path))
    refs = []
    for fit in e.c.FITS:
        for seat in (0, 1):
            for phase in range(3):
                record = copy.deepcopy(template)
                record.update(index=len(refs), fit=fit, seat=seat, phase=phase)
                # These unusable outcomes prove the forecast only consumes cost
                # measurements, shape and serialized endpoint-size accounting.
                for cell in record["cells"]:
                    for sample in cell["samples"]:
                        sample["result"]["outcome"] = {"not_an_outcome": True}
                        for point in sample["result"]["endpoints"].values(): point["z"] = "forbidden"
                path = tmp_path/f"timing_fixture_{len(refs)}.json"
                e.old.save_new(path, record); refs.append(e.c.artifact(path))
    e.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()-100})
    result = e.execution_forecast(refs, "fixture", tmp_path)
    assert result["outcome_blind"] and result["estimated_total_seconds"] >= 400
    assert result["main_candidate_prefixes"] == 344064
    assert result["maximum_audited_candidate_prefixes"] == (336*8+78*120)*8
    assert result["all_endpoint_checks"] == 168*128*8*4 and result["estimated_remaining_bytes"] > 0
    monkeypatch.setattr(e.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    assert not e.execution_forecast(refs, "fixture", tmp_path)["disk_reserve_pass"]

"""Actual lawful worlds, independent tensor export, and durable recovery fixtures."""
import copy
import json
from dataclasses import asdict

import pytest

from . import labels as p


def fixture_spec(state=None,index=0):
    """Pure isolated fixture; collection/audit perform actual128 terminal worlds."""
    env = p.c.old.SplendorEnv(seed=2408007000)
    if state is None:
        state = env.initial_state()
        for _ in range(2):
            state = env.step(state,next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2))
        state = env.step(state,next(a for a in env.legal_actions(state) if a.action_type.name == "TAKE_TOKENS"))
        for player in state.players:
            player.score = 14
            player.bonuses = {color:10 for color in player.bonuses}
    action = next(a for a in env.legal_actions(state) if a.action_type.name == "BUY_VISIBLE" and a.tier == 3)
    snap = p.c.old.serialize_state_snapshot(state)
    key = repr(p.c.old.canonical_successor_action_key(state,action))
    identity = {"union":{"keys":[key],"actions":[p.c.old.serialize_action(action)]},
                "anchor_keys":[key],"ranked_keys":[],"uniform_keys":[]}
    config = p.LegalActionAttentionScorerConfig(observation_size=295,action_feature_size=79,
        action_hidden_size=128,action_depth=2,state_hidden_size=256,state_depth=3,
        attention_heads=4,attention_layers=2,attention_feedforward_size=256,scorer_hidden_size=256,dropout=0.)
    return json.loads(json.dumps({"index":index,"row_index":12670+index,
        "row_id":f"independent_game_scaling_128:{index}:{state.turn_index}",
        "fit":161001,"seat":state.current_player,"phase":min(state.turn_index//20,2),"turn":state.turn_index,
        "privacy":p.c.baseline.privacy(state),"support":1,"canonical_support":1,"game_index":index,"pair":index,
        "source_stage":"train","search_stream":"fixture","snapshot_sha256":p.c.old.digest(snap),
        "fixture_payload":{"snapshot":snap,"identity":identity,"model_config":asdict(config)}}))


@pytest.fixture(scope="module")
def short_spec(): return fixture_spec()


def test_actual_collection_and_full_independent_tensor_audit(tmp_path,short_spec):
    ref = p.collect(short_spec,"fixture",tmp_path)
    root = p.validate_root(short_spec,"fixture",tmp_path,independent=True)
    assert root["worlds"] == root["recorded_continuations"] == 128 and root["sums_wdl"] == [128]
    checked = p.c.verified(p.audit(short_spec,"fixture",tmp_path,full=True))
    assert checked["root"] == {k:ref[k] for k in ("path","sha256")}
    assert checked["replayed_continuations"] == 128 and checked["full_inner_replay"]
    assert checked["independent_public_private_invariant_encoding"] and checked["all_canonical_memberships_checked"]
    assert p.audit(short_spec,"fixture",tmp_path,full=True)["sha256"] == p.c.artifact(p.audit_path(0,tmp_path))["sha256"]


@pytest.mark.parametrize("damage",["none","coherent_outcomes","unsealed"])
def test_interrupted_root_reuses_only_authenticated_completed_block(tmp_path,short_spec,monkeypatch,damage):
    original = p.c.old.save_new
    def interrupt(path,value):
        if path == p.root_path(0,tmp_path): raise RuntimeError("isolated interruption before root seal")
        return original(path,value)
    monkeypatch.setattr(p.c.old,"save_new",interrupt)
    with pytest.raises(RuntimeError): p.collect(short_spec,"fixture",tmp_path)
    monkeypatch.setattr(p.c.old,"save_new",original)
    path,receipt = p.block_path(0,tmp_path),p.block_receipt_path(0,tmp_path)
    assert path.exists() and receipt.exists() and not p.root_path(0,tmp_path).exists()
    if damage == "coherent_outcomes":
        value = p.c.old.read_json(path)
        for world in value["worlds"]: world["outcomes"][0].update(winner=0,wdl=-1.)
        value["sums_wdl"] = [-128]
        path.write_text(json.dumps(value))
    if damage == "unsealed": receipt.unlink()
    before = path.read_bytes()
    if damage == "none":
        p.collect(short_spec,"fixture",tmp_path)
        checked = p.c.verified(p.audit(short_spec,"fixture",tmp_path))
        assert checked["replayed_continuations"] == 8 and not checked["full_inner_replay"]
    else:
        with pytest.raises((ValueError,AssertionError)): p.collect(short_spec,"fixture",tmp_path)
    assert path.read_bytes() == before


def test_unknown_is_replayed_and_holds_without_target(tmp_path,short_spec,monkeypatch):
    unknown = {"reason":"repetition_cutoff","winner":None,"turns":8,"trace_sha256":"1"*64,"final_state_sha256":"2"*64}
    monkeypatch.setattr(p.c.old,"trajectory",lambda state,action:dict(unknown))
    calls = []
    def replay(world,action,actor):
        calls.append(actor)
        return {**unknown,"wdl":None}
    monkeypatch.setattr(p.continuation_audit,"replay_branch",replay)
    with pytest.raises(p.UnresolvedTarget): p.collect(short_spec,"fixture",tmp_path)
    assert len(calls) == 128 and not p.root_path(0,tmp_path).exists()
    receipt = p.c.old.read_json(tmp_path/"labels/unresolved/state_00000.json")
    assert receipt["all_unknown_branches_replayed"] and receipt["imputed"] is False


def test_independent_replay_detects_resealed_forged_trace(tmp_path,short_spec):
    p.collect(short_spec,"fixture",tmp_path)
    path = p.block_path(0,tmp_path)
    block = p.c.old.read_json(path)
    block["worlds"][p.ordinary_indices(0)[0]]["outcomes"][0]["final_state_sha256"] = "a"*64
    path.write_text(json.dumps(block))
    sealpath = p.block_receipt_path(0,tmp_path)
    seal = p.c.old.read_json(sealpath); seal["artifact"] = p.c.artifact(path)
    sealpath.write_text(json.dumps(seal))
    rootpath = p.root_path(0,tmp_path)
    root = p.c.old.read_json(rootpath); root["block"] = p.c.artifact(path,completion=p.c.artifact(sealpath))
    rootpath.write_text(json.dumps(root))
    p.validate_root(short_spec,"fixture",tmp_path,independent=True)
    with pytest.raises(AssertionError): p.audit(short_spec,"fixture",tmp_path)


def test_seed_namespaces_support_and_unknown_arithmetic():
    starts = [p.world_seeds(i)[0] for i in range(10800)]
    assert starts == list(range(2420000000,2420000000+10800*1024,1024))
    assert p.world_seeds(10799)[-1] == 2431058303
    assert len(set(p.ordinary_indices(10799))) == 8
    worlds = [{"outcomes":[{"wdl":v,"reason":"completed","winner":None} for v in (-1.,0.,1.)]} for _ in range(128)]
    assert p.sums(worlds,3) == ([-128,0,128],[]) and p.independent_sums(worlds,3) == [-128,0,128]
    worlds[1]["outcomes"][0].update(wdl=None,reason="repetition_cutoff")
    with pytest.raises(p.UnresolvedTarget): p.sums(worlds,3)
    with pytest.raises(AssertionError): p.independent_sums(worlds,3)


def test_teacher_expansion_preserves_anchor_ranked_ties_and_uniform_support():
    proposal = {"group_keys":list("abcdefghij"),"learned_group_scores":[1.]*10,
                "candidates":{"keys":list("abcd")},"reference":{"keys":list("abcd"),"actions":list("ABCD")},
                "group_representatives":list("ABCDEFGHIJ")}
    result = p.teacher_support(proposal,{"index":7})
    assert result["union"]["keys"][:6] == list("abcdef") and len(set(result["union"]["keys"])) == 8
    assert result["ranked_keys"] == list("ef") and set(result["uniform_keys"]) <= set("ghij")
    for n in range(1,9):
        small = copy.deepcopy(proposal)
        for field in ("group_keys","learned_group_scores","group_representatives"): small[field] = small[field][:n]
        for field in ("candidates","reference"):
            for key in small[field]: small[field][key] = small[field][key][:n]
        assert len(p.teacher_support(small,{"index":7})["union"]["keys"]) == n


def test_actual_original_anchor_proposal_independent_reconstruction_and_privacy(tmp_path):
    """Actual neural inference only on a synthetic isolated terminal fixture."""
    prior = p.c.old.read_json(p.c.ROOT/"outputs/proposal_breadth_learning_128/models_lock.json")
    registry = {"manifest":"fixture","source_only":True,
                "models":{f"anchor_{fit}":prior["models"][f"anchor_{fit}"] for fit in p.c.FITS}}
    p.c.old.save_new(tmp_path/"source_models_lock.json",registry)
    frozen_models,frozen_artifacts = dict(p.mature_engine._MODELS),dict(p.mature_engine._MODEL_ARTIFACTS)
    p.initialize_models(tmp_path)
    spec = fixture_spec()
    state = p.c.old.deserialize_state_snapshot(spec.pop("fixture_payload")["snapshot"])
    spec["canonical_support"] = len(p.c.old.actions_by_canonical_successor_key(state,p.c.old.SplendorEnv().legal_actions(state)))
    actual = p._identity(state,spec)
    expected = p._independent_identity(state,spec)
    p.mature_verify.compare(actual,expected)
    p.mature_verify.compare(expected,p._independent_identity(p.input_audit.hidden_alternative(state),spec))
    assert len(actual["union"]["keys"]) == min(8,spec["canonical_support"])
    assert p.mature_engine._MODELS == frozen_models and p.mature_engine._MODEL_ARTIFACTS == frozen_artifacts

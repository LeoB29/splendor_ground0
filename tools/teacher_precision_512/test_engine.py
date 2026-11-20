"""Private-model, original-teacher, interval and durable-selection regression checks."""
import copy
from pathlib import Path

import pytest

from tools.teacher_precision_512 import engine as p
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit


_FIXTURE_REGISTRY = None


def fixture_registry(base=None):
    global _FIXTURE_REGISTRY
    if _FIXTURE_REGISTRY is None:
        assert base is not None, "First fixture registry creation needs isolated output directory"
        base = Path(base)
        assert not base.resolve().is_relative_to(p.OUTPUT.resolve())
        from splendor_ai.alphazero.strong_policy import StrongPolicyInference, save_model
        import torch
        prior = p.old.read_json(p.ROOT/"outputs/proposal_breadth_learning_128/models_lock.json")
        models = {}
        for fit in p.c.FITS:
            anchor = copy.deepcopy(prior["models"][f"anchor_{fit}"])
            models[f"anchor_{fit}"] = anchor
            for arm in ("precision","control"):
                model = StrongPolicyInference(p.c.verify_hash(anchor)).model
                # Nonzero output exercises arithmetic and privacy in actual inference.
                with torch.no_grad():
                    model.score_head[-1].weight.add_(.01 if arm == "precision" else -.01)
                    model.score_head[-1].bias.add_(.02)
                path = base/"fixture_models"/f"{arm}_{fit}.pt"
                save_model(path,model,fit,{"source_fit":fit,"fixture":True,"update":0})
                models[f"{arm}_{fit}"] = p.c.artifact(path,source_fit=fit,arm=arm)
        _FIXTURE_REGISTRY = {"manifest":"fixture","models":models}
    return copy.deepcopy(_FIXTURE_REGISTRY)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    base = tmp_path_factory.mktemp("precision_transfer_engine")
    p.initialize_models(base, fixture_registry(base))


def private_state():
    env = p.old.SplendorEnv(seed=2189800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


def fixture_spec(index=0):
    state = private_state()
    snapshot = p.old.serialize_state_snapshot(state)
    return {"index": index, "pair": index, "game_index": index, "fit": 161001,
            "seat": state.current_player, "phase": 0, "turn": state.turn_index,
            "search_stream": "planner-transfer-isolated-source",
            "snapshot_sha256": p.old.digest(snapshot), "fixture_snapshot": snapshot}


def synthetic_union(size, fourths=None):
    keys = [f"k{j}" for j in range(size)]
    reps = {k: {"action": k} for k in keys}
    reference = {"keys": keys[:4], "actions": [reps[k] for k in keys[:4]]}
    fourths = fourths or (keys[min(size-1, 3)],)*3
    proposals = {}
    for i, arm in enumerate(p.PROPOSERS):
        offered = keys[:3]+([fourths[i]] if size >= 4 else [])
        proposals[arm] = {"candidates": {"keys": offered, "actions": [reps[k] for k in offered],
                                        "canonical_support": size, "legal_support": size},
                          "group_keys": keys, "learned_group_scores": [float(-j) for j in range(size)]}
    return reference, proposals, keys, reps, ("test", 0)


def worlds(values):
    return [{"outcomes": [{"wdl": v, "winner": None if v is None or v == 0 else (0 if v == 1 else 1),
                           "reason": "repetition_cutoff" if v is None else "completed"} for v in values]} for _ in range(128)]


def all_candidates(indices):
    return {arm: {"indices": list(indices)} for arm in p.ARMS}


def test_fresh_world_namespaces_are_disjoint():
    from tools.action_card_relations_128 import engine as previous
    seen, earlier = set(), set()
    for stage, count in (("main", 360), ("probe", 18), ("test", 18)):
        for index in range(count):
            for block in ("selection", "assessment"):
                seeds = p.world_seeds(stage, index, block)
                assert len(seeds) == len(set(seeds)) == 128 and seen.isdisjoint(seeds)
                seen.update(seeds)
                earlier.update(previous.world_seeds(stage, index, block))
    assert seen.isdisjoint(earlier) and min(seen) == 2300000000


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6, 7, 8, 14])
def test_teacher_is_exact_original_breadth_recipe(size):
    args = synthetic_union(size)
    actual = p.assemble_union(*args)
    anchor = args[1]["anchor"]["candidates"]["keys"]
    remaining = [i for i,k in enumerate(args[2]) if k not in anchor]
    ranked = remaining[:min(2, min(8,size)-len(anchor))]
    random = sorted([i for i in remaining if i not in ranked],key=lambda i:(
        p.old.digest([p.CONFIG["teacher_random_tag"], "test", 0, args[2][i]]),i))[:min(8,size)-len(anchor)-len(ranked)]
    expected = anchor+[args[2][i] for i in ranked+random]
    assert actual["candidates"]["teacher"]["keys"] == expected
    assert actual["candidates"]["teacher"]["actions"] == [args[3][key] for key in expected]
    assert actual["candidates"]["teacher"]["keys"][:4] == actual["candidates"]["anchor"]["keys"]
    assert len(actual["union"]["keys"]) <= 10
    for category,indices in (("ranked",ranked),("random",random)):
        assert [actual["union"]["keys"][i] for i in actual["teacher_added_indices"][category]] == [args[2][i] for i in indices]


def test_disjoint_learner_fourths_produce_ten_distinct_union_entries():
    args = synthetic_union(20)
    teacher = p.assemble_union(*args)["candidates"]["teacher"]["keys"]
    absent = [key for key in args[2] if key not in teacher]
    actual = p.assemble_union(*synthetic_union(20, ("k3", absent[0], absent[1])))
    assert len(actual["union"]["keys"]) == len(set(actual["union"]["keys"])) == 10
    assert actual["proposed_indices"]["precision"] == 4 and actual["proposed_indices"]["control"] == 5
    assert actual["proposed_indices"]["teacher"] == actual["proposed_indices"]["anchor"] == 3
    damaged = synthetic_union(8)
    damaged[1]["precision"]["candidates"]["keys"][0] = "k7"
    with pytest.raises(ValueError, match="common candidates"):
        p.assemble_union(*damaged)


def test_registry_is_transactional_and_rejects_production_fixtures(models, tmp_path):
    before = dict(p._MODEL_ARTIFACTS)
    for change in ("fit", "family"):
        registry = fixture_registry()
        if change == "fit":
            registry["models"]["precision_161001"]["source_fit"] = 161003
        else:
            del registry["models"]["control_161003"]
        with pytest.raises(ValueError):
            p.initialize_models(tmp_path, registry)
        assert p._MODEL_ARTIFACTS == before
    with pytest.raises(ValueError, match="production"):
        p.initialize_models(p.OUTPUT, fixture_registry())


@pytest.mark.parametrize("fit", [161001, 161002, 161003])
def test_proposals_preserve_hidden_information_and_ancestor_models(models, fit):
    from tools.proposal_breadth_learning_128 import engine as previous
    before = dict(previous._MODELS), dict(previous._MODEL_ARTIFACTS)
    state = private_state()
    snapshot = p.old.digest(p.old.serialize_state_snapshot(state))
    a = p.proposal_union(state, "planner-transfer-privacy", fit, ("test", 0))
    b = p.proposal_union(input_audit.hidden_alternative(state), "planner-transfer-privacy", fit, ("test", 0))
    for arm in p.PROPOSERS:
        x = {k: v for k, v in a["proposals"][arm].items() if k != "timing"}
        y = {k: v for k, v in b["proposals"][arm].items() if k != "timing"}
        input_audit.compare(x, y)
        assert x["model_id"] == f"{arm}_{fit}"
        assert x["model_sha256"] == fixture_registry()["models"][x["model_id"]]["sha256"]
    assert a["union"] == b["union"]
    assert snapshot == p.old.digest(p.old.serialize_state_snapshot(state))
    assert (previous._MODELS, previous._MODEL_ARTIFACTS) == before


def test_selection_preserves_arm_order_with_ties():
    candidates = all_candidates([0, 1, 2])
    candidates["anchor"]["indices"] = [2, 0, 1]
    selection = p.choose(worlds([1., 0., 1.]), candidates, 3)
    assert selection["means"] == [1., 0., 1.]
    assert selection["choices"] == {**dict.fromkeys(p.ARMS,0),"anchor":2}


def test_interval_certificate_does_not_impute_a_result():
    data = worlds([1., -1.])
    data[0]["outcomes"][0] = worlds([None])[0]["outcomes"][0]
    selection = p.choose(data, all_candidates([0, 1]), 2)
    assert selection["means"] == [None, -1.] and selection["choices"] == {a: 0 for a in p.ARMS}
    assert selection["means_bounds"][0] == [126/128, 1.]
    assert selection["unknown_counts"] == [1, 0]


def test_ambiguous_selection_requires_evidence_and_coupled_certificate_keeps_original_tie():
    data = worlds([0., 0.])
    data[0]["outcomes"] = worlds([None, None])[0]["outcomes"]
    with pytest.raises(p.AmbiguousDecision) as failure:
        p.choose(data, all_candidates([0, 1]), 2)
    assert all(v is None for v in failure.value.selection["choices"].values())
    context = {"fixture": "same future"}
    proofs = [{"world": 0, "candidate": j, "context": context, "context_sha256": p.old.digest(context)} for j in range(2)]
    selection = p.choose(data, all_candidates([0, 1]), 2, proofs)
    assert selection["choices"] == {a: 0 for a in p.ARMS}
    assert selection["means"] == [None, None] and len(selection["certificate"]["unknown_groups"]) == 1
    proofs[1]["context"] = {"fixture": "different future"}
    proofs[1]["context_sha256"] = p.old.digest(proofs[1]["context"])
    with pytest.raises(p.AmbiguousDecision):
        p.choose(data, all_candidates([0, 1]), 2, proofs)


def saved_coupled_fixture():
    """Reuse saved proof traces, never start another production continuation."""
    saved = p.old.read_json(p.ROOT/"outputs/equal_budget_breadth_arena_128/recovery_02/main/roots/game_0310/turn_058.json")
    assert saved["root_identity"] == ["main", 310, 58] and len(saved["worlds"]) == 64
    proofs = []
    for offset in (0, 64):
        for original in saved["equivalence_proofs"]:
            item = copy.deepcopy(original)
            item["world"] += offset
            proofs.append(item)
    assert {item["context_sha256"] for item in proofs} == {
        "3c456ec3296ebfc51e8e30af9755a7cee43cea83a8ad949eac434f82dafc7683"}
    return {"snapshot": saved["snapshot"], "union": saved["candidates"],
            "candidates": all_candidates(range(8)), "equivalence_proofs": proofs,
            "selection": {"candidate_indices": list(range(8)),
                          "world_seeds": saved["schedule"]["world_seeds"][:64]*2,
                          "worlds": copy.deepcopy(saved["worlds"])*2}}


def test_saved_repetition_proof_adapter_reconstructs_real_context_and_rejects_tampering(monkeypatch):
    record = saved_coupled_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("This check must not generate a new rollout")
    monkeypatch.setattr(p.old, "trajectory", forbidden)
    selected = p.selection_for_record(record)
    assert selected["choices"] == {arm: 0 for arm in p.ARMS}
    assert selected["means"][:2] == [None, None]
    assert selected["unknown_counts"][:2] == [2, 2]
    assert len(selected["certificate"]["unknown_groups"]) == 1
    record["equivalence_proofs"][0]["replay"]["final_state"]["players"][0]["score"] += 1
    with pytest.raises(ValueError, match="final state"):
        p.selection_for_record(record)


def test_ambiguous_selection_generates_and_accounts_for_only_required_saved_proofs(tmp_path, monkeypatch):
    record = saved_coupled_fixture()
    expected = record.pop("equivalence_proofs")
    lookup = {(item["world"], item["candidate"]): item for item in expected}
    calls = []
    def saved_replay(adapted, wi, ci):
        calls.append((wi, ci))
        assert adapted["candidates"] == record["union"]
        return copy.deepcopy(lookup[(wi, ci)])
    monkeypatch.setattr(p.repetition_proof, "build", saved_replay)
    with pytest.raises(p.AmbiguousDecision):
        p.selection_for_record(record, tmp_path)
    selection = p.selection_for_record(record, tmp_path, build_proofs=True)
    assert calls == [(42, 0), (42, 1), (106, 0), (106, 1)]
    assert record["proof_timing"]["replayed_continuations"] == 4
    assert selection["choices"] == {arm: 0 for arm in p.ARMS}
    assert p.selection_for_record(record, tmp_path) == selection
    assert len(calls) == 4


def test_assessment_keeps_unknown_bounds_and_exact_same_action_cancellation():
    choices = {"anchor": 0, "precision": 0, "control": 2, "teacher": 2}
    data = worlds([0., 1.])
    data[0]["outcomes"][0] = worlds([None])[0]["outcomes"][0]
    result = p.summarize(data, [0, 2], choices, {a: None for a in p.ARMS}, 3)
    assert result["arm_scores"]["anchor"] is None and result["arm_scores"]["precision"] is None
    assert result["arm_score_bounds"]["anchor"] == [127/256, 129/256]
    assert result["delta_bounds"]["precision_minus_anchor"] == [0., 0.]
    assert result["delta_bounds"]["precision_minus_control"] == [-129/256, -127/256]
    with pytest.raises(ValueError, match="locked choices"):
        p.summarize(worlds([0., 0., 1.]), [0, 1, 2], choices, {a: None for a in p.ARMS}, 3)


@pytest.mark.parametrize("reason,value", [("turn_limit", None), ("repetition_cutoff", 0.), ("completed", None), ("completed", True)])
def test_invalid_or_imputed_outcomes_rejected(reason, value):
    with pytest.raises(ValueError):
        p.outcome_value({"reason": reason, "winner": None, "wdl": value})


def test_real_isolated_root_is_sealed_before_sparse_assessment(models, tmp_path, monkeypatch):
    spec = fixture_spec()
    original, calls = p.old.trajectory, 0
    support = len(p.proposal_union(private_state(), spec["search_stream"], spec["fit"], ("test", 0))["union"]["keys"])
    def observed(world, action, **kwargs):
        nonlocal calls
        if calls == 128*support:
            seal = p.old.read_json(p.selection_path("test", 0, tmp_path))
            assert seal["selection"]["candidate_indices"] == list(range(support))
            assert "assessment" not in seal
        calls += 1
        return original(world, action, **kwargs)
    monkeypatch.setattr(p.old, "trajectory", observed)
    item = p.collect("test", spec, "fixture", tmp_path)
    record = p.c.verified(item)
    count = len(set(record["selection_lock"]["choices"].values()))
    assert calls == item["continuations"] == 128*(support+count)
    assert item["proof_continuations"] == 0 and count <= 4
    assert record["selection_lock"]["created_at_ns"] <= p.selection_path("test", 0, tmp_path).stat().st_mtime_ns <= record["assessment_started_at_ns"]
    row = p.statistics_row(record)
    assert row["pair"] == row["game_index"] and row["delta_bounds"] == record["summary"]["delta_bounds"]
    changed = copy.deepcopy(record)
    changed["summary"]["delta_bounds"]["precision_minus_anchor"][0] += .125
    with pytest.raises(ValueError, match="summary"):
        p.statistics_row(changed)
    before = p.old.file_hash(p.root_path("test", 0, tmp_path))
    with pytest.raises(ValueError, match="One-shot"):
        p.collect("test", spec, "fixture", tmp_path)
    assert p.old.file_hash(p.root_path("test", 0, tmp_path)) == before


def test_failed_assessment_preserves_selection_and_partial_root(models, tmp_path, monkeypatch):
    original = p.run_block
    def fail(state, actions, seeds, block, base):
        if seeds[0] == p.world_seeds("test", 0, "assessment")[0]:
            assert p.selection_path("test", 0, base).exists()
            raise RuntimeError("isolated assessment failure")
        return original(state, actions, seeds, block, base)
    monkeypatch.setattr(p, "run_block", fail)
    with pytest.raises(RuntimeError, match="isolated assessment failure"):
        p.collect("test", fixture_spec(), "fixture", tmp_path)
    assert p.old.read_json(p.selection_path("test", 0, tmp_path))["selection_lock"]["choices"]
    assert "failure" in p.old.read_json(p.root_path("test", 0, tmp_path))


def test_source_and_fixture_production_guards(models, tmp_path):
    with pytest.raises(ValueError, match="production"):
        p.collect("test", fixture_spec(), "fixture", p.OUTPUT)
    with pytest.raises(ValueError, match="Synthetic"):
        p.collect("main", fixture_spec(), "fixture", tmp_path)
    assert "failure" in p.old.read_json(p.root_path("main", 0, tmp_path))


def test_unsupported_checkpoint_kind_rejected_without_registry_mutation(models,tmp_path):
    import torch
    registry = fixture_registry()
    item = registry["models"]["precision_161001"]
    payload = torch.load(p.c.verify_hash(item), map_location="cpu", weights_only=True)
    payload["model_kind"] = "unexpected_residual_model"
    path = tmp_path/"bad_model.pt"
    torch.save(payload,path)
    registry["models"]["precision_161001"] = p.c.artifact(path,source_fit=161001)
    before = dict(p._MODEL_ARTIFACTS)
    with pytest.raises(ValueError,match="Unsupported strong-policy"):
        p.initialize_models(tmp_path,registry)
    assert p._MODEL_ARTIFACTS == before

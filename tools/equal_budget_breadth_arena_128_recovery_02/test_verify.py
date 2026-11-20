"""Independent committed-trace, guard-state and shared-unknown verification."""
import copy
from fractions import Fraction
from itertools import product

import pytest

from tools.equal_budget_breadth_arena_128_recovery_02 import arena as p, common as c, proof as collector, verify as a
from tools.equal_budget_breadth_arena_128_recovery_02.test_proof import synthetic
from tools.equal_budget_breadth_arena_128_recovery_01 import verify as previous
from tools.equal_budget_breadth_arena_128.test_arena import fixture_registry


@pytest.fixture(scope="module")
def saved_case():
    path = c.PREVIOUS/"main/roots/game_0310/turn_058.json"
    assert p.old.file_hash(path) == "a612be2ff084af6f89f83389c49b2f78058406d8e1e5faaba988f2fdfed65ae7"
    root = p.old.read_json(path)
    diagnostics = [p.old.read_json(c.SOURCE/f"recovery_diagnosis_02/candidate_{j}_reproduction.json") for j in (0,1)]
    proofs = []
    for j, diagnostic in enumerate(diagnostics):
        replay = diagnostic["reproduced"]
        context = collector.context_from_replay(root,42,j,replay)
        proofs.append({"world":42,"candidate":j,"replay":replay,
                       "context":context,"context_sha256":p.old.digest(context)})
    rebuilt = copy.deepcopy(root)
    rebuilt.pop("failure"); rebuilt.pop("hold")
    rebuilt.update(p.select(root["worlds"],root["counts"],proofs))
    rebuilt["equivalence_proofs"] = proofs
    return rebuilt,diagnostics


def test_standalone_guard_reconstructs_committed_traces_without_collector_or_inference(saved_case,monkeypatch):
    root,_ = saved_case
    def forbidden(*args,**kwargs): raise AssertionError("Verifier called collection/inference")
    monkeypatch.setattr(collector,"context_from_replay",forbidden)
    monkeypatch.setattr(collector,"canonical_snapshot",forbidden)
    monkeypatch.setattr(p,"select",forbidden)
    monkeypatch.setattr(p.old,"trajectory",forbidden)
    proofs = [a.independent_saved_proof(root,proof) for proof in root["equivalence_proofs"]]
    assert proofs[0]["context"] == proofs[1]["context"]
    result = a.certified_selection(root,root["counts"])
    assert result["choice"] == 0 and result["means"][:2] == [None,None]
    assert result["certificate"]["pairwise_lower_numerators"] == [0,0,256,0,128,128,0,128]


def test_independent_neural_replay_of_two_exported_branches_reconstructs_exact_proofs(saved_case):
    # Two bounded, previously diagnosed branches; no new root/world allocation.
    root,_ = saved_case
    p.old.initialize_worker()
    source = p.old.deserialize_state_snapshot(root["snapshot"])
    world = a.a154.independent_world(source,root["worlds"][42]["seed"])
    proofs = [a.independent_equivalence_proof(world,p.old.decode_action(root["candidates"]["actions"][j]),
                                           source.current_player,42,j) for j in (0,1)]
    assert p.old.digest(proofs) == p.old.digest(root["equivalence_proofs"])
    assert a.certified_selection(root,root["counts"],proofs)["choice"] == 0


def test_one_exported_coupled_root_with_authenticated_r1_prefix_and_sparse_audit(saved_case,tmp_path):
    original,_ = saved_case
    root = copy.deepcopy(original)
    source = c.PREVIOUS/"main/roots/game_0310/turn_058.json"
    ref = c.artifact(source,index=310,turn=58)
    binding = "isolated-coupled-root-audit"
    inventory = {"original_binding":c.ORIGINAL_BINDING,"previous_binding":c.PREVIOUS_BINDING,
        "sources":{ref["path"]:ref},"origins":{ref["path"]:{"generation":"recovery_01","stage":"main",
          "kind":"root","index":310,"turn":58,"launch":c.artifact(c.PREVIOUS/"main_launch.json")}},
        "replaceable_roots":{"main":{"310:58":ref}},"partial_games":{},
        "roots":{"main":{"310:58":ref}},"games":{}}
    p.old.save_new(tmp_path/"inventory.json",inventory)
    p.old.save_new(tmp_path/"freeze.json",{"binding":binding})
    p.old.save_new(tmp_path/"main_launch.json",{"manifest":c.ORIGINAL_BINDING,"recovery_binding":binding})
    root.update(recovery_binding=binding,recovery_origin=c.artifact(source),inherited_cells=512,new_cells=0,
                inherited_timing=root["timing"],timing={"wall":0.,"cpu":0.})
    path = p.root_path("main",310,58,tmp_path)
    p.old.save_new(path,root)
    p.initialize_models(tmp_path,fixture_registry())
    state = p.old.deserialize_state_snapshot(root["snapshot"])
    move = {"turn":58,"actor":0,"state_sha256":p.old.digest(root["snapshot"]),
            "root_sha256":p.old.file_hash(path),"action":root["candidates"]["actions"][0]}
    ordinary = a.ordinary_indices("main",310,58,64)
    selection = {"root_id":a.root_id(310,58),"ordinary_world_indices":ordinary,
                 "replay_world_indices":ordinary,"stress":False}
    receipt = a.audit_root("main",310,state,move,p.specs("main")[310],c.ORIGINAL_BINDING,tmp_path,selection)
    assert receipt["choice"] == 0 and receipt["coupled_certified_roots"] == 1
    assert receipt["equivalence_proof_cells"] == receipt["unresolved_continuations"] == 2
    assert receipt["coupled_unknown_groups"] == 1 and receipt["terminal_continuations"] == 510
    expected = a.independent_replay_cells(root["worlds"],ordinary)
    assert receipt["replayed_branches"] == len(expected["all"])
    assert receipt["extra_censored_replay_cells"] == len(expected["extra"])
    assert receipt["replayed_worlds"] == len({wi for wi,j in expected["all"]})
    assert receipt["inherited_timing"] == original["timing"]
    assert p.old.file_hash(source) == ref["sha256"]


@pytest.mark.parametrize("damage", ["forced_action","trace_state","trace_actor","trace_turn","final_turn",
    "trace_hash","reason","world","context_guard","context_idle","context_actor","context_policy",
    "context_final","missing","duplicate","reverse","numerator","choice","mean","invented_draw"])
def test_proof_and_certificate_corruptions_rejected_even_after_rehash(saved_case,damage):
    original,_ = saved_case
    root = copy.deepcopy(original)
    proof = root["equivalence_proofs"][0]
    if damage == "forced_action": proof["replay"]["trace"][0]["action"] = root["candidates"]["actions"][1]
    elif damage == "trace_state": proof["replay"]["trace"][0]["state_hash"] = "0"*64
    elif damage == "trace_actor": proof["replay"]["trace"][0]["actor"] = 1
    elif damage == "trace_turn": proof["replay"]["trace"][0]["turn"] += 2
    elif damage == "final_turn": proof["replay"]["final_state"]["turn_index"] += 2
    elif damage == "trace_hash": proof["replay"]["trace_sha256"] = "0"*64
    elif damage == "reason": proof["replay"]["reason"] = "max_turns"
    elif damage == "world": root["worlds"][42]["state_sha256"] = "0"*64
    elif damage == "context_guard": proof["context"]["live_repetition_counts"][-1]["count"] += 1
    elif damage == "context_idle": proof["context"]["no_progress_count"] += 1
    elif damage == "context_actor": proof["context"]["root_actor"] = 1
    elif damage == "context_policy": proof["context"]["policy_sha256"] = "0"*64
    elif damage == "context_final": proof["context"]["canonical_final_state_sha256"] = "0"*64
    elif damage == "missing": root["equivalence_proofs"].pop()
    elif damage == "duplicate": root["equivalence_proofs"].append(copy.deepcopy(proof))
    elif damage == "reverse": root["equivalence_proofs"].reverse()
    elif damage == "numerator": root["certificate"]["pairwise_lower_numerators"][1] = 1
    elif damage == "choice": root["choice"] = 1
    elif damage == "mean": root["means"][0] = -59/64
    else: root["worlds"][42]["outcomes"][0]["wdl"] = 0.
    if damage.startswith("context_"):
        proof["context_sha256"] = p.old.digest(proof["context"])
    with pytest.raises((AssertionError,ValueError,KeyError)):
        a.certified_selection(root,root["counts"])


def test_live_guard_context_includes_final_visit_and_excludes_irreversible_past(saved_case):
    root,diagnostics = saved_case
    history = [item["snapshot"] for item in diagnostics[0]["positions"]]
    expected = root["equivalence_proofs"][0]["context"]
    assert a._context_from_history(history,0,8,0) == expected
    assert a._context_from_history(history+[history[-1]],0,8,0) != expected
    assert a._context_from_history([history[0]]*7+history,0,8,0) == expected
    assert max(row["count"] for row in expected["live_repetition_counts"]) == 4
    assert a._context_from_history(history,1,8,0) != expected
    assert a._context_from_history(history,0,9,0) != expected


def test_only_purchased_order_is_quotiented_and_other_orders_remain_live(saved_case):
    _,diagnostics = saved_case
    left,right = [d["reproduced"]["final_state"] for d in diagnostics]
    assert left != right and a._canonical_purchased_order(left) == a._canonical_purchased_order(right)
    for field in ("turn_index","start_player","current_player","pending_round_end"):
        changed = copy.deepcopy(right)
        changed[field] = changed[field]+1
        assert a._canonical_purchased_order(left) != a._canonical_purchased_order(changed)
    changed = copy.deepcopy(right)
    changed["hidden_tier_decks"]["2"].reverse()
    assert a._canonical_purchased_order(left) != a._canonical_purchased_order(changed)
    for before,after in zip(left["players"],a._canonical_purchased_order(left)["players"]):
        assert {k:v for k,v in before.items() if k != "purchased_card_ids"} == {
            k:v for k,v in after.items() if k != "purchased_card_ids"}


def test_grouped_fraction_bounds_match_exhaustive_assignments_with_unequal_denominators():
    checked = 0
    for values in product((-1,0,1,None),repeat=5):
        columns = [list(values[:2]),list(values[2:])]
        cells = [(wi,j) for j,col in enumerate(columns) for wi,value in enumerate(col) if value is None]
        for by_world in (True,False):
            labels = {cell:cell[0] if by_world else 0 for cell in cells}
            worlds,proofs = synthetic(columns,labels)
            if a.independent_selection(worlds,[2,3])["choice"] is not None:
                continue
            result = a.independent_coupled_selection(worlds,[2,3],proofs)
            labels_order = list(dict.fromkeys(labels.values()))
            winners = set()
            for completion in product((-1,0,1),repeat=len(labels_order)):
                filled = dict(zip(labels_order,completion))
                scores = [Fraction(sum(value if value is not None else filled[labels[(wi,j)]]
                                      for wi,value in enumerate(column)),len(column))
                          for j,column in enumerate(columns)]
                winners.add(0 if scores[0] >= scores[1] else 1)
            assert result["choice"] == (next(iter(winners)) if len(winners)==1 else None)
            try: collected = p.select(worlds,[2,3],proofs)
            except p.AmbiguousDecision as error: collected = error.selection
            assert result == collected
            checked += 1
    assert checked > 500


def test_equal_unknown_does_not_cancel_when_candidate_denominators_differ():
    worlds,proofs = synthetic([[None,0],[None,0,0]],{(0,0):0,(0,1):0})
    result = a.independent_coupled_selection(worlds,[2,3],proofs)
    assert result["choice"] is None and result["certificate"]["pairwise_lower_numerators"] is None


def test_r1_certified_natural_and_uncertain_roots_keep_their_schema():
    for columns in ([[1,0],[-1,0]], [[1,1],[None,-1]]):
        worlds,_ = synthetic(columns,{(0,1):0})
        prior = previous.independent_selection(worlds,[2,2])
        root = {"worlds":worlds,**prior}
        assert a.certified_selection(root,[2,2]) == prior
    for size in (64,74,86,103,128):
        assert a.ordinary_indices("main",310,58,size) == previous.ordinary_indices("main",310,58,size)


def test_unknown_replay_plan_remains_complete_without_changing_sampling(saved_case):
    root,_ = saved_case
    ordinary = a.ordinary_indices("main",310,58,64)
    plan = a.independent_replay_cells(root["worlds"],ordinary)
    assert plan == previous.independent_replay_cells(root["worlds"],ordinary)
    assert plan["unresolved"] == {(42,0),(42,1)}
    assert plan["unresolved"] <= plan["all"]


def test_completed_r1_root_ancestor_stays_auditable_under_r2_inventory(tmp_path,monkeypatch):
    source = {key:{"fixture":key} for key in ("snapshot","privacy","root_identity","planner",
              "schedule","candidates","proposal","allocation","scores")}
    source.update(worlds=[{"candidate_indices":[0],"outcomes":[{"wdl":None}]}],
                  timing={"wall":2.,"cpu":1.},failure="original failure")
    path = tmp_path/"original_failed.json";p.old.save_new(path,source)
    root = {k:copy.deepcopy(v) for k,v in source.items() if k != "failure"}
    root.update(recovery_binding=c.PREVIOUS_BINDING,recovery_origin=c.artifact(path),
                inherited_timing=source["timing"],inherited_cells=1,new_cells=0)
    monkeypatch.setattr(c,"load_inventory",lambda base:{"replaceable_roots":{}})
    monkeypatch.setattr(c,"recovery_binding",lambda base:"new-r2-binding")
    monkeypatch.setattr(c,"lineage_ancestor",lambda record,base:path)
    monkeypatch.setattr(c,"validate_origin",lambda *args:True)
    a._check_root_prefix(root,"main",13,17,c.ORIGINAL_BINDING,tmp_path)
    assert a._timing_fields(root)["inherited_timing"] == source["timing"]
    root["worlds"][0]["outcomes"][0]["wdl"] = 0
    with pytest.raises(AssertionError):a._check_root_prefix(root,"main",13,17,c.ORIGINAL_BINDING,tmp_path)


def test_completed_r1_game_uses_its_actual_ancestor_not_current_partials(tmp_path,monkeypatch):
    inherited = {"spec":{"index":13},"initial_state_sha256":"a"*64,
                 "trace":[{"turn":0}],"timing":{"wall":2.,"cpu":1.},"failure":"original failure"}
    path = tmp_path/"original_game.json";p.old.save_new(path,inherited)
    game = {k:copy.deepcopy(v) for k,v in inherited.items() if k != "failure"}
    game.update(recovery_binding=c.PREVIOUS_BINDING,recovery_origin=c.artifact(path),
                inherited_turns=1,inherited_timing=inherited["timing"])
    monkeypatch.setattr(c,"load_inventory",lambda base:{"partial_games":{}})
    monkeypatch.setattr(c,"recovery_binding",lambda base:"new-r2-binding")
    monkeypatch.setattr(c,"lineage_ancestor",lambda record,base:path)
    monkeypatch.setattr(c,"validate_origin",lambda *args:True)
    assert a._check_game_prefix(game,"main",13,c.ORIGINAL_BINDING,tmp_path) == inherited
    game["trace"].clear()
    with pytest.raises(AssertionError):a._check_game_prefix(game,"main",13,c.ORIGINAL_BINDING,tmp_path)

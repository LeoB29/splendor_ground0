"""Actual saved-loop context reconstruction and exact shared-unknown algebra."""
import copy
from fractions import Fraction
from itertools import product

import pytest

from tools.equal_budget_breadth_arena_128_recovery_02 import proof as p, arena

DIAG = p.c.SOURCE/"recovery_diagnosis_02"
SOURCE = p.c.PREVIOUS/"main/roots/game_0310/turn_058.json"
SOURCE_SHA = "a612be2ff084af6f89f83389c49b2f78058406d8e1e5faaba988f2fdfed65ae7"


@pytest.fixture(scope="module")
def actual():
    assert p.old.file_hash(SOURCE) == SOURCE_SHA
    root = p.old.read_json(SOURCE)
    diagnostics = [p.old.read_json(DIAG/f"candidate_{ci}_reproduction.json") for ci in (0,1)]
    return root,diagnostics


def saved_proofs(root,diagnostics):
    values = []
    for ci,diagnostic in enumerate(diagnostics):
        replay = diagnostic["reproduced"]
        context = p.context_from_replay(root,42,ci,replay)
        values.append({"world":42,"candidate":ci,"replay":replay,"context":context,"context_sha256":p.old.digest(context)})
    return values


def test_actual_contexts_match_and_cancel_only_the_shared_unknown(actual):
    root,diagnostics = actual
    proofs = saved_proofs(root,diagnostics)
    assert p.validate(root,proofs)
    assert proofs[0]["context"] == proofs[1]["context"]
    context = proofs[0]["context"]
    assert context["forced_pass_count"] == 0 and context["no_progress_count"] == 8
    assert context["root_actor"] == 0 and context["policy_sha256"] == p.old.CHECKPOINT_SHA
    assert [item["signature_sha256"] for item in context["live_repetition_counts"]] == sorted(item["signature_sha256"] for item in context["live_repetition_counts"])
    assert sorted(item["count"] for item in context["live_repetition_counts"])[-2:] == [3,4]
    with pytest.raises(arena.AmbiguousDecision):arena.select(root["worlds"],root["counts"])
    result = arena.select(root["worlds"],root["counts"],proofs)
    assert result["choice"] == 0
    assert result["means"][:2] == [None,None]
    assert result["certificate"]["pairwise_lower_numerators"] == [0,0,256,0,128,128,0,128]
    assert result["certificate"]["groups"] == [{"context_sha256":proofs[0]["context_sha256"],"cells":[{"world":42,"candidate":0},{"world":42,"candidate":1}]}]


def test_actual_original_trajectory_replay_matches_saved_proofs(actual):
    root,diagnostics = actual
    p.old.initialize_worker()
    expected = saved_proofs(root,diagnostics)
    built = [p.build(root,42,ci) for ci in (0,1)]
    assert p.old.digest(built) == p.old.digest(expected)


@pytest.mark.parametrize("damage",["move","move_state","move_actor","move_turn","final_state","trace_hash","reason","context","context_hash","missing","reverse"])
def test_saved_equivalence_proof_tampering_is_rejected(actual,damage):
    root,diagnostics = actual
    proofs = copy.deepcopy(saved_proofs(root,diagnostics))
    item = proofs[0]
    if damage == "move":item["replay"]["trace"][0]["action"] = copy.deepcopy(root["candidates"]["actions"][2])
    elif damage == "move_state":item["replay"]["trace"][0]["state_hash"] = "0"*64
    elif damage == "move_actor":item["replay"]["trace"][0]["actor"] = 1
    elif damage == "move_turn":item["replay"]["trace"][0]["turn"] += 2
    elif damage == "final_state":item["replay"]["final_state"]["turn_index"] += 2
    elif damage == "trace_hash":item["replay"]["trace_sha256"] = "0"*64
    elif damage == "reason":item["replay"]["reason"] = "max_turns"
    elif damage == "context":item["context"]["no_progress_count"] += 1;item["context_sha256"] = p.old.digest(item["context"])
    elif damage == "context_hash":item["context_sha256"] = "0"*64
    elif damage == "missing":proofs.pop()
    else:proofs.reverse()
    with pytest.raises(ValueError):p.validate(root,proofs)


def test_canonicalization_changes_only_purchased_card_order(actual):
    _,diagnostics = actual
    a,b = [item["reproduced"]["final_state"] for item in diagnostics]
    assert a != b and p.canonical_snapshot(a) == p.canonical_snapshot(b)
    normalized = p.canonical_snapshot(a)
    for path in ("turn_index","start_player","pending_round_end","winner"):
        assert normalized[path] == a[path]
    for key in a:
        if key != "players":assert normalized[key] == a[key]
    for before,after in zip(a["players"],normalized["players"]):
        assert {k:v for k,v in before.items() if k != "purchased_card_ids"} == {k:v for k,v in after.items() if k != "purchased_card_ids"}
    changed = copy.deepcopy(a);changed["turn_index"] += 2
    assert p.canonical_snapshot(changed) != normalized
    changed = copy.deepcopy(a);changed["hidden_tier_decks"]["2"].reverse()
    assert p.canonical_snapshot(changed) != normalized


def synthetic(columns,labels):
    worlds,proofs = [],[]
    for wi in range(max(map(len,columns))):
        active = [ci for ci,column in enumerate(columns) if wi < len(column)]
        row = {"candidate_indices":active,"outcomes":[]}
        for ci in active:
            value = columns[ci][wi]
            row["outcomes"].append({"reason":"repetition_cutoff" if value is None else "completed","winner":None,"wdl":value})
            if value is None:
                context = {"synthetic_group":labels[(wi,ci)]}
                proofs.append({"world":wi,"candidate":ci,"context":context,"context_sha256":p.old.digest(context)})
        worlds.append(row)
    return worlds,proofs


def test_coupled_certificate_matches_exhaustive_shared_assignments_with_ragged_counts():
    for entries in product((-1,0,1,None),repeat=5):
        columns = [list(entries[:2]),list(entries[2:])]
        unknown = [(wi,ci) for ci,col in enumerate(columns) for wi,v in enumerate(col) if v is None]
        for grouping in ("all_shared","by_world","independent"):
            labels = {cell:(0 if grouping == "all_shared" else cell[0] if grouping == "by_world" else cell) for cell in unknown}
            groups = list(dict.fromkeys(labels.values()))
            winners = set()
            for assignment in product((-1,0,1),repeat=len(groups)):
                assigned = dict(zip(groups,assignment))
                sums = [sum(v if v is not None else assigned[labels[(wi,ci)]] for wi,v in enumerate(col)) for ci,col in enumerate(columns)]
                means = [Fraction(s,len(col)) for s,col in zip(sums,columns)]
                winners.add(0 if means[0] >= means[1] else 1)
            recorded,proofs = synthetic(columns,labels)
            try:result = arena.select(recorded,list(map(len,columns)),proofs)
            except arena.AmbiguousDecision as exc:result = exc.selection
            assert result["choice"] == (next(iter(winners)) if len(winners) == 1 else None)


@pytest.mark.parametrize("change",["new_context","reordered","missing","duplicate"])
def test_coupling_requires_exact_unknown_inventory(actual,change):
    root,diagnostics = actual
    proofs = copy.deepcopy(saved_proofs(root,diagnostics))
    if change == "new_context":
        proofs[1]["context"]["no_progress_count"] += 1
        proofs[1]["context_sha256"] = p.old.digest(proofs[1]["context"])
        with pytest.raises(arena.AmbiguousDecision):arena.select(root["worlds"],root["counts"],proofs)
        return
    if change == "reordered":proofs.reverse()
    elif change == "missing":proofs.pop()
    else:proofs.append(copy.deepcopy(proofs[-1]))
    with pytest.raises(ValueError):arena.select(root["worlds"],root["counts"],proofs)

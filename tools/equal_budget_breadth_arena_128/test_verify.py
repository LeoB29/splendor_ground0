"""Independent actual-game/world audits, proposal changes and sampled limits."""
import copy
from pathlib import Path

import pytest

from tools.equal_budget_breadth_arena_128 import arena as p, verify as a
from tools.equal_budget_breadth_arena_128.test_arena import (
    BINDING, models, tiny_games, private_state,
)


@pytest.fixture(scope="module")
def sealed_games(tiny_games):
    base, snapshot, games = tiny_games
    a.seal_sample("test", games, BINDING, base)
    a.validate_sample("test", BINDING, base)
    return base, snapshot, games


def test_all_fits_and_both_actors_actual_game_and_every_fixture_world(sealed_games):
    base, snapshot, games = sealed_games
    for item in games:
        receipt = a.audit_game("test", item["index"], item["sha256"], BINDING, base, snapshot, full_fixture=True)
        assert {r["planner"] for r in receipt["roots"]} == {"candidate", "reference"}
        assert receipt["full_inner_replay"]
        for root in receipt["roots"]:
            assert root["model_id"] == f"anchor_{root['assigned_fit']}"
            assert root["replayed_worlds"] == root["recorded_worlds"] == max(root["world_counts"])
            assert root["continuations"] == root["replayed_branches"] == sum(root["world_counts"]) == 128*min(root["canonical_support"],4)
            assert root["support"] == min(root["canonical_support"],8 if root["planner"] == "candidate" else 4)


def root_copy(tiny_games, target, index=0):
    base, snapshot, _ = tiny_games
    game = p.old.read_json(p.game_path("test", index, base))
    move = game["trace"][0]
    state = p.old.deserialize_state_snapshot(snapshot)
    p.old.save_new(target / "test_launch.json", {"manifest": BINDING})
    path = p.root_path("test", index, move["turn"], target)
    root = p.old.read_json(p.root_path("test", index, move["turn"], base))
    p.old.save_new(path, root)
    ordinary = a.ordinary_indices("test", index, move["turn"], root["allocation"]["world_count"])
    selection = {"root_id": a.root_id(index, move["turn"]), "ordinary_world_indices": ordinary,
                 "replay_world_indices": ordinary, "stress": False}
    return game["spec"], state, move, root, path, selection


@pytest.mark.parametrize("damage", ["model_id", "model_sha", "proposal_logits", "world", "sampled_trace", "means", "choice", "last_outcome", "root_identity", "planner", "addition", "prefix", "extra_world", "extra_cell", "allocation", "counts", "sums", "sample"])
def test_rehashed_semantic_corruption_is_rejected(tiny_games, tmp_path, damage):
    spec, state, move, root, path, selection = root_copy(tiny_games, tmp_path)
    if damage == "model_id":
        root["proposal"]["model_id"] = f"anchor_999999"
    elif damage == "model_sha":
        root["proposal"]["model_sha256"] = "0"*64
    elif damage == "proposal_logits":
        root["proposal"]["learned_logits"][0] += .5
    elif damage == "world":
        root["worlds"][-1]["state_sha256"] = "0"*64
    elif damage == "sampled_trace":
        root["worlds"][selection["replay_world_indices"][0]]["outcomes"][0]["trace_sha256"] = "0"*64
    elif damage == "means":
        root["means"][0] += .125
    elif damage == "choice":
        root["choice"] = (root["choice"]+1) % len(root["means"])
    elif damage == "last_outcome":
        root["worlds"][-1]["outcomes"].pop()
    elif damage == "root_identity":
        root["root_identity"][1] = 1
    elif damage == "planner":
        root["planner"] = "reference"
    elif damage == "addition":
        root["proposal"]["added_indices"]["ranked"].reverse()
    elif damage == "prefix":
        root["proposal"]["anchor_candidates"]["actions"].reverse()
    elif damage == "extra_world":
        root["worlds"].append(copy.deepcopy(root["worlds"][-1]))
    elif damage == "extra_cell":
        root["worlds"][-1]["candidate_indices"].append(root["worlds"][-1]["candidate_indices"][-1])
        root["worlds"][-1]["outcomes"].append(copy.deepcopy(root["worlds"][-1]["outcomes"][-1]))
    elif damage == "allocation":
        root["allocation"]["world_counts"][0] += 1
    elif damage == "counts":
        root["counts"][0] += 1
    elif damage == "sums":
        root["sums_wdl"][0] += 1
    else:
        selection["ordinary_world_indices"] = a.ordinary_indices("test",0,move["turn"],128)
        selection["replay_world_indices"] = selection["ordinary_world_indices"]
    p.old.atomic_write(path, root)
    move["root_sha256"] = p.old.file_hash(path)
    with pytest.raises((AssertionError, ValueError)):
        a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection)


def test_sampled_assurance_explicitly_leaves_unselected_trace_unreplayed(tiny_games, tmp_path):
    spec, state, move, root, path, selection = root_copy(tiny_games, tmp_path)
    wi = next(i for i in range(root["allocation"]["world_count"]) if i not in selection["replay_world_indices"])
    root["worlds"][wi]["outcomes"][0]["trace_sha256"] = "0"*64
    p.old.atomic_write(path, root)
    move["root_sha256"] = p.old.file_hash(path)
    result = a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection)
    assert result["replayed_branches"] == sum(wi < count for wi in selection["replay_world_indices"] for count in result["world_counts"])
    assert not result["full_inner_replay"] and result["audit_timing"]["stress_extra_replay_wall"] == 0


def test_sample_plan_outcome_independence_and_metadata_tamper(sealed_games):
    base, _, games = sealed_games
    inventory = a.metadata_inventory("test", games, BINDING, base)
    plan = a.sample_plan("test", inventory, BINDING, "a"*64)
    changed = copy.deepcopy(inventory)
    for row in changed:
        row.update(outcome=-1., score=1., choice=99, sha256="0"*64)
    alternative = a.sample_plan("test", changed, "changed", "b"*64)
    assert plan["selections"] == alternative["selections"]
    assert plan["stress_cells"] == alternative["stress_cells"]
    assert plan["coverage"]["stress_cells"] == 171
    assert plan["coverage"]["stress_roots"] <= 171
    bad = copy.deepcopy(plan)
    bad["selections"][0]["ordinary_world_indices"] = list(range(8))
    with pytest.raises((ValueError, AssertionError)):
        a.compare(bad, plan)
    with pytest.raises(AssertionError):
        a.metadata_inventory("test", games[:-1], BINDING, base)




def test_independent_paired_bootstrap_and_strict_score_units():
    rows = [{"index": s["index"], "pair": s["pair"], "seat": s["candidate_seat"],
             "fit": s["fit"], "score": (1., .5, 0., 1., 1., 0., 1.)[s["index"] % 7]}
            for s in p.specs("main")]
    a.compare(p.statistics(rows), a.independent_statistics(rows))
    for damage in ("fit","seat","pair","score","order"):
        changed=copy.deepcopy(rows)
        if damage == "fit":changed[0]["fit"]=999
        elif damage == "seat":changed[0]["seat"]=1
        elif damage == "pair":changed[0]["pair"]=1
        elif damage == "score":changed[0]["score"]=.51
        else:changed[0],changed[1]=changed[1],changed[0]
        with pytest.raises(AssertionError):a.independent_statistics(changed)
    assert not p.decision(.54,[.51,.57])["stage_pass"]
    assert p.decision(.54,[.51,.57])["classification"] == "positive_below_observed_target"
    assert not p.decision(.55,[.5,.6])["stage_pass"]
    assert p.decision(.44,[.39,.49])["classification"] == "demonstrated_harm"
    assert p.decision(.49,[.45,.54])["classification"] == "below_material_target"


def support_state(support):
    state=p.old.SplendorEnv(seed=2179810000+support).initial_state()
    cards=[card for tier in state.visible_tier_cards for card in state.visible_tier_cards[tier]+state.hidden_tier_decks[tier]]
    chosen=sorted([card for card in cards if card.points > 0],key=lambda card:card.card_id)[:support]
    state.visible_tier_cards={tier:[] for tier in state.visible_tier_cards}
    state.hidden_tier_decks={tier:[] for tier in state.hidden_tier_decks}
    state.deck_counts=dict.fromkeys(state.deck_counts,0);state.bank_tokens=dict.fromkeys(state.bank_tokens,0);state.nobles=[]
    for player in state.players:
        player.score=14;player.bonuses=dict.fromkeys(player.bonuses,10)
    reserved=chosen[:3]
    state.players[0].reserved_cards=reserved;state.players[0].reserved_card_sources=["deck"]*len(reserved)
    for card in chosen[3:]:state.visible_tier_cards[card.tier].append(card)
    state.players[1].purchased_cards=[card for card in cards if card not in chosen]
    return state


@pytest.mark.parametrize("support",[1,2,3,4,5,6,7,8,9])
def test_independent_expansion_all_short_and_ragged_supports(support):
    state=support_state(support)
    spec=a.independent_spec("test",0)
    source=f"anchor_{spec['fit']}"
    for breadth in (False,True):
        identity=["test",0,state.turn_index]
        expected=a.independent_proposal_for_root(state,spec["search_stream"],source,identity,breadth)
        actual=p.proposal_for_root(state,spec["search_stream"],source,identity,breadth)
        a.compare({k:v for k,v in actual.items() if k != "timing"},expected)
        assert expected["candidates"]["canonical_support"] == support
        assert len(expected["candidates"]["keys"]) == min(support,8 if breadth else 4)
        original=expected["anchor_candidates"]
        assert expected["candidates"]["actions"][:len(original["actions"])] == original["actions"]
        if support<=4:assert expected["added_indices"] == {"ranked":[],"random":[]}


def test_all171_metadata_stress_cells_and_probe_stays_sampled():
    inventory=[]
    def add(arm,fit,phase,actor,privacy,support):
        index=len(inventory);turn=phase*20
        offered=min(support,8 if arm == "candidate" else 4)
        q,r=divmod(128*min(support,4),offered)
        counts=[q+int(j<r) for j in range(offered)]
        inventory.append({"root_id":a.root_id(index,turn),"index":index,"turn":turn,
                          "fit":fit,"arm":arm,"actor":actor,"phase":phase,
                          "privacy_category":privacy,"support":min(support,8 if arm == "candidate" else 4),
                          "canonical_support":support,"worlds":max(counts),"world_counts":counts,
                          "continuations":sum(counts)})
    for arm in ("candidate","reference"):
        for fit in p.FITS:
            for phase in range(3):
                for actor in (0,1):
                    for privacy in range(4):add(arm,fit,phase,actor,privacy,8)
            for support in (1,2,3):add(arm,fit,0,0,0,support)
    for fit in p.FITS:
        for support in (5,6,7):add("candidate",fit,0,0,0,support)
    plan=a.sample_plan("main",inventory,BINDING,"a"*64)
    assert plan["coverage"]["stress_cells"] == 171
    assert not plan["coverage"]["missing_stress_cells"]
    for fit in p.FITS:
        for support in (5,6,7):
            cell=next(c for c in plan["stress_cells"] if c["cell"] == f"candidate_canonical_support|{fit}|{support}")
            assert cell["eligible_count"] == 1 and cell["selected_root"] in plan["full_stress_roots"]
    samecell=[]
    for index in range(12):
        row=copy.deepcopy(inventory[0]);row.update(index=index,root_id=a.root_id(index,0))
        samecell.append(row)
    probe=a.sample_plan("probe",samecell,BINDING,"b"*64)
    assert sum(r["stress"] for r in probe["selections"]) == 1
    assert {len(r["replay_world_indices"]) for r in probe["selections"]} == {4,64}


def test_independent_spec_namespaces_and_model_identity():
    seen=set()
    for stage in ("probe","main","test"):
        specs=a.independent_specs(stage)
        assert specs == p.specs(stage)
        seeds={s["environment_seed"] for s in specs}
        assert not seeds & seen
        seen.update(seeds)
        for spec in specs:
            assert a.independent_model_id(stage,spec["fit"],True) == a.independent_model_id(stage,spec["fit"],False) == f"anchor_{spec['fit']}"


@pytest.mark.parametrize("support", range(1, 13))
def test_independent_allocation_exact_counts_both_roles(support):
    identity = ["test", 0, 2]
    for breadth in (False, True):
        n = min(support, 8 if breadth else 4)
        candidates = {"canonical_support": support, "keys": [f"candidate{j}" for j in range(n)],
                      "actions": [{} for _ in range(n)]}
        expected = a.independent_allocation(candidates, identity, breadth)
        assert p.allocation_for_candidates(candidates, identity, breadth) == expected
        assert expected["continuations"] == 128*min(support, 4)
        assert expected["world_count"] == max(expected["world_counts"])
        if breadth and support >= 8:
            assert expected["world_counts"] == [64]*8
        if not breadth:
            assert expected["world_counts"] == [128]*n


def synthetic_worlds(counts, totals=None):
    worlds = []
    for wi in range(max(counts)):
        active = [j for j, count in enumerate(counts) if wi < count]
        outcomes = [{"wdl": 1 if totals is None or wi < totals[j] else 0, "reason": "completed"} for j in active]
        worlds.append({"candidate_indices": active, "outcomes": outcomes})
    return worlds


def test_exact_rational_selection_ties_and_unequal_counts():
    counts = [102, 103, 102, 103, 102]
    worlds = synthetic_worlds(counts)
    result = a.independent_selection(worlds, counts)
    assert result == p.select(worlds, counts)
    assert result["choice"] == 0 and result["means"] == [1.]*5
    assert max(range(5), key=lambda j: result["sums_wdl"][j]) == 1
    counts = [103, 102, 103, 102, 102]
    worlds = synthetic_worlds(counts, [51, 51, 0, 0, 0])
    result = a.independent_selection(worlds, counts)
    assert result == p.select(worlds, counts)
    assert result["choice"] == 1 and result["means"][1] == .5


@pytest.mark.parametrize("damage", ["extra_world", "extra_cell", "missing_cell", "index_order", "counts", "bool_value", "fraction_value"])
def test_sparse_selection_rejects_undeclared_or_missing_cells(damage):
    counts = [103, 102, 103, 102, 102]
    worlds = synthetic_worlds(counts)
    if damage == "extra_world": worlds.append(copy.deepcopy(worlds[-1]))
    elif damage == "extra_cell":
        worlds[-1]["candidate_indices"].append(1)
        worlds[-1]["outcomes"].append({"wdl": 1})
    elif damage == "missing_cell": worlds[-1]["outcomes"].pop()
    elif damage == "index_order": worlds[-1]["candidate_indices"].reverse()
    elif damage == "counts": counts[1] += 1
    elif damage == "bool_value": worlds[0]["outcomes"][0]["wdl"] = True
    else: worlds[0]["outcomes"][0]["wdl"] = .5
    with pytest.raises(AssertionError):
        a.independent_selection(worlds, counts)


@pytest.mark.parametrize("worlds,expected", [(1, 1), (2, 2), (16, 2), (32, 2), (64, 4), (74, 5), (86, 6), (103, 7), (128, 8)])
def test_ordinary_sampling_uses_actual_world_count(worlds, expected):
    sample = a.ordinary_indices("test", 0, 2, worlds)
    assert len(sample) == len(set(sample)) == expected
    assert sample == sorted(sample) and all(0 <= wi < worlds for wi in sample)


@pytest.mark.parametrize("support", [5, 6, 7])
def test_actual_ragged_root_all_allocated_cells_independently_replayed(tmp_path, support):
    state = support_state(support)
    spec = a.independent_spec("test", 0)
    p.old.save_new(tmp_path/"test_launch.json", {"manifest": BINDING})
    identity = ["test", 0, state.turn_index]
    path = p.root_path(*identity, base=tmp_path)
    action, root = p.choose_root(state, spec["search_stream"], path, BINDING,
                                 f"anchor_{spec['fit']}", identity, True, tmp_path)
    move = {"turn": state.turn_index, "actor": state.current_player,
            "state_sha256": p.old.digest(root["snapshot"]), "root_sha256": p.old.file_hash(path),
            "action": p.old.serialize_action(action)}
    count = root["allocation"]["world_count"]
    selection = {"root_id": a.root_id(0, state.turn_index),
                 "ordinary_world_indices": a.ordinary_indices("test", 0, state.turn_index, count),
                 "replay_world_indices": list(range(count)), "stress": True}
    receipt = a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection, full_fixture=True)
    assert receipt["recorded_worlds"] == receipt["replayed_worlds"] == {5: 103, 6: 86, 7: 74}[support]
    assert len(root["worlds"][-1]["candidate_indices"]) == {5: 2, 6: 2, 7: 1}[support]
    assert receipt["continuations"] == receipt["replayed_branches"] == 512


def test_sparse_sample_coverage_counts_delivered_last_world_cells():
    # Every distinct canonical-support stress cell gets full replay; these
    # three totals would be overcounted by multiplying maxworlds*support.
    rows = []
    for index, support in enumerate((5, 6, 7)):
        q, r = divmod(512, support)
        counts = [q+int(j < r) for j in range(support)]
        rows.append({"root_id": a.root_id(index, 0), "index": index, "turn": 0,
                     "fit": 161001, "arm": "candidate", "actor": 0, "phase": 0,
                     "privacy_category": 0, "support": support, "canonical_support": support,
                     "worlds": max(counts), "world_counts": counts, "continuations": 512})
    plan = a.sample_plan("main", rows, BINDING, "a"*64)
    assert plan["coverage"]["recorded_continuations"] == plan["coverage"]["replayed_continuations"] == 1536
    assert plan["coverage"]["recorded_worlds"] == plan["coverage"]["replayed_worlds"] == 263
    broken = copy.deepcopy(rows)
    broken[0]["continuations"] = 515
    with pytest.raises(AssertionError):
        a.sample_plan("main", broken, BINDING, "a"*64)

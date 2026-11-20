"""Frozen anchor preservation, public expansion identities and actual tiny games."""
import copy

import pytest

from tools.equal_budget_breadth_arena_128 import arena as p
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit

BINDING = "equal-budget-breadth-arena-isolated-fixture"


def fixture_registry():
    previous = p.old.read_json(p.c.SOURCE/"models_lock.json")["models"]
    return {"manifest":BINDING,"models":{f"anchor_{fit}":dict(previous[f"anchor_{fit}"]) for fit in p.FITS}}


@pytest.fixture(scope="module",autouse=True)
def models(tmp_path_factory):
    from tools.proposal_breadth_128 import engine as ancestor
    before_models,before_artifacts = dict(ancestor._MODELS),dict(ancestor._MODEL_ARTIFACTS)
    registry = fixture_registry()
    p.initialize_models(tmp_path_factory.mktemp("breadth_arena_models"),registry)
    assert ancestor._MODELS == before_models and ancestor._MODEL_ARTIFACTS == before_artifacts
    return registry


def private_state():
    env = p.old.SplendorEnv(seed=2179800000)
    state = env.initial_state()
    for _ in range(2):
        move = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state,move)
    for player in state.players:
        player.score = 14
        player.bonuses = {color:5 for color in player.bonuses}
    return state


def synthetic_anchor(size):
    keys = [f"key{i}" for i in range(size)]
    representatives = {key:{"canonical":key} for key in keys}
    offered = keys[:min(size,4)]
    anchor = {"group_keys":keys,"learned_group_scores":[1.]*size,
              "candidates":{"keys":offered,"actions":[{"incumbent":key} for key in offered],
                            "canonical_support":size,"legal_support":size}}
    return anchor,representatives


def test_balanced_disjoint_paired_schedules_and_same_model_for_both_actors():
    seen_seeds,seen_streams = set(),set()
    for stage,count in p.CONFIG["games"].items():
        schedule = p.specs(stage)
        assert len(schedule) == count
        seeds = {s["environment_seed"] for s in schedule}
        streams = {s["search_stream"] for s in schedule}
        assert len(seeds) == len(streams) == count//2
        assert not seeds.intersection(seen_seeds) and not streams.intersection(seen_streams)
        seen_seeds.update(seeds);seen_streams.update(streams)
        for fit in p.FITS:
            assert sum(s["fit"] == fit for s in schedule) == count//3
            assert p.model_id(stage,fit,True) == p.model_id(stage,fit,False) == f"anchor_{fit}"
        for first,second in zip(schedule[::2],schedule[1::2],strict=True):
            assert first["fit"] == second["fit"] and first["pair"] == second["pair"]
            assert first["environment_seed"] == second["environment_seed"] and first["search_stream"] == second["search_stream"]
            assert (first["candidate_seat"],second["candidate_seat"]) == (0,1)
    from tools.proposal_iteration_128 import arena as previous
    old_seeds = {s["environment_seed"] for stage in previous.CONFIG["games"] for s in previous.specs(stage)}
    assert not seen_seeds.intersection(old_seeds)


@pytest.mark.parametrize("support",range(1,12))
def test_short_support_preserves_incumbents_and_fixed_ranked_uniform_recipe(support):
    anchor,reps = synthetic_anchor(support)
    baseline = p.expanded_candidates(anchor,reps,["test",0,2],False)
    expanded = p.expanded_candidates(anchor,reps,["test",0,2],True)
    m,n = min(support,4),min(support,8)
    assert baseline["candidates"] == expanded["anchor_candidates"] == anchor["candidates"]
    assert baseline["added_indices"] == {"ranked":[],"random":[]}
    assert expanded["candidates"]["keys"][:m] == anchor["candidates"]["keys"]
    assert expanded["candidates"]["actions"][:m] == anchor["candidates"]["actions"]
    assert len(set(expanded["candidates"]["keys"])) == n
    assert expanded["added_indices"]["ranked"] == list(range(m,min(m+2,n)))
    assert expanded["added_indices"]["random"] == list(range(min(m+2,n),n))
    assert expanded["candidates"]["keys"][m:min(m+2,n)] == anchor["group_keys"][m:min(m+2,n)]
    assert expanded["candidates"]["actions"][m:] == [reps[key] for key in expanded["candidates"]["keys"][m:]]
    if support <= 4:
        assert expanded["candidates"] == baseline["candidates"]


def test_random_additions_use_full_public_game_turn_identity_without_replacement():
    anchor,reps = synthetic_anchor(20)
    for identity in (["test",0,2],["test",1,2],["test",0,3]):
        value = p.expanded_candidates(anchor,reps,identity,True)
        remaining = anchor["group_keys"][6:]
        expected = sorted(remaining,key=lambda key:(p.old.digest([p.CONFIG["random_tag"],*identity,key]),anchor["group_keys"].index(key)))[:2]
        assert value["candidates"]["keys"][6:] == expected
        assert value["root_identity"] == identity
        assert value == p.expanded_candidates(anchor,reps,identity,True)
    with pytest.raises(ValueError,match="identity"):
        p.expanded_candidates(anchor,reps,["test",0],True)
    with pytest.raises(ValueError,match="identity"):
        p.expanded_candidates(anchor,reps,["test",6,2],True)


@pytest.mark.parametrize("damage",["missing_group","duplicate_group","nan","duplicate_incumbent"])
def test_malformed_proposal_support_is_rejected(damage):
    anchor,reps = synthetic_anchor(10)
    if damage == "missing_group":
        reps.pop("key9")
    elif damage == "duplicate_group":
        anchor["group_keys"][-1] = "key8"
    elif damage == "nan":
        anchor["learned_group_scores"][-1] = float("nan")
    else:
        anchor["candidates"]["keys"][1] = "key0"
    with pytest.raises(ValueError):
        p.expanded_candidates(anchor,reps,["test",0,2],True)


@pytest.mark.parametrize("fit",p.FITS)
def test_exact_anchor_four_world_schedule_and_hidden_input_invariance(fit):
    state = private_state()
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    source,stream = f"anchor_{fit}","breadth-arena-private-fixture"
    anchor = p.learned_proposal(state,stream,source)
    expected = p.baseline.candidate_set(state,anchor["schedule"]["candidate_nonce"])
    assert anchor["reference"] == expected
    allowed = [j for j,key in enumerate(anchor["group_keys"]) if key not in expected["keys"][:3]]
    fourth = anchor["group_keys"][max(allowed,key=lambda j:anchor["learned_group_scores"][j])]
    assert anchor["candidates"]["keys"] == expected["keys"][:3]+[fourth]
    assert anchor["candidates"]["actions"][:3] == expected["actions"][:3]
    assert anchor["schedule"] == p.world128.root_schedule(state,stream,128)
    assert len(set(anchor["schedule"]["world_seeds"])) == 128
    assert anchor["model_sha256"] == p._MODEL_ARTIFACTS[source]["sha256"]
    alternate = input_audit.hidden_alternative(state)
    for breadth in (False,True):
        actual = p.proposal_for_root(state,stream,source,["test",0,state.turn_index],breadth)
        other = p.proposal_for_root(alternate,stream,source,["test",0,state.turn_index],breadth)
        input_audit.compare({k:v for k,v in actual.items() if k != "timing"},
                            {k:v for k,v in other.items() if k != "timing"})
        assert actual["anchor_candidates"] == anchor["candidates"]
    assert before == p.old.digest(p.old.serialize_state_snapshot(state))


def selection_worlds(values,counts):
    worlds = []
    for wi in range(max(counts)):
        indices = [j for j,count in enumerate(counts) if wi < count]
        worlds.append({"candidate_indices":indices,"outcomes":[{"wdl":values[j],"reason":"completed"} for j in indices]})
    return worlds


@pytest.mark.parametrize("support",range(1,12))
def test_allocation_exact_budget_public_remainders_and_no_extra_cells(support):
    anchor,reps = synthetic_anchor(support)
    for breadth in (False,True):
        candidates = p.expanded_candidates(anchor,reps,["test",0,2],breadth)["candidates"]
        allocation = p.allocation_for_candidates(candidates,["test",0,2],breadth)
        n,m = len(candidates["keys"]),min(support,4)
        q,r = divmod(128*m,n)
        expected_order = sorted(range(n),key=lambda j:(p.old.digest([p.CONFIG["remainder_tag"],"test",0,2,candidates["keys"][j]]),j))
        assert allocation["remainder_order"] == expected_order and allocation["remainder_indices"] == expected_order[:r]
        assert allocation["world_counts"] == [q+int(j in expected_order[:r]) for j in range(n)]
        assert allocation["continuations"] == 128*m and allocation["world_count"] == q+int(bool(r))
        worlds = selection_worlds([1]*n,allocation["world_counts"])
        result = p.select(worlds,allocation["world_counts"])
        assert result["choice"] == 0 and result["means"] == [1.]*n
        assert result["sums_wdl"] == result["counts"] == allocation["world_counts"]
        assert sum(len(w["outcomes"]) for w in worlds) == allocation["continuations"]
        if r:
            assert worlds[-1]["candidate_indices"] == sorted(expected_order[:r])


def test_selection_exact_rational_means_and_invalid_ragged_records():
    counts = [103,102,102,103,102]
    worlds = selection_worlds([1,1,0,0,0],counts)
    assert p.select(worlds,counts)["choice"] == 0  # Rationaltie, unequalinteger totals.
    # One lost unit makes candidate1 genuinely higher despite equalrawsums102.
    worlds[-1]["outcomes"][0]["wdl"] = 0
    assert p.select(worlds,counts)["sums_wdl"][:2] == [102,102]
    assert p.select(worlds,counts)["choice"] == 1
    for damage in ("extra_world","missing_world","active_order","missing_outcome","invalid_ending","invalid_wdl"):
        broken = copy.deepcopy(worlds)
        if damage == "extra_world":broken.append(copy.deepcopy(worlds[0]))
        elif damage == "missing_world":broken.pop()
        elif damage == "active_order":broken[-1]["candidate_indices"].reverse()
        elif damage == "missing_outcome":broken[-1]["outcomes"].pop()
        elif damage == "invalid_ending":broken[0]["outcomes"][0]["reason"] = "max_turns"
        else:broken[0]["outcomes"][0]["wdl"] = .5
        with pytest.raises(ValueError):p.select(broken,counts)


@pytest.mark.parametrize("support",range(1,5))
def test_short_support_select_matches_unchanged128_convention(support):
    counts = [128]*support
    worlds = selection_worlds([-1,0,1,1][:support],counts)
    result = p.select(worlds,counts)
    assert {k:result[k] for k in ("choice","means")} == p.world128.select(worlds,support,128)


@pytest.fixture(scope="module")
def tiny_games(tmp_path_factory):
    base = tmp_path_factory.mktemp("breadth_arena_games")
    snapshot = p.old.serialize_state_snapshot(private_state())
    p.old.save_new(base/"test_launch.json",{"manifest":BINDING})
    games = [p.play_game("test",spec,BINDING,base,snapshot) for spec in p.specs("test")]
    p.old.save_new(base/"test_games_lock.json",{"manifest":BINDING,"files":games})
    return base,snapshot,games


def test_all_fits_and_seats_complete_with_only_allocated_selection(tiny_games):
    base,snapshot,games = tiny_games
    roles = set()
    for item in games:
        path = p.game_path("test",item["index"],base)
        game = p.old.read_json(path)
        assert item["sha256"] == p.old.file_hash(path) and "failure" not in game
        assert game["reason"] in p.old.VALID_NATURAL_TERMINATIONS and game["turns"] >= 2
        for move in game["trace"]:
            rp = p.root_path("test",item["index"],move["turn"],base)
            root = p.old.read_json(rp)
            breadth = move["actor"] == game["spec"]["candidate_seat"]
            roles.add((game["spec"]["fit"],breadth))
            assert move["root_sha256"] == p.old.file_hash(rp)
            assert root["planner"] == ("candidate" if breadth else "reference")
            assert root["root_identity"] == ["test",item["index"],move["turn"]]
            assert root["proposal"]["breadth"] is breadth
            assert root["proposal"]["model_id"] == f"anchor_{game['spec']['fit']}"
            support = min(root["candidates"]["canonical_support"],8 if breadth else 4)
            assert len(root["candidates"]["keys"]) == support
            allocation = root["allocation"]
            assert allocation == root["proposal"]["allocation"]
            assert len(root["worlds"]) == allocation["world_count"] == max(allocation["world_counts"])
            assert sum(len(w["outcomes"]) for w in root["worlds"]) == allocation["continuations"] == 128*min(root["candidates"]["canonical_support"],4)
            assert [w["seed"] for w in root["worlds"]] == root["schedule"]["world_seeds"][:allocation["world_count"]]
            assert len(root["schedule"]["world_seeds"]) == 128
            for wi,world in enumerate(root["worlds"]):
                assert world["candidate_indices"] == [j for j,count in enumerate(allocation["world_counts"]) if wi < count]
            assert root["privacy"] == {"own":1,"opponent":1}
            assert "assessment" not in root and "extension" not in root
            assert p.select(root["worlds"],allocation["world_counts"]) == {k:root[k] for k in ("choice","means","sums_wdl","counts")}
            assert move["action"] == root["candidates"]["actions"][root["choice"]]
    assert roles == {(fit,breadth) for fit in p.FITS for breadth in (False,True)}
    original = p.old.file_hash(p.game_path("test",0,base))
    with pytest.raises(FileExistsError):
        p.play_game("test",p.specs("test")[0],BINDING,base,snapshot)
    assert p.old.file_hash(p.game_path("test",0,base)) == original


def test_world_budget_failure_preserves_partial_root_and_actual_game(tmp_path,monkeypatch):
    calls = 0
    def boundary(base):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise p.c.BudgetExceeded("fixture world boundary deadline")
    monkeypatch.setattr(p.c,"budget_boundary",boundary)
    snapshot = p.old.serialize_state_snapshot(private_state())
    with pytest.raises(p.c.BudgetExceeded,match="deadline"):
        p.play_game("test",p.specs("test")[0],BINDING,tmp_path,snapshot)
    game = p.old.read_json(p.game_path("test",0,tmp_path))
    root = p.old.read_json(p.root_path("test",0,2,tmp_path))
    assert game["trace"] == [] and game["failed_decision"]["root_sha256"] == p.old.file_hash(p.root_path("test",0,2,tmp_path))
    assert "fixture world boundary deadline" in game["failure"] and "failure" in root
    assert 0 < len(root["worlds"]) < 128
    assert game["final_state_sha256"] == p.old.digest(snapshot)


def support_state(support):
    state = p.old.SplendorEnv(seed=2179810000+support).initial_state()
    cards = [card for tier in state.visible_tier_cards for card in state.visible_tier_cards[tier]+state.hidden_tier_decks[tier]]
    chosen = sorted([card for card in cards if card.points > 0],key=lambda card:card.card_id)[:support]
    state.visible_tier_cards = {tier:[] for tier in state.visible_tier_cards}
    state.hidden_tier_decks = {tier:[] for tier in state.hidden_tier_decks}
    state.deck_counts = dict.fromkeys(state.deck_counts,0)
    state.bank_tokens = dict.fromkeys(state.bank_tokens,0)
    state.nobles = []
    for player in state.players:
        player.score = 14; player.bonuses = dict.fromkeys(player.bonuses,10)
    state.players[0].reserved_cards = chosen[:3]
    state.players[0].reserved_card_sources = ["deck"]*len(chosen[:3])
    for card in chosen[3:]:state.visible_tier_cards[card.tier].append(card)
    state.players[1].purchased_cards = [card for card in cards if card not in chosen]
    return state


@pytest.mark.parametrize("support",range(1,10))
def test_actual_private_short_and_ragged_roots_execute_exactly_their_allocation(tmp_path,monkeypatch,support):
    state = support_state(support)
    calls = []
    original = p.old.trajectory
    def trace(world,action):
        calls.append((world,action))
        return original(world,action)
    monkeypatch.setattr(p.old,"trajectory",trace)
    for index,breadth in enumerate((False,True)):
        before = len(calls)
        identity = ["test",index,state.turn_index]
        action,root = p.choose_root(state,"actual-ragged-fixture",p.root_path(*identity,base=tmp_path),
            BINDING,"anchor_161001",identity,breadth,tmp_path)
        counts = root["allocation"]["world_counts"]
        assert root["candidates"]["canonical_support"] == support
        assert len(calls)-before == sum(counts) == 128*min(support,4)
        assert len(root["worlds"]) == max(counts)
        assert all(len(w["outcomes"]) == sum(wi < count for count in counts) for wi,w in enumerate(root["worlds"]))
        assert action == p.old.decode_action(root["candidates"]["actions"][root["choice"]])
        if breadth and support >= 8:
            assert counts == [64]*8 and len(root["worlds"]) == 64
        assert "assessment" not in root and "teacher" not in root and "extension" not in root


def test_registry_path_and_snapshot_safety(models,tmp_path):
    with pytest.raises(ValueError,match="production"):
        p.initialize_models(p.OUTPUT,models)
    damaged = copy.deepcopy(models);damaged["models"]["anchor_161001"]["source_fit"] = 161002
    with pytest.raises(ValueError,match="fit"):
        p.initialize_models(tmp_path,damaged)
    damaged = copy.deepcopy(models);damaged["models"]["anchor_161001"]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        p.initialize_models(tmp_path,damaged)
    with pytest.raises(ValueError,match="production"):
        p.game_path("test",0,p.OUTPUT)
    snapshot = p.old.serialize_state_snapshot(private_state())
    for stage in ("main","probe"):
        with pytest.raises(ValueError,match="isolated test"):
            p.play_game(stage,p.specs(stage)[0],BINDING,tmp_path,snapshot)
        assert not p.game_path(stage,0,tmp_path).exists()
    with pytest.raises(ValueError,match="identity"):
        p.choose_root(private_state(),"fixture",tmp_path/"wrong.json",BINDING,"anchor_161001",["test",0,2],True,tmp_path)


def statistics_rows():
    return [{"index":s["index"],"pair":s["pair"],"seat":s["candidate_seat"],"fit":s["fit"],
             "score":(1.,.5,0.,1.,1.,0.,1.)[s["index"] % 7]} for s in p.specs("main")]


def test_paired_statistics_reject_missing_fit_pair_or_score_alignment():
    rows = statistics_rows()
    result = p.statistics(rows)
    assert result["share"] == sum(r["score"] for r in rows)/402
    assert result["wins"]+result["draws"]+result["losses"] == 402
    for damage in ("fit","pair","seat","score","missing"):
        changed = copy.deepcopy(rows)
        if damage == "fit":
            changed[0]["fit"] = next(f for f in p.FITS if f != changed[0]["fit"])
        elif damage == "pair":
            changed[0]["pair"] = 9
        elif damage == "seat":
            changed[0]["seat"] = 1
        elif damage == "score":
            changed[0]["score"] = .75
        else:
            changed.pop()
        with pytest.raises(ValueError):
            p.statistics(changed)
    # Seat-pair resampling must keep a win/loss pair together: no uncertainty here.
    for row in rows:
        row["score"] = 1.-row["seat"]
    assert p.statistics(rows)["ci95"] == [.5,.5]


@pytest.mark.parametrize("share,interval,label,passed",[
    (.56,[.51,.61],"positive_practical_progress",True),
    (.54,[.51,.57],"positive_below_observed_target",False),
    (.55,[.5,.6],"inconclusive",False),
    (.45,[.4,.49],"demonstrated_harm",False),
    (.5,[.46,.54],"below_material_target",False),
])
def test_practical_superiority_and_harm_are_distinct(share,interval,label,passed):
    result = p.decision(share,interval)
    assert result["classification"] == label and result["stage_pass"] is passed
    assert result["inferiority_established"] == (interval[1] < .5)

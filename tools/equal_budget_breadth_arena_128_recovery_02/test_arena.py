"""Bounded interval, exact-prefix reuse and one-shot recovery fixtures."""
import copy
from itertools import product

import pytest

from tools.equal_budget_breadth_arena_128_recovery_02 import arena as p
from tools.equal_budget_breadth_arena_128.test_arena import fixture_registry, private_state, support_state

BINDING = p.c.ORIGINAL_BINDING
RECOVERY_BINDING = "isolated-second-recovery-arena-fixture"


@pytest.fixture(scope="module",autouse=True)
def models(tmp_path_factory):
    registry = fixture_registry()
    p.initialize_models(tmp_path_factory.mktemp("recovery_models"),registry)
    return registry


def worlds(columns):
    return [{"candidate_indices":[j for j,col in enumerate(columns) if wi < len(col)],
             "outcomes":[{"reason":"repetition_cutoff" if col[wi] is None else "completed",
                          "winner":None,"wdl":col[wi]} for col in columns if wi < len(col)]}
            for wi in range(max(map(len,columns)))]


def install_resolvers(monkeypatch,root_origins=None,game_origins=None,partials=None):
    roots,games,partial = root_origins or {},game_origins or {},partials or {}
    def resolve_root(stage,index,turn,base):
        target = p.root_path(stage,index,turn,base)
        return target if target.exists() else roots.get((stage,index,turn))
    def resolve_game(stage,index,base):
        target = p.game_path(stage,index,base)
        return target if target.exists() else games.get((stage,index))
    monkeypatch.setattr(p.c,"resolve_root",resolve_root)
    monkeypatch.setattr(p.c,"resolve_game",resolve_game)
    monkeypatch.setattr(p.c,"resolve_partial_game",lambda stage,index,base:partial.get((stage,index)))
    monkeypatch.setattr(p.c,"validate_origin",lambda *args:True)


@pytest.mark.parametrize("columns,choice",[
    ([[1,1],[None,1]],0),       # Earlier candidate wins even at the later upper tie.
    ([[None,-1],[1,1]],1),      # Later candidate strictly exceeds the earlier upper bound.
    ([[None]],0),              # Singleton remains uniquely selected for every assignment.
    ([[1,1,1],[None,1]],0),     # Rational means, unequal integer totals.
    ([[-1,-1],[None,1]],1),
])
def test_certificates_obey_original_tie_order_and_unequal_counts(columns,choice):
    result = p.select(worlds(columns),list(map(len,columns)))
    assert result["choice"] == result["certificate"]["choice"] == choice
    for j,col in enumerate(columns):
        assert result["means"][j] is None if None in col else result["means"][j] == sum(col)/len(col)
    assert result["certificate"]["known_counts"] == [sum(v is not None for v in col) for col in columns]


@pytest.mark.parametrize("columns",[
    ([[None,1],[1,1]]),         # Later candidate cannot claim an earlier candidate's possible tie.
    ([[None,0],[0,0]]),
    ([[None],[None]]),
    ([[0,0],[None,1]]),
])
def test_ambiguous_decisions_hold_without_draws_or_midpoint_fallback(columns):
    with pytest.raises(p.AmbiguousDecision) as error:
        p.select(worlds(columns),list(map(len,columns)))
    assert error.value.selection["choice"] is None
    assert error.value.selection["certificate"]["choice"] is None


def test_interval_certificate_agrees_with_exhaustive_bounded_assignments():
    for entries in product((-1,0,1,None),repeat=4):
        columns = [list(entries[:2]),list(entries[2:])]
        unknown = [(j,k) for j,col in enumerate(columns) for k,value in enumerate(col) if value is None]
        expected = set()
        for values in product((-1,0,1),repeat=len(unknown)):
            assigned = copy.deepcopy(columns)
            for (j,k),value in zip(unknown,values):assigned[j][k] = value
            expected.add(max(range(2),key=lambda j:sum(assigned[j])))
        if len(expected) == 1:
            assert p.select(worlds(columns),[2,2])["choice"] == next(iter(expected))
        else:
            with pytest.raises(p.AmbiguousDecision):p.select(worlds(columns),[2,2])


@pytest.mark.parametrize("support",range(1,9))
def test_natural_arithmetic_is_exactly_the_frozen_operator(support):
    counts = [128]*support
    recorded = worlds([[[-1,0,1][j%3]]*128 for j in range(support)])
    assert p.select(recorded,counts) == p.original.select(recorded,counts)
    assert p.specs is p.original.specs and p.proposal_for_root is p.original.proposal_for_root
    assert p.statistics is p.original.statistics and p.model_id is p.original.model_id


@pytest.mark.parametrize("damage",["no_progress_cutoff","max_turns","stalled","invalid_wdl","invented_draw","invented_winner","missing_cell","extra_world","bad_active"])
def test_only_null_repetition_can_be_censored(damage):
    recorded = worlds([[1,1],[None,1]])
    if damage in ("no_progress_cutoff","max_turns","stalled"):recorded[0]["outcomes"][1]["reason"] = damage
    elif damage == "invalid_wdl":recorded[0]["outcomes"][0]["wdl"] = .5
    elif damage == "invented_draw":recorded[0]["outcomes"][1]["wdl"] = 0
    elif damage == "invented_winner":recorded[0]["outcomes"][1]["winner"] = 0
    elif damage == "missing_cell":recorded[1]["outcomes"].pop()
    elif damage == "extra_world":recorded.append(copy.deepcopy(recorded[-1]))
    else:recorded[0]["candidate_indices"].reverse()
    with pytest.raises(ValueError):p.select(recorded,[2,2])


@pytest.fixture
def inherited_failed_root(tmp_path,monkeypatch):
    origin,newbase = tmp_path/"origin",tmp_path/"recovery"
    state = support_state(4)
    identity = ["test",0,state.turn_index]
    path = p.original.root_path(*identity,base=origin)
    _,root = p.original.choose_root(state,"fixture-recovery-prefix",path,BINDING,"anchor_161001",identity,True,origin)
    # This is a constructed fixture origin, never a production artifact.
    failed = copy.deepcopy(root)
    failed["worlds"] = failed["worlds"][:115]
    failed["worlds"][-1]["outcomes"][3].update(reason="repetition_cutoff",winner=None,wdl=None)
    for key in ("choice","means","sums_wdl","counts"):failed.pop(key)
    failed["failure"] = "synthetic original repetition failure"
    partial = tmp_path/"constructed_partial_root.json"
    p.old.save_new(partial,failed)
    install_resolvers(monkeypatch,{tuple(identity):partial})
    return state,identity,partial,newbase,failed


def test_resume_preserves_460_cells_and_executes_only_remaining_52(inherited_failed_root,monkeypatch):
    state,identity,origin,base,failed = inherited_failed_root
    before = p.old.file_hash(origin)
    calls = []
    trajectory = p.old.trajectory
    def traced(world,action):
        calls.append(action)
        return trajectory(world,action)
    monkeypatch.setattr(p.old,"trajectory",traced)
    monkeypatch.setattr(p,"proposal_for_root",lambda *args:pytest.fail("Inherited proposal must not be recomputed"))
    path = p.root_path(*identity,base=base)
    action,record = p.choose_root(state,"fixture-recovery-prefix",path,BINDING,"anchor_161001",identity,True,base,RECOVERY_BINDING)
    assert len(calls) == record["new_cells"] == 52 and record["inherited_cells"] == 460
    assert record["worlds"][:115] == failed["worlds"] and len(record["worlds"]) == 128
    assert record["certificate"]["unresolved_counts"] == [0,0,0,1]
    assert record["choice"] == 0 and record["means"] == [1.,1.,1.,None]
    assert record["manifest"] == BINDING and record["recovery_binding"] == RECOVERY_BINDING
    assert p.old.file_hash(origin) == before
    action2,record2 = p.choose_root(state,"fixture-recovery-prefix",path,BINDING,"anchor_161001",identity,True,base,RECOVERY_BINDING)
    assert action2 == action and p.old.digest(record2) == p.old.digest(record) and len(calls) == 52


def test_ambiguous_root_finishes_all_allocated_cells_then_saves_hold(tmp_path,monkeypatch):
    install_resolvers(monkeypatch)
    state = support_state(4)
    identity = ["test",0,state.turn_index]
    calls = []
    def unresolved(world,action):
        calls.append(action)
        return {"reason":"repetition_cutoff","winner":None,"turns":18,"trace_sha256":"fixture","final_state_sha256":"fixture"}
    monkeypatch.setattr(p.old,"trajectory",unresolved)
    def distinct_proof(record,wi,ci):
        context = {"fixture_distinct_cell":[wi,ci]}
        return {"world":wi,"candidate":ci,"replay":{},"context":context,"context_sha256":p.old.digest(context)}
    monkeypatch.setattr(p.proof,"build",distinct_proof)
    path = p.root_path(*identity,base=tmp_path)
    with pytest.raises(p.AmbiguousDecision):
        p.choose_root(state,"fixture",path,BINDING,"anchor_161001",identity,True,tmp_path,RECOVERY_BINDING)
    record = p.old.read_json(path)
    assert len(calls) == record["new_cells"] == 512
    assert record["choice"] is None and record["hold"] == "ambiguous_coupled_repetition_interval"
    assert record["means"] == [None]*4
    with pytest.raises(ValueError,match="cannot be retried"):
        p.choose_root(state,"fixture",path,BINDING,"anchor_161001",identity,True,tmp_path,RECOVERY_BINDING)
    assert len(calls) == 512


def test_outer_prefix_replayed_with_original_hashes_and_no_repeated_continuations(tmp_path,monkeypatch):
    origin,base = tmp_path/"origin",tmp_path/"recovery"
    snapshot = p.old.serialize_state_snapshot(private_state())
    spec = p.specs("test")[0]
    p.original.play_game("test",spec,BINDING,origin,snapshot)
    complete = p.old.read_json(p.original.game_path("test",0,origin))
    assert len(complete["trace"]) >= 2
    roots = {("test",0,row["turn"]):p.original.root_path("test",0,row["turn"],origin) for row in complete["trace"]}
    partial = copy.deepcopy(complete)
    partial["trace"] = partial["trace"][:1]
    next_root = p.old.read_json(roots[("test",0,complete["trace"][1]["turn"])])
    partial.update(final_state=next_root["snapshot"],failure="synthetic original failure",reason="exception",winner=None)
    partial_path = tmp_path/"constructed_partial_game.json"
    p.old.save_new(partial_path,partial)
    original_hash = p.old.file_hash(partial_path)
    install_resolvers(monkeypatch,roots,partials={("test",0):partial_path})
    monkeypatch.setattr(p.old,"trajectory",lambda *args:pytest.fail("Prefix replay must not repeat simulations"))
    result = p.play_game("test",spec,BINDING,base,snapshot,RECOVERY_BINDING)
    resumed = p.old.read_json(p.game_path("test",0,base))
    assert resumed["trace"] == complete["trace"] and resumed["trace_sha256"] == complete["trace_sha256"]
    assert resumed["final_state"] == complete["final_state"] and resumed["winner"] == complete["winner"]
    assert resumed["inherited_turns"] == 1 and p.old.file_hash(partial_path) == original_hash
    assert p.play_game("test",spec,BINDING,base,snapshot,RECOVERY_BINDING) == result


def test_production_fixture_guard_and_required_binding(tmp_path,models):
    with pytest.raises(ValueError,match="production"):p.initialize_models(p.OUTPUT,models)
    with pytest.raises(ValueError,match="production"):p.game_path("test",0,p.OUTPUT)
    with pytest.raises(ValueError,match="recovery binding"):
        p.play_game("test",p.specs("test")[0],BINDING,tmp_path)


def test_full_ambiguous_original_root_needs_no_new_cells_and_preserves_previous_binding(tmp_path,monkeypatch):
    from tools.equal_budget_breadth_arena_128_recovery_02.test_proof import SOURCE,SOURCE_SHA,DIAG
    source = p.old.read_json(SOURCE)
    assert p.old.file_hash(SOURCE) == SOURCE_SHA
    identity = source["root_identity"]
    state = p.old.deserialize_state_snapshot(source["snapshot"])
    install_resolvers(monkeypatch,{tuple(identity):SOURCE})
    builds = []
    def from_saved(record,wi,ci):
        builds.append((wi,ci))
        replay = p.old.read_json(DIAG/f"candidate_{ci}_reproduction.json")["reproduced"]
        context = p.proof.context_from_replay(record,wi,ci,replay)
        return {"world":wi,"candidate":ci,"replay":replay,"context":context,"context_sha256":p.old.digest(context)}
    monkeypatch.setattr(p.proof,"build",from_saved)
    monkeypatch.setattr(p.old,"trajectory",lambda *args,**kwargs:pytest.fail("All512 cells already exist"))
    path = p.root_path(*identity,base=tmp_path)
    _,record = p.choose_root(state,p.specs("main")[310]["search_stream"],path,BINDING,"anchor_161003",identity,True,tmp_path,RECOVERY_BINDING)
    assert record["inherited_cells"] == 512 and record["new_cells"] == 0
    assert record["worlds"] == source["worlds"] and record["choice"] == 0
    assert record["recovery_binding"] != source["recovery_binding"]
    assert builds == [(42,0),(42,1)] and p.old.file_hash(SOURCE) == SOURCE_SHA
    monkeypatch.setattr(p.proof,"build",lambda *args:pytest.fail("Saved proof must be reused"))
    _,reused = p.choose_root(state,p.specs("main")[310]["search_stream"],path,BINDING,"anchor_161003",identity,True,tmp_path,RECOVERY_BINDING)
    assert p.old.digest(record) == p.old.digest(reused)

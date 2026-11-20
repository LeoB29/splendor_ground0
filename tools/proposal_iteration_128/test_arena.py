"""Role identity, paired inference, and complete tiny-game audit contracts."""
import copy
from pathlib import Path
import sys
import time

import pytest

from tools.proposal_iteration_128 import arena as p
from tools.proposal_iteration_128 import verify as a
from scripts.analysis import learned_proposal_screen as ancestor

BINDING = "proposal-iteration-isolated-fixture"


@pytest.fixture(scope="module", autouse=True)
def models(tmp_path_factory):
    base = tmp_path_factory.mktemp("iteration_models")
    # Different known tensors for the three logical arms make wrong-role model
    # use observable. This is a synthetic registry, never a production lineage.
    registry = {"manifest": BINDING, "models": {}}
    for offset, arm in enumerate(("anchor", "treatment", "replay")):
        for index, fit in enumerate(p.FITS):
            source = p.FITS[(index+offset) % 3]
            path = ancestor.model_path(source)
            registry["models"][f"{arm}_{fit}"] = {
                "path": path.relative_to(p.ROOT).as_posix(), "sha256": p.old.file_hash(path),
                "source_fit": fit, "selected_update": ancestor.MODELS[source][0]}
    previous = dict(ancestor._MODELS)
    p.initialize_models(base, registry)
    assert ancestor._MODELS == previous
    return registry


def private_state():
    env = p.old.SplendorEnv(seed=2079800000)
    state = env.initial_state()
    for _ in range(2):
        move = next(v for v in env.legal_actions(state)
                    if v.action_type.name == "RESERVE_DECK" and v.tier == 2)
        state = env.step(state, move)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


def test_balanced_disjoint_schedules_and_independent_model_roles():
    seen = set()
    for stage, count in p.CONFIG["games"].items():
        rows = p.specs(stage)
        assert rows == a.independent_specs(stage) and len(rows) == count
        seeds = {r["environment_seed"] for r in rows}
        assert len(seeds) == count // 2 and not seeds & seen
        seen.update(seeds)
        for fit in p.FITS:
            assert sum(r["fit"] == fit for r in rows) == count // 3
            for candidate in (False, True):
                expected = f"{'treatment' if candidate else 'replay' if stage in ('stage_b','probe_b') else 'anchor'}_{fit}"
                assert p.model_id(stage, fit, candidate) == a.independent_model_id(stage, fit, candidate) == expected
        for left, right in zip(rows[::2], rows[1::2]):
            assert left["fit"] == right["fit"] and left["search_stream"] == right["search_stream"]
            assert left["environment_seed"] == right["environment_seed"]
            assert (left["candidate_seat"], right["candidate_seat"]) == (0, 1)


@pytest.mark.parametrize("arm", ["anchor", "treatment", "replay"])
@pytest.mark.parametrize("fit", p.FITS)
def test_new_and_anchor_models_have_independent_lawful_proposals(arm, fit):
    state = private_state()
    source = f"{arm}_{fit}"
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    actual = p.learned_proposal(state, "dynamic-proposal-fixture", source)
    expected = a.independent_proposal(state, "dynamic-proposal-fixture", source)
    a.compare({k: v for k, v in actual.items() if k != "timing"}, expected)
    altered = a.a157.hidden_alternative(state)
    a.compare(expected, a.independent_proposal(altered, "dynamic-proposal-fixture", source))
    assert actual["candidates"]["keys"][:3] == actual["reference"]["keys"][:3]
    assert actual["model_sha256"] == p._MODEL_ARTIFACTS[source]["sha256"]
    assert len(actual["schedule"]["world_seeds"]) == 128
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == before


def test_role_tensors_are_distinct_and_ancestor_registry_unchanged():
    state = private_state()
    values = [p.learned_proposal(state, "role-sensitive", f"{arm}_161001")["learned_logits"]
              for arm in ("anchor", "treatment", "replay")]
    assert len({tuple(row) for row in values}) == 3


def test_registry_and_synthetic_production_guards(models, tmp_path):
    with pytest.raises(ValueError):
        p.initialize_models(p.OUTPUT, models)
    changed = copy.deepcopy(models)
    changed["models"]["treatment_161001"]["source_fit"] = 161002
    with pytest.raises(ValueError):
        p.initialize_models(tmp_path, changed)
    changed = copy.deepcopy(models)
    changed["models"]["replay_161003"]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        p.initialize_models(tmp_path, changed)
    snapshot = p.old.serialize_state_snapshot(private_state())
    with pytest.raises(ValueError):
        p.game_path("test", 0, p.OUTPUT)
    for stage in ("stage_a", "stage_b", "probe_a", "probe_b"):
        with pytest.raises(ValueError):
            p.play_game(stage, p.specs(stage)[0], BINDING, tmp_path, snapshot)


@pytest.fixture(scope="module")
def tiny_games(tmp_path_factory):
    base = tmp_path_factory.mktemp("iteration_games")
    snapshot = p.old.serialize_state_snapshot(private_state())
    p.old.save_new(base / "test_launch.json", {"manifest": BINDING})
    games = [p.play_game("test", spec, BINDING, base, snapshot) for spec in p.specs("test")]
    p.old.save_new(base / "test_games_lock.json", {"manifest": BINDING, "files": games})
    a.seal_sample("test", games, BINDING, base)
    a.validate_sample("test", BINDING, base)
    return base, snapshot, games


def test_six_complete_private_games_all_fits_and_seats_audited(tiny_games):
    base, snapshot, games = tiny_games
    branches, roots = 0, 0
    for item in games:
        receipt = a.audit_game("test", item["index"], item["sha256"], BINDING, base, snapshot)
        assert {r["planner"] for r in receipt["roots"]} == {"candidate", "reference"}
        assert not receipt["full_inner_replay"]
        for row in receipt["roots"]:
            expected_arm = "treatment" if row["planner"] == "candidate" else "anchor"
            assert row["model_id"] == f"{expected_arm}_{row['assigned_fit']}"
            assert row["stress"] and row["replayed_worlds"] == 128
            assert row["continuations"] == row["replayed_branches"] == 128*row["support"]
            branches += row["replayed_branches"]
            roots += 1
    plan = a.load_sample("test", BINDING, base)
    assert plan["coverage"]["roots"] == roots
    assert plan["coverage"]["replayed_continuations"] == branches


def root_copy(tiny_games, target, index=0):
    base, snapshot, _ = tiny_games
    game = p.old.read_json(p.game_path("test", index, base))
    move = game["trace"][0]
    state = p.old.deserialize_state_snapshot(snapshot)
    p.old.save_new(target / "test_launch.json", {"manifest": BINDING})
    path = p.root_path("test", index, move["turn"], target)
    root = p.old.read_json(p.root_path("test", index, move["turn"], base))
    p.old.save_new(path, root)
    ordinary = a.ordinary_indices("test", index, move["turn"])
    selection = {"root_id": a.root_id(index, move["turn"]), "ordinary_world_indices": ordinary,
                 "replay_world_indices": ordinary, "stress": False}
    return game["spec"], state, move, root, path, selection


@pytest.mark.parametrize("damage", ["model_id", "model_sha", "proposal_logits", "world", "sampled_trace", "means", "choice", "last_outcome"])
def test_rehashed_semantic_corruption_is_rejected(tiny_games, tmp_path, damage):
    spec, state, move, root, path, selection = root_copy(tiny_games, tmp_path)
    if damage == "model_id":
        root["proposal"]["model_id"] = f"replay_{spec['fit']}"
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
    else:
        root["worlds"][-1]["outcomes"].pop()
    p.old.atomic_write(path, root)
    move["root_sha256"] = p.old.file_hash(path)
    with pytest.raises((AssertionError, ValueError)):
        a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection)


def test_sampled_assurance_explicitly_leaves_unselected_trace_unreplayed(tiny_games, tmp_path):
    spec, state, move, root, path, selection = root_copy(tiny_games, tmp_path)
    wi = next(i for i in range(128) if i not in selection["replay_world_indices"])
    root["worlds"][wi]["outcomes"][0]["trace_sha256"] = "0"*64
    p.old.atomic_write(path, root)
    move["root_sha256"] = p.old.file_hash(path)
    result = a.audit_root("test", 0, state, move, spec, BINDING, tmp_path, selection)
    assert result["replayed_branches"] == 8*result["support"]
    assert not result["full_inner_replay"] and result["audit_timing"]["stress_extra_replay_wall"] == 0


def test_sample_plan_outcome_independence_and_metadata_tamper(tiny_games):
    base, _, games = tiny_games
    inventory = a.metadata_inventory("test", games, BINDING, base)
    plan = a.sample_plan("test", inventory, BINDING, "a"*64)
    changed = copy.deepcopy(inventory)
    for row in changed:
        row.update(outcome=-1., score=1., choice=99, sha256="0"*64)
    alternative = a.sample_plan("test", changed, "changed", "b"*64)
    assert plan["selections"] == alternative["selections"]
    assert plan["stress_cells"] == alternative["stress_cells"]
    assert plan["coverage"]["stress_cells"] == 162
    assert plan["coverage"]["stress_roots"] <= 162
    bad = copy.deepcopy(plan)
    bad["selections"][0]["ordinary_world_indices"] = list(range(8))
    with pytest.raises((ValueError, AssertionError)):
        a.compare(bad, plan)
    with pytest.raises(AssertionError):
        a.metadata_inventory("test", games[:-1], BINDING, base)


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
def test_independent_paired_statistics_and_stage_specific_gates(stage):
    rows = [{"index": s["index"], "pair": s["pair"], "seat": s["candidate_seat"],
             "fit": s["fit"], "score": (1., .5, 0., 1., 1., 0., 1.)[s["index"] % 7]}
            for s in p.specs(stage)]
    a.compare(p.statistics(rows, stage), a.independent_statistics(rows, stage))
    damaged = copy.deepcopy(rows)
    damaged[0]["fit"] = next(fit for fit in p.FITS if fit != damaged[0]["fit"])
    with pytest.raises(ValueError):
        p.statistics(damaged, stage)
    assert not p.decision(.54, [.51, .57], "stage_a")["stage_pass"]
    assert p.decision(.54, [.51, .57], "stage_b")["stage_pass"]
    assert not p.decision(.55, [.5, .6], stage)["stage_pass"]


def test_two_hidden_spawn_pools_use_new_registry_and_worker_initializer(tmp_path, models):
    base = tmp_path / "synthetic"
    base.mkdir()
    fixture = {"kind": "proposal-iteration-128-synthetic-v1", "base": str(base), "binding": BINDING,
               "snapshot": p.old.serialize_state_snapshot(private_state()), "full_fixture_replay": True,
               "registry": models}
    path = tmp_path / "fixture.json"
    p.old.save_new(path, fixture)
    launch = p.windows.detached_launch([sys.executable, "-u", "-m", "tools.proposal_iteration_128.verify", "--fixture", path],
                                       base / "out.txt", base / "err.txt", p.ROOT)
    result_path = base / "fixture_result.json"
    deadline = time.monotonic()+180
    while not result_path.exists():
        if time.monotonic() > deadline:
            logs = [entry.read_text(encoding="utf-8") for entry in base.rglob("*err*.txt")]
            pytest.fail(f"Synthetic hidden pools timed out PID {launch['pid']}: {logs}")
        time.sleep(.05)
    result = p.old.read_json(result_path)
    assert result["passed"] and result["games"] == 6 and result["roots"] >= 12
    assert result["continuations"] == result["replayed_branches"] and result["full_fixture_replay"]
    for kind in ("test_games", "test_audits"):
        healths = list((base / "workers" / kind).glob("*_health.json"))
        assert len(healths) == 2
        for path in healths:
            health = p.old.read_json(path)
            assert health["manifest"] == BINDING
            if sys.platform == "win32":
                assert Path(health["executable"]).name == Path(health["parent_executable"]).name == "pythonw.exe"
                assert health["console_window"] == health["console_process_count"] == 0

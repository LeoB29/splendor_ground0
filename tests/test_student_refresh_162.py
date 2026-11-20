import copy
from collections import Counter
from pathlib import Path

import pytest
import torch

from splendor_ai.alphazero import student_refresh_162 as h
from scripts.analysis import verify_student_refresh_162 as a
from splendor_ai.engine.actions import ActionType


@pytest.fixture(scope="module", autouse=True)
def initialize():
    h.old.initialize_worker()


def test_source_and_test_panels_splits_and_disjoint_streams():
    assert h.source_schedule() == [a.independent_spec("source", i) for i in range(384)]
    assert h.splits() == a.independent_split()
    assert sum(k.startswith("162:") and v == "validation" for k, v in h.splits().items()) == 26
    assert sum(k.startswith("160:") and v == "validation" for k, v in h.splits().items()) == 40
    source = h.source_schedule()
    for cluster in range(128):
        group = source[3*cluster:3*cluster+3]
        assert {s["seed"] for s in group} == set(h.SEEDS)
        assert len({s["environment_seed"] for s in group}) == len({s["stream"] for s in group}) == 1
    sets = [{s["environment_seed"] for s in source}]
    for stage, count in (("probe", 48), ("main", 4800)):
        panel = h.schedule(stage)
        assert panel == [a.independent_spec(stage, i) for i in range(count)]
        assert len(panel) == count
        sets.append({s["environment_seed"] for s in panel})
        for cluster in range(count//12):
            group = panel[12*cluster:12*cluster+12]
            assert len({(s["comparison"], s["seed"], s["seat"]) for s in group}) == 12
            assert len({s["stream"] for s in group}) == 1
    assert all(x.isdisjoint(y) for i, x in enumerate(sets) for y in sets[i+1:])
    assert all(set(h.model_keys(s)) == {f"refresh_{s['seed']}", f"{'replay' if s['comparison'] == 'transfer' else 'anchor'}_{s['seed']}"}
               for s in h.schedule("probe"))


@pytest.mark.parametrize("turns", [6, 19, 25, 45, 65])
def test_outcome_blind_sampling_phase_seat_coverage_and_shortage(turns):
    game = {"spec": h.source_schedule()[0], "winner": 0,
            "trace": [{"turn": i, "actor": i % 2, "phase": min(i//20, 2),
                       "evidence": {"choice": i % 4}} for i in range(turns)]}
    picked = h.sampled_turns(game)
    assert picked == a.independent_turns(game) and len(picked) == 6
    present = {(r["actor"], r["phase"]) for r in game["trace"]}
    assert {(game["trace"][i]["actor"], game["trace"][i]["phase"]) for i in picked} == present
    game["winner"] = 1
    for row in game["trace"]:
        row["evidence"] = {"choice": 999, "scores": [-1e10]}
    assert h.sampled_turns(game) == picked
    game["trace"].reverse()
    assert h.sampled_turns(game) == a.independent_turns(game) == picked


def test_sample_manifest_independent_audit_and_probe_no_overlap(tmp_path):
    files = []
    for spec in h.source_schedule():
        game = {"manifest": "fixture", "spec": spec,
                "trace": [{"turn": i, "actor": i % 2, "phase": min(i//20, 2)} for i in range(48)]}
        path = tmp_path / f"game_{spec['index']}.json"
        h.old.save_new(path, game)
        files.append(h.artifact(path, index=spec["index"]))
    samples = h.sample_inventory(files, "fixture")
    assert a.audit_samples(samples, files, "fixture")
    assert Counter(s["split"] for s in samples) == {"train": 1836, "validation": 468}
    assert Counter(s["seed"] for s in samples) == {seed: 768 for seed in h.SEEDS}
    probe = h.probe_indices(samples)
    assert probe == a.independent_probe_indices(samples) and len(set(probe)) == 18
    assert len({(samples[i]["seed"], samples[i]["seat"], samples[i]["phase"]) for i in probe}) == 18
    changed = copy.deepcopy(samples)
    changed[0]["split"] = "invalid"
    with pytest.raises(AssertionError):
        a.audit_samples(changed, files, "fixture")


def tiny_rows():
    root = h.old.read_json(h.teacher.root_path("main", 0, 0))
    state = h.old.deserialize_state_snapshot(root["snapshot"])
    template = h.f.make_row(state, root, {"experiment": 160, "index": 0, "pair": "160:0"}, "train", h.old._RAW._model.config)
    return [{**copy.deepcopy(template), "id": f"{160 if i < 3 else 162}:0:{i}",
             "pair": f"{160 if i < 3 else 162}:{i//2}", "split": "train" if i < 5 else "validation",
             "teacher": i % len(template["members"])} for i in range(7)]


def test_exact_matching_of_training_dose_and_old_only_replacements():
    training = [r for r in tiny_rows() if r["split"] == "train"]
    for seed in h.SEEDS:
        for epoch in range(4):
            plan = h.batch_plan(training, seed, epoch)
            assert plan == a.independent_plan(training, seed, epoch)
            slots, replay = [i for x, _ in plan for i in x], [i for _, x in plan for i in x]
            assert sorted(slots) == list(range(len(training)))
            assert len(slots) == len(replay)
            assert all(training[i]["id"].startswith("160:") for i in replay)
            assert all(i == j for i, j in zip(slots, replay) if training[i]["id"].startswith("160:"))
    with pytest.raises(AssertionError):
        h.batch_plan(tiny_rows(), 162001, 0)
    assert max(h.f.checkpoint_updates(11413)) == 716
    assert 4*11413 == 45652


def test_both_fits_complete_independent_bitwise_replay_and_checkpoint_tamper(tmp_path):
    rows = tiny_rows()
    for arm in h.ARMS:
        h.train_fit(arm, 162001, rows, tmp_path)
        a.replay_fit(arm, 162001, rows, tmp_path)
    results = [h.old.read_json(tmp_path / "fits" / f"{arm}_162001" / "result.json") for arm in h.ARMS]
    assert results[0]["slot_sequence_sha256"] == results[1]["slot_sequence_sha256"]
    assert results[0]["batch_sequence_sha256"] != results[1]["batch_sequence_sha256"]
    assert results[0]["updates"] == results[1]["updates"] == 4
    assert results[0]["validation_rows"] == results[1]["validation_rows"] == 2
    folder = tmp_path / "fits" / "refresh_162001"
    path = folder / "checkpoint_0001.pt"
    payload = torch.load(path, weights_only=True)
    next(iter(payload["model_state_dict"].values())).flatten()[0] += 1
    torch.save(payload, path)
    with pytest.raises(AssertionError):
        a.replay_fit("refresh", 162001, rows, tmp_path)
    with pytest.raises(AssertionError):
        h.train_fit("refresh", 162001, rows, h.OUTPUT)
    with pytest.raises(AssertionError):
        h.train_fit("refresh", 162001, rows, h.OUTPUT / "fixture")


def test_initial_checkpoint_eligible_and_earliest_loss_tie(tmp_path):
    rows = tiny_rows()
    for row in rows:
        row["members"] = row["members"][:1]
        row["teacher"] = 0
    h.train_fit("refresh", 162001, rows, tmp_path)
    a.replay_fit("refresh", 162001, rows, tmp_path)
    result = h.old.read_json(tmp_path / "fits/refresh_162001/result.json")
    assert result["selected_update"] == 0


def rich_state():
    env = h.old.SplendorEnv(seed=1629800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(m for m in env.legal_actions(state) if m.action_type == ActionType.RESERVE_DECK and m.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


@pytest.mark.parametrize("comparison", ["selfplay", "transfer", "progress"])
def test_complete_source_or_arena_game_replay_and_snapshot_tamper(tmp_path, comparison):
    spec = {**h.schedule("probe")[0], "comparison": comparison, "environment_seed": 1629800000}
    model = h.f.StrongPolicyInference(h.initializer(162001)["path"]).model
    keys = (f"anchor_{spec['seed']}",) if comparison == "selfplay" else h.model_keys(spec)
    for key in keys:
        h.MODELS[key] = model
    item = h.play_game("test", spec, "fixture", tmp_path, h.old.serialize_state_snapshot(rich_state()))
    a.replay_game("test", spec, item, "fixture", tmp_path)
    game = h.old.read_json(Path(item["path"]))
    if comparison == "selfplay":
        game["trace"][0]["snapshot"]["turn_index"] += 1
    else:
        game["trace"][0]["evidence"]["offered_scores"][0] += 1
    path = tmp_path / "tampered.json"
    h.old.save_new(path, game)
    with pytest.raises(AssertionError):
        a.replay_game("test", spec, h.artifact(path), "fixture", tmp_path / "reject")
    with pytest.raises(AssertionError):
        h.play_game("test", spec, "fixture", tmp_path)
    with pytest.raises(AssertionError):
        h.play_game("source", spec, "fixture", tmp_path / "not_test", h.old.serialize_state_snapshot(rich_state()))


def test_failed_game_is_preserved_and_no_retry(tmp_path):
    spec = h.schedule("probe")[0]
    state = rich_state()
    state.turn_index = 400
    with pytest.raises(AssertionError, match="max_turns"):
        h.play_game("test", spec, "fixture", tmp_path, h.old.serialize_state_snapshot(state))
    assert "failure" in h.old.read_json(h.game_path("test", 0, tmp_path))
    with pytest.raises(AssertionError, match="replacement"):
        h.play_game("test", spec, "fixture", tmp_path)


def test_teacher_query_full_world_checks_sampled_branch_replay_and_tamper(tmp_path):
    # A short, private-information-rich teacher fixture; no real source corpus is used.
    state = rich_state()
    source_spec = h.source_schedule()[0]
    _, evidence = h.f.choose(state, source_spec["stream"], h.f.StrongPolicyInference(h.initializer(162001)["path"]).model)
    move = {"turn": state.turn_index, "actor": state.current_player, "phase": min(state.turn_index//20, 2),
            "snapshot": h.old.serialize_state_snapshot(state), "evidence": evidence}
    move["state_sha256"] = h.old.digest(move["snapshot"])
    game = {"manifest": "fixture", "spec": source_spec,
            "trace": [move]+[{"turn": state.turn_index+i, "actor": i % 2, "phase": 0} for i in range(1, 6)]}
    path = tmp_path / "source.json"
    h.old.save_new(path, game)
    spec = {"index": 0, "source": h.artifact(path, index=0), "source_index": 0, "pair": "162:0",
            "seed": 162001, "turn": state.turn_index, "seat": state.current_player, "phase": 0,
            "stream": source_spec["stream"], "split": h.splits()["162:0"]}
    item = h.query_teacher(spec, "fixture", tmp_path)
    report = a.audit_teacher(spec, item, "fixture", tmp_path)
    check = h.old.read_json(Path(report["path"]))
    assert check["passed"] and len(check["replay_world_indices"]) == 2
    assert check["continuations"] == 64*check["replayed_branches"]
    with pytest.raises(FileExistsError):
        h.query_teacher(spec, "fixture", tmp_path)
    receipt = h.old.read_json(Path(item["path"]))
    root_path = Path(receipt["root"]["path"])
    root = h.old.read_json(root_path)
    root["worlds"][0]["state_sha256"] = "0"*64
    changed_root = tmp_path / "bad_root.json"
    h.old.save_new(changed_root, root)
    receipt["root"] = h.artifact(changed_root)
    changed = tmp_path / "bad_receipt.json"
    h.old.save_new(changed, receipt)
    with pytest.raises(AssertionError):
        a.audit_teacher(spec, h.artifact(changed, index=0), "fixture", tmp_path / "reject")


def test_two_world_selection_is_identity_only_and_fresh_namespace():
    selected = [a.replay_indices(i) for i in range(2304)]
    assert all(len(set(row)) == 2 and all(0 <= v < 128 for v in row) for row in selected)
    assert len({tuple(row) for row in selected}) > 1000


@pytest.mark.parametrize("wins", [200, 219, 220, 240])
def test_joint_statistics_independent_and_material_progress_gate(wins):
    rows = [{**s, "score": float(s["pair"] < wins), "margin": -10} for s in h.schedule("main")]
    result = h.statistics(rows)
    a.a.compare(result, a.independent_statistics(rows))
    assert result["progress"]["pass"] == (wins >= 220 and result["progress"]["ci95"][0] > .5)
    assert result["transfer"]["ci95"][1]-result["transfer"]["ci95"][0] > .08


def test_exact_halfpoint_boundary_and_no_material_transfer_gate():
    rows = [{**s, "score": 1. if s["pair"] < 240 and s["seed"] == 162001 and s["seat"] == 0 else .5,
             "margin": 0} for s in h.schedule("main")]
    result = h.statistics(rows)
    assert result["progress"]["score"] == .55 and result["both_pass"]
    next(r for r in rows if r["comparison"] == "progress" and r["score"] == 1.)["score"] = .5
    result = h.statistics(rows)
    assert result["progress"]["ci95"][0] > .5 and not result["progress"]["pass"]
    for row in rows:
        if row["comparison"] == "transfer" and row["pair"] >= 120:
            row["score"] = .5
    result = h.statistics(rows)
    assert result["transfer"]["score"] < .55 and result["transfer"]["pass"]

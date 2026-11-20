import copy
from pathlib import Path

import pytest
import torch

from splendor_ai.alphazero import teacher_budget_161 as h
from scripts.analysis import verify_teacher_budget_161 as a
from splendor_ai.engine.actions import ActionType


@pytest.fixture(scope="module", autouse=True)
def initialize():
    h.old.initialize_worker()


@pytest.mark.parametrize("support", [1, 2, 3, 4])
def test_exact_nested_labels_ties_and_invalid_outcomes(support):
    root = {"candidates": {"actions": list(range(support))}, "choice": 0, "means": [0.]*support,
            "worlds": [{"outcomes": [{"wdl": 0., "reason": "completed"} for _ in range(support)]} for _ in range(128)]}
    assert h.labels(root) == a.independent_labels(root) == {"world128": 0, "world32": 0}
    if support > 1:
        for i, world in enumerate(root["worlds"]):
            world["outcomes"][1]["wdl"] = -1 if i < 32 else 1
        root.update(choice=1, means=[0., .5]+[0.]*(support-2))
        assert h.labels(root) == a.independent_labels(root) == {"world128": 1, "world32": 0}
    root["worlds"][-1]["outcomes"][0]["reason"] = "max_turns"
    for function in (h.labels, a.independent_labels):
        with pytest.raises(AssertionError):
            function(root)


def test_whole_pair_split_and_shared_panels():
    assert h.splits() == a.independent_split()
    assert list(h.splits().values()).count("validation") == 40
    assert list(h.splits().values()).count("train") == 160
    for stage, count in (("probe", 48), ("main", 4800)):
        specs = h.schedule(stage)
        assert len(specs) == count
        assert specs == [a.independent_spec(stage, i) for i in range(count)]
        for pair in range(count//12):
            group = specs[pair*12:(pair+1)*12]
            assert len({s["stream"] for s in group}) == len({s["environment_seed"] for s in group}) == 1
            assert len({(s["comparison"], s["seed"], s["seat"]) for s in group}) == 12
        assert all(h.model_keys(s)[0].startswith("world128_") for s in specs)
    assert {s["environment_seed"] for s in h.schedule("main")}.isdisjoint(s["environment_seed"] for s in h.schedule("probe"))


@pytest.fixture(scope="module")
def source_spec():
    index = 0
    path = h.source.game_path("main", index)
    game = h.old.read_json(path)
    return {"experiment": 160, "index": 0, "pair": "160:0", "path": path.as_posix(),
            "sha256": h.old.file_hash(path), "parent_binding": game["manifest"],
            "roots": [{"turn": r["turn"], "sha256": r["root_sha256"]} for r in game["trace"]
                      if r["actor"] == game["spec"]["candidate_seat"]]}


def test_complete_source_export_reconstruction_and_tamper(tmp_path, source_spec):
    item = h.export_game(source_spec, h.splits()["160:0"], tmp_path)
    report = a.source_game(source_spec, item, tmp_path)
    assert h.old.read_json(Path(report["path"]))["passed"]
    rows = torch.load(item["path"], weights_only=True)
    rows[0]["features"][0, 0] += 1
    path = tmp_path / "tampered.pt"
    torch.save(rows, path)
    item2 = h.artifact(path, index=0, rows=len(rows))
    with pytest.raises(AssertionError):
        a.source_game(source_spec, item2, tmp_path / "rejected")
    with pytest.raises(AssertionError):
        h.export_game(source_spec, h.splits()["160:0"], tmp_path)


def tiny_rows():
    state = h.old.SplendorEnv(seed=1619800000).initial_state()
    root = h.old.read_json(h.source.root_path("main", 0, 0))
    state = h.old.deserialize_state_snapshot(root["snapshot"])
    template = h.f.make_row(state, root, {"experiment": 160, "index": 0, "pair": "160:0"}, "train", h.old._RAW._model.config)
    return [{**copy.deepcopy(template), "id": f"fixture:{i}", "pair": f"fixture:{i//2}",
             "split": "train" if i < 4 else "validation", "teacher": i % len(template["members"])} for i in range(6)]


def test_matched_incremental_fits_complete_bitwise_replay_and_tamper(tmp_path):
    rows = tiny_rows()
    for arm in h.ARMS:
        h.train_fit(arm, 161001, rows, tmp_path)
        a.replay_fit(arm, 161001, rows, tmp_path)
    left, right = [h.old.read_json(tmp_path / "fits" / f"{arm}_161001" / "result.json") for arm in h.ARMS]
    assert left["batch_sequence_sha256"] == right["batch_sequence_sha256"]
    assert left["selected_update"] == right["selected_update"]
    for update in (0, 1, 2, 3, 4):
        tensors = [torch.load(tmp_path / "fits" / f"{arm}_161001" / f"checkpoint_{update:04d}.pt", weights_only=True)["model_state_dict"] for arm in h.ARMS]
        assert all(torch.equal(tensors[0][n], tensors[1][n]) for n in tensors[0])
    folder = tmp_path / "fits" / "world128_161001"
    path = folder / "checkpoint_0001.pt"
    payload = torch.load(path, weights_only=True)
    first = next(iter(payload["model_state_dict"].values()))
    first.flatten()[0] += 1
    torch.save(payload, path)
    with pytest.raises(AssertionError):
        a.replay_fit("world128", 161001, rows, tmp_path)
    with pytest.raises(AssertionError):
        h.train_fit("world128", 161001, rows, h.OUTPUT)


def test_batching_padding_loss_parity_and_unoffered_exclusion():
    rows = tiny_rows()*12
    for epoch in range(4):
        assert h.f.batches(rows, 161001, epoch) == a.a.audit_batches(rows, 161001, epoch)
    for x, y in zip(h.f.collate(rows), a.a.audit_batch_tensors(rows)):
        assert torch.equal(x, y)
    logits = torch.randn((len(rows), rows[0]["features"].shape[0]))
    assert torch.equal(h.f.objective(logits, rows, "treatment")[0], a.a.audit_objective(logits, rows, "treatment")[0])
    assert h.f.checkpoint_updates(65) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_arm_views_change_only_labels_and_initial_checkpoint_wins_ties(tmp_path, monkeypatch):
    rows = tiny_rows()
    for row in rows:
        row["labels"] = {"world128": 1, "world32": 0}
    path = tmp_path / "data.pt"
    torch.save(rows, path)
    h.old.save_new(tmp_path / "export_lock.json", {"files": [h.artifact(path, rows=len(rows))]})
    monkeypatch.setitem(h.CONFIG, "rows", len(rows))
    high = h.load_data("world128", tmp_path)
    low = h.load_data("world32", tmp_path)
    for r, s in zip(high, low):
        assert r["teacher"] == 1 and s["teacher"] == 0
        for field in ("observation", "features", "groups"):
            assert torch.equal(r[field], s[field])
        assert all(torch.equal(x, y) for x, y in zip(r["members"], s["members"]))
    for row in high:
        row["members"] = row["members"][:1]
        row["teacher"] = 0
    h.train_fit("world128", 161001, high, tmp_path)
    a.replay_fit("world128", 161001, high, tmp_path)
    result = h.old.read_json(tmp_path / "fits/world128_161001/result.json")
    assert result["selected_update"] == 0 and result["updates"] == 4


def rich_state():
    env = h.old.SplendorEnv(seed=1619800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(m for m in env.legal_actions(state) if m.action_type == ActionType.RESERVE_DECK and m.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


@pytest.mark.parametrize("comparison", ["transfer", "progress"])
def test_complete_fast_game_privacy_replay_and_action_tamper(tmp_path, comparison):
    spec = {**h.schedule("probe")[0], "comparison": comparison, "environment_seed": 1619800000}
    model = h.f.StrongPolicyInference(h.initializer(161001)["path"]).model
    for key in h.model_keys(spec):
        h.MODELS[key] = model
    state = rich_state()
    chosen, evidence = h.f.choose(state, spec["stream"], model)
    other, expected = a.a.independent_decision(state, spec["stream"], model)
    assert chosen == other
    a.a.compare(evidence, expected)
    item = h.play_game("test", spec, "fixture", tmp_path, h.old.serialize_state_snapshot(state))
    a.replay_game("test", spec, item, "fixture", tmp_path)
    game = h.old.read_json(Path(item["path"]))
    game["trace"][0]["evidence"]["offered_scores"][0] += 1
    path = tmp_path / "tampered.json"
    h.old.save_new(path, game)
    with pytest.raises(AssertionError):
        a.replay_game("test", spec, h.artifact(path), "fixture", tmp_path / "reject")
    with pytest.raises(AssertionError):
        h.play_game("test", spec, "fixture", tmp_path)


def test_failed_game_preserves_evidence_and_rejects_retry(tmp_path):
    spec = h.schedule("probe")[0]
    state = rich_state()
    state.turn_index = 400
    with pytest.raises(AssertionError, match="max_turns"):
        h.play_game("test", spec, "fixture", tmp_path, h.old.serialize_state_snapshot(state))
    game = h.old.read_json(h.game_path("test", 0, tmp_path))
    assert "failure" in game
    with pytest.raises(AssertionError, match="replacement"):
        h.play_game("test", spec, "fixture", tmp_path)


def test_frozen160_protection_for_existing_and_empty_outputs(tmp_path):
    from scripts.analysis import verify_world_budget_160 as audit160
    spec = audit160.independent_spec("main", 0)
    snapshot = h.old.serialize_state_snapshot(rich_state())
    existing = h.source.game_path("main", 0)
    before = h.old.file_hash(existing)
    with pytest.raises(FileExistsError):
        h.source.play_game("main", spec, "fixture", h.source.OUTPUT, snapshot)
    assert h.old.file_hash(existing) == before
    with pytest.raises(ValueError, match="snapshot"):
        h.source.play_game("main", spec, "fixture", tmp_path / "empty", snapshot)
    assert not (tmp_path / "empty").exists()


def statistical_fixture(transfer_wins, progress_wins):
    rows = []
    for s in h.schedule("main"):
        limit = transfer_wins if s["comparison"] == "transfer" else progress_wins
        # Same result across six games in each cluster: uncertainty must not use4800 units.
        rows.append({**s, "score": float(s["pair"] < limit), "margin": -10})
    return rows


@pytest.mark.parametrize("transfer,progress", [(220, 220), (219, 219), (200, 240), (240, 200)])
def test_joint_cluster_bootstrap_exact_boundary_and_both_gates(transfer, progress):
    rows = statistical_fixture(transfer, progress)
    result = h.statistics(rows)
    a.a.compare(result, a.independent_statistics(rows))
    assert result["progress"]["pass"] == (progress >= 220 and result["progress"]["ci95"][0] > .5)
    assert result["transfer"]["pass"] == (result["transfer"]["ci95"][0] > .5)
    assert result["both_pass"] == (result["transfer"]["pass"] and result["progress"]["pass"])
    assert result["transfer"]["ci95"][1]-result["transfer"]["ci95"][0] > .08


def test_no_five_point_transfer_gate():
    rows = []
    for s in h.schedule("main"):
        within = (s["seed"]-161001)*2+s["seat"]
        # Low cluster variance lets transfer pass below55%.
        score = .5 if s["pair"] >= 24 or within >= 3 else 1.
        rows.append({**s, "score": score, "margin": -100})
    result = h.statistics(rows)
    assert result["transfer"]["score"] < .55 and result["transfer"]["pass"]
    assert not result["progress"]["pass"] and not result["both_pass"]


def test_progress_exact55_boundary_without_float_rounding():
    rows = []
    for spec in h.schedule("main"):
        score = 1. if spec["pair"] < 240 and spec["seed"] == 161001 and spec["seat"] == 0 else .5
        rows.append({**spec, "score": score, "margin": -100})
    result = h.statistics(rows)
    assert result["progress"]["score"] == .55 and result["both_pass"]
    row = next(r for r in rows if r["comparison"] == "progress" and r["score"] == 1.)
    row["score"] = .5
    result = h.statistics(rows)
    assert result["progress"]["ci95"][0] > .5 and not result["progress"]["pass"]

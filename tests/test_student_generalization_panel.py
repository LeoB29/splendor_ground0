import copy
from pathlib import Path

import numpy as np
import pytest

from splendor_ai.eval import student_generalization_panel as h
from splendor_ai.eval import run_student_generalization_panel as runner
from scripts.analysis import verify_student_generalization_panel as auditor


@pytest.fixture(scope="module", autouse=True)
def initialize():
    h.old.initialize_worker()
    model = h.student.StrongPolicyInference(h.parent.initializer(162001)["path"]).model
    h.MODELS["anchor_162001"] = model


def rich_state():
    env = h.old.SplendorEnv(seed=2000900000)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


def test_schedule_balances_every_arm_fit_opponent_and_seat():
    main, probe = h.schedule("main"), h.schedule("probe")
    assert len(main) == 5400 and len(probe) == 54
    assert {r["environment_seed"] for r in main}.isdisjoint(r["environment_seed"] for r in probe)
    existing = h.parent.source_schedule()+h.parent.schedule("main")+h.parent.schedule("probe")
    assert {r["environment_seed"] for r in main+probe}.isdisjoint(r["environment_seed"] for r in existing)
    for cluster in range(100):
        block = main[54*cluster:54*(cluster+1)]
        assert len({(r["arm"], r["seed"], r["opponent"], r["seat"]) for r in block}) == 54
        assert len({r["stream"] for r in block}) == len({r["environment_seed"] for r in block}) == 1
        assert {r["cluster"] for r in block} == {cluster}
    with pytest.raises(ValueError):
        h.schedule("source")


@pytest.mark.parametrize("opponent", h.OPPONENTS)
@pytest.mark.parametrize("seat", (0, 1))
def test_game_and_full_independent_replay(tmp_path, opponent, seat):
    spec = {**h.schedule("probe")[0], "opponent": opponent, "seat": seat}
    snapshot = h.old.serialize_state_snapshot(rich_state())
    item = h.play_game("test", spec, "fixture", tmp_path, snapshot)
    report_item = auditor.replay_game("test", spec, item, "fixture", tmp_path)
    report = h.old.read_json(Path(report_item["path"]))
    assert report["full_replay"] and report["game"]["reason"] in h.old.VALID_NATURAL_TERMINATIONS
    with pytest.raises(ValueError, match="already exists"):
        h.play_game("test", spec, "fixture", tmp_path, snapshot)
    game = h.old.read_json(Path(item["path"]))
    game["trace"][0]["state_sha256"] = "tampered"
    path = tmp_path / "tampered.json"
    h.old.save_new(path, game)
    with pytest.raises(ValueError, match="State mismatch"):
        auditor.replay_game("test", spec, h.parent.artifact(path), "fixture", tmp_path / "reject")


def test_invalid_game_preserved_and_no_production_test_override(tmp_path):
    spec = h.schedule("probe")[0]
    state = rich_state()
    state.turn_index = 400
    with pytest.raises(ValueError, match="max_turns"):
        h.play_game("test", spec, "fixture", tmp_path, h.old.serialize_state_snapshot(state))
    record = h.old.read_json(h.game_path("test", 0, tmp_path))
    assert "failure" in record and record["final_state"]["turn_index"] == 400
    with pytest.raises(ValueError):
        h.game_path("test", 0, h.OUTPUT / "fixture")
    with pytest.raises(ValueError, match="snapshot override"):
        h.play_game("probe", spec, "fixture", tmp_path, h.old.serialize_state_snapshot(state))


@pytest.mark.parametrize("opponent", h.OPPONENTS)
def test_complete_game_from_real_initial_state(tmp_path, opponent):
    spec = {**h.schedule("probe")[0], "opponent": opponent}
    item = h.play_game("test", spec, "fixture", tmp_path)
    auditor.replay_game("test", spec, item, "fixture", tmp_path)
    assert len(h.old.read_json(Path(item["path"]))["trace"]) > 10


def rows_fixture():
    # Cluster-level refresh benefits vary; every fit, seat, opponent correlated.
    return [{**s, "score": {"anchor": .5, "refresh": float(s["cluster"] % 4 != 0), "replay": .5}[s["arm"]],
             "reason": "completed"} for s in h.schedule("main")]


def test_joint_cluster_statistics_against_independent_resampling(monkeypatch):
    monkeypatch.setitem(h.CONFIG, "bootstrap_samples", 2000)
    rows = rows_fixture()
    result = h.statistics(list(reversed(rows)))
    rng = np.random.default_rng(h.CONFIG["bootstrap_seed"])
    # Derive cluster changes directly from symbolic fixture, no panel reshape.
    delta = np.array([float(i % 4 != 0)-.5 for i in range(100)])
    sample = delta[rng.integers(100, size=(2000, 100))].mean(axis=1)
    expected_ci = np.percentile(sample, [2.5, 97.5])
    for name in ("refresh_minus_anchor", "refresh_minus_replay"):
        comparison = result["comparisons"][name]
        assert comparison["delta"] == .25
        np.testing.assert_allclose(comparison["ci95"], expected_ci)
        for opponent in h.OPPONENTS:
            assert comparison["per_opponent"][opponent]["delta"] == .25
    assert result["scores"]["refresh"]["wins"] == 1350
    assert result["scores"]["refresh"]["losses"] == 450
    assert result["scores"]["anchor"]["draws"] == 1800
    assert result["promotion"] is False
    assert result["comparisons"]["replay_minus_anchor"]["delta"] == 0.
    assert result["comparisons"]["replay_minus_anchor"]["ci95"] == [0., 0.]
    assert result["primary"] == "refresh_minus_anchor"
    assert result["secondary"] == ["refresh_minus_replay", "replay_minus_anchor"]


def test_three_contrasts_share_clusters_with_nonzero_replay_benefit(monkeypatch):
    monkeypatch.setitem(h.CONFIG, "bootstrap_samples", 2000)
    rows = rows_fixture()
    for row in rows:
        if row["arm"] == "replay":
            row["score"] = float(row["cluster"] % 5 != 0)
    report = h.statistics(rows)
    expected = {"refresh_minus_anchor": np.array([float(i % 4 != 0)-.5 for i in range(100)]),
                "refresh_minus_replay": np.array([float(i % 4 != 0)-float(i % 5 != 0) for i in range(100)]),
                "replay_minus_anchor": np.array([float(i % 5 != 0)-.5 for i in range(100)])}
    indices = np.random.default_rng(h.CONFIG["bootstrap_seed"]).integers(100, size=(2000, 100))
    for name, changes in expected.items():
        result = report["comparisons"][name]
        assert result["delta"] == pytest.approx(changes.mean())
        interval = np.percentile(changes[indices].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose(result["ci95"], interval)
        for opponent in h.OPPONENTS:
            np.testing.assert_allclose(result["per_opponent"][opponent]["ci95"], interval)


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "wrong_seat", "score", "cutoff"))
def test_statistics_reject_corrupt_panels(corruption):
    rows = rows_fixture()
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0]
    elif corruption == "wrong_seat":
        rows[0]["seat"] = 1
    elif corruption == "score":
        rows[0]["score"] = float("nan")
    else:
        rows[0]["reason"] = "max_turns"
    with pytest.raises(ValueError):
        h.statistics(rows)


def test_average_does_not_hide_reported_opponent_regression(monkeypatch):
    monkeypatch.setitem(h.CONFIG, "bootstrap_samples", 1000)
    rows = rows_fixture()
    for row in rows:
        row["score"] = {"035": 1., "shallow": 0., "023": .5}[row["opponent"]] if row["arm"] == "refresh" else .5
    report = h.statistics(rows)
    comparison = report["comparisons"]["refresh_minus_anchor"]
    assert comparison["delta"] == 0.
    assert comparison["per_opponent"]["035"]["ci95"] == [.5, .5]
    assert comparison["per_opponent"]["shallow"]["ci95"] == [-.5, -.5]
    assert comparison["per_opponent"]["023"]["ci95"] == [0., 0.]


@pytest.mark.parametrize("transfer,progress", ((True, True), (True, False), (False, True), (False, False)))
def test_decision_map_preserves_all_four_interpretations(transfer, progress):
    result = {"all_prescribed_audits_pass": True, "statistics": {
        "transfer": {"pass": transfer}, "progress": {"pass": progress}, "both_pass": transfer and progress}}
    assert isinstance(h.decision_map(result), str)
    result["all_prescribed_audits_pass"] = False
    with pytest.raises(ValueError):
        h.decision_map(result)


def test_prepare_blocks_before_writing_when_parent_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(h.parent, "OUTPUT", tmp_path / "missing162")
    with pytest.raises(ValueError, match="must finish"):
        runner.prepare(tmp_path / "checks.json", tmp_path / "panel")
    assert not (tmp_path / "panel").exists()


def test_parent_incomplete_or_wrong_binding_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(h.parent, "OUTPUT", tmp_path)
    h.old.save_new(tmp_path / "final_status.json", {"complete": False, "all_prescribed_audits_pass": True})
    with pytest.raises(ValueError, match="not complete"):
        runner.parent_evidence()


def test_audit_inventory_rejects_duplicates_before_opening_files(tmp_path):
    items = [{"index": i} for i in range(54)]
    items[-1] = items[0]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        runner.verify_audits(items, "probe", "fixture", tmp_path)


def test_failed_worker_cancels_queued_jobs_and_retains_failure(tmp_path, monkeypatch):
    from concurrent.futures import Future
    manifest = {"parent": {"models": {}}}
    monkeypatch.setattr(runner, "validate", lambda base: (manifest, "fixture"))
    monkeypatch.setattr(h, "schedule", lambda stage: [{"index": i} for i in range(3)])
    monkeypatch.setattr(runner.windows, "configure_spawn", lambda: None)
    futures = []
    class FakePool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, *args):
            future = Future()
            if not futures:
                future.set_exception(ValueError("fixture worker failure"))
            futures.append(future)
            return future
    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakePool)
    with pytest.raises(ValueError, match="fixture worker failure"):
        runner.stage_run("probe", tmp_path)
    assert all(f.cancelled() for f in futures[1:])
    assert "fixture worker failure" in h.old.read_json(tmp_path / "probe_failure.json")["error"]
    with pytest.raises(ValueError, match="already attempted"):
        runner.stage_run("probe", tmp_path)


def test_manifest_detects_code_evidence_and_model_changes(tmp_path, monkeypatch):
    dependency = tmp_path / "dependency.txt"
    dependency.write_text("initial")
    monkeypatch.setattr(runner, "identity", lambda: {"code": h.old.file_hash(dependency)})
    checks_path = tmp_path / "checks.json"
    h.old.save_new(checks_path, {"identity": runner.identity(), "exit_code": 0, "logs": []})
    model = tmp_path / "model.pt"
    model.write_text("weights")
    monkeypatch.setattr(runner, "parent_evidence", lambda: {"files": [], "opponent_checkpoints": [],
                                                           "models": {"fixture": h.parent.artifact(model)}})
    base = tmp_path / "panel"
    binding = runner.prepare(checks_path, base)
    assert runner.validate(base)[1] == binding
    model.write_text("corrupt")
    with pytest.raises(ValueError, match="Selected model changed"):
        runner.validate(base)
    dependency.write_text("different code")
    with pytest.raises(ValueError, match="code/runtime changed"):
        runner.validate(base)

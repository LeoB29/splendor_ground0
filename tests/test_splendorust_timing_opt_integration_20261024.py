"""Model-free cross-module resource and clock checks for the bounded candidate."""
from types import SimpleNamespace

import pytest

from tools.splendorust_timing_opt_20261024 import common as c, runner, launch


@pytest.mark.parametrize("mode,klass,base", [
    ("baseline", c.BaselineStudyWorker, c.original.StudyWorker),
    ("candidate", c.CandidateStudyWorker, c.BatchedStudyWorker),
])
def test_both_parent_modes_enforce_aggregate_during_constructor_sampling(monkeypatch, mode, klass, base):
    telemetry = {"rss_bytes": 123, "process_tree": [{"pid": 4}]}
    seen = []
    def initialize(child, *args, **kwargs):
        child._failure = None
        child._failure_detail = None
        # Constructor sampling occurs before the surrounding runner can
        # register this worker in its resident map.
        child.sample_at_initialization = child._sample()
    monkeypatch.setattr(base, "__init__", initialize)
    monkeypatch.setattr(base, "_sample", lambda child: telemetry)
    child = klass(aggregate_sample_guard=lambda child, sample: seen.append((child, sample)))
    assert child.sample_at_initialization is telemetry
    assert seen == [(child, telemetry)]


@pytest.mark.parametrize("klass,base", [
    (c.BaselineStudyWorker, c.original.StudyWorker),
    (c.CandidateStudyWorker, c.BatchedStudyWorker),
])
@pytest.mark.parametrize("prior_failure", [None, "source_identity_failure"])
def test_aggregate_failure_uses_clock_receipt_path_and_preserves_primary(monkeypatch, klass, base, prior_failure):
    child = object.__new__(klass)
    child._failure = prior_failure
    child._failure_detail = "earlier failure" if prior_failure else None
    def fail(child, telemetry):
        raise ValueError("combined worker limit")
    child.aggregate_sample_guard = fail
    monkeypatch.setattr(base, "_sample", lambda child: {"rss_bytes": 123})
    assert child._sample() == {"rss_bytes": 123}
    if prior_failure:
        assert child._failure == prior_failure and child._failure_detail == "earlier failure"
    else:
        assert child._failure == "resource_guard" and "combined worker limit" in child._failure_detail


def test_constructing_worker_is_added_to_existing_resident_usage(monkeypatch):
    envelope = runner.Envelope({"monotonic_origin": 0., "unix_origin": 1000.})
    calls = []
    monkeypatch.setattr(envelope, "aggregate_worker_rss", lambda **kwargs: calls.append(kwargs))
    child = SimpleNamespace(process=SimpleNamespace(pid=42))
    envelope.guard_worker_sample(child, {"rss_bytes": 456})
    assert calls == [{"exclude_pid": 42, "additional_rss": 456}]


def test_late_collection_stage_never_resets_collection_cutoff(monkeypatch):
    envelope = runner.Envelope({"monotonic_origin": 100., "unix_origin": 1000.})
    monkeypatch.setattr(runner.time, "monotonic", lambda: 4550.)
    monkeypatch.setattr(runner.time, "time", lambda: 5450.)
    monkeypatch.setattr(envelope, "guard", lambda: None)
    receipts = []
    monkeypatch.setattr(c, "save", lambda path, value: receipts.append(value))
    envelope.begin("operational64")
    assert envelope.remaining() == 50.
    assert envelope.deadline == 100. + c.COLLECTION_CUTOFF_SECONDS
    assert envelope.wall_deadline == 1000. + c.COLLECTION_CUTOFF_SECONDS
    assert receipts[0]["cap_seconds"] == c.STAGES["operational64"]


def test_final_stage_is_clamped_to_original_global_clocks(monkeypatch):
    envelope = runner.Envelope({"monotonic_origin": 100., "unix_origin": 1000.})
    monkeypatch.setattr(runner.time, "monotonic", lambda: 5450.)
    monkeypatch.setattr(runner.time, "time", lambda: 6350.)
    monkeypatch.setattr(envelope, "guard", lambda: None)
    monkeypatch.setattr(c, "save", lambda path, value: None)
    envelope.begin("final")
    assert envelope.remaining() == 50.
    monkeypatch.setattr(runner.time, "time", lambda: 6401.)
    assert envelope.remaining() == -1.


def test_outer_cleanup_cannot_borrow_time_after_original_global_deadline(monkeypatch):
    monkeypatch.setattr(launch.time, "monotonic", lambda: 5499.)
    monkeypatch.setattr(launch.time, "time", lambda: 6399.)
    result = launch.cleanup_owned(None, None, (), monotonic_origin=100., unix_origin=1000.)
    assert result["effective_allowance_seconds"] == 1.
    assert result["errors"] == []


@pytest.mark.parametrize("challenger_seat", [0, 1])
def test_operational_64_replay_routes_both_players_as_search_with_exact_quota(monkeypatch, challenger_seat):
    # Minimal ledger trajectory isolates caller routing; the actual dictionary
    # ledger and scientific validator are separately covered by frozen tests.
    states = [
        {"state": {"current_player": 0, "terminal": False}},
        {"state": {"current_player": 1, "terminal": False}},
        {"state": {"current_player": 0, "terminal": True, "winner": 0}},
    ]
    spec = dict(stage="ops64", pair_index=0, game_index=0, challenger_seat=challenger_seat,
                challenger_quota=64, opponent_quota=32, challenger_mode="candidate", opponent_mode="candidate")
    rows = []
    for ply in (0, 1):
        role = "challenger" if ply == challenger_seat else "opponent"
        action, trace = {"ply": ply}, {"quota": spec[role+"_quota"]}
        rows.append(dict(stage="ops64", pair_index=0, game_index=0, ply=ply,
                         before=states[ply], after=states[ply+1], seat=ply, game_role=role,
                         quota=spec[role+"_quota"], mode="candidate", arm="search", observer={},
                         completed=True, action=action, trace=trace,
                         clock={"status": "ok", "result": {"action": action, "diagnostics": trace}}))
    ending = dict(spec, plies=2, reason="natural", winner=0, final=states[-1])
    calls = []
    monkeypatch.setattr(runner, "decode_training_snapshot", lambda value: value)
    monkeypatch.setattr(runner.ledger, "check_observer", lambda *args: None)
    monkeypatch.setattr(runner.ledger, "transition", lambda current, action: states[action["ply"]+1])
    monkeypatch.setattr(runner.audit, "check_clock", lambda value: .5)
    monkeypatch.setattr(runner.audit, "validate_result",
                        lambda result, observer, arm, expected_quota: calls.append((arm, expected_quota)))
    assert runner.replay_game(states[0], rows, ending, spec)["status"] == "passed"
    assert calls == [("search", 64 if ply == challenger_seat else 32) for ply in (0, 1)]
    rows[1-challenger_seat]["arm"] = "reference"
    with pytest.raises(ValueError, match="Operational player identity"):
        runner.replay_game(states[0], rows, ending, spec)

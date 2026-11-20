"""Model-free recovery-reader contracts; never inspect or replay real games."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.splendorust_reliability_recovery_v2_20261026 import review


def interruption():
    package = review.ROOT / "tools/splendorust_isolated_reliability_20261026"
    frames = (("runner.py", 763, "run", "coverage=run_stages(envelope,freeze,schedule,qualification_started=True)"),
              ("runner.py", 748, "run_stages", "result=final_checks(envelope,freeze,schedule,games,stop)"),
              ("runner.py", 721, "final_checks", "verify_runtime(freeze,envelope)"),
              ("runner.py", 199, "verify_runtime", "envelope.guard()"),
              ("runner.py", 144, "guard", "c.require(self.remaining() > 0, 'Stage/global real-clock deadline')"),
              ("common.py", 38, "require", "raise ValueError(message)"))
    error = "Traceback (most recent call last):\n" + "".join(
        f'  File "{package / file}", line {line}, in {function}\n    {code}\n'
        for file, line, function, code in frames) + "ValueError: Stage/global real-clock deadline\n"
    complete = dict(status="incomplete", stage="final", coverage=None, automatic_retries=0,
                    freeze_sha256=review.EXPECTED_FREEZE, failure=error)
    execution = dict(status="runner_failed", exit_code=1, freeze_sha256=review.EXPECTED_FREEZE,
        cleanup_errors=[], cleanup_receipt=dict(errors=[], unconfirmed=[]), charged_job_seconds=11317.83,
        monotonic_end=1900.45, unix_end=5900.45)
    failure = dict(stage="final", preserved_partial_evidence=True, partial_evidence_not_accepted=True, error=error)
    final = dict(cap_seconds=1800, monotonic=100., monotonic_deadline=1900., unix_time=4100., unix_deadline=5900.)
    markers = dict(coverage=False, verification=False, final_end=False, collection_stop=False)
    return complete, execution, failure, final, markers


def test_only_known_incomplete_attempt_can_be_recovered():
    result = review.validate_original_interruption(*interruption())
    assert result["original_status"] == "incomplete"
    assert result["original_exit_code"] == 1
    assert result["original_status_preserved"] is True
    assert result["original_final_input_loop_completed_index_unknown"] is True


def test_actual_saved_initial_fixture_uses_available_corrected_rule_interface():
    # This one initial-world fixture catches the wrong rules-module alias. It
    # does not inspect moves, play, load a model or run the recovery audit.
    saved = json.loads((review.OUT / "games/operational/p000_g0/initial.json").read_text(encoding="utf-8"))
    assert review.reconstruct_initial(saved["seed_decimal"]) == saved["initial"]


@pytest.mark.parametrize("target,key,value", [
    (0, "status", "completed_verified"), (0, "stage", "main"), (0, "coverage", {}),
    (0, "automatic_retries", 1), (0, "freeze_sha256", "0" * 64),
    (1, "exit_code", 0), (1, "status", "watchdog_deadline"),
    (1, "cleanup_errors", ["owner not confirmed"]), (1, "charged_job_seconds", 21600.),
    (2, "partial_evidence_not_accepted", False), (2, "stage", "qualification"),
    (3, "cap_seconds", 3600), (3, "monotonic_deadline", 1901),
    (4, "coverage", True), (4, "verification", True), (4, "final_end", True), (4, "collection_stop", True),
])
def test_wrong_status_budget_or_existing_final_marker_rejected(target, key, value):
    args = list(interruption())
    args[target][key] = value
    with pytest.raises(ValueError):
        review.validate_original_interruption(*args)


@pytest.mark.parametrize("before,after", [
    ("line 199, in verify_runtime", "line 720, in final_checks"),
    ("Stage/global real-clock deadline\n", "Pinned sampler runtime changed\n"),
    ("in verify_runtime", "in play_game"),
])
def test_other_failure_locus_is_not_relabelled_as_recoverable(before, after):
    args = list(interruption())
    changed = args[0]["failure"].replace(before, after)
    args[0]["failure"] = args[2]["error"] = changed
    with pytest.raises(ValueError):
        review.validate_original_interruption(*args)


def test_unconfirmed_original_owner_rejected():
    args = list(interruption())
    args[1]["cleanup_receipt"]["unconfirmed"] = [dict(pid=3, created=4)]
    with pytest.raises(ValueError, match="cleanup incomplete"):
        review.validate_original_interruption(*args)


def reproduction():
    return dict(completed=True, before={"state": [1, 2]}, observer={"visible": [3]},
                action={"type": "take"}, trace={"callbacks": [1, 2, 3]}, after={"state": [2, 3]},
                clock=dict(deadline_seconds=1., result={"action": {"type": "take"}, "full_bits": [42]}))


def test_saved_reproduction_checks_full_science_not_action_alone():
    original = reproduction()
    review.validate_saved_reproduction(original, deepcopy(original))
    repeated = deepcopy(original)
    repeated["clock"]["result"]["full_bits"][0] = 43
    with pytest.raises(ValueError, match="full science differs"):
        review.validate_saved_reproduction(original, repeated)


@pytest.mark.parametrize("key", ("before", "observer", "action", "trace", "after"))
def test_any_saved_reproduction_scientific_difference_fails(key):
    original, repeated = reproduction(), reproduction()
    repeated[key] = {"mutated": True}
    with pytest.raises(ValueError, match="full science differs"):
        review.validate_saved_reproduction(original, repeated)


def test_failed_or_short_clock_never_qualifies_as_saved_reproduction():
    for change in ("failed", "short"):
        original, repeated = reproduction(), reproduction()
        if change == "failed":
            repeated["completed"] = False
        else:
            repeated["clock"]["deadline_seconds"] = .99
        with pytest.raises(ValueError, match="Incomplete or shortened"):
            review.validate_saved_reproduction(original, repeated)


def candidate(game, sha, *, private="own"):
    return dict(stage="main", pair_index=game // 2, game_index=game, ply=2, player_role="challenger",
                phase="0..19", seat=0, private_category=private, selection_sha256=sha)


def test_streamed_selector_keeps_global_minimum_and_distinct_cells():
    winners, originals = {}, {}
    first, better, worse, other = candidate(0, "b"), candidate(2, "a"), candidate(4, "c"), candidate(6, "d", private="both")
    for selected in (first, better, worse, other):
        row = {key: selected[key] for key in review.IDENTITY_KEYS}
        review.merge_replay_cells(winners, originals, [selected], [row])
    key = tuple(better[k] for k in review.CELL_KEYS)
    assert len(winners) == 2 and winners[key] == better
    assert originals[key]["game_index"] == 2


def test_selector_cannot_invent_source_row():
    with pytest.raises(ValueError, match="invented"):
        review.merge_replay_cells({}, {}, [candidate(0, "a")], [])


def test_receipt_inventory_rejects_missing_duplicate_and_failed_proofs():
    valid = dict(status="passed", workers=2, receipts=[dict(worker="a"), dict(worker="b")])
    assert set(review.receipt_inventory(valid, {"a", "b"})) == {"a", "b"}
    for values in ([dict(worker="a")], [dict(worker="a"), dict(worker="a")],
                   [dict(worker="a"), dict(worker="unowned")]):
        changed = {**valid, "receipts": values}
        with pytest.raises(ValueError, match="inventory"):
            review.receipt_inventory(changed, {"a", "b"})
    with pytest.raises(ValueError, match="inventory"):
        review.receipt_inventory({**valid, "status": "failed"}, {"a", "b"})


def streams_fixture():
    folder = review.OUT / "workers/fixture"
    decoded = [dict(name=name, status="verified", codec="gzip", path=name + ".gz", raw_bytes=2,
                    raw_sha256="a" * 64, compressed_bytes=3, compressed_sha256="b" * 64)
               for name in ("stdout.log", "stderr.log", "events.jsonl")]
    cleanup = dict(log_compression={item["name"]: {k: v for k, v in item.items() if k != "name"} for item in decoded})
    log = dict(streams=3, decoded_streams=decoded, raw_bytes=6, compressed_bytes=9)
    raw = dict(stdout_log=deepcopy(cleanup["log_compression"]["stdout.log"]))
    sealed = {}
    for item in decoded:
        path = (folder / item["path"]).relative_to(review.ROOT).as_posix()
        sealed[path] = dict(path=path, bytes=3, sha256="b" * 64)
    return folder, cleanup, log, raw, sealed


def test_original_log_and_raw_proofs_bind_exact_sealed_bytes():
    args = list(streams_fixture())
    review.bind_closed_stream_receipts(*args)
    next(iter(args[4].values()))["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="sealed compressed bytes"):
        review.bind_closed_stream_receipts(*args)


def test_different_raw_packet_stdout_binding_is_rejected():
    args = list(streams_fixture())
    args[3]["stdout_log"]["raw_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="same exact stdout"):
        review.bind_closed_stream_receipts(*args)


def test_graph_receipt_is_bound_to_saved_identity_and_successful_request():
    digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
    identity = dict(weights_before_sha256="a" * 64, weights_after_sha256="a" * 64, backend="graph")
    receipt = dict(status="passed", fixtures=11, graph_identity_sha256=digest(identity), weights_sha256="a" * 64)
    summary = dict(graph_identity_sha256=digest(identity), full_proof="saved")
    clock = dict(status="ok", result=summary, request_id="request-00000001")
    review.bind_graph_receipt(receipt, identity, clock, deepcopy(clock), summary, digest)
    changed = deepcopy(identity)
    changed["backend"] = "changed"
    with pytest.raises(ValueError, match="identity/weights"):
        review.bind_graph_receipt(receipt, changed, clock, clock, summary, digest)
    with pytest.raises(ValueError, match="successful raw request"):
        review.bind_graph_receipt(receipt, identity, clock, {**clock, "status": "timeout"}, summary, digest)


def temp_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    out = tmp_path / "outputs/original"
    out.mkdir(parents=True)
    monkeypatch.setattr(review, "OUT", out)
    path = out / "summary.json"
    data = b'{"status":"passed"}'
    path.write_bytes(data)
    item = dict(path=path.relative_to(tmp_path).as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    freeze = dict(original_snapshot=[item], inputs=[])
    return review.Review(10, boundary=lambda: None, original_freeze={}, recovery_freeze=freeze), path


def test_summary_outside_completed_seal_still_bound_by_whole_snapshot(tmp_path, monkeypatch):
    reader, path = temp_reader(tmp_path, monkeypatch)
    assert reader.read(path) == {"status": "passed"}
    path.write_text('{"status":"failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="recovery snapshot"):
        reader.read(path)


def test_extra_unsealed_file_rejected_before_review(tmp_path, monkeypatch):
    reader, path = temp_reader(tmp_path, monkeypatch)
    (path.parent / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="directory inventory"):
        reader.run()


def test_parent_clock_failure_is_never_swallowed(tmp_path, monkeypatch):
    reader, _ = temp_reader(tmp_path, monkeypatch)
    def expired():
        raise TimeoutError("global UTC limit")
    reader.boundary = expired
    with pytest.raises(TimeoutError, match="global UTC"):
        reader.guard()


def test_no_additional900second_component_cap(monkeypatch):
    seen = {}
    class Envelope:
        def guard(self):
            pass
        def remaining(self):
            return 1400.
    class Reader:
        def __init__(self, seconds, **kwargs):
            seen["seconds"] = seconds
            seen.update(kwargs)
        def run(self):
            return dict(status="passed_saved_evidence_review", no_overall_recovered_acceptance_claim=True)
    monkeypatch.setattr(review, "Review", Reader)
    result = review.run_review(Envelope(), {}, {})
    assert seen["seconds"] == 1400.
    assert result["status"] == "passed_saved_evidence_review"
    assert result["no_overall_recovered_acceptance_claim"] is True

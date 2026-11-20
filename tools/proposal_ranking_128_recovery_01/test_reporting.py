"""Status locks and unavailable logs cannot abort scientific computation."""
import io
import json
from pathlib import Path

from tools.proposal_ranking_128_recovery_01 import reporting as p


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_unique_immutable_events_and_sessions_preserve_existing_logs(tmp_path, capsys):
    (tmp_path/"failure.json").write_text("original failure evidence", encoding="utf-8")
    (tmp_path/"stderr.txt").write_text("original stderr", encoding="utf-8")
    metadata = {"manifest":"binding", "nested":{"source":"original"}}
    first = p.StatusReporter(tmp_path, metadata)
    metadata["nested"]["source"] = "mutated outside"
    rows = [first.event("fit_complete", index=1), first.event("source_complete", index=2)]
    second = p.StatusReporter(tmp_path, {"manifest":"binding"})
    rows.append(second.event("probe_complete", index=3))
    events = sorted((tmp_path/"status_events").glob("*.json"))
    assert len(events) == 3 and [read(path) for path in events] == rows
    assert rows[0]["nested"]["source"] == "original"
    assert rows[0]["status_session"] == rows[1]["status_session"] != rows[2]["status_session"]
    assert [row["status_sequence"] for row in rows] == [1,2,1]
    assert len({row["status_event_id"] for row in rows}) == 3
    assert read(tmp_path/"progress.json") == rows[-1]
    assert not list(tmp_path.glob(".progress_*.tmp"))
    assert not first.last_errors and not second.last_errors
    assert (tmp_path/"failure.json").read_text(encoding="utf-8") == "original failure evidence"
    assert (tmp_path/"stderr.txt").read_text(encoding="utf-8") == "original stderr"
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_transient_windows_replace_lock_has_bounded_retry(tmp_path, monkeypatch):
    original = p.os.replace
    attempts, delays = [], []
    def locked(source, target):
        attempts.append((source,target))
        if len(attempts) < 3:
            raise PermissionError(13,"locked by reader")
        return original(source,target)
    monkeypatch.setattr(p.os,"replace",locked)
    monkeypatch.setattr(p.time,"sleep",delays.append)
    reporter = p.StatusReporter(tmp_path)
    row = reporter.event("collection", completed=123)
    assert len(attempts) == 3 and delays == [.025,.050] and sum(delays) < .2
    assert read(tmp_path/"progress.json") == row
    assert not reporter.last_errors
    assert read(next((tmp_path/"status_events").glob("*.json"))) == row
    assert not list(tmp_path.glob(".progress_*.tmp"))


def test_permanent_replace_lock_keeps_durable_event_old_progress_and_unique_temps(tmp_path, monkeypatch, capsys):
    (tmp_path/"progress.json").write_text('{"stage":"old"}', encoding="utf-8")
    attempts,delays = [],[]
    def locked(source,target):
        attempts.append((source,target))
        raise PermissionError(13,"permanent reader lock")
    monkeypatch.setattr(p.os,"replace",locked)
    monkeypatch.setattr(p.time,"sleep",delays.append)
    reporter = p.StatusReporter(tmp_path)
    rows = [reporter.event("collected", completed=1), reporter.event("verified", completed=2)]
    assert len(attempts) == 6 and len(delays) == 4 and sum(delays) < .2
    assert read(tmp_path/"progress.json") == {"stage":"old"}
    assert len(list((tmp_path/"status_events").glob("*.json"))) == 2
    temps = list(tmp_path.glob(".progress_*.tmp"))
    assert len(temps) == 2 and {read(path)["status_event_id"] for path in temps} == {r["status_event_id"] for r in rows}
    assert reporter.last_errors[0]["operation"] == "progress_replace" and reporter.last_errors[0]["attempts"] == 3
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(row["status_io_errors"][0]["error"] == "PermissionError" for row in output)


def test_non_permission_replace_error_is_not_retried(tmp_path, monkeypatch):
    calls = []
    def failed(source,target):
        calls.append(source)
        raise OSError(28,"disk full")
    monkeypatch.setattr(p.os,"replace",failed)
    monkeypatch.setattr(p.time,"sleep",lambda seconds: (_ for _ in ()).throw(AssertionError("unexpected retry")))
    reporter = p.StatusReporter(tmp_path)
    row = reporter.event("source_complete")
    assert len(calls) == 1 and reporter.last_errors[0]["attempts"] == 1
    assert read(next((tmp_path/"status_events").glob("*.json"))) == row


def test_failed_event_file_still_updates_advisory_progress_and_stdout(tmp_path, monkeypatch, capsys):
    original = Path.open
    def failed(path,*args,**kwargs):
        if path.parent.name == "status_events":
            raise PermissionError(13,"event directory unavailable")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",failed)
    reporter = p.StatusReporter(tmp_path)
    row = reporter.event("fit_audit_complete")
    assert read(tmp_path/"progress.json") == row
    assert reporter.last_errors[0]["operation"] == "immutable_event"
    assert json.loads(capsys.readouterr().out)["stage"] == "fit_audit_complete"


def test_failed_fsync_preserves_partial_evidence_and_continues(tmp_path, monkeypatch):
    def failed(fd):
        raise OSError(5,"failed event sync")
    monkeypatch.setattr(p.os,"fsync",failed)
    reporter = p.StatusReporter(tmp_path)
    row = reporter.event("audit")
    assert reporter.last_errors[0]["operation"] == "immutable_event"
    assert read(next((tmp_path/"status_events").glob("*.json"))) == row
    assert read(tmp_path/"progress.json") == row


class BrokenStream:
    def write(self,value):
        raise OSError(32,"status stream broken")
    def flush(self):
        raise AssertionError("write should fail first")


def test_stdout_failure_falls_back_to_stderr(tmp_path, monkeypatch):
    fallback = io.StringIO()
    monkeypatch.setattr(p.sys,"stdout",BrokenStream())
    monkeypatch.setattr(p.sys,"stderr",fallback)
    reporter = p.StatusReporter(tmp_path)
    row = reporter.event("main_complete")
    assert read(tmp_path/"progress.json") == row
    assert reporter.last_errors[0]["operation"] == "stdout"
    logged = json.loads(fallback.getvalue())
    assert logged["stage"] == "main_complete" and logged["status_io_errors"][0]["operation"] == "stdout"


def test_all_status_channels_unavailable_do_not_abort_caller(tmp_path, monkeypatch):
    reporter = p.StatusReporter(tmp_path)
    def denied(*args,**kwargs):
        raise PermissionError(13,"status storage unavailable")
    monkeypatch.setattr(Path,"mkdir",denied)
    monkeypatch.setattr(p.sys,"stdout",BrokenStream())
    monkeypatch.setattr(p.sys,"stderr",BrokenStream())
    progress = []
    progress.append("scientific work before")
    row = reporter.event("source_collection", completed=17)
    progress.append("scientific work after")
    assert progress == ["scientific work before","scientific work after"]
    assert row["completed"] == 17
    assert [error["operation"] for error in reporter.last_errors] == ["immutable_event","progress_temporary","stdout","stderr"]


def test_entropy_failure_reinitializes_without_reusing_identity(tmp_path, monkeypatch):
    def denied():
        raise OSError("entropy unavailable")
    monkeypatch.setattr(p.uuid,"uuid4",denied)
    monkeypatch.setattr(p.time,"time_ns",lambda:123)
    first,second = p.StatusReporter(tmp_path),p.StatusReporter(tmp_path)
    a,b = first.event("first"),second.event("second")
    assert a["status_event_id"] != b["status_event_id"]
    assert len(list((tmp_path/"status_events").glob("*.json"))) == 2

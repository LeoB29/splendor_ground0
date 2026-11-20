"""Source separation, exact targets and real frozen-input checks; no new play."""
import copy
from collections import Counter
import pytest
import torch

from . import common as c, source as s


@pytest.fixture(scope="module")
def parent():
    return s.closed_parent()


def teacher_root(sums):
    n = len(sums)
    # Integers -128..128 are realized with wins/draws/losses in 128 worlds.
    worlds = [{"outcomes": [{"wdl": (1 if total > 0 else -1) if i < abs(total) else 0,
                              "reason": "completed"} for total in sums]} for i in range(128)]
    return {"candidates": {"keys": [f"candidate-{i}" for i in range(n)],
                           "actions": list(range(n))}, "worlds": worlds,
            "choice": max(range(n), key=sums.__getitem__), "means": [v/128 for v in sums]}


def fixture_row(spec, n=8):
    values = [i-3 for i in range(n)]
    target = s.old_targets(teacher_root(values))
    return {"observation": torch.arange(5, dtype=torch.float32),
            "features": torch.arange(4*n, dtype=torch.float32).reshape(n, 4),
            "groups": torch.arange(n, dtype=torch.long), "group_count": n,
            "members": [torch.tensor([i]) for i in range(n)],
            "id": spec["source_rowid"], "pair": f"breadth128:{spec['pair']}",
            "split": spec["split"], "seat": spec["seat"], "phase": spec["phase"],
            "privacy": spec["privacy"], "teacher": n-1, "narrow_support": min(4, n),
            "narrow_teacher": min(4, n)-1, "index": spec["index"],
            "source_fit": spec["fit"], **target}


@pytest.mark.parametrize("n", range(1, 9))
def test_all_supports_and_outsider_absence(n):
    sums = list(range(-n, 0))
    target = s.old_targets(teacher_root(sums))
    assert target["old_sums_wdl"] == sums
    assert target["outsider_support"] == max(0, n-3)
    assert target["outsider_teacher"] == (n-1 if n > 3 else -1)
    assert target["outsider_label"] == (n-4 if n > 3 else -1)


def test_original_candidate_order_breaks_ties():
    value = s.old_targets(teacher_root([10, 3, 9, 8, 8, 7, 8, -1]))
    assert value["outsider_teacher"] == 3 and value["outsider_label"] == 0


@pytest.mark.parametrize("change", ["unknown", "invalid", "short_world", "missing_world", "mean", "choice", "duplicate"])
def test_invalid_historical_targets_fail_closed(change):
    root = teacher_root([5, 7, -4, 1, 3, 12, 4, 10])
    if change == "unknown": root["worlds"][0]["outcomes"][0].update(wdl=None, reason="repetition_cutoff")
    if change == "invalid": root["worlds"][0]["outcomes"][0]["wdl"] = 2
    if change == "short_world": root["worlds"][0]["outcomes"].pop()
    if change == "missing_world": root["worlds"].pop()
    if change == "mean": root["means"][0] = 1
    if change == "choice": root["choice"] = 0
    if change == "duplicate": root["candidates"]["keys"][-1] = root["candidates"]["keys"][0]
    with pytest.raises(ValueError): s.old_targets(root)


def test_real_parent_exact_completion_models_and_partition(parent):
    assert parent["source_binding"] == c.SOURCE_BINDING
    assert parent["result"]["sha256"] == s.RESULT_SHA and parent["audit"]["sha256"] == s.AUDIT_SHA
    assert s.validate_schedule(parent["schedule"])
    assert len(parent["models"]) == 9 and len(parent["evaluation_sources"]) == 336
    assert parent["inventory"] == {r["path"]: r["sha256"] for r in parent["evidence"]}
    assert parent["new_simulations"] == parent["historical_fit_replays"] == 0
    assert parent["assessment_used_for_selection"] is False


@pytest.mark.parametrize("change", ["missing", "duplicate", "cross_split", "phase", "source_fit", "order"])
def test_schedule_changes_rejected(parent, change):
    schedule = copy.deepcopy(parent["schedule"])
    if change == "missing": schedule.pop()
    if change == "duplicate": schedule[1] = schedule[0]
    if change == "cross_split": schedule[0]["split"] = "validation"
    if change == "phase": schedule[0]["phase"] = (schedule[0]["phase"]+1) % 3
    if change == "source_fit": schedule[0]["fit"] = 999
    if change == "order": schedule.reverse()
    with pytest.raises(ValueError): s.validate_schedule(schedule)


@pytest.mark.parametrize("change", ["assessment", "path", "wrong_target", "noninteger", "nan", "member", "group", "duplicate_group"])
def test_training_allowlist_and_canonical_membership(parent, change):
    row = fixture_row(parent["schedule"][0])
    assert s.validate_row(row)
    if change == "assessment": row["assessment"] = [1]*128
    if change == "path": row["source_root"] = "an-assessment-bearing-root.json"
    if change == "wrong_target": row["outsider_teacher"] = 3
    if change == "noninteger": row["old_sums_wdl"][3] = 0.5
    if change == "nan": row["features"][0, 0] = float("nan")
    if change == "member": row["members"][0] = torch.tensor([99])
    if change == "group": row["group_count"] = 999
    if change == "duplicate_group": row["members"][1] = row["members"][0]
    with pytest.raises(ValueError): s.validate_row(row)


def test_prepare_separates_partitions_and_train_read_never_loads_other_data(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    monkeypatch.setattr(s, "reconstruct_row", lambda spec, config: fixture_row(spec, 1+spec["index"] % 8))
    ref = s.prepare("fixture-binding", tmp_path)
    lock = c.verified(ref)
    assert lock["files"]["train"]["path"] != lock["files"]["validation"]["path"]
    assert lock["counts"]["train"]["rows"] == lock["counts"]["validation"]["rows"] == 168
    original = torch.load
    opened = []
    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        assert str(path).endswith("train.pt"), "Optimizer read a non-training artifact"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", guarded)
    rows = s.load_rows("fixture-binding", "train", tmp_path)
    assert len(opened) == 1 and len(rows) == 168
    assert all(set(row) == s.ROW_FIELDS and row["split"] == "train" for row in rows)
    assert Counter(row["source_fit"] for row in rows) == {fit: 56 for fit in c.FITS}
    assert all("assessment" not in field and "source_root" not in field for row in rows for field in row)
    with pytest.raises(FileExistsError): s.prepare("fixture-binding", tmp_path)


def test_payload_hash_and_binding_changes_rejected(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    monkeypatch.setattr(s, "reconstruct_row", lambda spec, config: fixture_row(spec))
    s.prepare("fixture-binding", tmp_path)
    with pytest.raises(ValueError): s.load_rows("other-binding", "train", tmp_path)
    with pytest.raises(ValueError): s.load_rows("fixture-binding", "test", tmp_path)
    with (tmp_path/"train.pt").open("ab") as stream: stream.write(b"corruption")
    with pytest.raises(ValueError, match="Changed artifact"):
        s.load_rows("fixture-binding", "train", tmp_path)


def test_old_inputs_read_only_pre_assessment_artifacts(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    monkeypatch.setattr(s, "reconstruct_row", lambda spec, config: fixture_row(spec))
    s.prepare("fixture-binding", tmp_path)
    read = c.verified
    opened = []
    def guarded(ref):
        opened.append(ref["path"])
        assert "/main/selections/" in ref["path"], "Prediction input read a fresh outcome artifact"
        return read(ref)
    monkeypatch.setattr(c, "verified", guarded)
    values = s.old_inputs("fixture-binding", tmp_path)
    assert len(values) == len(opened) == 336
    assert [value["index"] for value in values] == list(range(336))
    assert all(not any(value["old_unknown_counts"]) for value in values)
    assert all("assessment" not in value for value in values)
    assert values[0]["old_choices"]["teacher_outsider"] == max(range(3, len(values[0]["old_sums_wdl"])),
                                                               key=values[0]["old_sums_wdl"].__getitem__)


def test_real_all_336_old_inputs_reconstruct_without_inference_or_play(parent, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("New inference/trajectory is forbidden in source preparation")
    monkeypatch.setattr(c.old, "trajectory", forbidden)
    from splendor_ai.training.model import LegalActionAttentionScorerModel
    monkeypatch.setattr(LegalActionAttentionScorerModel, "forward", forbidden)
    config = s.LegalActionAttentionScorerConfig(**parent["model_config"])
    rows = [s.reconstruct_row(spec, config) for spec in parent["schedule"]]
    assert len(rows) == 336 and all(set(row) == s.ROW_FIELDS for row in rows)
    assert sum(row["outsider_support"] == 0 for row in rows) == 17
    assert Counter(row["split"] for row in rows) == {"train": 168, "validation": 168}
    # Independent old-selection receipts agree with the targets reconstructed
    # from the original teacher-game worlds and lawful exported observations.
    for row, descriptor in zip(rows, parent["evaluation_sources"], strict=True):
        saved = c.verified(descriptor["selection"])
        stats = saved["selection_lock"]["selection_statistics"]
        assert row["old_sums_wdl"] == stats["sums_wdl"]
        target = saved["selection_lock"]["choices"]["teacher_outsider"]
        assert row["outsider_teacher"] == (-1 if target is None else target)


def test_independent_tensor_mismatch_is_detected(parent, monkeypatch):
    spec = parent["schedule"][0]
    actual = s.inherited_source._export_row(spec)
    changed = copy.deepcopy(actual)
    changed["features"][0, 0] += 1
    monkeypatch.setattr(s.inherited_source, "_export_row", lambda spec: changed)
    with pytest.raises(ValueError, match="independent lawful"):
        s.reconstruct_row(spec, s.LegalActionAttentionScorerConfig(**parent["model_config"]))

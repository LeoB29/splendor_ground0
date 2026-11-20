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


def fixture_corpus(parent):
    """Actual original identities, synthetic lawful-shaped payloads and old labels."""
    panels = {spec["source_rowid"]: spec for spec in parent["schedule"]}
    targets = {entry["spec"]["source_rowid"]:
               c.verified(entry["selection"])["selection_lock"]["selection_statistics"]["sums_wdl"]
               for entry in parent["evaluation_sources"]}
    rows = []
    for shard in parent["corpus"]["shards"]:
        game = shard["spec"]
        for root in shard["roots"]:
            rowid = f"breadth128:{game['index']}:{root['turn']}"
            spec = {"index": len(rows), "source_rowid": rowid, "pair": game["pair"],
                    "fit": game["fit"], "split": parent["corpus"]["splits"][str(game["pair"])],
                    "seat": game["candidate_seat"], "phase": min(root["turn"]//20, 2),
                    "privacy": panels.get(rowid, {}).get("privacy", {"own": 0, "opponent": 0})}
            sums = targets.get(rowid, list(range(8)))
            row = fixture_row(spec, len(sums))
            row.update(s.old_targets(teacher_root(sums)), teacher=max(range(len(sums)), key=sums.__getitem__),
                       narrow_teacher=max(range(min(4, len(sums))), key=sums.__getitem__))
            rows.append(row)
    return rows


@pytest.fixture(scope="module")
def synthetic_corpus(parent):
    return fixture_corpus(parent)


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
    assert len(parent["models"]) == 12 and len(parent["evaluation_sources"]) == 336
    assert len(parent["corpus"]["shards"]) == 402
    assert sum(len(shard["roots"]) for shard in parent["corpus"]["shards"]) == 11950
    assert Counter(parent["corpus"]["splits"].values()) == {"train": 159, "validation": 42}
    assert all(parent["models"][f"specialist_{fit}"]["sha256"] == s.SPECIALIST_MODELS[fit] for fit in c.FITS)
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


def test_prepare_separates_partitions_and_train_read_never_loads_other_data(parent, synthetic_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    monkeypatch.setattr(s, "corpus_rows", lambda parent: synthetic_corpus)
    ref = s.prepare("fixture-binding", tmp_path)
    lock = c.verified(ref)
    assert lock["files"]["train"]["path"] != lock["files"]["validation"]["path"]
    assert lock["counts"]["train"]["rows"] == 9461 and lock["counts"]["validation"]["rows"] == 2489
    assert lock["counts"]["train"]["pairs"] == 159 and lock["counts"]["validation"]["pairs"] == 42
    original = torch.load
    opened = []
    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        assert str(path).endswith("train.pt"), "Optimizer read a non-training artifact"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", guarded)
    rows = s.load_rows("fixture-binding", "train", tmp_path)
    assert len(opened) == 1 and len(rows) == 9461
    assert all(set(row) == s.ROW_FIELDS and row["split"] == "train" for row in rows)
    assert len(s.train_probe_ids("fixture-binding", tmp_path)) == 168
    assert set(s.train_probe_ids("fixture-binding", tmp_path)).issubset({row["id"] for row in rows})
    assert len(opened) == 1  # Metadata-only original TRAIN probe accessor.
    assert all("assessment" not in field and "source_root" not in field for row in rows for field in row)
    with pytest.raises(FileExistsError): s.prepare("fixture-binding", tmp_path)


def test_payload_hash_and_binding_changes_rejected(parent, synthetic_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    monkeypatch.setattr(s, "corpus_rows", lambda parent: synthetic_corpus)
    s.prepare("fixture-binding", tmp_path)
    with pytest.raises(ValueError): s.load_rows("other-binding", "train", tmp_path)
    with pytest.raises(ValueError): s.load_rows("fixture-binding", "test", tmp_path)
    with (tmp_path/"train.pt").open("ab") as stream: stream.write(b"corruption")
    with pytest.raises(ValueError, match="Changed artifact"):
        s.load_rows("fixture-binding", "train", tmp_path)


def test_old_inputs_read_only_pre_assessment_artifacts(parent, synthetic_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    monkeypatch.setattr(s, "corpus_rows", lambda parent: synthetic_corpus)
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


def test_real_full_corpus_conversion_without_inference_or_play(parent, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("New inference/trajectory is forbidden in source preparation")
    monkeypatch.setattr(c.old, "trajectory", forbidden)
    from splendor_ai.training.model import LegalActionAttentionScorerModel
    monkeypatch.setattr(LegalActionAttentionScorerModel, "forward", forbidden)
    rows = s.corpus_rows(parent)
    assert len(rows) == 11950 and all(set(row) == s.ROW_FIELDS for row in rows)
    assert sum(row["outsider_support"] == 0 for row in rows) == 583
    assert Counter(row["split"] for row in rows) == {"train": 9461, "validation": 2489}
    assert all(not any("assessment" in key or "source_root" in key for key in row) for row in rows)
    panel = s.panel_rows(rows, parent)
    assert Counter(row["split"] for row in panel) == {"train": 168, "validation": 168}
    assert sum(row["outsider_support"] == 0 for row in panel) == 17
    # Independent old-selection receipts agree with the targets reconstructed
    # from the original teacher-game worlds and lawful exported observations.
    for row, descriptor in zip(panel, parent["evaluation_sources"], strict=True):
        saved = c.verified(descriptor["selection"])
        stats = saved["selection_lock"]["selection_statistics"]
        assert row["old_sums_wdl"] == stats["sums_wdl"]
        target = saved["selection_lock"]["choices"]["teacher_outsider"]
        assert row["outsider_teacher"] == (-1 if target is None else target)


@pytest.mark.parametrize("change", ["split", "order", "id", "source_fit", "pair", "missing"])
def test_full_corpus_ancestry_tampering_rejected(parent, synthetic_corpus, change):
    rows = list(synthetic_corpus)
    rows[0] = dict(rows[0])
    if change == "split": rows[0]["split"] = "validation" if rows[0]["split"] == "train" else "train"
    if change == "order": rows[0], rows[1] = rows[1], rows[0]
    if change == "id": rows[0]["id"] += ":bad"
    if change == "source_fit": rows[0]["source_fit"] = 161003
    if change == "pair": rows[0]["pair"] = "breadth128:999"
    if change == "missing": rows.pop()
    with pytest.raises(ValueError): s.validate_corpus(rows, parent)


def test_panel_is_exact_projection_and_controls_are_sealed(parent, synthetic_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    monkeypatch.setattr(s, "corpus_rows", lambda parent: synthetic_corpus)
    s.prepare("fixture-binding", tmp_path)
    panel = s.load_panel_rows("fixture-binding", tmp_path)
    assert [row["index"] for row in panel] == list(range(336))
    assert [row["id"] for row in panel] == [spec["source_rowid"] for spec in parent["schedule"]]
    models = s.control_models("fixture-binding", tmp_path)
    assert set(models) == {f"{family}_{fit}" for family in ("anchor", "specialist") for fit in c.FITS}
    assert all(models[f"specialist_{fit}"]["selected_update"] == 1784 for fit in c.FITS)
    with (tmp_path/"panel.pt").open("ab") as stream: stream.write(b"corruption")
    with pytest.raises(ValueError, match="Changed artifact"):
        s.load_panel_rows("fixture-binding", tmp_path)


def test_independent_tensor_mismatch_is_detected(parent, monkeypatch):
    spec = parent["schedule"][0]
    actual = s.inherited_source._export_row(spec)
    changed = copy.deepcopy(actual)
    changed["features"][0, 0] += 1
    monkeypatch.setattr(s.inherited_source, "_export_row", lambda spec: changed)
    with pytest.raises(ValueError, match="independent lawful"):
        s.reconstruct_row(spec, s.LegalActionAttentionScorerConfig(**parent["model_config"]))


def test_target_disagreement_with_independent_specialist_export_is_rejected(parent, monkeypatch):
    real_load = torch.load
    def changed(path, *args, **kwargs):
        rows = real_load(path, *args, **kwargs)
        if "proposal_specialist_128" in str(path) and str(path).endswith("game_0000.pt"):
            rows[0]["outsider_teacher"] = -123
        return rows
    monkeypatch.setattr(torch, "load", changed)
    with pytest.raises(ValueError, match="independently audited original specialist target"):
        s.corpus_rows(parent)

"""New-game independence, immutable splits and common128 label conversion."""
import copy
import json
import time
import uuid
from collections import Counter

import pytest
import torch

from . import common as c, training_source as s


@pytest.fixture(scope="module")
def parent():
    start = time.perf_counter()
    value = s.closed_parent()
    print(f"closed_parent_seconds={time.perf_counter()-start:.3f}")
    return value


@pytest.fixture(scope="module")
def prepared(parent):
    # Preserve this concrete preflight export. It is separate from production
    # and never trains a model or replays an old game/continuation.
    base = c.ROOT/"outputs/independent_game_coverage_128_recovery_01_checks"/f"source_fixture_{uuid.uuid4().hex}"
    start = time.perf_counter()
    ref = s.prepare("source-fixture", base, parent=parent)
    print(f"added720_export_seconds={time.perf_counter()-start:.3f}; fixture={base}")
    return base, c.verified(ref)


def fixture_copy(prepared, tmp_path):
    _, lock = prepared
    value = copy.deepcopy(lock)
    (tmp_path/"data_lock.json").write_text(json.dumps(value), encoding="utf-8")
    return value


def test_completed_sources_controls_and_split_ancestry(parent):
    assert len(parent["added_sources"]) == 720
    assert set(parent["models"]) == {f"{arm}_{f}" for arm in ("anchor", "lowrate", "soft") for f in c.FITS}
    assert parent["soft"]["result"]["sha256"] == s.SOFT_RESULT
    assert parent["soft"]["audit"]["sha256"] == s.SOFT_AUDIT
    assert parent["inventory"] == {r["path"]: r["sha256"] for r in parent["evidence"]}
    assert parent["dependencies"] and all(not path.startswith("outputs/") for path in parent["dependencies"])
    proof = parent["training_disjointness"]
    assert proof["historical_games"] == 1204 and proof["added_distinct_games"] == 720
    assert len(proof["old_split_environment_seeds"]["train"]) == 159
    assert len(proof["old_split_environment_seeds"]["validation"]) == 42
    assert set(proof["added_environment_seeds"]).isdisjoint(proof["historical_environment_seeds"])
    assert proof["prospective_source_seed_disjoint"]
    assert parent["retired_evaluation_panels"]["permanent"] is True
    assert len(set(parent["retired_evaluation_panels"]["ids"])) == 720
    assert parent["historical_fit_replays"] == parent["historical_trajectory_replays"] == 0
    for fit in c.FITS:
        assert parent["models"][f"soft_{fit}"]["sha256"] == s.SOFT_MODELS[fit]
        assert parent["models"][f"anchor_{fit}"]["sha256"] == s.inherited.inherited_models.EXPECTED[f"anchor_{fit}"]
    for descriptor in parent["added_sources"]:
        assert descriptor["source_spec"] == s.PANEL_MODULES[descriptor["source_panel"]].independent_spec("main", descriptor["spec"]["game_index"])
        assert descriptor["selection"]["sha256"] == parent["inventory"][descriptor["selection"]["path"]]


def test_real_720_conversion_all_cells_and_original_tensor_reuse(parent, prepared):
    base, lock = prepared
    original = c.verified(parent["reused_data"])
    assert lock["files"]["old"] == original["files"]["train"]
    assert lock["files"]["validation"] == original["files"]["validation"]
    assert lock["row_digests"]["old"] == original["row_digests"]["train"]
    assert lock["row_digests"]["validation"] == original["row_digests"]["validation"]
    rows = s.load_rows("source-fixture", "added", base)
    assert len(rows) == len({r["environment_seed"] for r in rows}) == 720
    assert Counter(tuple(r["cell"]) for r in rows) == {
        (p, f, seat, phase): 20 for p in s.PANELS for f in c.FITS for seat in (0, 1) for phase in range(3)}
    assert all(set(r) == s.inherited.ROW_FIELDS | s.EXTRA_FIELDS for r in rows)
    assert all(r["split"] == "train" for r in rows)
    assert [r["index"] for r in rows] == list(range(11950, 12670))
    assert lock["retired_evaluation_panels"]["ids"] == [r["id"] for r in rows]
    for row, descriptor in zip(rows, parent["added_sources"], strict=True):
        saved = c.verified(descriptor["selection"])
        matrix = [[o["wdl"] for o in w["outcomes"]] for w in saved["selection"]["worlds"]]
        assert row["old_sums_wdl"] == [int(sum(column)) for column in zip(*matrix, strict=True)]
        assert all(o in (-1., 0., 1.) for values in matrix for o in values)
    assert s.validate("source-fixture", base, authenticate_parent=False) == lock
    assert len(s.control_models("source-fixture", base)) == 9
    assert s.initializer(c.FITS[0], base) == parent["models"][f"anchor_{c.FITS[0]}"]


def test_real_original_train_validation_separately_without_other_tensor_reads(prepared, monkeypatch):
    base, lock = prepared
    real_load = torch.load
    for split, expected in (("old", 9461), ("validation", 2489)):
        seen = []
        def load(path, *args, **kwargs):
            seen.append(str(path))
            assert str(path) == str(c.ROOT/lock["files"][split]["path"])
            return real_load(path, *args, **kwargs)
        monkeypatch.setattr(torch, "load", load)
        rows = s.load_rows("source-fixture", split, base)
        assert len(rows) == expected and len(seen) == 1
        assert all(r["split"] == ("train" if split == "old" else "validation") for r in rows)
        del rows


def test_partition_three_disjoint_equal_stratum_subsets_and_fourth_block(prepared):
    base, lock = prepared
    identities = lock["added_identities"]
    allocation = lock["partition"]
    seen = set()
    for fit in c.FITS:
        indices = s.narrow_indices(fit, "source-fixture", base)
        assert len(indices) == 180 and seen.isdisjoint(indices)
        seen.update(indices)
        assert Counter(tuple(identities[i]["cell"]) for i in indices) == {
            (p, f, seat, phase): 5 for p in s.PANELS for f in c.FITS for seat in (0, 1) for phase in range(3)}
    assert len(seen) == 540 and len(allocation["blocks"][3]) == 180
    assert sorted(seen | set(allocation["blocks"][3])) == list(range(720))
    # Independently calculate row ranks using ONLY cell and identity.
    for cell in {tuple(r["cell"]) for r in identities}:
        ranked = sorted((i for i, r in enumerate(identities) if tuple(r["cell"]) == cell),
                        key=lambda i: (c.old.digest([c.CONFIG["partition_tag"], list(cell), identities[i]["id"]]), identities[i]["id"]))
        for b in range(4):
            assert set(ranked[b*5:(b+1)*5]) <= set(allocation["blocks"][b])
    changed = [{**r, "old_sums_wdl": [i-360]} for i, r in enumerate(identities)]
    assert s.partition(changed) == allocation


@pytest.mark.parametrize("damage", ["duplicate", "cell", "population"])
def test_partition_rejects_wrong_identity_or_strata(parent, damage):
    identities = [s.row_identity(d) for d in parent["added_sources"]]
    if damage == "duplicate": identities[0]["id"] = identities[1]["id"]
    if damage == "cell": identities[0]["cell"][1] = 0
    if damage == "population": identities.pop()
    with pytest.raises(ValueError): s.partition(identities)


def fake_selection(n, values):
    descriptor = {"source_panel": s.PANELS[0], "spec": {"index": 0}, "binding": "fixture"}
    identity = {"spec": descriptor["spec"], "union": {"keys": [str(i) for i in range(n)]},
                "candidates": {"breadth": {"keys": [str(i) for i in range(n)], "actions": [{}]*n, "indices": list(range(n))},
                               "anchor": {"keys": [str(i) for i in range(min(n, 4))]}}}
    selection = {"worlds": [{"outcomes": [{"wdl": x, "reason": "completed"} for x in values]} for _ in range(128)],
                 "world_seeds": list(range(128)), "candidate_indices": list(range(n))}
    lock = {"identity_digest": c.old.digest(identity), "selection_sha256": c.old.digest(selection),
            "value_scale": "terminal_wdl", "means": {"breadth": list(values)},
            "choices": {"breadth": max(range(n), key=values.__getitem__)}}
    return {"manifest": "fixture", "stage": "main", "index": 0, "identity": identity,
            "selection": selection, "selection_lock": lock}, descriptor


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_target_scale_ties_and_small_support(n):
    saved, descriptor = fake_selection(n, [0]*n)
    target = s.selection_targets(saved, descriptor)
    assert target["old_sums_wdl"] == [0]*n
    assert target["outsider_support"] == max(0, n-3)
    assert target["outsider_teacher"] == (3 if n > 3 else -1)
    assert target["outsider_label"] == (0 if n > 3 else -1)
    saved, descriptor = fake_selection(8, [-1, 0, 1, -1, 0, 1, 1, -1])
    target = s.selection_targets(saved, descriptor)
    assert target["old_sums_wdl"] == [-128, 0, 128, -128, 0, 128, 128, -128]
    assert target["teacher"] == 2 and target["outsider_teacher"] == 5 and target["outsider_label"] == 2


@pytest.mark.parametrize("damage", ["none", "fraction", "bool", "reason", "missing", "means", "choice", "seed"])
def test_target_rejects_censoring_wrong_means_or_world_support(damage):
    saved, descriptor = fake_selection(8, [0]*8)
    block = saved["selection"]
    if damage == "none": block["worlds"][0]["outcomes"][7]["wdl"] = None
    if damage == "fraction": block["worlds"][0]["outcomes"][7]["wdl"] = .5
    if damage == "bool": block["worlds"][0]["outcomes"][7]["wdl"] = True
    if damage == "reason": block["worlds"][0]["outcomes"][7]["reason"] = "truncated"
    if damage == "missing": block["worlds"][0]["outcomes"].pop()
    if damage == "means": saved["selection_lock"]["means"]["breadth"][7] = .01
    if damage == "choice": saved["selection_lock"]["choices"]["breadth"] = 1
    if damage == "seed": block["world_seeds"][0] = 1
    saved["selection_lock"]["selection_sha256"] = c.old.digest(block)
    with pytest.raises(ValueError): s.selection_targets(saved, descriptor)


def test_selection_targets_ignore_old_assessment_and_precision_extension():
    saved, descriptor = fake_selection(8, [0]*8)
    expected = s.selection_targets(saved, descriptor)
    saved["assessment"] = {"worlds": "forbidden junk"}
    saved["extension"] = {"worlds": "forbidden junk"}
    assert s.selection_targets(saved, descriptor) == expected


@pytest.mark.parametrize("damage", ["assessment", "split", "membership", "target", "index", "cell"])
def test_added_strict_payload_and_member_target_firewall(prepared, damage):
    base, _ = prepared
    row = copy.deepcopy(torch.load(base/"added.pt", map_location="cpu", weights_only=True)[0])
    if damage == "assessment": row["assessment"] = [1]
    if damage == "split": row["split"] = "validation"
    if damage == "membership": row["members"][0] = row["members"][1]
    if damage == "target": row["outsider_teacher"] += 1
    if damage == "index": row["index"] = 0
    if damage == "cell": row["cell"][1] = 0
    with pytest.raises(ValueError): s.validate_added_row(row)


@pytest.mark.parametrize("damage", ["split_file", "partition", "retirement", "ancestry"])
def test_data_seal_rejects_tampering(prepared, tmp_path, damage):
    lock = fixture_copy(prepared, tmp_path)
    if damage == "split_file": lock["files"]["old"] = lock["files"]["validation"]
    if damage == "partition": lock["partition"]["narrow_indices"]["161001"][0] = 719
    if damage == "retirement": lock["retired_evaluation_panels"]["permanent"] = False
    if damage == "ancestry": lock["parent"]["added_sources"][0]["source_spec"]["environment_seed"] = 0
    (tmp_path/"data_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError): s.validate("source-fixture", tmp_path, authenticate_parent=False)


def test_tensor_file_tamper_rejected(prepared, tmp_path):
    lock = fixture_copy(prepared, tmp_path)
    path = tmp_path/"changed.pt"
    path.write_bytes((c.ROOT/lock["files"]["added"]["path"]).read_bytes() + b"changed")
    lock["files"]["added"]["path"] = str(path)
    (tmp_path/"data_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError): s.load_rows("source-fixture", "added", tmp_path)


def test_all_ancestral_games_and_fresh_namespaces_disjoint(parent, monkeypatch):
    historical = []
    for family, count in (("world_budget_160", 400), ("learned_proposal_128", 402), ("proposal_breadth_arena_128", 402)):
        for i in range(count):
            historical.append({"family": family, "environment_seed": i})
    corpus = {"shards": [{"spec": {"environment_seed": i, "pair": i}} for i in range(201)],
              "splits": {str(i): "train" if i < 159 else "validation" for i in range(201)}}
    assert s.disjointness(parent["added_sources"], historical, corpus)["added_distinct_games"] == 720
    descriptors = copy.deepcopy(parent["added_sources"])
    descriptors[0]["source_spec"]["environment_seed"] = 0
    with pytest.raises(ValueError, match="overlap"): s.disjointness(descriptors, historical, corpus)
    config = copy.deepcopy(c.CONFIG)
    config["source_seed"]["main"] = 0
    monkeypatch.setattr(c, "CONFIG", config)
    with pytest.raises(ValueError, match="Prospective"): s.disjointness(parent["added_sources"], historical, corpus)


def test_prepare_is_one_shot_and_metadata_apis_never_load_tensors(prepared, monkeypatch):
    base, lock = prepared
    def forbidden(*args, **kwargs): raise AssertionError("No tensors or new play allowed")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(c.old, "trajectory", forbidden)
    assert len(s.narrow_indices(c.FITS[0], "source-fixture", base)) == 180
    assert len(s.control_models("source-fixture", base)) == 9
    with pytest.raises(ValueError, match="One-shot"):
        s.prepare("source-fixture", base, parent=lock["parent"])


def test_original_row_digest_includes_order_and_all_score_magnitudes(prepared):
    base, _ = prepared
    rows = torch.load(base/"added.pt", map_location="cpu", weights_only=True)[:2]
    before = s.row_digest(rows)
    assert s.row_digest(rows[::-1]) != before
    rows[0]["old_sums_wdl"][0] -= 1
    assert s.row_digest(rows) != before

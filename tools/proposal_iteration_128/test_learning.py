"""Bounded fixtures for extraction, controlled dose and exact fit reproduction."""
import copy
from collections import Counter

import pytest
import torch

from tools.proposal_iteration_128 import learning as h


@pytest.fixture(scope="module", autouse=True)
def initialize():
    h.old.initialize_worker()


def source_example():
    spec = h.source.specs("main")[0]
    root = h.old.read_json(h.source.root_path("main", 0, 0))
    state = h.old.deserialize_state_snapshot(root["snapshot"])
    row = h.f.make_row(state, root, {"experiment": "learned128", "index": 0, "pair": "learned128:0"},
                       h.splits()["learned128:0"], h.old._RAW._model.config)
    return spec, root, state, row


def tiny_rows():
    template = source_example()[3]
    return [{**copy.deepcopy(template), "id": f"{'160' if i < 3 else 'learned128'}:0:{i}",
             "pair": f"{'160' if i < 3 else 'learned128'}:{i//2}",
             "split": "train" if i < 5 else "validation", "teacher": i % len(template["members"])}
            for i in range(7)]


def test_whole_source_pairs_stratified_by_fit_and_retained_old_split():
    assert h.splits() == h.independent_splits()
    assert {k: v for k, v in h.splits().items() if k.startswith("160:")} == h.previous.splits()
    assert Counter(v for k, v in h.splits().items() if k.startswith("learned128:")) == {"train": 159, "validation": 42}
    for fit in h.FITS:
        specs = [s for s in h.source.specs("main") if s["fit"] == fit]
        assert Counter(h.splits()[f"learned128:{s['pair']}"] for s in specs) == {"train": 106, "validation": 28}
        assert len({s["pair"] for s in specs}) == 67
        for pair in {s["pair"] for s in specs}:
            assert {s["candidate_seat"] for s in specs if s["pair"] == pair} == {0, 1}


def test_independent_new_label_inputs_and_corruption_rejection():
    spec, root, state, row = source_example()
    assert h.inspect_row(row, state, root, spec, h.old._RAW._model.config)["rows"] == 1
    for field in ("teacher", "split", "pair", "members", "features"):
        changed = copy.deepcopy(row)
        if field == "teacher":
            changed[field] = (changed[field]+1) % len(changed["members"])
        elif field == "members":
            changed[field] = list(reversed(changed[field]))
        elif field == "features":
            changed[field].flatten()[0] += 1
        else:
            changed[field] = "corrupt"
        with pytest.raises(AssertionError):
            h.inspect_row(changed, state, root, spec, h.old._RAW._model.config)
    bad_root = copy.deepcopy(root)
    bad_root["worlds"][0]["outcomes"][0]["wdl"] = .25
    with pytest.raises(AssertionError):
        h.inspect_row(row, state, bad_root, spec, h.old._RAW._model.config)


def test_control_dose_all_original_slots_preserved_and_labels_do_not_select_replacements():
    training = [r for r in tiny_rows() if r["split"] == "train"]
    for seed in h.TRAIN_SEEDS.values():
        for epoch in range(4):
            plan = h.batch_plan(training, seed, epoch)
            assert plan == h.independent_plan(training, seed, epoch)
            slots = [i for chunk, _ in plan for i in chunk]
            replay = [i for _, chunk in plan for i in chunk]
            assert sorted(slots) == list(range(len(training)))
            assert len(slots) == len(replay)
            assert all(training[i]["id"].startswith("160:") for i in replay)
            assert all(a == b for a, b in zip(slots, replay) if training[a]["id"].startswith("160:"))
            changed = [{**r, "teacher": 999} for r in training]
            assert h.batch_plan(changed, seed, epoch) == plan
    with pytest.raises(AssertionError):
        h.batch_plan(tiny_rows(), 207001, 0)


def test_unoffered_legal_actions_are_not_negative_targets():
    row = source_example()[3]
    row = copy.deepcopy(row)
    row["members"] = row["members"][:2]
    row["teacher"] = 0
    logits = torch.zeros((1, len(row["features"])))
    included = set(int(i) for member in row["members"] for i in member)
    excluded = [i for i in range(logits.shape[1]) if i not in included]
    assert excluded
    before = h.f.objective(logits, [row], "treatment")[0]
    logits[0, excluded] = 1000
    after = h.f.objective(logits, [row], "treatment")[0]
    assert torch.equal(before, after)
    assert torch.equal(after, h.independent.audit_objective(logits, [row], "treatment")[0])


def test_both_arms_independent_bitwise_reproduction_and_checkpoint_tamper(tmp_path):
    rows = tiny_rows()
    for arm in h.ARMS:
        h.fit_job(161001, arm, "fixture", tmp_path, rows=rows)
        h.audit_fit_job(161001, arm, "fixture", tmp_path, rows=rows)
    results = [h.old.read_json(tmp_path / "fits" / f"{arm}_161001" / "result.json") for arm in h.ARMS]
    assert results[0]["slot_sequence_sha256"] == results[1]["slot_sequence_sha256"]
    assert results[0]["batch_sequence_sha256"] != results[1]["batch_sequence_sha256"]
    assert results[0]["updates"] == results[1]["updates"] == 4
    assert results[0]["row_exposures"] == results[1]["row_exposures"] == 20
    path = tmp_path / "fits/treatment_161001/checkpoint_0001.pt"
    payload = torch.load(path, weights_only=True)
    next(iter(payload["model_state_dict"].values())).flatten()[0] += 1
    torch.save(payload, path)
    with pytest.raises(AssertionError):
        h.audit_fit_job(161001, "treatment", "fixture", tmp_path, rows=rows)
    with pytest.raises(AssertionError):
        h.fit_job(161001, "treatment", "fixture", h.OUTPUT, rows=rows)
    with pytest.raises(AssertionError):
        h.fit_job(161001, "treatment", "fixture", h.OUTPUT / "fixture", rows=rows)


def test_initial_checkpoint_eligible_earliest_tie_and_tensor_identity(tmp_path):
    rows = tiny_rows()
    for row in rows:
        row["members"] = row["members"][:1]
        row["teacher"] = 0
    h.fit_job(161001, "treatment", "fixture", tmp_path, rows=rows)
    h.audit_fit_job(161001, "treatment", "fixture", tmp_path, rows=rows)
    result = h.old.read_json(tmp_path / "fits/treatment_161001/result.json")
    audit = h.old.read_json(tmp_path / "fits/treatment_161001/audit.json")
    assert result["selected_update"] == 0 and audit["selected_equals_initializer"]
    assert result["curves"][0]["checkpoint_sha256"] != h.initializer(161001)["sha256"]


def test_new_export_from_audited_source_actual_game_and_tensors(tmp_path):
    game_lock = h.old.read_json(h.source.OUTPUT / "main_games_lock.json")
    audit_lock = h.old.read_json(h.source.OUTPUT / "main_audits_lock.json")
    item = {"game": game_lock["files"][0], "audit": audit_lock["files"][0],
            "spec": h.source.specs("main")[0], "parent_binding": game_lock["manifest"]}
    exported, audit_item = h.export_game(item, "fixture", tmp_path)
    audit = h._json(audit_item)
    assert audit["passed"] and audit["actual_game_replayed"] and audit["every_row_hidden_input_invariance"]
    assert audit["continuations_replayed"] == 0
    assert audit["coverage"]["rows"] == exported["rows"] == 31
    rows = torch.load(h._path(exported), weights_only=True)
    assert {r["pair"] for r in rows} == {"learned128:0"}
    assert {r["seat"] for r in rows} == {0}
    with pytest.raises(AssertionError):
        h.export_game(item, "fixture", tmp_path)

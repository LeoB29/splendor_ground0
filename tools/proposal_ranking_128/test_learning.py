"""Bounded fixtures; no production preparation, fitting, or continuation search."""
import copy
import math

import pytest
import torch

from . import learning as h
from . import learning_data as d


@pytest.fixture(scope="module", autouse=True)
def initialize():
    h.old.initialize_worker()


def root_fixture(support=4):
    outcomes = [{"wdl": value, "reason": "completed"} for value in (1, -1, 0, 1)[:support]]
    totals = [outcome["wdl"] for outcome in outcomes]
    return {"candidates": {"actions": list(range(support)), "keys": [str(i) for i in range(support)]},
            "worlds": [{"outcomes": copy.deepcopy(outcomes)} for _ in range(128)],
            "means": totals, "choice": totals.index(max(totals))}


def tiny_rows():
    lock = h.old.read_json(h.SOURCE / "data_lock.json")
    templates = torch.load(h.c.verify_hash(lock["old_exports"][0]), weights_only=True, map_location="cpu")
    template = next(row for row in templates if len(row["members"]) == 4)
    return [{**copy.deepcopy(template), "id": f"{'160' if i < 3 else 'learned128'}:0:{i}",
             "pair": f"{'160' if i < 3 else 'learned128'}:{i//2}",
             "split": "train" if i < 5 else "validation", "teacher": i % 4,
             "ranking_delta": [0., .25, -.5, .125, -.125, .25, -.5][i]}
            for i in range(7)]


def test_recorded_score_delta_independence_short_support_and_corruption():
    root = root_fixture()
    assert d.delta(root) == d.independent_delta(root) == -.5
    root["worlds"][0]["outcomes"][2]["wdl"] = 1
    root["means"][2] = 1/128
    assert d.delta(root) == d.independent_delta(root) == -127/256
    assert d.delta(root_fixture(3)) == d.independent_delta(root_fixture(3)) == 0
    broken = copy.deepcopy(root)
    broken["means"][2] += 1/128
    with pytest.raises(AssertionError):
        d.delta(broken)
    with pytest.raises(AssertionError):
        d.independent_delta(broken)
    broken = root_fixture()
    broken["worlds"][0]["outcomes"][0]["wdl"] = .25
    with pytest.raises(AssertionError):
        d.independent_delta(broken)


def test_loss_uses_only_outsider_group_max_sign_magnitude_and_zero_rows():
    rows = tiny_rows()
    logits = torch.arange(len(rows[0]["features"]), dtype=torch.float32)[None].repeat(len(rows), 1).requires_grad_()
    first = h.objective(logits, rows, .25)
    other = h.audit_objective(logits, rows, .25)
    assert torch.equal(first, other)
    a = torch.autograd.grad(first.sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad(other.sum(), logits)[0]
    assert torch.equal(a, b) and torch.equal(a[0], torch.zeros_like(a[0]))
    outsider = set(int(i) for group in rows[0]["members"][2:4] for i in group)
    unused = [i for i in range(logits.shape[1]) if i not in outsider]
    changed = logits.detach().clone()
    changed[:, unused] = 10000
    assert torch.equal(first.detach(), h.objective(changed, rows, .25))
    assert torch.equal(h.objective(logits.detach(), rows, .5), first.detach()/2)
    short = copy.deepcopy(rows[1])
    short["members"] = short["members"][:3]
    short["ranking_delta"] = 0.
    assert h.objective(logits[:1], [short], .25).item() == 0
    short["ranking_delta"] = .25
    with pytest.raises(AssertionError):
        h.objective(logits[:1], [short], .25)


def test_pair_weighting_and_train_only_normalizer_includes_zeros(tmp_path):
    records = [{"pair": "a", "loss": 0.}, {"pair": "b", "loss": 2.}, {"pair": "b", "loss": 2.}]
    assert h.pair_loss(records) == 1.
    rows = tiny_rows()
    _, normalizer = h._inputs("fixture", tmp_path, rows)
    assert normalizer == 1/5
    for row in rows[5:]:
        row["ranking_delta"] = 1.
    assert h._inputs("fixture", tmp_path, rows)[1] == normalizer
    with pytest.raises(AssertionError):
        h._inputs("fixture", h.OUTPUT / "fixture", rows)


@pytest.mark.parametrize("kind", ["old", "new"])
def test_one_actual_saved_source_shard_targets_and_parent_tensors(tmp_path, kind):
    lock = h.old.read_json(h.SOURCE / "data_lock.json")
    audit = h.c.verified(lock["audit"])
    item, checked = lock[f"{kind}_exports"][0], audit[f"{kind}_export_audits"][0]
    exported, receipt = d.export_shard(kind, item, checked, "fixture", tmp_path)
    rows = torch.load(h.c.verify_hash(exported), weights_only=True, map_location="cpu")
    evidence = h.c.verified(receipt)
    assert evidence["passed"] and evidence["independent_targets"]
    assert evidence["new_continuation_simulations"] == 0
    assert len(rows) == item["rows"] == evidence["coverage"]["rows"]
    assert all(row["ranking_delta"] == d.independent_delta(h.c.verified(root))
               for row, root in zip(rows, evidence["source_roots"], strict=True))
    with pytest.raises(AssertionError):
        d.export_shard(kind, item, checked, "fixture", tmp_path)


@pytest.mark.parametrize("validation_tied", [False, True])
def test_full_tiny_fit_reproduction_saved_ce_reselection_initial_tie_and_tamper(tmp_path, validation_tied):
    rows = tiny_rows()
    if validation_tied:
        for row in rows[5:]:
            row["ranking_delta"] = 0.
    parent = tmp_path / "parent"
    target = tmp_path / "ranking"
    ce_rows = [{key: value for key, value in row.items() if key != "ranking_delta"} for row in rows]
    h.previous.fit_job(161001, "treatment", "fixture-parent", parent, rows=ce_rows)
    h.previous.audit_fit_job(161001, "treatment", "fixture-parent", parent, rows=ce_rows)
    result_item = h.fit(161001, "fixture", target, rows=rows, ce_base=parent)
    audit_item = h.audit_fit(161001, result_item, "fixture", target, rows=rows, ce_base=parent)
    result, audit = h.c.verified(result_item), h.c.verified(audit_item)
    assert result["updates"] == 4 and result["row_exposures"] == 20
    if validation_tied:
        assert result["selected_updates"] == {"ranking": 0, "ce": 0}
        assert audit["unchanged_from_anchor"] == {"ranking": True, "ce": True}
    else:
        for arm in ("ranking", "ce"):
            assert all(curve[f"{arm}_pair_loss"] > 0 for curve in result["curves"])
            chosen = min(result["curves"], key=lambda curve: curve[f"{arm}_pair_loss"])
            assert chosen["update"] == result["selected_updates"][arm] == audit["selected_updates"][arm]
    assert audit["complete_fit_bitwise_identical"] and audit["ce_selection_independently_reproduced"]
    assert result["curves"][0]["ranking_checkpoint"]["sha256"] != result["initializer"]["sha256"]
    original = h.c.verify_hash(result["curves"][1]["ce_checkpoint"])
    payload = torch.load(original, weights_only=True, map_location="cpu")
    next(iter(payload["model_state_dict"].values())).flatten()[0] += 1
    torch.save(payload, original)
    with pytest.raises((AssertionError, ValueError)):
        h.audit_fit(161001, result_item, "fixture", target, rows=rows, ce_base=parent)

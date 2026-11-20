"""Isolated continuation, cumulative resource and exact completed-root reuse checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from tools.control_variate_128 import common as c, experiment as parent
from . import recovery as r


def _disk(monkeypatch, free=10*1024**3):
    monkeypatch.setattr(r.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))


def _launch(base, epoch=1000., previous=600.):
    c.old.save_new(base/"launch.json", {"epoch_seconds": epoch, "previous_elapsed_seconds": previous})


def test_active_budget_counts_both_prior_runs_and_excludes_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUTPUT", tmp_path)
    _disk(monkeypatch); _launch(tmp_path, epoch=1_000_000., previous=600.)
    monkeypatch.setattr(r.time, "time", lambda: 1_007_001.)
    r.budget_boundary()
    assert r.MAXIMUM_RUNTIME_SECONDS == 10800
    monkeypatch.setattr(r.time, "time", lambda: 1_010_200.)
    r.budget_boundary()
    monkeypatch.setattr(r.time, "time", lambda: 1_010_200.001)
    with pytest.raises(c.BudgetExceeded):
        r.budget_boundary()


def test_budget_production_requires_launch_but_fixture_can_omit_it(tmp_path, monkeypatch):
    production, fixture = tmp_path/"production", tmp_path/"fixture"
    production.mkdir(); fixture.mkdir()
    monkeypatch.setattr(r, "OUTPUT", production); _disk(monkeypatch)
    with pytest.raises(ValueError, match="launch"):
        r.budget_boundary()
    r.budget_boundary(fixture)


def test_cooperative_abort_precedes_remaining_budget_and_preserves_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUTPUT", tmp_path); _disk(monkeypatch)
    _launch(tmp_path); monkeypatch.setattr(r.time, "time", lambda: 1001.)
    c.old.save_new(tmp_path/"abort.json", {"error": "fixture stop"})
    ref = c.artifact(tmp_path/"abort.json")
    with pytest.raises(c.BudgetExceeded):
        r.budget_boundary()
    c.verify_hash(ref)


def test_five_gib_disk_reserve_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OUTPUT", tmp_path); _launch(tmp_path)
    monkeypatch.setattr(r.time, "time", lambda: 1001.)
    _disk(monkeypatch, 5*1024**3); r.budget_boundary()
    _disk(monkeypatch, 5*1024**3-1)
    with pytest.raises(c.BudgetExceeded, match="disk"):
        r.budget_boundary()


def test_configuration_installs_second_recovery_hooks_without_scientific_changes(tmp_path, monkeypatch):
    before, original_output = copy.deepcopy(c.CONFIG), c.OUTPUT
    for module, key in ((parent, "OUTPUT"), (parent, "validate_freeze"),
                        (parent, "worker_initialize"), (c, "budget_boundary")):
        monkeypatch.setattr(module, key, getattr(module, key))
    r.configure(tmp_path)
    assert parent.OUTPUT == tmp_path
    assert parent.validate_freeze is r.validate_freeze
    assert parent.worker_initialize is r.worker_initialize
    assert c.budget_boundary is r.budget_boundary
    assert c.CONFIG == before and c.OUTPUT == original_output
    assert c.CONFIG["maximum_runtime_hours"] == 2.
    assert r.MAXIMUM_RUNTIME_SECONDS == 10800


def _spec(index):
    return {"index": index, "pair": index//4, "fit": 161001+index//56,
            "split": "train", "game_index": index//2, "phase": index%3,
            "seat": index%2, "support": 1+index%8, "reference": 0,
            "cv_stress": False, "cv_audit_indices": [0, 4, 8, 16, 32, 64, 96, 127]}


def _record(spec):
    support = spec["support"]
    identity = {key: spec[key] for key in
                ("index", "pair", "fit", "split", "game_index", "phase", "seat", "support", "reference")}
    y = [[float((wi+ci)%3)/2 for ci in range(support)] for wi in range(128)]
    z = {str(h): [[float((wi+ci+h)%9)/8 for ci in range(support)] for wi in range(128)]
         for h in (4, 8, 16)}
    worlds = [{"index": wi, "seed": 2211000000+spec["index"]*512+wi,
               "state_sha256": "a"*64,
               "candidates": [{str(h): {"horizon": h, "z": z[str(h)][wi][ci],
                                        "trace_sha256": "b"*64} for h in (4, 8, 16)}
                              for ci in range(support)]} for wi in range(128)]
    return {"manifest": r.PARENT_BINDING, "stage": "train", "spec": copy.deepcopy(spec),
            **identity, "horizons": [4, 8, 16], "complete": True,
            "forced_root_counts_as_ply": True, "new_terminal_simulations": 0,
            "worlds": worlds, "y": y, "z": z,
            "statistics": {**identity, "y": y, "z": z},
            "source": {"path": "fixture-source", "sha256": "c"*64},
            "evaluator": {"path": "fixture-evaluator", "sha256": "d"*64},
            "timing": {"cpu": 12.5, "wall": 15.5}}


def _root_evidence(base, indices=tuple(range(40))):
    refs = []
    for index in indices:
        path = base/"train"/"roots"/f"state_{index:04d}.json"
        c.old.save_new(path, _record(_spec(index)))
        refs.append(c.artifact(path, index=index))
    return {"train_roots": refs, "previous_elapsed_seconds": 600.}


def _rewrite_fixture(path, payload, base):
    path = Path(path).resolve()
    assert path.is_relative_to(base.resolve())
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_completed_root_transform_changes_only_binding_and_provenance():
    ref = {"path": "outputs/previous/train/roots/state_0000.json", "sha256": "e"*64, "index": 0}
    record = _record(_spec(0)); before = copy.deepcopy(record)
    transformed = r.transform_root(record, "new-binding", ref)
    assert transformed == {**before, "manifest": "new-binding", "reused_from": ref}
    assert record == before


def test_all_forty_completed_roots_import_exactly_without_altering_ancestors(tmp_path):
    evidence = _root_evidence(tmp_path/"prior")
    base = tmp_path/"recovery"
    refs = r.import_roots(base, "new-binding", evidence)
    assert refs == r.validate_roots(base, "new-binding", evidence)
    assert len(refs) == 40
    for ref, prior in zip(refs, evidence["train_roots"], strict=True):
        assert ref["index"] == prior["index"]
        assert c.verified(ref) == r.transform_root(c.verified(prior), "new-binding", prior)
        c.verify_hash(prior)
    assert not (base/"train_audits_lock.json").exists()
    assert not (base/"selection_lock.json").exists()
    assert not (base/"validation").exists()


@pytest.mark.parametrize("damage", ["endpoint", "target", "source", "spec", "complete", "reference"])
def test_reuse_rejects_changed_scientific_payload_even_after_link_rehash(tmp_path, damage):
    evidence = _root_evidence(tmp_path/"prior")
    base = tmp_path/"recovery"
    refs = r.import_roots(base, "new-binding", evidence)
    assert r.validate_roots(base, "new-binding", evidence) == refs
    path = c.verify_hash(refs[0]); record = c.old.read_json(path)
    if damage == "endpoint": record["worlds"][0]["candidates"][0]["4"]["z"] += .01
    elif damage == "target": record["y"][0][0] = .125
    elif damage == "source": record["source"]["sha256"] = "9"*64
    elif damage == "spec": record["spec"]["seat"] = 1
    elif damage == "complete": record["complete"] = False
    elif damage == "reference": record["reference"] = 7
    _rewrite_fixture(path, record, base)
    receipt_path = base/"train_reuse.json"
    receipt = c.old.read_json(receipt_path)
    receipt["files"][0] = c.artifact(path, index=0)
    _rewrite_fixture(receipt_path, receipt, base)
    with pytest.raises(ValueError):
        r.validate_roots(base, "new-binding", evidence)
    for ref in evidence["train_roots"]:
        c.verify_hash(ref)


def test_root_import_preserves_existing_output_instead_of_overwriting(tmp_path):
    evidence = _root_evidence(tmp_path/"prior")
    base = tmp_path/"recovery"
    refs = r.import_roots(base, "new-binding", evidence)
    with pytest.raises((ValueError, FileExistsError)):
        r.import_roots(base, "new-binding", evidence)
    for ref in refs+evidence["train_roots"]:
        c.verify_hash(ref)


def test_partition_preserves_original_order_short_support_and_exact_40_128_split(tmp_path):
    specs = [_spec(index) for index in range(168)]
    # Completed workers need not have returned or been discovered in root order.
    evidence = _root_evidence(tmp_path, tuple(reversed(range(40))))
    refs = evidence["train_roots"]
    before_specs, before_refs = copy.deepcopy(specs), copy.deepcopy(refs)
    missing, reused = r.plan_training(specs, refs)
    assert missing == specs[40:]
    assert {ref["index"] for ref in reused} == set(range(40))
    assert len(missing) == 128 and len(reused) == 40
    assert {s["index"] for s in missing}.isdisjoint({ref["index"] for ref in reused})
    assert sorted([s["index"] for s in missing]+[ref["index"] for ref in reused]) == list(range(168))
    assert {c.verified(ref)["support"] for ref in reused} == set(range(1, 9))
    assert specs == before_specs and refs == before_refs


@pytest.mark.parametrize("damage", ["duplicate_completed", "unknown_completed", "duplicate_schedule",
                                    "wrong_spec", "wrong_support", "incomplete", "wrong_stage",
                                    "missing_world", "extra_completed", "missing_completed",
                                    "extra_spec", "missing_spec"])
def test_partition_rejects_untrusted_or_incomplete_completed_roots(tmp_path, damage):
    specs = [_spec(index) for index in range(168)]
    evidence = _root_evidence(tmp_path)
    refs = evidence["train_roots"]
    if damage == "duplicate_completed":
        refs[-1] = refs[0]
    elif damage == "unknown_completed":
        record = _record(_spec(999))
        path = tmp_path/"unknown.json"; c.old.save_new(path, record)
        refs[-1] = c.artifact(path, index=999)
    elif damage == "duplicate_schedule":
        specs[-1] = specs[0]
    elif damage == "extra_completed":
        path = tmp_path/"extra.json"; c.old.save_new(path, _record(_spec(40)))
        refs.append(c.artifact(path, index=40))
    elif damage == "missing_completed":
        refs.pop()
    elif damage == "extra_spec":
        specs.append(_spec(168))
    elif damage == "missing_spec":
        specs.pop()
    else:
        path = c.verify_hash(refs[0]); record = c.old.read_json(path)
        if damage == "wrong_spec": record["spec"]["seat"] = 1
        elif damage == "wrong_support": record["support"] = 8
        elif damage == "incomplete": record["complete"] = False
        elif damage == "wrong_stage": record["stage"] = "validation"
        elif damage == "missing_world": record["worlds"].pop()
        _rewrite_fixture(path, record, tmp_path)
        refs[0] = c.artifact(path, index=0)
    with pytest.raises(ValueError):
        r.plan_training(specs, refs)


def test_resume_dispatches_only_missing_128_and_seals_all_168_without_claiming_audits(tmp_path, monkeypatch):
    specs = [_spec(index) for index in range(168)]
    evidence = _root_evidence(tmp_path/"prior")
    base = tmp_path/"recovery"
    r.import_roots(base, "new-binding", evidence)
    c.old.save_new(base/"manifest.json", {"held_parent": evidence})
    c.old.save_new(base/"freeze.json", {"manifest": c.artifact(base/"manifest.json")})
    monkeypatch.setattr(r, "OUTPUT", base)
    monkeypatch.setattr(r, "validate_freeze", lambda: "new-binding")
    monkeypatch.setattr(r, "budget_boundary", lambda *args: None)
    monkeypatch.setattr(c.windows, "configure_spawn", lambda: {"fixture": True})
    progress, launched = [], []
    monkeypatch.setattr(parent, "event", lambda kind, **kw: progress.append((kind, kw)))

    class FakePool:
        def __init__(self, **kwargs):
            assert kwargs["initializer"] is r.worker_initialize
            assert kwargs["initargs"] == ("train", "new-binding", base)
        def __enter__(self): return self
        def __exit__(self, *args): return False

    def collect(kind, spec, binding, base):
        assert kind == "train" and binding == "new-binding"
        launched.append(spec["index"])
        path = base/"train"/"roots"/f"state_{spec['index']:04d}.json"
        # Scheduling and seal integration fixture: no simulated outcomes are produced.
        c.old.save_new(path, {"manifest": binding, "index": spec["index"], "fixture": True})
        return c.artifact(path, index=spec["index"])

    def dispatch(pool, pending, job, workers, on_result, boundary):
        assert pending == specs[40:]
        assert workers == c.CONFIG["workers"]
        refs = []
        for spec in reversed(pending):
            refs.append(job(spec)); boundary(); on_result(len(refs))
        return refs

    monkeypatch.setattr(parent, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(parent, "run_job", collect)
    monkeypatch.setattr(parent, "dispatch", dispatch)
    combined = r.run_training(specs, "new-binding")
    assert len(launched) == 128 and set(launched) == set(range(40, 168))
    assert [ref["index"] for ref in combined] == list(range(168))
    lock = c.old.read_json(base/"train_lock.json")
    new_lock = c.verified(lock["new_collection"])
    reused = c.verified(lock["reused_collection"])
    assert lock["files"] == combined and lock["reused_roots"] == 40 and lock["new_roots"] == 128
    assert len(new_lock["files"]) == 128 and len(reused["files"]) == 40
    assert progress[0] == ("train", {"completed": 40, "total": 168, "reused": 40, "new_completed": 0})
    assert progress[-1] == ("train", {"completed": 168, "total": 168, "reused": 40, "new_completed": 128})
    assert not (base/"train_audits_lock.json").exists()
    assert not (base/"selection_lock.json").exists()
    assert not (base/"validation").exists()
    assert reused["independent_audits_pending"] is True
    for ref in evidence["train_roots"]:
        c.verify_hash(ref)

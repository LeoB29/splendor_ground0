"""Real tiny-network regression for elapsed-time inheritance and exact metrics."""
import copy
import json
import time

import pytest
import torch

from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from tools.independent_game_coverage_128_recovery_01.test_learning import row
from tools.teacher_precision_512 import common as c, learning
from . import diagnostics as d


@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(719)
    model = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(
        observation_size=6, action_feature_size=4, state_hidden_size=8,
        action_hidden_size=8, scorer_hidden_size=8, state_depth=1, action_depth=1,
        attention_layers=1, attention_heads=2, attention_feedforward_size=16))
    values = [row(i, support=n) for i, n in enumerate((3, 4, 5, 8, 6))]
    for value in values:
        sums = [4*v for v in value["old_sums_wdl"]]
        if len(sums) > 4:
            sums[3], sums[4] = 401, 79
        value.update(teacher_sums_wdl=sums, teacher_worlds=512,
                     teacher_outsider=max(range(3, len(sums)), key=lambda j: sums[j]) if len(sums) > 3 else -1)
    return model, values, {**learning.CONFIG, "batch_size": 2}


def receipt(base, *, launch_hours=0., active_hours=None):
    epoch = time.time()
    (base/"launch.json").write_text(json.dumps({"epoch_seconds": epoch-launch_hours*3600}))
    if active_hours is not None:
        (base/"active_run.json").write_text(json.dumps({
            "epoch_seconds": epoch, "prior_active_seconds": active_hours*3600}))


@pytest.mark.parametrize("precision", [False, True])
@pytest.mark.parametrize("with_initial", [False, True])
def test_exact_fresh_equivalence_normal_audit_and_previous(tmp_path, tiny, precision, with_initial):
    model, rows, config = tiny
    receipt(tmp_path)
    before_model = {k: v.clone() for k, v in model.state_dict().items()}
    initial = d.ORIGINAL_REPORT(model, rows, tmp_path, config, precision=precision) if with_initial else None
    ordinary = d.training_report(model, rows, tmp_path, config, initial, precision=precision)
    for audit in (False, True):
        previous = d.ORIGINAL_REPORT(model, rows, tmp_path, config, initial,
                                     precision=precision, audit=audit)
        repaired = d.training_report(model, rows, tmp_path, config, initial,
                                     precision=precision, audit=audit)
        assert previous == repaired
        learning._compare(ordinary, repaired)
    assert all(torch.equal(v, before_model[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    if precision:
        assert ordinary["target_worlds"] == 512
        units = sum(record["old_regret_units"] for record in ordinary["rows"])
        assert all(type(record["old_regret_units"]) is int for record in ordinary["rows"])
        assert any(record["old_regret_units"] % 4 for record in ordinary["rows"])
        assert ordinary["metrics"]["all_rows"]["mean_old_score_regret"] == units/(1024*len(rows))


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("precision", [False, True])
@pytest.mark.parametrize("launch_hours,active_hours", [(9., None), (80., 13.)])
def test_elapsed_ancestor_ceiling_cannot_stop_current_report(tmp_path, tiny, audit, precision,
                                                         launch_hours, active_hours):
    model, rows, config = tiny
    receipt(tmp_path, launch_hours=launch_hours, active_hours=active_hours)
    with pytest.raises(learning.prior.c.BudgetExceeded, match="Eight-hour"):
        d.ORIGINAL_REPORT(model, rows, tmp_path, config, audit=audit, precision=precision)
    actual = d.training_report(model, rows, tmp_path, config, audit=audit, precision=precision)
    assert len(actual["rows"]) == len(rows)
    if precision:
        assert actual["target_worlds"] == 512


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("precision", [False, True])
@pytest.mark.parametrize("hold", ["abort", "ceiling"])
def test_current_abort_and_active48_hour_ceiling_still_enforced(tmp_path, tiny, audit, precision, hold):
    model, rows, config = tiny
    receipt(tmp_path, launch_hours=80., active_hours=49. if hold == "ceiling" else 13.)
    if hold == "abort":
        (tmp_path/"abort.json").write_text("{}")
    with pytest.raises(c.BudgetExceeded, match="Forty-eight-hour" if hold == "ceiling" else "Cooperative abort"):
        d.training_report(model, rows, tmp_path, config, audit=audit, precision=precision)


def test_install_changes_only_current_report_and_preserves_frozen_code(tmp_path, tiny, monkeypatch):
    model, rows, config = tiny
    receipt(tmp_path, launch_hours=9.)
    ancestor_common = learning.prior.c
    ancestor_budget, ancestor_report = ancestor_common.budget_boundary, learning.prior.training_report
    ancestor_config = copy.deepcopy(ancestor_common.CONFIG)
    current_config = copy.deepcopy(c.CONFIG)
    monkeypatch.setattr(learning, "training_report", learning.training_report)
    assert d.install() is d.training_report
    assert d.install() is d.training_report
    assert learning.training_report is d.training_report
    assert learning.prior.training_report is ancestor_report is d.ORIGINAL_INNER_REPORT
    assert ancestor_common.budget_boundary is ancestor_budget
    assert ancestor_common.CONFIG == ancestor_config and c.CONFIG == current_config
    assert d.INNER_REPORT.__code__ is d.ORIGINAL_INNER_REPORT.__code__
    assert d.REPORT.__code__ is d.ORIGINAL_REPORT.__code__
    for repaired, original, replacement in ((d.INNER_REPORT, d.ORIGINAL_INNER_REPORT, "c"),
                                            (d.REPORT, d.ORIGINAL_REPORT, "prior")):
        assert repaired.__defaults__ == original.__defaults__
        assert repaired.__kwdefaults__ == original.__kwdefaults__
        assert repaired.__globals__.keys() <= original.__globals__.keys()
        assert all(value is original.__globals__[key] for key, value in repaired.__globals__.items()
                   if key not in (replacement, "training_report"))
    assert len(learning.training_report(model, rows, tmp_path, config, precision=True)["rows"]) == len(rows)
    with pytest.raises(ancestor_common.BudgetExceeded, match="Eight-hour"):
        ancestor_report(model, learning._diagnostic_view(rows, precision=True), tmp_path, config)


def test_current_boundary_is_looked_up_at_call_time(tmp_path, tiny, monkeypatch):
    model, rows, config = tiny
    calls = []
    def hold(base):
        calls.append(base)
        raise RuntimeError("current recovery boundary")
    monkeypatch.setattr(c, "budget_boundary", hold)
    for audit in (False, True):
        with pytest.raises(RuntimeError, match="current recovery boundary"):
            d.training_report(model, rows, tmp_path, config, audit=audit, precision=True)
    assert calls == [tmp_path, tmp_path]


def test_install_rejects_unexpected_replacement(monkeypatch):
    monkeypatch.setattr(learning, "training_report", lambda *a, **k: None)
    with pytest.raises(ValueError, match="Unexpected diagnostic replacement"):
        d.install()


def test_installed_report_reaches_fit_and_independent_reproduction_after8h(tmp_path, monkeypatch):
    from tools.teacher_precision_512.test_learning import fixture
    values = fixture(tmp_path, updates=4)
    base = tmp_path/"repaired"
    base.mkdir()
    receipt(base, launch_hours=9.)
    monkeypatch.setattr(learning, "training_report", learning.training_report)
    d.install()
    result_ref = learning.fit(learning.FITS[0], "recovery-fixture", base, **values)
    result = c.verified(result_ref)
    arm = c.verified(result["arms"]["precision"])
    assert arm["selected_update"] == 4
    verified = c.verified(learning.audit_fit(learning.FITS[0], result_ref,
                                            "recovery-fixture", base, **values))
    assert verified["passed"] and verified["complete_fit_bitwise_identical"]
    assert all(verified["arms"]["precision"][key] for key in learning.AUDIT_FLAGS)

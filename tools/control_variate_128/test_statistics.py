"""Finite-mean variance, TRAIN-only selection and whole-pair uncertainty checks."""
import copy
import math
from pathlib import Path
import pytest
from . import statistics as s,common as c
from tools.teacher_signal_capture_128.test_verify import specs


def population(split="train",horizons=(4,8,16),kind="perfect"):
    rows=[]
    for spec in specs():
        if spec["split"] != split: continue
        support=1 if spec["pair"] % 11 == 0 else 3
        y=[];z={str(h):[] for h in horizons}
        for w in range(128):
            values=[.5, float(w%2), float((w//2)%2)][:support]
            if kind == "zero_y": values=[.5]*support
            y.append(values)
            for h in horizons:
                if kind == "constant_z": control=[.3]*support
                elif kind == "negative": control=[1-v for v in values]
                elif kind == "unrelated": control=[.5,float((w//4)%2),float((w//8)%2)][:support]
                else: control=[v*(1+spec["pair"]%3/10) for v in values]
                z[str(h)].append(control)
        rows.append({**{k:spec[k] for k in ("index","pair","fit","split","game_index","phase","seat")},
                     "support":support,"reference":0,"y":y,"z":z})
    return rows


def timing(rows=None):
    rows=population() if rows is None else rows
    selected=[]
    for fit in c.FITS:
        for seat in (0,1):
            for phase in (0,1,2):
                row=next(r for r in rows if r["fit"] == fit and r["seat"] == seat and r["phase"] == phase)
                count=row["support"]*c.CONFIG["timing_worlds"]
                block=lambda rate:{"cpu":rate*count,"wall":rate*count,"cells":count}
                selected.append({**copy.deepcopy(row),"timing":{"full_only":block(1.),
                    "full_plus":{str(h):block(1+h/40) for h in c.HORIZONS},
                    "prefix_only":{str(h):block(h/40) for h in c.HORIZONS}}})
    return selected


@pytest.fixture(autouse=True)
def bounded_bootstrap(monkeypatch):
    monkeypatch.setitem(c.CONFIG,"bootstrap_samples",79)
    monkeypatch.setitem(c.CONFIG,"bootstrap_chunk",23)


@pytest.fixture(scope="module")
def selection():
    return s.calibrate(population(),timing())


def test_moments_use_within_root_differences_not_between_root_value_correlation():
    row=population()[4]
    for h in c.HORIZONS:
        s._same(s.moments(row,h),s.moments(row,h,audit=True))
    original=s.moments(row,4)
    shifted=copy.deepcopy(row)
    for w,values in enumerate(shifted["z"]["4"]):
        shifted["z"]["4"][w]=[v+1000*(w%2) for v in values]
    s._same(original,s.moments(shifted,4,audit=True))


def test_finite_auxiliary_mean_is_not_known_and_nested_formula_agrees():
    m={"var_y":.25,"var_z":.25,"covariance":.25}
    assert s._variance(m,1.,16,512) == pytest.approx(.25/512,abs=1e-14)
    assert s._variance(m,1.,16,16) == .25/16
    assert s._variance(m,0.,16,512) == .25/16
    for beta in (-2.,0.,.3,1.,2.):
        assert s._variance(m,beta,17,97) == pytest.approx(s._variance(m,beta,17,97,audit=True),abs=1e-14)


def test_selection_reproduces_moments_costs_allocation_and_is_train_only(selection):
    checked=s.calibrate(population(),timing(),audit=True)
    s._same(selection,checked)
    assert selection["status"] == "selected" and selection["config"]["horizon"] == 4
    assert selection["config"]["n"] < selection["config"]["M"] <= 512
    assert s._feasible(selection["config"],selection["cost_model"])
    assert selection["validation_read"] is False
    with pytest.raises(ValueError,match="identity"): s.calibrate(population("validation"),timing())


def test_constant_proxy_zero_beta_and_negative_correlation_are_lawful():
    constant=s.calibrate(population(kind="constant_z"),timing())
    assert constant["config"]["beta"] == 0.
    assert constant["selected_training_prediction"]["efficiency_ratio"] >= 1.
    negative=s.calibrate(population(kind="negative"),timing())
    assert negative["config"]["beta"] == pytest.approx(-1.)


def test_zero_training_variance_and_no_feasible_allocation_are_explicit():
    result=s.calibrate(population(kind="zero_y"),timing())
    assert result["status"] == "undefined_zero_training_variance" and result["config"] is None
    costs=timing()
    for row in costs:
        for h in c.HORIZONS: row["timing"]["full_plus"][str(h)]["cpu"]*=1000
    result=s.calibrate(population(),costs)
    assert result["status"] == "no_feasible_allocation" and result["config"] is None


def test_cpu_counts_every_sample_and_conservative_overhead(selection):
    costs=timing()
    for row in costs:
        row["timing"]["full_plus"]["4"]["cpu"]*=.5
    model=s.cost_model(costs)
    assert all(p["horizons"]["4"]["full_plus"] == p["full_only"] for p in model["phases"].values())
    row=population()[1];config={"horizon":4,"beta":1.,"n":16,"M":64}
    cv,baseline=s._cost(row,config,model)
    assert cv == pytest.approx(row["support"]*(16+48*.1))
    assert baseline == row["support"]*128


def test_validation_bootstrap_separate_reconstruction_and_source_pair_unit(selection):
    rows=population("validation",(selection["config"]["horizon"],))
    result=s.summarize(rows,selection)
    expected=s.summarize(rows,selection,audit=True)
    s._same(result,expected)
    assert result["states"] == 168 and result["pairs"] == 42 and result["passed"]
    assert result["ci95"][1] < 1 and result["bootstrap_samples"] == 79
    assert result["timing_uncertainty_in_ci"] is False
    assert not result["timing_sensitivity"]["used_for_selection_or_gate"]
    assert not result["actual_estimator_run"]


def test_frozen_training_config_not_refitted_when_validation_is_worse(selection):
    selected=copy.deepcopy(selection)
    rows=population("validation",(selection["config"]["horizon"],),kind="unrelated")
    result=s.summarize(rows,selection)
    assert selection == selected
    assert result["efficiency_ratio"] > 1 and not result["passed"]


def test_zero_validation_variance_not_imputed_or_dropped(selection):
    rows=population("validation",(selection["config"]["horizon"],),kind="zero_y")
    result=s.summarize(rows,selection)
    assert result["efficiency_ratio"] is None and result["ci95"] is None and not result["passed"]
    assert result["bootstrap_zero_denominator_count"] == 79


@pytest.mark.parametrize("damage",["unknown","world_count","reference","extra_horizon","pair","fit","phase","short_support"])
def test_malformed_saved_panel_rejected(selection,damage):
    rows=population("validation",(selection["config"]["horizon"],))
    row=next(r for r in rows if r["support"] > 1)
    if damage == "unknown": row["y"][0][0]=None
    elif damage == "world_count": row["y"].pop()
    elif damage == "reference": row["reference"]=row["support"]
    elif damage == "extra_horizon": row["z"]["99"]=row["z"][str(selection["config"]["horizon"])]
    elif damage == "pair": row["pair"]=-1
    elif damage == "fit": row["fit"]=-1
    elif damage == "phase": row["phase"]=5
    else: row["support"]=1
    with pytest.raises(ValueError): s.summarize(rows,selection)


@pytest.mark.parametrize("damage",["validation","missing_cell","mismatched_count","negative_cpu","NaN"])
def test_timing_calibration_identity_and_cpu_checks(damage):
    rows=timing()
    if damage == "validation": rows[0]["split"]="validation"
    elif damage == "missing_cell": rows[0]["seat"]=1-rows[0]["seat"]
    elif damage == "mismatched_count": rows[0]["timing"]["full_only"]["cells"]+=1
    elif damage == "negative_cpu": rows[0]["timing"]["full_only"]["cpu"]=-1
    else: rows[0]["timing"]["prefix_only"]["4"]["cpu"]=float("nan")
    with pytest.raises(ValueError): s.cost_model(rows)


def _artifacts(rows,binding,base,stem):
    result=[]
    for index,row in enumerate(rows):
        keys=("index","pair","fit","split","game_index","phase","seat","support")
        keys+=("timing",) if stem == "timing" else ("reference","y","z")
        projection={key:row[key] for key in keys}
        path=base/f"{stem}_{index}.json"
        c.old.save_new(path,{"manifest":binding,**projection,"complete":True,
                            "stage":row["split"],"statistics":projection})
        result.append(c.artifact(path))
    return result


def test_saved_selection_and_assessment_receipts_end_to_end(tmp_path):
    train=_artifacts(population(),"fixture",tmp_path,"train")
    times=_artifacts(timing(),"fixture",tmp_path,"timing")
    selected=s.select(train,times,"fixture",tmp_path)
    check=s.audit_selection(train,times,selected,"fixture",tmp_path)
    assert c.verified(check)["allocation_independently_reproduced"]
    horizon=c.verified(selected)["config"]["horizon"]
    val=_artifacts(population("validation",(horizon,)),"fixture",tmp_path,"val")
    result=s.assess(val,selected,times,"fixture",tmp_path)
    checked=c.verified(s.audit(val,selected,times,result,"fixture",tmp_path))
    assert checked["passed"] and checked["bootstrap_independently_reproduced"]
    with pytest.raises(ValueError): s.select(train,times,"fixture",tmp_path)


def test_selection_auditor_rejects_changed_allocation(tmp_path,selection):
    train=_artifacts(population(),"fixture",tmp_path,"train")
    times=_artifacts(timing(),"fixture",tmp_path,"timing")
    result=copy.deepcopy(selection);result.update(manifest="fixture",training_roots=train,timing_roots=times,time="fixture")
    result["config"]["M"]-=1
    path=tmp_path/"selection_lock.json";c.old.save_new(path,result)
    with pytest.raises(ValueError,match="statistic"): s.audit_selection(train,times,c.artifact(path),"fixture",tmp_path)


@pytest.mark.parametrize("damage",["projection","complete","stage","manifest"])
def test_root_receipts_reject_stale_projection_or_wrong_identity(tmp_path,damage):
    refs=_artifacts(population()[:1],"fixture",tmp_path,"train")
    record=c.verified(refs[0])
    if damage == "projection": record["statistics"]["y"][0][0]=1.;record["y"][0][0]=.5
    elif damage == "complete": record["complete"]=False
    elif damage == "stage": record["stage"]="validation"
    else: record["manifest"]="different"
    path=tmp_path/"tampered.json";c.old.save_new(path,record)
    with pytest.raises(ValueError): s._read_roots([c.artifact(path)],"fixture","train")


@pytest.mark.parametrize("damage",["missing","unlinked","failed","different_seal"])
def test_validation_outcome_firewall_rejects_missing_or_changed_selection_audit(tmp_path,monkeypatch,selection,damage):
    selected=copy.deepcopy(selection);selected.update(manifest="fixture",timing_roots=[])
    path=tmp_path/"selection_lock.json";c.old.save_new(path,selected)
    ref=c.artifact(path)
    receipt={"manifest":"fixture","passed":True,"selection":ref,"validation_read":False,
             "moments_independently_reconstructed":True,"allocation_independently_reproduced":True,
             "fixed_train_rule_preserved":True}
    if damage == "unlinked": receipt["selection"]={"path":"different","sha256":"bad"}
    elif damage == "failed": receipt["passed"]=False
    elif damage == "different_seal":
        other=tmp_path/"unsealed.json";c.old.save_new(other,selected);ref=c.artifact(other)
    if damage != "missing": c.old.save_new(tmp_path/"selection_audit.json",receipt)
    def no_outcome_read(*args,**kwargs): raise AssertionError("Outcomes read before selection firewall")
    monkeypatch.setattr(s,"_read_roots",no_outcome_read)
    with pytest.raises((ValueError,FileNotFoundError)):
        s.assess([],ref,[],"fixture",tmp_path)

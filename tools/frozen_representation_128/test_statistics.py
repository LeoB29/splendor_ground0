"""Pair-preserving inference, bounded outcomes and conjunctive development gates."""
import copy
import pytest
from . import assessment as a,statistics as st
from tools.teacher_signal_capture_128.test_engine import worlds,outcome
from tools.teacher_signal_capture_128.test_verify import specs


def population_rows(unknown=True):
    offers={family:{"offered_group_scores":[0,0,0,2,1] if family != "headonly" else [0,0,0,1,2]}
            for family in a.c.MODEL_ARMS}
    choices=a._choices([0,0,0,0,128],[0]*5,offers)
    blocks=[]
    for variation in range(7):
        block=worlds([0]*5)
        for j in range(variation): block[j]["outcomes"][4]=outcome(1)
        if unknown and variation == 6: block[-1]["outcomes"][4]=outcome(None)
        blocks.append(block)
    return [a._score(spec,choices,blocks[spec["pair"]%7]) for spec in specs()]


def test_whole_pair_bootstraps_match_separate_math_and_no_fit_tripling(monkeypatch):
    monkeypatch.setitem(a.c.CONFIG,"bootstrap_samples",79)
    monkeypatch.setitem(a.c.CONFIG,"bootstrap_chunk",23)
    rows=population_rows()
    actual=st.summarize(rows)
    # The independent path must not reuse collector draws or the reporting reduction.
    monkeypatch.setattr(st,"_report",lambda *a:pytest.fail("Shared statistic arithmetic"))
    checked=st.independent_summarize(rows)
    a.compare(actual,checked,rel_tol=0.,abs_tol=1e-12)
    assert actual["states"] == 336 and actual["pairs"] == 84
    assert actual["groups"]["validation"]["states"] == 168 and actual["groups"]["validation"]["pairs"] == 42
    assert all(actual["per_fit"][str(f)]["validation"]["pairs"] == 14 for f in a.c.FITS)
    assert actual["groups"]["validation"]["contrasts"][st.NAMES[0]]["mean"] is None
    assert actual["groups"]["validation"]["contrasts"]["teacher_minus_headonly_supported"]["ci95"] == [0.,0.]


def test_exact_mean_and_zero_contrasts(monkeypatch):
    monkeypatch.setitem(a.c.CONFIG,"bootstrap_samples",31)
    rows=population_rows(False)
    result=st.summarize(rows)
    for split in st.SPLITS:
        subset=[r for r in rows if r["split"] == split]
        for name,_,_ in st.CONTRASTS:
            assert result["groups"][split]["contrasts"][name]["mean"] == sum(r["delta_bounds"][name][0] for r in subset)/168


@pytest.mark.parametrize("damage",["missing","split","fit","leg","phase","score_grid","imputation","support","outsider","same_action","delta"])
def test_population_and_choice_corruption_rejected(damage):
    rows=population_rows();row=rows[0]
    if damage == "missing": rows.pop()
    elif damage == "split": row["split"]="validation"
    elif damage == "fit": row["fit"]=999
    elif damage == "leg": row["game_index"]+=2
    elif damage == "phase": rows[1]["phase"]=row["phase"]
    elif damage == "score_grid": row["arm_score_bounds"]["headonly_supported"][0]+=.0001
    elif damage == "imputation": row["arm_scores"]["headonly_supported"]=.9
    elif damage == "support": row["support"]=9
    elif damage == "outsider": row["choices"]["headonly_outsider"]=2
    elif damage == "same_action": row["choices"]["headonly_supported"]=row["choices"]["anchor"]
    else: row["delta_bounds"][st.NAMES[0]][0]+=.1
    with pytest.raises(ValueError): st.summarize(rows)


def passing_gate_inputs():
    primary,control=a.c.CONFIG["primary_endpoint"],a.c.CONFIG["required_control_endpoint"]
    statistics={"groups":{"validation":{"contrasts":{
        primary:{"mean":.005,"mean_bounds":[.005,.005],"ci95":[.001,.01]},
        control:{"mean":.003,"mean_bounds":[.003,.003],"ci95":[.0001,.005]}}}},
        "per_fit":{str(fit):{"validation":{"contrasts":{name:{"mean_bounds":[0.,0.]} for name in (primary,control)}}}
                   for fit in a.c.FITS}}
    validation={"families":{family:{"per_fit":{str(fit):{"pair_weighted":{"mean_old_score_regret":regret}}
                  for fit in a.c.FITS}} for family,regret in (("anchor",.06),("broad",.05),("headonly",.04))}}
    return statistics,validation,{str(fit):.5 for fit in a.c.FITS}


@pytest.mark.parametrize("damage",[None,"acquisition","missing_acquisition","old_tie_anchor","old_tie_broad","mean","primary_ci","control_ci","single_fit"])
def test_all_preregistered_gate_parts_required(damage):
    stats,val,capture=passing_gate_inputs()
    first=str(a.c.FITS[0]);primary=a.c.CONFIG["primary_endpoint"];control=a.c.CONFIG["required_control_endpoint"]
    if damage == "acquisition": capture[first]=.499
    elif damage == "missing_acquisition": capture[first]=None
    elif damage == "old_tie_anchor": val["families"]["anchor"]["per_fit"][first]["pair_weighted"]["mean_old_score_regret"]=.04
    elif damage == "old_tie_broad": val["families"]["broad"]["per_fit"][first]["pair_weighted"]["mean_old_score_regret"]=.04
    elif damage == "mean": stats["groups"]["validation"]["contrasts"][primary]["mean_bounds"][0]=.004999
    elif damage == "primary_ci": stats["groups"]["validation"]["contrasts"][primary]["ci95"][0]=0.
    elif damage == "control_ci": stats["groups"]["validation"]["contrasts"][control]["ci95"][0]=0.
    elif damage == "single_fit": stats["per_fit"][first]["validation"]["contrasts"][primary]["mean_bounds"][0]=-.0001
    result=st.development_screen(stats,val,capture)
    assert result["passed"] is (damage is None)
    assert result["reused_development"] and not result["automatic_successor"]


def test_pair_weighted_full_validation_does_not_weight_long_games_more():
    records=[]
    for pair,count,regret in (("short",1,256),("long",9,0)):
        records.extend({"pair":pair,"old_regret_units":regret,**dict.fromkeys(
            ("loss","soft_loss","target_kl","policy_entropy","target_entropy"),float(regret)/256)} for _ in range(count))
    assert a._pair_metrics(records)["mean_old_score_regret"] == .5
    assert a._pair_metrics(records) == a._pair_metrics(records,audit=True)

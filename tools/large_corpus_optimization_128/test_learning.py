"""Tiny real continuations, independent replay, dose and acquisition contracts."""
import copy
import time
from pathlib import Path

import pytest
import torch

from tools.independent_game_scaling_128.test_learning import fixture as scaling_fixture, schedule_pools
from . import learning as h


@pytest.fixture(scope="module")
def parent_values(tmp_path_factory):
    base = tmp_path_factory.mktemp("closed_scaling")
    values = scaling_fixture(base)
    # A fast-to-acquire toy problem; every continuation keeps this exact rate.
    values["config"]["lr"] = .003
    records = {}
    for fit in h.FITS:
        ref = h.inherited.fit(fit,"large","parent-fixture",base,**values)
        audit = h.inherited.audit_fit(fit,"large",ref,"parent-fixture",base,**values)
        result = h.c.verified(ref)
        resume_ref = h.c.artifact(base/"fits"/f"fit_{fit}"/"large"/"resume.json")
        records[fit] = {"result":ref,"audit":audit,"resume":resume_ref,
            "optimizer":h.c.verified(resume_ref)["checkpoint"],"model":result["selected_model"],
            "initializer":values["initializer"],"initial_training":result["checkpoints"][0]["training"],
            "final_training":result["selected_training"]}
    settings = {**h.CONFIG,**copy.deepcopy(values["config"]),"start_update":16,"updates":64,"extra_updates":48,
        "fresh_optimizer":False,"restore_saved_AdamW":True,"continuation_repeats":4,
        "diagnostic_updates":[16,32,48,64],"checkpoint_selection":"fixed_final_update_64"}
    return {**values,"config":settings},records


def fixture_kwargs(parent_values,fit=None):
    values,records = parent_values
    return {**copy.deepcopy(values),"continuation":copy.deepcopy(records[h.FITS[0] if fit is None else fit])}


def seal_age(base,hours):
    h.old.save_new(base/"launch.json",{"manifest":"fixture","epoch_seconds":time.time()-hours*3600})
    h.old.save_new(base/"launch_seal.json",{"manifest":"fixture","receipt":h.c.artifact(base/"launch.json")})


def test_production_schedule_exact_prefix_and_four_per_row_doses():
    pools,subsets = schedule_pools()
    fit = h.FITS[0]
    plans = h.inherited.batch_plan(pools,subsets,fit,h.inherited.CONFIG)
    paired = h.inherited._paired_receipt(plans,pools,h.inherited.CONFIG,fixture=False)
    parent = {"config":h.inherited.CONFIG,"identity":{"paired_batch_receipt":paired}}
    full,receipt = h._plans(pools,subsets,fit,parent,h.CONFIG)
    assert full[:7136] == plans["large"]
    assert all(full[i*7136:(i+1)*7136] == plans["large"] for i in range(4))
    assert receipt["total"]["row_exposures"] == 1824704
    assert receipt["total"]["old_exposures"] == receipt["total"]["added_exposures"] == 912352
    assert receipt["total"]["updates"] == 28544


def test_three_continuations_and_all_optimizer_states_independently_reproduced(parent_values,tmp_path,monkeypatch):
    seal_age(tmp_path,24)
    monkeypatch.setattr(h.inherited.c,"budget_boundary",lambda *a:pytest.fail("Inherited168h guard reached"))
    monkeypatch.setattr(h.inherited.prior.c,"budget_boundary",lambda *a:pytest.fail("Inherited8h guard reached"))
    fits,audits = [],[]
    for fit in h.FITS:
        kwargs = fixture_kwargs(parent_values,fit)
        ref = h.fit(fit,"fixture",tmp_path,**kwargs)
        audited = h.audit_fit(fit,ref,"fixture",tmp_path,**kwargs)
        result = h.c.verified(ref)
        assert all(h.c.verified(audited)[k] for k in h.AUDIT_FLAGS)
        assert [p["update"] for p in result["checkpoints"]] == [16,32,48,64]
        original = h.c.verified(kwargs["continuation"]["initial_training"])["full"]
        final = h.c.verified(result["selected_training"])["full"]
        for pool in ("old","added"):
            assert final[pool]["initial_old_regret_units"] == original[pool]["initial_old_regret_units"]
        assert h.fit(fit,"fixture",tmp_path,**kwargs) == ref
        fits.append(ref);audits.append(audited)
    gate = h.c.verified(h.acquisition_gate(fits,audits,"fixture",tmp_path))
    assert gate["thresholds"] == {"added":.95,"old":.90}
    assert set(gate["fits"]) == {str(fit) for fit in h.FITS}
    controls = {f"{arm}_{fit}":parent_values[1][fit]["initializer" if arm == "anchor" else "model"]
                for arm in ("anchor","control") for fit in h.FITS}
    if gate["passed"]:
        models = h.c.verified(h.models_lock(fits,audits,"fixture",tmp_path,controls=controls))
        assert len(models["models"]) == 9 and models["acquisition_gate"] == h.c.artifact(tmp_path/"acquisition_gate.json")
    else:
        with pytest.raises(ValueError,match="Acquisition gate failed"):
            h.models_lock(fits,audits,"fixture",tmp_path,controls=controls)
        assert not (tmp_path/"models_lock.json").exists()
        # Qualification arithmetic is tested separately. Stub only that gate
        # to exercise normal model sealing/validation with real audited fits.
        fake_gate = tmp_path/"qualification_stub.json"
        h.old.save_new(fake_gate,{"manifest":"fixture","passed":True,"fixture_qualification_stub":True})
        monkeypatch.setattr(h,"acquisition_gate",lambda *a,**k:h.c.artifact(fake_gate))
        monkeypatch.setattr(h.source,"control_models",lambda *a,**k:controls)
        models = h.c.verified(h.models_lock(fits,audits,"fixture",tmp_path,controls=controls))
        assert len(models["models"]) == 9
        assert h.validate_models("fixture",tmp_path) == models


def test_resumed_matches_uninterrupted_including_moments_and_rng(parent_values,tmp_path,monkeypatch):
    kwargs = fixture_kwargs(parent_values)
    continuous,interrupted = tmp_path/"continuous",tmp_path/"interrupted"
    reference = h.fit(h.FITS[0],"fixture",continuous,**kwargs)
    save = h._save_progress
    def stop(folder,model,optimizer,update,norms,identity):
        save(folder,model,optimizer,update,norms,identity)
        if update == 32:raise h.c.BudgetExceeded("synthetic interruption")
    monkeypatch.setattr(h,"_save_progress",stop)
    with pytest.raises(h.c.BudgetExceeded):h.fit(h.FITS[0],"fixture",interrupted,**kwargs)
    before = h.c.artifact(interrupted/"fits"/f"fit_{h.FITS[0]}"/"point_00032.json")
    monkeypatch.setattr(h,"_save_progress",save)
    resumed = h.fit(h.FITS[0],"fixture",interrupted,**kwargs)
    assert before == h.c.artifact(interrupted/"fits"/f"fit_{h.FITS[0]}"/"point_00032.json")
    left,right = (h.c.verified(r) for r in (reference,resumed))
    for lref,rref in zip(left["optimizer_checkpoints"],right["optimizer_checkpoints"],strict=True):
        lp,rp = (torch.load(h.c.verify_hash(r),map_location="cpu",weights_only=True) for r in (lref,rref))
        assert h._equal_torch(lp,rp)
    assert h.c.verified(h.audit_fit(h.FITS[0],resumed,"fixture",interrupted,**kwargs))["passed"]


@pytest.mark.parametrize("damage",["rate","target","partition","reset_optimizer","diagnostic","initial_reference"])
def test_continuation_recipe_and_history_firewall(parent_values,tmp_path,damage):
    kwargs = fixture_kwargs(parent_values)
    if damage == "rate":kwargs["config"]["lr"] *= 2
    elif damage == "target":kwargs["rows"]["old"][0]["old_sums_wdl"][0] += 1
    elif damage == "partition":kwargs["partitions"]["medium"] = [2,3,4,5]
    elif damage == "reset_optimizer":kwargs["config"]["fresh_optimizer"] = True
    elif damage == "diagnostic":kwargs["config"]["diagnostic_updates"] = [16,64]
    else:kwargs["continuation"]["initial_training"] = kwargs["continuation"]["final_training"]
    with pytest.raises(ValueError):h.fit(h.FITS[0],"fixture",tmp_path,**kwargs)


def test_independent_sampler_does_not_call_collector(parent_values,tmp_path,monkeypatch):
    kwargs = fixture_kwargs(parent_values)
    pools,initial,settings,subsets,record,parent,_ = h._inputs(h.FITS[0],"fixture",tmp_path,**kwargs)
    expected = h._plans(pools,subsets,h.FITS[0],parent,settings)
    monkeypatch.setattr(h.inherited,"batch_plan",lambda *a,**k:pytest.fail("Collector schedule called"))
    assert h._plans(pools,subsets,h.FITS[0],parent,settings,audit=True) == expected


@pytest.mark.parametrize("mode",["abort","ceiling"])
def test_current_budget_stops_reports_fit_and_audit(parent_values,tmp_path,mode):
    kwargs = fixture_kwargs(parent_values)
    if mode == "abort":h.old.save_new(tmp_path/"abort.json",{"fixture":True})
    else:seal_age(tmp_path,49)
    with pytest.raises(h.c.BudgetExceeded):h.fit(h.FITS[0],"fixture",tmp_path,**kwargs)
    with pytest.raises(h.c.BudgetExceeded):h.audit_fit(h.FITS[0],{},"fixture",tmp_path,**kwargs)
    model,*_ = h._model(kwargs["initializer"],"continued",42,kwargs["config"])
    for audit in (False,True):
        with pytest.raises(h.c.BudgetExceeded):
            h.training_report(model,kwargs["rows"]["old"],tmp_path,kwargs["config"],audit=audit)


def test_optimizer_corruption_is_not_masked_by_unchanged_model(parent_values,tmp_path):
    kwargs = fixture_kwargs(parent_values)
    ref = h.fit(h.FITS[0],"fixture",tmp_path,**kwargs)
    result = h.c.verified(ref)
    target = next(r for r in result["optimizer_checkpoints"] if r["update"] == 32)
    payload = torch.load(h.c.verify_hash(target),map_location="cpu",weights_only=True)
    payload["optimizer_state_dict"]["state"][0]["exp_avg"].add_(.01)
    path = tmp_path/"corrupted_optimizer.pt";torch.save(payload,path)
    result["optimizer_checkpoints"] = [h.c.artifact(path,update=32) if r["update"] == 32 else r for r in result["optimizer_checkpoints"]]
    changed = tmp_path/"changed_result.json";h.old.save_new(changed,result)
    with pytest.raises(ValueError,match="Independent optimizer checkpoint differs"):
        h.audit_fit(h.FITS[0],h.c.artifact(changed),"fixture",tmp_path,**kwargs)


def gate_fixture(tmp_path,captures):
    """Arithmetic-only receipts: no claim of executing a synthetic trajectory."""
    from types import SimpleNamespace
    result = []
    for fit in h.FITS:
        original,final = {},{}
        for pool in ("old","added"):
            before = 1000;after = round(before*(1-captures[fit][pool]))
            original[pool] = {"rows":[{"old_regret_units":before}]}
            final[pool] = {"rows":[{"old_regret_units":after}],"initial_old_regret_units":before,
                "metrics":{"all_rows":{"old_regret_units_sum":after}},"regret_capture_vs_initializer":(before-after)/before}
        start,end = tmp_path/f"original_{fit}.json",tmp_path/f"final_{fit}.json"
        h.old.save_new(start,{"full":original});h.old.save_new(end,{"full":final})
        result.append({"fit":fit,"selected_training":h.c.artifact(end),"continuation":{"initial_training":h.c.artifact(start)}})
    return result


def test_acquisition_gate_uses_every_fit_both_pools_original_reference(tmp_path,monkeypatch):
    captures = {fit:{"added":.95,"old":.90} for fit in h.FITS}
    records = gate_fixture(tmp_path,captures)
    monkeypatch.setattr(h,"_verified_fits",lambda *a,**k:records)
    gate = h.c.verified(h.acquisition_gate([],[],"fixture",tmp_path))
    assert gate["passed"] and all(r["passed"] for r in gate["fits"].values())
    other = tmp_path/"fail";other.mkdir();captures[h.FITS[2]]["old"] = .899
    records = gate_fixture(other,captures)
    assert not h.c.verified(h.acquisition_gate([],[],"fixture",other))["passed"]

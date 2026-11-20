"""Small complete trajectories, exact recovery, production dose and acquisition."""
import copy
import time
import pytest
import torch
from tools.independent_game_scaling_128.test_learning import fixture as old_fixture, schedule_pools
from splendor_ai.alphazero.strong_policy import save_model
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from . import learning as h


def fixture(tmp_path):
    values = old_fixture(tmp_path)
    for pool in values["rows"].values():
        for row in pool:
            obs = torch.zeros(295);obs[:6] = row["observation"]
            obs[33] = 1.;obs[41:46] = torch.tensor([2,3,1,0,4])/7
            actions = torch.zeros(len(row["features"]),79);actions[:,:4] = row["features"]
            row["observation"],row["features"] = obs,actions
    torch.manual_seed(719)
    model = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(
        observation_size=295,action_feature_size=79,state_hidden_size=8,action_hidden_size=8,
        scorer_hidden_size=8,state_depth=1,action_depth=1,attention_layers=1,
        attention_heads=2,attention_feedforward_size=16))
    path = tmp_path/"structured_initial.pt";save_model(path,model,0,{"fixture":True})
    prefix = values["config"]
    # A short synthetic trajectory; the true7136 large sampler is separately checked.
    settings = {**copy.deepcopy(h.CONFIG),**copy.deepcopy(prefix),"prefix_config":copy.deepcopy(prefix),
                "full_repeats":4,"updates":64,"diagnostic_updates":[0,16,32,48,64]}
    return {**values,"initializer":h.c.artifact(path),"config":settings}


def seal_age(base,hours):
    h.old.save_new(base/"launch.json",{"manifest":"fixture","epoch_seconds":time.time()-hours*3600})
    h.old.save_new(base/"launch_seal.json",{"manifest":"fixture","receipt":h.c.artifact(base/"launch.json")})


def test_production_schedule_exact_7136_course_four_times():
    pools,subsets = schedule_pools()
    fit = h.FITS[0]
    full,receipt = h._plans(pools,subsets,fit,h.CONFIG)
    expected = h.inherited.batch_plan(pools,subsets,fit,h.inherited.CONFIG)["large"]
    assert all(full[i*7136:(i+1)*7136] == expected for i in range(4))
    assert receipt["total"]["row_exposures"] == 1824704
    assert receipt["total"]["old_exposures"] == receipt["total"]["added_exposures"] == 912352
    assert full == h._plans(pools,subsets,fit,h.CONFIG,audit=True)[0]


def test_all_six_fits_and_optimizer_states_reproduced(tmp_path,monkeypatch):
    values = fixture(tmp_path)
    seal_age(tmp_path,24)
    monkeypatch.setattr(h.inherited.c,"budget_boundary",lambda *a:pytest.fail("Inherited guard reached"))
    monkeypatch.setattr(h.inherited.prior.c,"budget_boundary",lambda *a:pytest.fail("Inherited guard reached"))
    fits,audits = [],[]
    for fit in h.FITS:
        for arm in h.ARMS:
            ref = h.fit(fit,arm,"fixture",tmp_path,**values)
            audit = h.audit_fit(fit,arm,ref,"fixture",tmp_path,**values)
            assert all(h.c.verified(audit)[key] for key in h.AUDIT_FLAGS)
            result = h.c.verified(ref)
            assert result["fresh_optimizer"] and result["base_policy_trainable"]
            assert [p["update"] for p in result["checkpoints"]] == [0,16,32,48,64]
            fits.append(ref);audits.append(audit)
    results = h._verified_fits(fits,audits,"fixture",tmp_path)
    assert len(results) == 6
    gate = h.c.verified(h.acquisition_gate(fits,audits,"fixture",tmp_path))
    assert set(gate["fits"]) == {f"{arm}_{fit}" for fit in h.FITS for arm in h.ARMS}
    controls = {f"anchor_{fit}":values["initializer"] for fit in h.FITS}
    if gate["passed"]:
        registry = h.c.verified(h.models_lock(fits,audits,"fixture",tmp_path,controls=controls))
        assert len(registry["models"]) == 9
    else:
        with pytest.raises(ValueError,match="Acquisition gate failed"):
            h.models_lock(fits,audits,"fixture",tmp_path,controls=controls)


def test_interrupted_resume_is_bitwise_equal_every_checkpoint(tmp_path,monkeypatch):
    values = fixture(tmp_path);fit,arm = h.FITS[0],"relations"
    expected = h.fit(fit,arm,"fixture",tmp_path/"clean",**values)
    original = h._save_progress
    def stop(*args):
        original(*args)
        if args[3] == 32:raise h.c.BudgetExceeded("fixture interruption")
    monkeypatch.setattr(h,"_save_progress",stop)
    with pytest.raises(h.c.BudgetExceeded):h.fit(fit,arm,"fixture",tmp_path/"resume",**values)
    monkeypatch.setattr(h,"_save_progress",original)
    actual = h.fit(fit,arm,"fixture",tmp_path/"resume",**values)
    for left,right in zip(h.c.verified(expected)["optimizer_checkpoints"],h.c.verified(actual)["optimizer_checkpoints"],strict=True):
        a,b = [torch.load(h.c.verify_hash(ref),map_location="cpu",weights_only=True) for ref in (left,right)]
        assert h._equal_torch(a,b)
    assert h.c.verified(h.audit_fit(fit,arm,actual,"fixture",tmp_path/"resume",**values))["passed"]


@pytest.mark.parametrize("damage",["temperature","mixture","old_validation","new_assessment","partition","dose","diagnostics","fresh"])
def test_learning_firewall(tmp_path,damage):
    values = fixture(tmp_path)
    if damage == "temperature":values["config"]["target_temperature"] = .1
    elif damage == "mixture":values["config"]["old_fraction"] = .4
    elif damage == "old_validation":values["rows"]["old"][0]["split"] = "validation"
    elif damage == "new_assessment":values["rows"]["added"][0]["assessment_outcomes"] = []
    elif damage == "partition":values["partitions"]["small"] = [6,7]
    elif damage == "dose":values["config"]["updates"] = 63
    elif damage == "diagnostics":values["config"]["diagnostic_updates"] = [0,64]
    else:values["config"]["fresh_optimizer"] = False
    with pytest.raises(ValueError):h.fit(h.FITS[0],"raw","fixture",tmp_path,**values)


def test_optimizer_corruption_rejected_independently(tmp_path):
    values = fixture(tmp_path)
    fit,arm = h.FITS[0],"raw"
    ref = h.fit(fit,arm,"fixture",tmp_path,**values)
    result = h.c.verified(ref)
    target = next(ref for ref in result["optimizer_checkpoints"] if ref["update"] == 32)
    payload = torch.load(h.c.verify_hash(target),map_location="cpu",weights_only=True)
    payload["optimizer_state_dict"]["state"][0]["exp_avg"].add_(.01)
    path = tmp_path/"corrupt.pt";torch.save(payload,path)
    result["optimizer_checkpoints"] = [h.c.artifact(path,update=32) if ref["update"] == 32 else ref for ref in result["optimizer_checkpoints"]]
    out = tmp_path/"corrupt.json";h.old.save_new(out,result)
    with pytest.raises(ValueError,match="Independent optimizer checkpoint differs"):
        h.audit_fit(fit,arm,h.c.artifact(out),"fixture",tmp_path,**values)


def test_gate_requires_every_fit_and_both_pools(tmp_path,monkeypatch):
    def records(base,fail=False):
        items = []
        for fit in h.FITS:
            for arm in h.ARMS:
                original,final = {},{}
                for pool in ("old","added"):
                    before = 1000;after = (100 if pool == "old" else 50)
                    if fail and fit == h.FITS[-1] and arm == "raw" and pool == "old":after = 101
                    original[pool] = {"rows":[{"old_regret_units":before}]}
                    final[pool] = {"rows":[{"old_regret_units":after}],"initial_old_regret_units":before,
                        "metrics":{"all_rows":{"old_regret_units_sum":after}},"regret_capture_vs_initializer":(before-after)/before}
                a,b = base/f"start_{arm}_{fit}.json",base/f"end_{arm}_{fit}.json"
                h.old.save_new(a,{"full":original});h.old.save_new(b,{"full":final})
                items.append({"fit":fit,"arm":arm,"selected_training":h.c.artifact(b),"checkpoints":[{"training":h.c.artifact(a)}]})
        return items
    for fail in (False,True):
        base = tmp_path/str(fail);base.mkdir()
        fixed = records(base,fail)
        monkeypatch.setattr(h,"_verified_fits",lambda *a,**k:fixed)
        assert h.c.verified(h.acquisition_gate([],[],"fixture",base))["passed"] is not fail


def test_zero_correction_reproduces_original_initializer_diagnostics(tmp_path):
    from splendor_ai.alphazero.strong_policy import StrongPolicyInference
    values = fixture(tmp_path)
    original = StrongPolicyInference(str(h.c.verify_hash(values["initializer"]))).model.eval()
    selected = values["partitions"]["large"]
    settings = values["config"]
    for probe in (False,True):
        expected = h._diagnostics(original,values["rows"],selected,tmp_path,settings,probe=probe)
        for arm in h.ARMS:
            model,*_ = h._model(values["initializer"],arm,settings["train_seeds"][str(h.FITS[0])],settings)
            actual = h._diagnostics(model,values["rows"],selected,tmp_path,settings,probe=probe)
            h._compare(actual,expected)

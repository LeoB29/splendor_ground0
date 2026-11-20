"""Old-label firewall, private input reconstruction, distinct controls and receipts."""
import copy
from pathlib import Path
import pytest
import torch
from . import assessment as a,statistics as st,source,learning
from .test_statistics import population_rows
from splendor_ai.alphazero.strong_policy import encode_position,save_model
from splendor_ai.training.model import LegalActionAttentionScorerConfig,LegalActionAttentionScorerModel
from tools.teacher_signal_capture_128.test_engine import worlds,outcome,proposals,private_state
from tools.teacher_signal_capture_128.test_verify import specs
from tools.teacher_signal_capture_128 import verify as original_verify


def offered(n,head=3,broad=3):
    return {family:{"offered_group_scores":[float(j == (head if family == "weighted" else broad if family == "broad" else 3))
                for j in range(n)]} for family in a.c.MODEL_ARMS}


@pytest.mark.parametrize("n",range(1,9))
def test_short_supports_ties_and_independent_raw_scores(n):
    choice=a._choices([0]*n,[0]*n,offered(n))
    assert choice == a._choices([0]*n,[0]*n,offered(n),audit=True)
    assert choice["weighted_supported"] == 0
    assert choice["weighted_outsider"] == (3 if n > 3 else None)
    row=a._score(specs()[0],choice,worlds([0]*n))
    assert row == a._score(specs()[0],choice,worlds([0]*n),audit=True)
    st._validate_row(row)


def test_separate_controls_can_choose_distinct_outsiders_and_common_order_ties():
    p=offered(6,5,4)
    choices=a._choices([0,0,0,128,128,128],[0]*6,p)
    assert choices["anchor_outsider"] == choices["anchor"] == 3
    assert choices["broad_outsider"] == choices["broad_supported"] == 4
    assert choices["weighted_outsider"] == choices["weighted_supported"] == 5
    assert choices["teacher"] == 3
    assert choices == a._choices([0,0,0,128,128,128],[0]*6,p,audit=True)
    p["anchor"]["offered_group_scores"][4]=2.
    with pytest.raises(ValueError,match="anchor fourth"): a._choices([0]*6,[0]*6,p)


def test_ambiguous_old_labels_hold_unknown_assessment_is_not_imputed():
    for audit in (False,True):
        with pytest.raises(ValueError): a._choices([0]*5,[1,0,0,0,0],offered(5),audit=audit)
    choice=a._choices([0,0,0,0,128],[0]*5,offered(5,4))
    block=worlds([0]*5);block[0]["outcomes"][4]=outcome(None)
    row=a._score(specs()[0],choice,block)
    assert row == a._score(specs()[0],choice,block,audit=True)
    assert row["arm_scores"]["weighted_supported"] is None
    assert row["delta_bounds"]["weighted_supported_minus_anchor"] == [-1/256,1/256]
    assert row["delta_bounds"]["teacher_minus_weighted_supported"] == [0.,0.]
    block[0]["outcomes"][4]["wdl"]=0
    for audit in (False,True):
        with pytest.raises(ValueError): a._score(specs()[0],choice,block,audit=audit)


@pytest.mark.parametrize("sums,choice,regret",[
    ([100,0,0,80,90],3,0),      # A better outsider can still be irrelevant to the shortlist.
    ([100,0,0,80,110],3,10),
    ([100,0,0,105,110],3,5),
    ([-20,-30,-40,-50,-10],3,10),
    ([1,2,3],-1,0),
    ([0,0,0,10],3,0),
    ([100,0,0,80,110],4,0),
])
def test_marginal_regret_counts_only_lost_shortlist_opportunity(sums,choice,regret):
    rows=[{"id":"one","pair":8,"old_sums_wdl":sums}]
    report={"rows":[{"id":"one","pair":8,"choice":choice,"marginal_shortlist_regret_units":regret}],
            "marginal_shortlist":{"value_scale":"expected_score_integer_wdl_sum_units_divided_by_256","rows":1,"regret_units_sum":regret,"initial_regret_units_sum":regret,
                "mean_score_regret":regret/256,"regret_capture_vs_initializer":0. if regret else None}}
    assert a._marginal_regret(rows,report) == a._marginal_regret(rows,report,audit=True)
    report["rows"][0]["marginal_shortlist_regret_units"]+=1
    for audit in (False,True):
        with pytest.raises(ValueError,match="Marginal shortlist regret"): a._marginal_regret(rows,report,audit=audit)


@pytest.mark.parametrize("damage",["choice","id","pair","summary","source"])
def test_marginal_regret_rejects_invalid_choice_source_or_summary(damage):
    rows=[{"id":"one","pair":8,"old_sums_wdl":[0,0,0,10,20]}]
    report={"rows":[{"id":"one","pair":8,"choice":3,"marginal_shortlist_regret_units":10}],
            "marginal_shortlist":{"value_scale":"expected_score_integer_wdl_sum_units_divided_by_256","rows":1,"regret_units_sum":10,"initial_regret_units_sum":10,
                "mean_score_regret":10/256,"regret_capture_vs_initializer":0.}}
    if damage == "choice": report["rows"][0]["choice"]=0
    elif damage == "id": report["rows"][0]["id"]="elsewhere"
    elif damage == "pair": report["rows"][0]["pair"]=9
    elif damage == "summary": report["marginal_shortlist"]["regret_capture_vs_initializer"]=.75
    else: rows[0]["old_sums_wdl"][4]=129
    for audit in (False,True):
        with pytest.raises(ValueError): a._marginal_regret(rows,report,audit=audit)


def fixture_panel(tmp_path,monkeypatch):
    torch.manual_seed(932);torch.set_num_threads(1)
    config=LegalActionAttentionScorerConfig(state_hidden_size=8,action_hidden_size=8,scorer_hidden_size=8,
        state_depth=1,action_depth=1,attention_layers=1,attention_heads=2,attention_feedforward_size=16)
    model=LegalActionAttentionScorerModel(config);model.eval()
    registry={"manifest":"fixture","models":{},"fits":[]}
    for fit in a.c.FITS:
        checkpoint=tmp_path/f"model_{fit}.pt";save_model(checkpoint,model,1784,{"fit":fit})
        ref=a.c.artifact(checkpoint,source_fit=fit)
        for family in a.c.MODEL_ARMS: registry["models"][f"{family}_{fit}"]=ref
        initial_records=[{"id":"fixture:0","pair":0,"choice":3,"marginal_shortlist_regret_units":128},
                         {"id":"fixture:1","pair":1,"choice":-1,"marginal_shortlist_regret_units":0}]
        final_records=copy.deepcopy(initial_records);final_records[0].update(choice=4,marginal_shortlist_regret_units=0)
        initial=tmp_path/f"initial_{fit}.json";a.old.save_new(initial,{"added":{"rows":initial_records,
            "marginal_shortlist":{"value_scale":"expected_score_integer_wdl_sum_units_divided_by_256","initial_regret_units_sum":128,"regret_units_sum":128,
                "mean_score_regret":.25,"regret_capture_vs_initializer":0.,"rows":2}}})
        training=tmp_path/f"training_{fit}.json";a.old.save_new(training,{"added":{"rows":final_records,
            "regret_capture_vs_initializer":.1,"marginal_shortlist":{"value_scale":"expected_score_integer_wdl_sum_units_divided_by_256","initial_regret_units_sum":128,
            "regret_units_sum":0,"mean_score_regret":0.,"regret_capture_vs_initializer":1.,"rows":2}}})
        path=tmp_path/f"fit_{fit}.json";a.old.save_new(path,{"fit":fit,"selected_training":a.c.artifact(training),
            "checkpoints":[{"update":0,"training":a.c.artifact(initial)}]})
        registry["fits"].append(a.c.artifact(path))
    a.old.save_new(tmp_path/"models_lock.json",registry)
    monkeypatch.setattr(learning,"validate_models",lambda binding,base:a.old.read_json(Path(base)/"models_lock.json"))
    monkeypatch.setattr(source,"control_models",lambda binding,base:{k:v for k,v in registry["models"].items() if not k.startswith("weighted_")})
    state=private_state();encoded,canonical,legal=encode_position(state,config)
    reps=a.old.actions_by_canonical_successor_key(state,legal)
    all_members=[torch.where(encoded["groups"] == j)[0] for j in range(len(canonical))]
    scores=a.prior._predict({**encoded,"members":all_members},model)["offered_group_scores"]
    order=list(range(3))+sorted(range(3,len(canonical)),key=lambda j:(-scores[j],j))
    rows,inputs,descriptors=[],[],[]
    for index,n in enumerate((8,3)):
        chosen=order[:n];keys=[repr(canonical[j]) for j in chosen]
        candidates={"keys":keys,"actions":[a.old.serialize_action(reps[canonical[j]]) for j in chosen]}
        row={**encoded,"id":f"fixture:{index}","index":index,"source_fit":161001+index,
             "split":"validation","pair":index,"members":[all_members[j] for j in chosen],
             "old_sums_wdl":[0,0,0,0,128,-128,0,0][:n],"outsider_start":min(3,n),
             "outsider_support":max(0,n-3),"outsider_teacher":4 if n > 3 else -1,"outsider_label":1 if n > 3 else -1}
        spec={"index":index,"fit":row["source_fit"],"source_rowid":row["id"],"split":"validation",
              "pair":index,"game_index":2*index,"seat":0,"phase":1,"privacy":{"own":1,"opponent":1}}
        snapshot=a.old.serialize_state_snapshot(state)
        old_worlds=worlds([0,0,0,0,1,-1,0,0][:n]);predicted=proposals(n)
        selected=original_verify.independent_selection(old_worlds,predicted,n)
        identity={"spec":spec,"snapshot":snapshot,"candidates":candidates,"proposals":predicted}
        block={"worlds":old_worlds}
        lock={"choices":selected["choices"],"identity_digest":a.old.digest(identity),
              "old_selection_sha256":a.old.digest(block),"selection_statistics":selected}
        path=tmp_path/f"selection_{index}.json"
        a.old.save_new(path,{"manifest":a.c.SOURCE_BINDING,"stage":"main","index":index,
                            "identity":identity,"selection_lock":lock,"old_selection":block})
        selref=a.c.artifact(path)
        item={"index":index,"spec":spec,"snapshot":snapshot,"candidates":candidates,
              "old_sums_wdl":selected["sums_wdl"],"old_choices":selected["choices"],"selection_artifact":selref}
        fresh=worlds([1]+[0]*(n-1))
        for j,w in enumerate(fresh): w["seed"]=2211000000+index*512+j
        root={"manifest":a.c.SOURCE_BINDING,"stage":"main","spec":spec,"candidates":candidates,
              "selection_lock":lock,"selection_artifact":selref,
              "assessment":{"candidate_indices":list(range(n)),"world_seeds":[w["seed"] for w in fresh],"worlds":fresh},
              "summary":original_verify.independent_summary(fresh,selected["choices"])}
        path=tmp_path/f"source_root_{index}.json";a.old.save_new(path,root);rootref=a.c.artifact(path)
        path=tmp_path/f"source_audit_{index}.json";a.old.save_new(path,{"manifest":a.c.SOURCE_BINDING,"root":rootref,"passed":True})
        descriptors.append({"index":index,"spec":spec,"root":rootref,"audit":a.c.artifact(path),"selection":selref})
        rows.append(row);inputs.append(item)
    monkeypatch.setattr(source,"load_panel_rows",lambda binding,base:rows)
    monkeypatch.setattr(source,"old_inputs",lambda binding,base:inputs)
    monkeypatch.setattr(source,"evaluation_sources",lambda binding,base:descriptors)
    monkeypatch.setattr(source,"load_rows",lambda binding,split,base:rows)
    monkeypatch.setattr(source,"load_pools",lambda *args:pytest.fail("Acquisition should only load the added720 pool"))
    monkeypatch.setitem(a.c.CONFIG,"bootstrap_samples",19)
    synthetic=population_rows(False)
    summarize,independent=st.summarize,st.independent_summarize
    monkeypatch.setattr(st,"summarize",lambda rows,boundary=None:summarize(synthetic,boundary))
    monkeypatch.setattr(st,"independent_summarize",lambda rows,boundary=None:independent(synthetic,boundary))
    return rows,inputs,descriptors


def test_complete_assessment_locks_all_models_and_validation_before_outcomes(tmp_path,monkeypatch):
    rows,inputs,descriptors=fixture_panel(tmp_path,monkeypatch)
    original=a.c.verified;seen=[]
    roots={r["root"]["path"] for r in descriptors}
    def guarded(ref):
        if ref["path"] in roots:
            lock=a.old.read_json(tmp_path/"predictions_lock.json")
            assert lock["complete"] and len(lock["predictions"]) == len(rows)
            validation=original(lock["validation_old_targets"])
            assert set(validation["families"]) == set(a.c.MODEL_ARMS)
            assert all(len(family["per_fit"]) == 3 for family in validation["families"].values())
            seen.append(ref["path"])
        return original(ref)
    monkeypatch.setattr(a.c,"verified",guarded)
    ref=a.assess("fixture",tmp_path)
    assert len(seen) == 2
    checked=a.c.verified(a.audit("fixture",ref,tmp_path))
    assert checked["passed"] and checked["fixed_controls_reproduced"]
    assert checked["full_validation_old_label_metrics_independently_reproduced"]
    assert checked["marginal_shortlist_regret_independently_reproduced"]
    assert checked["acquisition_marginal_regret_independently_reproduced"]
    result=a.c.verified(ref);validation=a.c.verified(result["validation_old_targets"])
    assert result["acquisition"] == dict.fromkeys(map(str,a.c.FITS),1.)
    assert validation["all_models_on_all_rows"]
    assert all(r["states"] == 2 for family in validation["families"].values() for r in family["per_fit"].values())
    with pytest.raises(ValueError,match="Preserve"): a.assess("fixture",tmp_path)


@pytest.mark.parametrize("damage",["missing_seal","missing_control","wrong_fit","wrong_control_ref","changed_old_input"])
def test_preassessment_seal_identity_and_old_inputs_required(tmp_path,monkeypatch,damage):
    rows,inputs,descriptors=fixture_panel(tmp_path,monkeypatch)
    if damage == "missing_seal":
        with pytest.raises(ValueError,match="Seal"): a._open_assessment(descriptors[0],{}, {},tmp_path)
        return
    if damage == "changed_old_input": inputs[0]["old_sums_wdl"][0]+=1
    else:
        path=tmp_path/"models_lock.json";reg=a.old.read_json(path)
        if damage == "missing_control": reg["models"].pop("broad_161001")
        elif damage == "wrong_fit": reg["models"]["weighted_161001"]["source_fit"]=161002
        else: reg["models"]["broad_161001"]=reg["models"]["anchor_161002"]
        a.old.atomic_write(path,reg)
    with pytest.raises(ValueError): a.assess("fixture",tmp_path)
    assert not (tmp_path/"predictions_lock.json").exists()


@pytest.mark.parametrize("damage",["choice","model","timing","row","statistic","gate","validation"])
def test_auditor_rejects_rehashed_result_and_prediction_corruptions(tmp_path,monkeypatch,damage):
    fixture_panel(tmp_path,monkeypatch);ref=a.assess("fixture",tmp_path)
    path=tmp_path/"assessment_result.json";result=a.c.verified(ref)
    if damage in ("choice","model"):
        lockpath=tmp_path/"predictions_lock.json";lock=a.old.read_json(lockpath)
        if damage == "choice": lock["predictions"][0]["choices"]["weighted_outsider"]=7
        else: lock["predictions"][0]["models"]["weighted"]=lock["predictions"][1]["models"]["weighted"]
        a.old.atomic_write(lockpath,lock);result["predictions"]=a.c.artifact(lockpath)
        result["assessment_opened_at_ns"]=a.time.time_ns()
    elif damage == "timing": result["assessment_opened_at_ns"]=1
    elif damage == "row": result["rows"][0]["delta_bounds"][st.NAMES[0]][0]+=.01
    elif damage == "statistic": result["statistics"]["groups"]["validation"]["contrasts"][st.NAMES[0]]["ci95"][0]+=1e-9
    elif damage == "gate": result["development_screen"]["passed"]=not result["development_screen"]["passed"]
    else:
        vp=tmp_path/"validation_old_targets.json";v=a.old.read_json(vp)
        v["families"]["weighted"]["per_fit"]["161001"]["pair_weighted"]["mean_old_score_regret"]+=.01
        a.old.atomic_write(vp,v);result["validation_old_targets"]=a.c.artifact(vp)
    a.old.atomic_write(path,result)
    with pytest.raises(ValueError): a.audit("fixture",a.c.artifact(path),tmp_path)


def test_real_old_inputs_and_saved_anchor_broad_lawful_inference_no_assessment(monkeypatch):
    freeze=a.old.read_json(a.c.SOURCE/"freeze.json");manifest=a.c.verified(freeze["manifest"])
    schedule=a.c.verified(manifest["schedule"])["main"]
    registry=a.old.read_json(a.c.PARENT/"models_lock.json")
    for fit in a.c.FITS:
        spec=next(r for r in schedule if r["fit"] == fit and r["split"] == "train")
        original=a.c.verified(spec["source_root"])
        export=torch.load(a.c.verify_hash(spec["source_export"]),map_location="cpu",weights_only=True)
        row=next(r for r in export if r["id"] == spec["source_rowid"])
        item={"snapshot":original["snapshot"],"candidates":original["candidates"]}
        for family in ("anchor","broad"):
            model=a.StrongPolicyInference(a.c.verify_hash(registry["models"][f"{family}_{fit}"])).model
            expected=a.prior._predict(row,model)
            actual=a.prior._independent_proposal(row,item,model)
            a.compare(expected,actual)
            changed=a.prior.input_audit.hidden_alternative(a.old.deserialize_state_snapshot(item["snapshot"]))
            a.compare(actual,a.prior._independent_proposal(row,item,model,state=changed))

"""Corrected loader and fitter admission tests; no actual network or game run."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from tools.splendorust_selfplay_pilot_v2_20261102 import data as d, training as t, capture_adapter as a, design
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import bridge


def synthetic_row(case=None):
    """Synthetic data plumbing only; outputs are explicitly not model evidence."""
    if case is None:
        case = d.build_qualification_cases()["cases"][0]
    before = case["snapshot"]; state = d.guard.decode_training_snapshot(before)
    obs = rules.observation_payload(state)
    actions, indices, mask = a.main_support(obs)
    main = actions[0]
    legal = d.guard.legal_actions_for_snapshot(before)
    selected = next(x for x in legal if bridge._main(x) == main)
    forced = main == {"kind": "Pass"}
    excluded = forced or len(actions) == 1 or obs["observation"]["turn_index"] >= 124
    native = None if forced else dict(enabled=True, phase="Main", quota=512, authoritative_indices67=indices,
        retained_indices67=indices if excluded else [indices[0]], callback_count=0 if excluded else 2,
        root_callback_index=None if excluded else 0, root_callback_role=None if excluded else "root", simulations=0 if excluded else 512,
        visits67=None, q67=None, root_value=None, network_value=None, policy67=None)
    root = None
    if not excluded:
        visits = [0] * 67; visits[indices[0]] = 512
        native.update(visits67=visits, q67=[.5 if x else None for x in visits], root_value=.5, network_value=.5,
                      policy67=[float(x / 512) for x in visits])
        tokens = a.public_root_tokens(obs); outputs = [[0.] * 83]
        root = dict(tokens=tokens, outputs=outputs, batch=1, input_sha256=d.digest(tokens), output_sha256=d.digest(outputs))
    capture = a.capture_record(obs, 512, native, root, main)
    phase = dict(action=main, neural_callback_count=0 if excluded else 2, simulations=0 if excluded else 512,
                 inference=[] if excluded else [root, {"batch": 1, "input_sha256": "0" * 64, "output_sha256": "0" * 64}])
    trace = dict(phases=[] if forced else [phase], forced_pass=forced)
    action = rules.serialize_action(selected)
    return dict(spec={"purpose": "synthetic_model_free_test"}, ply=0, before=deepcopy(before), observer=obs, action=action,
                after=rules.serialize_snapshot(rules.CorrectedEnv().step(state, selected)), training_capture=capture, trace=trace,
                clock=dict(status="ok", accepted=True, result=dict(action=action, diagnostics=trace)))


def test_actual_loader_all_corrected_cases_and_negative_contracts_without_torch():
    plan = d.build_qualification_cases()
    rows = [synthetic_row(x) for x in plan["cases"] + [plan["private_variant"]]]
    was_loaded = "torch" in sys.modules
    proof = d.qualify_captured_rows(plan, rows)
    assert ("torch" in sys.modules) == was_loaded
    assert proof["six_corrected_faces"] == sorted(rules.CORRECTED_COSTS)
    assert len(proof["rows"]) == 16 and len(proof["actual_loader_rejections"]) == 13
    assert proof["actual_input_policy_batch"]["shape"][1:] == [31, 48]
    assert not proof["actual_input_policy_batch"]["terminal_labels_fabricated"]


@pytest.mark.parametrize("mutation", ["trace_input", "trace_output", "trace_action", "trace_count", "mask_pruned", "observer", "after", "bare_snapshot"])
def test_actual_row_rejects_unbound_or_legacy_data(mutation):
    row = synthetic_row()
    if mutation == "trace_input": row["trace"]["phases"][0]["inference"][0]["input_sha256"] = "0" * 64
    elif mutation == "trace_output": row["trace"]["phases"][0]["inference"][0]["outputs"][0][0] = .25
    elif mutation == "trace_action": row["trace"]["phases"][0]["action"] = {"kind": "Pass"}
    elif mutation == "trace_count": row["trace"]["phases"][0]["neural_callback_count"] += 1
    elif mutation == "mask_pruned": row["training_capture"]["legal_mask81"] = [x > 0 for x in row["training_capture"]["policy_target81"]]
    elif mutation == "observer": row["observer"]["catalogue_sha256"] = "old"
    elif mutation == "after": row["after"] = row["before"]
    else: row["before"] = row["before"]["state"]
    with pytest.raises((ValueError, AssertionError, KeyError)): d.decode_row(row)


def test_policy_mask_includes_pruned_legal_choices_and_only_main_channels():
    example = d.decode_row(synthetic_row())
    assert sum(example["mask"]) > sum(x > 0 for x in example["policy"])
    assert not any(example["mask"][60:]) and not any(example["policy"][60:])
    assert example["canonical_players"] == [example["actor"], 1 - example["actor"]]


@pytest.mark.parametrize("winner,actor,expected", [(0,0,[1.,-1.]),(0,1,[-1.,1.]),(1,0,[-1.,1.]),(1,1,[1.,-1.]),(None,0,[0.,0.]),(None,1,[0.,0.])])
def test_canonical_values_use_both_absolute_seats(monkeypatch,winner,actor,expected):
    monkeypatch.setattr(d.guard,"require_official_terminal_target",lambda snapshot,reason:winner)
    assert d.signed_outcomes({},"completed",actor)==expected


@pytest.mark.parametrize("reason", ["timeout", "deadlock", "turn_limit", "infrastructure_failure", "resignation"])
def test_nonofficial_or_unfinished_games_never_make_value_targets(reason):
    with pytest.raises((ValueError, AssertionError)): d.signed_outcomes(synthetic_row()["before"],reason,0)


def dummy_manifest():
    specs=design.schedule()
    items=[dict(path=f"synthetic/g{i}.json",bytes=1,sha256=str(i)*64) for i in range(4)]
    return d.build_manifest(items,specs,qualification_only=True),specs


def test_training_loader_does_not_open_or_decode_heldout_before_checkpoint(monkeypatch):
    manifest,specs=dummy_manifest();opened=[]
    def reader(item,**kwargs):
        opened.append(item["path"])
        assert item["path"] not in ("synthetic/g2.json","synthetic/g3.json"), "heldout was opened"
        return {"id":int(item["path"].split("g")[-1].split(".")[0])}
    def loader(document,spec,**kwargs):
        return dict(spec=spec,examples=[dict(split=spec["split"],game_index=spec["game_index"])],eligible_roots=1)
    monkeypatch.setattr(d,"read_artifact",reader);monkeypatch.setattr(d,"load_game",loader)
    result=d.load_corpus(manifest,expected_specs=specs,include_splits=["fixture_train"])
    assert opened==["synthetic/g0.json","synthetic/g1.json"] and result["opened_splits"]==["fixture_train"]
    assert len(result["examples"])==2


@pytest.mark.parametrize("mutation", ["missing_metadata","legacy_catalogue","old_origin","relabelled_inventory","duplicate_seed","wrong_split"])
def test_corpus_rejects_historical_or_changed_inventory(mutation):
    manifest,specs=dummy_manifest()
    if mutation=="missing_metadata":manifest["metadata"].pop("ruleset_id")
    elif mutation=="legacy_catalogue":manifest["metadata"]["catalogue_sha256"]="old"
    elif mutation=="old_origin":manifest["metadata"]["origin"]="historical"
    elif mutation=="relabelled_inventory":manifest["games"][0]["sha256"]="f"*64
    elif mutation=="duplicate_seed":
        specs[1]["seed_decimal"]=specs[0]["seed_decimal"];manifest=d.build_manifest(manifest["games"],specs,qualification_only=True)
    else:
        specs[3]["split"]="fixture_train";manifest=d.build_manifest(manifest["games"],specs,qualification_only=True)
    with pytest.raises((ValueError,KeyError,AssertionError)):d.validate_manifest(manifest,specs)


def test_same_game_weight_balances_roots_without_replacement():
    corpus=dict(qualification_only=True,games=[dict(spec=dict(game_index=i,split="fixture_train")) for i in (0,1)],
        examples=[dict(game_index=i,split="fixture_train") for i in [0,1,1,1]])
    examples,weights,counts=t.select_training_examples(corpus,t.qualification_plan())
    assert counts=={0:1,1:3} and sum(weights[:1])==pytest.approx(sum(weights[1:]))
    order=t.epoch_order(4,t.qualification_plan()["seed"],0)
    assert set(order)==set(range(4)) and len(order)==4
    assert order==sorted(range(4),key=lambda i:hashlib.sha256(json.dumps([202611011,0,i],separators=(",",":"),sort_keys=True).encode()).hexdigest())
    corpus["examples"]=corpus["examples"][:1]
    with pytest.raises(ValueError,match="dropped"):t.select_training_examples(corpus,t.qualification_plan())


def test_actual_tensor_conversion_and_fixed_short_batch_objective():
    import torch
    e=d.decode_row(synthetic_row());e["value"]=[1.,-1.]
    batch=t.collate([e],[1.])
    assert batch["tokens"].dtype==torch.float32 and batch["mask"].dtype==torch.bool
    assert batch["tokens"].reshape(1,-1).tolist()==[e["tokens"]]
    outputs=torch.zeros((1,83),dtype=torch.float32,requires_grad=True)
    loss,policy,value=t.loss_components(outputs,batch)
    assert float(loss)==pytest.approx((math.log(sum(e["mask"]))+1)/32,rel=1e-6)
    assert float(value[0])==1
    loss.backward();assert torch.isfinite(outputs.grad).all()
    assert outputs.grad[0,81]>0 if e["value"][0]<0 else outputs.grad[0,81]<0
    assert torch.count_nonzero(outputs.grad[0,60:81])==0


def test_original_parent_architecture_bytes_match_accepted_weight_identity():
    hashes=t.verify_parent_tensor_hashes()
    assert hashes["all"]=="32cdd9466befc527d265a356be65200c3666a4fa8a82cfefdc40b327b2f529db"
    assert sum(math.prod(x) for x in t.expected_state_shapes().values())==4_780_883


def test_fit_recipe_is_exact_and_direct_counts_exclude_search_callbacks():
    plan=t.qualification_plan();assert plan["optimizer_steps"]==1 and plan["base_direct_forward_invocations"]==53
    assert plan["direct_forward_count_excludes_entrypoint_callbacks"] and plan["learned_entrypoint_decisions"]==6
    assert t.pilot_plan()["epochs"]==2 and t.RECIPE["batch_reduction"].endswith("including_short_batch")
    bad=deepcopy(plan);bad["recipe"]["learning_rate"]=.01
    with pytest.raises(ValueError):t.validate_plan(bad)


def cli_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(t,"ROOT",tmp_path);monkeypatch.setattr(d,"ROOT",tmp_path)
    output=tmp_path/"new";output.mkdir()
    def write(name,value):
        path=output/name;path.write_text(d.canonical(value));return t.artifact(path)
    manifest,specs=dummy_manifest()
    wrapper=dict(training_plan=t.qualification_plan(),source_specs=specs,learned_entrypoint={},input_qualification={})
    plan=write("fit_plan.json",wrapper);fixtures=write("fit_parity_fixtures.json",dict(fixtures=[]))
    write("corpus_manifest.json",manifest)
    write("freeze.json",dict(phase="qualification",fit_plan=plan,fit_parity_fixtures=fixtures,training_plan={"intentionally":"bare recipe is not wrapper"}))
    args=SimpleNamespace(mode="qualification",out=str(output/"fit"),plan=str(output/"fit_plan.json"),
        fixtures=str(output/"fit_parity_fixtures.json"),manifest=str(output/"corpus_manifest.json"))
    checked=[]
    common=SimpleNamespace(SCOPE="qualification",OUT=output,verify=lambda items:checked.extend(d.read_artifact(x) for x in items))
    env=dict(STUDY_RESEARCH_FREEZE=str(output/"freeze.json"),STUDY_RESEARCH_FREEZE_SHA256=d.guard.sha256(output/"freeze.json"))
    return args,common,env,wrapper,checked


def test_actual_cli_admission_uses_wrapped_fit_plan_and_sha256(tmp_path,monkeypatch):
    args,c,env,wrapper,checked=cli_fixture(tmp_path,monkeypatch)
    folder,actual,manifest,fixtures=t.admit_fit_cli(args,common=c,environment=env)
    assert actual==wrapper and len(checked)==2 and folder==c.OUT/"fit"


@pytest.mark.parametrize("bad", ["plan","fixtures","manifest","out","hash","old_env"])
def test_actual_cli_rejects_wrong_artifact_or_old_environment(tmp_path,monkeypatch,bad):
    args,c,env,_,_=cli_fixture(tmp_path,monkeypatch)
    if bad in ("plan","fixtures","manifest","out"):setattr(args,bad,str(c.OUT/"wrong.json"))
    elif bad=="hash":env["STUDY_RESEARCH_FREEZE_SHA256"]="0"*64
    else:env["STUDY_RESEARCH_FREEZE_SHA"]=env.pop("STUDY_RESEARCH_FREEZE_SHA256")
    with pytest.raises((ValueError,KeyError)):t.admit_fit_cli(args,common=c,environment=env)


def test_actual_main_routes_exact_args_before_any_model(monkeypatch,tmp_path):
    from tools.splendorust_selfplay_pilot_v2_20261102 import common
    args,c,env,_,_=cli_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(common,"SCOPE",c.SCOPE);monkeypatch.setattr(common,"OUT",c.OUT);monkeypatch.setattr(common,"verify",c.verify)
    for k,v in env.items():monkeypatch.setenv(k,v)
    monkeypatch.setenv("SELFPLAY_MONOTONIC_DEADLINE",str(t.time.perf_counter()+30));monkeypatch.setenv("SELFPLAY_UNIX_DEADLINE",str(t.time.time()+30))
    monkeypatch.setattr(t,"startup",lambda:None)
    class StopBeforeTorch(Exception):pass
    def placement(folder):
        assert folder==c.OUT/"fit"
        raise StopBeforeTorch
    monkeypatch.setattr(t,"place_fitter",placement)
    monkeypatch.setattr(sys,"argv",["training","--mode","qualification","--manifest",args.manifest,"--plan",args.plan,"--fixtures",args.fixtures,"--out",args.out])
    with pytest.raises(StopBeforeTorch):t.main()


def test_import_and_counterexample_preparation_stay_out_of_torch():
    code="from tools.splendorust_selfplay_pilot_v2_20261102 import data,training; import sys; data.build_qualification_cases(); assert 'torch' not in sys.modules; print('model-free')"
    process=subprocess.run([sys.executable,"-B","-c",code],cwd=d.ROOT,text=True,capture_output=True,timeout=20)
    assert process.returncode==0,process.stderr
    assert process.stdout.strip()=="model-free"


def test_offline_opening_chronology_in_actual_fit_source():
    source=inspect.getsource(t._run_fit)
    assert source.index("write_checkpoint(")<source.index('heldout = data.load_corpus(')
    assert 'include_splits=[plan["training_split"]]' in source
    assert source.index('qualify_saved_tensor_inputs(')<source.index('train_step(')


def pilot_specs():
    return [dict(game_index=i,seed_decimal=str(100000+i),teacher_simulations=512,
                 split="train" if i<192 else "validation" if i<224 else "offline",
                 purpose="single_generation_learning_pilot") for i in range(256)]


def toy_example(game,ply=0,split="train",win=True):
    return dict(game_index=game,ply=ply,split=split,row_sha256=d.digest([game,ply,split]),
        root_input_sha256=d.digest(["tokens",game,ply]),tokens=[0.]*1488,
        mask=[True,True]+[False]*79,policy=[.5,.5]+[0.]*79,
        value=[1.,-1.] if win else [0.,0.])


def test_exact_pilot_split_inventory_and_no_qualification_relabeling():
    specs=pilot_specs()
    items=[dict(path=f"synthetic/p{i}.json",bytes=1,sha256=d.digest(i)) for i in range(256)]
    manifest=d.build_manifest(items,specs,qualification_only=False)
    assert d.validate_manifest(manifest,specs)==specs or manifest["qualification_only"] is False
    wrong=deepcopy(manifest);wrong["qualification_only"]=True
    with pytest.raises(ValueError):d.validate_manifest(wrong,specs)
    specs[-1]["split"]="train"
    with pytest.raises(ValueError):d.validate_manifest(d.build_manifest(items,specs,qualification_only=False),specs)


def test_streaming_corpus_retains_counts_without_examples_or_heldout_opening(monkeypatch):
    manifest,specs=dummy_manifest();opened=[]
    monkeypatch.setattr(d,"read_artifact",lambda item,**kw:opened.append(item["path"]) or {})
    def load(document,spec,**kw):
        return dict(spec=spec,examples=[toy_example(spec["game_index"],j,spec["split"]) for j in range(spec["game_index"]+1)])
    monkeypatch.setattr(d,"load_game",load)
    out=d.load_corpus(manifest,expected_specs=specs,include_splits=["fixture_train"],retain_examples=False)
    assert not out["examples"] and out["example_counts"]=={"fixture_train":3}
    assert out["game_counts"]=={"fixture_train":2} and len(opened)==2
    assert not any("g2" in x or "g3" in x for x in opened)


def test_pilot_game_weights_and_complete_without_replacement_epochs():
    examples=[toy_example(i,j) for i in range(192) for j in range(2 if i==0 else 1)]
    corpus=dict(qualification_only=False,games=[dict(spec=x) for x in pilot_specs()],examples=examples)
    actual,weights,counts=t.select_training_examples(corpus,t.pilot_plan())
    assert len(actual)==193 and len(counts)==192
    assert sum(weights[:2])==pytest.approx(weights[2])
    assert math.fsum(weights)==pytest.approx(193.)
    a0=t.epoch_order(193,t.pilot_plan()["seed"],0);a1=t.epoch_order(193,t.pilot_plan()["seed"],1)
    assert sorted(a0)==sorted(a1)==list(range(193)) and a0!=a1
    corpus["examples"]=examples[:-1]
    with pytest.raises(ValueError,match="dropped"):t.select_training_examples(corpus,t.pilot_plan())


def test_exact_forward_accounting_uses_real_split_roots_and_short_batches():
    q=t.direct_forward_budget(t.qualification_plan(),{"fixture_train":112,"fixture_validation":58,"fixture_assessment":56})
    assert q["direct_forward_invocations"]==61 and q["heldout_forward_calls"]==8
    p=t.direct_forward_budget(t.pilot_plan(),{"train":193,"validation":33,"offline":32})
    assert p["training_forward_backward_calls"]==14 and p["heldout_forward_calls"]==6 and p["direct_forward_invocations"]==72
    assert t.direct_forward_budget(t.pilot_plan())["direct_forward_invocations"]==32052
    with pytest.raises(ValueError):t.direct_forward_budget(t.pilot_plan(),{"train":193,"validation":0,"offline":32})


def update_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(t,"ROOT",tmp_path)
    folder=tmp_path/"fit";plan=t.pilot_plan()
    examples=[toy_example(i,j) for i in range(192) for j in range(2 if i==0 else 1)]
    corpus=dict(qualification_only=False,games=[dict(spec=x) for x in pilot_specs()],examples=examples)
    examples,weights,_=t.select_training_examples(corpus,plan);step=0;orders=[]
    for epoch in range(2):
        order=t.epoch_order(len(examples),plan["seed"],epoch);orders.append(d.digest(order))
        t.save_json(folder/"orders"/f"{epoch:03d}.json",dict(epoch=epoch,indices=order,sha256=d.digest(order)))
        for offset in range(0,len(order),32):
            indices=order[offset:offset+32];outputs=[[0.]*83 for _ in indices]
            losses=[t.prediction_losses(t.prediction_row(examples[i],out)) for i,out in zip(indices,outputs)]
            record=dict(epoch=epoch,step=step,example_indices=indices,batch_examples=len(indices),forward_backward_calls=1,batch_denominator=32,
                batch_source_sha256=d.digest([[examples[i]["game_index"],examples[i]["ply"],examples[i]["row_sha256"],weights[i]] for i in indices]),
                outputs83=outputs,loss=math.fsum(x["combined"]*weights[i] for i,x in zip(indices,losses))/32,
                policy_mean_unweighted=math.fsum(x["policy"] for x in losses)/len(losses),value_mean_unweighted=1.,gradient_norm_before_clip=.2,seconds=.1)
            t.save_json(folder/"steps"/f"{step:06d}.json",record);step+=1
    return folder,examples,weights,plan,dict(epoch_order_sha256=orders,optimizer_steps=step)


def test_saved_two_epoch_verifier_covers_all_batches_and_partial_final_batch(tmp_path,monkeypatch):
    args=update_fixture(tmp_path,monkeypatch)
    assert t.verify_update_records(*args)==14
    assert json.loads((args[0]/"steps/000006.json").read_text())["batch_examples"]==1
    assert json.loads((args[0]/"steps/000013.json").read_text())["batch_denominator"]==32


@pytest.mark.parametrize("mutation",["missing","extra","duplicate","short_denominator","loss","weight","nonfinite","epoch_order","step_count"])
def test_saved_two_epoch_verifier_rejects_dropped_changed_or_invented_updates(tmp_path,monkeypatch,mutation):
    args=update_fixture(tmp_path,monkeypatch);folder=args[0]
    path=folder/"steps/000013.json";record=json.loads(path.read_text())
    if mutation=="missing":path.unlink()
    elif mutation=="extra":t.save_json(folder/"steps/000014.json",record)
    elif mutation=="epoch_order":
        p=folder/"orders/001.json";r=json.loads(p.read_text());r["indices"][0]=r["indices"][1];p.write_text(json.dumps(r))
    elif mutation=="step_count":args[-1]["optimizer_steps"]-=1
    else:
        if mutation=="duplicate":record["example_indices"]=[0]
        elif mutation=="short_denominator":record["batch_denominator"]=1
        elif mutation=="loss":record["loss"]+=.1
        elif mutation=="weight":record["batch_source_sha256"]="f"*64
        else:record["outputs83"][0][0]=float("nan")
        path.write_text(json.dumps(record))
    with pytest.raises((ValueError,FileNotFoundError)):t.verify_update_records(*args)


def test_heldout_summary_weights_games_not_correlated_roots():
    rows=[t.prediction_row(toy_example(0,0,"offline",win=False),[0.]*83)]
    rows += [t.prediction_row(toy_example(1,i,"offline",win=True),[0.]*83) for i in range(3)]
    result=t.summarize_predictions(rows,[0,1])
    assert result["root_count"]==4 and result["mean_game_losses"]["value"]==.5
    with pytest.raises(ValueError,match="every scheduled"):t.summarize_predictions(rows,[0,1,2])


def heldout_fixture(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(t,"ROOT",tmp_path)
    plan=t.pilot_plan();specs=pilot_specs()[192:]
    examples=[toy_example(x["game_index"],j,x["split"]) for x in specs for j in range(2 if x["game_index"] in (192,224) else 1)]
    heldout=dict(games=[dict(spec=x) for x in specs],examples=examples)
    class Dummy:
        def __init__(self,name):self.name=name
        def eval(self):return self
    monkeypatch.setattr(t,"load_parent",lambda:Dummy("parent"))
    monkeypatch.setattr(t,"state_digest",lambda m:m.name)
    calls=[]
    def fake_forward(model,tokens):
        calls.append((model.name,len(tokens)))
        result=torch.zeros((len(tokens),83),dtype=torch.float32)
        if model.name=="candidate":result[:,81]=.5;result[:,82]=-.5
        return result
    monkeypatch.setattr(t,"tensor_forward",fake_forward)
    folder=tmp_path/"heldout";checkpoint={"path":"fixed_candidate","sha256":"a"*64,"bytes":1}
    result=t.run_heldout_diagnostics(Dummy("candidate"),heldout,plan,checkpoint,folder)
    return folder,heldout,plan,checkpoint,{"parent":"parent","candidate":"candidate"},result,calls


def test_full_heldout_path_saved_outputs_reconstructs_all64_games_without_network(tmp_path,monkeypatch):
    args=heldout_fixture(tmp_path,monkeypatch)
    actual=t.verify_heldout_diagnostics(*args[:5])
    assert actual==args[5] and actual["forward_calls"]==len(args[6])==8
    assert sum(count for _,count in args[6])==132
    assert actual["summaries"]["parent"]["offline"]["game_count"]==32
    assert actual["summaries"]["candidate"]["validation"]["mean_game_losses"]["value"]==.25
    assert actual["candidate_minus_parent_mean_game_losses"]["offline"]["value"]==-.75
    assert actual["strength_claim"] is False and actual["checkpoint_selection"] is False


@pytest.mark.parametrize("mutation",["mask","target","game","missing","extra","weights","summary"])
def test_heldout_verifier_rejects_incorrect_inputs_coverage_or_losses(tmp_path,monkeypatch,mutation):
    args=heldout_fixture(tmp_path,monkeypatch);folder=args[0]
    path=folder/"candidate/offline/000000.json";value=json.loads(path.read_text())
    if mutation=="missing":path.unlink()
    elif mutation=="extra":t.save_json(folder/"candidate/offline/000002.json",value)
    elif mutation=="summary":
        p=folder/"result.json";r=json.loads(p.read_text());r["summaries"]["candidate"]["offline"]["mean_game_losses"]["value"]=0.;p.write_text(json.dumps(r))
    else:
        if mutation=="mask":value["rows"][0]["legal_mask81"][2]=True
        elif mutation=="target":value["rows"][0]["value_target2"]=[0.,0.]
        elif mutation=="game":value["rows"][0]["game_index"]+=1
        elif mutation=="weights":value["weights_sha256"]="other"
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError,FileNotFoundError)):t.verify_heldout_diagnostics(*args[:5])


def test_current_capture_tensor_qualification_uses_real_collator_but_no_model(tmp_path,monkeypatch):
    monkeypatch.setattr(t,"ROOT",tmp_path);monkeypatch.setattr(d,"ROOT",tmp_path)
    study=tmp_path/"new";cases=d.build_qualification_cases()
    rows=[synthetic_row(x) for x in cases["cases"]+[cases["private_variant"]]]
    paths=[study/"capture_fixtures/capture_on"/f"{i:03d}.json" for i in range(16)]
    for p,row in zip(paths,rows):t.save_json(p,row)
    t.save_json(study/"capture_qualification.json",{"actual_loader":d.qualify_captured_rows(cases,rows)})
    item=t.save_json(study/"qualification_plan.json",{"loader_cases":cases})
    binding=dict(source="current_frozen_job",plan=item,row_paths=[p.relative_to(tmp_path).as_posix() for p in paths],proof_path="new/capture_qualification.json")
    actual=t.qualify_saved_tensor_inputs(study/"fit",binding,materialize=True)
    assert actual==t.qualify_saved_tensor_inputs(study/"fit",binding,materialize=False)
    assert actual["rows"]==13 and actual["no_network_forward_or_optimizer_update"]
    bad=deepcopy(binding);bad["row_paths"][0]="old/000.json"
    with pytest.raises(ValueError,match="current-run"):t.qualify_saved_tensor_inputs(study/"fit",bad,materialize=False)


def test_cli_rejects_wrong_job_scope_before_model(tmp_path,monkeypatch):
    args,c,env,_,_=cli_fixture(tmp_path,monkeypatch);c.SCOPE="pilot"
    with pytest.raises(ValueError,match="scope"):t.admit_fit_cli(args,common=c,environment=env)

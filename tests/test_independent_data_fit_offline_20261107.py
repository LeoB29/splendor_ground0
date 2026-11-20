from copy import deepcopy
import struct
import sys
import pytest
from tools.splendorust_independent_data_20261107 import fit, offline as o, sampling as s


def test_exact_accepted_math_and_no_tensor_import():
    assert fit.source_equivalence()["status"]=="unchanged_joint_update_math_verified"
    assert "torch" not in sys.modules


def metadata():
    return dict(**s.fit_identity(0,"broad"),**s.IDENTITY,study_id=s.STUDY,checkpoint_step=96,
        qualification_only=False,historical_data_relabelled=False,historical_warm_start_sha256=fit.accepted.PARENT_SHA256,
        training_records=3072,record_presentations=3072,fixed_final_step=96,complete_passes=1,
        source_training_games=1024,roots_per_game=3,parameter_count=fit.accepted.PARAMETERS,
        diagnostic_gradient_probes=0,checkpoint_selection=False,seed=s.SEEDS[0])


@pytest.mark.parametrize("field,value",[("checkpoint_step",96.),("repetition",False),
    ("training_records",3072.),("complete_passes",True),("source_training_games",192),
    ("roots_per_game",16),("qualification_only",0),("checkpoint_selection",1)])
def test_strict_checkpoint_metadata(field,value):
    m=metadata();fit.validate_checkpoint_metadata(m);m[field]=value
    with pytest.raises(ValueError):fit.validate_checkpoint_metadata(m)


def test_strict_complete_training_batch_inventory():
    loaded=dict(train_examples=[dict(split="train") for _ in range(3072)],train_weights=[1.]*3072,
                batches=[list(range(i,i+32)) for i in range(0,3072,32)],probe_batches=[])
    assert fit.validate_loaded(loaded) is loaded
    loaded["batches"][0][0]=False
    with pytest.raises(ValueError):fit.validate_loaded(loaded)


def f32(x):return struct.unpack("<f",struct.pack("<f",x))[0]


def row(game,split,logit=0.,value=0.,ply=0):
    return dict(game_index=game,ply=ply,split=split,root_input_sha256=s.digest([game,ply,"root"]),
        row_sha256=s.digest([game,ply,"row"]),legal_mask81=[True,True]+[False]*79,
        policy_target81=[1.,0.]+[0.]*79,value_target2=[0.,0.],
        output83=[f32(logit)]+[0.]*80+[f32(value),f32(-value)])


def bundles():
    return {model:{"train":[],"validation":[row(0,"validation",.25 if "broad" in model else 0.),row(1,"validation",.5 if "broad" in model else 0.)],
                  "offline":[row(2,"offline",.75 if "broad" in model else 0.),row(3,"offline",1. if "broad" in model else 0.)]}
            for model in o.MODEL_IDS}


def analyze(b):return o.analyze_predictions(b,draws=100,seed=123,strict=False,expected_games={"validation":[0,1],"offline":[2,3]})


def test_primary_sign_and_stratified_same_games():
    result=analyze(bundles())
    assert result["primary"]["difference"]<0 and result["primary"]["ci95"][1]<0
    assert result["primary"]["metric"]=="policy" and result["primary"]["split"]=="pooled"
    assert sum(x["primary"] for x in result["contrasts"])==1
    assert result["independent_heldout_games"]==4 and result["fixed_fits"]==4
    assert result["bootstrap"]["same_draws_all_models"]
    assert result["checkpoint_selection"] is False and result["offline_strength_admission_gate"] is False
    means=[x["difference"] for x in result["contrasts"] if x["contrast"]=="broad_minus_narrow" and x["metric"]=="policy" and x["split"]!="pooled"]
    assert result["primary"]["difference"]==sum(means)/2


def test_bootstrap_repeatable_and_strata_distinct():
    a=analyze(bundles());b=analyze(bundles());assert a==b
    assert a["bootstrap"]["strata"]["validation"]["index_sha256"]!=a["bootstrap"]["strata"]["offline"]["index_sha256"]


def test_equal_game_mean_not_root_weighted():
    b=bundles()
    for model in o.MODEL_IDS:
        first=b[model]["validation"][0]
        b[model]["validation"]=[dict(deepcopy(first),ply=i) for i in range(5)]+[b[model]["validation"][1]]
    assert analyze(b)["primary"]["difference"]==analyze(bundles())["primary"]["difference"]


@pytest.mark.parametrize("mutation",[
    lambda b:b.pop("r1_narrow_joint"),
    lambda b:b["r0_broad_joint"]["validation"].pop(),
    lambda b:b["r0_broad_joint"]["validation"][0].__setitem__("game_index",False),
    lambda b:b["r0_broad_joint"]["validation"][0].__setitem__("value_target2",[True,-1.]),
    lambda b:b["r0_broad_joint"]["validation"][0]["output83"].__setitem__(0,float("nan")),
    lambda b:b["r0_broad_joint"]["validation"][0]["output83"].__setitem__(0,.1),
    lambda b:b["r0_broad_joint"]["validation"].append(deepcopy(b["r0_broad_joint"]["validation"][0])),
    lambda b:b["r0_broad_joint"]["validation"][0].__setitem__("row_sha256","f"*64),
])
def test_missing_mutated_or_unpaired_predictions_rejected(mutation):
    b=bundles();mutation(b)
    with pytest.raises((ValueError,KeyError)):analyze(b)


def test_degenerate_intervals_disclosed_not_strength_claim():
    b=bundles()
    for model in o.MODEL_IDS:
        for split in o.SPLITS:
            for r in b[model][split]:r["output83"]=[0.]*83
    result=analyze(b)
    assert result["primary"]["degenerate_interval"] and result["primary"]["ci95"]==[0.,0.]
    assert result["strength_improvement_established"] is False


def test_no_heldout_forward_before_commitment(monkeypatch):
    called=[]
    monkeypatch.setattr(fit,"validate_commitment",lambda *a,**k:(_ for _ in ()).throw(ValueError("missing finals")))
    monkeypatch.setattr(fit,"engine",lambda *a,**k:called.append("engine"))
    with pytest.raises(ValueError,match="missing finals"):o.run_evaluation({}, {}, "unused")
    assert called==[]


def commitment_fixture(monkeypatch):
    def desc(name):return dict(path=str(name),bytes=1,sha256="a"*64)
    docs={};selection={"source_receipt":{"accepted":True}}
    rows=[]
    for spec in s.fits():
        name=spec["fit_id"];m=metadata();m.update(spec)
        m.update(seed=s.SEEDS[spec["repetition"]],source_training_games=192 if spec["breadth"]=="narrow" else 1024,
                 roots_per_game=16 if spec["breadth"]=="narrow" else 3,selection_sha256=s.digest(selection))
        endpoint=desc(name+"_manifest");result=desc(name+"_fit");proof=desc(name+"_proof");actual_exit=desc(name+"_exit")
        docs[endpoint["path"]]={"metadata":m,"checkpoint":desc(name+"_pt"),"tensor_manifest":desc(name+"_tensors")}
        docs[result["path"]]=dict(spec,status="fit_complete",selection_sha256=s.digest(selection),checkpoints={"96":endpoint})
        docs[proof["path"]]=dict(spec,status="fit_saved_verified",result=result,selection_sha256=s.digest(selection),
                                checkpoints={"96":endpoint},optimizer_calls=96,replay_optimizer_calls=1)
        docs[actual_exit["path"]]=dict(status="passed",worker_exit_code=0,cleanup_errors=[],error=None,mode="fit",
                                      fit_id=name,plan_sha256="a"*64)
        rows.append(dict(spec=spec,fit_result=result,saved_verification=proof,final_checkpoint=endpoint,actual_exit=actual_exit))
    value=dict(schema="independent_data_all_fits_committed_v1",status="all_four_fits_committed",study_id=s.STUDY,
        selection_sha256=s.digest(selection),source_receipt_sha256=s.digest(selection["source_receipt"]),
        optimizer_calls=384,replay_optimizer_calls=4,checkpoint_selection=False,fits=rows,
        freeze=desc(fit.c.PLAN),selection=desc(fit.c.SELECTION),source_receipt=selection["source_receipt"])
    docs["commitment"]=value
    monkeypatch.setattr(fit,"read_json",lambda item,**kw:deepcopy(docs[item["path"]]))
    monkeypatch.setattr(fit.c,"artifact",desc);monkeypatch.setattr(fit,"artifact",desc)
    monkeypatch.setattr(fit.c,"verify",lambda *a,**kw:None)
    return docs,selection,desc("commitment")


def test_commitment_requires_four_actual_verified_exits(monkeypatch):
    docs,selection,item=commitment_fixture(monkeypatch)
    value=fit.validate_commitment(selection,item)
    assert len(value["fits"])==4


@pytest.mark.parametrize("mutation",[
    lambda d:d["commitment"]["fits"].pop(),
    lambda d:d["r0_broad_joint_exit"].__setitem__("worker_exit_code",False),
    lambda d:d["r0_broad_joint_exit"].__setitem__("error","failed"),
    lambda d:d["r0_broad_joint_exit"].__setitem__("cleanup_errors",["owned child still alive"]),
    lambda d:d["r0_broad_joint_manifest"]["metadata"].__setitem__("checkpoint_step",95),
    lambda d:d["r0_broad_joint_manifest"]["metadata"].__setitem__("repetition",False),
    lambda d:d["r0_broad_joint_fit"].__setitem__("status","failed"),
    lambda d:d["r0_broad_joint_proof"].__setitem__("replay_optimizer_calls",True),
    lambda d:d["commitment"].__setitem__("optimizer_calls",384.),
    lambda d:d["commitment"].__setitem__("selection_sha256","0"*64),
])
def test_commitment_negative_reader_cases(monkeypatch,mutation):
    docs,selection,item=commitment_fixture(monkeypatch);mutation(docs)
    with pytest.raises((ValueError,KeyError)):fit.validate_commitment(selection,item)


def test_model_free_figure_fixture(tmp_path,monkeypatch):
    result=analyze(bundles());monkeypatch.setattr(o.c,"OUT",tmp_path)
    artifact=o.render_chart(result,tmp_path)
    assert artifact["bytes"]>1000 and (tmp_path/"prediction_transfer.png").is_file()


def test_independent_offline_arithmetic_and_false_contrast_rejection():
    b=bundles();result=analyze(b)
    proof=o.verify_arithmetic(b,result)
    assert proof["status"]=="independent_saved_arithmetic_verified" and proof["contrasts"]==42
    bad=deepcopy(result);bad["contrasts"][0]["difference"]+=.01
    with pytest.raises(ValueError,match="independent fixed contrast"):o.verify_arithmetic(b,bad)


def test_independent_bootstrap_stream_rejection():
    b=bundles();result=analyze(b);result["bootstrap"]["strata"]["offline"]["index_sha256"]="0"*64
    with pytest.raises(ValueError,match="draw stream"):o.verify_arithmetic(b,result)

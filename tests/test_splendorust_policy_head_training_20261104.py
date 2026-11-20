"""Model-free control and saved-data qualification; no network calls or Torch."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from tools.splendorust_policy_head_20261104 import training as t,data as d
from tools.splendorust_selfplay_pilot_resume2_20261103 import training as old


class Parameter:
    def __init__(self,count):self.count=count;self.requires_grad=True;self.grad=None
    def numel(self):return self.count
    def requires_grad_(self,value):self.requires_grad=value;return self


class Model:
    def __init__(self):
        self.names={'shared':Parameter(4759552),'policy.weight':Parameter(20736),
                    'policy.bias':Parameter(81),'value.weight':Parameter(512),'value.bias':Parameter(2)}
        self.hashes={'shared':'shared-original','policy':'policy-original','value':'value-original'}
    def named_parameters(self):return self.names.items()
    def parameters(self):return self.names.values()
    def train(self):return self


class Scalar:
    def __init__(self,value,backward=lambda:None):self.value=value;self.backward=backward
    def detach(self):return self
    def mean(self):return self
    def __float__(self):return float(self.value)


def scope_model(monkeypatch):
    model=Model();monkeypatch.setattr(t,'group_digests',lambda model:dict(model.hashes))
    t.configure_trainability(model)
    optimizer=SimpleNamespace(param_groups=[{'params':[model.names[n] for n in t.TRAINABILITY['trainable_names']]}])
    return model,optimizer


def test_trainability_is_sole_recipe_change_and_fixed_orders():
    assert t.RECIPE==old.RECIPE
    assert t.pilot_plan()['seed']==old.pilot_plan()['seed']==202611012
    assert t.epoch_order(10660,202611012,0)==old.epoch_order(10660,202611012,0)
    assert t.epoch_order(10660,202611012,1)==old.epoch_order(10660,202611012,1)
    assert t.TRAINABILITY['trainable_parameters']==20817
    assert t.TRAINABILITY['frozen_parameters']+20817==t.PARAMETERS


def test_exact_added_engineering_work_budget():
    b=t.direct_forward_budget(t.pilot_plan(),{'train':10660,'validation':1802,'offline':1755})
    assert b['direct_forward_invocations']==947 and b['training_forward_backward_calls']==668
    assert b['heldout_forward_calls']==224 and b['engineering_probe_forward_calls']==3
    assert b['engineering_probe_optimizer_steps']==b['engineering_probe_backward_calls']==1
    assert t.pilot_plan()['engineering_probe']['examples']=='first32_actual_epoch0_training_indices_with_official_targets'


def test_only_two_policy_tensors_are_trainable(monkeypatch):
    model,optimizer=scope_model(monkeypatch)
    assert {n for n,p in model.names.items() if p.requires_grad}=={'policy.weight','policy.bias'}
    assert t.verify_trainability(model,optimizer)==[model.names['policy.weight'],model.names['policy.bias']]


@pytest.mark.parametrize('mutation',['shared_requires','value_gradient','extra_optimizer','duplicate_optimizer','missing_bias'])
def test_scope_guard_rejects_frozen_or_malformed_optimizer(monkeypatch,mutation):
    model,opt=scope_model(monkeypatch)
    if mutation=='shared_requires':model.names['shared'].requires_grad=True
    elif mutation=='value_gradient':model.names['value.weight'].grad=object()
    elif mutation=='extra_optimizer':opt.param_groups[0]['params'].append(model.names['value.weight'])
    elif mutation=='duplicate_optimizer':opt.param_groups[0]['params'].append(model.names['policy.weight'])
    else:opt.param_groups[0]['params'].pop()
    with pytest.raises(ValueError):t.verify_trainability(model,opt)


def fake_train_step(monkeypatch,change='policy',gradient=True):
    model,opt=scope_model(monkeypatch);clipped=[]
    def zero_grad(**kwargs):
        for p in model.parameters():p.grad=None
    def backward():
        if gradient:
            for n in t.TRAINABILITY['trainable_names']:model.names[n].grad=object()
    def step():model.hashes[change]+='-changed'
    opt.zero_grad=zero_grad;opt.step=step
    fake=SimpleNamespace(isfinite=lambda value:SimpleNamespace(all=lambda:True),
        nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda params,*a,**kw:clipped.extend(params) or 2.)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    outputs=SimpleNamespace(detach=lambda:SimpleNamespace(tolist=lambda:[[0.]*83]))
    monkeypatch.setattr(t,'tensor_forward',lambda model,tokens:outputs)
    monkeypatch.setattr(t,'loss_components',lambda outputs,batch:(Scalar(1.,backward),Scalar(.5),Scalar(.5)))
    return model,opt,clipped


def test_actual_train_step_control_clips_only_policy_and_records_frozen_hashes(monkeypatch):
    model,opt,clipped=fake_train_step(monkeypatch)
    result=t.train_step(model,opt,{'tokens':[0]})
    assert clipped==[model.names[n] for n in t.TRAINABILITY['trainable_names']]
    assert result['before_group_sha256']['shared']==result['after_group_sha256']['shared']
    assert result['before_group_sha256']['value']==result['after_group_sha256']['value']
    assert result['before_group_sha256']['policy']!=result['after_group_sha256']['policy']


@pytest.mark.parametrize('change',['shared','value'])
def test_actual_update_control_rejects_frozen_tensor_change(monkeypatch,change):
    model,opt,_=fake_train_step(monkeypatch,change=change)
    with pytest.raises(ValueError,match='preserves every'):t.train_step(model,opt,{'tokens':[0]})


def test_actual_update_control_rejects_missing_policy_gradient(monkeypatch):
    model,opt,_=fake_train_step(monkeypatch,gradient=False)
    with pytest.raises(ValueError,match='finite gradients'):t.train_step(model,opt,{'tokens':[0]})


def test_checkpoint_scope_invariants_require_policy_change():
    parent=dict(all='a',shared='s',policy='p',value='v');candidate=dict(all='b',shared='s',policy='q',value='v')
    metadata=dict(trainability=deepcopy(t.TRAINABILITY),frozen_group_sha256={'shared':'s','value':'v'})
    assert t.verify_policy_head_checkpoint_invariants(metadata,parent,candidate)['status']=='exact_policy_only_checkpoint'
    for key in ('shared','value'):
        bad=dict(candidate);bad[key]+='x'
        with pytest.raises(ValueError):t.verify_policy_head_checkpoint_invariants(metadata,parent,bad)
    with pytest.raises(ValueError):t.verify_policy_head_checkpoint_invariants(metadata,parent,parent)


def test_value_equality_is_bitwise_and_ignores_legitimate_policy_change():
    a=[[0.]*83];b=deepcopy(a);b[0][0]=1.
    assert t.verify_value_pair(a,b,'fixture')['equal']
    b[0][81]=-0.
    with pytest.raises(ValueError,match='float32 bits'):t.verify_value_pair(a,b,'fixture')


def test_saved_update_chain_rejects_joint_update_or_missing_link():
    before={'shared':'s','policy':'p','value':'v'}
    step=dict(trainability=deepcopy(t.TRAINABILITY),before_group_sha256=before,
              after_group_sha256={'shared':'s','policy':'q','value':'v'},frozen_gradients_absent=True,optimizer_policy_only=True)
    assert t.verify_step_scope(step,before)['policy']=='q'
    bad=deepcopy(step);bad['after_group_sha256']['shared']='changed'
    with pytest.raises(ValueError):t.verify_step_scope(bad,before)
    with pytest.raises(ValueError):t.verify_step_scope(step,{'shared':'s','policy':'other','value':'v'})


@pytest.fixture(scope='module')
def accepted():
    declaration={k:t.artifact(d.ROOT/d.ACCEPTED_SOURCE_OUTPUT/name) for k,(name,sha) in d.ACCEPTED_SOURCE_HASHES.items()}
    docs=d.accepted_source_documents(declaration)
    return declaration,docs


def test_actual_accepted256_corpus_keeps_old_mixed_paths(accepted):
    declaration,docs=accepted
    proof=d.validate_accepted_corpus_reuse(docs['corpus_manifest'],docs['source_specs'],declaration)
    assert proof['source_games']==256 and proof['new_collection_games']==0
    assert proof['training_games']==192 and proof['validation_games']==proof['offline_games']==32


@pytest.mark.parametrize('mutation',['new_path','drop_game','change_split','wrong_closeout'])
def test_reused_corpus_rejects_relabel_or_unaccepted_source(accepted,mutation):
    declaration,docs=deepcopy(accepted);manifest=docs['corpus_manifest'];specs=docs['source_specs']
    if mutation=='new_path':manifest['games'][48]['path']='outputs/new/games/g048/game.json'
    elif mutation=='drop_game':manifest['games'].pop()
    elif mutation=='change_split':specs[0]['split']='validation'
    else:declaration['accepted_source_closeout']['sha256']='0'*64
    with pytest.raises(ValueError):d.validate_accepted_corpus_reuse(manifest,specs,declaration)


def test_new_loader_qualifies_exact_sixteen_saved_corrected_cases(accepted):
    declaration,docs=accepted;freeze=docs['accepted_source_freeze']
    binding=d.read_artifact(freeze['fit_plan'])['input_qualification']
    assert d.validate_accepted_input_qualification(binding,declaration)['training_examples_added']==0
    cases=d.read_artifact(binding['plan'])['loader_cases']
    rows=[json.loads((d.ROOT/p).read_text()) for p in binding['row_paths']]
    actual=d.qualify_captured_rows(cases,rows)
    saved=json.loads((d.ROOT/binding['proof_path']).read_text())['actual_loader']
    assert actual==saved and len(actual['actual_loader_rejections'])==13
    assert len(actual['six_corrected_faces'])==6


def test_no_model_runtime_imported():
    assert 'torch' not in sys.modules


def test_actual_cli_binds_reused_manifest_instead_of_new_output(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'ROOT',tmp_path)
    out=tmp_path/'new';out.mkdir();oldroot=tmp_path/'accepted';oldroot.mkdir()
    def write(path,value):
        path.write_text(d.canonical(value));return t.artifact(path)
    manifest=write(oldroot/'corpus.json',{'saved':'accepted'})
    wrapper=dict(training_plan=t.pilot_plan(),source_specs=[],learned_entrypoint={},input_qualification={'fixture':'old'})
    plan=write(out/'fit_plan.json',wrapper);fixtures=write(out/'fixtures.json',{'fixtures':[]})
    declaration={'actual':'source_reuse'}
    freeze=write(out/'freeze.json',dict(phase='pilot',fit_plan=plan,fit_parity_fixtures=fixtures,
                                      reused_corpus=manifest,source_reuse=declaration))
    checked=[]
    monkeypatch.setattr(d,'validate_accepted_corpus_reuse',lambda m,s,x:checked.append(('corpus',m,s,x)))
    monkeypatch.setattr(d,'validate_accepted_input_qualification',lambda b,x:checked.append(('fixtures',b,x)))
    def verify(items):
        assert all(t.artifact(tmp_path/i['path'])==i for i in items)
    common=SimpleNamespace(OUT=out,SCOPE='pilot',verify=verify)
    args=SimpleNamespace(mode='pilot',out=str(out/'fit'),plan=str(out/'fit_plan.json'),
                         fixtures=str(out/'fixtures.json'),manifest=str(oldroot/'corpus.json'))
    env=dict(STUDY_RESEARCH_FREEZE=str(out/'freeze.json'),STUDY_RESEARCH_FREEZE_SHA256=freeze['sha256'])
    assert t.admit_fit_cli(args,common=common,environment=env)[2]=={'saved':'accepted'}
    assert checked==[('corpus',{'saved':'accepted'},[],declaration),('fixtures',{'fixture':'old'},declaration)]
    args.manifest=str(out/'corpus_manifest.json')
    with pytest.raises(ValueError,match='accepted reused corpus'):t.admit_fit_cli(args,common=common,environment=env)


def test_new_fitter_runtime_source_guard_executes_once(monkeypatch):
    calls=[];monkeypatch.setattr(t,'_STARTUP_RECEIPT',None)
    monkeypatch.setattr(t.guard,'verify_runtime_sources',lambda:calls.append(1) or {'verified':'fixture'})
    first=t.startup();second=t.startup();assert calls==[1] and first==second
    first['runtime_sources']['verified']='changed'
    assert t.startup()['runtime_sources']['verified']=='fixture'

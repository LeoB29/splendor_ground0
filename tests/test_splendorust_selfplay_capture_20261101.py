"""Model-free root-capture contract and saved/native public tensor checks."""
from copy import deepcopy
import ast
import inspect
import io
import json
import re
import subprocess
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_selfplay_20261101 import capture_adapter as a, common as c
from tools.splendorust_selfplay_20261101.build_capture import validate_build, NATIVE
from tools.splendorust_search512_20261101 import corpus
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import rules


FIXTURES={f['id']:f for f in corpus.build_fixtures()}
SUITE_PATH=c.ACCEPTED_OUT/'workers/operational_g000_candidate/entrypoint_clock.json'


def observer(name='face_T1_05'):
    return rules.observation_payload(decode_training_snapshot(FIXTURES[name]['snapshot']))


def synthetic_record(name='face_T1_05'):
    obs=observer(name);actions,indices,mask=a.main_support(obs)
    visits=[0]*67;visits[indices[0]]=512
    native=dict(enabled=True,phase='Main',quota=512,authoritative_indices67=indices,retained_indices67=[indices[0]],
        callback_count=2,root_callback_index=0,root_callback_role='root',simulations=512,visits67=visits,
        q67=[.5 if value else None for value in visits],root_value=.5,network_value=.5,
        policy67=[a.f32(value/512) for value in visits])
    root=dict(tokens=a.public_root_tokens(obs),outputs=[[0.]*83])
    return obs,a.capture_record(obs,512,native,root,actions[0])


def test_main_mapping_uses_exact60_channels_and_retains_legal_before_pruning():
    assert len(a.INDEX67_TO81)==60 and set(a.INDEX67_TO81.values())==set(range(60))
    assert set(range(67))-set(a.INDEX67_TO81)=={0,15,23,27,29,30,31}
    obs,record=synthetic_record()
    assert a.validate_capture(record,obs) is record
    assert sum(record['legal_mask81'])>len(record['retained_indices67'])==1
    assert sum(record['policy_target81'])==1
    assert all(record['policy_target81'][i]==0 for i in range(60,81))


@pytest.mark.parametrize('field,value',[
    ('schema','old'),('ruleset_id','old'),('catalogue_sha256','0'*64),('turn_contract','old'),
    ('observer_sha256','0'*64),('seed_decimal','1'),('quota',128),('quota',True),('actor_seat',3),
    ('canonical_actor_slot',1),('eligible',False),('exclusion_reason','single_choice'),
    ('root_callback_index',1),('root_input_sha256','0'*64),('root_output_sha256','0'*64),
    ('retained_indices67',[]),('callback_count',0),('visits67',[0]*67),('policy_target81',[0.]*81),
])
def test_capture_rejects_identity_or_target_corruption(field,value):
    obs,record=synthetic_record();record[field]=value
    with pytest.raises((ValueError,TypeError,KeyError)):a.validate_capture(record,obs)


def test_no_mask_conditioning_on_only_retained_search_choices():
    obs,record=synthetic_record();record['legal_mask81']=[v>0 for v in record['policy_target81']]
    with pytest.raises(ValueError,match='before search pruning'):a.validate_capture(record,obs)


@pytest.mark.parametrize('change', ['tensor','nonfinite','float64','value_units','unvisited_q','native_policy','root_aggregate','leaf_role','hidden_field'])
def test_capture_rejects_scientific_corruption(change):
    obs,record=synthetic_record()
    if change=='tensor':record['root_tokens'][0]+=.1;record['root_input_sha256']=a.digest(record['root_tokens'])
    elif change=='nonfinite':record['root_output83'][0]=float('nan')
    elif change=='float64':record['root_output83'][0]=.1
    elif change=='value_units':record['root_output83'][81]=2.
    elif change=='unvisited_q':record['native_capture']['q67'][0]=.4
    elif change=='native_policy':record['native_capture']['policy67'][0]=1.
    elif change=='root_aggregate':record['native_capture']['root_value']=.1
    elif change=='leaf_role':record['native_capture']['root_callback_role']='leaf'
    else:record['native_capture']['true_hidden_card']='T1_01'
    with pytest.raises(ValueError):a.validate_capture(record,obs)


def test_actual_saved12_roots_full_tensor_exact_without_forward():
    suite=c.read(SUITE_PATH)['result']; checked=[]
    for row in suite['fixtures']:
        phases=row['results']['search']['diagnostics']['phases']
        if phases and phases[0]['neural_callback_count']:
            expected=phases[0]['inference'][0]['tokens']
            actual=a.public_root_tokens(observer(row['id']))
            assert actual==expected and len(actual)==1488
            checked.append(row['id'])
    assert len(checked)==12
    assert all(name in checked for name in ('face_T1_05','face_T1_07','face_T2_10','face_T2_12','face_T2_13','face_T2_14',
        'noble_nonpurchase','multiple_nobles','noble_final_round','private_reserves','gold_returns','turn123'))


def test_public_tensor_ignores_inaccessible_true_cards():
    state=decode_training_snapshot(FIXTURES['private_reserves']['snapshot'])
    alternate=corpus.hidden_variant(state)
    left=rules.observation_payload(state);right=rules.observation_payload(alternate)
    assert left==right and rules.serialize_snapshot(state)!=rules.serialize_snapshot(alternate)
    assert a.public_root_tokens(left)==a.public_root_tokens(right)
    bad=deepcopy(left);bad['hidden_tier_decks']={}
    with pytest.raises(ValueError):a.public_root_tokens(bad)


def test_late_and_single_choice_exclude_targets_mechanically():
    count=0
    for name in FIXTURES:
        obs=observer(name);actions,indices,_=a.main_support(obs)
        if len(actions)!=1 and obs['observation']['turn_index']<124:continue
        forced=actions==[{'kind':'Pass'}]
        native=None if forced else dict(enabled=True,phase='Main',quota=512,authoritative_indices67=indices,
            retained_indices67=indices,callback_count=0,root_callback_index=None,root_callback_role=None,
            simulations=0,visits67=None,q67=None,root_value=None,network_value=None,policy67=None)
        record=a.capture_record(obs,512,native,None,actions[0]);assert not record['eligible']
        assert record['root_tokens'] is None and record['visits67'] is None
        record['root_tokens']=[0.]*1488
        with pytest.raises(ValueError,match='fabricated'):a.validate_capture(record,obs)
        count+=1
    assert count>=2


def test_native_model_free_encode_matches_independent_public_tensors():
    build=validate_build();argv=[str(c.ROOT/build['binary']['path'])]
    requests=[];expected=[]
    for name in FIXTURES:
        obs=observer(name)
        if obs['observation']['turn_index']>=124:continue
        expected.append(a.public_root_tokens(obs))
        requests.append(dict(op='encode',observation=a.bridge.encode_observation(obs),seed=int(a.digest(obs)[:16],16)))
    # Only catalogue/encoding operations. No select request, model output,
    # neural network, search simulation or new game can enter this subprocess.
    data=''.join(json.dumps(value,separators=(',',':'))+'\n' for value in requests)
    result=subprocess.run(argv,input=data,text=True,encoding='utf-8',capture_output=True,timeout=20,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0 and not result.stderr
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]=={'kind':'ready','bridge':a.BRIDGE_ID,'source_pin':a.PIN}
    assert len(rows)==1+len(expected)
    for row,tokens in zip(rows[1:],expected,strict=True):
        assert row['kind']=='result' and row['tokens']==tokens


def test_copied_native_algorithm_and_semantic_index_constants_unchanged():
    proof=validate_build();assert proof['algorithm_sources_unchanged'] and proof['only_bridge_capture_added']
    source=(NATIVE/'crates/splendor-agents/src/transfer_data.rs').read_text()
    for name,values in [('TAKE_MASKS',a.TAKE_MASKS),('CARD_SLOT',a.CARD_SLOT),('CARD_GROUP',a.CARD_GROUP)]:
        literal=re.search(r'pub const '+name+r':[^=]+=\s*\[([^]]+)\]',source).group(1)
        assert tuple(int(value) for value in literal.replace('\n','').split(',') if value.strip())==values
    main=(NATIVE/'bridge/src/main.rs').read_text()
    assert main.count('agent.select_action(&o,&choices)')==1
    assert 'agent.capture_search_targets=enabled' in main
    assert 'CALLBACKS.fetch_add' in main and 'root_callback_index' in main


def test_off_marker_does_not_validate_teacher_targets_or_change_science(monkeypatch):
    adapter=object.__new__(a.CaptureOpponent);adapter.capture_targets=False;adapter.profile={'iterations':128}
    action={'unchanged':'action'};trace={'unchanged':'trace'}
    def underlying(self,*args,**kwargs):
        self._capture_decision_deadline=time.perf_counter()+9.
        return action,trace
    monkeypatch.setattr(a.BaselineOpponent,'choose',underlying)
    monkeypatch.setattr(a,'validate_capture',lambda *args,**kwargs:pytest.fail('OFF must not validate teacher capture'))
    obs=observer();result=adapter.choose(obs,diagnostic_capture=True)
    assert result==(action,trace) and result[0] is action and result[1] is trace
    assert adapter.last_training_capture==dict(schema='selfplay_root_capture_disabled_v1',enabled=False,
        observer_sha256=a.digest(obs),quota=128,training_eligible=False)


def test_postcapture_work_uses_original_deadline_not_a_new_allowance(monkeypatch):
    adapter=object.__new__(a.CaptureOpponent);adapter.capture_targets=False;adapter.profile={'iterations':128}
    def underlying(self,*args,**kwargs):
        self._capture_decision_deadline=time.perf_counter()-1.
        self.active_deadline=None
        return {},{}
    monkeypatch.setattr(a.BaselineOpponent,'choose',underlying)
    with pytest.raises(TimeoutError,match='original9s'):
        adapter.direct_choose(observer())
    assert adapter.last_training_capture['enabled'] is False


def test_capture_uses_exact_original_tensor_inference_and_no_model_load():
    assert a.CaptureOpponent._infer is a.BaselineOpponent._infer
    source=inspect.getsource(a.CaptureOpponent.__init__)
    assert 'shared_from.model,shared_from.reference' in source
    assert 'load_model(' not in source and 'torch.load(' not in source

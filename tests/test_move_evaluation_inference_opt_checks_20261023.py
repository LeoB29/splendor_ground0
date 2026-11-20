"""Synthetic Stage B regressions using frozen, model-free fixture builders."""
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
import runpy

import pytest
from tools.move_evaluation_inference_opt_20261023 import checks as c

ROOT = Path(__file__).resolve().parents[1]
A = runpy.run_path(str(ROOT/'tests/test_move_evaluation_inference_checks_20261023.py'))
R = runpy.run_path(str(ROOT/'tests/test_move_evaluation_reuse_checks_20261023.py'))


def fixture_args():
    value=A['fixture']()
    return value,deepcopy(value),deepcopy(value),A['fine_measurement'](c.FIXTURE_CALLS)


def test_fixture_keeps_reference_and_fine_counts():
    result=c.validate_fixture(*fixture_args())
    assert result['status']=='passed' and result['inference_scopes']['inference_callbacks']==14
    assert result['scope']['eligibility_receipt_authenticated_separately_by_runner']


@pytest.mark.parametrize('index',(0,1,2))
def test_fixture_rejects_scientific_difference(index):
    args=list(fixture_args()); args[index]['changed']=True
    with pytest.raises(ValueError,match='scientific output'): c.validate_fixture(*args)


def test_panel_keeps_exact_capture_and_identity_gate(monkeypatch):
    monkeypatch.setattr(c.stage_a,'_panel_identities',lambda:(A['identities'](),{'path':'synthetic'}))
    value=A['panel']()
    result=c.validate_panel(value,deepcopy(value),deepcopy(value),A['panel_measurement']())
    assert result['decisions']==144 and result['inference_scopes']['inference_callbacks']==144
    assert result['scope']['eligibility_receipt_authenticated_separately_by_runner']
    wrong=deepcopy(value); wrong['rows'][0]['trace']['observer_seed']+=1
    with pytest.raises(ValueError,match='seed/hash'):
        c.validate_panel(wrong,deepcopy(wrong),deepcopy(wrong),A['panel_measurement']())


def selection(tmp_path,monkeypatch,*,one_ply=False,neural=True):
    output=tmp_path/'outputs/current'
    path=output/'workers/r00/timed/continuations.jsonl.gz'
    path.parent.mkdir(parents=True)
    rows=R['trace_rows'](one_ply=one_ply,neural=neural)
    data=gzip.compress(''.join(c.frozen.canonical(r)+'\n' for r in rows).encode(),mtime=0)
    path.write_bytes(data)
    monkeypatch.setattr(c,'ROOT',tmp_path); monkeypatch.setattr(c,'OUT',output)
    base=R['selection_measurement'](one_ply=one_ply,neural=neural)
    counts={r['label']:r['calls'] for r in base['rows']}
    counts.update(dict.fromkeys(c.FINE_LABELS,counts['entity_infer']))
    measurement=A['fine_measurement'](counts)
    science={'decision':{'action':{},'record':{}},
        'trace':{'path':path.relative_to(tmp_path).as_posix(),'bytes':len(data),
                 'sha256':hashlib.sha256(data).hexdigest(),'records':512},
        'profile_id':'fixed4_raw035_own_greedy_entity_opponent_20261023_v1',
        'observer_sha256':'a'*64,'ruleset_id':'splendor_base_qualified_rules_20261018_v1',
        'catalogue_sha256':'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'}
    return science,measurement,path


@pytest.mark.parametrize('one_ply,neural',((False,True),(False,False),(True,False)))
def test_selection_authenticates_trace_and_fine_census(tmp_path,monkeypatch,one_ply,neural):
    science,measurement,path=selection(tmp_path,monkeypatch,one_ply=one_ply,neural=neural)
    before=deepcopy(science)
    result=c.validate_selection(science,measurement)
    assert result['status']=='passed' and result['trajectory_records']==512
    assert result['entity_inference_calls']==(512 if neural and not one_ply else 0)
    assert result['trace']['sha256']==science['trace']['sha256']
    assert result['scientific_sha256']==c.frozen.digest(science)
    assert science==before
    assert result['scope']['original_policy_replay_required_separately']
    assert result['scope']['exact_previous_science_comparison_required_separately']


@pytest.mark.parametrize('label',c.FINE_LABELS+('entity_infer','shared_attention_forward',
    'observer_decode','legal','reference_greedy_choose','legacy_bridge_prepare'))
def test_selection_rejects_wrong_measured_work(tmp_path,monkeypatch,label):
    science,measurement,path=selection(tmp_path,monkeypatch)
    counts={r['label']:r['calls'] for r in measurement['rows']}; counts[label]+=1
    with pytest.raises(ValueError,match=label):
        c.validate_selection(science,A['fine_measurement'](counts))


@pytest.mark.parametrize('damage',('wrapper','profile','ruleset','catalogue','observer_hash',
    'trace_extra','bytes','bool_bytes','hash','records','bool_records','absolute','traversal','outside','basename','file'))
def test_selection_rejects_bad_artifact_or_identity(tmp_path,monkeypatch,damage):
    science,measurement,path=selection(tmp_path,monkeypatch)
    trace=science['trace']
    if damage=='wrapper': science['extra']=1
    elif damage=='profile': science['profile_id']='changed'
    elif damage=='ruleset': science['ruleset_id']='historical'
    elif damage=='catalogue': science['catalogue_sha256']='0'*64
    elif damage=='observer_hash': science['observer_sha256']='bad'
    elif damage=='trace_extra': trace['extra']=1
    elif damage=='bytes': trace['bytes']+=1
    elif damage=='bool_bytes': trace['bytes']=True
    elif damage=='hash': trace['sha256']='0'*64
    elif damage=='records': trace['records']=511
    elif damage=='bool_records': trace['records']=True
    elif damage=='absolute': trace['path']=str(path)
    elif damage=='traversal': trace['path']='outputs/current/../current/workers/r00/timed/continuations.jsonl.gz'
    elif damage in ('outside','basename'):
        moved=(tmp_path/'continuations.jsonl.gz') if damage=='outside' else path.with_name('other.gz')
        moved.write_bytes(path.read_bytes()); trace['path']=moved.relative_to(tmp_path).as_posix()
    else: path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError): c.validate_selection(science,measurement)


def test_selection_rejects_broken_fine_partition(tmp_path,monkeypatch):
    science,measurement,path=selection(tmp_path,monkeypatch)
    row=next(r for r in measurement['rows'] if r['label']=='infer_transformer')
    row['inclusive_ns']+=1
    with pytest.raises(ValueError,match='selected descendants'):
        c.validate_selection(science,measurement)


def test_selection_rejects_unknown_region_unwind(tmp_path,monkeypatch):
    science,measurement,path=selection(tmp_path,monkeypatch)
    row=next(r for r in measurement['rows'] if r['label']=='infer_transformer')
    row['returns']-=1; row['unwinds']=1
    with pytest.raises(ValueError,match='unwind'): c.validate_selection(science,measurement)

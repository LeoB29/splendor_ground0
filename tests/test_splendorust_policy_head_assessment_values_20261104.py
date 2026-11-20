"""Model-free value invariance, actual saved handlers, and checkpoint routing."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import zipfile
import pytest
from tools.splendorust_policy_head_20261104.assessment import (
    common as c, value_invariance as v, checkpoint, adapters, qualification, worker, control)


@pytest.fixture(scope='module')
def saved():
    pins=v.sources()
    return c.read(c.ROOT/pins['graph']['path']),c.read(c.ROOT/pins['entrypoint']['path'])['result']


@pytest.fixture
def local_out(monkeypatch):
    base=c.parent.OUT/'prelaunch';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='policy_assessment_',dir=base) as name:
        folder=Path(name);monkeypatch.setattr(c,'OUT',folder)
        yield folder


def test_all_actual_same_input_value_paths_pass(saved):
    graph,suite=saved
    a=v.compare_graph(graph,graph);b=v.compare_native(suite,suite)
    assert a['fixtures']==11 and a['batch_rows']==27 and a['new_model_calls']==0
    assert b['same_input_probes']==46 and b['new_model_calls']==0
    assert a['original_value_float_and_bits_exact'] and b['original_value_float_and_bits_exact']


def test_changed_policy_allowed_but_no_search_trajectory_comparison(saved):
    original_graph,original_suite=saved;graph=deepcopy(original_graph);suite=deepcopy(original_suite)
    for e in graph['evidence']:
        for field in ('eager_outputs','candidate_outputs'):
            for row in e[field]:row[0]+=1.
    for arm in ('search','reference'):
        for row in suite['fresh_numeric_parity'][arm]['rows']:
            for field in ('host_policy','native_policy'):row['response'][field][0]+=1.
    suite['fixtures']=[];suite['direct_same_quota']=[]  # Deliberately not cross-policy comparison targets.
    assert v.compare_graph(graph,original_graph)['status']=='passed'
    assert v.compare_native(suite,original_suite)['status']=='passed'


@pytest.mark.parametrize('field',['eager_outputs','candidate_outputs'])
@pytest.mark.parametrize('fault',['changed','nonfinite','missing','signed_zero'])
def test_graph_values_fail_closed(saved,field,fault):
    original=deepcopy(saved[0]);actual=deepcopy(original);row=actual['evidence'][0][field][0]
    if fault=='changed':row[81]+=1e-6
    elif fault=='nonfinite':row[81]=float('nan')
    elif fault=='missing':row.pop()
    else:original['evidence'][0][field][0][81]=0.;row[81]=-0.
    with pytest.raises(ValueError):v.compare_graph(actual,original)


@pytest.mark.parametrize('fault',['input','order','batch','omission'])
def test_graph_requires_exact_original_inputs(saved,fault):
    original=saved[0];actual=deepcopy(original)
    if fault=='input':actual['fixtures'][0]['tokens'][0]+=1.
    elif fault=='order':actual['evidence'][0],actual['evidence'][1]=actual['evidence'][1],actual['evidence'][0]
    elif fault=='batch':actual['evidence'][0]['batch']=2
    else:actual['evidence'].pop()
    with pytest.raises(ValueError):v.compare_graph(actual,original)


@pytest.mark.parametrize('arm',['reference','search'])
@pytest.mark.parametrize('fault',['host','native','input','order','omission'])
def test_native_values_and_fixture_identity_fail_closed(saved,arm,fault):
    original=saved[1];actual=deepcopy(original);rows=actual['fresh_numeric_parity'][arm]['rows']
    if fault in ('host','native'):rows[0]['response'][fault+'_value'][0]+=1e-6
    elif fault=='input':rows[0]['observer_sha256']='0'*64
    elif fault=='order':rows[0],rows[1]=rows[1],rows[0]
    else:rows.pop()
    with pytest.raises(ValueError):v.compare_native(actual,original)


def fixture_freeze(folder):
    plan=qualification.build_plan();path=folder/'qualification_plan.json';c.save(path,plan)
    return {'qualification_plan':c.artifact(path)},plan


def test_sources_bound_to_acceptance_and_exact_freeze(local_out):
    freeze,_=fixture_freeze(local_out)
    assert v.frozen_sources(freeze)==v.SOURCES
    doc=c.read(local_out/'qualification_plan.json');doc['value_invariance_sources']['graph']['sha256']='0'*64
    (local_out/'qualification_plan.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError):v.frozen_sources(freeze)
    freeze['qualification_plan']=c.artifact(local_out/'qualification_plan.json')
    with pytest.raises(ValueError):v.frozen_sources(freeze)


def test_actual_saved_full67_handler_inventory_and_value_probes(local_out,saved):
    source=c.ROOT/v.SOURCES['entrypoint']['path'];old=source.parent
    freeze,plan=fixture_freeze(local_out)
    freeze.update(profiles={'control':{str(q):c.artifact(adapters.profile_path(q,route='control')) for q in (16,128)}},
        checkpoint_bindings={'control':{'native_reference':c.artifact(checkpoint.MODEL_BIN)}})
    (local_out/'entrypoint_forensics.jsonl').write_bytes((old/'entrypoint_forensics.jsonl').read_bytes())
    receipt=c.read(source)
    proof=v.native_proof(receipt['result'],freeze);c.save(local_out/'value_invariance_native.json',proof)
    checked=qualification.verify_entrypoint(local_out,receipt,plan,freeze)
    assert checked['complete_decisions']==67 and checked['native_numeric_parity']==46
    assert checked['original_value_invariance']==proof
    assert qualification.compare_control_parity(receipt['result'],plan)['status']=='passed'
    wrong=deepcopy(receipt);wrong['result']['fresh_numeric_parity']['search']['rows'][0]['response']['host_value'][0]+=1e-6
    with pytest.raises(ValueError,match='Same input/encoding'):qualification.verify_entrypoint(local_out,wrong,plan,freeze)


def test_actual_graph_saved_reader_recomputes_invariance(local_out,saved):
    graph=saved[0];source=c.ROOT/v.SOURCES['graph']['path'];original_freeze=c.read(source.parent.parent.parent/'freeze.json')
    freeze,_=fixture_freeze(local_out);freeze['graph_parity_plan']=original_freeze['graph_parity_plan']
    c.save(local_out/'graph_parity.json',graph);c.save(local_out/'player_identity.json',{'route':'control'})
    proof=v.graph_proof(local_out,freeze);c.save(local_out/'value_invariance_graph.json',proof)
    assert worker.verify_graph_admission(local_out,freeze,graph['summary'])['original_value_invariance']==proof
    changed=dict(proof,fixtures=10);(local_out/'value_invariance_graph.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='value-invariance'):worker.verify_graph_admission(local_out,freeze,graph['summary'])


def test_live_graph_handler_publishes_proof_before_return(local_out,monkeypatch):
    context={'folder':local_out,'freeze':{}};calls=[]
    monkeypatch.setattr(worker,'original_graph_parity',lambda *a:(calls.append('actual_graph') or {'accepted':True}))
    monkeypatch.setattr(v,'graph_proof',lambda *a:(calls.append('values') or {'status':'passed'}))
    assert worker.handle(context,{'operation':'graph_parity'})=={'accepted':True}
    assert calls==['actual_graph','values'] and c.read(local_out/'value_invariance_graph.json')=={'status':'passed'}


@pytest.mark.parametrize('pass_values',[True,False])
def test_live_entrypoint_checks_original_values_before_qualification_pass(local_out,monkeypatch,pass_values):
    context=dict(folder=local_out,freeze={},graph_parity_passed=True,diagnostic_capture=False,
        entrypoint_attempted=False,entrypoint_passed=False,request_started=None,
        adapters={'search':SimpleNamespace(admission_guard=None)},route='candidate')
    suite={'counts':{'complete_decisions':67}}
    monkeypatch.setattr(worker.corpus,'run_entrypoint_checks',lambda *a,**kw:suite)
    def probe(*a,**kw):
        assert context['entrypoint_passed'] is False
        if not pass_values:raise ValueError('changed original values')
        return {'same_input_probes':46}
    monkeypatch.setattr(v,'native_proof',probe)
    if pass_values:
        assert worker.handle(context,{'operation':'entrypoint_checks'})==suite
        assert context['entrypoint_passed'] and c.read(local_out/'value_invariance_native.json')['same_input_probes']==46
    else:
        with pytest.raises(ValueError,match='changed original'):worker.handle(context,{'operation':'entrypoint_checks'})
        assert not context['entrypoint_passed']
        assert 'entrypoint_failed' in (local_out/'entrypoint_forensics.jsonl').read_text()


def candidate_fixture(folder,changed='policy.bias'):
    """Modify one serialized float only, without Torch, loading a network or fitting."""
    entries=checkpoint.parent_entries();entry=next(e for e in entries if e['name']==changed)
    path=folder/'fixture.pt'
    with zipfile.ZipFile(checkpoint.training.PARENT_CHECKPOINT) as source,zipfile.ZipFile(path,'w') as target:
        for name in source.namelist():
            raw=source.read(name)
            if name==entry['zip_member']:
                raw=struct.pack('<f',struct.unpack('<f',raw[:4])[0]+.001)+raw[4:]
                entry['sha256']=hashlib.sha256(raw).hexdigest()
            target.writestr(name,raw)
    hashes=checkpoint.training._verify_tensor_archive(path,entries)
    parent=checkpoint.training.verify_parent_tensor_hashes()
    metadata=dict(weights_sha256=hashes['all'],trainability=deepcopy(checkpoint.training.TRAINABILITY),
        frozen_group_sha256={k:parent[k] for k in ('shared','value')})
    tensors=folder/'tensors.json';c.save(tensors,dict(checkpoint=c.artifact(path),weights_sha256=hashes['all'],tensors=entries))
    manifest=folder/'manifest.json';c.save(manifest,dict(schema='selfplay_candidate_checkpoint_v1',
        checkpoint=c.artifact(path),tensor_manifest=c.artifact(tensors),
        model_definition=c.artifact(Path(checkpoint.training.__file__)),metadata=metadata))
    return c.artifact(manifest)


def test_exact_policy_only_archive_export_and_profiles(local_out):
    descriptor=candidate_fixture(local_out);item=checkpoint.export_candidate(descriptor)
    assert checkpoint.verify_candidate_export(descriptor,item)['tensors']==81
    with pytest.raises(FileExistsError):checkpoint.export_candidate(descriptor)
    for quota in (16,128):
        original=adapters.profile(quota,route='control');learned=adapters.candidate_profile(quota,descriptor,item)
        for key in ('iterations','world_pool','cpuct','depth','dynamic_fpu','fpu_reduction','inference_batch_size','native_binary','internal_budget_seconds'):
            assert original[key]==learned[key]
        assert learned['checkpoint']==c.read(c.ROOT/descriptor['path'])['checkpoint']['path']
        assert 'policy_head' in learned['profile_id'] and learned['native_reference_model']==item
    target=c.ROOT/item['path'];raw=bytearray(target.read_bytes());raw[-1]^=1;target.write_bytes(raw)
    with pytest.raises(ValueError):checkpoint.verify_candidate_export(descriptor,c.artifact(target))


@pytest.mark.parametrize('changed',['value.bias','norm.bias'])
def test_correctly_hashed_changed_frozen_group_is_rejected(local_out,changed):
    descriptor=candidate_fixture(local_out,changed)
    with pytest.raises(ValueError,match='original shared and value'):checkpoint.candidate_sources(descriptor)


def test_old_joint_candidate_is_never_accepted():
    path=c.ROOT/'outputs/splendorust_selfplay_pilot_resume2_20261103_01/fit/checkpoint_manifest.json'
    with pytest.raises(ValueError,match='policy-head-only'):checkpoint.candidate_sources(c.artifact(path))


def test_accepted_controller_and_search_science_are_unmodified():
    old=c.ROOT/'tools/splendorust_selfplay_pilot_resume2_20261103/assessment'
    new=Path(control.__file__).parent
    for name in ('control.py','transport.py','scientific.py','corpus.py','graph.py'):
        expected=(old/name).read_text().replace('tools.splendorust_selfplay_pilot_resume2_20261103','tools.splendorust_policy_head_20261104')
        assert (new/name).read_text()==expected,name
    assert c.ROUTE_QUOTAS=={'control':128,'candidate':128} and c.MAIN_GAMES==400
    assert len(control.lifecycle_inventory())==804 and c.FORECAST_POLICY=='advisory_only'


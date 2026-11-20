"""Saved256 admission-failure reproduction; no learned model or native process."""
import ast
from copy import deepcopy
import gzip
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import patch

import pytest

from tools import splendorust_search256_accounting_20261029 as fixed
from tools.splendorust_search128_resume_20261029 import accounting as original
from tools.splendorust_search256_strength_20261029 import common as c, transport, worker
from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile

ATTEMPT=c.ROOT/'outputs/splendorust_search256_strength_20261029_01'
FAILED=ATTEMPT/'workers/qualification_research'
CONTROL=ATTEMPT/'workers/qualification_frozen'
RID='request-00000001'


def payload():return c.read(FAILED/'startup_inputs'/f'{RID}.json')


def make_account(tmp_path, module):
    folder=tmp_path/'workers'/'fixture'
    account=module.WorkerAccounting(folder)
    folder.mkdir(parents=True)
    child=SimpleNamespace(output_dir=folder,_log_lock=threading.Lock(),
        _logs={name:SimpleNamespace() for name in module.LOG_NAMES},max_log_bytes=128*module.MiB,_closed=False)
    for name in module.LOG_NAMES:(folder/name).write_bytes(b'')
    account.bind_child(child);account.finish_initialization()
    return account,child,folder


def transport_child(tmp_path,module,quota):
    account,child,folder=make_account(tmp_path,module)
    identity=c.read(FAILED/'initialization.json')['identity']
    graph=c.read(FAILED/'boundary_timing_identity.json')['graph_identity_sha256']
    child.recovery_accounting=account;child._counter=0;child.challenger_quota=quota
    child.process=SimpleNamespace(pid=identity['pid']);child.placement_mode='isolated'
    child._placement_freeze={'unchanged_graph_backend_identity':{'graph_identity_sha256':graph}}
    child.recovery_envelope=SimpleNamespace(time_guard=lambda:None,remaining=lambda:60.)
    child.calls=[]
    def request(request_payload,timeout,validate=None):
        child.calls.append((deepcopy(request_payload),timeout));child._counter+=1
        return dict(status='ok',request_id=RID,deadline_seconds=timeout,
                    result={'model_free_transport_fixture':True})
    child.request=request
    return account,child,folder


def save_fixture(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')


def test_saved_failure_is_before_dispatch_and_not_a_player_timeout():
    failure=c.read(ATTEMPT/'failure.json')
    assert failure['stage']=='qualification'
    assert 'publication_identity' in failure['error'] and 'Fixed scientific quota' in failure['error']
    assert payload()['operation']=='graph_parity'
    assert gzip.decompress((FAILED/'events.jsonl.gz').read_bytes())==b''
    assert gzip.decompress((FAILED/'stderr.log.gz').read_bytes())==b''
    stdout=gzip.decompress((FAILED/'stdout.log.gz').read_bytes()).splitlines()
    assert len(stdout)==1 and json.loads(stdout[0])['type']=='ready'
    assert not list(FAILED.glob('request-*.json'))
    assert not list(FAILED.glob('*child_profile*')) and not (FAILED/'inputs').exists()
    assert not (FAILED/'graph_parity.json').exists()
    assert (FAILED/'graph_identity.json').exists()  # Initialization did build a model graph.
    cleanup=c.read(FAILED/'cleanup.json')
    assert cleanup['status']=='passed' and not cleanup['alive'] and not cleanup['errors']
    for name in ('stdout.log','stderr.log','events.jsonl'):
        raw=gzip.decompress((FAILED/(name+'.gz')).read_bytes());row=cleanup['log_compression'][name]
        assert len(raw)==row['raw_bytes'] and hashlib.sha256(raw).hexdigest()==row['raw_sha256']


def test_actual_saved256_payload_reproduces_original_predispatch_rejection(tmp_path):
    account,child,_=transport_child(tmp_path,original,256)
    with patch.object(transport.placement,'verify_runtime',return_value={'model_free_fixture':True}):
        with pytest.raises(original.AccountingError,match='Fixed scientific quota'):
            transport.request(child,payload(),60.)
    assert child.calls==[] and child._counter==0
    assert account.state=='idle' and account.active_request_id is None


@pytest.mark.parametrize('quota',[128,256])
@pytest.mark.parametrize('operation',['graph_parity','entrypoint_checks','ordinary'])
def test_corrected_live_admission_exact_publication_and_saved_readback(tmp_path,quota,operation):
    account,child,folder=transport_child(tmp_path,fixed,quota)
    request_payload=(payload() if operation=='graph_parity' else {'operation':'entrypoint_checks'}
        if operation=='entrypoint_checks' else c.read(CONTROL/'inputs/request-00000003.json'))
    seconds={'graph_parity':60.,'entrypoint_checks':300.,'ordinary':10.}[operation]
    with patch.object(transport.placement,'verify_runtime',return_value={'model_free_fixture':True}),patch.object(c,'save',side_effect=save_fixture):
        receipt=transport.request(child,request_payload,seconds)
    assert child.calls==[(request_payload,seconds)] and child._counter==1
    assert account.state=='request' and account.active_request_id==RID
    assert not child.last_placement_error and receipt['deadline_seconds']==seconds
    publication=deepcopy(account._publication)
    assert publication['expected']['quota']==quota
    raw=(json.dumps(dict(type='request',request_id=RID,payload=request_payload),separators=(',',':'),allow_nan=False)+'\n').encode()
    assert publication['expected']['request_raw_sha256']==hashlib.sha256(raw).hexdigest()
    # The real publisher writes a bounded sidecar; this is metadata-only, never scientific evidence.
    publish_profile(folder,RID,publication['expected'],batch=operation!='ordinary')
    proof=account.finish_request(require_published=True)
    assert proof['state']=='idle' and proof['reserved_bytes']==0
    child._closed=True
    assert account.close()['state']=='closed'


@pytest.mark.parametrize('quota',[0,1,32,255,257,512,True,False,256.,'256',None])
def test_other_quotas_and_boolean_coercions_still_rejected(tmp_path,quota):
    account,child,_=transport_child(tmp_path,fixed,quota)
    with patch.object(transport.placement,'verify_runtime',return_value={'model_free_fixture':True}):
        with pytest.raises(fixed.AccountingError,match='Fixed scientific quota'):
            transport.request(child,payload(),60.)
    assert child.calls==[] and account.state=='idle' and account.active_request_id is None


@pytest.mark.parametrize('quota',[128,256])
def test_saved_metadata_cannot_lie_about_quota(tmp_path,quota):
    account,child,folder=make_account(tmp_path,fixed)
    publication=account.begin_request(RID,payload(),pid=101,quota=quota)
    value=deepcopy(publication['publication']['expected']);value['quota']=128 if quota==256 else 256
    publish_profile(folder,RID,value,batch=True)
    with pytest.raises(fixed.AccountingError,match='identity differs'):
        account.finish_request(require_published=True)


@pytest.mark.parametrize('quota,route',[(128,'frozen'),(256,'research')])
@pytest.mark.parametrize('rid',[1,2,3])
def test_saved_reader_supports_exact256_graph_suite_and_ordinary_sidecars(quota,route,rid):
    # Metadata-only fixture derived from authentic completed128 clocks. No256
    # scientific output or timing is asserted; this exercises quota propagation.
    request_id=f'request-{rid:08d}'
    directory='startup_inputs' if rid<3 else 'inputs'
    request_payload=c.read(CONTROL/directory/(request_id+'.json'))
    parent=c.read(CONTROL/'parent_profiles'/(request_id+'.json'))
    sidecar=deepcopy(c.read(CONTROL/(request_id+'_child_profile.json')))
    receipt=c.read(CONTROL/(request_id+'.json'))
    sidecar['quota']=quota
    assert transport.validate_profiles(parent,sidecar,receipt,request_payload,route)['status']=='passed'
    sidecar['quota']=128 if quota==256 else 256
    with pytest.raises(ValueError,match='request/quota'):
        transport.validate_profiles(parent,sidecar,receipt,request_payload,route)


def test_new_accounting_source_has_only_the_one_allowlist_addition():
    before=Path(original.__file__).read_text();after=Path(fixed.__file__).read_text()
    marker="quota in (16, 64, 128), 'Fixed scientific quota'"
    assert before.count(marker)==1
    assert after==before.replace(marker,"quota in (16, 64, 128, 256), 'Fixed scientific quota'")
    for quota in (16,64,128):
        kwargs=dict(pid=101,quota=quota,graph_identity_sha256='a'*64)
        assert fixed.publication_identity(RID,payload(),**kwargs)==original.publication_identity(RID,payload(),**kwargs)


def test_frozen_chain_uses_general_helpers_and_local256_reader():
    # Older helpers contain quota-specific entrypoints; the frozen256 route
    # intentionally imports only their general byte/clock primitives.
    source=Path(transport.__file__).read_text()
    assert 'from tools.splendorust_live_boundary_20261025.raw_transport import _packet_evidence,_bind_parent_wire,_bind_child_wire,REQUEST_ID,MAX_FAILED_LINES' in source
    assert 'quota in (128,256)' in inspect.getsource(transport.verify_raw_transport)
    assert 'research_child_identity(' in inspect.getsource(transport.validate_profiles)
    assert 'boundaries._child_identity' not in source and 'boundaries.validate_startup_profiles' not in source
    assert 'original_graph_parity(context,request)' in inspect.getsource(worker.handle)
    assert 'from tools.splendorust_search128_resume_20261029.accounting import WorkerAccounting' in (c.ROOT/'tools/splendorust_search256_strength_20261029/runner.py').read_text()
    freeze=c.read(ATTEMPT/'freeze.json')
    paths={str(Path(original.__file__).relative_to(c.ROOT).as_posix()),'tools/splendorust_search256_strength_20261029/runner.py',
        'tools/splendorust_search256_strength_20261029/transport.py','tools/splendorust_search256_strength_20261029/worker.py'}
    pins=[x for x in freeze['inputs'] if x['path'] in paths]
    assert len(pins)==len(paths);c.verify(pins)

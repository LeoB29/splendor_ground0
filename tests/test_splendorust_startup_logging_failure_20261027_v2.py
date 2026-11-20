"""Saved-only failure evidence; no worker or learned model may run."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.splendorust_startup_logging_20261027_02 import failure_audit as f


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True),encoding='utf-8')


@pytest.fixture(scope='session')
def frozen():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    verify_runtime_sources()
    return load(f.PRIOR/'freeze.json')


@pytest.fixture(autouse=True)
def no_child_process(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Saved-only verifier attempted a subprocess')
    monkeypatch.setattr(subprocess,'Popen',forbidden)


@pytest.fixture
def clone(tmp_path):
    attempt=tmp_path/'attempt'
    folder=attempt/'workers/qualification_frozen_cpu2'
    shutil.copytree(f.PRIOR/'workers/qualification_frozen_cpu2',folder)
    for name in ('launch.json','freeze.json','watchdog_claim.json','guardian_os_exit.json',
                 'watchdog_execution.json','execution.json'):
        shutil.copyfile(f.PRIOR/name,attempt/name)
    return folder


def mutate_json(path,change):
    value=load(path);change(value);save(path,value)


def rechain(path,change):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    events=[row['event'] for row in rows];change(events)
    previous=None;lines=[]
    for index,event in enumerate(events):
        row=dict(sequence=index,previous_sha256=previous,event=event)
        lines.append(f.canonical(row));previous=f.digest(row)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def test_actual_failed_startup_is_verified_but_never_qualified(frozen):
    result=f.verify_failed_startup(f.PRIOR/'workers/qualification_frozen_cpu2',frozen,verify_owners=True)
    assert result['status']=='failed_startup_evidence_verified'
    assert result['qualification_passed'] is False and result['attempt_accepted'] is False
    assert result['complete_suite'] is False and result['failed_result_remains_unaccepted']
    assert result['original_clock']['deadline_seconds']==300
    assert result['original_clock']['status']=='worker_error'
    assert result['partial_science']['complete_decisions']==27
    assert result['partial_science']['successful_handlers']==18
    assert result['partial_science']['direct_references']==9
    assert result['partial_science']['failed_case_id']=='private_reserves'
    assert result['partial_science']['failed_result_accepted'] is False
    assert result['graph_cases']==11 and result['raw_requests']==2
    assert result['forensic_events']==153 and result['placement_requests']==2
    assert result['owner_exits']['guardian_exit']==result['owner_exits']['watchdog_exit']==result['owner_exits']['runner_exit']==1


def test_historical_finalization_preserves_incomplete_attempt():
    result=f.verify_saved_attempt()
    assert result['status']=='historical_failed_startup_forensics_verified'
    assert result['original_completion_unchanged']=='incomplete'
    assert result['qualification_passed'] is False and result['attempt_accepted'] is False
    assert result['original_evidence_files']==112
    assert result['required_experiment_exit_nonzero'] is True


def test_raw_error_packet_corruption_with_consistent_compression_hashes_rejected(clone,frozen):
    path=clone/'stdout.log.gz';raw=gzip.decompress(path.read_bytes())
    rows=raw.splitlines(keepends=True);packet=json.loads(rows[-1]);packet['error']='forged error'
    rows[-1]=(json.dumps(packet)+'\r\n').encode();raw=b''.join(rows)
    compressed=gzip.compress(raw,compresslevel=1,mtime=0);path.write_bytes(compressed)
    def update(value):
        value['log_compression']['stdout.log'].update(raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),
            compressed_bytes=len(compressed),compressed_sha256=hashlib.sha256(compressed).hexdigest())
    mutate_json(clone/'cleanup.json',update)
    with pytest.raises(ValueError,match='raw packet'):
        f.verify_failed_startup(clone,frozen)


@pytest.mark.parametrize('field,value',[('deadline_seconds',1.),('accepted',True),('status','ok'),('received_before_deadline',False)])
def test_failed_admin_clock_cannot_be_relabelled(clone,frozen,field,value):
    mutate_json(clone/'request-00000002.json',lambda x:x.update({field:value}))
    with pytest.raises(ValueError,match='administrative clock'):
        f.verify_failed_startup(clone,frozen)


def test_forensic_hashchain_corruption_rejected(clone,frozen):
    path=clone/'entrypoint_forensics.jsonl';rows=path.read_text().splitlines()
    row=json.loads(rows[20]);row['event']['corruption']=True;rows[20]=json.dumps(row)
    path.write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError,match='hash chain'):
        f.verify_failed_startup(clone,frozen)


def test_rehashed_missing_scientific_result_cannot_inflate_partial_counts(clone,frozen):
    def change(events):
        index=next(i for i,e in enumerate(events) if e.get('operation')=='ordinary_worker_request' and e['event']=='result')
        del events[index]
    rechain(clone/'entrypoint_forensics.jsonl',change)
    with pytest.raises(ValueError,match='partial handler counts'):
        f.verify_failed_startup(clone,frozen)


def test_rehashed_saved_scientific_decision_corruption_rejected(clone,frozen):
    def change(events):
        event=next(e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='validated')
        event['check']['corrupted']=True
    rechain(clone/'entrypoint_forensics.jsonl',change)
    with pytest.raises(ValueError,match='saved scientific check'):
        f.verify_failed_startup(clone,frozen)


def test_wrong_core_placement_rejected(clone,frozen):
    mutate_json(clone/'placement_requests/request-00000002.json',lambda x:x.update(logical_cpu=0))
    with pytest.raises((ValueError,RuntimeError)):
        f.verify_failed_startup(clone,frozen)


def test_missing_partial_handle_cannot_claim_same_completed_count(clone,frozen):
    mutate_json(clone/'request-00000002_child_profile.json',lambda x:x['profiles'].pop(0))
    mutate_json(clone/'entrypoint_profiles.json',lambda x:x['child_profile']['profiles'].pop(0))
    with pytest.raises(ValueError,match='partial boundary handle counts'):
        f.verify_failed_startup(clone,frozen)


def test_failed_handle_cannot_be_promoted_to_success(clone,frozen):
    mutate_json(clone/'request-00000002_child_profile.json',lambda x:x['profiles'][-1].update(handle_failed=False))
    mutate_json(clone/'entrypoint_profiles.json',lambda x:x['child_profile']['profiles'][-1].update(handle_failed=False))
    with pytest.raises((ValueError,RuntimeError)):
        f.verify_failed_startup(clone,frozen)


def test_imported_validator_source_must_be_frozen(frozen):
    changed=copy.deepcopy(frozen)
    item=next(x for x in changed['inputs'] if x['path']=='tools/splendorust_pool6_core_compare_20261027_02/runner.py')
    item['sha256']='0'*64
    with pytest.raises(ValueError,match='artifact hash differs'):
        f.verify_failed_startup(f.PRIOR/'workers/qualification_frozen_cpu2',changed)


def test_clean_owner_exit_cannot_promote_failed_experiment(clone,frozen):
    attempt=clone.parent.parent
    mutate_json(attempt/'execution.json',lambda x:x.update(exit_code=0))
    with pytest.raises(ValueError,match='nonzero actual exits'):
        f.verify_owner_exits(attempt,frozen)


def test_unconfirmed_owner_cleanup_rejected(clone,frozen):
    attempt=clone.parent.parent
    mutate_json(attempt/'watchdog_execution.json',lambda x:x['cleanup_receipt']['unconfirmed'].append({'pid':1,'created':1}))
    with pytest.raises(ValueError,match='owner cleanup errors'):
        f.verify_owner_exits(attempt,frozen)


def test_bounded_reader_detects_mid_audit_mutation(tmp_path):
    path=tmp_path/'receipt.json';path.write_text('{}')
    reader=f.Reader(tmp_path,lambda:None);reader.json(path);path.write_text('{"x":1}')
    with pytest.raises(ValueError,match='input changed'):
        reader.finish()


def test_unknown_startup_module_rejected_before_import(frozen):
    with pytest.raises(ValueError,match='explicit supported startup implementation'):
        f.verify_failed_startup(f.PRIOR/'workers/qualification_frozen_cpu2',frozen,
            source_module_prefix='tools.not_a_qualified_startup')


def test_explicit_new_module_requires_new_frozen_source_pins(frozen):
    with pytest.raises(ValueError,match='new startup validator source is not frozen'):
        f.verify_failed_startup(f.PRIOR/'workers/qualification_frozen_cpu2',frozen,
            source_module_prefix='tools.splendorust_startup_logging_20261027_02')


def test_test_only_new_controller_projection_retains_failed_status(clone,frozen):
    """Synthetic source-binding projection only, not a new worker execution."""
    prefix='tools.splendorust_startup_logging_20261027_02';changed=copy.deepcopy(frozen)
    for name in ('runner','placement','common','worker','logging_probe'):
        relative=prefix.replace('.','/')+'/'+name+'.py';path=f.ROOT/relative
        changed['inputs'].append(dict(path=relative,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    mutate_json(clone/'route_identity.json',lambda x:x.update(module=prefix+'.worker'))
    proof=f.verify_failed_startup(clone,changed,source_module_prefix=prefix)
    assert proof['source_module_prefix']==prefix
    assert proof['qualification_passed'] is False and proof['partial_science']['complete_decisions']==27

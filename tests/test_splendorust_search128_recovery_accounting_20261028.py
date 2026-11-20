"""Model-free deterministic reproduction and accounting boundary regressions."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import patch

import pytest

from tools.splendorust_search128_recovery_20261028 import accounting as a

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/splendorust_search128_strength_20261028_01'
WORKER = OLD/'workers/main_g38_challenger'
ID = 'request-00000021'
SAVED = WORKER/(ID+'_child_profile.json')


def saved():
    raw = SAVED.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '3ddc4f450fb98bdb74e3c48e318cd0a21a711761ccd808fa507452cfd388a0ff'
    value = json.loads(raw)
    payload = json.loads((WORKER/'inputs'/f'{ID}.json').read_text())
    return raw, value, payload


def make_account(tmp_path, *, reserve=1024*1024, max_entries=a.MAX_ENTRIES):
    folder = tmp_path/'workers'/'test_worker'
    account = a.WorkerAccounting(folder, growth_reserve_bytes=reserve, max_entries=max_entries)
    folder.mkdir(parents=True)
    child = SimpleNamespace(output_dir=folder, _log_lock=threading.Lock(),
        _logs={n: SimpleNamespace() for n in a.LOG_NAMES}, max_log_bytes=128*a.MiB, _closed=False)
    for name in a.LOG_NAMES:
        (folder/name).write_bytes(b'')
    account.bind_child(child)
    account.finish_initialization()
    return account, child, folder


def begin(account, payload=None):
    raw, value, old_payload = saved()
    saved_fixture = payload is None
    payload = old_payload if payload is None else payload
    account.begin_request(ID, payload, pid=value['pid'], quota=value['quota'],
                          graph_identity_sha256=value['graph_identity_sha256'])
    if saved_fixture:
        # The saved JSON input was canonicalized, losing original object-key
        # order. For this historical fixture ONLY, authenticate exact raw input
        # bytes through the original parent observer. Runtime begin_request uses
        # the live payload and computes its exact wire bytes before dispatch.
        parent = json.loads((WORKER/'parent_profiles'/f'{ID}.json').read_text())
        assert parent['request_raw_sha256'] == value['request_raw_sha256']
        account._publication['expected']['request_raw_sha256'] = parent['request_raw_sha256']
    return raw, value, payload


def test_exact_saved_failure_metadata_and_bytes_are_read_only():
    raw, sidecar, payload = saved()
    receipt = json.loads((WORKER/f'{ID}.json').read_text())
    assert receipt['status'] == 'resource_guard' and receipt['accepted'] is False
    assert receipt['result'] is None and receipt['receipt_elapsed_seconds'] is None
    assert 'request-00000021_child_profile.pending' in receipt['detail']
    assert sidecar['outcome'] == 'response_flushed' and sidecar['quota'] == 128
    identity = a.publication_identity(ID, payload, pid=sidecar['pid'], quota=128)
    assert all(sidecar[k] == v for k, v in identity['expected'].items() if k != 'request_raw_sha256')
    parent = json.loads((WORKER/'parent_profiles'/f'{ID}.json').read_text())
    assert parent['request_raw_sha256'] == sidecar['request_raw_sha256']
    assert len(raw) < a.MAX_PROFILE_BYTES


def test_original_scan_exact_pending_rename_deterministically_raises(tmp_path):
    """The inherited is_file/stat sequence fails; no model or live worker exists."""
    raw, _, _ = saved()
    folder = tmp_path/'original_race'; folder.mkdir()
    pending = folder/(ID+'_child_profile.pending')
    final = folder/(ID+'_child_profile.json')
    pending.write_bytes(raw)
    real = Path.is_file
    def is_file(path):
        answer = real(path)
        if path == pending and answer:
            os.rename(pending, final)
        return answer
    with patch.object(Path, 'is_file', is_file):
        with pytest.raises(FileNotFoundError) as error:
            sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
    assert error.value.filename == str(pending)
    assert final.read_bytes() == raw and not pending.exists()


def test_fixed_scan_accepts_only_authenticated_exact_saved_publication(tmp_path, monkeypatch):
    account, _, folder = make_account(tmp_path)
    raw, _, _ = begin(account)
    pending = folder/(ID+'_child_profile.pending'); pending.write_bytes(raw)
    final = folder/(ID+'_child_profile.json')
    real = os.stat
    triggered = []
    def stat(path, *args, **kwargs):
        if Path(path) == pending and not triggered:
            triggered.append(True); os.rename(pending, final)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(a.os, 'stat', stat)
    proof = account.snapshot()
    assert triggered and proof['pending_publications'][0]['request_id'] == ID
    assert proof['pending_publications'][0]['sha256'] == hashlib.sha256(raw).hexdigest()
    assert proof['accounted_bytes'] >= sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    assert final.read_bytes() == raw
    end = account.finish_request(require_published=True)
    assert end['reserved_bytes'] == 0 and end['observed_bytes'] == len(raw)


def test_pending_seen_before_rename_keeps_actual_file_identity(tmp_path):
    account, _, folder = make_account(tmp_path); raw, _, _ = begin(account)
    pending = folder/(ID+'_child_profile.pending'); pending.write_bytes(raw)
    before = account.snapshot()
    os.rename(pending, folder/(ID+'_child_profile.json'))
    proof = account.snapshot()
    assert proof['pending_publications'] and proof['accounted_bytes'] == before['accounted_bytes']
    assert account.finish_request(require_published=True)['reserved_bytes'] == 0


@pytest.mark.parametrize('field,value', [('pid', 1), ('quota',64), ('request_id','request-00000022'),
    ('payload_sha256','0'*64), ('request_raw_sha256','0'*64), ('graph_identity_sha256','0'*64),
    ('backend','research_graph'), ('schema','wrong')])
def test_publication_identity_corruption_fails(tmp_path, field, value):
    account, _, folder = make_account(tmp_path); raw, sidecar, _ = begin(account)
    pending = folder/(ID+'_child_profile.pending'); pending.write_bytes(raw); account.snapshot()
    sidecar[field] = value
    pending.write_bytes(json.dumps(sidecar).encode()+b'\n')
    os.rename(pending, folder/(ID+'_child_profile.json'))
    with pytest.raises(a.AccountingError): account.snapshot()


def test_pending_delete_and_different_replacement_is_not_rename(tmp_path):
    account, _, folder = make_account(tmp_path); raw, _, _ = begin(account)
    pending = folder/(ID+'_child_profile.pending'); pending.write_bytes(raw); account.snapshot()
    (folder/(ID+'_child_profile.json')).write_bytes(raw)
    pending.unlink()
    with pytest.raises(a.AccountingError, match='deleted/replaced'): account.snapshot()


def test_pending_delete_without_target_is_fatal(tmp_path):
    account, _, folder = make_account(tmp_path); raw, _, _ = begin(account)
    path = folder/(ID+'_child_profile.pending'); path.write_bytes(raw); account.snapshot(); path.unlink()
    with pytest.raises(FileNotFoundError): account.snapshot()


@pytest.mark.parametrize('name', ['other.pending', 'request-00000022_child_profile.pending',
                                 'request-21_child_profile.pending'])
def test_unregistered_pending_names_are_fatal(tmp_path, name):
    account, _, folder = make_account(tmp_path); begin(account); (folder/name).write_bytes(b'{}')
    with pytest.raises(a.AccountingError, match='Unauthenticated'): account.snapshot()


@pytest.mark.parametrize('mutation', ['delete','replace','shrink'])
def test_unrelated_evidence_loss_is_fatal(tmp_path, mutation):
    account, _, folder = make_account(tmp_path); path = folder/'evidence.json'; path.write_bytes(b'123456')
    account.snapshot()
    if mutation == 'delete': path.unlink()
    elif mutation == 'replace':
        other = folder/'replacement'; other.write_bytes(b'123456'); os.replace(other, path)
    else: path.write_bytes(b'1')
    with pytest.raises(a.AccountingError): account.snapshot()


def test_no_growing_directory_rescan_and_fixed_reserve_does_not_slide(tmp_path, monkeypatch):
    account, _, folder = make_account(tmp_path, reserve=100); begin(account)
    real = os.stat; added = []
    def stat(path, *args, **kwargs):
        if Path(path) == folder/'events.jsonl' and not added:
            added.append(True); (folder/'late.json').write_bytes(b'x'*90)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(a.os, 'stat', stat)
    proof = account.snapshot()
    assert proof['metadata_bytes'] == 0 and proof['accounted_bytes'] == 100
    nextproof = account.snapshot()
    assert nextproof['metadata_bytes'] == 90 and nextproof['accounted_bytes'] == 100
    (folder/'overflow.json').write_bytes(b'x'*11)
    with pytest.raises(a.AccountingError, match='reserve exceeded'): account.snapshot()


def test_byte_reserve_exact_boundary_and_strict_global_cap(tmp_path):
    account, _, folder = make_account(tmp_path, reserve=100); begin(account)
    (folder/'exact.json').write_bytes(b'x'*100)
    assert account.snapshot()['accounted_bytes'] == 100
    assert a.enforce_cap(100,101) == 100
    for cap in (99,100):
        with pytest.raises(a.AccountingError, match='ceiling'): a.enforce_cap(100,cap)
    with pytest.raises(a.AccountingError): a.enforce_cap(True,100)


def convert(child, folder, name, data=b'compressed'):
    target = folder/(name+'.gz'); target.write_bytes(data)
    child._logs[name] = SimpleNamespace(path=target, compressed_bytes=len(data))
    (folder/name).unlink()
    return target


def test_only_owned_empty_log_rotation_allowed(tmp_path):
    account, child, folder = make_account(tmp_path); begin(account)
    with child._log_lock: convert(child, folder, 'stdout.log')
    proof = account.snapshot()
    assert proof['empty_log_conversions'] == [dict(empty_placeholder='stdout.log',compressed='stdout.log.gz')]
    assert proof['compressed_stream_bytes'] == len(b'compressed')


def test_real_unchanged_compressor_and_lock_are_compatible(tmp_path):
    from tools.splendorust_search_scale_20261024.compression import StreamingGzipLog
    account, child, folder = make_account(tmp_path); begin(account)
    with child._log_lock:
        writer = StreamingGzipLog(folder/'stdout.log.gz',max_raw_bytes=128*a.MiB)
        child._logs['stdout.log'] = writer; (folder/'stdout.log').unlink()
        writer.write(b'unchanged transport bytes\n')
    proof = account.snapshot()
    assert proof['compressed_stream_bytes'] == writer.compressed_bytes
    with child._log_lock: writer.close_and_verify()
    assert account.snapshot()['compressed_stream_bytes'] == writer.compressed_bytes


@pytest.mark.parametrize('failure', ['missing_gzip','wrong_owner','nonempty','both'])
def test_invalid_log_rotation_fails(tmp_path, failure):
    account, child, folder = make_account(tmp_path); begin(account)
    if failure == 'missing_gzip': (folder/'stdout.log').unlink()
    elif failure == 'nonempty': (folder/'stdout.log').write_bytes(b'x')
    elif failure == 'wrong_owner':
        (folder/'stdout.log.gz').write_bytes(b'x'); (folder/'stdout.log').unlink()
    else:
        (folder/'stdout.log.gz').write_bytes(b'x')
        child._logs['stdout.log'] = SimpleNamespace(path=folder/'stdout.log.gz',compressed_bytes=1)
    with pytest.raises(a.AccountingError): account.snapshot()


def test_log_lock_timeout_is_bounded_and_does_not_mutate_evidence(tmp_path):
    account, child, folder = make_account(tmp_path); before = dict(account._known)
    with child._log_lock:
        with pytest.raises(a.AccountingError, match='log-lock deadline'): account.snapshot(lock_timeout=0)
    assert account._known == before
    assert account.snapshot()['stream_snapshot_under_original_log_lock']


def test_guard_is_not_called_under_an_unbounded_wait_and_reentry_fails(tmp_path):
    account, _, _ = make_account(tmp_path)
    def deadline(): raise TimeoutError('shared deadline')
    with pytest.raises(TimeoutError): account.snapshot(boundary=deadline)
    with pytest.raises(a.AccountingError, match='reenter'): account.snapshot(boundary=lambda: account.snapshot())
    assert account.snapshot()['state'] == 'idle'


def test_failed_close_retains_reserve_until_late_parent_evidence_collected(tmp_path):
    account, child, folder = make_account(tmp_path, reserve=4096); begin(account)
    child._closed = True
    closed = account.close()
    assert closed['state'] == 'request' and closed['accounted_bytes'] == 4096
    (folder/'late_parent.json').write_bytes(b'x'*100)
    assert account.snapshot()['accounted_bytes'] == 4096
    ended = account.finish_request(require_published=False)
    assert ended['state'] == 'closed' and ended['accounted_bytes'] == 100
    (folder/'final_audit.json').write_bytes(b'123')
    assert account.close()['accounted_bytes'] == 103


def test_failed_pending_bytes_preserved_without_fabricating_final(tmp_path):
    account, child, folder = make_account(tmp_path); begin(account)
    path = folder/(ID+'_child_profile.pending'); path.write_bytes(b'{"partial":')
    child._closed=True; account.close()
    proof = account.finish_request(require_published=False)
    assert proof['state']=='closed' and proof['metadata_bytes']==len(b'{"partial":')
    assert not (folder/(ID+'_child_profile.json')).exists()
    path.unlink()
    with pytest.raises(a.AccountingError, match='disappeared'): account.snapshot()


def test_success_needs_final_and_forbids_overlapping_requests(tmp_path):
    account, _, _ = make_account(tmp_path); _, value, payload = begin(account)
    with pytest.raises(a.AccountingError, match='Overlapping'): account.begin_request('request-00000022',payload,pid=value['pid'],quota=128)
    with pytest.raises(a.AccountingError, match='lacks'): account.finish_request(require_published=True)
    assert account.active_request_id==ID


def test_initialization_missing_folder_reserve_then_bound_tree(tmp_path):
    folder=tmp_path/'workers'/'fresh'
    account=a.WorkerAccounting(folder)
    assert account.snapshot()['accounted_bytes']==128*a.MiB
    folder.mkdir(parents=True)
    with pytest.raises(a.AccountingError, match='original log lock'): account.snapshot()


def test_entry_count_bound_and_directory_identity(tmp_path):
    account, _, folder = make_account(tmp_path,max_entries=5)
    (folder/'a').write_bytes(b'');(folder/'b').write_bytes(b'');(folder/'c').write_bytes(b'')
    with pytest.raises(a.AccountingError, match='entry bound'): account.snapshot()


def test_bound_folder_disappearance_and_access_failure_are_fatal(tmp_path, monkeypatch):
    account, _, folder = make_account(tmp_path)
    real=os.stat
    def missing(path,*args,**kwargs):
        if Path(path)==folder: raise FileNotFoundError(str(folder))
        return real(path,*args,**kwargs)
    monkeypatch.setattr(a.os,'stat',missing)
    with pytest.raises(a.AccountingError,match='disappeared'): account.snapshot()
    def denied(path,*args,**kwargs):
        if Path(path)==folder: raise PermissionError(str(folder))
        return real(path,*args,**kwargs)
    monkeypatch.setattr(a.os,'stat',denied)
    with pytest.raises(PermissionError): account.snapshot()


def test_startup_profile_uses512k_bound_ordinary8k():
    assert a.publication_identity(ID,{'operation':'entrypoint_checks'},pid=10,quota=128)['maximum_bytes']==512*1024+1
    assert a.publication_identity(ID,{'player':'search','observer':{}},pid=10,quota=128)['maximum_bytes']==8193


def test_live_payload_wire_hash_matches_unchanged_original_encoding():
    payload={'player':'search','observer':{'z':1,'a':2}}
    expected=(json.dumps({'type':'request','request_id':ID,'payload':payload},separators=(',', ':'),allow_nan=False)+'\n').encode()
    assert a.publication_identity(ID,payload,pid=10,quota=128)['expected']['request_raw_sha256']==hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize('request_id', ['../request-00000021','request-00000021\n','request-1',False])
def test_path_request_rejections(request_id):
    with pytest.raises(a.AccountingError): a.publication_identity(request_id,{},pid=10,quota=128)


def test_no_model_imports_or_frozen_source_mutation_in_accounting():
    import ast
    tree=ast.parse(Path(a.__file__).read_text())
    imports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imports|={name.name for node in ast.walk(tree) if isinstance(node,ast.Import) for name in node.names}
    assert all(not name or name.split('.')[0] in {'__future__','contextlib','hashlib','json','math','os','pathlib','re','stat','threading'} for name in imports)

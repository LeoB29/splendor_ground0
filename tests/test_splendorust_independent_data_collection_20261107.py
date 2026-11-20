"""Model-free storage, source isolation, schedule and rejection counterexamples."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import time
import zlib
import pytest

from tools.splendorust_independent_data_20261107.collection import storage,common as cc,private,data,collector,qualification,audit,control,worker
from tools.splendorust_independent_data_20261107 import common as root,schedule


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(root,'ROOT',tmp_path)
    monkeypatch.setattr(root,'OUT',tmp_path/'new')
    monkeypatch.setattr(cc,'ROOT',tmp_path)
    monkeypatch.setattr(cc,'OUT',tmp_path/'new/collection')
    monkeypatch.setattr(root,'WRITE_OBSERVER',None)
    return cc.OUT


@pytest.mark.parametrize('raw',[b'{"x":1}\n',b'\r\n { "x" : [1.0000001, -0.0], "y":"caf\xc3\xa9"}\n',b'{"x":"'+b'x'*100000+b'"}\n'],ids=['simple','exact_float_unicode','large_repetition'])
def test_exact_bytes_roundtrip(raw):
    encoded=storage.encode_bytes(raw)
    assert storage.decode_bytes(encoded)==raw
    assert storage.encode_bytes(raw)==encoded
    assert storage.parse_bytes(encoded)==json.loads(raw)


@pytest.mark.parametrize('mode',['header','payload','crc','hash','trailing','member','length','huge'])
def test_compressed_corruption_rejected(mode):
    raw=b'{"x":"'+b'abcde'*1000+b'"}\n';encoded=storage.encode_bytes(raw)
    if mode=='header':bad=encoded[:20]
    elif mode=='payload':bad=encoded[:-10]
    elif mode=='crc':bad=encoded[:-5]+bytes([encoded[-5]^1])+encoded[-4:]
    elif mode=='hash':bad=encoded[:16]+bytes([encoded[16]^1])+encoded[17:]
    elif mode=='trailing':bad=encoded+b'x'
    elif mode=='member':bad=encoded+gzip.compress(b'',mtime=0)
    elif mode=='length':bad=encoded[:8]+struct.pack('<Q',len(raw)-1)+encoded[16:]
    else:bad=encoded[:8]+struct.pack('<Q',storage.MAX_RAW+1)+encoded[16:]
    with pytest.raises((ValueError,zlib.error)):storage.decode_bytes(bad)


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'])
def test_strict_json_semantics(raw):
    with pytest.raises(ValueError):storage.parse_bytes(storage.encode_bytes(raw))


def test_real_save_reader_physical_authentication(isolated):
    value={'signed_zero':-0.0,'nested':{'cost':[0,2,7]},'unicode':'é'}
    path=isolated/'worker/cell/games/g0/moves/000000.json'
    item=cc.save(path,value)
    assert path.read_bytes().startswith(storage.MAGIC)
    assert data.read_artifact(item)==value and cc.read(path)==value
    assert item['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):cc.save(path,value)
    bad=deepcopy(item);bad['sha256']='0'*64
    with pytest.raises(ValueError):data.read_artifact(bad)
    bad=deepcopy(item);bad['bytes']=True
    with pytest.raises(ValueError):data.read_artifact(bad)
    plain=isolated/'worker/cell/games/g0/clocks/000000.json';cc.save(plain,value)
    assert not plain.read_bytes().startswith(storage.MAGIC)


def test_journal_durable_seal_and_stream_roundtrip(isolated):
    path=isolated/'worker/runtime/inference.jsonl';path.parent.mkdir(parents=True)
    raw=b''.join((json.dumps({'seq':i,'x':'abc'*17},sort_keys=True)+'\n').encode() for i in range(100))
    path.write_bytes(raw);commit=storage.seal_journal(path)
    receipt=cc.read(path.with_name(path.name+'.seal_receipt.json'))
    assert receipt['raw']['sha256']==hashlib.sha256(raw).hexdigest()
    sink=io.BytesIO();checked=storage.stream_journal(path,sink)
    assert sink.getvalue()==raw and checked==receipt['roundtrip']
    assert cc.read(cc.resolve(commit))['stored']==cc.artifact(path)
    assert not path.with_name(path.name+'.sealing').exists()


@pytest.mark.parametrize('mode',['truncated','trailing','member','hash','length','crc'])
def test_journal_saved_failure_rejection(isolated,mode):
    path=isolated/'worker/runtime/inference.jsonl';path.parent.mkdir(parents=True)
    path.write_bytes(b'{"event":0}\n'*100);storage.seal_journal(path);raw=path.read_bytes()
    if mode=='truncated':bad=raw[:-1]
    elif mode=='trailing':bad=raw+b'x'
    elif mode=='member':bad=raw+gzip.compress(b'',mtime=0)
    elif mode=='hash':bad=raw[:16]+bytes([raw[16]^1])+raw[17:]
    elif mode=='length':bad=raw[:8]+struct.pack('<Q',99)+raw[16:]
    else:bad=raw[:-7]+bytes([raw[-7]^1])+raw[-6:]
    path.write_bytes(bad)
    with pytest.raises((ValueError,zlib.error)):storage.stream_journal(path)


def test_failed_sealing_preserves_original_and_partial(isolated,monkeypatch):
    path=isolated/'worker/runtime/inference.jsonl';path.parent.mkdir(parents=True);raw=b'{"event":0}\n'*100;path.write_bytes(raw)
    monkeypatch.setattr(storage,'stream_journal',lambda *a,**k:(_ for _ in ()).throw(ValueError('saved failure fixture')))
    with pytest.raises(ValueError):storage.seal_journal(path)
    assert path.read_bytes()==raw and path.with_name(path.name+'.sealing').exists()
    assert not path.with_name(path.name+'.seal_commit.json').exists()


def test_original_sources_and_public_module_context_unchanged():
    from tools.splendorust_gpu_collector_20261105 import collector as old,data as olddata
    original_common=old.c;original_data=old.data
    own=private.load('collector')
    assert own is not old and own.c is cc and old.c is original_common and old.data is original_data
    for name in ('decide_snapshot','_play_game','run_replays','run_cell','warmup_runtime'):
        assert getattr(own,name).__code__.co_code==getattr(old,name).__code__.co_code
    assert data.decode_row is olddata.decode_row and data.collate_arrays is olddata.collate_arrays


def test_frozen_scientific_collection_counts():
    specs=collector.build_plan()['specifications'];cells=collector.cells()
    assert len(specs)==1152 and len(cells)==72
    assert [s for cell in cells for s in cell['specifications']]==specs
    assert all(cell['concurrency']==8 and cell['scheduler']=='original8' and cell['route']=='cpu' for cell in cells)
    assert all(s['quota']==512 and s['world_pool']==3 and s['cpuct']==.4 for s in specs)
    assert len({s['seed_decimal'] for s in specs})==1152
    engineering=schedule.qualification_schedule()
    assert len(engineering)==2 and not {s['seed_decimal'] for s in specs}&{s['seed_decimal'] for s in engineering}
    assert qualification.build_plan()['decisions_per_route']==116


def test_production_manifest_cannot_be_engineering_or_bool_substitute():
    specs=schedule.collection_schedule()[:1];items=[{'path':'new.json','bytes':1,'sha256':'a'*64}]
    manifest=data.build_manifest(items,specs,route='cpu',cell_id='c000');data.validate_manifest(manifest,specs)
    assert manifest['purpose']=='fresh_corrected_production_training_source'
    bad=deepcopy(manifest);bad['teacher_simulations']=512.0
    bad['metadata']['data_sha256']=cc.digest({k:v for k,v in bad.items() if k!='metadata'})
    with pytest.raises(ValueError):data.validate_manifest(bad,specs)
    bad=deepcopy(manifest);bad['purpose']='collector_qualification_no_training'
    bad['metadata']['data_sha256']=cc.digest({k:v for k,v in bad.items() if k!='metadata'})
    with pytest.raises(ValueError):data.validate_manifest(bad,specs)


def test_live_registry_paths_are_bounded():
    p=Path('new/worker')
    assert control.active_registry_paths(p,'qualification')==[p/'cpu_runtime/active_decisions.json']
    assert control.active_registry_paths(p,'games')==[p/'runtime/active_decisions.json']
    assert control.active_registry_paths(p,'audit_cell')==[]


def test_atomic_status_helper_admits_only_supported_names(isolated):
    for name in ('stage.json', 'progress.json'):
        path=isolated/name
        cc.atomic_json(path, {'step':1})
        cc.atomic_json(path, {'step':2})
        assert cc.read(path)=={'step':2}
    with pytest.raises(ValueError):cc.atomic_json(isolated/'active_decisions.json', {})


def test_saved_real_journal_unchanged_scanner(isolated,monkeypatch):
    # Saved original real callbacks, no inference, simulation or fabricated times.
    source=Path(__file__).resolve().parents[1]/'outputs/splendorust_gpu_collector_20261105_01/cells/b1_cpu/worker/runtime/inference.jsonl'
    with source.open('rb') as stream:raw=b''.join(next(stream) for _ in range(100))
    path=isolated/'worker/runtime/inference.jsonl';path.parent.mkdir(parents=True);path.write_bytes(raw)
    storage.seal_journal(path)
    claim=isolated/'claim.json';cc.save(claim,dict(monotonic_origin=time.monotonic()-1,unix_origin=time.time()-1))
    monkeypatch.setattr(cc,'CLAIM',claim)
    with audit.JournalIndex(path,'cpu','original8') as journal:
        assert journal.request_count==journal.batch_count==100
        assert journal.journal_artifact==cc.artifact(path)
        ids=[x[0] for x in journal.db.execute('SELECT id FROM requests')]
        journal.mark(ids,'model_free_saved_fixture')
        proof=journal.complete_attribution()
        assert proof['lossless_storage']['raw']['sha256']==hashlib.sha256(raw).hexdigest()
        assert proof['all_callbacks_attributed_once'] is True
    assert list((isolated/'audit_scratch').glob('journal_*'))==[]


def test_actual_target_parser_rejects_wrong_identity_and_value():
    from tools.splendorust_independent_data_20261107 import corpus
    source=Path(__file__).resolve().parents[1]/'outputs/splendorust_gpu_collector_20261105_01/cells/b1_cpu/worker/cell/targets/g000.json'
    target=json.loads(source.read_bytes());spec=deepcopy(target['spec']);spec['split']='engineering';target['spec']=spec
    assert corpus._parse_target(target,spec)['examples']
    bad=deepcopy(target);bad['origin']='historical'
    with pytest.raises(ValueError):corpus._parse_target(bad,spec)
    bad=deepcopy(target);bad['examples'][0]['value']=[1.,1.]
    with pytest.raises(ValueError):corpus._parse_target(bad,spec)
    bad=deepcopy(target);bad['examples'][0]['mask'][0]=1
    with pytest.raises(ValueError):corpus._parse_target(bad,spec)

"""Independent saved-boundary, single-use amendment, and clock counterexamples."""
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

import psutil
import pytest

from tools.splendorust_full_corpus_recovery_20261111 import boundary as b, common as c


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


@pytest.fixture
def authorization(tmp_path, monkeypatch):
    old = tmp_path/'old'; new = tmp_path/'new'
    monkeypatch.setattr(c,'ORIGINAL_OUT',old)
    monkeypatch.setattr(c,'OUT',new)
    monkeypatch.setattr(c,'PLAN',new/'freeze.json')
    monkeypatch.setattr(c,'AUTHORIZATION',new/'authorization.json')
    write(old/'freeze.json',{'original':True})
    monkeypatch.setattr(c,'ORIGINAL_PLAN_SHA',c.sha(old/'freeze.json'))
    inspection = write(tmp_path/'inspection.json',dict(root_actual_exit_code=1,interrupted_game=b.PARTIAL,
        completed_plies=22,next_ply=22,ambiguous_worker='main_g030_control',ambiguous_request=70))
    value = dict(schema=b.AUTH_SCHEMA,retry_spec=deepcopy(c.RETRY_SPEC),user_reply='Authorize one disclosed retry.',
        original_freeze=c.artifact(old/'freeze.json'),inspection=c.artifact(inspection),
        original_attempt_remains_incomplete=True,deadline_extension_seconds=0,new_training_calls=0,replacement_games=0)
    write(c.AUTHORIZATION,value)
    return old,new,value


def test_exact_amendment_accepts_and_retains_dispatch_ambiguity(authorization):
    assert b.validate_retry_authorization()['retry_spec']==c.RETRY_SPEC


@pytest.mark.parametrize('field,value',[
    ('ply',21),('ply',23),('ply',True),('game_index',33),('repetition',0),('repetition',True),
    ('matchup','mixed_parent'),('route','candidate'),('stage','operational'),('old_request_id',69),
    ('old_worker','main_g030_candidate'),('maximum_retries',2),('internal_seconds',10.),
    ('complete_response_seconds',11.),
])
def test_amendment_cannot_widen_to_another_decision(authorization,field,value):
    auth=deepcopy(authorization[2]);auth['retry_spec'][field]=value
    with pytest.raises(ValueError):b.validate_retry_authorization(auth)


@pytest.mark.parametrize('field,value',[
    ('original_attempt_remains_incomplete',False),('replacement_games',1),('new_training_calls',1),
    ('deadline_extension_seconds',1),('deadline_extension_seconds',False),('user_reply',''),
])
def test_amendment_cannot_relabel_or_extend_the_original(authorization,field,value):
    auth=deepcopy(authorization[2]);auth[field]=value
    with pytest.raises(ValueError):b.validate_retry_authorization(auth)


def test_authorization_rejects_changed_inspection_bytes(authorization):
    auth=deepcopy(authorization[2])
    write(c.resolve(auth['inspection']),{'next_ply':22})
    with pytest.raises(ValueError):b.validate_retry_authorization(auth)


@pytest.fixture
def consumer(authorization,monkeypatch):
    old,new,auth=authorization
    before={'immutable_beforestate':'the original row 21 afterstate'}
    monkeypatch.setattr(b,'retry_beforestate',lambda:deepcopy(before))
    write(old/b.PARTIAL/'moves/000021.json',{'after':before})
    write(old/b.WORKERS/'main_g030_control/inputs/request-00000070.json',{'observer':'original'})
    mono,wall=time.monotonic(),time.time()
    base=dict(monotonic_origin=mono-100.,unix_origin=wall-100.,total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS)
    write(old/'claim.json',base);write(new/'claim.json',dict(base,pid=100,created=1.))
    phase=dict(monotonic=mono-50.,unix_time=wall-50.,monotonic_deadline=mono+300.,unix_deadline=wall+300.,
               stage='strength',cap_seconds=64*3600)
    write(old/'stages/strength_start.json',phase);write(new/'stages/strength_start.json',phase)
    write(c.PLAN,{'inputs':[c.artifact(c.AUTHORIZATION)]})
    start_path,segment_path=b._consumer_paths()
    segment=dict(matchup='terminal_parent',repetition=1,mode='resume_partial',block_start=None,
                 claim=c.artifact(new/'claim.json'),phase_clock=c.artifact(new/'stages/strength_start.json'),
                 freeze_sha256=c.sha(c.PLAN))
    write(segment_path,segment)
    start=dict(owner=dict(pid=os.getpid(),created=psutil.Process().create_time()),phase='resume_partial',
        matchup='terminal_parent',repetition=1,root_claim=segment['claim'],root_phase_clock=segment['phase_clock'],
        segment_claim=c.artifact(segment_path),monotonic=mono-1.,unix_time=wall-1.)
    write(start_path,start)
    return dict(before=before,path=new/'authorized_retry_consumed.json',args=dict(matchup='terminal_parent',repetition=1,
        stage='main',game_index=32,ply=22,route='control',worker_name='main_g032_control',request_id='request-00000003'))


def consume(fixture,**changed):
    return b.consume_authorized_retry(fixture['path'],fixture['before'],**dict(fixture['args'],**changed))


def test_authorized_retry_is_durable_and_nonrenewable(consumer):
    item=consume(consumer)
    value=b.verify_retry_consumption()
    assert c.artifact(consumer['path'])==item
    assert value['consumed_before_transport'] is True and value['reusable_after_failure'] is False
    with pytest.raises(FileExistsError):consume(consumer)
    assert c.artifact(consumer['path'])==item


@pytest.mark.parametrize('changed',[
    {'ply':21},{'ply':23},{'ply':True},{'game_index':33},{'repetition':0},{'repetition':True},
    {'route':'candidate'},{'matchup':'mixed_parent'},{'worker_name':'main_g030_control'},
    {'request_id':'request-00000004'},{'request_id':3},
])
def test_other_turn_or_worker_never_spends_the_exception(consumer,changed):
    with pytest.raises(ValueError):consume(consumer,**changed)
    assert not consumer['path'].exists()


def test_changed_beforestate_or_token_destination_rejected(consumer):
    with pytest.raises(ValueError):
        b.consume_authorized_retry(consumer['path'],{'different':True},**consumer['args'])
    with pytest.raises(ValueError):
        b.consume_authorized_retry(consumer['path'].with_name('another.json'),consumer['before'],**consumer['args'])


@pytest.mark.parametrize('filename,key',[
    ('claim.json','monotonic_origin'),('claim.json','unix_origin'),('claim.json','total_seconds'),
    ('claim.json','model_seconds'),('stages/strength_start.json','monotonic'),
    ('stages/strength_start.json','unix_time'),('stages/strength_start.json','monotonic_deadline'),
    ('stages/strength_start.json','unix_deadline'),('stages/strength_start.json','cap_seconds'),
])
def test_retry_cannot_restart_any_original_clock(consumer,filename,key):
    path=c.OUT/filename;value=c.read(path);value[key]+=1;write(path,value)
    with pytest.raises(ValueError):consume(consumer)
    assert not consumer['path'].exists()


@pytest.mark.parametrize('clock',['monotonic','time'])
def test_either_original_deadline_expiry_rejects_token(consumer,monkeypatch,clock):
    phase=c.read(c.OUT/'stages/strength_start.json')
    deadline=phase['monotonic_deadline' if clock=='monotonic' else 'unix_deadline']
    monkeypatch.setattr(b.time,clock,lambda:deadline)
    with pytest.raises(ValueError):consume(consumer)
    assert not consumer['path'].exists()


def test_wrong_controller_cannot_consume_token(consumer):
    path,_=b._consumer_paths();value=c.read(path);value['owner']['created']+=1;write(path,value)
    with pytest.raises(ValueError):consume(consumer)


def test_authorization_must_be_in_exact_recovery_freeze(consumer):
    write(c.PLAN,{'inputs':[]})
    with pytest.raises(ValueError):consume(consumer)


def test_consumed_token_mutation_rejected(consumer):
    consume(consumer);value=c.read(consumer['path']);value['amendment_retry_number']=2
    with pytest.raises(ValueError):b.verify_retry_consumption(value)


def unclosed_gzip(raw):
    encoder=zlib.compressobj(wbits=16+zlib.MAX_WBITS)
    return encoder.compress(raw)+encoder.flush(zlib.Z_SYNC_FLUSH)


def test_only_complete_jsonl_records_in_unclosed_prefix_are_read(tmp_path):
    path=tmp_path/'stream.gz';path.write_bytes(unclosed_gzip(b'{"request_id":"request-00000069"}\n'))
    original=path.read_bytes()
    lines,proof=b._read_unclosed_stream(path,c.sha(path))
    assert len(lines)==1 and proof['gzip_eof'] is False and proof['unclosed_stream_preserved'] is True
    assert path.read_bytes()==original


@pytest.mark.parametrize('raw',[
    b'{"id":1}',b'{"id":1}\n{"partial":',b'{"id":1,"id":2}\n',b'{"v":NaN}\n',b'not-json\n',
])
def test_partial_malformed_or_duplicate_raw_frames_rejected(tmp_path,raw):
    path=tmp_path/'stream.gz';path.write_bytes(unclosed_gzip(raw))
    with pytest.raises((ValueError,json.JSONDecodeError)):b._read_unclosed_stream(path,c.sha(path))


def test_footer_repair_or_changed_compressed_bytes_not_accepted(tmp_path):
    path=tmp_path/'stream.gz';path.write_bytes(gzip.compress(b'{"id":1}\n'))
    with pytest.raises(ValueError):b._read_unclosed_stream(path,c.sha(path))
    path.write_bytes(unclosed_gzip(b'{"id":1}\n'));before=c.sha(path)
    path.write_bytes(unclosed_gzip(b'{"id":2}\n'))
    with pytest.raises(ValueError):b._read_unclosed_stream(path,before)


@pytest.fixture
def original_inventory(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ORIGINAL_OUT',tmp_path/'original')
    for matchup,repetitions in c.COMPLETED_PREFIXES.items():
        for rep,count in repetitions.items():
            for stage,limit in (('main',count),('operational',6)):
                for index in range(limit):
                    folder=c.ORIGINAL_OUT/f'strength/{matchup}/r{rep}/games/{stage}/g{index:03d}'
                    write(folder/'result.json',dict(complete=True,reason='natural',game_index=index,stage=stage,
                                                  matchup=matchup,repetition=rep))
                    write(folder/'online_replay.json',{})
                    write(folder/'cost.json',{})
    partial=c.ORIGINAL_OUT/b.PARTIAL
    write(partial/'spec.json',{})
    write(partial/'initial.json',{})
    write(partial/'workers.json',dict(control='main_g030_control',candidate='main_g030_candidate'))
    for ply in range(22):write(partial/'moves'/f'{ply:06d}.json',{})
    return partial


def test_exact_212_game_inventory_and_22_move_prefix(original_inventory):
    assert b._prefix_inventory()==212


@pytest.mark.parametrize('mutation',[
    'missing_move','extra_move','invented_partial_result','invented_partial_replay',
    'hidden_extra_game','wrong_completed_seat_cohort','missing_original_replay','replacement_forfeit',
])
def test_completed_prefix_and_partial_boundary_cannot_be_replaced(original_inventory,mutation):
    partial=original_inventory
    if mutation=='missing_move':(partial/'moves/000021.json').unlink()
    elif mutation=='extra_move':write(partial/'moves/000022.json',{})
    elif mutation=='invented_partial_result':write(partial/'result.json',{})
    elif mutation=='invented_partial_replay':write(partial/'online_replay.json',{})
    elif mutation=='hidden_extra_game':write(partial.parent/'g033/spec.json',{})
    else:
        folder=partial.parent/'g031'
        if mutation=='missing_original_replay':(folder/'online_replay.json').unlink()
        else:
            value=c.read(folder/'result.json')
            value['repetition' if mutation=='wrong_completed_seat_cohort' else 'reason']=0 if mutation=='wrong_completed_seat_cohort' else 'forfeit'
            write(folder/'result.json',value)
    with pytest.raises(ValueError):b._prefix_inventory()


def test_actual_saved_boundary_full_corrected_and_raw_audit():
    result=b.run_saved_audit()
    assert result['checked_partial_moves']==22 and result['checked_completed_block_moves']==112
    assert result['raw_requests']==138 and result['failed_saved_requests']==0
    assert result['unknown_dispatch_ply']==22 and result['unknown_dispatch_request']==70
    assert result['dispatch_never_started_proven'] is False
    assert result['models']==result['games']==0
    assert result['before_sha256']=='44d0664f2b03ce38b39aba4cf6c9b849986764c80fa2519ea74af0611291fce7'
    assert [row['prepared_extra_request'] for row in result['workers']]==['request-00000070',None]
    assert result['helper_execution']['worker_exit_code']==0 and result['helper_execution']['identity_absent'] is True
    assert result['helper_execution']['original_python_sources_checked']>0
    assert result['runtime_sources']
    assert 'torch' not in sys.modules

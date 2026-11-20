"""Model-free original-byte parity and truthful total-callback evidence."""
import copy
import hashlib
import json
from pathlib import Path
import time

import pytest

from tools.splendorust_minimal_logging_20261028 import logging_probe as p
from tools.splendorust_search_scale_20261024.worker import _forensic_recorder


def event(index=0):
    return dict(event='complete_result_before_worker_validation', marker=index,
                observer=dict(public='only'), action=dict(take_tokens=['white','blue','red']),
                diagnostics=dict(callbacks=[dict(output=[.1,-.2,0.])]))


def fixture(tmp_path, enabled=True):
    recorder=p.make_recorder(tmp_path/'forensics.jsonl',enabled)
    start=time.perf_counter_ns();recorder.record(event());end=time.perf_counter_ns()
    profile=dict(profiles=[dict(handle_start_ns=start,handle_end_ns=end,handle_failed=False)])
    value=recorder.snapshot('request-00000003',0,profile,
        dict(kind='response',pre_write_ns=end,post_flush_ns=time.perf_counter_ns()),dict(sequence=0))
    recorder.close()
    return recorder,value,profile


@pytest.mark.parametrize('enabled',[False,True])
def test_exact_original_bytes_and_chain_for_complete_events(tmp_path,enabled):
    original,stream=_forensic_recorder(tmp_path/'original.jsonl')
    probe=p.make_recorder(tmp_path/'probe.jsonl',enabled)
    for index in range(4):original(event(index));probe.record(event(index))
    stream.close();probe.close()
    assert (tmp_path/'original.jsonl').read_bytes()==(tmp_path/'probe.jsonl').read_bytes()


def test_off_calls_only_original_recorder_and_reconstructs_entries_after_emission(tmp_path,monkeypatch):
    recorder=p.make_recorder(tmp_path/'off.jsonl',False)
    def forbidden():raise AssertionError('OFF callback must not read timing clock')
    monkeypatch.setattr(p.time,'perf_counter_ns',forbidden)
    recorder.record(event())
    assert recorder.record_count==1 and not hasattr(recorder,'entries')
    profile=dict(profiles=[])
    value=recorder.snapshot('r',0,profile,dict(post_flush_ns=100))
    recorder.close()
    assert len(value['records'])==1 and all(v is None for v in value['records'][0]['timestamps'].values())
    assert p.validate_probe_sidecar(value,recorder.path,False,'r',profile)['prefix_verified']
    assert value['summary']['total_ns'] is None


@pytest.mark.parametrize('enabled',[False,True])
def test_request_range_prefix_survives_later_appends(tmp_path,enabled):
    recorder,first,profile=fixture(tmp_path,enabled)
    # fixture closed the stream; exercise the real persistent path separately.
    recorder=p.make_recorder(tmp_path/'persistent.jsonl',enabled)
    recorder.record(event());first=recorder.snapshot('one',0,dict(profiles=[]),dict(post_flush_ns=time.perf_counter_ns()))
    begin=recorder.record_count;recorder.record(event(1))
    second=recorder.snapshot('two',begin,dict(profiles=[]),dict(post_flush_ns=time.perf_counter_ns()))
    recorder.close()
    assert p.validate_probe_sidecar(first,recorder.path,enabled,'one',dict(profiles=[]))['persisted_rows']==1
    assert p.validate_probe_sidecar(second,recorder.path,enabled,'two',dict(profiles=[]))['persisted_rows']==2
    assert [r['sequence'] for r in second['records']]==[1]


def test_total_span_and_original_handler_scope_recomputed(tmp_path):
    recorder,value,profile=fixture(tmp_path)
    proof=p.validate_probe_sidecar(value,recorder.path,True,'request-00000003',profile)
    row=proof['records'][0]
    stamps=value['records'][0]['timestamps']
    assert row['elapsed_ns']==stamps['end_ns']-stamps['start_ns']==proof['total_ns']
    assert set(stamps)=={'start_ns','end_ns'} and 'durations_ns' not in row
    assert row['scope']==dict(kind='inside_original_handler',handler_index=0)
    assert proof['original_player_outcome_not_inferred']


@pytest.mark.parametrize('change',['negative','bool','order','missing','extra','summary','child','mode','sequence','event','prefix','request','log'])
def test_timing_and_byte_proof_corruption_rejected(tmp_path,change):
    recorder,value,profile=fixture(tmp_path)
    row=value['records'][0]
    if change=='negative':row['timestamps']['start_ns']=-1
    if change=='bool':row['timestamps']['start_ns']=True
    if change=='order':row['timestamps']['end_ns']=row['timestamps']['start_ns']-1
    if change=='missing':row['timestamps']['end_ns']=None
    if change=='extra':row['timestamps']['invented_ns']=1
    if change=='summary':value['summary']['total_ns']+=1
    if change=='child':profile['profiles'][0]['handle_end_ns']+=1
    if change=='mode':value['instrumented']=False
    if change=='sequence':row['sequence']=3
    if change=='event':row['event']='different'
    if change=='prefix':value['prefix_sha256']='0'*64
    if change=='request':value['request_id']='other'
    if change=='log':value['log']='other.jsonl'
    with pytest.raises(ValueError):p.validate_probe_sidecar(value,recorder.path,True,'request-00000003',profile)


def test_changed_raw_bytes_cannot_pass_a_valid_timer_record(tmp_path):
    recorder,value,profile=fixture(tmp_path)
    recorder.path.write_bytes(recorder.path.read_bytes().replace(b'white',b'black'))
    with pytest.raises(ValueError,match='prefix hash'):
        p.validate_probe_sidecar(value,recorder.path,True,'request-00000003',profile)


def test_off_never_fabricates_timer_values(tmp_path):
    recorder,value,profile=fixture(tmp_path,False)
    value['records'][0]['timestamps']['start_ns']=1
    with pytest.raises(ValueError,match='OFF fabricated'):
        p.validate_probe_sidecar(value,recorder.path,False,'request-00000003',profile)


def test_fsync_failure_preserves_total_attempt_span_and_never_means_player_success(tmp_path,monkeypatch):
    recorder=p.make_recorder(tmp_path/'fail.jsonl',True)
    def fail(_):raise OSError('fixture fsync failure')
    monkeypatch.setattr(p.os,'fsync',fail)
    with pytest.raises(OSError,match='fsync failure'):recorder.record(event())
    profile=dict(profiles=[],outcome='failed_before_successful_emission')
    value=recorder.snapshot('failed',0,profile,dict(kind='before_error',prepared_ns=time.perf_counter_ns(),
        pre_write_ns=None,post_flush_ns=None))
    recorder.close()
    proof=p.validate_probe_sidecar(value,recorder.path,True,'failed',profile)
    assert not proof['measurement_accepted'] and proof['failed_record_count']==1
    assert proof['records'][0]['elapsed_ns'] is not None
    assert set(value['records'][0]['timestamps'])=={'start_ns','end_ns'}
    assert value['records'][0]['timestamps']['end_ns'] is not None
    assert value['records'][0]['error']==dict(type='OSError',detail='fixture fsync failure')
    assert value['persistence_disposition']=='after_failure_before_error_emission'
    assert value['timing_metadata_written_after_response_or_error_emission'] is False
    value['emission']['post_flush_ns']=time.perf_counter_ns()
    with pytest.raises(ValueError,match='invented flush'):
        p.validate_probe_sidecar(value,recorder.path,True,'failed',profile)


def test_timer_metadata_does_not_change_the_actual_saved_failed_scientific_event(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'outputs/splendorust_pool6_core_compare_20261027_02/workers/qualification_frozen_cpu2/entrypoint_forensics.jsonl'
    raw=path.read_bytes();before=hashlib.sha256(raw).hexdigest()
    rows=[json.loads(line) for line in raw.splitlines()]
    start=next(i for i,row in enumerate(rows) if row['event'].get('case_id')=='private_reserves' and
               row['event'].get('arm')=='search' and row['event']['event']=='started')
    actual=rows[start+1]['event'];assert actual['event']=='complete_result_before_worker_validation'
    original,stream=_forensic_recorder(tmp_path/'original.jsonl')
    probe=p.make_recorder(tmp_path/'timed.jsonl',True)
    original(actual);probe.record(copy.deepcopy(actual));stream.close();probe.close()
    assert (tmp_path/'original.jsonl').read_bytes()==(tmp_path/'timed.jsonl').read_bytes()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_finite_bound_covers_actual_largest_accepted_startup(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'outputs/splendorust_pool6_core_compare_20261027_01/workers/qualification_research_cpu0/entrypoint_forensics.jsonl'
    assert sum(1 for _ in path.open(encoding='utf-8'))==517<p.MAX_RECORDS
    assert path.stat().st_size<p.MAX_LOG_BYTES


@pytest.mark.parametrize('bad',[0,1,None,'false'])
def test_instrumentation_is_an_exact_boolean(tmp_path,bad):
    with pytest.raises(ValueError,match='Literal'):p.make_recorder(tmp_path/'bad.jsonl',bad)
    assert not (tmp_path/'bad.jsonl').exists()


def test_atomic_publication_exposes_only_complete_json_and_never_replaces(tmp_path,monkeypatch):
    original=p.os.rename;seen=[]
    def rename(source,target):
        assert not target.exists()
        assert json.loads(source.read_bytes())==dict(proof='complete')
        seen.append(True);return original(source,target)
    monkeypatch.setattr(p.os,'rename',rename)
    target=p.publish_probe(tmp_path/'probe','request-00000003',dict(proof='complete'))
    assert seen==[True] and json.loads(target.read_bytes())==dict(proof='complete')
    with pytest.raises(FileExistsError):p.publish_probe(tmp_path/'probe','request-00000003',dict(proof='replacement'))
    assert json.loads(target.read_bytes())==dict(proof='complete')


def test_atomic_publish_failure_preserves_complete_pending_bytes(tmp_path,monkeypatch):
    def fail(*_):raise OSError('fixture rename failure')
    monkeypatch.setattr(p.os,'rename',fail)
    with pytest.raises(OSError):p.publish_probe(tmp_path/'probe','request-00000003',dict(proof='pending'))
    assert not (tmp_path/'probe/request-00000003.json').exists()
    assert json.loads((tmp_path/'probe/request-00000003.pending').read_bytes())==dict(proof='pending')


@pytest.mark.parametrize('enabled',[False,True])
def test_same_immutable_delegate_and_only_two_on_clock_reads(tmp_path,monkeypatch,enabled):
    real_factory=p._forensic_recorder;calls=[];clocks=[]
    def factory(path):
        original,stream=real_factory(path)
        def record(value):
            calls.append(value)
            return original(value)
        return record,stream
    monkeypatch.setattr(p,'_forensic_recorder',factory)
    monkeypatch.setattr(p.time,'perf_counter_ns',lambda:clocks.append(len(clocks)+100) or clocks[-1])
    recorder=p.make_recorder(tmp_path/'same.jsonl',enabled)
    item=event();recorder.record(item);recorder.close()
    assert calls==[item] and calls[0] is item and len(clocks)==(2 if enabled else 0)
    entry=recorder.recorded_entries()[0]
    assert entry['timestamps']==dict(start_ns=100 if enabled else None,end_ns=101 if enabled else None)
    assert recorder.record_count==1


@pytest.mark.parametrize('enabled',[False,True])
def test_both_modes_defer_log_read_and_error_formatting(tmp_path,monkeypatch,enabled):
    recorder=p.make_recorder(tmp_path/'deferred.jsonl',enabled)
    original_read=recorder._read_records
    def forbidden(*_):raise AssertionError('No metadata/parsing/formatting in timed callback')
    monkeypatch.setattr(recorder,'_read_records',forbidden)
    monkeypatch.setattr(p,'_failure',forbidden)
    recorder.record(event())
    class DeferredError(Exception):
        def __str__(self):raise AssertionError('No exception formatting in timed callback')
    error=DeferredError()
    def fail(_):raise error
    recorder.original_record=fail
    with pytest.raises(DeferredError) as raised:recorder.record(event(1))
    assert raised.value is error and recorder.record_count==2
    recorder.close()
    monkeypatch.setattr(recorder,'_read_records',original_read)
    monkeypatch.setattr(p,'_failure',lambda e:dict(type=type(e).__name__,detail='deferred'))
    entries=recorder.recorded_entries()
    assert len(entries)==2 and entries[1]['error']['detail']=='deferred'
    assert all(set(e['timestamps'])=={'start_ns','end_ns'} for e in entries)


@pytest.mark.parametrize('enabled',[False,True])
def test_record_bound_is_preallocated_and_rejects_before_callback(tmp_path,monkeypatch,enabled):
    monkeypatch.setattr(p,'MAX_RECORDS',1)
    recorder=p.make_recorder(tmp_path/'bounded.jsonl',enabled)
    recorder.record(event());before=recorder.path.read_bytes()
    with pytest.raises(ValueError,match='bound/closure'):recorder.record(event(1))
    assert recorder.record_count==1 and recorder.path.read_bytes()==before
    recorder.close()
    with pytest.raises(ValueError,match='bound/closure'):recorder.record(event(2))


def test_missing_failed_total_is_unavailable_not_zero(tmp_path):
    recorder,value,profile=fixture(tmp_path)
    entry=value['records'][0]
    entry['completed']=False;entry['error']=dict(type='InterruptedError',detail='synthetic missing marker')
    entry['timestamps']['end_ns']=None;value['status']='log_failure'
    proof=p.validate_profile(value,expected_enabled=True,child_profile=profile)
    assert proof['total_ns'] is None and proof['records'][0]['elapsed_ns'] is None
    assert not proof['measurement_accepted']


def test_timed_callback_has_no_metadata_construction_or_new_logging_operations():
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(p.Recorder.record)))
    assert not any(isinstance(n,(ast.Dict,ast.List,ast.DictComp,ast.ListComp)) for n in ast.walk(tree))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    attrs=[n.func.attr for n in calls if isinstance(n.func,ast.Attribute)]
    assert attrs.count('perf_counter_ns')==2 and attrs.count('original_record')==1
    assert not set(attrs)&{'dumps','loads','write','flush','fsync','snapshot','recorded_entries'}

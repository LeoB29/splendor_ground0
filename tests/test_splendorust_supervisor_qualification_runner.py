"""No models: control-timeout continuation, independent lifecycle and fatal faults."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.splendorust_supervisor_qualification_20261026 import runner as r, common as c, preparation as p


class Envelope:
    def __init__(self):self.stage=None;self.events=[];self.resident={}
    def begin(self,stage):self.stage=stage;self.events.append(('begin',stage))
    def finish(self):self.events.append(('finish',self.stage))
    def guard(self):pass
    def admit(self):pass
    def remaining(self):return 10000.
    def timeout(self,maximum):return maximum
    def effective_deadline(self):return 99999.


def setup_collection(monkeypatch,tmp_path,faults=None,*,profile_error=None,audit_failure=None):
    faults=faults or {};events=[];attempts=[];workers={}
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    # These fixtures exercise scheduling and evidence binding, not the separately
    # qualified fsync primitive; avoid thousands of real disk flushes per test.
    def fixture_save(path,value):
        path=r.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf8') as stream:stream.write(c.canonical(value)+'\n')
    monkeypatch.setattr(c,'save',fixture_save)
    fixtures=[dict(fixture_id=str(i),snapshot={'i':i},observer={'i':i}) for i in range(24)]
    plan=dict(fixtures=fixtures,schedule=r.panel.make_schedule(fixtures))
    monkeypatch.setattr(r,'switch_placement',lambda env,freeze,mode:events.append(('switch',mode)))
    monkeypatch.setattr(r,'decode_training_snapshot',lambda value:value)
    class Worker:
        def __init__(self,name,quota,mode):
            self.name=name;self.output_dir=tmp_path/'workers'/name;self.quota=quota;self.placement_mode=mode;self.counter=0
            self.closed=False;workers[name]=self;events.append(('open',name))
            c.save(self.output_dir/'placement_admission.json',{'supervisor':{'monotonic_ns':r.time.perf_counter_ns()}})
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True;events.append(('close',self.name))
    monkeypatch.setattr(r,'make_worker',lambda name,*args,quota,placement_mode,**kwargs:Worker(name,quota,placement_mode))
    def decide(child,state,envelope,role):
        index=len(attempts);spec=plan['schedule'][index];attempts.append(index);child.counter+=1
        clockname=f'request-{child.counter:08d}.json';now=r.time.perf_counter_ns()
        status=faults.get(index,'ok')
        row=dict(completed=status=='ok',player_role=role,quota=child.quota,mode='candidate' if child.quota==64 else 'baseline',
            placement_mode=child.placement_mode,clock={'status':status,'request_id':clockname[:-5]},
            worker_clock_file=(child.output_dir/clockname).relative_to(tmp_path).as_posix(),
            parent_profile={'request_start_ns':now,'capture_snapshot_ns':now+1})
        if index==profile_error:row['profile_capture_error']='corrupt sidecar'
        c.save(child.output_dir/'inputs'/clockname,{'scheduled_index':index})
        events.append(('decision',index,child.name,status));return row
    monkeypatch.setattr(r,'decision',decide)
    monkeypatch.setattr(r,'verify_saved_decision',lambda *args:None)
    monkeypatch.setattr(r.audit,'classify_failure',lambda receipt:receipt['status'])
    def control(row):
        if row['placement_mode']!='original' or row['completed'] or row['clock']['status']!='timeout':
            raise ValueError('Not an authorized control timeout')
        return 'timeout'
    monkeypatch.setattr(r.assessment,'control_timeout',control)
    def verify(folder,freeze,envelope):
        child=workers[folder.name]
        assert child.closed
        events.append(('audit',child.name))
        if audit_failure==child.name:raise ValueError('raw evidence corruption')
        return dict(quota=child.quota,placement_mode=child.placement_mode)
    monkeypatch.setattr(r,'verify_closed_worker',verify)
    monkeypatch.setattr(r,'validate_timing_rows',lambda rows,*a,**kw:dict(status='passed',requests=len(rows)))
    monkeypatch.setattr(r.panel,'timing_gate',lambda rows:dict(qualified=len(rows)==192,
        status='passed' if len(rows)==192 else 'prefix',complete_inventory=len(rows)==192))
    return plan,events,attempts,workers


@pytest.mark.parametrize('faults', [{}, {0:'timeout'}, {0:'timeout',1:'timeout',23:'timeout',73:'timeout'},
    {i:'timeout' for i in list(range(24))+list(range(72,96))+list(range(96,120))+list(range(168,192))}])
def test_each_control_attempt_once_and_close_audit_precedes_next_worker(monkeypatch,tmp_path,faults):
    plan,events,attempts,workers=setup_collection(monkeypatch,tmp_path,faults)
    rows,gate,stop=r.collect_timing(Envelope(),{},plan)
    assert attempts==list(range(192)) and stop is None and gate['qualified']
    assert len({row['index'] for row in rows})==192
    checks=r.verify_timing_lifecycle(rows,plan,{},Envelope())
    extra=sum(plan['schedule'][i]['ordinal']<23 for i in faults)
    assert checks['workers']==8+extra and checks['continuation_workers']==extra and checks['control_timeouts']==len(faults)
    for index in faults:
        row=rows[index];name=row['timing_worker_name'];closed=events.index(('close',name));audited=events.index(('audit',name))
        assert closed<audited and not row['completed']
        if plan['schedule'][index]['ordinal']<23:
            next_name=rows[index+1]['timing_worker_name']
            assert next_name!=name and audited<events.index(('open',next_name))
    assert len(workers)<=100
    manifest=c.read(tmp_path/'control_timeout_manifest.json')
    assert manifest['count']==len(faults) and not manifest['failed_indices_retried']
    assert [item['index'] for item in manifest['failures']]==sorted(faults)


def test_original_timeout_on_block_last_row_does_not_create_unused_worker(monkeypatch,tmp_path):
    plan,events,attempts,workers=setup_collection(monkeypatch,tmp_path,{23:'timeout'})
    rows,gate,stop=r.collect_timing(Envelope(),{},plan)
    assert len(workers)==8 and rows[23]['timing_worker_name'].endswith('segment_00')
    assert not any('timing_block_0_segment_01' in str(e) for e in events)


def test_isolated_failure_stops_all_later_requests_preserving_earlier_control_timeout(monkeypatch,tmp_path):
    plan,events,attempts,workers=setup_collection(monkeypatch,tmp_path,{0:'timeout',25:'timeout'})
    rows,gate,stop=r.collect_timing(Envelope(),{},plan)
    assert attempts==list(range(26)) and stop['index']==25 and not gate['qualified']
    checks=r.verify_timing_lifecycle(rows,plan,{},Envelope())
    assert checks['control_timeouts']==1 and checks['workers']==3
    assert c.read(tmp_path/'timing_stop.json')==stop


@pytest.mark.parametrize('failure',['illegal','worker_exit','malformed'])
def test_nontimeout_original_failure_is_fatal_and_never_continues(monkeypatch,tmp_path,failure):
    plan,events,attempts,workers=setup_collection(monkeypatch,tmp_path,{0:failure})
    with pytest.raises(ValueError,match='authorized control timeout'):r.collect_timing(Envelope(),{},plan)
    assert attempts==[0] and len(workers)==1 and next(iter(workers.values())).closed
    assert (tmp_path/'timing_rows'/'0000.json').exists()
    assert not (tmp_path/'control_timeout_manifest.json').exists()


@pytest.mark.parametrize('kind',['profile','raw_audit'])
def test_evidence_corruption_prevents_continuation_after_timeout(monkeypatch,tmp_path,kind):
    opts={'profile_error':0} if kind=='profile' else {'audit_failure':'timing_block_0_segment_00'}
    plan,events,attempts,workers=setup_collection(monkeypatch,tmp_path,{0:'timeout'},**opts)
    with pytest.raises(ValueError):r.collect_timing(Envelope(),{},plan)
    assert attempts==[0] and len(workers)==1 and next(iter(workers.values())).closed
    assert (tmp_path/'timing_rows'/'0000.json').exists()


@pytest.mark.parametrize('tamper',['segment','extra_input','timeout_manifest','chronology','wrong_worker'])
def test_independent_lifecycle_rejects_omission_retry_and_restart_drift(monkeypatch,tmp_path,tamper):
    plan,events,attempts,workers=setup_collection(monkeypatch,tmp_path,{0:'timeout'})
    rows,gate,stop=r.collect_timing(Envelope(),{},plan)
    if tamper=='segment':rows[1]['timing_segment']=0
    elif tamper=='extra_input':c.save(tmp_path/'workers'/rows[0]['timing_worker_name']/'inputs'/'request-99999999.json',{})
    elif tamper=='timeout_manifest':
        path=tmp_path/'control_timeout_manifest.json';value=c.read(path);value['count']=0;path.write_text(c.canonical(value))
    elif tamper=='chronology':
        path=tmp_path/'timing_worker_segments'/'0_01.json';value=c.read(path);value['started_ns']=1;path.write_text(c.canonical(value))
    else:rows[1]['worker_clock_file']=rows[0]['worker_clock_file']
    with pytest.raises(ValueError):r.verify_timing_lifecycle(rows,plan,{},Envelope())


def test_operational_gate_uses_isolated_qualification_not_control_success(monkeypatch):
    env=Envelope();called=[];final=[]
    monkeypatch.setattr(r,'qualify',lambda *a:{})
    monkeypatch.setattr(r,'switch_placement',lambda *a:None)
    monkeypatch.setattr(r,'collect_timing',lambda *a:([{'completed':False,'placement_mode':'original'}],
        {'qualified':True,'relative_comparison':{'status':'not_estimable_with_control_timeouts'}},None))
    def game(spec,*args):called.append(spec['game_index']);return spec,[],0.,None
    monkeypatch.setattr(r,'play_game',game)
    monkeypatch.setattr(r,'final_checks',lambda *args:final.append(args) or {})
    r.run_stages(env,{}, {},p.operational_schedule())
    assert called==list(range(6)) and len(final[0][-2])==6


def test_isolated_nonpass_skips_games_but_reaches_final(monkeypatch):
    env=Envelope();final=[]
    monkeypatch.setattr(r,'qualify',lambda *a:{})
    monkeypatch.setattr(r,'collect_timing',lambda *a:([],{'qualified':False},None))
    monkeypatch.setattr(r,'play_game',lambda *a:pytest.fail('Isolated nonpass dispatched games'))
    monkeypatch.setattr(r,'final_checks',lambda *args:final.append(args) or {})
    r.run_stages(env,{}, {},p.operational_schedule())
    assert final and [x[1] for x in env.events if x[0]=='begin']==['qualification','timing','final']


def test_refresh_accounts_many_sidecars_written_after_trusted_timeout_closes_worker(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    env=r.Envelope({'monotonic_origin':r.time.monotonic(),'unix_origin':r.time.time()})
    env.enable_evidence_accounting()
    for index in range(96):
        folder=tmp_path/'workers'/f'failed_{index}'
        folder.mkdir(parents=True);(folder/'early.bin').write_bytes(b'0'*128)
        env.register_worker(folder);env.close_worker(folder)
        # Original request has already closed; the adapter then preserves both
        # the trusted parent snapshot and the final placement evidence.
        (folder/'late_profile.bin').write_bytes(b'1'*2048)
        (folder/'late_placement.bin').write_bytes(b'2'*2048)
        env.close_worker(folder)
    assert env.accounted_bytes()>=r.bytes_used()
    assert sum(env.evidence['workers'].values())==96*(128+4096)
    assert all(value is not None for value in env.evidence['workers'].values())


def test_closed_worker_verification_refreshes_postclose_cache_before_guard(monkeypatch,tmp_path):
    events=[]
    env=SimpleNamespace(close_worker=lambda folder:events.append('refresh'),guard=lambda:events.append('guard'),
        effective_deadline=lambda:9999.)
    def failure(*args,**kwargs):events.append('raw_checks');raise ValueError('test exit')
    monkeypatch.setattr(r,'verify_worker_logs',failure)
    with pytest.raises(ValueError,match='test exit'):r.verify_closed_worker(tmp_path,{},env)
    assert events==['refresh','guard','raw_checks']

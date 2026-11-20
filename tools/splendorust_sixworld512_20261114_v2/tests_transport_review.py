"""Independent model-free transport, custody and replay regression tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import gzip,time
import pytest
from . import common as c,adapters as a,qualification as q,transport,verifier
from tools.splendorust_teacher_variability_20261113 import ownership,qualification as prior


def clock(result=None,*,status='ok',request_id='request-00000001',limit=120.):
    accepted=status=='ok';timeout=status=='timeout'
    return dict(clock='perf_counter_ns',clock_implementation='QueryPerformanceCounter()',request_id=request_id,
        deadline_seconds=limit,elapsed_seconds=limit+.1 if timeout else .2,
        receipt_elapsed_seconds=None if timeout else .1,received_before_deadline=not timeout,
        accepted=accepted,status=status,result=result,detail=None,telemetry={})


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'attempt')
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json');monkeypatch.setattr(c,'PRIOR_OUT',tmp_path/'prior')
    c.save(c.PLAN,{'fixture':True});monkeypatch.setattr(c,'exact_alive',lambda item:False)
    monkeypatch.setattr(ownership,'validate_snapshot',lambda *args:None)
    monkeypatch.setattr(ownership,'validate_book',lambda *args:None)
    monkeypatch.setattr(prior,'verify_graph_evidence',lambda *args:None)
    return tmp_path


def life_fixture(*,failed=False,alter=None):
    folder=c.OUT/'live/services/life_0000';service=folder/'service';wire=folder/'transport'
    identity={'pid':1234,'created':123.0};role='main'
    binary=c.save(c.OUT/'binary.json',{'fixture_native':True})
    profile=c.save(c.OUT/'profile.json',dict(native_binary=binary['path'],native_binary_sha256=binary['sha256']))
    graph=c.save(c.OUT/'graph_plan.json',{'fixtures':[]})
    plan=dict(control_profiles={'legacy_p3_q16':profile},research_profiles={
        'legacy_p3_q16':profile,'legacy_p3_q512':profile},console_executable='fixture',
        graph_parity_plan=graph,corrected_source_receipt={'sources':'fixed'})
    metadata=dict(identity=identity,role=role,freeze_sha256=c.sha(c.PLAN),live_schema=a.LIVE_SCHEMA,result_encoding=a.PACK_SCHEMA)
    init=c.save(wire/'initialization.json',{'result':metadata})
    runtime=c.save(service/'variability_runtime_identity.json',{'native_pids':{'reference':1,'control':2,'search':3}})
    c.save(folder/'native_startup.json',{'rows':[]})
    c.save(service/'runtime_sources.json',{'sources':plan['corrected_source_receipt']})
    c.save(service/'graph_parity.json',{'passed':True,'fixtures':[]})
    scientific={'action':{'kind':'Pass'},'diagnostics':{'calls':0},'research':{'phases':[]}}
    entries=[(dict(operation='graph_parity',fixtures=[]),{'passed':True},300.,'ok'),
        (q.payload({'test':1},6),scientific,120.,'ok')]
    if failed:entries.append((q.payload({'test':2},3),None,120.,'timeout'))
    links=[];packets=[{'type':'ready','metadata':metadata}];raws=[];inputs=[]
    for i,(payload,result,limit,status) in enumerate(entries,1):
        rid=f'request-{i:08d}';packed=None if result is None else a.pack_result(result)
        raw=clock(packed,status=status,request_id=rid,limit=limit);decoded=deepcopy(raw)
        if result is not None:decoded['result']=result
        source=c.save(wire/f'{rid}.json',raw)
        inp=c.save_gzip(folder/'inputs'/f'{rid}.json.gz',dict(payload=payload,request_id=rid,
            identity=identity,freeze_sha256=c.sha(c.PLAN),time='fixed'))
        link=dict(source=source,input=inp,decoded_sha256=c.digest(decoded),
            result_sha256=None if result is None else c.digest(result),identity=identity,request_id=rid,timeout=limit)
        if alter:alter(i,link,raw,payload,result,folder)
        links.append(c.save(folder/'links'/f'{rid}.json',link));raws.append(source);inputs.append(inp)
        if result is not None:packets.append(dict(type='response',request_id=rid,result=packed))
    (wire/'stdout.log').write_text(''.join(c.canonical(p)+'\n' for p in packets),encoding='utf-8')
    event=dict(sequence=0,previous_sha256=None,event=dict(transport_request_id='request-00000002',
        event=dict(event='complete_result_before_worker_validation',arm='search',observer={'test':1},**scientific)))
    path=service/'ordinary_forensics.jsonl.gz';path.write_bytes(gzip.compress((c.canonical(event)+'\n').encode(),mtime=0))
    c.save(service/'forensic_summary.json',dict(events=1,terminal_sha256=c.digest(event),artifact=c.artifact(path),
        lossless=True,schema='sixworld512_compressed_forensic_chain_v1'))
    ending=dict(status='closed_after_failed_request' if failed else 'passed',error=None,role=role,
        exact_owned_absence=True,identity=identity,owned_identities=[],runtime=runtime,initialization=init,
        failed_clock=raws[-1] if failed else None,exit_code=1 if failed else 0,
        book={'service':identity,'startup':[]},requests=links)
    item=c.save(folder/'actual_exit.json',ending)
    return dict(item=item,plan=plan,folder=folder,raws=raws,scientific=scientific,metadata=metadata)


def test_complete_wire_and_compressed_forensic_prefix_verify(workspace):
    fixture=life_fixture();proof=transport.verify_services([fixture['item']],fixture['plan'])
    assert proof[0]['requests']==2 and proof[0]['forensics']['accepted_events']==1


def test_genuine_timeout_life_retains_prior_complete_evidence(workspace):
    fixture=life_fixture(failed=True);proof=transport.verify_services([fixture['item']],fixture['plan'])
    assert proof[0]['requests']==3 and proof[0]['forensics']['failed_request_preserved']


def test_packed_wire_mismatch_rejected(workspace):
    fixture=life_fixture();path=fixture['folder']/'transport/stdout.log'
    lines=path.read_text().splitlines();packet=c.read(c.resolve(fixture['raws'][1]))
    wrong={'type':'response','request_id':packet['request_id'],'result':a.pack_result({'changed':True})}
    lines[-1]=c.canonical(wrong);path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):transport.verify_services([fixture['item']],fixture['plan'])


def test_unaccounted_extra_response_rejected(workspace):
    fixture=life_fixture();path=fixture['folder']/'transport/stdout.log'
    with path.open('a') as f:f.write(c.canonical({'type':'response','request_id':'request-00000003','result':{}})+'\n')
    with pytest.raises(ValueError):transport.verify_services([fixture['item']],fixture['plan'])


def test_cross_life_source_substitution_rejected(workspace):
    def substitute(i,link,raw,payload,result,folder):
        if i==2:link['source']=c.save(c.OUT/'different_life/transport/request-00000002.json',raw)
    fixture=life_fixture(alter=substitute)
    with pytest.raises(ValueError):transport.verify_services([fixture['item']],fixture['plan'])


def test_changed_link_deadline_rejected(workspace):
    def change(i,link,*rest):
        if i==2:link['timeout']=121.
    fixture=life_fixture(alter=change)
    with pytest.raises(ValueError):transport.verify_services([fixture['item']],fixture['plan'])


def test_cross_life_input_substitution_rejected(workspace):
    def substitute(i,link,raw,payload,result,folder):
        if i==2:link['input']=c.save_gzip(c.OUT/'different_life/inputs/request-00000002.json.gz',
            c.read(c.resolve(link['input'])))
    fixture=life_fixture(alter=substitute)
    with pytest.raises(ValueError):transport.verify_services([fixture['item']],fixture['plan'])


def test_readiness_owner_mismatch_rejected(workspace):
    fixture=life_fixture();path=c.resolve(fixture['item']);ending=c.read(path)
    ending['identity']={'pid':9999,'created':123.0}
    # Rehash the containing receipt to exercise semantic owner binding.
    path.write_text(c.canonical(ending)+'\n',encoding='utf-8')
    with pytest.raises(ValueError):transport.verify_services([c.artifact(path)],fixture['plan'])


def test_partial_wire_only_tolerated_for_preserved_final_forfeit(workspace):
    fixture=life_fixture(failed=True);path=fixture['folder']/'transport/stdout.log'
    with path.open('ab') as f:f.write(b'{"type":"response","request_id":"request-00000003","result":')
    proof=transport.verify_services([fixture['item']],fixture['plan'])
    assert proof[0]['forensics']['accepted_events']==1


def test_clean_life_rejects_partial_wire(workspace):
    fixture=life_fixture();path=fixture['folder']/'transport/stdout.log'
    with path.open('ab') as f:f.write(b'{"type":"response"')
    with pytest.raises(ValueError):transport.verify_services([fixture['item']],fixture['plan'])


def manager_fixture(*,seconds=1000,status='ok'):
    manager=transport.Services.__new__(transport.Services);folder=c.OUT/'request_life'
    manager.owner={'limits':{'monotonic_deadline':time.monotonic()+seconds}}
    manager.boundary=lambda:None;manager.active={'folder':folder,'requests':[]};calls=[]
    class Child:
        _counter=1;identity={'pid':444,'created':22.}
        def request(self,payload,timeout):
            rid='request-00000002'
            assert c.read(folder/'inputs'/f'{rid}.json.gz')['payload']==payload
            calls.append((payload,timeout));self._counter+=1
            value=None if status!='ok' else a.pack_result({'saved':True})
            raw=clock(value,status=status,request_id=rid,limit=timeout)
            c.save(folder/'transport'/f'{rid}.json',raw)
            return raw
    manager.child=Child();manager.close=lambda **kwargs:calls.append(('closed',kwargs))
    return manager,calls


def test_input_is_durable_before_trusted_clock_dispatch(workspace):
    manager,calls=manager_fixture();result=manager.request({'test':'input'},120)
    assert calls==[({'test':'input'},120)] and result['clock']['result']=={'saved':True}
    assert a.unpack_result(c.read(c.resolve(result['source']))['result'])=={'saved':True}


def test_insufficient_remaining_allowance_never_dispatches_or_resets_clock(workspace):
    manager,calls=manager_fixture(seconds=100)
    with pytest.raises(ValueError):manager.request({'test':'input'},120)
    assert calls==[] and not (c.OUT/'request_life/inputs').exists()


def test_timeout_request_closes_once_without_retry(workspace):
    manager,calls=manager_fixture(status='timeout');result=manager.request({'test':'input'},120)
    assert len(calls)==2 and calls[1][0]=='closed' and result['clock']['result'] is None
    assert calls[1][1]['failed_clock']==result['source']


def test_owner_guard_refusal_prevents_dispatch_without_a_loss(workspace):
    manager,calls=manager_fixture()
    def refused():raise ValueError('actual owner budget exhausted')
    manager.boundary=refused
    with pytest.raises(ValueError,match='actual owner budget exhausted'):manager.request({'test':'input'},120)
    assert calls==[] and not (c.OUT/'request_life').exists()


def test_replay_binds_full_result_not_action_only(workspace,monkeypatch):
    fixture=life_fixture();source=fixture['raws'][1]
    monkeypatch.setattr(q,'validate_live_result',lambda *args:None)
    chosen=dict(replay_id='fixed',source=source,observer={'test':1},pool=6,result_sha256=c.digest(fixture['scientific']))
    changed=deepcopy(fixture['scientific']);changed['research']['phases']=[{'callback_bits':'changed'}]
    replay=c.save(c.OUT/'replay/request-00000002.json',clock(a.pack_result(changed),request_id='request-00000002'))
    with pytest.raises(ValueError):verifier.verify_replays([chosen],[{'replay_id':'fixed','source':replay}],fixture['plan'])
    with pytest.raises(ValueError):verifier.verify_replays([chosen],[{'replay_id':'fixed','source':source}],fixture['plan'])


@pytest.fixture
def affinity_launch(workspace,monkeypatch):
    """No child process; fake constructor enforces the actual clock precondition."""
    from tools.local_player_qualification_128 import clock as trusted
    existing=life_fixture();state=dict(mask=[0,1,2,3],masks=[],constructors=[],closed=[],fail_init=False,fail_restore=False)
    graph_identity={k:'fixed' for k in ('checkpoint_sha256','weights_before_sha256','weights_after_sha256',
        'torch_version','numpy_version','backend','options')}
    historical=c.PRIOR_OUT/'jobs/qualification/worker/service/variability_runtime_identity.json'
    historical.parent.mkdir(parents=True);historical.write_text(c.canonical({'graph_identity':graph_identity}))
    class Controller:
        def cpu_affinity(self,mask=None):
            if mask is None:return list(state['mask'])
            if mask==[2] and state['fail_restore']:
                state['fail_restore']=False
                raise RuntimeError('fixture restoration failure')
            assert set(mask)<=set((0,1,2,3))
            state['mask']=list(mask);state['masks'].append(list(mask))
    monkeypatch.setattr(transport.psutil,'Process',lambda *args:Controller())
    monkeypatch.setattr(transport,'CONTROLLER_LAUNCH_AFFINITY',(0,1,2,3))
    monkeypatch.setattr(ownership,'inspect_service_tree',lambda *args:[])
    def factory(module,out,**kwargs):
        requested=kwargs['affinity']
        # Same admission test as the unchanged trusted PersistentWorker.
        if len(requested)!=1 or requested[0] not in state['mask']:
            raise ValueError('Worker must use one allowed logical processor')
        state['constructors'].append(dict(parent_mask=list(state['mask']),service_mask=list(requested),
            module=module,options=kwargs))
        if state['fail_init']:raise ValueError('fixture initialization failure')
        folder=Path(out).parent;ident={'pid':5000+len(state['constructors']),'created':123.0}
        metadata=dict(identity=ident,role=kwargs['env']['SIXWORLD_ROLE'],freeze_sha256=c.sha(c.PLAN),
            live_schema=a.LIVE_SCHEMA,result_encoding=a.PACK_SCHEMA)
        initial={'result':metadata};c.save(Path(out)/'initialization.json',initial)
        c.save(folder/'service/variability_runtime_identity.json',dict(native_pids={'reference':1,'search':2,'control':3},
            graph_identity=graph_identity))
        return SimpleNamespace(identity=ident,initialization=initial,affinity=requested,
            close=lambda:state['closed'].append(ident))
    monkeypatch.setattr(trusted,'PersistentWorker',factory)
    def manager(name,role='main'):
        value=transport.Services(c.OUT/name,existing['plan'],{'limits':{'monotonic_deadline':time.monotonic()+1000}},
            lambda:None,role,3)
        def graph_call(request,timeout,**kwargs):
            assert state['mask']==[2] and value.child.affinity==[0]
            assert request=={'operation':'graph_parity','fixtures':[]} and timeout==300
            return {'clock':{'status':'ok','result':{'passed':True}}}
        value.request=graph_call
        return value
    return state,manager,factory


def test_saved_affinity_failure_is_reproduced_and_fixed_for_repeated_lives_and_replay(affinity_launch):
    state,make_manager,factory=affinity_launch
    first=make_manager('launch_main');assert state['mask']==[2]
    with pytest.raises(ValueError,match='one allowed logical processor'):
        factory('fixture',c.OUT/'unused',affinity=[0])
    assert not state['constructors']
    for _ in range(2):
        first.start();assert state['mask']==[2] and first.child.affinity==[0]
        first.child.close();first.child=None;first.active=None
    second=make_manager('launch_replay','replay')
    assert state['mask']==[2] and second.launch_affinity==[0,1,2,3]
    second.start();assert state['mask']==[2] and second.child.affinity==[0]
    assert len(state['constructors'])==3
    for call in state['constructors']:
        assert call['parent_mask']==[0,1,2,3] and call['service_mask']==[0]
        assert call['options']['initialization_timeout']==240
        assert call['options']['max_message_bytes']==16*c.MiB


def test_controller_affinity_is_restored_after_constructor_failure(affinity_launch):
    state,make_manager,_=affinity_launch;manager=make_manager('launch_failure')
    state['fail_init']=True
    with pytest.raises(ValueError,match='fixture initialization failure'):manager.start()
    assert state['mask']==[2] and manager.child is None and manager.lives==[]
    assert state['constructors'][0]['parent_mask']==[0,1,2,3]


def test_successfully_constructed_child_is_closed_if_controller_restoration_fails(affinity_launch):
    state,make_manager,_=affinity_launch;manager=make_manager('restore_failure')
    state['fail_restore']=True
    with pytest.raises(RuntimeError,match='restoration failure'):manager.start()
    assert len(state['closed'])==1 and manager.child is None and manager.lives==[]


def test_original_allowed_universe_cannot_be_invented(affinity_launch,monkeypatch):
    state,make_manager,_=affinity_launch;monkeypatch.setattr(transport,'CONTROLLER_LAUNCH_AFFINITY',(2,))
    with pytest.raises(ValueError,match='Original controller mask'):make_manager('invalid_original_mask')
    assert not state['constructors']

"""Model-free resource, no-retry and conditional operational-run contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_inference_opt_20261025 import runner


class Envelope:
    def __init__(self): self.stage=None; self.events=[]
    def begin(self,stage): self.stage=stage; self.events.append(('begin',stage))
    def finish(self): self.events.append(('finish',self.stage))
    def guard(self): self.events.append(('guard',self.stage))
    def admit(self): self.events.append(('admit',self.stage))
    def remaining(self): return 100.
    def timeout(self,allowance): return allowance


def test_exact_six_new_swapped_games_have_no_main_branch():
    rows=runner.operational_schedule()
    assert len(rows)==6 and len({row['seed_decimal'] for row in rows})==3
    assert {row['stage'] for row in rows}=={'operational'}
    assert set(runner.c.STAGES)=={'qualification','operational','final'}
    assert runner.c.JOB_SECONDS==3600 and runner.c.COLLECTION_CUTOFF_SECONDS==2700
    for first,second in zip(rows[::2],rows[1::2],strict=True):
        assert first['seed_decimal']==second['seed_decimal']
        assert first['challenger_seat']==1-second['challenger_seat']
        assert first['challenger_quota']==64 and first['opponent_quota']==32
        assert first['challenger_mode']==first['opponent_mode']=='candidate'
    from tools.splendorust_timing_opt_20261024.runner import operational_schedule
    assert not {row['seed_decimal'] for row in rows}&{row['seed_decimal'] for row in operational_schedule()}


@pytest.mark.parametrize('panel,ops,stages,admitted',[
    (False,None,['qualification','final'],False),
    (True,False,['qualification','operational','final'],False),
    (True,True,['qualification','operational','final'],True),
])
def test_negative_timing_gate_still_verifies_evidence_without_launching_strength(monkeypatch,panel,ops,stages,admitted):
    envelope=Envelope(); events=[]; saved=[]
    monkeypatch.setattr(runner,'qualify_panel',lambda *args:{'qualified':panel})
    def operational(*args):
        assert panel
        events.append('six_ops')
        return {'qualified':ops}
    monkeypatch.setattr(runner,'operational',operational)
    monkeypatch.setattr(runner,'final_checks',lambda *args:events.append(('final',deepcopy(args[-1]))))
    monkeypatch.setattr(runner.c,'save',lambda path,value:saved.append(value))
    result=runner.run_stages(envelope,{}, {},[])
    assert [stage for event,stage in envelope.events if event=='begin']==stages
    assert events[-1]==('final',result)
    assert ('six_ops' in events) is panel
    assert saved[-1]['inference_graph_qualified'] is admitted
    assert saved[-1]['strength_games_launched']==0 and saved[-1]['automatic_strength_launch'] is False


def test_startup_qualification_clock_is_not_reset(monkeypatch):
    envelope=Envelope(); envelope.begin('qualification')
    monkeypatch.setattr(runner,'qualify_panel',lambda *args:{'qualified':False})
    monkeypatch.setattr(runner,'final_checks',lambda *args:None)
    monkeypatch.setattr(runner.c,'save',lambda *args:None)
    runner.run_stages(envelope,{}, {},[],qualification_started=True)
    assert envelope.events.count(('begin','qualification'))==1


def test_actual_deadline_failure_stops_without_retry_or_more_model_work(monkeypatch):
    calls=[]
    def failed(*args): calls.append('once'); raise TimeoutError('preserved hard failure')
    monkeypatch.setattr(runner,'qualify_panel',failed)
    monkeypatch.setattr(runner,'operational',lambda *args:pytest.fail('No operational work after hard failure'))
    monkeypatch.setattr(runner,'final_checks',lambda *args:pytest.fail('No fresh reproduction after hard failure'))
    with pytest.raises(TimeoutError,match='preserved'): runner.run_stages(Envelope(),{}, {},[])
    assert calls==['once']


@pytest.mark.parametrize('result',[{}, {'qualified':1},{'passed':'true'}])
def test_only_explicit_boolean_gate_admits_operations(result):
    with pytest.raises(ValueError): runner.gate_passed(result)


def test_failed_receipt_is_saved_before_any_downstream_guard(monkeypatch):
    events=[]; row={'completed':False,'clock':{'status':'timeout','detail':'retained'}}
    monkeypatch.setattr(runner.c,'save',lambda path,value:events.append(('saved',value)))
    envelope=SimpleNamespace(guard=lambda:events.append(('guard',None)))
    with pytest.raises(ValueError,match='retained without retries'):
        runner.save_and_check(row,Path('unused'),envelope)
    assert events==[('saved',row),('guard',None)]


def test_failed_actual_entrypoint_is_preserved_without_retry(monkeypatch):
    saved=[]; calls=[]; receipt={'status':'worker_error','detail':'counterexample'}
    child=SimpleNamespace(request=lambda *args:calls.append(args) or receipt)
    monkeypatch.setattr(runner.c,'save',lambda path,value:saved.append(value))
    with pytest.raises(ValueError,match='entrypoint failed without retry'):
        runner.actual_entrypoint(child,Envelope(),'64_candidate')
    assert calls==[({'operation':'entrypoint_checks'},300)] and saved==[receipt]


@pytest.mark.parametrize('result',[
    {'status':'entrypoint_counterexamples_passed','challenger_quota':32,'counts':{'complete_decisions':52}},
    {'status':'entrypoint_counterexamples_passed','challenger_quota':64,'counts':{'complete_decisions':51}},
])
def test_actual_entrypoint_inventory_bound_to_exact_quota(monkeypatch,result):
    monkeypatch.setattr(runner.c,'save',lambda *args:None)
    child=SimpleNamespace(challenger_quota=64,request=lambda *args:{'status':'ok','result':result})
    with pytest.raises(ValueError,match='quota/decision inventory'):
        runner.actual_entrypoint(child,Envelope(),'candidate')


@pytest.mark.parametrize('first_remaining,computed',[(.999,1.),(1.,.999)])
def test_partial_move_allowance_cannot_be_dispatched(monkeypatch,tmp_path,first_remaining,computed):
    state=runner.frozen_q.reference._initial()
    child=SimpleNamespace(_counter=0,output_dir=tmp_path,challenger_quota=64,
                          request=lambda *args,**kwargs:pytest.fail('Partial clock dispatched'))
    envelope=Envelope()
    monkeypatch.setattr(envelope,'remaining',lambda:first_remaining)
    monkeypatch.setattr(envelope,'timeout',lambda allowance:computed)
    monkeypatch.setattr(runner.c,'save',lambda *args:None)
    with pytest.raises(ValueError,match='one-second|shortened'):
        runner.decision(child,state,envelope)


def test_all_four_entrypoints_precede_panel_with_only_two_resident_workers(monkeypatch):
    opened=[]; live=set(); peak=[]; entrypoints=[]; saved=[]
    class Child:
        def __init__(self,name,mode,quota): self.name=name; self.runtime_mode=mode; self.challenger_quota=quota
        def __enter__(self): live.add(self.name); opened.append(self.name); peak.append(len(live)); return self
        def __exit__(self,*args): live.remove(self.name)
    def worker(name,envelope,freeze,*,mode,quota): return Child(name,mode,quota)
    monkeypatch.setattr(runner,'make_worker',worker)
    monkeypatch.setattr(runner,'actual_entrypoint',lambda child,envelope,label:entrypoints.append(label))
    monkeypatch.setattr(runner.plan_rules,'snapshot_for_request',lambda plan,spec:spec['index'])
    monkeypatch.setattr(runner,'decode_training_snapshot',lambda value:value)
    def decision(child,state,envelope):
        assert entrypoints==['64_baseline','64_candidate','32_baseline','32_candidate']
        return dict(quota=child.challenger_quota,mode=child.runtime_mode,marker=state)
    monkeypatch.setattr(runner,'decision',decision)
    monkeypatch.setattr(runner,'save_and_check',lambda row,path,envelope:saved.append(row))
    monkeypatch.setattr(runner.c,'save',lambda *args:None)
    def validate(rows,plan,**kwargs):
        assert len(rows)==288 and all(row['marker']==row['index'] for row in rows)
        return {'qualified':False}
    monkeypatch.setattr(runner.plan_rules,'validate_panel',validate)
    schedule=[dict(index=i,quota=64 if i<144 else 32,mode=('baseline','candidate')[i%2]) for i in range(288)]
    assert runner.qualify_panel(Envelope(),{}, {'timed_requests':schedule})=={'qualified':False}
    assert opened==['entrypoint_64_baseline','entrypoint_64_candidate','entrypoint_32_baseline','entrypoint_32_candidate',
        'qualification_64_baseline','qualification_64_candidate','qualification_32_baseline','qualification_32_candidate']
    assert entrypoints==['64_baseline','64_candidate','32_baseline','32_candidate']
    assert max(peak)==2 and not live and len(saved)==288


def operational_rows(arrivals):
    return [dict(stage='operational',pair_index=0,game_index=0,ply=i,quota=64 if i%2==0 else 32,
        mode='candidate',completed=True,clock=dict(status='ok',deadline_seconds=1.,receipt_elapsed_seconds=value))
        for i,value in enumerate(arrivals)]


def test_operational_summary_retains_soft_misses_and_never_uses_game_winners():
    rows=operational_rows([.2,.7,.70001])
    result=runner.summarize_operational(rows,6)
    assert result['qualified'] is False and result['complete_decisions']==3
    assert result['margin_failures']==[dict(stage='operational',pair_index=0,game_index=0,ply=2,quota=64,arrival_seconds=.70001)]
    assert result['maximum_arrival_seconds']==.70001 and result['outcomes_used_for_admission'] is False
    assert runner.summarize_operational(operational_rows([.7]),6)['qualified'] is True


@pytest.mark.parametrize('field,value',[('mode','baseline'),('completed',False),('quota',16)])
def test_operational_summary_rejects_wrong_player_or_failed_move(field,value):
    rows=operational_rows([.1]); rows[0][field]=value
    with pytest.raises(ValueError,match='six complete games'): runner.summarize_operational(rows,6)


def test_nested_artifacts_bind_only_explicit_input_triples():
    artifact={'path':'a.json','sha256':'x','bytes':4}
    document={'wrapper':[{'source':artifact}, {'path':'not-an-artifact.json'}]}
    assert list(runner.nested_artifacts(document))==[artifact]


def test_previous_reader_and_resource_envelope_are_exact_qualified_dependencies():
    import ast
    from tools.splendorust_timing_opt_20261024 import common, runner as prior
    assert runner.c.BaselineStudyWorker is common.CandidateStudyWorker
    assert issubclass(runner.c.CandidateStudyWorker,common.CandidateStudyWorker)
    def envelope(path):
        return next(node for node in ast.parse(Path(path).read_text()).body
                    if isinstance(node,ast.ClassDef) and node.name=='Envelope')
    assert ast.dump(envelope(runner.__file__))==ast.dump(envelope(prior.__file__))


def mock_game(monkeypatch):
    states=[{'state':{'current_player':i%2,'terminal':i==2,'winner':1 if i==2 else None}} for i in range(3)]
    spec=runner.operational_schedule()[0]
    rows=[]
    for i in range(2):
        action,trace={'move':i},{'immutable':i}
        rows.append(dict(ply=i,seat=i,arm='search',game_role='challenger' if i==0 else 'opponent',
            quota=64 if i==0 else 32,mode='candidate',stage='operational',pair_index=0,game_index=0,
            before=states[i],after=states[i+1],observer={'seat':i},action=action,trace=trace,
            completed=True,clock={'status':'ok','result':{'action':action,'diagnostics':trace}}))
    ending=dict(**spec,plies=2,reason='natural',winner=1,final=states[-1],complete=True)
    validations=[]
    monkeypatch.setattr(runner,'decode_training_snapshot',lambda value:value)
    monkeypatch.setattr(runner.ledger,'check_observer',lambda *args:None)
    monkeypatch.setattr(runner.ledger,'transition',lambda before,action:states[action['move']+1])
    monkeypatch.setattr(runner.audit,'check_clock',lambda *args:None)
    monkeypatch.setattr(runner.audit,'validate_result',lambda result,observer,arm,expected_quota:validations.append((arm,expected_quota)))
    return states,spec,rows,ending,validations


def test_independent_game_replay_uses_both_exact_search_quotas(monkeypatch):
    states,spec,rows,ending,validated=mock_game(monkeypatch)
    assert runner.replay_game(states[0],rows,ending,spec)['transitions']==2
    assert validated==[('search',64),('search',32)]


@pytest.mark.parametrize('field,value',[('quota',16),('mode','baseline'),('arm','reference')])
def test_game_replay_rejects_wrong_player_identity(monkeypatch,field,value):
    states,spec,rows,ending,_=mock_game(monkeypatch); rows[1][field]=value
    with pytest.raises(ValueError,match='player identity changed'): runner.replay_game(states[0],rows,ending,spec)


def test_game_replay_rejects_disconnected_action_and_receipt(monkeypatch):
    states,spec,rows,ending,_=mock_game(monkeypatch); rows[1]['action']={'move':0}
    with pytest.raises(ValueError,match='result coupling changed'): runner.replay_game(states[0],rows,ending,spec)


def parity_fixture():
    fixtures=[dict(fixture_id=f'fixed_{i}',batch=1,tokens=[float(i)]) for i in range(11)]
    output=[[0.0]*83]
    sha,bits=runner.exact_output_hashes(output,1)
    rows=[dict(fixture_id=item['fixture_id'],batch=1,input_sha256=runner.c.digest(item['tokens']),
        exact_float_equal=True,float32_bits_equal=True,eager_output_sha256=sha,candidate_output_sha256=sha,
        eager_float32_sha256=bits,candidate_float32_sha256=bits) for item in fixtures]
    return fixtures,dict(schema='splendorust_inference_graph_parity_v1',status='passed',passed=True,
        fixtures_count=11,fixture_payload_sha256=runner.c.digest(fixtures),rows=rows)


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(passed=1),lambda r:r.update(status='failed'),lambda r:r.update(fixtures_count=10),
    lambda r:r.update(fixture_payload_sha256='wrong'),lambda r:r['rows'][0].update(input_sha256='wrong'),
    lambda r:r['rows'][0].update(exact_float_equal=False),lambda r:r['rows'][0].update(float32_bits_equal=False),
    lambda r:r['rows'][0].update(candidate_float32_sha256='wrong'),
    lambda r:r['rows'][0].update(candidate_output_sha256='wrong'),
    lambda r:r['rows'][0].update(fixture_id='different'),lambda r:r['rows'][0].update(batch=8),
])
def test_exact_parity_rejects_every_relaxed_or_mismatched_gate(mutate):
    fixtures,result=parity_fixture()
    runner.validate_graph_parity(result,fixtures)
    mutate(result)
    with pytest.raises(ValueError): runner.validate_graph_parity(result,fixtures)


def test_independent_float32_hash_detects_signed_zero_and_rounding():
    positive=[[0.0]*83]; negative=deepcopy(positive); negative[0][0]=-0.0
    assert positive==negative
    assert runner.exact_output_hashes(positive,1)!=runner.exact_output_hashes(negative,1)
    rounded=deepcopy(positive); rounded[0][0]=0.1
    with pytest.raises(ValueError,match='exact float32'): runner.exact_output_hashes(rounded,1)
    for bad in ([],[[float('nan')]*83],[[0]*83],[[0.0]*82]):
        with pytest.raises(ValueError): runner.exact_output_hashes(bad,1)


def test_graph_parity_failure_is_saved_and_cannot_reach_decisions(monkeypatch,tmp_path):
    fixtures,result=parity_fixture();result['passed']=False;result['status']='failed'
    calls=[];saved=[]
    child=SimpleNamespace(output_dir=tmp_path,
        request=lambda payload,timeout:calls.append((payload,timeout)) or dict(status='ok',result=result))
    monkeypatch.setattr(runner.c,'verify',lambda items:None)
    monkeypatch.setattr(runner.c,'read',lambda path:{'fixtures':fixtures})
    monkeypatch.setattr(runner.c,'save',lambda path,value:saved.append((path,value)))
    with pytest.raises(ValueError,match='parity gate failed'):
        runner.graph_parity(child,Envelope(),{'graph_parity_plan':{'path':'plan'}})
    assert calls==[({'operation':'graph_parity','fixtures':fixtures},60)]
    assert len(saved)==1 and saved[0][0].name=='graph_parity_clock.json'


@pytest.mark.parametrize('mode',['baseline','candidate'])
def test_worker_uses_same_raw_parent_and_parity_before_return(monkeypatch,tmp_path,mode):
    from tools.splendorust_search_scale_20261024 import adapters
    events=[];seen=[]
    metadata={'scientific':'unchanged'}
    if mode=='candidate': metadata['inference_backend']=runner.BACKEND
    class Worker:
        def __init__(self,module,folder,**kwargs):
            seen.append((module,kwargs));self.output_dir=folder;self.process=SimpleNamespace(pid=123)
            self.initialization={'result':deepcopy(metadata)};self._failure=None
        def _sample(self): return {'process_tree':[]}
        def close(self): events.append('close')
    envelope=Envelope()
    envelope.register_worker=lambda folder:None
    envelope.guard_worker_sample=lambda *args:None
    envelope.effective_deadline=lambda:999
    envelope.own_worker=lambda child:events.append('own')
    monkeypatch.setattr(runner.c,'OUT',tmp_path)
    monkeypatch.setattr(runner.c,'BaselineStudyWorker',Worker)
    monkeypatch.setattr(runner.c,'CandidateStudyWorker',Worker)
    monkeypatch.setattr(adapters,'validate_ready_metadata',lambda value,**kwargs:events.append(('ready',value)))
    monkeypatch.setattr(runner,'graph_parity',lambda child,*args:events.append('parity'))
    child=runner.make_worker('fixture',envelope,{'native_artifacts':[],
        'topology':{'selected_logical_processor':2}},mode=mode,quota=64)
    expected='tools.splendorust_search_scale_20261024.worker' if mode=='baseline' else 'tools.splendorust_inference_opt_20261025.worker'
    assert seen[0][0]==expected and seen[0][1]['initialization_timeout']==60
    assert events==(['own',('ready',{'scientific':'unchanged'})]+(['parity'] if mode=='candidate' else []))
    assert child.initialization['result']==metadata and child.runtime_mode==mode


def graph_evidence_files(monkeypatch,tmp_path):
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    fixtures,result=parity_fixture()
    output=[[0.0]*83]
    monkeypatch.setattr(runner.c,'ROOT',tmp_path)
    folder=tmp_path/'candidate'; folder.mkdir()
    def write(path,value): path.write_text(runner.c.canonical(value),encoding='utf-8')
    plan={'fixtures':fixtures,'provenance_by_fixture':{item['fixture_id']:{} for item in fixtures}}
    write(tmp_path/'parity.json',plan)
    graph=b'fixed graph\n'; (folder/'graph_ir.txt').write_bytes(graph)
    identity=dict(schema='splendorust_inference_graph_identity_v1',backend=runner.BACKEND,
        checkpoint_sha256=CHECKPOINT_SHA256,weights_before_sha256='same',weights_after_sha256='same',
        parameter_count=4780883,dtype='torch.float32',device='cpu',threads=1,interop_threads=1,
        supported_batches=[1,2,8],input_shape=['batch',31,48],output_shape=['batch',83],
        serialized_graph_saved=False,warmup_is_parity_proof=False,numpy_version='1',torch_version='2+cpu',
        graph_ir_artifact='graph_ir.txt',graph_ir_bytes=len(graph),graph_ir_sha256=runner.hashlib.sha256(graph).hexdigest(),
        options=dict(trace_batch=1,trace_check_trace=False,trace_strict=True,freeze_optimize_numerics=False,
            warm_batches=[1,2,8],trace_and_warm_inputs='architecture_zero_float32_only',
            automatic_fallback=False,alternative_candidates=0))
    result['graph_identity_sha256']=runner.c.digest(identity)
    write(folder/'graph_identity.json',identity)
    metadata={'inference_backend':runner.BACKEND}
    write(folder/'inference_execution_identity.json',dict(backend=runner.BACKEND,
        graph_identity_sha256=runner.c.digest(identity),metadata=metadata))
    write(folder/'initialization.json',{'result':metadata})
    receipt=dict(status='ok',receipt_elapsed_seconds=.1,deadline_seconds=60.,request_id='request-00000001',result=result)
    write(folder/'graph_parity_clock.json',receipt);write(folder/'request-00000001.json',receipt)
    evidence=dict(schema='splendorust_inference_graph_parity_v1',summary=result,fixtures=fixtures,
        evidence=[dict(fixture_id=item['fixture_id'],batch=1,eager_outputs=deepcopy(output),
            candidate_outputs=deepcopy(output),comparison=row) for item,row in zip(fixtures,result['rows'])])
    write(folder/'graph_parity.json',evidence)
    freeze=dict(graph_parity_plan={'path':'parity.json'},numpy_version='1',torch_runtime_version='2+cpu')
    return folder,freeze,write


@pytest.mark.parametrize('tamper',['output','bits','ir','receipt','historical'])
def test_full_graph_evidence_rejects_tampering_even_when_summary_claims_pass(monkeypatch,tmp_path,tamper):
    folder,freeze,write=graph_evidence_files(monkeypatch,tmp_path)
    assert runner.verify_graph_evidence(folder,freeze)['fixtures']==11
    if tamper in ('output','bits'):
        evidence=runner.c.read(folder/'graph_parity.json')
        evidence['evidence'][0]['candidate_outputs'][0][0]=1.0 if tamper=='output' else -0.0
        write(folder/'graph_parity.json',evidence)
    elif tamper=='ir': (folder/'graph_ir.txt').write_text('different',encoding='utf-8')
    elif tamper=='receipt':
        receipt=runner.c.read(folder/'request-00000001.json');receipt['receipt_elapsed_seconds']=.2
        write(folder/'request-00000001.json',receipt)
    else:
        plan=runner.c.read(tmp_path/'parity.json')
        plan['provenance_by_fixture']['fixed_0']={'historical_outputs':[[1.0]*83],
            'historical_output_sha256':'different'}
        write(tmp_path/'parity.json',plan)
    with pytest.raises(ValueError): runner.verify_graph_evidence(folder,freeze)

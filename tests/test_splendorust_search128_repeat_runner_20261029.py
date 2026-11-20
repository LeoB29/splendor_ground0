"""Model-free counterbalance, preserved-prefix and retained-tree regressions."""
import copy
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.splendorust_search128_repeat_20261029 import common as c,design,runner,preparation,transport,guardian


@pytest.fixture
def output(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'new')
    monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    c.OUT.mkdir()
    return c.OUT


def synthetic_panel():
    return {'roots':[dict(root_id=f'{group}_{i:02}',group=group,observer={'id':i},before={})
        for i in range(1,11) for group in ('slow','systematic')]}


def test_actual_saved20_panel_and_fixed120_schedule():
    panel=design.build_panel()
    matrix=design.build_matrix(panel)
    assert len(panel['roots'])==20 and len(matrix)==120
    assert panel['roots'][0]['original_receipt_seconds']==2.0121378
    assert all(r['original_result128']['simulations_quota']==128 for r in panel['roots'])
    assert [r['group'] for r in panel['roots']]==['slow','systematic']*10
    assert len({(s['root_id'],s['route'],s['repeat']) for s in matrix})==120
    assert sum(s['route']=='research' for s in matrix[::2])==30
    assert all(matrix[i]['root_id']==matrix[i+1]['root_id'] and
               {matrix[i]['route'],matrix[i+1]['route']}=={'research','frozen'} for i in range(0,120,2))


def test_runtime_contract_exact_and_no_latency_gate():
    contract=preparation.contract()
    assert contract['job_seconds']==3600 and contract['collection_cutoff_seconds']==2400
    assert contract['parent_requests']==124 and contract['qualification_decisions']==119
    assert contract['ordinary_requests']==120 and contract['graph_cases']==22
    assert contract['seconds_per_move']==10 and contract['internal_seconds']==9
    assert contract['max_evidence_bytes']==1024**3 and contract['max_workers']==2
    assert contract['initial_available_ram_bytes']==4*1024**3 and contract['minimum_available_ram_bytes']==2*1024**3
    assert contract['maximum_ram_wait_seconds']==300
    assert contract['seven_hundred_ms_gate'] is contract['one_second_gate'] is False
    good=dict(contract,schema='splendorust_search128_repeat_v1',same_workers_retained_for_repeats=True,numerical_player_change=False)
    runner.validate_freeze_contract(good)
    for field,value in [('seconds_per_move',1.),('ordinary_requests',121),('max_workers',3)]:
        with pytest.raises(ValueError):runner.validate_freeze_contract(dict(good,**{field:value}))


def test_shared_stages_do_not_reset_global_or_collection_clock(output,monkeypatch):
    clock=[100.]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runner.time,'time',lambda:1000+clock[0])
    envelope=runner.Envelope(dict(monotonic_origin=100.,unix_origin=1100.))
    monkeypatch.setattr(envelope,'guard',lambda:None)
    envelope.begin('qualification');assert envelope.deadline==1000
    clock[0]=1100
    envelope.begin('diagnostic');assert envelope.deadline==2500
    clock[0]=2490
    envelope.begin('final');assert envelope.deadline==3690 and envelope.end==3700
    clock[0]=3701
    assert envelope.remaining()<0


def test_stage_rebinding_requires_idle_qualified_trees(output):
    env=SimpleNamespace(stage='diagnostic',resident={1:None,2:None},start=0,end=3600,deadline=2400,
                        effective_deadline=lambda:2000.)
    def child(route):
        return SimpleNamespace(_lock=threading.Lock(),_counter=2,_closed=False,absolute_deadline=900.,
            identity={'pid':1 if route=='research' else 2,'created':1.},output_dir=c.OUT/'workers'/('qualification_repeat_'+route))
    workers={r:child(r) for r in ('research','frozen')}
    result=runner.bind_diagnostic_deadline(env,workers)
    assert result['ordinary_deadline_seconds']==10 and result['internal_seconds']==9
    assert result['request_clocks_reset'] is False
    assert all(x.absolute_deadline==2000 for x in workers.values())
    workers['frozen']._counter=3
    with pytest.raises(ValueError,match='idle'):runner.bind_diagnostic_deadline(env,workers)


def _collect_fixture(monkeypatch,rows_for_requests):
    from tools.splendorust_search128_repeat_20261029 import analysis
    calls=[]
    def ordinary(child,spec,observer,env,freeze):
        calls.append(spec['sequence'])
        item=copy.deepcopy(rows_for_requests[len(calls)-1]);item.update(spec=spec,observer=observer,route=spec['route'])
        return item
    monkeypatch.setattr(transport,'ordinary',ordinary)
    monkeypatch.setattr(transport,'verify_saved_record',lambda row,freeze:None)
    monkeypatch.setattr(transport,'deadline_failure',lambda clock,route:clock['status']=='timeout')
    monkeypatch.setattr(runner,'validate_transition',lambda row,root:({},dict(status='passed')))
    return calls,analysis


def test_slow_completed_requests_are_saved_and_not_gated(output,monkeypatch):
    row=dict(completed=True,clock=dict(status='ok',receipt_elapsed_seconds=2.0),action={},trace={})
    calls,analysis=_collect_fixture(monkeypatch,[row,row])
    checked=[]
    def exact(row,spec,*args,**kwargs):
        assert (c.OUT/'records'/f"{spec['sequence']:04d}.json").exists()
        checked.append(spec['sequence'])
    monkeypatch.setattr(analysis,'validate_repeat_row',exact)
    panel=synthetic_panel();matrix=design.build_matrix(panel)[:2];rows=[]
    runner.collect(SimpleNamespace(guard=lambda:None),{},panel,matrix,{'research':object(),'frozen':object()},rows)
    assert calls==checked==[0,1] and len(rows)==2
    assert not(c.OUT/'collection_stop.json').exists()


def test_original_failure_saved_before_stop_no_retry(output,monkeypatch):
    failed=dict(completed=False,clock=dict(status='timeout',accepted=False,deadline_seconds=10.,result=None),action=None,trace=None)
    calls,analysis=_collect_fixture(monkeypatch,[failed])
    monkeypatch.setattr(analysis,'validate_repeat_row',lambda *a,**k:pytest.fail('No accepted science for failure'))
    panel=synthetic_panel();matrix=design.build_matrix(panel);rows=[]
    runner.collect(SimpleNamespace(guard=lambda:None),{},panel,matrix,{'research':object(),'frozen':object()},rows)
    assert calls==[0] and len(rows)==1
    stop=c.read(c.OUT/'collection_stop.json')
    assert stop['clock']==failed['clock'] and stop['no_retry'] is True


def test_scientific_mismatch_preserved_before_any_next_request(output,monkeypatch):
    row=dict(completed=True,clock=dict(status='ok',receipt_elapsed_seconds=.6),action={},trace={})
    calls,analysis=_collect_fixture(monkeypatch,[row])
    def reject(*args,**kwargs):raise ValueError('exact scientific mismatch')
    monkeypatch.setattr(analysis,'validate_repeat_row',reject)
    panel=synthetic_panel();matrix=design.build_matrix(panel);rows=[]
    with pytest.raises(ValueError,match='scientific mismatch'):
        runner.collect(SimpleNamespace(guard=lambda:None),{},panel,matrix,{'research':object(),'frozen':object()},rows)
    assert calls==[0] and (c.OUT/'records/0000.json').exists()


def test_static_source_closure_contains_unchanged_transport_and_runtime():
    files=preparation.python_dependencies([Path(runner.__file__),Path(transport.__file__)])
    paths={p.relative_to(c.ROOT).as_posix() for p in files}
    assert 'tools/local_player_qualification_128/clock.py' in paths
    assert 'tools/splendorust_search128_resume_20261029/accounting.py' in paths
    assert 'tools/future_training_rules_20261021.py' in paths
    assert 'tools/splendorust_search128_repeat_20261029/runner.py' in paths


def test_real_saved_transition_and_corrupted_successor():
    panel=design.build_panel();root=panel['roots'][0]
    row=dict(observer=root['observer'],action=root['original_result128']['action'],quota=128)
    after,proof=runner.validate_transition(row,root)
    assert after==root['after'] and proof['corrected_engine_and_independent_ledger'] is True
    corrupted=dict(root,after={})
    with pytest.raises(ValueError,match='successor'):runner.validate_transition(row,corrupted)


def test_transition_error_saved_before_next_request(output,monkeypatch):
    row=dict(completed=True,clock=dict(status='ok',receipt_elapsed_seconds=.6),action={},trace={})
    calls,analysis=_collect_fixture(monkeypatch,[row])
    def reject(*args):raise ValueError('dictionary ledger mismatch')
    monkeypatch.setattr(runner,'validate_transition',reject)
    panel=synthetic_panel();rows=[]
    with pytest.raises(ValueError,match='transition failed'):
        runner.collect(SimpleNamespace(guard=lambda:None),{},panel,design.build_matrix(panel),
                       {'research':object(),'frozen':object()},rows)
    saved=c.read(c.OUT/'records/0000.json')
    assert calls==[0] and 'dictionary ledger mismatch' in saved['transition_error']
    assert saved['clock']==row['clock'] and saved['after'] is None


def test_owner_publication_and_no_relaunch_source_contract():
    import ast
    source=Path(guardian.__file__).read_text()
    tree=ast.parse(source)
    for name in ('guardian_main','inner_main'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        text=ast.get_source_segment(source,node)
        assert "c.JOB_SECONDS" in text and "freeze_sha256" in text
        assert 'process.returncode' in text
    assert "c.atomic_save(c.OUT/'guardian_claim.json'" in source
    assert "c.atomic_save(c.OUT/'guardian_owner.json'" in source
    assert 'SPLENDOR_SEARCH128_REPEAT_PHASE' in source
    assert 'tools.splendorust_search128_repeat_20261029.runner' in source


def test_reader_clock_clips_to_original_final_and_global():
    from tools.splendorust_search128_repeat_20261029.independent_review import review_deadlines
    claim=dict(monotonic_origin=100.,unix_origin=1000.)
    final=dict(monotonic_deadline=3500.,unix_deadline=4400.)
    assert review_deadlines(claim,final,3400.,4300.)==(3500.,4400.)
    assert review_deadlines(claim,dict(monotonic_deadline=9000.,unix_deadline=10000.),3650.,4550.)==(3700.,4600.)
    assert review_deadlines(claim,final,2000.,2900.)==(2600.,3500.)


def test_reader_expired_clock_rejects_before_resource_calls(monkeypatch):
    from tools.splendorust_search128_repeat_20261029 import independent_review as review
    value=object.__new__(review.Guard);value.deadline=-1.;value.wall_deadline=-1.
    monkeypatch.setattr(review.psutil,'Process',lambda:pytest.fail('No resource query after expired clock'))
    with pytest.raises(ValueError,match='original final'):value.guard()


def test_actual_owner_failure_cannot_be_accepted(output):
    from tools.splendorust_search128_repeat_20261029 import independent_review as review
    c.save(c.OUT/'execution.json',dict(status='runner_failed',exit_code=1,freeze_sha256='f',
                                    cleanup_errors=[],charged_job_seconds=100))
    with pytest.raises(ValueError,match='actual runner'):review.check_successful_owners('f')


@pytest.mark.parametrize('exit_code,claimed_approved,expected',[(0,True,0),(1,True,1),(0,False,1)])
def test_reader_owned_exit_and_output_both_required(output,monkeypatch,exit_code,claimed_approved,expected):
    from tools.splendorust_search128_repeat_20261029 import independent_review as review
    c.save(c.OUT/'stages/final_start.json',dict(monotonic_deadline=9999.,unix_deadline=9999.))
    c.save(c.OUT/'independent_review.json',dict(approved=claimed_approved,freeze_sha256='f',reviewer=dict(pid=72,created=7.)))
    monkeypatch.setattr(review,'review_deadlines',lambda *a:(9999.,9999.))
    monkeypatch.setattr(guardian.time,'monotonic',lambda:100.)
    monkeypatch.setattr(guardian.time,'time',lambda:1000.)
    process=SimpleNamespace(pid=72,returncode=exit_code,poll=lambda:exit_code)
    launched=[]
    def launch(command,**kwargs):
        launched.append((command,kwargs));return process
    monkeypatch.setattr(guardian.subprocess,'Popen',launch)
    monkeypatch.setattr(guardian.psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:7.))
    monkeypatch.setattr(guardian,'_WindowsJob',lambda pid:object())
    cleaned=[]
    def cleanup(*args,**kwargs):cleaned.append(kwargs);return dict(errors=[])
    monkeypatch.setattr(guardian,'cleanup_owned',cleanup)
    code=guardian.independent_main(SimpleNamespace(freeze_sha256='f'),dict(monotonic_origin=10.,unix_origin=900.),{})
    assert code==expected and len(launched)==1 and len(cleaned)==1
    assert cleaned[0]['extra_deadlines']==(9999.,9999.)
    assert launched[0][1]['creationflags']==getattr(guardian.subprocess,'DETACHED_PROCESS',0)
    receipt=c.read(c.OUT/'independent_review_os_exit.json')
    assert receipt['exit_code']==exit_code and receipt['reviewer']==dict(pid=72,created=7.)
    assert receipt['status']==('completed' if expected==0 else 'failed')
    assert receipt['review']==(c.artifact(c.OUT/'independent_review.json') if expected==0 else None)


def test_reader_postspawn_failure_keeps_cleanup_and_actual_exit(output,monkeypatch):
    from tools.splendorust_search128_repeat_20261029 import independent_review as review
    c.save(c.OUT/'stages/final_start.json',{})
    monkeypatch.setattr(review,'review_deadlines',lambda *a:(9999.,9999.))
    monkeypatch.setattr(guardian.time,'monotonic',lambda:100.)
    monkeypatch.setattr(guardian.time,'time',lambda:1000.)
    process=SimpleNamespace(pid=72,returncode=None,poll=lambda:None)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:process)
    monkeypatch.setattr(guardian.psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:7.))
    def fail(pid):raise RuntimeError('job admission failed')
    monkeypatch.setattr(guardian,'_WindowsJob',fail)
    def cleanup(*args,**kwargs):process.returncode=1;return dict(errors=[])
    monkeypatch.setattr(guardian,'cleanup_owned',cleanup)
    assert guardian.independent_main(SimpleNamespace(freeze_sha256='f'),dict(monotonic_origin=10.,unix_origin=900.),{})==1
    receipt=c.read(c.OUT/'independent_review_os_exit.json')
    assert receipt['exit_code']==1 and 'job admission failed' in receipt['error']


def test_absolute_runtime_descriptor_preserved_and_tamper_rejected(output,tmp_path):
    import gzip
    runtime=Path(gzip.__file__).resolve()
    item=dict(path=str(runtime),bytes=runtime.stat().st_size,sha256=c.file_hash(runtime))
    original=dict(item);c.verify([item]);assert item==original
    file=tmp_path/'runtime.py';file.write_bytes(b'abc')
    pin=dict(path=str(file.resolve()),bytes=3,sha256=c.file_hash(file));c.verify([pin])
    with pytest.raises(ValueError,match='Changed frozen'):c.verify([dict(pin,bytes=4)])
    file.write_bytes(b'abd')
    with pytest.raises(ValueError,match='Changed frozen'):c.verify([pin])
    with pytest.raises(ValueError,match='Canonical workspace'):c.verify([dict(pin,path='../escape.py')])

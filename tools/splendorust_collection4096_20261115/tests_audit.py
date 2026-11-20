"""Model-free failures and complete owned-receipt fixture for final acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from . import common as c,audit


@pytest.fixture
def work(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'study')
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json');c.OUT.mkdir()
    # Synthetic receipt checks test authenticated contents, not physical flush latency.
    monkeypatch.setattr(c.os,'fsync',lambda fd:None)
    return tmp_path


def rewrite(path,value):
    Path(path).write_text(c.canonical(value)+'\n',encoding='utf8')


@pytest.fixture
def frozen(work):
    plan=dict(inputs=[],corrected_source_receipt={'fixture_guard':'corrected'})
    item=c.save(c.PLAN,plan)
    claim=dict(freeze=item,plan_sha256=item['sha256'],total_seconds=c.TOTAL_SECONDS,
        model_seconds=c.MODEL_SECONDS,automatic_retries=0,monotonic_origin=0.,unix_origin=1700000000.)
    claimitem=c.save(c.OUT/'claim.json',claim)
    for name,started,ended in [('qualification',10.,900.),('collection',1000.,100000.),
                               ('corpus',101000.,109000.),('final',110000.,None)]:
        cap=c.STAGES[name];cutoff=c.TOTAL_SECONDS if name=='final' else c.MODEL_SECONDS
        phase=dict(stage=name,claim=claimitem,cap_seconds=cap,monotonic=started,unix_time=claim['unix_origin']+started,
            monotonic_deadline=min(started+cap,cutoff),unix_deadline=claim['unix_origin']+min(started+cap,cutoff))
        phaseitem=c.save(c.OUT/'stages'/f'{name}_start.json',phase)
        if ended is not None:c.save(c.OUT/'stages'/f'{name}_end.json',dict(stage=name,start=phaseitem,
            monotonic=ended,unix_time=claim['unix_origin']+ended))
    return plan,claim,audit.verify_phases(claim)


def job_fixture(job,frozen,index=0):
    plan,claim,phases=frozen;folder=c.OUT/job['path'];phase=phases[job['phase']]
    start=phase['start']['monotonic']+1.+index*10.;wall=claim['unix_origin']+start
    deadline=min(start+job['seconds'],phase['start']['monotonic_deadline'])
    walldeadline=claim['unix_origin']+deadline
    args=['python.exe','-B','-u','-m',job['module'],'--mode',job['mode'],'--plan-sha256',c.sha(c.PLAN),
        '--output',str(folder/'worker')]
    if job['phase']=='collection':
        ident=folder.name;cellpath=c.OUT/'collection/cell_clocks'/f'{ident}.json'
        args+=['--cell-id',ident]
        if not cellpath.exists():c.save(cellpath,dict(cell_id=ident,phase=phase['start_item'],
            monotonic_origin=start,unix_origin=wall,monotonic_deadline=start+4*3600,unix_deadline=wall+4*3600))
        cell=c.read(cellpath);deadline=min(deadline,cell['monotonic_deadline']);walldeadline=min(walldeadline,cell['unix_deadline'])
    identity=dict(pid=40000+index,created=1600000000.+index)
    owner=dict(identity=identity,mode=job['mode'],fit_id=None,plan_sha256=c.sha(c.PLAN),windows_job_assigned=True,
        model_work=job['model'],global_claim=c.artifact(c.OUT/'claim.json'),phase_clock=phase['start_item'],
        maximum_seconds=job['seconds'],maximum_rss_bytes=job['rss'],args=args,monotonic_origin=start,unix_origin=wall,
        monotonic_deadline=deadline,unix_deadline=walldeadline)
    c.save(folder/'owner.json',owner)
    value=dict(status='passed',worker_exit_code=0,cleanup_errors=[],error=None,retries=0,identity=identity,
        observed_owned_identities=[dict(pid=50000+index,created=1600000000.+index)],mode=job['mode'],fit_id=None,
        plan_sha256=c.sha(c.PLAN),maximum_seconds=job['seconds'],peak_rss_bytes=1024,
        monotonic_origin=start,unix_origin=wall,monotonic_end=start+1.,unix_end=wall+1.,
        monotonic_deadline=deadline,unix_deadline=walldeadline)
    c.save(folder/'execution.json',value)
    c.save(folder/'worker/runtime_sources.json',dict(sources=plan['corrected_source_receipt'],identity=identity,
        original_claim=owner['global_claim'],original_phase=owner['phase_clock'],mode=job['mode'],fit_id=None))
    c.save(folder/'worker/result.json',dict(status='passed',mode=job['mode'],identity=identity,result={}))
    (folder/'worker/stderr.log').write_bytes(b'')
    return folder


def test_exact_collection_only_execution_count():
    rows=audit.expected_jobs()
    assert len(rows)==421 and len({r['path'] for r in rows})==421
    assert sum(r['mode']=='games' for r in rows)==208
    assert sum(r['mode']=='audit_cell' for r in rows)==208
    assert rows[-1]['path']=='union_build'
    assert not any(r['mode'] in ('fit','offline','main_block','operational') for r in rows)


@pytest.mark.parametrize('mutation',['extended','claim_changed','end_late','phase_overlap'])
def test_phase_clock_failures(frozen,mutation):
    plan,claim,phases=frozen
    path=c.OUT/'stages/collection_start.json';row=c.read(path)
    if mutation=='extended':row['monotonic_deadline']+=1
    elif mutation=='claim_changed':row['claim']['sha256']='0'*64
    elif mutation=='phase_overlap':row['monotonic']=500
    else:
        path=c.OUT/'stages/collection_end.json';row=c.read(path);row['unix_time']=phases['collection']['start']['unix_deadline']
    rewrite(path,row)
    with pytest.raises(ValueError):audit.verify_phases(claim)


@pytest.mark.parametrize('index',[0,1,2,3,4,419,420])
def test_each_original_worker_role(frozen,index):
    job=audit.expected_jobs()[index];job_fixture(job,frozen,index)
    absent=[];proof=audit.verify_execution(job,*frozen,absent=absent.append)
    assert len(absent)==2 and proof['execution']['path'].endswith('execution.json')


@pytest.mark.parametrize('mutation',['nonzero_exit','bool_exit','cleanup','error','retry','wrong_identity','wrong_mode',
    'clock_extended','late','missing_absence','wrong_source_guard','result_failed','extra_args','model_flag','rss'])
def test_owned_job_failures(frozen,mutation):
    job=audit.expected_jobs()[0];folder=job_fixture(job,frozen)
    path=folder/'execution.json';row=c.read(path)
    if mutation=='nonzero_exit':row['worker_exit_code']=1
    elif mutation=='bool_exit':row['worker_exit_code']=False
    elif mutation=='cleanup':row['cleanup_errors']=['still running']
    elif mutation=='error':row['error']='original failure'
    elif mutation=='retry':row['retries']=1
    elif mutation=='wrong_identity':row['identity']['created']+=1
    elif mutation=='wrong_mode':row['mode']='fit'
    elif mutation=='clock_extended':row['monotonic_deadline']+=1
    elif mutation=='late':row['unix_end']=row['unix_deadline']
    elif mutation=='rss':row['peak_rss_bytes']=job['rss']+1
    elif mutation=='wrong_source_guard':
        path=folder/'worker/runtime_sources.json';row=c.read(path);row['sources']={}
    elif mutation=='result_failed':
        path=folder/'worker/result.json';row=c.read(path);row['status']='failed'
    elif mutation in ('extra_args','model_flag'):
        path=folder/'owner.json';row=c.read(path)
        if mutation=='extra_args':row['args']+=['--fit-id','unexpected']
        else:row['model_work']=False
    rewrite(path,row)
    def absent(identity):
        if mutation=='missing_absence':raise ValueError('exact owned process still exists')
    with pytest.raises(ValueError):audit.verify_execution(job,*frozen,absent=absent)


def test_cell_pair_uses_original_shared_clock(frozen):
    jobs=audit.expected_jobs()[3:5]
    for i,job in enumerate(jobs):job_fixture(job,frozen,i)
    for job in jobs:audit.verify_execution(job,*frozen,absent=lambda i:None)
    clock=c.OUT/'collection/cell_clocks/c000.json';row=c.read(clock);row['unix_deadline']+=10;rewrite(clock,row)
    with pytest.raises(ValueError):audit.verify_execution(jobs[1],*frozen,absent=lambda i:None)


@pytest.fixture
def complete_receipts(frozen):
    for i,job in enumerate(audit.expected_jobs()):job_fixture(job,frozen,i)
    folder=c.OUT/'final_audit/worker';folder.mkdir(parents=True)
    return frozen,folder


def test_complete_421_saved_executions(complete_receipts):
    frozen,folder=complete_receipts
    assert len(audit.verify_execution_inventory(folder))==421
    for job in audit.expected_jobs():audit.verify_execution(job,*frozen,absent=lambda i:None)


@pytest.mark.parametrize('mutation',['missing','extra','failure','worker_failure'])
def test_no_omissions_hidden_jobs_or_failures(complete_receipts,mutation):
    frozen,folder=complete_receipts
    if mutation=='missing':(c.OUT/'union_build/execution.json').unlink()
    elif mutation=='extra':c.save(c.OUT/'unexpected/execution.json',dict(worker_exit_code=0))
    else:c.save(c.OUT/'collection/cells/c000'/('failure.json' if mutation=='failure' else 'worker_failure.json'),dict(error='preserve'))
    with pytest.raises(ValueError):audit.verify_execution_inventory(folder)


def test_inventory_streams_exact_closed_bytes(work):
    c.save(c.OUT/'closed.json',dict(a=1));c.status(c.OUT/'stage.json',dict(active=True))
    (c.OUT/'launch').mkdir();(c.OUT/'launch/stdout.log').write_text('still active',encoding='utf8')
    (c.OUT/'launch/stderr.log').write_bytes(b'')
    folder=c.OUT/'final_audit/worker';folder.mkdir(parents=True)
    c.save(folder/'active.json',dict(active=True))
    proof=audit.preserve_evidence(folder)
    assert proof['preserved_file_count']==1
    rows=[json.loads(line) for line in c.resolve(proof['physical_inventory']).read_text().splitlines()]
    assert rows==[c.artifact(c.OUT/'closed.json')]


@pytest.mark.parametrize('mutation',['none','changed','unbound'])
def test_frozen_prelaunch_failures_preserved_not_production_failures(work,mutation):
    folder=c.OUT/'final_audit/worker';folder.mkdir(parents=True)
    for container in audit.expected_executions():c.save(container/'execution.json',dict(worker_exit_code=0))
    probe=c.save(c.OUT/'prelaunch/launcher_probe_a/failure/execution.json',dict(worker_exit_code=7,status='failed'))
    failed=c.save(c.OUT/'prelaunch/launcher_probe_a/failure/failure.json',dict(intentional_engineering_failure=True))
    c.save(c.PLAN,dict(prelaunch_evidence=[probe,failed]))
    if mutation=='changed':rewrite(c.resolve(probe),dict(worker_exit_code=0,status='passed'))
    if mutation=='unbound':c.save(c.OUT/'prelaunch/unapproved/execution.json',dict(worker_exit_code=0))
    if mutation!='none':
        with pytest.raises(ValueError):audit.verify_execution_inventory(folder)
    else:
        assert len(audit.verify_execution_inventory(folder))==421
        proof=audit.preserve_evidence(folder)
        preserved=[json.loads(x)['path'] for x in c.resolve(proof['physical_inventory']).read_text().splitlines()]
        assert probe['path'] in preserved and failed['path'] in preserved


def test_preservation_rejects_concurrent_closed_evidence_change(work,monkeypatch):
    path=c.OUT/'closed.json';c.save(path,dict(a=1));folder=c.OUT/'final_audit/worker';folder.mkdir(parents=True)
    original=c.artifact
    def modified(target):
        result=original(target)
        if target==path:path.write_text('changed closed evidence',encoding='utf8')
        return result
    monkeypatch.setattr(c,'artifact',modified)
    with pytest.raises(ValueError,match='Closed evidence changed'):audit.preserve_evidence(folder)


def test_full_output_bits_keep_signed_zero_and_reject_nonfloat32():
    rows=[[0.] *83 for _ in range(32)];original=audit.full_outputs(rows)
    rows[0][0]=-0.;assert audit.full_outputs(rows)!=original
    rows[0][0]=.1
    with pytest.raises(ValueError):audit.full_outputs(rows)


def test_collection_final_orchestrates_all_saved_checks(complete_receipts,monkeypatch):
    frozen,folder=complete_receipts;plan,claim,phases=frozen
    original=audit.verify_execution
    monkeypatch.setattr(audit,'verify_execution',lambda job,p,c0,ph,**kw:original(job,p,c0,ph,absent=lambda i:None))
    calls=[]
    monkeypatch.setattr(audit,'verify_collection_receipts',lambda p,**kw:(calls.append('accepted') or {'accepted_fixture':True}))
    proof=audit.final(plan,folder,lambda:None)
    assert proof['planned_predecessors']==421 and proof['scientific_optimizer_calls']==0
    assert proof['new_scientific_source_games']==3328 and proof['fresh_assessment_games']==256
    assert proof['owner_exit_still_requires_observation'] and calls==['accepted']
    assert 'torch' not in sys.modules


def test_module_import_and_final_source_do_not_import_tensor_runtime():
    assert 'torch' not in sys.modules
    import ast
    tree=ast.parse(Path(audit.__file__).read_text())
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any((isinstance(n,ast.ImportFrom) and n.module and 'fit' in n.module)
        or (isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)) for n in imports)

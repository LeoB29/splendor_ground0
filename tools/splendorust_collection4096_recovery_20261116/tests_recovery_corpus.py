"""Recovery-specific provenance, immutable-prefix and execution inventory tests."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from . import common as c, corpus, audit
from .collection import common as cc


@pytest.mark.parametrize('number,segment',[(0,'original_passed'),(40,'original_passed'),(41,'original_recovered'),
    (42,'recovery_new'),(207,'recovery_new')])
def test_exact_cell_segments(number,segment):
    value,path=corpus.cell_source(f'c{number:03d}')
    assert value==segment
    expected=c.ORIGINAL_OUT if number<42 else c.OUT
    assert path==expected/'collection/cells'/f'c{number:03d}'


@pytest.mark.parametrize('bad',['c-01','c208','c999','c01','c0000','C001','../c001',41,True,None])
def test_cell_resolver_rejects_unplanned_or_ambiguous_ids(bad):
    with pytest.raises(ValueError):corpus.cell_source(bad)


def test_exact_recovery_jobs_no_duplicated_games_or_qualifier_fit():
    jobs=audit.expected_jobs()
    assert len(jobs)==339
    assert len({j['path'] for j in jobs})==339
    assert [j['mode'] for j in jobs[:5]]==['saved_boundary','recovery_qualification','audit_recovery_qualification',
        'supplemental_replays','supplemental_audit']
    assert all(j['phase']=='supplemental' and j['seconds']==3600 for j in jobs[:5])
    fresh=[j for j in jobs if j['mode']=='games']
    assert [Path(j['path']).name for j in fresh]==[f'c{i:03d}' for i in range(42,208)]
    assert all(j['seconds']==4*3600 for j in fresh)
    assert not any('.fit' in j['module'] for j in jobs)


def test_first_audit_mapping_retains_original_failure():
    assert corpus.first_audit_path('c040')==c.ORIGINAL_OUT/'collection/cell_audits/c040/worker/saved_audit.json'
    assert corpus.first_audit_path('c041')==cc.OUT/'recovery_audit/worker/saved_audit.json'
    assert corpus.first_audit_path('c042')==cc.OUT/'cell_audits/c042/worker/saved_audit.json'


@pytest.fixture
def toy(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'recovery')
    monkeypatch.setattr(c,'ORIGINAL_OUT',tmp_path/'original')
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json')
    monkeypatch.setattr(cc,'OUT',c.OUT/'collection')
    monkeypatch.setattr(cc,'PLAN',cc.OUT/'plan.json')
    c.OUT.mkdir();c.ORIGINAL_OUT.mkdir()
    return tmp_path


def original_inventory():
    a=c.ORIGINAL_OUT/'failure.json';a.write_text('{"status":"incomplete"}')
    b=c.ORIGINAL_OUT/'record.bin';b.write_bytes(b'unchanged saved original')
    entries=[c.artifact(a),c.artifact(b)]
    path=c.OUT/'original_inventory.jsonl'
    path.write_text(''.join(c.canonical(e)+'\n' for e in entries))
    return dict(original_preservation_inventory=c.artifact(path)),entries


def test_original_preservation_includes_failure_and_exact_bytes(toy):
    plan,entries=original_inventory();result=audit.verify_original_preservation(plan)
    assert result['preserved_files']==2
    assert result['preserved_bytes']==sum(x['bytes'] for x in entries)
    assert result['original_failed_attempt_preserved'] and result['all_original_bytes_unchanged']


@pytest.mark.parametrize('kind',['changed','missing','extra','duplicate','outside'])
def test_preservation_rejects_changes_omission_extra_and_escape(toy,kind):
    plan,entries=original_inventory();path=c.resolve(plan['original_preservation_inventory'])
    if kind=='changed':(c.ORIGINAL_OUT/'record.bin').write_bytes(b'tampered')
    elif kind=='missing':(c.ORIGINAL_OUT/'record.bin').unlink()
    elif kind=='extra':(c.ORIGINAL_OUT/'invented.json').write_text('{}')
    elif kind=='duplicate':
        with path.open('a') as f:f.write(c.canonical(entries[0])+'\n')
        plan['original_preservation_inventory']=c.artifact(path)
    else:
        outside=c.OUT/'outside.json';outside.write_text('{}')
        with path.open('a') as f:f.write(c.canonical(c.artifact(outside))+'\n')
        plan['original_preservation_inventory']=c.artifact(path)
    with pytest.raises((ValueError,FileNotFoundError)):audit.verify_original_preservation(plan)


def test_real_saved_original_prefix_checks_actual_exit_and_failed_tail():
    observer=c.ROOT/'outputs/splendorust_collection4096_launch_20261115_01/observed_launcher_exit.json'
    plan=dict(original_freeze=c.artifact(c.ORIGINAL_OUT/'freeze.json'),inputs=[c.artifact(observer)])
    proof=audit.verify_original_prefix(plan)
    assert len(proof['successful_actual_exits'])==85
    assert proof['original_attempt_status']=='incomplete'
    assert proof['failed_execution']==c.artifact(c.ORIGINAL_OUT/'collection/cells/c041/execution.json')
    assert proof['training_qualification']['optimizer_calls']==2


def test_real_saved_cell_authenticates_complete_moves_without_models():
    import sys
    plan=cc.read(c.ORIGINAL_OUT/'collection/plan.json')
    before='torch' in sys.modules
    proof=corpus.authenticate_original_cell(plan,plan['cells'][0])
    assert proof['games']==16 and proof['full_cohort_verified']
    assert proof['replay']['count']==48
    assert ('torch' in sys.modules)==before


def test_original_game_schedule_substitution_rejected_before_reuse():
    plan=cc.read(c.ORIGINAL_OUT/'collection/plan.json');bad=deepcopy(plan)
    bad['cells'][0]['specifications'][0]['seed_decimal']='0'
    with pytest.raises(ValueError):corpus.authenticate_original_cell(bad,bad['cells'][0])



def recovered_acceptance():
    from .collection import data
    (c.ORIGINAL_OUT/'freeze.json').write_text('{}')
    c.save(c.PLAN,{})
    c.save(cc.PLAN,{})
    value=dict(schema='collection4096_new_corpus_acceptance_v1',status='accepted',passed=True,
        game_count=3328,independent_setup_count=3328,partition_counts=c.NEW_GAME_COUNTS,
        source_plan=c.artifact(cc.PLAN),catalogue=dict(path='catalogue.json',bytes=0,sha256='a'*64),
        original_attempt_status='incomplete',amended_recovered_verification=True,
        original_collection_freeze=c.artifact(c.ORIGINAL_OUT/'freeze.json'),recovery_freeze=c.artifact(c.PLAN),
        original_completed_games=672,newly_collected_games=2656,gameplay_retries=0,verification_retries=1,
        interrupted_verification=dict(game_id='g0666',ply=40),**data.IDENTITY)
    return value


def test_recovered_acceptance_discloses_retry_and_original_incomplete(toy):
    value=recovered_acceptance();item=c.save(cc.OUT/'corpus_acceptance.json',value)
    assert corpus._receipt(item,lambda:None)['acceptance']==item


@pytest.mark.parametrize('change',[
    {'original_attempt_status':'passed'}, {'amended_recovered_verification':False}, {'original_completed_games':671},
    {'newly_collected_games':2657}, {'gameplay_retries':1}, {'verification_retries':0}, {'verification_retries':2},
    {'interrupted_verification':dict(game_id='g0666',ply=20)}])
def test_recovered_acceptance_cannot_hide_or_expand_retry(toy,change):
    value=recovered_acceptance();value.update(change);item=c.save(cc.OUT/'corpus_acceptance.json',value)
    with pytest.raises(ValueError):corpus._receipt(item,lambda:None)


@pytest.mark.parametrize('technical',['prelaunch','test_artifacts'])
def test_only_exact_frozen_technical_files_excluded(toy,monkeypatch,technical):
    technical_path=c.OUT/technical/'saved/execution.json'
    item=c.save(technical_path,dict(worker_exit_code=1,preserved_intentional_fixture=True))
    c.save(c.PLAN,dict(prelaunch_evidence=[item]))
    monkeypatch.setattr(audit,'expected_executions',lambda:[])
    assert audit.verify_execution_inventory(c.OUT/'final_audit/worker')==set()
    c.save(technical_path.parent/'unbound.json',{})
    with pytest.raises(ValueError):audit.verify_execution_inventory(c.OUT/'final_audit/worker')


def execution_fixture():
    job=audit.expected_jobs()[0];container=c.OUT/job['path'];worker=container/'worker'
    freeze=c.save(c.PLAN,{})
    claim=dict(monotonic_origin=0.,unix_origin=1000.)
    claim_item=c.save(c.OUT/'claim.json',claim)
    phase=dict(stage='supplemental',monotonic=100.,unix_time=1100.,monotonic_deadline=3700.,unix_deadline=4700.)
    phase_item=c.save(c.OUT/'stages/supplemental_start.json',phase)
    phases=dict(supplemental=dict(start=phase,start_item=phase_item,end=dict(monotonic=200.,unix_time=1200.)))
    identity=dict(pid=12345,created=1000.)
    args=['python.exe','-B','-u','-m',job['module'],'--mode',job['mode'],'--plan-sha256',freeze['sha256'],
        '--output',str(worker)]
    owner=dict(identity=identity,mode=job['mode'],fit_id=None,plan_sha256=freeze['sha256'],windows_job_assigned=True,
        model_work=False,global_claim=claim_item,phase_clock=phase_item,maximum_seconds=3600,maximum_rss_bytes=job['rss'],
        args=args,monotonic_origin=100.,unix_origin=1100.,monotonic_deadline=3700.,unix_deadline=4700.)
    execution=dict(identity=identity,mode=job['mode'],fit_id=None,plan_sha256=freeze['sha256'],status='passed',
        worker_exit_code=0,cleanup_errors=[],error=None,retries=0,maximum_seconds=3600,peak_rss_bytes=100,
        monotonic_origin=100.,unix_origin=1100.,monotonic_end=120.,unix_end=1120.,
        monotonic_deadline=3700.,unix_deadline=4700.,observed_owned_identities=[])
    c.save(container/'owner.json',owner);c.save(container/'execution.json',execution)
    sources=dict(source='fixed')
    c.save(worker/'runtime_sources.json',dict(sources=sources,identity=identity,original_claim=claim_item,
        original_phase=phase_item,mode=job['mode'],fit_id=None))
    c.save(worker/'result.json',dict(status='passed',mode=job['mode'],identity=identity))
    (worker/'stderr.log').write_text('')
    return job,dict(corrected_source_receipt=sources),claim,phases,container,execution


def test_actual_recovery_execution_receipt_verification(toy):
    job,plan,claim,phases,container,_=execution_fixture()
    proof=audit.verify_execution(job,plan,claim,phases,absent=lambda _:None)
    assert proof['execution']==c.artifact(container/'execution.json')


@pytest.mark.parametrize('change',[{'worker_exit_code':1},{'status':'failed'},{'retries':1},{'cleanup_errors':['left alive']},
    {'mode':'games'},{'maximum_seconds':7200},{'unix_end':4800.},{'peak_rss_bytes':9*1024**3}])
def test_execution_rejects_failed_retry_wrong_role_and_exceeded_bounds(toy,change):
    job,plan,claim,phases,container,value=execution_fixture();value.update(change)
    (container/'execution.json').write_text(c.canonical(value))
    with pytest.raises(ValueError):audit.verify_execution(job,plan,claim,phases,absent=lambda _:None)



def test_repeated_cell_proof_uses_exact_json_roundtrip_for_integer_histogram_keys(toy,monkeypatch):
    cell=dict(cell_id='c042',specifications=[dict(game_id='g0672')])
    proof=dict(game_proofs=[dict(game_id='g0672')],backend_metrics=dict(occupied_rows_histogram={1:2,8:11}))
    first=corpus.first_audit_path('c042');descriptor=c.save(first,proof)
    c.save(first.parent/'result.json',dict(status='passed',result=descriptor))
    monkeypatch.setattr(corpus.audit,'set_boundary',lambda _:None)
    monkeypatch.setattr(corpus.audit,'audit_cell',lambda *args:deepcopy(proof))
    assert corpus.audit_source_cell({},cell)==proof
    changed=deepcopy(proof);changed['backend_metrics']['occupied_rows_histogram'][8]=12
    monkeypatch.setattr(corpus.audit,'audit_cell',lambda *args:changed)
    with pytest.raises(ValueError):corpus.audit_source_cell({},cell)

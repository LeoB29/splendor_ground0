"""Synthetic amendment/provenance tests. No models or production analysis."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from tools import analyze_matched_strength_recovery_20261022 as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def plan():
    return {'schema':'matched_strength_amended_recovery_plan_v1',
        'source_base':a.SOURCE.relative_to(a.ROOT).as_posix(),
        'completed_games':list(a.old.GAME_ORDER[:-1]),
        'partial_game':{'game':59,'next_ply':43,'state_sha256':'a'*64},
        'amendment':{'retry_identity':'games/g059/t0043','new_allowance_seconds':120,
            'maximum_retries':1,'partial_search_reused':False,'original_frozen_status':'incomplete'},
        'preserved_timeout':{'game':39,'ply':7,'status':'forfeit_timeout'},
        'original_primary_complete':False,'validation_reused':True,'new_validation_requests':0,'outcomes_closed':True}


def scores(value=1):
    return [{'game':g,'pair':g//2,'fit':a.old.FITS[(g//2)%3],
        'local_seat':1-((g//2+g%2)%2),'local_score_halfpoints':value,
        'termination':'natural','forfeiting_player':None} for g in range(60)]


def test_exact_prospective_amendment_is_accepted():
    a.check_plan(plan())


@pytest.mark.parametrize('change',[
    lambda p:p.update(original_primary_complete=True),
    lambda p:p.update(outcomes_closed=False),
    lambda p:p.update(new_validation_requests=12),
    lambda p:p.update(validation_reused=False),
    lambda p:p['completed_games'].pop(),
    lambda p:p['partial_game'].update(game=58),
    lambda p:p['partial_game'].update(next_ply=42),
    lambda p:p['amendment'].update(retry_identity='games/g039/t0007'),
    lambda p:p['amendment'].update(new_allowance_seconds=121),
    lambda p:p['amendment'].update(maximum_retries=2),
    lambda p:p['amendment'].update(partial_search_reused=True),
    lambda p:p['amendment'].update(original_frozen_status='complete'),
    lambda p:p['preserved_timeout'].update(status='interruption'),
    lambda p:p['amendment'].update(maximum_retries=True),
    lambda p:p.update(new_validation_requests=False),
])
def test_amendment_cannot_expand_or_relabel_original(change):
    value=plan();change(value)
    with pytest.raises(ValueError):a.check_plan(value)


@pytest.mark.parametrize('name,original',[
    ('validation/v000',True),('validation/v011',True),('games/g000/t0000',True),
    ('games/g039/t0007',True),('games/g059/t0042',True),('games/g059/t0043',False),('games/g059/t0044',False),
])
def test_request_origin_has_exact_retry_boundary(name,original):
    assert a.source_decision(name) is original


@pytest.mark.parametrize('name',['validation/v012','games/g060/t0000','games/g059/t0400',
    'games/g59/t43','games/g059/t0043/extra','../../games/g059/t0043','audits/g059-t0043'])
def test_request_origin_rejects_noncanonical_or_outside_fixed_sample(name):
    with pytest.raises(ValueError):a.source_decision(name)


def test_original_missing_game_bounds_do_not_use_recovered_result():
    low=scores();high=deepcopy(low)
    low[59]['local_score_halfpoints']=0;high[59]['local_score_halfpoints']=2
    lower=a.missing_original_game_sensitivity(low);upper=a.missing_original_game_sensitivity(high)
    assert lower==upper
    assert lower['lower_mean_score']==59/120
    assert lower['upper_mean_score']==61/120
    assert lower['maximum_point_estimate_width']==1/60
    assert lower['confidence_interval'] is None
    assert lower['recovered_game_result_used_for_original'] is False
    assert lower['original_frozen_status']=='incomplete'


@pytest.mark.parametrize('score,expected',[(0,(0,1/60)),(2,(59/60,1))])
def test_missing_game_bounds_are_correct_at_score_scale_extremes(score,expected):
    result=a.missing_original_game_sensitivity(scores(score))
    assert (result['lower_mean_score'],result['upper_mean_score'])==expected


def test_missing_game_sensitivity_retains_genuine_original_timeout_loss():
    rows=scores()
    rows[39].update(local_score_halfpoints=0,termination='forfeit',forfeiting_player='local')
    result=a.missing_original_game_sensitivity(rows)
    assert result['known_score_halfpoints']==58
    assert result['lower_mean_score']==58/120
    rows[39]['local_score_halfpoints']=1
    with pytest.raises(ValueError):a.missing_original_game_sensitivity(rows)


def test_incomplete_amended_sample_cannot_open_sensitivity():
    with pytest.raises(ValueError):a.missing_original_game_sensitivity(scores()[:-1])


def test_scientific_referee_world_and_manifest_methods_are_frozen_inheritance():
    for name in ('local_record','illegal_response','proof','manifest','context','process_identity'):
        assert getattr(a.Audit,name) is getattr(a.old.Audit,name)


def test_context_origin_restores_after_failure_and_does_not_change_old_globals(tmp_path):
    audit=a.Audit(tmp_path/'new');before=a.old.BASE
    with pytest.raises(RuntimeError):
        with audit.at(tmp_path/'old'):
            assert audit.base==(tmp_path/'old').resolve()
            raise RuntimeError('synthetic check failure')
    assert audit.base==(tmp_path/'new').resolve() and a.old.BASE==before


def test_import_binding_requires_both_source_and_destination_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(a.old,'ROOT',tmp_path)
    source=tmp_path/'source/moves/0000.json';destination=tmp_path/'new/moves/0000.json'
    write(source,{'safe':'original'});write(destination,{'safe':'original'})
    audit=a.Audit(tmp_path/'new')
    value={'source':a.old.artifact(source),'destination':a.old.artifact(destination)}
    audit.imports[destination.resolve()]=value
    assert audit.bound(value['source'],destination)==destination.resolve()
    write(destination,{'safe':'changed'})
    with pytest.raises(ValueError):audit.bound(value['source'],destination)
    write(destination,{'safe':'original'});write(source,{'safe':'changed'})
    with pytest.raises(ValueError):audit.bound(value['source'],destination)


def test_unmapped_source_descriptor_cannot_substitute_new_output(tmp_path,monkeypatch):
    monkeypatch.setattr(a.old,'ROOT',tmp_path)
    source=tmp_path/'source/row.json';destination=tmp_path/'new/row.json'
    write(source,{});write(destination,{})
    audit=a.Audit(tmp_path/'new')
    with pytest.raises(ValueError):audit.bound(a.old.artifact(source),destination)


def test_trace_binding_only_redirects_before_retry_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(a.old,'ROOT',tmp_path)
    audit=a.Audit(tmp_path/'new');audit.source=(tmp_path/'source').resolve()
    prior= audit.source/'workers/games/g059/t0042/continuations.jsonl.gz'
    amended= audit.output/'workers/games/g059/t0043/continuations.jsonl.gz'
    prior.parent.mkdir(parents=True);prior.write_bytes(b'old synthetic trace')
    amended.parent.mkdir(parents=True);amended.write_bytes(b'new synthetic trace')
    assert audit.bound(a.old.artifact(prior),audit.output/prior.relative_to(audit.source))==prior
    assert audit.bound(a.old.artifact(amended),amended)==amended
    wrong=audit.source/'workers/games/g059/t0043/continuations.jsonl.gz'
    wrong.parent.mkdir(parents=True);wrong.write_bytes(b'partial old trace')
    with pytest.raises(ValueError):audit.bound(a.old.artifact(wrong),amended)


@pytest.mark.parametrize('which',['original_wall','original_mono','recovery_wall','recovery_mono'])
def test_either_original_or_recovery_clock_blocks_analysis_at_final_reserve(monkeypatch,tmp_path,which):
    audit=a.Audit(tmp_path)
    audit.source_launch={'epoch_seconds':100.,'monotonic_start':200.}
    audit.launch={'epoch_seconds':200000.,'monotonic_start':200100.}
    wall,mono=200500.,200600.
    if which=='original_wall':wall=100+259200-600
    elif which=='original_mono':mono=200+259200-600
    elif which=='recovery_wall':wall=200000+43200-600
    else:mono=200100+43200-600
    monkeypatch.setattr(a.time,'time',lambda:wall);monkeypatch.setattr(a.time,'monotonic',lambda:mono)
    monkeypatch.setattr(a.shutil,'disk_usage',lambda root:SimpleNamespace(free=10*1024**3))
    with pytest.raises(ValueError,match='dual caps'):audit.boundary()


def test_boundary_does_not_refund_original_time_and_requires_disk(monkeypatch,tmp_path):
    audit=a.Audit(tmp_path);audit.source_launch={'epoch_seconds':100.,'monotonic_start':200.}
    audit.launch={'epoch_seconds':200000.,'monotonic_start':200100.}
    monkeypatch.setattr(a.time,'time',lambda:200500.);monkeypatch.setattr(a.time,'monotonic',lambda:200600.)
    monkeypatch.setattr(a.shutil,'disk_usage',lambda root:SimpleNamespace(free=10*1024**3))
    audit.boundary()
    monkeypatch.setattr(a.shutil,'disk_usage',lambda root:SimpleNamespace(free=5*1024**3-1))
    with pytest.raises(ValueError,match='disk reserve'):audit.boundary()


def test_exact_partial_state_hash_required_before_retry(monkeypatch,tmp_path):
    audit=a.Audit(tmp_path);snapshot={'synthetic':'prefix'};audit.plan=plan()
    observer={'synthetic':'lawful'};row={'observer':observer}
    audit.plan['preserved_interrupted_request']=[{'synthetic':'preserved'}]
    write(audit.output/'amended_retry.json',{'time':'2026-10-22T12:00:00+00:00','amendment':audit.plan['amendment'],
        'observer_sha256':a.digest(observer),'before_sha256':a.digest(snapshot),
        'preserved_original_request':audit.plan['preserved_interrupted_request'],
        'original_primary_complete':False,'outcomes_closed':True})
    called=[]
    monkeypatch.setattr(a.old.Audit,'decision',lambda self,*args:called.append(args) or 'verified')
    with pytest.raises(ValueError,match='reconstructed prefix'):
        audit.decision(row,snapshot,'games/g059/t0043','local','games')
    assert called==[]
    audit.plan['partial_game']['state_sha256']=a.digest(snapshot)
    assert audit.decision(row,snapshot,'games/g059/t0043','local','games')=='verified'
    with pytest.raises(ValueError):audit.decision(row,snapshot,'games/g059/t0043','opponent','games')
    with pytest.raises(ValueError):audit.decision(row,snapshot,'games/g059/t0043','local','validation')


def test_original_validation_uses_only_original_proofs_and_restores_context(monkeypatch,tmp_path):
    audit=a.Audit(tmp_path/'new');audit.source=(tmp_path/'old').resolve();seen=[]
    monkeypatch.setattr(a.old,'ROOT',tmp_path)
    write(audit.output/'adapter_validation.json',{'synthetic':'passed'})
    write(audit.output/'games_admission.json',{'validation_reused':True,'outcomes_closed':True,
        'adapter_validation':a.artifact(audit.output/'adapter_validation.json'),'remaining_seconds':40000})
    monkeypatch.setattr(a.old.Audit,'validation',lambda self:seen.append(self.base))
    audit.validation()
    assert seen==[audit.source] and audit.base==audit.output


@pytest.mark.parametrize('field,value',[('before_sha256','b'*64),('observer_sha256','c'*64),
    ('original_primary_complete',True),('outcomes_closed',False),('preserved_original_request',[])])
def test_retry_authorization_receipt_cannot_rebind_prefix_or_drop_partial(field,value):
    p=plan();p['preserved_interrupted_request']=[{'synthetic':'original'}]
    snapshot={'synthetic':'state'};observer={'synthetic':'view'}
    receipt={'time':'2026-10-22T12:00:00+00:00','amendment':p['amendment'],
        'before_sha256':a.digest(snapshot),'observer_sha256':a.digest(observer),
        'preserved_original_request':p['preserved_interrupted_request'],
        'original_primary_complete':False,'outcomes_closed':True}
    a.check_retry_receipt(receipt,p,snapshot,observer)
    receipt[field]=value
    with pytest.raises(ValueError):a.check_retry_receipt(receipt,p,snapshot,observer)


@pytest.mark.parametrize('which',['original_wall','original_mono','recovery_wall','recovery_mono'])
def test_game_admission_keeps_original18h_and_recovery6h_reserves(which):
    source={'epoch_seconds':0.,'monotonic_start':100.}
    launch={'epoch_seconds':180000.,'monotonic_start':180100.}
    value={'name':'games/g059/t0043','finished_epoch_seconds':181000.,'finished_monotonic':181100.}
    a.check_recovery_gate_reserves(value,source,launch)
    if which=='original_wall':value['finished_epoch_seconds']=259200-64800-179
    elif which=='original_mono':value['finished_monotonic']=100+259200-64800-179
    elif which=='recovery_wall':value['finished_epoch_seconds']=180000+43200-21600-179
    else:value['finished_monotonic']=180100+43200-21600-179
    with pytest.raises(ValueError,match='reserves'):a.check_recovery_gate_reserves(value,source,launch)


def fake_receipts(tmp_path,monkeypatch,*,name='games/g059/t0043',player='local',audit_request=False):
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a.old,'ROOT',tmp_path)
    audit=a.Audit(tmp_path/'new');audit.source=(tmp_path/'source').resolve()
    origin=audit.expected_origin(name,audit=audit_request)
    folder=origin/('audits' if audit_request else 'workers')/name
    module=(('tools.splendorust_120_qualification_20261018.worker' if player=='opponent' else
             'tools.matched_strength_20261020.worker') if origin==audit.source else
            a.OPPONENT_MODULE if player=='opponent' else a.RECOVERY_MODULE)
    init={'command':['python','-B','-u','-m',module],'identity':{'pid':123,'created':20.}}
    clock={'status':'ok','accepted':True,'deadline_seconds':600 if audit_request else 120,
           'elapsed_seconds':2.,'receipt_elapsed_seconds':1.8}
    write(folder/'initialization.json',init);write(folder/'request-00000001.json',{**clock,'result':{'synthetic':True}})
    gate_name='audits/'+name if audit_request else name
    gate=origin/'memory_admissions'/(gate_name+'.json');write(gate,{'synthetic':True})
    row={'initialization':{},'initialization_artifact':a.artifact(folder/'initialization.json'),
        'receipt':a.artifact(folder/'request-00000001.json'),'clock':clock,'memory_admission':a.artifact(gate)}
    audit.corrected_runtime_sources={'synthetic.py':'a'*64}
    if origin==audit.output:
        write(folder/'runtime_sources.json',{'schema':'corrected_runtime_sources_v1','ruleset_id':a.old.RULESET,
            'catalogue_sha256':a.old.CATALOGUE,'sources':audit.corrected_runtime_sources})
        row['runtime_sources']=a.artifact(folder/'runtime_sources.json')
    audit.frozen={};audit.cpu=0;seen=[]
    def initialization(value,saved,freeze,expected_module):
        assert value['command'][-1]==expected_module
        seen.append(expected_module)
    monkeypatch.setattr(a.old,'check_initialization',initialization)
    audit.memory_gate=lambda descriptor,name:10.
    audit.check_clock=lambda *args:None
    audit.process_identity=lambda *args:None
    return audit,row,folder,seen


@pytest.mark.parametrize('name,player,is_audit,module',[
    ('games/g039/t0007','local',False,'tools.matched_strength_20261020.worker'),
    ('games/g059/t0043','local',False,a.RECOVERY_MODULE),
    ('games/g059/t0044','opponent',False,a.OPPONENT_MODULE),
    ('games/g059/t0042','opponent',False,'tools.splendorust_120_qualification_20261018.worker'),
    ('validation/v001','audit',True,'tools.matched_strength_20261020.worker'),
    ('g059-t0043','audit',True,a.RECOVERY_MODULE),
])
def test_receipt_origin_and_new_wrapper_module_are_independent_of_supplied_path(tmp_path,monkeypatch,name,player,is_audit,module):
    audit,row,folder,seen=fake_receipts(tmp_path,monkeypatch,name=name,player=player,audit_request=is_audit)
    value=audit.receipts(row,name,player,audit=is_audit)
    assert value['status']=='ok' and seen==[module]
    assert folder in (audit.audit_files if is_audit else audit.worker_files)


@pytest.mark.parametrize('damage',['wrong_origin','clock_copy','runtime_changed','runtime_missing','duplicate_request','reused_worker'])
def test_new_receipt_rejects_relabelled_clock_runtime_and_process(tmp_path,monkeypatch,damage):
    audit,row,folder,_=fake_receipts(tmp_path,monkeypatch)
    if damage=='wrong_origin':
        other=audit.source/'workers/games/g059/t0043/request-00000001.json'
        write(other,a.read(folder/'request-00000001.json'));row['receipt']=a.artifact(other)
    elif damage=='clock_copy':row['clock']['elapsed_seconds']=3.
    elif damage=='runtime_changed':
        value=a.read(folder/'runtime_sources.json');value['sources']['synthetic.py']='b'*64
        write(folder/'runtime_sources.json',value);row['runtime_sources']=a.artifact(folder/'runtime_sources.json')
    elif damage=='runtime_missing':row.pop('runtime_sources')
    elif damage=='duplicate_request':write(folder/'request-00000002.json',{})
    else:audit.worker_owners[(123,20.)]=audit.source/'workers/games/g059/t0043'
    with pytest.raises((ValueError,KeyError)):audit.receipts(row,'games/g059/t0043','local')


@pytest.mark.parametrize('damage',[None,'reset_count','refund_old_wait','prior_seed','extra_gate','early_analyzer','log'])
def test_combined_memory_inventory_includes_interrupted_gate_and_never_refunds_it(tmp_path,monkeypatch,damage):
    from test_matched_strength_analyzer_20261020 import memory_receipt
    import psutil
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a.old,'ROOT',tmp_path)
    audit=a.Audit(tmp_path/'new');audit.source=(tmp_path/'old').resolve();audit.config=deepcopy(a.old.CONFIG)
    audit.source_launch={'epoch_seconds':90.,'monotonic_start':190.}
    audit.launch={'epoch_seconds':490.,'monotonic_start':590.}
    sequence=[(audit.source,'initial'),(audit.source,'games/g059/t0043'),
              (audit.output,'initial'),(audit.output,'games/g059/t0043'),
              (audit.output,'audits/g059-t0043'),(audit.output,'analysis')]
    before=0.;start=100.;original_paths=[];prior=None
    for index,(base,name) in enumerate(sequence):
        if index==2:start=500.
        value=memory_receipt(name,start=start,before=before,count=index)
        if base==audit.output:
            if name.startswith('games/'):value['reserve_seconds']=21600
            for row in value['readings']:row['remaining_seconds']=40000-row['elapsed_seconds']
        if damage=='reset_count' and index==2:value['prior_admission_count']=0
        if damage=='refund_old_wait' and index==2:
            value['cumulative_before_seconds']=0.;value['cumulative_after_seconds']=value['charged_seconds']
        path=base/'memory_admissions'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        log=path.with_suffix('.jsonl');log.write_text(''.join(json.dumps(row)+'\n' for row in value['readings']))
        value['readings_log']={'path':log.relative_to(base).as_posix(),'sha256':a.file_hash(log),'bytes':log.stat().st_size}
        write(path,value)
        if base==audit.source:original_paths.append(path)
        if name!='initial' and name!='analysis':audit.memory_gate(a.artifact(path),name)
        before+=value['charged_seconds'];start=value['finished_epoch_seconds']+1
        if index==1:
            prior={'schema':'matched_strength_prior_memory_v1','admission_count':2,'charged_seconds':before,
                   'inputs':[a.artifact(p) for p in sorted(original_paths)]}
    if damage=='prior_seed':prior['charged_seconds']-=1
    write(audit.output/'prior_memory_state.json',prior)
    if damage=='extra_gate':write(audit.output/'memory_admissions/extra.json',{})
    if damage=='log':(audit.output/'memory_admissions/analysis.jsonl').write_text('{}\n')
    monkeypatch.setattr(psutil,'Process',lambda:SimpleNamespace(create_time=lambda:value['finished_epoch_seconds']+(-1 if damage=='early_analyzer' else 1)))
    if damage is None:
        audit.memory_inventory()
        assert audit.memory_summary['admissions']==6 and audit.memory_summary['charged_seconds']>=240
        assert audit.memory_summary['original_charges_retained'] is True
    else:
        with pytest.raises(ValueError):audit.memory_inventory()


def test_combined_verification_failure_prevents_statistical_opening(tmp_path,monkeypatch):
    from tools.matched_strength_20261020 import statistics
    audit=a.Audit(tmp_path);audit.schedule=[{'game':i} for i in range(60)]
    audit.freeze=lambda:None;audit.validation=lambda:None;audit.game=lambda spec:None
    def failed():raise ValueError('combined replay mismatch')
    audit.manifest=failed
    monkeypatch.setattr(statistics,'statistics',lambda *args,**kwargs:pytest.fail('Premature statistical opening'))
    with pytest.raises(ValueError,match='replay mismatch'):audit.run()


@pytest.mark.parametrize('damage',[None,'ruleset','catalogue','source','missing'])
def test_supervisor_runtime_sources_require_exact_startup_receipt(damage):
    sources={'synthetic_rules.py':'a'*64,'synthetic_oracle.py':'b'*64}
    value=a.runtime_source_receipt(dict(sources))
    if damage=='ruleset':value['ruleset_id']='historical'
    elif damage=='catalogue':value['catalogue_sha256']='0'*64
    elif damage=='source':value['sources']['synthetic_rules.py']='c'*64
    elif damage=='missing':value['sources'].pop('synthetic_oracle.py')
    if damage is None:a.check_supervisor_runtime_sources(value,sources)
    else:
        with pytest.raises(ValueError,match='startup source'):a.check_supervisor_runtime_sources(value,sources)


@pytest.mark.parametrize('damage',[None,'job','affinity','executable','command','pid','creation','owner','freeze','retry'])
def test_job_isolated_launcher_and_runner_are_bound_to_launch(tmp_path,damage):
    executable=str(tmp_path/'python.exe');sha='a'*64
    frozen={'executable':executable,'topology':{'allowed_logical_processors':[0,1,2]}}
    owner={'pid':100,'created':1000.,'executable':executable,'affinity':[0,1,2],
        'in_any_windows_job':False,'command':[executable,'-B',str(a.ROOT/'tools/matched_strength_recovery_20261022/launch.py'),'--freeze-sha256',sha]}
    child={**deepcopy(owner),'pid':101,'created':1001.,'command':[executable,'-B','-m','tools.matched_strength_recovery_20261022.runner','run','--freeze-sha256',sha]}
    launcher={'identity':owner,'freeze_sha256':sha,'automatic_retries':0,'detachment_method':'WMI_breakaway_no_window'}
    process={'identity':child,'supervisor_identity':deepcopy(owner),'freeze_sha256':sha,'automatic_retries':0}
    launch={'pid':101,'identity':{'pid':101,'created':1001.}}
    if damage=='job':child['in_any_windows_job']=True
    elif damage=='affinity':child['affinity']=[2]
    elif damage=='executable':child['executable']=str(tmp_path/'other.exe')
    elif damage=='command':child['command'][-1]='b'*64
    elif damage=='pid':launch['pid']=102
    elif damage=='creation':launch['identity']['created']+=1
    elif damage=='owner':process['supervisor_identity']['created']+=1
    elif damage=='freeze':launcher['freeze_sha256']='b'*64
    elif damage=='retry':process['automatic_retries']=1
    if damage is None:a.check_launcher_receipts(launcher,process,launch,frozen,sha)
    else:
        with pytest.raises(ValueError):a.check_launcher_receipts(launcher,process,launch,frozen,sha)

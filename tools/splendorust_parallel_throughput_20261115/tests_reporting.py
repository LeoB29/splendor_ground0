"""Synthetic receipt integration; no games, native workers or learned models."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import itertools,time
import pytest
from . import common as c,acceptance,report,execution,owner

@pytest.fixture
def evidence(tmp_path,monkeypatch):
    from .collection import common as cc,launch
    from .collection import audit as ca,identity
    from .live import audit as la
    monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'PREP',tmp_path/'prep')
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json');monkeypatch.setattr(cc,'OUT',c.OUT/'collection')
    sources={'fixture':'corrected'}
    monkeypatch.setattr('tools.future_training_rules_20261021.verify_runtime_sources',lambda:sources)
    monkeypatch.setattr(c,'exact_alive',lambda ident:False)
    monkeypatch.setattr(la,'verify_lane',lambda row,*args:row['synthetic_audit'])
    monkeypatch.setattr(la,'compare_arms',lambda *args:dict(status='passed',setups=16))
    monkeypatch.setattr(identity,'compare_cells',lambda *args,**kwargs:dict(status='passed'))
    monkeypatch.setattr(ca,'audit_cell',lambda *args,**kwargs:dict(status='passed'))
    phases=[];phase_items=[]
    for i,(arm,seconds) in enumerate((('A1',100.),('A4',40.),('A4',40.),('A1',100.))):
        audit=dict(timings=[1.,2.,3.],decisions=8)
        phase=dict(arm=arm,elapsed_seconds=seconds,games=[dict(game={'synthetic':j}) for j in range(8)],
                   lanes=[dict(synthetic_audit=audit)],audits=[audit])
        phases.append(dict(result=phase));phase_items.append(c.save(c.OUT/'live'/f'p{i}.json',phase))
    identity_item=c.save(c.OUT/'live/identity.json',dict(elapsed_seconds=20.,proof={'status':'passed'}))
    live_metrics=execution.live_metrics(phases,20.)
    live_item=c.save(c.OUT/'live/result.json',dict(status='passed',metrics=live_metrics,phases=phase_items,
                                                   identity={'matched':identity_item}))
    records={}
    cells={}
    times={('control','X'):(0.,120.,0.),('candidate','X'):(140.,210.,10.),
           ('candidate','Y'):(180.,260.,0.),('control','Y'):(280.,400.,10.)}
    for (arm,cell),(start,end,shared) in times.items():
        folder=cc.OUT/'cells'/f'{arm}_{cell}'
        started=dict(started_perf_counter=start);checked=dict(verified_perf_counter=end,shared_identity_seconds=shared)
        cc.save(folder/'started.json',started);cc.save(folder/'verified.json',checked)
        cc.save(folder/'cohort.json',dict(plies=16,eligible_roots=15))
        for index,duration in enumerate((1.,2.,4.)):
            cc.save(folder/'games/g0000/clocks'/f'c{index}.json',dict(elapsed_seconds=duration))
        records[(arm,cell)]=dict(folder=folder,started=started,checked=checked)
        cells[cell]={'specifications':[]}
    collected=launch.measure(records)
    collection_item=c.save(cc.OUT/'result.json',dict(status='passed',metrics=collected,
        qualifies=collected['candidate']['verified_games_per_hour']/collected['control']['verified_games_per_hour']>=1.3))
    ident=dict(pid=9999999,created=123.)
    value=c.save(c.OUT/'prior/worker/result.json',dict(status='passed',identity=ident))
    previous=c.save(c.OUT/'prior/execution.json',dict(status='passed',exit_code=0,error=None,cleanup_errors=[],
        retries=0,result=value,identity=ident,observed_owned_identities=[]))
    plan=dict(inputs=[],corrected_source_receipt=sources,collection={'cells':cells},
              topology={'validation':{'logical_processor_count':20}})
    c.save(c.PLAN,plan)
    request=dict(mode='acceptance',live=live_item,collection=collection_item,executions=[previous])
    resources=dict(arm_cpu_seconds={'A1':100.,'A4':200.,'B-control':300.,'B-candidate':400.},
                   arm_peak_rss={'A1':1000,'A4':4000,'B-control':2000,'B-candidate':5000})
    return SimpleNamespace(plan=plan,request=request,resources=resources,records=records,
                           live=live_metrics,collection=collected,previous=previous)

def accepted(evidence,monkeypatch):
    # Every separately timed final operation consumes exactly one synthetic second.
    monkeypatch.setattr(acceptance.time,'monotonic',lambda:next(clock))
    clock=itertools.count(1000)
    return acceptance.run(c.OUT/'acceptance/worker',evidence.plan,{},lambda:None,evidence.request)

def test_acceptance_integrates_real_raw_metric_schemas_and_charges_every_check(evidence,monkeypatch):
    result=accepted(evidence,monkeypatch)
    assert result['status']=='passed' and result['models_run']==result['games_run']==0
    assert result['reconstructed_live']==evidence.live
    assert result['reconstructed_collection']==evidence.collection
    assert result['final_verification_seconds_by_arm']=={'A1':2.5,'A4':2.5,'B-control':3.,'B-candidate':3.}
    assert result['reconstructed_live']['A1']['response_seconds']['median']==2.
    assert result['reconstructed_collection']['control']['complete_response_seconds']['p50']==2.

def test_final_verification_is_charged_and_shared_overhead_cannot_help_candidate(evidence,monkeypatch):
    result=accepted(evidence,monkeypatch)
    final=report.final_metrics(result,dict(monotonic_start=100.,monotonic_end=131.),evidence.resources,20)
    rows=final['arms']
    assert sum(r['final_verification_seconds_charged'] for r in rows.values())==31.
    for arm in rows:
        source=evidence.live[arm] if arm.startswith('A') else evidence.collection[arm[2:]]
        assert rows[arm]['verified_seconds']==source['verified_seconds']+result['final_verification_seconds_by_arm'][arm]+5.
        assert rows[arm]['verified_games_per_hour']<source.get('games_per_hour',source.get('verified_games_per_hour'))
    assert final['live_speedup']<evidence.live['speedup']
    original_b=evidence.collection['candidate']['verified_games_per_hour']/evidence.collection['control']['verified_games_per_hour']
    assert final['collection_speedup']<original_b
    assert rows['B-candidate']['cpu_seconds_approximate']==400.
    assert rows['B-candidate']['peak_rss_bytes_sampled']==5000

def test_final_audit_can_make_preliminary_pass_a_valid_negative(evidence,monkeypatch):
    result=accepted(evidence,monkeypatch)
    assert evidence.live['adopt']
    final=report.final_metrics(result,dict(monotonic_start=0.,monotonic_end=4000.),evidence.resources,20)
    assert final['live_adopt'] is False and final['collection_adopt'] is False
    assert final['all_audits_passed'] and final['all_identity_checks_exact']
    assert final['engineering_only'] and final['strength_claim'] is False

def test_final_verifier_cpu_and_rss_are_included_without_losing_prior_jobs(evidence,monkeypatch):
    result=accepted(evidence,monkeypatch)
    final=report.final_metrics(result,dict(monotonic_start=0.,monotonic_end=31.,
        cpu_seconds=62.,peak_tree_rss_bytes=6000),evidence.resources,20)
    rows=final['arms']
    assert sum(row['cpu_seconds_approximate'] for row in rows.values())==1062.
    for arm,row in rows.items():
        assert row['cpu_seconds_approximate']==evidence.resources['arm_cpu_seconds'][arm]+2*row['final_verification_seconds_charged']
        assert row['peak_rss_bytes_sampled']==6000

@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_invalid_final_charge_cannot_inflate_candidate_speed(evidence,monkeypatch,value):
    result=accepted(evidence,monkeypatch);result['final_verification_seconds_by_arm']['A4']=value
    with pytest.raises((ValueError,AssertionError)):
        report.final_metrics(result,dict(monotonic_start=0.,monotonic_end=31.),evidence.resources,20)

@pytest.mark.parametrize('start,end',[(10.,10.),(10.,9.),(0.,5.)])
def test_invalid_worker_duration_or_overallocated_charge_rejected(evidence,monkeypatch,start,end):
    result=accepted(evidence,monkeypatch)
    with pytest.raises((ValueError,AssertionError)):
        report.final_metrics(result,dict(monotonic_start=start,monotonic_end=end),evidence.resources,20)

@pytest.mark.parametrize('mutation',['raw_clock','cohort','actual_identity','live_audit'])
def test_final_acceptance_reconstructs_instead_of_trusting_summaries(evidence,monkeypatch,mutation):
    from .collection import common as cc
    if mutation=='raw_clock':
        path=cc.OUT/'cells/candidate_X/games/g0000/clocks/c0.json'
        path.write_text(c.canonical(dict(elapsed_seconds=20.)),encoding='utf-8')
    elif mutation=='cohort':
        path=cc.OUT/'cells/control_Y/cohort.json';path.write_text(c.canonical(dict(plies=17,eligible_roots=15)),encoding='utf-8')
    elif mutation=='actual_identity':
        monkeypatch.setattr(c,'exact_alive',lambda ident:True)
    else:
        from .live import audit
        monkeypatch.setattr(audit,'verify_lane',lambda *args:dict(timings=[999.],decisions=0))
    with pytest.raises(ValueError):accepted(evidence,monkeypatch)

def test_write_supports_live_median_and_collection_p50_after_actual_exit(evidence,monkeypatch):
    result=accepted(evidence,monkeypatch)
    value=c.save(c.OUT/'acceptance/worker/result.json',dict(status='passed',payload=result))
    receipt=c.save(c.OUT/'acceptance/execution.json',dict(monotonic_start=0.,monotonic_end=31.,result=value))
    c.save(c.OUT/'complete.json',dict(payload=dict(independent_acceptance=receipt,
        live=evidence.request['live'],collection=evidence.request['collection'])))
    c.save(c.OUT/'owner_final.json',evidence.resources)
    c.save(c.OUT/'observed_owner_exit.json',dict(status='passed',actual_root_exit=0,cleanup_errors=[]))
    item=report.write();saved=c.read(c.resolve(item))
    assert saved['status']=='verified_engineering_result'
    text=(c.OUT/'results.md').read_text(encoding='utf-8')
    assert all(arm in text for arm in ('A1','A4','B-control','B-candidate'))
    assert (c.OUT/'throughput.png').stat().st_size>0

def test_cpu_totals_are_cumulative_and_overlapped_rss_is_summed(evidence,monkeypatch):
    begun=time.monotonic();wall=time.time()
    claim=c.OUT/'claim.json';c.save(claim,dict(monotonic_origin=begun,unix_origin=wall))
    plan=dict(limits=dict(process_rss=10000,aggregate_rss=30000,disk_floor=0,evidence_bytes=10000000))
    jobs=owner.Jobs(plan,c.PLAN,claim);jobs.last_disk=float('inf')
    paths=[c.OUT/'live/phase0/lane0/worker',c.OUT/'collection/cells/candidate_X/replays/worker',
           c.OUT/'collection/cells/candidate_Y/players/p0/worker',c.OUT/'collection/cells/candidate_Y/players/p1/worker']
    samples={999:[0.,10],101:[10.,100],201:[20.,200],202:[30.,300],203:[0.,200]}
    class Process:
        def __init__(self,pid=999):self.pid=pid
        def children(self,recursive=True):return []
        def create_time(self):return float(self.pid)
        def memory_info(self):return SimpleNamespace(rss=samples[self.pid][1])
        def cpu_times(self):return SimpleNamespace(user=samples[self.pid][0],system=0.)
    monkeypatch.setattr(owner.psutil,'Process',Process);monkeypatch.setattr(c,'exact_alive',lambda identity:True)
    jobs.running={str(p):dict(pid=pid,created=float(pid)) for p,pid in zip(paths[:3],(101,201,202))}
    jobs.guard();assert jobs.arm_peaks['B-candidate']==500
    samples.update({101:[12.,110],201:[25.,50],202:[37.,400]});jobs.guard()
    assert jobs.arm_peaks['B-candidate']==500 and sum(jobs.cpu.values())==74.
    del jobs.running[str(paths[1])];jobs.running[str(paths[3])]=dict(pid=203,created=203.)
    samples[202][0]=38.;samples[203][0]=5.;jobs.guard()
    assert sum(jobs.cpu.values())==80.
    assert jobs.arm_peaks['B-candidate']==600
    candidate_cpu=sum(sum(values.values()) for path,values in jobs.job_cpu.items() if owner.metric_arm(Path(path))=='B-candidate')
    assert candidate_cpu==68.  # Includes the exited X replay, without recounting earlier samples.

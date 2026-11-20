"""Synthetic guard/IPC ownership tests; no models, games, or production evidence."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools import operational_final_validation_repair_20261020 as m


class Process:
    def __init__(self, pid, created, exe, *, parent=0, name='python.exe', rss=100, affinity=None):
        self.pid, self.created, self.executable = pid, created, str(exe)
        self.parent_pid, self.label, self.rss = parent, name, rss
        self.affinity = [2] if affinity is None else affinity
        self.descendants, self.live, self.killed = [], True, False
    def create_time(self): return self.created
    def exe(self): return self.executable
    def name(self): return self.label
    def ppid(self): return self.parent_pid
    def cpu_affinity(self, value=None):
        if value is not None: self.affinity = value
        return self.affinity
    def memory_info(self): return SimpleNamespace(rss=self.rss)
    def children(self, recursive=True): return [p for p in self.descendants if p.live]
    def kill(self): self.killed, self.live = True, False
    def wait(self, timeout=5): self.live = False; return 0


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root=tmp_path; source_base=root/'completed'; base=root/'repair'
    source_base.mkdir();base.mkdir()
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'SOURCE_BASE',source_base);monkeypatch.setattr(m,'BASE',base)
    helperfile=root/'System32'/'conhost.exe';helperfile.parent.mkdir();helperfile.write_bytes(b'known OS helper')
    python=root/'python.exe';python.write_bytes(b'python')
    parent=Process(10001,100,python)
    helper=Process(10002,100.01,helperfile,parent=parent.pid,name='conhost.exe',rss=25)
    parent.descendants=[helper];registry={parent.pid:parent,helper.pid:helper}
    def process(pid):
        value=registry.get(pid)
        if value is None or not value.live: raise psutil.NoSuchProcess(pid)
        return value
    monkeypatch.setattr(m.psutil,'Process',process)
    plan={'analyzer_command':[str(python),'-B','adapter.py','--repair-freeze-sha256','f'*64],
        'console_helper':{'executable':str(helperfile),'name':'conhost.exe','sha256':m.sha(helperfile),'maximum_distinct_helpers':1},
        'runner_identity':{'pid':20001,'created':1},'epoch_seconds':m.time.time()-10,
        'monotonic_start':m.time.monotonic()-10,
        'analyzer':{'path':'adapter.py','sha256':'a'*64,'bytes':1},
        'repair_freeze':{'path':'repair/repair_freeze.json','sha256':'f'*64,'bytes':1}}
    return SimpleNamespace(root=root,base=base,source_base=source_base,parent=parent,helper=helper,
        helperfile=helperfile,registry=registry,plan=plan)


def test_exact_helper_is_included_in_combined_memory_and_identity(setup):
    s=setup;tree=m.ProcessTree(m.identity(s.parent),s.plan)
    assert tree.sample()==125
    observed=list(tree.observed.values())[0]
    assert observed['owned_parent']=={'pid':10001,'created':100}
    assert observed['parent_pid']==10001 and observed['allowed_console_helper'] is True
    assert observed['executable_sha256']==m.sha(s.helperfile)
    assert tree.peak_rss==125


@pytest.mark.parametrize('field,value,label',[
    ('parent_pid',999,'unexpected_analyzer_descendant'),('label','not-conhost.exe','unexpected_analyzer_descendant'),
    ('affinity',[0],'unexpected_analyzer_descendant'),('executable','C:/unexpected.exe','unexpected_analyzer_descendant')])
def test_helper_requires_exact_parent_name_path_and_affinity(setup,field,value,label):
    s=setup;setattr(s.helper,field,value);tree=m.ProcessTree(m.identity(s.parent),s.plan)
    with pytest.raises(m.Hold,match=label):tree.sample()
    assert (s.helper.pid,s.helper.created) in tree.observed


def test_helper_hash_mismatch_is_rejected(setup):
    s=setup;s.helperfile.write_bytes(b'replaced binary');tree=m.ProcessTree(m.identity(s.parent),s.plan)
    with pytest.raises(m.Hold,match='console_helper_hash'):tree.sample()


def test_replacement_helper_is_rejected_even_when_first_is_closed(setup):
    s=setup;tree=m.ProcessTree(m.identity(s.parent),s.plan);tree.sample();s.helper.live=False
    other=Process(10003,101,s.helperfile,parent=s.parent.pid,name='conhost.exe')
    s.registry[other.pid]=other;s.parent.descendants.append(other)
    with pytest.raises(m.Hold,match='console_helper_replacement'):tree.sample()


def test_descendant_identity_drift_fails(setup):
    s=setup;tree=m.ProcessTree(m.identity(s.parent),s.plan);tree.sample();s.helper.parent_pid=7
    with pytest.raises(m.Hold,match='descendant_identity_changed'):tree.sample()


def test_analyzer_identity_reuse_and_affinity_drift_rejected(setup):
    s=setup;tree=m.ProcessTree(m.identity(s.parent),s.plan);s.parent.created=101
    with pytest.raises(m.Hold,match='analyzer_identity_changed'):tree.sample()
    s.parent.created=100;s.parent.affinity=[0]
    with pytest.raises(m.Hold,match='analyzer_executable_or_affinity'):tree.sample()


def test_combined_rss_includes_console_helper(setup,monkeypatch):
    s=setup;monkeypatch.setattr(m,'RSS_LIMIT',120)
    with pytest.raises(m.Hold,match='combined_analyzer_tree_rss'):m.ProcessTree(m.identity(s.parent),s.plan).sample()


def test_cleanup_closes_owned_helper_after_normal_parent_exit(setup):
    s=setup;tree=m.ProcessTree(m.identity(s.parent),s.plan);tree.sample();s.parent.live=False
    child=SimpleNamespace(poll=lambda:0,wait=lambda timeout:0)
    result=tree.cleanup(child)
    assert result['parent_closed'] and result['descendants_closed'] and not result['errors']
    assert s.helper.killed and result['terminated_owned_identities']==[{'pid':10002,'created':100.01}]


def test_cleanup_does_not_kill_reused_pid(setup):
    s=setup;tree=m.ProcessTree(m.identity(s.parent),s.plan);tree.sample();s.parent.live=False
    replacement=Process(s.helper.pid,900,s.helperfile);s.registry[s.helper.pid]=replacement
    result=tree.cleanup(SimpleNamespace(poll=lambda:0,wait=lambda timeout:0))
    assert result['descendants_closed'] and not replacement.killed


def test_cleanup_cannot_kill_unrelated_runner_identity(setup):
    s=setup;s.helper.pid=s.plan['runner_identity']['pid']
    tree=m.ProcessTree(m.identity(s.parent),s.plan)
    with pytest.raises(m.Hold,match='invalid_descendant_ownership'):tree.sample()
    assert tree.observed=={}


def test_both_original_clocks_bound_remaining_time(setup,monkeypatch):
    plan=setup.plan;plan['epoch_seconds']=1000;plan['monotonic_start']=2000
    monkeypatch.setattr(m.time,'time',lambda:1100);monkeypatch.setattr(m.time,'monotonic',lambda:2200)
    assert m.remaining(plan)==43000
    monkeypatch.setattr(m.time,'time',lambda:1500)
    assert m.remaining(plan)==42700


def test_combined_evidence_and_final_receipt_reserve(setup,monkeypatch):
    s=setup;monkeypatch.setattr(m,'MAX_BYTES',1000);monkeypatch.setattr(m,'EVIDENCE_RESERVE',100)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.MIN_FREE))
    (s.source_base/'evidence').write_bytes(b'x'*600);(s.base/'new').write_bytes(b'y'*310)
    with pytest.raises(m.Hold,match='combined_evidence_limit'):m.resources(s.plan)
    assert m.resources(s.plan,final=True)['combined_evidence_bytes']==910
    (s.base/'more').write_bytes(b'z'*90)
    with pytest.raises(m.Hold,match='combined_evidence_limit'):m.resources(s.plan,final=True)


def test_final_reserve_and_disk_thresholds(setup,monkeypatch):
    s=setup;monkeypatch.setattr(m,'remaining',lambda _:600)
    with pytest.raises(m.Hold,match='original_deadline_or_reserve'):m.resources(s.plan)
    monkeypatch.setattr(m,'remaining',lambda _:601)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.MIN_FREE-1))
    with pytest.raises(m.Hold,match='disk_reserve'):m.resources(s.plan)


def prepare_dispatch(s,monkeypatch):
    plan_path=s.base/'launcher_plan.json';m.save(plan_path,s.plan);m.save(s.base/'launch.json',{'test':True})
    launcher=m.Launcher(plan_path,'test','source');monkeypatch.setattr(launcher,'authenticate',lambda **kw:s.plan)
    calls=[]
    def guard(*args,**kwargs):calls.append('resource');return {'remaining_seconds':10000}
    monkeypatch.setattr(m,'resources',guard)
    def admission(source,base,launch,prior_state,*,resources):
        calls.append('admission');resources();value={'status':'passed'}
        m.save(base/'memory_admission.json',value);(base/'memory_admission.jsonl').write_text('{}\n')
        return value
    monkeypatch.setattr(m,'memory_module',lambda:SimpleNamespace(wait_for_memory=admission))
    m.save(s.source_base/'launch.json',{'test':True});s.plan['prior_memory_state']={'count':458}
    class Child:
        def __init__(self,*args,**kwargs):
            calls.append('spawn');self.pid=s.parent.pid;self.returncode=None
            assert kwargs['stdin'] is m.subprocess.DEVNULL and kwargs['cwd']==s.root
            assert kwargs['env']['PYTHONDONTWRITEBYTECODE']=='1'
            m.save(s.base/'independent_analysis.json',{'status':'passed','outcomes_closed':True,
                'strength_analysis_games':0,'freeze_sha256':m.ORIGINAL_FREEZE_SHA256})
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=0;s.parent.live=False;return 0
        def kill(self):self.returncode=-1;s.parent.live=False
    monkeypatch.setattr(m.subprocess,'Popen',Child)
    return launcher,calls


def test_single_dispatch_fresh_admission_and_closure_receipt(setup,monkeypatch):
    s=setup;launcher,calls=prepare_dispatch(s,monkeypatch)
    result=launcher.execute(s.plan)
    assert calls.index('admission')<calls.index('spawn') and calls.count('spawn')==1
    assert result['exit_code']==0 and result['error'] is None
    assert result['cleanup']['parent_closed'] and result['cleanup']['descendants_closed']
    assert result['observed_descendants'][0]['allowed_console_helper'] is True
    assert result['peak_sampled_combined_rss_bytes']==125
    assert result['memory_admission']==m.artifact(s.base/'memory_admission.json')
    assert result['prior_hold_preserved'] and not result['games_or_replays_rerun']
    assert read_json(s.base/'execution.json')==result


def read_json(path):return json.loads(path.read_text())


def test_forbidden_child_is_closed_and_failed_execution_retained(setup,monkeypatch):
    s=setup;launcher,calls=prepare_dispatch(s,monkeypatch);s.helper.label='unknown.exe'
    with pytest.raises(m.Hold,match='final_validation_failed'):launcher.execute(s.plan)
    execution=read_json(s.base/'execution.json')
    assert execution['error']=='Hold' and execution['cleanup']['parent_closed'] and execution['cleanup']['descendants_closed']
    assert s.helper.killed and (s.base/'failure.json').exists() and calls.count('spawn')==1


def test_log_limit_closes_child_without_accepting_output(setup,monkeypatch):
    s=setup;launcher,calls=prepare_dispatch(s,monkeypatch);monkeypatch.setattr(m,'MAX_LOG',-1)
    with pytest.raises(m.Hold,match='final_validation_failed'):launcher.execute(s.plan)
    execution=read_json(s.base/'execution.json')
    assert execution['error']=='Hold' and execution['cleanup']['parent_closed']
    assert calls.count('spawn')==1


def test_claim_is_exclusive_and_never_overwrites_existing_attempt(setup,monkeypatch):
    s=setup;plan_path=s.base/'launcher_plan.json';m.save(plan_path,s.plan)
    launcher=m.Launcher(plan_path,'hash','hash');monkeypatch.setattr(launcher,'authenticate',lambda:s.plan)
    current=Process(70001,1,'python');old=m.psutil.Process
    monkeypatch.setattr(m.psutil,'Process',lambda pid=None:current if pid is None else old(pid))
    (s.base/'execution.json').write_bytes(b'preserve')
    with pytest.raises(m.Hold,match='repair_already_attempted'):launcher.run()
    assert (s.base/'execution.json').read_bytes()==b'preserve' and not launcher.claimed


def test_wrong_plan_path_or_hash_rejected_before_build(setup):
    s=setup;path=s.base/'launcher_plan.json';m.save(path,s.plan)
    with pytest.raises(m.Hold,match='launcher_plan_identity'):m.Launcher(path,'wrong','source').authenticate()
    other=s.base/'wrong_plan.json';m.save(other,s.plan)
    with pytest.raises(m.Hold,match='launcher_plan_identity'):m.Launcher(other,m.sha(other),'source').authenticate()


def test_artifacts_cannot_escape_workspace_or_change(setup):
    s=setup;path=s.root/'item';path.write_bytes(b'original');value=m.artifact(path)
    path.write_bytes(b'changed')
    with pytest.raises(m.Hold,match='artifact_changed'):m.bound(value)
    with pytest.raises(m.Hold,match='artifact_path'):m.bound({'path':'../outside','sha256':'a'*64,'bytes':1})


def accepted_dispatch(s,monkeypatch):
    launcher,calls=prepare_dispatch(s,monkeypatch);execution=launcher.execute(s.plan)
    s.plan['source']={'path':'launcher.py','sha256':'s'*64,'bytes':1}
    # All edits here are isolated synthetic receipt construction.
    (s.base/'repair_freeze.json').write_text('{}')
    s.plan['repair_freeze']=m.artifact(s.base/'repair_freeze.json')
    (s.base/'launcher_plan.json').write_text(json.dumps(s.plan))
    plan_sha=m.sha(s.base/'launcher_plan.json')
    launch={'schema':'final_validation_repair_launch_v1','source':s.plan['source'],
        'plan':m.artifact(s.base/'launcher_plan.json'),'plan_sha256':plan_sha,'expected_source_sha256':'s'*64,
        'cpu':2,'original_epoch_seconds':s.plan['epoch_seconds'],'original_monotonic_start':s.plan['monotonic_start'],
        'maximum_seconds':m.CAP,'final_reserve_seconds':m.RESERVE,'outcomes_closed':True,
        'prior_hold_preserved':True,'automatic_retries':0,'identity':{'pid':30001,'created':1}}
    (s.base/'launch.json').write_text(json.dumps(launch))
    admission={'status':'passed','finished_epoch_seconds':99}
    (s.base/'memory_admission.json').write_text(json.dumps(admission))
    execution.update(plan=m.artifact(s.base/'launcher_plan.json'),launch=m.artifact(s.base/'launch.json'),
        repair_freeze=s.plan['repair_freeze'],memory_admission=m.artifact(s.base/'memory_admission.json'))
    (s.base/'execution.json').write_text(json.dumps(execution))
    monkeypatch.setattr(m.Launcher,'authenticate',lambda self,**kw:s.plan)
    return execution,launch,(s.plan['repair_freeze']['sha256'],plan_sha,m.sha(s.base/'execution.json'))


def test_finalization_validator_checks_success_and_current_closure(setup,monkeypatch):
    execution,launch,hashes=accepted_dispatch(setup,monkeypatch)
    assert m.validate_execution_for_finalization(*hashes)==execution
    setup.parent.live=True
    with pytest.raises(m.Hold,match='authenticated_dispatch_processes_not_closed'):
        m.validate_execution_for_finalization(*hashes)


@pytest.mark.parametrize('mutation,label',[
    (lambda v:v.update(exit_code=1),'successful_dispatch_binding'),
    (lambda v:v.update(error='Hold'),'successful_dispatch_binding'),
    (lambda v:v['cleanup'].update(descendants_closed=False),'dispatch_cleanup_incomplete'),
    (lambda v:v.update(peak_sampled_combined_rss_bytes=m.RSS_LIMIT+1),'dispatch_rss_bound'),
    (lambda v:v['observed_descendants'][0].update(parent_pid=77),'authenticated_console_helper_not_closed'),
    (lambda v:v['observed_descendants'][0].update(executable_sha256='0'*64),'authenticated_console_helper_not_closed'),
    (lambda v:v['cleanup'].update(terminated_owned_identities=[{'pid':999,'created':1}]),'cleanup_identity_scope'),
    (lambda v:v.update(elapsed_seconds=v['elapsed_seconds']+1),'analysis_and_cleanup_retained_original_final_reserve'),
])
def test_finalization_rejects_corrupt_receipt_even_when_new_digest_given(setup,monkeypatch,mutation,label):
    s=setup;value,_,hashes=accepted_dispatch(s,monkeypatch);mutation(value)
    (s.base/'execution.json').write_text(json.dumps(value))
    with pytest.raises(m.Hold,match=label):
        m.validate_execution_for_finalization(hashes[0],hashes[1],m.sha(s.base/'execution.json'))


def test_finalization_cannot_accept_late_analysis_or_new_hold(setup,monkeypatch):
    s=setup;value,_,hashes=accepted_dispatch(s,monkeypatch)
    value['finished_monotonic']=s.plan['monotonic_start']+m.CAP-m.RESERVE
    (s.base/'execution.json').write_text(json.dumps(value))
    with pytest.raises(m.Hold,match='analysis_and_cleanup_retained_original_final_reserve'):
        m.validate_execution_for_finalization(hashes[0],hashes[1],m.sha(s.base/'execution.json'))
    m.save(s.base/'hold.json',{'status':'preserved_hold'})
    with pytest.raises(m.Hold,match='repair_hold_prevents_acceptance'):
        m.validate_execution_for_finalization(hashes[0],hashes[1],m.sha(s.base/'execution.json'))


def test_finalization_may_use_last_reserve_but_does_not_reset_clock(setup,monkeypatch):
    s=setup;value,_,hashes=accepted_dispatch(s,monkeypatch)
    modes=[]
    def auth(self,*,final=False):modes.append(final);return s.plan
    monkeypatch.setattr(m.Launcher,'authenticate',auth)
    def guard(plan,*,final=False):
        assert final is True
        return {'remaining_seconds':200}
    monkeypatch.setattr(m,'resources',guard)
    assert m.validate_execution_for_finalization(*hashes)==value and modes==[True]


def test_finalization_rejects_output_or_admission_tamper(setup,monkeypatch):
    s=setup;_,_,hashes=accepted_dispatch(s,monkeypatch)
    (s.base/'independent_analysis.json').write_text('{}')
    with pytest.raises(m.Hold,match='artifact_changed'):m.validate_execution_for_finalization(*hashes)


def preparation_fixture(s,monkeypatch):
    for name in ('SOURCE','ANALYZER','MEMORY_SOURCE','FROZEN_ANALYZER'):
        path=s.root/(name.lower()+'.py');path.write_text(name);monkeypatch.setattr(m,name,path)
    monkeypatch.setenv('SystemRoot',str(s.root))
    for name in m.PRESERVED_NAMES:
        path=s.source_base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    def write(name,value): (s.source_base/name).write_text(json.dumps(value))
    write('freeze.json',{'executable':m.sys.executable})
    monkeypatch.setattr(m,'ORIGINAL_FREEZE_SHA256',m.sha(s.source_base/'freeze.json'))
    write('launch.json',{'freeze':m.artifact(s.source_base/'freeze.json'),'maximum_seconds':m.CAP,
        'outcomes_closed':True,'epoch_seconds':s.plan['epoch_seconds'],'monotonic_start':s.plan['monotonic_start']})
    write('completion.json',{'status':'completed_pending_independent_analysis','failure':None,
        'worker_processes_closed':True,'completed_games':18,'outcomes_closed':True,'strength_analysis_games':0})
    runner={'pid':40001,'created':1};write('runner_process.json',{'identity':runner})
    write('runner_execution.json',{'exit_code':0,'runner_identity':runner})
    write('coordinator_process.json',{'identity':{'pid':40002,'created':2}})
    write('coordinator_hold.json',{'status':'preserved_hold'})
    write('independent_analysis_execution_01.json',{'error':'Hold','analyzer_process_closed':True,
        'owned_descendants_closed':True,'output':None,'analyzer_process':{'pid':40003,'created':3},'owned_descendants':[]})
    probe={'exit_code':0,'parent_closed':True,'models_or_games_executed':False,'cpu':2,
        'creationflags':'CREATE_NO_WINDOW','parent':{'pid':41,'created':10},'descendants':[
        {'executable':str(s.helperfile),'name':'conhost.exe','ppid':41,'created':11,
         'closed_after_parent_exit':True,'affinity':[2],'executable_sha256':m.sha(s.helperfile)}]}
    m.save(s.base/'launcher_probe_01.json',probe);m.save(s.base/'source_inventory.json',{'files':[]})
    frozen={'schema':'operational_final_validation_repair_freeze_v1','source_base':'completed','repair_base':'repair',
        'outcomes_closed':True,'larger_data_deferred':True,'maximum_seconds':m.CAP,'final_reserve_seconds':m.RESERVE,
        'original_freeze':m.artifact(s.source_base/'freeze.json'),'original_launch':m.artifact(s.source_base/'launch.json'),
        'frozen_analyzer':m.artifact(m.FROZEN_ANALYZER),'source_inventory':m.artifact(s.base/'source_inventory.json'),
        'inputs':[m.artifact(p) for p in (m.SOURCE,m.ANALYZER,m.MEMORY_SOURCE,s.base/'launcher_probe_01.json',s.base/'source_inventory.json')]}
    m.save(s.base/'repair_freeze.json',frozen)
    monkeypatch.setattr(m,'memory_module',lambda:SimpleNamespace(prior_memory_state=lambda base:{'count':458,'charged_seconds':123,'receipts':[]}))
    monkeypatch.setattr(m,'resources',lambda plan,**kw:{'remaining_seconds':1000})
    return m.sha(s.base/'repair_freeze.json'),m.sha(m.SOURCE)


def test_plan_pins_original_scope_and_probe_and_distinct_dispatch(setup,monkeypatch):
    hashes=preparation_fixture(setup,monkeypatch);plan=m.build_plan(*hashes)
    assert plan['maximum_seconds']==43200 and plan['final_reserve_seconds']==600
    assert plan['epoch_seconds']==setup.plan['epoch_seconds'] and plan['monotonic_start']==setup.plan['monotonic_start']
    assert plan['analysis_attempts']==1 and plan['automatic_retries']==0 and not plan['games_or_replays_rerun']
    assert plan['console_helper']['sha256']==m.sha(setup.helperfile) and plan['prior_memory_state']['count']==458
    assert plan['analyzer_command'][-2:]==['--repair-freeze-sha256',hashes[0]]


def test_plan_rejects_changed_new_source_and_old_closure(setup,monkeypatch):
    hashes=preparation_fixture(setup,monkeypatch)
    m.SOURCE.write_text('changed')
    with pytest.raises(m.Hold,match='launcher_source_identity'):m.build_plan(*hashes)
    m.SOURCE.write_text('SOURCE')
    value=read_json(setup.source_base/'completion.json');value['completed_games']=17
    (setup.source_base/'completion.json').write_text(json.dumps(value))
    with pytest.raises(m.Hold,match='completed_collection_required'):m.build_plan(*hashes)

"""Model-free stage admission, raw-receipt preservation and resource bounds."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.splendorust_component_timing_20261025 import runner as r, common as c


class Envelope:
    def __init__(self): self.stage=None; self.calls=[]
    def begin(self,stage): self.stage=stage; self.calls.append(('begin',stage))
    def finish(self): self.calls.append(('finish',self.stage))


@pytest.mark.parametrize('passed,stages',[(False,['overhead','final']),(True,['overhead','diagnostic','final'])])
def test_overhead_gate_controls_diagnostic_without_game_branch(monkeypatch,passed,stages):
    envelope=Envelope(); saved=[]
    monkeypatch.setattr(r,'run_panel',lambda env,freeze,plan,stage:{'qualified':passed,'stage':stage})
    monkeypatch.setattr(r,'final_checks',lambda *args:None)
    monkeypatch.setattr(c,'save',lambda path,value:saved.append(value))
    result=r.run_stages(envelope,{}, {})
    assert [stage for event,stage in envelope.calls if event=='begin']==stages
    assert ('diagnostic' in result) is passed
    assert saved[-1]['fresh_games']==saved[-1]['strength_games']==0


def test_hard_failure_cannot_run_followup_or_acceptance(monkeypatch):
    calls=[]
    def fail(*args): raise ValueError('timeout record preserved')
    monkeypatch.setattr(r,'run_panel',fail)
    monkeypatch.setattr(r,'final_checks',lambda *args:calls.append('final'))
    with pytest.raises(ValueError,match='timeout'): r.run_stages(Envelope(),{}, {})
    assert not calls


def test_raw_scientific_response_is_not_mutated_when_stripping_profile():
    result={'action':{'move':1},'timing_profile':{'wall':1}}
    before=deepcopy(result)
    assert r.science_result(result,'profiled')=={'action':{'move':1}}
    assert result==before
    with pytest.raises(ValueError): r.science_result(result,'baseline')


def test_profile_sidecar_is_separate_from_clock(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path); monkeypatch.setattr(c,'OUT',tmp_path/'output')
    monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    receipt={'request_id':'request-00000001','status':'ok'}
    child=SimpleNamespace(runtime_mode='profiled',last_parent_profile={'request_id':receipt['request_id'],'spans':[]},
        output_dir=c.OUT/'workers/one')
    profile,path=r.save_profile(child,receipt)
    assert c.read(c.ROOT/path)==profile and receipt=={'request_id':'request-00000001','status':'ok'}
    assert 'parent_profiles' in path


def test_stage_and_global_cutoffs_and_final_evidence_reserve(monkeypatch,tmp_path):
    now=SimpleNamespace(mono=2690.,wall=3690.)
    monkeypatch.setattr(r.time,'monotonic',lambda:now.mono)
    monkeypatch.setattr(r.time,'time',lambda:now.wall)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda root:SimpleNamespace(free=10*c.GiB))
    monkeypatch.setattr(c,'save',lambda *args:None)
    envelope=r.Envelope({'monotonic_origin':0.,'unix_origin':1000.})
    monkeypatch.setattr(envelope,'accounted_bytes',lambda:448*c.MiB-1)
    envelope.begin('diagnostic')
    assert envelope.remaining()==10 and envelope.deadline==2700
    monkeypatch.setattr(envelope,'accounted_bytes',lambda:448*c.MiB)
    with pytest.raises(ValueError,match='evidence'): envelope.guard()
    envelope.begin('final')
    assert envelope.deadline==3590
    now.wall=4600.
    with pytest.raises(ValueError,match='real-clock'): envelope.guard()


def test_constructor_aggregate_counts_unregistered_second_tree():
    envelope=r.Envelope({'monotonic_origin':0.,'unix_origin':0.})
    root=SimpleNamespace(pid=1,create_time=lambda:1.,children=lambda recursive:[],
        memory_info=lambda:SimpleNamespace(rss=c.GiB))
    envelope.resident[(1,1.)]=root
    with pytest.raises(ValueError,match='Combined'):
        envelope.guard_worker_sample(SimpleNamespace(process=SimpleNamespace(pid=2)),{'rss_bytes':c.GiB+1})

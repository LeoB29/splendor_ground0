"""Synthetic clocks and source gates; no actual waiting or model execution."""
import json
from pathlib import Path

import pytest

from tools import operational_validation_memory_20261020 as m


class Clock:
    def __init__(self):
        self.mono,self.epoch=1000.,10000.
        self.sleeps=[]
    def sleep(self,seconds):
        self.sleeps.append(seconds);self.mono+=seconds;self.epoch+=seconds


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def rig(tmp_path,monkeypatch):
    source,base=tmp_path/'source',tmp_path/'repair'
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'SOURCE',source);monkeypatch.setattr(m,'BASE',base)
    monkeypatch.setattr(m,'PRIOR_COUNT',3)
    clock=Clock();monkeypatch.setattr(m.time,'monotonic',lambda:clock.mono)
    monkeypatch.setattr(m.time,'time',lambda:clock.epoch);monkeypatch.setattr(m.time,'sleep',clock.sleep)
    launch={'maximum_seconds':43200,'epoch_seconds':9900.,'monotonic_start':900.}
    write(source/'launch.json',launch)
    for index,name in enumerate(('initial','games/g021/t0011','analysis')):
        write(source/'memory_admissions'/(name+'.json'),{
            'schema':'operational_memory_admission_v2','status':'passed','failure_category':None,'failure_detail':None,
            'name':name,'charged_seconds':2.,'started_epoch_seconds':9000.+index*10,'finished_epoch_seconds':9002.+index*10,
            'started_monotonic':100.+index*10,'finished_monotonic':102.+index*10,
            'cumulative_before_seconds':index*2.,'cumulative_after_seconds':(index+1)*2.,'prior_admission_count':index})
    monkeypatch.setattr(m,'memory',lambda:{'available_physical':m.PHYSICAL_BYTES,'available_page':m.COMMIT_BYTES})
    return {'source':source,'base':base,'clock':clock,'launch':launch,'prior':m.prior_memory_state(source)}


def admission(rig,resources=lambda:None):
    return m.wait_for_memory(rig['source'],rig['base'],rig['launch'],rig['prior'],resources=resources)


def receipt(rig):return m.read(rig['base']/'memory_admission.json')


def test_always_requires_fresh_two_minute_window_and_preserves_sources(rig):
    before={p:p.read_bytes() for p in rig['source'].rglob('*') if p.is_file()}
    result=admission(rig)
    assert result['status']=='passed' and result['charged_seconds']==120
    assert len(result['readings'])==13 and rig['clock'].sleeps==[10]*12
    assert result['prior_admission_count']==3 and result['cumulative_before_seconds']==6
    assert result['cumulative_after_seconds']==126 and result['prior_gate_inputs']==rig['prior']['receipts']
    assert result['supplemental'] is True and result['initial'] is False and result['stability_required'] is True
    assert result['name']=='analysis_resume_02' and result['readings_log']['path']=='memory_admission.jsonl'
    assert result==receipt(rig)
    assert before=={p:p.read_bytes() for p in rig['source'].rglob('*') if p.is_file()}
    with pytest.raises(m.ResourceHold,match='already attempted'):admission(rig)


def test_bad_reading_resets_full_stability_window(rig,monkeypatch):
    readings=iter([True]*6+[False]+[True]*13)
    monkeypatch.setattr(m,'memory',lambda:{'available_physical':m.PHYSICAL_BYTES if next(readings) else 0,
                                          'available_page':m.COMMIT_BYTES})
    result=admission(rig)
    assert result['charged_seconds']==190 and len(result['readings'])==20
    assert result['readings'][6]['consecutive_admissible_samples']==0
    assert result['readings'][-1]['consecutive_admissible_samples']==13


@pytest.mark.parametrize('damage',['missing','extra','changed','wrong_count','wrong_sum','wrong_charge','duplicate_index'])
def test_seeded_source_inventory_is_exact(rig,damage):
    path=rig['source']/'memory_admissions/analysis.json'
    value=m.read(path)
    if damage=='missing':path.rename(path.with_suffix('.retained'))
    elif damage=='extra':write(path.with_name('extra.json'),value)
    elif damage=='wrong_count':rig['prior']['count']+=1
    elif damage=='wrong_sum':rig['prior']['charged_seconds']+=1
    else:
        if damage=='changed':value['extra_field']='tampered'
        elif damage=='wrong_charge':value['charged_seconds']+=1
        else:value['prior_admission_count']=1
        write(path,value)
    with pytest.raises(m.ResourceHold):admission(rig)
    assert not rig['clock'].sleeps and not (rig['base']/'memory_admission.jsonl').exists()


def test_episode_limit_stops_with_held_receipt(rig,monkeypatch):
    monkeypatch.setattr(m,'memory',lambda:{'available_physical':0,'available_page':m.COMMIT_BYTES})
    with pytest.raises(m.ResourceHold,match='allowance'):admission(rig)
    assert receipt(rig)['status']=='preserved_hold' and receipt(rig)['charged_seconds']==7200


def test_cumulative_limit_can_stop_before_episode_limit(rig,monkeypatch):
    monkeypatch.setattr(m,'CUMULATIVE_WAIT',126)
    with pytest.raises(m.ResourceHold,match='allowance'):admission(rig)
    assert receipt(rig)['charged_seconds']==120 and receipt(rig)['cumulative_after_seconds']==126


def test_forward_wall_jump_consumes_allowance(rig,monkeypatch):
    def memory():
        rig['clock'].epoch+=7200
        return {'available_physical':m.PHYSICAL_BYTES,'available_page':m.COMMIT_BYTES}
    monkeypatch.setattr(m,'memory',memory)
    with pytest.raises(m.ResourceHold,match='allowance'):admission(rig)
    assert receipt(rig)['charged_seconds']==7200 and receipt(rig)['elapsed_seconds']==0


def test_backward_wall_clock_never_refunds_wait(rig,monkeypatch):
    def sleep(seconds):
        rig['clock'].mono+=seconds;rig['clock'].epoch-=seconds
    monkeypatch.setattr(m.time,'sleep',sleep)
    result=admission(rig)
    assert result['charged_seconds']==120 and result['finished_epoch_seconds']<result['started_epoch_seconds']


def test_original_deadline_and_final_reserve_remain_enforced(rig):
    rig['clock'].mono=rig['launch']['monotonic_start']+43200-700
    with pytest.raises(m.ResourceHold,match='reserve'):admission(rig)
    assert sum(rig['clock'].sleeps)==40
    assert receipt(rig)['status']=='preserved_hold'


def test_resource_guard_latency_is_not_refunded(rig):
    rig['clock'].mono=rig['launch']['monotonic_start']+43200-680
    def resources():rig['clock'].mono+=3;rig['clock'].epoch+=3
    with pytest.raises(m.ResourceHold,match='reserve'):admission(rig,resources)
    assert sum(rig['clock'].sleeps)<=11


@pytest.mark.parametrize('late_failure',['time','fsync','resources'])
def test_final_flush_and_guard_cannot_create_false_pass(rig,monkeypatch,late_failure):
    def fsync(fd):
        if late_failure=='time':rig['clock'].mono+=7200
        elif late_failure=='fsync':raise OSError('synthetic flush failure')
    monkeypatch.setattr(m.os,'fsync',fsync)
    calls=[0]
    def resources():
        calls[0]+=1
        if late_failure=='resources' and calls[0]==15:raise m.ResourceHold('synthetic disk failure')
    with pytest.raises((m.ResourceHold,OSError)):admission(rig,resources)
    assert receipt(rig)['status']=='preserved_hold' and receipt(rig)['failure_category'] is not None


@pytest.mark.parametrize('bad',[True,-1,1.5])
def test_malformed_memory_fails_closed(rig,monkeypatch,bad):
    monkeypatch.setattr(m,'memory',lambda:{'available_physical':bad,'available_page':m.COMMIT_BYTES})
    with pytest.raises(m.ResourceHold,match='reading'):admission(rig)
    assert receipt(rig)['status']=='preserved_hold' and not receipt(rig)['readings']


def test_wrong_launch_and_path_are_rejected_without_writes(rig):
    rig['launch']['maximum_seconds']=86400
    with pytest.raises(m.ResourceHold,match='launch'):admission(rig)
    rig['launch']['maximum_seconds']=43200
    with pytest.raises(m.ResourceHold,match='paths'):
        m.wait_for_memory(rig['source'],rig['source'],rig['launch'],rig['prior'],resources=lambda:None)
    assert not rig['base'].exists()

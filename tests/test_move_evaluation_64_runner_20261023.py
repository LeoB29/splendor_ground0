"""Deadline, admission and immutable-plan regression checks without models."""
from types import SimpleNamespace

import pytest

from tools.move_evaluation_64_20261023 import common as c, launch, runner as r


def test_wall_phase_clamps_before_monotonic_phase():
    envelope = r.Envelope({'monotonic':100.,'unix_time':1000.})
    assert r.clamp_timeout(120,160,envelope=envelope,clock=lambda:110,wall_clock=lambda:1030) == 30
    assert r.clamp_timeout(120,160,envelope=envelope,clock=lambda:150,wall_clock=lambda:1030) == 10
    with pytest.raises(ValueError,match='deadline'):
        r.clamp_timeout(120,160,envelope=envelope,clock=lambda:110,wall_clock=lambda:1060)


def test_setup_cannot_reset_request_phase():
    envelope = r.Envelope({'monotonic':100.,'unix_time':1000.})
    assert r.clamp_timeout(60,160,envelope=envelope,clock=lambda:110,wall_clock=lambda:1010) == 50
    assert r.clamp_timeout(120,160,envelope=envelope,clock=lambda:145,wall_clock=lambda:1045) == 15


def test_effective_deadline_uses_earlier_wall(monkeypatch):
    envelope = r.Envelope({'monotonic':100.,'unix_time':1000.})
    monkeypatch.setattr(r.time,'monotonic',lambda:110.)
    monkeypatch.setattr(r.time,'time',lambda:1030.)
    assert envelope.effective_deadline(160) == 140


@pytest.mark.parametrize('seconds,okay',[(95.9,True),(96.,True),(96.00001,False),(120.,False),(float('nan'),False),(-1,False),(True,False)])
def test_margin_is_per_complete_receipt(seconds,okay):
    assert r.margin_passed(dict(status='ok',accepted=True,receipt_elapsed_seconds=seconds)) is okay
    assert not r.margin_passed(dict(status='timeout',accepted=False,receipt_elapsed_seconds=seconds))


def test_every_saved_request_in_fixed_order_and_counts():
    lock=c.read(c.PREP/'selection_lock.json'); requests=lock['timed_request_order']; roots=lock['roots']
    assert [(q['index'],q['arm']) for q in requests[:12]] == [
        (0,'A'),(0,'B'),(6,'B'),(6,'A'),(12,'A'),(12,'B'),
        (0,'B'),(0,'A'),(6,'A'),(6,'B'),(12,'B'),(12,'A')]
    assert len(requests)==54
    assert sum(64*len(roots[q['index']]['payload']['candidates']['actions']) for q in requests)==13696
    core=[q for q in requests if q['role']=='core']
    assert len({(q['index'],q['arm']) for q in core})==36
    assert sum(64*len(roots[q['index']]['payload']['candidates']['actions']) for q in core)==9088


def test_existing_launch_claim_forbids_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    launch.ensure_unlaunched()
    (tmp_path/'watchdog_claim.json').write_text('{}')
    with pytest.raises(ValueError,match='relaunch'): launch.ensure_unlaunched()


def test_freeze_addition_cannot_replace_inherited_hash():
    inputs={'a':{'path':'a','bytes':1,'sha256':'original'}}
    r.add_input(inputs,dict(inputs['a']))
    with pytest.raises(ValueError,match='collision'):
        r.add_input(inputs,{'path':'a','bytes':1,'sha256':'different'})


def test_postresponse_phase_guard_rejects_forward_clock_jump():
    calls=[]
    class Envelope:
        start=0; wall_start=r.time.time(); end=10**15
        def guard(self,deadline):
            calls.append('guard')
            raise ValueError('phase expired')
    child=SimpleNamespace(_study_envelope=Envelope(),request=lambda payload,limit:
                          dict(status='ok',accepted=True,receipt_elapsed_seconds=.1))
    with pytest.raises(ValueError,match='phase expired'):
        r.request(child,{},1,10**15)
    assert calls==['guard']


def test_ram_shortage_restarts_stability_but_not_study_clock(monkeypatch):
    current=[0.]; saved=[]
    monkeypatch.setattr(r.time,'monotonic',lambda:current[0])
    monkeypatch.setattr(r.time,'time',lambda:current[0])
    monkeypatch.setattr(r.time,'sleep',lambda seconds:current.__setitem__(0,current[0]+seconds))
    monkeypatch.setattr(r.psutil,'virtual_memory',lambda:SimpleNamespace(available=(1 if 10<=current[0]<20 else 3)*c.GiB))
    monkeypatch.setattr(r.shutil,'disk_usage',lambda path:SimpleNamespace(free=9*c.GiB))
    monkeypatch.setattr(r,'bytes_used',lambda:0)
    monkeypatch.setattr(c,'save',lambda path,data:saved.append(data))
    envelope=r.Envelope({'monotonic':0.,'unix_time':0.})
    envelope.admit(envelope.pilot_end)
    assert envelope.start==envelope.wall_start==0 and envelope.waited==80
    assert saved[-1]['minimum_available_bytes']==c.GiB and not saved[-1]['fast_path']
    envelope.admit(envelope.pilot_end)
    assert saved[-1]['fast_path'] and envelope.waited==80


def test_guard_rejects_phase_wall_even_when_global_is_open(monkeypatch):
    envelope=r.Envelope({'monotonic':100.,'unix_time':1000.})
    monkeypatch.setattr(r.time,'monotonic',lambda:110.)
    monkeypatch.setattr(r.time,'time',lambda:1060.)
    with pytest.raises(ValueError,match='real-clock'):
        envelope.guard(160,memory=False)

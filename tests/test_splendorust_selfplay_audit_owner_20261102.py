"""Model-free original-clock and saved-only owner bounds."""
from types import SimpleNamespace
import pytest
from tools.splendorust_selfplay_audit_20261102 import common as a, owner


@pytest.mark.parametrize('claim_m,claim_w,mono,wall,limit,expected',[
    (1000.,2000.,100.,200.,50.,(150.,250.)),
    (130.,240.,100.,200.,50.,(130.,240.)),
    (99999.,99999.,7100.,7200.,500.,(7200.,7300.)),
    (1000.,230.,100.,200.,50.,(150.,230.)),
])
def test_deadlines_clip_new_phase_allowance_audit_claim_and_both_original_clocks(claim_m,claim_w,mono,wall,limit,expected):
    claim=dict(monotonic_deadline=claim_m,unix_deadline=claim_w)
    old=dict(monotonic_origin=0.,unix_origin=100.)
    assert a.deadlines(claim,old,mono,wall,limit)==expected


def setup_guard(monkeypatch,tmp_path):
    clock=SimpleNamespace(mono=100.,wall=200.)
    monkeypatch.setattr(a,'time',SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall))
    process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=100),children=lambda recursive:[])
    monkeypatch.setattr(a.psutil,'Process',lambda:process)
    monkeypatch.setattr(a.shutil,'disk_usage',lambda p:SimpleNamespace(free=10*a.c.GiB))
    monkeypatch.setattr(a,'sys',SimpleNamespace(modules={}))
    monkeypatch.setattr(a,'OUT',tmp_path/'audit');monkeypatch.setattr(a.c,'OUT',tmp_path/'original')
    guard=a.Guard(dict(monotonic_deadline=1000.,unix_deadline=2000.),dict(monotonic_origin=0.,unix_origin=100.),50.)
    return guard,clock,process


def test_guard_retains_original_origins_for_historical_checks_but_new_expiry(monkeypatch,tmp_path):
    guard,clock,_=setup_guard(monkeypatch,tmp_path)
    assert (guard.start,guard.wall_start,guard.end)==(0.,100.,7200.)
    assert (guard.deadline,guard.wall_deadline)==(150.,250.)
    assert guard.remaining()==50. and guard.effective_deadline()==150.
    guard.guard();clock.mono=150.
    with pytest.raises(ValueError,match='deadline'):guard.guard()


def test_wall_clock_alone_expires_guard(monkeypatch,tmp_path):
    guard,clock,_=setup_guard(monkeypatch,tmp_path)
    clock.wall=251.
    assert guard.remaining()==-1.
    with pytest.raises(ValueError,match='deadline'):guard.guard()


@pytest.mark.parametrize('fault',['torch','child','rss','disk','evidence','live_close'])
def test_saved_only_guard_rejects_model_runtime_children_and_resource_failures(monkeypatch,tmp_path,fault):
    guard,clock,p=setup_guard(monkeypatch,tmp_path)
    if fault=='torch':a.sys.modules['torch']=object()
    elif fault=='child':p.children=lambda recursive:[object()]
    elif fault=='rss':p.memory_info=lambda:SimpleNamespace(rss=2*a.c.GiB+1)
    elif fault=='disk':monkeypatch.setattr(a.shutil,'disk_usage',lambda p:SimpleNamespace(free=5*a.c.GiB-1))
    elif fault=='evidence':
        a.OUT.mkdir();(a.OUT/'extra').write_bytes(b'x'*11);monkeypatch.setattr(a,'MAX_BYTES',10)
    with pytest.raises(ValueError):
        if fault=='live_close':guard.close_worker(tmp_path,child=object())
        else:guard.guard()


def test_guard_does_not_skip_clock_or_torch_checks_during_resource_throttle(monkeypatch,tmp_path):
    guard,clock,_=setup_guard(monkeypatch,tmp_path);guard.guard()
    clock.mono+=.1;a.sys.modules['torch']=object()
    with pytest.raises(ValueError,match='model runtime'):guard.guard()


@pytest.mark.parametrize('expired_clock',['monotonic','utc'])
def test_owner_rejects_acceptance_after_child_pass_if_either_final_clock_expired(monkeypatch,tmp_path,expired_clock):
    clock=SimpleNamespace(mono=100.,wall=200.)
    monkeypatch.setattr(owner,'time',SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall))
    monkeypatch.setattr(a,'OUT',tmp_path/'audit')
    saved=[]
    fake_c=SimpleNamespace(ROOT=tmp_path,OUT=tmp_path/'original',JOB_SECONDS=7200,GiB=1024**3,
        require=a.c.require,file_hash=lambda p:'freeze-sha',verify=lambda items:None,
        read=lambda p:dict(approved=True,freeze_sha256='freeze-sha') if p.name=='prelaunch_review.json' else dict(inputs=[]))
    monkeypatch.setattr(owner,'c',fake_c)
    monkeypatch.setattr(a,'check_original',lambda:(dict(monotonic_origin=0.,unix_origin=100.),[]))
    monkeypatch.setattr(a,'save',lambda name,value:saved.append((name,value)))
    monkeypatch.setattr(a.psutil,'Process',lambda:SimpleNamespace(create_time=lambda:50.))
    monkeypatch.setattr(a.psutil,'virtual_memory',lambda:SimpleNamespace(available=10*fake_c.GiB))
    monkeypatch.setattr(owner.shutil,'disk_usage',lambda p:SimpleNamespace(free=10*fake_c.GiB))
    monkeypatch.setattr(owner.placement,'configure_execution',lambda *args,**kwargs:None)
    def child(mode,sha,claim,old):
        if mode=='independent':
            if expired_clock=='monotonic':clock.mono=claim['monotonic_deadline']
            else:clock.wall=claim['unix_deadline']
        return True
    monkeypatch.setattr(owner,'run_child',child)
    with pytest.raises(ValueError,match='Final audit owner acceptance'):owner.owner('freeze-sha')
    assert [name for name,_ in saved]==['claim.json']

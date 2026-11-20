"""Deferred checkpoint provenance and exact owning-job handshake counterexamples."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
os.environ.setdefault('INDEPENDENT_DATA_REPETITION','0')
os.environ.setdefault('FULL_CORPUS_MATCHUP','mixed_parent')
from tools.splendorust_full_corpus_20261110 import common as root, owner, fit
from tools.splendorust_full_corpus_20261110.strength import common as c, checkpoint as cp, segment, envelope


@pytest.fixture
def isolate(tmp_path,monkeypatch):
    monkeypatch.setattr(root,'ROOT',tmp_path);monkeypatch.setattr(root,'OUT',tmp_path/'study')
    monkeypatch.setattr(root,'PLAN',root.OUT/'freeze.json');monkeypatch.setattr(root,'SELECTION',root.OUT/'selection.json')
    monkeypatch.setattr(root,'WRITE_OBSERVER',None)
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'STUDY_OUT',root.OUT)
    monkeypatch.setattr(c,'OUT',root.OUT/'strength/r0')
    monkeypatch.setattr(owner.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*root.GiB))
    return root.OUT


def commit_fixture(out,monkeypatch):
    freeze=root.save(out/'freeze.json',{'unit_fixture':True})
    corpus=root.save(out/'corpus/accepted.json',{'status':'saved_corpus_accepted'})
    body={'accepted_corpus':corpus,'new_games':1152,'historical_data_relabelled':False}
    selection=root.save(out/'selection.json',{'source_receipt':body})
    rows=[]
    for rep in (0,1):
        for breadth in ('terminal','mixed'):
            identity=f'r{rep}_{breadth}_joint'
            final=root.save(out/'fits'/identity/'worker/checkpoints/1782/checkpoint_manifest.json',{'fixture':identity})
            actual=root.save(out/'fits'/identity/'execution.json',{'worker_exit_code':0,'error':None,'cleanup_errors':[]})
            rows.append({'spec':next(spec for spec in root.fits() if spec['fit_id']==identity),'final_checkpoint':final,'actual_exit':actual})
    value={'schema':'full_corpus_all_fits_committed_v1','study_id':cp.SOURCE_STUDY,
        'freeze':freeze,'selection':selection,'source_receipt':body,'fits':rows}
    root.save(out/'all_fits_committed.json',value)
    seen=[]
    monkeypatch.setattr(fit,'validate_commitment',lambda selected,descriptor,**kw:seen.append((selected,descriptor)) or deepcopy(value))
    return value,seen


def test_deferred_import_requires_both_fixed_fits_commitment(isolate,monkeypatch):
    value,seen=commit_fixture(isolate,monkeypatch)
    assert cp.committed_fits()==value and len(seen)==1
    for rep in (0,1):
        descriptor=cp.source_descriptor(rep)
        assert descriptor['path'].endswith(f'fits/r{rep}_mixed_joint/worker/checkpoints/1782/checkpoint_manifest.json')
    assert len(seen)==3


def test_deferred_import_rejects_missing_commitment(isolate):
    with pytest.raises(FileNotFoundError):cp.committed_fits()


@pytest.mark.parametrize('mutation',['wrong_final','wrong_study','old_freeze','unclean_exit'])
def test_deferred_import_rejects_uncommitted_or_wrong_artifact(isolate,monkeypatch,mutation):
    value,_=commit_fixture(isolate,monkeypatch)
    original=root.read
    bad=deepcopy(value)
    if mutation=='wrong_final':bad['fits'][1]['final_checkpoint']=bad['fits'][0]['final_checkpoint']
    if mutation=='wrong_study':bad['study_id']='historical_study'
    if mutation=='old_freeze':bad['freeze']=root.save(isolate/'other_freeze.json',{'old':True})
    def read(path):
        if Path(path).name=='all_fits_committed.json':return deepcopy(bad)
        result=original(path)
        if mutation=='unclean_exit' and Path(path).name=='execution.json':result['error']='failed'
        return result
    monkeypatch.setattr(c,'read',read)
    monkeypatch.setattr(fit,'validate_commitment',lambda *a,**k:deepcopy(bad))
    with pytest.raises(ValueError):cp.source_descriptor(0)


def owning_documents(out,monkeypatch,*,mutation=None):
    freeze=root.save(out/'freeze.json',{'fixture':True})
    parent={'pid':123,'created':44.,'monotonic_origin':0.,'unix_origin':9000.,'freeze':freeze}
    claim=root.save(out/'claim.json',parent)
    phase=root.save(out/'stages/strength_start.json',{'stage':'strength','claim':claim,'monotonic':1000.,'unix_time':10000.,
        'monotonic_deadline':145000.,'unix_deadline':154000.,'cap_seconds':144000})
    folder=out/'segments/qualification_r0'
    local={'claim':claim,'phase_clock':phase};item=root.save(folder/'claim.json',local)
    value={'identity':{'pid':888,'created':99.},'windows_job_assigned':True,'segment_claim':item}
    if mutation=='pid':value['identity']['pid']=999
    if mutation=='creation':value['identity']['created']=99.1
    if mutation=='unassigned':value['windows_job_assigned']=False
    if mutation=='claim':value['segment_claim']=freeze
    root.save(folder/'owner.json',value)
    monkeypatch.setattr(segment.os,'getpid',lambda:888)
    monkeypatch.setattr(segment.psutil,'Process',lambda pid=None:SimpleNamespace(create_time=lambda:99. if pid is None else 44.,is_running=lambda:True))
    monkeypatch.setattr(segment.time,'monotonic',lambda:1001.)
    monkeypatch.setattr(segment.time,'time',lambda:10001.)
    return folder,local


def test_exact_assigned_job_handshake_before_strength_work(isolate,monkeypatch):
    folder,claim=owning_documents(isolate,monkeypatch)
    assert segment.owning_handshake(folder,claim)==root.artifact(folder/'owner.json')


@pytest.mark.parametrize('mutation',['pid','creation','unassigned','claim'])
def test_wrong_owner_handshake_refused(isolate,monkeypatch,mutation):
    folder,claim=owning_documents(isolate,monkeypatch,mutation=mutation)
    with pytest.raises(ValueError,match='Exact strength owner'):segment.owning_handshake(folder,claim)


@pytest.mark.parametrize('mode',['qualification','operational','main_block'])
def test_strength_modes_share_same_original_phase(isolate,monkeypatch,mode):
    monkeypatch.setattr(owner.time,'monotonic',lambda:1000.)
    monkeypatch.setattr(owner.time,'time',lambda:10000.)
    freeze=root.save(isolate/'freeze.json',{'fixture':True})
    claim={'monotonic_origin':0.,'unix_origin':9000.,'freeze':freeze}
    claimitem=root.save(isolate/'claim.json',claim)
    phase=owner.begin_phase(claim,'strength')
    ident=owner.segment_id(mode,0,0 if mode=='main_block' else None)
    segmentitem=root.save(isolate/'segments'/ident/'claim.json',dict(segment_id=ident,mode=mode,claim=claimitem,phase_clock=phase,reader_clock=None))
    env=envelope.SegmentEnvelope(claimitem,phase,segmentitem)
    assert env.deadline == 1000. + root.STAGES['strength'] and env.wall_deadline == 10000. + root.STAGES['strength']
    with pytest.raises(ValueError,match='root owner'):env.begin('strength')

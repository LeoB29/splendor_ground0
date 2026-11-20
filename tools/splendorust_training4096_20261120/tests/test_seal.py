from copy import deepcopy
import pytest
from tools.splendorust_training4096_20261120 import common as c,seal,corpus


def test_missing_commitment_rejects_before_expected_or_protected(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'COMMITMENT',tmp_path/'absent.json')
    monkeypatch.setattr(seal,'_expected',lambda *a,**k:pytest.fail('must not inspect fits before missing seal rejected'))
    with pytest.raises(ValueError):seal.validate_commitment()


def test_forged_release_capability_rejected():
    with pytest.raises(ValueError):seal.require_release(None)
    with pytest.raises(ValueError):seal.require_release(seal.Release(object(),{},()))


@pytest.mark.parametrize('field',['models','evaluation','assessment_binding','fits'])
def test_mutated_commitment_rejected_before_commit_exit(tmp_path,monkeypatch,field):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'COMMITMENT',tmp_path/'seal.json')
    expected=dict(models=['all7'],evaluation={'fixed':True},assessment_binding={'games':256},fits=['all4'])
    bad=deepcopy(expected);bad[field]=[];c.save(c.COMMITMENT,bad)
    monkeypatch.setattr(seal,'_expected',lambda *a,**k:expected)
    from tools.splendorust_training4096_20261120 import control
    monkeypatch.setattr(control,'passed_job',lambda *a,**k:pytest.fail('mutated seal must fail first'))
    with pytest.raises(ValueError):seal.validate_commitment()


def test_commit_exit_required_after_complete_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'COMMITMENT',tmp_path/'seal.json')
    expected={'models':['all7']};c.save(c.COMMITMENT,expected)
    monkeypatch.setattr(seal,'_expected',lambda *a,**k:expected)
    from tools.splendorust_training4096_20261120 import control
    def reject(*a,**k):raise ValueError('commit still running')
    monkeypatch.setattr(control,'passed_job',reject)
    with pytest.raises(ValueError):seal.validate_commitment()
    assert seal.validate_commitment(require_commit_exit=False)==expected


def test_stale_release_invalidated_before_protected_read(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);path=tmp_path/'seal.json';c.save(path,{'fixed':True})
    release=seal.Release(seal._TOKEN,c.artifact(path),(seal._stamp(path),))
    monkeypatch.setattr(seal,'_RELEASE',release)
    seal.require_release(release)
    path.write_text('{"fixed":false}')
    with pytest.raises(ValueError):seal.require_release(release)


def test_assessment_builder_calls_release_before_selection(tmp_path,monkeypatch):
    def reject(*a,**k):raise ValueError('sealed')
    monkeypatch.setattr(seal,'acquire_release',reject)
    monkeypatch.setattr(corpus.selection,'load',lambda *a,**k:pytest.fail('selection not opened before failed gate'))
    with pytest.raises(ValueError):corpus.build_assessment(tmp_path)

@pytest.fixture
def fit_graph(tmp_path,monkeypatch):
    from tools.splendorust_training4096_20261120 import control,fit
    monkeypatch.setattr(c,'OUT',tmp_path); proofs={};jobs={}
    for fit_id in c.FIT_IDS:
        folder=tmp_path/'jobs'/fit_id/'worker';checkpoints={}
        for step in (0,7100,7101):
            place=folder/'checkpoints'/str(step)
            payload=c.save(place/'checkpoint.json',{'fake_tensor_bits':step})
            tensor=c.save(place/'tensor_manifest.json',{'step':step})
            checkpoints[str(step)]=c.save(place/'manifest.json',dict(schema='training4096_checkpoint_manifest_v1',checkpoint=payload,tensor_manifest=tensor,
                metadata=dict(fit_id=fit_id,checkpoint_step=step,qualification_only=False,weights_sha256='parent' if step==0 else fit_id,**c.IDENTITY),tree_sha256={'state_dict':str(step)}))
        update=c.save(folder/'update.json',{'step':7101})
        replay=c.save(folder/'replay.json',dict(status='exact_final_update_reproduced',optimizer_calls=1,step=7101,from_checkpoint=checkpoints['7100'],
            expected_checkpoint=checkpoints['7101'],tree_sha256={'state_dict':'7101'},actual_checkpoint=checkpoints['7101'],expected_update=update))
        final=dict(fit_id=fit_id,status='fit_complete',optimizer_calls=7101,replay_optimizer_calls=1,record_presentations=227203,endpoint=checkpoints['7101'],checkpoints=checkpoints,replay=replay)
        item=c.save(folder/'fit_result.json',final);execution=c.save(folder.parent/'actual_exit.json',{'passed':True})
        jobs[fit_id]=(execution,{'result':{'fit_result':item}})
        proofs[str(folder)]=dict(status='training4096_fit_saved_verified',result=item,endpoint=final['endpoint'],checkpoints=checkpoints,optimizer_calls=7101,replay_optimizer_calls=1,record_presentations=227203)
    monkeypatch.setattr(control,'passed_job',lambda role,boundary=lambda:None:jobs[role])
    monkeypatch.setattr(fit,'verify_fit',lambda folder,boundary=lambda:None:proofs[str(folder)])
    return jobs


def test_complete_four_endpoint_graph_commits_exact_proofs(fit_graph):
    records,parent=seal._fit_records(lambda:None,full=True)
    assert len(records)==4 and parent=='parent'
    assert seal._fit_records(lambda:None)==(records,parent)


def test_incomplete_fourth_fit_never_produces_complete_seal(fit_graph):
    from tools.splendorust_training4096_20261120 import control
    del fit_graph[c.FIT_IDS[-1]]
    with pytest.raises(KeyError):seal._fit_records(lambda:None,full=True)


def test_fit_exit_result_cannot_substitute_another_fit(fit_graph):
    ident=c.FIT_IDS[0];other=c.FIT_IDS[1]
    fit_graph[ident]=(fit_graph[ident][0],fit_graph[other][1])
    with pytest.raises(ValueError):seal._fit_records(lambda:None,full=True)


def test_replay_must_be_exact_full_state(fit_graph):
    import json
    folder=c.OUT/'jobs'/c.FIT_IDS[0]/'worker';final=c.read(folder/'fit_result.json');path=c.resolve(final['replay'])
    replay=c.read(path);replay['tree_sha256']={'state_dict':'changed'};path.write_text(c.canonical(replay))
    final['replay']=c.artifact(path);(folder/'fit_result.json').write_text(c.canonical(final))
    fit_graph[c.FIT_IDS[0]][1]['result']['fit_result']=c.artifact(folder/'fit_result.json')
    with pytest.raises(ValueError):seal._fit_records(lambda:None,full=True)

"""Small provenance/dispatch fixtures; production training is never entered."""
from concurrent.futures import Future
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from . import recovery as r


def fixture(base, count=5):
    settings={'updates':4,'diagnostic_updates':[0,4],'resume_checkpoint_interval':2}
    arguments=r.jobs()
    initial=base/'initial.pt';initial.write_bytes(b'fixture model bytes')
    initializers={fit:r.c.artifact(initial) for fit,_ in arguments}
    files=[]
    for index,(fit,arm) in enumerate(arguments[:count]):
        folder=base/'fits'/f'fit_{fit}'/arm;folder.mkdir(parents=True)
        identity={'manifest':'fixture','fit':fit,'arm':arm,'config':settings,'initializer':initializers[fit],
                  'model_config':r.e.model.MODEL_CONFIG,'batch_receipt':{'fixture':True}}
        points=[]
        for update in settings['diagnostic_updates']:
            model=folder/f'checkpoint_{update:05d}.pt';model.write_bytes(str(update).encode())
            training=folder/f'training_{update:05d}.json';r.c.old.save_new(training,{'fixture':True})
            point={'update':update,'identity_sha256':r.c.old.digest(identity),
                   'checkpoint':r.c.artifact(model),'training':r.c.artifact(training)}
            receipt=folder/f'point_{update:05d}.json';r.c.old.save_new(receipt,point)
            points.append({**point,'receipt':r.c.artifact(receipt)})
        optimizers=[]
        for update in r.e.learning._progress_updates(settings):
            path=folder/f'optimizer_{update:05d}.pt';path.write_bytes(str(update).encode())
            optimizers.append(r.c.artifact(path,update=update))
        result={**identity,'identity':identity,'complete':True,'selected_update':4,
                'checkpoints':points,'optimizer_checkpoints':optimizers,'selected_model':points[-1]['checkpoint'],
                'selected_training':points[-1]['training'],'assessment_read':False,'validation_read':False}
        result_path=folder/'result.json';r.c.old.save_new(result_path,result)
        ref=r.c.artifact(result_path,fit=fit,arm=arm,index=index);files.append(ref)
        r.c.old.save_new(base/'completed_jobs/learning_fits'/f'{index:06d}.json',
                        {'manifest':'fixture','stage':'learning','kind':'fits','argument_sha256':r.c.old.digest(arguments[index]),'artifact':ref})
    for name in ('freeze.json','data_lock.json','training_feature_audit.json'):
        r.c.old.save_new(base/name,{'fixture':True})
    r.c.old.save_new(base/'learning_fits_launch.json',{'manifest':'fixture','stage':'learning','kind':'fits',
             'expected_indices':list(range(6)),'arguments_sha256':r.c.old.digest(arguments),
             'dependencies':[r.c.artifact(path) for path in r.upstream(base)]})
    return arguments,settings,initializers,files


def test_completed_five_authenticate_without_deserializing_models(tmp_path,monkeypatch):
    args,settings,initializers,files=fixture(tmp_path)
    monkeypatch.setattr(r.e.torch,'load',lambda *a,**kw:pytest.fail('Model deserialization forbidden in receipt authentication'))
    actual,evidence=r.completed_jobs(args,'fixture',tmp_path,range(5),settings,initializers)
    assert actual==files and len(evidence)>5
    r.authenticate_stage_launch(args,'fixture',r.upstream(tmp_path),tmp_path)


@pytest.mark.parametrize('damage',['argument','index','manifest','checkpoint','launch_dependency','duplicate'])
def test_changed_provenance_is_rejected(tmp_path,damage):
    args,settings,initializers,files=fixture(tmp_path)
    path=tmp_path/'completed_jobs/learning_fits/000004.json'
    receipt=r.c.old.read_json(path)
    if damage=='argument':receipt['argument_sha256']='changed'
    if damage=='index':receipt['artifact']['index']=0
    if damage=='manifest':receipt['manifest']='changed'
    if damage in ('argument','index','manifest'):path.write_text(json.dumps(receipt))
    if damage=='checkpoint':(tmp_path/'fits/fit_161003/raw/optimizer_00002.pt').write_bytes(b'changed')
    if damage=='duplicate':(tmp_path/'completed_jobs/learning_fits/000005.json').write_text(json.dumps(receipt))
    if damage=='launch_dependency':(tmp_path/'training_feature_audit.json').write_text('{}')
    with pytest.raises((AssertionError,ValueError)):
        r.authenticate_stage_launch(args,'fixture',r.upstream(tmp_path),tmp_path)
        r.completed_jobs(args,'fixture',tmp_path,range(5),settings,initializers)


def test_only_missing_fit_dispatched_partial_state_untouched_and_original_launch_preserved(tmp_path,monkeypatch):
    args,settings,initializers,files=fixture(tmp_path,count=6)
    # Completed sixth output is synthetic worker output, revealed by fake dispatch.
    existing=tmp_path/'completed_jobs/learning_fits/000005.json'
    sixth=existing.read_bytes();existing.unlink()
    partial=tmp_path/'fits/fit_161003/relations/resume.json'
    r.c.old.save_new(partial,{'update':9216,'checkpoint':files[5]})
    launch_before=(tmp_path/'learning_fits_launch.json').read_bytes();partial_before=partial.read_bytes()
    recovery=tmp_path/'recovery';recovery.mkdir();r.c.old.save_new(recovery/'freeze.json',{'fixture':True})
    monkeypatch.setattr(r,'BASE',tmp_path);monkeypatch.setattr(r,'OUTPUT',recovery);monkeypatch.setattr(r,'BINDING','fixture')
    monkeypatch.setattr(r.e,'event',lambda *a,**kw:None)
    monkeypatch.setattr(r.c.windows,'configure_spawn',lambda:{'fixture':True})
    boundaries=[];monkeypatch.setattr(r.c,'budget_boundary',lambda base:boundaries.append(base))
    class Pool:
        def __init__(self,**kw):
            assert kw['initializer'] is r.e.worker_initialize
            assert kw['max_workers']==1 and kw['initargs']==('learning','fits','fixture',tmp_path)
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(r.e,'ProcessPoolExecutor',Pool)
    dispatched=[]
    def dispatch(pool,arguments,job,workers,on_result,boundary):
        dispatched.extend(arguments);assert job.func is r.e.run_job
        boundary();existing.write_bytes(sixth);on_result(1);return [files[5]]
    monkeypatch.setattr(r.e,'dispatch',dispatch)
    manifest={'original_snapshot':{'completed':files[:5],'evidence':files[:5],
              'partial':{'pointer_at_freeze':r.c.artifact(partial),'resume_pointer':{'update':9216,'checkpoint':files[5]}}}}
    result=r.recovered_fit_pool('learning','fits',args,'fixture',r.upstream(tmp_path),manifest,lambda *a:pytest.fail('Old fit pool must not relaunch'))
    assert result==files and dispatched==[args[5]] and boundaries==[tmp_path,tmp_path]
    assert partial.read_bytes()==partial_before and (tmp_path/'learning_fits_launch.json').read_bytes()==launch_before
    assert r.c.old.read_json(tmp_path/'learning_fits_lock.json')['reused_indices']==list(range(5))
    assert r.recovered_fit_pool('learning','fit_audits',[], 'fixture',[],manifest,lambda *a:'original')=='original'


def test_adapters_restore_after_exception_and_reuse_only_authenticated_feature_audit(tmp_path,monkeypatch):
    original_pool,original_features=r.e.run_pool,r.e.training_source.audit_features
    monkeypatch.setattr(r,'BASE',tmp_path);monkeypatch.setattr(r,'BINDING','fixture')
    calls=[]
    monkeypatch.setattr(r,'authenticate_stage_launch',lambda *a,**kw:calls.append('launch'))
    monkeypatch.setattr(r.e.training_source,'validate_features',lambda *a:calls.append('features'))
    r.c.old.save_new(tmp_path/'training_feature_audit.json',{'fixture':True})
    with pytest.raises(RuntimeError):
        with r.recovery_adapters({}):
            assert r.e.training_source.audit_features('fixture',tmp_path)==r.c.artifact(tmp_path/'training_feature_audit.json')
            raise RuntimeError('fixture')
    assert calls==['launch','features']
    assert r.e.run_pool is original_pool and r.e.training_source.audit_features is original_features


def test_pipeline_keeps_original_gate_and_all_future_stages(monkeypatch):
    calls=[]
    monkeypatch.setattr(r.e,'build_learning',lambda b:calls.append('learning') or False)
    monkeypatch.setattr(r.e,'build_sources',lambda b:calls.append('sources'))
    monkeypatch.setattr(r.e,'build_stage',lambda s,b:calls.append(s))
    monkeypatch.setattr(r.e,'operational_decision',lambda b:calls.append('decision') or True)
    monkeypatch.setattr(r.e,'finish',lambda b:calls.append('finish'))
    monkeypatch.setattr(r.c,'budget_boundary',lambda b:calls.append('boundary'))
    r.continue_pipeline('fixture');assert calls==['learning']
    calls.clear();monkeypatch.setattr(r.e,'build_learning',lambda b:calls.append('learning') or True)
    r.continue_pipeline('fixture')
    assert calls==['learning','boundary','sources','boundary','probe','decision','main','boundary','finish']


def test_actual_hidden_spawn_keeps_original_windowless_plumbing(tmp_path):
    output=tmp_path/'health.json'
    code='import sys;sys.path.insert(0,'+repr(str(r.ROOT/'src'))+');from pathlib import Path;from splendor_ai.alphazero import windowless_workers_160 as w;import json;Path('+repr(str(output))+').write_text(json.dumps(w.assert_windowless_worker()))'
    process=r.c.windows.detached_launch([sys.executable,'-c',code],tmp_path/'stdout.txt',tmp_path/'stderr.txt',r.ROOT)
    import psutil
    started=time.monotonic()
    while not output.exists() and time.monotonic()-started<20:time.sleep(.05)
    assert output.exists(),(tmp_path/'stderr.txt').read_text()
    observed=json.loads(output.read_text())
    assert observed['pid']==process['pid'] and observed['console_window']==observed['console_process_count']==0
    while psutil.pid_exists(process['pid']) and time.monotonic()-started<20:time.sleep(.05)
    assert not psutil.pid_exists(process['pid'])


def test_recovery_snapshot_json_roundtrip_and_changed_pointer(tmp_path,monkeypatch):
    identity={'fixture':True}
    snapshot={'arguments':[(161003,'relations')],'partial':{'argument':(161003,'relations'),'update':9216}}
    monkeypatch.setattr(r,'OUTPUT',tmp_path)
    monkeypatch.setattr(r,'own_identity',lambda:identity)
    monkeypatch.setattr(r,'stopped_snapshot',lambda:snapshot)
    r.c.old.save_new(tmp_path/'checks.json',{'identity':identity,'exit_code':0,'logs':[]})
    manifest={'identity':identity,'original_snapshot':snapshot,'checks':r.c.artifact(tmp_path/'checks.json'),'owner_authorized':True}
    r.c.old.save_new(tmp_path/'manifest.json',manifest)
    r.c.old.save_new(tmp_path/'freeze.json',{'manifest':r.c.artifact(tmp_path/'manifest.json'),'binding':r.c.old.digest(manifest)})
    assert r.validate_recovery()[0]==r.c.old.digest(manifest)
    snapshot['partial']['update']=9472
    with pytest.raises((ValueError,AssertionError),match='stopped state changed'):r.validate_recovery()

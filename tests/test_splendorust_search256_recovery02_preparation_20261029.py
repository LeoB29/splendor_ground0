from pathlib import Path
from types import SimpleNamespace
import hashlib,json,subprocess
import pytest
from tools.splendorust_search256_strength_20261029_02 import common as c,design,preparation,schedule,adapters,runner,analysis


def test_absolute_artifact_paths_accepted_relative_escape_rejected_and_tampering_caught(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(c,'ROOT',root)
    path=root/'evidence.json';path.write_text('{"passed":true}')
    pin=c.artifact(path);c.verify([pin]);c.verify([dict(pin,path=str(path.resolve()))])
    with pytest.raises(ValueError,match='Canonical workspace'):c.verify([dict(pin,path='../outside.json')])
    path.write_text('{"passed":false}')
    with pytest.raises(ValueError,match='Changed frozen input'):c.verify([pin])


def test_outputs_are_exclusive_and_cannot_rewrite_earlier_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'new')
    path=c.OUT/'control.json';c.atomic_save(path,dict(complete=True));saved=path.read_bytes()
    assert not path.with_suffix('.json.pending').exists()
    with pytest.raises(ValueError,match='already published'):c.atomic_save(path,dict(complete=False))
    with pytest.raises(FileExistsError):c.save(path,dict(complete=False))
    with pytest.raises(ValueError,match='outside'):c.save(tmp_path/'prior/control.json',{})
    assert path.read_bytes()==saved and not (tmp_path/'prior').exists()


def test_static_dependency_closure_includes_function_local_and_relative_imports(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    folder=tmp_path/'tools/study';folder.mkdir(parents=True)
    files={folder/'__init__.py':'',folder/'entry.py':'from . import common\ndef run():\n from .child import helper\n return helper()\n',
        folder/'common.py':'from tools.policy import RULE\n',folder/'child.py':'def helper(): return 1\n',tmp_path/'tools/policy.py':'RULE=1\n'}
    for path,text in files.items():path.write_text(text)
    dependencies=set(preparation.python_dependencies([folder/'entry.py']))
    assert dependencies=={p.resolve() for p in files}


def test_lifecycle_allowlist_does_not_raise_actual_worker_ceiling():
    lifecycle=design.lifecycle_inventory()
    assert len(lifecycle)==910
    actual={name for name in lifecycle if not name.startswith('audit_')}
    assert len(actual)==814
    for i in range(48):
        name=design.replay_worker_name('main','research',i,'research' if i%2 else 'frozen')
        assert lifecycle[name]['ordinary_requests']==1;actual.add(name)
    assert len(actual)==c.MAX_WORKERS==862
    envelope=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START))
    envelope.worker_accounting={i:None for i in range(862)}
    with pytest.raises(ValueError,match='worker bound'):envelope.register_worker(Path('forbidden_extra_slot'))


def test_prepare_is_idempotent_but_cannot_replace_registered_schedule(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'outputs/current')
    first=design.prepare();second=design.prepare();assert first==second
    saved=c.read(c.OUT/'schedule.json');saved['main'][0]['seed_decimal']='1'
    (c.OUT/'schedule.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='differs'):design.prepare()


def seed_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'outputs/current')
    prior=tmp_path/'outputs/old/schedule.json';prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({stage:schedule.make_schedule(stage) for stage in ('operational','main')}))
    preservation=tmp_path/'outputs/old_postfailure/preservation.json';preservation.parent.mkdir(parents=True)
    preservation.write_text('{}')
    result=dict(reusable_schedule=c.artifact(prior),preservation=c.artifact(preservation),prior_game_count=0,prior_research_requests_dispatched=0)
    monkeypatch.setattr(preparation,'verify_prior_attempt',lambda:result)
    return prior,result


@pytest.mark.parametrize('collision',['none','game','statistics'])
def test_registered_seed_collision_is_detected_without_replacement(tmp_path,monkeypatch,collision):
    prior,receipt=seed_fixture(tmp_path,monkeypatch)
    path=tmp_path/'outputs/unrelated/schedule.json';path.parent.mkdir(parents=True)
    seed=0 if collision=='none' else (analysis.SEED if collision=='statistics' else int(schedule.make_schedule('main')[0]['seed_decimal']))
    path.write_text(json.dumps(dict(frozen_seed=seed)))
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='\n'.join(x.relative_to(tmp_path).as_posix() for x in (prior,path))))
    before=path.read_bytes()
    if collision=='none':
        result=design.seed_inventory()
        assert result['status']=='passed' and result['game_setups']==203 and len(result['checked'])==1
        assert result['reused_registered_schedule']==[c.artifact(prior)]
    else:
        with pytest.raises(ValueError,match='seed'):design.seed_inventory()
    assert path.read_bytes()==before and not c.OUT.exists()


def test_seed_exclusion_never_skips_entire_prior_folder(tmp_path,monkeypatch):
    prior,_=seed_fixture(tmp_path,monkeypatch)
    another=prior.with_name('other_schedule.json');another.write_text(prior.read_text())
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='\n'.join(x.relative_to(tmp_path).as_posix() for x in (prior,another))))
    with pytest.raises(ValueError,match='seed collision'):design.seed_inventory()


def test_seed_reuse_requires_exact_preserved_schedule(tmp_path,monkeypatch):
    prior,_=seed_fixture(tmp_path,monkeypatch)
    prior.write_text('{}')
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=prior.relative_to(tmp_path).as_posix()))
    with pytest.raises(ValueError,match='unused schedule unchanged'):design.seed_inventory()


def test_seed_reuse_requires_prior_qualification_and_no_game_admission(tmp_path,monkeypatch):
    prior,_=seed_fixture(tmp_path,monkeypatch)
    def fail():raise ValueError('Prior game or dispatched request cannot be reused')
    monkeypatch.setattr(preparation,'verify_prior_attempt',fail)
    with pytest.raises(ValueError,match='cannot be reused'):design.seed_inventory()


def test_exact_original_clocks_wait_and_resources_remain_charged():
    recovery=preparation.recovery_contract()
    assert recovery['monotonic_origin']==564433.078
    assert recovery['unix_origin']==1793308578.3462226
    assert recovery['qualification_monotonic_deadline']==564433.078+7200
    assert recovery['qualification_unix_deadline']==1793308578.3462226+7200
    assert recovery['prior_actual_qualification_monotonic_deadline']==571633.625
    assert recovery['prior_ram_wait_seconds']==60.06299999996554
    assert recovery['prior_worker_trees']==2 and recovery['maximum_combined_worker_trees']==864
    assert c.MAX_EVIDENCE_BYTES+c.PRIOR_EVIDENCE_BYTES==24*c.GiB
    assert c.PREFINAL_EVIDENCE_BYTES+c.PRIOR_EVIDENCE_BYTES==16*c.GiB
    assert c.COLLECTION_CUTOFF_SECONDS==36*3600 and c.JOB_SECONDS==48*3600


def test_actual_saved_failure_is_closed_preserved_and_safe_for_unused_schedule_reuse():
    result=preparation.verify_prior_attempt()
    assert result['status']=='passed' and len(result['files'])==265 and len(result['source_inputs'])==891
    assert result['prior_identities_absent']==15 and result['prior_game_count']==0 and result['prior_research_requests_dispatched']==0
    assert result['reusable_schedule']==c.artifact(c.PRIOR_ATTEMPT_OUT/'schedule.json')


@pytest.mark.parametrize('mutation',['old_source','old_evidence','clock','dispatch','active_owner'])
def test_preservation_refuses_changed_sources_clocks_dispatch_and_live_owners(monkeypatch,mutation):
    import gzip,io,psutil
    real_verify=c.verify;real_read=c.read
    if mutation in ('old_source','old_evidence'):
        def changed(items):
            if len(items)==(891 if mutation=='old_source' else 265):raise ValueError('Changed frozen input')
            return real_verify(items)
        monkeypatch.setattr(c,'verify',changed)
    elif mutation=='clock':monkeypatch.setattr(c,'PRIOR_ORIGIN',c.PRIOR_ORIGIN+1)
    elif mutation=='dispatch':monkeypatch.setattr(gzip,'open',lambda *args,**kwargs:io.BytesIO(b'published'))
    else:
        identities=c.read(preparation.PRESERVATION)['identities'];target=identities[0]
        real_process=psutil.Process
        monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:target['created']) if pid==target['pid'] else real_process(pid))
    with pytest.raises(ValueError):preparation.verify_prior_attempt()


def test_original128_profile_bytes_unchanged_and256_only_quota_science_differs():
    original=adapters.profile_path(128,route='frozen');before=original.read_bytes()
    assert hashlib.sha256(before).hexdigest()=='048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb'
    baseline=adapters.profile(128,route='frozen');candidate=adapters.profile(256,route='research')
    changed={key for key in baseline.keys()|candidate.keys() if baseline.get(key)!=candidate.get(key)}
    assert changed=={'iterations','profile_id','wrapper_id','execution_policy'}
    assert candidate['iterations']==256 and baseline['iterations']==128
    assert candidate['native_binary']==baseline['native_binary']
    assert candidate['execution_policy']['changed_scientific_fields']==['iterations']
    assert original.read_bytes()==before


def test_freeze_contract_tampering_rejected_before_runtime():
    frozen=dict(preparation.contract(),schema='splendorust_search256_strength_v1',only_scientific_change='simulations_128_to_256')
    assert runner.validate_freeze_contract(frozen)==preparation.contract()
    frozen['main_games']=399
    with pytest.raises(ValueError,match='contract'):runner.validate_freeze_contract(frozen)

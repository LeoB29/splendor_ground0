from pathlib import Path
from types import SimpleNamespace
import hashlib,json,subprocess
import pytest
from tools.splendorust_search256_strength_20261029 import common as c,design,preparation,schedule,adapters,runner,analysis


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
    envelope=runner.Envelope(dict(monotonic_origin=1.,unix_origin=1.))
    envelope.worker_accounting={i:None for i in range(862)}
    with pytest.raises(ValueError,match='worker bound'):envelope.register_worker(Path('forbidden_extra_slot'))


def test_prepare_is_idempotent_but_cannot_replace_registered_schedule(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'outputs/current')
    first=design.prepare();second=design.prepare();assert first==second
    saved=c.read(c.OUT/'schedule.json');saved['main'][0]['seed_decimal']='1'
    (c.OUT/'schedule.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='differs'):design.prepare()


@pytest.mark.parametrize('collision',['none','game','statistics'])
def test_registered_seed_collision_is_detected_without_replacement(tmp_path,monkeypatch,collision):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'outputs/current')
    path=tmp_path/'outputs/prior/schedule.json';path.parent.mkdir(parents=True)
    seed=0 if collision=='none' else (analysis.SEED if collision=='statistics' else int(schedule.make_schedule('main')[0]['seed_decimal']))
    path.write_text(json.dumps(dict(frozen_seed=seed)))
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=path.relative_to(tmp_path).as_posix()+'\n'))
    before=path.read_bytes()
    if collision=='none':
        result=design.seed_inventory()
        assert result['status']=='passed' and result['game_setups']==203 and len(result['checked'])==1
    else:
        with pytest.raises(ValueError,match='seed'):design.seed_inventory()
    assert path.read_bytes()==before and not c.OUT.exists()


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

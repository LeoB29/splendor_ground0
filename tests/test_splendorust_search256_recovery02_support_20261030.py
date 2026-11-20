"""Model-free checks of cumulative resources and both prior ownership trees."""
from copy import deepcopy
import pytest
from tools.splendorust_search256_recovery_20261030_02 import preparation_support as s


@pytest.fixture
def saved(monkeypatch):
    actual=s._read;values={}
    def read(path):
        if path not in values:values[path]=actual(path)
        return deepcopy(values[path])
    monkeypatch.setattr(s,'_read',read)
    return values,read


def test_both_original_and_previous_saved_identity_trees_are_bound():
    identities=s.recorded_old_identities()
    assert len(identities)==len({s._pair(x) for x in identities})==30
    assert {p['pid'] for p in identities[15:18]}=={21392,68552,29928}
    assert len(s.recorded_previous_identities())==15


def test_all_thirty_absence_checks_are_repeated(monkeypatch):
    seen=[]
    def absent(identity):seen.append(s._pair(identity));return dict(identity,alive=False)
    monkeypatch.setattr(s.original,'_absent',absent)
    first=s.verify_old_absence();second=s.old_owner_identities()
    assert first==second and len(seen)==60 and all(x['alive'] is False for x in first)


def test_any_previous_live_identity_prevents_admission(monkeypatch):
    def absent(identity):
        if identity['pid']==29928:raise ValueError('still alive')
        return dict(identity,alive=False)
    monkeypatch.setattr(s.original,'_absent',absent)
    with pytest.raises(ValueError,match='still alive'):s.verify_old_absence()


@pytest.mark.parametrize('name,key,value',[
    ('execution.json','exit_code',0),('execution.json','cleanup_errors',['unfinished']),
    ('execution.json','status','completed'),('watchdog_execution.json','exit_code',0),
    ('watchdog_execution.json','freeze_sha256','0'*64),('completion.json','status','completed_verified'),
    ('failure.json','stage','main_verification')])
def test_failed_exit_provenance_cannot_be_laundered(saved,name,key,value):
    values,read=saved;path=s.PREVIOUS_OUT/name;read(path);values[path][key]=value
    with pytest.raises(ValueError):s.previous_exit_provenance()


@pytest.mark.parametrize('key,value',[('input_created',True),('request_dispatched',True),
    ('move_clock_started',True),('additional_retry_required',True),('ply',51),('quota',256)])
def test_clean_boundary_is_exact(saved,key,value):
    values,read=saved;path=s.DIAGNOSIS_OUT/'boundary_01.json';read(path);values[path]['next_move'][key]=value
    with pytest.raises(ValueError,match='between-move'):s.previous_exit_provenance()


def test_guardian_exit_requires_actual_retained_observation(saved):
    values,read=saved;path=s.DIAGNOSIS_OUT/'status_01.json';read(path)
    values[path]['guardian_actual_exit_code']=None
    with pytest.raises(ValueError,match='guardian exit'):s.previous_exit_provenance()


def test_changed_helper_creation_identity_is_rejected(saved):
    values,read=saved;path=s.PREVIOUS_OUT/'workers'/s.WORKERS[0]/'request-00000026.json';read(path)
    values[path]['telemetry']['process_tree'][2]['created']+=1
    with pytest.raises(ValueError,match='process identities'):s.recorded_previous_identities()


def test_incomplete_worker_cleanup_is_rejected(saved):
    values,read=saved;path=s.PREVIOUS_OUT/'workers'/s.WORKERS[1]/'cleanup.json';read(path)
    values[path]['alive']=[123]
    with pytest.raises(ValueError,match='helper exits'):s.recorded_previous_identities()


def test_ram_counter_is_conservatively_carried_without_reset():
    first=s.ram_wait_accounting();second=s.ram_wait_accounting()
    assert first==second
    assert first['last_durable_wait_seconds']==1722.90699999989
    assert 1820<first['cumulative_upper_bound_seconds']<1821
    assert first['charged_prior_ram_wait_seconds']==1821
    assert first['charged_prior_ram_wait_seconds']>1803
    assert first['fast_admissions_not_silently_omitted']
    assert not first['counters_reset'] and not first['exact_actual_wait_seconds_known']


def test_changed_last_ram_counter_rejected(saved):
    values,read=saved;path=sorted((s.PREVIOUS_OUT/'memory_admissions').glob('*.json'))[-1];read(path)
    values[path]['cumulative_wait_seconds']=0
    with pytest.raises(ValueError,match='durable cumulative'):s.ram_wait_accounting()


def test_cumulative_worker_cap_counts_every_failed_tree():
    proof=s.old_tree_counts()
    assert proof['used_worker_trees']==488+120==608
    assert proof['maximum_new_worker_trees']==2+105*2+48==260
    assert proof['maximum_combined_worker_trees']==868
    assert proof['completed_graph_parities']==proof['used_graph_builds']-1


def test_changed_previous_lifecycle_count_rejected(monkeypatch):
    original=s.original._tree_count
    def changed(path):
        value=original(path)
        if path==s.PREVIOUS_OUT:value['worker_trees']+=1
        return value
    monkeypatch.setattr(s.original,'_tree_count',changed)
    with pytest.raises(ValueError,match='lifecycle counts'):s.old_tree_counts()


def test_exact_metadata_census_includes_both_diagnoses():
    proof=s.prior_evidence_census()
    assert [(x['files'],x['bytes']) for x in proof['trees']]==[(23163,1043690409),(2,12655)]
    assert proof['total_prior_evidence_bytes']==5115862569


def test_support_imports_no_model_or_execution_runtime():
    import ast
    tree=ast.parse(s.Path(s.__file__).read_text())
    modules=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert modules==['datetime','pathlib','tools.splendorust_search256_recovery_20261030']

"""Saved failed qualification and old ledger regressions; no model execution."""
from copy import deepcopy
import os
import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION','0')
from tools.splendorust_independent_data_disk_recovery_20261109_02.strength import common as c
from tools.splendorust_independent_data_disk_recovery_20261109_02.strength import profile_parity as p,analysis,lineage,transport
from tools.splendorust_independent_data_recovery_20261109.strength import common as oldc


@pytest.fixture(scope='module')
def actual_data():
    root=c.ROOT
    old=root/'outputs/splendorust_independent_data_recovery_20261109_01/strength/r1'
    failed=root/'outputs/splendorust_independent_data_disk_recovery_20261109_01/strength/r1'
    left,right=c.read(old/'freeze.json'),c.read(failed/'freeze.json')
    manifests=[c.read(root/f['candidate_manifest']['path']) for f in (left,right)]
    tensors=[c.read(root/m['tensor_manifest']['path']) for m in manifests]
    profiles=[c.read(root/f['profiles']['candidate']['128']['path']) for f in (left,right)]
    suites=[c.read(base/'workers'/name/'entrypoint_clock.json')['result'] for base,name in
            ((old,'qualification_candidate'),(failed,'main_g159_candidate'))]
    return dict(old=old,failed=failed,left=left,right=right,manifests=manifests,tensors=tensors,
                profiles=profiles,suites=suites)


@pytest.fixture
def actual(actual_data,monkeypatch):
    d=actual_data
    monkeypatch.setattr(c,'REPETITION',1);monkeypatch.setattr(c,'OUT',d['failed'])
    monkeypatch.setattr(c,'STUDY_OUT',d['failed'].parent.parent)
    monkeypatch.setattr(c,'LIVE_SOURCE_OUT',d['old'])
    monkeypatch.setattr(oldc,'REPETITION',1);monkeypatch.setattr(oldc,'OUT',d['old'])
    monkeypatch.setattr(lineage,'old_freeze',lambda:d['left'])
    yield d


def result_objects(value):
    if type(value) is dict:
        if 'profile_sha256' in value:yield value
        else:
            for item in value.values():yield from result_objects(item)
    elif type(value) is list:
        for item in value:yield from result_objects(item)


def test_actual_saved_failure_only_authenticated_profile_relocation(actual):
    d=actual;before=[c.digest(v) for v in d['suites']]
    assert d['suites'][0]!=d['suites'][1]
    proof=p.compare_suite(*d['suites'],d['left'],d['right'],'candidate')
    assert proof['status']=='passed' and proof['scientific_fields_exact'] is True
    assert len(proof['normalized_profile_fields'])==len(list(result_objects(d['suites'][0])))
    assert {path.rsplit('/',1)[-1] for path in proof['normalized_profile_fields']}=={'profile_sha256'}
    assert before==[c.digest(v) for v in d['suites']]
    assert proof['profile_binding']['exact_model_bytes'] and proof['profile_binding']['exact_tensor_inventory']


@pytest.mark.parametrize('mutation',['action','trace','callback','unknown_profile','quota','profile_id','extra','missing','bool'])
def test_actual_saved_suite_scientific_difference_rejected(actual,mutation):
    d=actual;changed=deepcopy(d['suites'][1]);result=next(result_objects(changed))
    if mutation=='action':result['action']['action_type']='INVALID_TEST'
    elif mutation=='trace':result['diagnostics']['observer_seed']+=1
    elif mutation=='callback':result['diagnostics']['phases'][0]['inference'][0]['output_sha256']='0'*64
    elif mutation=='unknown_profile':result['profile_sha256']='0'*64
    elif mutation=='quota':result['simulations_quota']=32
    elif mutation=='profile_id':result['profile_id']='different'
    elif mutation=='extra':result['unexpected']=0
    elif mutation=='missing':del result['action']
    elif mutation=='bool':result['simulations_quota']=True
    with pytest.raises(ValueError):p.compare_suite(d['suites'][0],changed,d['left'],d['right'],'candidate')


@pytest.mark.parametrize('mutation',['weights','checkpoint_bytes','training_metadata','source','tensor_hash','tensor_shape','tensor_order'])
def test_candidate_manifest_and_tensors_cannot_be_changed(actual_data,mutation):
    d=actual_data;a,b=deepcopy(d['manifests']);x,y=deepcopy(d['tensors'])
    if mutation=='weights':b['checkpoint']['sha256']='0'*64
    elif mutation=='checkpoint_bytes':b['checkpoint']['bytes']+=1
    elif mutation=='training_metadata':b['metadata']['checkpoint_step']=95
    elif mutation=='source':b['source_manifest']['sha256']='0'*64
    elif mutation=='tensor_hash':y['tensors'][0]['sha256']='0'*64
    elif mutation=='tensor_shape':y['tensors'][0]['shape'][0]+=1
    elif mutation=='tensor_order':y['tensors'][0],y['tensors'][1]=y['tensors'][1],y['tensors'][0]
    with pytest.raises(ValueError):p.relocated_candidate(a,b,x,y)


@pytest.mark.parametrize('mutation',['cpuct','iterations','world_pool','depth','internal_budget_seconds','native_binary','unknown_artifact','duplicate_artifact'])
def test_profile_settings_cannot_hide_behind_relocation(actual_data,mutation):
    d=actual_data;a,b=deepcopy(d['profiles'])
    if mutation in ('cpuct','iterations','world_pool','depth','internal_budget_seconds'):b[mutation]+=1
    elif mutation=='native_binary':b[mutation]='other.exe'
    elif mutation=='unknown_artifact':b['artifacts'][0]['sha256']='0'*64
    elif mutation=='duplicate_artifact':a['artifacts'].append(deepcopy(d['left']['candidate_manifest']))
    with pytest.raises(ValueError):p.relocate_profile(a,b,d['left']['candidate_manifest'],d['right']['candidate_manifest'],*d['manifests'])


def test_actual_complete_old_game_passes_new_scientific_ledger(actual):
    d=actual;folder=d['old']/'games/main/g000'
    paths=sorted((folder/'moves').glob('*.json'));hashes=[c.file_hash(path) for path in paths]
    rows=[c.read(path) for path in paths]
    candidate=next(row for row in rows if row['variant']=='candidate')
    with pytest.raises(ValueError):analysis.validate_completed(candidate)
    view=p.audit_view(candidate,d['right']);analysis.validate_completed(view)
    assert view['clock']['result']['profile_sha256']!=candidate['clock']['result']['profile_sha256']
    assert view['action']==candidate['action'] and view['trace']==candidate['trace'] and view['after']==candidate['after']
    proof=p.audit_game(c.read(folder/'initial.json')['initial'],rows,c.read(folder/'result.json'),c.read(folder/'spec.json'),freeze=d['right'])
    assert proof==c.read(folder/'online_replay.json')
    assert hashes==[c.file_hash(path) for path in paths]


def test_actual_old_row_tampering_is_rejected_before_normalization(actual):
    d=actual;row=c.read(d['old']/'games/main/g159/moves/000025.json')
    row['after']['state']['current_player']=1-row['after']['state']['current_player']
    with pytest.raises(ValueError):p.audit_view(row,d['right'])


def test_actual_six_old_operational_games_keep_exact_summary(actual):
    d=actual;folders=[d['old']/f'games/operational/g{index:03d}' for index in range(6)]
    games=[c.read(folder/'result.json') for folder in folders]
    rows=(c.read(path) for folder in folders for path in sorted((folder/'moves').glob('*.json')))
    proof=p.operational_summary(games,rows,freeze=d['right'])
    assert proof==c.read(d['old']/'operational_analysis.json')


def test_exact_replay_comparison_accepts_only_profile_metadata(actual):
    d=actual;folder=d['old']/'games/main/g000'
    original=next(row for row in (c.read(path) for path in sorted((folder/'moves').glob('*.json'))) if row['variant']=='candidate')
    replayed=p.audit_view(original,d['right'])
    selected={key:original[key] for key in ('stage','game_index','pair_index','ply','variant','player_role','seat',
        'repetition','checkpoint_name','global_game_index','global_pair_index')}
    selected['observer_sha256']=c.digest(original['observer'])
    proof=p.verify_replay(original,replayed,selected,freeze=d['right'])
    assert proof['full_scientific_output_exact'] and proof['raw_receipts_unchanged']
    assert proof['result_sha256']==c.digest(original['clock']['result'])
    assert proof['original_raw_result_sha256']!=proof['replayed_raw_result_sha256']
    replayed['trace']['observer_seed']+=1
    with pytest.raises(ValueError):p.verify_replay(original,replayed,selected,freeze=d['right'])


def test_cache_skips_large_rehash_after_full_validation(actual,monkeypatch):
    d=actual;proof=p.profile_map(d['left'],d['right'],'candidate')
    monkeypatch.setattr(p.checkpoint,'candidate_sources',lambda *a,**k:pytest.fail('large model revalidated per row'))
    assert p.profile_map(d['left'],d['right'],'candidate')==proof
    changed=deepcopy(d['right']);changed['checkpoint_bindings']['candidate']['weights_sha256']='0'*64
    with pytest.raises(ValueError):p.profile_map(d['left'],changed,'candidate')


def test_worker_namespace_and_science_sources_unchanged():
    assert transport.OBSERVER_MODULE=='tools.splendorust_independent_data_disk_recovery_20261109_02.strength.worker'
    old=c.ROOT/'tools/splendorust_independent_data_recovery_20261109/strength'
    new=c.ROOT/'tools/splendorust_independent_data_disk_recovery_20261109_02/strength'
    for name in ('worker.py','scientific.py','graph.py','adapters.py','analysis.py','qualification.py','corpus.py',
                 'checkpoint.py','design.py','boundary_adapter.py','parent_profile.py','accounting.py'):
        assert c.file_hash(old/name)==c.file_hash(new/name),name

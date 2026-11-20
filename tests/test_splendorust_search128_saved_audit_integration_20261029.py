"""Real saved closure/path integration; no models, games, or old-output writes."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import pytest

from tools.splendorust_search128_resume_20261029 import independent_review as frozen
from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime
from tools.splendorust_search128_saved_audit_20261029.path_runtime import AuthorizedRuntimeReview

RESUME_SHA='0e4eb49a3f32bf4e52480cd24d6472b7bbe42e131c85c289cc396ee8d4ff5aaf'


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def saved_closure():
    """Use the complete real nested inventory, never the former verify stub."""
    reader=AuthorizedRuntimeReview(frozen.ROOT,frozen.OUT,seconds=180.)
    freeze=reader.read('freeze.json')
    frozen.source_authentication(reader,freeze,RESUME_SHA)
    manifest=frozen.bound_read(reader,freeze['resume_manifest'])
    protected=[Path(frozen.__file__),Path(ReviewRuntime.__module__.replace('.','/')).with_suffix('.py')]
    protected += [frozen.OUT/name for name in ('owner_failure.json','postrun/independent_review_failure.json',
        'postrun/independent_review_os_exit.json','postrun/acceptance_invalidated.json','completion.json')]
    before={str(path):file_digest(path) for path in protected}
    result=frozen.review_previous_closure(reader,freeze,manifest)
    yield reader,freeze,manifest,result
    reader.revalidate_identities()
    assert before=={str(path):file_digest(path) for path in protected}


def test_unchanged_saved_closure_full_real_absolute_and_relative_authentication(saved_closure):
    reader,freeze,manifest,value=saved_closure
    items=value['consumed_inputs']+value['imported_frozen_sources_and_runtime']+[value['checker_source']]
    assert len(items)==449 and sum(Path(item['path']).is_absolute() for item in items)==95
    assert len(reader.absolute_runtime_descriptors)==95
    for item in items:
        path=reader.dispatch_descriptor(item)
        got=reader.pin(path)
        assert (got['bytes'],got['sha256'])==(item['bytes'],item['sha256'])
    assert value['status']=='qualified_saved_closure_checker_only'
    assert value['benchmark_accepted'] is False and value['original_attempt_accepted'] is False
    assert value['new_model_calls']==value['new_games']==0
    assert value['actual_entrypoint_decisions']==119 and value['graph_cases']==22
    assert value['ordinary_saved_moves']==14 and value['segmented_game']['proof']['transitions']==54
    assert len(value['closed_workers'])==2 and all(p['status']=='passed' for p in value['closed_workers'])
    assert sum(p['raw']['requests'] for p in value['closed_workers'])==18
    assert reader.revalidate_identities()['status']=='passed'


def test_original_real_absolute_failure_reproduces_without_altering_its_receipt(saved_closure):
    reader,_,_,value=saved_closure
    absolute=next(x for x in value['imported_frozen_sources_and_runtime'] if Path(x['path']).is_absolute())
    original=ReviewRuntime(frozen.ROOT,frozen.OUT,seconds=20.)
    with pytest.raises(ValueError,match='Canonical project-relative path|Safe path'):
        original.verify(absolute)
    reader.verify(absolute)
    # General source/game path parsing remains strict and project-relative.
    with pytest.raises(ValueError,match='Canonical project-relative path|Safe path'):
        reader.physical(absolute['path'])


@pytest.mark.parametrize('field',['bytes','sha256','path'])
def test_actual_absolute_authority_rejects_changed_descriptor(saved_closure,field):
    reader,_,_,value=saved_closure
    item=deepcopy(next(x for x in value['imported_frozen_sources_and_runtime'] if Path(x['path']).is_absolute()))
    if field=='bytes':item[field]+=1
    elif field=='sha256':item[field]='0'*64
    else:item[field]=str(Path(item[field]).parent/'unapproved_runtime.dll')
    with pytest.raises(ValueError):reader.verify(item)


def test_saved_terminal_failure_remains_failed_after_component_verification(saved_closure):
    reader,_,_,value=saved_closure
    failed=reader.read('postrun/independent_review_failure.json')
    terminal=reader.read('postrun/independent_review_os_exit.json')
    owner=reader.read('owner_failure.json');invalid=reader.read('postrun/acceptance_invalidated.json')
    assert failed['status']=='failed_no_acceptance' and 'review_previous_closure' in failed['traceback']
    assert 'Canonical project-relative path' in failed['error']
    assert terminal['exit_code']==1 and terminal['component_sha256'] is None
    assert terminal['observed_by']=='owned Process handle' and terminal['cleanup_reason'] is None
    assert owner['status']=='not_accepted' and invalid['accepted'] is False
    assert not(frozen.OUT/'postrun/independent_completion_review.json').exists()
    assert not(frozen.OUT/'owner_completion.json').exists()
    for root in (frozen.ORIGINAL,frozen.FIRST_RECOVERY):
        assert reader.read_path(root/'completion.json')['status']=='incomplete'
    assert value['benchmark_accepted'] is False


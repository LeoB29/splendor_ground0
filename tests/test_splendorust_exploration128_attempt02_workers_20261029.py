"""Read-only scientific identity and new-attempt write-boundary regressions."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

import pytest

from tools.splendorust_exploration128_20261029_02 import (
    adapters as a, common as c, corpus, scientific as s, transport as t,
    worker as w, qualification as q, placement,
)
from tools.splendorust_exploration128_20261029 import (
    adapters as old_a, common as old_c, worker as old_w, transport as old_t,
    qualification as old_q,
)

OLD = c.ROOT/'outputs/splendorust_exploration128_20261029_01'
PRIOR_FREEZE = old_c.read(OLD/'freeze.json')
OLD_SOURCE = c.ROOT/'tools/splendorust_exploration128_20261029'


def syntax(obj):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(obj))),include_attributes=False)


@pytest.mark.parametrize('name',['BaselineOpponent','ConfiguredOpponent','initialize_adapters',
    'validate_native_configuration','verify_native_configuration_receipts','_phase_configuration_expected',
    '_configuration_expected','validate_result','validate_execution_identity','ordinary_payload'])
def test_scientific_and_native_attestation_methods_unchanged(name):
    assert syntax(getattr(a,name))==syntax(getattr(old_a,name))


@pytest.mark.parametrize('module',[s,corpus,q,placement])
def test_scientific_qualification_and_placement_byte_identical(module):
    assert Path(module.__file__).read_bytes()==(OLD_SOURCE/Path(module.__file__).name).read_bytes()


def test_worker_only_attempt_environment_changes():
    text=Path(w.__file__).read_text().replace('SPLENDOR_EXPLORATION128_ATTEMPT02_PHASE','SPLENDOR_EXPLORATION128_PHASE')
    assert text==Path(old_w.__file__).read_text()


def test_transport_only_module_and_attempt_environment_changes():
    text=Path(t.__file__).read_text().replace('tools.splendorust_exploration128_20261029_02.worker',
        'tools.splendorust_exploration128_20261029.worker').replace('SPLENDOR_EXPLORATION128_ATTEMPT02_PHASE','SPLENDOR_EXPLORATION128_PHASE')
    assert text==Path(old_t.__file__).read_text()
    assert t.OBSERVER_MODULE=='tools.splendorust_exploration128_20261029_02.worker'
    assert c.OUT==c.ROOT/'outputs/splendorust_exploration128_20261029_02'
    assert c.OUT!=a.SCIENTIFIC_OUT==OLD


@pytest.mark.parametrize('route',a.ROUTES)
def test_exact_preserved_profiles_ids_binaries_and_metadata(route):
    for quota in (16,128):
        descriptor=PRIOR_FREEZE['profiles'][route][str(quota)]
        path=c.ROOT/descriptor['path']
        assert a.profile_path(quota,route=route)==path==old_a.profile_path(quota,route=route)
        assert a.profile(quota,route=route)==old_c.read(path)==old_a.profile(quota,route=route)
        assert a.profile_sha256(quota,route=route)==descriptor['sha256']
        assert a.profile_id(quota,route=route)==old_a.profile_id(quota,route=route)
        assert a.wrapper_id(quota,route=route)==old_a.wrapper_id(quota,route=route)
        assert a.native_descriptor(route,quota)==old_a.native_descriptor(route,quota)
    assert a.native_artifacts(route)==PRIOR_FREEZE['native_artifacts_by_route'][route]
    assert a.native_roles(route)==PRIOR_FREEZE['native_roles_by_route'][route]


def test_profile_preparation_cannot_write_historical_evidence():
    with patch.object(c,'save',side_effect=AssertionError('No profile writes authorized')):
        value=a.prepare_profiles()
    for field in ('profiles','native_artifacts_by_route','native_roles_by_route','native_build_receipt'):
        assert value[field]==PRIOR_FREEZE[field]
    assert c.ROOT/value['native_build_receipt']['path']==OLD/'native_build/build_receipt.json'


def test_reused_native_build_receipt_is_explicitly_hash_bound():
    assert a.native_build_receipt()==old_a.native_build_receipt()
    assert a.NATIVE_BUILD_RECEIPT_SHA256==PRIOR_FREEZE['native_build_receipt']['sha256']
    with patch.object(a,'file_hash',return_value='f'*64):
        with pytest.raises(ValueError,match='first-attempt native build receipt'):a.native_build_receipt()


@pytest.mark.parametrize('route',['low','high','parity'])
def test_configured_profile_sha_is_bound_to_first_attempt(route):
    assert a.CONFIGURED_PROFILE_SHAS[route]==PRIOR_FREEZE['profiles'][route]['128']['sha256']
    original_hash=a.file_hash
    def corrupt(path):
        return 'f'*64 if Path(path)==a.profile_path(128,route=route) else original_hash(path)
    with patch.object(a,'file_hash',side_effect=corrupt):
        with pytest.raises(ValueError,match='preserved configured profile'):a.profile(128,route=route)


def test_foreign_worker_output_rejected_before_models(monkeypatch):
    monkeypatch.setenv('STUDY_WORKER_DIR',str(OLD/'workers/qualification_low'))
    observed=[]
    def source_guard():observed.append('sources');return {}
    with patch.object(w,'verify_runtime_sources',source_guard),patch.object(a,'initialize_adapters',side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError,match='evidence boundary'):w.initialize('low')
    assert observed==['sources']


def test_foreign_freeze_rejected_before_models(monkeypatch):
    monkeypatch.setenv('STUDY_WORKER_DIR',str(c.OUT/'workers/qualification_low'))
    monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1')
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE',str(OLD/'freeze.json'))
    with patch.object(w,'verify_runtime_sources',return_value={}),patch.object(a,'initialize_adapters',side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError,match='freeze binding'):w.initialize('low')


def test_once_only_actual_source_guard_preserved():
    nodes=ast.walk(ast.parse(inspect.getsource(w.initialize)))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources' for n in nodes)==1
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('route',q.QUAL_ROUTES)
def test_saved_actual_entrypoint_evidence_passes_identical_new_checker(route):
    # Checker regression only; this does not qualify the new worker or reuse game results.
    folder=OLD/'workers'/('qualification_'+route)
    receipt=c.read(OLD/'entrypoint_checks'/f'{route}.json')
    before=deepcopy(receipt)
    actual=q.verify_entrypoint(folder,receipt,q.build_plan(),PRIOR_FREEZE)
    expected=old_q.verify_entrypoint(folder,receipt,old_q.build_plan(),PRIOR_FREEZE)
    assert actual==expected and actual['raw_execution_decisions_bound']==67
    assert receipt==before
    assert a.verify_native_configuration_receipts(folder,route)==old_a.verify_native_configuration_receipts(folder,route)


def test_previous_frozen_worker_sources_preserved():
    by_path={item['path']:item for item in PRIOR_FREEZE['inputs']}
    for path in sorted(OLD_SOURCE.glob('*')):
        if not path.is_file():continue
        key=path.relative_to(c.ROOT).as_posix()
        assert key in by_path
        c.verify([by_path[key]])


def test_qualification_inventory_unchanged():
    assert q.build_plan()==old_q.build_plan()
    plan=q.build_plan()
    assert (plan['scientific_decisions'],plan['graph_cases'],plan['ordinary_requests'])==(364,44,96)

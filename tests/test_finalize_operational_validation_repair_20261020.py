from copy import deepcopy
import json

import pytest

from tools import finalize_operational_validation_repair_20261020 as final


@pytest.fixture
def analysis():
    timing = {'mean': 20., 'p95': 80., 'max': 119.}
    return {'status': 'passed', 'outcomes_closed': True, 'strength_analysis_games': 0,
        'larger_data_deferred': True, 'operational_schedule': {'games': 18, 'pairs': 9},
        'per_stage': {
            'games:local': {'attempted': 365, 'statuses': {'completed': 361, 'forfeit_timeout': 4},
                            'completed_receipt_seconds': timing},
            'games:opponent': {'attempted': 365, 'statuses': {'completed': 365},
                               'completed_receipt_seconds': timing}},
        'scope': {'original_timeout_forfeits_preserved': 4},
        'source_preservation': {'all_hashes_and_sizes_match': True},
        'actual_continuation_verification': {'roots': 361, 'full_stress_roots': 45,
            'sample_only_roots': 316, 'replayed_union': {'worlds': 100, 'continuations': 400, 'transitions': 9000},
            'covered_cells': 70, 'cells': 81},
        'resources': {'original_attempt_elapsed_seconds': 10000., 'first_recovery_elapsed_seconds': 20000.,
                      'production_elapsed_seconds': 10362., 'including_final_analysis_seconds': 25000.},
        'combined_memory_waiting': {'admissions': 459, 'charged_seconds': 1218.},
        'final_validation_repair': {'elapsed_from_original_launch_seconds': 25060.}}


def report(data):
    return final.operational_report(data, {}, {'source_files': 5172, 'original_frozen_inputs': 27935})


def test_report_keeps_failures_and_distinguishes_elapsed_hold(analysis):
    text = report(analysis)
    assert 'four earlier local planner timeouts' in text
    assert 'not all active computation' in text
    assert 'original coordinator hold remains preserved' in text
    assert 'Matchup outcomes remain closed' in text
    assert '361 / 365' in text


@pytest.mark.parametrize('path,value', [
    (('status',), 'failed'), (('outcomes_closed',), False), (('strength_analysis_games',), 18),
    (('larger_data_deferred',), False), (('operational_schedule', 'games'), 17),
    (('scope', 'original_timeout_forfeits_preserved'), 3),
    (('per_stage', 'games:local', 'statuses', 'forfeit_timeout'), 0),
    (('actual_continuation_verification', 'roots'), 360),
    (('actual_continuation_verification', 'full_stress_roots'), 44),
    (('actual_continuation_verification', 'sample_only_roots'), 315),
    (('source_preservation', 'all_hashes_and_sizes_match'), False),
])
def test_report_rejects_incomplete_acceptance(analysis, path, value):
    changed = deepcopy(analysis)
    at = changed
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = value
    with pytest.raises(RuntimeError):
        report(changed)


def test_artifact_recheck_rejects_changed_content(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'ROOT', tmp_path)
    p = tmp_path / 'evidence.json'
    p.write_text('{"passed":true}')
    row = final.artifact(p)
    final.check_artifacts([row])
    p.write_text('{"passed":false}')
    with pytest.raises(RuntimeError, match='Bound artifact changed'):
        final.check_artifacts([row])


def test_artifact_recheck_rejects_duplicate_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'ROOT', tmp_path)
    p = tmp_path / 'evidence.json'
    p.write_text('{}')
    row = final.artifact(p)
    with pytest.raises(RuntimeError, match='Unique'):
        final.check_artifacts([row, row])


def test_artifact_recheck_observes_deadline_between_files(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'ROOT', tmp_path)
    p = tmp_path / 'evidence.json'
    p.write_text('{}')
    def expired():
        raise RuntimeError('expired')
    with pytest.raises(RuntimeError, match='expired'):
        final.check_artifacts([final.artifact(p)], expired)


def test_source_preservation_requires_exact_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'ROOT', tmp_path)
    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setattr(final, 'SOURCE', source)
    (source / 'freeze.json').write_text(json.dumps({'inputs': []}))
    (source / 'coordinator_hold.json').write_text(json.dumps({'status': 'preserved_hold'}))
    snapshot = tmp_path / 'inventory.json'
    snapshot.write_text(json.dumps({'source_base': 'source', 'outcomes_closed': True,
                                   'files': [final.artifact(p) for p in source.iterdir()]}))
    freeze = {'inputs': [final.artifact(snapshot)], 'source_inventory': final.artifact(snapshot)}
    assert final.source_preservation(freeze)['historical_hold_retained'] is True
    (source / 'extra.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Exact preserved'):
        final.source_preservation(freeze)

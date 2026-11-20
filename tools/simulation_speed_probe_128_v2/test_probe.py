"""Pure plan/guard contracts, no model load or trajectory generation."""
from pathlib import Path
import pytest
from . import probe


def test_metadata_only_selection_balanced_by_phase_and_support():
    specs = [{'phase': p, 'index': p*10+i, 'canonical_support': 5+i, 'unread_outcome': i%2}
             for p in range(3) for i in range(6)]
    selected = probe.choose_specs(list(reversed(specs)))
    assert [r['spec']['index'] for r in selected] == [0,3,5,10,13,15,20,23,25]
    assert [r['support_band'] for r in selected] == ['minimum','median','maximum']*3


def test_safe_output_rejects_production_and_nested_directories():
    for path in [probe.OLD, probe.ROOT/'outputs/structured_proposer_128',
                 probe.ROOT/'outputs/simulation_speed_probe_128_01/nested',
                 probe.ROOT/'tools/simulation_speed_probe_128']:
        with pytest.raises(ValueError): probe.safe_output(path)
    assert probe.safe_output(probe.ROOT/'outputs/simulation_speed_probe_128_test').parent == probe.ROOT/'outputs'


def test_probe_does_not_import_running_study_or_enable_production():
    assert probe.CONFIG['production_enabled'] is False
    assert probe.CONFIG['threads'] == 1
    assert probe.CONFIG['world_indices'] == [0,17,35,53,71,89,107,127]
    assert probe.CONFIG['micro_active_seconds'] <= 180
    assert probe.CONFIG['rollout_active_seconds'] <= 900


def test_resource_boundaries_and_running_phase_firewall():
    info = {'available_ram_bytes': probe.CONFIG['minimum_available_ram_bytes']*2,
            'commit_headroom_bytes': probe.CONFIG['minimum_commit_headroom_bytes']*2,
            'process_rss_bytes': probe.CONFIG['maximum_process_rss_bytes']//2}
    assert probe.resource_ok(info, 6*1024**3)
    for field, value in [('available_ram_bytes', 0), ('commit_headroom_bytes', 0),
                         ('process_rss_bytes', 3*1024**3)]:
        assert not probe.resource_ok({**info, field: value}, 6*1024**3)
    assert not probe.resource_ok(info, 4*1024**3)
    assert not probe.resource_ok({**info,'process_peak_rss_bytes':probe.CONFIG['maximum_process_rss_bytes']+1},6*1024**3)
    for phase in ('learning_data_validation','learning_fits','learning_fit_audits'):
        assert probe.permitted_stage(phase)
    for phase in ('source_main_collection','probe_collection','main_collection','main_audits',None):
        assert not probe.permitted_stage(phase)
        assert probe.permitted_stage(phase, True)

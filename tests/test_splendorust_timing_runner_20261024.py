"""No-model checks of diagnostic invariants, failure retention and bounds."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.splendorust_timing_20261024 import common as c, runner as r, launch
from tools.splendorust_timing_20261024.fixtures import repeated_schedule


def clock(result=None, seconds=.4):
    return dict(status='ok', clock='perf_counter_ns', request_id='request-00000001',
        deadline_seconds=1., elapsed_seconds=seconds+.01, receipt_elapsed_seconds=seconds,
        received_before_deadline=True, accepted=True, result=result or {})


def fixtures():
    return dict(development=[dict(fixture_id='root', snapshot={'saved': 1}, observer={'lawful': 1},
        preserved_variants=[dict(snapshot={'saved': 2})], historical32=dict(action={'move': 1}, trace={'hash': 9}))])


def rows_for(manifest):
    schedule = repeated_schedule(manifest['development'])
    rows = []
    for spec in schedule:
        item = manifest['development'][spec['fixture_index']]
        before = item['snapshot'] if spec['variant'] == 'original' else item['preserved_variants'][0]['snapshot']
        result = {}
        if spec['mode'] == 'profiled':
            result['timing_profile'] = {'totals': {'host_inference': {'wall_ns': 10_000_000}}}
        receipt = clock(result, .42 if spec['mode'] == 'profiled' else .4)
        if spec['mode'] == 'profiled':
            receipt['parent_timing'] = {'spans': {'resource_guard': {'wall_ns': 1_000_000}}}
        rows.append(dict(spec, before=before, observer=item['observer'], action={'move': 1},
                         trace={'hash': 9}, completed=True, clock=receipt))
    return rows, schedule


def test_only_nonscientific_profile_field_is_removed():
    original = dict(action={'move': 1}, diagnostics={'callback': 'same'})
    decorated = dict(original, timing_profile={'diagnostic': True})
    assert r.clean_result(decorated, 'profiled') == original
    assert decorated['timing_profile'] == {'diagnostic': True}
    assert r.clean_result(original, 'baseline') == original
    assert r.clean_result(dict(decorated, unexpected=1), 'profiled')['unexpected'] == 1
    # Unknown scientific fields reach the frozen strict validator, not a broad
    # envelope filter that could hide protocol changes.


@pytest.mark.parametrize('result,mode', [({}, 'profiled'), ({'timing_profile': []}, 'profiled'),
    ({'timing_profile': {}}, 'baseline'), ({}, 'unknown')])
def test_mode_binding_is_fail_closed(result, mode):
    with pytest.raises(ValueError):
        r.clean_result(result, mode)


def test_exact_repeats_and_hidden_variant_are_compared_to_historical_output():
    manifest = fixtures()
    rows, schedule = rows_for(manifest)
    checked = r.validate_reproductions(rows, manifest, schedule)
    assert checked == dict(status='passed', decisions=8, exact_reproduced_roots=1, historical_comparisons=8)


@pytest.mark.parametrize('field,value', [('action', {'move': 2}), ('trace', {'hash': 8}),
    ('observer', {'illegal_hidden': 1}), ('before', {'other': 1}), ('completed', False), ('mode', 'baseline')])
def test_reproduction_rejects_changed_behavior_or_identity(field, value):
    manifest = fixtures()
    rows, schedule = rows_for(manifest)
    rows[1][field] = value
    with pytest.raises(ValueError):
        r.validate_reproductions(rows, manifest, schedule)


def test_duplicate_and_missing_rows_cannot_pass():
    manifest = fixtures()
    rows, schedule = rows_for(manifest)
    with pytest.raises(ValueError):
        r.validate_reproductions(rows[:-1], manifest, schedule)
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError):
        r.validate_reproductions(rows, manifest, schedule)


def test_exact_reproduction_distinguishes_bool_from_numeric_fields():
    manifest = fixtures()
    manifest['development'][0]['historical32']['trace'] = {'hash': 1}
    rows, schedule = rows_for(manifest)
    for row in rows:
        row['trace'] = {'hash': 1}
    rows[1]['trace'] = {'hash': True}
    with pytest.raises(ValueError, match='exact decision'):
        r.validate_reproductions(rows, manifest, schedule)


def test_summary_keeps_margin_exceedance_and_observed_instrumentation_cost():
    rows, _ = rows_for(fixtures())
    rows[0]['clock'] = clock(seconds=.8)
    summary = r.summarize(rows)
    assert summary['modes']['baseline']['above_margin'] == 1
    assert summary['modes']['baseline']['above_deadline'] == 0
    assert summary['modes']['baseline']['arrival_seconds']['maximum'] == .8
    assert summary['paired_profile_minus_baseline_seconds']['count'] == 4
    assert summary['observed_nested_spans_seconds']['worker']['host_inference']['mean'] == .01
    assert summary['observed_nested_spans_seconds']['parent']['resource_guard']['mean'] == .001
    assert summary['overlapping_spans_must_not_be_summed'] is True


def test_summary_rejects_missing_partner():
    rows, _ = rows_for(fixtures())
    with pytest.raises(ValueError, match='paired sample incomplete'):
        r.summarize(rows[:-1])


def test_distribution_does_not_silently_accept_invalid_clocks():
    for values in ([], [float('nan')], [True], [float('inf')]):
        with pytest.raises(ValueError):
            r.distribution(values)
    assert r.distribution([1., 2., 3.])['p95'] == pytest.approx(2.9)


@pytest.fixture
def envelope(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'attempt')
    c.OUT.mkdir()
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: SimpleNamespace(free=8*c.GiB))
    monkeypatch.setattr(r.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(r.time, 'time', lambda: 1000.)
    return r.Envelope(dict(monotonic_origin=100., unix_origin=1000.))


def test_profiling_cutoff_reserves_final_time_and_never_resets_job(envelope, monkeypatch):
    envelope.begin('qualification')
    assert envelope.deadline == 700.
    monkeypatch.setattr(r.time, 'monotonic', lambda: 700.)
    monkeypatch.setattr(r.time, 'time', lambda: 1600.)
    envelope.begin('profiling')
    assert envelope.deadline == 2200.
    monkeypatch.setattr(r.time, 'monotonic', lambda: 2190.)
    monkeypatch.setattr(r.time, 'time', lambda: 3090.)
    envelope.begin('final')
    assert envelope.deadline == 2790.
    assert envelope.end == 2800.


def test_wall_deadline_alone_stops_attempt(envelope, monkeypatch):
    envelope.begin('qualification')
    monkeypatch.setattr(r.time, 'time', lambda: 1600.)
    with pytest.raises(ValueError, match='real-clock deadline'):
        envelope.guard()


def test_evidence_reserves_64mib_until_final(envelope, monkeypatch):
    monkeypatch.setattr(envelope, 'accounted_bytes', lambda: 449*c.MiB)
    envelope.stage = 'profiling'
    with pytest.raises(ValueError, match='evidence ceiling'):
        envelope.guard()
    envelope.stage = 'final'
    envelope.guard()


def fake_process(pid, created, rss, descendants=()):
    return SimpleNamespace(pid=pid, create_time=lambda: created,
        children=lambda recursive: list(descendants), memory_info=lambda: SimpleNamespace(rss=rss))


def test_aggregate_rss_is_one_cap_across_resident_roots_and_helpers(envelope):
    native = fake_process(3, 3., int(.2*c.GiB))
    one = fake_process(1, 1., int(.9*c.GiB), [native])
    two = fake_process(2, 2., int(.9*c.GiB))
    envelope.resident = {(1, 1.): one, (2, 2.): two}
    assert envelope.aggregate_worker_rss() <= 2*c.GiB
    two.memory_info = lambda: SimpleNamespace(rss=c.GiB)
    with pytest.raises(ValueError, match='Combined resident'):
        envelope.aggregate_worker_rss()


def test_starting_worker_tree_counts_before_readiness(envelope):
    envelope.resident = {(1, 1.): fake_process(1, 1., int(1.1*c.GiB))}
    child = SimpleNamespace(process=SimpleNamespace(pid=2))
    with pytest.raises(ValueError, match='Combined resident'):
        envelope.guard_worker_sample(child, {'rss_bytes': c.GiB})


def test_current_worker_is_not_double_counted(envelope):
    envelope.resident = {(1, 1.): fake_process(1, 1., c.GiB)}
    envelope.guard_worker_sample(SimpleNamespace(process=SimpleNamespace(pid=1)), {'rss_bytes': c.GiB})
    assert envelope.peak_combined_rss == c.GiB


def test_changed_pid_identity_fails_aggregate_supervision(envelope):
    envelope.resident = {(1, 1.): fake_process(1, 2., 1)}
    with pytest.raises(ValueError, match='PID identity'):
        envelope.aggregate_worker_rss()


def test_launch_has_fixed_diagnostic_caps_and_new_module():
    source = (c.ROOT/'tools/splendorust_timing_20261024/launch.py').read_text()
    assert 'tools.splendorust_timing_20261024.runner' in source
    assert 'tools.splendorust_search_scale_20261024.runner' not in source
    assert '21600' not in source and '19800' not in source
    assert c.JOB_SECONDS == 2700 and c.COLLECTION_CUTOFF_SECONDS == 2100
    assert c.STAGES == {'qualification': 600, 'profiling': 1500, 'final': 600}
    assert launch.c.OUT == c.OUT


def test_no_strength_or_optimization_launch_path():
    source = (c.ROOT/'tools/splendorust_timing_20261024/runner.py').read_text()
    assert 'strength_games=0' in source and 'diagnosis_only=True' in source
    assert 'play_game(' not in source
    assert 'automatic_retries=0' in source
    assert "q.model_free_checks(32)" in source


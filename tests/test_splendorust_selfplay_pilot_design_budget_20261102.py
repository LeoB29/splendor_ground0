"""Model-free prospective scope, split, resource and preservation regressions."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest

from tools.splendorust_selfplay_pilot_budget_20261102 import common as c, design, resources, preparation


@pytest.mark.parametrize('scope,count,seconds,models,evidence', [
    ('qualification', 4, 9000, 7200, 4), ('pilot', 256, 259200, 216000, 48)])
def test_actual_import_binds_scope_out_and_bounds_without_model(scope, count, seconds, models, evidence):
    code = ('import json,sys; from tools.splendorust_selfplay_pilot_budget_20261102 import common as c; '
            'print(json.dumps([c.SCOPE,c.MAX_SOURCE_GAMES,c.JOB_SECONDS,c.MODEL_CUTOFF_SECONDS,'
            'c.MAX_EVIDENCE_BYTES//c.GiB,str(c.OUT),"torch" in sys.modules]))')
    env = {**os.environ, 'SELFPLAY_JOB_SCOPE': scope, 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=c.ROOT, env=env,
                            capture_output=True, text=True, timeout=15,
                            creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value[:5] == [scope, count, seconds, models, evidence]
    assert value[-1] is False
    assert value[5].endswith('splendorust_selfplay_pilot_qualification_20261102_02' if scope == 'qualification'
                             else 'splendorust_selfplay_pilot_20261102_02')


def test_invalid_scope_cannot_silently_fall_back():
    result = subprocess.run([sys.executable, '-B', '-c',
        'from tools.splendorust_selfplay_pilot_budget_20261102 import common'], cwd=c.ROOT,
        env={**os.environ, 'SELFPLAY_JOB_SCOPE': 'resume', 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=15, creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
    assert result.returncode != 0 and 'SELFPLAY_JOB_SCOPE' in result.stderr


def test_independent_game_splits_and_all_scopes_have_fresh_disjoint_seeds():
    qualification, pilot = design.schedule('qualification'), design.schedule('pilot')
    assert [x['split'] for x in qualification] == ['fixture_train', 'fixture_train', 'fixture_validation', 'fixture_assessment']
    assert {split: sum(x['split'] == split for x in pilot) for split in ('train', 'validation', 'offline')} == {
        'train': 192, 'validation': 32, 'offline': 32}
    seeds = [x['seed_decimal'] for x in qualification + pilot]
    assert len(seeds) == len(set(seeds)) == 260
    assert all(x['teacher_simulations'] == 512 for x in qualification + pilot)
    assert all(x['purpose'] == 'single_generation_learning_pilot' for x in pilot)
    assert all(x['purpose'] == 'engineering_qualification_only' for x in qualification)
    assert design.schedule('pilot') == pilot


@pytest.fixture
def pilot_scope(monkeypatch):
    values = dict(SCOPE='pilot', MAX_SOURCE_GAMES=256, MAX_SOURCE_WORKER_LIFECYCLES=34,
        MAX_SOURCE_REPLAY_DECISIONS=48, MAX_ASSESSMENT_GAMES=406, MAX_ASSESSMENT_REPLAY_DECISIONS=96,
        MAX_REPLAY_DECISIONS=144, RAM_WAIT_SECONDS=10800, MODEL_CUTOFF_SECONDS=216000,
        JOB_SECONDS=259200, PREFINAL_EVIDENCE_BYTES=40*c.GiB, FINAL_RESERVE_BYTES=8*c.GiB,
        STAGES={'collection': 34*3600, 'fit': 4*3600, 'assessment': 20*3600})
    for k, v in values.items(): monkeypatch.setattr(c, k, v)


def test_rotations_have_exactly_eight_games_and_fixed_replay_panel(pilot_scope):
    blocks = design.collection_blocks()
    assert len(blocks) == 32 and blocks[0]['worker'] == 'capture_on' and blocks[-1]['worker'] == 'collect_031'
    assert [i for b in blocks for i in b['game_indices']] == list(range(256))
    assert all(len(b['specs']) == 8 for b in blocks)
    inventory = design.source_lifecycle_inventory()
    assert len(inventory) == 34 and inventory['capture_on']['ordinary_requests'] == 16016
    assert inventory['collect_031']['ordinary_requests'] == 16000
    assert inventory['capture_off']['capture_targets'] is False
    assert inventory['replay_on']['ordinary_requests'] == 48
    assert all(x['startup_requests'] == 2 and x['quota'] == 512 for x in inventory.values())
    assert design.replay_game_indices() == list(range(0, 256, 17))


def test_scope_spec_cannot_admit_legacy_seed_or_enriched_record():
    spec = design.schedule()[0]
    assert design.validate_spec(spec) == spec
    for bad in [dict(spec, seed_decimal='1'), dict(spec, split='train'), dict(spec, extra=True),
                dict(spec, game_index=True)]:
        with pytest.raises(ValueError): design.validate_spec(bad)


@pytest.mark.parametrize('value,expected', [
    ({'seconds': 10., 'ram_wait_seconds': 4.}, 6.), ({'seconds': 10.}, 10.),
    ({'seconds': 10., 'ram_wait_seconds': 4., 'nonwait_seconds': 6.}, 6.)])
def test_ram_is_removed_only_when_explicit_and_valid(value, expected):
    assert resources.nonwait_seconds(value) == expected


@pytest.mark.parametrize('value', [
    {'seconds': 10., 'ram_wait_seconds': 11.}, {'seconds': 10., 'ram_wait_seconds': -1.},
    {'seconds': 10., 'ram_wait_seconds': math.nan}, {'seconds': 0.}, {'nonwait_seconds': math.inf},
    {'nonwait_seconds': True}])
def test_invalid_measurement_is_never_a_discount(value):
    with pytest.raises(ValueError): resources.nonwait_seconds(value)


@pytest.fixture
def forecast_fixture(tmp_path, monkeypatch, pilot_scope):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'new')
    plan = dict(scope='pilot', factor=1.5,
        prices={name: dict(seconds=seconds, bytes=bytes_) for name, seconds, bytes_ in [
            ('source_game', 407.484, 75_372_237), ('source_worker', 145.5, 30_996_900),
            ('assessment_game', 70., 16_000_000), ('assessment_worker', 50., 20_000_000)]},
        fit_seconds_reserve=14400, fit_bytes_reserve=2*c.GiB, source_replay_seconds_each=11.,
        assessment_replay_seconds_each=11., replay_bytes_each=2*c.MiB,
        bookkeeping_seconds=1800, bookkeeping_bytes=2*c.GiB, assessment_normal_workers=84,
        source_collection_bookkeeping_seconds=600, final_seconds_reserve=43200)
    descriptor = c.save(c.OUT/'prices.json', plan)
    snapshot = dict(monotonic_now=360., unix_now=360., original_monotonic_origin=0., original_unix_origin=0.,
        stage_monotonic_deadline=122760., stage_unix_deadline=122760., accounted_bytes=128*c.MiB,
        cumulative_ram_wait_seconds=60., completed_games=0, qualified_source_workers=2)
    return dict(resource_plan=descriptor), snapshot, plan


def test_forecast_reserves_every_remaining_downstream_phase_once(forecast_fixture):
    freeze, snapshot, plan = forecast_fixture
    result = resources.source_forecast(freeze, snapshot)
    assert result['affordable'] and all(result['checks'].values())
    assert result['remaining_source_games'] == 256 and result['remaining_source_workers'] == 32
    assert result['reserved_remaining_ram_seconds'] == 10740
    expected = 256*407.484 + 32*145.5 + 48*11 + 600 + 10740
    assert result['projected_collection_seconds'] == pytest.approx(expected)
    assert result['projected_downstream_model_seconds'] == 14400 + 406*70 + 84*50 + 96*11 + 1800
    assert result['reserved_final_seconds'] == 43200 and result['reserved_final_bytes'] == 8*c.GiB
    assert result['outcomes_used'] is False and result['changes_to_sample_or_deadlines'] is False
    assert snapshot == result['snapshot']


def test_saved_forecast_uses_both_clocks_and_does_not_reset_on_block(forecast_fixture):
    freeze, snapshot, _ = forecast_fixture
    snapshot.update(completed_games=8, qualified_source_workers=2, unix_now=210000.)
    result = resources.source_forecast(freeze, snapshot)
    assert not result['affordable'] and not result['checks']['model'] and not result['checks']['collection']
    assert result['remaining_source_games'] == 248
    assert result['remaining_model_seconds'] == 6000.


def test_faster_live_games_cannot_reduce_frozen_prices(forecast_fixture):
    freeze, snapshot, _ = forecast_fixture
    snapshot['completed_games'] = 1
    result = resources.source_forecast(freeze, snapshot, source_costs=[dict(seconds=1., accounted_growth_bytes=1)])
    assert result['effective_prices']['source_game'] == dict(seconds=407.484, bytes=75_372_237)
    result = resources.source_forecast(freeze, snapshot, source_costs=[dict(seconds=1000., accounted_growth_bytes=10**9)])
    assert result['effective_prices']['source_game'] == dict(seconds=1500., bytes=1_500_000_000)
    assert not result['affordable']


@pytest.mark.parametrize('key,value', [('accounted_bytes', 40*1024**3),
    ('cumulative_ram_wait_seconds', 10801), ('stage_monotonic_deadline', 361.)])
def test_whole_fixed_sample_rejects_budget_or_clock_shortage(forecast_fixture, key, value):
    freeze, snapshot, _ = forecast_fixture
    snapshot[key] = value
    assert not resources.source_forecast(freeze, snapshot)['affordable']


def test_snapshot_rejects_unbound_counters_and_new_outcome_fields(forecast_fixture):
    freeze, snapshot, _ = forecast_fixture
    for bad in [dict(snapshot, score=.7), dict(snapshot, completed_games=True),
                dict(snapshot, qualified_source_workers=35), dict(snapshot, unix_now=math.nan)]:
        with pytest.raises(ValueError): resources.source_forecast(freeze, bad)
    with pytest.raises(ValueError):
        resources.source_forecast(freeze, snapshot, source_costs=[dict(seconds=1., accounted_growth_bytes=1)])


def test_forecast_authenticates_saved_price_file(forecast_fixture):
    freeze, snapshot, _ = forecast_fixture
    (c.ROOT/freeze['resource_plan']['path']).write_text('{}')
    with pytest.raises(ValueError): resources.source_forecast(freeze, snapshot)


def test_pilot_cannot_price_missing_new_matched128_preflight(tmp_path, monkeypatch, pilot_scope):
    monkeypatch.setattr(c, 'QUALIFICATION_OUT', tmp_path/'missing-new-preflight')
    with pytest.raises(FileNotFoundError): resources.resource_plan()


def test_pilot_prices_new_source_startups_and_keeps_old_maximum(monkeypatch, pilot_scope):
    def rows(paths):
        return [dict(seconds=300. if p.parent == c.QUALIFICATION_OUT/'startup_costs' else 100.,
                     bytes=3000 if p.parent == c.QUALIFICATION_OUT/'startup_costs' else 1000,
                     artifact=dict(path=p.as_posix(), sha256='test', bytes=1)) for p in paths]
    monkeypatch.setattr(resources, '_cost_rows', rows)
    result = resources.resource_plan()
    assert len(result['measured']['source_worker']) == 9
    assert result['prices']['source_worker'] == dict(seconds=450., bytes=4500)


def test_real_recovered_acceptance_stays_engineering_only():
    result = preparation.accepted_engineering()
    assert result['original_attempt_remains_incomplete'] is True
    assert result['recovered_acceptance']['sha256'] == c.RECOVERED_ACCEPTANCE_SHA
    assert result['independent_acceptance']['sha256'] == c.RECOVERED_INDEPENDENT_ACCEPTANCE_SHA


def test_new_pilot_cannot_substitute_old_recovered_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'QUALIFICATION_OUT', tmp_path)
    with pytest.raises(FileNotFoundError): preparation.accepted_new_entrypoint()


def test_exclusive_new_scope_writer_cannot_touch_old_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path); monkeypatch.setattr(c, 'OUT', tmp_path/'new')
    old = tmp_path/'old'/'receipt.json'; old.parent.mkdir(); old.write_bytes(b'unchanged')
    with pytest.raises(ValueError): c.save(old, {'new': True})
    assert old.read_bytes() == b'unchanged'
    c.save(c.OUT/'receipt.json', {'new': True})
    with pytest.raises(FileExistsError): c.save(c.OUT/'receipt.json', {'new': False})


def test_recipe_admission_is_bound_to_current_entrypoint_tensor_fixtures():
    source = Path(preparation.__file__).read_text()
    assert 'rglob(\'*.py\')' in source
    assert "input_qualification=dict(source='current_frozen_job'" in source
    assert "capture_fixtures/capture_on" in source and "proof_path=(c.OUT/'capture_qualification.json')" in source
    assert "training.direct_forward_budget(training_plan)" in source


def test_v2_identity_and_fresh_source_namespace_preserve_recipe():
    from tools.splendorust_selfplay_pilot_20261102 import design as stopped
    assert c.SCHEMA == 'splendorust_selfplay_pilot_v2'
    assert design.DOMAIN != stopped.DOMAIN
    old = {x['seed_decimal'] for scope in ('qualification', 'pilot') for x in stopped.schedule(scope)}
    new = {x['seed_decimal'] for scope in ('qualification', 'pilot') for x in design.schedule(scope)}
    assert len(old) == len(new) == 260 and not old.intersection(new)
    assert design.FIT_SEED == stopped.FIT_SEED == 202611012
    assert c.MAX_SOURCE_WORKER_LIFECYCLES == 3 and c.MAX_ASSESSMENT_WORKER_LIFECYCLES == 4
    source = Path(preparation.__file__).read_text()
    assert "glob('test_splendorust_selfplay_pilot*_budget_20261102.py')" in source
    assert 'failed_attempt_provenance()' in source


def test_failed_interval_measurements_do_not_qualify_or_reuse_training(monkeypatch):
    def rows(paths):
        return [dict(seconds=100., bytes=1000,
                     artifact=dict(path=p.as_posix(), sha256='test', bytes=1)) for p in paths]
    monkeypatch.setattr(resources, '_cost_rows', rows)
    value = resources.resource_plan()
    assert len(value['measured']['source_game']) == 8
    assert len(value['measured']['source_worker']) == 6
    assert value['failed_attempt_costs_are_qualification'] is False
    assert value['failed_attempt_training_examples_reused'] == 0
    assert not any('learned_entrypoint' in x['path'] for x in value['inputs'])
    assert 'same-domain' in value['source_time_provenance']


@pytest.fixture
def preserved_failure(tmp_path, monkeypatch):
    q, w, out = tmp_path/'failed_qualification', tmp_path/'failed_workflow', tmp_path/'new'
    q.mkdir(); w.mkdir(); out.mkdir()
    source = tmp_path/'old_source.py'; source.write_text('unchanged = True\n')
    for key, value in dict(ROOT=tmp_path, OUT=out, FAILED_QUALIFICATION_OUT=q,
                           FAILED_WORKFLOW_OUT=w, FAILED_PRESERVATION=out/'preserved.json').items():
        monkeypatch.setattr(c, key, value)
    (q/'completion.json').write_text(json.dumps(dict(status='incomplete')))
    (q/'freeze.json').write_text(json.dumps(dict(inputs=[c.artifact(source)])))
    (w/'failure.json').write_text(json.dumps(dict(status='stopped')))
    monkeypatch.setattr(c, 'FAILED_FREEZE_SHA', c.file_hash(q/'freeze.json'))
    inputs = [c.artifact(source)]
    trees = []
    for root in [q, w]:
        rows = [c.artifact(p) for p in sorted(root.rglob('*')) if p.is_file()]
        inputs.extend(rows)
        trees.append(dict(root=root.relative_to(tmp_path).as_posix(), files=len(rows),
                          bytes=sum(x['bytes'] for x in rows), paths=[x['path'] for x in rows]))
    value = dict(schema='selfplay_failed_attempt_preservation_v1', original_status='incomplete',
        full_pilot_started=False, training_examples_reused=0, checkpoint_reused=False,
        retrospective_acknowledgements=0, original_freeze=c.artifact(q/'freeze.json'),
        trees=trees, inputs=inputs)
    descriptor = c.save(c.FAILED_PRESERVATION, value)
    return descriptor, value, source, q, w


def test_complete_failed_inventory_and_original_sources_are_verified(preserved_failure):
    descriptor, value, *_ = preserved_failure
    assert c.verify_failed_attempt_preservation(descriptor) == value


@pytest.mark.parametrize('mutation', ['source', 'extra_file', 'missing_file', 'accepted'])
def test_failed_inventory_cannot_be_mutated_or_retroactively_accepted(preserved_failure, mutation):
    descriptor, value, source, q, w = preserved_failure
    if mutation == 'source': source.write_text('changed = True\n')
    elif mutation == 'extra_file': (w/'new_ack.json').write_text('{}')
    elif mutation == 'missing_file': (w/'failure.json').unlink()
    else: (q/'qualification_acceptance.json').write_text('{}')
    with pytest.raises(ValueError): c.verify_failed_attempt_preservation(descriptor)


def test_engineering_acceptance_is_not_new_clock_qualification():
    value = preparation.accepted_engineering()
    assert value['new_clock_path_qualified'] is False
    assert 'not certification' in value['role']

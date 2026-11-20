from copy import deepcopy

import pytest

from tools.matched_strength_recovery_20261022 import completion


def report():
    return {'status': 'passed_amended', 'final_acceptance': False,
        'outcomes_opened': True, 'freeze_sha256': 'frozen',
        'strength_analysis_games': 60, 'completed_games': 60, 'completed_pairs': 30,
        'statistics': {}, 'study_label': 'amended_recovered_benchmark',
        'original_frozen_status': 'incomplete',
        'amendment': {'retry_identity': 'games/g059/t0043', 'additional_attempts_authorized': 1,
            'fresh_seconds': 120, 'original_partial_preserved': True},
        'original_missing_game_sensitivity': {'missing_game': 59,
            'maximum_point_estimate_width': 1/60}}


def test_only_separately_labelled_complete_amended_report_is_accepted(monkeypatch):
    monkeypatch.setattr(completion.c, 'config', lambda: {'pair_count': 30})
    completion.verify_report(report(), 'frozen')


@pytest.mark.parametrize('field,value', [
    ('status', 'passed'), ('study_label', 'original_frozen_benchmark'),
    ('original_frozen_status', 'complete'), ('freeze_sha256', 'changed'),
    ('completed_games', 59), ('completed_pairs', 29), ('outcomes_opened', False),
    ('final_acceptance', True), ('strength_analysis_games', 59),
])
def test_rejects_scope_or_provenance_relaxation(monkeypatch, field, value):
    monkeypatch.setattr(completion.c, 'config', lambda: {'pair_count': 30})
    candidate = report(); candidate[field] = value
    with pytest.raises(ValueError):
        completion.verify_report(candidate, 'frozen')


@pytest.mark.parametrize('field,value', [
    ('retry_identity', 'games/g039/t0007'), ('additional_attempts_authorized', 2),
    ('fresh_seconds', 121), ('original_partial_preserved', False),
])
def test_rejects_timeout_retry_or_extra_exception(monkeypatch, field, value):
    monkeypatch.setattr(completion.c, 'config', lambda: {'pair_count': 30})
    candidate = deepcopy(report()); candidate['amendment'][field] = value
    with pytest.raises(ValueError):
        completion.verify_report(candidate, 'frozen')


def test_rejects_missing_original_game_sensitivity(monkeypatch):
    monkeypatch.setattr(completion.c, 'config', lambda: {'pair_count': 30})
    candidate = report(); candidate['original_missing_game_sensitivity']['missing_game'] = 39
    with pytest.raises(ValueError):
        completion.verify_report(candidate, 'frozen')


@pytest.mark.parametrize('damage',['old_wall','old_mono','new_wall','new_mono','reset','reserve'])
def test_file_guard_retains_both_dual_clocks_and_analysis_reserve(tmp_path,monkeypatch,damage):
    c=completion.c;monkeypatch.setattr(c,'OUTPUT',tmp_path/'new');monkeypatch.setattr(c,'SOURCE',tmp_path/'old')
    c.save_new(c.SOURCE/'launch.json',{'epoch_seconds':1000.,'monotonic_start':100.,'maximum_seconds':259200})
    c.save_new(c.OUTPUT/'launch.json',{'epoch_seconds':2000.,'monotonic_start':1100.,'maximum_seconds':43200})
    mono,epoch=1101.,2001.
    if damage=='old_wall':epoch=260200.
    elif damage=='old_mono':mono=259300.
    elif damage=='new_wall':epoch=45200.
    elif damage=='new_mono':mono=44300.
    elif damage=='reset':mono=99.
    elif damage=='reserve':mono=43700.
    monkeypatch.setattr(completion.time,'time',lambda:epoch)
    monkeypatch.setattr(completion.time,'monotonic',lambda:mono)
    with pytest.raises((ValueError,c.ResourceHold)):completion.make_clock_guard(44300.,reserve=600)()


def test_verify_stops_immediately_when_hash_consumes_remaining_time(monkeypatch):
    calls=[]
    item={'path':'synthetic','sha256':'a'*64,'bytes':1}
    def guard():
        if calls:raise completion.c.ResourceHold('expired after first hash')
    monkeypatch.setattr(completion.c,'artifact',lambda path:calls.append(str(path)) or item)
    with pytest.raises(completion.c.ResourceHold):completion.verify_files([item,item],guard)
    assert len(calls)==1


def test_inventory_checks_clock_between_files_and_retains_live_log_exclusions(tmp_path,monkeypatch):
    for name in ('first.json','second.json','runner.stdout.log'):(tmp_path/name).write_text('{}')
    calls=[]
    monkeypatch.setattr(completion.c,'artifact',lambda path:calls.append(path.name) or {'path':path.name})
    def guard():
        if len(calls)==1:raise completion.c.ResourceHold('expired between files')
    with pytest.raises(completion.c.ResourceHold):completion.final_artifact_inventory(tmp_path,guard)
    assert calls==['first.json']
    calls.clear()
    result=completion.final_artifact_inventory(tmp_path,lambda:None)
    assert result==[{'path':'first.json'},{'path':'second.json'}]

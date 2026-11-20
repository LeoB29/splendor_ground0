"""State-machine and independent action-check regressions; no opponent inference."""
from copy import deepcopy
import json
import pytest

from tools.splendorust_120_qualification_20261018 import runner as r
from tools.analyze_external_strength_screen_20261018_02 import main_action


def completed_fixture():
    state = r.rules.CorrectedEnv(seed=41).initial_state()
    snapshot = r.rules.serialize_snapshot(state)
    observer = r.rules.observation_payload(state)
    legal = r.rules.legal_actions(state)
    actions = [r.c.serialize_action(a) for a in legal]
    action = next(a for a in actions if a['action_type'] == 'TAKE_TOKENS')
    profile = r.c.make_profile(256)
    result = {'status': 'completed', 'action': action, 'player': 'opponent',
        'observer_sha256': r.io.digest(observer), 'legal_actions_sha256': r.c.legal_digest(legal),
        'legal_count': len(legal), 'ruleset': r.rules.RULESET_ID,
        'catalogue_sha256': r.rules.CATALOGUE_SHA256, 'profile_sha256': 'a'*64,
        'diagnostics': {'opponent_id': 'splendorust', 'profile_id': profile['profile_id'],
            'observer_seed': int(r.io.digest(observer)[:16], 16), 'forced_pass': False,
            'inference_calls': 257, 'inference_rows': 257,
            'phases': [{'action': main_action(action), 'simulations': 256, 'leaf_calls': 257,
                'native_legal_count': len({r.io.canonical(main_action(a)) for a in actions}),
                'heuristic_phase': False}]}}
    return snapshot, observer, profile, result


def test_completed_result_uses_independent_dictionary_successor():
    snapshot, observer, profile, result = completed_fixture()
    after = r.check_result(result, observer, profile, 'a'*64, snapshot)
    assert after == r.audit.transition(snapshot, result['action'])
    assert after['state']['turn_index'] == 1


@pytest.mark.parametrize('damage', ['observer', 'profile', 'legal_digest', 'illegal', 'quota', 'inferences', 'extra', 'state'])
def test_completed_result_rejects_wrong_identity_work_or_transition(damage):
    snapshot, observer, profile, result = completed_fixture()
    if damage == 'observer': result['observer_sha256'] = '0'*64
    elif damage == 'profile': result['profile_sha256'] = '0'*64
    elif damage == 'legal_digest': result['legal_actions_sha256'] = '0'*64
    elif damage == 'illegal': result['action']['take_tokens'] = ['gold']
    elif damage == 'quota': result['diagnostics']['phases'][0]['simulations'] = 255
    elif damage == 'inferences': result['diagnostics']['inference_calls'] = 999
    elif damage == 'extra': result['partial'] = True
    else: snapshot['state']['bank_tokens']['white'] = 5
    with pytest.raises(ValueError): r.check_result(result, observer, profile, 'a'*64, snapshot)


def test_exhaustion_never_supplies_partial_choice():
    snapshot, observer, profile, result = completed_fixture()
    del result['action']; del result['diagnostics']
    result.update(status='internal_budget_exhausted', internal_budget_seconds=108,
        opponent_id='splendorust', profile_id=profile['profile_id'], reason='Internal fixed-work deadline exceeded')
    assert r.check_result(result, observer, profile, 'a'*64, snapshot) is None
    result['action'] = {}
    with pytest.raises(ValueError): r.check_result(result, observer, profile, 'a'*64, snapshot)


@pytest.mark.parametrize('receipt,correct,status,wanted', [(96,True,'ok',True), (96.001,True,'ok',False),
    (1,False,'ok',False), (None,False,'timeout',False)])
def test_margin_uses_complete_receipt_not_validation_time(receipt, correct, status, wanted):
    assert r.margin_passed({'correct': correct, 'clock': {'status': status,
        'receipt_elapsed_seconds': receipt, 'elapsed_seconds': 200}}) is wanted


def fake_run(tmp_path, monkeypatch, reject_quota=None, reject_ordinal=0, final_timeout=None,
             final_mismatch=False, hard_failure=False):
    base = tmp_path / 'study'; base.mkdir()
    monkeypatch.setattr(r, 'BASE', base)
    monkeypatch.setattr(r.io, 'ROOT', tmp_path)
    monkeypatch.setattr(r.io, 'boundary', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(r, 'verify', lambda *_: None)
    r.io.save_new(base / 'freeze.json', {'inputs': [], 'topology': {'selected_logical_processor': 0}})
    for quota in r.GRID: r.io.save_new(base / 'profiles' / f'q{quota:05d}.json', {'iterations': quota})
    r.io.save_new(base / 'qualification_schedule.json', [{'index':2*i+j,'fixture':i,'kind':kind}
        for i in range(23) for j,kind in enumerate(('original','hidden_variant'))])
    calls = []
    def attempt(stage, name, quota, fixture, kind, *_):
        calls.append((stage, name, quota, fixture, kind))
        timeout = stage == 'calibration' and quota == reject_quota and int(name.rsplit('_',1)[1]) == reject_ordinal
        timeout |= stage == 'qualification' and int(name) == final_timeout
        row = {'stage':stage,'quota':quota,'fixture':fixture,'kind':kind,'correct':not timeout,
            'clock':{'status':'timeout' if timeout else 'ok','receipt_elapsed_seconds':None if timeout else 50.}}
        if not timeout: row['result'] = {'fixture':fixture,'quota':quota,'value':kind if final_mismatch and stage == 'qualification' else 'same'}
        r.io.save_new(base / 'rows' / stage / (name+'.json'), row)
        if hard_failure: raise ValueError('Synthetic protocol failure')
        return row
    monkeypatch.setattr(r, 'one_attempt', attempt)
    # The real controller excludes Torch. Other test files may have imported it.
    monkeypatch.delitem(r.sys.modules, 'torch', raising=False)
    code = r.run()
    return base, calls, code, r.io.read(base / 'qualification_result.json')


def test_largest_complete_calibration_selected_and_all_pairs_fixed(tmp_path, monkeypatch):
    base, calls, code, result = fake_run(tmp_path, monkeypatch)
    assert code == 0 and result['qualified'] and result['privacy_pairs_matched'] == 23
    assert len(calls) == 74 and result['selected_quota'] == 6400
    assert r.io.read(base/'selected_profile.json') == {'iterations':6400}


def test_first_timing_rejection_keeps_previous_quota_and_never_tests_higher(tmp_path, monkeypatch):
    base, calls, code, result = fake_run(tmp_path, monkeypatch, reject_quota=4096, reject_ordinal=2)
    assert code == 0 and result['qualified'] and result['selected_quota'] == 1024
    assert result['calibration_attempted'] == 17 and result['qualification_attempted'] == 46
    assert all(c[2] != 6400 for c in calls)
    assert r.io.read(base/'calibration_selection.json')['attempts'][-1]['passed_margin'] is False


def test_no_calibration_pass_is_completed_negative_without_final_requests(tmp_path, monkeypatch):
    base, calls, code, result = fake_run(tmp_path, monkeypatch, reject_quota=256)
    assert code == 0 and result['complete_schedule'] and not result['qualified']
    assert len(calls) == 1 and result['qualification_attempted'] == 0 and not (base/'selected_profile.json').exists()


def test_final_timing_failure_does_not_downshift_or_replace_and_finishes_panel(tmp_path, monkeypatch):
    base, calls, code, result = fake_run(tmp_path, monkeypatch, final_timeout=0)
    assert code == 0 and result['complete_schedule'] and not result['qualified']
    assert result['qualification_attempted'] == 46 and result['verified_decisions'] == 45
    assert result['privacy_pairs_matched'] == 22 and result['selected_quota'] == 6400


@pytest.mark.parametrize('damage', ['mismatch','hard'])
def test_scientific_or_protocol_failure_stops_and_is_preserved(tmp_path, monkeypatch, damage):
    base, calls, code, result = fake_run(tmp_path, monkeypatch, final_mismatch=damage=='mismatch', hard_failure=damage=='hard')
    assert code == 1 and not result['complete_schedule'] and not result['qualified']
    assert (base/'failure.json').exists()
    assert r.io.read(base/'completion.json')['status'] == 'preserved_failure'
    assert result['qualification_attempted'] == (2 if damage=='mismatch' else 0)

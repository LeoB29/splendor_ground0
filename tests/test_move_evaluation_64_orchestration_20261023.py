"""Exercise the real bounded parent schedule with synthetic worker responses.

No model entry point is called. These tests target orchestration interfaces and
failure barriers; separate tests cover the actual scientific implementations.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.move_evaluation_64_20261023 import runner as r, checks, assessment
from tools.external_strength_screen_20261018_02 import runner as privacy


@pytest.fixture
def execution(monkeypatch, tmp_path):
    c = r.c
    prepared = c.read(c.PREP/'selection_lock.json')
    roots = []
    for row in prepared['roots']:
        root = deepcopy(row['payload'])
        root.update({key: row[key] for key in ('id', 'fit', 'game', 'seat', 'phase')})
        roots.append(root)
    locked = {'roots': roots}
    by_id = {row['id']: row for row in roots}
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'run')
    c.save(c.OUT/'sealed/panel_states.json',
           {row['id']: {'id': row['id'], 'hidden': False} for row in roots})
    events, control = [], {}
    envelope = SimpleNamespace(start=0., pilot_end=3600., qual_end=10800.,
                               guard=lambda deadline: events.append(('guard', deadline)))
    monkeypatch.setattr(r, 'decode_training_snapshot', deepcopy)
    monkeypatch.setattr(r.rules, 'serialize_snapshot', deepcopy)
    monkeypatch.setattr(r.rules, 'observation_payload',
                        lambda state: r.external_observer(by_id[state['id']]['observer']))
    monkeypatch.setattr(r.dictionary, 'check_observer', lambda *args: None)
    monkeypatch.setattr(privacy, 'hidden_variant', lambda state: {**state, 'hidden': True})

    class Worker:
        def __init__(self, name, mode, deadline, arm):
            self.name, self.mode, self.deadline, self.arm = name, mode, deadline, arm
            self.output_dir = c.OUT/'workers'/name
            self.requests = 0

        def __enter__(self):
            events.append(('open', self.name, self.mode, self.arm, self.deadline))
            return self

        def __exit__(self, *_):
            events.append(('close', self.name))

    def make_worker(name, mode, env, deadline, freeze, *, arm=None):
        assert env is envelope
        return Worker(name, mode, deadline, arm)

    def request(child, payload, seconds, deadline):
        child.requests += 1
        assert child.requests == 1, 'Fresh qualification, selector and replay workers'
        assert deadline == child.deadline
        events.append(('request', child.name, child.mode, seconds, deadline))
        elapsed = 10.
        if child.mode == 'qualify':
            assert payload == {'operation': 'entrypoint_checks'}
            assert seconds == 300 and deadline == envelope.pilot_end
            result = {'status': 'failed' if control.get('qualification') == child.arm
                      else 'entrypoint_counterexamples_passed', 'arm': child.arm}
        elif child.mode == 'select':
            ordinal = int(child.name.split('/q')[1][:2])
            entry = prepared['timed_request_order'][ordinal]
            row = roots[entry['index']]
            assert payload == c.selector_payload(row), 'Only lawful selector inputs are sent'
            assert seconds == 120
            assert deadline == (envelope.pilot_end if ordinal < 12 else envelope.qual_end)
            choice = 0 if child.arm == 'A' else 1
            result = {'observer_sha256': c.digest(payload['observer']),
                      'decision': {'action': row['candidates']['actions'][choice],
                                   'record': {'choice': choice, 'means': [0.]*len(row['candidates']['actions'])}},
                      'trace': {'path': f'synthetic/{child.name}.gz',
                                'sha256': c.digest([row['id'], child.arm]), 'bytes': 1,
                                'records': 64*len(row['candidates']['actions'])}}
            if control.get('margin') == ordinal:
                elapsed = 96.00001
        else:
            assert child.mode == 'audit_selection'
            assert set(payload) == {'selector', 'result', 'trace'}
            assert payload['trace'] == payload['result']['trace']
            assert set(payload['trace']) == {'path', 'sha256', 'bytes', 'records'}
            assert seconds == 300
            result = {'status': 'failed' if control.get('replay') == child.name else 'passed',
                      'audited_continuations': payload['trace']['records']}
        receipt = {'request_id': 'request-00000001', 'status': 'ok', 'accepted': True,
                   'receipt_elapsed_seconds': elapsed, 'result': result}
        c.save(child.output_dir/'request-00000001.json', receipt)
        return receipt

    def check_selection(result, payload, arm):
        events.append(('checked', result['trace']['path']))
        return {'status': 'passed', 'scientific_replay_still_required': True}

    def prefix(row, result, arm, index):
        events.append(('prefix', index, arm))
        if control.get('prefix') == (index, arm):
            raise ValueError('Synthetic prefix failure')
        return {'status': 'passed'}

    def duplicate(result, prior):
        ordinal = int(result['trace']['path'].split('/q')[1][:2])
        events.append(('duplicate', ordinal))
        if control.get('duplicate') == ordinal:
            raise ValueError('Synthetic duplicate failure')
        left, right = deepcopy(result), deepcopy(prior)
        del left['trace']['path']; del right['trace']['path']
        assert left == right

    def qualify_assessor(env, freeze, make, req):
        events.append(('assessor_qualification',))
        assert (c.OUT/'sealed/choices.json').exists()
        assert (c.OUT/'sealed/evaluated_parent_lock.json').exists()
        assert not (c.OUT/'qualification_complete.json').exists()
        if control.get('assessor'):
            raise ValueError('Synthetic assessor failure')

    monkeypatch.setattr(r, 'make_worker', make_worker)
    monkeypatch.setattr(r, 'request', request)
    monkeypatch.setattr(checks, 'check_selection', check_selection)
    monkeypatch.setattr(r, 'prefix_check', prefix)
    monkeypatch.setattr(r, 'duplicate_check', duplicate)
    monkeypatch.setattr(assessment, 'qualify_assessor', qualify_assessor)
    return SimpleNamespace(c=c, locked=locked, events=events, control=control,
                           envelope=envelope, prepared=prepared)


def test_actual_parent_schedule_replays_pilot_before_expansion_and_seals_after_all_checks(execution):
    x = execution
    choices = r.selection(x.envelope, {}, x.locked)
    requests = [event for event in x.events if event[0] == 'request']
    selects = [event for event in requests if event[2] == 'select']
    audits = [event for event in requests if event[2] == 'audit_selection']
    assert [event[1] for event in requests[:2]] == ['entrypoint_A', 'entrypoint_B']
    assert len(selects) == 54 and len(audits) == 36
    assert [event[1] for event in requests[14:20]] == [
        f'audit/r{index:02d}_{arm}' for index in (0, 6, 12) for arm in ('A', 'B')]
    assert requests[20][1].startswith('selection/q12_')
    for event in audits:
        index = int(event[1].split('/r')[1][:2])
        assert event[4] == (3600. if index in (0, 6, 12) else 10800.)
    privacy = x.prepared['timed_request_order'][42:]
    assert len(privacy) == 12 and all(entry['role'] == 'privacy_repeat' for entry in privacy)
    assert len([event for event in x.events if event[0] == 'duplicate']) == 18
    assert len(choices) == 18
    proof = x.c.read(x.c.OUT/'qualification_complete.json')
    assert proof['fully_replayed_trajectories'] == 9088 and proof['timed_requests'] == 54
    assert proof['choices'] == x.c.artifact(x.c.OUT/'sealed/choices.json')
    assert proof['lock'] == x.c.artifact(x.c.OUT/'sealed/evaluated_parent_lock.json')
    assert x.events.index(('assessor_qualification',)) > x.events.index(selects[-1])
    assert all(event[0] != 'open' or event[2] != 'assess' for event in x.events)


@pytest.mark.parametrize('ordinal', [0, 6, 42])
def test_margin_failure_preserves_complete_response_and_blocks_dependent_work(execution, ordinal):
    x = execution; x.control['margin'] = ordinal
    with pytest.raises(ValueError, match='96-second margin'):
        r.selection(x.envelope, {}, x.locked)
    selections = [event for event in x.events if event[0] == 'request' and event[2] == 'select']
    assert len(selections) == ordinal+1
    saved = x.c.read(x.c.OUT/'selections'/f'q{ordinal:02d}.json')
    assert saved['complete_response_seconds'] == 96.00001 and saved['margin_passed'] is False
    assert not (x.c.OUT/'qualification_complete.json').exists()
    assert not (x.c.OUT/'sealed/choices.json').exists()
    assert ('assessor_qualification',) not in x.events


@pytest.mark.parametrize('failure,expected_selects', [
    ('qualification', 0), ('prefix', 2), ('duplicate', 7), ('replay', 12)])
def test_failed_prerequisite_stops_at_exact_prefix(execution, failure, expected_selects):
    x = execution
    x.control[failure] = {'qualification': 'B', 'prefix': (0, 'B'),
                          'duplicate': 6, 'replay': 'audit/r00_A'}[failure]
    with pytest.raises(ValueError):
        r.selection(x.envelope, {}, x.locked)
    selects = [event for event in x.events if event[0] == 'request' and event[2] == 'select']
    assert len(selects) == expected_selects
    assert not (x.c.OUT/'pilot_complete.json').exists()
    assert not (x.c.OUT/'qualification_complete.json').exists()
    assert not (x.c.OUT/'sealed/choices.json').exists()
    assert ('assessor_qualification',) not in x.events


def test_assessor_failure_retains_sealed_choices_but_never_admits_assessment(execution):
    x = execution; x.control['assessor'] = True
    with pytest.raises(ValueError, match='assessor failure'):
        r.selection(x.envelope, {}, x.locked)
    assert (x.c.OUT/'sealed/choices.json').exists()
    assert not (x.c.OUT/'qualification_complete.json').exists()
    assert len([e for e in x.events if e[0] == 'request' and e[2] == 'select']) == 54
    assert len([e for e in x.events if e[0] == 'request' and e[2] == 'audit_selection']) == 36

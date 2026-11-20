"""Model-free counterexamples for narrowly routed saved-only final acceptance."""
import json
from pathlib import Path
import pytest

from . import common as c, audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def roots(tmp_path, monkeypatch):
    source = tmp_path/'source'; out = tmp_path/'out'
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'SOURCE', source)
    monkeypatch.setattr(c, 'OUT', out)
    return source, out


def good_exit():
    return dict(status='passed', worker_exit_code=0, cleanup_errors=[], error=None,
                identity=dict(pid=1, created=2.), observed_owned_identities=[])


def roles(source):
    return [source/'segments'/name for name in ('saved_mixed_terminal_r1', *audit.REPLACEMENTS)]


def make_inventory(source, out):
    expected = roles(source)
    for path in expected:
        if path.name not in audit.REPLACEMENTS:
            put(path/'execution.json', good_exit())
    for name in audit.REPLACEMENTS:
        put(out/'jobs'/name/'execution.json', good_exit())
    (out/'jobs/final_acceptance/worker').mkdir(parents=True)
    return audit.execution_routes(expected)


def test_exact_two_missing_roles_route_to_new_jobs_only(roots):
    source, out = roots
    rows = audit.execution_routes(roles(source))
    assert [row['source'] for row in rows] == ['preserved_recovery', 'finalization', 'finalization']
    assert rows[0]['path'] == source/'segments/saved_mixed_terminal_r1/execution.json'
    assert {row['path'] for row in rows[1:]} == {out/'jobs'/name/'execution.json' for name in audit.REPLACEMENTS}


def test_completed_reader_cannot_be_replaced(roots):
    source, out = roots
    changed = {name: out/'jobs'/name/'execution.json' for name in audit.REPLACEMENTS}
    changed['saved_mixed_terminal_r1'] = out/'jobs/saved_mixed_terminal_r1/execution.json'
    with pytest.raises(ValueError, match='two named'):
        audit.execution_routes(roles(source), changed)


def test_missing_original_role_cannot_silently_shrink_coverage(roots):
    source, _ = roots
    with pytest.raises(ValueError, match='Both unfinished'):
        audit.execution_routes(roles(source)[:-1])


def test_complete_expected_execution_inventory(roots):
    source, out = roots
    audit.verify_execution_inventory(make_inventory(source, out))


@pytest.mark.parametrize('missing', ['old', 'new'])
def test_missing_success_receipt_rejects_final_acceptance(roots, missing):
    source, out = roots; routes = make_inventory(source, out)
    path = routes[0 if missing == 'old' else 1]['path']; path.unlink()
    with pytest.raises(ValueError): audit.verify_execution_inventory(routes)


def test_extra_execution_is_not_an_authorized_replacement(roots):
    source, out = roots; routes = make_inventory(source, out)
    put(out/'jobs/extra_saved_reader/execution.json', good_exit())
    with pytest.raises(ValueError, match='Exactly two'):
        audit.verify_execution_inventory(routes)


@pytest.mark.parametrize('key,value', [('worker_exit_code', None), ('worker_exit_code', True),
    ('worker_exit_code', 1), ('status', 'pending'), ('cleanup_errors', ['still alive']), ('error', 'bad')])
def test_completion_file_or_unknown_exit_does_not_count_as_passed(key, value):
    receipt = good_exit(); receipt[key] = value
    with pytest.raises(ValueError): audit.passed_execution(receipt)


def interrupted_source(source):
    put(source/'claim.json', dict(pid=62252, created=1794362749.6136715))
    put(source/'segments'/audit.REPLACEMENTS[0]/'owner.json',
        dict(identity=dict(pid=39812, created=1794501812.5410624)))


def test_unknown_source_root_exit_is_truthfully_preserved(roots):
    source, _ = roots; interrupted_source(source); observed = []
    result = audit.source_interruption(observed.append)
    assert result['status'] == 'incomplete_preserved'
    assert result['actual_root_exit_code'] is None
    assert result['actual_root_exit_observed'] is False
    assert result['clean_root_exit_claimed'] is False
    assert [identity['pid'] for identity in observed] == [62252, 39812]


@pytest.mark.parametrize('path', ['execution.json', 'completion.json', 'failure.json',
    'segments/independent_mixed_terminal_r1/execution.json',
    'segments/independent_mixed_terminal_r1/result.json'])
def test_source_exit_or_missing_reader_cannot_be_fabricated(roots, path):
    source, _ = roots; interrupted_source(source); put(source/path, good_exit())
    with pytest.raises(ValueError): audit.source_interruption(lambda identity: None)


def test_dead_pid_without_exact_absence_does_not_admit_finalization(roots):
    source, _ = roots; interrupted_source(source)
    def still_alive(identity): raise ValueError('exact identity still alive')
    with pytest.raises(ValueError, match='still alive'): audit.source_interruption(still_alive)


def test_changed_frozen_source_is_rejected(roots):
    source, _ = roots; path = source/'saved.json'; put(path, dict(status='passed'))
    item = c.artifact(path); c.verify([item])
    put(path, dict(status='failed'))
    with pytest.raises(ValueError): c.verify([item])


@pytest.mark.parametrize('mutation', ['edit', 'add', 'remove'])
def test_complete_source_inventory_detects_every_mutation(roots, mutation):
    source, out = roots; path = source/'saved.json'; put(path, dict(status='passed'))
    out.mkdir(parents=True)
    inventory = out/'source_inventory.jsonl'
    inventory.write_text(json.dumps(c.artifact(path))+'\n', encoding='utf-8')
    plan = dict(source_inventory=c.artifact(inventory))
    accepted = c.verify_source_inventory(plan)
    assert accepted['files'] == 1 and accepted['bytes'] == path.stat().st_size
    if mutation == 'edit': put(path, dict(status='changed'))
    elif mutation == 'add': put(source/'extra.json', dict(status='extra'))
    else: path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        c.verify_source_inventory(plan)


def test_family_routes_only_missing_group_acceptance_to_new_output(roots):
    source, out = roots
    assert audit.acceptance_path('mixed_terminal') == out/'strength/mixed_terminal/report/independent_acceptance.json'
    assert audit.acceptance_path('mixed_parent') == source/'strength/mixed_parent/report/independent_acceptance.json'
    assert audit.acceptance_path('terminal_parent') == source/'strength/terminal_parent/report/independent_acceptance.json'

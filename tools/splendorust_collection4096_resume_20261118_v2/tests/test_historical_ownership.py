"""Saved PID-reuse counterexamples; no live process is launched or terminated."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from tools.splendorust_collection4096_resume_20261118_v2 import historical_ownership as h


def chain():
    return [dict(identity=identity, name='Discord.exe', exe=h.EXPECTED_IMAGE,
        ppid=35020 if n == 0 else 9908, live_identity_matched=True)
        for n, identity in enumerate(h.exception_identities())]


def test_actual_psutil_recursive_bug_on_saved_impossible_parentage():
    result = h.reproduce_installed_ancestry_bug()
    assert result['false_inclusion_reproduced'] is True
    assert {pid for pid, _ in h.FOREIGN} <= set(result['actual_recursive_result'])
    assert result['impossible_immediate_parent_edges'][0]['child'] == 9908
    assert result['process_launches'] == result['model_calls'] == 0


def test_exact_saved_foreign_chain_accepted():
    h.validate_foreign_chain(chain())


@pytest.mark.parametrize('change', [
    lambda rows: rows[0]['identity'].update(created=rows[0]['identity']['created'] + 1),
    lambda rows: rows[0].update(ppid=29196),
    lambda rows: rows[1].update(ppid=35020),
    lambda rows: rows[1].update(exe='C:/unknown/Discord.exe'),
    lambda rows: rows.append(deepcopy(rows[-1])),
])
def test_no_pid_name_or_broad_parentage_exception(change):
    rows = chain(); change(rows)
    with pytest.raises(ValueError): h.validate_foreign_chain(rows)


def test_live_worker_identity_never_excluded_by_same_application_name():
    rows = chain(); rows[0]['identity'] = dict(pid=26260, created=1794929941.0990553)
    with pytest.raises(ValueError): h.validate_foreign_chain(rows)


def test_all_authoritative_saved_worker_identities_absent_and_foreign_disjoint():
    history = h.authenticated_history()
    assert history['authenticated_worker_executions'] == 192
    assert len(history['authoritative_worker_identities']) == 4898
    assert set(h.FOREIGN).isdisjoint(h.key(x) for x in history['authoritative_worker_identities'])


def test_actual_persisted_proof_json_roundtrip():
    proof = h.c.read(h.PROOF)
    assert h.validate_proof(proof, require_frozen=False) == h.exception_identities()


def test_true_worker_still_alive_remains_a_hard_failure():
    with patch.object(h, 'require_absent', side_effect=ValueError('exact real worker alive')):
        with pytest.raises(ValueError, match='exact real worker alive'):
            h.authenticated_history()

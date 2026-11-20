"""Exact historical PID-reuse diagnosis; never grants an application-wide exception.

The prior long-lived observer used root-relative psutil ancestry. Its six
foreign observations are retained, but are not evidence of Windows-job ownership.
Every authenticated worker identity still has to be absent. No process is killed.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil

from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent

PROOF = c.OUT / 'prelaunch/historical_ownership01.json'
FOREIGN = (
    (9908, 1794917493.9600725),
    (24532, 1794917494.3812823),
    (47168, 1794917494.5134084),
    (40728, 1794917494.567165),
    (46384, 1794917496.0325174),
    (44532, 1794917497.4329777),
)
REUSED_PARENT = ((35020, 1794929941.1094468), (35020, 1794939565.5650935))
EXPECTED_IMAGE = r'C:\Users\leoba\AppData\Local\Discord\app-1.0.9268\Discord.exe'


def key(identity):
    return identity['pid'], identity['created']


def exception_identities():
    return [dict(pid=pid, created=birth) for pid, birth in FOREIGN]


def worker_containers():
    base = c.PREVIOUS_OUT / 'collection'
    values = [base / name for name in ('recovery_boundary', 'recovery_qualification',
        'recovery_qualification_audit', 'recovery_replays', 'recovery_audit')]
    for index in range(42, 135):
        values.extend(base / kind / f'c{index:03d}' for kind in ('cells', 'cell_audits'))
    return values + [base / 'cells/c135']


def authenticated_history(*, check_absence=True):
    c.require(c.sha(c.PREVIOUS_OUT/'freeze.json') == c.PREVIOUS_FREEZE_SHA, 'Exact predecessor freeze')
    root_execution_path = c.PREVIOUS_OUT/'execution.json'
    root_execution = c.read(root_execution_path)
    launch_path = c.PREVIOUS_OUT/'launch/execution.json'; launch = c.read(launch_path)
    root_claim_path = c.PREVIOUS_OUT/'claim.json'; root_claim = c.read(root_claim_path)
    launcher_claim_path = c.PREVIOUS_OUT/'launcher_claim.json'; launcher_claim = c.read(launcher_claim_path)
    containers = worker_containers(); expected = [c.artifact(path/'execution.json') for path in containers]
    c.require(len(expected) == 192 and root_execution['segments'] == expected[:-1]
        and root_execution['status'] == 'incomplete' and root_execution['error'], 'Exactly191 passed predecessors and preservedfailedc135')
    c.verify(root_execution['segments'])
    authoritative = {}; bindings = [c.artifact(p) for p in (root_execution_path, launch_path,
        root_claim_path, launcher_claim_path, c.PREVIOUS_OUT/'freeze.json')]
    reused = []
    for index, container in enumerate(containers):
        owner_path = container/'owner.json'; path = container/'execution.json'
        owner, execution = c.read(owner_path), c.read(path)
        c.require(owner['identity'] == execution['identity'] and owner['windows_job_assigned'] is True
            and owner['global_claim'] == c.artifact(root_claim_path)
            and owner['plan_sha256'] == execution['plan_sha256'] == c.PREVIOUS_FREEZE_SHA
            and execution['retries'] == 0, 'Authenticated assigned worker identity and exact source')
        if index < len(containers)-1:
            c.require(execution['status'] == 'passed' and execution['worker_exit_code'] == 0
                and execution['error'] is None and not execution['cleanup_errors'], 'Prior successful worker exit retained')
        else:
            c.require(execution['status'] == 'failed' and execution['worker_exit_code'] == 1
                and execution['error'] and not execution['cleanup_errors'], 'c135 actual failure retained')
        bindings.extend((c.artifact(owner_path), c.artifact(path)))
        for identity in execution['observed_owned_identities'] + [execution['identity']]:
            authoritative[key(identity)] = identity
            if key(identity) in REUSED_PARENT:
                reused.append(dict(identity=identity, container=container.relative_to(c.ROOT).as_posix(),
                    execution=c.artifact(path), owner_identity=owner['identity'],
                    worker_unix_origin=execution['unix_origin'], worker_unix_end=execution['unix_end']))
    foreign = set(FOREIGN)
    c.require(not foreign.intersection(authoritative), 'No exception is an authenticated worker or worker-observed descendant')
    c.require({key(x['identity']) for x in reused} == set(REUSED_PARENT)
        and all(identity['created'] > FOREIGN[0][1] for identity in (x['identity'] for x in reused)),
        'Discord ancestor is older than both observed incarnations of its staleparentPID')
    observations = {key(x): x for x in launch['observed_owned_identities']}
    c.require(foreign <= observations.keys() and set(REUSED_PARENT) <= observations.keys(),
        'Preserved broad observer includes both impossibleparentedges and exactforeignbranch')
    c.require(launch['identity'] == dict(pid=root_claim['pid'], created=root_claim['created'])
        and launch['parent'] == launcher_claim['identity'] and launch['status'] == 'failed'
        and launch['worker_exit_code'] == 1 and launch['cleanup_errors'] ==
        ["ValueError('Exact previous process still alive')"] * 6, 'All six historical cleanup observations remain failed and unchanged')
    roots = [dict(pid=root_claim['pid'], created=root_claim['created']), launcher_claim['identity']]
    if check_absence:
        for identity in list(authoritative.values()) + roots:
            require_absent(identity)
        # Every other historical observation remains subject to strict absence.
        for identity in observations.values():
            if key(identity) not in foreign:
                require_absent(identity)
    return dict(bindings=bindings, authoritative_worker_identities=sorted(authoritative.values(), key=key),
        authenticated_worker_executions=192, reused_parent_identities=reused,
        observed_foreign_identities=exception_identities(), original_cleanup_errors=launch['cleanup_errors'],
        root_identity=roots[0], launcher_identity=roots[1], strict_absence_checked=check_absence)


def validate_foreign_chain(rows):
    c.require(type(rows) is list and [key(row['identity']) for row in rows] == list(FOREIGN), 'Only exact six frozen identities')
    first = rows[0]
    c.require(first['ppid'] == 35020, 'Exact stale ancestor parentPID')
    for row in rows:
        c.require(row['name'] == 'Discord.exe' and row['exe'] == EXPECTED_IMAGE and row['live_identity_matched'] is True,
            'Authenticated read-only native image and birth snapshot; names alone never grant exemption')
    for row in rows[1:]:
        c.require(row['ppid'] == first['identity']['pid'] and row['identity']['created'] >= first['identity']['created'],
            'Each remaining foreign observation descends from exact older ancestor')


def reproduce_installed_ancestry_bug():
    """Call installed psutil's actual recursive function on an immutable fake map."""
    c.require(psutil.__version__ == '5.9.0', 'Bind the historical installed observer implementation')
    root_pid, worker_pid = 29196, 26260
    births = {root_pid: 1794860744.4922292, worker_pid: 1794929941.0990553,
              REUSED_PARENT[0][0]: REUSED_PARENT[0][1], **dict(FOREIGN)}
    parents = {worker_pid: root_pid, 35020: worker_pid, FOREIGN[0][0]: 35020,
               **{pid: FOREIGN[0][0] for pid, _ in FOREIGN[1:]}}
    actual = psutil.Process.children.__wrapped__
    class SavedProcess:
        def __init__(self, pid): self.pid = pid
        def create_time(self): return births[self.pid]
    with patch.object(psutil, '_ppid_map', return_value=parents), patch.object(psutil, 'Process', SavedProcess):
        observed = [p.pid for p in actual(SavedProcess(root_pid), recursive=True)]
    expected = set(parents)
    c.require(set(observed) == expected, 'Actual installedroot-relative observer reproduces all six falseobservations')
    rejected = []
    for child, parent in parents.items():
        if births[child] < births[parent]: rejected.append(dict(child=child, parent=parent,
            child_created=births[child], parent_created=births[parent]))
    c.require(rejected == [dict(child=9908, parent=35020, child_created=FOREIGN[0][1], parent_created=REUSED_PARENT[0][1])],
        'The actual immediateparentedge is chronologically impossible')
    return dict(psutil_version=psutil.__version__, installed_source=c.artifact(Path(inspect.getsourcefile(actual))),
        model_calls=0, process_launches=0, fake_parent_map={str(k):v for k,v in parents.items()},
        fake_births={str(k):v for k,v in births.items()},
        actual_recursive_result=observed, impossible_immediate_parent_edges=rejected,
        false_inclusion_reproduced=True, reason='recursivechildren compares descendantbirth against originalroot, not immediateparent')


def capture_proof():
    history = authenticated_history(); rows = []
    for identity in exception_identities():
        process = psutil.Process(identity['pid'])
        c.require(process.create_time() == identity['created'] and process.is_running(), 'Exact live observation during diagnostic capture')
        rows.append(dict(identity=identity, name=process.name(), exe=process.exe(), ppid=process.ppid(), live_identity_matched=True))
    validate_foreign_chain(rows)
    result = dict(schema='collection4096_historical_ownership_exception_v1', status='proven_false_parent_lineage',
        exact_exception_identities=exception_identities(), observed_native_processes=rows, history=history,
        reproduction=reproduce_installed_ancestry_bug(),
        all_original_failures_preserved=True, any_worker_exception=False, application_name_filter=False,
        process_mutations=0, process_terminations=0, models=0, games=0, time=c.now(),
        limitations='Original closed Windowsjob handles cannot be queried retroactively. Classification uses exactimpossibleparentchronology, nativeimage/chain snapshot and absence from allauthenticatedworkerregistries; newobservation must use livejobmembership.')
    c.save(PROOF, result)
    validate_proof(result, require_frozen=False)
    return result


def validate_proof(proof, *, require_frozen=True):
    c.require(proof['schema'] == 'collection4096_historical_ownership_exception_v1'
        and proof['status'] == 'proven_false_parent_lineage'
        and proof['exact_exception_identities'] == exception_identities()
        and proof['all_original_failures_preserved'] is True and proof['any_worker_exception'] is False
        and proof['application_name_filter'] is False and proof['process_mutations'] == proof['process_terminations'] == 0,
        'Exact narrowly frozen historical exception, never omitted worker cleanup')
    validate_foreign_chain(proof['observed_native_processes'])
    if require_frozen:
        c.require(c.PLAN.exists() and c.artifact(PROOF) in c.read(c.PLAN)['inputs'], 'Historical exception proof must be an exact frozen input')
    c.verify(proof['history']['bindings'])
    current = authenticated_history()
    c.require(current == proof['history'], 'Revalidated exact predecessor ownership and absence inventory')
    reproduction = proof['reproduction']
    c.verify([reproduction['installed_source']])
    c.require(reproduction == reproduce_installed_ancestry_bug(), 'Reproduced installedobserver lineage bug unchanged')
    return exception_identities()


def excluded_launch_identities():
    """Only use to classify the previous launcher observation list, never worker lists."""
    return validate_proof(c.read(PROOF))


if __name__ == '__main__':
    result = capture_proof()
    print(c.canonical(dict(status=result['status'], exact_exceptions=len(FOREIGN),
        authenticated_worker_executions=result['history']['authenticated_worker_executions'],
        authoritative_identities=len(result['history']['authoritative_worker_identities']), models=0, process_mutations=0)))

"""Authenticated native and Windows console descendants, including cleanup IDs."""
from copy import deepcopy
import math
import os
from pathlib import Path

from . import common as c

SCHEMA = 'splendorust_proxy_owned_service_tree_v1'
NATIVE_ROLES = {'reference', 'search', 'control'}
FIELDS = {'pid', 'created', 'parent_pid', 'role', 'executable', 'executable_sha256',
          'executable_bytes', 'affinity'}


def _identity(value):
    c.require(type(value) is dict and set(value) == {'pid', 'created'} and
        type(value['pid']) is int and value['pid'] > 0 and
        type(value['created']) in (int, float) and math.isfinite(value['created']) and value['created'] > 0,
        'Exact process birth identity')
    return value


def _expected(expected_binaries, console_artifact):
    c.require(type(expected_binaries) is dict and set(expected_binaries) == NATIVE_ROLES,
        'Exactly three native executable roles')
    for item in [*expected_binaries.values(), console_artifact]:
        c.require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'} and
            type(item['bytes']) is int and item['bytes'] > 0 and
            type(item['sha256']) is str and len(item['sha256']) == 64 and
            all(x in '0123456789abcdef' for x in item['sha256']), 'Pinned executable artifact')
    console_path = (Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve()
    c.require(c.resolve(console_artifact) == console_path, 'Only pinned System32 console host allowed')


def validate_snapshot(service_identity, native_pids, rows, expected_binaries, console_artifact, *, affinity=(0,)):
    """Pure readback: enforce roles, parentage, exact hashes and finite identities."""
    _identity(service_identity)
    _expected(expected_binaries, console_artifact)
    c.require(type(native_pids) is dict and set(native_pids) == NATIVE_ROLES and
        all(type(x) is int and x > 0 for x in native_pids.values()) and len(set(native_pids.values())) == 3 and
        service_identity['pid'] not in native_pids.values(), 'Three distinct declared native PIDs')
    c.require(type(rows) is list and 3 <= len(rows) <= 7, 'Three natives and at most four console hosts')
    roles_by_pid = {pid: name for name, pid in native_pids.items()}
    native_rows = {}
    pids, console_parents = set(), set()
    for row in rows:
        c.require(type(row) is dict and set(row) == FIELDS, 'Complete descendant identity fields')
        _identity({key: row[key] for key in ('pid', 'created')})
        c.require(row['pid'] not in pids and type(row['parent_pid']) is int and
                  row['pid'] != service_identity['pid'], 'Distinct observed descendant PIDs')
        pids.add(row['pid'])
        c.require(type(row['affinity']) is list and bool(row['affinity']) and
            all(type(x) is int and 0 <= x < os.cpu_count() for x in row['affinity']) and
            row['affinity'] == sorted(set(row['affinity'])), 'Finite observed descendant affinity')
        if row['pid'] in roles_by_pid:
            c.require(row['affinity'] == list(affinity), 'Unchanged native compute affinity')
            role = roles_by_pid[row['pid']]
            c.require(row['role'] == role and row['parent_pid'] == service_identity['pid'],
                'Native role is a direct child of exact service')
            item = expected_binaries[role]
            native_rows[role] = row
        else:
            # Windows may create its console before PersistentWorker pins the
            # Python service. Record that helper's actual finite affinity;
            # only learned/native computation is required to run on CPU0.
            parent_role = 'service' if row['parent_pid'] == service_identity['pid'] else roles_by_pid.get(row['parent_pid'])
            c.require(parent_role is not None and row['role'] == 'console:' + parent_role,
                'Only console children of exact service or native are allowed')
            c.require(row['parent_pid'] not in console_parents, 'At most one console per owned process')
            console_parents.add(row['parent_pid'])
            item = console_artifact
        c.require(type(row['executable']) is str and Path(row['executable']).is_absolute() and
            Path(row['executable']).resolve() == c.resolve(item) and
            row['executable_sha256'] == item['sha256'] and
            type(row['executable_bytes']) is int and row['executable_bytes'] == item['bytes'],
            'Exact planned descendant executable path/hash/size')
    c.require(set(native_rows) == NATIVE_ROLES, 'All three native processes observed')
    return native_rows


def inspect_service_tree(service_identity, native_pids, expected_binaries, console_artifact, *, affinity=(0,)):
    import psutil
    _identity(service_identity)
    _expected(expected_binaries, console_artifact)
    service = psutil.Process(service_identity['pid'])
    c.require(service.create_time() == service_identity['created'] and service.is_running(),
        'Exact service identity remains live')
    roles_by_pid = {pid: name for name, pid in native_pids.items()}
    rows = []
    for child in service.children(recursive=True):
        parent = child.ppid()
        role = roles_by_pid.get(child.pid)
        if role is None:
            parent_role = 'service' if parent == service_identity['pid'] else roles_by_pid.get(parent)
            c.require(parent_role is not None, 'Unexpected descendant ownership chain')
            role = 'console:' + parent_role
        executable = Path(child.exe()).resolve()
        item = c.artifact(executable)
        rows.append(dict(pid=child.pid, created=child.create_time(), parent_pid=parent, role=role,
            executable=str(executable), executable_sha256=item['sha256'], executable_bytes=item['bytes'],
            affinity=child.cpu_affinity()))
    rows.sort(key=lambda row: (row['role'], row['pid']))
    validate_snapshot(service_identity, native_pids, rows, expected_binaries, console_artifact, affinity=affinity)
    return rows


def make_book(service_identity, native_pids, startup, end, expected_binaries, console_artifact, *, affinity=(0,)):
    first = validate_snapshot(service_identity, native_pids, startup, expected_binaries, console_artifact, affinity=affinity)
    last = validate_snapshot(service_identity, native_pids, end, expected_binaries, console_artifact, affinity=affinity)
    c.require(first == last, 'All exact native identities and executables persist through service lifetime')
    identities = {(row['pid'], row['created']): dict(pid=row['pid'], created=row['created'])
                  for row in startup + end}
    return dict(schema=SCHEMA, service=deepcopy(service_identity), native_pids=deepcopy(native_pids),
        startup=deepcopy(startup), end=deepcopy(end), affinity=list(affinity),
        all_owned_identities=[identities[key] for key in sorted(identities)],
        native_identities={name: {key: first[name][key] for key in ('pid', 'created')} for name in sorted(first)},
        expected_binaries=deepcopy(expected_binaries), console_artifact=deepcopy(console_artifact))


def validate_book(book, expected_binaries, console_artifact):
    c.require(type(book) is dict and set(book) == {'schema', 'service', 'native_pids', 'startup', 'end', 'affinity',
        'all_owned_identities', 'native_identities', 'expected_binaries', 'console_artifact'}, 'Complete ownership book')
    c.require(book['expected_binaries'] == expected_binaries and book['console_artifact'] == console_artifact and
        book['affinity'] == [0], 'Frozen ownership policy')
    expected = make_book(book['service'], book['native_pids'], book['startup'], book['end'],
                         expected_binaries, console_artifact, affinity=book['affinity'])
    c.require(book == expected, 'Exact startup/final ownership and complete cleanup union')
    return dict(status='passed', native_roles=sorted(NATIVE_ROLES), native_processes=3,
                observed_owned_descendants=len(book['all_owned_identities']), all_console_identities_retained=True)

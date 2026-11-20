"""Explicit host/role placement for the separate supervisor experiment.

Windows SetProcessAffinityMask constrains all process threads and is inherited
by newly created threads/processes. We query thread masks without changing them.
Only this host's single processor group is supported; unobservable state fails.
API contracts: https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups
https://learn.microsoft.com/en-us/windows/win32/api/processtopologyapi/nf-processtopologyapi-getthreadgroupaffinity
"""
from copy import deepcopy
import ctypes
import hashlib
import json
import math
import os
import threading
import time
import weakref

import psutil
from tools.local_player_qualification_128 import clock

SCHEMA = 'splendorust_fixed_core_placement_v1'
PLAYER_CPUS = (0, 2)
NORMAL_PRIORITY = 0x20
MAX_THREADS = 128
_workers = weakref.WeakSet()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plan(host_topology):
    """Freeze host eligibility separately from each process role's CPU mask."""
    host = deepcopy(host_topology)
    require(host['api'] == 'GetLogicalProcessorInformationEx(RelationProcessorCore)', 'Windows physical topology required')
    require(host['allowed_logical_processors'] == list(range(20)), 'Unexpected original host eligibility')
    require(host['selected_logical_processor'] == 0, 'Original player processor must remain zero')
    cores = [row for row in host['cores'] if 0 in row['logical_processors']]
    require(len(cores) == 1 and cores[0]['logical_processors'] == [0, 1], 'Original physical player core changed')
    require(sorted(cpu for row in host['cores'] for cpu in row['logical_processors']) == list(range(20)), 'Unexpected processor inventory')
    require(host['selected_core_index'] == cores[0]['core_index'], 'Selected physical core identity differs')
    chosen = max(host['cores'], key=lambda row: (row['efficiency_class'], len(row['logical_processors']), -min(row['logical_processors'])))
    require(chosen['core_index'] == host['selected_core_index'] and min(chosen['logical_processors']) == 0,
            'Frozen host selection does not follow original admission rule')
    other = [row for row in host['cores'] if 2 in row['logical_processors']]
    require(len(other) == 1 and other[0]['logical_processors'] == [2, 3] and
            other[0]['core_index'] != cores[0]['core_index'] and
            other[0]['efficiency_class'] == chosen['efficiency_class'],
            'Comparison CPU must be a distinct physical core in the same highest efficiency class')
    return dict(schema=SCHEMA, host_topology=host, host_topology_sha256=_digest(host),
                player_masks={'0': [0], '2': [2]}, physical_core_masks={'0': [0, 1], '2': [2, 3]},
                role_masks={'original': list(range(20)), 'isolated': list(range(4, 20))},
                fixed_player_affinity_for_entire_lifetime=True, live_affinity_switches=False,
                normal_priority_class=NORMAL_PRIORITY, processor_group=0,
                process_mask_constrains_all_threads=True,
                placement_is_not_exclusive_reservation_against_external_processes=True)


def _planned(freeze, mode):
    require(mode in ('original', 'isolated'), 'Unknown placement mode')
    expected = plan(freeze['topology'])
    require(freeze['placement'] == expected, 'Placement plan differs from frozen host topology')
    return expected


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [('Mask', ctypes.c_size_t), ('Group', ctypes.c_ushort), ('Reserved', ctypes.c_ushort * 3)]


def _api():
    require(os.name == 'nt', 'Supervisor placement requires Windows')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.GetProcessAffinityMask.restype = ctypes.c_int
    api.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    api.OpenThread.restype = ctypes.c_void_p
    api.GetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(GROUP_AFFINITY)]
    api.GetThreadGroupAffinity.restype = ctypes.c_int
    api.GetThreadPriority.argtypes = [ctypes.c_void_p]
    api.GetThreadPriority.restype = ctypes.c_int
    api.GetProcessIdOfThread.argtypes = [ctypes.c_void_p]
    api.GetProcessIdOfThread.restype = ctypes.c_uint32
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    api.CloseHandle.restype = ctypes.c_int
    return api


def _bits(mask):
    return [index for index in range(ctypes.sizeof(ctypes.c_size_t)*8) if mask & (1 << index)]


def _process_masks():
    api = _api()
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not api.GetProcessAffinityMask(api.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    require(process.value and system.value, 'Unobservable or multiple-group process affinity')
    return _bits(process.value), _bits(system.value)


def _thread_rows(process, expected_mask, *, require_current=True):
    api = _api()
    threads = process.threads()
    require(0 < len(threads) <= MAX_THREADS, 'Supervisor thread inventory bound')
    result = []
    for item in threads:
        handle = api.OpenThread(0x0800, False, item.id)  # THREAD_QUERY_LIMITED_INFORMATION
        if not handle:
            # A thread which ended before observation is explicitly accounted;
            # an extant but unqueryable thread is never silently accepted.
            require(item.id not in {row.id for row in process.threads()}, 'Live supervisor thread is unqueryable')
            result.append(dict(thread_id=item.id, state='exited_before_query'))
            continue
        try:
            require(api.GetProcessIdOfThread(handle) == process.pid, 'Thread ownership differs')
            affinity = GROUP_AFFINITY()
            if not api.GetThreadGroupAffinity(handle, ctypes.byref(affinity)):
                raise ctypes.WinError(ctypes.get_last_error())
            priority = api.GetThreadPriority(handle)
            require(priority != 0x7fffffff, 'Thread priority is unobservable')
            mask = _bits(affinity.Mask)
            require(affinity.Group == 0 and mask == expected_mask, 'Supervisor thread affinity differs')
            require(priority == 0, 'Supervisor thread priority differs from normal')
            result.append(dict(thread_id=item.id, state='observed', group=int(affinity.Group),
                               mask=mask, relative_priority=priority))
        finally:
            if not api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
    require(any(row['state'] == 'observed' and (not require_current or
                row['thread_id'] == threading.get_native_id()) for row in result),
            'Required current/owned thread absent from placement evidence')
    return result


def verify_host(freeze):
    """Read raw physical kernel topology; never use role eligibility as host eligibility."""
    expected = plan(freeze['topology'])
    require(freeze['placement'] == expected, 'Frozen placement mismatch')
    current = clock.processor_topology()
    # These fields are obtained directly from kernel physical topology, not the
    # current process's role mask or the resulting selected logical processor.
    for field in ('api', 'cores', 'selection_rule', 'distinct_efficiency_classes'):
        require(current[field] == expected['host_topology'][field], 'Physical host topology changed: ' + field)
    _, system = _process_masks()
    require(system == expected['host_topology']['allowed_logical_processors'], 'Host system processor eligibility changed')
    return deepcopy(expected['host_topology'])


def verify_runtime(freeze, placement_mode, *, role='runner', full=True):
    """Read-only role guard; full snapshots run outside decision clocks."""
    expected = _planned(freeze, placement_mode)
    require(role in ('runner', 'watchdog'), 'Unknown supervisory role')
    process = psutil.Process()
    mask, system = _process_masks()
    wanted = expected['role_masks'][placement_mode]
    require(system == expected['host_topology']['allowed_logical_processors'], 'System affinity changed')
    require(mask == wanted and process.cpu_affinity() == wanted, 'Supervisor process affinity changed')
    require(int(process.nice()) == NORMAL_PRIORITY, 'Supervisor priority changed')
    rows = _thread_rows(process, wanted) if full else None
    return dict(schema=SCHEMA, role=role, placement_mode=placement_mode, pid=process.pid,
                created=process.create_time(), monotonic_ns=time.perf_counter_ns(),
                current_thread_id=threading.get_native_id(),
                process_mask=mask, system_mask=system, priority_class=NORMAL_PRIORITY,
                physical_core_masks=expected['physical_core_masks'], player_masks=expected['player_masks'],
                host_topology_sha256=expected['host_topology_sha256'],
                full_thread_inventory=full, threads=rows,
                short_lived_threads_constrained_by_process_mask=True)


def configure_execution(freeze, placement_mode, *, role='runner'):
    """Change only our role mask before workers exist; never repair a live request."""
    expected = _planned(freeze, placement_mode)
    require(role in ('runner', 'watchdog'), 'Unknown supervisory role')
    require(not any(_resident(worker) for worker in tuple(_workers)),
            'Cannot switch supervisor placement with resident workers')
    verify_host(freeze)
    process = psutil.Process()
    require(int(process.nice()) == NORMAL_PRIORITY, 'Existing supervisor priority is not normal')
    before, _ = _process_masks()
    require(before in expected['role_masks'].values(), 'Unexpected inherited supervisor mask')
    process.cpu_affinity(expected['role_masks'][placement_mode])
    receipt = verify_runtime(freeze, placement_mode, role=role)
    return dict(receipt, previous_process_mask=before, mutation_scope='current_owned_process_affinity_only')


def admit_worker_topology(freeze, placement_mode):
    verify_runtime(freeze, placement_mode, full=False)
    return verify_host(freeze)


def register_worker(worker):
    _workers.add(worker)


def _resident(worker):
    if getattr(worker, '_placement_constructing', False) or not getattr(worker, '_closed', True):
        return True
    process = getattr(worker, 'process', None)
    if process is not None and process.poll() is None:
        return True
    if any(thread.is_alive() for thread in getattr(worker, '_threads', ())):
        return True
    cleanup = getattr(worker, 'cleanup_receipt', None)
    return cleanup is not None and cleanup.get('status') != 'passed'


def verify_saved_runtime(value, freeze, placement_mode, *, role='runner', identity=None):
    expected = _planned(freeze, placement_mode)
    require(type(value) is dict and value.get('schema') == SCHEMA and value.get('role') == role and
            value.get('placement_mode') == placement_mode, 'Saved supervisory role identity differs')
    require(type(value.get('pid')) is int and value['pid'] > 0 and type(value.get('created')) in (int, float) and
            math.isfinite(value['created']) and value['created'] > 0 and
            type(value.get('monotonic_ns')) is int and value['monotonic_ns'] > 0,
            'Saved supervisory identity/time is invalid')
    if identity is not None:
        require((value['pid'], value['created']) == (identity['pid'], identity['created']), 'Saved supervisor ownership differs')
    require(value.get('process_mask') == expected['role_masks'][placement_mode] and
            value.get('system_mask') == expected['host_topology']['allowed_logical_processors'] and
            value.get('physical_core_masks') == expected['physical_core_masks'] and
            value.get('player_masks') == expected['player_masks'] and
            value.get('priority_class') == NORMAL_PRIORITY and value.get('host_topology_sha256') == expected['host_topology_sha256'],
            'Saved supervisor mask/priority/topology differs')
    rows = value.get('threads')
    require(value.get('full_thread_inventory') is True and type(rows) is list and 0 < len(rows) <= MAX_THREADS,
            'Saved supervisor thread inventory absent/bounded incorrectly')
    seen, observed = set(), set()
    for row in rows:
        require(type(row) is dict and type(row.get('thread_id')) is int and row['thread_id'] > 0 and
                row['thread_id'] not in seen, 'Saved thread identity invalid/duplicated')
        seen.add(row['thread_id'])
        require(row.get('state') in ('observed', 'exited_before_query'), 'Saved thread observation state invalid')
        if row['state'] == 'observed':
            require(row.get('group') == 0 and row.get('mask') == expected['role_masks'][placement_mode] and
                    row.get('relative_priority') == 0, 'Saved supervisor thread placement differs')
            observed.add(row['thread_id'])
    require(type(value.get('current_thread_id')) is int and value['current_thread_id'] in observed,
            'Saved current supervisory thread was not observed')
    require(value.get('short_lived_threads_constrained_by_process_mask') is True, 'Thread inheritance guarantee absent')
    return value


def verify_saved_request(value, freeze, placement_mode, *, request_id, player_pid,
                         logical_cpu, supervisor_identity=None, parent_profile=None, require_complete=True,
                         player_identity=None, expected_processes=None, original_clock=None, cleanup_receipt=None):
    require(type(value) is dict and value.get('schema') == SCHEMA and value.get('placement_mode') == placement_mode and
            value.get('request_id') == request_id and value.get('player_pid') == player_pid and
            type(value.get('logical_cpu')) is int and value.get('logical_cpu') == logical_cpu,
            'Saved placement/request identity differs')
    before = verify_saved_runtime(value['before'], freeze, placement_mode, identity=supervisor_identity)
    player_before = verify_saved_player_runtime(value['player_before'], freeze, logical_cpu,
        player_identity=player_identity, expected_processes=expected_processes)
    require(player_before['player_pid'] == player_pid, 'Player/request ownership differs')
    require(original_clock is not None and value.get('original_request_status') == original_clock['status'],
            'Original request status binding missing')
    if value.get('player_after') is None:
        require(original_clock['status'] != 'ok' and original_clock.get('accepted') is False and
                value.get('player_closed_by_original_request') is True,
                'Missing player aftermath without original failed request closure')
        require(type(cleanup_receipt) is dict and cleanup_receipt.get('status') == 'passed' and
                cleanup_receipt.get('errors') == [] and cleanup_receipt.get('alive') == [],
                'Missing clean owned closure proof')
        confirmed = {(row['pid'], row['created']) for row in cleanup_receipt['confirmed']
                     if row['state'] in ('absent', 'original_identity_absent', 'exit_wait_confirmed')}
        require(all((row['pid'], row['created']) in confirmed for row in player_before['processes']),
                'Closed player identity inventory incomplete')
    else:
        require(value.get('player_closed_by_original_request') is False,
                'Live player aftermath conflicts with closure claim')
        player_after = verify_saved_player_runtime(value['player_after'], freeze, logical_cpu,
            player_identity=dict(pid=player_before['player_pid'], created=player_before['player_created']),
            expected_processes=player_before['processes'])
        require(player_after['start_ns'] >= player_before['monotonic_ns'], 'Player placement chronology reversed')
    if require_complete:
        require(value.get('error') is None and value.get('after') is not None, 'Placement request evidence incomplete/failed')
    if value.get('after') is not None:
        after = verify_saved_runtime(value['after'], freeze, placement_mode, identity=before)
        require(after['monotonic_ns'] >= before['monotonic_ns'], 'Placement request chronology reversed')
    else:
        require(type(value.get('error')) is str and value['error'], 'Missing placement aftermath must disclose error')
    require(value.get('checks_outside_original_request_clock') is True and
            value.get('sample_checks_inside_original_clock') is True and
            value.get('original_clock_and_failure_receipt_preserved') is True, 'Saved placement clock contract differs')
    if parent_profile is not None:
        require(before['pid'] == parent_profile['pid'] and player_pid == parent_profile['child_pid'] and
                request_id == parent_profile['request_id'], 'Placement/parent observer ownership differs')
        require(before['monotonic_ns'] <= parent_profile['request_start_ns'], 'Placement precheck follows trusted request start')
        require(player_before['monotonic_ns'] <= parent_profile['request_start_ns'],
                'Player precheck follows trusted request start')
        if value.get('after') is not None:
            require(value['after']['monotonic_ns'] >= parent_profile['capture_snapshot_ns'],
                    'Placement aftermath precedes original observer completion')
        if value.get('player_after') is not None:
            require(value['player_after']['start_ns'] >= parent_profile['capture_snapshot_ns'],
                    'Player aftermath precedes original observer completion')
    return value


def verify_worker_placement(folder, freeze, placement_mode, logical_cpu=None):
    """Validate the immutable admission evidence in addition to old child checks."""
    from pathlib import Path
    folder = Path(folder)
    expected = _planned(freeze, placement_mode)
    proof = json.loads((folder/'placement_admission.json').read_text(encoding='utf-8'))
    initial = json.loads((folder/'initialization.json').read_text(encoding='utf-8'))
    core = proof.get('logical_cpu')
    require(type(core) is int and core in PLAYER_CPUS and (logical_cpu is None or logical_cpu == core),
            'Saved player core differs')
    require(initial['topology'] == expected['host_topology'] and initial['affinity'] == [core], 'Saved player admission topology/mask differs')
    require(proof['schema'] == SCHEMA and proof['placement_mode'] == placement_mode and
            proof['player_pid'] == initial['identity']['pid'] and proof['player_created'] == initial['identity']['created'],
            'Saved player placement identity differs')
    verify_saved_runtime(proof['supervisor'], freeze, placement_mode)
    require(proof['player_mask'] == [core] and proof['player_priority_class'] == NORMAL_PRIORITY,
            'Saved player placement or priority differs')
    verify_saved_player_runtime(proof['player'], freeze, core, player_identity=initial['identity'],
                                expected_processes=initial['telemetry']['process_tree'])
    return proof


def _process_identity_rows(rows):
    require(type(rows) is list and 3 <= len(rows) <= 6, 'Bounded owned tree inventory required')
    roles = [row['role'] for row in rows]
    require(len(set(roles)) == len(roles) and {'python', 'search', 'reference'} <= set(roles) and
            set(roles) <= {'python', 'search', 'reference', 'console:python', 'console:search', 'console:reference'},
            'Owned process roles differ')
    result = []
    for row in rows:
        require(type(row['pid']) is int and row['pid'] > 0 and type(row['created']) in (int, float) and
                math.isfinite(row['created']) and row['created'] > 0 and
                type(row['parent_pid']) is int and row['parent_pid'] > 0 and
                type(row['executable']) is str and row['executable'], 'Invalid owned process identity')
        result.append({key: row[key] for key in ('role', 'pid', 'created', 'parent_pid', 'executable')})
    require(len({row['pid'] for row in result}) == len(result), 'Duplicate owned PID')
    by_role = {row['role']: row for row in result}
    for row in result:
        owner = 'python' if row['role'] in ('search', 'reference') else row['role'].removeprefix('console:')
        if row['role'] != 'python':
            require(row['parent_pid'] == by_role[owner]['pid'], 'Owned process parent differs')
    return sorted(result, key=lambda row: row['role'])


def verify_player_runtime(child, freeze, logical_cpu):
    """Read fixed placement of all authenticated owned processes outside clocks."""
    from pathlib import Path
    _planned(freeze, 'isolated')
    require(type(logical_cpu) is int and logical_cpu in PLAYER_CPUS and child.affinity == [logical_cpu],
            'Fixed player affinity differs')
    started = time.perf_counter_ns()
    telemetry = child._sample()  # Original ownership/executable/aggregate checks.
    require(not child._failure and not child._closed, 'Player unavailable during placement check')
    identities = _process_identity_rows(telemetry['process_tree'])
    rows = []
    for identity in identities:
        process = psutil.Process(identity['pid'])
        require(process.create_time() == identity['created'] and process.ppid() == identity['parent_pid'] and
                str(Path(process.exe()).resolve()) == identity['executable'], 'Player identity changed during placement')
        require(process.cpu_affinity() == [logical_cpu] and int(process.nice()) == NORMAL_PRIORITY,
                'Player mask/priority differs')
        threads = _thread_rows(process, [logical_cpu], require_current=False)
        require(psutil.Process(process.pid).create_time() == identity['created'] and
                process.cpu_affinity() == [logical_cpu] and int(process.nice()) == NORMAL_PRIORITY,
                'Player identity/mask/priority changed during thread read')
        rows.append(dict(identity, process_mask=[logical_cpu], priority_class=NORMAL_PRIORITY, threads=threads))
    result = dict(schema=SCHEMA, logical_cpu=logical_cpu, player_pid=child.identity['pid'],
        player_created=child.identity['created'], start_ns=started, monotonic_ns=time.perf_counter_ns(),
        full_thread_inventory=True, process_mask_constrains_all_threads=True, processes=rows)
    verify_saved_player_runtime(result, freeze, logical_cpu, player_identity=child.identity,
                                expected_processes=telemetry['process_tree'])
    return result


def verify_saved_player_runtime(value, freeze, logical_cpu, *, player_identity=None, expected_processes=None):
    _planned(freeze, 'isolated')
    require(type(logical_cpu) is int and logical_cpu in PLAYER_CPUS and type(value) is dict and
            value.get('schema') == SCHEMA and type(value.get('logical_cpu')) is int and
            value.get('logical_cpu') == logical_cpu and
            value.get('full_thread_inventory') is True and value.get('process_mask_constrains_all_threads') is True,
            'Saved player placement contract differs')
    require(type(value.get('start_ns')) is int and type(value.get('monotonic_ns')) is int and
            0 < value['start_ns'] <= value['monotonic_ns'], 'Invalid player placement times')
    identities = _process_identity_rows(value['processes'])
    if expected_processes is not None:
        require(identities == _process_identity_rows(expected_processes), 'Saved player owned identity inventory differs')
    main = next(row for row in identities if row['role'] == 'python')
    require((value['player_pid'], value['player_created']) == (main['pid'], main['created']),
            'Saved player root identity differs')
    if player_identity is not None:
        require((main['pid'], main['created']) == (player_identity['pid'], player_identity['created']),
                'Saved player owner differs')
    seen_threads = set()
    for process in value['processes']:
        require(process['process_mask'] == [logical_cpu] and process['priority_class'] == NORMAL_PRIORITY,
                'Saved player mask/priority differs')
        rows = process['threads']
        require(type(rows) is list and 0 < len(rows) <= MAX_THREADS, 'Saved player thread inventory bound')
        observed = False
        for row in rows:
            require(type(row.get('thread_id')) is int and row['thread_id'] > 0 and
                    row['thread_id'] not in seen_threads, 'Saved player thread identity duplicated/invalid')
            seen_threads.add(row['thread_id'])
            require(row.get('state') in ('observed', 'exited_before_query'), 'Saved player thread state invalid')
            if row['state'] == 'observed':
                require(row.get('group') == 0 and row.get('mask') == [logical_cpu] and row.get('relative_priority') == 0,
                        'Saved player thread placement/priority differs')
                observed = True
        require(observed, 'No observed live player thread')
    return value

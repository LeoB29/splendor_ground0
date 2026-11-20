"""Bounded retrospective extraction; no model, environment, or replay execution.

prepare authenticates and freezes a deterministic panel. run requires that exact
freeze SHA, extracts saved markers, and rehashes consumed inputs before success.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/splendorust_search128_saved_timing_20261029_01'
PROTOCOL = 'docs/engineering/splendorust_search128_saved_timing_20261029.md'
TEST = 'tests/test_splendorust_search128_saved_timing_20261029.py'
ANCHOR = {'path': 'outputs/splendorust_search128_saved_audit_20261029_continuity/recovered_verification_accepted.json',
          'bytes': 3081, 'sha256': '8571ce7f3ee6adc52a3aee4b7e18b9dba3587dae1878ab629613d30e9b098680'}
SOURCE_OUT = 'outputs/splendorust_search128_resume_20261029_01'
LIMITS = dict(maximum_seconds=900, maximum_rss_bytes=1024**3,
              maximum_output_bytes=64*1024**2, minimum_available_ram_bytes=2*1024**3,
              minimum_free_disk_bytes=1024**3, maximum_detailed_rows=50)
RULE = dict(slow=10, same_game='nearest ply <=1 second excluding slow10; tie earlier',
            adjacent='both adjacent opponent64 ordinary moves if present',
            systematic_games=list(range(20, 381, 40)), systematic_own_turn=10,
            ties='game_index then ply', reuse='record all memberships; deduplicate rows',
            population='400 main games; 6 operational games separate; bridge excluded')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and '\\' not in relative and ':' not in relative,
            'noncanonical/path escape')
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and relative == rel.as_posix() and
            all(x not in ('', '.', '..') for x in rel.parts), 'path escape')
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts)
    require(target.resolve().is_relative_to(root), 'path escape/redirect')
    for p in [target, *target.parents]:
        if p == root:
            break
        st = p.lstat()
        require(not p.is_symlink() and not getattr(st, 'st_file_attributes', 0) & 0x400,
                'symlink/reparse input')
    return target


def descriptor(path, root=ROOT):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=path.resolve().relative_to(Path(root).resolve()).as_posix(),
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class Guard:
    def __init__(self, out, seconds=900):
        self.out = Path(out)
        self.start = time.monotonic()
        self.start_unix = time.time()
        self.seconds = seconds
        self.last_resource = -1.
        self.peak_rss = 0
        self.check(initial=True)

    def check(self, initial=False):
        require(time.monotonic()-self.start <= self.seconds and
                time.time()-self.start_unix <= self.seconds, 'saved diagnostic deadline')
        now = time.monotonic()
        if not initial and now-self.last_resource < 1.:
            return
        self.last_resource = now
        if os.name == 'nt':
            class Memory(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ('total', 'available', 'page_total',
                                                     'page_available', 'virtual_total',
                                                     'virtual_available', 'extended')]
            class ProcessMemory(ctypes.Structure):
                _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
                    (n, ctypes.c_size_t) for n in ('peak', 'working', 'paged_peak',
                        'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
            mem, proc = Memory(), ProcessMemory()
            mem.length, proc.cb = ctypes.sizeof(mem), ctypes.sizeof(proc)
            require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)), 'RAM API')
            ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            memory_api = ctypes.windll.psapi.GetProcessMemoryInfo
            memory_api.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong]
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            require(memory_api(handle, ctypes.byref(proc), proc.cb), 'RSS API')
            self.peak_rss = max(self.peak_rss, proc.working)
            require(proc.working <= LIMITS['maximum_rss_bytes'], 'RSS cap')
            if initial:
                require(mem.available >= LIMITS['minimum_available_ram_bytes'], 'initial available RAM')
        require(shutil.disk_usage(self.out.parent if self.out.parent.exists() else ROOT).free >=
                LIMITS['minimum_free_disk_bytes'], 'disk reserve')
        if self.out.exists():
            require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <=
                    LIMITS['maximum_output_bytes'], 'output cap')


class Reader:
    def __init__(self, root=ROOT, guard=lambda: None):
        self.root, self.guard = Path(root), guard
        self.pins, self.consumed = {}, {}
        self.read_bytes = 0

    def add(self, item):
        require(isinstance(item, dict) and set(('path', 'bytes', 'sha256')) <= item.keys(), 'bad descriptor')
        d = {k: item[k] for k in ('path', 'bytes', 'sha256')}
        require(type(d['bytes']) is int and d['bytes'] >= 0 and
                isinstance(d['sha256'], str) and len(d['sha256']) == 64, 'bad descriptor types')
        # Path syntax is checked even before physical access; inventory entries are local.
        p = PurePosixPath(d['path'])
        require(not p.is_absolute() and ':' not in d['path'] and '\\' not in d['path'] and
                d['path'] == p.as_posix() and '..' not in p.parts, 'path escape')
        old = self.pins.get(d['path'])
        require(old is None or old == d, 'conflicting input pins')
        self.pins[d['path']] = d
        return d

    def read(self, item):
        self.guard()
        d = self.pins[item] if isinstance(item, str) else self.add(item)
        path = safe_path(self.root, d['path'])
        before = path.stat()
        require(before.st_size == d['bytes'], 'input byte mismatch: '+d['path'])
        data = path.read_bytes()
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'input changed during read')
        require(len(data) == d['bytes'] and hashlib.sha256(data).hexdigest() == d['sha256'],
                'input hash mismatch: '+d['path'])
        self.consumed[d['path']] = d
        self.read_bytes += len(data)
        return json.loads(data)

    def recheck(self):
        for d in list(self.consumed.values()):
            self.guard()
            path = safe_path(self.root, d['path'])
            before = path.stat()
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024*1024), b''):
                    self.guard()
                    h.update(block)
            after = path.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and before.st_size == d['bytes'] and
                    h.hexdigest() == d['sha256'], 'consumed input changed: '+d['path'])


def authority(reader):
    accepted = reader.read(ANCHOR)
    require(accepted['status'] == 'separately_recovered_verification_accepted' and
            accepted['actual_owner_exit_code'] == accepted['reader_exit_code'] ==
            accepted['acceptance_exit_code'] == 0 and accepted['root_session_closed'] is True,
            'accepted historical anchor')
    reader.read(accepted['acceptance'])
    reader.read(accepted['owner_completion'])
    frozen = reader.read(accepted['audit_freeze'])
    preserved = reader.read(frozen['preservation_manifest'])
    require(preserved['status'] == 'preserved_before_new_audit' and
            accepted['preserved']['status'] == 'passed', 'historical preservation not passed')
    for tree in preserved['trees']:
        for pin in tree['artifacts']:
            reader.add(pin)
    del preserved
    source = reader.read(accepted['source_freeze'])
    # Current extraction imports no player/runtime module; source pins needed only
    # for the once-per-process corrected-source check and interpretation metadata.
    for pin in source['inputs']:
        if not Path(pin['path']).is_absolute() and ':' not in pin['path']:
            reader.add(pin)
    return accepted


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)
    return value


def population(reader):
    manifest = reader.read(SOURCE_OUT+'/game_evidence_manifest.json')
    require(len(manifest['games']) == 406, 'game inventory')
    rows, bridges, games_seen = [], [], set()
    for game in manifest['games']:
        reader.guard()
        spec = game['spec']
        key = spec['stage'], spec['game_index']
        require(key not in games_seen, 'duplicate game')
        games_seen.add(key)
        require(spec['challenger_quota'] == 128 and spec['opponent_quota'] == 64 and
                spec['challenger_mode'] == 'research' and spec['opponent_mode'] == 'frozen',
                'player/quota mismatch')
        initial = reader.read(game['initial'])
        require(initial['initial']['state']['current_player'] == 0, 'initial seat mapping')
        proof = reader.read(game['online_replay'])
        require(proof['status'] == 'passed' and proof['failed_moves'] == 0 and
                proof['transitions'] == len(game['events']), 'compact replay inventory')
        by_role = {'challenger': [], 'opponent': []}
        for ply, event in enumerate(game['events']):
            if event['kind'] == 'recovery_bridge':
                bridges.append(dict(stage=spec['stage'], game_index=spec['game_index'], ply=ply,
                                    descriptor=event['bridge']))
                continue
            require(event['kind'] == 'ordinary' and int(Path(event['row']['path']).stem) == ply,
                    'ordinary event/ply identity')
            require(reader.pins[event['row']['path']] == event['row'], 'row not in accepted inventory')
            role = 'challenger' if ply % 2 == spec['challenger_seat'] else 'opponent'
            by_role[role].append(dict(stage=spec['stage'], game_index=spec['game_index'],
                pair_index=spec['pair_index'], ply=ply, seat=ply % 2, role=role,
                quota=128 if role == 'challenger' else 64, row=event['row'],
                source_freeze_sha256=event['source_freeze_sha256']))
        for role, rr in by_role.items():
            times = proof['latencies'][role]
            require(len(times) == len(rr), 'latency/event reconciliation')
            for row, seconds in zip(rr, times):
                row['receipt_elapsed_seconds'] = number(seconds, 'invalid compact timing')
                require(seconds <= 10, 'accepted original research clock')
                rows.append(row)
    require(games_seen == {('main', i) for i in range(400)} | {('operational', i) for i in range(6)},
            'fixed schedule scope')
    require(len(bridges) == 1 and bridges[0]['game_index'] == 38 and bridges[0]['ply'] == 39,
            'one preserved bridge only')
    require(sum(r['stage'] == 'main' and r['quota'] == 128 for r in rows) == 11139 and
            sum(r['stage'] == 'main' and r['quota'] == 64 for r in rows) == 11140 and
            len(rows) == 22619, 'population denominator')
    return rows, bridges


def select_panel(rows):
    main = [r for r in rows if r['stage'] == 'main']
    candidates = [r for r in main if r['quota'] == 128]
    slow = sorted(candidates, key=lambda r: (-r['receipt_elapsed_seconds'], r['game_index'], r['ply']))[:10]
    require(len(slow) == 10, 'slow10 incomplete')
    excluded = {r['row']['path'] for r in slow}
    members, slots = {}, []
    def add(label, row, target=None):
        slots.append(dict(group=label, target=target, row=None if row is None else row['row']['path']))
        if row is not None:
            entry = members.setdefault(row['row']['path'], dict(row, memberships=[]))
            entry['memberships'].append(dict(group=label, target=target))
    for rank, r in enumerate(slow, 1):
        add('slow10', r, rank)
        same = [x for x in candidates if x['game_index'] == r['game_index'] and
                x['row']['path'] not in excluded and x['receipt_elapsed_seconds'] <= 1.]
        add('same_game_128_control', min(same, key=lambda x: (abs(x['ply']-r['ply']), x['ply'])) if same else None, rank)
        for direction in (-1, 1):
            adjacent = next((x for x in main if x['game_index'] == r['game_index'] and
                             x['ply'] == r['ply']+direction and x['quota'] == 64), None)
            add('adjacent64_before' if direction < 0 else 'adjacent64_after', adjacent, rank)
    for game_number in RULE['systematic_games']:
        same = sorted((r for r in candidates if r['game_index'] == game_number-1), key=lambda x: x['ply'])
        require(same, 'systematic game missing')
        add('systematic128', same[min(9, len(same)-1)], game_number)
    require(len(members) <= 50 and len(slots) == 50, 'panel cap')
    return dict(rule=RULE, rows=list(members.values()), slots=slots,
                distinct_rows=len(members), missing_slots=sum(s['row'] is None for s in slots),
                reused_slots=sum(s['row'] is not None for s in slots)-len(members))


def bound_row(reader, selected):
    row = reader.read(selected['row'])
    for field in ('stage', 'game_index', 'pair_index', 'ply', 'seat', 'quota'):
        require(row[field] == selected[field], 'row '+field+' mismatch')
    require(row['player_role'] == selected['role'] and row['completed'] is True and
            row['arm'] == 'search' and row['route'] == ('research' if row['quota'] == 128 else 'frozen'),
            'player/quota mismatch')
    clock = reader.read(row['worker_clock_file'])
    parent = reader.read(row['parent_profile_file'])
    child = reader.read(row['child_profile_file'])
    require(clock == row['clock'] and parent == row['parent_profile'] and child == row['child_profile'],
            'saved sidecar/clock mismatch')
    require(clock['status'] == 'ok' and clock['accepted'] is True and
            clock['received_before_deadline'] is True and clock['deadline_seconds'] == 10. and
            clock['receipt_elapsed_seconds'] == selected['receipt_elapsed_seconds'], 'accepted ordinary clock')
    result = clock['result']
    require(row['trace'] == result['diagnostics'] and row['action'] == result['action'] and
            result['simulations_quota'] == row['quota'] and result['player'] == 'search' and
            result['observer_sha256'] == digest(row['observer']), 'result/observer/trace identity')
    require(child['quota'] == row['quota'] and child['backend'] == 'qualified_graph64' and
            child['outcome'] == 'response_flushed' and child['error'] is None and
            child['result_sha256'] == digest(result), 'child result identity')
    request_id = clock['request_id']
    require(parent['request_id'] == child['request_id'] == request_id and
            parent['child_pid'] == child['pid'] and parent['request_raw_sha256'] == child['request_raw_sha256'],
            'request identity')
    payload = reader.read(row['worker_folder']+'/inputs/'+request_id+'.json')
    require(payload == dict(player='search', observer=row['observer']) and
            digest(payload) == child['payload_sha256'], 'input/observer binding')
    placement = reader.read(row['worker_folder']+'/placement_requests/'+request_id+'.json')
    require(placement == row['placement_evidence'] and placement['error'] is None and
            placement['request_id'] == request_id and placement['player_pid'] == child['pid'], 'placement binding')
    return row


def extract(row):
    p, c, clock = row['parent_profile'], row['child_profile'], row['clock']
    require(len(c['profiles']) == 1, 'ordinary single handler required')
    h = c['profiles'][0]
    require(h['handle_failed'] is False and h['pid'] == c['pid'], 'handler identity')
    values = [p.get('request_start_ns'), c.get('request_read_complete_ns'), h.get('handle_start_ns'),
              h.get('handle_end_ns'), c.get('pre_write_ns'), p.get('receipt_arrival_ns')]
    require(all(type(v) is int and v > 0 for v in values), 'missing/invalid ordered markers')
    require(values == sorted(values), 'unordered boundary markers')
    start, read, hs, he, write, arrival = values
    total = arrival-start
    require(abs(total/1e9-clock['receipt_elapsed_seconds']) <= 1e-9, 'partition disagrees with trusted receipt')
    names = ['input_delivery', 'input_read_to_handler', 'whole_handler', 'handler_to_write', 'response_delivery']
    pieces = dict(zip(names, [b-a for a, b in zip(values, values[1:])]))
    require(sum(pieces.values()) == total, 'partition reconciliation')
    for name in ('dispatch_complete_ns', 'stdin_write_start_ns', 'stdin_write_end_ns',
                 'stdin_flush_start_ns', 'stdin_flush_end_ns'):
        require(type(p.get(name)) is int, 'missing parent '+name)
    require(type(c.get('post_flush_ns')) is int and c['post_flush_ns'] >= write, 'missing/invalid postflush')
    t, tr = clock['telemetry'], row['trace']
    phases = [{k: phase[k] for k in ('heuristic_phase', 'simulations', 'leaf_calls',
        'native_legal_count', 'neural_callback_count', 'neural_callback_rows')} for phase in tr['phases']]
    require(sum(x['neural_callback_count'] for x in phases) == tr['inference_calls'] and
            sum(x['neural_callback_rows'] for x in phases) == tr['inference_rows'], 'callback count identity')
    cpu = t.get('request_cpu_seconds')
    if cpu is not None:
        number(cpu, 'sampled request CPU')
    return dict(game_number=row['game_index']+1, ply=row['ply'], move_number=row['ply']+1,
        individual_turn=row['ply']//2+1, seat=row['seat'], quota=row['quota'], role=row['player_role'],
        action=row['action'], receipt_elapsed_seconds=clock['receipt_elapsed_seconds'],
        parent_elapsed_including_validation_seconds=clock['elapsed_seconds'],
        partition_ns=pieces, partition_total_ns=total, original_markers_ns=dict(zip(
            ['request_start','child_read','handler_start','handler_end','prewrite','trusted_arrival'], values)),
        overlapping_diagnostics_ns=dict(parent_dispatch=p['dispatch_complete_ns']-start,
            parent_write=p['stdin_write_end_ns']-p['stdin_write_start_ns'],
            parent_flush=p['stdin_flush_end_ns']-p['stdin_flush_start_ns'],
            child_emission=c['post_flush_ns']-write,
            child_read_minus_parent_dispatch=read-p['dispatch_complete_ns'],
            arrival_minus_child_flush=arrival-c['post_flush_ns']),
        response_raw_bytes=p['response_raw_bytes'], request_raw_bytes=p['request_raw_bytes'],
        sampled_request_tree_cpu_seconds=cpu, sampled_rss_bytes=t.get('rss_bytes'),
        peak_sampled_rss_bytes=t.get('peak_sampled_rss_bytes'),
        observed_threads=t.get('observed_threads'), affinity=t.get('affinity'),
        phases=phases, inference_calls=tr['inference_calls'], inference_rows=tr['inference_rows'],
        unavailable=['per_phase_wall_time','neural_inference_wall_time','native_search_wall_time',
                     'host_CPU_contention','paging_timeline','per_request_free_system_RAM'],
        cpu_scope='sampled worker-tree CPU delta from pre-dispatch to after packet validation; not handler-aligned')


def distribution(values):
    values = sorted(values)
    return dict(count=len(values), mean=statistics.fmean(values) if values else None,
        median=statistics.median(values) if values else None,
        p95_nearest_rank=values[math.ceil(.95*len(values))-1] if values else None,
        maximum=max(values) if values else None,
        above_700ms=sum(v > .7 for v in values), above_900ms=sum(v > .9 for v in values),
        above_1s=sum(v > 1 for v in values))


def summarize_population(rows):
    groups = []
    for stage in ('main', 'operational'):
        for quota in (128, 64):
            selected = [r for r in rows if r['stage'] == stage and r['quota'] == quota]
            games = []
            for game in sorted({r['game_index'] for r in selected}):
                rr = [r for r in selected if r['game_index'] == game]
                games.append(dict(game_number=game+1, **distribution([r['receipt_elapsed_seconds'] for r in rr])))
            segments = []
            for sha in sorted({r['source_freeze_sha256'] for r in selected}):
                rr = [r for r in selected if r['source_freeze_sha256'] == sha]
                segments.append(dict(source_freeze_sha256=sha,
                    **distribution([r['receipt_elapsed_seconds'] for r in rr])))
            blocks = []
            if stage == 'main':
                for label, lower, upper in [('original_games_1_38',0,37),
                        ('recovered_game_39',38,38), ('resumed_games_40_400',39,399)]:
                    rr = [r for r in selected if lower <= r['game_index'] <= upper]
                    blocks.append(dict(label=label, **distribution([r['receipt_elapsed_seconds'] for r in rr])))
            groups.append(dict(stage=stage, quota=quota, games=games, source_segments=segments,
                               fixed_game_blocks=blocks,
                               **distribution([r['receipt_elapsed_seconds'] for r in selected])))
    return groups


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical(value)+'\n').encode()
    require(len(data) <= LIMITS['maximum_output_bytes'], 'single output cap')
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return descriptor(path)


def prepare(out=OUT):
    guard = Guard(out)
    reader = Reader(guard=guard.check)
    authority(reader)
    rows, bridges = population(reader)
    panel = select_panel(rows)
    for selected in panel['rows']:
        bound_row(reader, selected)  # bind only; detailed timing components are not ranked/analysed
    for bridge in bridges:
        reader.read(bridge['descriptor'])
    for name in ('boundary_analysis.json', 'performance.json'):
        reader.read(SOURCE_OUT+'/'+name)
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    for relative, sha in SOURCE_HASHES.items():
        pin = descriptor(ROOT/relative)
        require(pin['sha256'] == sha, 'corrected source changed')
        reader.add(pin)
        reader.consumed[relative] = pin
    for path in (Path(__file__), ROOT/PROTOCOL, ROOT/TEST, ROOT/'tools/future_training_rules_20261021.py'):
        pin = descriptor(path)
        reader.add(pin)
        reader.consumed[pin['path']] = pin
    for path in sorted((out/'prelaunch').glob('*')):
        if path.is_file():
            pin = descriptor(path)
            reader.add(pin)
            reader.consumed[pin['path']] = pin
    reader.recheck()
    panel_pin = save(out/'panel.json', panel)
    inputs_pin = save(out/'inputs.json', dict(scope='only consumed inputs; historical audit reused',
                    artifacts=sorted(reader.consumed.values(), key=lambda x:x['path'])))
    freeze = dict(schema='search128_saved_timing_v1', accepted_anchor=ANCHOR,
        limits=LIMITS, selection=RULE, panel=panel_pin, inputs=inputs_pin,
        protocol=descriptor(ROOT/PROTOCOL), implementation=descriptor(Path(__file__)),
        tests=descriptor(ROOT/TEST), model_calls=0, games=0, retrospective=True,
        no_player_admission=True, source_evidence_modified=False)
    result = save(out/'freeze.json', freeze)
    guard.check()
    print(canonical(dict(status='prepared_only', freeze=result, detailed_rows=panel['distinct_rows'])), flush=True)
    return result


def run(expected_sha, out=OUT):
    guard = Guard(out)
    fp = descriptor(out/'freeze.json')
    require(fp['sha256'] == expected_sha, 'freeze SHA mismatch')
    reader = Reader(guard=guard.check)
    freeze = reader.read(fp)
    require(freeze['limits'] == LIMITS and freeze['selection'] == RULE and freeze['accepted_anchor'] == ANCHOR,
            'frozen contract changed')
    require(freeze['implementation'] == descriptor(Path(__file__)) and
            freeze['protocol'] == descriptor(ROOT/PROTOCOL), 'execution source/protocol changed')
    save(out/'run_claim.json', dict(pid=os.getpid(), freeze_sha256=expected_sha,
        started_unix=guard.start_unix, started_monotonic=guard.start, limits=LIMITS))
    try:
        inputs = reader.read(freeze['inputs'])
        for pin in inputs['artifacts']:
            reader.add(pin)
        # Authenticate every finite input before the only corrected-source import.
        reader.consumed.update(reader.pins)
        reader.recheck()
        # A fresh run has only the finite consumed-file pins. Restore the exact
        # authenticated inventory authority for all compact-manifest row bindings;
        # do not reread/hash unconsumed historical files.
        authority(reader)
        from tools.future_training_rules_20261021 import verify_runtime_sources
        runtime = verify_runtime_sources()  # exactly once; no environment/model construction
        rows, bridges = population(reader)
        panel = reader.read(freeze['panel'])
        require(panel == select_panel(rows), 'frozen selection mismatch')
        detailed = []
        for selected in panel['rows']:
            row = bound_row(reader, selected)
            detailed.append(dict(source_row=selected['row'], memberships=selected['memberships'],
                                 source_freeze_sha256=selected['source_freeze_sha256'], **extract(row)))
            del row
        for bridge in bridges:
            reader.read(bridge['descriptor'])
        report = dict(schema='search128_saved_timing_report_v1', status='saved_timing_extracted',
            freeze_sha256=expected_sha, population=summarize_population(rows), panel=detailed,
            selection_slots=panel['slots'], reused_slots=panel['reused_slots'], missing_slots=panel['missing_slots'],
            bridge_excluded=bridges, scope='400 main games; 6 operations separately; no accepted bridge clock',
            interpretation=dict(retrospective=True, causal_claim=False, panel_is_population_estimate=False,
                exact_cpu_wait_partition=False, phase_wall_times_available=False,
                note='Handler includes search/inference/IPC/bookkeeping. CPU scope differs. Adjacent controls are different positions.'),
            model_calls=0, new_games=0, scientific_replays=0, bootstrap_runs=0,
            existing_summary_descriptors=[reader.pins[SOURCE_OUT+'/'+name] for name in
                                          ('boundary_analysis.json','performance.json')])
        reader.recheck()
        report_pin = save(out/'report.json', report)
        guard.check()
        completion = dict(status='saved_timing_extraction_completed', freeze_sha256=expected_sha,
            report=report_pin, runtime_sources=runtime, consumed_inputs=sorted(reader.consumed.values(), key=lambda x:x['path']),
            input_preservation='passed', detailed_rows=len(detailed), ordinary_rows=len(rows),
            elapsed_seconds=time.monotonic()-guard.start, peak_rss_bytes=guard.peak_rss,
            model_calls=0, new_games=0, source_evidence_modified=False,
            actual_process_exit_still_required=True, independent_readback_still_required=True)
        save(out/'completion.json', completion)
        guard.check()
        print(canonical({k:completion[k] for k in ('status','detailed_rows','ordinary_rows','elapsed_seconds')}), flush=True)
        return completion
    except BaseException as exc:
        save(out/'failure.json', dict(status='incomplete', error=repr(exc), traceback=traceback.format_exc(),
            freeze_sha256=expected_sha, model_calls=0, new_games=0, elapsed_seconds=time.monotonic()-guard.start))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        require(args.freeze_sha256 is not None, 'run requires --freeze-sha256')
        run(args.freeze_sha256)


if __name__ == '__main__':
    main()

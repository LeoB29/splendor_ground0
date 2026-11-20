"""Finite, outcome-blind saved256 panel; no model calls or evidence mutation.

The literal selection rule is saved before extraction. Twelve hash-selected
controls and six deliberately slow controls are reported separately. They are
not a probability sample for population latency or playing-strength estimates.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = ROOT / 'outputs/splendorust_search256_recovery_20261030_02'
CLOSEOUT = ROOT / 'outputs/splendorust_search256_recovery_20261030_02_closeout'
FREEZE_SHA = '6cf5f864e451926cf530c8b78468ac7f496fff159f51cab0d62d29b5c80d8149'
ACCEPTANCE_SHA = '40eb825c191ba57fd7e17738f4dd38426404592c7a29997f60c4fcc9ccc7c202'
EXIT_SHA = '640db910f69e377a6a24d340302a97a7aa6a1596eca612293e9a4dca739048d5'
SELECTION_DOMAIN = 'splendorust_search256_efficiency_20261031_systematic_v1'
PROGRESS = ('early', 'middle', 'late')
GROUPS = ('systematic', 'tail')
MODES = ('off', 'on')
SCHEMA = 'splendorust_search256_efficiency_panel_v1'


def require(ok, message):
    if not ok:
        raise ValueError('SEARCH256_EFFICIENCY_PANEL: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'workspace artifact')
    content = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def selection_rule():
    return dict(schema='search256_efficiency_selection_rule_v1',
        population='all11066 accepted main-game256 decisions, all400 games',
        hash_domain=SELECTION_DOMAIN,
        hash_encoding='UTF-8 domain + | + decimal game_index + | + decimal ply',
        progress='min(2, floor(3 * ply / final_game_plies)); early,middle,late',
        systematic='two ascending SHA256 priorities per progress/seat cell, globally distinct games',
        tail='one maximum trusted arrival per progress/seat cell, excluding all selected games',
        cell_order=[dict(progress=p, seat=s) for p in PROGRESS for s in (0, 1)],
        tie_break=['game_index', 'ply'], roots=18, systematic_roots=12, tail_roots=6,
        outcome_used=False, historical_latency_used_only_for_tail=True,
        population_latency_estimate=False, groups_must_be_reported_separately=True,
        source_seams_preserved=True, hypothetical_one_second_losses=False)


def priority(game, ply):
    return hashlib.sha256(f'{SELECTION_DOMAIN}|{game}|{ply}'.encode()).hexdigest()


def progress(ply, plies):
    require(type(ply) is int and type(plies) is int and 0 <= ply < plies,
            'genuine completed-game progress')
    return PROGRESS[min(2, 3 * ply // plies)]


def select_positions(candidates):
    """Pure fixed rule; only identity, progress, seat and latency are inspected."""
    candidates = list(candidates)
    require(len({(r['game_index'], r['ply']) for r in candidates}) == len(candidates),
            'unique source decision identities')
    cells = {(p, s): [] for p in PROGRESS for s in (0, 1)}
    for row in candidates:
        require(type(row['game_index']) is int and 0 <= row['game_index'] < 400 and
                type(row['seat']) is int and row['seat'] in (0, 1), 'game/seat domain')
        require(row['progress'] == progress(row['ply'], row['game_plies']), 'progress rule')
        seconds = row['original_receipt_seconds']
        require(type(seconds) in (float, int) and math.isfinite(seconds) and 0 < seconds <= 10.,
                'accepted finite research arrival')
        cells[row['progress'], row['seat']].append(row)
    selected = []
    used_games = set()
    for group in GROUPS:
        for cell, values in cells.items():
            order = sorted(values, key=(lambda r: (priority(r['game_index'], r['ply']), r['game_index'], r['ply']))
                           if group == 'systematic' else
                           (lambda r: (-r['original_receipt_seconds'], r['game_index'], r['ply'])))
            count = 2 if group == 'systematic' else 1
            admitted = 0
            for row in order:
                if row['game_index'] in used_games:
                    continue
                selected.append({**row, 'group': group,
                                 'selection_priority': priority(row['game_index'], row['ply'])})
                used_games.add(row['game_index'])
                admitted += 1
                if admitted == count:
                    break
            require(admitted == count, 'sufficient distinct games in each fixed cell')
    require(len(selected) == 18, 'fixed eighteen roots')
    return selected


class Reader:
    def __init__(self, boundary=lambda: None):
        self.boundary = boundary
        self.pins = {}
        self.consumed = {}

    def add(self, item):
        require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'artifact descriptor')
        path = (ROOT / item['path']).resolve()
        require(path.is_relative_to(ROOT), 'artifact stays inside workspace')
        old = self.pins.setdefault(item['path'], item)
        require(old == item, 'no contradictory artifact identities')

    def bound(self, path, *, expected_sha=None):
        self.boundary()
        path = Path(path).resolve()
        key = path.relative_to(ROOT).as_posix()
        content = path.read_bytes()
        item = dict(path=key, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        require((expected_sha is not None and item['sha256'] == expected_sha) or
                (expected_sha is None and self.pins.get(key) == item), 'authenticated source: ' + key)
        self.add(item)
        self.consumed[key] = item
        return json.loads(content)


def _source_context(reader):
    freeze = reader.bound(ACCEPTED / 'freeze.json', expected_sha=FREEZE_SHA)
    approval = reader.bound(CLOSEOUT / 'independent_acceptance_01.json', expected_sha=ACCEPTANCE_SHA)
    exit_receipt = reader.bound(CLOSEOUT / 'actual_guardian_exit_01.json', expected_sha=EXIT_SHA)
    require(approval['approved'] is True and approval['freeze_sha256'] == FREEZE_SHA and
            approval['actual_runner_watchdog_reader_exit_codes'] == [0, 0, 0] and
            approval['coverage']['main_games'] == 400 and approval['coverage']['model_reproductions'] == 48,
            'accepted complete400-game benchmark')
    # The owning exit is bound independently of any completion sentinel.
    require(exit_receipt is not None and approval['guardian_actual_exit']['exit_code'] == 0,
            'actual guardian exit evidence')
    for item in freeze['inputs'] + approval['inputs']:
        # The scientific freeze also pins external Python/runtime binaries.
        # This finite saved-game reader never consumes those external files.
        if (ROOT / item['path']).resolve().is_relative_to(ROOT):
            reader.add(item)
    manifest = reader.bound(ROOT / freeze['recovery_manifest']['path'])
    for key in ('original_tree', 'previous_tree'):
        reader.add(manifest[key])
        inventory = reader.bound(ROOT / manifest[key]['path'])
        for item in inventory['files']:
            reader.add(item)
    seal = reader.bound(ACCEPTED / 'completed_evidence_seal.json')
    for item in seal['artifacts']:
        reader.add(item)
    main = reader.bound(ACCEPTED / 'main_analysis.json')
    require(main['games'] == 400 if 'games' in main else True, 'accepted analysis scope')
    schedule = reader.bound(ACCEPTED / 'schedule.json')['main']
    require(len(schedule) == 400 and [s['game_index'] for s in schedule] == list(range(400)),
            'exact fixed main schedule')
    return freeze, schedule


def compact_census(reader, schedule):
    """400 accepted small replay proofs; no full-game audit or all-row reread."""
    from tools.splendorust_search256_recovery_20261030_02.verification import source_game_folder
    candidates = []
    for spec in schedule:
        folder = source_game_folder(spec)
        initial = reader.bound(folder / 'initial.json')
        proof = reader.bound(folder / 'online_replay.json')
        require(proof['status'] == 'passed' and proof['failed_moves'] == 0,
                'accepted completed compact game proof')
        plies = proof['transitions']
        require(all(initial[key] == value for key, value in spec.items()),
                'compact initial wrapper binds exact frozen game spec')
        start = initial['initial']['state']['current_player']
        seat = spec['challenger_seat']
        require(type(start) is int and start in (0, 1), 'initial acting seat')
        require(proof['scientific_arms']['challenger'] == dict(arm='search', mode='research', quota=256),
                'actual original256 compact timing source')
        latencies = proof['latencies']['challenger']
        indices = list(range((seat - start) % 2, plies, 2))
        require(len(indices) == len(latencies) and
                sum(len(x) for x in proof['latencies'].values()) == plies,
                'complete alternating-seat compact timing census')
        for ply, seconds in zip(indices, latencies, strict=True):
            candidates.append(dict(game_index=spec['game_index'], ply=ply, seat=seat,
                game_plies=plies, progress=progress(ply, plies), original_receipt_seconds=seconds))
    require(len(candidates) == 11066, 'all11066 accepted main256 decisions')
    return candidates


def bind_root(reader, selected, spec, freeze, index):
    from tools.splendorust_search256_recovery_20261030_02 import verification, adapters
    from tools.external_strength_screen_20261018 import audit as ledger
    from tools.future_training_rules_20261021 import decode_training_snapshot
    path = verification.source_move_path(spec, selected['ply'])
    row = reader.bound(path)
    require(row['game_index'] == selected['game_index'] and row['ply'] == selected['ply'] and
            row['stage'] == 'main' and row['player_role'] == 'challenger' and row['seat'] == selected['seat'] and
            row['mode'] == row['route'] == 'research' and row['quota'] == 256 and row['completed'] is True,
            'selected compact row matches full saved256 decision')
    require(row['clock']['status'] == 'ok' and row['clock']['accepted'] is True and
            row['clock']['receipt_elapsed_seconds'] == selected['original_receipt_seconds'],
            'compact timing exactly matches accepted original receipt')
    clock = reader.bound(ROOT / row['worker_clock_file'])
    require(clock == row['clock'], 'original trusted request receipt')
    worker = ROOT / row['worker_folder']
    request = reader.bound(worker / 'inputs' / (clock['request_id'] + '.json'))
    require(request == dict(player='search', observer=row['observer']), 'exact original lawful input')
    decode_training_snapshot(row['before']); decode_training_snapshot(row['after'])
    ledger.check_observer(row['before'], row['observer'])
    require(ledger.transition(row['before'], row['action']) == row['after'], 'independent selected transition replay')
    adapters.validate_result(clock['result'], row['observer'], 'research',
                             identity=adapters.profile_identity(freeze, 'research'))
    require(row['action'] == clock['result']['action'] and row['trace'] == clock['result']['diagnostics'],
            'full scientific action and trace binding')
    observer_seed = int(digest(row['observer'])[:16], 16)
    require(row['trace']['observer_seed'] == observer_seed, 'exact deterministic lawful-observation seed')
    name = f"{selected['group']}_{selected['progress']}_s{selected['seat']}_{index:02d}"
    return dict(**selected, panel_id=name, root_id=name,
        source_row=reader.consumed[path.relative_to(ROOT).as_posix()],
        source_clock=reader.consumed[row['worker_clock_file']],
        source_input=reader.consumed[(worker / 'inputs' / (clock['request_id'] + '.json')).relative_to(ROOT).as_posix()],
        before=row['before'], observer=row['observer'], original_result256=clock['result'],
        action=row['action'], trace=row['trace'], after=row['after'], observer_seed=observer_seed,
        original_game_seed_decimal=spec['seed_decimal'],
        historical_result_includes_full_callback_arrays=True,
        historical_retry_position=(selected['game_index'], selected['ply']) == (235, 30))


def build_panel(boundary=lambda: None):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    source_guard = verify_runtime_sources()
    reader = Reader(boundary)
    freeze, schedule = _source_context(reader)
    census = compact_census(reader, schedule)
    selected = select_positions(census)
    roots = [bind_root(reader, row, schedule[row['game_index']], freeze, i)
             for i, row in enumerate(selected)]
    return dict(schema=SCHEMA, selection=selection_rule(), roots=roots,
        source_guard=source_guard, accepted_source_freeze_sha256=FREEZE_SHA,
        census=dict(decisions=len(census), games=400, scalar_rows_sha256=digest(census),
            counts_by_cell={f'{p}_s{s}': sum(r['progress'] == p and r['seat'] == s for r in census)
                            for p in PROGRESS for s in (0, 1)}),
        inputs=sorted(reader.consumed.values(), key=lambda x: x['path']),
        source_scope='Accepted compact400-game replay proofs and eighteen full selected inputs/results/transitions; no repeated full-tree audit',
        new_models=0, new_games=0, source_evidence_modified=False)


def request_matrix(panel, phase):
    require(phase in ('overhead', 'diagnostic'), 'fixed measured phase')
    roots = panel['roots']
    require(len(roots) == 18 and len({r['root_id'] for r in roots}) == 18, 'fixed distinct roots')
    if phase == 'overhead':
        roots = [next(r for r in roots if r['group'] == 'systematic' and r['progress'] == p and r['seat'] == s)
                 for p in PROGRESS for s in (0, 1)]
    matrix = []
    pair_index = 0
    for repeat in range(4):
        ordered = list(enumerate(roots))
        shift = repeat * 5 % len(ordered)
        ordered = ordered[shift:] + ordered[:shift]
        if repeat % 2:
            ordered.reverse()
        for root_index, root in ordered:
            modes = MODES if (root_index + repeat) % 2 == 0 else tuple(reversed(MODES))
            for mode in modes:
                matrix.append(dict(phase=phase, index=len(matrix), sequence=len(matrix),
                    panel_id=root['root_id'], root_id=root['root_id'], pair_index=pair_index,
                    repeat=repeat, mode=mode, group=root['group'], route='research', quota=256))
            pair_index += 1
    require(len(matrix) == (48 if phase == 'overhead' else 144), 'fixed complete measured matrix')
    return matrix


build_matrix = request_matrix

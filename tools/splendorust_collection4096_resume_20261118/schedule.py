"""Fresh collection-only reservation with whole-game splits fixed before collection.

No model or search is used. A collision fails the reservation; there is no seed
redraw. Registry coverage is finite and disclosed rather than claiming all logs.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import subprocess
import sys
import time
from . import common as c

DOMAIN = 'splendorust_collection4096_20261115_v1'
RESERVED = c.OUT / 'reserved'
MAX_SCAN_BYTES = 8 * 1024**2
TRAIN_GAMES, VALIDATION_GAMES, OFFLINE_GAMES = 3072, 128, 128
COLLECTION_GAMES = TRAIN_GAMES + VALIDATION_GAMES + OFFLINE_GAMES
GAMES_PER_CELL, CELL_COUNT = 16, 208
# Qualification import compatibility; no scientific fitting is authorized here.
FIT_SEEDS = {'r0': 202611151, 'r1': 202611152}
BOOTSTRAP_SEEDS = {'offline': 202611151283}
EXPLICIT_PRIOR = (
    ('outputs/splendorust_independent_data_20261107_01/reserved/collection_schedule.json', 1152, 1152),
    ('outputs/splendorust_independent_data_20261107_01/reserved/qualification_schedule.json', 2, 2),
    ('outputs/splendorust_onepass_strength_preparation_20261106_01/schedule.json', 412, 206),
    ('outputs/splendorust_policy_head_20261104_01/assessment/schedule.json', 406, 203),
    ('outputs/splendorust_selfplay_pilot_resume2_20261103_01/assessment/schedule.json', 406, 203),
    ('outputs/splendorust_selfplay_pilot_resume2_20261103_01/game_schedule.json', 256, 256),
)
ACTIVE_SCORE_FREE_SCHEDULE = Path('outputs/splendorust_sixworld512_20261114_01/schedule.json')
ACTIVE_PREFIX = 'splendorust_sixworld512_'


def seed(*identity):
    return str(int.from_bytes(hashlib.sha256(c.canonical([DOMAIN, *identity]).encode()).digest(), 'big'))


def collection_spec(index):
    c.require(type(index) is int and 0 <= index < COLLECTION_GAMES, 'Fixed collection index')
    split = 'train' if index < TRAIN_GAMES else 'validation' if index < TRAIN_GAMES + VALIDATION_GAMES else 'offline'
    return dict(game_id=f'g{index:04d}', game_index=index, cell_index=index // GAMES_PER_CELL,
                split=split, seed_decimal=seed('collection', split, index),
                start_player=index % 2, quota=512, world_pool=3, cpuct=.4,
                purpose='independent_corrected_training_source')


def collection_schedule():
    return [collection_spec(index) for index in range(COLLECTION_GAMES)]


def qualification_schedule():
    return [dict(game_id=f'qualification_g{i}', game_index=COLLECTION_GAMES+2+i,
                 split='engineering', seed_decimal=seed('engineering_qualification', i),
                 start_player=i, quota=512, world_pool=3, cpuct=.4,
                 purpose='engineering_qualification_only') for i in (0, 1)]


def validate(collection):
    c.require(c.canonical(collection) == c.canonical(collection_schedule()), 'Exact typed collection reservation')
    setups = {x['seed_decimal'] for x in collection}
    engineering = {x['seed_decimal'] for x in qualification_schedule()}
    c.require(len(setups) == COLLECTION_GAMES and len(engineering) == 2 and not setups & engineering,
              'Fresh independent sources and disjoint excluded engineering starts')
    auxiliary = list(BOOTSTRAP_SEEDS.values()) + list(FIT_SEEDS.values())
    c.require(len(set(auxiliary)) == len(auxiliary) and not set(map(int, setups | engineering)) & set(auxiliary),
              'Disjoint qualification and future analysis seed reservations')
    for split, count in [('train', TRAIN_GAMES), ('validation', VALIDATION_GAMES), ('offline', OFFLINE_GAMES)]:
        rows = [x for x in collection if x['split'] == split]
        c.require(len(rows) == count and all(sum(x['start_player'] == seat for x in rows) == count // 2
                                           for seat in (0, 1)), 'Balanced split starts')
    c.require([x['game_index'] for x in collection] == list(range(COLLECTION_GAMES)), 'Complete fresh collection order')
    c.require(CELL_COUNT * GAMES_PER_CELL == COLLECTION_GAMES, 'Complete fixed collection cells')
    return dict(collection_games=COLLECTION_GAMES, train_games=TRAIN_GAMES,
                validation_games=VALIDATION_GAMES, offline_games=OFFLINE_GAMES,
                fresh_held_out_games=VALIDATION_GAMES + OFFLINE_GAMES,
                fresh_assessment_policy='Pool both fixed held-out partitions for the primary256-game assessment; no selection between them',
                strength_games=0, scientific_optimizer_updates=0, engineering_games=2,
                all_independent_setups=COLLECTION_GAMES + 2, balanced_starts=True,
                cells=CELL_COUNT, games_per_cell=GAMES_PER_CELL, auxiliary_seeds=len(auxiliary))


def registered_seed_inventory(boundary=lambda: None):
    targets = {int(x['seed_decimal']) for x in collection_schedule() + qualification_schedule()}
    targets.update(BOOTSTRAP_SEEDS.values()); targets.update(FIT_SEEDS.values())
    checked, skipped, explicit, collisions, blinded = [], [], [], [], []
    for relative, count, unique in EXPLICIT_PRIOR:
        boundary(); path = c.ROOT / relative; rows = c.read(path)
        previous = {int(x['seed_decimal']) for x in rows}
        c.require(len(rows) == count and len(previous) == unique, 'Complete explicit prior schedule')
        c.require(not targets & previous, 'Prior setup seed collision; no redraw')
        explicit.append(dict(artifact=c.artifact(path), rows=count, independent_setups=unique))
    boundary(); active = c.ROOT / ACTIVE_SCORE_FREE_SCHEDULE
    active_rows = c.read(active)
    c.require(type(active_rows) is dict and set(active_rows) == {'main', 'operational'},
              'Read only the original score-free current-study reservation')
    c.require(len(active_rows['main']) == 3600 and len(active_rows['operational']) == 6,
              'Complete current-study score-free reservation')
    active_seeds = {int(row['seed_decimal']) for rows in active_rows.values() for row in rows}
    c.require(len(active_seeds) == 1803 and not targets & active_seeds,
              'Disjoint from current study without reading its results')
    explicit.append(dict(artifact=c.artifact(active), rows=3606, independent_setups=1803, score_free=True))
    result = subprocess.run(['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json',
        '-g', '*schedule*.json', '-g', '*seed_registry*.json', '-g', '*reserved_panel*.json',
        '-g', '*freeze*.json', '-g', '*selection*.json'], cwd=c.ROOT, capture_output=True,
        text=True, check=True, timeout=120)
    def visit(value, name):
        if type(value) is int or type(value) is str and value.isdecimal():
            if int(value) in targets: collisions.append(name)
        elif type(value) is list:
            for item in value: visit(item, name)
        elif type(value) is dict:
            for item in value.values(): visit(item, name)
    for name in sorted(result.stdout.splitlines()):
        boundary(); path = (c.ROOT / name).resolve()
        if path.is_relative_to(c.OUT.resolve()): continue
        relative = path.relative_to(c.ROOT)
        if len(relative.parts) > 1 and relative.parts[1].startswith(ACTIVE_PREFIX):
            blinded.append(relative.as_posix()); continue
        if path.stat().st_size > MAX_SCAN_BYTES:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(), bytes=path.stat().st_size)); continue
        visit(c.read(path), name); checked.append(c.artifact(path))
    c.require(not collisions, 'Registered seed collision; preserve refusal and never redraw')
    return dict(status='passed', checked=checked, skipped_oversized=skipped, explicit_prior=explicit,
                current_study_result_files_not_opened=blinded, collisions=collisions, setup_seeds=COLLECTION_GAMES+2, bootstrap_seeds=BOOTSTRAP_SEEDS, fit_seeds=FIT_SEEDS,
                scope='Finite registered manifest/schedule/seed_registry/reserved_panel/freeze/selection JSON files up to8MiB; own output excluded; active six-world files blinded except its original score-free schedule. Not all raw logs.')


def _initial(seed_decimal, start_player):
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from tools.external_strength_screen_20261018 import rules
    state = rules.CorrectedEnv(seed=int(seed_decimal)).initial_state()
    state.current_player = state.start_player = start_player
    snapshot = rules.serialize_snapshot(state)
    decoded = decode_training_snapshot(snapshot)
    c.require(not decoded.terminal and decoded.turn_index == 0 and
              decoded.current_player == decoded.start_player == start_player and
              rules.serialize_snapshot(decoded) == snapshot, 'Corrected initial snapshot roundtrip')
    return snapshot


def prepare():
    """Exclusive bounded startup; root calls once before the execution freeze."""
    from tools.future_training_rules_20261021 import verify_runtime_sources
    mono, wall = time.monotonic(), time.time()
    def boundary():
        c.require(time.monotonic()-mono < 600 and time.time()-wall < 600, 'Reservation ten-minute real clock')
    c.require(not RESERVED.exists(), 'Exclusive reservation; no regeneration')
    sources = verify_runtime_sources()
    collection = collection_schedule()
    checks = validate(collection)
    inventory = registered_seed_inventory(boundary)
    initials = []
    for row in collection:
        boundary(); initials.append(dict(spec=row, initial=_initial(row['seed_decimal'], row['start_player'])))
    value = dict(collection_schedule=c.save(RESERVED/'collection_schedule.json', collection),
                 collection_initials=c.save(RESERVED/'collection_initial_states.json', initials),
                 qualification_schedule=c.save(RESERVED/'qualification_schedule.json', qualification_schedule()),
                 qualification_initials=c.save(RESERVED/'qualification_initial_states.json', [dict(spec=row,
                    initial=_initial(row['seed_decimal'], row['start_player'])) for row in qualification_schedule()]),
                 seed_inventory=c.save(RESERVED/'seed_inventory.json', inventory),
                 sampling=c.save(RESERVED/'sampling.json', dict(domain=DOMAIN,
                    fit_seeds=FIT_SEEDS, bootstrap_seeds=BOOTSTRAP_SEEDS, counts=checks,
                    scientific_training_authorized=False, strength_games_authorized=False)))
    c.require('torch' not in sys.modules, 'Model-free reservation only')
    value['reservation_receipt'] = c.save(RESERVED/'reservation_receipt.json', dict(
        status='reserved', domain=DOMAIN, artifacts=value, runtime_sources=sources, checks=checks,
        elapsed_seconds=time.monotonic()-mono, search_decisions=0, model_calls=0, games_played=0))
    return value


if __name__ == '__main__':
    print(c.canonical(prepare()))

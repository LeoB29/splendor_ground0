"""One prospective reservation for independent collection and matched assessment.

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

DOMAIN = 'splendorust_independent_data_20261107_v1'
RESERVED = c.OUT / 'reserved'
MAX_SCAN_BYTES = 8 * 1024**2
BOOTSTRAP_SEEDS = {'strength_primary': 202611071280, 'strength_r0': 202611071281,
                   'strength_r1': 202611071282, 'offline': 202611071283}
# Existing training seeds also seed NumPy's uint32 legacy RNG unchanged.
FIT_SEEDS = {'r0': 202611071, 'r1': 202611072}
EXPLICIT_PRIOR = (
    ('outputs/splendorust_onepass_strength_preparation_20261106_01/schedule.json', 412, 206),
    ('outputs/splendorust_policy_head_20261104_01/assessment/schedule.json', 406, 203),
    ('outputs/splendorust_selfplay_pilot_resume2_20261103_01/assessment/schedule.json', 406, 203),
    ('outputs/splendorust_selfplay_pilot_resume2_20261103_01/game_schedule.json', 256, 256),
)


def seed(*identity):
    return str(int.from_bytes(hashlib.sha256(c.canonical([DOMAIN, *identity]).encode()).digest(), 'big'))


def collection_spec(index):
    c.require(type(index) is int and 0 <= index < 1152, 'Fixed collection index')
    split = 'train' if index < 1024 else 'validation' if index < 1088 else 'offline'
    return dict(game_id=f'g{index:04d}', game_index=index, cell_index=index // 16,
                split=split, seed_decimal=seed('collection', split, index),
                start_player=index % 2, quota=512, world_pool=3, cpuct=.4,
                purpose='independent_corrected_training_source')


def collection_schedule():
    return [collection_spec(index) for index in range(1152)]


def qualification_schedule():
    return [dict(game_id=f'qualification_g{i}', game_index=1152+i,
                 split='engineering', seed_decimal=seed('engineering_qualification', i),
                 start_player=i, quota=512, world_pool=3, cpuct=.4,
                 purpose='engineering_qualification_only') for i in (0, 1)]


def narrow_ids():
    """Outcome-blind common subset with balanced first seats and no fit selection."""
    result = []
    for seat in (0, 1):
        choices = range(seat, 1024, 2)
        result.extend(sorted(choices, key=lambda i: seed('narrow_subset', i))[:96])
    return sorted(result)


def strength_spec(stage, repetition, pair, order):
    c.require(type(stage) is str and stage in ('operational', 'main'), 'Fixed strength stage')
    c.require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
    count = 3 if stage == 'operational' else 100
    c.require(type(pair) is int and 0 <= pair < count and type(order) is int and order in (0, 1),
              'Fixed paired-game indices')
    stage_pair = repetition * 3 + pair if stage == 'operational' else pair // 5 * 10 + repetition * 5 + pair % 5
    global_pair = stage_pair if stage == 'operational' else 6 + stage_pair
    return dict(stage=stage, candidate=f'r{repetition}_broad_joint', reference='original_ef8a',
                repetition=repetition, local_pair_index=pair, stage_pair_index=stage_pair,
                pair_index=global_pair, game_index=2 * global_pair + order,
                stage_game_index=2 * stage_pair + order, order_in_pair=order,
                seed_decimal=seed('strength', stage, repetition, pair),
                candidate_seat=(repetition + pair + order) % 2,
                candidate_simulations=128, reference_simulations=128,
                worlds_each=3, cpuct_each=.4, depth_each=64)


def strength_schedule():
    rows = [strength_spec('operational', rep, pair, order)
            for rep in (0, 1) for pair in range(3) for order in (0, 1)]
    rows += [strength_spec('main', rep, pair, order)
             for block in range(20) for rep in (0, 1)
             for pair in range(block * 5, block * 5 + 5) for order in (0, 1)]
    return rows


def validate(collection, strength):
    c.require(c.canonical(collection) == c.canonical(collection_schedule()), 'Exact typed collection reservation')
    c.require(c.canonical(strength) == c.canonical(strength_schedule()), 'Exact typed strength reservation')
    a, b = {x['seed_decimal'] for x in collection}, {x['seed_decimal'] for x in strength}
    c.require(len(a) == 1152 and len(b) == 206 and a.isdisjoint(b), 'Disjoint independent setup seeds')
    engineering = {x['seed_decimal'] for x in qualification_schedule()}
    c.require(len(engineering) == 2 and not engineering & (a | b), 'Disjoint excluded engineering starts')
    auxiliary = list(BOOTSTRAP_SEEDS.values()) + list(FIT_SEEDS.values())
    c.require(len(set(auxiliary)) == len(auxiliary) and not (set(map(int, a | b | engineering)) & set(auxiliary)),
              'Disjoint fit and analysis seeds')
    for split, count in [('train', 1024), ('validation', 64), ('offline', 64)]:
        rows = [x for x in collection if x['split'] == split]
        c.require(len(rows) == count and all(sum(x['start_player'] == seat for x in rows) == count // 2
                                           for seat in (0, 1)), 'Balanced split starts')
    c.require(len(narrow_ids()) == len(set(narrow_ids())) == 192, 'Fixed narrow subset')
    c.require([x['game_index'] for x in strength] == list(range(412)), 'Complete strength order')
    for i in range(0, 412, 2):
        pair = strength[i:i+2]
        c.require(pair[0]['seed_decimal'] == pair[1]['seed_decimal'] and
                  {x['candidate_seat'] for x in pair} == {0, 1}, 'Swapped-seat pair')
    return dict(collection_games=1152, train_games=1024, validation_games=64, offline_games=64,
                strength_games=412, strength_pairs=206, engineering_games=2, all_independent_setups=1360,
                narrow_subset_games=192, balanced_starts=True, auxiliary_seeds=len(auxiliary))


def registered_seed_inventory(boundary=lambda: None):
    targets = {int(x['seed_decimal']) for x in collection_schedule() + strength_schedule() + qualification_schedule()}
    targets.update(BOOTSTRAP_SEEDS.values()); targets.update(FIT_SEEDS.values())
    checked, skipped, explicit, collisions = [], [], [], []
    for relative, count, unique in EXPLICIT_PRIOR:
        boundary(); path = c.ROOT / relative; rows = c.read(path)
        previous = {int(x['seed_decimal']) for x in rows}
        c.require(len(rows) == count and len(previous) == unique, 'Complete explicit prior schedule')
        c.require(not targets & previous, 'Prior setup seed collision; no redraw')
        explicit.append(dict(artifact=c.artifact(path), rows=count, independent_setups=unique))
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
        if path.stat().st_size > MAX_SCAN_BYTES:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(), bytes=path.stat().st_size)); continue
        visit(c.read(path), name); checked.append(c.artifact(path))
    c.require(not collisions, 'Registered seed collision; preserve refusal and never redraw')
    return dict(status='passed', checked=checked, skipped_oversized=skipped, explicit_prior=explicit,
                collisions=collisions, setup_seeds=1360, bootstrap_seeds=BOOTSTRAP_SEEDS, fit_seeds=FIT_SEEDS,
                scope='Finite registered manifest/schedule/seed_registry/reserved_panel/freeze/selection JSON files up to8MiB; own output excluded. Not all raw logs.')


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
    collection, strength = collection_schedule(), strength_schedule()
    checks = validate(collection, strength)
    inventory = registered_seed_inventory(boundary)
    initials = []
    for row in collection:
        boundary(); initials.append(dict(spec=row, initial=_initial(row['seed_decimal'], row['start_player'])))
    paired = []
    for row in strength:
        boundary(); paired.append(dict(spec=row, initial=_initial(row['seed_decimal'], 0)))
    for i in range(0, 412, 2):
        c.require(c.canonical(paired[i]['initial']) == c.canonical(paired[i+1]['initial']), 'Paired exact initial states')
    value = dict(collection_schedule=c.save(RESERVED/'collection_schedule.json', collection),
                 collection_initials=c.save(RESERVED/'collection_initial_states.json', initials),
                 qualification_schedule=c.save(RESERVED/'qualification_schedule.json', qualification_schedule()),
                 qualification_initials=c.save(RESERVED/'qualification_initial_states.json', [dict(spec=row,
                    initial=_initial(row['seed_decimal'], row['start_player'])) for row in qualification_schedule()]),
                 strength_schedule=c.save(RESERVED/'strength_schedule.json', strength),
                 strength_initials=c.save(RESERVED/'strength_initial_states.json', paired),
                 seed_inventory=c.save(RESERVED/'seed_inventory.json', inventory),
                 sampling=c.save(RESERVED/'sampling.json', dict(domain=DOMAIN, narrow_game_ids=narrow_ids(),
                    fit_seeds=FIT_SEEDS, bootstrap_seeds=BOOTSTRAP_SEEDS, counts=checks)))
    c.require('torch' not in sys.modules, 'Model-free reservation only')
    value['reservation_receipt'] = c.save(RESERVED/'reservation_receipt.json', dict(
        status='reserved', domain=DOMAIN, artifacts=value, runtime_sources=sources, checks=checks,
        elapsed_seconds=time.monotonic()-mono, search_decisions=0, model_calls=0, games_played=0))
    return value


if __name__ == '__main__':
    print(c.canonical(prepare()))

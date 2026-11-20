"""Fresh outcome-blind paired starts; training data and fit seeds are reused.

The finite prior registry is disclosed. No model or search is used and a setup
collision refuses the reservation rather than drawing a replacement seed.
"""
from pathlib import Path
import hashlib
import subprocess
import sys
import time
from . import common as c

DOMAIN = 'splendorust_value_targets_20261110_v1'
RESERVED = c.OUT / 'reserved'
MAX_SCAN_BYTES = 8 * 1024**2
BOOTSTRAP_SEEDS = {'strength_primary': 202611101280, 'strength_r0': 202611101281,
                   'strength_r1': 202611101282, 'offline': 202611101283}
FIT_SEEDS = {'r0': 202611071, 'r1': 202611072}  # Deliberately paired with saved terminal controls.
EXPLICIT_PRIOR = (
    ('outputs/splendorust_independent_data_20261107_01/reserved/strength_schedule.json', 412, 206),
    ('outputs/splendorust_independent_data_20261107_01/reserved/collection_schedule.json', 1152, 1152),
    ('outputs/splendorust_independent_data_20261107_01/reserved/qualification_schedule.json', 2, 2),
    ('outputs/splendorust_onepass_strength_preparation_20261106_01/schedule.json', 412, 206),
    ('outputs/splendorust_policy_head_20261104_01/assessment/schedule.json', 406, 203),
    ('outputs/splendorust_selfplay_pilot_resume2_20261103_01/assessment/schedule.json', 406, 203),
    ('outputs/splendorust_selfplay_pilot_resume2_20261103_01/game_schedule.json', 256, 256),
)


def seed(*identity):
    return str(int.from_bytes(hashlib.sha256(c.canonical([DOMAIN, *identity]).encode()).digest(), 'big'))


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


def validate(strength):
    c.require(c.canonical(strength) == c.canonical(strength_schedule()), 'Exact typed fresh reservation')
    setups = {x['seed_decimal'] for x in strength}
    c.require(len(setups) == 206, '206 distinct new setups')
    auxiliary = list(BOOTSTRAP_SEEDS.values())
    c.require(len(set(auxiliary)) == len(auxiliary) and not set(map(int, setups)) & set(auxiliary),
              'Independent analysis and setup seeds')
    c.require(not set(auxiliary) & set(FIT_SEEDS.values()), 'New analysis seeds do not collide with reused fits')
    c.require([x['game_index'] for x in strength] == list(range(412)), 'Complete fixed assessment order')
    for i in range(0, 412, 2):
        pair = strength[i:i+2]
        c.require(pair[0]['seed_decimal'] == pair[1]['seed_decimal'] and
                  {x['candidate_seat'] for x in pair} == {0, 1}, 'Swapped seats, same setup')
    for rep in (0, 1):
        for stage, games in (('operational', 6), ('main', 200)):
            rows = [x for x in strength if x['repetition'] == rep and x['stage'] == stage]
            c.require(len(rows) == games and all(sum(x['candidate_seat'] == seat for x in rows) == games // 2
                                               for seat in (0, 1)), 'Balanced fixed repetition and stage')
    return dict(strength_games=412, main_games=400, operational_games=12, strength_pairs=206,
                all_independent_setups=206, new_collection_games=0, balanced_seats=True,
                new_auxiliary_seeds=len(auxiliary), reused_fit_seeds=FIT_SEEDS)


def registered_seed_inventory(boundary=lambda: None):
    # Fit seeds are intentionally reused for the controlled target comparison.
    targets = {int(x['seed_decimal']) for x in strength_schedule()} | set(BOOTSTRAP_SEEDS.values())
    checked, skipped, explicit, collisions = [], [], [], []
    for relative, count, unique in EXPLICIT_PRIOR:
        boundary(); path = c.ROOT / relative; rows = c.read(path)
        previous = {int(x['seed_decimal']) for x in rows}
        c.require(len(rows) == count and len(previous) == unique, 'Complete explicit prior schedule')
        c.require(not targets & previous, 'Prior setup collision; no redraw')
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
    c.require(not collisions, 'Registered seed collision; retain refusal and never redraw')
    return dict(status='passed', checked=checked, skipped_oversized=skipped, explicit_prior=explicit,
                collisions=collisions, setup_seeds=206, bootstrap_seeds=BOOTSTRAP_SEEDS,
                fit_seeds=FIT_SEEDS, fit_seed_policy='intentional paired reuse; excluded from new-seed collision check',
                scope='Finite registered manifest/schedule/seed_registry/reserved_panel/freeze/selection JSON up to8MiB; own output excluded. Not all raw logs.')


def _initial(seed_decimal, start_player=0):
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
    from tools.future_training_rules_20261021 import verify_runtime_sources
    mono, wall = time.monotonic(), time.time()
    def boundary():
        c.require(time.monotonic()-mono < 600 and time.time()-wall < 600, 'Reservation ten-minute real clock')
    c.require(not RESERVED.exists(), 'Exclusive reservation; no regeneration')
    sources = verify_runtime_sources(); strength = strength_schedule()
    checks = validate(strength); inventory = registered_seed_inventory(boundary)
    paired = []
    for row in strength:
        boundary(); paired.append(dict(spec=row, initial=_initial(row['seed_decimal'])))
    for i in range(0, 412, 2):
        c.require(c.canonical(paired[i]['initial']) == c.canonical(paired[i+1]['initial']), 'Paired exact initial states')
    value = dict(strength_schedule=c.save(RESERVED/'strength_schedule.json', strength),
                 strength_initials=c.save(RESERVED/'strength_initial_states.json', paired),
                 seed_inventory=c.save(RESERVED/'seed_inventory.json', inventory))
    c.require('torch' not in sys.modules, 'Model-free reservation only')
    value['reservation_receipt'] = c.save(RESERVED/'reservation_receipt.json', dict(
        status='reserved', domain=DOMAIN, artifacts=value, runtime_sources=sources, checks=checks,
        elapsed_seconds=time.monotonic()-mono, search_decisions=0, model_calls=0, games_played=0))
    return value


if __name__ == '__main__':
    print(c.canonical(prepare()))

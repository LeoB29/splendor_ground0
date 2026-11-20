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

DOMAIN = 'splendorust_full_corpus_20261110_v1'
RESERVED = c.ORIGINAL_OUT / 'reserved'
MAX_SCAN_BYTES = 8 * 1024**2
MATCHUPS = ('mixed_parent', 'terminal_parent', 'mixed_terminal')
MATCHUP_ARMS = {'mixed_parent':{'candidate':'mixed','control':'parent'},
                'terminal_parent':{'candidate':'terminal','control':'parent'},
                'mixed_terminal':{'candidate':'mixed','control':'terminal'}}
BOOTSTRAP_SEEDS = {m:{'strength_primary':202611101500+10*i,
                      'strength_r0':202611101501+10*i,'strength_r1':202611101502+10*i}
                   for i,m in enumerate(MATCHUPS)}
BOOTSTRAP_SEEDS['offline'] = 202611101530
FIT_SEEDS = {'r0': 202611071, 'r1': 202611072}  # Paired from-scratch warm starts for terminal/mixed full-corpus fits.
EXPLICIT_PRIOR = (
    ('outputs/splendorust_value_targets_20261110_01/reserved/strength_schedule.json', 412, 206),
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


def strength_spec(stage, repetition, pair, order, matchup):
    c.require(type(matchup) is str and matchup in MATCHUPS, 'Fixed matchup')
    c.require(type(stage) is str and stage in ('operational','main'), 'Fixed strength stage')
    c.require(type(repetition) is int and repetition in (0,1), 'Fixed repetition')
    count=3 if stage=='operational' else 100
    c.require(type(pair) is int and 0<=pair<count and type(order) is int and order in (0,1), 'Fixed pair identity')
    m=MATCHUPS.index(matchup);arms=MATCHUP_ARMS[matchup]
    stage_pair=m*6+repetition*3+pair if stage=='operational' else pair//5*30+m*10+repetition*5+pair%5
    global_pair=stage_pair if stage=='operational' else 18+stage_pair
    return dict(stage=stage,matchup=matchup,candidate=f'r{repetition}_{arms["candidate"]}_joint',
        reference='original_ef8a' if arms['control']=='parent' else f'r{repetition}_{arms["control"]}_joint',
        repetition=repetition,local_pair_index=pair,stage_pair_index=stage_pair,
        pair_index=global_pair,game_index=2*global_pair+order,stage_game_index=2*stage_pair+order,
        order_in_pair=order,seed_decimal=seed('strength',matchup,stage,repetition,pair),
        candidate_seat=(repetition+pair+order)%2,candidate_simulations=128,reference_simulations=128,
        worlds_each=3,cpuct_each=.4,depth_each=64)


def strength_schedule(matchup=None):
    c.require(matchup is None or matchup in MATCHUPS,'Known optional matchup filter')
    rows=[strength_spec('operational',r,p,o,m) for m in MATCHUPS for r in (0,1) for p in range(3) for o in (0,1)]
    rows += [strength_spec('main',r,p,o,m) for block in range(20) for m in MATCHUPS for r in (0,1)
             for p in range(block*5,block*5+5) for o in (0,1)]
    return rows if matchup is None else [r for r in rows if r['matchup']==matchup]


def auxiliary_seeds():
    return [v for m in MATCHUPS for v in BOOTSTRAP_SEEDS[m].values()]+[BOOTSTRAP_SEEDS['offline']]


def validate(strength):
    c.require(c.canonical(strength)==c.canonical(strength_schedule()),'Exact typed fresh three-matchup schedule')
    setups={x['seed_decimal'] for x in strength};auxiliary=auxiliary_seeds()
    c.require(len(setups)==618,'618 disjoint paired setups')
    c.require(len(set(auxiliary))==len(auxiliary) and not set(map(int,setups))&set(auxiliary), 'Separate analysis/setup seeds')
    c.require(not set(auxiliary)&set(FIT_SEEDS.values()), 'Analysis seeds distinct from paired fitting seeds')
    c.require([x['game_index'] for x in strength]==list(range(1236)), 'Complete fixed global assessment order')
    for i in range(0,1236,2):
        pair=strength[i:i+2]
        c.require(pair[0]['seed_decimal']==pair[1]['seed_decimal'] and {x['candidate_seat'] for x in pair}=={0,1},'Swapped-seat pair')
    for m in MATCHUPS:
        for rep in (0,1):
            for stage,n in (('operational',6),('main',200)):
                rows=[x for x in strength if x['matchup']==m and x['repetition']==rep and x['stage']==stage]
                c.require(len(rows)==n and all(sum(x['candidate_seat']==seat for x in rows)==n//2 for seat in (0,1)), 'Balanced fixed matchup/seed/stage')
    return dict(strength_games=1236,main_games=1200,operational_games=36,strength_pairs=618,
        all_independent_setups=618,new_collection_games=0,balanced_seats=True,matchups=list(MATCHUPS),
        new_auxiliary_seeds=len(auxiliary),paired_fit_seeds=FIT_SEEDS)


def registered_seed_inventory(boundary=lambda: None):
    # Fit seeds are intentionally reused for the controlled target comparison.
    targets = {int(x['seed_decimal']) for x in strength_schedule()} | set(auxiliary_seeds())
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
                collisions=collisions, setup_seeds=618, bootstrap_seeds=BOOTSTRAP_SEEDS,
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
        c.require(time.monotonic()-mono < 1800 and time.time()-wall < 1800, 'Reservation thirty-minute real clock')
    c.require(not RESERVED.exists(), 'Exclusive reservation; no regeneration')
    sources = verify_runtime_sources(); strength = strength_schedule()
    checks = validate(strength); inventory = registered_seed_inventory(boundary)
    paired = []
    for row in strength:
        boundary(); paired.append(dict(spec=row, initial=_initial(row['seed_decimal'])))
    for i in range(0, 1236, 2):
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

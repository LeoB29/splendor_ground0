"""Fixed fresh source schedules, lifecycle limits and replayable resource prices."""
import hashlib
import subprocess
from . import common as c
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import rules

DOMAIN = 'splendorust_selfplay_pilot_20261102_v1'
FIT_SEED = 202611012

def schedule(scope=None):
    scope = c.SCOPE if scope is None else scope
    c.require(scope in ('qualification', 'pilot'), 'Declared self-play scope')
    rows = []
    for index in range(4 if scope == 'qualification' else 256):
        seed = str(int.from_bytes(hashlib.sha256(c.canonical([DOMAIN, scope, 'source', index]).encode()).digest(), 'big'))
        if scope == 'qualification':
            split = 'fixture_train' if index < 2 else 'fixture_validation' if index == 2 else 'fixture_assessment'
        else:
            split = 'train' if index < 192 else 'validation' if index < 224 else 'offline'
        rows.append(dict(game_index=index, seed_decimal=seed, split=split, teacher_simulations=512,
                         purpose='engineering_qualification_only' if scope == 'qualification' else 'single_generation_learning_pilot'))
    return rows

def validate_spec(spec):
    c.require(type(spec) is dict and type(spec.get('game_index')) is int and 0 <= spec['game_index'] < c.MAX_SOURCE_GAMES,
              'Fixed source-game index')
    c.require(spec == schedule()[spec['game_index']], 'Exact prospectively frozen source specification')
    return spec

def initial_states():
    rows = []
    for spec in schedule():
        initial = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        state = decode_training_snapshot(initial)
        c.require(not state.terminal and state.turn_index == 0, 'Fresh corrected source setup')
        rows.append(dict(spec=spec, initial=initial))
    return rows

def collection_blocks():
    specs = schedule()
    return [dict(worker='capture_on' if start == 0 else f'collect_{start // c.SOURCE_GAMES_PER_WORKER:03d}',
                 block_index=start // c.SOURCE_GAMES_PER_WORKER,
                 game_indices=list(range(start, min(start + c.SOURCE_GAMES_PER_WORKER, len(specs)))),
                 specs=specs[start:start + c.SOURCE_GAMES_PER_WORKER])
            for start in range(0, len(specs), c.SOURCE_GAMES_PER_WORKER)]

def replay_game_indices():
    return list(range(4)) if c.SCOPE == 'qualification' else [17 * i for i in range(16)]

def source_lifecycle_inventory():
    limits = {'capture_off': 16}
    for block in collection_blocks():
        limits[block['worker']] = len(block['game_indices']) * c.MAX_PLIES + (16 if block['block_index'] == 0 else 0)
    limits['replay_on'] = c.MAX_SOURCE_REPLAY_DECISIONS
    result = {name: dict(route='candidate', quota=512, capture_targets=name != 'capture_off',
                        graph_request_id='request-00000001', entrypoint_request_id='request-00000002',
                        ordinary_first_request_id='request-00000003', ordinary_requests=count, startup_requests=2)
              for name, count in limits.items()}
    c.require(len(result) == c.MAX_SOURCE_WORKER_LIFECYCLES, 'Exact source lifecycle ceiling')
    return result

def fixed(path, value):
    if path.exists():
        c.require(c.read(path) == value, 'Prepared immutable document changed')
        return c.artifact(path)
    return c.save(path, value)

def seed_inventory(extra_seeds=()):
    own = {int(row['seed_decimal']) for row in schedule()}
    extra = {int(x) for x in extra_seeds}
    other_scope = 'pilot' if c.SCOPE == 'qualification' else 'qualification'
    c.require(len(own) == c.MAX_SOURCE_GAMES and not own.intersection(extra)
              and not own.intersection(int(x['seed_decimal']) for x in schedule(other_scope)),
              'Disjoint source, assessment and other-scope seed sets')
    seeds = own | extra
    found = subprocess.run(['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json',
                            '-g', '*schedule*.json', '-g', '*seed_registry*.json',
                            '-g', '*reserved_panel*.json', '-g', '*freeze*.json'], cwd=c.ROOT,
                           capture_output=True, text=True, check=True, timeout=120)
    checked, skipped, collisions = [], [], []
    def visit(value, name):
        if type(value) is int or type(value) is str and value.isdecimal():
            if int(value) in seeds:
                collisions.append(name)
        elif type(value) is list:
            for child in value: visit(child, name)
        elif type(value) is dict:
            for child in value.values(): visit(child, name)
    for name in sorted(found.stdout.splitlines()):
        path = (c.ROOT / name).resolve()
        if path.is_relative_to(c.OUT.resolve()): continue
        if path.stat().st_size > 8 * c.MiB:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(), bytes=path.stat().st_size)); continue
        visit(c.read(path), name); checked.append(c.artifact(path))
    c.require(not collisions, 'Existing registered seed collision; no substitution after observation')
    return dict(status='passed', domain=DOMAIN, scope=c.SCOPE, seeds=sorted(str(n) for n in seeds),
                checked=checked, skipped_oversized=skipped, collisions=collisions,
                search_scope='Registered JSON manifests/schedules/reserved panels/seed registries/freezes <=8MiB; no arbitrary logs')

def prepare(extra_seeds=()):
    result = dict(game_schedule=fixed(c.OUT / 'game_schedule.json', schedule()),
                  initial_states=fixed(c.OUT / 'initial_states.json', initial_states()),
                  collection_blocks=fixed(c.OUT / 'collection_blocks.json', collection_blocks()))
    path = c.OUT / 'seed_freshness.json'
    if not path.exists(): c.save(path, seed_inventory(extra_seeds))
    receipt = c.read(path)
    expected = sorted({x['seed_decimal'] for x in schedule()} | {str(int(x)) for x in extra_seeds})
    c.require(receipt['status'] == 'passed' and receipt['scope'] == c.SCOPE and not receipt['collisions']
              and receipt['domain'] == DOMAIN and receipt['seeds'] == expected, 'Exact fixed seed inventory')
    c.verify(receipt['checked']); result['seed_freshness'] = c.artifact(path)
    return result

from .resources import resource_plan, source_forecast
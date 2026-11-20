"""Fixed engineering fixtures, distinct from every later learning sample."""
import hashlib
import json
import subprocess
from . import common as c
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import rules

DOMAIN = 'splendorust_selfplay_engineering_qualification_20261101_v1'
FIT_SEED = 202611011

def schedule():
    result = []
    for index in range(4):
        seed = str(int.from_bytes(hashlib.sha256(c.canonical([DOMAIN, index]).encode()).digest(), 'big'))
        split = 'fixture_train' if index < 2 else 'fixture_validation' if index == 2 else 'fixture_assessment'
        result.append(dict(game_index=index, seed_decimal=seed, split=split,
                           teacher_simulations=512, purpose='engineering_qualification_only'))
    return result

def validate_spec(spec):
    c.require(type(spec) is dict and type(spec.get('game_index')) is int
              and 0 <= spec['game_index'] < 4, 'Fixed qualification game index')
    c.require(spec == schedule()[spec['game_index']], 'Exact frozen fixture specification')
    return spec

def initial_states():
    result = []
    for spec in schedule():
        initial = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        state = decode_training_snapshot(initial)
        c.require(not state.terminal and state.turn_index == 0, 'Fresh corrected fixture setup')
        result.append(dict(spec=spec, initial=initial))
    return result

def fixed(path, value):
    if path.exists():
        c.require(c.read(path) == value, 'Prepared fixture document changed')
        return c.artifact(path)
    return c.save(path, value)

def seed_inventory():
    seeds = {int(item['seed_decimal']) for item in schedule()}
    c.require(len(seeds) == 4, 'Four independent fixture seeds')
    command = ['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json',
               '-g', '*schedule*.json', '-g', '*seed_registry*.json',
               '-g', '*reserved_panel*.json', '-g', '*freeze*.json']
    found = subprocess.run(command, cwd=c.ROOT, capture_output=True, text=True, check=True, timeout=120)
    checked, skipped, collisions = [], [], []
    def visit(value, name):
        if type(value) is int or type(value) is str and value.isdecimal():
            if int(value) in seeds:
                collisions.append(name)
        elif type(value) is list:
            for child in value:
                visit(child, name)
        elif type(value) is dict:
            for child in value.values():
                visit(child, name)
    for name in sorted(found.stdout.splitlines()):
        path = (c.ROOT / name).resolve()
        if path.is_relative_to(c.OUT.resolve()):
            continue
        if path.stat().st_size > 8*c.MiB:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(), bytes=path.stat().st_size))
            continue
        visit(c.read(path), name)
        checked.append(c.artifact(path))
    c.require(not collisions, 'Existing registered seed collision; never substitute after observation')
    return dict(status='passed', domain=DOMAIN, seeds=sorted(str(n) for n in seeds),
                checked=checked, skipped_oversized=skipped, collisions=collisions,
                scope='Registered JSON manifests/schedules/reserved panels/seed registries/freezes <=8MiB; no arbitrary logs',
                purpose='engineering_only_never_training_or_strength_sample')

def prepare():
    result = dict(game_schedule=fixed(c.OUT/'game_schedule.json', schedule()),
                  initial_states=fixed(c.OUT/'initial_states.json', initial_states()))
    path = c.OUT/'seed_freshness.json'
    if not path.exists():
        c.save(path, seed_inventory())
    receipt = c.read(path)
    c.require(receipt['status'] == 'passed' and not receipt['collisions'] and receipt['domain'] == DOMAIN,
              'Exact fixed fixture seed namespace')
    c.verify(receipt['checked'])
    result['seed_freshness'] = c.artifact(path)
    return result

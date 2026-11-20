"""Strict views of the already reserved two-checkpoint paired assessment.

No starts are regenerated. Each local runtime retains immutable global identities.
"""
from copy import deepcopy
from functools import lru_cache
from itertools import product
import hashlib
import json
import subprocess
from . import common as c

DOMAIN = 'splendorust_onepass_strength_20261106_v1'
BOOTSTRAP_SEED = 20261106128
INDIVIDUAL_SEEDS = (20261106129, 20261106130)
PAIRS = {'operational': 3, 'main': 100}
VARIANTS = ('control', 'candidate')
PHASES = ('0..19', '20..39', '40+')
PRIVATE = ('neither', 'own', 'opponent', 'both')
PREPARATION = c.ROOT / 'outputs/splendorust_onepass_strength_preparation_20261106_01'
PINS = {
    'schedule.json': 'a936e49600b48abfb72ca4efd8b77196d6fbcb4f041f10a109244ac6f77dd3e3',
    'initial_states.json': 'e82c5379ada5e63407fe0da6bf0b8014d1afe4d6cbc7d0d1d2f9c4036d1c12cc',
}


def require(value, message):
    if not value:
        raise ValueError('ONEPASS_ASSESSMENT_DESIGN: ' + message)


def repetition(value=None):
    value = c.REPETITION if value is None else value
    require(type(value) is int and value in (0, 1), 'exact repetition identity')
    return value


def _same_type_value(actual, expected):
    return type(actual) is type(expected) and actual == expected


def _reserved_spec(stage, rep, pair, order):
    """Independent strict expected metadata; never create an environment."""
    require(type(stage) is str and stage in PAIRS, 'stage')
    rep = repetition(rep)
    require(type(pair) is int and 0 <= pair < PAIRS[stage], 'local pair')
    require(type(order) is int and order in (0, 1), 'pair order')
    payload = json.dumps([DOMAIN, stage, rep, pair], sort_keys=True, separators=(',', ':'))
    seed = str(int.from_bytes(hashlib.sha256(payload.encode()).digest(), 'big'))
    stage_pair = rep * 3 + pair if stage == 'operational' else pair // 5 * 10 + rep * 5 + pair % 5
    global_pair = stage_pair if stage == 'operational' else 6 + stage_pair
    return dict(stage=stage, candidate=f'r{rep}_broad_joint', reference='original_ef8a',
                repetition=rep, local_pair_index=pair, stage_pair_index=stage_pair,
                pair_index=global_pair, game_index=2 * global_pair + order,
                stage_game_index=2 * stage_pair + order, order_in_pair=order,
                seed_decimal=seed, candidate_seat=(rep + pair + order) % 2,
                candidate_simulations=128, reference_simulations=128, worlds_each=3,
                cpuct_each=.4, depth_each=64)


def validate_reserved(rows, initials):
    require(type(rows) is list and type(initials) is list and len(rows) == len(initials) == 412,
            'complete reserved sample')
    expected = [_reserved_spec('operational', rep, pair, order)
                for rep in (0, 1) for pair in range(3) for order in (0, 1)]
    expected += [_reserved_spec('main', rep, pair, order)
                 for block in range(20) for rep in (0, 1)
                 for pair in range(block * 5, block * 5 + 5) for order in (0, 1)]
    for row, initial, wanted in zip(rows, initials, expected):
        require(type(row) is dict and row.keys() == wanted.keys() and
                all(_same_type_value(row[k], v) for k, v in wanted.items()),
                'strict reserved specification including numeric types')
        require(type(initial) is dict and initial.keys() == {'spec', 'initial'} and
                type(initial['spec']) is dict and initial['spec'].keys() == wanted.keys() and
                all(_same_type_value(initial['spec'][k], v) for k, v in wanted.items()),
                'strict initial-state specification binding')
        state = initial['initial']
        require(type(state) is dict and state.get('ruleset_id') == 'splendor_base_qualified_rules_20261018_v1'
                and state.get('catalogue_sha256') ==
                'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'
                and type(state.get('state')) is dict and state['state'].get('terminal') is False
                and type(state['state'].get('turn_index')) is int and state['state']['turn_index'] == 0,
                'corrected initial identity and fresh start')
    for i in range(0, 412, 2):
        require(c.canonical(initials[i]['initial']) == c.canonical(initials[i + 1]['initial']),
                'same full initial state within swapped pair')
    require(len({r['seed_decimal'] for r in rows}) == 206, 'all setup seeds distinct')
    return rows


@lru_cache(maxsize=1)
def _reserved():
    values = {}
    for name, sha in PINS.items():
        raw = (PREPARATION / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == sha, 'reserved artifact changed: ' + name)
        values[name] = json.loads(raw)
    rows, initials = values['schedule.json'], values['initial_states.json']
    validate_reserved(rows, initials)
    return {(r['stage'], r['repetition'], r['local_pair_index'], r['order_in_pair']): (r, i['initial'])
            for r, i in zip(rows, initials)}


def _spec(stage, pair, order, rep=None):
    rep = repetition(rep)
    require(type(stage) is str and stage in PAIRS and type(pair) is int and 0 <= pair < PAIRS[stage]
            and type(order) is int and order in (0, 1), 'local game identity')
    row, _ = _reserved()[stage, rep, pair, order]
    return dict(stage=stage, candidate='candidate', pair_index=pair, game_index=2 * pair + order,
                seed_decimal=row['seed_decimal'], challenger_seat=row['candidate_seat'],
                challenger_quota=128, opponent_quota=128, challenger_mode='candidate', opponent_mode='control',
                challenger_variant='candidate', opponent_variant='control', challenger_cpuct=.4, opponent_cpuct=.4,
                repetition=rep, checkpoint_name=row['candidate'], global_pair_index=row['pair_index'],
                global_game_index=row['game_index'], reserved_stage_pair_index=row['stage_pair_index'])


def expected_seed(stage, pair_index, rep=None):
    return _spec(stage, pair_index, 0, rep)['seed_decimal']


def make_schedule(stage=None, repetition=None):
    rep = globals()['repetition'](repetition)
    if stage is None:
        return make_schedule('operational', rep) + make_schedule('main', rep)
    require(type(stage) is str and stage in PAIRS, 'fixed stage')
    return [_spec(stage, pair, order, rep) for pair in range(PAIRS[stage]) for order in (0, 1)]


def validate_spec(value, repetition=None):
    require(type(value) is dict, 'game object')
    rep = globals()['repetition'](repetition)
    stage, pair, index = (value.get(k) for k in ('stage', 'pair_index', 'game_index'))
    require(type(stage) is str and stage in PAIRS and type(pair) is int and 0 <= pair < PAIRS[stage]
            and type(index) is int and index in (2 * pair, 2 * pair + 1), 'exact local game identity')
    expected = _spec(stage, pair, index % 2, rep)
    require(all(k in value and _same_type_value(value[k], v) for k, v in expected.items()),
            'exact typed specification and correct repetition/global identity')
    return expected


def validate_schedule(values, stage=None, repetition=None):
    wanted = make_schedule(stage, repetition)
    require(type(values) is list and len(values) == len(wanted), 'complete fixed schedule')
    for value, expected in zip(values, wanted):
        require(validate_spec(value, repetition) == expected and value.keys() == expected.keys(),
                'fixed order with no enriched schedule records')
    return values


def initial_snapshot(spec):
    from tools.future_training_rules_20261021 import decode_training_snapshot
    value = validate_spec(spec)
    state = deepcopy(_reserved()[value['stage'], value['repetition'], value['pair_index'], value['game_index'] % 2][1])
    decode_training_snapshot(state)
    return state


def replay_domain():
    return DOMAIN + '_r' + str(repetition()) + '_replay_v1'


def replay_rule():
    return dict(schema='onepass_assessment_replay_rule_v1', repetition=repetition(), domain=replay_domain(),
                maximum=96, main_primary_cells='variant x acting seat x absolute-ply phase x hidden-reserve category',
                main_primary_maximum=48, operational_cells='local game_index x variant', operational_maximum=12,
                main_temporal_cells='local pair-index third x variant x acting seat x absolute-ply phase',
                main_temporal_maximum=36, pair_index_thirds=[[0, 32], [33, 66], [67, 99]],
                phases=list(PHASES), private_categories=list(PRIVATE),
                per_cell='minimum SHA256 of [domain,stage,pair_index,game_index,ply,variant]',
                selection_order='main primary, operational, main temporal; fixed Cartesian cell order; deduplicate identities',
                duplicate_selection=False, selection_uses_outcomes=False, selection_uses_timing=False,
                failed_moves_excluded=True, all_available_cells_selected=True,
                within_variant_full_scientific_result_equality_required=True, cross_variant_output_equality_required=False)


def fixed(path, value):
    if path.exists():
        require(c.canonical(c.read(path)) == c.canonical(value), 'prepared artifact changed')
        return c.artifact(path)
    return c.save(path, value)


def prepare():
    rows = make_schedule()
    initials = [dict(**spec, initial=initial_snapshot(spec)) for spec in rows]
    return dict(schedule=fixed(c.OUT / 'schedule.json', rows),
                initial_states=fixed(c.OUT / 'initial_states.json', initials),
                replay_rule=fixed(c.OUT / 'replay_rule.json', replay_rule()))


def seed_inventory():
    """Recheck registered seeds; only authenticated own reservation is excluded."""
    _reserved()
    targets = {int(row[0]['seed_decimal']) for row in _reserved().values()} | {BOOTSTRAP_SEED, *INDIVIDUAL_SEEDS}
    found = subprocess.run(['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json', '-g', '*schedule*.json',
                            '-g', '*seed_registry*.json', '-g', '*reserved_panel*.json', '-g', '*freeze*.json'],
                           cwd=c.ROOT, capture_output=True, text=True, check=True, timeout=120)
    checked, skipped, collisions = [], [], []
    def visit(value, name):
        if type(value) is int or type(value) is str and value.isdecimal():
            if int(value) in targets:
                collisions.append(name)
        elif type(value) is list:
            for item in value: visit(item, name)
        elif type(value) is dict:
            for item in value.values(): visit(item, name)
    own = c.STUDY_OUT.resolve()
    for name in sorted(found.stdout.splitlines()):
        path = (c.ROOT / name).resolve()
        if path.is_relative_to(own) or path.is_relative_to(PREPARATION.resolve()):
            continue
        if path.stat().st_size > 8 * c.MiB:
            skipped.append(dict(path=name, bytes=path.stat().st_size)); continue
        visit(c.read(path), name); checked.append(c.artifact(path))
    require(not collisions, 'registered seed collision; never replace reserved seeds')
    return dict(status='passed', domain=DOMAIN, new_setup_count=206, bootstrap_seed=BOOTSTRAP_SEED,
                individual_seeds=list(INDIVIDUAL_SEEDS), prior_seed_reuse=False,
                checked=checked, skipped_oversized=skipped, collisions=collisions,
                authenticated_own_reservation=[c.artifact(PREPARATION / n) for n in PINS],
                scope='Finite registered files up to8MiB, excluding this execution and authenticated own preparation; not all raw logs.')


def phase(ply): return PHASES[0] if ply < 20 else PHASES[1] if ply < 40 else PHASES[2]
def temporal_block(pair): return 0 if pair < 33 else 1 if pair < 67 else 2


def select_replay_sample(rows, maximum=None):
    if maximum is None: maximum = c.MAX_REPLAY_DECISIONS
    require(type(maximum) is int and maximum == c.MAX_REPLAY_DECISIONS == 96, 'fixed per-repetition replay bound')
    primary, ops, temporal, seen = {}, {}, {}, set()
    eligible = failed = 0
    for row in rows:
        spec = validate_spec(row); stage, game, pair = spec['stage'], spec['game_index'], spec['pair_index']
        ply, variant = row.get('ply'), row.get('variant')
        require(type(ply) is int and 0 <= ply < c.MAX_PLIES and variant in VARIANTS, 'replay identity')
        identity = (stage, game, ply)
        require(identity not in seen, 'duplicate played decision'); seen.add(identity)
        require(type(row.get('completed')) is bool, 'explicit completion')
        if not row['completed']: failed += 1; continue
        seat = row.get('seat'); role = 'challenger' if seat == spec['challenger_seat'] else 'opponent'
        require(type(seat) is int and seat in (0, 1) and row.get('player_role') == role
                and variant == spec[role + '_variant'] and type(row.get('quota')) is int
                and row['quota'] == c.ROUTE_QUOTAS[variant] and row.get('arm') == 'search'
                and row.get('route') == row.get('mode') == variant, 'role/seat/quota replay binding')
        view = row['observer']['observation']
        require(type(view['current_player']) is int and view['current_player'] == seat, 'acting observer seat')
        own = any(x['source'] == 'deck' for x in view['players'][seat]['reserves'])
        other = any(x['source'] == 'deck' for x in view['players'][1 - seat]['reserves'])
        private = 'both' if own and other else 'own' if own else 'opponent' if other else 'neither'
        key = hashlib.sha256(json.dumps([replay_domain(), stage, pair, game, ply, variant], separators=(',', ':')).encode()).hexdigest()
        item = dict(stage=stage, candidate='candidate', repetition=spec['repetition'], checkpoint_name=spec['checkpoint_name'],
                    game_index=game, pair_index=pair, global_game_index=spec['global_game_index'],
                    global_pair_index=spec['global_pair_index'], ply=ply, variant=variant, player_role=role, seat=seat,
                    quota=c.ROUTE_QUOTAS[variant], route=variant, phase=phase(ply), private_category=private,
                    selection_sha256=key, observer_sha256=c.digest(row['observer']))
        maps = [(ops, (game, variant))] if stage == 'operational' else [
            (primary, (variant, seat, phase(ply), private)), (temporal, (temporal_block(pair), variant, seat, phase(ply)))]
        for mapping, cell in maps:
            if cell not in mapping or key < mapping[cell]['selection_sha256']: mapping[cell] = item
        eligible += 1
    groups = [('main_primary', primary, list(product(VARIANTS, (0, 1), PHASES, PRIVATE))),
              ('operational', ops, list(product(range(6), VARIANTS))),
              ('main_temporal', temporal, list(product(range(3), VARIANTS, (0, 1), PHASES)))]
    decisions, selected, coverage = [], set(), {}
    for name, mapping, cells in groups:
        for cell in cells:
            item = mapping.get(cell)
            if item is None: continue
            identity = (item['stage'], item['game_index'], item['ply'])
            if identity not in selected: decisions.append(item); selected.add(identity)
        coverage[name] = dict(available_cells=len(mapping), maximum_cells=len(cells),
                             empty_cells=[list(cell) for cell in cells if cell not in mapping],
                             cells=[dict(cell=list(cell), stage=mapping[cell]['stage'], game_index=mapping[cell]['game_index'],
                                         ply=mapping[cell]['ply']) for cell in cells if cell in mapping])
    require(len(decisions) <= maximum, 'bounded coverage selection')
    return dict(schema='onepass_assessment_replay_selection_v1', repetition=repetition(), rule=replay_rule(),
                maximum_decisions=maximum, eligible_completed_decisions=eligible, excluded_failed_decisions=failed,
                decisions=decisions, coverage=coverage, selection_uses_outcomes=False,
                selection_uses_timing=False, failed_moves_excluded=True)

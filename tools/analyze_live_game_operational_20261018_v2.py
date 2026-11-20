"""Versioned initial-envelope repair; sealed outcomes never enter its report."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/live_game_operational_20261018_01'
FITS = (161001, 161002, 161003)
GAME_ORDER = (0, 2, 4, 1, 3, 5)
LIVE_NAMESPACE = 'splendor_corrected_live_operational_20261018_v1'
SEARCH_TAG = 'splendor_live_operational_search_v1'
SEED_TAG = 'splendor_live_operational_referee_v1'
AUDIT_DOMAIN = 'splendor_live_operational_world_audit_v1'
AUDIT_SEED = 202610186401
LOCAL_PROFILE = 'four_candidate128_corrected_live_operational_20261018_v1'
LOCAL_SCHEMA = 'corrected_live_planner_observer_v1'
RULESET = 'splendor_base_qualified_rules_20261018_v1'
CATALOGUE = 'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'
OPPONENT_PROFILE = 'outputs/splendorust_120_qualification_20261018_01/selected_profile.json'
OPPONENT_SHA256 = '6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599'
PRIOR_LOCAL = ROOT / 'outputs/corrected_planner_qualification_20261018_01'
TRANSFER_INDICES = (0, 6, 12)
VALIDATION_GAMES = (1, 2, 5)
METADATA = frozenset(('command', 'identity', 'affinity', 'topology', 'thread_environment'))
THREADS = frozenset(('BLIS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
ORIGINAL_ANALYZER = 'tools/analyze_live_game_operational_20261018.py'
ORIGINAL_SHA256 = '7995dcadac82dce6369bbf7cbd6f0266ea44edb6b80c4d9de3fa31baef839b35'
REPAIR_PROTOCOL = 'docs/engineering/live_game_operational_20261018_analysis_repair.md'
REPAIR_TEST = 'tests/test_live_game_operational_analyzer_repair_20261018.py'


def require(value, message):
    # Messages are check labels only: never interpolate sealed values or payloads.
    if not value: raise ValueError('Independent operational audit: ' + message)


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def finite(value): return type(value) in (int, float) and math.isfinite(value)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def check_preflight(value):
    require(type(value['returncode']) is int and value['returncode'] == 0, 'Successful frozen preflight')


def local_profile():
    return {'profile_id': LOCAL_PROFILE, 'schema': LOCAL_SCHEMA, 'ruleset_id': RULESET,
        'catalogue_sha256': CATALOGUE, 'fits': list(FITS), 'candidates': 4, 'worlds': 128,
        'backend': 'unchanged_serial', 'parent_profile_id': 'four_candidate128_corrected_rules_20261018_v1',
        'public_namespace': LIVE_NAMESPACE, 'public_search_tag': SEARCH_TAG,
        'fresh_worker_each_decision': True, 'trace_delivery_inside_move_clock': True}


def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def verify_repair_freeze(base, expected_sha256):
    """Authenticate only the prospective repair; the original freeze remains intact."""
    require(type(expected_sha256) is str and len(expected_sha256) == 64 and
            all(c in '0123456789abcdef' for c in expected_sha256), 'Explicit repair freeze authentication required')
    path = base/'analyzer_repair_02/freeze.json'
    require(file_hash(path) == expected_sha256, 'Repair freeze hash')
    value = read(path)
    require(value['schema'] == 'live_operational_analyzer_repair_v2' and
            value['original_freeze'] == artifact(base/'freeze.json') and
            value['original_analyzer'] == artifact(ROOT/ORIGINAL_ANALYZER) and
            value['original_analyzer']['sha256'] == ORIGINAL_SHA256, 'Repair lineage authentication')
    launch = read(base/'launch.json')
    require(value['original_launch'] == artifact(base/'launch.json') and
            value['original_completion'] == artifact(base/'completion.json') and
            value['original_final_input_check'] == artifact(base/'final_input_check.json') and
            value['original_deadline_epoch_seconds'] == launch['epoch_seconds']+43200 and
            value['maximum_seconds'] == launch['maximum_seconds'] == 43200 and
            value['final_reserve_seconds'] == 600, 'Original execution boundary retained')
    expected_failure = [artifact(base/name) for name in ('independent_analysis_execution_01.log',
        'independent_analysis_execution_01.json', 'sealed/independent_analysis_failure_01.json')]
    require(value['preserved_failure'] == expected_failure, 'Original failure preservation')
    paths = set()
    for item in value['inputs']:
        require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'}, 'Repair artifact descriptor')
        source = (ROOT/item['path']).resolve()
        require(source.is_relative_to(ROOT) and source not in paths and artifact(source) == item, 'Repair input hash/size/path')
        paths.add(source)
    required = (Path(__file__), ROOT/REPAIR_TEST, ROOT/REPAIR_PROTOCOL,
                base/'analyzer_repair_02/tests.json', base/'analyzer_repair_02/tests.log', base/'analyzer_repair_02/analyzer.diff')
    require(all(p.resolve() in paths for p in required), 'Complete repair source/test/protocol evidence')
    check_preflight(read(base/'analyzer_repair_02/tests.json'))
    return {'freeze_sha256': expected_sha256, 'original_analyzer_sha256': ORIGINAL_SHA256,
            'files': len(paths), 'all_hashes_and_sizes_match': True}


def schedule():
    return [{'game': game, 'pair': game//2, 'fit': FITS[game//2],
             'opponent_seat': (game//2+game%2)%2, 'local_seat': 1-(game//2+game%2)%2,
             'seed': str(int(digest([SEED_TAG, LIVE_NAMESPACE, game//2]), 16))} for game in GAME_ORDER]


def context_for(game, turn):
    require(type(game) is int and game in range(6) and type(turn) is int and turn in range(400), 'Operational context range')
    return {'mode': 'operational', 'namespace': LIVE_NAMESPACE, 'pair': game//2, 'game': game,
            'turn': turn, 'fit': FITS[game//2], 'stream': digest([SEARCH_TAG, LIVE_NAMESPACE, game]),
            'identity': [LIVE_NAMESPACE, game, turn]}


def sample_worlds(identity):
    priority = [(digest([AUDIT_DOMAIN, 'world', AUDIT_SEED, identity, wi]), wi) for wi in range(128)]
    return sorted(wi for _, wi in sorted(priority)[:8])


def root_cells(observer, support):
    view = observer['observation']; actor = view['current_player']; fit = observer['context']['fit']
    private = [any(r['source'] != 'visible' for r in p['reserves']) for p in view['players']]
    own, other = private[actor], private[1-actor]
    label = 'both' if own and other else 'own' if own else 'opponent' if other else 'neither'
    phase = 0 if view['turn_index'] < 20 else 1 if view['turn_index'] < 40 else 2
    return [f'ordinary:{fit}:{phase}:{actor}:{label}'] + ([f'support:{fit}:{support}'] if support in (1, 2, 3) else [])


def stress_selection(roots):
    """Outcome-free roots contain id/game/ply/observer/candidate_support only."""
    cells = {f'ordinary:{fit}:{phase}:{seat}:{privacy}': {'eligible': [], 'selected': None}
             for fit in FITS for phase in range(3) for seat in (0, 1)
             for privacy in ('neither', 'own', 'opponent', 'both')}
    cells.update({f'support:{fit}:{support}': {'eligible': [], 'selected': None}
                  for fit in FITS for support in (1, 2, 3)})
    ordered = sorted(roots, key=lambda r: (r['game'], r['ply']))
    require(len({r['id'] for r in roots}) == len(roots), 'Duplicate local root identity')
    lookup = {r['id']: r for r in ordered}
    def priority(identifier):
        row = lookup[identifier]
        return (digest([AUDIT_DOMAIN, 'stress', AUDIT_SEED, row['observer']['context']['identity']]), row['game'], row['ply'])
    for root in ordered:
        for key in root_cells(root['observer'], root['candidate_support']): cells[key]['eligible'].append(root['id'])
    for value in cells.values():
        if value['eligible']: value['selected'] = min(value['eligible'], key=priority)
    selected = sorted({v['selected'] for v in cells.values() if v['selected'] is not None})
    return cells, selected


def check_initialization(value, saved, freeze, module):
    require(METADATA <= value.keys() and {k: v for k, v in value.items() if k not in METADATA and k != 'result'} == saved,
            'Initialization receipt copy')
    require(value['command'] == [freeze['executable'], '-B', '-u', '-m', module], 'Worker command identity')
    require(value['affinity'] == [freeze['topology']['selected_logical_processor']] and value['topology'] == freeze['topology'],
            'Worker CPU/topology')
    require(value['thread_environment'] == {k: '1' for k in THREADS}, 'Numerical thread environment')
    identity = value['identity']; python = [p for p in value['telemetry']['process_tree'] if p['role'] == 'python']
    require(type(identity) is dict and set(identity) == {'pid', 'created'} and type(identity['pid']) is int and
            identity['pid'] > 0 and finite(identity['created']) and len(python) == 1 and
            python[0]['pid'] == identity['pid'] and python[0]['created'] == identity['created'] and
            Path(python[0]['executable']).resolve() == Path(freeze['executable']).resolve(), 'Worker process identity')
    require(value['status'] == 'ok' and value['result']['pid'] == identity['pid'] and value['result']['threads'] == 1,
            'Worker initialization success')


def check_proof(proof, observer, wrapper, mode):
    record = wrapper['decision']['record']; support = len(record['candidates']['actions'])
    indices = list(range(128)) if mode == 'full' else sample_worlds(observer['context']['identity'])
    require(mode in ('full', 'sample'), 'Unknown audit mode')
    transitions = sum(o['turns'] for wi in indices for o in record['worlds'][wi]['outcomes'])
    expected = {'status': 'passed', 'profile_id': LOCAL_PROFILE, 'ruleset_id': RULESET, 'catalogue_sha256': CATALOGUE,
        'observer_sha256': digest(observer), 'result_sha256': digest(wrapper), 'decision_sha256': digest(wrapper['decision']),
        'trace_sha256': wrapper['trace']['sha256'], 'trace_bytes': wrapper['trace']['bytes'],
        'root_identity': observer['context']['identity'], 'fit': observer['context']['fit'], 'mode': mode,
        'recorded_worlds': 128, 'recorded_continuations': 128*support, 'candidate_support': support,
        'audited_world_indices': indices, 'audited_worlds': len(indices), 'audited_continuations': len(indices)*support,
        'audited_transitions': transitions, 'raw035_choices_reexecuted': transitions-len(indices)*support,
        'proposal_model_reexecuted': True, 'proposal_selection_independent': True, 'all_world_records_checked': True,
        'all_root_checks_passed': True, 'all_continuations_replayed': mode == 'full', 'old_outcomes_used_as_ground_truth': False,
        'transition_oracle': 'frozen_external_screen_dictionary_ledger',
        'shared_dependencies': ['qualified corrected model/feature kernels', 'corrected legal enumeration/catalogue/serialization',
                                'frozen independent world/proposal/trajectory audit primitives']}
    require(canonical(proof) == canonical(expected), 'Complete bound audit proof')
    return {'worlds': len(indices), 'continuations': len(indices)*support, 'transitions': transitions,
            'raw_choices': transitions-len(indices)*support}


PLAYER_FAILURES = frozenset(('timeout', 'worker_exit', 'malformed', 'wrong_request', 'illegal', 'unsolicited', 'write_error'))


def check_game_ending(game, state, failure):
    """Check sealed outcome fields internally; return no winner or score."""
    require(game['complete'] is True and game['final'] == state, 'Sealed final state')
    if failure is None:
        require(game['reason'] == 'natural' and state['state']['terminal'] is True and
                game['winner'] == state['state']['winner'] and 'failing_player' not in game, 'Natural ending/tiebreak')
    else:
        status, actor, player = failure
        require(status in {'forfeit_'+s for s in PLAYER_FAILURES} | {'forfeit_internal_budget_exhausted'} and
                state['state']['terminal'] is False and game['reason'] == status and
                type(game['winner']) is int and game['winner'] == 1-actor and game['failing_player'] == player,
                'Authorized player forfeit')
    require(game['winner'] is None or type(game['winner']) is int and game['winner'] in (0, 1), 'Sealed outcome type')


class Audit:
    def __init__(self, base, repair_freeze_sha256=None):
        self.base = Path(base).resolve()
        self.repair_freeze_sha256 = repair_freeze_sha256
        self.worker_files, self.audit_files, self.decision_files = set(), set(), set()
        self.clocks, self.initializations, self.proofs = [], [], []
        self.statuses, self.work = {}, Counter()
        self.latencies = {}; self.roots = []; self.actual_transitions = 0
        self.local_recorded_worlds = self.local_recorded_continuations = 0
        self.identities = set(); self.game_costs = []

    def boundary(self):
        require(0 <= time.time()-self.launch['epoch_seconds'] < 43200, 'Shared twelve-hour execution/audit cap')

    def bound(self, item, expected=None):
        require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'}, 'Artifact descriptor')
        path = (ROOT/item['path']).resolve()
        require(path.is_relative_to(ROOT) and (expected is None or path == Path(expected).resolve()) and
                artifact(path) == item, 'Artifact hash/size/path binding')
        return path

    def freeze(self):
        self.repair = verify_repair_freeze(self.base, self.repair_freeze_sha256)
        self.completion = read(self.base/'completion.json'); self.progress = read(self.base/'operational_progress.json')
        self.frozen = read(self.base/'freeze.json'); self.launch = read(self.base/'launch.json')
        require(self.completion['status'] == 'completed_pending_independent_analysis' and
                self.completion['failure'] is None and self.completion['worker_processes_closed'] is True and
                self.completion['progress'] == artifact(self.base/'operational_progress.json') and
                {k: v for k, v in self.completion.items() if k not in ('worker_processes_closed', 'progress')} == self.progress,
                'Completed supervisor and progress binding')
        require(self.launch['freeze'] == artifact(self.base/'freeze.json') and self.launch['maximum_seconds'] == 43200 and
                self.launch['torch_loaded_in_supervisor'] is False and self.launch['outcomes_closed'] is True and
                finite(self.launch['epoch_seconds']), 'Original launch/freeze identity')
        self.boundary()
        paths, sizes = set(), 0
        for item in self.frozen['inputs']:
            self.boundary(); path = self.bound(item)
            require(path not in paths, 'Duplicate frozen input'); paths.add(path); sizes += item['bytes']
        constants = {'ruleset_id': RULESET, 'catalogue_sha256': CATALOGUE, 'local_profile_id': LOCAL_PROFILE,
            'games': 6, 'pairs': 3, 'seconds_per_move': 120, 'initialization_seconds': 60, 'audit_seconds': 600,
            'execution_cap_seconds': 43200, 'post_game_reserve_seconds': 14400, 'final_reserve_seconds': 600,
            'evidence_cap_bytes': 2*1024**3, 'minimum_free_bytes': 5*1024**3, 'max_process_tree_rss_bytes': 2*1024**3,
            'maximum_plies': 400, 'validation_requests': 12, 'sample_worlds': 8, 'audit_seed': AUDIT_SEED,
            'audit_domain': AUDIT_DOMAIN, 'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True}
        require(all(canonical(self.frozen.get(k)) == canonical(v) for k, v in constants.items()), 'Frozen operational protocol')
        prior = ROOT/'outputs/splendorust_120_qualification_20261018_01'
        require(all((ROOT/i['path']).resolve() in paths for i in read(prior/'freeze.json')['inputs']) and
                all(p.resolve() in paths for p in prior.rglob('*') if p.is_file() and '__pycache__' not in p.parts),
                'Previous qualification preservation')
        require(self.frozen['opponent_profile'] == artifact(ROOT/OPPONENT_PROFILE) and
                file_hash(ROOT/OPPONENT_PROFILE) == OPPONENT_SHA256, 'Qualified fixed opponent profile')
        self.profile = read(ROOT/OPPONENT_PROFILE)
        require(read(self.base/'local_profile.json') == local_profile(), 'Exact transferred local profile')
        check_preflight(read(self.base/'preflight_tests.json'))
        required = [self.base/name for name in ('local_profile.json', 'preflight_tests.json', 'schedule.json',
                    'validation_cases.json', 'seed_collision_check.json')]
        required += [ROOT/ORIGINAL_ANALYZER, ROOT/'tools/corrected_planner_qualification_20261018/verify.py',
            ROOT/'tools/analyze_external_strength_screen_20261018_02.py',
            ROOT/'tools/external_strength_screen_20261018/rules.py', ROOT/'tools/external_strength_screen_20261018/audit.py']
        required += list((ROOT/'tools/live_game_operational_20261018').glob('*.py'))
        require(all(p.resolve() in paths for p in required), 'Analyzer/profile/dependency freeze coverage')
        self.registry = read(ROOT/'outputs/proposal_breadth_arena_128/models_lock.json')
        self.sources = {'files': len(paths), 'bytes': sizes, 'all_hashes_and_sizes_match': True}
        require(read(self.base/'schedule.json') == schedule(), 'Independent paired schedule')
        self.cpu = self.frozen['topology']['selected_logical_processor']
        for path in (ROOT, ROOT/'src'):
            if str(path) not in sys.path: sys.path.insert(0, str(path))
        from tools.external_strength_screen_20261018 import rules, audit as ledger
        from tools.corrected_planner_qualification_20261018 import verify as recorded
        from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics, initial_snapshot, descriptive
        self.rules, self.ledger, self.recorded = rules, ledger, recorded
        self.check_clock, self.check_diagnostics = check_clock, check_diagnostics
        self.initial_snapshot, self.descriptive = initial_snapshot, descriptive
        require('torch' not in sys.modules and rules.RULESET_ID == RULESET and rules.CATALOGUE_SHA256 == CATALOGUE,
                'Model-free corrected-rule dependencies')
        self.catalogue = read(ROOT/'outputs/external_strength_screen_20261018_02/catalogue.json')
        require(digest(self.catalogue) == CATALOGUE, 'Corrected catalogue semantic identity')
        self.seed_collisions(paths)

    def seed_collisions(self, paths):
        value = read(self.base/'seed_collision_check.json'); targets = {int(s['seed']) for s in schedule()}
        require(value['namespace'] == LIVE_NAMESPACE and value['distinct_seeds'] == 3 and value['collisions'] == [], 'Seed collision receipt')
        def contains(item):
            if type(item) is int: return item in targets
            if type(item) is str: return item.isdecimal() and int(item) in targets
            if type(item) is list: return any(contains(v) for v in item)
            if type(item) is dict: return any(contains(v) for v in item.values())
            return False
        checked = set()
        for field in ('checked', 'skipped_oversize'):
            for item in value[field]:
                path = self.bound(item)
                require(path in paths and path not in checked and not path.is_relative_to(self.base), 'Historical seed inventory identity')
                checked.add(path)
                require((item['bytes'] <= 8*1024**2) is (field == 'checked'), 'Seed inventory size rule')
                if field == 'checked': require(not contains(read(path)), 'Historical referee seed collision')
        self.seed_scope = {'checked_files': len(value['checked']), 'oversize_files_not_scanned': len(value['skipped_oversize']),
                           'scope': 'Declared frozen manifests/schedules/seed registries, excluding files over 8 MiB'}

    def process_identity(self, receipts, player):
        owners = {}
        binary = (ROOT/self.profile['native_binary']).resolve()
        native = [i for i in self.profile['artifacts'] if (ROOT/i['path']).resolve() == binary]
        require(len(native) == 1, 'Native binary artifact')
        allowed = {'python', 'console:python'} | ({'rust', 'console:rust'} if player == 'opponent' else set())
        for receipt in receipts:
            for process in receipt['telemetry']['process_tree']:
                role = process['role']; executable = Path(process['executable']).resolve()
                require(role in allowed and type(process['pid']) is int and process['pid'] > 0 and
                        finite(process['created']), 'Allowed process tree')
                if role == 'python': require(executable == Path(self.frozen['executable']).resolve(), 'Python executable')
                elif role == 'rust':
                    require(executable == binary and process['executable_sha256'] == native[0]['sha256'], 'Native executable')
                else:
                    require(executable == (Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve(), 'Exact console helper')
                identity = (process['pid'], process['created'], str(executable), process['executable_sha256'],
                            canonical({k: v for k, v in process.items() if k.startswith('startup_affinity_')}))
                require(role not in owners or owners[role] == identity, 'Worker replacement or identity mutation')
                owners[role] = identity
                self.identities.add((process['pid'], process['created']))
            require(receipt['telemetry']['processes_observed'] == len(owners) and 'python' in owners and
                    (player != 'opponent' or 'rust' in owners), 'Complete descendant accounting')

    def receipts(self, row, name, player, audit=False):
        folder = self.base/('audits' if audit else 'workers')/name
        require({p.name for p in folder.glob('request-*.json')} == {'request-00000001.json'}, 'Exactly one request per worker')
        init = read(self.bound(row['initialization_artifact'], folder/'initialization.json'))
        receipt = read(self.bound(row['receipt'], folder/'request-00000001.json'))
        module = 'tools.splendorust_120_qualification_20261018.worker' if player == 'opponent' else 'tools.live_game_operational_20261018.worker'
        check_initialization(init, row['initialization'], self.frozen, module)
        require({k: v for k, v in receipt.items() if k != 'result'} == row['clock'], 'Copied decision/audit clock')
        self.check_clock(init, None, 60, self.cpu); self.check_clock(receipt, 'request-00000001', 600 if audit else 120, self.cpu)
        self.process_identity([init, receipt], player)
        self.initializations.append(init); self.clocks.append((player, audit, receipt))
        (self.audit_files if audit else self.worker_files).add(folder.resolve())
        return receipt

    def context(self, observer):
        context = observer['context']
        if context['mode'] == 'operational':
            expected = context_for(context['game'], context['turn'])
            require(observer['observation']['current_player'] == 1-((context['game']//2+context['game']%2)%2), 'Local acting seat')
        else:
            require(context['mode'] == 'legacy_transfer' and type(context['fixture_index']) is int and
                    context['fixture_index'] in TRANSFER_INDICES, 'Transfer context')
            old = read(PRIOR_LOCAL/'observations'/f'{context["fixture_index"]:02d}.json')
            expected = {'mode': 'legacy_transfer', 'fixture_index': context['fixture_index'], **old['context']}
            require(observer['observation'] == old['observation'], 'Transfer lawful observation')
        require(canonical(context) == canonical(expected) and observer['schema'] == LOCAL_SCHEMA and
                observer['observation']['turn_index'] == context['identity'][2], 'Complete public context identity')

    def local_record(self, row, wrapper, legal):
        observer = row['observer']; self.context(observer)
        require(set(wrapper) == {'decision', 'trace', 'ruleset_id', 'catalogue_sha256', 'profile_id', 'observer_sha256'} and
                wrapper['profile_id'] == LOCAL_PROFILE and wrapper['ruleset_id'] == RULESET and
                wrapper['catalogue_sha256'] == CATALOGUE and wrapper['observer_sha256'] == digest(observer), 'Local wrapper identity')
        trace = self.bound(row['trace'], self.base/'workers'/row['id']/'continuations.jsonl.gz')
        require(wrapper['trace'] == {'bytes': trace.stat().st_size, 'sha256': file_hash(trace),
                    'records': 128*len(wrapper['decision']['record']['candidates']['actions'])}, 'Local full trace artifact')
        decision = wrapper['decision']; record = decision['record']
        require(set(decision) == {'action', 'record'} and record['profile'] == LOCAL_PROFILE and
                record['ruleset_id'] == RULESET and record['catalogue_sha256'] == CATALOGUE and
                record['root_identity'] == observer['context']['identity'] and record['planner'] == 'reference', 'Local record identity')
        root = self.recorded.world_from_observer(observer)
        model = self.registry['models'][f'anchor_{observer["context"]["fit"]}']
        proposal = self.recorded.derive_proposal(observer, root, legal, record['scores'], record['proposal']['learned_logits'], model)
        require(record['proposal'] == proposal and record['schedule'] == proposal['schedule'] and
                record['candidates'] == proposal['candidates'], 'Recorded model/proposal/schedule reconstruction')
        support = len(record['candidates']['actions']); require(1 <= support <= 4 and len(record['worlds']) == 128, 'Complete fixed local budget')
        actor = observer['observation']['current_player']; turn = observer['observation']['turn_index']
        expected_privacy = {label: sum(r['source'] != 'visible' for r in observer['observation']['players'][seat]['reserves'])
                            for label, seat in (('own', actor), ('opponent', 1-actor))}
        require(record['privacy'] == expected_privacy, 'Local public privacy metadata')
        totals = [0]*support
        for wi, world in enumerate(record['worlds']):
            self.boundary(); seed = proposal['schedule']['world_seeds'][wi]
            expected_world = self.recorded.world_from_observer(observer, seed)
            require(set(world) == {'seed', 'state_sha256', 'outcomes'} and world['seed'] == seed and
                    world['state_sha256'] == digest(expected_world['state']) and len(world['outcomes']) == support, 'All-world identity/completeness')
            for ci, outcome in enumerate(world['outcomes']):
                require(set(outcome) == {'reason', 'winner', 'turns', 'trace_sha256', 'final_state_sha256', 'wdl'} and
                        outcome['reason'] in ('completed', 'forced_pass_deadlock') and
                        (outcome['winner'] is None or type(outcome['winner']) is int and outcome['winner'] in (0, 1)) and
                        type(outcome['turns']) is int and 1 <= outcome['turns'] <= 400-turn and
                        type(outcome['wdl']) in (int, float) and outcome['wdl'] == (0 if outcome['winner'] is None else 1 if outcome['winner'] == actor else -1),
                        'Admissible all-world outcome entry')
                require(all(type(outcome[k]) is str and len(outcome[k]) == 64 and all(c in '0123456789abcdef' for c in outcome[k])
                            for k in ('trace_sha256', 'final_state_sha256')), 'Continuation hash fields')
                totals[ci] += int(outcome['wdl'])
        choice = max(range(support), key=lambda index: totals[index])
        require(type(record['choice']) is int and record['choice'] == choice and record['means'] == [v/128 for v in totals] and
                decision['action'] == record['candidates']['actions'][choice] and decision['action'] in legal, 'All-world arithmetic/stable tie/action')
        return support

    def decision(self, row, snapshot, name, player, stage):
        self.boundary()
        require(row['id'] == name and row['player'] == player and type(row['correct']) is bool, 'Decision identity')
        path = self.base/'decision_attempts'/(name+'.json')
        direct = read(path); self.decision_files.add(path.resolve())
        require(all(k in row and row[k] == value for k, value in direct.items()), 'Stored decision attempt binding')
        observer = row['observer']
        clean = {**{k: deepcopy(v) for k, v in observer.items() if k != 'context'}, 'schema': self.rules.OBSERVATION_SCHEMA}
        self.ledger.check_observer(snapshot, clean if player == 'local' else observer)
        if player == 'local': self.context(observer)
        receipt = self.receipts(row, name, player)
        key = stage+':'+player
        self.statuses.setdefault(key, Counter())[row['status']] += 1
        if receipt['status'] != 'ok':
            require(receipt['status'] in PLAYER_FAILURES and row['status'] == 'forfeit_'+receipt['status'] and
                    row['correct'] is False and 'after_sha256' not in row and 'result_sha256' not in row,
                    'Unscored hard error or promoted failed decision')
            return None
        result = receipt['result']; require(row['result_sha256'] == digest(result), 'Decision result digest')
        legal = [self.rules.serialize_action(a) for a in self.rules.legal_actions(self.rules.deserialize_snapshot(snapshot))]
        if player == 'local':
            self.local_record(row, result, legal); action = result['decision']['action']
        else:
            identity = {'player': 'opponent', 'observer_sha256': digest(observer), 'legal_actions_sha256': digest(legal),
                'legal_count': len(legal), 'ruleset': RULESET, 'catalogue_sha256': CATALOGUE, 'profile_sha256': OPPONENT_SHA256}
            require(type(result['legal_count']) is int and all(result.get(k) == v for k, v in identity.items()), 'Opponent result identity')
            if result['status'] == 'internal_budget_exhausted':
                require(result == {**identity, 'status': 'internal_budget_exhausted', 'internal_budget_seconds': 108,
                    'opponent_id': 'splendorust', 'profile_id': self.profile['profile_id'],
                    'reason': 'Internal fixed-work deadline exceeded'} and row['status'] == 'forfeit_internal_budget_exhausted' and
                    row['correct'] is False and 'after_sha256' not in row, 'Internal exhaustion handling')
                return None
            require(set(result) == set(identity) | {'status', 'action', 'diagnostics'} and result['status'] == 'completed', 'Opponent completed schema')
            action = result['action']; require(action in legal, 'Opponent legal action')
            self.work.update(self.check_diagnostics(result['diagnostics'], observer, action, legal, self.profile))
        require(row['status'] == 'completed' and row['correct'] is True, 'Completed decision status')
        after = self.ledger.transition(snapshot, action)
        require(row['after_sha256'] == digest(after), 'Actual dictionary successor')
        self.latencies.setdefault(key, []).append(receipt['receipt_elapsed_seconds'])
        return result, after

    def proof(self, name, decision_row, wrapper, mode):
        row_path = self.base/'audit_rows'/(name+'.json'); row = read(row_path)
        require(row['id'] == name and row['mode'] == mode and row['correct'] is True and
                row['observer_sha256'] == digest(decision_row['observer']) and row['result_sha256'] == digest(wrapper), 'Audit row identity')
        receipt = self.receipts(row, name, 'audit', audit=True)
        require(receipt['status'] == 'ok' and receipt['result'] == row['verification'], 'Completed full/sample audit receipt')
        require(finite(row['total_seconds']) and row['total_seconds'] >=
                row['initialization']['elapsed_seconds']+row['clock']['elapsed_seconds']-0.1, 'Audit wall-cost accounting')
        counts = check_proof(row['verification'], decision_row['observer'], wrapper, mode)
        self.proofs.append((name, mode, counts))
        return counts

    def validation(self):
        cases = read(self.base/'validation_cases.json'); require(len(cases) == 12, 'Twelve transfer cases')
        results, counts = [], Counter()
        for index, case in enumerate(cases):
            fixture = TRANSFER_INDICES[index//4]; game = VALIDATION_GAMES[index//4]
            mode = 'legacy_transfer' if index%4 < 2 else 'operational'
            kind = 'original' if index%2 == 0 else 'hidden_variant'
            require(set(case) == {'index', 'fixture', 'mode', 'kind', 'fit', 'snapshot', 'observer', 'reference_receipt'} and
                    canonical([case['index'], case['fixture'], case['mode'], case['kind'], case['fit']]) ==
                    canonical([index, fixture, mode, kind, FITS[index//4]]), 'Validation order/identity')
            original = read(PRIOR_LOCAL/'fixtures'/f'{fixture:02d}.json')['snapshot']
            require(case['snapshot'] == (original if kind == 'original' else hidden_variant(original, self.catalogue)),
                    'Transferred hidden-state fixture identity')
            expected_context = ({'mode': mode, 'fixture_index': fixture,
                **read(PRIOR_LOCAL/'observations'/f'{fixture:02d}.json')['context']} if mode == 'legacy_transfer'
                else context_for(game, original['state']['turn_index']))
            require(case['observer']['context'] == expected_context, 'Validation public context')
            reference = self.bound(case['reference_receipt'], PRIOR_LOCAL/'attempts'/f'{23+fixture:03d}'/'request-00000001.json')
            name = f'validation/v{index:03d}'; decision_path = self.base/'decision_attempts'/(name+'.json')
            row = read(decision_path)
            require(row['observer'] == case['observer'], 'Validation observer binding')
            outcome = self.decision(row, case['snapshot'], name, 'local', 'validation')
            require(outcome is not None, 'Validation failed; operational admission forbidden')
            wrapper, _ = outcome
            if mode == 'legacy_transfer':
                old = read(reference)['result']
                def projection(value):
                    value = deepcopy(value); value['record'].pop('profile'); return value
                require(projection(wrapper['decision']) == projection(old['decision']) and wrapper['trace'] == old['trace'],
                        'Exact legacy scientific/trace transfer')
            if kind == 'hidden_variant':
                require(cases[index-1]['observer'] == case['observer'] and cases[index-1]['snapshot'] != case['snapshot'] and
                        results[-1] == wrapper, 'Validation exact hidden-state repeat')
            results.append(wrapper); counts.update(self.proof(name, row, wrapper, 'full'))
            saved = read(self.base/'validation/rows'/f'{index:03d}.json')
            require(saved == {'case': index, 'decision': artifact(decision_path),
                'audit': artifact(self.base/'audit_rows'/(name+'.json')), 'passed': True}, 'Validation row artifacts')
        require({p.name for p in (self.base/'validation/rows').glob('*.json')} == {f'{i:03d}.json' for i in range(12)},
                'Validation row inventory')
        expected = {'status': 'passed', 'requests': 12, 'legacy_transfer_requests': 6, 'new_namespace_requests': 6,
            'exact_privacy_pairs': 6, 'all_full_replays': True, 'worlds_replayed': counts['worlds'],
            'continuations_replayed': counts['continuations']}
        require(read(self.base/'adapter_validation.json') == expected, 'Adapter validation summary')
        admission = read(self.base/'games_admission.json')
        require(admission['adapter_validation'] == artifact(self.base/'adapter_validation.json') and
                admission['outcomes_closed'] is True and finite(admission['remaining_seconds']) and
                14400 < admission['remaining_seconds'] <= 43200, 'Games admitted only after validation')
        self.validation_counts = dict(counts)

    def game(self, spec):
        folder = self.base/'sealed/games'/f'{spec["game"]:03d}'
        initial = read(folder/'initial.json')
        snapshot = self.initial_snapshot(spec['seed'], self.catalogue,
            {'ruleset_id': RULESET, 'catalogue_sha256': CATALOGUE})
        require(initial == {**spec, 'initial': snapshot, 'initial_sha256': digest(snapshot)}, 'Independent paired initial world')
        self.ledger.validate_ledger(snapshot)
        game = read(folder/'result.json')
        require(all(game.get(k) == v for k, v in spec.items()) and game['initial_sha256'] == digest(snapshot) and
                1 <= len(game['moves']) <= 400, 'Six-game schedule/initial binding/ply bound')
        require({p.name for p in (folder/'moves').glob('*.json')} == {f'{i:04d}.json' for i in range(len(game['moves']))},
                'Complete actual-move inventory')
        failure = None; receipt_seconds = 0.; init_seconds = 0.; total_seconds = 0.
        for ply, descriptor in enumerate(game['moves']):
            require(not snapshot['state']['terminal'] and failure is None, 'Moves continued after ending')
            path = self.bound(descriptor, folder/'moves'/f'{ply:04d}.json'); row = read(path)
            actor = snapshot['state']['current_player']; player = 'local' if actor == spec['local_seat'] else 'opponent'
            require(canonical([row[k] for k in ('game', 'pair', 'fit', 'ply', 'actor', 'before_sha256')]) ==
                canonical([spec['game'], spec['pair'], spec['fit'], ply, actor, digest(snapshot)]), 'Actual move identity/chain')
            if player == 'local': require(row['observer']['context'] == context_for(spec['game'], ply), 'Actual local fit/stream context')
            name = f'games/g{spec["game"]:03d}/t{ply:04d}'
            outcome = self.decision(row, snapshot, name, player, 'games')
            # Verify observer invariance at every actual edge without executing a policy.
            alternate = hidden_variant(snapshot, self.catalogue)
            cleaned = {**{k: deepcopy(v) for k, v in row['observer'].items() if k != 'context'}, 'schema': self.rules.OBSERVATION_SCHEMA}
            self.ledger.check_observer(alternate, cleaned)
            require(finite(row['total_seconds']) and row['total_seconds'] >= row['initialization']['elapsed_seconds']+
                    row['clock']['elapsed_seconds']-0.1, 'Decision wall-cost accounting')
            total_seconds += row['total_seconds']; init_seconds += row['initialization']['elapsed_seconds']
            receipt_seconds += row['clock']['elapsed_seconds']
            if outcome is None:
                failure = (row['status'], actor, player)
            else:
                wrapper, snapshot = outcome; self.actual_transitions += 1
                if player == 'local':
                    support = len(wrapper['decision']['record']['candidates']['actions'])
                    self.local_recorded_worlds += 128; self.local_recorded_continuations += 128*support
                    self.roots.append({'id': f'g{spec["game"]:03d}-t{ply:04d}', 'game': spec['game'], 'ply': ply,
                        'observer': row['observer'], 'candidate_support': support, 'row': row, 'wrapper': wrapper, 'path': path})
        check_game_ending(game, snapshot, failure)
        self.game_costs.append({'game': spec['game'], 'pair': spec['pair'], 'fit': spec['fit'],
            'attempted_decisions': len(game['moves']), 'decision_supervision_seconds': receipt_seconds,
            'initialization_supervision_seconds': init_seconds, 'decision_attempt_wall_seconds': total_seconds})

    def manifest(self):
        cells, full = stress_selection(self.roots); expected = []
        for root in sorted(self.roots, key=lambda r: (r['game'], r['ply'])):
            row = root['row']; identity = row['observer']['context']['identity']; sample = sample_worlds(identity)
            mode = 'full' if root['id'] in full else 'sample'
            expected.append({'id': root['id'], 'game': root['game'], 'ply': root['ply'], 'fit': row['fit'],
                'identity': identity, 'observer_sha256': digest(row['observer']), 'move': artifact(root['path']),
                'receipt': row['receipt'], 'trace': row['trace'], 'candidate_support': root['candidate_support'],
                'mode': mode, 'world_indices': list(range(128)) if mode == 'full' else sample,
                'sample_world_indices': sample,
                'sample_hashes': [{'world_index': wi, 'sha256': digest([AUDIT_DOMAIN, 'world', AUDIT_SEED, identity, wi])} for wi in sample],
                'stress_priority_sha256': digest([AUDIT_DOMAIN, 'stress', AUDIT_SEED, identity]),
                'stress_cells': root_cells(row['observer'], root['candidate_support'])})
        require(read(self.base/'audit_manifest.json') == {'schema': 'operational_continuation_manifest_v1',
            'entries': expected, 'cells': cells, 'full_stress_ids': full}, 'Independent exact sample/stress manifest')
        lookup = {r['id']: r for r in self.roots}; counts = Counter()
        for entry in expected:
            root = lookup[entry['id']]; counts.update(self.proof(entry['id'], root['row'], root['wrapper'], entry['mode']))
        return {'roots': len(expected), 'sample_only_roots': len(expected)-len(full), 'full_stress_roots': len(full),
            'cells': 81, 'covered_cells': sum(bool(v['eligible']) for v in cells.values()),
            'uncovered_cells': [k for k, v in cells.items() if not v['eligible']],
            'by_fit': {str(fit): sum(r['observer']['context']['fit'] == fit for r in self.roots) for fit in FITS},
            'recorded_worlds': self.local_recorded_worlds, 'recorded_continuations': self.local_recorded_continuations,
            'replayed_union': dict(counts), 'sample_within_full_not_double_counted': True}

    def run(self):
        self.freeze(); self.validation()
        for spec in schedule(): self.game(spec)
        coverage = self.manifest()
        require({p.name for p in (self.base/'sealed/games').iterdir() if p.is_dir()} == {f'{i:03d}' for i in range(6)}, 'Exact game inventory')
        require({p.resolve() for p in (self.base/'decision_attempts').rglob('*.json')} == self.decision_files, 'No extra decision attempts')
        require({p.parent.resolve() for p in (self.base/'workers').rglob('initialization.json')} == self.worker_files and
                {p.parent.resolve() for p in (self.base/'audits').rglob('initialization.json')} == self.audit_files, 'Worker inventory coverage')
        require({p.resolve() for p in (self.base/'audit_rows').rglob('*.json')} ==
                {(self.base/'audit_rows'/(name+'.json')).resolve() for name, _, _ in self.proofs}, 'Audit proof inventory coverage')
        expected = {'validation_passed': True, 'completed_games': 6, 'outcomes_closed': True,
                    'strength_analysis_games': 0, 'larger_data_deferred': True}
        require(all(canonical(self.progress.get(k)) == canonical(v) for k, v in expected.items()), 'Completed operational coverage')
        final = read(self.base/'final_input_check.json')
        require(final['all_hashes_and_sizes_match'] is True and final['files'] == self.sources['files'], 'Final frozen preservation receipt')
        # Check that each authenticated worker/descendant identity has actually exited.
        import psutil
        for pid, created in self.identities:
            try: require(psutil.Process(pid).create_time() != created, 'Authenticated worker still alive')
            except psutil.NoSuchProcess: pass
        self.boundary(); elapsed = time.time()-self.launch['epoch_seconds']
        utc_elapsed = (datetime.fromisoformat(self.progress['time'])-datetime.fromisoformat(self.launch['time'])).total_seconds()
        require(finite(self.progress['elapsed_seconds']) and 0 < self.progress['elapsed_seconds'] <= elapsed and
                abs(self.progress['elapsed_seconds']-utc_elapsed) < 3, 'Original launch elapsed reconciliation')
        sizes = sum(p.stat().st_size for p in self.base.rglob('*') if p.is_file())
        require(sizes < 2*1024**3 and 'torch' not in sys.modules, 'Evidence ceiling/model-free analysis')
        clocks = [v for _, _, v in self.clocks]; telemetry = [r['telemetry'] for r in self.initializations+clocks]
        proof_costs = [read(self.base/'audit_rows'/(name+'.json'))['total_seconds'] for name, _, _ in self.proofs]
        return {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(), 'analyzer': artifact(__file__),
            'analyzer_repair': self.repair,
            'freeze_sha256': file_hash(self.base/'freeze.json'), 'source_preservation': self.sources,
            'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True,
            'operational_schedule': {'games': 6, 'pairs': 3, 'fits': list(FITS), 'order': list(GAME_ORDER),
                'identical_worlds_with_reversed_seats_per_fit': True, 'actual_dictionary_transitions': self.actual_transitions},
            'validation': {'requests': 12, 'legacy_transfer_requests': 6, 'new_context_requests': 6,
                'exact_hidden_state_repeat_pairs': 6, 'full_replay': self.validation_counts},
            'actual_continuation_verification': coverage, 'seed_collision_check': self.seed_scope,
            'per_stage': {key: {'attempted': sum(counter.values()), 'statuses': dict(counter),
                'completed_receipt_seconds': self.descriptive(self.latencies.get(key, []))} for key, counter in sorted(self.statuses.items())},
            'reported_external_work_validated': dict(self.work), 'game_operational_costs': self.game_costs,
            'resources': {'initializations': len(self.initializations), 'initialization_receipt_seconds':
                self.descriptive([r['receipt_elapsed_seconds'] for r in self.initializations]),
                'audit_attempt_wall_seconds': self.descriptive(proof_costs), 'audit_wall_seconds_total': sum(proof_costs),
                'peak_sampled_combined_rss_bytes': max(t['peak_sampled_rss_bytes'] for t in telemetry),
                'sum_last_observed_worker_cpu_seconds': sum(r['telemetry']['cpu_seconds'] for r in clocks),
                'authenticated_exited_process_identities': len(self.identities), 'worker_cleanup_checked': True,
                'production_elapsed_seconds': self.progress['elapsed_seconds'], 'including_final_analysis_seconds': elapsed,
                'evidence_bytes_before_this_receipt': sizes, 'sampled_not_continuous': True},
            'scope': {'sealed_endings_internally_checked_without_outcome_summaries': True,
                'all_actual_actions_and_dictionary_successors_audited': True,
                'all128_worlds_and_recorded_outcome_arithmetic_checked_per_completed_local_decision': True,
                'raw_model_reexecution_performed_by_bound_audit_workers_not_this_analyzer': True,
                'unsampled_continuations_structurally_checked_by_bound_audit_workers_not_replayed': True,
                'external_neural_outputs_and_internal_search_edges_not_reexecuted': True,
                'finite_operational_exercise_not_strength_evidence': True, 'timeout_latencies_not_imputed': True,
                'no_additional_games_or_data_authorized_by_this_analysis': True}}


def hidden_variant(snapshot, catalogue):
    """Independent literal hidden reallocation; no environment/policy execution."""
    alternate = deepcopy(snapshot); state = alternate['state']; tiers = {c['card_id']: str(c['tier']) for c in catalogue['cards']}
    for deck in state['hidden_tier_decks'].values(): deck.reverse()
    opponent = state['players'][1-state['current_player']]
    for index, card in enumerate(opponent['reserved_card_ids']):
        if opponent['reserved_card_sources'][index] == 'deck':
            deck = state['hidden_tier_decks'][tiers[card]]
            if deck: opponent['reserved_card_ids'][index], deck[0] = deck[0], card
    return alternate


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, default=BASE); parser.add_argument('--output', type=Path)
    parser.add_argument('--repair-freeze-sha256', required=True)
    args = parser.parse_args(); destination = args.output or args.base/'independent_analysis_v2.json'
    require(not destination.exists(), 'Never overwrite independent analysis')
    try:
        audit = Audit(args.base, args.repair_freeze_sha256); result = audit.run()
        payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
        audit.boundary()
        require(sum(p.stat().st_size for p in args.base.rglob('*') if p.is_file())+len(payload.encode()) < 2*1024**3, 'Final evidence cap')
        with destination.open('x', encoding='utf-8', newline='\n') as stream: stream.write(payload)
    except BaseException:
        diagnostic = {'time': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(), 'outcomes_closed': True}
        folder = args.base/'sealed'; folder.mkdir(exist_ok=True)
        with (folder/'independent_analysis_failure_02.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(diagnostic, stream, sort_keys=True, indent=2)
        raise
    print(json.dumps({'status': 'passed', 'output': str(destination), 'outcomes_closed': True,
                      'operational_games_checked': 6, 'strength_analysis_games': 0}, sort_keys=True))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        # Dependency exceptions may contain sealed state; do not expose their text or traceback.
        print(json.dumps({'status': 'failed', 'category': type(error).__name__, 'outcomes_closed': True}), file=sys.stderr)
        raise SystemExit(1)

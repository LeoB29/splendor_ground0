"""Independent matched-strength audit; statistics follow the complete fixed-sample verification gate."""
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
BASE = ROOT / 'outputs/matched_strength_20261020_01'
PRIOR_PILOT = ROOT / 'outputs/live_game_operational_20261018_01'
FITS = (161001, 161002, 161003)
GAMES = tuple(range(60))
GAME_ORDER = tuple(g for b in range(0, 60, 6) for g in (b, b+2, b+4, b+1, b+3, b+5))
LIVE_NAMESPACE = 'splendor_matched_strength_20261020_v1'
PRIVATE_NAMESPACE = 'splendor_matched_strength_referee_20261020_v1'
ADAPTER_ID = 'matched_strength_adapter_20261020_v1'
SEARCH_TAG = 'splendor_live_operational_search_v1'
SEED_TAG = 'splendor_live_operational_referee_v1'
AUDIT_DOMAIN = 'splendor_live_operational_world_audit_v1'
AUDIT_SEED = 202610206401
LOCAL_PROFILE = 'four_candidate128_corrected_live_operational_20261019_v1'
LOCAL_SCHEMA = 'corrected_live_planner_observer_v1'
RULESET = 'splendor_base_qualified_rules_20261018_v1'
CATALOGUE = 'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'
OPPONENT_PROFILE = 'outputs/splendorust_120_qualification_20261018_01/selected_profile.json'
OPPONENT_SHA256 = '6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599'
PRIOR_LOCAL = ROOT / 'outputs/corrected_planner_qualification_20261018_01'
TRANSFER_INDICES = (0, 6, 12)
VALIDATION_GAMES = (1, 2, 5)
PILOT_CASES = (2, 6, 10)
PILOT_HASHES = {
    'freeze.json': 'e5c635c1cd5f0e4e9e499a421ab25f8a34b82076707ee946e10d3f0644f904ba',
    'analyzer_repair_02/freeze.json': 'fa27d136c589f54f0e2e993beb3b706e150520e1e744375e81daf9ee705611e6',
    'independent_analysis_v2.json': 'd3259938221b8e5836bd79d1a7bdbcf30762f3bd5cfc57413ac04c914986a0db',
    'finalization.json': '23fa728c97290b3a9e21edd7401c0e539fee75ef36fc3d33d1d07e74807bcee6',
}
METADATA = frozenset(('command', 'identity', 'affinity', 'topology', 'thread_environment'))
THREADS = frozenset(('BLIS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))



def require(value, message):
    # Messages are check labels only: never interpolate sealed values or payloads.
    if not value: raise ValueError('Independent operational audit: ' + message)


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def finite(value): return type(value) in (int, float) and math.isfinite(value)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


CONFIG = {'schema': 'matched_strength_config_v1', 'pair_count': 30,
    'execution_cap_seconds': 259200, 'post_game_reserve_seconds': 64800,
    'evidence_cap_bytes': 8589934592, 'cumulative_memory_wait_seconds': 21600,
    'bootstrap_seed': 202610206701, 'bootstrap_replicates': 50000, 'confidence_level': .95}
PRIOR_COMPLETED = ROOT/'outputs/live_game_operational_recovery_20261020_02'
PRIOR_ACCEPTANCE = ROOT/'outputs/live_game_operational_recovery_20261020_02_validation_02'
PRIOR_FREEZE_SHA = '8540930acd731cb12b2574365e9d919e9d3ee508c56b9aa7e3d61b442a3e65ca'
PRIOR_ACCEPTANCE_SHA = '6a994ab23a4b0817753fbd8fa02b7315313966125a6a42fabb861f8d6c506a48'


def check_config(value):
    require(canonical(value) == canonical(CONFIG), 'Exact authorized sixty-game prospective configuration')
    return deepcopy(value)


def score_row(spec, game):
    # Called only after the full dictionary ending and deadline-forfeit verification.
    winner = game['winner']
    require(winner is None or type(winner) is int and winner in (0, 1), 'Verified winner type')
    return {'game': spec['game'], 'pair': spec['pair'], 'fit': spec['fit'], 'local_seat': spec['local_seat'],
        'local_score_halfpoints': 1 if winner is None else 2 if winner == spec['local_seat'] else 0,
        'termination': 'natural' if game['reason'] == 'natural' else 'forfeit',
        'forfeiting_player': None if game['reason'] == 'natural' else game['failing_player']}


def check_preflight(value):
    require(type(value['returncode']) is int and value['returncode'] == 0, 'Successful frozen preflight')


def local_profile():
    return {'profile_id': LOCAL_PROFILE, 'adapter_id': ADAPTER_ID, 'schema': LOCAL_SCHEMA, 'ruleset_id': RULESET,
        'catalogue_sha256': CATALOGUE, 'fits': list(FITS), 'candidates': 4, 'worlds': 128,
        'backend': 'unchanged_serial', 'parent_profile_id': LOCAL_PROFILE,
        'public_namespace': LIVE_NAMESPACE, 'public_search_tag': SEARCH_TAG,
        'fresh_worker_each_decision': True, 'trace_delivery_inside_move_clock': True,
        'deadline_policy': 'hard120_complete_response_or_forfeit_no_fallback'}


def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def check_pilot_lineage(descriptors, frozen_items):
    """Read only accepted operational metadata; preserve all prior evidence by hash."""
    expected = [artifact(PRIOR_PILOT/name) for name in PILOT_HASHES]
    require(descriptors == expected and all(item['sha256'] == value for item, value in zip(expected, PILOT_HASHES.values(), strict=True)),
            'Accepted pilot lineage identity')
    analysis = read(PRIOR_PILOT/'independent_analysis_v2.json'); final = read(PRIOR_PILOT/'finalization.json')
    require(analysis['status'] == 'passed' and final['status'] == 'accepted' and analysis['outcomes_closed'] is True and
            final['outcomes_closed'] is True and type(analysis['strength_analysis_games']) is int and
            type(final['strength_analysis_games']) is int and analysis['strength_analysis_games'] == final['strength_analysis_games'] == 0 and
            final['accepted_analysis_sha256'] == PILOT_HASHES['independent_analysis_v2.json'] and
            final['original_freeze_sha256'] == PILOT_HASHES['freeze.json'] and
            final['repair_freeze_sha256'] == PILOT_HASHES['analyzer_repair_02/freeze.json'], 'Accepted closed-outcome pilot status')
    for name in ('freeze.json', 'analyzer_repair_02/freeze.json'):
        require(all(frozen_items.get((ROOT/item['path']).resolve()) == item for item in read(PRIOR_PILOT/name)['inputs']),
                'Prior original/repair frozen input preservation')
    repair = read(PRIOR_PILOT/'analyzer_repair_02/freeze.json')
    immutable = [item for item in final['artifacts'] if item['path'] not in ('AGENT_CURRENT.md', 'AGENT_HANDOFF.md')]
    immutable += [repair[key] for key in ('original_analyzer', 'original_completion', 'original_final_input_check',
                                         'original_freeze', 'original_launch')]
    immutable += repair['preserved_failure']
    require(all(frozen_items.get((ROOT/item['path']).resolve()) == item for item in immutable),
            'Prior accepted artifacts and failed-execution preservation')
    require(all(p.resolve() in frozen_items for p in PRIOR_PILOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts),
            'Complete prior pilot evidence preservation')
    return {'accepted_analysis_sha256': PILOT_HASHES['independent_analysis_v2.json'],
            'finalization_sha256': PILOT_HASHES['finalization.json'], 'all_four_lineage_artifacts_authenticated': True,
            'all_original_repair_and_pilot_evidence_preserved': True,
            'accepted_immutable_artifacts_and_prior_failure_preserved': True,
            'historical_mutable_handoff_exemptions': ['AGENT_CURRENT.md', 'AGENT_HANDOFF.md'], 'prior_strength_analysis_games': 0}


def schedule(seed_material):
    require(type(seed_material) is dict and set(seed_material) == {'schema', 'private_nonce'} and
            seed_material['schema'] == 'matched_strength_private_seed_material_v1' and
            type(seed_material['private_nonce']) is str and len(seed_material['private_nonce']) == 64 and
            all(ch in '0123456789abcdef' for ch in seed_material['private_nonce']), 'Private referee material')
    return [{'game': game, 'pair': game//2, 'fit': FITS[(game//2)%3],
             'opponent_seat': (game//2+game%2)%2, 'local_seat': 1-(game//2+game%2)%2,
             'seed': str(int(digest([SEED_TAG, PRIVATE_NAMESPACE, seed_material['private_nonce'], game//2]), 16))} for game in GAME_ORDER]


def context_for(game, turn):
    require(type(game) is int and game in GAMES and type(turn) is int and turn in range(400), 'Operational context range')
    return {'mode': 'operational', 'namespace': LIVE_NAMESPACE, 'pair': game//2, 'game': game,
            'turn': turn, 'fit': FITS[(game//2)%3], 'stream': digest([SEARCH_TAG, LIVE_NAMESPACE, game]),
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


def check_memory_receipt(value, name, launch, config):
    initial = name == 'initial'
    reserve = 600 if name in ('initial', 'analysis') else 4200 if name.startswith('audits/') else config['post_game_reserve_seconds']
    request = 0 if name == 'initial' else 3600 if name == 'analysis' else 600 if name.startswith('audits/') else 120
    cap = config['execution_cap_seconds']; cumulative_limit = config['cumulative_memory_wait_seconds']
    constants = {'schema': 'operational_memory_admission_v2', 'name': name, 'status': 'passed',
        'failure_category': None, 'failure_detail': None, 'initial': initial,
        'maximum_wait_seconds': 7200, 'maximum_cumulative_wait_seconds': cumulative_limit,
        'minimum_available_physical_bytes': 2*1024**3, 'minimum_available_commit_bytes': 4*1024**3,
        'sample_interval_seconds': 10, 'stable_seconds': 120, 'required_consecutive_samples': 13,
        'reserve_seconds': reserve, 'request_seconds': request, 'initialization_seconds': 60,
        'no_worker_during_admission': True, 'outcomes_closed': True}
    require(all(canonical(value.get(k)) == canonical(v) for k, v in constants.items()), 'Fixed memory-wait contract')
    numeric = ('started_epoch_seconds', 'finished_epoch_seconds', 'started_monotonic', 'finished_monotonic',
               'elapsed_seconds', 'charged_seconds', 'cumulative_before_seconds', 'cumulative_after_seconds')
    require(all(finite(value[k]) for k in numeric), 'Finite memory-wait clocks')
    begin, end = value['started_monotonic'], value['finished_monotonic']
    wall_begin, wall_end = value['started_epoch_seconds'], value['finished_epoch_seconds']
    charge = max(end-begin, wall_end-wall_begin, 0.)
    require(value['elapsed_seconds'] == end-begin and value['charged_seconds'] == charge and
            0 <= charge < 7200 and 0 <= value['cumulative_before_seconds'] <= value['cumulative_after_seconds'] < cumulative_limit and
            value['cumulative_after_seconds'] == value['cumulative_before_seconds']+charge and
            0 <= begin-launch['monotonic_start'] <= end-launch['monotonic_start'] < cap and
            0 <= wall_begin-launch['epoch_seconds'] < cap and
            0 <= wall_end-launch['epoch_seconds'] < cap,
            'Unpaused bounded memory-wait clock charges')
    rows = value['readings']; require(type(rows) is list and 1 <= len(rows) <= 721, 'Bounded memory-reading inventory')
    needs = initial; streak = 0; first_good = None; previous = None
    for index, row in enumerate(rows):
        require(set(row) == {'epoch_seconds', 'monotonic_seconds', 'elapsed_seconds', 'available_physical_bytes',
                'available_commit_bytes', 'admissible', 'consecutive_admissible_samples', 'remaining_seconds'},
                'Memory reading schema')
        mono, epoch = row['monotonic_seconds'], row['epoch_seconds']
        physical, commit = row['available_physical_bytes'], row['available_commit_bytes']
        require(all(finite(row[k]) for k in ('epoch_seconds', 'monotonic_seconds', 'elapsed_seconds', 'remaining_seconds')) and
                begin <= mono <= end and 0 <= epoch-launch['epoch_seconds'] < cap and
                row['elapsed_seconds'] == mono-begin and
                (previous is None or mono-previous >= 9.95) and type(physical) is int and type(commit) is int and
                min(physical, commit) >= 0 and reserve+60+request <= row['remaining_seconds'] <= cap,
                'Memory values, spacing and retained work reserve')
        good = physical >= 2*1024**3 and commit >= 4*1024**3
        needs = needs or not good; streak = streak+1 if good else 0
        first_good = (mono if streak == 1 else first_good) if good else None
        require(row['admissible'] is good and type(row['consecutive_admissible_samples']) is int and
                row['consecutive_admissible_samples'] == streak, 'Independent threshold and consecutive-good count')
        eligible = good and (not needs or streak >= 13 and mono-first_good >= 120)
        require(eligible is (index == len(rows)-1), 'First eligible fast/stable memory admission')
        previous = mono
    require(value['stability_required'] is needs and type(value['prior_admission_count']) is int and
            value['prior_admission_count'] >= 0, 'Memory stability and serial-count identity')
    return wall_end


class Audit:
    def __init__(self, base):
        self.base = Path(base).resolve()
        self.worker_files, self.audit_files, self.decision_files = set(), set(), set()
        self.clocks, self.initializations, self.proofs = [], [], []
        self.statuses, self.work = {}, Counter()
        self.latencies = {}; self.roots = []; self.actual_transitions = 0
        self.local_recorded_worlds = self.local_recorded_continuations = 0
        self.identities = set(); self.game_costs = []; self.score_rows = []
        self.memory_paths = set(); self.memory_values = {}
        self.worker_owners = {}

    def boundary(self):
        require(0 <= time.time()-self.launch['epoch_seconds'] < self.config['execution_cap_seconds']-600 and
                0 <= time.monotonic()-self.launch['monotonic_start'] < self.config['execution_cap_seconds']-600,
                'Original dual-clock execution/audit cap with final reserve')

    def bound(self, item, expected=None):
        require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'}, 'Artifact descriptor')
        path = (ROOT/item['path']).resolve()
        require(path.is_relative_to(ROOT) and (expected is None or path == Path(expected).resolve()) and
                artifact(path) == item, 'Artifact hash/size/path binding')
        return path

    def memory_gate(self, descriptor, name):
        path = self.bound(descriptor, self.base/'memory_admissions'/(name+'.json'))
        value = read(path); finished = check_memory_receipt(value, name, self.launch, self.config)
        log = path.with_suffix('.jsonl'); raw = log.read_bytes()
        require(value['readings_log'] == {'path': log.relative_to(self.base).as_posix(),
                'sha256': file_hash(log), 'bytes': len(raw)} and
                [json.loads(line) for line in raw.decode('utf-8').splitlines()] == value['readings'],
                'Immutable complete memory-reading log')
        self.memory_paths.add(path)
        require(name not in self.memory_values or self.memory_values[name] == value, 'Stable memory gate reference')
        self.memory_values[name] = value
        return finished

    def memory_inventory(self):
        import psutil
        passed = self.memory_gate(artifact(self.base/'memory_admissions/analysis.json'), 'analysis')
        require(psutil.Process().create_time() >= passed, 'Analyzer launched after analysis memory admission')
        ordered = sorted(self.memory_values.values(), key=lambda value: value['prior_admission_count'])
        require([v['prior_admission_count'] for v in ordered] == list(range(len(ordered))) and
                ordered[0]['name'] == 'initial' and ordered[-1]['name'] == 'analysis',
                'Exact serial memory gate inventory')
        previous = []
        for value in ordered:
            charge = sum(v['charged_seconds'] for v in sorted(previous, key=lambda row: row['name']+'.json'))
            require(math.isclose(value['cumulative_before_seconds'], charge, rel_tol=0., abs_tol=1e-6),
                    'Independent cumulative memory-wait charge')
            if previous:
                require(value['started_monotonic'] >= previous[-1]['finished_monotonic'],
                        'Nonoverlapping memory admissions')
            previous.append(value)
        require({p.resolve() for p in (self.base/'memory_admissions').rglob('*.json')} == self.memory_paths and
                {p.resolve() for p in (self.base/'memory_admissions').rglob('*.jsonl')} ==
                {p.with_suffix('.jsonl') for p in self.memory_paths}, 'No omitted or extra memory gates/logs')
        self.memory_summary = {'admissions': len(ordered), 'stability_required_admissions':
            sum(v['stability_required'] for v in ordered), 'charged_seconds': sum(v['charged_seconds'] for v in ordered)}

    def accepted_history(self, items):
        require(file_hash(PRIOR_ACCEPTANCE/'finalization_validation.json') == PRIOR_ACCEPTANCE_SHA and
                self.frozen['prior_acceptance'] == artifact(PRIOR_ACCEPTANCE/'finalization_validation.json') and
                not (PRIOR_ACCEPTANCE/'hold.json').exists(), 'Accepted closed operational predecessor')
        acceptance = read(PRIOR_ACCEPTANCE/'finalization_validation.json')
        final_path = self.bound(acceptance['finalization'], PRIOR_ACCEPTANCE/'finalization.json')
        final = read(final_path)
        require(acceptance['status'] == 'passed' and acceptance['outcomes_closed'] is True and
                final['status'] == 'operational_checks_passed_with_preserved_timeouts' and
                final['games'] == 18 and final['preserved_local_timeout_forfeits'] == 4 and
                final['outcomes_closed'] is True and final['strength_analysis_games'] == 0,
                'Prior operational acceptance retains four timing failures without strength opening')
        require(file_hash(PRIOR_COMPLETED/'freeze.json') == PRIOR_FREEZE_SHA, 'Prior completed source freeze')
        original = read(PRIOR_COMPLETED/'freeze.json'); repair = read(PRIOR_ACCEPTANCE/'repair_freeze.json')
        inventory = read(self.bound(repair['source_inventory']))
        required = original['inputs']+repair['inputs']+final['artifacts']+inventory['files']
        require(all(items.get((ROOT/i['path']).resolve()) == i for i in required),
                'Previously accepted sources and artifacts retain their accepted hashes')
        require({p.relative_to(ROOT).as_posix() for p in PRIOR_COMPLETED.rglob('*') if p.is_file()} ==
                {i['path'] for i in inventory['files']}, 'Complete preserved operational source inventory')
        require(all(p.resolve() in items for p in PRIOR_ACCEPTANCE.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts), 'Whole accepted repair evidence preserved')
        return {'accepted': artifact(PRIOR_ACCEPTANCE/'finalization_validation.json'),
                'source_freeze_sha256': PRIOR_FREEZE_SHA, 'old_outcomes_opened': False,
                'previous_operational_games_excluded': 24, 'historical_timeouts_preserved': 4}

    def freeze(self):
        self.completion = read(self.base/'completion.json'); self.progress = read(self.base/'operational_progress.json')
        self.frozen = read(self.base/'freeze.json'); self.launch = read(self.base/'launch.json')
        self.config = check_config(read(self.base/'config.json'))
        require(self.completion['status'] == 'completed_pending_independent_analysis' and
                self.completion['failure'] is None and self.completion['worker_processes_closed'] is True and
                self.completion['progress'] == artifact(self.base/'operational_progress.json') and
                {k: v for k, v in self.completion.items() if k not in ('worker_processes_closed', 'progress')} == self.progress,
                'Completed supervisor and progress binding')
        require(self.launch['freeze'] == artifact(self.base/'freeze.json') and self.launch['maximum_seconds'] == self.config['execution_cap_seconds'] and
                self.launch['torch_loaded_in_supervisor'] is False and self.launch['outcomes_closed'] is True and
                finite(self.launch['epoch_seconds']), 'Original launch/freeze identity')
        self.boundary()
        paths, sizes = set(), 0
        for item in self.frozen['inputs']:
            self.boundary(); path = self.bound(item)
            require(path not in paths, 'Duplicate frozen input'); paths.add(path); sizes += item['bytes']
        constants = {'ruleset_id': RULESET, 'catalogue_sha256': CATALOGUE, 'local_profile_id': LOCAL_PROFILE,
            'games': 60, 'pairs': 30, 'seconds_per_move': 120, 'initialization_seconds': 60, 'audit_seconds': 600,
            'execution_cap_seconds': self.config['execution_cap_seconds'], 'post_game_reserve_seconds': self.config['post_game_reserve_seconds'], 'final_reserve_seconds': 600,
            'analysis_seconds': 3600, 'cumulative_memory_wait_seconds': self.config['cumulative_memory_wait_seconds'],
            'evidence_cap_bytes': self.config['evidence_cap_bytes'], 'minimum_free_bytes': 5*1024**3, 'max_process_tree_rss_bytes': 2*1024**3,
            'maximum_plies': 400, 'validation_requests': 12, 'sample_worlds': 8, 'audit_seed': AUDIT_SEED,
            'audit_domain': AUDIT_DOMAIN, 'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True,
            'schema': 'matched_strength_freeze_v1', 'adapter_id': ADAPTER_ID, 'planned_strength_games': 60,
            'prepared_only': True, 'preparation_launches_execution': False}
        require(all(canonical(self.frozen.get(k)) == canonical(v) for k, v in constants.items()), 'Frozen operational protocol')
        require(self.frozen['config'] == artifact(self.base/'config.json'), 'Frozen prospective configuration')
        self.prior_lineage = self.accepted_history({(ROOT/item['path']).resolve(): item for item in self.frozen['inputs']})
        self.pilot_lineage = check_pilot_lineage(self.frozen['prior_pilot_lineage'],
            {(ROOT/item['path']).resolve(): item for item in self.frozen['inputs']})
        prior = ROOT/'outputs/splendorust_120_qualification_20261018_01'
        require(all((ROOT/i['path']).resolve() in paths for i in read(prior/'freeze.json')['inputs']) and
                all(p.resolve() in paths for p in prior.rglob('*') if p.is_file() and '__pycache__' not in p.parts),
                'Previous qualification preservation')
        require(self.frozen['opponent_profile'] == artifact(ROOT/OPPONENT_PROFILE) and
                file_hash(ROOT/OPPONENT_PROFILE) == OPPONENT_SHA256, 'Qualified fixed opponent profile')
        self.profile = read(ROOT/OPPONENT_PROFILE)
        require(read(self.base/'local_profile.json') == local_profile(), 'Exact transferred local profile')
        check_preflight(read(self.base/'preflight_tests.json'))
        required = [self.base/name for name in ('config.json', 'local_profile.json', 'preflight_tests.json', 'sealed/schedule.json',
                    'sealed/referee_seed_material.json', 'validation_cases.json', 'seed_collision_check.json')]
        required += [Path(__file__), ROOT/'tools/corrected_planner_qualification_20261018/verify.py',
            ROOT/'tools/analyze_external_strength_screen_20261018_02.py',
            ROOT/'tools/external_strength_screen_20261018/rules.py', ROOT/'tools/external_strength_screen_20261018/audit.py']
        required += list((ROOT/'tools/matched_strength_20261020').glob('*.py'))
        required += [ROOT/'docs/engineering/matched_strength_20261020.md',
            ROOT/'docs/engineering/matched_strength_20261020_review.md', ROOT/'docs/engineering/verification_policy.md']
        required += [ROOT/'outputs/matched_strength_20261020_preparation/statistics_runtime.json',
            ROOT/'outputs/matched_strength_20261020_preparation/completion_contract.json']
        require(all(p.resolve() in paths for p in required), 'Analyzer/profile/dependency freeze coverage')
        from tools.matched_strength_20261020.statistics import runtime_contract
        self.statistics_runtime = read(ROOT/'outputs/matched_strength_20261020_preparation/statistics_runtime.json')
        require(canonical(runtime_contract()) == canonical(self.statistics_runtime), 'Frozen numerical runtime identity')
        self.registry = read(ROOT/'outputs/proposal_breadth_arena_128/models_lock.json')
        self.sources = {'files': len(paths), 'bytes': sizes, 'all_hashes_and_sizes_match': True}
        self.schedule = schedule(read(self.base/'sealed/referee_seed_material.json'))
        require(read(self.base/'sealed/schedule.json') == self.schedule, 'Independent private paired schedule')
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
        value = read(self.base/'seed_collision_check.json'); targets = {int(s['seed']) for s in self.schedule}
        require(value['namespace'] == PRIVATE_NAMESPACE and value['distinct_seeds'] == 30 and
                len(targets) == 30 and value['collisions'] == [] and value['skipped_oversize'] == [], 'Seed collision receipt')
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
                require(field == 'checked' and item['bytes'] <= 64*1024**2, 'Every seed inventory file is scanned')
                if field == 'checked': require(not contains(read(path)), 'Historical referee seed collision')
        discovered = {p.resolve() for p in (ROOT/'outputs').rglob('*.json')
            if not p.resolve().is_relative_to(self.base) and
            (p.name in ('manifest.json', 'schedule.json') or 'seed_registry' in p.name)}
        require(discovered == checked, 'Complete historical manifest and seed-registry inventory')
        self.seed_scope = {'checked_files': len(value['checked']), 'oversize_files_not_scanned': len(value['skipped_oversize']),
                           'scope': 'All discovered manifests/schedules/seed registries scanned; not every historical game log'}

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
        module = 'tools.splendorust_120_qualification_20261018.worker' if player == 'opponent' else 'tools.matched_strength_20261020.worker'
        check_initialization(init, row['initialization'], self.frozen, module)
        owner = (init['identity']['pid'], init['identity']['created'])
        require(owner not in self.worker_owners or self.worker_owners[owner] == folder.resolve(),
                'Fresh independent process for every player and audit request')
        self.worker_owners[owner] = folder.resolve()
        gate_name = 'audits/'+name if audit else name
        passed = self.memory_gate(row['memory_admission'], gate_name)
        require(init['identity']['created'] >= passed, 'Worker starts after fresh memory admission')
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
            require(context['mode'] == 'pilot_transfer' and type(context['fixture_index']) is int and
                    context['fixture_index'] in TRANSFER_INDICES, 'Transfer context')
            old = read(PRIOR_PILOT/'decision_attempts/validation'/f'v{PILOT_CASES[TRANSFER_INDICES.index(context["fixture_index"])]:03d}.json')['observer']
            expected = {**old['context'], 'mode': 'pilot_transfer', 'fixture_index': context['fixture_index']}
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
                    row['correct'] is False and 'after_sha256' not in row and
                    (receipt['status'] == 'illegal' or 'result_sha256' not in row),
                    'Unscored hard error or promoted failed decision')
            if receipt['status'] == 'illegal': self.illegal_response(row, snapshot, receipt, player)
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

    def illegal_response(self, row, snapshot, receipt, player):
        """A retained, recognized response proves illegality; unknown failures hold."""
        result, observer = receipt['result'], row['observer']
        require(receipt['received_before_deadline'] is True and row['result_sha256'] == digest(result),
                'Timely illegal response payload binding')
        legal = [self.rules.serialize_action(action) for action in
                 self.rules.legal_actions(self.rules.deserialize_snapshot(snapshot))]
        if player == 'local':
            require(type(result) is dict and set(result) == {'decision', 'trace', 'ruleset_id',
                'catalogue_sha256', 'profile_id', 'observer_sha256'} and result['profile_id'] == LOCAL_PROFILE and
                result['ruleset_id'] == RULESET and result['catalogue_sha256'] == CATALOGUE and
                result['observer_sha256'] == digest(observer), 'Illegal local response identity')
            decision = result['decision']; require(type(decision) is dict and set(decision) == {'action', 'record'},
                                                   'Illegal local decision envelope')
            action, record = decision['action'], decision['record']
            candidates = record['candidates']['actions']; support = len(candidates)
            require(record['profile'] == LOCAL_PROFILE and record['ruleset_id'] == RULESET and
                    record['catalogue_sha256'] == CATALOGUE and record['root_identity'] == observer['context']['identity'] and
                    record['proposal']['fit'] == observer['context']['fit'] and
                    record['proposal']['model_id'] == 'anchor_'+str(observer['context']['fit']) and
                    record['planner'] == 'reference' and 1 <= support <= 4 and len(record['worlds']) == 128 and
                    all(len(world['outcomes']) == support for world in record['worlds']) and
                    type(record['choice']) is int and 0 <= record['choice'] < support and
                    candidates[record['choice']] == action, 'Recognized local work and selected-action coupling')
            trace = self.bound(row['trace'], self.base/'workers'/row['id']/'continuations.jsonl.gz')
            require(result['trace'] == {'sha256': file_hash(trace), 'bytes': trace.stat().st_size,
                                       'records': 128*support}, 'Illegal local complete trace binding')
        else:
            identity = {'player': 'opponent', 'observer_sha256': digest(observer), 'legal_actions_sha256': digest(legal),
                'legal_count': len(legal), 'ruleset': RULESET, 'catalogue_sha256': CATALOGUE, 'profile_sha256': OPPONENT_SHA256}
            require(type(result) is dict and set(result) == set(identity) | {'status', 'action', 'diagnostics'} and
                    result['status'] == 'completed' and type(result['legal_count']) is int and
                    all(result.get(k) == v for k, v in identity.items()) and
                    result['diagnostics']['opponent_id'] == 'splendorust' and
                    result['diagnostics']['profile_id'] == self.profile['profile_id'], 'Illegal opponent response identity')
            self.check_diagnostics(result['diagnostics'], observer, result['action'], legal, self.profile)
            action = result['action']
        require(self.rules.serialize_action(self.rules.deserialize_action(action)) == action and action not in legal,
                'Canonical selected action is independently illegal')

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
            mode = 'pilot_transfer' if index%4 < 2 else 'operational'
            kind = 'original' if index%2 == 0 else 'hidden_variant'
            require(set(case) == {'index', 'fixture', 'mode', 'kind', 'fit', 'snapshot', 'observer', 'reference_receipt'} and
                    canonical([case['index'], case['fixture'], case['mode'], case['kind'], case['fit']]) ==
                    canonical([index, fixture, mode, kind, FITS[index//4]]), 'Validation order/identity')
            original = read(PRIOR_LOCAL/'fixtures'/f'{fixture:02d}.json')['snapshot']
            require(case['snapshot'] == (original if kind == 'original' else hidden_variant(original, self.catalogue)),
                    'Transferred hidden-state fixture identity')
            expected_context = ({**read(PRIOR_PILOT/'decision_attempts/validation'/f'v{PILOT_CASES[index//4]:03d}.json')['observer']['context'],
                'mode': mode, 'fixture_index': fixture} if mode == 'pilot_transfer'
                else context_for(game, original['state']['turn_index']))
            require(case['observer']['context'] == expected_context, 'Validation public context')
            reference = self.bound(case['reference_receipt'], PRIOR_PILOT/'workers/validation'/f'v{PILOT_CASES[index//4]:03d}'/'request-00000001.json')
            name = f'validation/v{index:03d}'; decision_path = self.base/'decision_attempts'/(name+'.json')
            row = read(decision_path)
            require(row['observer'] == case['observer'], 'Validation observer binding')
            outcome = self.decision(row, case['snapshot'], name, 'local', 'validation')
            require(outcome is not None, 'Validation failed; operational admission forbidden')
            wrapper, _ = outcome
            if mode == 'pilot_transfer':
                old = read(reference)['result']
                def projection(value):
                    value = deepcopy(value); value['record'].pop('profile'); return value
                require(projection(wrapper['decision']) == projection(old['decision']) and wrapper['trace'] == old['trace'],
                        'Exact accepted pilot scientific/trace transfer')
            if kind == 'hidden_variant':
                require(cases[index-1]['observer'] == case['observer'] and cases[index-1]['snapshot'] != case['snapshot'] and
                        results[-1] == wrapper, 'Validation exact hidden-state repeat')
            results.append(wrapper); counts.update(self.proof(name, row, wrapper, 'full'))
            saved = read(self.base/'validation/rows'/f'{index:03d}.json')
            require(saved == {'case': index, 'decision': artifact(decision_path),
                'audit': artifact(self.base/'audit_rows'/(name+'.json')), 'passed': True}, 'Validation row artifacts')
        require({p.name for p in (self.base/'validation/rows').glob('*.json')} == {f'{i:03d}.json' for i in range(12)},
                'Validation row inventory')
        expected = {'status': 'passed', 'requests': 12, 'pilot_transfer_requests': 6, 'new_namespace_requests': 6,
            'exact_privacy_pairs': 6, 'all_full_replays': True, 'worlds_replayed': counts['worlds'],
            'continuations_replayed': counts['continuations']}
        require(read(self.base/'adapter_validation.json') == expected, 'Adapter validation summary')
        admission = read(self.base/'games_admission.json')
        require(admission['adapter_validation'] == artifact(self.base/'adapter_validation.json') and
                admission['outcomes_closed'] is True and finite(admission['remaining_seconds']) and
                self.config['post_game_reserve_seconds'] < admission['remaining_seconds'] <= self.config['execution_cap_seconds'], 'Games admitted only after validation')
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
                1 <= len(game['moves']) <= 400, 'Fixed-sample schedule/initial binding/ply bound')
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
        self.score_rows.append(score_row(spec, game))
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
        self.freeze()
        self.memory_gate(artifact(self.base/'memory_admissions/initial.json'), 'initial')
        self.validation()
        for spec in self.schedule: self.game(spec)
        coverage = self.manifest()
        self.memory_inventory()
        require({p.name for p in (self.base/'sealed/games').iterdir() if p.is_dir()} == {f'{i:03d}' for i in GAMES}, 'Exact game inventory')
        require({p.resolve() for p in (self.base/'decision_attempts').rglob('*.json')} == self.decision_files, 'No extra decision attempts')
        require({p.parent.resolve() for p in (self.base/'workers').rglob('initialization.json')} == self.worker_files and
                {p.parent.resolve() for p in (self.base/'audits').rglob('initialization.json')} == self.audit_files, 'Worker inventory coverage')
        require({p.resolve() for p in (self.base/'audit_rows').rglob('*.json')} ==
                {(self.base/'audit_rows'/(name+'.json')).resolve() for name, _, _ in self.proofs}, 'Audit proof inventory coverage')
        expected = {'validation_passed': True, 'completed_games': 60, 'outcomes_closed': True,
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
        require(sizes < self.config['evidence_cap_bytes'] and 'torch' not in sys.modules, 'Evidence ceiling/model-free analysis')
        clocks = [v for _, _, v in self.clocks]; telemetry = [r['telemetry'] for r in self.initializations+clocks]
        proof_costs = [read(self.base/'audit_rows'/(name+'.json'))['total_seconds'] for name, _, _ in self.proofs]
        result = {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(), 'analyzer': artifact(__file__),
            'prior_pilot_lineage': self.pilot_lineage, 'prior_operational_lineage': self.prior_lineage,
            'freeze_sha256': file_hash(self.base/'freeze.json'), 'source_preservation': self.sources,
            'outcomes_closed': False, 'strength_analysis_games': 60, 'larger_data_deferred': True,
            'final_acceptance': False, 'outcomes_opened': True, 'completed_games': 60, 'completed_pairs': 30,
            'memory_waiting': self.memory_summary,
            'operational_schedule': {'games': 60, 'pairs': 30, 'fits': list(FITS), 'order': list(GAME_ORDER),
                'identical_worlds_with_reversed_seats_per_fit': True, 'actual_dictionary_transitions': self.actual_transitions},
            'validation': {'requests': 12, 'pilot_transfer_requests': 6, 'new_context_requests': 6,
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
            'scope': {'all_fixed_sample_endings_checked_before_statistics': True,
                'all_actual_actions_and_dictionary_successors_audited': True,
                'all128_worlds_and_recorded_outcome_arithmetic_checked_per_completed_local_decision': True,
                'raw_model_reexecution_performed_by_bound_audit_workers_not_this_analyzer': True,
                'unsampled_continuations_structurally_checked_by_bound_audit_workers_not_replayed': True,
                'external_neural_outputs_and_internal_search_edges_not_reexecuted': True,
                'fresh_matched_screen_includes_genuine_timeout_losses': True, 'timeout_latencies_not_imputed': True,
                'no_additional_games_or_data_authorized_by_this_analysis': True}}
        # Opening occurs only after every schedule, referee, clock, replay and preservation gate.
        from tools.matched_strength_20261020.statistics import statistics
        result['statistics'] = statistics(self.score_rows, pairs_per_fit=10,
            bootstrap_seed=self.config['bootstrap_seed'], boundary=self.boundary)
        result['statistics_runtime'] = self.statistics_runtime
        # No report is emitted until the scientific inputs are reauthenticated.
        for item in self.frozen['inputs']:
            self.boundary(); self.bound(item)
        from tools.matched_strength_20261020.statistics import runtime_contract
        require(canonical(runtime_contract()) == canonical(self.statistics_runtime),
                'Numerical runtime unchanged after statistics')
        result['source_preservation']['post_analysis_all_hashes_and_sizes_match'] = True
        self.boundary()
        result['resources']['including_final_analysis_seconds'] = time.time()-self.launch['epoch_seconds']
        result['time'] = datetime.now(timezone.utc).isoformat()
        return result


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
    args = parser.parse_args(); destination = args.output or args.base/'independent_analysis.json'
    require(not destination.exists(), 'Never overwrite independent analysis')
    try:
        audit = Audit(args.base); result = audit.run()
        payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
        audit.boundary()
        require(sum(p.stat().st_size for p in args.base.rglob('*') if p.is_file())+len(payload.encode()) < audit.config['evidence_cap_bytes'], 'Final evidence cap')
        with destination.open('x', encoding='utf-8', newline='\n') as stream: stream.write(payload)
    except BaseException:
        diagnostic = {'time': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(), 'outcomes_closed': True}
        folder = args.base/'sealed'; folder.mkdir(exist_ok=True)
        with (folder/'independent_analysis_failure_01.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(diagnostic, stream, sort_keys=True, indent=2)
        raise
    print(json.dumps({'status': 'passed', 'output': str(destination), 'outcomes_closed': False,
                      'games_checked': 60, 'strength_analysis_games': 60}, sort_keys=True))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        # Dependency exceptions may contain sealed state; do not expose their text or traceback.
        print(json.dumps({'status': 'failed', 'category': type(error).__name__, 'outcomes_closed': True}), file=sys.stderr)
        raise SystemExit(1)

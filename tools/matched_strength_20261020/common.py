"""Fresh matched schedule and bounded resources; unchanged scientific contracts.

Derived from the accepted 20261019 common module. Private referee entropy never
enters the public player context or configuration.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import shutil
import time

from tools.local_player_qualification_128.common import (
    ROOT, ResourceHold, now, read, canonical, digest, file_hash, artifact,
    save_new, memory, directory_bytes, execution_lease,
)
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02.common import (
    serialize_action, deserialize_action, legal_digest,
)

OUTPUT = ROOT / 'outputs/matched_strength_20261020_01'
PREPARATION = ROOT / 'outputs/matched_strength_20261020_preparation'
PRIOR_ACCEPTANCE = ROOT / 'outputs/live_game_operational_recovery_20261020_02_validation_02'
PRIOR_COMPLETED = ROOT / 'outputs/live_game_operational_recovery_20261020_02'
PRIOR_COMPLETED_FREEZE_SHA256 = '8540930acd731cb12b2574365e9d919e9d3ee508c56b9aa7e3d61b442a3e65ca'
PRIOR_PILOT = ROOT / 'outputs/live_game_operational_20261018_01'
PRIOR_LOCAL = ROOT / 'outputs/corrected_planner_qualification_20261018_01'
PRIOR_EXTERNAL = ROOT / 'outputs/splendorust_120_qualification_20261018_01'
EXTERNAL_PROFILE_PATH = PRIOR_EXTERNAL / 'selected_profile.json'
EXTERNAL_PROFILE_SHA256 = '6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599'
LOCAL_PROFILE_ID = 'four_candidate128_corrected_live_operational_20261019_v1'
ADAPTER_ID = 'matched_strength_adapter_20261020_v1'
SCHEMA = 'corrected_live_planner_observer_v1'
LIVE_NAMESPACE = 'splendor_matched_strength_20261020_v1'
SEARCH_TAG = 'splendor_live_operational_search_v1'
SEED_TAG = 'splendor_live_operational_referee_v1'
PRIVATE_SEED_NAMESPACE = 'splendor_matched_strength_referee_20261020_v1'
AUDIT_DOMAIN = 'splendor_live_operational_world_audit_v1'
AUDIT_SEED = 202610206401
FITS = (161001, 161002, 161003)
TRANSFER_INDICES = (0, 6, 12)
PILOT_CASE_INDICES = (2, 6, 10)
PILOT_VALIDATION_GAMES = (1, 2, 5)
PILOT_NAMESPACE = 'splendor_corrected_live_operational_20261018_v1'
VALIDATION_GAMES = (1, 2, 5)
RULESET_ID = rules.RULESET_ID
CATALOGUE_SHA256 = rules.CATALOGUE_SHA256
MOVE_SECONDS, INIT_SECONDS, AUDIT_SECONDS = 120, 60, 600
ANALYSIS_SECONDS = 3600
FINAL_RESERVE = 600
MIN_FREE, MAX_WORKER_RSS = 5*1024**3, 2*1024**3
EVIDENCE_RESERVE = 32*1024**2
MAX_PLIES = 400


def require(value, reason):
    if not value:
        raise ValueError(reason)


def validate_config(value):
    fields = {'schema', 'pair_count', 'execution_cap_seconds', 'post_game_reserve_seconds',
              'evidence_cap_bytes', 'bootstrap_seed', 'bootstrap_replicates', 'confidence_level',
              'cumulative_memory_wait_seconds'}
    require(type(value) is dict and set(value) == fields and
            value['schema'] == 'matched_strength_config_v1', 'Prospective configuration schema')
    for name in fields-{'schema', 'confidence_level'}:
        require(type(value[name]) is int and value[name] > 0, 'Positive integer configuration: '+name)
    require(3 <= value['pair_count'] <= 300 and value['pair_count'] % 3 == 0,
            'Complete balanced three-fit pair allocation')
    require(value['execution_cap_seconds'] > value['post_game_reserve_seconds']+2*INIT_SECONDS+MOVE_SECONDS+AUDIT_SECONDS+FINAL_RESERVE,
            'Execution cap must protect validation and replay work')
    require(value['post_game_reserve_seconds'] > INIT_SECONDS+AUDIT_SECONDS+ANALYSIS_SECONDS+FINAL_RESERVE and
            value['cumulative_memory_wait_seconds'] >= 7200 and
            value['evidence_cap_bytes'] >= 1024**3 and value['bootstrap_replicates'] == 50000 and
            type(value['confidence_level']) is float and value['confidence_level'] == .95,
            'Frozen resource/statistical contract')
    return deepcopy(value)


def config():
    return validate_config(read(OUTPUT/'config.json'))


def game_order(pairs=None):
    pairs = config()['pair_count'] if pairs is None else pairs
    require(type(pairs) is int and 3 <= pairs <= 300 and pairs % 3 == 0, 'Balanced pair count')
    return tuple(game for start in range(0, 2*pairs, 6)
                 for game in (start, start+2, start+4, start+1, start+3, start+5))


def verify(items):
    for item in items:
        require(artifact(ROOT/item['path']) == item, 'Changed frozen input: '+item['path'])


def local_seat(game):
    require(type(game) is int and 0 <= game < 2*config()['pair_count'], 'Invalid strength game')
    return 1-((game//2+game%2) % 2)


def make_schedule(seed_material):
    require(type(seed_material) is dict and set(seed_material) == {'schema', 'private_nonce'} and
            seed_material['schema'] == 'matched_strength_private_seed_material_v1' and
            type(seed_material['private_nonce']) is str and len(seed_material['private_nonce']) == 64 and
            all(ch in '0123456789abcdef' for ch in seed_material['private_nonce']), 'Private referee seed material')
    return [{'game': game, 'pair': game//2, 'fit': FITS[(game//2) % 3],
             'opponent_seat': 1-local_seat(game), 'local_seat': local_seat(game),
             'seed': str(int(digest([SEED_TAG, PRIVATE_SEED_NAMESPACE, seed_material['private_nonce'], game//2]), 16))}
            for game in game_order()]


@lru_cache(maxsize=3)
def pilot_observer(index):
    """Read only the accepted pilot's lawful observer, never its hidden fixture."""
    require(type(index) is int and index in TRANSFER_INDICES, 'Transfer fixture outside fixed panel')
    position = TRANSFER_INDICES.index(index)
    case, game = PILOT_CASE_INDICES[position], PILOT_VALIDATION_GAMES[position]
    row = read(PRIOR_PILOT/'decision_attempts'/'validation'/f'v{case:03d}.json')
    require(row['id'] == f'validation/v{case:03d}' and row['status'] == 'completed' and
            row['correct'] is True, 'Accepted pilot decision source required')
    observer = row['observer']
    require(set(observer) == {'schema', 'context', 'ruleset_id', 'catalogue_sha256', 'observation'} and
            observer['schema'] == SCHEMA and observer['ruleset_id'] == RULESET_ID and
            observer['catalogue_sha256'] == CATALOGUE_SHA256, 'Pilot observer source identity')
    turn = observer['observation']['turn_index']
    require(type(turn) is int and 0 <= turn < MAX_PLIES and
            type(observer['observation']['current_player']) is int and
            observer['observation']['current_player'] == 0, 'Pilot fixture public turn/actor')
    expected = {'mode': 'operational', 'namespace': PILOT_NAMESPACE, 'pair': game//2,
                'game': game, 'turn': turn, 'fit': FITS[position],
                'stream': digest([SEARCH_TAG, PILOT_NAMESPACE, game]),
                'identity': [PILOT_NAMESPACE, game, turn]}
    require(canonical(observer['context']) == canonical(expected), 'Pilot source context assignment')
    return observer


def pilot_context(index):
    return {**deepcopy(pilot_observer(index)['context']), 'mode': 'pilot_transfer',
            'fixture_index': index}


def operational_context(game, turn):
    local_seat(game)
    require(type(turn) is int and 0 <= turn < MAX_PLIES, 'Invalid live turn')
    return {'mode': 'operational', 'namespace': LIVE_NAMESPACE, 'pair': game//2,
            'game': game, 'turn': turn, 'fit': FITS[(game//2) % 3],
            'stream': digest([SEARCH_TAG, LIVE_NAMESPACE, game]),
            'identity': [LIVE_NAMESPACE, game, turn]}


def check_context(value):
    require(isinstance(value, dict), 'Context must be an object')
    if value.get('mode') == 'pilot_transfer':
        index = value.get('fixture_index')
        expected = pilot_context(index)
    elif value.get('mode') == 'operational':
        expected = operational_context(value.get('game'), value.get('turn'))
    else:
        raise ValueError('Unsupported live context mode')
    require(canonical(value) == canonical(expected), 'Context provenance or public stream mismatch')
    return deepcopy(expected)


def make_observer_payload(snapshot, context):
    context = check_context(context)
    state = rules.deserialize_snapshot(snapshot)
    view = rules.observation_payload(state)
    require(not state.terminal and state.winner is None and
            context['identity'][2] == state.turn_index, 'Expected matching live turn')
    if context['mode'] == 'pilot_transfer':
        old = pilot_observer(context['fixture_index'])
        require(view['observation'] == old['observation'], 'Transfer fixture lawful view changed')
    else:
        require(state.current_player == local_seat(context['game']), 'Operational local seat mismatch')
    return {**view, 'schema': SCHEMA, 'context': context}


def opponent_observer(observer):
    return {**{k: deepcopy(v) for k, v in observer.items() if k != 'context'},
            'schema': rules.OBSERVATION_SCHEMA}


def sample_worlds(identity):
    return sorted(sorted(range(128), key=lambda wi:
                         (digest([AUDIT_DOMAIN, 'world', AUDIT_SEED, identity, wi]), wi))[:8])


def stress_priority(identity):
    return digest([AUDIT_DOMAIN, 'stress', AUDIT_SEED, identity])


def stress_cells(observer, support):
    context, obs = observer['context'], observer['observation']
    actor = obs['current_player']
    counts = [sum(r['source'] != 'visible' for r in p['reserves']) for p in obs['players']]
    own, other = bool(counts[actor]), bool(counts[1-actor])
    privacy = 'both' if own and other else 'own' if own else 'opponent' if other else 'neither'
    phase = min(2, obs['turn_index']//20)
    cells = [f"ordinary:{context['fit']}:{phase}:{actor}:{privacy}"]
    if support in (1, 2, 3): cells.append(f"support:{context['fit']}:{support}")
    return cells


def expected_stress_cells():
    return ([f'ordinary:{fit}:{phase}:{seat}:{privacy}' for fit in FITS
             for phase in range(3) for seat in range(2)
             for privacy in ('neither', 'own', 'opponent', 'both')]
            + [f'support:{fit}:{support}' for fit in FITS for support in (1, 2, 3)])


def boundary(base, deadline, *, memory_headroom=True):
    cfg = config()
    launch = read(Path(base)/'launch.json')
    require(launch['maximum_seconds'] == cfg['execution_cap_seconds'], 'Frozen execution cap changed')
    receipt = {'time': now(), 'remaining_seconds': min(deadline-time.monotonic(),
               launch['epoch_seconds']+cfg['execution_cap_seconds']-time.time()),
               'free_disk_bytes': shutil.disk_usage(ROOT).free,
               'artifact_bytes': directory_bytes(base), 'memory': memory()}
    if receipt['remaining_seconds'] <= 0: raise ResourceHold('Frozen dual-clock execution ceiling reached')
    if receipt['free_disk_bytes'] < MIN_FREE: raise ResourceHold('Five-GiB disk reserve reached')
    if receipt['artifact_bytes'] >= cfg['evidence_cap_bytes']-EVIDENCE_RESERVE:
        raise ResourceHold('Frozen evidence ceiling approached;32MiB in-flight/final reserve')
    if memory_headroom and receipt['memory']['available_physical'] < 1024**3:
        raise ResourceHold('Insufficient physical memory headroom')
    if memory_headroom and receipt['memory']['available_page'] < 3*1024**3:
        raise ResourceHold('Insufficient commit headroom')
    return receipt


def check_wrapper(value, observer, legal, trace_path):
    require(isinstance(value, dict) and set(value) == {'decision', 'trace', 'ruleset_id',
            'catalogue_sha256', 'profile_id', 'observer_sha256'}, 'Local result schema')
    require(value['ruleset_id'] == RULESET_ID and value['catalogue_sha256'] == CATALOGUE_SHA256 and
            value['profile_id'] == LOCAL_PROFILE_ID and value['observer_sha256'] == digest(observer),
            'Local result identity')
    decision = value['decision']
    require(isinstance(decision, dict) and set(decision) == {'action', 'record'} and decision['action'] in legal,
            'Local action/support mismatch')
    record = decision['record']; support = len(record['candidates']['actions'])
    require(1 <= support <= 4 and len(record['worlds']) == 128 and record['planner'] == 'reference' and
            all(len(w['outcomes']) == support for w in record['worlds']), 'Delivered local work mismatch')
    trace = value['trace']
    require(set(trace) == {'sha256', 'bytes', 'records'} and trace['records'] == 128*support and
            Path(trace_path).stat().st_size == trace['bytes'] and file_hash(trace_path) == trace['sha256'],
            'Local trace binding')
    return True


def check_audit_proof(proof, observer, result, mode):
    require(mode in ('sample', 'full'), 'Unsupported audit mode')
    indices = list(range(128)) if mode == 'full' else sample_worlds(observer['context']['identity'])
    support = len(result['decision']['record']['candidates']['actions'])
    expected = {'status': 'passed', 'profile_id': LOCAL_PROFILE_ID, 'ruleset_id': RULESET_ID,
        'catalogue_sha256': CATALOGUE_SHA256, 'observer_sha256': digest(observer),
        'result_sha256': digest(result), 'decision_sha256': digest(result['decision']),
        'trace_sha256': result['trace']['sha256'], 'trace_bytes': result['trace']['bytes'],
        'root_identity': observer['context']['identity'], 'fit': observer['context']['fit'],
        'mode': mode, 'recorded_worlds': 128, 'recorded_continuations': 128*support,
        'candidate_support': support, 'audited_world_indices': indices, 'audited_worlds': len(indices),
        'audited_continuations': len(indices)*support, 'proposal_model_reexecuted': True,
        'proposal_selection_independent': True, 'all_world_records_checked': True,
        'all_root_checks_passed': True, 'all_continuations_replayed': mode == 'full',
        'old_outcomes_used_as_ground_truth': False}
    require(isinstance(proof, dict) and all(proof.get(k) == v for k, v in expected.items()), 'Audit identity/coverage')
    require(type(proof.get('audited_transitions')) is int and proof['audited_transitions'] >= len(indices)*support and
            type(proof.get('raw035_choices_reexecuted')) is int and
            proof['raw035_choices_reexecuted'] == proof['audited_transitions']-len(indices)*support,
            'Audit transition/policy counts')
    return True

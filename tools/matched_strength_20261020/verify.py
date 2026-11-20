"""All-root validation plus prespecified whole-world or exhaustive replay.

Source: tools/live_game_operational_20261018/verify.py.
Frozen corrected-verifier primitives independently rebuild worlds and proposals,
reexecute policy scores and use the dictionary transition ledger. This version
changes context admission and replay selection only. Unreplayed outcomes receive
structural checks, not claims of trajectory correctness.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from tools.corrected_planner_qualification_20261018 import verify as frozen
from tools.external_strength_screen_20261018 import rules

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = 'four_candidate128_corrected_live_operational_20261019_v1'
SCHEMA = 'corrected_live_planner_observer_v1'
NAMESPACE = 'splendor_matched_strength_20261020_v1'
PILOT_NAMESPACE = 'splendor_corrected_live_operational_20261018_v1'
SEARCH_TAG = 'splendor_live_operational_search_v1'
AUDIT_DOMAIN = 'splendor_live_operational_world_audit_v1'
AUDIT_SEED = 202610206401
FITS = (161001, 161002, 161003)
TRANSFER_INDICES = (0, 6, 12)
PILOT_CASE_INDICES = (2, 6, 10)
PILOT_GAMES = (1, 2, 5)
digest, file_hash, require = frozen.digest, frozen.file_hash, frozen.require
OUTCOME_FIELDS = frozenset(frozen.OUTCOME_FIELDS)
STEP_FIELDS = frozenset(('state_hash', 'action', 'turn', 'actor', 'legal_count',
                         'selected_index', 'forced', 'scores_sha256'))


def check_context(context, observation=None):
    require(type(context) is dict, 'Live context must be an object')
    if context.get('mode') == 'pilot_transfer':
        index = context.get('fixture_index')
        require(type(index) is int and index in TRANSFER_INDICES, 'Unregistered transfer fixture')
        position = TRANSFER_INDICES.index(index)
        case, game = PILOT_CASE_INDICES[position], PILOT_GAMES[position]
        path = ROOT / 'outputs/live_game_operational_20261018_01/decision_attempts/validation' / f'v{case:03d}.json'
        row = json.loads(path.read_text(encoding='utf-8'))
        require(row['id'] == f'validation/v{case:03d}' and row['status'] == 'completed' and
                row['correct'] is True, 'Accepted pilot source decision')
        prior = row['observer']
        require(set(prior) == {'schema', 'context', 'ruleset_id', 'catalogue_sha256', 'observation'} and
                prior['schema'] == SCHEMA and prior['ruleset_id'] == rules.RULESET_ID and
                prior['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'Pilot observer source identity')
        turn = prior['observation']['turn_index']
        require(type(turn) is int and 0 <= turn < 400 and
                type(prior['observation']['current_player']) is int and
                prior['observation']['current_player'] == 1 - ((game//2 + game%2) % 2),
                'Pilot source public turn/seat')
        old_expected = {'mode': 'operational', 'namespace': PILOT_NAMESPACE, 'pair': game//2,
                        'game': game, 'turn': turn, 'fit': FITS[position],
                        'stream': digest([SEARCH_TAG, PILOT_NAMESPACE, game]),
                        'identity': [PILOT_NAMESPACE, game, turn]}
        require(json.dumps(prior['context'], sort_keys=True, allow_nan=False) ==
                json.dumps(old_expected, sort_keys=True, allow_nan=False), 'Pilot source context assignment')
        expected = {**old_expected, 'mode': 'pilot_transfer', 'fixture_index': index}
        if observation is not None:
            require(observation == prior['observation'], 'Pilot lawful observer changed')
    elif context.get('mode') == 'operational':
        game, turn = context.get('game'), context.get('turn')
        cfg = json.loads((ROOT/'outputs/matched_strength_20261020_01/config.json').read_text(encoding='utf-8'))
        pairs = cfg['pair_count']
        require(type(pairs) is int and 3 <= pairs <= 300 and pairs % 3 == 0, 'Balanced frozen pair count')
        require(type(game) is int and 0 <= game < 2*pairs and type(turn) is int and 0 <= turn < 400,
                'Operational game/turn')
        expected = {'mode': 'operational', 'namespace': NAMESPACE, 'pair': game // 2,
                    'game': game, 'turn': turn, 'fit': FITS[(game // 2) % 3],
                    'stream': digest([SEARCH_TAG, NAMESPACE, game]), 'identity': [NAMESPACE, game, turn]}
        if observation is not None:
            require(type(observation['current_player']) is int and observation['current_player'] ==
                    1 - ((game // 2 + game % 2) % 2), 'Operational acting seat')
    else:
        raise ValueError('Unknown live context mode')
    require(json.dumps(context, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), 'Live context schedule/fields')
    if observation is not None:
        require(type(observation['turn_index']) is int and observation['turn_index'] == expected['identity'][2],
                'Observer/context turn')
    return expected


def sample_worlds(identity):
    return sorted(sorted(range(128), key=lambda wi:
                         (digest([AUDIT_DOMAIN, 'world', AUDIT_SEED, identity, wi]), wi))[:8])


def _hash(value):
    return type(value) is str and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


def check_outcome(value, actor, root_turn):
    require(type(value) is dict and set(value) == OUTCOME_FIELDS | {'wdl'}, 'Continuation outcome fields')
    require(value['reason'] in ('completed', 'forced_pass_deadlock') and
            (value['winner'] is None or type(value['winner']) is int and value['winner'] in (0, 1)) and
            type(value['turns']) is int and 1 <= value['turns'] <= 400 - root_turn and
            _hash(value['trace_sha256']) and _hash(value['final_state_sha256']), 'Admissible continuation fields')
    expected_wdl = 0 if value['winner'] is None else 1 if value['winner'] == actor else -1
    require(type(value['wdl']) in (int, float) and value['wdl'] == expected_wdl, 'Winner/WDL consistency')


def check_trace_structure(row, saved, wi, ci, seed, world, forced_action, legal):
    """Check every delivered record without pretending to replay its transitions."""
    require(type(row) is dict and set(row) == OUTCOME_FIELDS | {
        'world_index', 'candidate_index', 'seed', 'initial_state_sha256', 'steps'}, 'Trace row fields')
    require(type(row['world_index']) is int and row['world_index'] == wi and
            type(row['candidate_index']) is int and row['candidate_index'] == ci and
            row['seed'] == seed and row['initial_state_sha256'] == digest(world['state']),
            'Trace world/candidate/seed coupling')
    require(all(row[k] == saved[k] for k in OUTCOME_FIELDS) and
            type(row['turns']) is int and (row['winner'] is None or type(row['winner']) is int),
            'Trace and outcome headers differ')
    steps = row['steps']
    require(type(steps) is list and len(steps) == saved['turns'], 'Trace length differs from recorded work')
    hasher = hashlib.sha256()
    for index, step in enumerate(steps):
        require(type(step) is dict and set(step) == STEP_FIELDS and _hash(step['state_hash']) and
                type(step['turn']) is int and step['turn'] == world['state']['turn_index'] + index and
                type(step['actor']) is int and step['actor'] == (world['state']['current_player'] + index) % 2 and
                type(step['legal_count']) is int and step['legal_count'] >= 1 and
                type(step['selected_index']) is int and 0 <= step['selected_index'] < step['legal_count'] and
                step['forced'] is (index == 0), 'Trace step structure')
        action = step['action']
        require(type(action) is dict and rules.serialize_action(rules.deserialize_action(action)) == action,
                'Noncanonical trace action')
        if index == 0:
            require(step['state_hash'] == digest(world['state']) and step['scores_sha256'] is None and
                    step['legal_count'] == len(legal) and step['action'] == forced_action and
                    step['selected_index'] == legal.index(forced_action), 'Forced root action coupling')
        else:
            require(_hash(step['scores_sha256']), 'Continuation score digest missing')
        hasher.update(digest(action).encode())
    require(hasher.hexdigest() == saved['trace_sha256'], 'Complete recorded action sequence digest')


def audit_decision(observer, result, traces_path, *, mode='sample', boundary=lambda: None):
    boundary()
    require(mode in ('sample', 'full'), 'Unregistered continuation audit mode')
    require(type(observer) is dict and set(observer) ==
            {'schema', 'context', 'ruleset_id', 'catalogue_sha256', 'observation'} and
            observer['schema'] == SCHEMA and observer['ruleset_id'] == rules.RULESET_ID and
            observer['catalogue_sha256'] == rules.CATALOGUE_SHA256 and
            observer['observation']['terminal'] is False and observer['observation']['winner'] is None,
            'Live observer envelope')
    check_context(observer['context'], observer['observation'])
    indices = list(range(128)) if mode == 'full' else sample_worlds(observer['context']['identity'])
    chosen = set(indices)
    path, wrapper = Path(traces_path), result
    require(type(wrapper) is dict and set(wrapper) ==
            {'decision', 'trace', 'ruleset_id', 'catalogue_sha256', 'profile_id', 'observer_sha256'} and
            wrapper['profile_id'] == PROFILE_ID and wrapper['ruleset_id'] == rules.RULESET_ID and
            wrapper['catalogue_sha256'] == rules.CATALOGUE_SHA256 and
            wrapper['observer_sha256'] == digest(observer), 'Live wrapper identity')
    trace = wrapper['trace']
    require(type(trace) is dict and set(trace) == {'bytes', 'sha256', 'records'} and
            type(trace['bytes']) is int and trace['bytes'] >= 0 and type(trace['records']) is int and
            path.stat().st_size == trace['bytes'] and file_hash(path) == trace['sha256'], 'Trace artifact binding')
    decision, record = wrapper['decision'], wrapper['decision']['record']
    require(type(decision) is dict and set(decision) == {'action', 'record'} and type(record) is dict and
            set(record) == {'root_identity', 'planner', 'privacy', 'schedule', 'candidates', 'proposal',
                           'scores', 'worlds', 'profile', 'ruleset_id', 'catalogue_sha256', 'choice', 'means'} and
            record['profile'] == PROFILE_ID and record['ruleset_id'] == rules.RULESET_ID and
            record['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'Decision/profile fields')
    root = frozen.world_from_observer(observer)
    state = rules.deserialize_snapshot(root)
    legal_objects = rules.legal_actions(state)
    legal = [rules.serialize_action(a) for a in legal_objects]
    raw, logits, artifact = frozen.root_scores(state, legal_objects, f"anchor_{observer['context']['fit']}")
    proposed = frozen.derive_proposal(observer, root, legal, raw, logits, artifact)
    require(record['proposal'] == proposed and record['scores'] == raw and
            record['candidates'] == proposed['candidates'] and record['schedule'] == proposed['schedule'] and
            record['root_identity'] == observer['context']['identity'], 'Independent live proposal/schedule reconstruction')
    privacy = {name: sum(r['source'] != 'visible' for r in observer['observation']['players'][seat]['reserves'])
               for name, seat in (('own', state.current_player), ('opponent', 1 - state.current_player))}
    require(record['privacy'] == privacy and record['planner'] == 'reference', 'Planner/privacy role')
    support = len(proposed['candidates']['actions'])
    require(1 <= support <= 4 and type(record['worlds']) is list and len(record['worlds']) == 128 and
            trace['records'] == 128 * support, 'Complete delivered world/candidate budget')
    replayed, transitions, choices, recorded = 0, 0, 0, 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for wi, (seed, saved) in enumerate(zip(proposed['schedule']['world_seeds'], record['worlds'])):
            boundary()
            world = frozen.world_from_observer(observer, seed)
            legal_world = [rules.serialize_action(a) for a in rules.legal_actions(rules.deserialize_snapshot(world))]
            require(type(saved) is dict and set(saved) == {'seed', 'state_sha256', 'outcomes'} and
                    saved['seed'] == seed and saved['state_sha256'] == digest(world['state']) and
                    type(saved['outcomes']) is list and len(saved['outcomes']) == support, 'Every world seed/state coupling')
            for ci, action in enumerate(proposed['candidates']['actions']):
                require(action in legal_world, 'Candidate illegal in sampled hidden world')
                outcome = saved['outcomes'][ci]
                check_outcome(outcome, state.current_player, state.turn_index)
                line = stream.readline(8 * 1024**2)
                require(line and line.endswith('\n'), 'Missing/truncated/oversized trace row')
                row = json.loads(line)
                check_trace_structure(row, outcome, wi, ci, seed, world, action, legal_world)
                recorded += 1
                if wi in chosen:
                    expected, raw_count = frozen.audit_trajectory(world, action, row, boundary)
                    wdl = 0.0 if expected['winner'] is None else 1.0 if expected['winner'] == state.current_player else -1.0
                    require(outcome == {**expected, 'wdl': wdl}, 'Replayed continuation differs')
                    replayed += 1; transitions += expected['turns']; choices += raw_count
        require(stream.read(1) == '', 'Extra continuation traces')
    choice, means = frozen.select_outcomes(record['worlds'], support)
    require(type(record['choice']) is int and record['choice'] == choice and record['means'] == means and
            decision['action'] == proposed['candidates']['actions'][choice], 'Full128 aggregate/first maximum/action')
    boundary()
    return {'status': 'passed', 'profile_id': PROFILE_ID, 'ruleset_id': rules.RULESET_ID,
            'catalogue_sha256': rules.CATALOGUE_SHA256, 'observer_sha256': digest(observer),
            'result_sha256': digest(wrapper), 'decision_sha256': digest(decision),
            'trace_sha256': file_hash(path), 'trace_bytes': path.stat().st_size,
            'root_identity': observer['context']['identity'], 'fit': observer['context']['fit'], 'mode': mode,
            'recorded_worlds': 128, 'recorded_continuations': recorded, 'candidate_support': support,
            'audited_world_indices': indices, 'audited_worlds': len(indices), 'audited_continuations': replayed,
            'audited_transitions': transitions, 'raw035_choices_reexecuted': choices,
            'proposal_model_reexecuted': True, 'proposal_selection_independent': True,
            'all_world_records_checked': True, 'all_root_checks_passed': True,
            'all_continuations_replayed': mode == 'full',
            'transition_oracle': 'frozen_external_screen_dictionary_ledger',
            'shared_dependencies': ['qualified corrected model/feature kernels',
                                    'corrected legal enumeration/catalogue/serialization',
                                    'frozen independent world/proposal/trajectory audit primitives'],
            'old_outcomes_used_as_ground_truth': False}

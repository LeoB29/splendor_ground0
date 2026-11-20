"""Independent ledger replay and fixed-panel statistics for the prepared study.

No models load on import. Neural kernels/checkpoints and corrected legal
enumeration are disclosed shared dependencies; transitions, world reconstruction,
proposal arithmetic and terminal labels are independently checked. PUCT internal
edges are not replayed. The owning worker authenticates sources once before model initialization.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

from tools.corrected_planner_qualification_20261018 import verify as reference
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.future_training_rules_20261021 import (
    decode_training_snapshot, require_official_terminal_target, verify_runtime_sources)
from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics

ROOT = Path(__file__).resolve().parents[2]
from .checks import PROFILES, arm_profile
from . import common as c
NAMESPACE = 'splendor_matched_strength_20261020_v1'
FITS = (161001, 161002, 161003)
PUCT_PROFILE = ROOT/'outputs/external_strength_screen_20261018_02/opponent_profile.json'
PUCT_SHA256 = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
OUTCOME_FIELDS = frozenset(reference.OUTCOME_FIELDS)
STEP_FIELDS = {'state_hash','action','turn','actor','legal_count','selected_index',
               'forced','scores_sha256','policy','policy_trace'}
BOOTSTRAP_SEED = 2026102301
BOOTSTRAP_REPLICATES = 50000
digest, file_hash = reference.digest, reference.file_hash


def require(value, message):
    if not value:
        raise ValueError('Move-evaluation independent audit: '+message)


def _equal(a, b):
    """JSON equality preserves integer/bool and float/integer distinctions."""
    return digest(a) == digest(b)


def select_outcomes64(worlds, support):
    """Independent integer accumulation; no selector reducer is reused."""
    require(type(worlds) is list and len(worlds) == 64 and type(support) is int and 1 <= support <= 4,
            'Complete64-world support')
    totals = [0] * support
    for world in worlds:
        require(len(world['outcomes']) == support, 'Incomplete candidate assessments')
        for i, row in enumerate(world['outcomes']):
            require(type(row['wdl']) in (int, float) and row['wdl'] in (-1, 0, 1), 'WDL domain')
            totals[i] += int(row['wdl'])
    return max(range(support), key=lambda i: totals[i]), [n / 64 for n in totals]


def _raw_scores(state, legal):
    from tools.corrected_planner_qualification_20261018.local import score_actions
    return score_actions(state, legal)


def _root_scores(state, legal, model_id):
    return reference.root_scores(state, legal, model_id)


def _action(value):
    result = value if isinstance(value, dict) else rules.serialize_action(value)
    require(_equal(rules.serialize_action(rules.deserialize_action(result)), result), 'Canonical action schema')
    return result


def _observer(snapshot):
    state = decode_training_snapshot(snapshot)
    value = rules.observation_payload(state)
    ledger.check_observer(snapshot, value)
    return value


def _policy_choice(policy, observer):
    selected, diagnostics = policy.choose(observer) if hasattr(policy, 'choose') else policy(observer)
    return _action(selected), diagnostics


def check_payload(payload):
    require(type(payload) is dict and set(payload) == {'observer','fit','candidates','selection_schedule'},
            'Selector payload allowlist')
    observer = payload['observer']
    require(type(observer) is dict and set(observer) == {'schema','context','ruleset_id','catalogue_sha256','observation'} and
            observer['schema'] == 'corrected_live_planner_observer_v1' and
            observer['ruleset_id'] == rules.RULESET_ID and observer['catalogue_sha256'] == rules.CATALOGUE_SHA256,
            'Corrected observer identity')
    ctx, view = observer['context'], observer['observation']
    game, turn = ctx.get('game'), ctx.get('turn')
    require(type(game) is int and 0 <= game < 60 and type(turn) is int and 0 <= turn < 400, 'Original root identity')
    expected = {'mode':'operational','namespace':NAMESPACE,'pair':game//2,'game':game,'turn':turn,
                'fit':FITS[(game//2)%3], 'stream':digest(['splendor_live_operational_search_v1',NAMESPACE,game]),
                'identity':[NAMESPACE,game,turn]}
    require(_equal(ctx, expected) and type(payload['fit']) is int and payload['fit'] == expected['fit'], 'Original context/fit')
    require(type(view['turn_index']) is int and view['turn_index'] == turn and
            type(view['current_player']) is int and view['current_player'] == 1-((game//2+game%2)%2) and
            view['terminal'] is False and view['winner'] is None, 'Live root/actor')
    world = reference.world_from_observer(observer)
    decode_training_snapshot(world)
    return world


def audit_trajectory(world, forced_action, row, entity_policy=None, *, arm, boundary=lambda:None, scorer=None):
    """Reproduce every B action; step only through the independent dictionary."""
    arm_profile(arm)
    scorer = _raw_scores if scorer is None else scorer
    decode_training_snapshot(world)
    require(row['initial_state_sha256'] == digest(world['state']) and type(row['steps']) is list and row['steps'],
            'Initial world or empty trajectory')
    current, seen, forced, stalled = deepcopy(world), Counter(), 0, 0
    root_actor = world['state']['current_player']
    counts = Counter(raw035_choices_reexecuted=0, entity_choices_reexecuted=0)
    hasher, reason, winner = hashlib.sha256(), None, None
    for index, step in enumerate(row['steps']):
        boundary(); s = current['state']; seen[reference.signature(current)] += 1
        require(not s['terminal'] and seen[reference.signature(current)] < 4 and
                s['turn_index'] < 400 and stalled < 60 and forced < 2, 'Continuation past terminal/cutoff')
        state = decode_training_snapshot(current)
        objects = rules.legal_actions(state); legal = [rules.serialize_action(a) for a in objects]
        expected_fields = STEP_FIELDS if arm == 'B' else STEP_FIELDS-{'policy', 'policy_trace'}
        require(legal and type(step) is dict and set(step) == expected_fields, 'Step fields/legal support')
        require(step['state_hash'] == digest(s) and type(step['turn']) is int and step['turn'] == s['turn_index'] and
                type(step['actor']) is int and step['actor'] == s['current_player'] and
                type(step['legal_count']) is int and step['legal_count'] == len(legal) and
                step['forced'] is (index == 0), 'Step identity/forced position')
        if index == 0:
            action, policy, score_hash, diagnostics = forced_action, 'forced', None, None
        elif arm == 'A' or s['current_player'] == root_actor:
            scores = list(map(float, scorer(state, objects))); reference.finite_scores(scores, len(legal))
            action = legal[max(range(len(legal)), key=lambda i:scores[i])]
            policy, score_hash, diagnostics = 'raw035', digest(scores), None
            counts['raw035_choices_reexecuted'] += 1
        else:
            action, diagnostics = _policy_choice(entity_policy, _observer(current))
            policy, score_hash = 'greedy_entity', None
            counts['entity_choices_reexecuted'] += 1
        require(action in legal and _equal(step['action'], action) and type(step['selected_index']) is int and
                step['selected_index'] == legal.index(action), 'Policy choice/first-tie/legal action')
        require(step['scores_sha256'] == score_hash, 'Raw score identity')
        if arm == 'B':
            require(step['policy'] == policy and _equal(step['policy_trace'], diagnostics),
                    'Policy identity/diagnostic replay')
        _action(step['action']); hasher.update(digest(action).encode())
        successor = ledger.transition(current, action)
        changed = reference.progress(current, successor, action)
        forced = forced+1 if len(legal) == 1 and legal[0]['action_type'] == 'PASS' and action['noble_id'] is None else 0
        stalled = 0 if changed else stalled+1
        current = successor
        if forced >= 2:
            reason, winner = 'forced_pass_deadlock', reference.adjudicate(current)
        elif current['state']['terminal']:
            reason, winner = 'completed', current['state']['winner']
        require(stalled < 60, 'Unresolved no-progress ending')
        if reason is not None:
            require(index == len(row['steps'])-1, 'Extra trace after termination')
    require(reason is not None, 'Incomplete/unresolved trajectory')
    outcome = {'reason':reason,'winner':winner,'turns':len(row['steps']),
               'trace_sha256':hasher.hexdigest(),'final_state_sha256':digest(current['state'])}
    require(_equal({k:row[k] for k in OUTCOME_FIELDS}, outcome), 'Outcome/hash/type mismatch')
    return outcome, dict(counts)


def audit_selection(payload, result, traces_path, entity_policy=None, boundary=lambda:None, *, arm):
    profile = arm_profile(arm)
    boundary(); root = check_payload(payload); observer = payload['observer']
    require(type(result) is dict and set(result) == {'decision','trace','profile_id','observer_sha256','ruleset_id','catalogue_sha256'} and
            result['profile_id'] == profile and result['observer_sha256'] == digest(observer) and
            result['ruleset_id'] == rules.RULESET_ID and result['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'B wrapper identity')
    path = Path(traces_path); trace_info = result['trace']
    require(type(trace_info) is dict and set(trace_info) == {'path','sha256','bytes','records'} and
            (ROOT/trace_info['path']).resolve() == path.resolve() and
            type(trace_info['bytes']) is int and type(trace_info['records']) is int and
            path.stat().st_size == trace_info['bytes'] and file_hash(path) == trace_info['sha256'], 'Compressed trace identity')
    decision = result['decision']; record = decision['record']
    require(set(decision) == {'action','record'} and set(record) == {'root_identity','planner','privacy','schedule',
            'candidates','proposal','scores','worlds','profile','ruleset_id','catalogue_sha256','choice','means'} and
            record['profile'] == profile and record['ruleset_id'] == rules.RULESET_ID and
            record['catalogue_sha256'] == rules.CATALOGUE_SHA256 and record['planner'] == ('reference' if arm == 'A' else 'opponent_policy_substitution'),
            'B record identity')
    state = decode_training_snapshot(root); objects = rules.legal_actions(state)
    legal = [rules.serialize_action(a) for a in objects]
    raw, logits, artifact = _root_scores(state, objects, f"anchor_{payload['fit']}")
    proposed = reference.derive_proposal(observer, root, legal, raw, logits, artifact)
    # Independently derive the fixed prefix rather than using selector helper.
    full = proposed['schedule']
    require(type(full['world_seeds']) is list and len(full['world_seeds']) == 128 and
            len(set(full['world_seeds'])) == 128, 'Independent full128 schedule')
    expected_schedule = {**deepcopy(full), 'world_seeds': full['world_seeds'][:64]}
    prepared = c.read(c.PREP/'selection_lock.json')['roots']
    locked = [r for r in prepared if _equal(r['payload']['observer'], observer)]
    require(len(locked) == 1 and _equal(locked[0]['payload'], payload) and
            locked[0]['original_full_schedule_sha256'] == digest(full), 'Prepared root/full128 digest')
    require(_equal(record['proposal'], proposed) and _equal(record['scores'], raw) and
            _equal(record['candidates'], payload['candidates']) and _equal(payload['candidates'], proposed['candidates']) and
            _equal(record['schedule'], payload['selection_schedule']) and _equal(payload['selection_schedule'], expected_schedule) and
            _equal(record['root_identity'], observer['context']['identity']), 'Independent proposal/lock/schedule reconstruction')
    privacy = {name:sum(r['source'] != 'visible' for r in observer['observation']['players'][seat]['reserves'])
               for name,seat in [('own',state.current_player),('opponent',1-state.current_player)]}
    require(_equal(record['privacy'], privacy), 'Root privacy counts')
    actions = proposed['candidates']['actions']; support = len(actions)
    require(1 <= support <= 4 and type(record['worlds']) is list and len(record['worlds']) == 64 and
            trace_info['records'] == 64*support, 'Full64 work')
    counts = Counter(audited_continuations=0,audited_transitions=0,raw035_choices_reexecuted=0,entity_choices_reexecuted=0)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for wi,(seed,saved) in enumerate(zip(expected_schedule['world_seeds'],record['worlds'], strict=True)):
            boundary(); world = reference.world_from_observer(observer,seed)
            require(set(saved) == {'seed','state_sha256','outcomes'} and saved['seed'] == seed and
                    saved['state_sha256'] == digest(world['state']) and len(saved['outcomes']) == support, 'World seed/state/support')
            for ci,action in enumerate(actions):
                line = stream.readline(8*1024**2)
                require(line and line.endswith('\n'), 'Missing/oversized trace row')
                row = json.loads(line)
                require(set(row) == OUTCOME_FIELDS|{'world_index','candidate_index','seed','initial_state_sha256','steps'} and
                        type(row['world_index']) is int and row['world_index'] == wi and type(row['candidate_index']) is int and
                        row['candidate_index'] == ci and row['seed'] == seed, 'Trace order/identity')
                outcome, choices = audit_trajectory(world, action, row, entity_policy, arm=arm, boundary=boundary)
                wdl = 0. if outcome['winner'] is None else (1. if outcome['winner'] == state.current_player else -1.)
                require(_equal(saved['outcomes'][ci], {**outcome,'wdl':wdl}), 'Saved WDL disagrees with replay')
                counts.update(choices); counts['audited_continuations'] += 1; counts['audited_transitions'] += outcome['turns']
        require(not stream.read(1), 'Extra trace records')
    choice, means = select_outcomes64(record['worlds'], support)
    require(type(record['choice']) is int and record['choice'] == choice and _equal(record['means'],means) and
            _equal(decision['action'], actions[choice]), 'Means/first-max/action')
    boundary()
    require(file_hash(path) == trace_info['sha256'] and path.stat().st_size == trace_info['bytes'], 'Trace changed during replay')
    return {'status':'passed','profile_id':profile,'ruleset_id':rules.RULESET_ID,'catalogue_sha256':rules.CATALOGUE_SHA256,
            'payload_sha256':digest(payload),'observer_sha256':digest(observer),'result_sha256':digest(result),
            'trace_sha256':file_hash(path),'trace_bytes':path.stat().st_size,'audited_worlds':64,'arm':arm,'candidate_support':support,
            **dict(counts),'proposal_model_reexecuted':True,'all_continuations_replayed':True,
            'transition_oracle':'frozen_external_screen_dictionary_ledger',
            'shared_dependencies':['corrected legal enumeration/catalogue/serialization','frozen model/feature kernels',
                                   'qualified Entity policy parity/transport'], 'old_outcomes_used_as_ground_truth':False}



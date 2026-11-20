"""Fresh A64/B64 decisions with unchanged full128 proposals and fixed prefix.

The full proposal remains explicitly full128 provenance. Only the separately
validated record schedule is64; neither historical configuration nor policy is
modified. No checkpoint loads on import.
"""
from copy import deepcopy

from tools.corrected_planner_qualification_20261018 import local as q
from tools.move_evaluation_20261023 import evaluator as original
from tools.external_strength_screen_20261018 import rules
from . import common as c

PROFILES = {'A': 'fixed4_world64_raw035_both_seats_20261023_v1',
            'B': 'fixed4_world64_raw035_own_greedy_entity_opponent_20261023_v1'}
WORLD_COUNT = 64
decode_observer, sample_world, select_raw = original.decode_observer, original.sample_world, original.select_raw
require, digest = c.require, c.digest


def arm_profile(arm):
    require(type(arm) is str and arm in PROFILES, 'Explicit A64/B64 arm required')
    return PROFILES[arm]


def prefix_schedule(full):
    require(type(full) is dict and set(full) == {'belief_sha256', 'candidate_nonce', 'world_seeds'},
            'Full128 schedule schema')
    seeds = full['world_seeds']
    require(type(seeds) is list and len(seeds) == 128 and
            all(type(seed) is str for seed in seeds) and len(set(seeds)) == 128,
            'Original complete128 unique seed schedule required')
    return {**deepcopy(full), 'world_seeds': list(seeds[:WORLD_COUNT])}


def locked_root(payload):
    require(type(payload) is dict and set(payload) == {'observer', 'fit', 'candidates', 'selection_schedule'},
            'Selector accepts lawful allowlisted fields only')
    require(type(payload['fit']) is int and payload['fit'] in (161001, 161002, 161003), 'Assigned fit type')
    # This file contains lawful selector inputs and full-schedule digests only.
    # It contains neither historical outcomes nor assessment-world schedules.
    rows = c.read(c.PREP/'selection_lock.json')['roots']
    matching = [row for row in rows if digest(row['payload']['observer']) == digest(payload['observer'])]
    require(len(matching) == 1, 'Observer must identify exactly one prepared root')
    row = matching[0]
    require(digest(payload) == digest(row['payload']), 'Prepared lawful payload differs')
    return row


def reproduce(payload):
    lock = locked_root(payload)
    state = decode_observer(payload['observer'])
    context = payload['observer']['context']
    require(payload['fit'] == context['fit'], 'Assigned fit mismatch')
    legal = rules.CorrectedEnv().legal_actions(state)
    scores = q.score_actions(state, legal)
    proposal = q.proposal_for_root(state, context, legal, scores)
    require(digest(proposal['candidates']) == digest(payload['candidates']), 'Fixed candidate reproduction differs')
    require(digest(proposal['schedule']) == lock['original_full_schedule_sha256'],
            'Original full128 schedule reproduction differs')
    selected_schedule = prefix_schedule(proposal['schedule'])
    require(digest(selected_schedule) == digest(payload['selection_schedule']), 'Fixed64 prefix schedule differs')
    require(q.score_actions(state, legal) == scores, 'Repeated root raw035 scores differ')
    return state, legal, proposal, scores


def trajectory(state, root_action, opponent, *, arm, boundary=lambda: None, capture=False):
    arm_profile(arm)
    if arm == 'A':
        require(opponent is None, 'A64 must not receive an Entity policy')
        return q.trajectory(state, root_action, boundary=boundary, capture=capture)
    require(opponent is not None, 'B64 needs its fixed greedy Entity policy')
    return original.trajectory(state, root_action, opponent, boundary=boundary, capture=capture)


def select64(worlds, support):
    """Original serial float sums/first maximum with an explicit64 denominator."""
    require(type(support) is int and 1 <= support <= 4 and type(worlds) is list and len(worlds) == 64,
            'Complete64 support required for selection')
    require(all(len(w['outcomes']) == support for w in worlds), 'Incomplete forced-action support')
    require(all(o['reason'] in q.VALID_NATURAL_TERMINATIONS and type(o['wdl']) in (float, int)
                and o['wdl'] in (-1., 0., 1.) for w in worlds for o in w['outcomes']), 'Invalid inner ending')
    means = [sum(w['outcomes'][i]['wdl'] for w in worlds)/64 for i in range(support)]
    return {'choice': max(range(support), key=lambda i: means[i]), 'means': means}


def evaluate_proposal(state, proposal, scores, context, opponent, *, arm,
                      boundary=lambda: None, trace_sink=None):
    """Actual complete64 production body, also exercised by terminal fixtures."""
    profile = arm_profile(arm)
    schedule = prefix_schedule(proposal['schedule'])
    record = {'root_identity': list(context['identity']),
              'planner': 'reference' if arm == 'A' else 'opponent_policy_substitution',
              'privacy': q.arena.baseline.privacy(state), 'schedule': schedule,
              'candidates': proposal['candidates'], 'proposal': proposal, 'scores': scores, 'worlds': [],
              'profile': profile, 'ruleset_id': rules.RULESET_ID,
              'catalogue_sha256': rules.CATALOGUE_SHA256}
    actions = [q.old.decode_action(a) for a in proposal['candidates']['actions']]
    require(1 <= len(actions) <= 4, 'Unexpected fixed candidate support')
    for wi, seed in enumerate(schedule['world_seeds']):
        boundary()
        world = sample_world(state, seed)
        row = {'seed': seed, 'state_sha256': digest(q.serialize_state_snapshot(world)), 'outcomes': []}
        legal = rules.CorrectedEnv().legal_actions(world)
        for ci, action in enumerate(actions):
            require(action in legal, 'Illegal corrected forced proposal')
            value, trace = trajectory(world, action, opponent, arm=arm, boundary=boundary,
                                      capture=trace_sink is not None)
            if trace_sink is not None:
                trace_sink(q._copy({'world_index': wi, 'candidate_index': ci, 'seed': seed,
                    'initial_state_sha256': row['state_sha256'], 'steps': trace, **value}))
            require(value['reason'] in q.VALID_NATURAL_TERMINATIONS, 'Unresolved64 continuation')
            value['wdl'] = q.outcome_value(value['winner'], state.current_player)
            row['outcomes'].append(value)
        record['worlds'].append(row)
    require(len(record['worlds']) == WORLD_COUNT and
            all(len(row['outcomes']) == len(actions) for row in record['worlds']), 'Incomplete64 workload')
    record.update(select64(record['worlds'], len(actions)))
    return q._copy({'action': q.serialize_action(actions[record['choice']]), 'record': record})


def choose(payload, opponent=None, *, arm, boundary=lambda: None, trace_sink=None):
    boundary()
    arm_profile(arm)
    state, _, proposal, scores = reproduce(payload)
    return evaluate_proposal(state, proposal, scores, payload['observer']['context'], opponent,
                             arm=arm, boundary=boundary, trace_sink=trace_sink)

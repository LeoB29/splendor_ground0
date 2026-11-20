"""Versioned live context; unchanged qualified corrected search primitives.

Source: tools/live_game_operational_20261018/local.py.
The explicit choose orchestration below mirrors the accepted pilot adapter.
Only observer/context admission and the profile label change. Models, encoders,
proposal construction, world sampling, trajectories and reductions are reused
without rebinding any historical module globals.
"""
from copy import deepcopy
import random

from tools.corrected_planner_qualification_20261018 import local as qualified
from tools.external_strength_screen_20261018 import rules
from tools.live_game_operational_20261019 import common as c

registered_model = qualified.registered_model
score_actions = qualified.score_actions
trajectory = qualified.trajectory
proposal_for_root = qualified.proposal_for_root
require = qualified.require


def initialize():
    metadata = qualified.initialize()
    return {**metadata, 'source_profile': metadata['profile'], 'profile': c.LOCAL_PROFILE_ID,
            'binding_strategy': 'versioned_live_context_unchanged_corrected_primitives'}


def make_observer_payload(snapshot, context):
    return c.make_observer_payload(snapshot, context)


def state_from_observer_payload(payload):
    require(isinstance(payload, dict) and set(payload) ==
            {'schema', 'context', 'ruleset_id', 'catalogue_sha256', 'observation'} and
            payload['schema'] == c.SCHEMA, 'Invalid live planner envelope')
    context = payload['context']
    c.check_context(context)
    source = {k: deepcopy(v) for k, v in payload.items() if k != 'context'}
    source['schema'] = rules.OBSERVATION_SCHEMA
    state = rules.state_from_observation(source)
    require(not state.terminal and state.winner is None and context['identity'][2] == state.turn_index,
            'Expected a live matching decision')
    if context['mode'] == 'pilot_transfer':
        prior = c.pilot_observer(context['fixture_index'])
        require(payload['observation'] == prior['observation'], 'Pilot transfer observation changed')
    else:
        expected_actor = 1 - ((context['game'] // 2 + context['game'] % 2) % 2)
        require(state.current_player == expected_actor, 'Operational player/seat mismatch')
    qualified.assert_corrected_cards(state)
    return state


def choose(payload, *, boundary=lambda: None, trace_sink=None):
    """Retain every full4x128 operation, first tie and complete trace work."""
    require(qualified._INITIALIZED, 'Live corrected worker not initialized')
    require(trace_sink is None or callable(trace_sink), 'Invalid continuation trace sink')
    boundary()
    state = state_from_observer_payload(payload)
    context = payload['context']
    legal = rules.CorrectedEnv().legal_actions(state)
    proposal_scores = score_actions(state, legal)
    proposal = proposal_for_root(state, context, legal, proposal_scores)
    scores = score_actions(state, legal)
    require(scores == proposal_scores, 'Repeated deterministic root scores changed')
    record = {'root_identity': list(context['identity']), 'planner': 'reference',
              'privacy': qualified.arena.baseline.privacy(state), 'schedule': proposal['schedule'],
              'candidates': proposal['candidates'], 'proposal': proposal, 'scores': scores, 'worlds': [],
              'profile': c.LOCAL_PROFILE_ID, 'ruleset_id': rules.RULESET_ID,
              'catalogue_sha256': rules.CATALOGUE_SHA256}
    actions = [qualified.old.decode_action(a) for a in proposal['candidates']['actions']]
    require(1 <= len(actions) <= 4, 'Unexpected four-candidate support')
    for world_index, seed in enumerate(proposal['schedule']['world_seeds']):
        boundary()
        world = qualified.determinize_for_observer(state, state.current_player, random.Random(seed))
        qualified.assert_corrected_cards(world)
        row = {'seed': seed, 'state_sha256': qualified.old.digest(qualified.serialize_state_snapshot(world)),
               'outcomes': []}
        legal_world = rules.CorrectedEnv().legal_actions(world)
        for candidate_index, action in enumerate(actions):
            require(action in legal_world, 'Illegal corrected forced proposal')
            value, trace = trajectory(world, action, boundary=boundary, capture=trace_sink is not None)
            if trace_sink is not None:
                trace_sink(qualified._copy({'world_index': world_index, 'candidate_index': candidate_index,
                    'seed': seed, 'initial_state_sha256': row['state_sha256'], 'steps': trace, **value}))
            require(value['reason'] in qualified.VALID_NATURAL_TERMINATIONS,
                    'Invalid corrected continuation ending')
            value['wdl'] = qualified.outcome_value(value['winner'], state.current_player)
            row['outcomes'].append(value)
        record['worlds'].append(row)
    require(len(record['worlds']) == 128, 'Incomplete corrected world schedule')
    record.update(qualified.arena.select(record['worlds'], len(actions)))
    return qualified._copy({'action': qualified.serialize_action(actions[record['choice']]), 'record': record})

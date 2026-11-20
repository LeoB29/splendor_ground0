"""Fixed candidates; raw035 own turns and greedy Entity opposing turns."""
import hashlib
import random

from tools.corrected_planner_qualification_20261018 import local as q
from tools.matched_strength_20261020 import local as live
from tools.external_strength_screen_20261018 import rules
from tools.future_training_rules_20261021 import decode_training_snapshot

PROFILE = 'fixed4_raw035_own_greedy_entity_opponent_20261023_v1'
require = q.require
digest = q.old.digest


def decode_observer(observer):
    if observer.get('schema') == rules.OBSERVATION_SCHEMA:
        state = rules.state_from_observation(observer)
    else:
        state = live.state_from_observer_payload(observer)
    q.assert_corrected_cards(state)
    return state


def sample_world(state, seed):
    world = q.determinize_for_observer(state, state.current_player, random.Random(seed))
    q.assert_corrected_cards(world)
    # Explicit strict corrected decoder and independent card/token ledger.
    return decode_training_snapshot(rules.serialize_snapshot(world))


def select_raw(observer):
    state = decode_observer(observer)
    legal = rules.CorrectedEnv().legal_actions(state)
    require(bool(legal), 'No live raw035 action')
    scores = q.score_actions(state, legal)
    selected = max(range(len(legal)), key=lambda i: scores[i])
    return legal[selected], {'policy': 'raw035', 'scores_sha256': digest(scores),
                            'selected_index': selected, 'legal_count': len(legal)}


def reproduce(payload):
    require(type(payload) is dict and set(payload) == {'observer', 'fit', 'candidates', 'selection_schedule'},
            'Selector accepts lawful allowlisted fields only')
    observer = payload['observer']
    state = decode_observer(observer)
    context = observer['context']
    require(payload['fit'] == context['fit'], 'Assigned fit mismatch')
    legal = rules.CorrectedEnv().legal_actions(state)
    scores = q.score_actions(state, legal)
    proposal = q.proposal_for_root(state, context, legal, scores)
    require(proposal['candidates'] == payload['candidates'], 'Fixed candidate reproduction differs')
    require(proposal['schedule'] == payload['selection_schedule'], 'Fixed selection schedule differs')
    scores_again = q.score_actions(state, legal)
    require(scores_again == scores, 'Repeated root raw035 scores differ')
    return state, legal, proposal, scores


def trajectory(state, root_action, opponent, *, boundary=lambda: None, capture=False):
    actor = state.current_player
    env = rules.CorrectedEnv()
    seen, forced, no_progress, turns = {}, 0, 0, 0
    hasher, trace = hashlib.sha256(), []
    reason, winner = 'completed', None
    while not state.terminal:
        boundary()
        signature = q.state_signature(state)
        seen[signature] = seen.get(signature, 0) + 1
        if seen[signature] >= q.old.CONFIG['repetition_limit']:
            reason = 'repetition_cutoff'; break
        if state.turn_index >= q.old.CONFIG['max_turns']:
            reason = 'max_turns'; break
        legal = env.legal_actions(state)
        if not legal:
            reason = 'stalled'; break
        first = turns == 0
        scores, policy_trace = None, None
        if first:
            action, policy = root_action, 'forced'
        elif state.current_player == actor:
            scores = q.score_actions(state, legal)
            action, policy = legal[max(range(len(legal)), key=lambda i: scores[i])], 'raw035'
        else:
            action, policy_trace = opponent.choose(rules.observation_payload(state))
            policy = 'greedy_entity'
        require(action in legal, 'Illegal simulated action')
        selected = legal.index(action)
        payload = q.serialize_action(action)
        hasher.update(digest(payload).encode())
        if capture:
            trace.append({'state_hash': digest(q.serialize_state_snapshot(state)), 'action': payload,
                          'turn': state.turn_index, 'actor': state.current_player,
                          'legal_count': len(legal), 'selected_index': selected, 'forced': first,
                          'scores_sha256': None if scores is None else digest(scores),
                          'policy': policy, 'policy_trace': policy_trace})
        successor = env.step(state, action)
        turns += 1
        forced = forced + 1 if q.is_forced_pass_turn(legal) and action.noble_id is None else 0
        no_progress = 0 if q.is_progress_transition(state, successor, action) else no_progress + 1
        state = successor
        if forced >= 2:
            reason, winner = 'forced_pass_deadlock', q.adjudicate_scores(state); break
        if no_progress >= q.old.CONFIG['no_progress_limit']:
            reason = 'no_progress_cutoff'; break
    if reason == 'completed':
        winner = state.winner
    result = {'reason': reason, 'winner': winner, 'turns': turns, 'trace_sha256': hasher.hexdigest(),
              'final_state_sha256': digest(q.serialize_state_snapshot(state))}
    return result, trace


def choose(payload, opponent, *, boundary=lambda: None, trace_sink=None):
    boundary()
    state, legal, proposal, scores = reproduce(payload)
    context = payload['observer']['context']
    record = {'root_identity': list(context['identity']), 'planner': 'opponent_policy_substitution',
              'privacy': q.arena.baseline.privacy(state), 'schedule': proposal['schedule'],
              'candidates': proposal['candidates'], 'proposal': proposal, 'scores': scores, 'worlds': [],
              'profile': PROFILE, 'ruleset_id': rules.RULESET_ID,
              'catalogue_sha256': rules.CATALOGUE_SHA256}
    actions = [q.old.decode_action(a) for a in proposal['candidates']['actions']]
    require(1 <= len(actions) <= 4 and len(proposal['schedule']['world_seeds']) == 128,
            'Unexpected full fixed workload')
    for wi, seed in enumerate(proposal['schedule']['world_seeds']):
        boundary()
        world = sample_world(state, seed)
        row = {'seed': seed, 'state_sha256': digest(q.serialize_state_snapshot(world)), 'outcomes': []}
        legal_world = rules.CorrectedEnv().legal_actions(world)
        for ci, action in enumerate(actions):
            require(action in legal_world, 'Illegal corrected forced proposal')
            value, trace = trajectory(world, action, opponent, boundary=boundary, capture=trace_sink is not None)
            if trace_sink is not None:
                trace_sink(q._copy({'world_index': wi, 'candidate_index': ci, 'seed': seed,
                    'initial_state_sha256': row['state_sha256'], 'steps': trace, **value}))
            require(value['reason'] in q.VALID_NATURAL_TERMINATIONS, 'Unresolved B selection continuation')
            value['wdl'] = q.outcome_value(value['winner'], state.current_player)
            row['outcomes'].append(value)
        record['worlds'].append(row)
    require(len(record['worlds']) == 128, 'Incomplete corrected world schedule')
    record.update(q.arena.select(record['worlds'], len(actions)))
    return q._copy({'action': q.serialize_action(actions[record['choice']]), 'record': record})

"""Actual new64 arm paths; frozen fixtures remain reusable data and oracles."""
from copy import deepcopy

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.move_evaluation_20261023 import qualification as old
from tools.corrected_planner_qualification_20261018 import local
from . import evaluator

require, digest = old.require, old.digest


def verify_actual_trajectory(opponent, fixtures, *, arm, boundary=lambda: None):
    initial = guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures
                                                  if f['id'] == 'noble_final_round'))
    initial.current_player, initial.turn_index = 1, 1
    initial = guard.decode_training_snapshot(rules.serialize_snapshot(initial))
    forced = next(a for a in rules.legal_actions(initial) if a.action_type.name == 'TAKE_TOKENS')
    result, trace = evaluator.trajectory(initial, forced, opponent, arm=arm,
                                         capture=True, boundary=boundary)
    if arm == 'B':
        require([row['policy'] for row in trace] == ['forced', 'greedy_entity', 'raw035'],
                'Actual B64 trajectory dispatch')
    else:
        require(len(trace) == 3 and all('policy' not in row for row in trace), 'Actual A64 raw trajectory')
    state = initial
    for row in trace:
        boundary()
        require(row['state_hash'] == local.old.digest(local.serialize_state_snapshot(state)), 'Trajectory state hash')
        state = old._step_check(state, rules.deserialize_action(row['action']))
    require(state.terminal and result['reason'] == 'completed' and result['winner'] == state.winner and
            result['turns'] == 3 and result['final_state_sha256'] == local.old.digest(local.serialize_state_snapshot(state)),
            'Actual64 trajectory terminal/scoring')
    require(guard.require_official_terminal_target(rules.serialize_snapshot(state), 'completed') == result['winner'],
            'Actual64 unofficial target')
    return {'status': 'passed', 'arm': arm, 'result': result, 'trace': trace, 'independent_transitions': 3}


def complete64_checks(opponent, fixtures, *, arm, boundary=lambda: None):
    """Actual production evaluator loop on a bounded naturally terminal world."""
    initial = guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures if f['id'] == 'forced_pass'))
    legal = rules.legal_actions(initial)
    require(len(legal) == 1, 'Terminal fixture support')
    rows = []
    for fit in local.FITS:
        boundary()
        context = old._context(initial, fit)
        scores = local.score_actions(initial, legal)
        proposal = local.proposal_for_root(initial, context, legal, scores)
        traces = []
        decision = evaluator.evaluate_proposal(initial, proposal, scores, context, opponent,
            arm=arm, boundary=boundary, trace_sink=traces.append)
        record = decision['record']
        require(len(traces) == len(record['worlds']) == 64 and
                record['schedule']['world_seeds'] == proposal['schedule']['world_seeds'][:64] and
                len(record['proposal']['schedule']['world_seeds']) == 128 and record['choice'] == 0,
                'Actual production64 loop completeness/provenance/choice')
        for index, (trace, world) in enumerate(zip(traces, record['worlds'], strict=True)):
            require(trace['world_index'] == index and trace['candidate_index'] == 0 and
                    trace['seed'] == world['seed'] and trace['reason'] == 'completed' and trace['turns'] == 1,
                    'Actual64 fixture world/candidate ordering')
            sampled = evaluator.sample_world(initial, world['seed'])
            after = old._step_check(sampled, legal[0])
            require(after.terminal and world['outcomes'][0]['winner'] == after.winner,
                    'Actual64 fixture independent transition')
        rejected = []
        for count in (0, 63, 64, 65, 127, 129):
            bad = deepcopy(proposal['schedule']); bad['world_seeds'] = ['fixture-'+str(i) for i in range(count)]
            rejected.append(old._rejected(lambda bad=bad: evaluator.prefix_schedule(bad), 'full_schedule_count_'+str(count)))
        duplicate = deepcopy(proposal['schedule']); duplicate['world_seeds'][-1] = duplicate['world_seeds'][0]
        rejected.append(old._rejected(lambda: evaluator.prefix_schedule(duplicate), 'duplicate_full_seed'))
        rows.append({'fit': fit, 'decision_sha256': digest(decision), 'continuations': len(traces),
                     'independent_transitions': len(traces), 'schedule_rejections': rejected})
    return {'status': 'passed', 'arm': arm, 'fixtures': rows, 'actual_continuations': 192,
            'model_bearing_production_loop': True}


def selector_contract_checks(*, boundary=lambda: None):
    """Exercise the actual locked reproduction path without extra panel rollouts."""
    from . import common as c
    lock = c.read(c.PREP/'selection_lock.json')
    rows = []
    for index in (0, 6, 12):
        boundary()
        payload = lock['roots'][index]['payload']
        _, _, proposal, _ = evaluator.reproduce(payload)
        rejected = []
        for count in (0, 63, 65, 128):
            bad = deepcopy(payload)
            bad['selection_schedule']['world_seeds'] = proposal['schedule']['world_seeds'][:count]
            rejected.append(old._rejected(lambda bad=bad: evaluator.reproduce(bad), 'selector_worlds_'+str(count)))
        bad = deepcopy(payload); bad['assessment_world_seeds_parent_only'] = ['forbidden']
        rejected.append(old._rejected(lambda: evaluator.reproduce(bad), 'selector_parent_secret'))
        for field, value in [('ruleset_id', 'historical'), ('catalogue_sha256', '0'*64)]:
            bad = deepcopy(payload); bad['observer'][field] = value
            rejected.append(old._rejected(lambda bad=bad: evaluator.reproduce(bad), 'selector_'+field))
        rows.append({'root_id': lock['roots'][index]['id'], 'fit': payload['fit'],
                     'full_proposal_sha256': digest(proposal), 'rejected': rejected})
    return {'status': 'passed', 'actual_reproduction_roots': rows, 'extra_panel_rollouts': 0}


def run_entrypoint_checks(opponent, raw_callback=None, *, arm, boundary=lambda: None):
    """Owning worker guards once before models; this function never reguards."""
    evaluator.arm_profile(arm)
    require(local._INITIALIZED, 'Actual proposal/raw models not initialized')
    require((opponent is None) == (arm == 'A'), 'Actual arm/opponent dispatch')
    raw_callback = evaluator.select_raw if raw_callback is None else raw_callback
    fixtures = old.build_fixtures()
    rows, proposals, privacy = [], [], []
    for fixture in fixtures:
        boundary()
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        decoded = evaluator.decode_observer(observer)
        require(rules.observation_payload(decoded) == observer, 'Actual decoder differs')
        seed = 'move-evaluation-entrypoint|' + fixture['id']
        world = evaluator.sample_world(decoded, seed)
        require(rules.observation_payload(world) == observer, 'Actual sampler changed lawful view')
        local.assert_corrected_cards(world)
        encoding = old._encoded_costs(world)
        legal = rules.legal_actions(world)
        handle, seen = old._forward_probe(local._RAW._model, world, legal)
        try:
            action, diag = raw_callback(observer)
        finally:
            handle.remove()
        require(bool(seen), 'Raw production route did not reach checked encoder')
        old._step_check(world, action)
        row = {'id': fixture['id'], 'observer_sha256': digest(observer),
               'sampled_world_sha256': digest(rules.serialize_snapshot(world)), 'encoding': encoding,
               'raw_model_inputs': seen, 'raw_action': rules.serialize_action(action), 'raw': diag}
        if arm == 'B':
            selected, trace = opponent.choose(observer, diagnostic_capture=True)
            old._step_check(world, selected)
            row.update({'greedy_action': rules.serialize_action(selected), 'greedy': trace})
        rows.append(row)
        if 'corrected_face' in fixture['tags']:
            for fit in local.FITS:
                boundary()
                model, artifact = local.registered_model(f'anchor_{fit}')
                legal = rules.legal_actions(decoded)
                handle, checked = old._forward_probe(model, decoded, legal)
                try:
                    proposal = local.proposal_for_root(decoded, old._context(decoded, fit), legal,
                                                       local.score_actions(decoded, legal))
                finally:
                    handle.remove()
                require(bool(checked) and proposal['fit'] == fit, 'Actual proposal encoder route')
                derived = evaluator.prefix_schedule(proposal['schedule'])
                proposals.append({'id': fixture['id'], 'fit': fit, 'model_sha256': artifact['sha256'],
                                  'model_inputs': checked, 'proposal_sha256': digest(proposal),
                                  'derived64_schedule_sha256': digest(derived)})
        if fixture['id'] == 'private_reserves':
            changed = old._private_variant(state); other = rules.observation_payload(changed)
            require(other == observer and rules.serialize_snapshot(changed) != fixture['snapshot'],
                    'Privacy variant is not hidden-only')
            repeated, repeated_diag = raw_callback(other)
            require(repeated == action and repeated_diag == diag, 'Raw hidden-state dependence')
            if arm == 'B':
                repeated, repeated_trace = opponent.choose(other, diagnostic_capture=True)
                require(repeated == selected and repeated_trace == trace, 'Greedy hidden-state dependence')
            alternate = evaluator.sample_world(evaluator.decode_observer(other), seed)
            require(rules.serialize_snapshot(alternate) == rules.serialize_snapshot(world), 'Sampler hidden-state dependence')
            privacy.append({'id': fixture['id'], 'equal': True})
    trajectory = verify_actual_trajectory(opponent, fixtures, arm=arm, boundary=boundary)
    native = old.verify_native_contract(opponent, fixtures, rows, boundary=boundary) if arm == 'B' else None
    auxiliary = old.verify_actual_auxiliary(opponent, fixtures, boundary=boundary) if arm == 'B' else None
    complete = complete64_checks(opponent, fixtures, arm=arm, boundary=boundary)
    selector = selector_contract_checks(boundary=boundary)
    require(len(rows) == 15 and len(proposals) == 18, 'Required actual admission coverage')
    return {'status': 'entrypoint_counterexamples_passed', 'arm': arm,
            'profile_id': evaluator.arm_profile(arm), 'ruleset_id': guard.RULESET_ID,
            'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
            'fixtures': rows, 'proposal_encoder_checks': proposals, 'privacy': privacy,
            'semantic_edges': old._semantic_edges(fixtures), 'rejections': old.identity_rejections(fixtures[0]),
            'native_contract': native, 'actual_auxiliary': auxiliary, 'actual_mixed_trajectory': trajectory,
            'complete64_production_checks': complete, 'training_loader_qualified': False,
            'actual_selector_contract': selector,
            'qualification_scope': 'actual arm-specific bounded fixtures; full panel/timing/replay still required'}

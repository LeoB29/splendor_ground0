"""Independent ledger replay and fixed-panel statistics for the prepared study.

No models load on import. Neural kernels/checkpoints and corrected legal
enumeration are disclosed shared dependencies; transitions, world reconstruction,
proposal arithmetic and terminal labels are independently checked. PUCT internal
edges are not replayed. Call initialize_audit once at worker startup.
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
PROFILE = 'fixed4_raw035_own_greedy_entity_opponent_20261023_v1'
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


def initialize_audit():
    return {'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256,
            'runtime_sources': verify_runtime_sources(), 'models_loaded_by_guard': False}


def _equal(a, b):
    """JSON equality preserves integer/bool and float/integer distinctions."""
    return digest(a) == digest(b)


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


def audit_trajectory(world, forced_action, row, entity_policy, *, boundary=lambda:None, scorer=None):
    """Reproduce every B action; step only through the independent dictionary."""
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
        require(legal and type(step) is dict and set(step) == STEP_FIELDS, 'Step fields/legal support')
        require(step['state_hash'] == digest(s) and type(step['turn']) is int and step['turn'] == s['turn_index'] and
                type(step['actor']) is int and step['actor'] == s['current_player'] and
                type(step['legal_count']) is int and step['legal_count'] == len(legal) and
                step['forced'] is (index == 0), 'Step identity/forced position')
        if index == 0:
            action, policy, score_hash, diagnostics = forced_action, 'forced', None, None
        elif s['current_player'] == root_actor:
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
        require(step['policy'] == policy and step['scores_sha256'] == score_hash and
                _equal(step['policy_trace'], diagnostics), 'Policy identity/score/diagnostic replay')
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


def audit_selection(payload, result, traces_path, entity_policy, boundary=lambda:None):
    boundary(); root = check_payload(payload); observer = payload['observer']
    require(type(result) is dict and set(result) == {'decision','trace','profile_id','observer_sha256','ruleset_id','catalogue_sha256'} and
            result['profile_id'] == PROFILE and result['observer_sha256'] == digest(observer) and
            result['ruleset_id'] == rules.RULESET_ID and result['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'B wrapper identity')
    path = Path(traces_path); trace_info = result['trace']
    require(type(trace_info) is dict and set(trace_info) == {'path','sha256','bytes','records'} and
            (ROOT/trace_info['path']).resolve() == path.resolve() and
            type(trace_info['bytes']) is int and type(trace_info['records']) is int and
            path.stat().st_size == trace_info['bytes'] and file_hash(path) == trace_info['sha256'], 'Compressed trace identity')
    decision = result['decision']; record = decision['record']
    require(set(decision) == {'action','record'} and set(record) == {'root_identity','planner','privacy','schedule',
            'candidates','proposal','scores','worlds','profile','ruleset_id','catalogue_sha256','choice','means'} and
            record['profile'] == PROFILE and record['ruleset_id'] == rules.RULESET_ID and
            record['catalogue_sha256'] == rules.CATALOGUE_SHA256 and record['planner'] == 'opponent_policy_substitution',
            'B record identity')
    state = decode_training_snapshot(root); objects = rules.legal_actions(state)
    legal = [rules.serialize_action(a) for a in objects]
    raw, logits, artifact = _root_scores(state, objects, f"anchor_{payload['fit']}")
    proposed = reference.derive_proposal(observer, root, legal, raw, logits, artifact)
    require(_equal(record['proposal'], proposed) and _equal(record['scores'], raw) and
            _equal(record['candidates'], payload['candidates']) and _equal(payload['candidates'], proposed['candidates']) and
            _equal(record['schedule'], payload['selection_schedule']) and _equal(payload['selection_schedule'], proposed['schedule']) and
            _equal(record['root_identity'], observer['context']['identity']), 'Independent proposal/lock/schedule reconstruction')
    privacy = {name:sum(r['source'] != 'visible' for r in observer['observation']['players'][seat]['reserves'])
               for name,seat in [('own',state.current_player),('opponent',1-state.current_player)]}
    require(_equal(record['privacy'], privacy), 'Root privacy counts')
    actions = proposed['candidates']['actions']; support = len(actions)
    require(1 <= support <= 4 and len(record['worlds']) == 128 and trace_info['records'] == 128*support, 'Full128 work')
    counts = Counter(audited_continuations=0,audited_transitions=0,raw035_choices_reexecuted=0,entity_choices_reexecuted=0)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for wi,(seed,saved) in enumerate(zip(proposed['schedule']['world_seeds'],record['worlds'])):
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
                outcome, choices = audit_trajectory(world, action, row, entity_policy, boundary=boundary)
                wdl = 0. if outcome['winner'] is None else (1. if outcome['winner'] == state.current_player else -1.)
                require(_equal(saved['outcomes'][ci], {**outcome,'wdl':wdl}), 'Saved WDL disagrees with replay')
                counts.update(choices); counts['audited_continuations'] += 1; counts['audited_transitions'] += outcome['turns']
        require(not stream.read(1), 'Extra trace records')
    choice, means = reference.select_outcomes(record['worlds'], support)
    require(type(record['choice']) is int and record['choice'] == choice and _equal(record['means'],means) and
            _equal(decision['action'], actions[choice]), 'Means/first-max/action')
    boundary()
    require(file_hash(path) == trace_info['sha256'] and path.stat().st_size == trace_info['bytes'], 'Trace changed during replay')
    return {'status':'passed','profile_id':PROFILE,'ruleset_id':rules.RULESET_ID,'catalogue_sha256':rules.CATALOGUE_SHA256,
            'payload_sha256':digest(payload),'observer_sha256':digest(observer),'result_sha256':digest(result),
            'trace_sha256':file_hash(path),'trace_bytes':path.stat().st_size,'audited_worlds':128,'candidate_support':support,
            **dict(counts),'proposal_model_reexecuted':True,'all_continuations_replayed':True,
            'transition_oracle':'frozen_external_screen_dictionary_ledger',
            'shared_dependencies':['corrected legal enumeration/catalogue/serialization','frozen model/feature kernels',
                                   'qualified Entity policy parity/transport'], 'old_outcomes_used_as_ground_truth':False}


audit_decision = audit_selection


def audit_assessment_initial(observer, seed, snapshot):
    """Independently regenerate one of the exact 32 fresh parent-only worlds."""
    ctx = observer['context']
    check_payload({'observer':observer,'fit':ctx['fit'],'candidates':{},'selection_schedule':{}})
    root_id = f"games/g{ctx['game']:03d}/t{ctx['turn']:04d}"
    seeds = [int(digest(['move_evaluation_20261023_assessment_v1',root_id,j]),16) for j in range(32)]
    require(type(seed) is int and seed >= 0 and seed in seeds, 'Fresh prescribed integer assessment seed')
    require(str(seed) not in reference.schedule(observer)['world_seeds'], 'Assessment/selection seed overlap')
    expected = reference.world_from_observer(observer,seed)
    require(_equal(snapshot,expected), 'Independent assessment initial world differs')
    return {'status':'passed','root_id':root_id,'world_index':seeds.index(seed),'world_seed':seed,
            'initial_sha256':digest(expected),'observer_sha256':digest(observer)}


def _artifact_json(item):
    require(type(item) is dict and set(item) == {'path','sha256','bytes'} and type(item['bytes']) is int, 'Receipt artifact schema')
    path = (ROOT/item['path']).resolve()
    require(path.is_relative_to(ROOT) and path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], 'Receipt artifact bytes')
    return json.loads(path.read_text(encoding='utf-8'))


def audit_branch(branch, boundary=lambda:None, *, scorer=None):
    """Official terminal assessment: every actual edge, no internal PUCT replay."""
    scorer = _raw_scores if scorer is None else scorer
    require(type(branch) is dict and set(branch) == {'initial','forced_action','root_actor','steps','final','score','reason'}, 'Branch fields')
    current = deepcopy(branch['initial']); decode_training_snapshot(current)
    require(type(branch['root_actor']) is int and branch['root_actor'] == current['state']['current_player'] and
            type(branch['steps']) is list and branch['steps'], 'Branch actor/steps')
    require(file_hash(PUCT_PROFILE) == PUCT_SHA256, 'Qualified PUCT16 profile identity')
    profile = json.loads(PUCT_PROFILE.read_text(encoding='utf-8'))
    clocks, raw_count, opponent_count, work = [], 0, 0, Counter()
    for index,step in enumerate(branch['steps']):
        boundary(); state = decode_training_snapshot(current)
        require(not state.terminal and state.turn_index < 400, 'Assessment continued after terminal/turn ceiling')
        require(set(step) == {'before','observer','action','after','player','clock','receipt'} and
                _equal(step['before'],current), 'Assessment state chain/step fields')
        ledger.check_observer(current,step['observer'])
        objects = rules.legal_actions(state); legal = [rules.serialize_action(a) for a in objects]
        action = _action(step['action']); require(action in legal, 'Illegal assessment action')
        player = 'forced' if index == 0 else 'raw035' if state.current_player == branch['root_actor'] else 'opponent'
        require(step['player'] == player, 'Assessment actor policy')
        if index == 0:
            require(_equal(action,branch['forced_action']) and step['clock'] is None and step['receipt'] is None, 'Forced branch action')
        else:
            receipt = step['clock']; require(_equal(_artifact_json(step['receipt']),receipt), 'Clock receipt copy')
            request_id = receipt['request_id']
            require(type(request_id) is str and request_id.startswith('request-'), 'Assessment request identity')
            check_clock(receipt,request_id,1,0)
            require(receipt['status'] == 'ok', 'Incomplete assessment decision')
            result = receipt['result']
            require(set(result) == {'action','diagnostics','profile_sha256','player','observer_sha256','legal_actions_sha256',
                                   'legal_count','ruleset','catalogue_sha256'} and
                    _equal(result['action'],action) and result['player'] == player and
                    result['observer_sha256'] == digest(step['observer']) and
                    result['legal_actions_sha256'] == digest(legal) and type(result['legal_count']) is int and
                    result['legal_count'] == len(legal) and result['ruleset'] == rules.RULESET_ID and
                    result['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'Assessment receipt identity/action')
            clocks.append(receipt['receipt_elapsed_seconds'])
            if player == 'raw035':
                scores = list(map(float,scorer(state,objects))); reference.finite_scores(scores,len(legal))
                selected = max(range(len(legal)),key=lambda i:scores[i])
                require(action == legal[selected] and result['profile_sha256'] is None and
                        _equal(result['diagnostics'], {'policy':'raw035','scores_sha256':digest(scores),
                               'selected_index':selected,'legal_count':len(legal)}), 'Assessment raw035 first argmax/score digest')
                raw_count += 1
            else:
                require(result['profile_sha256'] == PUCT_SHA256, 'Assessment PUCT16 profile')
                work.update(check_diagnostics(result['diagnostics'],step['observer'],action,legal,profile))
                opponent_count += 1
        successor = ledger.transition(current,action)
        require(_equal(successor,step['after']), 'Independent assessment transition')
        current = successor
    require(_equal(current,branch['final']), 'Assessment final snapshot')
    winner = require_official_terminal_target(current,branch['reason'])
    score = .5 if winner is None else float(winner == branch['root_actor'])
    require(type(branch['score']) in (int,float) and branch['score'] == score, 'Official assessment score')
    boundary()
    return {'status':'passed','branch_sha256':digest(branch),'initial_sha256':digest(branch['initial']),
            'final_sha256':digest(current),'forced_action_sha256':digest(branch['forced_action']),
            'score_halfpoints':int(2*score),'audited_transitions':len(branch['steps']),
            'raw035_choices_reexecuted':raw_count,'puct16_actual_edges_checked':opponent_count,
            'puct16_internal_edges_reexecuted':False,'maximum_move_receipt_seconds':max(clocks,default=0.),
            'work':dict(work),'official_terminal':True}


def intervention_breadth(rows, candidate_lock):
    roots = candidate_lock['roots'] if isinstance(candidate_lock,dict) else candidate_lock
    require(type(roots) is list and len(roots) == 18 and type(rows) is list and len(rows) == 18, 'Fixed eighteen-root panel')
    require(len({r['id'] for r in roots}) == 18 and Counter(r['fit'] for r in roots) == Counter({f:6 for f in FITS}), 'Panel identity/fit coverage')
    changed = Counter({f:0 for f in FITS})
    for row,root in zip(rows,roots):
        require(row['id'] == root['id'] and type(row['fit']) is int and row['fit'] == root['fit'] and
                type(row['control_choice']) is int and row['control_choice'] == root['control_choice'] and
                type(row['alternative_choice']) is int and
                0 <= row['alternative_choice'] < len(root['candidates']['actions']), 'Ordered locked root/choice')
        changed[row['fit']] += row['alternative_choice'] != row['control_choice']
    total = sum(changed.values())
    return {'changed_roots':total,'changed_roots_by_fit':{str(f):changed[f] for f in FITS},
            'passes':total >= 6 and all(changed[f] >= 1 for f in FITS)}


def bootstrap_totals(differences, boundary=lambda:None):
    """Exact replicate/root/world RNG order, including every structural zero."""
    import numpy as np
    values = np.asarray(differences,dtype=np.int64)
    require(values.shape == (18,32), 'Bootstrap fixed root/world shape')
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    totals = np.empty(BOOTSTRAP_REPLICATES,dtype=np.int64)
    # Chunking changes neither C-order draws nor integer totals; it bounds memory.
    for start in range(0,BOOTSTRAP_REPLICATES,100):
        boundary(); n = min(100,BOOTSTRAP_REPLICATES-start)
        indices = rng.integers(0,32,size=(n,18,32),dtype=np.int64)
        totals[start:start+n] = np.take_along_axis(values[None,:,:],indices,axis=2).sum(axis=(1,2),dtype=np.int64)
    return totals


def statistics(rows, candidate_lock, *, checks_complete, boundary=lambda:None):
    import numpy as np
    require(checks_complete is True, 'Every prescribed qualification/assessment check must pass before analysis')
    breadth = intervention_breadth(rows,candidate_lock)
    require(breadth['passes'], 'Insufficient intervention breadth; do not assess or impute outcomes')
    fields = {'id','fit','control_choice','alternative_choice','differences_halfpoints','a_scores_halfpoints','b_scores_halfpoints'}
    differences, totals_by_fit, arm_totals, structural = [], Counter(), Counter(), 0
    for row in rows:
        require(set(row) == fields, 'Exact assessment row schema')
        ds = row['differences_halfpoints']; a = row['a_scores_halfpoints']; b = row['b_scores_halfpoints']
        require(type(ds) is list and len(ds) == 32 and all(type(d) is int and -2 <= d <= 2 for d in ds), 'Complete paired difference integers')
        if row['control_choice'] == row['alternative_choice']:
            require(a is None and b is None and ds == [0]*32, 'Structural zero without redundant/imputed outcomes')
            structural += 1
        else:
            require(all(type(v) is list and len(v) == 32 and all(type(x) is int and x in (0,1,2) for x in v) for v in (a,b)), 'Complete natural terminal scores')
            require(ds == [y-x for x,y in zip(a,b)], 'Paired score arithmetic')
            arm_totals['a'] += sum(a); arm_totals['b'] += sum(b)
        differences.append(ds); totals_by_fit[row['fit']] += sum(ds)
    totals = bootstrap_totals(differences,boundary)
    numerator = sum(map(sum,differences)); mean = numerator/1152
    ci = np.percentile(totals.astype(np.float64)/1152,[2.5,97.5],method='linear').tolist()
    by_fit = {str(f):totals_by_fit[f]/384 for f in FITS}
    saturated = arm_totals['a'] == arm_totals['b'] == 0 and ci == [0.,0.]
    promising = mean >= .02 and ci[0] > 0 and all(v >= 0 for v in by_fit.values())
    boundary()
    return {'schema':'move_evaluation_conditional_statistics_v1','status':'complete_verified_fixed_panel',
            'roots':18,'worlds_per_root':32,'source_pairs':16,'equal_root_weights':True,'equal_fit_weights':True,
            'structural_zero_roots':structural,'terminal_branches':2*32*(18-structural),'intervention_breadth':breadth,
            'integer_difference_numerator':numerator,'denominator':1152,'mean_gain':mean,'mean_gain_percentage_points':100*mean,
            'by_fit_mean_gain':by_fit,'conditional_ci95':ci,'bootstrap_replicates':BOOTSTRAP_REPLICATES,
            'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_rng':'NumPy PCG64','bootstrap_draw_order':'replicate,locked_root,32_int64_world_indices',
            'bootstrap_quantile':'linear','bootstrap_integer_totals_sha256':hashlib.sha256(totals.astype('<i8').tobytes(order='C')).hexdigest(),
            'verified_rows_sha256':digest(rows),'all_checks_complete':True,'promising_on_development_panel':promising,
            'saturated_endpoint':saturated,'classification':'saturated_uninformative' if saturated else
                'promising_on_development_panel' if promising else 'did_not_meet_development_investment_rule',
            'saturation_scope':'all observed changed-root branches lose; unchanged-root contrasts are structural zero',
            'scope':'conditional Monte Carlo uncertainty for these eighteen known roots and fixed reference policies',
            'shared_entity_checkpoint':True,'population_strength_claim':False,'fresh_game_launch_authorized':False,
            'larger_data_deferred':True}

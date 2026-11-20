"""Independent saved-evidence qualification for the 13 November proxy adapter.

Numerical and completed-turn checks are copied explicitly from the accepted
27 October v2 checker, with a new bound through2048 and new identity/seed/clock
allowlists. No historical checker globals or source are modified.
"""
from __future__ import annotations
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.splendorust_search_scale_20261024 import qualification as old
from tools.splendorust_search_scale_20261024 import audit as original
from tools.splendorust_search_scale_20261024 import audit as original_audit
from tools.splendorust_search_scale_20261024 import common as original_profiles
from tools.splendorust_decision_quality_20261027_v2 import qualification as old_qualification
from tools.splendorust_decision_quality_20261027_v2.audit import (
    require, same, finite, integer, sha, f32, validate_callback_capture,
    initial_network_win_credit, check_clock, BASE_PHASE, TRACE_FIELDS, TELEMETRY_FIELDS)
from . import adapters
ROOT=Path(__file__).resolve().parents[2]
QUOTAS=adapters.ALL_QUOTAS
RESEARCH_SCHEMA=adapters.SCHEMA
scientific_core=old_qualification.scientific_core


def frozen_projection(result):
    """Disclosed checker view only; raw sampling seeds are validated separately.

    The historical structural checker insists on its canonical seed annotation.
    Replace that annotation and profile labels only, never actions, callback
    bytes, work counts or numerical values. This is not trajectory equivalence.
    """
    quota=result['simulations_quota']
    require(quota in (16,64), 'projection restricted to corrected-entrypoint suite')
    projected={key:deepcopy(result[key]) for key in original.RESULT_FIELDS}
    projected.update(profile_id=original_profiles.profile_id(quota),
        profile_sha256=original_profiles.profile_sha256(quota),wrapper_id=original_profiles.wrapper_id(quota))
    projected['diagnostics']['profile_id']=original_profiles.profile_id(quota)
    projected['diagnostics']['observer_seed']=int(result['observer_sha256'][:16],16)
    return projected


def build_plan(observer, root_id='fixed_qualification_root'):
    state=rules.state_from_observation(observer)
    require(not state.terminal and state.turn_index<124 and len(bridge.main_actions(observer))>1,
            'qualification root must actually search')
    specs=[]
    def add(player,quota,pool,mode,world,encoding,telemetry):
        specs.append(dict(index=len(specs),player=player,quota=quota,pool=pool,mode=mode,
            world_seed_label=world,encoding_seed_label=encoding,telemetry=telemetry,
            root_id=root_id,observer=deepcopy(observer),observer_sha256=digest(observer)))
    add('control',16,3,'legacy','A','A',True)
    for telemetry in (False,True):add('reference',16,3,'legacy','A','A',telemetry)
    for quota in (512,2048):
        add('control',quota,3,'legacy','A','A',True)
        for pool in (3,6):
            for telemetry in (False,True):add('search',quota,pool,'legacy','A','A',telemetry)
            for world in ('A','B'):
                for encoding in ('A','B'):
                    for telemetry in (False,True):add('search',quota,pool,'split',world,encoding,telemetry)
    routes=[dict(mode=mode,pool=pool) for mode in ('legacy','split') for pool in (3,6)]
    return dict(schema='teacher_variability_qualification_v1',original_suite=old.build_plan(64),
        entrypoint_routes=routes,parity_requests=specs,complete_decisions=52*len(routes)+len(specs),graph_cases=11,
        complete_response_seconds=adapters.COMPLETE_SECONDS,internal_seconds=adapters.INTERNAL_SECONDS,
        no_retries=True,original_binary_controls=[16,512,2048],telemetry_on_off_pairs=21,
        exact_scientific_and_float32_parity=True,split_crossed_world_input_checks=True,
        runtime_guard_required_once_per_worker=True,
        checker_projection='profile and canonical-seed annotations only after independent raw binding; all science retained',
        corrected_identity=dict(ruleset_id=guard.RULESET_ID,catalogue_sha256=guard.CATALOGUE_SHA256,
                                turn_contract=guard.TURN_CONTRACT))


class _ProjectedAdapter:
    def __init__(self,adapter,arm,evidence,mode,pool):
        self._adapter,self._arm,self._record,self._mode,self._pool=adapter,arm,evidence,mode,pool
        self.profile=original_profiles.profile_for(16 if arm=='reference' else 64)
    def __getattr__(self,name):return getattr(self._adapter,name)
    def exchange(self,request):
        actual=deepcopy(request)
        if request.get('op')=='select':
            require(request['observation']['phase']['kind'] in ('Payment','Return','Noble'),
                    'direct native fixture is auxiliary only')
            actual.update(sampling_mode=self._mode,world_pool=self._pool,world_seed=request['seed'])
        result=self._adapter.exchange(actual)
        if request.get('op')=='select':
            sampling=deepcopy(self._adapter.last_sampling)
            validate_sampling(sampling,pool=self._pool,mode=self._mode,world_seed=request['seed'],
                encoding_seed=request['seed'],heuristic=True)
            self._record(dict(event='actual_auxiliary_native_sampling',arm=self._arm,request=actual,result=result,sampling=sampling))
        return result
    def reference_choose(self,observer,**kwargs):
        action,trace=self._adapter.reference_choose(observer,**kwargs)
        self._record(dict(event='research_direct_reference_before_projection',arm=self._arm,
            observer=observer,action=rules.serialize_action(action),diagnostics=trace,
            mode=self._mode,pool=self._pool,research_phases=deepcopy(self._adapter.last_research_phases)))
        projected=deepcopy(trace);projected['profile_id']=original_profiles.profile_id(16)
        projected['observer_seed']=int(digest(observer)[:16],16)
        return action,projected


def run_entrypoint_checks(context,request_callback,*,mode,pool,boundary=lambda:None,evidence_callback=lambda row:None):
    adapters.profile_key(16,pool,mode)
    def routed(payload):
        actual=dict(payload,quota=16 if payload.get('player')=='reference' else 64,
            mode=mode,pool=pool,world_seed_label='A',encoding_seed_label='A',telemetry=True)
        result=request_callback(actual)
        evidence_callback(dict(event='raw_research_result_before_checker_projection',request=actual,result=result))
        projected=frozen_projection(result)
        evidence_callback(dict(event='explicit_frozen_checker_projection',raw_result_sha256=old.digest(result),
            projected_result=projected,projection='validated profile/seed annotations only; raw diagnostic identity retained'))
        return projected
    proxies={arm:_ProjectedAdapter(context['adapters'][arm],arm,evidence_callback,mode,pool) for arm in old.ARMS}
    summary=old.run_entrypoint_checks(proxies,boundary=boundary,request_callback=routed,evidence_callback=evidence_callback)
    require(summary['counts']['complete_decisions']==52,'original52 structural counterexamples')
    return dict(status='research_entrypoint_counterexamples_passed',mode=mode,pool=pool,counts=summary['counts'],
        challenger_quota=64,reference_quota=16,normalized_suite=summary,original_checker_unchanged=True,
        normalization='explicit profile and seed annotation projection; full raw science preserved')


def ordinary_payload(spec,observer=None):
    return dict(player=spec.get('player','search'),quota=spec['quota'],pool=spec['pool'],mode=spec['mode'],
        world_seed_label=spec['world_seed_label'],encoding_seed_label=spec['encoding_seed_label'],
        telemetry=spec.get('telemetry',True),observer=spec['observer'] if observer is None else observer)


def validate_raw_qualification_annotations(result,observer,freeze,*,mode,pool):
    quota=result['simulations_quota'];require(type(quota) is int and quota in (16,64),'qualification16/64')
    item=freeze['research_profiles'][adapters.profile_key(quota,pool,mode)]
    return validate_result(result,observer,quota=quota,world_seed_label='A',mode=mode,pool=pool,encoding_seed_label='A',
        profile_id=adapters.profile_id(quota,pool,mode),profile_sha256=item['sha256'],
        wrapper_id=adapters.wrapper_id(quota,pool,mode),native_binary_sha256=adapters.NATIVE_SHA256,
        player='reference' if quota==16 else 'search')


def validate_parity_rows(rows,plan,profiles,profile_artifacts,controls,control_artifacts):
    require(len(rows)==len(plan['parity_requests'])==45,'fixed45 parity matrix')
    results={}
    for row,spec in zip(rows,plan['parity_requests']):
        require(same(row['spec'],spec),'ordered parity specification')
        clock=row['clock'];check_clock(clock,limit=adapters.COMPLETE_SECONDS)
        require(clock['status']=='ok','failed parity request cannot qualify')
        result=clock['result'];control=spec['player']=='control'
        key=adapters.profile_key(spec['quota'],spec['pool'],spec['mode'])
        document=(controls if control else profiles)[key];item=(control_artifacts if control else profile_artifacts)[key]
        validate_result(result,spec['observer'],quota=spec['quota'],world_seed_label=spec['world_seed_label'],
            encoding_seed_label=spec['encoding_seed_label'],pool=spec['pool'],mode=spec['mode'],
            profile_id=document['profile_id'],profile_sha256=item['sha256'],wrapper_id=document['wrapper_id'],
            native_binary_sha256=document['native_binary_sha256'],telemetry_required=spec['telemetry'],player=spec['player'])
        identity=tuple(spec[k] for k in ('player','quota','pool','mode','world_seed_label','encoding_seed_label','telemetry'))
        require(identity not in results,'unique parity condition');results[identity]=result
    checks=[]
    for identity,on in results.items():
        player,quota,pool,mode,world,encoding,flag=identity
        if player=='control' or not flag:continue
        off=results[player,quota,pool,mode,world,encoding,False]
        require(same(scientific_core(off),scientific_core(on)),'telemetry changed full scientific trajectory')
        require(same([p['callback_bits'] for p in off['research']['phases']],
                     [p['callback_bits'] for p in on['research']['phases']]),'telemetry changed float32 callbacks')
        require(same([p['sampling'] for p in off['research']['phases']],
                     [p['sampling'] for p in on['research']['phases']]),'telemetry changed sampled worlds or RNG boundary')
        checks.append(dict(quota=quota,pool=pool,mode=mode,world=world,encoding=encoding,exact=True))
    for quota in (16,512,2048):
        control=results['control',quota,3,'legacy','A','A',True]
        research=results['reference' if quota==16 else 'search',quota,3,'legacy','A','A',True]
        left,right=scientific_core(control),scientific_core(research);left.pop('player');right.pop('player')
        require(same(left,right),'legacy pool3 differs from original native complete trajectory')
        require(same([{k:p[k] for k in ('telemetry','callback_bits')} for p in control['research']['phases']],
                     [{k:p[k] for k in ('telemetry','callback_bits')} for p in research['research']['phases']]),
                'legacy pool3 original exact float32/root statistics')
    sampling_checks=validate_crossed_sampling(results)
    return dict(status='passed',complete_decisions=45,telemetry_comparisons=checks,
        original_native_exact_quotas=[16,512,2048],sampling_checks=sampling_checks,models_or_decisions_replayed=False)


def validate_crossed_sampling(results):
    """All cells checked from raw native worlds and initial callback identity."""
    checks=[];quotas=sorted({key[1] for key in results if key[0]=='search'})
    require(bool(quotas) and set(quotas)<={512,2048},'fixed crossed sampling budgets')
    def first(quota,pool,world,encoding,mode='split'):
        return results['search',quota,pool,mode,world,encoding,True]['research']['phases'][0]
    for quota in quotas:
        for world in ('A','B'):
            for pool in (3,6):
                a,b=first(quota,pool,world,'A'),first(quota,pool,world,'B')
                require(a['sampling']['root_world_debug']==b['sampling']['root_world_debug'],
                        'encoding seed changed fixed root worlds')
            for encoding in ('A','B'):
                small,large=first(quota,3,world,encoding),first(quota,6,world,encoding)
                require(small['sampling']['root_world_debug']==large['sampling']['root_world_debug'][:3],
                        'three worlds must be nested prefix of six')
                require(small['callback_bits'][0]==large['callback_bits'][0],
                        'pool size changed pre-search root input/output')
        for pool in (3,6):
            for encoding in ('A','B'):
                a,b=first(quota,pool,'A',encoding),first(quota,pool,'B',encoding)
                require(a['callback_bits'][0]==b['callback_bits'][0],
                        'root-world seed changed fixed initial neural input/output')
        legacy_labels=sorted({key[4] for key in results if key[0]=='search' and key[1]==quota and key[3]=='legacy'})
        for label in legacy_labels:
            small,large=first(quota,3,label,label,'legacy'),first(quota,6,label,label,'legacy')
            require(small['sampling']['root_world_debug']==large['sampling']['root_world_debug'][:3],
                    'legacy three worlds must be nested prefix of six')
            require(small['callback_bits'][0]==large['callback_bits'][0],
                    'legacy pool size changed initial neural input/output')
        checks.append(dict(quota=quota,nested_root_worlds=True,encoding_seed_does_not_change_worlds=True,
            world_seed_does_not_change_initial_neural_input=True,legacy_nested_labels=legacy_labels,
            interpretation='controlled sensitivity; not additive causal variance'))
    if quotas==[512,2048]:
        for key,result in results.items():
            if key[0]!='search' or key[1]!=512 or key[-1] is not True:continue
            deep_key=tuple(2048 if i==1 else value for i,value in enumerate(key))
            shallow=result['research']['phases'][0];deep=results[deep_key]['research']['phases'][0]
            require(shallow['sampling']==deep['sampling'] and shallow['callback_bits'][0]==deep['callback_bits'][0],
                    'quota changed initial sampled worlds or neural input')
    return checks


def verify_graph_evidence(path, expected_fixtures):
    """Recompute exact float32 comparisons from retained eager/graph values."""
    from tools.splendorust_inference_opt_20261025 import graph
    packet = json.loads(Path(path).read_text(encoding='utf-8'))
    require(same(packet['fixtures'], expected_fixtures), 'frozen graph fixture inventory')
    graph.validate_parity_fixtures(expected_fixtures)
    require(len(packet['evidence']) == 11, 'eleven graph output pairs')
    rows = []
    for fixture, event in zip(expected_fixtures, packet['evidence']):
        require(event['fixture_id'] == fixture['fixture_id'] and event['batch'] == fixture['batch'],
                'graph fixture coupling')
        row = graph.compare_outputs(fixture, event['eager_outputs'], event['candidate_outputs'])
        require(row['exact_float_equal'] and row['float32_bits_equal'] and row == event['comparison'],
                'exact graph output/float32 comparison')
        rows.append(row)
    require(packet['summary']['status'] == 'passed' and packet['summary']['passed'] is True and
            packet['summary']['fixtures_count'] == 11 and packet['summary']['rows'] == rows,
            'graph successful receipt binding')
    return dict(status='passed', fixtures=11, exact_float32_outputs=True, model_calls=0)


def dynamic_fpu_rounding_bound(visits):
    """Absolute bound for two float32 reports and positive float64 arithmetic.

    The exact running mean is (initial + sum(backups))/(N+1). Both the
    reported all-edge mean and reported node mean are separately rounded to
    float32; each has at most 2**-24 absolute error on [0,1] (conservative).
    At most 6*N+16 binary64 operations cover the positive running-mean path,
    edge sums and their regrouping, and this reconstruction. gamma_k bounds
    their error; the values and convex means are all in [0,1]. This includes
    source arithmetic error rather than treating rounded Q as exact backups.
    """
    integer(visits, 'dynamic-FPU visits', 1, 2048)
    k_u = (6 * visits + 16) * 2.**-53
    return 2.**-23 + k_u / (1. - k_u)


def validate_root_statistics(telemetry, native_observer, choices, action, quota, raw83):
    require(type(telemetry) is dict and set(telemetry) == TELEMETRY_FIELDS, 'root statistic fields')
    visits, qs = telemetry['visits'], telemetry['q']
    require(type(visits) is list and type(qs) is list and len(visits) == len(qs) == 67,
            'canonical67 root arrays')
    safe = original.safe_choices(native_observer, choices)
    supported = {original.canonical_index(a) for a in safe}
    for i, (n, q) in enumerate(zip(visits, qs)):
        integer(n, 'edge visit count', 0, quota)
        require((q is None) == (n == 0), 'unvisited Q must be missing, visited Q present')
        require(i in supported or n == 0, 'visits outside safe legal support')
        if q is not None:
            finite(q, 'edge win credit', 0., 1.)
            require(f32(q) == q, 'edge Q float32')
    require(sum(visits) == quota, 'root visits equal completed quota')
    selected = original.canonical_index(action)
    require(selected in supported and visits[selected] > 0 and visits[selected] == max(visits),
            'selected root edge must attain maximum visits')
    for field in ('root_value', 'network_value', 'selected_edge_value'):
        finite(telemetry[field], field, 0., 1.)
        require(f32(telemetry[field]) == telemetry[field], field+' float32')
    require(telemetry['selected_edge_value'] == qs[selected], 'selected-edge value/Q')
    # Each reported Q and aggregate was separately rounded to float32. The
    # weighted reconstruction therefore has at most two float32 roundings.
    weighted = sum(n*q for n, q in zip(visits, qs) if q is not None)/quota
    require(abs(weighted-telemetry['root_value']) <= 2**-22, 'root visit-weighted value')
    require(native_observer['viewer'] == native_observer['current'], 'root acting-player perspective')
    initial = initial_network_win_credit(raw83)
    # The unchanged native source updates node.value on each dynamic-FPU
    # backup, then exports it under the historical name "network_value".
    expected_node = (initial + quota * telemetry['root_value']) / (quota + 1)
    bound = dynamic_fpu_rounding_bound(quota)
    residual = abs(telemetry['network_value'] - expected_node)
    require(residual <= bound, 'post-search dynamic-FPU node mean')
    return dict(selected_index=selected, visited_edges=sum(n>0 for n in visits),
                safe_edges=len(supported), missing_safe_edges=sum(visits[i] == 0 for i in supported),
                initial_network_win_credit=initial,
                post_search_root_node_value=telemetry['network_value'],
                dynamic_fpu_expected_node_mean=expected_node,
                dynamic_fpu_absolute_residual=residual,
                dynamic_fpu_absolute_rounding_bound=bound)


def validate_sampling(packet,*,pool,mode,world_seed,encoding_seed,heuristic,control=False):
    if control:
        require(packet is None,'original native cannot carry diagnostic sampling evidence');return
    fields={'schema','mode','world_pool','world_seed','encoding_seed','root_world_debug',
        'encoding_rng_before_worlds','encoding_rng_after_worlds','root_world_sha256',
        'root_worlds_sha256','root_world_prefix3_sha256'}
    require(type(packet) is dict and set(packet)==fields,'exact native sampling fields')
    require(packet['schema']=='teacher_variability_sampling_v1' and packet['mode']==mode and
            type(packet['world_pool']) is int and packet['world_pool']==pool and
            type(packet['world_seed']) is int and packet['world_seed']==world_seed and
            type(packet['encoding_seed']) is int and packet['encoding_seed']==encoding_seed,'sampling request identity')
    worlds=packet['root_world_debug']
    require(type(worlds) is list and len(worlds)==(0 if heuristic else pool) and
            all(type(w) is str and w.startswith('GameState {') and len(w)<100000 for w in worlds),
            'full bounded hypothetical world inventory')
    require(packet['root_world_sha256']==[hashlib.sha256(w.encode()).hexdigest() for w in worlds] and
            packet['root_worlds_sha256']==digest(worlds) and packet['root_world_prefix3_sha256']==digest(worlds[:3]),
            'independently recomputed full/prefix root world hashes')
    before,after=packet['encoding_rng_before_worlds'],packet['encoding_rng_after_worlds']
    if heuristic:require(before is None and after is None,'heuristic route generated no root world pool')
    else:
        integer(before,'encoding RNG before worlds',0,2**64-1);integer(after,'encoding RNG after worlds',0,2**64-1)
        if mode=='split':require(before==after,'root world sampler consumed neural-input/residual RNG')


def validate_result(result, observer, *, quota, world_seed_label, profile_id,
                             profile_sha256, wrapper_id, native_binary_sha256,
                             telemetry_required=True, schema=RESEARCH_SCHEMA, player='search',
                             pool=3, mode='legacy', encoding_seed_label=None):
    """Full completed action/phase/root work audit, without model inference."""
    require(type(quota) is int and quota in QUOTAS, 'research quota')
    require(player in ('search','reference','control'), 'player identity')
    require((player != 'reference' or quota == 16) and (player != 'search' or quota != 16), 'qualification reference quota')
    require(player != 'control' or (pool == 3 and mode == 'legacy'), 'original control configuration')
    require(type(result) is dict and set(result) == original.RESULT_FIELDS | {'research'}, 'research result fields')
    state = rules.state_from_observation(observer)
    require(not state.terminal and same(rules.observation_payload(state), observer), 'strict lawful observer')
    legal = [rules.serialize_action(a) for a in rules.legal_actions(state)]
    require(result['player'] == player and result['observer_sha256'] == digest(observer) and
            result['legal_actions_sha256'] == digest(legal) and type(result['legal_count']) is int and
            result['legal_count'] == len(legal), 'observer/legal envelope binding')
    require(result['ruleset'] == guard.RULESET_ID and result['catalogue_sha256'] == guard.CATALOGUE_SHA256 and
            result['turn_contract'] == guard.TURN_CONTRACT, 'corrected identity')
    require(type(result['simulations_quota']) is int and result['simulations_quota'] == quota and
            result['profile_id'] == profile_id and result['profile_sha256'] == profile_sha256 and
            result['wrapper_id'] == wrapper_id, 'externally bound profile/quota identity')
    sha(profile_sha256, 'profile'); sha(native_binary_sha256, 'native binary')
    require(any(same(result['action'], a) for a in legal), 'complete action legality')
    trace, research = result['diagnostics'], result['research']
    require(type(trace) is dict and set(trace) == TRACE_FIELDS, 'original scientific trace fields')
    encoding_seed_label=world_seed_label if encoding_seed_label is None else encoding_seed_label
    world_seed,encoding_seed=adapters.seed_pair(observer,mode,world_seed_label,encoding_seed_label)
    fields={'schema','quota','mode','pool','world_seed_label','encoding_seed_label','world_seed','encoding_seed',
            'telemetry_enabled','internal_seconds','native_binary_sha256','phases'}
    require(type(research) is dict and set(research)==fields,'research annotation fields')
    require(research['schema']==schema and type(research['quota']) is int and research['quota']==quota and
            research['mode']==mode and type(research['pool']) is int and research['pool']==pool and
            research['world_seed_label']==world_seed_label and research['encoding_seed_label']==encoding_seed_label and
            type(research['world_seed']) is int and research['world_seed']==world_seed and
            type(research['encoding_seed']) is int and research['encoding_seed']==encoding_seed and
            type(research['telemetry_enabled']) is bool and research['telemetry_enabled'] is telemetry_required and
            research['internal_seconds']==adapters.INTERNAL_SECONDS and research['native_binary_sha256']==native_binary_sha256,
            'research seed/clock/native/mode identity')
    require(trace['opponent_id']=='splendorust' and trace['profile_id']==profile_id and
            type(trace['observer_seed']) is int and trace['observer_seed']==encoding_seed,'scientific seed/profile')
    phases, extra = trace['phases'], research['phases']
    require(type(phases) is list and type(extra) is list and len(phases) == len(extra) <= 4,
            'finite phase inventory')
    mains = bridge.main_actions(observer)
    forced = mains == [{'kind': 'Pass'}]
    require(type(trace['forced_pass']) is bool and trace['forced_pass'] == forced, 'forced pass')
    cursor = 0; calls = leaves = simulations = 0; checks = []
    def select(native, choices):
        nonlocal cursor, calls, leaves, simulations
        require(cursor < len(phases), 'missing atomic phase')
        phase, annotation = phases[cursor], extra[cursor]
        require(type(phase) is dict and set(phase) == BASE_PHASE | {'inference'}, 'phase fields/capture')
        require(type(annotation) is dict and set(annotation) == {'phase_index', 'phase_kind', 'telemetry', 'callback_bits', 'sampling'} and
                type(annotation['phase_index']) is int and annotation['phase_index'] == cursor and
                annotation['phase_kind'] == native['phase']['kind'], 'research phase coupling')
        cursor += 1
        retained = original.safe_choices(native, choices)
        require(any(same(phase['action'], a) for a in retained), 'phase action outside safe support')
        require(type(phase['native_legal_count']) is int and phase['native_legal_count'] == len(choices),
                'native full legal count')
        heuristic = native['phase']['kind'] != 'Main' or len(choices) == 1 or native['turns'] >= 124
        require(type(phase['heuristic_phase']) is bool and phase['heuristic_phase'] == heuristic,
                'native shortcut classification')
        validate_sampling(annotation['sampling'],pool=pool,mode=mode,world_seed=world_seed,
            encoding_seed=encoding_seed,heuristic=heuristic,control=player=='control')
        n = integer(phase['simulations'], 'phase simulations', 0, quota)
        leaf = integer(phase['leaf_calls'], 'leaf calls', 0, quota+1)
        count = integer(phase['neural_callback_count'], 'callbacks', 0, quota+1)
        require(n == (0 if heuristic else quota), 'actual fixed simulation quota')
        raw83 = validate_callback_capture(phase, annotation['callback_bits'])
        if heuristic:
            require(leaf == count == 0 and annotation['telemetry'] is None,
                    'heuristic phase has no search/telemetry')
        else:
            require(1 <= count <= leaf <= quota+1, 'searched leaf/callback bounds')
            if native['turns']+min(quota, 64) < 124:
                require(count == leaf, 'heuristic leaf unreachable')
            elif native['turns'] == 123:
                require(count == 1, 'turn123 root-only neural call')
            if telemetry_required:
                checks.append(validate_root_statistics(annotation['telemetry'], native, choices,
                                                       phase['action'], quota, raw83))
            else:
                require(annotation['telemetry'] is None, 'telemetry-off route must remain off')
        calls += count; leaves += leaf; simulations += n
        return phase['action']
    main = mains[0] if forced else select(bridge.encode_observation(observer), mains)
    resolved = rules.serialize_action(bridge.resolve_compound(observer, main, select))
    require(cursor == len(phases) and same(resolved, result['action']), 'complete atomic phase/action reconstruction')
    require(type(trace['inference_calls']) is int and trace['inference_calls'] == calls and
            type(trace['inference_rows']) is int and trace['inference_rows'] == calls, 'aggregate neural work')
    return dict(status='passed', main_action=deepcopy(main), phases=cursor, simulations=simulations,
                neural_callbacks=calls, leaf_calls=leaves, root_statistic_checks=checks)

def verify_entrypoint(folder,receipt,plan,freeze,boundary=lambda:None,*,mode,pool):
    """Read back finite raw forensics; no model, process or checker rerun."""
    import math
    from collections import Counter
    from tools.external_strength_screen_20261018 import audit as ledger
    from tools.external_strength_screen_20261018_02 import bridge
    from tools.future_training_rules_20261021 import decode_training_snapshot
    folder=Path(folder);boundary()
    require(receipt['status']=='ok','Entrypoint successful original receipt required')
    packet=receipt['result'];research=packet['status']=='research_entrypoint_counterexamples_passed'
    require(research or packet['status']=='entrypoint_counterexamples_passed','Entrypoint route status')
    suite=packet['normalized_suite'] if research else packet
    require(suite['status']=='entrypoint_counterexamples_passed' and suite['challenger_quota']==64 and
        suite['reference_quota']==16 and suite['counts']=={'complete_decisions':52},'Exact original suite coverage')
    require(packet.get('mode')==mode and packet.get('pool')==pool,'entrypoint mode/pool receipt binding')
    path=folder/f'entrypoint_{mode}_p{pool}_forensics.jsonl'
    require(path.stat().st_size<=64*1024**2,'Finite entrypoint forensic byte bound')
    events=[];previous=None
    with path.open('r',encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            boundary();require(index<2048 and len(line.encode())<=4*1024**2,'Finite forensic event bounds')
            row=json.loads(line)
            require(set(row)=={'sequence','previous_sha256','event'} and row['sequence']==index and
                row['previous_sha256']==previous,'Forensic chain/sequence')
            previous=old.digest(row);events.append(row['event'])
    require(not any(e.get('event') in ('failed','entrypoint_failed','worker_decision_failed','worker_research_decision_failed')
                    for e in events),'Failed scientific event cannot be accepted')
    ordinary=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='result']
    validated=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='validated']
    rejected=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='rejected']
    direct=[e for e in events if e.get('operation')=='reference_choose' and e['event']=='result']
    require(len(ordinary)==len(validated)==37 and len(direct)==15 and len(rejected)==26,
            'Exact37 handler/15 direct/26 rejection forensic coverage')
    expected_rejections=Counter((arm,'worker_rejection/'+name) for arm in old.ARMS for name in old.rejection_case_names())
    require(Counter((e['arm'],e['case_id']) for e in rejected)==expected_rejections,'Every actual rejection case retained')
    after={(e['arm'],e['case_id']):e for e in validated}
    require(len(after)==37,'Distinct actual handler fixture identities')
    for event in ordinary:
        boundary();request=event['request'];result=event['result'];arm=event['arm']
        original_audit.validate_result(result,request['observer'],arm,expected_quota=64 if arm=='search' else 16)
        decoded=decode_training_snapshot(event['before_snapshot'])
        ledger.check_observer(event['before_snapshot'],request['observer'])
        saved_after=after[arm,event['case_id']]['after_snapshot']
        ledger.check_step(event['before_snapshot'],result['action'],saved_after)
        require(decoded.current_player==request['observer']['observation']['current_player'],'Entrypoint actor binding')
    fixtures={row['id']:row for row in suite['fixtures']}
    frozen_cases=plan['original_suite']['counterexamples']
    require(set(fixtures)=={row['id'] for row in frozen_cases} and len(fixtures)==15,'Original15 counterexample fixtures')
    for row in suite['fixtures']:
        for arm in old.ARMS:
            matches=[e for e in ordinary if e['case_id']==row['id'] and e['arm']==arm]
            require(len(matches)==1 and matches[0]['result']==row['results'][arm],'Summary binds actual handler result')
    reference_traces={row['id']:row['trace'] for row in suite['reference_reproductions']}
    require(len(reference_traces)==15,'Reference15 summary inventory')
    for event in direct:
        case=event['case_id'];result=event['result'];baseline=fixtures[case]['results']['reference']
        require(result['diagnostics']==reference_traces[case],'Direct reference saved trace binding')
        require(old.scientific_result('reference',result['action'],result['diagnostics'])==
                old.scientific_result('reference',baseline['action'],baseline['diagnostics']) and
                old.callback_receipts(result['diagnostics'])==old.callback_receipts(baseline['diagnostics']),
                'Reference exact science/callback reproduction')
    require(len(suite['worker_rejections'])==26 and len(suite['actual_encoded_costs'])==30,'Rejection/actual encoder coverage')
    for fixture in frozen_cases:
        state=decode_training_snapshot(fixture['snapshot'])
        for arm in old.ARMS:
            encoded=old._actual_encoded_costs(state,fixtures[fixture['id']]['results'][arm]['diagnostics'],
                corrected_face='corrected_face' in fixture['tags'])
            expected={'case_id':fixture['id'],'arm':arm,**encoded}
            require(expected in suite['actual_encoded_costs'],'Actual encoded corrected costs retained')
    contract=suite['native_contract'];require(contract['status']=='passed' and contract['count']==len(contract['checks'])==28,'28 native ordered supports')
    for check in contract['checks']:
        fixture=next(x for x in frozen_cases if x['id']==check['case_id'])
        observer=rules.observation_payload(decode_training_snapshot(fixture['snapshot']))
        require(check['inspection']['legal']==bridge.main_actions(observer),'Native ordered support readback')
    parity=suite['fresh_numeric_parity']
    require(set(parity)==set(old.ARMS),'Both reference/search numerical parity')
    for arm in old.ARMS:
        value=parity[arm];require(value['status']=='passed' and value['checks']==len(value['rows'])==23 and
            value['max_abs_tolerance']==.0005,'Fresh numerical parity23 per arm')
        for row in value['rows']:
            response=row['response'];left=response['host_policy']+response['host_value'];right=response['native_policy']+response['native_value']
            require(len(left)==len(right)==83 and all(type(x) in (float,int) and math.isfinite(x) for x in left+right),
                    'Finite parity83 outputs')
            require(max(abs(x-y) for x,y in zip(left,right))<=.0005 and 0<=response['max_abs_delta']<=.0005,'Saved native/host output parity')
    actual_aux=[e for e in events if e.get('event')=='actual_auxiliary_native_sampling']
    original_aux=[e for e in events if e.get('event')=='result' and e.get('operation')=='exchange'
                  and e['request']['native_request'].get('op')=='select']
    require(len(actual_aux)==len(original_aux)==6,'all six actual auxiliary sampling probes retained')
    for actual_event,prior in zip(actual_aux,original_aux):
        actual_request=actual_event['request'];original_request=prior['request']['native_request']
        require(actual_event['arm']==prior['arm'] and actual_event['result']==prior['result'] and
                actual_request==dict(original_request,sampling_mode=mode,world_pool=pool,world_seed=original_request['seed']),
                'explicit auxiliary request extension binds retained original fixture')
        validate_sampling(actual_event['sampling'],pool=pool,mode=mode,world_seed=original_request['seed'],
            encoding_seed=original_request['seed'],heuristic=True)
    raw_count=0
    if research:
        raw=[e for e in events if e.get('event')=='raw_research_result_before_checker_projection']
        projected=[e for e in events if e.get('event')=='explicit_frozen_checker_projection']
        raw_direct=[e for e in events if e.get('event')=='research_direct_reference_before_projection']
        require(len(raw)==len(projected)==37 and len(raw_direct)==15,'All raw new identities retained before projection')
        for raw_event,projected_event,event in zip(raw,projected,ordinary):
            result=raw_event['result'];q=result['simulations_quota']
            validate_raw_qualification_annotations(result,raw_event['request']['observer'],freeze,mode=mode,pool=pool)
            require(result['profile_sha256']==freeze['research_profiles'][adapters.profile_key(q,pool,mode)]['sha256'] and
                result['profile_id']==adapters.profile_id(q,pool,mode) and result['wrapper_id']==adapters.wrapper_id(q,pool,mode),
                'Raw research qualification identity binding')
            require(projected_event['raw_result_sha256']==old.digest(result) and
                frozen_projection(result)==projected_event['projected_result']==event['result'],
                'Exact disclosed identity-only projection')
        for event,normalized in zip(raw_direct,direct):
            trace=deepcopy(event['diagnostics']);observer=event['observer']
            require(event['mode']==mode and event['pool']==pool and trace['profile_id']==adapters.profile_id(16,pool,mode),
                    'raw direct research16 mode/pool identity')
            baseline=next(e['result'] for e in raw if e['request']['player']=='reference' and e['request']['observer']==observer)
            require(trace==baseline['diagnostics'] and event['research_phases']==baseline['research']['phases'] and
                    event['action']==baseline['action'],'direct reference exact raw science/callback/sampling reproduction')
            trace['profile_id']=original_profiles.profile_id(16);trace['observer_seed']=int(digest(observer)[:16],16)
            require(trace==normalized['result']['diagnostics'] and event['action']==normalized['result']['action'],
                    'exact disclosed direct reference annotation projection')
        raw_count=len(raw)+len(raw_direct)
    return dict(status='passed',route='research' if research else 'frozen',mode=mode,pool=pool,complete_decisions=52,
        ordinary_handlers=37,direct_references=15,negative_handler_cases=26,actual_encoded_cost_checks=30,
        native_ordered_supports=28,native_numeric_parity=46,independent_saved_transitions=37,
        raw_research_decisions_bound=raw_count,forensic_events=len(events),forensic_terminal_sha256=previous,
        forensic_artifact=old.artifact(path),model_calls=0,qualification_rerun=False)

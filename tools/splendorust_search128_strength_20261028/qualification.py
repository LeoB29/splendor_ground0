"""Finite actual-entrypoint qualification for genuine128 versus64, both pool3.

Raw new identities and actual quotas are validated before any historical
identity/capture-only comparison. No model is loaded by preparation or readback.
"""
from copy import deepcopy
from collections import Counter
import json
import math
from pathlib import Path
from . import adapters, common as c, corpus, scientific
from tools.external_strength_screen_20261018 import audit as ledger, rules
from tools.external_strength_screen_20261018_02 import bridge
from tools.future_training_rules_20261021 import decode_training_snapshot, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT

PRIOR=c.ROOT/'outputs/splendorust_sampling_stability_20261027_01'
MATRIX=PRIOR/'prepared_request_matrix.json'
PANEL=c.ROOT/'outputs/decision_quality_preparation_20261027_01/panel.json'
EDGES=('turn123','turn124','forced_pass','gold_returns','multiple_nobles','noble_final_round')
require=c.require

def build_plan():
    matrix=c.read(MATRIX);panel=c.read(PANEL);reserved=c.read(c.OUT/'reserved_panel.json')
    require(len(panel['roots'])==24 and len(reserved['positions'])==18,'Fixed24 saved and18 fresh roots')
    suites={str(q):corpus.build_plan(q) for q in (64,128)}
    rows=[]
    def add(route,kind,case_id,observer,before,**extra):
        rows.append(dict(index=len(rows),route=route,kind=kind,case_id=case_id,
            quota=adapters.route_quota(route),world_pool=3,seed_label='A',telemetry=False,
            observer=observer,observer_sha256=c.digest(observer),before_snapshot=before,**extra))
    for root in panel['roots']:
        matches=[x for x in matrix['requests'] if x['root_id']==root['root_id'] and x['route']=='frozen' and x['world_pool']==3 and x['seed_label']=='A']
        require(len(matches)==1,'One original64 historical target')
        c.verify([root['source']]);before=c.read(c.ROOT/root['source']['path'])['before']
        add('frozen','saved_reproduction',root['root_id'],root['observer'],before,
            expected_record=c.artifact(PRIOR/'records'/f"{matches[0]['sequence']:04d}.json"),root_id=root['root_id'])
    for index,pos in enumerate(reserved['positions']):
        for route in (('frozen','research') if index%2==0 else ('research','frozen')):
            add(route,'fresh_reserved',str(pos['id']),pos['observer'],pos['before_snapshot'],reserved_id=pos['id'],
                reserved_seed_decimal=pos['seed_decimal'],target_ply=pos['target_ply'],seat=pos['seat'])
    for index,case_id in enumerate(EDGES):
        fixture=next(x for x in suites['64']['counterexamples'] if x['id']==case_id)
        observer=rules.observation_payload(decode_training_snapshot(fixture['snapshot']))
        for route in (('frozen','research') if index%2==0 else ('research','frozen')):
            add(route,'live_edge',case_id,observer,fixture['snapshot'],expected_case=case_id)
    # Fixed first six fresh roots, selected before any model outputs or timings.
    for index,pos in enumerate(reserved['positions'][:6]):
        for route in (('research','frozen') if index%2==0 else ('frozen','research')):
            first=next(x for x in rows if x['route']==route and x['kind']=='fresh_reserved' and x['case_id']==str(pos['id']))
            add(route,'exact_repeat',str(pos['id']),pos['observer'],pos['before_snapshot'],repeat_of=first['index'])
    require(len(rows)==84 and Counter(x['route'] for x in rows)==dict(frozen=54,research=30),'Fixed84 ordinary inventory')
    return dict(schema='search128_strength_qualification_plan_v1',
        graph_requests=[dict(route=r,fixtures=11) for r in ('frozen','research')],
        entrypoint_requests=[dict(route='frozen',complete_decisions=52,ordinary_handlers=37,negative_handlers=26,direct_reference=15,direct_same_quota=0),
            dict(route='research',complete_decisions=67,ordinary_handlers=37,negative_handlers=26,direct_reference=15,direct_same_quota=15)],
        parity_requests=rows,complete_decisions=203,ordinary_requests=84,administrative_parent_requests=4,
        total_parent_requests=88,graph_cases=22,qualification_worker_trees=2,maximum_simultaneous_qualification_trees=2,
        stage_seconds=3600,entrypoint_external_seconds=300,graph_external_seconds=60,
        ordinary_frozen_external_seconds=10.,ordinary_research_external_seconds=10.,ordinary_internal_seconds=9.,
        original_suite=suites['64'],actual_suites=suites,world_pool=3,saved_reproductions=24,
        edge_cases=list(EDGES),edge_requests=12,fresh_reserved_positions=18,fresh_reserved_requests=36,exact_repeats=12,
        normalization=dict(raw_identity_checked_first=True,identity_fields=['profile_id','profile_sha256','wrapper_id','diagnostics.profile_id'],
            historical_capture_only_fields=['diagnostics.phases[].inference'],quota_never_normalized=True,raw_clock_never_normalized=True),
        fixture_source=c.artifact(PANEL),request_matrix=c.artifact(MATRIX),reserved_panel=c.artifact(c.OUT/'reserved_panel.json'),
        no_retries=True,no_replacements=True,timing_margins_are_descriptive=True,model_execution_during_preparation=False)

def scientific_core(result,*,capture=True):
    value=deepcopy(result)
    for key in ('profile_id','profile_sha256','wrapper_id'):value.pop(key,None)
    value['diagnostics'].pop('profile_id',None)
    if not capture:
        for phase in value['diagnostics']['phases']:phase.pop('inference',None)
    return value

def _saved(item,cache,boundary):
    key=(item['path'],item['sha256'],item['bytes']);boundary()
    if key not in cache.setdefault('artifacts',{}):
        c.verify([item]);cache['artifacts'][key]=c.read(c.ROOT/item['path'])
    return cache['artifacts'][key]

def validate_admission_row(row,spec,cache,*,boundary=lambda:None,frozen_entrypoint=None):
    boundary();scientific.check_clock(row['clock'],limit=10.)
    require(row['spec']==spec and row['completed'] is True and row['clock']['accepted'] is True and row['clock']['status']=='ok','Exact completed qualification request')
    result=row['clock']['result'];route=spec['route'];quota=adapters.route_quota(route)
    require(spec['quota']==quota and spec['world_pool']==3 and spec['seed_label']=='A' and spec['telemetry'] is False,'Fixed scientific condition')
    if 'freeze' not in cache:cache['freeze']=c.read(c.OUT/'freeze.json')
    identity=adapters.profile_identity(cache['freeze'],route,quota,boundary=boundary)
    adapters.validate_result(result,spec['observer'],route,identity=identity)
    require(result['observer_sha256']==spec['observer_sha256']==c.digest(spec['observer']),'Observer identity')
    ledger.check_observer(spec['before_snapshot'],spec['observer'])
    kind=spec['kind'];late=0
    if kind=='saved_reproduction':
        require(route=='frozen' and quota==64,'Historical exact64 only')
        prior=_saved(spec['expected_record'],cache,boundary)
        require(prior['completed'] and prior['clock']['status']=='ok' and prior['observer']==spec['observer'],'Accepted historical input')
        require(scientific_core(result,capture=False)==scientific_core(prior['clock']['result'],capture=False),
            'Original64 actions, numeric callback hashes and work must reproduce exactly')
    elif kind=='live_edge':
        key='entrypoint_'+route
        if key not in cache:cache[key]=frozen_entrypoint if route=='frozen' and frozen_entrypoint is not None else c.read(c.OUT/'entrypoint_checks'/f'{route}.json')
        suite=cache[key];require(suite['status']=='ok','Current actual suite source')
        expected=next(x for x in suite['result']['fixtures'] if x['id']==spec['expected_case'])['results']['search']
        require(result==expected,'Complete edge output and captured callbacks must match actual suite')
        if spec['case_id']=='turn124':
            require(result['diagnostics']['inference_calls']==0 and all(p['heuristic_phase'] and p['simulations']==p['leaf_calls']==0 for p in result['diagnostics']['phases']),'Late shortcut cannot search')
            other=cache.setdefault('late_actions',{})
            view=dict(action=result['action'],diagnostics=deepcopy(result['diagnostics']));view['diagnostics'].pop('profile_id')
            if other:require(view==next(iter(other.values())),'Quota must not change exact late heuristic science')
            other[route]=view;late=1
    elif kind=='fresh_reserved':pass
    elif kind=='exact_repeat':
        require(spec['repeat_of'] in cache.setdefault('results',{}),'Repeat follows its original root')
        require(result==cache['results'][spec['repeat_of']],'Exact full result and callback arrays differ on repeat')
    else:raise ValueError('Unknown qualification request kind')
    require(spec['index'] not in cache.setdefault('results',{}),'No repeated scheduled request')
    cache['results'][spec['index']]=deepcopy(result)
    return dict(status='passed',category=kind,exact_late_shortcuts=late,model_calls=0)

def validate_parity_rows(rows,plan,*,boundary=lambda:None,frozen_entrypoint=None,allow_prefix=False):
    expected=plan['parity_requests'];require(len(expected)==84 and len(rows)<=84,'Bounded84 inventory')
    if not allow_prefix:require(len(rows)==84,'Complete84 inventory')
    cache={};counts=Counter();failures=[];late=0
    for index,row in enumerate(rows):
        spec=expected[index];require(row['spec']==spec,'Exact ordered qualification inventory');scientific.check_clock(row['clock'],limit=10.)
        if row['clock']['status']!='ok' or row['clock']['accepted'] is not True:
            require(allow_prefix and index==len(rows)-1 and row['completed'] is False and row['clock']['accepted'] is False,'Only terminal failure retained')
            failures.append(dict(index=index,status=row['clock']['status'],detail=row['clock'].get('detail')));continue
        proof=validate_admission_row(row,spec,cache,boundary=boundary,frozen_entrypoint=frozen_entrypoint)
        counts[proof['category']]+=1;late+=proof['exact_late_shortcuts']
    complete=len(rows)==84 and not failures
    if complete:require(counts==dict(saved_reproduction=24,fresh_reserved=36,live_edge=12,exact_repeat=12) and late==2,'All frozen categories and both late routes')
    return dict(status='passed' if complete else 'verified_prefix',complete=complete,complete_decisions=sum(counts.values()),attempted=len(rows),
        counts=dict(counts),failures=failures,exact_late_shortcuts=late,raw_quotas_preserved=True,model_calls=0,no_retries=True)

def verify_entrypoint(folder,receipt,plan,freeze,boundary=lambda:None):
    """Authenticate actual handler/forensic outputs; no fresh model invocation."""
    folder=Path(folder);boundary();require(receipt['status']=='ok' and receipt['accepted'] is True,'Successful actual suite receipt')
    suite=receipt['result'];route=suite['execution_route'];quota=adapters.route_quota(route);extra=15 if quota==128 else 0
    require(suite['execution_schema']==adapters.EXECUTION_SCHEMA and suite['status']=='entrypoint_counterexamples_passed' and
        suite['challenger_quota']==quota and suite['reference_quota']==16 and suite['world_pool']==3 and
        suite['counts']==dict(complete_decisions=52+extra) and suite['original_counts']==dict(complete_decisions=52),'Actual suite identity/counts')
    require(suite['ruleset_id']==RULESET_ID and suite['catalogue_sha256']==CATALOGUE_SHA256 and suite['turn_contract']==TURN_CONTRACT,'Corrected actual suite rules')
    path=folder/'entrypoint_forensics.jsonl';require(path.stat().st_size<=64*1024**2,'Forensic byte bound')
    events=[];previous=None
    with path.open('r',encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            boundary();require(index<2048 and len(line.encode())<=4*1024**2,'Forensic record bounds')
            item=json.loads(line);require(set(item)=={'sequence','previous_sha256','event'} and item['sequence']==index and item['previous_sha256']==previous,'Exact durable forensic chain')
            previous=c.digest(item);events.append(item['event'])
    require(not any(e.get('event') in ('failed','entrypoint_failed','worker_decision_failed') for e in events),'Failure cannot pass qualification')
    ordinary=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='result']
    validated=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='validated']
    rejected=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='rejected']
    direct=[e for e in events if e.get('operation')=='reference_choose' and e['event']=='result']
    completed=[e for e in events if e.get('event')=='complete_result_before_worker_validation']
    same=[e for e in events if e.get('operation')=='direct_same_quota_choose' and e['event']=='result']
    require(len(ordinary)==len(validated)==len(completed)==37 and len(direct)==15 and len(rejected)==26 and len(same)==extra,'Complete actual entrypoint inventory')
    require(Counter((e['arm'],e['case_id']) for e in rejected)==Counter((a,'worker_rejection/'+n) for a in corpus.ARMS for n in corpus.rejection_case_names()),'All negative handler cases')
    after={(e['arm'],e['case_id']):e for e in validated};require(len(after)==37,'Distinct actual handler identities')
    identities={q:adapters.profile_identity(freeze,route,q,boundary=boundary) for q in (16,quota)}
    for event,raw in zip(ordinary,completed,strict=True):
        request=event['request'];result=event['result'];arm=event['arm'];q=quota if arm=='search' else 16
        require(request==dict(player=arm,observer=request['observer']),'Exact ordinary two-field request')
        adapters.validate_result(result,request['observer'],route,identity=identities[q])
        ledger.check_observer(event['before_snapshot'],request['observer']);decode_training_snapshot(event['before_snapshot'])
        ledger.check_step(event['before_snapshot'],result['action'],after[arm,event['case_id']]['after_snapshot'])
        require(raw['arm']==arm and raw['observer']==request['observer'] and raw['action']==result['action'] and raw['diagnostics']==result['diagnostics'],'Prevalidation raw output binding')
    fixtures={r['id']:r for r in suite['fixtures']};cases=plan['actual_suites'][str(quota)]['counterexamples']
    require(set(fixtures)=={x['id'] for x in cases} and len(fixtures)==15,'All15 structural fixtures')
    for case in cases:
        for arm in corpus.ARMS:
            matching=[e for e in ordinary if e['case_id']==case['id'] and e['arm']==arm]
            require(len(matching)==1 and matching[0]['result']==fixtures[case['id']]['results'][arm] and matching[0]['before_snapshot']==case['snapshot'],'Summary binds actual fixture')
    reference_traces={x['id']:x['trace'] for x in suite['reference_reproductions']};require(len(reference_traces)==15,'Direct reference inventory')
    for event in direct:
        result=event['result'];baseline=fixtures[event['case_id']]['results']['reference']
        require(result==dict(action=baseline['action'],diagnostics=baseline['diagnostics']) and result['diagnostics']==reference_traces[event['case_id']],'Exact same16 action/full callbacks')
    require(len(suite['direct_same_quota'])==extra,'Direct128 summary count')
    same_summary={x['id']:x for x in suite['direct_same_quota']}
    require(set(same_summary)==(set(fixtures) if extra else set()),'Direct128 exact fixture inventory')
    for event in same:
        case=event['case_id'];expected=fixtures[case]['results']['search'];summary=same_summary[case]
        require(event['request']==dict(observer=next(e for e in ordinary if e['case_id']==case and e['arm']=='search')['request']['observer'],quota=128) and
            event['result']==summary['result']==dict(action=expected['action'],diagnostics=expected['diagnostics']) and summary['exact_callback_arrays_and_hashes'] is True,'Direct genuine128 exact native/host science')
    require(len(suite['worker_rejections'])==26 and len(suite['actual_encoded_costs'])==30,'Negative/encoded-cost inventory')
    for case in cases:
        state=decode_training_snapshot(case['snapshot'])
        for arm in corpus.ARMS:
            encoded=corpus._actual_encoded_costs(state,fixtures[case['id']]['results'][arm]['diagnostics'],corrected_face='corrected_face' in case['tags'])
            require(dict(case_id=case['id'],arm=arm,**encoded) in suite['actual_encoded_costs'],'Actual corrected encoding evidence')
    contract=suite['native_contract'];require(contract['status']=='passed' and contract['count']==len(contract['checks'])==28,'Native support inventory')
    for check in contract['checks']:
        case=next(x for x in cases if x['id']==check['case_id']);observer=rules.observation_payload(decode_training_snapshot(case['snapshot']))
        require(check['inspection']['legal']==bridge.main_actions(observer),'Native ordered support')
    parity=suite['fresh_numeric_parity'];require(set(parity)==set(corpus.ARMS),'Both native numeric paths')
    for value in parity.values():
        require(value['status']=='passed' and value['checks']==len(value['rows'])==23 and value['max_abs_tolerance']==.0005,'23 exact configured numeric fixtures')
        for row in value['rows']:
            response=row['response'];left=response['host_policy']+response['host_value'];right=response['native_policy']+response['native_value']
            require(len(left)==len(right)==83 and all(type(x) in (int,float) and math.isfinite(x) for x in left+right),'Finite native/host outputs')
            require(max(abs(x-y) for x,y in zip(left,right))<=.0005 and 0<=response['max_abs_delta']<=.0005,'Native/host output tolerance')
    return dict(status='passed',route=route,complete_decisions=52+extra,original_complete_decisions=52,
        ordinary_handlers=37,negative_handler_cases=26,direct_references=15,direct_same_quota=extra,
        actual_encoded_cost_checks=30,native_ordered_supports=28,native_numeric_parity=46,
        independent_saved_transitions=37,raw_execution_decisions_bound=52+extra,
        forensic_events=len(events),forensic_terminal_sha256=previous,forensic_artifact=c.artifact(path),model_calls=0,qualification_rerun=False)

ordinary_payload=adapters.ordinary_payload

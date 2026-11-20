"""Actual entrypoint forensics for the unchanged-player saved repeat."""
from collections import Counter
import json
import math
from pathlib import Path
from . import adapters,common as c,corpus,scientific
from tools.external_strength_screen_20261018 import audit as ledger,rules
from tools.external_strength_screen_20261018_02 import bridge
from tools.future_training_rules_20261021 import decode_training_snapshot,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
require=c.require
QUAL_ROUTES=('frozen','parity','low','high')
PANEL_REPEAT_INDICES=(0,4,8,9,13,17)


def build_plan():
    return dict(schema='exploration128_qualification_v1',
        actual_suites={'128':corpus.build_plan(128)},
        entrypoint_requests=[dict(route=route,complete_decisions=67,ordinary_handlers=37,negative_handlers=26,
            direct_reference=15,direct_same_quota=15) for route in QUAL_ROUTES],
        entrypoint_scientific_decisions=268,ordinary_requests=96,scientific_decisions=364,
        graph_cases=44,graph_requests=4,entrypoint_parent_requests=4,administrative_parent_requests=8,
        total_parent_requests=104,worker_trees=4,maximum_simultaneous_trees=2,
        entrypoint_external_seconds=300.,graph_external_seconds=60.,ordinary_external_seconds=10.,ordinary_internal_seconds=9.,
        source_guard_once_each=True,no_retries=True,panel_requests=build_panel_requests(),
        panel_repeat_indices=list(PANEL_REPEAT_INDICES),panel_source=corpus.artifact(corpus.PANEL))


def canonical_scientific_result(result):
    """Exclude only transport/profile labels; retain every scientific output bit."""
    from copy import deepcopy
    value=deepcopy(result)
    for key in ('profile_id','profile_sha256','wrapper_id'):value.pop(key,None)
    value['diagnostics'].pop('profile_id')
    return value


def compare_default_parity(frozen_suite,parity_suite):
    require(frozen_suite['execution_route']=='frozen' and parity_suite['execution_route']=='parity','Distinct old/new0.4 routes')
    frozen={row['id']:row for row in frozen_suite['fixtures']};new={row['id']:row for row in parity_suite['fixtures']}
    require(set(frozen)==set(new) and len(frozen)==15,'Exact15 suite fixture inventory')
    for name in frozen:
        for arm in ('reference','search'):
            require(c.digest(canonical_scientific_result(frozen[name]['results'][arm]))==
                    c.digest(canonical_scientific_result(new[name]['results'][arm])),'New0.4 versus original exact action, arrays, callback hashes and work')
    return dict(status='passed',cases=15,complete_results=30,compared='all scientific result fields except declared profile labels',
        old_native_sha256=adapters.SEARCH_BINARY_SHA256,new_native_sha256=adapters.native_descriptor('parity')['sha256'])


def verify_entrypoint(folder,receipt,plan,freeze,boundary=lambda:None):
    """Authenticate actual handler/forensic outputs; no fresh model invocation."""
    folder=Path(folder);boundary();require(receipt['status']=='ok' and receipt['accepted'] is True,'Successful actual suite receipt')
    suite=receipt['result'];route=suite['execution_route'];quota=adapters.route_quota(route);extra=15
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


def build_panel_requests():
    saved=corpus.read(corpus.PANEL)
    core=[item for item in saved if item['source']['reason'][0]=='core']
    require(len(core)==18 and len({item['source']['root_id'] for item in core})==18,'Exact predetermined18 core positions')
    requests=[]
    for route in QUAL_ROUTES:
        for within,index in enumerate(list(range(18))+list(PANEL_REPEAT_INDICES)):
            item=core[index];before=item['snapshot'];observer=rules.observation_payload(decode_training_snapshot(before))
            ledger.check_observer(before,observer)
            requests.append(dict(index=len(requests),route=route,quota=128,root_id=item['source']['root_id'],
                fixture_index=index,repeat=int(within>=18),before=before,observer=observer,
                meaning='Previously corrected counterfactual structural fixture; no historical outcome label reused'))
    require(len(requests)==96,'Fixed96 qualification panel decisions');return requests


def validate_panel_rows(rows,plan,freeze,*,boundary=lambda:None):
    require(plan['panel_requests']==build_panel_requests(),'Exact predeclared qualification panel')
    require(len(rows)==len(plan['panel_requests'])==96,'Complete96 panel requests')
    first={};repeats=0;parity=0;timings={route:[] for route in QUAL_ROUTES}
    for row,spec in zip(rows,plan['panel_requests'],strict=True):
        boundary();require(row['spec']==spec and row['route']==spec['route'] and row['observer']==spec['observer'],
            'Panel row order/request binding')
        require(row['completed'] is True and row['clock']['status']=='ok' and row['clock']['deadline_seconds']==10.,
            'Qualification requires completed research-clock decisions')
        scientific.check_clock(row['clock'],limit=10.)
        identity=adapters.profile_identity(freeze,spec['route'],128,boundary=boundary)
        result=row['clock']['result'];adapters.validate_result(result,spec['observer'],spec['route'],identity=identity)
        require(row['action']==result['action'] and row['trace']==result['diagnostics'],'Exact panel returned fields')
        ledger.check_observer(spec['before'],spec['observer']);after=ledger.transition(spec['before'],result['action']);decode_training_snapshot(after)
        key=(spec['route'],spec['root_id'])
        if spec['repeat']:
            require(result==first[key],'Exact full configured output repeat');repeats+=1
        else:
            require(key not in first,'No duplicate initial panel root');first[key]=result
        if spec['route']=='parity':
            require(c.digest(canonical_scientific_result(result))==c.digest(canonical_scientific_result(first['frozen',spec['root_id']])),
                'New0.4 native versus original0.4 full panel parity');parity+=1
        timings[spec['route']].append(row['clock']['receipt_elapsed_seconds'])
    require(len(first)==72 and repeats==24 and parity==24,'Full cross-route/repeat inventories')
    return dict(status='passed',ordinary_decisions=96,roots=18,exact_repeats=24,default_parity_comparisons=24,
        matched_default_first_results=18,matched_default_repeat_results=6,independent_transitions=96,
        timing_seconds={route:dict(count=len(values),maximum=max(values),over_one_second=sum(x>1. for x in values),
            over_700ms=sum(x>.7 for x in values)) for route,values in timings.items()},
        timing_descriptive_only=True,thresholds_do_not_gate=True)

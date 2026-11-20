"""Actual corrected entrypoint and original256 saved parity; no model imports."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from . import adapters,common as c,corpus,scientific
from tools.external_strength_screen_20261018 import audit as ledger,rules
from tools.external_strength_screen_20261018_02 import bridge
from tools.future_training_rules_20261021 import decode_training_snapshot,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
require=c.require
QUAL_ROUTES=adapters.ROUTES
BASELINE_EVIDENCE=c.ROOT/'outputs/splendorust_network_live_20261031_01'
BASELINE_ACCEPTANCE=c.ROOT/'outputs/splendorust_network_live_20261031_01_closeout/independent_acceptance.json'
BASELINE_ACCEPTANCE_SHA256='a01a03648ba8ede8691aaba82e5c0640d4e0a690e2e66542c09479876e6b80a4'


def baseline_parity_sources():
    require(c.file_hash(BASELINE_ACCEPTANCE)==BASELINE_ACCEPTANCE_SHA256 and c.read(BASELINE_ACCEPTANCE)['approved'] is True,
        'Accepted immutable original256 evidence')
    source=BASELINE_EVIDENCE/'workers/live_control/entrypoint_clock.json'
    seal=BASELINE_EVIDENCE/'completed_evidence_seal.json'
    seal_item=next(x for x in c.read(BASELINE_ACCEPTANCE)['inputs'] if x['path']==seal.relative_to(c.ROOT).as_posix())
    c.verify([seal_item])
    item=next(x for x in c.read(seal)['artifacts'] if x['path']==source.relative_to(c.ROOT).as_posix())
    c.verify([item])
    return dict(acceptance=c.artifact(BASELINE_ACCEPTANCE),entrypoint=item,seal=seal_item)


def build_plan():
    return dict(schema='search512_qualification_v1',
        actual_suites={str(quota):corpus.build_plan(quota) for quota in (256,512)},
        entrypoint_requests=[dict(route=route,complete_decisions=67,ordinary_handlers=37,negative_handlers=26,
            direct_reference=15,direct_same_quota=15) for route in QUAL_ROUTES],
        per_worker=dict(graph_cases=11,scientific_decisions=67,ordinary_requests=0,parent_requests=2),
        entrypoint_scientific_decisions=134,ordinary_requests=0,scientific_decisions=134,
        graph_cases=22,graph_requests=2,entrypoint_parent_requests=2,administrative_parent_requests=4,
        total_parent_requests=4,worker_trees=2,maximum_simultaneous_trees=2,
        entrypoint_external_seconds=300.,graph_external_seconds=60.,ordinary_external_seconds=10.,ordinary_internal_seconds=9.,
        source_guard_once_each=True,no_failed_request_retries=True,panel_requests=[],
        baseline_parity_sources=baseline_parity_sources(),
        repeatability='Every512 suite has15 exact direct same-quota repeats; every later lifecycle repeats the first route suite exactly; selected played decisions reproduce in fresh qualified processes.')


def canonical_scientific_result(result):
    """Retain every scientific field; only outer study suite labels are normalized elsewhere."""
    return deepcopy(result)


def compare_control_parity(control_suite,plan,*,boundary=lambda:None):
    sources=plan['baseline_parity_sources']
    require(sources==baseline_parity_sources(),'Frozen accepted original256 source inventory')
    boundary();c.verify(list(sources.values()))
    receipt=c.read(c.ROOT/sources['entrypoint']['path']);old=receipt['result']
    require(receipt['status']=='ok' and receipt['accepted'] is True and
        control_suite['execution_route']=='control' and old['execution_route']=='research',
        'Passed original256 and current control route')
    require(control_suite['execution_schema']==adapters.EXECUTION_SCHEMA and
        old['execution_schema']=='splendorust_search256_strength_execution_v1','Explicit original/new suite schemas')
    left=deepcopy(control_suite);right=deepcopy(old)
    for key in ('execution_schema','execution_route'):left.pop(key);right.pop(key)
    require(c.digest(left)==c.digest(right),'Exact full original256 suite science, arrays, callbacks, work and support')
    return dict(status='passed',complete_scientific_decisions=67,fixture_complete_results=30,direct_same_quota=15,
        compared='entire actual suite except execution_schema and execution_route labels',
        original_source=sources['entrypoint'],native_binary_sha256=adapters.SEARCH_BINARY_SHA256)


def compare_baseline_parity(control_suite,plan,*,boundary=lambda:None):
    return compare_control_parity(control_suite,plan,boundary=boundary)


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
    require(len(suite['direct_same_quota'])==extra,'Direct same-quota summary count')
    same_summary={x['id']:x for x in suite['direct_same_quota']}
    require(set(same_summary)==(set(fixtures) if extra else set()),'Direct same-quota exact fixture inventory')
    for event in same:
        case=event['case_id'];expected=fixtures[case]['results']['search'];summary=same_summary[case]
        require(event['request']==dict(observer=next(e for e in ordinary if e['case_id']==case and e['arm']=='search')['request']['observer'],quota=quota) and
            event['result']==summary['result']==dict(action=expected['action'],diagnostics=expected['diagnostics']) and summary['exact_callback_arrays_and_hashes'] is True,'Direct genuine configured quota exact native/host science')
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


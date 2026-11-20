"""Finite corrected-entrypoint and telemetry parity qualification.

This module imports no model. The original scientific checker is reused through
an explicitly labelled projection; raw research identities are preserved first.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

from tools.external_strength_screen_20261018 import rules
from tools.splendorust_search_scale_20261024 import qualification as old
from tools.splendorust_search_scale_20261024 import common as original
from tools.splendorust_search_scale_20261024 import audit as original_audit
from . import adapters, common as c
from tools.external_strength_screen_20261018 import audit as ledger

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/'outputs/splendorust_sampling_stability_20261027_01'
MATRIX = PRIOR/'prepared_request_matrix.json'
PANEL = ROOT/'outputs/decision_quality_preparation_20261027_01/panel.json'

def require(value,message):
    if not value: raise ValueError('DECISION_QUALIFICATION: '+message)

def build_plan():
    from . import common as c
    matrix=json.loads(MATRIX.read_text(encoding='utf-8'))
    panel=json.loads(PANEL.read_text(encoding='utf-8'))
    reserved=c.read(c.QUAL_OUT/'reserved_panel.json')
    require(len(reserved['positions'])==18,'Exactly18 frozen reserved positions')
    suite=old.build_plan(64);edges=('turn123','turn124','forced_pass','gold_returns','multiple_nobles','noble_final_round')
    parity=[]
    def add(route,kind,case_id,observer,before,**extra):
        parity.append(dict(index=len(parity),route=route,kind=kind,case_id=case_id,
            quota=64,world_pool=3 if route=='frozen' else 6,seed_label='A',telemetry=False,
            observer=observer,observer_sha256=old.digest(observer),before_snapshot=before,**extra))
    for route in ('frozen','research'):
        for root in panel['roots']:
            matches=[x for x in matrix['requests'] if x['root_id']==root['root_id'] and x['route']==route and
                x['world_pool']==(3 if route=='frozen' else 6) and x['seed_label']=='A' and
                x['stage']!='research_reproduction']
            require(len(matches)==1,'One accepted saved target per route/root')
            target=PRIOR/'records'/f"{matches[0]['sequence']:04d}.json"
            require(old.artifact(ROOT/root['source']['path'])==root['source'],'Historical root source changed')
            before=c.read(ROOT/root['source']['path'])['before']
            add(route,'saved_reproduction',root['root_id'],root['observer'],before,
                expected_record=old.artifact(target),root_id=root['root_id'])
        if route=='research':
            for case_id in edges:
                fixture=next(x for x in suite['counterexamples'] if x['id']==case_id)
                # Use the mandatory strict corrected decoder, never a legacy snapshot route.
                from tools.future_training_rules_20261021 import decode_training_snapshot
                state=decode_training_snapshot(fixture['snapshot'])
                add(route,'live_edge',case_id,rules.observation_payload(state),fixture['snapshot'],
                    expected_entrypoint=old.artifact(PRIOR/'entrypoint_checks/research.json'),expected_case=case_id,
                    exact_frozen_shortcut=case_id=='turn124')
        for pos in reserved['positions']:
            add(route,'fresh_reserved',str(pos['id']),pos['observer'],pos['before_snapshot'],
                reserved_id=pos['id'],reserved_seed_decimal=pos['seed_decimal'],target_ply=pos['target_ply'],seat=pos['seat'])
    require(len(parity)==90,'Finite90 ordinary live qualification requests')
    paired=[x for x in parity if x['route']=='frozen']
    lookup={(x['route'],x['kind'],x['case_id']):x for x in parity}
    edge_rows=[x for x in parity if x['kind']=='live_edge'];ordered=[]
    for index,left in enumerate(paired):
        arms=('frozen','research') if index%2==0 else ('research','frozen')
        for route in arms:ordered.append(lookup[route,left['kind'],left['case_id']])
        if (index+1)%7==0:ordered.append(edge_rows[(index+1)//7-1])
    parity=[dict(x,index=i,source_plan_index=x['index']) for i,x in enumerate(ordered)]
    return dict(schema='pool6_strength_qualification_plan_v1',
        graph_requests=[dict(route=r,fixtures=11) for r in ('frozen','research')],
        entrypoint_requests=[dict(route=r,complete_decisions=52) for r in ('frozen','research')],
        parity_requests=parity,complete_decisions=194,graph_cases=22,
        qualification_worker_trees=2,maximum_simultaneous_qualification_trees=2,
        stage_seconds=1800,entrypoint_external_seconds=300,graph_external_seconds=60,
        ordinary_frozen_external_seconds=10,ordinary_research_external_seconds=10,
        ordinary_internal_seconds=9.,candidate_world_pool=6,reference_world_pool=3,
        original_suite=suite,saved_reproductions_per_route=24,edge_cases=list(edges),
        fresh_reserved_positions=18,fresh_reserved_requests=36,
        normalization=dict(raw_evidence_preserved_before_projection=True,
            removed_envelope_fields=['research'],
            identity_replacements=['profile_id','profile_sha256','wrapper_id','diagnostics.profile_id'],
            fields_not_normalized=['action','quota','observer_seed','callback hashes','work counts','numeric values']),
        fixture_source=old.artifact(PANEL),request_matrix=old.artifact(MATRIX),reserved_panel=old.artifact(c.QUAL_OUT/'reserved_panel.json'),
        no_retries=True,no_replacements=True,model_execution_during_preparation=False)

def frozen_projection(result,observer,route,*,identity):
    return adapters.original_projection(result,observer,route,identity=identity)


def scientific_core(result):
    """Preserve all science, dropping only explicitly prospective identity/capture."""
    out=deepcopy(result)
    out.pop('research',None)
    for name in ('profile_id','profile_sha256','wrapper_id'): out.pop(name,None)
    out['diagnostics'].pop('profile_id',None)
    for phase in out['diagnostics']['phases']: phase.pop('inference',None)
    return out

class _ProjectedAdapter:
    def __init__(self,adapter,arm,evidence,context):
        self._adapter,self._arm,self._record,self._context=adapter,arm,evidence,context
        self.profile=original.profile_for(16 if arm=='reference' else 64)
    def __getattr__(self,name):return getattr(self._adapter,name)
    def reference_choose(self,observer,**kwargs):
        action,trace=self._adapter.reference_choose(observer,**kwargs)
        route=self._context['route'];identity=self._context['execution_identities'][16]
        self._record(dict(event='execution_direct_reference_before_projection',route=route,arm=self._arm,
            observer=observer,action=rules.serialize_action(action),diagnostics=trace,world_pool=3,
            research_phases=deepcopy(getattr(self._adapter,'last_research_phases',None))))
        require(trace['profile_id']==identity['profile_id'],'Raw direct execution identity')
        projected=deepcopy(trace);projected['profile_id']=original.profile_id(16)
        return action,projected


def run_entrypoint_checks(context,request_callback,*,boundary=lambda:None,evidence_callback=lambda row:None):
    route=context['route']
    def routed(payload):
        actual=dict(payload)  # Negative fields reach the real handler unchanged.
        result=request_callback(actual)
        evidence_callback(dict(event='raw_execution_result_before_checker_projection',route=route,request=actual,result=result))
        identity=context['execution_identities'][result['simulations_quota']]
        if route=='research':validate_raw_qualification_annotations(result,actual['observer'],context['freeze'])
        projected=frozen_projection(result,actual['observer'],route,identity=identity)
        evidence_callback(dict(event='explicit_frozen_checker_projection',route=route,
            raw_result_sha256=old.digest(result),projected_result=projected,
            projection='authenticated execution identity only; raw clock never projected'))
        return projected
    proxies={arm:_ProjectedAdapter(adapter,arm,evidence_callback,context) for arm,adapter in context['adapters'].items()}
    summary=old.run_entrypoint_checks(proxies,boundary=boundary,request_callback=routed,evidence_callback=evidence_callback)
    require(summary['counts']['complete_decisions']==52,'Original52 science inventory')
    base=deepcopy(summary) if route=='frozen' else dict(status='research_entrypoint_counterexamples_passed',
        counts=summary['counts'],challenger_quota=64,reference_quota=16,challenger_world_pool=6,reference_world_pool=3)
    base.update(normalized_suite=summary,execution_route=route,execution_schema=adapters.EXECUTION_SCHEMA,
        normalization='authenticated identity-only frozen-checker projection; original raw execution results retained',
        original_checker_unchanged=True,raw_execution_forensics='entrypoint_forensics.jsonl')
    return base


def exact_scientific_outputs(result):
    """Identity-only view retaining every recorded callback output and hash."""
    value=deepcopy(result);value.pop('research',None)
    for key in ('profile_id','profile_sha256','wrapper_id'):value.pop(key,None)
    value['diagnostics'].pop('profile_id',None)
    return value

def _saved(item, cache, boundary):
    boundary()
    key = (item['path'], item['sha256'], item['bytes'])
    saved = cache.setdefault('artifacts', {})
    if key not in saved:
        c.verify([item])
        saved[key] = c.read(c.ROOT/item['path'])
    return saved[key]


def validate_admission_row(row, spec, cache, *, boundary=lambda: None, frozen_entrypoint=None):
    """Exact original scientific gates immediately after saving each response.

    Raw receipts retain the symmetric10s hard guard;700ms/1s are descriptive.
    Raw execution identities are authenticated before any scientific projection.
    """
    boundary()
    from .clock_policy import check_clock
    check_clock(row['clock'],limit=10.)
    c.require(row['spec'] == spec, 'Qualification exact ordered specification')
    clock = row['clock']
    c.require(clock['status'] == 'ok' and clock['accepted'] is True and row.get('completed', True) is True,
              'Qualification request failed; no retry')
    result = clock['result']
    if 'execution_freeze' not in cache:cache['execution_freeze']=c.read(c.OUT/'freeze.json')
    identity=adapters.profile_identity(cache['execution_freeze'],spec['route'],64,boundary=boundary)
    adapters.validate_result(result,spec['observer'],spec['route'],identity=identity)
    c.require(result['observer_sha256'] == spec['observer_sha256'] == c.digest(spec['observer']),
              'Parity observer binding')
    c.require(result['simulations_quota'] == spec['quota'] == 64, 'Parity simulation quota')
    c.require(result['diagnostics']['observer_seed'] == adapters.observer_seed(spec['observer'], spec['seed_label']),
              'Parity observer-only seed')
    ledger.check_observer(spec['before_snapshot'], spec['observer'])
    if spec['route'] == 'research':
        c.require(result['research']['world_pool'] == 6 and result['research']['seed_label'] == 'A' and
                  result['research']['telemetry_enabled'] is False and
                  all(x['telemetry'] is None for x in result['research']['phases']),
                  'Fixed live pool6-A/telemetryOFF')
    late = 0
    if spec['kind'] == 'saved_reproduction':
        prior = _saved(spec['expected_record'], cache, boundary)
        c.require(prior['completed'] and prior['clock']['status'] == 'ok' and prior['observer'] == spec['observer'],
                  'Accepted saved target input/receipt')
        c.require(exact_scientific_outputs(result) == exact_scientific_outputs(prior['clock']['result']),
                  'Live saved action/output/callback reproduction differs')
        if spec['route'] == 'research':
            c.require([p['callback_bits'] for p in result['research']['phases']] ==
                      [p['callback_bits'] for p in prior['clock']['result']['research']['phases']],
                      'Live callback float32 bits differ from accepted sampling')
            category = 'candidate_saved'
        else:
            category = 'frozen_saved'
    elif spec['kind'] == 'live_edge':
        previous = _saved(spec['expected_entrypoint'], cache, boundary)
        c.require(previous['status'] == 'ok', 'Accepted edge source receipt')
        expected = next(x for x in previous['result']['normalized_suite']['fixtures']
                        if x['id'] == spec['expected_case'])['results']['search']
        c.require(exact_scientific_outputs(result) == exact_scientific_outputs(expected),
                  'Live edge differs from accepted exact science')
        if spec['exact_frozen_shortcut']:
            if frozen_entrypoint is None:
                frozen_entrypoint = c.read(c.OUT/'entrypoint_checks/frozen.json')
            c.require(frozen_entrypoint['status'] == 'ok', 'Current unchanged64 suite receipt')
            baseline = next(x for x in frozen_entrypoint['result']['fixtures']
                            if x['id'] == spec['expected_case'])['results']['search']
            c.require(exact_scientific_outputs(result) == exact_scientific_outputs(baseline),
                      'Late heuristic changed from frozen64')
            c.require(result['diagnostics']['inference_calls'] == 0 and
                      all(p['heuristic_phase'] and p['simulations'] == p['leaf_calls'] == 0
                          for p in result['diagnostics']['phases']),
                      'Late heuristic unexpectedly searched')
            late = 1
        category = 'live_edges'
    elif spec['kind'] == 'fresh_reserved':
        category = 'fresh_reserved'
    else:
        raise ValueError('Unknown live qualification kind')
    return dict(status='passed', category=category, exact_frozen_late_shortcuts=late,
                model_calls=0, no_retries=True)


def validate_parity_rows(rows, plan, *, boundary=lambda: None, frozen_entrypoint=None, allow_prefix=False):
    expected = plan['parity_requests']
    c.require(len(expected) == 90 and len(rows) <= 90, 'Finite90 live qualification request bound')
    if not allow_prefix:
        c.require(len(rows) == 90, 'Finite90 live qualification request coverage')
    counts = dict(frozen_saved=0, candidate_saved=0, live_edges=0, fresh_reserved=0,
                  exact_frozen_late_shortcuts=0)
    cache, failures = {}, []
    successful = 0
    for index, row in enumerate(rows):
        spec = expected[index]
        c.require(row['spec'] == spec, 'Qualification exact ordered specification')
        clock = row['clock']
        from .clock_policy import check_clock
        check_clock(clock,limit=10.)
        if clock['status'] != 'ok' or clock['accepted'] is not True:
            c.require(allow_prefix and index == len(rows)-1 and clock['accepted'] is False and
                      row.get('completed', False) is False,
                      'Only final failed prefix request; never continue or invent successful science')
            failures.append(dict(index=index, status=clock['status'], detail=clock.get('detail')))
            continue
        proof = validate_admission_row(row, spec, cache, boundary=boundary, frozen_entrypoint=frozen_entrypoint)
        counts[proof['category']] += 1
        counts['exact_frozen_late_shortcuts'] += proof['exact_frozen_late_shortcuts']
        successful += 1
    complete = len(rows) == 90 and not failures
    if complete:
        c.require(counts == dict(frozen_saved=24, candidate_saved=24, live_edges=6, fresh_reserved=36,
                                exact_frozen_late_shortcuts=1), 'All mandatory fixed live qualification cells')
    return dict(status='passed' if complete else 'verified_prefix', complete=complete,
        complete_decisions=successful, attempted=len(rows), failures=failures, counts=counts,
        exact_recorded_numeric_outputs=True, exact_candidate_callback_bits=True,
        model_calls=0, no_retries=True)


def ordinary_payload(spec,observer):
    return adapters.ordinary_payload(spec,observer)

def validate_raw_qualification_annotations(result,observer,freeze):
    """Bind retained new-route identity before the labelled old-checker view."""
    q=result['simulations_quota'];research=result['research']
    require(type(q) is int and q in (16,64),'Qualification annotation quota')
    fields={'schema','quota','world_pool','seed_label','seed','telemetry_enabled','internal_seconds','native_binary_sha256','phases'}
    require(type(research) is dict and set(research)==fields and research['schema']==adapters.SCHEMA and
        type(research['quota']) is int and research['quota']==q and research['seed_label']=='A' and
        type(research['world_pool']) is int and research['world_pool']==(3 if q==16 else 6) and
        type(research['seed']) is int and research['seed']==adapters.observer_seed(observer,'A') and
        research['seed']==result['diagnostics']['observer_seed'] and research['telemetry_enabled'] is False and
        research['internal_seconds']==9. and research['native_binary_sha256']==freeze['research_native_sha256'],
        'Raw qualification research identity/clock/native/capture binding')
    _validate_qualification_phases(result['diagnostics'],research['phases'],observer,q)

def _validate_qualification_phases(trace,annotations,observer,quota):
    from tools.splendorust_sampling_stability_20261027 import audit as scientific
    from tools.external_strength_screen_20261018_02 import bridge
    phases=trace['phases']
    require(type(annotations) is list and len(annotations)==len(phases)<=4,'Raw qualification phase inventory')
    cursor=0
    def select(native,choices):
        nonlocal cursor
        require(cursor<len(phases),'Raw qualification missing phase')
        phase,annotation=phases[cursor],annotations[cursor]
        require(type(annotation) is dict and set(annotation)=={'phase_index','phase_kind','telemetry','callback_bits'} and
            type(annotation['phase_index']) is int and annotation['phase_index']==cursor and
            annotation['phase_kind']==native['phase']['kind'],'Raw qualification phase identity')
        raw83=scientific.validate_callback_capture(phase,annotation['callback_bits'])
        require(annotation['telemetry'] is None,'Live qualification telemetry must remain OFF')
        cursor+=1
        return phase['action']
    main=bridge.main_actions(observer)
    selected=main[0] if main==[{'kind':'Pass'}] else select(bridge.encode_observation(observer),main)
    bridge.resolve_compound(observer,selected,select)
    require(cursor==len(phases),'Raw qualification unconsumed phase')

def verify_entrypoint(folder,receipt,plan,freeze,boundary=lambda:None):
    """Read back finite raw forensics; no model, process or checker rerun."""
    import math
    from collections import Counter
    from tools.external_strength_screen_20261018 import audit as ledger
    from tools.external_strength_screen_20261018_02 import bridge
    from tools.future_training_rules_20261021 import decode_training_snapshot
    folder=Path(folder);boundary()
    require(receipt['status']=='ok','Entrypoint successful original receipt required')
    packet=receipt['result'];route=packet['execution_route'];research=route=='research'
    require(route in ('frozen','research') and packet['execution_schema']==adapters.EXECUTION_SCHEMA,'New suite execution identity')
    require(packet['status']==('research_entrypoint_counterexamples_passed' if research else 'entrypoint_counterexamples_passed'),'Entrypoint route status')
    if research:
        require(packet['challenger_world_pool']==6 and packet['reference_world_pool']==3,'Actual suite pool bindings')
    suite=packet['normalized_suite']
    require(suite['status']=='entrypoint_counterexamples_passed' and suite['challenger_quota']==64 and
        suite['reference_quota']==16 and suite['counts']=={'complete_decisions':52},'Exact original suite coverage')
    path=folder/'entrypoint_forensics.jsonl'
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
    raw=[e for e in events if e.get('event')=='raw_execution_result_before_checker_projection']
    projected=[e for e in events if e.get('event')=='explicit_frozen_checker_projection']
    raw_direct=[e for e in events if e.get('event')=='execution_direct_reference_before_projection']
    completed=[e for e in events if e.get('event')==('complete_research_result_before_worker_validation' if research else 'complete_result_before_worker_validation')]
    require(len(raw)==len(projected)==len(completed)==37 and len(raw_direct)==15,'All raw execution identities retained')
    identities={q:adapters.profile_identity(freeze,route,q,boundary=boundary) for q in (16,64)}
    for raw_event,projected_event,event,complete in zip(raw,projected,ordinary,completed,strict=True):
        result=raw_event['result'];q=result['simulations_quota'];observer=raw_event['request']['observer']
        require(raw_event['route']==projected_event['route']==route and
            raw_event['request']==dict(player=result['player'],observer=event['request']['observer']), 'Raw execution request binding')
        if research:validate_raw_qualification_annotations(result,observer,freeze)
        normalized=frozen_projection(result,observer,route,identity=identities[q])
        require(projected_event['raw_result_sha256']==old.digest(result) and normalized==projected_event['projected_result']==event['result'],
            'Exact disclosed identity-only projection')
        require(complete['observer']==observer and complete['arm']==result['player'] and
            complete['action']==result['action'] and complete['diagnostics']==result['diagnostics'], 'Completed prevalidation raw science')
        if research:require(complete['research']==result['research'],'Raw completed candidate annotation')
    for event,normalized in zip(raw_direct,direct,strict=True):
        trace=deepcopy(event['diagnostics'])
        require(event['route']==route and trace['profile_id']==identities[16]['profile_id'] and event['world_pool']==3 and
            event['observer']==normalized['request']['observer'],'Raw direct16 execution identity')
        if research:_validate_qualification_phases(trace,event['research_phases'],event['observer'],16)
        else:require(event['research_phases'] is None,'Baseline direct reference has no candidate annotation')
        trace['profile_id']=original.profile_id(16)
        require(trace==normalized['result']['diagnostics'] and event['action']==normalized['result']['action'],'Exact direct reference projection')
    raw_count=len(raw)+len(raw_direct)
    return dict(status='passed',route='research' if research else 'frozen',complete_decisions=52,
        ordinary_handlers=37,direct_references=15,negative_handler_cases=26,actual_encoded_cost_checks=30,
        native_ordered_supports=28,native_numeric_parity=46,independent_saved_transitions=37,
        raw_execution_decisions_bound=raw_count,forensic_events=len(events),forensic_terminal_sha256=previous,
        forensic_artifact=old.artifact(path),model_calls=0,qualification_rerun=False)

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
from . import adapters

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT/'outputs/decision_quality_design_20261027_01/prepared_request_matrix.json'
PANEL = ROOT/'outputs/decision_quality_preparation_20261027_01/panel.json'

def require(value,message):
    if not value: raise ValueError('DECISION_QUALIFICATION: '+message)

def build_plan():
    matrix=json.loads(MATRIX.read_text(encoding='utf-8'))
    panel=json.loads(PANEL.read_text(encoding='utf-8'))
    pilot=matrix['pilot_roots'][0]
    if isinstance(pilot,dict): pilot=pilot['root_id']
    root=next(x for x in panel['roots'] if x['root_id']==pilot)
    observer=root['observer']
    parity=[dict(index=0,route='frozen',quota=64,seed_label='A',telemetry=False,
                 root_id=pilot,observer=observer,observer_sha256=old.digest(observer))]
    for quota in adapters.QUOTAS:
        for label in ('A','B','C'):
            for capture in (False,True):
                parity.append(dict(index=len(parity),route='research',quota=quota,seed_label=label,
                    telemetry=capture,root_id=pilot,observer=observer,observer_sha256=old.digest(observer)))
    return dict(schema='saved_decision_quality_qualification_plan_v1',
        graph_requests=[dict(route=r,fixtures=11) for r in ('frozen','research')],
        entrypoint_requests=[dict(route=r,complete_decisions=52) for r in ('frozen','research')],
        parity_requests=parity,complete_decisions=123,graph_cases=22,global_graph_cases=44,
        maximum_initialized_workers=4,maximum_simultaneous_worker_trees=1,
        worker_lifecycle=['frozen_qualification','research_qualification','frozen_saved','research_saved'],
        stage_seconds=900,entrypoint_external_seconds=300,graph_external_seconds=60,
        ordinary_frozen_external_seconds=1,ordinary_research_external_seconds=30,
        original_suite=old.build_plan(64),
        normalization=dict(raw_evidence_preserved_before_projection=True,
            removed_envelope_fields=['research'],
            identity_replacements=['profile_id','profile_sha256','wrapper_id','diagnostics.profile_id'],
            fields_not_normalized=['action','quota','observer_seed','callback hashes','work counts','numeric values']),
        fixture_source=old.artifact(PANEL),request_matrix=old.artifact(MATRIX),
        no_retries=True,no_replacements=True,model_execution_during_preparation=False)

def frozen_projection(result):
    """Only valid for quota64/16-A; never used to relabel raw stored evidence."""
    q=result['simulations_quota']
    require(q in (16,64),'Projection only qualification reference16/research64')
    research=result['research']
    require(research['seed_label']=='A' and research['seed']==result['diagnostics']['observer_seed'],
            'Projection canonical seed binding')
    projected={key:deepcopy(result[key]) for key in original_audit.RESULT_FIELDS}
    projected.update(profile_id=original.profile_id(q),profile_sha256=original.profile_sha256(q),
                     wrapper_id=original.wrapper_id(q))
    projected['diagnostics']['profile_id']=original.profile_id(q)
    return projected

def scientific_core(result):
    """Preserve all science, dropping only explicitly prospective identity/capture."""
    out=deepcopy(result)
    out.pop('research',None)
    for name in ('profile_id','profile_sha256','wrapper_id'): out.pop(name,None)
    out['diagnostics'].pop('profile_id',None)
    for phase in out['diagnostics']['phases']: phase.pop('inference',None)
    return out

class _ProjectedAdapter:
    def __init__(self,adapter,arm,evidence):
        self._adapter,self._arm,self._record=adapter,arm,evidence
        self.profile=original.profile_for(16 if arm=='reference' else 64)
    def __getattr__(self,name): return getattr(self._adapter,name)
    def reference_choose(self,observer,**kwargs):
        action,trace=self._adapter.reference_choose(observer,**kwargs)
        self._record(dict(event='research_direct_reference_before_projection',arm=self._arm,
            observer=observer,action=rules.serialize_action(action),diagnostics=trace,
            research_phases=deepcopy(self._adapter.last_research_phases)))
        projected=deepcopy(trace);projected['profile_id']=original.profile_id(16)
        return action,projected

def run_entrypoint_checks(context,request_callback,*,boundary=lambda:None,evidence_callback=lambda row:None):
    def routed(payload):
        # Keep malformed fields from the original negative cases: the actual
        # new allowlist, decoder and guard must reject them, not this wrapper.
        actual=dict(payload,quota=16 if payload.get('player')=='reference' else 64,
                    seed_label='A',telemetry=True)
        result=request_callback(actual)
        evidence_callback(dict(event='raw_research_result_before_checker_projection',
            request=actual,result=result))
        projected=frozen_projection(result)
        evidence_callback(dict(event='explicit_frozen_checker_projection',
            raw_result_sha256=old.digest(result),projected_result=projected,
            projection='research64/16-A identity/capture only; not raw player identity'))
        return projected
    proxies={arm:_ProjectedAdapter(adapter,arm,evidence_callback) for arm,adapter in context['adapters'].items()}
    summary=old.run_entrypoint_checks(proxies,boundary=boundary,request_callback=routed,
                                      evidence_callback=evidence_callback)
    require(summary['counts']['complete_decisions']==52,'Original52 science inventory')
    return dict(status='research_entrypoint_counterexamples_passed',counts=summary['counts'],
        challenger_quota=64,reference_quota=16,normalized_suite=summary,
        normalization='explicit identity-only frozen-checker projection, original raw research results retained',
        original_checker_unchanged=True,raw_research_forensics='entrypoint_forensics.jsonl')

def validate_parity_rows(rows,plan):
    require(len(rows)==len(plan['parity_requests'])==19,'Finite19 parity request coverage')
    by={}
    for row,spec in zip(rows,plan['parity_requests']):
        require(row['spec']==spec,'Qualification exact ordered specification')
        clock=row['clock'];require(clock['status']=='ok','Qualification request failed; no retry')
        result=clock['result'];require(result['observer_sha256']==spec['observer_sha256'],'Parity observer binding')
        require(result['simulations_quota']==spec['quota'],'Parity simulation quota')
        require(result['diagnostics']['observer_seed']==adapters.observer_seed(spec['observer'],spec['seed_label']),
                'Parity observer-only seed')
        by[spec['route'],spec['quota'],spec['seed_label'],spec['telemetry']]=result
    checks=[]
    for q in adapters.QUOTAS:
        for label in ('A','B','C'):
            off,on=(by['research',q,label,flag] for flag in (False,True))
            require(scientific_core(off)==scientific_core(on),'Telemetry changed full science/callback stream')
            require([p['callback_bits'] for p in off['research']['phases']]==
                    [p['callback_bits'] for p in on['research']['phases']],'Telemetry changed float32 callback bits')
            require(all(p['telemetry'] is None for p in off['research']['phases']),'Disabled telemetry unexpectedly present')
            checks.append(dict(quota=q,seed_label=label,exact_scientific_core=True,exact_callback_float32_bits=True))
    frozen=by['frozen',64,'A',False]
    research=by['research',64,'A',True]
    require(scientific_core(frozen)==scientific_core(research),'Research64-A differs from exact frozen64 science')
    return dict(status='passed',complete_decisions=19,telemetry_comparisons=checks,
                frozen64_research64_exact=True,main_root=plan['parity_requests'][0]['root_id'])

def ordinary_payload(spec,observer):
    if spec['route']=='frozen': return dict(player='search',observer=observer)
    require(spec['quota'] in adapters.QUOTAS,'Ordinary research cannot use qualification-only16')
    return dict(player='search',observer=observer,quota=spec['quota'],seed_label=spec['seed_label'],
                telemetry=spec.get('telemetry',True))

def validate_raw_qualification_annotations(result,observer,freeze):
    """Bind retained new-route identity before the labelled old-checker view."""
    q=result['simulations_quota'];research=result['research']
    require(type(q) is int and q in (16,64),'Qualification annotation quota')
    fields={'schema','quota','seed_label','seed','telemetry_enabled','internal_seconds','native_binary_sha256','phases'}
    require(type(research) is dict and set(research)==fields and research['schema']==adapters.SCHEMA and
        type(research['quota']) is int and research['quota']==q and research['seed_label']=='A' and
        type(research['seed']) is int and research['seed']==adapters.observer_seed(observer,'A') and
        research['seed']==result['diagnostics']['observer_seed'] and research['telemetry_enabled'] is True and
        research['internal_seconds']==29. and research['native_binary_sha256']==freeze['research_native_sha256'],
        'Raw qualification research identity/clock/native/capture binding')
    _validate_qualification_phases(result['diagnostics'],research['phases'],observer,q)

def _validate_qualification_phases(trace,annotations,observer,quota):
    from . import audit as scientific
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
        if phase['heuristic_phase']:
            require(annotation['telemetry'] is None,'Heuristic qualification has no search targets')
        else:
            scientific.validate_root_statistics(annotation['telemetry'],native,choices,phase['action'],quota,raw83)
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
    packet=receipt['result'];research=packet['status']=='research_entrypoint_counterexamples_passed'
    require(research or packet['status']=='entrypoint_counterexamples_passed','Entrypoint route status')
    suite=packet['normalized_suite'] if research else packet
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
    raw_count=0
    if research:
        raw=[e for e in events if e.get('event')=='raw_research_result_before_checker_projection']
        projected=[e for e in events if e.get('event')=='explicit_frozen_checker_projection']
        raw_direct=[e for e in events if e.get('event')=='research_direct_reference_before_projection']
        require(len(raw)==len(projected)==37 and len(raw_direct)==15,'All raw new identities retained before projection')
        for raw_event,projected_event,event in zip(raw,projected,ordinary):
            result=raw_event['result'];q=result['simulations_quota']
            validate_raw_qualification_annotations(result,raw_event['request']['observer'],freeze)
            require(result['profile_sha256']==freeze['research_profiles'][str(q)]['sha256'] and
                result['profile_id']==adapters.profile_id(q) and result['wrapper_id']==adapters.wrapper_id(q),
                'Raw research qualification identity binding')
            require(projected_event['raw_result_sha256']==old.digest(result) and
                frozen_projection(result)==projected_event['projected_result']==event['result'],
                'Exact disclosed identity-only projection')
        for event,normalized in zip(raw_direct,direct):
            trace=deepcopy(event['diagnostics']);require(trace['profile_id']==adapters.profile_id(16),'Raw direct research16 identity')
            _validate_qualification_phases(trace,event['research_phases'],event['observer'],16)
            trace['profile_id']=original.profile_id(16)
            require(trace==normalized['result']['diagnostics'] and event['action']==normalized['result']['action'],
                    'Exact direct reference projection')
        raw_count=len(raw)+len(raw_direct)
    return dict(status='passed',route='research' if research else 'frozen',complete_decisions=52,
        ordinary_handlers=37,direct_references=15,negative_handler_cases=26,actual_encoded_cost_checks=30,
        native_ordered_supports=28,native_numeric_parity=46,independent_saved_transitions=37,
        raw_research_decisions_bound=raw_count,forensic_events=len(events),forensic_terminal_sha256=previous,
        forensic_artifact=old.artifact(path),model_calls=0,qualification_rerun=False)

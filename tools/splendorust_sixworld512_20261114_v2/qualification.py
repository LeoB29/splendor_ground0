"""Actual live-entrypoint qualification and independent saved-evidence checks."""
from copy import deepcopy
import gzip
import json
import zlib
from pathlib import Path
from . import common as c, adapters
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.splendorust_teacher_variability_20261113 import qualification as prior


def payload(observer,pool,*,player='search',quota=512,telemetry=True):
    return dict(player=player,observer=observer,quota=quota,pool=pool,mode='legacy',
        world_seed_label='A',encoding_seed_label='A',telemetry=telemetry)


def validate_live_result(result,observer,pool,plan):
    c.require(type(pool) is int and pool in (3,6),'Declared pool3 or6')
    key=adapters.profile_key(512,pool,'legacy')
    item=plan['research_profiles'][key];document=c.read(c.resolve(item))
    return prior.validate_result(result,observer,quota=512,pool=pool,mode='legacy',
        world_seed_label='A',encoding_seed_label='A',profile_id=document['profile_id'],
        profile_sha256=item['sha256'],wrapper_id=document['wrapper_id'],
        native_binary_sha256=document['native_binary_sha256'],telemetry_required=True,player='search')


def build_plan():
    """Two fixed development roots, chosen by identity hash, plus all old cases."""
    c.require(c.sha(c.PRIOR_FREEZE)==c.PRIOR_FREEZE_SHA and c.sha(c.PRIOR_CLOSEOUT)==c.PRIOR_CLOSEOUT_SHA,
        'Accepted unchanged diagnostic and closeout')
    original=c.read(c.PRIOR_FREEZE)
    q=c.read(c.resolve(original['qualification_plan']))
    panel=c.read(c.resolve(original['panel']))
    boundary_path=c.ROOT/'outputs/splendorust_teacher_variability_crash_review_20261113_01/crash_boundary.json'
    recovered=c.ROOT/'outputs/splendorust_teacher_variability_recovery_20261113_01'
    acceptance=c.read(recovered/'jobs/acceptance/worker/independent_acceptance.json')
    c.require(acceptance['approved'] and acceptance['saved_evidence']['ordinary']['requests']==3072,
        'Only accepted complete prior decisions supply exact expectations')
    boundary=c.read(boundary_path)
    rows=[dict(spec=r['spec'],receipt=r['receipt']) for r in boundary['prefix_rows']]
    rows+=c.read(recovered/'jobs/diagnostic/worker/index.json')['requests']
    c.require(len(rows)==3072,'Original/recovered source union')
    eligible={r['spec']['root_id'] for r in rows if r['spec']['quota']==512}
    roots=sorted((r for r in panel['roots'] if r['root_id'] in eligible),
        key=lambda r:c.digest(['sixworld512_live_qualification',r['root_id']]))[:2]
    c.require(len(roots)==2,'Two existing512 development roots')
    selected=[]
    for root in roots:
        expected={}
        for pool in (3,6):
            matches=[r for r in rows if r['spec']['root_id']==root['root_id'] and r['spec']['quota']==512
                and r['spec']['pool']==pool and r['spec']['mode']=='legacy'
                and r['spec']['world_seed_label']==r['spec']['encoding_seed_label']=='A']
            c.require(len(matches)==1,'Exact canonical prior condition')
            expected[str(pool)]=matches[0]['receipt'];c.verify([matches[0]['receipt']])
        selected.append(dict(root_id=root['root_id'],observer=root['observer'],expected=expected))
    edge=[]
    for name in ('forced_pass','turn123','turn124'):
        row=next(r for r in q['original_suite']['counterexamples'] if r['id']==name)
        state=guard.decode_training_snapshot(row['snapshot'])
        edge.append(dict(case_id=name,snapshot=row['snapshot'],observer=rules.observation_payload(state)))
    return dict(schema='sixworld512_live_qualification_v1',original_suite=q['original_suite'],
        entrypoint_routes=[dict(mode='legacy',pool=p) for p in (3,6)],saved_parity_roots=selected,
        edge_cases=edge,complete_decisions=120,corrected_entrypoint_decisions=104,
        ordinary_live_decisions=10,qualification_decisions=6,graph_cases=11,
        live_late_and_forcedpass_route_without_qualification_bypass=True,
        source_bindings=[c.artifact(c.PRIOR_FREEZE),c.artifact(c.PRIOR_CLOSEOUT),original['panel'],
            original['qualification_plan'],c.artifact(boundary_path),
            c.artifact(recovered/'jobs/diagnostic/worker/index.json'),
            c.artifact(recovered/'jobs/acceptance/worker/independent_acceptance.json')])


def _read_bound_call(row):
    c.require(type(row) is dict and set(row)=={'payload','source'},'Saved qualification call binding')
    c.verify([row['source']]);raw=c.read(c.resolve(row['source']))
    clock=deepcopy(raw)
    if raw['result'] is not None:clock['result']=adapters.unpack_result(raw['result'])
    return clock


def _validate_request_result(result,request,plan):
    control=request['player']=='control';key=adapters.profile_key(request['quota'],request['pool'],'legacy')
    item=plan['control_profiles' if control else 'research_profiles'][key]
    document=c.read(c.resolve(item))
    return prior.validate_result(result,request['observer'],quota=request['quota'],pool=request['pool'],mode='legacy',
        world_seed_label='A',encoding_seed_label='A',profile_id=document['profile_id'],profile_sha256=item['sha256'],
        wrapper_id=document['wrapper_id'],native_binary_sha256=document['native_binary_sha256'],
        telemetry_required=request['telemetry'],player=request['player'])


def verify_live_qualification(rows,plan,*,service_folder,boundary=lambda:None):
    """Read-only counterpart of every actual handler request; no models executed."""
    q=c.read(c.resolve(plan['qualification_plan']));c.require(len(rows)==18,'Two suites plus16 explicit requests')
    cursor=0;entry=[];fixed=[];edges=[];expected_requests=[]
    def take(request,limit=120):
        nonlocal cursor
        boundary();row=rows[cursor];cursor+=1
        c.require(row['payload']==request,'Exact prospective qualification request order')
        clock=_read_bound_call(row);prior.check_clock(clock,limit=limit)
        c.require(clock['status']=='ok' and clock['accepted'] and clock['received_before_deadline'],'Accepted complete qualification response')
        expected_requests.append(dict(request_id=clock['request_id'],payload=request,result=clock['result']))
        return clock
    for pool in (3,6):
        request=dict(operation='entrypoint_checks',mode='legacy',pool=pool)
        clock=take(request,6000)
        entry.append(prior.verify_entrypoint(service_folder,clock,q,plan,boundary,mode='legacy',pool=pool))
    for root in q['saved_parity_roots']:
        current={}
        for pool in (3,6):
            request=payload(root['observer'],pool);clock=take(request)
            validate_live_result(clock['result'],root['observer'],pool,plan)
            expected=c.read(c.resolve(root['expected'][str(pool)]))['result']
            c.verify([root['expected'][str(pool)]])
            c.require(c.canonical(clock['result'])==c.canonical(expected),'Complete archived canonical result reproduced exactly')
            current[pool]=clock['result']
        request=payload(root['observer'],3,player='control')
        clock=take(dict(operation='qualification_decision',request=request))
        _validate_request_result(clock['result'],request,plan)
        original=clock['result'];left,right=prior.scientific_core(original),prior.scientific_core(current[3])
        left.pop('player');right.pop('player')
        c.require(left==right,'Live pool3 equals unchanged original native science')
        c.require([{k:p[k] for k in ('telemetry','callback_bits')} for p in original['research']['phases']]
            ==[{k:p[k] for k in ('telemetry','callback_bits')} for p in current[3]['research']['phases']],
            'Exact original native callback bits and root statistics')
        for pool in (3,6):
            request=payload(root['observer'],pool,telemetry=False)
            off=take(dict(operation='qualification_decision',request=request))['result']
            _validate_request_result(off,request,plan)
            c.require(prior.scientific_core(off)==prior.scientific_core(current[pool]),'Telemetry leaves complete science unchanged')
            for key in ('callback_bits','sampling'):
                c.require([p[key] for p in off['research']['phases']]==[p[key] for p in current[pool]['research']['phases']],
                    'Telemetry leaves callback bits and sampled worlds unchanged')
        fixed.append(dict(root_id=root['root_id'],archived_exact_pools=[3,6],original_native_exact=True,telemetry_invariant=True))
    for edge in q['edge_cases']:
        for pool in (3,6):
            clock=take(payload(edge['observer'],pool))
            checked=validate_live_result(clock['result'],edge['observer'],pool,plan)
            if edge['case_id']=='forced_pass':
                c.require(clock['result']['diagnostics']['forced_pass'] and checked['simulations']==0,'Actual live forced pass')
            if edge['case_id']=='turn124':c.require(checked['simulations']==checked['neural_callbacks']==0,'Actual live late native shortcut')
            if edge['case_id']=='turn123':c.require(checked['simulations']==512 and checked['neural_callbacks']==1,'Actual live final searched root')
            edges.append(dict(case_id=edge['case_id'],pool=pool,checked=checked))
    c.require(cursor==len(rows),'Complete qualification inventory')
    return dict(status='passed',complete_decisions=120,entrypoint=entry,fixed_parity=fixed,live_edge_cases=edges,
        graph_fixtures=11,qualification_requests=18,no_models_executed_by_saved_verifier=True,
        expected_forensic_requests=[r for r in expected_requests if r['payload'].get('operation')!='entrypoint_checks'])


def run_live_qualification(call_fn,plan,*,service_folder,boundary=lambda:None):
    """Caller has already passed fresh service graph fixtures before this call."""
    q=c.read(c.resolve(plan['qualification_plan']));rows=[]
    def call(request,timeout=120):
        boundary();value=call_fn(request,timeout)
        c.require(type(value) is dict and 'clock' in value and 'source' in value,'Owning clock/source call interface')
        rows.append(dict(payload=deepcopy(request),source=value['source']))
        raw=_read_bound_call(rows[-1])
        c.require(c.canonical(raw)==c.canonical(value['clock']),'Decoded complete response binds packed source')
        prior.check_clock(raw,limit=timeout)
        c.require(raw['status']=='ok' and raw['accepted'] and raw['received_before_deadline'],'Qualification failure preserved; no retry')
    for pool in (3,6):call(dict(operation='entrypoint_checks',mode='legacy',pool=pool),6000)
    for root in q['saved_parity_roots']:
        for pool in (3,6):call(payload(root['observer'],pool))
        call(dict(operation='qualification_decision',request=payload(root['observer'],3,player='control')))
        for pool in (3,6):call(dict(operation='qualification_decision',request=payload(root['observer'],pool,telemetry=False)))
    for edge in q['edge_cases']:
        for pool in (3,6):call(payload(edge['observer'],pool))
    proof=verify_live_qualification(rows,plan,service_folder=service_folder,boundary=boundary)
    return dict(status='passed',requests=rows,verification=proof)


def _forensic_lines(path,state,allow_truncated,boundary):
    """Bounded single-member reader; a killed owner's unaccepted tail is explicit."""
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS);buffer=b''
    with Path(path).open('rb') as stream:
        while True:
            chunk=stream.read(65536)
            if not chunk:break
            pending=chunk
            while pending:
                boundary();buffer+=decoder.decompress(pending,adapters.MAX_RAW_BYTES+1-len(buffer))
                pending=decoder.unconsumed_tail
                c.require(not decoder.unused_data,'Single forensic gzip member; no appended bytes')
                while b'\n' in buffer:
                    line,buffer=buffer.split(b'\n',1)
                    c.require(len(line)+1<=adapters.MAX_RAW_BYTES,'Finite forensic row')
                    yield line+b'\n'
                c.require(len(buffer)<=adapters.MAX_RAW_BYTES,'Finite partial forensic row')
    state.update(gzip_complete=decoder.eof,partial_row_bytes=len(buffer))
    c.require(allow_truncated or (decoder.eof and not buffer),'Complete forensic gzip and row framing')


def verify_service_forensics(service_folder,expected,*,boundary=lambda:None,failed_request=None):
    """Stream successful results; preserve and separately label a failed tail."""
    folder=Path(service_folder);summary_path=folder/'forensic_summary.json'
    summary=c.read(summary_path) if summary_path.exists() else None
    item=c.artifact(folder/'ordinary_forensics.jsonl.gz')
    if summary is not None:
        c.require(summary['artifact']==item and summary['lossless'] is True,'Exact lossless forensic artifact')
    else:c.require(failed_request is not None,'Clean service requires final forensic summary')
    if failed_request is not None:
        c.require(set(failed_request)=={'request_id','payload'} and type(failed_request['request_id']) is str,
            'Bound final failed request')
    iterator=iter(r for r in expected if r['payload'].get('operation') in (None,'qualification_decision'))
    previous=None;count=0;accepted=0;tail=[];state={}
    for line in _forensic_lines(folder/'ordinary_forensics.jsonl.gz',state,failed_request is not None,boundary):
        boundary();row=json.loads(line,object_pairs_hook=c._unique,
            parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite forensic JSON')))
        c.require(set(row)=={'sequence','previous_sha256','event'} and type(row['sequence']) is int
            and row['sequence']==count and row['previous_sha256']==previous,'Independent forensic sequence/hash chain')
        c.require(c.canonical(row).encode('utf-8')+b'\n'==line,'Canonical full forensic row')
        packet=row['event'];c.require(set(packet)=={'transport_request_id','event'},'Exact forensic packet')
        event=packet['event'];wanted=next(iterator,None)
        if wanted is not None:
            request=wanted['payload'].get('request',wanted['payload']);result=wanted['result']
            c.require(packet['transport_request_id']==wanted['request_id'] and event==dict(
                event='complete_result_before_worker_validation',arm=request['player'],observer=request['observer'],
                action=result['action'],diagnostics=result['diagnostics'],research=result['research']),
                'Full prevalidation observer/action/callback/world evidence binds returned result')
            accepted+=1
        else:
            c.require(failed_request is not None and len(tail)<2,'No unsolicited forensic row')
            request=failed_request['payload'].get('request',failed_request['payload'])
            kind=event.get('event')
            c.require(packet['transport_request_id']==failed_request['request_id'] and event.get('arm')==request['player']
                and event.get('observer')==request['observer'],'Failed tail belongs only to final failed request')
            c.require(kind in ('complete_result_before_worker_validation','worker_research_decision_failed')
                and kind not in tail and (not tail or kind=='worker_research_decision_failed'),'Known ordered failed forensic tail')
            tail.append(kind)
        previous=c.digest(row);count+=1
    c.require(next(iterator,None) is None,'Every accepted result has complete forensic evidence')
    if summary is not None:
        c.require(state['gzip_complete'] and not state['partial_row_bytes'] and count==summary['events']
            and previous==summary['terminal_sha256'],'Complete terminal forensic chain')
    return dict(status='passed',events=count,accepted_events=accepted,terminal_sha256=previous,artifact=item,
        lossless=True,gzip_complete=state['gzip_complete'],partial_row_bytes=state['partial_row_bytes'],
        failed_tail_events=tail,failed_request_preserved=failed_request is not None,
        failed_tail_is_not_an_accepted_decision=True,summary_present=summary is not None)

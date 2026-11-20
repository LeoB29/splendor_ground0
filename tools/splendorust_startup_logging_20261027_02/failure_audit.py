"""Read-only, nonqualifying audit of a failed original startup-handler prefix.

This never makes a worker request or edits old evidence. Its deliberately finite
contract is the frozen64 graph worker's entrypoint counterexample prefix. Other
failure shapes are rejected, not silently promoted to completed qualification.
Callers authenticate their evaluator runtime once at process startup and enforce
their existing global audit boundary through ``boundary``.
"""
from collections import Counter
from datetime import datetime
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/'outputs/splendorust_pool6_core_compare_20261027_02'
PRIOR_FREEZE_SHA = '92e35152b24276baaa676572b8891b723ae64eec4c4320be8389147eb5915745'
PRIOR_REVIEW_SHA = '4a5cdd7aa91e92f10bebf70621b990bead9ad2afb31d524ff268c933e5131cb0'
PRIOR_INVENTORY_SHA = '192c8d834bda33f00e76ab3b6f03f86787b3037460a75cb22b092fa86bc8273e'
MiB = 1024**2


def require(condition, message):
    if not condition:
        raise ValueError('Failed-startup forensic audit: '+message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Reader:
    """Bounded immutable reads, with before/after authentication of every input."""
    def __init__(self,root,boundary):
        self.root=Path(root).resolve();self.boundary=boundary;self.items={}

    def bytes(self,path,maximum=64*MiB):
        self.boundary();path=Path(path).resolve()
        require(path.is_file() and path.stat().st_size<=maximum,'bounded input '+str(path))
        value=path.read_bytes();require(len(value)<=maximum,'input grew beyond bound')
        key=path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else str(path)
        item=dict(path=key,bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
        require(key not in self.items or self.items[key]==item,'input changed during audit')
        self.items[key]=item;return value

    def json(self,path,maximum=64*MiB):
        return json.loads(self.bytes(path,maximum))

    def artifact(self,item):
        path=Path(item['path'])
        require(not path.is_absolute() and '..' not in path.parts,'project artifact path')
        value=self.bytes(self.root/path)
        require(len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256'],
                'artifact hash differs: '+item['path'])
        return value

    def finish(self):
        before=list(self.items.values())
        for item in before:
            path=Path(item['path']);path=path if path.is_absolute() else self.root/path
            self.bytes(path)
        return sorted(self.items.values(),key=lambda item:item['path'])


def _absence(identities,boundary):
    seen=set()
    for identity in identities:
        boundary();key=(identity['pid'],identity['created'])
        require(type(key[0]) is int and key[0]>0 and type(key[1]) in (int,float) and
                math.isfinite(key[1]) and key[1]>0,'valid owned identity')
        if key in seen:continue
        seen.add(key)
        try:process=psutil.Process(key[0])
        except psutil.NoSuchProcess:continue
        require(process.create_time()!=key[1],'original owned process remains alive')
    return len(seen)


def _sources(reader,freeze,modules):
    pins={item['path']:item for item in freeze['inputs']}
    require(len(pins)==len(freeze['inputs']),'duplicate frozen input paths')
    sources=[]
    for module in modules:
        path=Path(inspect.getsourcefile(module)).resolve()
        require(path.is_relative_to(reader.root),'validator outside project')
        relative=path.relative_to(reader.root).as_posix()
        require(relative in pins,'validator source absent from supplied freeze: '+relative)
        reader.artifact(pins[relative]);sources.append(pins[relative])
    return sources


def _events(reader,path):
    data=reader.bytes(path);require(data.endswith(b'\n'),'complete forensic line suffix')
    previous=None;events=[]
    for index,line in enumerate(data.splitlines()):
        reader.boundary();require(index<2048 and len(line)<=4*MiB,'forensic event bound')
        row=json.loads(line)
        require(set(row)=={'sequence','previous_sha256','event'} and row['sequence']==index and
                row['previous_sha256']==previous,'forensic hash chain/order')
        previous=digest(row);events.append(row['event'])
    require(events,'nonempty failed forensic stream')
    return events,previous


def verify_partial_science(events,plan,clock,child,*,boundary=lambda:None):
    """Authenticate a terminal failed handler, not a passed52 suite.

    This supports failures during the initial fifteen counterexamples. Successful
    handlers, their direct16 reproductions, corrected encoded faces and native
    ordered legal support are checked independently; later suite stages are not
    asserted to have run.
    """
    from tools.splendorust_search_scale_20261024 import qualification as q,audit as science
    from tools.external_strength_screen_20261018 import audit as ledger,rules
    from tools.external_strength_screen_20261018_02 import bridge
    from tools.future_training_rules_20261021 import decode_training_snapshot
    fixtures=plan['counterexamples'];cases={x['id']:x for x in fixtures}
    require(len(fixtures)==len(cases)==15,'original counterexample plan')
    starts=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='started']
    results=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='result']
    checked=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='validated']
    failures=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='failed']
    expected=[(case['id'],arm) for case in fixtures for arm in q.ARMS]
    require(1<=len(starts)<=30 and [(e['case_id'],e['arm']) for e in starts]==expected[:len(starts)],
            'ordered initial counterexample-handler prefix')
    require(len(failures)==1 and len(results)==len(checked)==len(starts)-1,'partial handler counts')
    require([(e['case_id'],e['arm']) for e in results]==expected[:len(results)] and
            [(e['case_id'],e['arm']) for e in checked]==expected[:len(results)],'successful prefix identities')
    failed=failures[0]
    require((failed['case_id'],failed['arm'])==expected[len(results)] and
            events[-1]['event']=='entrypoint_failed' and events[-2]==failed and
            events[-3]['event']=='worker_decision_failed','terminal failure ordering')
    require(all(e.get('event') not in ('failed','entrypoint_failed','worker_decision_failed','rejected')
                for e in events[:-3]),'unexpected earlier failure/rejection')
    require(failed['error']==events[-3]['error']==events[-1]['error']==clock['detail'],
            'original error and forensic failure binding')
    require(child['profiles'] and len(child['profiles'])==len(starts) and
            [p['handle_failed'] for p in child['profiles']]==[False]*len(results)+[True],
            'partial boundary handle counts')
    for event in starts:
        boundary();case=cases[event['case_id']]
        require(event['before_snapshot']==case['snapshot'],'exact failed-suite root')
        state=decode_training_snapshot(event['before_snapshot'])
        observer=rules.observation_payload(state)
        require(event['request']==dict(player=event['arm'],observer=observer),'exact lawful ordinary payload')
    for result,saved,start in zip(results,checked,starts):
        boundary();arm=result['arm'];observer=start['request']['observer'];quota=64 if arm=='search' else 16
        require(result['request']==saved['request']==start['request'] and
                result['before_snapshot']==saved['before_snapshot']==start['before_snapshot'],'result root binding')
        science.validate_result(result['result'],observer,arm,expected_quota=quota)
        require(q.validate_scientific_decision(arm,observer,result['result']['action'],
                    result['result']['diagnostics'],quota=quota)==saved['check'],'saved scientific check')
        ledger.check_observer(start['before_snapshot'],observer)
        ledger.check_step(start['before_snapshot'],result['result']['action'],saved['after_snapshot'])
    direct=[e for e in events if e.get('operation')=='reference_choose' and e['event']=='result']
    finished_fixtures=fixtures[:len(results)//2]
    require([e['case_id'] for e in direct]==[f['id'] for f in finished_fixtures] and
            all(e['arm']=='reference' for e in direct),'direct reference prefix inventory')
    for e in direct:
        boundary();baseline=next(x['result'] for x in results if x['case_id']==e['case_id'] and x['arm']=='reference')
        require(q.scientific_result('reference',e['result']['action'],e['result']['diagnostics'])==
                q.scientific_result('reference',baseline['action'],baseline['diagnostics']) and
                q.callback_receipts(e['result']['diagnostics'])==q.callback_receipts(baseline['diagnostics']),
                'direct16 exact scientific/callback reproduction')
    encoded=[e for e in events if e.get('operation')=='actual_entity_encoded_costs' and e['event']=='validated']
    require([(e['case_id'],e['arm']) for e in encoded]==expected[:len(results)],'actual encoded-cost prefix counts')
    for e,result in zip(encoded,results):
        case=cases[e['case_id']];state=decode_training_snapshot(case['snapshot'])
        require(q._actual_encoded_costs(state,result['result']['diagnostics'],
            corrected_face='corrected_face' in case['tags'])==e['result'],'actual corrected encoded faces')
    inspected=[e for e in events if e.get('operation')=='exchange' and e['event']=='result' and
               e['request']['native_request']['op']=='inspect']
    expected_inspections=[(f['id'],arm) for f in finished_fixtures if f['id']!='forced_pass' for arm in q.ARMS]
    require([(e['case_id'],e['arm']) for e in inspected]==expected_inspections,'native ordered-support prefix counts')
    for e in inspected:
        observer=rules.observation_payload(decode_training_snapshot(cases[e['case_id']]['snapshot']))
        require(e['result']['legal']==bridge.main_actions(observer),'native ordered complete support')
    catalogues=[e for e in events if e.get('operation')=='exchange' and e['event']=='result' and
                e['request']['native_request']['op']=='catalogue']
    require([e['arm'] for e in catalogues]==list(q.ARMS),'both native catalogue receipts')
    # This supplies saved responses only. The pure checker performs no exchange.
    q._native_catalogues({e['arm']:SimpleNamespace(exchange=lambda request,v=e['result']:v) for e in catalogues})
    complete=[e for e in events if e.get('event')=='complete_result_before_worker_validation']
    require(len(complete) in (len(results),len(starts)),'partial full-result count')
    for e,result in zip(complete,results):
        require(e['arm']==result['arm'] and e['observer']==result['request']['observer'] and
                e['action']==result['result']['action'] and e['diagnostics']==result['result']['diagnostics'],
                'prevalidation/result exact binding')
    failed_science=False
    if len(complete)==len(starts):
        evidence=complete[-1];observer=starts[-1]['request']['observer'];arm=starts[-1]['arm']
        require(evidence['observer']==observer and evidence['arm']==arm,'failed prevalidation identity')
        q.validate_scientific_decision(arm,observer,evidence['action'],evidence['diagnostics'],quota=64 if arm=='search' else 16)
        failed_science=True
    require(events[-3]['observer']==starts[-1]['request']['observer'] and events[-3]['arm']==starts[-1]['arm'],
            'failed callback observer identity')
    return dict(complete_decisions=len(results)+len(direct),successful_handlers=len(results),direct_references=len(direct),
        failed_handlers=1,attempted_handlers=len(starts),saved_transitions=len(results),actual_encoded_cost_checks=len(encoded),
        native_ordered_supports=len(inspected),native_catalogues=2,failed_case_id=failed['case_id'],failed_arm=failed['arm'],
        failed_prevalidation_science_checked=failed_science,failed_result_accepted=False,
        full_suite_passed=False,unreached_checks_not_claimed=True)


def verify_failed_startup(folder,freeze,*,boundary=lambda:None,root=ROOT,verify_owners=False,
        source_module_prefix='tools.splendorust_pool6_core_compare_20261027_02'):
    """Return forensic validity only; never an admission or successful final seal.

    ``verify_owners`` adds the three completed owner-exit proofs and is false when
    invoked by a still-running finalizer. Cleanup of the failed worker is always
    required. Full runtime/source inventory verification remains the caller's
    responsibility; this function pins every imported validator source directly.
    """
    require(source_module_prefix in ('tools.splendorust_pool6_core_compare_20261027_02',
        'tools.splendorust_startup_logging_20261027_02'),'explicit supported startup implementation')
    reader=Reader(root,boundary)
    pins={item['path']:item for item in freeze['inputs']}
    require(len(pins)==len(freeze['inputs']),'duplicate frozen input paths')
    modules=('runner','placement','common')
    if source_module_prefix=='tools.splendorust_startup_logging_20261027_02':
        modules+=('worker','logging_probe')
    for name in modules:
        path=source_module_prefix.replace('.','/')+'/'+name+'.py'
        require(path in pins,'new startup validator source is not frozen: '+path)
        reader.artifact(pins[path])
    old=importlib.import_module(source_module_prefix+'.runner')
    placement=importlib.import_module(source_module_prefix+'.placement')
    from tools.splendorust_search_scale_20261024 import qualification as q,audit as science,compression,adapters as original_adapters
    from tools.splendorust_boundary_timing_20261025 import analyze as timing
    from tools.splendorust_live_boundary_20261025 import raw_transport
    from tools.external_strength_screen_20261018 import rules,audit as ledger
    from tools.external_strength_screen_20261018_02 import bridge
    from tools import future_training_rules_20261021 as corrected
    folder=Path(folder).resolve();attempt=folder.parent.parent
    source_pins=_sources(reader,freeze,[old,old.c,placement,q,science,compression,original_adapters,timing,raw_transport,rules,ledger,bridge,corrected])
    source_pins+=[pins[source_module_prefix.replace('.','/')+'/'+name+'.py'] for name in modules if name not in ('runner','placement','common')]
    for path in ('tools/splendorust_search_scale_20261024/worker.py','tools/splendorust_boundary_timing_20261025/worker.py'):
        descriptor=next(x for x in freeze['inputs'] if x['path']==path);reader.artifact(descriptor);source_pins.append(descriptor)
    init=reader.json(folder/'initialization.json');identity=reader.json(folder/'route_identity.json')
    require(identity['route']=='frozen' and identity['logical_cpu'] in (0,2) and
            identity['arm_id']==f"frozen_cpu{identity['logical_cpu']}" and
            folder.name=='qualification_'+identity['arm_id'],'supported fixed original64 startup tree')
    expected_worker=(old.OBSERVER_MODULE if source_module_prefix.endswith('core_compare_20261027_02')
                     else source_module_prefix+'.worker')
    require(identity['module']==expected_worker and identity['metadata']==init['result'] and
            identity['native_artifacts']==freeze['native_artifacts'] and identity['native_roles'] is None,'unchanged route/READY/native identity')
    metadata=dict(init['result']);require(metadata.pop('inference_backend')==old.BACKEND,'unchanged graph backend')
    original_adapters.validate_ready_metadata(metadata,expected_sources=corrected.SOURCE_HASHES,worker_pid=init['identity']['pid'],
        process_tree=init['telemetry']['process_tree'],expected_quota=64)
    reader.bytes(folder/'boundary_timing_identity.json')
    old.verify_instrumentation_identity(folder,freeze,64,init['result'])
    paths=sorted(path for path in folder.glob('request-*.json') if raw_transport.REQUEST_ID.fullmatch(path.stem))
    require([p.stem for p in paths]==['request-00000001','request-00000002'],'one graph and one failed suite; no retry')
    clocks=[reader.json(p) for p in paths];graph_clock,clock=clocks
    require(graph_clock['status']=='ok' and clock['status']=='worker_error' and clock['accepted'] is False and
            clock['result'] is None and clock['received_before_deadline'] is True and clock['deadline_seconds']==300.,
            'original failed startup administrative clock')
    science.check_clock(clock,limit=300.);science.check_clock(graph_clock,limit=60.)
    require(graph_clock['deadline_seconds']==60. and clock==reader.json(folder/'entrypoint_clock.json'),
            'original admin allowance/clock aliases')
    require(not list((folder/'inputs').glob('*.json')),'no ordinary work after failed qualification')
    require({p.stem for p in (folder/'startup_inputs').glob('*.json')}=={p.stem for p in paths},'startup input inventory')
    cleanup=reader.json(folder/'cleanup.json')
    require(cleanup['status']=='passed' and not cleanup['errors'] and not cleanup['alive'] and
            cleanup['deadline_clamped_to_stage_and_global'] is True,'failed worker cleanup')
    require(cleanup['primary_failure']=={key:clock[key] for key in ('request_id','status','detail')},'cleanup preserves original failure')
    owned=init['telemetry']['process_tree'];confirmed={(p['pid'],p['created']) for p in cleanup['confirmed']}
    require({(p['pid'],p['created']) for p in owned}==confirmed,'exact owned cleanup identity inventory')
    absent=_absence(cleanup['confirmed'],boundary)
    raw={};log_proofs=[]
    require(set(cleanup['log_compression'])=={'stdout.log','stderr.log','events.jsonl'},'three original raw streams')
    require({p.name for p in folder.glob('*.gz')}=={'stdout.log.gz','stderr.log.gz','events.jsonl.gz'},'compressed stream inventory')
    for name,item in cleanup['log_compression'].items():
        boundary();path=folder/(name+'.gz');reader.bytes(path,16*MiB*16+128)
        require(item['path']==path.name and item['status']=='verified' and not(folder/name).exists(),'qualified compressed identity')
        chunks=[]
        proof=compression.verify_log(path,item,max_raw_bytes=16*MiB,max_compressed_bytes=16*MiB*16+128,
            deadline=time.monotonic()+60.,sink=lambda chunk: (boundary(),chunks.append(chunk)))
        raw[name]=b''.join(chunks);log_proofs.append(proof)
    lines=raw['stdout.log'].splitlines(keepends=True)
    require(len(lines)==3 and all(line.endswith(b'\n') and len(line)<=4*MiB for line in lines),
            'exact READY/graph/error raw inventory')
    packets=[raw_transport._packet_evidence(line) for line in lines]
    require(packets[0]['packet']==dict(type='ready',metadata=init['result']),'raw READY binding')
    require(packets[1]['packet']==dict(type='response',request_id=graph_clock['request_id'],result=graph_clock['result']),
            'raw successful graph response')
    require(packets[2]['packet']==dict(type='error',request_id=clock['request_id'],error=clock['detail']),'exact original failed raw packet')
    parents=[];children=[];payloads=[]
    require({p.name for p in(folder/'parent_profiles').glob('*.json')}=={p.name for p in paths} and
            {p.name for p in(folder/'placement_requests').glob('*.json')}=={p.name for p in paths},'complete boundary/placement inventory')
    require({p.name for p in folder.glob('request-*_child_profile.json')}=={p.stem+'_child_profile.json' for p in paths},'complete saved child prefix')
    admission=reader.json(folder/'placement_admission.json');placement.verify_worker_placement(folder,freeze,'isolated',logical_cpu=identity['logical_cpu'])
    launch=reader.json(attempt/'launch.json')
    require((launch['pid'],launch['created'])==(admission['supervisor']['pid'],admission['supervisor']['created']),
            'owning supervisor identity')
    for i,path in enumerate(paths):
        rid=path.stem;receipt=clocks[i];payload=reader.json(folder/'startup_inputs'/path.name)
        parent=reader.json(folder/'parent_profiles'/path.name);child=reader.json(folder/(rid+'_child_profile.json'))
        require(receipt['request_id']==rid and parent['request_id']==rid==child['request_id'],'original request identities')
        raw_transport._bind_parent_wire(parent,[packets[i+1]],completed=i==0)
        raw_transport._bind_child_wire(child,[packets[i+1]],receipt)
        require(child['request_raw_sha256']==parent['request_raw_sha256'] and child['payload_sha256']==digest(payload),
                'actual transported request coupling')
        require(type(parent['request_raw_bytes']) is int and 0<parent['request_raw_bytes']<=MiB and
                parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False,
                'complete bounded transported request')
        proof=reader.json(folder/'placement_requests'/path.name)
        placement.verify_saved_request(proof,freeze,'isolated',request_id=rid,player_pid=init['identity']['pid'],
            supervisor_identity=admission['supervisor'],parent_profile=parent,logical_cpu=identity['logical_cpu'],
            player_identity=init['identity'],expected_processes=owned,original_clock=receipt,cleanup_receipt=cleanup)
        parents.append(parent);children.append(child);payloads.append(payload)
    for index,name in enumerate(('graph_parity_profiles.json','entrypoint_profiles.json')):
        alias=reader.json(folder/name)
        require(alias['parent_profile']==parents[index] and alias['child_profile']==children[index],
                'saved aggregate boundary alias')
    require(payloads[1]=={'operation':'entrypoint_checks'},'actual original entrypoint operation')
    require(graph_clock==reader.json(folder/'graph_parity_clock.json'),'graph clock alias')
    graph_plan=json.loads(reader.artifact(freeze['graph_parity_plan']))
    require(payloads[0]==dict(operation='graph_parity',fixtures=graph_plan['fixtures']),'exact actual graph fixtures')
    timing.validate_startup_profiles(parents[0],children[0],graph_clock,payloads[0],64,worker_pid=init['identity']['pid'])
    for name in ('graph_identity.json','inference_execution_identity.json','graph_ir.txt','graph_parity.json'):
        reader.bytes(folder/name)
    graph=old.verify_graph_evidence(folder,freeze,boundary=boundary)
    parent,child=parents[1],children[1]
    start,arrival,_=timing._validate_parent(parent,clock,completed=False)
    timing._child_identity(child,parent,clock,payloads[1],64,completed=False,startup=True,worker_pid=init['identity']['pid'])
    require(child['error']==clock['detail'] and child['outcome']=='failed_before_successful_emission' and
            child['result_sha256'] is None and child['serialized_packet_utf8_sha256'] is None and
            child['pre_write_ns'] is None and child['post_flush_ns'] is None,'no failed success emission invented')
    previous=child['request_read_complete_ns']
    require(type(child['profiles']) is list and 1<=len(child['profiles'])<=30,'bounded failed initial handler prefix')
    for i,profile in enumerate(child['profiles']):
        _,left,right=timing._validate_handle(profile,child,completed=i<len(child['profiles'])-1,startup=True)
        require(previous<=left<=right<=arrival,'ordered partial handler timings');previous=right
    error=reader.json(folder/'worker_error.json')
    require(error['request_id']==clock['request_id'] and error['error']==clock['detail'],'original worker error binding')
    events,terminal=_events(reader,folder/'entrypoint_forensics.jsonl')
    qualification=json.loads(reader.artifact(freeze['qualification_plan']))
    partial=verify_partial_science(events,qualification['original_suite'],clock,child,boundary=boundary)
    owners=verify_owner_exits(attempt,freeze,boundary=boundary,reader=reader) if verify_owners else None
    return dict(schema='failed_startup_forensic_receipt_v1',status='failed_startup_evidence_verified',
        qualification_passed=False,attempt_accepted=False,admission_passed=False,complete_suite=False,
        failed_request_original_status=clock['status'],original_clock=clock,partial_science=partial,
        worker=folder.name,logical_cpu=identity['logical_cpu'],graph_cases=graph['fixtures'],
        raw_requests=2,raw_error_bytes=parents[1]['response_raw_bytes'],raw_error_sha256=parents[1]['response_raw_sha256'],
        forensic_events=len(events),forensic_terminal_sha256=terminal,placement_requests=2,owned_identities_absent=absent,
        logs=log_proofs,owner_exits=owners,source_pins=source_pins,consumed_inputs=reader.finish(),
        source_scope='Direct verifier pins only; full frozen source/runtime inventory remains a caller obligation',
        source_module_prefix=source_module_prefix,
        no_models=True,no_new_games=True,no_requests=True,no_retry=True,required_experiment_exit_nonzero=True,
        original_clock_unchanged=True,failed_result_remains_unaccepted=True)


def verify_owner_exits(attempt,freeze,*,boundary=lambda:None,reader=None):
    attempt=Path(attempt);reader=Reader(ROOT,boundary) if reader is None else reader
    frozen_bytes=reader.bytes(attempt/'freeze.json');sha=hashlib.sha256(frozen_bytes).hexdigest()
    require(json.loads(frozen_bytes)==freeze,'owner freeze binding')
    claim=reader.json(attempt/'watchdog_claim.json');launch=reader.json(attempt/'launch.json')
    guardian=reader.json(attempt/'guardian_os_exit.json');watchdog=reader.json(attempt/'watchdog_execution.json');execution=reader.json(attempt/'execution.json')
    for value in (claim,launch,guardian,watchdog,execution):require(value['freeze_sha256']==sha,'actual owner freeze identity')
    require(all(value['exit_code']!=0 for value in (guardian,watchdog,execution)),'failed startup must retain nonzero actual exits')
    require(guardian['pid']==claim['guardian_pid'] and guardian['observer']=='owned PowerShell Process handle' and
        abs(datetime.fromisoformat(guardian['created_utc'].replace('Z','+00:00')).timestamp()-claim['guardian_created'])<.00001,
        'actual guardian identity/exit observer')
    require(watchdog['exit_code_observed_by']=='guardian Popen handle' and
            watchdog['observed_process']=='owning_watchdog','actual watchdog OS exit observer')
    identities=[dict(pid=claim['guardian_pid'],created=claim['guardian_created']),
        dict(pid=claim['watchdog_pid'],created=claim['watchdog_created']),dict(pid=launch['pid'],created=launch['created'])]
    for value in (execution,watchdog):
        cleanup=value['cleanup_receipt'];require(not cleanup['errors'] and not cleanup['unconfirmed'],'owner cleanup errors')
        identities+=cleanup['confirmed']
    absent=_absence(identities,boundary)
    return dict(status='failed_actual_exits_verified',guardian_exit=guardian['exit_code'],watchdog_exit=watchdog['exit_code'],
        runner_exit=execution['exit_code'],owned_identities_absent=absent,experiment_passed=False)


def verify_saved_attempt(attempt_dir=PRIOR,*,boundary=lambda:None):
    """Authenticate the accepted failure review and finite old evidence inventory.

    This reuses its already verified full16156-source/93-runtime audit explicitly;
    it does not recursively rehash that ancestry or revise the failed attempt.
    """
    from tools.future_training_rules_20261021 import verify_runtime_sources
    runtime_sources=verify_runtime_sources()
    attempt=Path(attempt_dir).resolve();require(attempt==PRIOR.resolve(),'explicit pinned historical failure only')
    reader=Reader(ROOT,boundary);freeze=reader.json(attempt/'freeze.json')
    require(reader.items[(attempt/'freeze.json').relative_to(ROOT).as_posix()]['sha256']==PRIOR_FREEZE_SHA,'known prior freeze')
    review_path=attempt/'postrun/independent_failure_review.json';review=reader.json(review_path)
    require(reader.items[review_path.relative_to(ROOT).as_posix()]['sha256']==PRIOR_REVIEW_SHA,'known prior failure review')
    require(review['review_passed'] is True and review['accepted'] is False and review['qualification_passed'] is False and
            review['original_completion_status']=='incomplete' and review['original_status_unchanged'] is True and
            review['freeze_sha256']==PRIOR_FREEZE_SHA,'failure review is not successful qualification')
    inventory_path=attempt/'postrun/failure_evidence_inventory.json';inventory=reader.json(inventory_path)
    require(reader.items[inventory_path.relative_to(ROOT).as_posix()]['sha256']==PRIOR_INVENTORY_SHA,'known failed evidence inventory')
    require(len(inventory['artifacts'])==inventory['file_count']==112 and
            sum(x['bytes'] for x in inventory['artifacts'])==inventory['total_bytes'],'exact finite preservation counts')
    expected={x['path'] for x in inventory['artifacts']}
    actual={p.relative_to(ROOT).as_posix() for p in attempt.rglob('*') if p.is_file() and 'postrun' not in p.relative_to(attempt).parts}
    require(actual==expected,'original failed attempt gained/lost evidence')
    for item in inventory['artifacts']:reader.artifact(item)
    for item in review['consumed_inputs']:reader.artifact(item)
    require(reader.json(attempt/'completion.json')['status']=='incomplete' and
            not(attempt/'completed_evidence_seal.json').exists(),'preserved original incomplete attempt')
    failed=verify_failed_startup(attempt/'workers/qualification_frozen_cpu2',freeze,boundary=boundary,verify_owners=True)
    return dict(schema='saved_failed_startup_finalization_v1',status='historical_failed_startup_forensics_verified',
        original_completion_unchanged='incomplete',qualification_passed=False,attempt_accepted=False,
        failed_worker=failed,prior_full_audit_reused=dict(review_sha256=PRIOR_REVIEW_SHA,freeze_sha256=PRIOR_FREEZE_SHA,
            scope='Full source/runtime/previous-attempt audit reused from authenticated independent failure review'),
        original_evidence_files=112,consumed_inputs=reader.finish(),runtime_sources=runtime_sources,
        no_models=True,no_new_games=True,no_retry=True,required_experiment_exit_nonzero=True)

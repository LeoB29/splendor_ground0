"""Bounded saved-only qualification of the two retained-worker closure checks.

No frozen module is modified or rebound. These two copied proof functions retain
all original raw/scientific/placement/log checks; only the administrative
request authorization is made explicit for the frozen resumed worker lifecycle.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import time
import traceback
import psutil
from tools.splendorust_search128_recovery_20261028 import common as c
from tools.splendorust_search128_recovery_20261028 import transport as held
from tools.splendorust_search128_recovery_20261028 import runner,qualification,recovery
from tools.future_training_rules_20261021 import verify_runtime_sources

OUT=c.ROOT/'outputs/splendorust_search128_closure_check_20261029_01'
ATTEMPT_SHA='524daba942dd6478ccbfc2a9293f391b1502ffca94163228aef8d1d3dba718f4'
WORKERS={'main_g38_challenger':'research','main_g38_opponent':'frozen'}
MAX_SECONDS=900.
adapters=held.adapters
OBSERVER_MODULE=held.OBSERVER_MODULE
REQUEST_ID=held.REQUEST_ID
MAX_FAILED_LINES=held.MAX_FAILED_LINES
_packet_evidence=held._packet_evidence
_bind_parent_wire=held._bind_parent_wire
_bind_child_wire=held._bind_child_wire
verify_log=held.verify_log
validate_profiles=held.validate_profiles
verify_worker_logs=held.verify_worker_logs
verify_graph_evidence=held.verify_graph_evidence
verify_placement_request=held.verify_placement_request


def validate_lifecycle_data(name,route,payloads,summary,entrypoint,clock,graph_clock,initialization,identity):
    c.require(name in WORKERS and route==WORKERS[name],'Only the two exact frozen retained trees')
    c.require(payloads=={'request-00000001':'graph_parity','request-00000002':'entrypoint_checks'},
              'Graph1 then exactly one entrypoint2, never an injected/repeated suite')
    c.require(summary['status']=='passed' and summary['scientific_decisions']==119 and
        summary['parent_requests']==2 and summary['original_scientific_decisions']==203 and
        summary['combined_scientific_decisions']==322 and summary['same_workers_retained_for_continuation'] is True and
        summary['failed_game_decision_retried'] is False,'Exact frozen119-suite summary')
    receipts=summary['receipts']
    c.require(len(receipts)==2 and {x['route'] for x in receipts}=={'frozen','research'},'Both distinct suite routes')
    selected=[x for x in receipts if x['route']==route]
    c.require(len(selected)==1,'Unique owning suite')
    pin=selected[0]
    c.require(pin['worker_folder']==(c.OUT/'workers'/name).relative_to(c.ROOT).as_posix(),
              'Suite binds exact owning worker path')
    c.require(entrypoint==clock and clock['status']=='ok' and clock['request_id']=='request-00000002' and
        clock['result']['counts']['complete_decisions']==(67 if route=='research' else 52),
        'Exact same actual entrypoint receipt identity')
    c.require(pin['verification']['status']=='passed' and pin['verification']['route']==route and
        pin['verification']['raw_execution_decisions_bound']==(67 if route=='research' else 52),
        'Correct route/count forensic proof')
    c.require(graph_clock['request_id']=='request-00000001' and graph_clock['status']=='ok',
              'Original completed graph request1')
    c.require(identity['route']==route and identity['module']==OBSERVER_MODULE and
        identity['metadata']==initialization['result'],'Original READY route identity')
    return dict(startup_requests=2,graph_request_id='request-00000001',entrypoint_request_id='request-00000002',
        route=route,worker=name,administrative_requests_before_all_ordinary=True)


def authenticated_lifecycle(folder,freeze):
    folder=Path(folder)
    c.require(folder.resolve()==(c.OUT/'workers'/folder.name).resolve() and folder.name in WORKERS,
              'Exact frozen recovery worker location')
    c.require(c.file_hash(c.OUT/'freeze.json')==ATTEMPT_SHA and c.read(c.OUT/'freeze.json')==freeze,
              'Exact incomplete recovery freeze')
    c.require(c.read(c.OUT/'completion.json')['status']=='incomplete','Failed attempt stays incomplete')
    descriptor=freeze['recovery_manifest'];c.verify([descriptor]);manifest=c.read(c.ROOT/descriptor['path'])
    c.require(manifest['partial_game_index']==38 and manifest['continuation_ply']==40 and
        manifest['completed_prefix_plies']==39 and manifest['original_attempt_status']=='incomplete',
              'Frozen continuation boundary')
    identity=c.read(folder/'route_identity.json');route=identity['route']
    summary=c.read(c.OUT/'recovery_qualification_summary.json')
    for entry in summary['receipts']:c.verify([entry['clock']])
    selected=next((x for x in summary['receipts'] if x['route']==route),None)
    c.require(selected is not None and selected['clock']==c.artifact(c.OUT/'recovery_entrypoint_checks'/f'{route}.json'),
              'Bound owning actual entrypoint artifact')
    payloads={p.stem:c.read(p).get('operation') for p in (folder/'startup_inputs').glob('*.json')}
    lifecycle=validate_lifecycle_data(folder.name,route,payloads,summary,
        c.read(c.OUT/'recovery_entrypoint_checks'/f'{route}.json'),c.read(folder/'request-00000002.json'),
        c.read(folder/'graph_parity_clock.json'),c.read(folder/'initialization.json'),identity)
    graph=c.read(folder/'startup_inputs/request-00000001.json')
    plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
    c.require(graph=={'operation':'graph_parity','fixtures':plan['fixtures']} and
        c.read(folder/'startup_inputs/request-00000002.json')=={'operation':'entrypoint_checks'},'Exact administrative payloads')
    ordinary=[p.stem for p in (folder/'inputs').glob('*.json')]
    c.require(all(int(x.split('-')[1])>=3 for x in ordinary),'No ordinary request before administrative checks')
    return lifecycle


def verify_raw_transport(folder,freeze,envelope):
    folder=Path(folder);cleanup=c.read(folder/'cleanup.json');init=c.read(folder/'initialization.json')
    metadata=init['result'];route=c.read(folder/'route_identity.json')['route'];quota=adapters.route_quota(route);mode=route
    c.require(route in ('frozen','research'),'Only frozen64 and research routes')
    lifecycle=authenticated_lifecycle(folder,freeze)
    paths=sorted(path for path in folder.glob('request-*.json') if REQUEST_ID.fullmatch(path.stem))
    c.require(len(paths)<=2002,'Bounded request inventory')
    c.require([p.stem for p in paths]==[f'request-{i:08d}' for i in range(1,len(paths)+1)],
              'Contiguous original worker request inventory')
    receipts={path.stem:c.read(path) for path in paths};ids=list(receipts)
    failures=[rid for rid,receipt in receipts.items() if receipt['status']!='ok']
    c.require(not failures or failures==[ids[-1]],'At most one final failed request per worker')
    failed_id=failures[0] if failures else None
    packets={rid:[] for rid in ids};pending=bytearray();ready=False;index=0;partial=None
    def packet_line(raw):
        nonlocal ready,index
        envelope.guard();c.require(len(raw)<=4*c.MiB and raw.endswith(b'\n'),'Raw packet frame bound')
        item=_packet_evidence(raw);packet=item['packet']
        if not ready:
            c.require(packet=={'type':'ready','metadata':metadata},'Exact raw READY metadata changed')
            ready=True;return
        c.require(index<len(ids),'Unsolicited extra raw packet after successful final request')
        rid=ids[index];receipt=receipts[rid]
        if receipt['status']=='ok':
            c.require(type(packet) is dict and set(packet)=={'type','request_id','result'} and
                packet['type']=='response' and packet['request_id']==rid and packet['result']==receipt['result'],
                'Completed raw response differs from original trusted receipt')
            packets[rid].append(item);index+=1
        else:
            c.require(rid==failed_id and len(packets[rid])<MAX_FAILED_LINES,'Bounded failed raw packet suffix')
            packets[rid].append(item)
    def sink(chunk):
        pending.extend(chunk)
        while True:
            end=pending.find(b'\n')
            if end<0:break
            raw=bytes(pending[:end+1]);del pending[:end+1];packet_line(raw)
        c.require(len(pending)<=4*c.MiB,'Raw pending response frame bound');envelope.guard()
    verified=verify_log(folder/'stdout.log.gz',cleanup['log_compression']['stdout.log'],
        max_raw_bytes=128*c.MiB,max_compressed_bytes=128*c.MiB*16+128,
        deadline=envelope.effective_deadline(),sink=sink)
    c.require(ready,'Closed worker raw READY missing')
    if pending:
        c.require(failed_id is not None and index==len(ids)-1,'Partial output without final failed request')
        partial=dict(bytes=len(pending),sha256=hashlib.sha256(pending).hexdigest(),complete_packet=False)
    c.require(index==len(ids) if failed_id is None else index==len(ids)-1,
              'Missing completed raw response')
    input_ids=[p.stem for directory in ('inputs','startup_inputs') for p in (folder/directory).glob('*.json')]
    c.require(len(input_ids)==len(ids) and set(input_ids)==set(ids),'Exact input request inventory')
    expected_profiles={rid+'_child_profile.json' for rid in ids}
    c.require({p.name for p in folder.glob('request-*_child_profile.json')}<=expected_profiles,
              'Unexpected child sidecar inventory')
    c.require({p.stem for p in (folder/'parent_profiles').glob('*.json')}==set(ids),
              'Complete parent sidecar inventory required')
    startup_count=0;checks=[]
    for rid,receipt in receipts.items():
        envelope.guard();c.require(receipt['request_id']==rid,'Raw trusted receipt identity')
        startup=folder/'startup_inputs'/(rid+'.json');ordinary=folder/'inputs'/(rid+'.json')
        c.require(startup.is_file()!=ordinary.is_file(),'Exactly one saved request payload type required')
        if startup.is_file():
            payload=c.read(startup);startup_count+=1
            c.require(receipt['status']=='ok','Admission failure cannot be accepted as a live-player failure')
            if payload=={'operation':'entrypoint_checks'}:
                c.require(rid==lifecycle['entrypoint_request_id'],'Exact authenticated retained-worker entrypoint request')
            else:
                c.require(rid==lifecycle['graph_request_id'],'Exact graph request precedes actual suite')
                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
                c.require(quota in (64,128) and payload=={'operation':'graph_parity','fixtures':plan['fixtures']},
                          'Exact original graph startup payload required')
        else:
            payload=c.read(ordinary)
            c.require(set(payload)=={'player','observer'} and payload['player']=='search','Exact fixed live payload')
            c.require(receipt['deadline_seconds']==c.SECONDS_PER_MOVE,'Exact full research route allowance required')
            for packet in packets[rid]:
                c.require(packet['raw_bytes']<=4*c.MiB,'Common ordinary response exceeds4MiB cap')
        parent=c.read(folder/'parent_profiles'/(rid+'.json'))
        child_path=folder/(rid+'_child_profile.json')
        if child_path.exists():
            c.require(child_path.stat().st_size<=(512*1024+1 if startup.is_file() else 8193),
                      'Original bounded child sidecar size')
            child=c.read(child_path)
        else:
            c.require(receipt['status']!='ok','Completed child sidecar missing');child=None
        c.require(parent['request_id']==rid,'Parent sidecar request binding')
        parent_observed=_bind_parent_wire(parent,packets[rid],completed=receipt['status']=='ok')
        if child is not None:
            c.require(child['request_id']==rid and child['payload_sha256']==c.digest(payload) and
                parent['request_raw_sha256']==child['request_raw_sha256'] and
                type(parent['request_raw_bytes']) is int and 0<parent['request_raw_bytes']<=c.MiB and
                parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False,
                'Actual parent written bytes differ from child raw request')
        binding=_bind_child_wire(child,packets[rid],receipt)
        if receipt['status']=='illegal' or receipt.get('result') is not None and receipt['status']!='ok':
            # The original parent retained this decoded value before rejecting
            # it. Bind it to the actual trusted packet without accepting it.
            observed=[item for item in packets[rid] if item['raw_sha256']==parent['response_raw_sha256']]
            c.require(len(observed)==1,'Exactly one trusted rejected response packet')
            packet=observed[0]['packet']
            c.require(type(packet) is dict and set(packet)=={'type','request_id','result'} and
                packet['type']=='response' and packet['request_id']==rid and
                c.canonical(packet['result'])==c.canonical(receipt['result']),
                'Retained rejected result differs from original trusted raw response')
        validate_profiles(parent,child,receipt,payload,route,worker_pid=init['identity']['pid'])
        if receipt['status']=='worker_error':
            errors=[p['packet'] for p in packets[rid] if type(p['packet']) is dict and p['packet'].get('type')=='error']
            c.require(len(errors)==1 and errors[0].get('request_id')==rid and
                str(errors[0].get('error'))[:4096]==receipt['detail'],'Original worker error/raw packet binding')
        checks.append(dict(request_id=rid,status=receipt['status'],startup=startup.is_file(),
            parent_trusted_packet_observed=parent_observed,raw_packets=len(packets[rid]),**binding,
            failed_packets=[{key:item[key] for key in ('raw_sha256','raw_bytes','decode_error')}
                            for item in packets[rid]] if receipt['status']!='ok' else []))
    expected_startups=lifecycle['startup_requests']
    c.require(startup_count==expected_startups,'Original startup request inventory differs')
    return dict(status='passed',worker=folder.name,requests=len(ids),startup_requests=startup_count,
        observed=True,quota=quota,scientific_mode=mode,stdout_log=verified,exact_bytes_checked=True,
        failed_requests=len(failures),failed_request_id=failed_id,partial_failed_output=partial,requests_checked=checks,
        newline_normalization='terminal CRLF to LF only; JSON bytes never reserialized',
        failed_packets_remain_unaccepted=True,original_trusted_receipts_unchanged=True)

def verify_closed_worker(folder,freeze,envelope):
    folder=Path(folder);envelope.close_worker(folder);envelope.guard()
    identity=c.read(folder/'route_identity.json');route=identity['route']
    logs=verify_worker_logs(folder,envelope,route)
    raw=verify_raw_transport(folder,freeze,envelope)
    init=c.read(folder/'initialization.json');metadata=init['result']
    expected_artifacts=freeze['research_native_artifacts'] if route=='research' else freeze['native_artifacts']
    expected_roles=freeze['research_native_roles'] if route=='research' else None
    c.require(identity==dict(route=route,module=OBSERVER_MODULE,
        native_artifacts=expected_artifacts,native_roles=expected_roles,metadata=metadata), 'Saved READY/role/module identity binding')
    adapters.verify_saved_ready(folder,freeze,route=route)
    cleanup=c.read(folder/'cleanup.json')
    c.require(not cleanup['alive'] and not cleanup['errors'] and cleanup['deadline_clamped_to_stage_and_global'] is True,
              'Owned worker cleanup errors remain failures')
    for owned in cleanup['confirmed']:
        envelope.guard()
        try:proc=psutil.Process(owned['pid'])
        except psutil.NoSuchProcess:continue
        c.require(proc.create_time()!=owned['created'],'Owned worker identity remains alive after cleanup')
    graph=verify_graph_evidence(folder,freeze,boundary=envelope.guard)
    parents=sorted((folder/'parent_profiles').glob('request-*.json'))
    c.require([p.name for p in parents]==[p.name for p in sorted((folder/'placement_requests').glob('request-*.json'))],
              'Complete per-request placement inventory')
    for path in parents:
        envelope.guard();verify_placement_request(folder,path.stem,freeze,c.read(path))
    envelope.close_worker(folder);envelope.guard()
    return dict(status='passed',worker=folder.name,route=route,logs=logs,raw=raw,graph=graph,
                placement_requests=len(parents),placement_mode='isolated')


class Boundary:
    def __init__(self,seconds=MAX_SECONDS):
        c.require(0<seconds<=MAX_SECONDS,'Bounded saved checker duration')
        self.started=time.monotonic();self.wall=time.time()
        self.end=min(self.started+seconds,c.ORIGINAL_MONOTONIC+c.JOB_SECONDS)
        self.wall_end=min(self.wall+seconds,c.ORIGINAL_UNIX+c.JOB_SECONDS)
        self.last_resource=-float('inf');self.peak_rss=0
    def guard(self):
        c.require(min(self.end-time.monotonic(),self.wall_end-time.time())>0,'Saved checker/shared original deadline')
        now=time.monotonic()
        if now-self.last_resource>=1:
            self.last_resource=now;p=psutil.Process();self.peak_rss=max(self.peak_rss,p.memory_info().rss)
            c.require(self.peak_rss<=2*c.GiB and not p.children(recursive=True),'Saved-only checker process/memory bound')
    def effective_deadline(self):
        self.guard();return time.monotonic()+min(self.end-time.monotonic(),self.wall_end-time.time())
    def close_worker(self,folder,child=None):
        c.require(child is None and Path(folder).resolve() in {(c.OUT/'workers'/name).resolve() for name in WORKERS},
                  'Saved-only closure cannot own or terminate a live worker')


def authenticate_sources(freeze,boundary):
    index={}
    for item in freeze['inputs']:
        path=Path(item['path']);path=path if path.is_absolute() else c.ROOT/path
        key=str(path.resolve()).casefold()
        c.require(key not in index or index[key]['sha256']==item['sha256'],'Conflicting source pins')
        index[key]=item
    items={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.suffix!='.py' or not path.is_relative_to(c.ROOT) or path==Path(__file__).resolve():continue
        boundary.guard();key=str(path).casefold();c.require(key in index,'Imported saved verifier is not frozen: '+str(path))
        item=index[key];c.verify([item]);items[key]=item
    for group in ('torch_runtime','external_runtime','native_artifacts','research_native_artifacts'):
        for item in freeze[group]:
            boundary.guard();path=Path(item.get('absolute_path',item.get('path')))
            if not path.is_absolute():path=c.ROOT/path
            c.require(path.stat().st_size==item['bytes'] and c.file_hash(path)==item['sha256'],'Frozen native/runtime artifact changed')
            items[str(path).casefold()]=dict(path=str(path),bytes=item['bytes'],sha256=item['sha256'])
    c.require(sys.version==freeze['python'] and c.file_hash(sys.executable)==freeze['python_executable_sha256'],'Original Python runtime')
    return list(items.values())


def verify_saved_game(freeze,boundary):
    old=runner.original_freeze()
    saved=c.read(c.OUT/'recovered_bridge.json')
    proved=recovery.verify_bridge(saved['bridge'],old,boundary=boundary.guard)
    c.require(saved==proved,'Existing bridge unchanged')
    folder=c.OUT/'games/main/p019_g38';before=c.read(folder/'initial.json')
    original_folder=c.OLD_OUT/'games/main/p019_g38'
    c.require(before==c.read(original_folder/'initial.json'),'Same frozen initial snapshot')
    spec={k:v for k,v in before.items() if k!='initial'}
    prefix=[c.read(original_folder/'moves'/f'{i:04d}.json') for i in range(39)]
    paths=sorted((folder/'moves').glob('*.json'))
    c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(40,54)],'Exactly14 saved continuation moves')
    rows=[c.read(p) for p in paths]
    c.require(all(row['completed'] is True for row in rows),'Every saved continuation move completed')
    events=[dict(kind='ordinary',value=row) for row in prefix]+[dict(kind='recovery_bridge')]+[dict(kind='ordinary',value=row) for row in rows]
    ending=c.read(folder/'result.json')
    c.require(ending['reason']=='natural' and ending['plies']==54,'Original saved natural ending')
    proof=runner.audit_segmented_game(before['initial'],events,ending,spec,{'bridge':proved,'old_freeze':old},freeze,boundary.guard)
    c.require(proof['status']=='passed' and proof['transitions']==54 and proof['recovered_transitions']==1 and
        proof['failed_moves']==0 and sum(map(len,proof['latencies'].values()))==53,'Complete saved segmented ledger')
    return dict(status='passed_saved_game_only',proof=proof,new_ordinary_moves=14,prior_accepted_moves=39,
        recovered_actions=1,model_calls=0,game_rerun=False,original_attempt_accepted=False,
        source_initial=c.artifact(folder/'initial.json'),source_result=c.artifact(folder/'result.json'))


def verify_owner_exits(boundary):
    path=c.ROOT/'outputs/splendorust_search128_recovery_20261028_continuity/root_owner_exit_observation.json'
    value=c.read(path)
    c.require(value['actual_exit_observed'] is True and value['actual_exit_code']==1 and value['session_closed'] is True,
              'Observed original owner failure exit preserved')
    for item in value['original_identities']:
        boundary.guard()
        try:p=psutil.Process(item['pid'])
        except psutil.NoSuchProcess:continue
        c.require(p.create_time()!=item['created'],'Original owning identity remains live')
    for name in ('execution.json','watchdog_execution.json','guardian_os_exit.json'):
        result=c.read(c.OUT/name)
        c.require(result['exit_code']==1 and result['freeze_sha256']==ATTEMPT_SHA,'Three actual failed owner exits')
        if 'cleanup_errors' in result:c.require(not result['cleanup_errors'],'Runner tree cleanup errors')
        if 'cleanup_receipt' in result:c.require(not result['cleanup_receipt']['errors'] and not result['cleanup_receipt']['unconfirmed'],'Owned cleanup verified')
    return dict(status='passed_failed_owner_cleanup',actual_exit_codes=[1,1,1,1],
        owner_observation=c.artifact(path),experiment_accepted=False)


def selected_evidence():
    paths=set()
    for sub in ('workers','games/main/p019_g38','recovery_entrypoint_checks'):
        paths.update(p for p in (c.OUT/sub).rglob('*') if p.is_file())
    for name in ('freeze.json','completion.json','failure.json','incomplete_verification.json','incomplete_evidence_seal.json',
        'recovery_qualification_summary.json','recovery_stage_transition.json','recovery_manifest.json','recovered_bridge.json',
        'launch.json','watchdog.json','watchdog_claim.json','guardian_claim.json','execution.json','watchdog_execution.json',
        'guardian_os_exit.json','owner_failure.json','owner_launch.json','launcher_placement.json'):
        paths.add(c.OUT/name)
    paths.add(c.ROOT/'outputs/splendorust_search128_recovery_20261028_continuity/root_owner_exit_observation.json')
    bridge=c.read(c.OUT/'recovered_bridge.json')
    for item in bridge['bridge']['consumed_inputs']:
        path=Path(item['path']);paths.add(path if path.is_absolute() else c.ROOT/path)
    paths.add(c.OLD_OUT/'games/main/p019_g38/initial.json')
    paths.update(c.OLD_OUT/'games/main/p019_g38/moves'/f'{i:04d}.json' for i in range(39))
    return sorted(paths)


def _artifact(path):
    path=Path(path);return dict(path=path.relative_to(c.ROOT).as_posix() if path.is_relative_to(c.ROOT) else str(path),
        bytes=path.stat().st_size,sha256=c.file_hash(path))


def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf8',newline='\n') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run():
    boundary=Boundary();start_identity=dict(pid=os.getpid(),created=psutil.Process().create_time())
    save('checker_launch.json',dict(**start_identity,maximum_seconds=900,original_global_deadline=c.ORIGINAL_UNIX+c.JOB_SECONDS))
    try:
        runtime_sources=verify_runtime_sources()
        c.require(c.file_hash(c.OUT/'freeze.json')==ATTEMPT_SHA,'Exact saved incomplete freeze')
        freeze=c.read(c.OUT/'freeze.json');sources=authenticate_sources(freeze,boundary)
        before=[]
        for path in selected_evidence():boundary.guard();before.append(_artifact(path))
        owners=verify_owner_exits(boundary);old_failures=[];proofs=[];suites=[]
        plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
        for name,route in WORKERS.items():
            folder=c.OUT/'workers'/name
            try:held.verify_raw_transport(folder,freeze,boundary)
            except ValueError as error:
                c.require(str(error)=='Unexpected entrypoint request during play','Different original failure cannot be relabelled')
                old_failures.append(dict(worker=name,error=str(error),original_source_unchanged=True))
            else:raise ValueError('Original lifecycle rejection was not reproduced')
            proofs.append(verify_closed_worker(folder,freeze,boundary))
            receipt=c.read(c.OUT/'recovery_entrypoint_checks'/f'{route}.json')
            suites.append(qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=boundary.guard))
        game=verify_saved_game(freeze,boundary)
        for item in before:boundary.guard();c.verify([item])
        c.require(selected_evidence()==[Path(x['path']) if Path(x['path']).is_absolute() else c.ROOT/x['path'] for x in before],
                  'No evidence inventory additions or deletions')
        final_sources=authenticate_sources(freeze,boundary)
        final_index={x['path']:x for x in final_sources}
        c.require(all(final_index.get(x['path'])==x for x in sources),'Imported verifier/runtime sources unchanged')
        sources=final_sources
        value=dict(status='qualified_saved_closure_checker_only',freeze_sha256=ATTEMPT_SHA,
            original_attempt_status='incomplete',original_attempt_accepted=False,benchmark_accepted=False,
            original_failures_reproduced=old_failures,closed_workers=proofs,actual_entrypoint_verification=suites,
            actual_entrypoint_decisions=sum(x['complete_decisions'] for x in suites),graph_cases=22,
            ordinary_saved_moves=14,segmented_game=game,owner_cleanup=owners,consumed_inputs=before,
            imported_frozen_sources_and_runtime=sources,corrected_runtime_sources=runtime_sources,
            checker_source=_artifact(Path(__file__).resolve()),reader=start_identity,
            elapsed_seconds=time.monotonic()-boundary.started,peak_rss_bytes=boundary.peak_rss,
            original_files_unchanged=True,new_model_calls=0,new_games=0,original_clocks_rewritten=False,
            limitation='Saved closure and one completed game verified separately; no resumed run or full400-game acceptance.')
        save('closure_check.json',value);boundary.guard();print(json.dumps({'status':value['status'],'receipt':_artifact(OUT/'closure_check.json')}),flush=True)
        return 0
    except BaseException:
        save('checker_failure.json',dict(error=traceback.format_exc(),original_attempt_accepted=False,reader=start_identity))
        raise


if __name__=='__main__':raise SystemExit(run())

"""Bounded logging evidence and nonqualifying failure finalization."""
import hashlib
import json
import time
from pathlib import Path
from . import common as c,design

def run_saved_replay(envelope,panel):
    from .logging_probe import make_recorder
    from .analysis import replay_overhead_gate
    source=design.BASELINE/'workers/qualification_frozen_cpu0/entrypoint_forensics.jsonl'
    events=[json.loads(line)['event'] for line in source.read_text(encoding='utf-8').splitlines()]
    event_by_case={}
    for pos in panel['positions']:
        matches=[e for e in events if e.get('event')=='complete_result_before_worker_validation' and e.get('arm')=='search' and e.get('observer')==pos['observer']]
        c.require(matches and all(c.canonical(e)==c.canonical(matches[0]) for e in matches),'Same saved prevalidation science')
        event_by_case[pos['case_id']]=matches[0]
    failure_source=design.PRESERVED_ATTEMPT/'workers/qualification_frozen_cpu2/entrypoint_forensics.jsonl'
    failed_events=[json.loads(line)['event'] for line in failure_source.read_text(encoding='utf-8').splitlines()]
    c.require(event_by_case['private_reserves'] in failed_events,'Replay target equals the actual failed prevalidation record')
    folder=c.OUT/'saved_replays';folder.mkdir(exist_ok=True);rows=[]
    for block in range(4):
        for step,enabled in enumerate((False,True,True,False,True,False,False,True)):
            positions=panel['positions'] if (block+step)%2==0 else list(reversed(panel['positions']))
            for pos in positions:
                envelope.guard();case=pos['case_id'];event=event_by_case[case];index=len(rows)
                path=folder/f'{index:04d}.jsonl';recorder=make_recorder(path,enabled)
                try:
                    start=time.perf_counter_ns();recorder.record(event);end=time.perf_counter_ns()
                finally:recorder.close()
                raw=path.read_bytes();expected=json.dumps(dict(sequence=0,previous_sha256=None,event=event),separators=(',',':'),sort_keys=True,allow_nan=False).encode()
                verified=raw in (expected+b'\n',expected+b'\r\n')
                row=dict(index=index,case_id=case,logging_timing_enabled=enabled,elapsed_seconds=(end-start)/1e9,
                    start_ns=start,end_ns=end,bytes_verified=verified,log=c.artifact(path),source=c.artifact(source),
                    original_event_sha256=c.digest(event),timing_records=recorder.entries)
                c.save(folder/f'{index:04d}.json',row);rows.append(row)
                c.require(verified,'Saved replay changed original bytes')
    c.require(len(rows)==64,'Fixed replay inventory')
    c.save(c.OUT/'saved_replay_results.json',dict(rows=rows,new_model_calls=0,new_games=0))
    gate=replay_overhead_gate(rows);c.save(c.OUT/'saved_replay_gate.json',gate);return gate

def verify_replay():
    from .analysis import replay_overhead_gate
    from .logging_probe import MARKERS
    saved=c.read(c.OUT/'saved_replay_results.json');rows=saved['rows']
    c.require(saved['new_model_calls']==saved['new_games']==0 and len(rows)==64,'Exact saved replay count')
    panel=c.read(c.OUT/'saved_panel.json');c.require(panel==design.build_panel(),'Frozen replay panel changed')
    expected_order=[]
    for block in range(4):
        for step,enabled in enumerate((False,True,True,False,True,False,False,True)):
            positions=panel['positions'] if (block+step)%2==0 else list(reversed(panel['positions']))
            expected_order.extend((p,enabled) for p in positions)
    cache={}
    for index,row in enumerate(rows):
        c.require(row['index']==index and c.read(c.OUT/'saved_replays'/f'{index:04d}.json')==row,'Replay order and original receipt')
        pos,enabled=expected_order[index]
        c.require(row['case_id']==pos['case_id'] and row['logging_timing_enabled'] is enabled and row['log']['path']==(c.OUT/'saved_replays'/f'{index:04d}.jsonl').relative_to(c.ROOT).as_posix(),'Exact counterbalanced replay allocation')
        c.require(row['source']==c.artifact(design.BASELINE/'workers/qualification_frozen_cpu0/entrypoint_forensics.jsonl'),'Exact replay source')
        c.verify([row['log']]);src=row['source']
        if src['path'] not in cache:
            c.verify([src]);cache[src['path']]=[json.loads(x)['event'] for x in (c.ROOT/src['path']).read_text(encoding='utf-8').splitlines()]
        raw=(c.ROOT/row['log']['path']).read_bytes();event=json.loads(raw)['event']
        c.require(event in cache[src['path']] and c.digest(event)==row['original_event_sha256'],'Replay original event binding')
        c.require(event.get('event')=='complete_result_before_worker_validation' and event.get('arm')=='search' and event.get('observer')==pos['observer'],'Exact fixture prevalidation event')
        expected=json.dumps(dict(sequence=0,previous_sha256=None,event=event),separators=(',',':'),sort_keys=True,allow_nan=False).encode()
        c.require(raw in (expected+b'\n',expected+b'\r\n') and row['bytes_verified'] is True,'Original replay bytes')
        c.require(type(row['start_ns']) is int and row['end_ns']>=row['start_ns'] and row['elapsed_seconds']==(row['end_ns']-row['start_ns'])/1e9,'Replay original clock span')
        timings=row['timing_records']
        if enabled:
            c.require(len(timings)==1 and timings[0]['sequence']==0 and timings[0]['event']==event['event'] and timings[0]['completed'] is True and timings[0]['error'] is None,'Exact ON replay timing event')
            stamps=timings[0]['timestamps'];c.require(set(stamps)==set(MARKERS),'All six replay markers')
            values=[stamps[k] for k in MARKERS]
            c.require(all(type(x) is int for x in values) and values==sorted(values) and row['start_ns']<=values[0]<=values[-1]<=row['end_ns'],'Replay marker clock brackets')
        else:c.require(timings==[],'Exact immutable OFF replay has no added timers or bookkeeping')
    gate=replay_overhead_gate(rows);c.require(c.read(c.OUT/'saved_replay_gate.json')==gate,'Saved replay gate changed');return gate

def probe_path(folder,profile):
    name=profile['log'];c.require(name in ('entrypoint_forensics.jsonl','off.jsonl','on.jsonl'),'Allowlisted forensic log')
    return folder/name if name=='entrypoint_forensics.jsonl' else folder/'logging_forensics'/name

def attach_probe(row,folder,envelope,*,expected_mode):
    path=folder/'logging_probe'/(row['clock']['request_id']+'.json');end=min(envelope.effective_deadline(),time.monotonic()+1.)
    while not path.exists() and time.monotonic()<end:time.sleep(.005)
    if not path.exists():
        row['logging_profile']=None;row['logging_profile_error']='Required timing sidecar missing';return
    row['logging_profile']=c.read(path);row['logging_profile_file']=c.artifact(path)

def verify_probe_row(row,folder):
    from .logging_probe import validate_probe_sidecar
    c.require(not row.get('logging_profile_error') and row.get('logging_profile') is not None,'Timing evidence unavailable')
    item=row['logging_profile_file'];c.verify([item]);profile=row['logging_profile'];spec=row['spec']
    c.require(c.read(c.ROOT/item['path'])==profile and (c.ROOT/item['path']).resolve()==(folder/'logging_probe'/(row['clock']['request_id']+'.json')).resolve(),'Exact timing sidecar binding')
    binding=profile['binding'];c.require(binding['sequence']==spec['sequence'] and binding['arm_id']==spec['arm_id'] and binding['case_id']==spec['case_id'] and binding['observer_sha256']==c.digest(row['observer']),'Timing allocation binding')
    return validate_probe_sidecar(profile,probe_path(folder,profile),spec['logging_timing_enabled'],row['clock']['request_id'],row['child_profile'])

def verify_worker_probes(folder,freeze):
    from .logging_probe import SCHEMA,validate_probe_sidecar
    folder=Path(folder);identity=c.read(folder/'logging_identity.json');route=c.read(folder/'route_identity.json')
    c.require(identity==dict(schema=SCHEMA,route=route['route'],arm_id=route['arm_id'],matrix=freeze['request_matrix'],
        ordinary_requests=24,startup_timer_enabled=True,ordinary_payload_unchanged=True,numerical_route_unchanged=True,
        extra_host_sampler_enabled=False,recorder_modes=[False,True],record_operations_unchanged=True,
        synchronous_fsync_retained=True,mode_selected_by_frozen_worker_local_index=True),'Logging identity changed')
    specs=[s for s in c.read(c.ROOT/freeze['request_matrix']['path'])['requests'] if s['arm_id']==route['arm_id']]
    expected=[];checks=[]
    clocks=sorted(p for p in folder.glob('request-*.json') if p.stem[8:].isdigit())
    for path in clocks:
        rid=path.stem
        if rid=='request-00000001':continue
        sidepath=folder/'logging_probe'/(rid+'.json');expected.append(sidepath.name)
        c.require(sidepath.is_file(),'All attempted startup/ordinary probes required')
        profile=c.read(sidepath);child=c.read(folder/(rid+'_child_profile.json'))
        if rid=='request-00000002':
            mode=True;c.require(profile['binding']==dict(operation='entrypoint_checks',arm_id=route['arm_id']),'Startup timing binding')
        else:
            local=int(rid[8:])-3;spec=specs[local];mode=spec['logging_timing_enabled']
            c.require(profile['binding']==dict(sequence=spec['sequence'],arm_id=spec['arm_id'],case_id=spec['case_id'],worker_local_index=local,observer_sha256=spec['observer_sha256']),'Exact local logging allocation')
        checks.append(validate_probe_sidecar(profile,probe_path(folder,profile),mode,rid,child))
    c.require(sorted(p.name for p in(folder/'logging_probe').glob('*.json'))==expected,'No extra/missing logging sidecars')
    return dict(status='passed',requests=len(checks),checks=checks)

def close_proof(folder,freeze,envelope):
    from . import runner,failure_audit
    path=folder/'entrypoint_clock.json'
    if path.exists() and c.read(path)['status']!='ok':
        proof=failure_audit.verify_failed_startup(folder,freeze,boundary=envelope.guard,verify_owners=False,
            source_module_prefix='tools.splendorust_startup_logging_20261027')
        proof['logging_probe']=verify_worker_probes(folder,freeze)
        return proof
    return runner.verify_closed_worker(folder,freeze,envelope)

def verify_gates(rows):
    from .analysis import overhead_gate
    replay=verify_replay();c.require(replay['passed'],'Models require saved replay gate')
    selected=[r for r in rows if r['spec']['stage']=='overhead'];path=c.OUT/'overhead_gate.json'
    if path.exists():c.require(c.read(path)==overhead_gate(selected),'Actual overhead gate changed')
    else:c.require(len(selected)<64 or not all(r['completed'] for r in selected),'Missing completed overhead gate')
    if any(r['spec']['stage']=='diagnostic' for r in rows):c.require(path.exists() and c.read(path)['passed'],'No detailed work before overhead admission')

def failed_startup_finalization(envelope,freeze,panel,matrix,plan):
    from . import runner,qualification
    c.require(not list((c.OUT/'records').glob('*.json')),'No ordinary work before qualification')
    replay=verify_replay();folders=sorted((c.OUT/'workers').glob('*'));proofs=[];failed=[]
    expected=['qualification_'+arm for arm in runner.ARM_IDS]
    c.require(len(folders)<=len(expected) and [p.name for p in folders]==expected[:len(folders)] and
              all(p.is_dir() for p in folders),'Exact attempted startup worker prefix, without substitutions or retries')
    top_paths={p.name for p in(c.OUT/'entrypoint_checks').glob('*.json')}
    represented=set()
    for index,folder in enumerate(folders):
        envelope.guard();proof=close_proof(folder,freeze,envelope);proofs.append(proof)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{folder.name}.json'),'Closed failure proof changed')
        clock=c.read(folder/'entrypoint_clock.json')
        c.require(clock==c.read(folder/(clock['request_id']+'.json')),'Original startup worker clock binding')
        top_name=runner.ARM_IDS[index]+'.json';top=c.OUT/'entrypoint_checks'/top_name
        if top.exists():
            c.require(c.read(top)==clock,'Top-level original startup clock binding');represented.add(top_name)
        if clock['status']=='ok':
            c.require(not failed and top.exists(),'Preceding successful startup requires original top-level receipt')
            qualification.verify_entrypoint(folder,clock,plan,freeze,boundary=envelope.guard)
        else:
            c.require(index==len(folders)-1 and not failed,'The first failed startup must be the last attempted tree')
            c.require(proof['status']=='failed_startup_evidence_verified' and proof['qualification_passed'] is False and
                proof['attempt_accepted'] is False and proof['complete_suite'] is False and
                proof['original_clock']==clock,'Nonqualifying failure proof and original clock are mandatory')
            failed.append(proof)
    c.require(top_paths==represented,'No extra or unbound startup summary receipts')
    c.require({p.name for p in(c.OUT/'closed_workers').glob('*.json')}=={p.name+'.json' for p in folders},
              'Every attempted worker has exactly one closed evidence proof')
    c.require((replay['passed'] and len(failed)==1) or (not replay['passed'] and not folders),'Exactly one stopped startup or negative replay gate')
    runner.verify_runtime(freeze,envelope);preserved=runner.verify_prior_preservation(freeze,envelope)
    placement=runner.verify_exchanges(envelope,freeze)
    value=dict(status='verified_failure_evidence_only',complete=False,failed_startup_verified=bool(failed),
        startup_qualification_passed=False,attempt_accepted=False,ordinary_requests=0,failed_workers=failed,
        verified_worker_proofs=proofs,saved_replay_gate=replay,placement=placement,prior_preservation=preserved,
        new_games=0,new_models_during_finalization=0,original_failure_preserved=True)
    c.save(c.OUT/'failed_startup_finalization.json',value)
    items=[]
    for sub in ('workers','closed_workers','saved_replays','placement_requests','placement_acks','stages'):
        for path in sorted((c.OUT/sub).rglob('*')):
            if path.is_file():envelope.guard();items.append(c.artifact(path))
    for name in ('failure.json','failed_startup_finalization.json','saved_replay_gate.json','saved_replay_results.json','prior_preservation_after.json'):
        if(c.OUT/name).exists():items.append(c.artifact(c.OUT/name))
    c.save(c.OUT/'failure_evidence_seal.json',dict(status='failure_preservation_only',attempt_accepted=False,artifacts=items))
    for item in items:envelope.guard();c.verify([item])
    return value

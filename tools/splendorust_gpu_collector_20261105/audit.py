"""Independent saved-evidence checks for the bounded full-game collector.

No model imports or gameplay. Large immutable journals are scanned once into a
temporary derived index; the original records remain the scientific evidence.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
import struct
import sys
import tempfile
import time
import uuid
from copy import deepcopy
from itertools import combinations

from . import common as c
from tools.splendorust_gpu_throughput_20261105 import audit as prior

read,digest,bits,timing=prior.read,prior.digest,prior.bits,prior.timing


def require(value,message):
    if not value:raise ValueError('COLLECTOR_AUDIT: '+message)


def guard_audit_clock(claim=None):
    claim=read(c.OUT/'claim.json') if claim is None else claim
    require(0<=time.monotonic()-claim['monotonic_origin']<c.TOTAL_SECONDS and
            0<=time.time()-claim['unix_origin']<c.TOTAL_SECONDS,'independent audit remains inside original overall dual clock')


class JournalIndex:
    """One-cell streaming verifier; SQLite stores only derived original rows.

    There is no numerical inference, recovery or rewritten journal. At most one
    FIFO backlog and one physical batch are held while scanning. Request use is
    marked by the independent move/replay checks to reject orphans and reuse.
    """
    def __init__(self,path,route,scheduler,*,incomplete=False):
        self.path=Path(path);self.route=route;self.scheduler=scheduler;self.incomplete=incomplete
        self.scratch=tempfile.TemporaryDirectory(prefix='splendor_collector_audit_')
        self.db=sqlite3.connect(str(Path(self.scratch.name)/'requests.sqlite'))
        self.db.execute('PRAGMA journal_mode=OFF');self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('CREATE TABLE requests(id TEXT PRIMARY KEY,client TEXT,seq INTEGER,slot INTEGER,deadline REAL,submitted REAL,completed REAL,batch INTEGER,body TEXT,used TEXT)')
        self.db.execute('CREATE INDEX decision ON requests(client,deadline,seq)')
        self.db.execute('CREATE INDEX physical_batch ON requests(batch)')
        self.db.execute('CREATE TABLE batches(id INTEGER PRIMARY KEY,body TEXT)')
        self.batch_count=0;self.request_count=0;self.dispatches=0;self.enqueued=0
        self.clients={};self.histogram=Counter();self.queue_seconds=[];self.forward_seconds=[]
        self.started=time.perf_counter();self.scan_seconds=None;self.journal_artifact=None
        self.claim=read(c.OUT/'claim.json') if (c.OUT/'claim.json').exists() else None
    def __enter__(self):
        try:self.scan();return self
        except BaseException as error:self.close(error);raise
    def __exit__(self,typ,error,tb):self.close(error)
    def close(self,error=None):
        if self.db is None:return
        if error is not None:
            folder=c.OUT.with_name(c.OUT.name+'_audit_scratch_receipts');folder.mkdir(parents=True,exist_ok=True)
            receipt=dict(status='derived_index_audit_failed',journal=str(self.path.resolve()),
                journal_artifact=self.journal_artifact,route=self.route,scheduler=self.scheduler,
                indexed_batches=self.batch_count,indexed_requests=self.request_count,
                scratch_index_was_not_source=True,error_type=type(error).__name__,error=str(error),
                models=0,games=0,original_journal_unchanged=True)
            with (folder/(uuid.uuid4().hex+'.json')).open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True)
        self.db.close();self.db=None;self.scratch.cleanup()
    def scan(self):
        if self.claim is not None:guard_audit_clock(self.claim)
        require(self.route in ('cpu','gpu') and self.scheduler in ('original8','row_fifo32'),'journal route/scheduler')
        require(self.route=='gpu' or self.scheduler=='original8','unchanged CPU scheduler')
        before=self.path.stat();hasher=hashlib.sha256();deferred=[];pending=None;last_ticket=0
        physical=8 if self.route=='gpu' else 1
        with self.path.open('rb') as stream:
            for raw in stream:
                hasher.update(raw);event=json.loads(raw)
                if event['kind']=='scheduler_dispatch':
                    require(self.scheduler=='row_fifo32' and pending is None,'one scheduler record before its physical completion')
                    self.dispatches+=1
                    require(event['scheduler']==self.scheduler and event['scheduler_dispatch_index']==self.dispatches,'original scheduler sequence')
                    ready=event['ready'];require(ready and ready[:len(deferred)]==deferred,'retained oldest deferred prefix')
                    require(len(ready)<=32 and len({x['client_id'] for x in ready})==len(ready),'bounded one-pending-client backlog')
                    tickets=[x['enqueue_sequence'] for x in ready];ids=[x['request_id'] for x in ready]
                    require(tickets==sorted(tickets) and len(set(tickets))==len(tickets) and len(set(ids))==len(ids),'unique ordered FIFO tickets')
                    for row in ready[len(deferred):]:
                        require(type(row['enqueue_sequence']) is int and row['enqueue_sequence']==last_ticket+1,'contiguous new enqueue ticket')
                        require(row['submitted_perf']<=event['observed_perf'] and row['submitted_perf']<row['deadline_perf'],'original queued request clock')
                        require(self.db.execute('SELECT 1 FROM requests WHERE id=?',(row['request_id'],)).fetchone() is None,'completed request cannot reenter queue')
                        last_ticket+=1
                    selected=[];keep=[];occupied=set()
                    for row in ready:
                        require(type(row['slot']) is int and 0<=row['slot']<8,'physical queue row')
                        if row['slot'] in occupied:keep.append(row)
                        else:occupied.add(row['slot']);selected.append(row)
                    require(event['selected_request_ids']==[x['request_id'] for x in selected] and
                            event['deferred_request_ids']==[x['request_id'] for x in keep],'exact greedy earliest request per row')
                    pending=(event,selected);deferred=keep;continue
                require(event['kind']=='inference_batch' and event['route']==self.route,'original physical event route/kind')
                rows=event['requests'];require(event['physical_batch']==physical and 1<=len(rows)<=physical,'unchanged physical batch shape')
                occupied=sorted(x['slot'] for x in rows)
                require(len(set(occupied))==len(occupied) and all(type(x) is int and 0<=x<8 for x in occupied),'distinct physical batch rows')
                if self.scheduler=='row_fifo32':
                    require(pending is not None and event['batch_id']==self.dispatches,'completed physical batch binds preceding dispatch')
                    scheduled,selected=pending
                    require(all(scheduled['observed_perf']<x['deadline_perf'] for x in scheduled['ready']),'completed dispatch has no already-expired ready request')
                    require([x['request_id'] for x in selected]==[x['request_id'] for x in rows],'actual physical request order equals FIFO selection')
                    for row,item in zip(rows,selected):
                        require(all(row[k]==item[k] for k in ('request_id','client_id','slot','submitted_perf','deadline_perf')) and
                                scheduled['observed_perf']<=row['dispatch_perf'] and scheduled['observed_perf']<row['deadline_perf'],
                                'actual forward retains queued identity and original deadline')
                    pending=None
                try:self.db.execute('INSERT INTO batches VALUES(?,?)',(event['batch_id'],json.dumps(event,separators=(',',':'))))
                except sqlite3.IntegrityError as error:raise ValueError('COLLECTOR_AUDIT: duplicate physical batch') from error
                for row in rows:
                    require(row['route']==self.route and row['batch_id']==event['batch_id'] and row['native_batch']==1 and
                            row['physical_batch']==physical and row['occupied_slots']==occupied,'actual request batch identity')
                    require(row['submitted_perf']<=row['dispatch_perf']<row['complete_readback_perf']<row['deadline_perf'] and
                            row['queue_seconds']==row['dispatch_perf']-row['submitted_perf'] and
                            row['forward_roundtrip_seconds']==row['complete_readback_perf']-row['dispatch_perf'],'complete original callback clocks')
                    client=row['client_id'];sequence=int(row['request_id'].rsplit(':',1)[1]);previous=self.clients.get(client)
                    require(row['request_id']==client+':'+str(sequence) and sequence==(1 if previous is None else previous['seq']+1),'one complete callback sequence per client')
                    require(previous is None or previous['completed']<=row['submitted_perf'] and previous['slot']==row['slot'],'one outstanding callback and permanent client row')
                    try:self.db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?,NULL)',(row['request_id'],client,sequence,row['slot'],row['deadline_perf'],row['submitted_perf'],row['complete_readback_perf'],row['batch_id'],json.dumps(row,separators=(',',':'))))
                    except sqlite3.IntegrityError as error:raise ValueError('COLLECTOR_AUDIT: duplicate completed request') from error
                    self.clients[client]=dict(seq=sequence,completed=row['complete_readback_perf'],slot=row['slot'])
                    self.request_count+=1;self.queue_seconds.append(row['queue_seconds'])
                self.batch_count+=1;self.histogram[len(rows)]+=1;self.forward_seconds.append(rows[0]['forward_roundtrip_seconds'])
                if self.batch_count%2048==0:
                    self.db.commit()
                    if self.claim is not None:guard_audit_clock(self.claim)
        after=self.path.stat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'journal unchanged while streaming')
        self.journal_artifact=dict(path=self.path.resolve().relative_to(c.ROOT).as_posix(),sha256=hasher.hexdigest(),bytes=after.st_size)
        limits=self.db.execute('SELECT MIN(id),MAX(id),COUNT(*) FROM batches').fetchone()
        require(limits==(None,None,0) if not self.batch_count else limits==(1,self.batch_count,self.batch_count),'complete contiguous physical batch inventory')
        if self.scheduler=='row_fifo32':
            require(self.dispatches==self.batch_count or self.incomplete and self.dispatches==self.batch_count+1,'every dispatch completes except authenticated final failure')
            require(self.incomplete or pending is None and not deferred and last_ticket==self.request_count,'all queued requests complete with none dropped')
        else:require(self.dispatches==0,'original control has no candidate scheduler events')
        self.enqueued=last_ticket;self.db.commit();self.scan_seconds=time.perf_counter()-self.started
        if self.claim is not None:guard_audit_clock(self.claim)
        return self
    def __len__(self):return self.request_count
    def __getitem__(self,key):
        row=self.db.execute('SELECT body FROM requests WHERE id=?',(key,)).fetchone()
        if row is None:raise KeyError(key)
        return json.loads(row[0])
    def client_rows(self,client):
        return [json.loads(row[0]) for row in self.db.execute('SELECT body FROM requests WHERE client=? ORDER BY seq',(client,))]
    def decision(self,client,started,*,owner,slot=None):
        saved=self.db.execute('SELECT id,body,used FROM requests WHERE client=? AND deadline=? ORDER BY seq',(client,started+60.)).fetchall()
        rows=[]
        for key,body,used in saved:
            row=json.loads(body);require(used is None,'callback belongs to exactly one original decision')
            require(row['submitted_perf']>=started and (slot is None or row['slot']==slot),'decision origin and permanent row')
            self.db.execute('UPDATE requests SET used=? WHERE id=?',(owner,key));rows.append(row)
        return rows
    def mark(self,ids,owner):
        for key in ids:
            row=self.db.execute('SELECT used FROM requests WHERE id=?',(key,)).fetchone()
            require(row is not None and row[0] is None,'unique existing callback attribution')
            self.db.execute('UPDATE requests SET used=? WHERE id=?',(owner,key))
    def complete_attribution(self):
        require(self.db.execute('SELECT COUNT(*) FROM requests WHERE used IS NULL').fetchone()[0]==0,'no orphan callbacks outside warmup/qualification/game/replay')
        self.db.commit()
        return dict(status='streamed_original_journal_verified',journal=self.journal_artifact,requests=self.request_count,
                    physical_batches=self.batch_count,clients=len(self.clients),scheduler=self.scheduler,
                    dispatches=self.dispatches,enqueued=self.enqueued,all_callbacks_attributed_once=True,
                    derived_index_is_not_evidence=True,model_calls=0)
    def metrics(self,prefix):
        query='SELECT body FROM requests WHERE used LIKE ?';rows=self.db.execute(query,(prefix+'%',))
        count=0;queues=[];forwards=[];batch_count=0;histogram=Counter()
        for (body,) in rows:
            row=json.loads(body);count+=1;queues.append(row['queue_seconds'])
        for (body,) in self.db.execute('SELECT body FROM batches WHERE id IN (SELECT DISTINCT batch FROM requests WHERE used LIKE ?)',(prefix+'%',)):
            event=json.loads(body);batch_count+=1;histogram[len(event['requests'])]+=1
            require(self.db.execute('SELECT COUNT(*) FROM requests WHERE batch=? AND used LIKE ?',(event['batch_id'],prefix+'%')).fetchone()[0]==len(event['requests']),
                    'a physical batch does not mix measured gameplay and excluded work')
            forwards.append(event['requests'][0]['forward_roundtrip_seconds'])
        return dict(completed_callback_prefixes=count,physical_batches=batch_count,occupied_rows_histogram=dict(sorted(histogram.items())),
                    mean_occupied_rows=count/batch_count if batch_count else None,queue_seconds=timing(queues) if queues else None,
                    physical_forward_readback_seconds=timing(forwards) if forwards else None,
                    scope='Original attributed gameplay callbacks; includes physical transfers/readback, not pure GPU kernels')


def independent_example(row,*,winner=None,terminal=False):
    """Independently map true native visits and official actor-oriented outcomes."""
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules
    from tools.external_strength_screen_20261018_02 import bridge
    from tools.splendorust_selfplay_pilot_resume2_20261103.scientific import canonical_index
    from tools.splendorust_selfplay_pilot_resume2_20261103.capture_adapter import public_root_tokens,validate_capture
    from tools.splendorust_selfplay_pilot_resume2_20261103.data import check_encoded_costs
    state=guard.decode_training_snapshot(row['before']);observer=rules.observation_payload(state)
    capture=row['training_capture'];validate_capture(capture,observer,expected_quota=512)
    actions=bridge.main_actions(observer)
    reason=('forced_pass' if actions==[{'kind':'Pass'}] else 'single_choice' if len(actions)==1 else
            'late_heuristic' if state.turn_index>=124 else None)
    require(capture['eligible'] is (reason is None) and capture['exclusion_reason']==reason,'independent mechanical target eligibility')
    if reason is not None:return dict(eligible=False,reason=reason,ply=row['ply'],actor=state.current_player)
    take_masks=[sum(1<<i for i in which) for n in (1,2,3) for which in combinations(range(5),n)]
    mapping={mask:30+i for i,mask in enumerate(take_masks)}
    for source,target,count in ((32,55,5),(37,12,12),(49,24,3),(52,0,12),(64,27,3)):
        mapping.update({source+i:target+i for i in range(count)})
    indices=[canonical_index(action) for action in actions];mask=[False]*81
    for index in indices:mask[mapping[index]]=True
    visits=capture['native_capture']['visits67']
    require(type(visits) is list and len(visits)==67 and all(type(x) is int and x>=0 for x in visits) and sum(visits)==512,
            'actual complete fixed512 root visit mass')
    target=[0.]*81
    for index,count in enumerate(visits):
        if count:
            require(index in indices and index in capture['retained_indices67'],'search visits on retained authoritative actions')
            target[mapping[index]]=count/512.
    require(capture['visits67']==visits and capture['policy_target81']==target and capture['legal_mask81']==mask,
            'independently normalized full legal root policy, not selected-action imitation')
    tokens=public_root_tokens(observer);require(tokens==capture['root_tokens'],'lawful public encoding independent of hidden state')
    main=row['trace']['phases'][0];root=main['inference'][0]
    require(root['tokens']==tokens and root['outputs']==[capture['root_output83']] and root['input_sha256']==digest(tokens) and
            root['output_sha256']==digest(root['outputs']),'target root is first actual same-decision inference')
    result=dict(eligible=True,ply=row['ply'],actor=state.current_player,tokens=deepcopy(tokens),policy=target,mask=mask,
                canonical_players=[state.current_player,1-state.current_player],observer_sha256=digest(observer),
                row_sha256=digest(row),root_input_sha256=digest(tokens),encoded_cost_checks=check_encoded_costs(row['before'],tokens),
                authoritative_legal_actions=[rules.serialize_action(action) for action in guard.legal_actions_for_snapshot(row['before'])])
    if terminal:
        require(winner is None or type(winner) is int and winner in (0,1),'official winner or draw')
        value=0. if winner is None else 1. if winner==state.current_player else -1.
        result.update(value=[value,-value if value else 0.],game_id=row['game_id'],game_index=row['spec']['game_index'],route=row['route'])
    return result


def audit_decision(folder,spec,ply,journal,*,before,owner,client=None,slot=None):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import audit as ledger,rules
    from tools.splendorust_gpu_throughput_20261105 import scientific
    folder=Path(folder);name=f'{ply:06d}.json';row=read(folder/'moves'/name)
    state=guard.decode_training_snapshot(before);observer=rules.observation_payload(state);request=dict(player='search',observer=observer)
    attempt=read(folder/'attempts'/name);clock=read(folder/'clocks'/name);raw=read(folder/'raw_results'/name)
    require(prior.validate_clock(clock) and clock['status']=='ok' and clock['completed'] is True,'successful unchanged original decision clock')
    require(row['before']==before and row['ply']==ply and row['game_id']==spec['game_id'] and row['spec']==spec and
            row['clock']==clock and row['observer']==observer and row['actor']==state.current_player and row['route']==journal.route,
            'every original contiguous corrected source decision')
    actual_client=row['client_id'];actual_slot=row['slot']
    require((client is None or client==actual_client) and (slot is None or slot==actual_slot),'assigned original client and physical slot')
    require(attempt['client_id']==actual_client and attempt['slot']==actual_slot and attempt['game_id']==spec['game_id'] and
            attempt['ply']==ply and attempt['started_perf_counter']==clock['started_perf_counter'] and
            attempt['request_sha256']==digest(request) and attempt['before_sha256']==digest(before) and attempt['no_retry'] is True and
            read(folder/'requests'/name)==request,'durable original request/origin with no clock reset')
    result=row['result'];scientific.validate_result(result,observer,'search',expected_quota=512)
    scientific.audit_decision(observer,result['action'],result['diagnostics'],'search',before=before,after=row['after'],expected_quota=512)
    require(row['action']==result['action'] and row['trace']==result['diagnostics'] and raw['result']==result and raw['observer']==observer and
            raw['action']==result['action'] and raw['diagnostics']==result['diagnostics'],'full raw response, action and trace binding')
    require(row['after']==ledger.transition(before,result['action']),'independent full dictionary-ledger transition')
    guard.check_training_transition(before,result['action'],row['after'])
    callbacks=journal.decision(actual_client,clock['started_perf_counter'],owner=owner,slot=actual_slot)
    receipts=read(folder/'service_calls'/name)
    require(len(receipts)==len(callbacks) and prior.callback_hashes(result)==[digest([x['input_sha256'],x['output_sha256'],1]) for x in callbacks],
            'all native callbacks bind original service request and output hashes')
    for receipt,entry in zip(receipts,callbacks):
        require({k:v for k,v in receipt.items() if k!='client_received_perf'}==entry and
                entry['complete_readback_perf']<=receipt['client_received_perf']<entry['deadline_perf'] and
                receipt['client_received_perf']<=clock['ended_perf_counter'],'returned callback before same complete original clock')
    require(row['training_capture']==read(folder/'captures'/name),'original separate native training-capture evidence')
    example=independent_example(row)
    if example['eligible']:
        require(callbacks and callbacks[0]['input_sha256']==example['root_input_sha256'] and
                callbacks[0]['output_sha256']==digest([row['training_capture']['root_output83']]),'actual root callback binds policy target source')
    return row,example,len(callbacks)


def qualification_gate(proof):
    routes=proof.get('routes',{});passed=proof.get('status')=='passed' and proof.get('passed') is True and set(routes)=={'cpu','gpu'}
    if passed:
        passed=(proof.get('complete_decisions')==232 and proof.get('ordinary_games')==proof.get('training_updates')==0 and
            proof.get('compilation_fingerprint')=='2a720c18d129310700dd850eb9b6ae8c99b4a6345bdbf437bc924fef6feb9dbb')
        for route,row in routes.items():
            passed=passed and row['route']==route and row['complete_decisions']==116 and row['entrypoint']['proof']['actual_decisions']==67 and (
                row['parity_decisions']==32 and len(row['parity'])==32 and all(x['exact_full_result_and_after'] is True for x in row['parity']) and
                row['loader_decisions']==16 and row['official_continuations']==1 and row['loader_row_checks']['status']=='actual_loader_counterexamples_passed' and
                row['full_loader_checks']['status']=='actual_complete_fixture_loader_passed' and row['full_loader_checks']['eligible_roots']>0 and
                row['journal']['all_callbacks_attributed_once'] is True and row['runtime_closure']['final_weights_verified'] is True)
    return dict(status='collector_admitted' if passed else 'collector_not_admitted',passed=passed,
                actual_collector_loader_qualified=passed,exact_original_search_qualified=passed,no_retry=True,
                scope='Only this fixed64-trajectory throughput comparison; no training or promotion')


_PRIVATE={}
LIBRARIES={
    'batchfill':('tools/splendorust_gpu_batchfill_20261105/audit.py','ef3e37d43fac0522cb44a903f37d35d3abe21b1b9f149040792e74f276f96c13'),
    'compiled':('tools/splendorust_gpu_compiled_20261105/audit.py','d5b70b33857ff6cab59c818d14cc4bcf7c95f80c2966d993220a0336a97ec5e2')}


def private_checker(name):
    relative,sha=LIBRARIES[name];path=c.ROOT/relative
    require(c.sha(path)==sha,'authenticated independent historical checker source')
    if name not in _PRIVATE:
        module_name='.'.join(Path(relative).with_suffix('').parts[:-1])+f'._private_collector_{name}'
        specification=importlib.util.spec_from_file_location(module_name,path);module=importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        for key in ('ROOT','INTERNAL_SECONDS','OUTER_SECONDS','EXECUTION_POLICY_ID'):
            require(getattr(module.c,key)==getattr(c,key),'unchanged scientific clock semantics '+key)
        original=sys.modules.get('.'.join(Path(relative).with_suffix('').parts));previous=None if original is None else original.c
        module.c=c;require(original is None or original.c is previous,'private context does not mutate historical public modules')
        _PRIVATE[name]=module
    require(_PRIVATE[name].c is c,'only new explicit evidence context');return _PRIVATE[name]


def source_context_proof():
    for name in LIBRARIES:private_checker(name)
    return dict(status='private_saved_checker_context_verified',libraries=[c.artifact(c.ROOT/path) for path,_ in LIBRARIES.values()],
        context=c.artifact(Path(c.__file__)),binding='Private module.c only; frozen historical files/public modules untouched',
        resource_bounds='New12h total/10h model/actual8GiB RSS12GiB output/1GiB disk; original60/75 decision clocks retained',models=0,games=0)


class CountOnly:
    def __init__(self,count):self.count=count
    def __len__(self):return self.count


def runtime_closure(plan,folder,route,journal,*,incomplete=False,failed=False):
    closure=prior.audit_runtime(plan,folder,route,CountOnly(journal.batch_count),journal,incomplete=incomplete)
    identity=read(Path(folder)/'runtime_identity.json');scheduler='original8' if route=='cpu' else 'row_fifo32'
    require(identity['scheduler']==scheduler and identity['maximum_logical_clients']==(8 if route=='cpu' else 32) and
            identity['physical_batch_unchanged']==8 and 'timing_observer_enabled' not in identity,'unchanged CPU8 or compiled GPU16 service')
    if route=='gpu':
        compiled=private_checker('compiled').compilation_identity(folder,'compiled')
        require(compiled['compilation_fingerprint']==plan['compilation_fingerprint'],'exact prior accepted compiled graph')
        closure['compiled_graph']=compiled
    else:
        require(identity['implementation']=='eager' and identity['compilation_fingerprint'] is None and identity['compilation_receipt'] is None,
                'original CPU reference remains unchanged')
    require(incomplete or failed or closure['service_failure'] is None,'no latent service failure in completed cell')
    return closure


def audit_warmup(plan,folder,journal):
    folder=Path(folder);warmup=read(folder/'warmup.json')
    require(warmup['status']=='passed' and len(warmup['calls'])==5 and warmup['inside_cell_timing'] is False,'five original untimed service warmups')
    baseline=warmup['calls'][0]['output']
    source=plan['cases'][0]['expected_results'][journal.route];c.verify([source])
    expected=prior.root_output(plan['cases'][0],read(c.resolve(source))['result'])
    require(warmup['accepted_source']==source and warmup['exact_accepted_root_bits'] is True and baseline==[expected] and
            bits(baseline[0])==bits(expected),'fresh warmup matches accepted same-route source output bits')
    for i,row in enumerate(warmup['calls']):
        require(row==read(folder/'warmup_raw'/f'{i:02d}.json') and row['index']==i and row['clock']==read(folder/'warmup_clocks'/f'{i:02d}.json') and
                prior.validate_clock(row['clock']) and row['output']==baseline and bits(row['output'][0])==bits(baseline[0]),'actual exact same-route warmup clocks and raw outputs')
        receipt=row['receipt'];key='warmup:'+str(i+1);saved=journal[key]
        require({k:v for k,v in receipt.items() if k!='client_received_perf'}==saved and saved['slot']==0 and saved['client_id']=='warmup' and
                saved['deadline_perf']==row['clock']['started_perf_counter']+60. and saved['input_sha256']==digest(plan['cases'][0]['expected_root_tokens']) and
                saved['output_sha256']==digest(row['output']) and saved['complete_readback_perf']<=receipt['client_received_perf']<saved['deadline_perf'],
                'warmup binds original full service receipt and deadline')
        journal.mark([key],'warmup')
    return warmup


def audit_entrypoint(plan,folder,journal):
    from tools.splendorust_gpu_throughput_20261105 import qualification as q
    folder=Path(folder);proof=q.verify_qualification(plan['runtime'],folder)
    require(proof==read(folder/'saved_qualification_audit.json'),'actual67 entrypoint proof independently reconstructed')
    events,_=q.read_forensics(folder/'entrypoint_forensics.jsonl');lib=prior.load_saved_library(folder)
    hashes={'search':[],'reference':[]};clocks=[x for x in events if x.get('operation') in ('outer_clock','outer_ordinary_clock')]
    for clock in clocks:prior.validate_clock(clock)
    for event in events:
        if event.get('event')=='result' and event.get('operation') in ('ordinary_worker_request','reference_choose','direct_same_quota_choose','exchange'):
            hashes[event['arm']].extend(lib.result_callback_hashes(event['result']))
    for arm,slot in (('search',0),('reference',1)):
        rows=journal.client_rows('qualification_'+arm)
        require(hashes[arm]==[digest([x['input_sha256'],x['output_sha256'],1]) for x in rows],'all actual entrypoint callbacks independently matched')
        for row in rows:
            matching=[clock for clock in clocks if row['deadline_perf']==clock['started_perf_counter']+60. and
                      clock['started_perf_counter']<=row['submitted_perf'] and row['complete_readback_perf']<=clock['started_perf_counter']+clock['elapsed_seconds']]
            require(row['slot']==slot and len(matching)==1 and matching[0]['status']=='ok','actual entrypoint callback retains original operation deadline')
        journal.mark([x['request_id'] for x in rows],'entrypoint:'+arm)
    return dict(proof=proof,callbacks={arm:len(values) for arm,values in hashes.items()})


def audit_initialization(container,runtime_folder,route,first_clock=None):
    runtime_folder=Path(runtime_folder);item=runtime_folder.parent/'initialization'/(runtime_folder.name+'.json');row=read(item)
    execution=read(Path(container)/'execution.json');implementation='eager' if route=='cpu' else 'compiled'
    require(row['schema']=='collector_runtime_initialization_v1' and row['route']==route and row['implementation']==implementation and
            Path(row['runtime_folder']).resolve()==runtime_folder.resolve() and row['status']=='completed' and row['error'] is None and
            row['no_retry'] is True and row['inside_collection_timing'] is False,'one successful runtime preparation before collection')
    require(row['elapsed_seconds']==row['ended_perf_counter']-row['started_perf_counter']>=0 and
            execution['unix_origin']<=row['started_unix']<=row['ended_unix']<=execution['unix_end'],'original bounded runtime initialization clock')
    if first_clock is not None:require(row['ended_perf_counter']<=first_clock,'all requests follow actual runtime initialization')
    return dict(artifact=c.artifact(item),route=route,implementation=implementation,elapsed_seconds=row['elapsed_seconds'],scope=row['scope'])


def audit_compilation_admission(plan,runtime_folder):
    runtime_folder=Path(runtime_folder);row=read(runtime_folder/'compilation_admission.json')
    require(row==dict(actual=c.artifact(runtime_folder/'runtime_identity.json'),qualified=plan['compiled_identity'],
        compilation_fingerprint=plan['compilation_fingerprint'],checked_before_requests=True),'fresh runtime accepted compiled graph binding before requests')
    c.verify([row['qualified']]);require(read(c.resolve(row['qualified']))['compilation_fingerprint']==row['compilation_fingerprint'],
        'authenticated prior exact graph identity')
    return row


def audit_full_loader_fixture(folder,document,spec,rows):
    from . import data
    from tools import future_training_rules_20261021 as guard
    folder=Path(folder);require(read(folder/'fixture_target_game.json')==document,'exact complete qualification fixture document')
    manifest=read(folder/'fixture_corpus_manifest.json');expected=data.build_manifest([c.artifact(folder/'fixture_target_game.json')],[spec],route=document['route'],cell_id='qualification_fixture')
    require(manifest==expected,'actual fixture corpus binds only declared engineering continuation')
    examples=[];loaded=data.load_corpus(manifest,expected_specs=[spec],sink=lambda spec,values:examples.extend(values))
    winner=guard.require_official_terminal_target(document['final'],'completed')
    independent=[independent_example(row,winner=winner,terminal=True) for row in rows]
    independent=[x for x in independent if x['eligible']]
    require(examples==independent and read(folder/'fixture_loaded_targets.json')==dict(examples=examples,summary=loaded),
            'actual full loader targets match independent visits and official actor outcomes')
    negative=data.qualify_manifest_rejections(manifest,[spec])
    def reject(operation,name):
        try:operation()
        except (ValueError,KeyError,TypeError,AssertionError):return dict(case=name,rejected=True)
        raise ValueError('COLLECTOR_AUDIT: bad actual loader fixture admitted '+name)
    for reason in ('cutoff','deadlock','timeout'):
        bad=deepcopy(document);bad['reason']=reason;negative.append(reject(lambda bad=bad:data.load_game(bad,spec),reason+'_cannot_label'))
    bad=deepcopy(document);bad['final']=document['initial'];negative.append(reject(lambda:data.load_game(bad,spec),'nonterminal_cannot_label'))
    for key in data.IDENTITY:
        bad=deepcopy(document);bad.pop(key);negative.append(reject(lambda bad=bad:data.load_game(bad,spec),'missing_game_'+key))
    actor0=0. if winner is None else 1. if winner==0 else -1.;actor1=-actor0 if actor0 else 0.
    proof=dict(status='actual_complete_fixture_loader_passed',corpus=c.artifact(folder/'fixture_corpus_manifest.json'),
        loaded=c.artifact(folder/'fixture_loaded_targets.json'),eligible_roots=loaded['eligible_roots'],official_winner=winner,
        actor0_value=[actor0,actor1],actor1_value=[actor1,actor0],negative_checks=negative,
        ordinary_games=0,engineering_fixture_only=True,optimizer_updates=0)
    require(read(folder/'full_loader_checks.json')==proof,'actual identity/hash/nonterminal refusal inventory and outcome orientation')
    return proof


def audit_qualification(plan,container):
    from . import qualification,data,backend
    guard_audit_clock()
    container=Path(container);folder=container/'worker';execution,absent=private_checker('batchfill').audit_stage_execution(plan,container,mode='qualification')
    require(execution['status']=='completed' and execution['worker_exit_code']==0,'actual collector qualification finished normally')
    require(read(folder/'runtime_sources.json')['sources']==plan['source_guard'] and plan['qualification']==qualification.build_plan(),
            'actual once-startup corrected guard and complete fixed collector qualification')
    require(backend.source_equivalence()==plan['backend_source_equivalence'] and backend.capture_provenance()==plan['native_capture'],
            'accepted native capture and unchanged model/search source provenance')
    routes={};summaries={}
    for route in ('cpu','gpu'):
        root=folder/route;runtime=folder/(route+'_runtime');scheduler='original8' if route=='cpu' else 'row_fifo32'
        with JournalIndex(runtime/'inference.jsonl',route,scheduler) as journal:
            entrypoint=audit_entrypoint(plan,root/'entrypoint',journal);parity=[]
            for case in plan['cases']:
                spec=dict(game_id=case['case_id'],game_index=case['index'],purpose='engineering_qualification_only')
                row,example,_=audit_decision(root/'parity'/case['case_id'],spec,0,journal,before=case['snapshot'],
                    owner='parity:'+case['case_id'],client='parity_'+str(case['slot']),slot=case['slot'])
                source=read(c.resolve(case['expected_results'][route]));c.verify([case['expected_results'][route]])
                require(row['result']==source['result'] and row['after']==source['after'] and
                        bits(prior.root_output(case,row['result']))==bits(prior.root_output(case,source['result'])),
                        'capture preserves entire accepted original result, callback bits and afterstate')
                witness=dict(case_id=case['case_id'],actual=c.artifact(root/'parity'/case['case_id']/'moves'/'000000.json'),
                             expected=case['expected_results'][route],exact_full_result_and_after=True)
                require(read(root/'parity'/case['case_id']/'comparison.json')==witness,'exact parity witness');parity.append(witness)
            require({x.name for x in (root/'parity').iterdir() if x.is_dir()}=={x['case_id'] for x in plan['cases']},'no replaced or extra parity cases')
            loader=plan['qualification']['loader'];cases=loader['cases']+[loader['private_variant']];rows=[]
            for i,case in enumerate(cases):
                spec=qualification.fixture_spec(case,i)
                row,_,_=audit_decision(root/'loader'/f'{i:02d}',spec,0,journal,before=case['snapshot'],owner=f'loader:{i}',client='loader',slot=0)
                rows.append(row)
            rowproof=data.qualify_captured_rows(loader,rows)
            require(rowproof==read(root/'loader_row_checks.json') and len(rows)==16,'all actual loader counterexamples and encoded corrected costs')
            i=next(i for i,x in enumerate(cases) if x.get('id')=='noble_final_round');first=rows[i]
            last,_,_=audit_decision(root/'loader_terminal',first['spec'],1,journal,before=first['after'],owner='loader:terminal',client='loader',slot=0)
            document=read(root/'fixture_target_game.json')
            require(document['initial']==first['before'] and document['final']==last['after'] and document['rows']==[
                    c.artifact(root/'loader'/f'{i:02d}'/'moves'/'000000.json'),c.artifact(root/'loader_terminal'/'moves'/'000001.json')],
                    'actual fixture has only initial native decision plus one official final-seat continuation')
            full=audit_full_loader_fixture(root,document,first['spec'],[first,last])
            closure=runtime_closure(plan,runtime,route,journal)
            if route=='gpu':audit_compilation_admission(plan,runtime)
            first_clock=journal.db.execute('SELECT MIN(submitted) FROM requests').fetchone()[0]
            initialization=audit_initialization(container,runtime,route,first_clock)
            summary=read(root/'qualification.json');expected=dict(status='qualified',route=route,actual_entrypoint_decisions=67,
                accepted_saved_decisions=32,actual_loader_decisions=16,official_fixture_continuation_decisions=1,complete_decisions=116,
                entrypoint=c.artifact(root/'entrypoint'/'qualification.json'),parity=parity,loader_rows=c.artifact(root/'loader_row_checks.json'),
                full_loader=c.artifact(root/'full_loader_checks.json'),actual_collector_and_loader_qualified=True,
                ordinary_games=0,training_updates=0,no_retry=True)
            require(summary==expected,'complete original actual qualification route accounting');summaries[route]=summary
            routes[route]=dict(route=route,complete_decisions=116,entrypoint=entrypoint,parity=parity,parity_decisions=32,loader_decisions=16,
                official_continuations=1,loader_row_checks=rowproof,full_loader_checks=full,runtime_closure=closure,
                initialization=initialization,journal=journal.complete_attribution())
    worker=read(folder/'worker_result.json')
    require(worker['mode']=='qualification' and worker['status']=='passed' and worker['exit_code']==0 and worker['reference_weights_unchanged'] is True and
            worker['result']==dict(status='qualified',routes=summaries,complete_decisions=232,new_games=0,engineering_fixture_trajectories=2,training_updates=0) and
            read(folder/'cleanup.json')['success'] is True,'actual completed qualification and original owned cleanup')
    cross=private_checker('batchfill').cross_route_sources(plan)
    guard_audit_clock()
    return dict(status='passed',passed=True,routes=routes,complete_decisions=232,ordinary_games=0,training_updates=0,
                compilation_fingerprint=plan['compilation_fingerprint'],cross_route_numerical_sources=cross,
                native_capture=plan['native_capture'],owned_processes_absent=absent,model_calls=0)


def verify_gate_binding(plan,folder):
    gate=read(c.OUT/'qualification_gate.json');proof=read(c.OUT/'qualification/saved_audit.json')
    source=[c.artifact(c.OUT/'qualification/saved_audit.json')]
    require(gate==dict(qualification_gate(proof),plan_sha256=c.sha(c.PLAN),source_audits=source) and gate['passed'] is True,
            'original complete qualified collector admission')
    binding=read(Path(folder)/'qualification_gate_binding.json')
    require(binding==dict(admission=c.artifact(c.OUT/'qualification_gate.json'),source_audits=source,passed=True,
            independently_recomputed=True,verified_before_model_creation=True),'actual worker independently admitted before models')
    return binding


def audit_game(plan,spec,game,folder,journal,assignment):
    from . import data
    from tools import future_training_rules_20261021 as guard
    from tools.splendorust_gpu_throughput_20261105 import games as old_games
    guard_audit_clock();folder=Path(folder);initial=data.initial_snapshot(game)
    require(read(folder/'spec.json')==game and read(folder/'initial.json')==initial,'frozen fresh initial setup, no replacement')
    started=read(folder/'started.json');require(started['game_id']==game['game_id'] and started['client_id']==assignment['client_id'] and
        started['slot']==assignment['slot'] and assignment['assigned_perf_counter']<=started['started_perf_counter'],
        'game starts only after fixed queue assignment')
    moves=sorted((folder/'moves').glob('*.json'));names=[f'{i:06d}.json' for i in range(len(moves))]
    require(names and len(names)<=2000 and [x.name for x in moves]==names,'complete contiguous bounded game')
    for key in ('attempts','clocks','requests','raw_results','service_calls','captures'):
        require(sorted(x.name for x in (folder/key).glob('*.json'))==names,'all original per-move evidence inventory '+key)
    before=initial;rows=[];examples=[];exclusions=Counter();durations=[];previous_pass=False;callback_count=0
    for ply in range(len(moves)):
        row,example,count=audit_decision(folder,game,ply,journal,before=before,owner=f'game:{game["game_id"]}:{ply}',
            client=assignment['client_id'],slot=assignment['slot'])
        require(started['started_perf_counter']<=row['clock']['started_perf_counter'] and
                (not rows or rows[-1]['clock']['ended_perf_counter']<=row['clock']['started_perf_counter']),
                'original game decision clocks follow one sequential trajectory')
        is_pass=row['action']['action_type']=='PASS';require(not(previous_pass and is_pass) and row['after']!=before,'no deadlock or fabricated terminal')
        if example['eligible']:examples.append(example)
        else:exclusions[example['reason']]+=1
        before=row['after'];previous_pass=is_pass;rows.append(row);durations.append(row['clock']['elapsed_seconds']);callback_count+=count
    winner=guard.require_official_terminal_target(before,'completed');require(examples,'every complete game has usable original search targets')
    for example in examples:
        value=0. if winner is None else 1. if winner==example['actor'] else -1.
        example.update(value=[value,-value if value else 0.],game_id=game['game_id'],game_index=game['game_index'],route=spec['route'])
    document=dict(schema=data.GAME_SCHEMA,**data.IDENTITY,route=spec['route'],spec=game,reason='completed',initial=initial,final=before,winner=winner,
        rows=[c.artifact(path) for path in moves])
    require(read(folder/'target_game.json')==document,'target source document binds every legal move and official final outcome')
    selection=old_games.replay_selection(rows)
    ending=dict(status='naturally_completed',route=spec['route'],game_id=game['game_id'],specification=game,
        client_id=assignment['client_id'],slot=assignment['slot'],plies=len(rows),winner=winner,final=before,
        model_replay_selection=selection,target_game=c.artifact(folder/'target_game.json'))
    require(read(folder/'result.json')==ending,'natural game result independently reconstructed')
    finished=read(folder/'finished.json');require(finished['game_id']==game['game_id'] and
        rows[-1]['clock']['ended_perf_counter']<=finished['ended_perf_counter'] and started['started_unix']<=finished['ended_unix'],
        'durable game result endpoint follows final complete decision')
    outcome=dict(game_id=game['game_id'],status='completed',completed_moves=len(rows),finished=finished,no_retry=True)
    require(read(folder/'outcome.json')==outcome,'original completed game outcome')
    targets=dict(schema='gpu_collector_examples_v1',**data.IDENTITY,spec=game,examples=examples,official_terminal_only=True,teacher_simulations=512,training_updates=0)
    target_path=folder.parent.parent/'targets'/(game['game_id']+'.json');require(read(target_path)==targets,'every retained target independently reconstructed')
    batches=[dict(start=i,count=min(32,len(examples)-i),shape=[min(32,len(examples)-i),31,48]) for i in range(0,len(examples),32)]
    loaded=dict(spec=game,route=spec['route'],plies=len(rows),eligible_roots=len(examples),exclusions=dict(exclusions),winner=winner,batches=batches,
        initial_sha256=digest(initial),final_sha256=digest(before),game_sha256=digest(document))
    # Hash only game states/actions for cross-route workload comparison: clocks,
    # client placement and floating point neural traces are deliberately excluded.
    trajectory=digest(dict(initial=initial,actions=[row['action'] for row in rows],final=before))
    return dict(game_id=game['game_id'],loaded=loaded,ending=ending,outcome=outcome,finished=finished,
        target=c.artifact(target_path),document=c.artifact(folder/'target_game.json'),trajectory_sha256=trajectory,
        descriptive_latency=prior.descriptive_latency(durations),callback_count=callback_count)


def audit_replays(plan,spec,folder,journal,game_proofs,cohort):
    folder=Path(folder);summary=read(folder/'model_replay_summary.json');receipts=[];first=None;last=None
    expected_files=[]
    for game,proof in zip(spec['specifications'],game_proofs):
        ending=proof['ending'];slot=(ending['slot']+1)%8;client='replay_'+game['game_id']
        for chosen in ending['model_replay_selection']:
            ply=chosen['ply'];source_path=folder/'games'/game['game_id']/'moves'/f'{ply:06d}.json';source=read(source_path)
            destination=folder/'model_replays'/game['game_id']/f'{ply:06d}';expected_files.append(destination/'comparison.json')
            row,_,_=audit_decision(destination,game,ply,journal,before=source['before'],owner=f'replay:{game["game_id"]}:{ply}',client=client,slot=slot)
            require(row['result']==source['result'] and row['training_capture']==source['training_capture'] and row['after']==source['after'],
                'selected shifted-row model replay preserves complete result, true visits and afterstate exactly')
            clock=row['clock'];require(clock['started_perf_counter']>=cohort['ended_perf_counter'] and
                (last is None or last<=clock['started_perf_counter']),'selected isolated replays occur after collection with no overlap')
            first=clock['started_perf_counter'] if first is None else first;last=clock['ended_perf_counter']
            receipt=dict(game_id=game['game_id'],ply=ply,source=c.artifact(source_path),replay=c.artifact(destination/'moves'/f'{ply:06d}.json'),
                original_slot=ending['slot'],replay_slot=slot,other_clients_inactive=True,exact_full_result_capture_and_after=True)
            require(read(destination/'comparison.json')==receipt,'exact preselected replay comparison');receipts.append(receipt)
    require(sorted((folder/'model_replays').rglob('comparison.json'))==sorted(expected_files),'all and only fixed first-neural-band replays')
    require(summary['status']=='exact_replays_passed' and summary['count']==summary['model_calls']==len(receipts) and 0<len(receipts)<=48 and
            summary['receipts']==receipts and summary['inside_collection_timing'] is False and summary['wall_seconds']>=last-first,
            'complete bounded original model-replay accounting and timing')
    return summary


def audit_cell(plan,spec,container):
    from . import data
    guard_audit_clock()
    container=Path(container);folder=container/'worker';cell=folder/'cell'
    execution,absent=private_checker('batchfill').audit_stage_execution(plan,container,mode='games',cell=spec)
    require(read(folder/'runtime_sources.json')['sources']==plan['source_guard'],'actual collector once-startup corrected source guard')
    verify_gate_binding(plan,folder)
    if execution['status']!='completed':return audit_incomplete_cell(plan,spec,container,execution,absent)
    require(execution['worker_exit_code']==0 and read(folder/'cleanup.json')['success'] is True,'completed collector actual exit and cleanup')
    with JournalIndex(folder/'runtime'/'inference.jsonl',spec['route'],spec['scheduler']) as journal:
        warmup=audit_warmup(plan,cell,journal);started=read(cell/'started.json');cohort=read(cell/'cohort.json')
        require(started['cell_id']==spec['cell_id'] and cohort['started_perf_counter']==started['started_perf_counter'] and
                cohort['started_unix']==started['started_unix'] and warmup['calls'][-1]['clock']['ended_perf_counter']<started['started_perf_counter'],
                'original complete-collection clock after warmups')
        assignments=[read(path) for path in sorted((cell/'assignments').glob('*.json'))]
        require([path.name for path in sorted((cell/'assignments').glob('*.json'))]==[f'{i:02d}.json' for i in range(16)] and
                len(assignments)==16,'one original assignment per fixed queued game')
        games=[];worker_last={}
        for i,(game,assigned) in enumerate(zip(spec['specifications'],assignments)):
            worker=assigned['worker_index'];require(type(worker) is int and 0<=worker<spec['concurrency'] and assigned['queue_index']==i and
                assigned['game_id']==game['game_id'] and assigned['client_id']==f'worker_{worker}' and assigned['slot']==worker%8 and
                started['started_perf_counter']<=assigned['assigned_perf_counter'] and
                (i==0 or assignments[i-1]['assigned_perf_counter']<=assigned['assigned_perf_counter']) and
                (worker not in worker_last or worker_last[worker]<=assigned['assigned_perf_counter']),
                'ascending fixed queue assigned only to an idle persistent worker')
            proof=audit_game(plan,spec,game,cell/'games'/game['game_id'],journal,assigned);games.append(proof)
            worker_last[worker]=proof['finished']['ended_perf_counter']
        require({path.name for path in (cell/'games').iterdir() if path.is_dir()}=={x['game_id'] for x in spec['specifications']},'no extra/replaced games')
        queue=read(cell/'queue_outcomes.json');require(queue==dict(assignments=assignments,outcomes={x['game_id']:x['outcome'] for x in games},errors=[],never_started=[]),
            'all fixed game queue outcomes durably accounted')
        manifest=data.build_manifest([x['document'] for x in games],spec['specifications'],route=spec['route'],cell_id=spec['cell_id'])
        require(read(cell/'corpus_manifest.json')==manifest,'exact complete source corpus manifest')
        targets=sum(x['loaded']['eligible_roots'] for x in games);plies=sum(x['loaded']['plies'] for x in games)
        loaded=dict(status='actual_collector_loader_passed',manifest_sha256=digest(manifest),games=[x['loaded'] for x in games],
            game_count=16,eligible_roots=targets,training_updates=0,actual_training_loader_used=True,historical_training_loader_used=False,**data.IDENTITY)
        require(read(cell/'loader_result.json')==loaded,'actual loader full results independently reconstructed')
        require(read(cell/'target_manifest.json')==dict(schema='gpu_collector_loaded_targets_v1',**data.IDENTITY,corpus=c.artifact(cell/'corpus_manifest.json'),
            loader=c.artifact(cell/'loader_result.json'),targets=[x['target'] for x in games],count=targets,training_updates=0),'all actual usable targets and identity hashes')
        require(cohort['status']=='completed' and cohort['cell_id']==spec['cell_id'] and cohort['route']==spec['route'] and cohort['concurrency']==spec['concurrency'] and
            cohort['games']==16 and cohort['plies']==plies and cohort['eligible_roots']==targets and cohort['full_cohort_completed'] is True and cohort['corpus_produced'] is True and
            cohort['retries']==cohort['replacements']==cohort['training_updates']==0,'complete official corpus with no retry/replacement')
        elapsed=cohort['ended_perf_counter']-started['started_perf_counter']
        require(elapsed==cohort['wall_seconds']>0 and cohort['games_per_hour']==16*3600/elapsed and cohort['eligible_roots_per_hour']==targets*3600/elapsed and
            max(x['finished']['ended_perf_counter'] for x in games)<=cohort['gameplay_end_perf_counter']<=cohort['ended_perf_counter'] and
            cohort['gameplay_seconds']==cohort['gameplay_end_perf_counter']-started['started_perf_counter'] and
            cohort['target_loader_seconds']==cohort['ended_perf_counter']-cohort['gameplay_end_perf_counter'] and
            execution['unix_origin']<=started['started_unix']<=cohort['gameplay_end_unix']<=cohort['ended_unix']<=execution['unix_end'],
            'primary clock includes natural tails and complete durable target/loader work')
        replay=audit_replays(plan,spec,cell,journal,games,cohort);closure=runtime_closure(plan,folder/'runtime',spec['route'],journal)
        if spec['route']=='gpu':audit_compilation_admission(plan,folder/'runtime')
        initialization=audit_initialization(container,folder/'runtime',spec['route'],warmup['calls'][0]['clock']['started_perf_counter'])
        worker=read(folder/'worker_result.json');require(worker['status']=='passed' and worker['mode']=='games' and worker['cell_id']==spec['cell_id'] and
            worker['exit_code']==0 and worker['result']==dict(cohort,model_replay=replay) and worker['reference_weights_unchanged'] is True,'worker complete result and original weights closure')
        metrics=journal.metrics('game:');accounting=journal.complete_attribution()
    guard_audit_clock()
    return dict(status='passed',cell_id=spec['cell_id'],route=spec['route'],block=spec['block'],concurrency=spec['concurrency'],
        full_cohort_verified=True,games=16,plies=plies,eligible_roots=targets,wall_seconds=elapsed,games_per_hour=16*3600/elapsed,
        eligible_roots_per_hour=targets*3600/elapsed,gameplay_seconds=cohort['gameplay_seconds'],target_loader_seconds=cohort['target_loader_seconds'],
        seed_decimals=[x['seed_decimal'] for x in spec['specifications']],trajectory_hashes=[x['trajectory_sha256'] for x in games],
        game_proofs=games,replay=replay,replay_seconds=replay['wall_seconds'],backend_metrics=metrics,journal=accounting,
        runtime_closure=closure,initialization=initialization,owned_processes_absent=absent,model_calls=0,training_updates=0)


def audit_incomplete_cell(plan,spec,container,execution,absent):
    """Preserve verified move prefixes and original failure clocks, never a rate."""
    from . import data
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules
    folder=Path(container)/'worker';cell=folder/'cell';outer=execution['outer_deadline_observed']
    require(execution['status']=='deadline_failed','only authenticated deadline can end an incomplete cohort normally')
    require(not any((cell/name).exists() for name in ('corpus_manifest.json','target_manifest.json','loader_result.json','model_replay_summary.json')),
            'incomplete cohort has no corpus, selected replay or completed-cohort rate')
    if outer is not None:
        require(outer['kind']=='ordinary' and outer['execution_policy_id']==c.EXECUTION_POLICY_ID and
            abs(outer['deadline_perf']-outer['started_perf']-75.)<1e-5 and abs(outer['deadline_unix']-outer['started_unix']-75.)<1e-5 and
            (outer['observed_perf']>=outer['deadline_perf'] or outer['observed_unix']>=outer['deadline_unix']),
            'original outer75 deadline independently witnessed')
    else:require(execution['worker_exit_code']==20,'returned original inner-deadline failure exit')
    summaries=[];failures=[];bound_outer=False;triggers=[];assignments=[]
    with JournalIndex(folder/'runtime'/'inference.jsonl',spec['route'],spec['scheduler'],incomplete=True) as journal:
        audit_warmup(plan,cell,journal);started=read(cell/'started.json');previous={}
        paths=sorted((cell/'assignments').glob('*.json'))
        require([x.name for x in paths]==[f'{i:02d}.json' for i in range(len(paths))] and 0<len(paths)<=16,'preserved exact ascending assignment prefix')
        for i,path in enumerate(paths):
            assigned=read(path);game=spec['specifications'][i];worker=assigned['worker_index'];assignments.append(assigned)
            require(assigned['queue_index']==i and assigned['game_id']==game['game_id'] and 0<=worker<spec['concurrency'] and
                assigned['client_id']==f'worker_{worker}' and assigned['slot']==worker%8 and started['started_perf_counter']<=assigned['assigned_perf_counter'] and
                (worker not in previous or previous[worker]<=assigned['assigned_perf_counter']),'partial assignment keeps original idle worker and queue identity')
            gamefolder=cell/'games'/game['game_id'];before=data.initial_snapshot(game);complete=0;failed_attempt=None
            require(read(gamefolder/'spec.json')==game and read(gamefolder/'initial.json')==before,'partial game keeps original fresh setup')
            attempts=sorted((gamefolder/'attempts').glob('*.json'))
            require([x.name for x in attempts]==[f'{ply:06d}.json' for ply in range(len(attempts))],'contiguous original attempts')
            for ply,attempt_path in enumerate(attempts):
                if (gamefolder/'moves'/attempt_path.name).exists():
                    row,_,_=audit_decision(gamefolder,game,ply,journal,before=before,owner=f'game:{game["game_id"]}:{ply}',
                        client=assigned['client_id'],slot=assigned['slot']);before=row['after'];complete+=1;continue
                require(ply==len(attempts)-1,'no decision retry or continuation after a failed original move')
                attempt=read(attempt_path);observer=rules.observation_payload(guard.decode_training_snapshot(before));request=dict(player='search',observer=observer)
                origin=attempt['started_perf_counter'];require(attempt['game_id']==game['game_id'] and attempt['ply']==ply and
                    attempt['client_id']==assigned['client_id'] and attempt['slot']==assigned['slot'] and attempt['no_retry'] is True and
                    attempt['before_sha256']==digest(before) and attempt['request_sha256']==digest(request) and
                    read(gamefolder/'requests'/attempt_path.name)==request,'failed request retains exact original input and clock origin')
                callbacks=journal.decision(assigned['client_id'],origin,owner=f'game:{game["game_id"]}:failed',slot=assigned['slot'])
                clockpath=gamefolder/'clocks'/attempt_path.name;clock=read(clockpath) if clockpath.exists() else None
                if clock is not None:
                    prior.validate_clock(clock);require(clock['started_perf_counter']==origin and clock['completed'] is False,'preserved unsuccessful original response clock')
                    require(all(x['complete_readback_perf']<=origin+clock['elapsed_seconds'] for x in callbacks),'completed callback prefix precedes failed response')
                else:require(outer is not None,'missing original clock only after actual supervisor expiry')
                receiptpath=gamefolder/'service_calls'/attempt_path.name
                if receiptpath.exists():
                    receipts=read(receiptpath);require(len(receipts)<=len(callbacks),'returned receipt prefix')
                    for receipt,entry in zip(receipts,callbacks):
                        require({k:v for k,v in receipt.items() if k!='client_received_perf'}==entry and
                            entry['complete_readback_perf']<=receipt['client_received_perf']<entry['deadline_perf'],'original partial returned receipt')
                if outer is not None and outer['client_id']==assigned['client_id'] and outer['started_perf']==origin:bound_outer=True
                failed_attempt=dict(ply=ply,original_origin=origin,clock=clock,completed_callback_prefix=len(callbacks))
            require(sorted(x.name for x in (gamefolder/'moves').glob('*.json'))==[f'{ply:06d}.json' for ply in range(complete)],'only validated original completed move prefix')
            failurepath=gamefolder/'failure.json'
            if failurepath.exists():
                failure=read(failurepath);require(failure['game_id']==game['game_id'] and failure['no_retry'] is True and
                    failure['completed_moves']==complete and failure['next_snapshot_sha256']==digest(before) and
                    failed_attempt is not None and failure['clock']==failed_attempt['clock'],'original failed move evidence binds preserved prefix')
                classification=prior.deadline_classification(failure['error_chain'])
                require(classification in ('deadline_failed','cancelled') and failure['status']==classification,'known original deadline or direct propagated peer failure')
                if classification=='deadline_failed' or prior.wrapped_deadline_proof(failure) is not None:triggers.append(game['game_id'])
                failures.append(failure)
            if (gamefolder/'result.json').exists():
                ending=read(gamefolder/'result.json');winner=guard.require_official_terminal_target(before,'completed')
                require(ending['status']=='naturally_completed' and ending['final']==before and ending['winner']==winner and ending['plies']==complete,
                        'a finished peer remains an official complete trajectory, not a replacement')
                previous[worker]=read(gamefolder/'finished.json')['ended_perf_counter']
            else:previous[worker]=float('inf')
            summaries.append(dict(game_id=game['game_id'],completed_moves=complete,attempts=len(attempts),failed_attempt=failed_attempt,
                                  naturally_completed=(gamefolder/'result.json').exists()))
        require(bound_outer if outer is not None else bool(triggers),'at least one exact original failed deadline, never generic fault relabelled')
        closure=runtime_closure(plan,folder/'runtime',spec['route'],journal,incomplete=outer is not None,failed=True)
        accounting=journal.complete_attribution()
    guard_audit_clock()
    return dict(status='deadline_failed',cell_id=spec['cell_id'],route=spec['route'],block=spec['block'],concurrency=spec['concurrency'],
        full_cohort_verified=False,games=None,plies=sum(x['completed_moves'] for x in summaries),eligible_roots=None,wall_seconds=None,
        games_per_hour=None,eligible_roots_per_hour=None,preserved_game_prefixes=summaries,failures=failures,supervisor_expiry=outer,
        original_deadline_triggers=triggers,journal=accounting,runtime_closure=closure,owned_processes_absent=absent,model_calls=0,training_updates=0)


def audit_cell_cost(proof):
    container=c.OUT/'cells'/proof['cell_id'];cost=read(container/'cell_cost.json');execution=read(container/'execution.json')
    require(cost['schema']=='collector_verified_cell_cost_v1' and cost['cell_id']==proof['cell_id'] and cost['route']==proof['route'] and
        cost['execution']==c.artifact(container/'execution.json') and cost['saved_audit']==c.artifact(container/'saved_audit.json') and
        cost['auditor']==c.artifact(Path(__file__)),'original verified-data cost binds actual execution and durable acceptance')
    for label,key in (('monotonic','monotonic'),('unix','unix')):
        require(cost['worker_started_'+label]==execution[key+'_origin'] and cost['worker_ended_'+label]==execution[key+'_end'] and
            cost['worker_started_'+label]<=cost['worker_ended_'+label]<=cost['audit_started_'+label]<=cost['audit_ended_'+label]<=cost['verified_ended_'+label],
            'unextended ordered original worker, saved audit and durable-proof clocks')
    require(cost['worker_seconds']==execution['monotonic_end']-execution['monotonic_origin'] and
        cost['saved_audit_seconds']==cost['audit_ended_monotonic']-cost['audit_started_monotonic'] and
        cost['verified_seconds']==cost['verified_ended_monotonic']-execution['monotonic_origin']>0,'verified-data wall includes all startup/replay/audit work')
    claim=read(c.OUT/'claim.json')
    require(cost['verified_ended_monotonic']<claim['monotonic_origin']+c.TOTAL_SECONDS and cost['verified_ended_unix']<claim['unix_origin']+c.TOTAL_SECONDS,
            'saved acceptance stays inside original overall study clock')
    return dict(artifact=c.artifact(container/'cell_cost.json'),verified_seconds=cost['verified_seconds'],audit_seconds=cost['saved_audit_seconds'],worker_seconds=cost['worker_seconds'])


def validate_plan(plan):
    from . import design,collector,qualification,backend,compilation
    from tools import future_training_rules_20261021 as guard
    require(plan['schema']=='splendorust_compiled_gpu_collector_v1' and plan['source_guard']==guard.SOURCE_HASHES,
            'corrected prospective collector identity')
    c.verify([plan['prior']['plan'],plan['prior']['audit'],plan['prior']['closeout']])
    require(plan['prior']['plan']['sha256']==design.PLAN_SHA and plan['prior']['audit']['sha256']==design.AUDIT_SHA and
        plan['prior']['closeout']['sha256']==design.CLOSEOUT_SHA,'exact accepted compiled predecessor')
    previous=read(c.resolve(plan['prior']['plan']));accepted=read(c.resolve(plan['prior']['audit']))
    require(accepted['status']=='saved_compiled_comparison_verified' and plan['prior']['frozen_inputs']==previous['inputs'] and
        plan['cases']==previous['cases'] and plan['compiled_identity']==accepted['qualification']['compiled_identity'] and
        plan['compilation_fingerprint']==accepted['qualification']['compilation_fingerprint'] and plan['compilation']==previous['compilation'],
        'only exact accepted same-route positions, outputs and compiled graph reused')
    expected_runtime=deepcopy(previous['runtime']);expected_runtime['native_capture_binary']=plan['native_capture']['binary']
    require(plan['runtime']==expected_runtime and plan['native_capture']==backend.capture_provenance() and
        plan['backend_source_equivalence']==backend.source_equivalence(),'only authenticated target-only capture added to original runtime')
    require(plan['qualification']==qualification.build_plan() and plan['games']==collector.build_plan() and plan['cells']==design.cells(plan['games']),
        'complete predetermined actual-loader qualification and fresh matched32-setup inventory')
    require(plan['worker_limits']==dict(qualification_seconds=3600,games_seconds=14400) and
        plan['total_seconds']==c.TOTAL_SECONDS==43200 and plan['model_seconds']==c.MODEL_SECONDS==36000 and
        plan['maximum_worker_rss_bytes']==c.MAX_RSS and plan['maximum_output_bytes']==c.MAX_OUTPUT and plan['minimum_free_disk_bytes']==c.MIN_DISK,
        'new fixed bounded study with original decision safeguards')
    require(plan['internal_seconds']==60. and plan['outer_seconds']==75. and plan['execution_policy_id']==c.EXECUTION_POLICY_ID and
        plan['worker_affinity']==list(range(19)) and plan['supervisor_affinity']==[19] and
        plan['available_ram_reserve_bytes']==0 and plan['forecast_admission_gate'] is False and plan['timing_observer'] is False,
        'unchanged response clocks, CPU separation and no forecast/RAM admission')
    require(plan['maximum_new_games']==64 and plan['independent_setup_seeds']==32 and plan['generated_trajectories_per_setup']==2 and
        plan['maximum_model_replays']==192 and plan['qualification_decisions_per_route']==116 and plan['qualification_routes']==['cpu','gpu'] and
        plan['qualification_fixture_trajectories']==plan['qualification_extra_continuations']==2 and
        plan['retries']==plan['replacements']==plan['training_updates']==0 and plan['larger_data_deferred'] is True and
        plan['automatic_followup_training'] is plan['automatic_followup_collection'] is False,'no extra games, retry, replacement or training')
    require(plan['practical_advantage']==dict(metric='verified_completed_games_per_hour',minimum_gpu_over_cpu_each_block=1.05,
        complete_exact_cells_required=4,targets_rate_reported_separately=True,statistical_significance_claim=False,
        automatic_promotion=False,never_stops_collection_early=True),'fixed complete verified-throughput criterion')
    return True


def audit_study(plan,output,*,expected_plan_sha,controller_session,controller_exit_code,controller_chunk,allow_failure=False):
    output=Path(output);require(output.resolve()==c.OUT.resolve() and c.sha(c.PLAN)==expected_plan_sha,'exact separately frozen study')
    guard_audit_clock()
    validate_plan(plan);c.verify(plan['inputs'])
    require(sum(path.stat().st_size for path in output.rglob('*') if path.is_file())<plan['maximum_output_bytes'],'actual retained evidence bound')
    claim,execution=read(output/'claim.json'),read(output/'execution.json')
    require(claim['plan_sha256']==execution['plan_sha256']==expected_plan_sha and claim['total_seconds']==plan['total_seconds'] and
        claim['model_seconds']==execution['model_seconds']==plan['model_seconds'] and claim['monotonic_origin']==execution['monotonic_origin'] and
        claim['unix_origin']==execution['unix_origin'] and 0<=execution['monotonic_end']-execution['monotonic_origin']<plan['total_seconds'] and
        0<=execution['unix_end']-execution['unix_origin']<plan['total_seconds'],'one unchanged absolute overall dual clock')
    require(execution['retries']==execution['replacements']==execution['training_updates']==0 and not execution['cleanup_errors'] and
        execution['execution_policy_id']==c.EXECUTION_POLICY_ID,'no hidden retries, training or cleanup failure')
    absent=prior.require_processes_absent(execution['observed_owned_identities']+[dict(pid=claim['pid'],created=claim['created'])])
    require(type(controller_session) is int and controller_session>0 and bool(controller_chunk),'actual retained controller exit supplied')
    base=dict(plan_sha256=expected_plan_sha,auditor=c.artifact(Path(__file__)),source_context=source_context_proof(),
        frozen_inputs_unchanged=len(plan['inputs']),exact_owned_processes_absent=absent,
        retained_controller_exit=dict(session_id=controller_session,exit_code=controller_exit_code,tool_chunk=controller_chunk,
            source='Actual retained execution supplied by root; exact owner absence independently checked'),
        auditor_models_launched=0,auditor_games_launched=0,training_updates=0)
    if execution['status']=='failed':
        require(allow_failure and controller_exit_code!=0 and read(output/'failure.json')==execution and execution['error'],
                'technical failure only in explicit preservation-only scope')
        return dict(**base,status='preserved_failed_attempt_verified',eligible=False,failure=execution['error'],
                    scope='Preserved failed evidence, original overall clock and exact owner absence; no collector acceptance')
    require(controller_exit_code==0 and execution['error'] is None and not(output/'failure.json').exists(),'actual successful controller exit')
    qualifier=audit_qualification(plan,output/'qualification');require(qualifier==read(output/'qualification/saved_audit.json') and
        qualification_gate(qualifier)['passed'],'all actual collector and target qualification independently reproduced')
    artifacts=[c.artifact(output/'qualification/saved_audit.json')];proofs=[];costs=[];names=['qualification'];incomplete=None
    require(read(output/'qualification_gate.json')==dict(qualification_gate(qualifier),plan_sha256=expected_plan_sha,source_audits=list(artifacts)),
            'one exact original prospective game admission')
    for spec in plan['cells']:
        guard_audit_clock()
        container=output/'cells'/spec['cell_id'];proof=audit_cell(plan,spec,container)
        require(proof==read(container/'saved_audit.json'),'every original game/capture/target/replay proof independently reconstructed')
        proofs.append(proof);names.append(spec['cell_id']);artifacts.append(c.artifact(container/'saved_audit.json'));costs.append(audit_cell_cost(proof)['artifact'])
        if proof['status']!='passed':incomplete=spec['cell_id'];break
    require({path.name for path in (output/'cells').iterdir() if path.is_dir()}==set(names[1:]),'all and only predeclared completed or failed stage prefix')
    stages=[]
    for name in names:
        row=read((output/'qualification' if name=='qualification' else output/'cells'/name)/'execution.json')
        stages.append(dict(stage_id=name,status=row['status'],worker_exit_code=row['worker_exit_code']))
    expected='completed_collector_comparison_pending_independent_acceptance' if incomplete is None else 'incomplete_collector_comparison_pending_independent_acceptance'
    require(execution['status']==expected and execution['incomplete_cell']==incomplete and execution['stages']==stages and
        execution['game_cells_started']==len(proofs),'exact entire study or first-failure prefix')
    comparison=comparison_summary(proofs)
    if incomplete is None:require(read(output/'comparison.json')==comparison and comparison['eligible'],'original complete-data throughput reconstructed')
    else:require(not(output/'comparison.json').exists() and not comparison['eligible'],'no complete-cohort rate from incomplete evidence')
    completion=read(output/'completion.json');require(completion['status']==expected and completion['execution']==c.artifact(output/'execution.json') and
        completion['plan']==c.artifact(c.PLAN) and completion['audited_stages']==len(artifacts) and completion['stage_audits']==artifacts and
        completion['cell_costs']==costs and completion['incomplete_cell']==incomplete and completion['training_updates']==0 and completion['inputs_unchanged'] is True,
        'completion binds actual verified stage inventory, cost receipts and owner exit')
    c.verify(plan['inputs'])
    guard_audit_clock()
    return dict(**base,status='saved_collector_comparison_verified' if incomplete is None else 'saved_incomplete_collector_verified',
        qualification=qualifier,cells=proofs,comparison=comparison,incomplete_cell=incomplete,independent_setup_seeds=32,
        generated_trajectories=sum(x['games'] or 0 for x in proofs),
        scope='Engineering collector throughput only; no training, playing-strength improvement, production promotion or one-second deployment claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-plan-sha',required=True);parser.add_argument('--controller-session',type=int,required=True)
    parser.add_argument('--controller-exit-code',type=int,required=True);parser.add_argument('--controller-chunk',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--allow-failure',action='store_true');args=parser.parse_args()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources();plan=read(c.PLAN);require(plan['source_guard']==sources,'saved audit once-startup corrected source guard')
    result=audit_study(plan,c.OUT,expected_plan_sha=args.expected_plan_sha,controller_session=args.controller_session,
        controller_exit_code=args.controller_exit_code,controller_chunk=args.controller_chunk,allow_failure=args.allow_failure)
    destination=Path(args.output).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    require(not destination.is_relative_to(c.OUT),'final independent acceptance outside frozen study evidence')
    with destination.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(c.canonical(dict(status=result['status'],output=c.artifact(destination),models=0,games=0)));return 0


def comparison_summary(proofs,*,costs=None):
    expected=['b1_cpu','b1_gpu','b2_gpu','b2_cpu']
    require([x['cell_id'] for x in proofs]==expected[:len(proofs)],'exact predeclared counterbalanced cell prefix')
    complete=len(proofs)==4 and all(x['status']=='passed' and x['full_cohort_verified'] and x['games']==16 for x in proofs)
    pairs=[];aggregates={}
    if complete:
        if costs is None:costs={x['cell_id']:audit_cell_cost(x) for x in proofs}
        for block,cpu,gpu in ((1,proofs[0],proofs[1]),(2,proofs[3],proofs[2])):
            require(cpu['route']=='cpu' and gpu['route']=='gpu' and cpu['block']==gpu['block']==block and
                    cpu['seed_decimals']==gpu['seed_decimals'],'same fresh setup seeds within each opposite-order block')
            require(all(type(x['wall_seconds']) in (int,float) and math.isfinite(x['wall_seconds']) and x['wall_seconds']>0 for x in (cpu,gpu)),
                    'positive complete original collection wall clocks')
            pairs.append(dict(block=block,cpu_seconds=cpu['wall_seconds'],gpu_seconds=gpu['wall_seconds'],
                gpu_over_cpu_games_per_hour=cpu['wall_seconds']/gpu['wall_seconds'],
                gpu_over_cpu_targets_per_hour=(gpu['eligible_roots']/gpu['wall_seconds'])/(cpu['eligible_roots']/cpu['wall_seconds']),
                cpu_plies=cpu['plies'],gpu_plies=gpu['plies'],cpu_targets=cpu['eligible_roots'],gpu_targets=gpu['eligible_roots'],
                exact_same_trajectories=cpu['trajectory_hashes']==gpu['trajectory_hashes'],
                cpu_verified_seconds=costs[cpu['cell_id']]['verified_seconds'],gpu_verified_seconds=costs[gpu['cell_id']]['verified_seconds'],
                gpu_over_cpu_verified_games_per_hour=costs[cpu['cell_id']]['verified_seconds']/costs[gpu['cell_id']]['verified_seconds'],
                passed_useful_threshold=costs[cpu['cell_id']]['verified_seconds']/costs[gpu['cell_id']]['verified_seconds']>=1.05))
        for route in ('cpu','gpu'):
            cells=[x for x in proofs if x['route']==route];seconds=sum(x['wall_seconds'] for x in cells);targets=sum(x['eligible_roots'] for x in cells)
            aggregates[route]=dict(games=32,collection_seconds=seconds,eligible_roots=targets,games_per_hour=32*3600/seconds,
                eligible_roots_per_hour=targets*3600/seconds,plies=sum(x['plies'] for x in cells),
                model_replay_seconds=sum(x['replay_seconds'] for x in cells),
                saved_audit_seconds=sum(costs[x['cell_id']]['audit_seconds'] for x in cells))
            verified=sum(costs[x['cell_id']]['verified_seconds'] for x in cells)
            aggregates[route].update(complete_verified_stage_seconds=verified,verified_games_per_hour=32*3600/verified,
                                     verified_targets_per_hour=targets*3600/verified)
    return dict(status='complete_collector_comparison' if complete else 'incomplete_collector_comparison',eligible=complete,
                pairs=pairs,aggregates=aggregates,
                pooled_gpu_over_cpu_games_per_hour=aggregates['gpu']['games_per_hour']/aggregates['cpu']['games_per_hour'] if complete else None,
                pooled_gpu_over_cpu_targets_per_hour=aggregates['gpu']['eligible_roots_per_hour']/aggregates['cpu']['eligible_roots_per_hour'] if complete else None,
                pooled_gpu_over_cpu_verified_games_per_hour=aggregates['gpu']['verified_games_per_hour']/aggregates['cpu']['verified_games_per_hour'] if complete else None,
                useful_improvement=complete and all(x['passed_useful_threshold'] for x in pairs),minimum_each_block_verified_rate_ratio=1.05,
                independent_setups=32,generated_trajectories=64,training_updates=0,
                scope='Descriptive complete verified-data throughput under matched setups; no strength or one-second deployment claim')


if __name__=='__main__':raise SystemExit(main())


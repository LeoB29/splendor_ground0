"""Owned live service with unchanged search and complete lossless evidence."""
from contextlib import contextmanager
import gzip
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from . import common as c
from .adapters import pack_result,unpack_result

ROLES=('qualification','operational','main','replay')
MAX_REQUESTS=c.GAMES_PER_BLOCK*c.MAX_PLIES+4096


def serve():
    from tools.local_player_qualification_128.clock import _decode
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from . import adapters
    folder=Path(os.environ['SIXWORLD_SERVICE_FOLDER']).resolve()
    expected=os.environ['SIXWORLD_FREEZE_SHA256'];role=os.environ['SIXWORLD_ROLE']
    c.require(folder.is_relative_to(c.OUT) and role in ROLES,'Owned live service boundary')
    folder.mkdir(parents=True,exist_ok=False)
    context=None;error=None;sequence=0;forensic=None;raw_forensic=None
    forensic_sequence=0;previous=None;request_id=None;requests=0
    def emit(packet):
        sys.stdout.write(c.canonical(packet)+'\n');sys.stdout.flush()
    def record(event):
        nonlocal forensic_sequence,previous
        row=dict(sequence=forensic_sequence,previous_sha256=previous,
            event=dict(transport_request_id=request_id,event=event))
        encoded=(c.canonical(row)+'\n').encode('utf-8')
        c.require(len(encoded)<=adapters.MAX_RAW_BYTES and forensic_sequence<MAX_REQUESTS,'Finite ordinary forensic evidence')
        forensic.write(encoded);forensic.flush();raw_forensic.flush();os.fsync(raw_forensic.fileno())
        previous=c.digest(row);forensic_sequence+=1
    @contextmanager
    def decision_guard():
        nonlocal sequence
        sequence+=1;ordinal=sequence;started=time.perf_counter();active_request=request_id;started_time=c.now()
        def expired():
            try:c.save(folder/'decision_timeout.json',dict(ordinal=ordinal,request_id=active_request,
                deadline_seconds=120.0,elapsed_seconds=time.perf_counter()-started,
                started_perf_counter=started,started_time=started_time,
                reason='sixworld512_decision_watchdog_120s',role=role,
                identity=c.identity(),freeze_sha256=expected,time=c.now()))
            finally:os._exit(124)
        timer=threading.Timer(120,expired);timer.daemon=True;timer.start()
        try:yield
        finally:timer.cancel()
    try:
        c.require(c.sha(c.PLAN)==expected,'Exact frozen live service')
        plan=c.read(c.PLAN);c.verify(plan['worker_inputs'])
        sources=verify_runtime_sources()
        c.require(sources==plan['corrected_source_receipt'],'Corrected service source guard')
        c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),**c.IDENTITY))
        profiles={q:c.read(c.resolve(item)) for q,item in plan['research_profiles'].items()}
        controls={q:c.read(c.resolve(item)) for q,item in plan['control_profiles'].items()}
        context=adapters.initialize(folder,profiles,sources,save_json=c.save,
            control_profiles=controls,profile_artifacts=plan['research_profiles'],
            control_profile_artifacts=plan['control_profiles'],decision_guard=decision_guard)
        context['qualification_enabled']=role=='qualification'
        raw_forensic=(folder/'ordinary_forensics.jsonl.gz').open('xb')
        forensic=gzip.GzipFile(fileobj=raw_forensic,mode='wb',filename='',mtime=0,compresslevel=1)
        emit(dict(type='ready',metadata=dict(identity=c.identity(),role=role,freeze_sha256=expected,
            live_schema=adapters.LIVE_SCHEMA,result_encoding=adapters.PACK_SCHEMA)))
        while True:
            raw=sys.stdin.buffer.readline(1024**2+1)
            if not raw:break
            context['request_started']=time.perf_counter();requests+=1
            c.require(requests<=MAX_REQUESTS and len(raw)<=1024**2 and raw.endswith(b'\n'),'Finite bounded live request framing')
            envelope=_decode(raw)
            c.require(set(envelope)=={'type','request_id','payload'} and envelope['type']=='request'
                and type(envelope['request_id']) is str,'Exact request envelope')
            payload=envelope['payload'];request_id=envelope['request_id']
            operation=payload.get('operation') if type(payload) is dict else None
            c.require(operation not in ('entrypoint_checks','qualification_decision') or role=='qualification',
                'Qualification route only in declared service')
            if operation in (None,'qualification_decision'):context['evidence_callback']=record
            else:context.pop('evidence_callback',None)
            result=adapters.handle(context,payload)
            packed=pack_result(result)
            encoded=c.canonical(dict(type='response',request_id=request_id,result=packed))+'\n'
            c.require(len(encoded.encode('utf-8'))<=16*c.MiB,'Bounded full compressed response wire')
            if operation in (None,'qualification_decision'):
                c.require(time.perf_counter()<context['request_started']+119,
                    'VARIABILITY_INTERNAL_DEADLINE: complete compressed serialized response exceeded119seconds')
            sys.stdout.write(encoded);sys.stdout.flush()
    except BaseException:
        error=traceback.format_exc()
        c.save(folder/'failure.json',dict(error=error,request_id=request_id,time=c.now()))
        emit(dict(type='error',error=error[-4096:]))
    finally:
        try:
            if forensic is not None:forensic.close()
            if raw_forensic is not None:
                raw_forensic.flush();os.fsync(raw_forensic.fileno());raw_forensic.close()
                c.save(folder/'forensic_summary.json',dict(events=forensic_sequence,
                    terminal_sha256=previous,artifact=c.artifact(folder/'ordinary_forensics.jsonl.gz'),
                    lossless=True,schema='sixworld512_compressed_forensic_chain_v1'))
        except BaseException:error=(error or '')+traceback.format_exc()
        if context is not None:
            try:adapters.close_context(context)
            except BaseException:error=(error or '')+traceback.format_exc()
        c.save(folder/'service_exit.json',dict(status='passed' if error is None else 'failed',
            error=error,identity=c.identity(),guarded_decisions=sequence,requests=requests,time=c.now()))
    return 0 if error is None else 1


if __name__=='__main__':raise SystemExit(serve())

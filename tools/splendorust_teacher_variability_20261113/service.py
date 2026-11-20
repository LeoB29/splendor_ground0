"""One immutable model service; trusted response clock lives in its parent."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from . import common as c


def serve():
    from tools.local_player_qualification_128.clock import _decode
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from . import adapters
    folder=Path(os.environ['VARIABILITY_SERVICE_FOLDER']).resolve()
    expected=os.environ['VARIABILITY_FREEZE_SHA256'];role=os.environ['VARIABILITY_ROLE']
    c.require(folder.is_relative_to(c.OUT) and role in c.MODEL_JOBS,'Owned service boundary')
    folder.mkdir(parents=True,exist_ok=False)
    context=None;error=None;sequence=0;forensic=None;forensic_sequence=0;previous=None;request_id=None
    def emit(packet):
        sys.stdout.write(c.canonical(packet)+'\n');sys.stdout.flush()
    def record(event):
        nonlocal forensic_sequence,previous
        row=dict(sequence=forensic_sequence,previous_sha256=previous,
            event=dict(transport_request_id=request_id,event=event))
        encoded=c.canonical(row)+'\n'
        c.require(len(encoded.encode())<=16*1024**2 and forensic_sequence<6000,'Finite ordinary forensic evidence')
        forensic.write(encoded);forensic.flush();os.fsync(forensic.fileno())
        previous=c.digest(row);forensic_sequence+=1
    @contextmanager
    def decision_guard():
        # Nested qualification decisions receive an independent real watchdog.
        # Ordinary requests additionally have the parent arrival clock.
        nonlocal sequence
        sequence+=1;ordinal=sequence;started=time.perf_counter()
        def expired():
            try:c.save(folder/'decision_timeout.json',dict(ordinal=ordinal,seconds=120,started=started,time=c.now()))
            finally:os._exit(124)
        timer=threading.Timer(120,expired);timer.daemon=True;timer.start()
        try:yield
        finally:timer.cancel()
    try:
        c.require(c.sha(c.PLAN)==expected,'Exact frozen service')
        plan=c.read(c.PLAN);c.verify(plan['worker_inputs'])
        sources=verify_runtime_sources()
        c.require(sources==plan['corrected_source_receipt'],'Corrected service source guard')
        c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),**c.IDENTITY))
        profiles={q:c.read(c.resolve(item)) for q,item in plan['research_profiles'].items()}
        controls={q:c.read(c.resolve(item)) for q,item in plan['control_profiles'].items()}
        context=adapters.initialize(folder,profiles,sources,save_json=c.save,
            control_profiles=controls,profile_artifacts=plan['research_profiles'],
            control_profile_artifacts=plan['control_profiles'],decision_guard=decision_guard)
        forensic=(folder/'ordinary_forensics.jsonl').open('x',encoding='utf-8',newline='\n')
        emit(dict(type='ready',metadata=dict(identity=c.identity(),role=role,freeze_sha256=expected)))
        while True:
            raw=sys.stdin.buffer.readline(1024**2+1)
            if not raw:break
            context['request_started']=time.perf_counter()
            c.require(len(raw)<=1024**2 and raw.endswith(b'\n'),'Bounded request framing')
            envelope=_decode(raw)
            c.require(set(envelope)=={'type','request_id','payload'} and envelope['type']=='request'
                and type(envelope['request_id']) is str,'Exact request envelope')
            payload=envelope['payload']
            request_id=envelope['request_id']
            operation=payload.get('operation') if type(payload) is dict else None
            c.require(operation!='entrypoint_checks' or role=='qualification','Finite qualification route')
            context['qualification_active']=role=='qualification' and operation is None
            if operation is None:context['evidence_callback']=record
            else:context.pop('evidence_callback',None)
            result=adapters.handle(context,payload)
            packet=dict(type='response',request_id=envelope['request_id'],result=result)
            encoded=c.canonical(packet)+'\n'
            if operation is None:
                c.require(time.perf_counter()<context['request_started']+119,'Complete serialized internal response deadline')
            sys.stdout.write(encoded);sys.stdout.flush()
    except BaseException:
        error=traceback.format_exc()
        c.save(folder/'failure.json',dict(error=error,time=c.now()))
        emit(dict(type='error',error=error[-4096:]))
    finally:
        if forensic is not None:forensic.close()
        if context is not None:
            try:adapters.close_context(context)
            except BaseException:error=(error or '')+traceback.format_exc()
        c.save(folder/'service_exit.json',dict(status='passed' if error is None else 'failed',
            error=error,identity=c.identity(),guarded_decisions=sequence,time=c.now()))
    return 0 if error is None else 1


if __name__=='__main__':raise SystemExit(serve())

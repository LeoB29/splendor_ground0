"""Separate repeat-study entry point; exact frozen live player and transport."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import ROOT,digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent,CHECKPOINT_SHA256
from tools.local_player_qualification_128.clock import THREAD_ENV,_decode
from tools.splendorust_search_scale_20261024.worker import _save,_forensic_recorder,_forensic_action,decode_observer
from tools.splendorust_inference_opt_20261025 import graph
from tools.splendorust_inference_opt_20261025.worker import graph_parity as original_graph_parity
from tools.splendorust_boundary_timing_20261025.worker import make_evidence,save_sidecar_error
from tools.splendorust_boundary_timing_20261025.child_profile import SCHEMA,MAX_BATCH_PROFILES,publish_profile
from . import adapters,qualification
from .adapters import require,INTERNAL_SECONDS

MAX_RESPONSE_BYTES=4*1024**2

def response_packet(value):
    encoded=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    # Windows text stdout may expand the final newline to CRLF.
    require(len(encoded.encode('utf-8'))+1<=MAX_RESPONSE_BYTES,'Research response exceeds4MiB transport cap')
    return encoded

def initialize():
    sources=verify_runtime_sources()  # Exactly once before loading profiles or model.
    from . import common as c
    folder=Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder.is_relative_to(c.OUT.resolve()),'New worker evidence boundary')
    require(os.environ.get('STUDY_BOUNDARY_PROFILE')=='1','Boundary observer required')
    freeze_path=Path(os.environ['STUDY_RESEARCH_FREEZE']).resolve()
    require(freeze_path==c.OUT/'freeze.json' and c.file_hash(freeze_path)==os.environ['STUDY_RESEARCH_FREEZE_SHA256'],
            'Exact execution freeze binding')
    freeze=c.read(freeze_path)
    require(freeze.get('phase')==c.PHASE==os.environ.get('SPLENDOR_POOL6_CORECOMPARE01_PHASE','corecompare01'),
            'Execution phase/freeze/environment binding')
    descriptors=freeze['research_profiles']
    require(set(descriptors)==set(adapters.PROFILE_KEYS),'Frozen sampling profile inventory')
    c.verify(list(descriptors.values()))
    profiles={q:c.read(ROOT/item['path']) for q,item in descriptors.items()}
    for text,profile in profiles.items():
        q,pool=map(int,text.split('-'))
        require(profile==adapters.profile(q,pool),'Frozen sampling profile full identity')
    runtime=dict(sources=sources,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)
    _save(folder/'runtime_sources.json',runtime)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    instances=adapters.initialize_adapters(folder,profiles)
    context=dict(adapters=instances,profiles=profiles,profile_artifacts=descriptors,folder=folder,quota=64,
        diagnostic_capture=False,request_started=None,component_profiles=[],boundary_handle_active=False,
        component_backend='research_graph',graph_parity_passed=False,graph_parity_attempted=False,
        inference_binding=SimpleNamespace(original=Opponent._infer))
    try:
        forward=graph.build_graph(instances['search'].model,folder,_save)
        for instance in instances.values(): instance.forward=forward
        context.update(frozen_forward=forward,component_backend_identity=digest(forward.identity))
        metadata=dict(schema=adapters.SCHEMA,route='research',phase=c.PHASE,pid=os.getpid(),threads=1,
            inference_backend=graph.BACKEND,checkpoint_sha256=CHECKPOINT_SHA256,
            native_pids={role:instance.proc.pid for role,instance in instances.items()},
            research_profiles={q:item['sha256'] for q,item in descriptors.items()},
            native_binary_sha256=profiles['64-6']['native_binary_sha256'],
            request_quotas=list(adapters.QUOTAS),qualification_reference_quota=16,
            world_pools=list(adapters.WORLD_POOLS),seed_labels=list(adapters.SEED_LABELS),qualification_reference_world_pool=3,
            telemetry_enabled=False,live_turns_unrestricted=True,
            internal_seconds=INTERNAL_SECONDS,freeze_sha256=os.environ['STUDY_RESEARCH_FREEZE_SHA256'],**runtime)
        context['metadata']=metadata
        _save(folder/'player_identity.json',metadata)
        _save(folder/'inference_execution_identity.json',dict(backend=graph.BACKEND,
            graph_identity_sha256=digest(forward.identity),metadata=metadata))
        _save(folder/'boundary_timing_identity.json',dict(schema=SCHEMA,backend='research_graph',quota=64,
            variable_request_quota=True,variable_world_pool=False,graph_identity_sha256=context['component_backend_identity'],
            metadata=metadata,scientific_response_fields_unchanged=False,
            additive_research_response=True,clock='perf_counter_ns',
            clock_implementation=time.get_clock_info('perf_counter').implementation,
            evidence='atomic exclusive request sidecars after successful stdout flush'))
        return context
    except BaseException:
        close_context(context);raise

def _ordinary(context,request):
    require(type(request) is dict and set(request)=={'player','observer'},'Fixed live request allowlist')
    arm=request['player']
    require(arm=='search' or context['diagnostic_capture'] and arm=='reference','Live arm allowlist')
    q=16 if arm=='reference' else 64;pool=adapters.pool_for(q)
    context['quota']=q
    observer=request['observer'];state=decode_observer(observer)
    # Late games are lawful live inputs. The unchanged native selector retains
    # its existing turn>=124 heuristic; this wrapper must not reject that path.
    seed=adapters.observer_seed(observer,'A')
    legal=rules.CorrectedEnv().legal_actions(state)
    require(bool(legal),'Empty complete support')
    started=context['request_started']
    if started is None: started=time.perf_counter()
    deadline=started+INTERNAL_SECONDS
    instance=context['adapters'][arm];instance.request_deadline=deadline
    try:
        action,trace=instance.choose(observer,diagnostic_capture=True)
        research=dict(schema=adapters.SCHEMA,quota=q,world_pool=pool,seed_label='A',seed=seed,
            telemetry_enabled=False,internal_seconds=INTERNAL_SECONDS,
            native_binary_sha256=context['profiles'][f'{q}-{pool}']['native_binary_sha256'],
            phases=copy.deepcopy(instance.last_research_phases))
        if context.get('evidence_callback') is not None:
            context['evidence_callback'](dict(event='complete_research_result_before_worker_validation',
                arm=arm,observer=observer,action=_forensic_action(action),diagnostics=trace,research=research))
        require(action in legal,'PLAYER_ILLEGAL_ACTION: research action outside corrected support')
        result=dict(player=arm,action=rules.serialize_action(action),diagnostics=trace,
            profile_id=adapters.profile_id(q,pool),profile_sha256=context['profile_artifacts'][f'{q}-{pool}']['sha256'],
            wrapper_id=adapters.wrapper_id(q,pool),simulations_quota=q,observer_sha256=digest(observer),
            legal_actions_sha256=digest([rules.serialize_action(a) for a in legal]),legal_count=len(legal),
            ruleset=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,research=research)
        json.dumps(result,separators=(',',':'),allow_nan=False)
        if time.perf_counter()>=deadline:
            raise TimeoutError('SAMPLING_INTERNAL_DEADLINE: complete response exceeded0.900s')
        return result
    except BaseException as error:
        if context.get('evidence_callback') is not None:
            context['evidence_callback'](dict(event='worker_research_decision_failed',arm=arm,observer=observer,
                error=repr(error),last_phase_callbacks=instance.callbacks,
                partial_research_phases=instance.last_research_phases,last_callback_bits=instance.callback_bits))
        raise
    finally:
        instance.request_deadline=None

def handle(context,request):
    if type(request) is dict and request.get('operation')=='graph_parity':
        return original_graph_parity(context,request)
    require(context['graph_parity_passed'],'Research decisions require graph parity')
    if request=={'operation':'entrypoint_checks'}:
        import psutil
        def boundary():
            require(psutil.virtual_memory().available>=2*1024**3,'Qualification available RAM below2GiB')
        require(not context['diagnostic_capture'],'Nested qualification')
        context['diagnostic_capture']=True
        outer=context['request_started'];context['request_started']=None
        for instance in context['adapters'].values(): instance.admission_guard=boundary
        record,stream=_forensic_recorder(context['folder']/'entrypoint_forensics.jsonl')
        context['evidence_callback']=record
        try:
            boundary()
            return qualification.run_entrypoint_checks(context,lambda payload:handle(context,payload),
                boundary=boundary,evidence_callback=record)
        except BaseException as error:
            record(dict(event='entrypoint_failed',error=repr(error),traceback=traceback.format_exc()));raise
        finally:
            stream.close();context.pop('evidence_callback',None)
            for instance in context['adapters'].values(): instance.admission_guard=None
            context['diagnostic_capture']=False;context['request_started']=outer;context['quota']=64
    require(not context['boundary_handle_active'],'Nested research handle')
    require(len(context['component_profiles'])<MAX_BATCH_PROFILES,'Boundary profile count bound')
    context['boundary_handle_active']=True
    failed=True;started=time.perf_counter_ns()
    try:
        result=_ordinary(context,request);failed=False
        context['quota']=result['simulations_quota']
        return result
    finally:
        ended=time.perf_counter_ns();context['boundary_handle_active']=False
        context['component_profiles'].append(dict(schema=SCHEMA,pid=os.getpid(),clock='perf_counter_ns',
            measurement_units='nanoseconds',request_read_complete_ns=context.get('request_read_complete_ns')
                if context.get('request_started') is not None else None,
            handle_start_ns=started,handle_end_ns=ended,handle_failed=failed,
            times_are_diagnostic_not_trusted_deadline_receipts=True))

def close_context(context):
    for instance in context.get('adapters',{}).values(): instance.close()

def serve(*,max_request_bytes=1024*1024):
    for name in THREAD_ENV: os.environ[name]='1'
    from threadpoolctl import threadpool_limits
    def emit(value,deadline=None,markers=None):
        encoded=response_packet(value)
        if deadline is not None and time.perf_counter()>=deadline:
            raise TimeoutError('SAMPLING_INTERNAL_DEADLINE: response serialization exceeded0.900s')
        before=time.perf_counter_ns()
        if markers is not None: markers['pre_write_ns']=before
        sys.stdout.write(encoded);sys.stdout.flush()
        after=time.perf_counter_ns()
        if markers is not None: markers['post_flush_ns']=after
        return before,after,encoded
    context=None
    with threadpool_limits(limits=1):
        try:
            context=initialize();emit(dict(type='ready',metadata=context['metadata']))
            while True:
                raw=sys.stdin.buffer.readline(max_request_bytes+1)
                if not raw:return 0
                context['request_started']=time.perf_counter()
                context['request_read_complete_ns']=time.perf_counter_ns()
                context['component_profiles']=[]
                request_id=payload=result=None;markers={}
                try:
                    require(len(raw)<=max_request_bytes and raw.endswith(b'\n'),'Request size/termination')
                    request=_decode(raw)
                    require(set(request)=={'type','request_id','payload'} and request['type']=='request'
                        and type(request['request_id']) is str,'Transport envelope')
                    request_id,payload=request['request_id'],request['payload']
                    result=handle(context,payload)
                    startup=payload=={'operation':'entrypoint_checks'} or type(payload) is dict and set(payload)=={'operation','fixtures'} and payload['operation']=='graph_parity'
                    deadline=None if startup else context['request_started']+INTERNAL_SECONDS
                    pre,post,encoded=emit(dict(type='response',request_id=request_id,result=result),deadline,markers)
                except BaseException as error:
                    _save(context['folder']/'worker_error.json',dict(request_id=request_id,error=repr(error),traceback=traceback.format_exc(),
                        partial_adapters={key:dict(callbacks=value.callbacks,callback_bits=value.callback_bits,
                            phases=value.last_research_phases) for key,value in context['adapters'].items()}))
                    if request_id is not None:
                        try:
                            evidence=make_evidence(context,request_id,raw,payload,result,
                                outcome='failed_before_successful_emission',error=repr(error)[:4096],
                                pre_write_ns=markers.get('pre_write_ns'),post_flush_ns=markers.get('post_flush_ns'))
                            publish_profile(context['folder'],request_id,evidence,batch=len(context['component_profiles'])>1)
                        except BaseException as caught: save_sidecar_error(context,request_id,caught,primary_error=repr(error))
                    emit(dict(type='error',request_id=request_id,error=repr(error)[:4096]));return 1
                try:
                    evidence=make_evidence(context,request_id,raw,payload,result,outcome='response_flushed',
                        pre_write_ns=pre,post_flush_ns=post,encoded=encoded)
                    publish_profile(context['folder'],request_id,evidence,batch=startup)
                except BaseException as error:
                    save_sidecar_error(context,request_id,error);return 1
        except BaseException as error:
            emit(dict(type='error',error=repr(error)[:4096]));return 1
        finally:
            if context is not None:close_context(context)

if __name__=='__main__':
    raise SystemExit(serve())

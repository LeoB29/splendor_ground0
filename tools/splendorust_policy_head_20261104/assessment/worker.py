"""Parent or fixed learned checkpoint at128, exact graph and full evidence."""
from __future__ import annotations
import copy
import json
import os
import re
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
from . import graph
from tools.splendorust_inference_opt_20261025.worker import graph_parity as original_graph_parity
from tools.splendorust_boundary_timing_20261025.worker import make_evidence,save_sidecar_error
from tools.splendorust_boundary_timing_20261025.child_profile import SCHEMA,MAX_BATCH_PROFILES,publish_profile
from . import adapters,corpus,value_invariance
from .adapters import require,INTERNAL_SECONDS

MAX_RESPONSE_BYTES=8*1024**2

def response_packet(value):
    encoded=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    # Windows text stdout may expand the final newline to CRLF.
    require(len(encoded.encode('utf-8'))+1<=MAX_RESPONSE_BYTES,'Research response exceeds8MiB transport cap')
    return encoded

def lifecycle_route(folder,freeze,*,route=None):
    """A declared lifetime has one immutable quota; no implicit reused worker."""
    from . import common as c
    folder=Path(folder).resolve();name=folder.name
    require(re.fullmatch(r'[a-z0-9_]+',name) is not None and folder==(c.OUT/'workers'/name).resolve(),
        'Exact worker evidence folder')
    planned=freeze['declared_worker_lifecycles']
    require(type(planned) is dict and name in planned,'Explicit frozen worker lifetime')
    item=planned[name];expected=item['route'];quota=adapters.route_quota(expected)
    require(route is None or route==expected,'Worker route and declared lifetime agree')
    require(type(item['quota']) is int and item['quota']==quota and
        item['startup_requests']==2 and item['graph_request_id']=='request-00000001' and
        item['entrypoint_request_id']=='request-00000002' and
        item['ordinary_first_request_id']=='request-00000003' and
        type(item['ordinary_requests']) is int and 0<=item['ordinary_requests']<=c.MAX_LIVE_DECISIONS,
        'Frozen quota, qualification prefix and finite ordinary suffix')
    return expected


def execution_identity(context,metadata):
    return dict(schema='splendorust_selfplay_assessment_execution_identity_v1',
        worker_lifecycle=context['folder'].name,route=context['route'],quota=context['quota'],
        original_full_network=True,network_implementation_changed=False,numerical_tolerance_relaxation=False,
        graph_identity_sha256=context['component_backend_identity'],metadata=metadata)


def verify_execution_identity(folder,freeze,metadata):
    from . import common as c
    folder=Path(folder);route=lifecycle_route(folder,freeze,route=metadata['route'])
    identity=c.read(folder/'graph_identity.json');graph_sha=c.digest(identity)
    require(identity['checkpoint_sha256']==freeze['checkpoint_bindings'][route]['checkpoint']['sha256'] and
        identity['weights_before_sha256']==identity['weights_after_sha256']==freeze['checkpoint_bindings'][route]['weights_sha256'],
        'Original exact full network and weights')
    expected=execution_identity(dict(folder=folder,route=route,quota=adapters.route_quota(route),
        component_backend_identity=graph_sha),metadata)
    require(c.read(folder/'search_execution_identity.json')==expected,'Declared original-network execution identity')
    require(not any((folder/name).exists() for name in ('candidate_ir.txt','candidate_identity.json',
        'practical_execution_identity.json','numerical_checks.json')),'No alternative graph or relaxed numerical execution')
    return dict(status='passed',route=route,quota=adapters.route_quota(route),worker_lifecycle=folder.name,
        graph_identity_sha256=graph_sha,original_full_network=True)


def metadata_route(folder):
    from . import common as c
    return c.read(Path(folder)/'player_identity.json')['route']


def verify_graph_admission(folder,freeze,result,*,boundary=lambda:None):
    """Recompute exact arrays and float32-bit comparisons, without model calls."""
    from . import common as c
    from tools.splendorust_boundary_timing_20261025.runner import validate_graph_parity
    boundary();c.verify([freeze['graph_parity_plan']])
    plan=c.read(c.ROOT/freeze['graph_parity_plan']['path']);fixtures=plan['fixtures']
    validate_graph_parity(result,fixtures)
    saved=c.read(Path(folder)/'graph_parity.json')
    require(saved['schema']==graph.PARITY_SCHEMA and saved['summary']==result and
        saved['fixtures']==fixtures and len(saved['evidence'])==11,'Exact graph case inventory and response')
    for fixture,observed,row in zip(fixtures,saved['evidence'],result['rows'],strict=True):
        boundary()
        require(observed['fixture_id']==fixture['fixture_id'] and observed['batch']==fixture['batch'],
            'Graph fixture identity')
        comparison=graph.compare_outputs(fixture,observed['eager_outputs'],observed['candidate_outputs'])
        require(comparison==row==observed['comparison'] and row['exact_float_equal'] is True and
            row['float32_bits_equal'] is True,'Full exact original graph outputs and bits')
        provenance=plan['provenance_by_fixture'][fixture['fixture_id']]
        if metadata_route(folder)=='control' and 'historical_outputs' in provenance:
            require(observed['eager_outputs']==provenance['historical_outputs'] and
                row['eager_output_sha256']==provenance['historical_output_sha256'],'Original historical graph outputs')
    invariant=value_invariance.graph_proof(Path(folder),freeze,boundary=boundary)
    require(c.read(Path(folder)/'value_invariance_graph.json')==invariant,'Saved actual graph value-invariance proof')
    return dict(status='passed',accepted=True,fixtures=11,graph_identity_sha256=result['graph_identity_sha256'],
        original_value_invariance=invariant)


def initialize(route='control'):
    sources=verify_runtime_sources()  # Exactly once before any model/profile load.
    from . import common as c
    require(route in adapters.ROUTES,'Explicit numerical route')
    require(c.INTERNAL_SECONDS==INTERNAL_SECONDS and c.SECONDS_PER_MOVE==adapters.SECONDS_PER_MOVE,'Frozen clock policy')
    folder=Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder.parent==(c.OUT/'workers').resolve(),'New worker evidence boundary')
    require(os.environ.get('STUDY_BOUNDARY_PROFILE')=='1','Boundary observer required')
    freeze_path=Path(os.environ['STUDY_RESEARCH_FREEZE']).resolve()
    require(freeze_path==c.OUT/'freeze.json' and c.file_hash(freeze_path)==os.environ['STUDY_RESEARCH_FREEZE_SHA256'],
            'Exact execution freeze binding')
    freeze=c.read(freeze_path)
    lifecycle_route(folder,freeze,route=route)
    require(freeze.get('phase')==c.PHASE==os.environ.get('SELFPLAY_JOB_SCOPE'),
            'Execution phase/freeze/environment binding')
    descriptors=adapters.profile_descriptors(freeze,route)
    require(set(descriptors)==({'16',str(adapters.route_quota(route))}),'Frozen profile inventory')
    c.verify(list(descriptors.values()))
    profiles={key:c.read(ROOT/item['path']) for key,item in descriptors.items()}
    identities={q:adapters.profile_identity(freeze,route,q) for q in (16,adapters.route_quota(route))}
    runtime=dict(sources=sources,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)
    _save(folder/'runtime_sources.json',runtime)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    instances=adapters.initialize_adapters(folder,profiles,route=route,freeze=freeze)
    context=dict(adapters=instances,profiles=profiles,profile_artifacts=descriptors,folder=folder,quota=adapters.route_quota(route),route=route,
        execution_identities=identities,freeze=freeze,diagnostic_capture=False,request_started=None,
        component_profiles=[],boundary_handle_active=False,
        component_backend='qualified_graph64',
        graph_parity_passed=False,graph_parity_attempted=False,entrypoint_attempted=False,entrypoint_passed=False,
        inference_binding=SimpleNamespace(original=Opponent._infer))
    try:
        forward=graph.build_graph(instances['search'].model,folder,_save,checkpoint_sha256=freeze['checkpoint_bindings'][route]['checkpoint']['sha256'])
        for instance in instances.values():instance.forward=forward
        context.update(frozen_forward=forward,component_backend_identity=digest(forward.identity))
        require(forward.identity['weights_before_sha256']==freeze['checkpoint_bindings'][route]['weights_sha256'],
            'Exact frozen checkpoint tensors in unchanged full-network graph')
        metadata=adapters.ready_metadata(route,freeze,pid=os.getpid(),native_pids={key:instance.proc.pid for key,instance in instances.items()},
            sources=sources,freeze_sha256=os.environ['STUDY_RESEARCH_FREEZE_SHA256'],graph_identity_sha256=digest(forward.identity))
        context['metadata']=metadata
        _save(folder/'search_execution_identity.json',execution_identity(context,metadata))
        _save(folder/'player_identity.json',metadata)
        _save(folder/'inference_execution_identity.json',dict(backend=graph.BACKEND,
            graph_identity_sha256=digest(forward.identity),metadata=metadata))
        _save(folder/'boundary_timing_identity.json',dict(schema=SCHEMA,backend=context['component_backend'],quota=adapters.route_quota(route),
            variable_request_quota=True,variable_world_pool=False,graph_identity_sha256=context['component_backend_identity'],
            metadata=metadata,scientific_response_fields_unchanged=False,execution_policy_changed=True,
            additive_research_response=False,clock='perf_counter_ns',
            clock_implementation=time.get_clock_info('perf_counter').implementation,
            evidence='atomic exclusive request sidecars after successful stdout flush'))
        return context
    except BaseException:
        close_context(context);raise


def _ordinary(context,request):
    # Original baseline handler body with only explicit profile/clock bindings.
    started=context.get('request_started')
    if started is None:started=time.perf_counter()
    deadline=started+INTERNAL_SECONDS
    require(type(request) is dict and set(request)=={'player','observer'} and request['player'] in ('search','reference'),'Request allowlist')
    require(request['player']=='search' or context['diagnostic_capture'],'Reference only during qualification')
    observer,arm=request['observer'],request['player']
    state=decode_observer(observer);legal=rules.CorrectedEnv().legal_actions(state)
    require(bool(legal),'Empty complete-action support')
    instance=context['adapters'][arm];instance.request_deadline=deadline
    q=adapters.route_quota(context['route']) if arm=='search' else 16;identity=context['execution_identities'][q]
    try:
        action,trace=instance.choose(observer,diagnostic_capture=True)
        if context.get('evidence_callback') is not None:
            context['evidence_callback'](dict(event='complete_result_before_worker_validation',arm=arm,
                observer=observer,action=_forensic_action(action),diagnostics=trace))
        require(action in legal,'PLAYER_ILLEGAL_ACTION: returned action outside corrected support')
        require(trace['profile_id']==identity['profile_id'],'Scientific execution identity')
        result=dict(player=arm,action=rules.serialize_action(action),diagnostics=trace,
            profile_id=identity['profile_id'],profile_sha256=identity['profile_sha256'],wrapper_id=identity['wrapper_id'],
            simulations_quota=q,observer_sha256=digest(observer),
            legal_actions_sha256=digest([rules.serialize_action(a) for a in legal]),legal_count=len(legal),
            ruleset=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)
        json.dumps(result,separators=(',',':'),allow_nan=False)
        if time.perf_counter()>=deadline:raise TimeoutError('STRENGTH_INTERNAL_DEADLINE: complete response exceeded9s')
        return result
    except BaseException as error:
        if context.get('evidence_callback') is not None:
            context['evidence_callback'](dict(event='worker_decision_failed',arm=arm,observer=observer,
                error=repr(error),last_phase_callbacks=getattr(instance,'callbacks',None)))
        raise
    finally:instance.request_deadline=None


def handle(context,request):
    if type(request) is dict and request.get('operation')=='graph_parity':
        result=original_graph_parity(context,request)
        _save(context['folder']/'value_invariance_graph.json',
            value_invariance.graph_proof(context['folder'],context['freeze']))
        return result
    require(context['graph_parity_passed'],'Research decisions require graph parity')
    if request=={'operation':'entrypoint_checks'}:
        import psutil
        def boundary():
            return None  # Available host RAM is advisory; external actual guards remain.
        require(not context['diagnostic_capture'] and not context['entrypoint_attempted'],'Single actual qualification prefix')
        context['entrypoint_attempted']=True
        context['diagnostic_capture']=True
        outer=context['request_started'];context['request_started']=None
        for instance in context['adapters'].values(): instance.admission_guard=boundary
        record,stream=_forensic_recorder(context['folder']/'entrypoint_forensics.jsonl')
        context['evidence_callback']=record
        try:
            boundary()
            result=corpus.run_entrypoint_checks(context['adapters'],request_callback=lambda payload:handle(context,payload),
                boundary=boundary,evidence_callback=record,route=context['route'])
            require(result['counts']['complete_decisions']==67,'Complete actual entrypoint suite')
            _save(context['folder']/'value_invariance_native.json',
                value_invariance.native_proof(result,context['freeze'],boundary=boundary))
            context['entrypoint_passed']=True
            return result
        except BaseException as error:
            record(dict(event='entrypoint_failed',error=repr(error),traceback=traceback.format_exc()));raise
        finally:
            stream.close();context.pop('evidence_callback',None)
            for instance in context['adapters'].values(): instance.admission_guard=None
            context['diagnostic_capture']=False;context['request_started']=outer;context['quota']=adapters.route_quota(context['route'])
    require(context['entrypoint_passed'] or context['diagnostic_capture'],'Actual corrected entrypoint qualification before play')
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

def serve(*,route='control',max_request_bytes=1024*1024):
    for name in THREAD_ENV: os.environ[name]='1'
    from threadpoolctl import threadpool_limits
    def emit(value,deadline=None,markers=None):
        encoded=response_packet(value)
        if deadline is not None and time.perf_counter()>=deadline:
            raise TimeoutError('STRENGTH_INTERNAL_DEADLINE: response serialization exceeded9s')
        before=time.perf_counter_ns()
        if markers is not None: markers['pre_write_ns']=before
        sys.stdout.write(encoded);sys.stdout.flush()
        after=time.perf_counter_ns()
        if markers is not None: markers['post_flush_ns']=after
        return before,after,encoded
    context=None
    with threadpool_limits(limits=1):
        try:
            context=initialize(route);emit(dict(type='ready',metadata=context['metadata']))
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
                        partial_adapters={key:dict(callbacks=value.callbacks,callback_bits=getattr(value,'callback_bits',None),
                            phases=getattr(value,'last_research_phases',None)) for key,value in context['adapters'].items()}))
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
    raise SystemExit(serve(route=os.environ.get('STUDY_ROUTE','control')))

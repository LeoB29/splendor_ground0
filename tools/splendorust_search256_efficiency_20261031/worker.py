"""Separate optional timing observer around the exact accepted256 player.

All scientific decisions call the immutable accepted ordinary handler directly.
Component observations never change deadlines, outputs, precision or search.
"""
from __future__ import annotations
import copy
import json
import hashlib
import math
import re
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
from tools.splendorust_search256_recovery_20261030_02 import adapters,corpus,worker as accepted
from tools.splendorust_search256_recovery_20261030_02.adapters import require,INTERNAL_SECONDS

MAX_RESPONSE_BYTES=4*1024**2
EFFICIENCY_SCHEMA='splendorust_search256_efficiency_v1'
EFFICIENCY_LABELS=('compiled_forward','inference_bridge','callback_total',
    'native_exchange','native_read_wait','native_write','native_flush')
EFFICIENCY_FIELDS=('calls','inclusive_wall_ns','inclusive_thread_cpu_ns',
    'exclusive_wall_ns','exclusive_thread_cpu_ns','max_wall_ns','failures')
MAX_EFFICIENCY_BYTES=16*1024
MAX_EFFICIENCY_BATCH_BYTES=1024*1024
_MISSING=object()


def observer_mode(freeze,folder,value):
    require(value in ('off','on'),'Explicit efficiency observer mode')
    require(freeze.get('efficiency_observers')=={'observer_off':'off','observer_on':'on'},
        'Frozen exact two-worker observer inventory')
    require(freeze['efficiency_observers'].get(folder.name)==value,'Worker observer identity')
    return value


def efficiency_identity(mode,metadata,graph_identity):
    return dict(schema=EFFICIENCY_SCHEMA,observer_mode=mode,profile_enabled=mode=='on',
        metadata=metadata,graph_identity_sha256=digest(graph_identity),labels=list(EFFICIENCY_LABELS),
        scientific_handler='tools.splendorust_search256_recovery_20261030_02.worker._ordinary',
        scientific_output_changed=False,clock_policy_changed=False,native_binary_changed=False,
        wall_clock='perf_counter_ns',cpu_clock='thread_time_ns',
        thread_cpu_clock_implementation=time.get_clock_info('thread_time').implementation,
        thread_cpu_clock_resolution_seconds=time.get_clock_info('thread_time').resolution,
        native_cpu='selected_native_process_user_plus_system_boundary_delta',
        per_inference_events=False,extra_host_sampler=False,
        efficiency_publication='exclusive_fsync_after_scientific_flush_before_atomic_child_profile')


class NativeCPUProbe:
    """Two process CPU snapshots per handler; never a sampler or clock correction."""
    def __init__(self,pid):
        import psutil
        self.pid=pid
        self.created=psutil.Process(pid).create_time()
        self.read()  # Availability is qualified at initialization, before requests.

    def read(self):
        import psutil
        process=psutil.Process(self.pid)
        require(process.create_time()==self.created,'Native CPU process identity changed')
        counters=process.cpu_times()
        values=(counters.user,counters.system)
        require(all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in values),
            'Native CPU counters unavailable or malformed')
        return round(sum(values)*1_000_000_000)


class EfficiencyProfiler:
    """Bounded nested aggregates, preserving the original scientific return value."""
    def __init__(self,probe,*,wall_clock=time.perf_counter_ns,cpu_clock=time.thread_time_ns):
        self.probe=probe;self.wall_clock=wall_clock;self.cpu_clock=cpu_clock
        self.totals={label:[0]*len(EFFICIENCY_FIELDS) for label in EFFICIENCY_LABELS}
        self.stack=[];self.ended_wall=None;self.failed=False
        self.started_wall=wall_clock();self.started_cpu=cpu_clock()
        self.native_started=probe.read()

    def begin(self,label):
        require(label in self.totals and self.ended_wall is None and len(self.stack)<16,
            'Invalid bounded efficiency span')
        frame=[label,self.wall_clock(),self.cpu_clock(),0,0]
        self.stack.append(frame);return frame

    def end(self,frame,failed=False):
        cpu,wall=self.cpu_clock(),self.wall_clock()
        require(bool(self.stack) and self.stack.pop() is frame,'Efficiency span nesting changed')
        wall_delta,cpu_delta=wall-frame[1],cpu-frame[2]
        require(min(wall_delta,cpu_delta)>=0 and frame[3]<=wall_delta and frame[4]<=cpu_delta,
            'Efficiency clocks or nesting inconsistent')
        row=self.totals[frame[0]]
        row[0]+=1;row[1]+=wall_delta;row[2]+=cpu_delta
        row[3]+=wall_delta-frame[3];row[4]+=cpu_delta-frame[4]
        row[5]=max(row[5],wall_delta);row[6]+=bool(failed)
        if self.stack:self.stack[-1][3]+=wall_delta;self.stack[-1][4]+=cpu_delta

    def finish(self,failed=False):
        require(not self.stack and self.ended_wall is None,'Cannot finish active efficiency spans twice')
        self.native_ended=self.probe.read()
        self.ended_cpu=self.cpu_clock();self.ended_wall=self.wall_clock();self.failed=bool(failed)
        require(self.native_ended>=self.native_started,'Native CPU counter decreased')

    def snapshot(self):
        require(self.ended_wall is not None and not self.stack,'Unfinished efficiency profile')
        wall=self.ended_wall-self.started_wall;cpu=self.ended_cpu-self.started_cpu
        wall_residual=wall-sum(row[3] for row in self.totals.values())
        cpu_residual=cpu-sum(row[4] for row in self.totals.values())
        require(min(wall_residual,cpu_residual)>=0,'Component totals exceed handler')
        return dict(schema=EFFICIENCY_SCHEMA,handler_start_ns=self.started_wall,handler_end_ns=self.ended_wall,
            handler_wall_ns=wall,handler_thread_cpu_ns=cpu,handler_failed=self.failed,
            native_cpu_ns=self.native_ended-self.native_started,native_pid=self.probe.pid,native_created=self.probe.created,
            totals={label:dict(zip(EFFICIENCY_FIELDS,row,strict=True)) for label,row in self.totals.items()},
            handler_unattributed_wall_ns=wall_residual,handler_unattributed_thread_cpu_ns=cpu_residual,
            measurement_units='nanoseconds',cpu_scope='current_python_thread_only',
            inclusive_labels_overlap=True,exclusive_totals_are_disjoint=True,
            native_cpu_is_separate_dimension=True,native_read_includes_search_pipe_and_scheduling=True,
            residual_includes_decode_encode_python_and_measurement=True,
            wall_minus_cpu_is_unexplained_not_proven_scheduling=True,
            times_are_diagnostic_not_trusted_deadline_receipts=True)


class _StreamProxy:
    def __init__(self,stream,bindings):
        self.original=stream
        for method,label in (('write','native_write'),('flush','native_flush'),('readline','native_read_wait')):
            if hasattr(stream,method):setattr(self,method,bindings.wrap(label,getattr(stream,method)))
    def __getattr__(self,name):return getattr(self.original,name)


class EfficiencyBindings:
    """Only this worker's instances change; frozen modules and classes never do."""
    def __init__(self,*,probe_factory=NativeCPUProbe):
        self.active=None;self.restorations=[];self.probes={};self.installed=False
        self.probe_factory=probe_factory

    def replace(self,owner,name,value):
        old=vars(owner).get(name,_MISSING)
        setattr(owner,name,value);self.restorations.append((owner,name,old))

    def wrap(self,label,function):
        def measured(*args,**kwargs):
            profiler=self.active
            if profiler is None:return function(*args,**kwargs)
            frame=profiler.begin(label);failed=True
            try:
                value=function(*args,**kwargs);failed=False;return value
            finally:
                primary=sys.exc_info()[1]
                try:profiler.end(frame,failed)
                except BaseException as error:
                    if primary is None:raise
                    primary.add_note('Efficiency span also failed: '+repr(error))
        return measured

    def install(self,instances,forward):
        require(not self.installed,'Efficiency hooks cannot be installed twice')
        self.installed=True
        try:
            self.replace(forward,'compiled',self.wrap('compiled_forward',forward.compiled))
            self.replace(forward,'infer',self.wrap('inference_bridge',forward.infer))
            for arm,instance in instances.items():
                require(instance.forward is forward,'Observer must wrap the exact shared graph')
                self.probes[arm]=self.probe_factory(instance.proc.pid)
                self.replace(instance,'_infer',self.wrap('callback_total',instance._infer))
                self.replace(instance,'exchange',self.wrap('native_exchange',instance.exchange))
                self.replace(instance.proc,'stdin',_StreamProxy(instance.proc.stdin,self))
                self.replace(instance.proc,'stdout',_StreamProxy(instance.proc.stdout,self))
            return self
        except BaseException:self.close();raise

    def start(self,arm):
        require(self.installed and self.active is None and arm in self.probes,'Ordinary observer activation')
        profiler=EfficiencyProfiler(self.probes[arm]);self.active=profiler;return profiler

    def finish(self,profiler,*,failed=False):
        require(self.active is profiler,'Efficiency owner identity changed')
        self.active=None;profiler.finish(failed)

    def close(self):
        require(self.active is None,'Cannot restore active efficiency hooks')
        for owner,name,old in reversed(self.restorations):
            if old is _MISSING:delattr(owner,name)
            else:setattr(owner,name,old)
        self.restorations.clear();self.probes.clear();self.installed=False


def publish_evidence(context,request_id,boundary_evidence,*,batch=False):
    """Existing child-profile publication is the complete-efficiency-file barrier."""
    require(type(request_id) is str and re.fullmatch(r'request-[0-9]{8}',request_id) is not None,
        'Efficiency request identity')
    require(len(context['efficiency_profiles'])<=MAX_BATCH_PROFILES,'Efficiency profile count bound')
    mode=context['observer_mode']
    require(mode in ('off','on') and (mode=='on' or not context['efficiency_profiles']),
        'OFF observer cannot emit component measurements')
    keys=('request_raw_sha256','payload_sha256','result_sha256','serialized_packet_utf8_sha256',
        'serialized_packet_utf8_bytes','outcome','error','pre_write_ns','post_flush_ns')
    evidence=dict(schema=EFFICIENCY_SCHEMA,request_id=request_id,pid=os.getpid(),observer_mode=mode,
        profile_enabled=mode=='on',route='research',quota=context['quota'],profiles=context['efficiency_profiles'],
        **{key:boundary_evidence[key] for key in keys},scientific_response_fields_unchanged=True,
        success_sidecar_after_response_flush=True,boundary_sidecar_is_publication_barrier=True,
        times_are_diagnostic_not_trusted_deadline_receipts=True)
    encoded=json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    require(len(encoded)<=(MAX_EFFICIENCY_BATCH_BYTES if batch else MAX_EFFICIENCY_BYTES),'Efficiency evidence bound')
    target=context['folder']/(request_id+'_efficiency_profile.json')
    with target.open('xb') as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    return publish_profile(context['folder'],request_id,boundary_evidence,batch=batch)


def verify_efficiency_identity(folder,freeze,metadata):
    """Model-free identity check for the copied transport and saved reader."""
    from . import common as c
    folder=Path(folder)
    require(folder.resolve().is_relative_to((c.OUT/'workers').resolve()),'Efficiency identity output boundary')
    mode=observer_mode(freeze,folder,freeze['efficiency_observers'].get(folder.name))
    identity=c.read(folder/'efficiency_identity.json')
    graph_identity=c.read(folder/'graph_identity.json')
    require(identity==efficiency_identity(mode,metadata,graph_identity),'Exact efficiency hook identity')
    return dict(status='passed',observer_mode=mode,profile_enabled=mode=='on',
        identity_sha256=digest(identity))


def verify_efficiency_profile(profile,*,mode,request_id,payload,receipt,parent_profile,child_profile,worker_pid):
    """Authenticate disjoint aggregates without reinterpreting trusted clocks."""
    require(type(profile) is dict and profile.get('schema')==EFFICIENCY_SCHEMA,'Efficiency sidecar schema')
    require(mode in ('off','on') and profile.get('observer_mode')==mode and
        profile.get('profile_enabled') is (mode=='on') and profile.get('pid')==worker_pid and
        profile.get('request_id')==request_id==receipt['request_id'],'Efficiency owner/request identity')
    require(profile.get('route')=='research' and profile.get('quota')==256,'Exact256 efficiency route')
    keys=('request_raw_sha256','payload_sha256','result_sha256','serialized_packet_utf8_sha256',
        'serialized_packet_utf8_bytes','outcome','error','pre_write_ns','post_flush_ns')
    require(all(profile.get(key)==child_profile.get(key) for key in keys),'Efficiency/boundary raw bindings')
    require(profile['payload_sha256']==digest(payload) and
        profile['request_raw_sha256']==parent_profile['request_raw_sha256'],'Efficiency trusted request binding')
    for field in ('scientific_response_fields_unchanged','success_sidecar_after_response_flush',
        'boundary_sidecar_is_publication_barrier','times_are_diagnostic_not_trusted_deadline_receipts'):
        require(profile.get(field) is True,'Efficiency scope '+field)
    rows=profile.get('profiles');boundary_rows=child_profile['profiles']
    require(type(rows) is list and len(rows)<=MAX_BATCH_PROFILES,'Bounded efficiency profile inventory')
    if mode=='off':
        require(rows==[],'OFF must not record CPU/component observations')
        return dict(status='passed',observer_mode=mode,profiles=0)
    require(len(rows)==len(boundary_rows),'Every ON ordinary handle has one component profile')
    number=lambda x:type(x) is int and 0<=x<2**63
    for row,outer in zip(rows,boundary_rows,strict=True):
        require(type(row) is dict and row.get('schema')==EFFICIENCY_SCHEMA and
            row.get('measurement_units')=='nanoseconds' and row.get('cpu_scope')=='current_python_thread_only',
            'Efficiency handler identity/clock scope')
        for field in ('handler_start_ns','handler_end_ns','handler_wall_ns','handler_thread_cpu_ns','native_cpu_ns',
            'handler_unattributed_wall_ns','handler_unattributed_thread_cpu_ns'):
            require(number(row.get(field)),'Finite nonnegative efficiency counter '+field)
        require(outer['handle_start_ns']<=row['handler_start_ns']<=row['handler_end_ns']<=outer['handle_end_ns'] and
            row['handler_wall_ns']==row['handler_end_ns']-row['handler_start_ns'],'Nested handler wall interval')
        require(type(row.get('handler_failed')) is bool and row['handler_failed']==outer['handle_failed'],
            'Expected counterexample failure remains explicit')
        require(type(row.get('native_pid')) is int and row['native_pid']>0 and
            type(row.get('native_created')) in (int,float) and math.isfinite(row['native_created']) and row['native_created']>0,
            'Exact native process counter owner')
        totals=row.get('totals');require(type(totals) is dict and set(totals)==set(EFFICIENCY_LABELS),'Fixed component labels')
        for label,values in totals.items():
            require(type(values) is dict and set(values)==set(EFFICIENCY_FIELDS) and all(number(x) for x in values.values()),
                'Fixed finite aggregate fields')
            require(values['exclusive_wall_ns']<=values['inclusive_wall_ns']<=row['handler_wall_ns'] and
                values['exclusive_thread_cpu_ns']<=values['inclusive_thread_cpu_ns']<=row['handler_thread_cpu_ns'] and
                values['max_wall_ns']<=values['inclusive_wall_ns'] and values['failures']<=values['calls'],
                'Bounded component aggregate')
            require(values['calls']>0 or all(x==0 for x in values.values()),'Unused component must have zero measurements')
        require(sum(x['exclusive_wall_ns'] for x in totals.values())+row['handler_unattributed_wall_ns']==row['handler_wall_ns'] and
            sum(x['exclusive_thread_cpu_ns'] for x in totals.values())+row['handler_unattributed_thread_cpu_ns']==row['handler_thread_cpu_ns'],
            'Exact disjoint handler decomposition')
        for field in ('inclusive_labels_overlap','exclusive_totals_are_disjoint','native_cpu_is_separate_dimension',
            'native_read_includes_search_pipe_and_scheduling','residual_includes_decode_encode_python_and_measurement',
            'wall_minus_cpu_is_unexplained_not_proven_scheduling','times_are_diagnostic_not_trusted_deadline_receipts'):
            require(row.get(field) is True,'Efficiency interpretation '+field)
    if type(payload) is dict and set(payload)=={'player','observer'} and receipt.get('status')=='ok':
        require(len(rows)==1 and not rows[0]['handler_failed'],'Successful ordinary profile')
        trace=receipt['result']['diagnostics'];totals=rows[0]['totals']
        callbacks=trace['inference_calls'];phases=len(trace['phases'])
        require(all(totals[label]['calls']==callbacks and totals[label]['failures']==0
            for label in ('compiled_forward','inference_bridge','callback_total')),'Inference counts equal exact scientific trace')
        require(totals['native_exchange']['calls']==phases and
            all(totals[label]['calls']==callbacks+phases for label in ('native_write','native_flush','native_read_wait')),
            'Native protocol counters equal exact scientific trace')
    return dict(status='passed',observer_mode=mode,profiles=len(rows))

def response_packet(value):
    encoded=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    # Windows text stdout may expand the final newline to CRLF.
    require(len(encoded.encode('utf-8'))+1<=MAX_RESPONSE_BYTES,'Research response exceeds4MiB transport cap')
    return encoded

def initialize(route='frozen'):
    sources=verify_runtime_sources()  # Exactly once before any model/profile load.
    from . import common as c
    require(route=='research','Only the accepted256 player is admitted')
    require(c.INTERNAL_SECONDS==INTERNAL_SECONDS and c.SECONDS_PER_MOVE==adapters.SECONDS_PER_MOVE,'Frozen clock policy')
    folder=Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder.is_relative_to(c.OUT.resolve()),'New worker evidence boundary')
    require(os.environ.get('STUDY_BOUNDARY_PROFILE')=='1','Boundary observer required')
    freeze_path=Path(os.environ['STUDY_RESEARCH_FREEZE']).resolve()
    require(freeze_path==c.OUT/'freeze.json' and c.file_hash(freeze_path)==os.environ['STUDY_RESEARCH_FREEZE_SHA256'],
            'Exact execution freeze binding')
    freeze=c.read(freeze_path)
    mode=observer_mode(freeze,folder,os.environ.get('STUDY_EFFICIENCY_OBSERVER'))
    require(freeze.get('phase')==c.PHASE==os.environ.get('SPLENDOR_SEARCH256_STRENGTH_PHASE','strength'),
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
    instances=adapters.initialize_adapters(folder,profiles,route=route)
    context=dict(adapters=instances,profiles=profiles,profile_artifacts=descriptors,folder=folder,quota=adapters.route_quota(route),route=route,
        execution_identities=identities,freeze=freeze,diagnostic_capture=False,request_started=None,
        component_profiles=[],boundary_handle_active=False,
        component_backend='qualified_graph64',
        graph_parity_passed=False,graph_parity_attempted=False,inference_binding=SimpleNamespace(original=Opponent._infer))
    try:
        forward=graph.build_graph(instances['search'].model,folder,_save)
        for instance in instances.values():instance.forward=forward
        context.update(frozen_forward=forward,component_backend_identity=digest(forward.identity))
        metadata=adapters.ready_metadata(route,freeze,pid=os.getpid(),native_pids={key:instance.proc.pid for key,instance in instances.items()},
            sources=sources,freeze_sha256=os.environ['STUDY_RESEARCH_FREEZE_SHA256'])
        context['metadata']=metadata
        context.update(observer_mode=mode,efficiency_profiles=[],efficiency_bindings=None)
        if mode=='on':context['efficiency_bindings']=EfficiencyBindings().install(instances,forward)
        _save(folder/'efficiency_identity.json',efficiency_identity(mode,metadata,forward.identity))
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
    """Identity-preserving delegation; no rewritten scientific implementation."""
    return accepted._ordinary(context,request)


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
            return corpus.run_entrypoint_checks(context['adapters'],request_callback=lambda payload:handle(context,payload),
                boundary=boundary,evidence_callback=record,route=context['route'])
        except BaseException as error:
            record(dict(event='entrypoint_failed',error=repr(error),traceback=traceback.format_exc()));raise
        finally:
            stream.close();context.pop('evidence_callback',None)
            for instance in context['adapters'].values(): instance.admission_guard=None
            context['diagnostic_capture']=False;context['request_started']=outer;context['quota']=adapters.route_quota(context['route'])
    require(not context['boundary_handle_active'],'Nested research handle')
    require(len(context['component_profiles'])<MAX_BATCH_PROFILES,'Boundary profile count bound')
    context['boundary_handle_active']=True
    failed=True;started=time.perf_counter_ns()
    profiler=None;bindings=context.get('efficiency_bindings')
    try:
        if bindings is not None:
            arm=request.get('player') if type(request) is dict else None
            # Invalid-arm fixtures must still fail in the original handler.
            # The idle search process is a counter owner for such rejected input.
            profiler=bindings.start(arm if type(arm) is str and arm in bindings.probes else 'search')
        result=_ordinary(context,request);failed=False
        context['quota']=result['simulations_quota']
        return result
    finally:
        primary=sys.exc_info()[1]
        try:
            if profiler is not None:
                try:
                    bindings.finish(profiler,failed=failed)
                    context['efficiency_profiles'].append(profiler.snapshot())
                except BaseException as error:
                    failed=True
                    if primary is None:raise
                    primary.add_note('Efficiency finalization also failed: '+repr(error))
                finally:bindings.active=None
        finally:
            ended=time.perf_counter_ns();context['boundary_handle_active']=False
            context['component_profiles'].append(dict(schema=SCHEMA,pid=os.getpid(),clock='perf_counter_ns',
                measurement_units='nanoseconds',request_read_complete_ns=context.get('request_read_complete_ns')
                    if context.get('request_started') is not None else None,
                handle_start_ns=started,handle_end_ns=ended,handle_failed=failed,
                times_are_diagnostic_not_trusted_deadline_receipts=True))

def close_context(context):
    bindings=context.get('efficiency_bindings')
    try:
        if bindings is not None:bindings.close()
    finally:
        for instance in context.get('adapters',{}).values(): instance.close()

def serve(*,route='frozen',max_request_bytes=1024*1024):
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
                context['efficiency_profiles']=[]
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
                            publish_evidence(context,request_id,evidence,batch=len(context['component_profiles'])>1)
                        except BaseException as caught: save_sidecar_error(context,request_id,caught,primary_error=repr(error))
                    emit(dict(type='error',request_id=request_id,error=repr(error)[:4096]));return 1
                try:
                    evidence=make_evidence(context,request_id,raw,payload,result,outcome='response_flushed',
                        pre_write_ns=pre,post_flush_ns=post,encoded=encoded)
                    publish_evidence(context,request_id,evidence,batch=startup)
                except BaseException as error:
                    save_sidecar_error(context,request_id,error);return 1
        except BaseException as error:
            emit(dict(type='error',error=repr(error)[:4096]));return 1
        finally:
            if context is not None:close_context(context)

if __name__=='__main__':
    raise SystemExit(serve(route=os.environ.get('STUDY_ROUTE','frozen')))

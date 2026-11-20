"""Separate logging-diagnostic entry point; unchanged numerical player routes.

Synchronous qualification records stay inside the original .900-second guard.
Timer metadata follows a successful response; on failure it is preserved before
the error packet so immediate parent cleanup cannot destroy it.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent, CHECKPOINT_SHA256
from tools.local_player_qualification_128.clock import THREAD_ENV, _decode
from tools.splendorust_search_scale_20261024 import worker as original_worker
from tools.splendorust_search_scale_20261024 import qualification as original_qualification
from tools.splendorust_search_scale_20261024.worker import _save
from tools.splendorust_inference_opt_20261025 import graph
from tools.splendorust_inference_opt_20261025.worker import graph_parity as original_graph_parity
from tools.splendorust_boundary_timing_20261025 import worker as boundary_worker
from tools.splendorust_boundary_timing_20261025.worker import make_evidence, save_sidecar_error
from tools.splendorust_boundary_timing_20261025.child_profile import SCHEMA, MAX_BATCH_PROFILES, publish_profile
from tools.splendorust_pool6_core_compare_20261027_02.worker import _ordinary as candidate_ordinary
from . import adapters, qualification, logging_probe
from .adapters import require, INTERNAL_SECONDS

MAX_RESPONSE_BYTES=4*1024**2
FROZEN_ORDINARY = original_worker.handle

def _initialize_research():
    sources=verify_runtime_sources()  # Exactly once before loading profiles or model.
    from . import common as c
    folder=Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder.is_relative_to(c.OUT.resolve()),'New worker evidence boundary')
    require(os.environ.get('STUDY_BOUNDARY_PROFILE')=='1','Boundary observer required')
    freeze_path=Path(os.environ['STUDY_RESEARCH_FREEZE']).resolve()
    require(freeze_path==c.OUT/'freeze.json' and c.file_hash(freeze_path)==os.environ['STUDY_RESEARCH_FREEZE_SHA256'],
            'Exact execution freeze binding')
    freeze=c.read(freeze_path)
    require(freeze.get('phase')==c.PHASE==os.environ.get('SPLENDOR_STARTUP_LOGGING_PHASE','startuplog01'),
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


def _study_contract():
    from . import common as c
    route = os.environ['STUDY_LOGGING_ROUTE']
    arm_id = os.environ['STUDY_LOGGING_ARM_ID']
    require(route in ('frozen', 'research') and arm_id in (route+'_cpu0', route+'_cpu2'),
            'Logging route/fixed arm allowlist')
    folder = Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder.is_relative_to(c.OUT.resolve()), 'Logging evidence directory')
    freeze_path = Path(os.environ['STUDY_RESEARCH_FREEZE']).resolve()
    require(freeze_path == c.OUT/'freeze.json' and
            c.file_hash(freeze_path) == os.environ['STUDY_RESEARCH_FREEZE_SHA256'], 'Logging execution freeze')
    freeze = c.read(freeze_path)
    require(freeze['phase'] == c.PHASE == os.environ.get('SPLENDOR_STARTUP_LOGGING_PHASE', 'startuplog01'),
            'Logging execution phase')
    item = freeze['request_matrix']; c.verify([item]); matrix = c.read(ROOT/item['path'])
    specs = [s for s in matrix['requests'] if s['arm_id'] == arm_id]
    require(len(specs) == 24 and all(s['route'] == route and s['quota'] == 64 and
            type(s['logging_timing_enabled']) is bool for s in specs), 'Frozen logging request allocation')
    require(os.environ.get('STUDY_SEARCH_SIMULATIONS') == '64', 'Only unchanged graph64 players')
    return dict(route=route, arm_id=arm_id, folder=folder, specs=specs, matrix=item)


def initialize():
    contract = _study_contract()  # No model before exact route/freeze admission.
    # Each chosen initializer calls the corrected runtime guard exactly once.
    context = boundary_worker.initialize() if contract['route'] == 'frozen' else _initialize_research()
    context.update(logging_contract=contract, logging_index=0, probe_active=None,
                   qualification_active=False, request_id=None, probe_recorders={})
    try:
        logdir = context['folder']/'logging_forensics'; logdir.mkdir()
        for enabled in (False, True):
            context['probe_recorders'][enabled] = logging_probe.make_recorder(
                logdir/('on.jsonl' if enabled else 'off.jsonl'), enabled)
        _save(context['folder']/'logging_identity.json', dict(schema=logging_probe.SCHEMA,
            route=contract['route'], arm_id=contract['arm_id'], matrix=contract['matrix'],
            ordinary_requests=24, startup_timer_enabled=True, ordinary_payload_unchanged=True,
            numerical_route_unchanged=True, extra_host_sampler_enabled=False,
            recorder_modes=[False, True], record_operations_unchanged=True,
            synchronous_fsync_retained=True, mode_selected_by_frozen_worker_local_index=True))
        return context
    except BaseException:
        close_context(context)
        raise


def _ordinary_boundary(context, request):
    require(not context['boundary_handle_active'], 'Nested ordinary logging handler')
    require(len(context['component_profiles']) < MAX_BATCH_PROFILES, 'Boundary profile count bound')
    context['boundary_handle_active'] = True
    failed = True; started = time.perf_counter_ns()
    try:
        result = (FROZEN_ORDINARY(context, request) if context['logging_contract']['route'] == 'frozen'
                  else candidate_ordinary(context, request))
        failed = False
        if context['logging_contract']['route'] == 'research':
            context['quota'] = result['simulations_quota']
        return result
    finally:
        ended = time.perf_counter_ns(); context['boundary_handle_active'] = False
        context['component_profiles'].append(dict(schema=SCHEMA, pid=os.getpid(), clock='perf_counter_ns',
            measurement_units='nanoseconds', request_read_complete_ns=context.get('request_read_complete_ns')
            if context.get('request_started') is not None else None,
            handle_start_ns=started, handle_end_ns=ended, handle_failed=failed,
            times_are_diagnostic_not_trusted_deadline_receipts=True))


def _entrypoint(context):
    import psutil
    def admission_boundary():
        if psutil.virtual_memory().available < 2*1024**3:
            raise RuntimeError('STUDY_RESOURCE: qualification batch available RAM below 2 GiB')
    require(not context['qualification_active'], 'Nested qualification')
    context['qualification_active'] = True; context['diagnostic_capture'] = True
    outer = context['request_started']; context['request_started'] = None
    for instance in context['adapters'].values(): instance.admission_guard = admission_boundary
    recorder = logging_probe.make_recorder(context['folder']/'entrypoint_forensics.jsonl', True)
    context['probe_active'] = dict(recorder=recorder, start_record=0,
        binding=dict(operation='entrypoint_checks', arm_id=context['logging_contract']['arm_id']))
    context['evidence_callback'] = recorder.record
    try:
        admission_boundary()
        if context['logging_contract']['route'] == 'frozen':
            return original_qualification.run_entrypoint_checks(context['adapters'], boundary=admission_boundary,
                request_callback=lambda payload: handle(context, payload), evidence_callback=recorder.record)
        return qualification.run_entrypoint_checks(context, lambda payload: handle(context, payload),
            boundary=admission_boundary, evidence_callback=recorder.record)
    except BaseException as error:
        recorder.record(dict(event='entrypoint_failed', error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        recorder.close(); context.pop('evidence_callback', None)
        for instance in context['adapters'].values(): instance.admission_guard = None
        context['diagnostic_capture'] = False; context['qualification_active'] = False
        context['request_started'] = outer; context['quota'] = 64


def handle(context, request):
    if type(request) is dict and request.get('operation') == 'graph_parity':
        return original_graph_parity(context, request)
    require(context['graph_parity_passed'], 'Decisions require exact graph parity first')
    if request == {'operation':'entrypoint_checks'}:
        return _entrypoint(context)
    if context['qualification_active']:
        return _ordinary_boundary(context, request)
    # This is study control only. The original scientific request is unchanged.
    index = context['logging_index']; specs = context['logging_contract']['specs']
    require(index < len(specs), 'No extra logging diagnostic request')
    spec = specs[index]
    require(type(request) is dict and set(request) == {'player','observer'} and
            request['player'] == 'search' and request['observer'] == spec['observer'],
            'Exact frozen diagnostic fixture/payload order')
    context['logging_index'] += 1  # Failed attempts cannot repeat this allocation.
    recorder = context['probe_recorders'][spec['logging_timing_enabled']]
    context['probe_active'] = dict(recorder=recorder, start_record=len(recorder.entries),
        binding=dict(sequence=spec['sequence'], arm_id=spec['arm_id'], case_id=spec['case_id'],
                     worker_local_index=index, observer_sha256=digest(spec['observer'])))
    context['evidence_callback'] = recorder.record
    context['diagnostic_capture'] = True  # Exact startup capture, including callback arrays.
    try:
        return _ordinary_boundary(context, request)
    finally:
        context.pop('evidence_callback', None); context['diagnostic_capture'] = False


def _publish_probe(context, request_id, evidence, emission):
    active = context.get('probe_active')
    if active is None: return
    value = active['recorder'].snapshot(request_id, active['start_record'], evidence, emission, active['binding'])
    folder = context['folder']/'logging_probe'
    logging_probe.publish_probe(folder,request_id,value)
    context['probe_active'] = None


def close_context(context):
    failure = None
    try:
        for recorder in context.get('probe_recorders', {}).values():
            try:
                if not recorder.closed: recorder.close()
            except BaseException as error:
                if failure is None: failure = (error, error.__traceback__)
    finally:
        try:
            if context.get('logging_contract', {}).get('route') == 'frozen':
                boundary_worker.close_context(context)
            else:
                for instance in context.get('adapters', {}).values(): instance.close()
        except BaseException as error:
            if failure is None: failure = (error, error.__traceback__)
    if failure is not None:
        raise failure[0].with_traceback(failure[1])


def serve(*, max_request_bytes=1024*1024):
    for name in THREAD_ENV: os.environ[name] = '1'
    from threadpoolctl import threadpool_limits
    context = None
    def emit(value, deadline=None, markers=None):
        encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
        require(len(encoded.encode('utf-8'))+1 <= MAX_RESPONSE_BYTES, 'Logging response exceeds original4MiB cap')
        if deadline is not None and time.perf_counter() >= deadline:
            prefix = 'ONE_SECOND_INTERNAL_DEADLINE:' if context['logging_contract']['route'] == 'frozen' else 'SAMPLING_INTERNAL_DEADLINE:'
            raise TimeoutError(prefix+' response serialization exceeded0.900s')
        pre = time.perf_counter_ns()
        if markers is not None: markers['pre_write_ns'] = pre
        sys.stdout.write(encoded); sys.stdout.flush()
        post = time.perf_counter_ns()
        if markers is not None: markers['post_flush_ns'] = post
        return pre, post, encoded
    with threadpool_limits(limits=1):
        try:
            context = initialize(); emit(dict(type='ready', metadata=context['metadata']))
            while True:
                raw = sys.stdin.buffer.readline(max_request_bytes+1)
                if not raw: return 0
                context['request_started'] = time.perf_counter()
                context['request_read_complete_ns'] = time.perf_counter_ns()
                context['component_profiles'] = []; context['probe_active'] = None
                request_id = payload = result = None; markers = {}
                try:
                    require(len(raw) <= max_request_bytes and raw.endswith(b'\n'), 'Request size/termination')
                    request = _decode(raw)
                    require(set(request) == {'type','request_id','payload'} and request['type'] == 'request' and
                            type(request['request_id']) is str, 'Transport request envelope')
                    request_id, payload = request['request_id'], request['payload']
                    context['request_id'] = request_id
                    result = handle(context, payload)
                    startup = payload == {'operation':'entrypoint_checks'} or type(payload) is dict and set(payload) == {'operation','fixtures'} and payload['operation'] == 'graph_parity'
                    deadline = None if startup else context['request_started']+INTERNAL_SECONDS
                    pre, post, encoded = emit(dict(type='response', request_id=request_id, result=result), deadline, markers)
                except BaseException as error:
                    _save(context['folder']/'worker_error.json', dict(request_id=request_id, error=repr(error), traceback=traceback.format_exc()))
                    evidence = None
                    if request_id is not None:
                        try:
                            evidence = make_evidence(context, request_id, raw, payload, result,
                                outcome='failed_before_successful_emission', error=repr(error)[:4096],
                                pre_write_ns=markers.get('pre_write_ns'), post_flush_ns=markers.get('post_flush_ns'))
                            publish_profile(context['folder'], request_id, evidence, batch=len(context['component_profiles'])>1)
                        except BaseException as caught:
                            save_sidecar_error(context, request_id, caught, primary_error=repr(error))
                    if evidence is not None:
                        try:
                            _publish_probe(context, request_id, evidence, dict(kind='before_error',
                                prepared_ns=time.perf_counter_ns(),pre_write_ns=None,post_flush_ns=None))
                        except BaseException as caught:
                            save_sidecar_error(context, request_id, caught, primary_error=repr(error))
                    emit(dict(type='error', request_id=request_id, error=repr(error)[:4096]))
                    return 1
                try:
                    evidence = make_evidence(context, request_id, raw, payload, result, outcome='response_flushed',
                        pre_write_ns=pre, post_flush_ns=post, encoded=encoded)
                    publish_profile(context['folder'], request_id, evidence, batch=startup)
                    _publish_probe(context, request_id, evidence, dict(kind='response',pre_write_ns=pre,post_flush_ns=post))
                except BaseException as error:
                    save_sidecar_error(context, request_id, error); return 1
        except BaseException as error:
            emit(dict(type='error',error=repr(error)[:4096])); return 1
        finally:
            if context is not None: close_context(context)


if __name__ == '__main__':
    raise SystemExit(serve())

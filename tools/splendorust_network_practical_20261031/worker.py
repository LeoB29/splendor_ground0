"""Separate numerical-tolerance qualification of the one unchanged FFN candidate."""
from __future__ import annotations
import copy
import json
import hashlib
import importlib.util
import math
import os
import re
import statistics
from collections import Counter
from contextlib import nullcontext
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
EXECUTION_VARIANTS=('control','candidate')
EXECUTION_SCHEMA='splendorust_network_practical_identity_v1'
ADMISSION_SCHEMA='splendorust_network_practical_admission_v1'
NUMERICAL_INTERNAL_SECONDS=45.
TOLERANCES={'policy_atol':1e-5,'policy_rtol':1e-6,'value_atol':1e-6,'value_rtol':0.0}

def response_packet(value):
    encoded=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    # Windows text stdout may expand the final newline to CRLF.
    require(len(encoded.encode('utf-8'))+1<=MAX_RESPONSE_BYTES,'Research response exceeds4MiB transport cap')
    return encoded

def initialize(route='research'):
    sources=verify_runtime_sources()  # Exactly once before any model/profile load.
    from . import common as c
    require(route=='research','Only unchanged scientific256 route')
    variant=os.environ.get('STUDY_EXECUTION_VARIANT')
    require(variant in EXECUTION_VARIANTS,'Declared execution variant')
    require(c.INTERNAL_SECONDS==INTERNAL_SECONDS and c.SECONDS_PER_MOVE==adapters.SECONDS_PER_MOVE,'Frozen clock policy')
    folder=Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder==(c.OUT/'workers'/variant).resolve(),'Exact execution variant evidence folder')
    require(os.environ.get('STUDY_BOUNDARY_PROFILE')=='1','Boundary observer required')
    freeze_path=Path(os.environ['STUDY_RESEARCH_FREEZE']).resolve()
    require(freeze_path==c.OUT/'freeze.json' and c.file_hash(freeze_path)==os.environ['STUDY_RESEARCH_FREEZE_SHA256'],
            'Exact execution freeze binding')
    freeze=c.read(freeze_path)
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
        context.update(execution_variant=variant,numerical_attempted=False,numerical_passed=False,entrypoint_attempted=False,entrypoint_passed=False)
        install_execution(context)
        metadata=adapters.ready_metadata(route,freeze,pid=os.getpid(),native_pids={key:instance.proc.pid for key,instance in instances.items()},
            sources=sources,freeze_sha256=os.environ['STUDY_RESEARCH_FREEZE_SHA256'])
        context['metadata']=metadata
        _save(folder/'practical_execution_identity.json',execution_identity(context,metadata))
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




def numerical_contract(freeze):
    require(freeze.get('numerical_tolerances')==TOLERANCES,
        'Exact prospectively frozen numerical tolerances')
    return dict(TOLERANCES)


def candidate_source(freeze):
    from . import common as c
    item=c.artifact(c.CANDIDATE_SOURCE)
    require(item['sha256']==c.CANDIDATE_SOURCE_SHA and item in freeze['inputs'],
        'The sole unchanged candidate source is frozen')
    return item


def install_execution(context):
    from . import common as c
    variant=context['execution_variant'];forward=context['frozen_forward']
    numerical_contract(context['freeze'])
    require(digest(forward.identity)==context['freeze']['unchanged_graph_backend_identity']['graph_identity_sha256'],
        'Original graph identity before selecting execution')
    original_text=str(forward.compiled.inlined_graph)+'\n'
    require(hashlib.sha256(original_text.encode()).hexdigest()==forward.identity['graph_ir_sha256'],
        'Original graph IR before candidate installation')
    context['original_compiled']=forward.compiled
    context['candidate_descriptor']=None
    if variant=='control':return
    descriptor=candidate_source(context['freeze'])
    require(c.CANDIDATE_MODULE not in sys.modules,'Candidate module imported only once')
    spec=importlib.util.spec_from_file_location(c.CANDIDATE_MODULE,c.CANDIDATE_SOURCE)
    require(spec is not None and spec.loader is not None,'Original candidate loader')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c.verify([descriptor]);require(module.CANDIDATE_ID==c.CANDIDATE_ID,'Only the held implementation')
    import torch
    model=context['adapters']['search'].model
    before=graph._weight_digest(model,torch)
    chosen=module.build_candidate(forward.compiled,model,context['folder'],_save)
    require(callable(chosen),'One candidate callable')
    require(graph._weight_digest(model,torch)==before==forward.identity['weights_before_sha256'],
        'Candidate preserves original float32 weights')
    require(torch.get_num_threads()==torch.get_num_interop_threads()==1 and
        all(not m.training for m in model.modules()),'One-thread evaluation unchanged')
    require(str(context['original_compiled'].inlined_graph)+'\n'==original_text,'Original graph unchanged')
    forward.compiled=chosen;context['candidate_descriptor']=descriptor;c.verify([descriptor])


def execution_identity(context,metadata):
    from . import common as c
    variant=context['execution_variant'];forward=context['frozen_forward']
    return dict(schema=EXECUTION_SCHEMA,execution_variant=variant,metadata=metadata,
        scientific_route='research',simulations=256,
        original_graph_identity_sha256=digest(forward.identity),
        original_graph_ir_sha256=forward.identity['graph_ir_sha256'],
        original_graph_identity_is_reference_only=variant=='candidate',
        exact_output_identity_claimed=variant=='control',
        candidate_module=context['candidate_descriptor'],
        candidate_id=c.CANDIDATE_ID if variant=='candidate' else None,
        selected_callable_type=type(forward.compiled).__module__+'.'+type(forward.compiled).__qualname__,
        numerical_tolerances=numerical_contract(context['freeze']),
        weights_changed=False,precision_changed=False,scientific_settings_changed=False,
        qualification_reference_uses_same_execution_variant=True,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)


def verify_execution_identity(folder,freeze,metadata):
    from . import common as c
    folder=Path(folder).resolve();variant=folder.name
    require(variant in EXECUTION_VARIANTS and folder==(c.OUT/'workers'/variant).resolve(),
        'Exact practical worker folder')
    value=c.read(folder/'practical_execution_identity.json');original=c.read(folder/'graph_identity.json')
    require(value['schema']==EXECUTION_SCHEMA and value['execution_variant']==variant and
        value['metadata']==metadata and value['scientific_route']=='research' and value['simulations']==256,
        'Practical identity fields')
    require(digest(original)==value['original_graph_identity_sha256']==
        freeze['unchanged_graph_backend_identity']['graph_identity_sha256'] and
        value['original_graph_ir_sha256']==original['graph_ir_sha256'],'Original reference graph binding')
    require(value['original_graph_identity_is_reference_only'] is (variant=='candidate') and
        value['exact_output_identity_claimed'] is (variant=='control'),'No false unchanged-candidate numerical identity')
    require(value['weights_changed'] is value['precision_changed'] is value['scientific_settings_changed'] is False and
        value['qualification_reference_uses_same_execution_variant'] is True,'Scientific configuration and reference route')
    require(value['numerical_tolerances']==numerical_contract(freeze),'Frozen tolerance identity')
    require((value['ruleset_id'],value['catalogue_sha256'],value['turn_contract'])==
        (RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT),'Corrected rules identity')
    if variant=='candidate':
        require(value['candidate_module']==candidate_source(freeze) and value['candidate_id']==c.CANDIDATE_ID,
            'Only the previously held candidate implementation')
    else:require(value['candidate_module'] is value['candidate_id'] is None,'Exact control has no candidate')
    return dict(status='passed',execution_variant=variant,identity_sha256=digest(value))


def validate_numerical_fixtures(fixtures):
    require(type(fixtures) is list and len(fixtures)==38,'Exactly38 fresh numerical fixtures')
    ids=set();singles=set();batches=[]
    for fixture in fixtures:
        require(type(fixture) is dict and set(fixture)=={'fixture_id','batch','tokens'},'Numerical fixture fields')
        ident,batch,tokens=fixture['fixture_id'],fixture['batch'],fixture['tokens']
        require(type(ident) is str and ident and ident not in ids,'Unique numerical fixture identity');ids.add(ident)
        require(type(batch) is int and batch in (1,2,8) and type(tokens) is list and
            len(tokens)==batch*31*48,'Numerical input tensor shape')
        require(all(type(x) in (int,float) and math.isfinite(x) for x in tokens),'Finite numerical input')
        batches.append(batch)
        if batch==1:
            h=digest(tokens);require(h not in singles,'Distinct saved batch-one inputs');singles.add(h)
    require(sorted(batches)==[1]*36+[2,8],'Fixed36 singles and two composite batches')
    return fixtures


def numerical_fixtures(freeze):
    from . import common as c
    item=freeze['numerical_plan'];c.verify([item]);document=c.read(c.ROOT/item['path'])
    return validate_numerical_fixtures(document['fixtures'])


def verify_historical_numerical_shadows(freeze,fixtures,evidence):
    """Anchor only saved batch-one inputs; batching has its own eager reference."""
    from . import common as c
    item=freeze['numerical_plan'];c.verify([item]);document=c.read(c.ROOT/item['path'])
    require(document['fixtures']==fixtures,'Historical anchor uses exact frozen numerical inputs')
    provenance=document['provenance']
    require(type(provenance) is list and len(provenance)==36,'Exactly36 historical batch-one anchors')
    require(type(evidence) is list and len(evidence)==len(fixtures),'Complete anchored numerical evidence')
    singles=[(fixture,row) for fixture,row in zip(fixtures,evidence,strict=True) if fixture['batch']==1]
    require(len(singles)==36,'Historical anchors apply only to36 batch-one fixtures')
    for (fixture,row),origin in zip(singles,provenance,strict=True):
        require(origin['fixture_id']==fixture['fixture_id']==row['fixture_id'] and row['batch']==1 and
            origin['input_sha256']==digest(fixture['tokens']),'Historical numerical input/provenance binding')
        comparison=assess_outputs(fixture,[origin['historical_output83']],row['eager_outputs'],'control')
        require(comparison['accepted'],'Eager shadow must exactly reproduce historical float32 values and bits')
    return 36


def assess_outputs(fixture,eager,candidate,variant,*,tolerances=None):
    """Only elementwise compare arrays produced from this identical input."""
    tolerances=TOLERANCES if tolerances is None else tolerances
    require(variant in EXECUTION_VARIANTS and tolerances==TOLERANCES,'Declared numerical admission policy')
    comparison=graph.compare_outputs(fixture,eager,candidate)
    # compare_outputs independently enforces finite float values and exact83 shape.
    import struct
    require(all(struct.unpack('<f',struct.pack('<f',value))[0]==value
        for outputs in (eager,candidate) for row in outputs for value in row),
        'Saved outputs must be actual finite float32 values')
    groups={}
    for name,start,stop in (('policy',0,81),('value',81,83)):
        errors=[];allowed=[];bit_differences=0;numeric_differences=0
        for left,right in zip(eager,candidate,strict=True):
            for reference,observed in zip(left[start:stop],right[start:stop],strict=True):
                errors.append(abs(observed-reference))
                allowed.append(tolerances[name+'_atol']+tolerances[name+'_rtol']*abs(reference))
                bit_differences+=struct.pack('<f',reference)!=struct.pack('<f',observed)
                numeric_differences+=reference!=observed
        groups[name]=dict(float_count=len(errors),unequal_float32_bits=bit_differences,
            unequal_numeric_values=numeric_differences,max_absolute_error=max(errors),
            mean_absolute_error=sum(errors)/len(errors),
            outside_tolerance=sum(error>limit for error,limit in zip(errors,allowed,strict=True)),
            maximum_error_to_allowed_ratio=max(error/limit for error,limit in zip(errors,allowed,strict=True)))
    within=all(group['outside_tolerance']==0 for group in groups.values())
    exact=comparison['exact_float_equal'] and comparison['float32_bits_equal']
    return dict(comparison=comparison,policy=groups['policy'],value=groups['value'],
        within_tolerance=within,accepted=exact if variant=='control' else within)


def summarize_admission(operation,fixtures,evidence,exact_summary,variant,freeze):
    require(operation in ('graph_parity','numerical_checks') and variant in EXECUTION_VARIANTS,
        'Declared admission operation and variant')
    tolerances=numerical_contract(freeze)
    if operation=='graph_parity':graph.validate_parity_fixtures(fixtures)
    else:validate_numerical_fixtures(fixtures)
    require(type(evidence) is list and len(evidence)==len(fixtures),'Complete numerical evidence')
    rows=[]
    for fixture,item in zip(fixtures,evidence,strict=True):
        require(set(item)=={'fixture_id','batch','eager_outputs','candidate_outputs','comparison'} and
            item['fixture_id']==fixture['fixture_id'] and item['batch']==fixture['batch'],'Fixture/output binding')
        row=assess_outputs(fixture,item['eager_outputs'],item['candidate_outputs'],variant,tolerances=tolerances)
        require(row['comparison']==item['comparison'],'Complete exact numerical hashes recompute')
        rows.append(row)
    require(exact_summary['rows']==[row['comparison'] for row in rows] and
        exact_summary['fixtures_count']==len(fixtures) and exact_summary['fixture_payload_sha256']==digest(fixtures),
        'Original exact summary binds all numerical evidence')
    exact=all(row['comparison']['exact_float_equal'] and row['comparison']['float32_bits_equal'] for row in rows)
    require(exact_summary['passed'] is exact and exact_summary['status']==('passed' if exact else 'failed'),
        'Exact rejection remains explicit')
    accepted=all(row['accepted'] for row in rows)
    return dict(schema=ADMISSION_SCHEMA,operation=operation,execution_variant=variant,
        status='passed' if accepted else 'numerically_rejected',accepted=accepted,
        admission_policy='exact_float32' if variant=='control' else 'prospectively_frozen_tolerance',
        numerical_tolerances=tolerances,fixtures_count=len(fixtures),
        fixture_payload_sha256=digest(fixtures),exact_summary=exact_summary,rows=rows,
        compared_only_identical_inputs=True,decisions_compared=False,automatic_retries=0)


def _verify_admission(folder,freeze,result,operation,variant,boundary):
    from . import common as c
    boundary();folder=Path(folder).resolve();variant=folder.name if variant is None else variant
    require(variant in EXECUTION_VARIANTS and folder==(c.OUT/'workers'/variant).resolve(),'Admission worker identity')
    numerical_contract(freeze)
    fixtures=freeze['graph_parity_fixtures'] if operation=='graph_parity' else numerical_fixtures(freeze)
    evidence_name='graph_parity.json' if operation=='graph_parity' else 'numerical_checks.json'
    admission_name='graph_admission.json' if operation=='graph_parity' else 'numerical_admission.json'
    path=folder/evidence_name;require(path.stat().st_size<=4*1024**2,'Bounded full numerical evidence')
    document=c.read(path);require(document['fixtures']==fixtures,'Exact frozen numerical inputs')
    require(document['schema']==(graph.PARITY_SCHEMA if operation=='graph_parity' else 'splendorust_practical_numerical_evidence_v1'),
        'Numerical evidence schema')
    require('error' not in document,'Partial failed numerical evidence cannot be accepted')
    if operation=='numerical_checks':verify_historical_numerical_shadows(freeze,fixtures,document['evidence'])
    recomputed=summarize_admission(operation,fixtures,document['evidence'],document['summary'],variant,freeze)
    require(result==recomputed==c.read(folder/admission_name),'Saved numerical verdict independently recomputes')
    require(document['summary']['graph_identity_sha256']==freeze['unchanged_graph_backend_identity']['graph_identity_sha256'],
        'Original graph reference identity in numerical evidence')
    boundary()
    return dict(status='verified_numerical_admission',accepted=recomputed['accepted'],
        execution_variant=variant,operation=operation,fixtures_count=len(fixtures),
        exact_fixtures=sum(row['comparison']['float32_bits_equal'] and row['comparison']['exact_float_equal'] for row in recomputed['rows']),
        within_tolerance_fixtures=sum(row['within_tolerance'] for row in recomputed['rows']),
        verdict_sha256=digest(recomputed),full_evidence=c.artifact(path))


def verify_graph_admission(folder,freeze,result,variant=None,*,boundary=lambda:None):
    return _verify_admission(folder,freeze,result,'graph_parity',variant,boundary)


def verify_numerical_admission(folder,freeze,result,variant=None,*,boundary=lambda:None):
    return _verify_admission(folder,freeze,result,'numerical_checks',variant,boundary)


def practical_graph_parity(context,request):
    from . import common as c
    require(request==dict(operation='graph_parity',fixtures=context['freeze']['graph_parity_fixtures']),
        'Original frozen graph compatibility inputs')
    exact=original_graph_parity(context,request)
    document=c.read(context['folder']/'graph_parity.json')
    result=summarize_admission('graph_parity',request['fixtures'],document['evidence'],exact,
        context['execution_variant'],context['freeze'])
    _save(context['folder']/'graph_admission.json',result)
    context['graph_parity_passed']=result['accepted']
    return result


def run_numerical_checks(context):
    require(context['graph_parity_passed'] and not context['numerical_attempted'] and
        not context['entrypoint_attempted'],'One numerical panel after graph admission and before actual suite')
    context['numerical_attempted']=True
    fixtures=numerical_fixtures(context['freeze']);evidence=[];rows=[]
    path=context['folder']/'numerical_checks.json'
    deadline=context['request_started']+NUMERICAL_INTERNAL_SECONDS
    def boundary():
        require(time.perf_counter()<deadline,'Numerical qualification internal45s bound')
    base=dict(schema='splendorust_practical_same_input_exact_summary_v1',
        fixture_payload_sha256=digest(fixtures),graph_identity_sha256=digest(context['frozen_forward'].identity))
    try:
        adapter=context['adapters']['search'];forward=context['frozen_forward']
        for fixture in fixtures:
            boundary()
            item=dict(fixture_id=fixture['fixture_id'],batch=fixture['batch'],eager_outputs=None,candidate_outputs=None)
            evidence.append(item)
            item['eager_outputs']=context['inference_binding'].original(adapter,fixture['tokens'],fixture['batch'])
            boundary()
            item['candidate_outputs']=forward.infer(adapter,fixture['tokens'],fixture['batch'])
            boundary()
            item['comparison']=graph.compare_outputs(fixture,item['eager_outputs'],item['candidate_outputs'])
            rows.append(item['comparison'])
        exact=all(row['exact_float_equal'] and row['float32_bits_equal'] for row in rows)
        summary=dict(base,status='passed' if exact else 'failed',passed=exact,fixtures_count=len(rows),rows=rows)
        _save(path,dict(schema='splendorust_practical_numerical_evidence_v1',fixtures=fixtures,evidence=evidence,summary=summary))
        verify_historical_numerical_shadows(context['freeze'],fixtures,evidence)
        result=summarize_admission('numerical_checks',fixtures,evidence,summary,context['execution_variant'],context['freeze'])
        _save(context['folder']/'numerical_admission.json',result)
        context['numerical_passed']=result['accepted']
        return result
    except BaseException as error:
        if not path.exists():
            _save(path,dict(schema='splendorust_practical_numerical_evidence_v1',fixtures=fixtures,evidence=evidence,
                summary=dict(base,status='error',passed=False,fixtures_count=len(rows),rows=rows),
                error=repr(error),traceback=traceback.format_exc()))
        raise


def handle(context,request):
    if type(request) is dict and request.get('operation')=='graph_parity':return practical_graph_parity(context,request)
    require(context['graph_parity_passed'],'Practical graph admission before any further model call')
    if request=={'operation':'numerical_checks'}:return run_numerical_checks(context)
    require(context['numerical_passed'],'Fresh same-input numerical admission before actual scientific suite')
    if request=={'operation':'entrypoint_checks'}:
        require(not context['entrypoint_attempted'],'Actual suite is never retried')
        context['entrypoint_attempted']=True
        result=accepted.handle(context,request)
        require(result['counts']['complete_decisions']==67,'Complete unchanged actual corrected-rule suite')
        context['entrypoint_passed']=True
        return result
    require(context['entrypoint_passed'],'Actual suite admission before ordinary decisions')
    # No cross-variant action, callback-path or historical equality condition here.
    return accepted.handle(context,request)

def close_context(context):
    accepted.close_context(context)

def serve(*,route='research',max_request_bytes=1024*1024):
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
                    numerical=payload=={'operation':'numerical_checks'}
                    startup=numerical or payload=={'operation':'entrypoint_checks'} or type(payload) is dict and set(payload)=={'operation','fixtures'} and payload['operation']=='graph_parity'
                    deadline=context['request_started']+NUMERICAL_INTERNAL_SECONDS if numerical else (None if startup else context['request_started']+INTERNAL_SECONDS)
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
    raise SystemExit(serve(route=os.environ.get('STUDY_ROUTE','research')))

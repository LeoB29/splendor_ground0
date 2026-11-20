"""Exact256 operator localization and one separately pinned execution candidate."""
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
EXECUTION_VARIANTS=('localization','control','candidate')
EXECUTION_SCHEMA='splendorust_network_execution_identity_v1'
PROFILE_SCHEMA='splendorust_network_operator_profile_v1'
PROFILE_INTERNAL_SECONDS=90.
PROFILE_EXTERNAL_SECONDS=120.
MAX_SINGLE_FORWARD_SECONDS=10.
PROFILE_FIXTURE_IDS=tuple(f'recorded_root_{i:02d}' for i in range(1,7))
PROFILE_FORWARD_CALLS=60
MAX_OPERATOR_ROWS=256
MAX_RAW_EVENTS=16384
PROFILE_OPTIONS=dict(activities=['CPU'],record_shapes=False,profile_memory=False,
    with_stack=False,with_flops=False,with_modules=False)

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
        context.update(execution_variant=variant,operator_profile_attempted=False,entrypoint_attempted=False,entrypoint_passed=False)
        install_execution(context)
        metadata=adapters.ready_metadata(route,freeze,pid=os.getpid(),native_pids={key:instance.proc.pid for key,instance in instances.items()},
            sources=sources,freeze_sha256=os.environ['STUDY_RESEARCH_FREEZE_SHA256'])
        context['metadata']=metadata
        _save(folder/'network_execution_identity.json',execution_identity(context,metadata))
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



def profiling_plan(freeze):
    """A fixed small lawful tensor panel; no outcome-dependent input selection."""
    graph.validate_parity_fixtures(freeze['graph_parity_fixtures'])
    fixtures={x['fixture_id']:x for x in freeze['graph_parity_fixtures']}
    selected=[copy.deepcopy(fixtures[name]) for name in PROFILE_FIXTURE_IDS]
    require(all(x['batch']==1 for x in selected),'Profile batch-one inputs only')
    calls=[]
    for fixture in selected:
        # The first pair checks dispatch before repeated localization.
        for mode in ('off','on'):
            calls.append(dict(index=len(calls),fixture_id=fixture['fixture_id'],mode=mode,
                purpose='profiler_exactness',repeat=-1))
        for repeat in range(4):
            order=('off','on') if repeat%2==0 else ('on','off')
            for mode in order:
                calls.append(dict(index=len(calls),fixture_id=fixture['fixture_id'],mode=mode,
                    purpose='operator_localization',repeat=repeat))
    require(len(calls)==PROFILE_FORWARD_CALLS,'Fixed 60 raw forwards')
    return dict(schema=PROFILE_SCHEMA,fixtures=selected,calls=calls,
        input_hashes={f['fixture_id']:digest(f['tokens']) for f in selected},
        profiler=PROFILE_OPTIONS,raw_forward_calls=60,profiled_calls=30,unprofiled_calls=30,
        internal_seconds=PROFILE_INTERNAL_SECONDS,external_seconds=PROFILE_EXTERNAL_SECONDS,
        maximum_single_forward_seconds=MAX_SINGLE_FORWARD_SECONDS,
        maximum_operator_rows=MAX_OPERATOR_ROWS,maximum_raw_events=MAX_RAW_EVENTS,
        expected_fused_transformer_calls_per_forward=6,automatic_fallback=False,
        profiling_overhead_is_descriptive=True,final_speed_requires_observer_off=True)


def validate_candidate_supplement(supplement,freeze_sha):
    """Model-free checks run before the selected module can be imported."""
    from . import common as c
    require(type(supplement) is dict and
        supplement.get('schema')=='splendorust_network_execution_candidate_v1' and
        supplement.get('phase1_freeze_sha256')==freeze_sha,'Candidate supplement identity')
    require(supplement.get('candidate_function')=='build_candidate','One declared candidate factory')
    name=supplement.get('candidate_import_module')
    require(type(name) is str and re.fullmatch(
        r'tools\.splendorust_network_execution_20261031\.candidate_[a-z0-9_]+',name) is not None,
        'One separately versioned local candidate module')
    descriptor=supplement['candidate_module']
    require(descriptor['path']==name.replace('.','/')+'.py','Candidate import/path identity')
    require(type(supplement.get('candidate_id')) is str and bool(supplement['candidate_id']),
        'Declared single candidate identity')
    inputs=supplement['inputs'];require(type(inputs) is list and descriptor in inputs,
        'Candidate module belongs to supplement input inventory')
    require(len({i['path'] for i in inputs})==len(inputs),'Unique candidate supplement pins')
    for key in ('localization_result','localization_verification','decision','model_free_tests','independent_review'):
        c.verify([supplement[key]])
    c.verify(inputs)
    tests=c.read(c.ROOT/supplement['model_free_tests']['path'])
    review=c.read(c.ROOT/supplement['independent_review']['path'])
    require(tests['exit_code']==0 and tests.get('models',0)==tests.get('games',0)==0,
        'Candidate model-free tests passed')
    require(review['approved'] is True and not review.get('blocking_findings',[]),
        'Candidate independent review approved')
    return descriptor


def graph_operator_inventory(text):
    return dict(sorted(Counter(re.findall(r'=\s*(aten::[a-zA-Z0-9_]+)\(',text)).items()))


def install_execution(context):
    """Keep the reference graph truthful; import at most one pinned factory."""
    from . import common as c
    variant=context['execution_variant'];forward=context['frozen_forward']
    require(digest(forward.identity)==context['freeze']['unchanged_graph_backend_identity']['graph_identity_sha256'],
        'Original frozen graph identity before execution selection')
    original_text=str(forward.compiled.inlined_graph)+'\n'
    require(hashlib.sha256(original_text.encode()).hexdigest()==forward.identity['graph_ir_sha256'],
        'Actual original graph IR')
    require(graph_operator_inventory(original_text).get('aten::_transformer_encoder_layer_fwd')==6,
        'Original six fused transformer blocks')
    context['original_compiled']=forward.compiled
    context['execution_supplement']=None
    context['candidate_descriptor']=None
    if variant!='candidate':
        require(not os.environ.get('STUDY_CANDIDATE_SUPPLEMENT_SHA256') and
            not os.environ.get('STUDY_CANDIDATE_SUPPLEMENT'),'No candidate binding for unchanged graph')
        return
    path=Path(os.environ['STUDY_CANDIDATE_SUPPLEMENT']).resolve()
    require(path==(c.OUT/'candidate_supplement.json').resolve() and
        c.file_hash(path)==os.environ['STUDY_CANDIDATE_SUPPLEMENT_SHA256'],'Exclusive supplement path/hash')
    supplement=c.read(path)
    descriptor=validate_candidate_supplement(supplement,os.environ['STUDY_RESEARCH_FREEZE_SHA256'])
    module_path=(c.ROOT/descriptor['path']).resolve()
    require(module_path.parent==Path(__file__).resolve().parent,'Candidate module is in this new package')
    name=supplement['candidate_import_module'];require(name not in sys.modules,'Candidate module imported once')
    spec=importlib.util.spec_from_file_location(name,module_path)
    require(spec is not None and spec.loader is not None,'Candidate source loader available')
    module=importlib.util.module_from_spec(spec)
    # All input and approval checks above precede executable candidate code.
    spec.loader.exec_module(module)
    c.verify([descriptor])
    factory=getattr(module,'build_candidate',None);require(callable(factory),'Declared callable candidate factory')
    import torch
    model=context['adapters']['search'].model
    before=graph._weight_digest(model,torch)
    chosen=factory(forward.compiled,model,context['folder'],_save)
    require(callable(chosen),'Candidate factory returned one callable')
    require(graph._weight_digest(model,torch)==before==forward.identity['weights_before_sha256'],
        'Candidate cannot change weights or float32 precision')
    require(torch.get_num_threads()==torch.get_num_interop_threads()==1 and
        all(not m.training for m in model.modules()),'Candidate preserves one-thread evaluation mode')
    forward.compiled=chosen
    context['execution_supplement']=c.artifact(path)
    context['candidate_descriptor']=descriptor
    c.verify([descriptor])


def execution_identity(context,metadata):
    chosen=context['frozen_forward'].compiled
    return dict(schema=EXECUTION_SCHEMA,execution_variant=context['execution_variant'],
        metadata=metadata,scientific_route='research',simulations=256,
        original_graph_identity_sha256=digest(context['frozen_forward'].identity),
        original_graph_identity_is_reference_only=context['execution_variant']=='candidate',
        original_graph_ir_sha256=context['frozen_forward'].identity['graph_ir_sha256'],
        selected_callable_type=type(chosen).__module__+'.'+type(chosen).__qualname__,
        candidate_module=context.get('candidate_descriptor'),supplement=context.get('execution_supplement'),
        weights_changed=False,precision_changed=False,scientific_settings_changed=False,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)


def verify_execution_identity(folder,freeze,metadata,supplement=None):
    from . import common as c
    folder=Path(folder).resolve();variant=folder.name
    require(variant in EXECUTION_VARIANTS and folder==(c.OUT/'workers'/variant).resolve(),
        'Exact execution identity folder')
    value=c.read(folder/'network_execution_identity.json')
    require(value['schema']==EXECUTION_SCHEMA and value['execution_variant']==variant and
        value['metadata']==metadata and value['scientific_route']=='research' and value['simulations']==256,
        'Execution identity fields')
    original=c.read(folder/'graph_identity.json')
    require(digest(original)==value['original_graph_identity_sha256']==
        freeze['unchanged_graph_backend_identity']['graph_identity_sha256'] and
        value['original_graph_ir_sha256']==original['graph_ir_sha256'],
        'Original reference graph identity remains explicit')
    require(value['original_graph_identity_is_reference_only'] is (variant=='candidate') and
        value['weights_changed'] is value['precision_changed'] is value['scientific_settings_changed'] is False,
        'No scientific model/settings change')
    require((value['ruleset_id'],value['catalogue_sha256'],value['turn_contract'])==
        (RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT),'Corrected execution identity')
    if variant=='candidate':
        require(supplement is not None,'Candidate identity requires supplemental evidence')
        descriptor=validate_candidate_supplement(supplement,c.file_hash(c.OUT/'freeze.json'))
        require(value['candidate_module']==descriptor and
            value['supplement']==c.artifact(c.OUT/'candidate_supplement.json'),'Chosen factory binding')
    else:require(value['candidate_module'] is value['supplement'] is None,'Unchanged variant has no candidate')
    return dict(status='passed',execution_variant=variant,identity_sha256=digest(value))


def profiler_factory(torch):
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,profile_memory=False,with_stack=False,with_flops=False,with_modules=False)


def operator_rows(profile):
    events=profile.events()
    require(0<len(events)<=MAX_RAW_EVENTS,'Bounded nonempty CPU operator event inventory')
    hierarchy=[]
    for event in events:
        require(str(event.device_type)=='DeviceType.CPU' and not event.is_async,
            'Only synchronous CPU profiler events are attributable')
        hierarchy.append(dict(event_id=event.id,operator=event.name,thread_id=event.thread,
            parent_id=None if event.cpu_parent is None else event.cpu_parent.id,
            children_ids=[child.id for child in event.cpu_children],
            start_us=float(event.time_range.start),end_us=float(event.time_range.end),
            inclusive_elapsed_us=float(event.cpu_time_total),self_elapsed_us=float(event.self_cpu_time_total)))
    rows=[]
    for event in profile.key_averages():
        name=event.key;count=event.count
        inclusive=float(event.cpu_time_total);exclusive=float(event.self_cpu_time_total)
        require(type(name) is str and 0<len(name)<=512 and type(count) is int and count>0,
            'Valid aggregate operator identity/count')
        require(all(math.isfinite(x) and x>=0 for x in (inclusive,exclusive)) and exclusive<=inclusive+1e-6,
            'Valid profiler elapsed operator counters')
        rows.append(dict(operator=name,calls=count,inclusive_elapsed_us=inclusive,self_elapsed_us=exclusive))
    require(0<len(rows)<=MAX_OPERATOR_ROWS and len({x['operator'] for x in rows})==len(rows),
        'Bounded unique aggregate operators')
    return sorted(rows,key=lambda x:x['operator']),hierarchy


def verify_operator_hierarchy(operators,events):
    require(type(events) is list and 0<len(events)<=MAX_RAW_EVENTS,'Bounded saved operator hierarchy')
    require(type(operators) is list and 0<len(operators)<=MAX_OPERATOR_ROWS,'Bounded saved operator aggregates')
    indexed={e['event_id']:e for e in events}
    require(len(indexed)==len(events),'Unique profiler event identifiers')
    totals={}
    for event in events:
        require(set(event)=={'event_id','operator','thread_id','parent_id','children_ids','start_us','end_us',
            'inclusive_elapsed_us','self_elapsed_us'},'Exact saved event schema')
        require(type(event['event_id']) is int and type(event['thread_id']) is int and
            type(event['operator']) is str and 0<len(event['operator'])<=512,'Profiler event identity')
        require(all(type(event[k]) in (int,float) and math.isfinite(event[k]) and event[k]>=0
            for k in ('start_us','end_us','inclusive_elapsed_us','self_elapsed_us')),'Finite event intervals')
        require(event['start_us']<=event['end_us'] and
            abs(event['inclusive_elapsed_us']-(event['end_us']-event['start_us']))<=.001,
            'Profiler inclusive elapsed independently recomputes')
        children=event['children_ids']
        require(type(children) is list and len(set(children))==len(children) and
            all(x in indexed and x!=event['event_id'] for x in children),'Exact bounded child identifiers')
        ordered=sorted((indexed[x] for x in children),key=lambda x:x['start_us'])
        previous=event['start_us']
        for child in ordered:
            require(child['parent_id']==event['event_id'] and child['thread_id']==event['thread_id'] and
                child['start_us']>=previous-.001 and child['end_us']<=event['end_us']+.001,
                'Profiler child intervals belong to the immediate parent without overlap')
            previous=child['end_us']
        if event['parent_id'] is not None:
            require(event['parent_id'] in indexed and
                event['event_id'] in indexed[event['parent_id']]['children_ids'],'Exact reverse event parent link')
        ancestors={event['event_id']};parent=event['parent_id']
        while parent is not None:
            require(parent not in ancestors and parent in indexed,'Acyclic profiler event hierarchy')
            ancestors.add(parent);parent=indexed[parent]['parent_id']
        computed=event['inclusive_elapsed_us']-sum(x['inclusive_elapsed_us'] for x in ordered)
        require(computed>=-.001 and abs(computed-event['self_elapsed_us'])<=.001,
            'Profiler self elapsed independently recomputes from immediate children')
        total=totals.setdefault(event['operator'],dict(calls=0,inclusive_elapsed_us=0.,self_elapsed_us=0.))
        total['calls']+=1;total['inclusive_elapsed_us']+=event['inclusive_elapsed_us'];total['self_elapsed_us']+=event['self_elapsed_us']
    require(len({r['operator'] for r in operators})==len(operators) and
        set(totals)=={r['operator'] for r in operators},'Operator aggregates match complete event hierarchy')
    for row in operators:
        require(set(row)=={'operator','calls','inclusive_elapsed_us','self_elapsed_us'} and
            type(row['calls']) is int and row['calls']==totals[row['operator']]['calls'],'Aggregate operator call counts')
        require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0 and
            abs(row[k]-totals[row['operator']][k])<=.001 for k in ('inclusive_elapsed_us','self_elapsed_us')),
            'Aggregate operator elapsed recomputes from raw intervals')
    return dict(status='passed',events=len(events),operators=len(operators))


def measured_forward(compiled,tensor,enabled,torch,*,clock=time.perf_counter_ns,factory=profiler_factory):
    """Measure only the already-built forward; tensor conversion is outside."""
    outer_started=clock();profile=factory(torch) if enabled else None
    with profile if enabled else nullcontext():
        started=clock();output=compiled(tensor);ended=clock()
    outer_ended=clock()
    rows,events=operator_rows(profile) if enabled else ([],[])
    return output,dict(forward_wall_ns=ended-started,outer_wall_ns=outer_ended-outer_started,
        profiler_setup_teardown_wall_ns=(outer_ended-outer_started)-(ended-started),
        operators=rows,events=events,raw_event_count=len(events))


def check_profile_row(row,fixture,expected):
    require(row['input_sha256']==digest(fixture['tokens']),'Exact lawful profile input hash')
    comparison=graph.compare_outputs(fixture,expected,row['outputs'])
    require(row['comparison']==comparison and comparison['exact_float_equal'] and comparison['float32_bits_equal'],
        'Profiler-active and unprofiled full outputs must remain exact')
    metrics=row['measurement']
    require(all(type(metrics[k]) is int and metrics[k]>=0 for k in
        ('forward_wall_ns','outer_wall_ns','profiler_setup_teardown_wall_ns')),
        'Finite monotonic forward measurements')
    require(metrics['forward_wall_ns']>0 and metrics['outer_wall_ns']>=metrics['forward_wall_ns'] and
        metrics['outer_wall_ns']-metrics['forward_wall_ns']==metrics['profiler_setup_teardown_wall_ns'],
        'Exact forward and outer overhead partition')
    require(metrics['outer_wall_ns']<MAX_SINGLE_FORWARD_SECONDS*1e9,'One raw forward/profiler call exceeds10s')
    if row['spec']['mode']=='on':
        verify_operator_hierarchy(metrics['operators'],metrics['events'])
        require(metrics['raw_event_count']==len(metrics['events']),'Exact saved event count')
        fused=[x for x in metrics['operators'] if x['operator']=='aten::_transformer_encoder_layer_fwd']
        require(len(fused)==1 and fused[0]['calls']==6,'Profiler changed or hid the six fused transformer calls')
    else:require(metrics['operators']==metrics['events']==[] and metrics['raw_event_count']==0,'OFF has no profiler events')
    return comparison


def run_profile_loop(plan,expected,invoke,save_call,boundary):
    """Pure orchestration helper; failures retain full output before rejection."""
    fixtures={f['fixture_id']:f for f in plan['fixtures']};rows=[]
    for spec in plan['calls']:
        boundary();fixture=fixtures[spec['fixture_id']]
        row=dict(spec=copy.deepcopy(spec),input_sha256=digest(fixture['tokens']),outputs=None,
            comparison=None,measurement=None,error=None)
        try:
            row['outputs'],row['measurement']=invoke(fixture,spec['mode']=='on')
            row['comparison']=graph.compare_outputs(fixture,expected[fixture['fixture_id']],row['outputs'])
            check_profile_row(row,fixture,expected[fixture['fixture_id']]);boundary()
        except BaseException as error:
            row['error']=repr(error);save_call(row);raise
        save_call(row);rows.append(row)
    require(len(rows)==60,'Complete fixed raw-forward matrix')
    return rows


def profile_summary(plan,rows,call_artifacts):
    pairs=[]
    for fixture in plan['fixtures']:
        for repeat in range(4):
            pair={r['spec']['mode']:r for r in rows if r['spec']['fixture_id']==fixture['fixture_id'] and
                r['spec']['purpose']=='operator_localization' and r['spec']['repeat']==repeat}
            off,on=pair['off']['measurement'],pair['on']['measurement']
            pairs.append(dict(fixture_id=fixture['fixture_id'],repeat=repeat,
                forward_ratio=on['forward_wall_ns']/off['forward_wall_ns'],
                outer_ratio=on['outer_wall_ns']/off['outer_wall_ns'],
                forward_extra_ns=on['forward_wall_ns']-off['forward_wall_ns'],
                outer_extra_ns=on['outer_wall_ns']-off['outer_wall_ns']))
    return dict(schema=PROFILE_SCHEMA,status='passed',exact_outputs=True,exact_float32_bits=True,
        full_fused_operator_inventory_preserved=True,raw_forward_calls=60,profiled_calls=30,
        unprofiled_calls=30,plan_sha256=digest(plan),input_hashes=plan['input_hashes'],
        call_artifacts=call_artifacts,paired_overhead=pairs,
        median_forward_ratio=statistics.median(p['forward_ratio'] for p in pairs),
        median_outer_ratio=statistics.median(p['outer_ratio'] for p in pairs),
        times_are_profiler_elapsed_not_os_cpu=True,inclusive_operators_overlap=True,
        interpretation='Coarse operator localization; profiler overhead is descriptive, final speed is observer OFF.',
        new_games=0,training=False)


def verify_operator_profile(folder,freeze,result,boundary=lambda:None):
    """Saved-only complete output/bit/timer/hierarchy reconciliation."""
    from . import common as c
    folder=Path(folder).resolve()
    require(folder==(c.OUT/'workers/localization').resolve(),'Localization evidence belongs to the declared tree')
    base=folder/'operator_profile';plan=profiling_plan(freeze)
    require(c.read(base/'plan.json')==plan,'Exact frozen profile input and call matrix')
    expected_files={'plan.json','summary.json'}|{f'calls/{i:04d}.json' for i in range(60)}
    require({p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==expected_files,
        'Exact full profile evidence inventory; no omitted failed call or retry')
    parity=c.read(folder/'graph_parity.json');require(parity['summary']['passed'] is True,'Qualified original graph')
    expected={x['fixture_id']:x['candidate_outputs'] for x in parity['evidence']}
    fixtures={f['fixture_id']:f for f in plan['fixtures']};rows=[];pins=[]
    require(len(result['call_artifacts'])==60,'All raw-forward artifacts declared')
    for spec,item in zip(plan['calls'],result['call_artifacts'],strict=True):
        boundary();path=base/'calls'/f"{spec['index']:04d}.json"
        require(item==c.artifact(path),'Exact raw-forward artifact binding');c.verify([item])
        row=c.read(path)
        require(set(row)=={'spec','input_sha256','outputs','comparison','measurement','error'} and
            row['spec']==spec and row['error'] is None,'Complete accepted fixed profile row')
        check_profile_row(row,fixtures[spec['fixture_id']],expected[spec['fixture_id']])
        rows.append(row);pins.append(item)
    require(result==c.read(base/'summary.json')==profile_summary(plan,rows,pins),
        'Full independently recomputed localization summary')
    return dict(status='passed',raw_forward_calls=60,profiled_calls=30,unprofiled_calls=30,
        exact_outputs=60,exact_float32_bits=60,complete_saved_event_hierarchies=30,
        profiler_counter_scope='elapsed CPU-operator intervals, not OS thread CPU',
        plan_sha256=digest(plan))


def operator_profile(context,request):
    from . import common as c
    require(request=={'operation':'operator_profile'} and context['execution_variant']=='localization',
        'Profile operation belongs only to unchanged localization worker')
    require(context['graph_parity_passed'] and context['entrypoint_passed'] and
        not context['operator_profile_attempted'],'Profiling once after complete actual qualification')
    context['operator_profile_attempted']=True
    plan=profiling_plan(context['freeze']);folder=context['folder']/'operator_profile';folder.mkdir()
    _save(folder/'plan.json',plan)
    start=context.get('request_started') or time.perf_counter();deadline=start+PROFILE_INTERNAL_SECONDS
    def boundary():
        import psutil
        require(time.perf_counter()<deadline,'Operator localization90s internal clock')
        require(psutil.virtual_memory().available>=2*1024**3,'Operator localization requires2GiB available RAM')
    forward=context['frozen_forward'];model=context['adapters']['search'].model
    original_ir=context['folder']/'graph_ir.txt'
    baseline=c.read(context['folder']/'graph_parity.json')
    expected={r['fixture_id']:r['candidate_outputs'] for r in baseline['evidence']}
    rows=[];call_artifacts=[]
    try:
        require(baseline['summary']['passed'] is True,'Exact original graph parity evidence')
        import torch
        require(torch.get_num_threads()==torch.get_num_interop_threads()==1,'Single-thread profiling')
        with torch.inference_mode():
            tensors={f['fixture_id']:torch.tensor(f['tokens'],dtype=torch.float32).reshape(1,31,48)
                for f in plan['fixtures']}
            def invoke(fixture,enabled):
                boundary()
                output,measurement=measured_forward(forward.compiled,tensors[fixture['fixture_id']],enabled,torch)
                require(tuple(output.shape)==(1,83) and output.dtype==torch.float32 and
                    output.device.type=='cpu' and bool(torch.isfinite(output).all()),'Original float32 output shape/type')
                return output.tolist(),measurement
            def save_call(row):
                path=folder/'calls'/f"{row['spec']['index']:04d}.json"
                path.parent.mkdir(exist_ok=True);_save(path,row);call_artifacts.append(c.artifact(path))
            rows=run_profile_loop(plan,expected,invoke,save_call,boundary)
        require(forward.compiled is context['original_compiled'] and
            c.file_hash(original_ir)==forward.identity['graph_ir_sha256'] and
            hashlib.sha256((str(forward.compiled.inlined_graph)+'\n').encode()).hexdigest()==forward.identity['graph_ir_sha256'] and
            graph._weight_digest(model,torch)==forward.identity['weights_before_sha256'],
            'Profiling leaves compiled graph and weights unchanged')
        result=profile_summary(plan,rows,call_artifacts)
        _save(folder/'summary.json',result);return result
    except BaseException as error:
        _save(folder/'failure.json',dict(status='failed_preserved',error=repr(error),
            traceback=traceback.format_exc(),persisted_calls=call_artifacts,automatic_retries=0))
        raise


def handle(context,request):
    if type(request) is dict and request.get('operation')=='graph_parity':
        return accepted.handle(context,request)
    require(context['graph_parity_passed'],'Exact graph parity before any scientific call')
    if request=={'operation':'entrypoint_checks'}:
        require(not context['entrypoint_attempted'],'Actual suite is never retried')
        context['entrypoint_attempted']=True
        result=accepted.handle(context,request)
        require(result['counts']['complete_decisions']==67,'Complete corrected actual suite')
        context['entrypoint_passed']=True
        return result
    if request=={'operation':'operator_profile'}:return operator_profile(context,request)
    require(context['execution_variant'] in ('control','candidate') and context['entrypoint_passed'],
        'Ordinary decisions only after actual control/candidate qualification')
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
                    profiling=payload=={'operation':'operator_profile'}
                    startup=profiling or payload=={'operation':'entrypoint_checks'} or type(payload) is dict and set(payload)=={'operation','fixtures'} and payload['operation']=='graph_parity'
                    deadline=context['request_started']+PROFILE_INTERNAL_SECONDS if profiling else (None if startup else context['request_started']+INTERNAL_SECONDS)
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

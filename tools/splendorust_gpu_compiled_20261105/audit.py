"""Independent saved-evidence acceptance for one exact compiled GPU candidate.

Imports no model and starts no games. Historical checker code is authenticated,
loaded privately and rebound only to this study's evidence context.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys

from . import common as c
from tools.splendorust_gpu_throughput_20261105 import audit as old,qualification as q

LIBRARY=c.ROOT/'tools/splendorust_gpu_batchfill_20261105/audit.py'
LIBRARY_SHA='ef3e37d43fac0522cb44a903f37d35d3abe21b1b9f149040792e74f276f96c13'
_REUSED=None
read,digest,bits,timing=old.read,old.digest,old.bits,old.timing


def require(condition,message):
    if not condition:raise ValueError('COMPILED_GPU_AUDIT: '+message)


def load_reused():
    global _REUSED
    require(c.sha(LIBRARY)==LIBRARY_SHA,'authenticated unchanged saved checker')
    if _REUSED is None:
        spec=importlib.util.spec_from_file_location('tools.splendorust_gpu_batchfill_20261105._private_compiled_context',LIBRARY)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for key in ('ROOT','TOTAL_SECONDS','MODEL_SECONDS','INTERNAL_SECONDS','OUTER_SECONDS','EXECUTION_POLICY_ID','MAX_RSS','MAX_OUTPUT','MIN_DISK'):
            require(getattr(module.c,key)==getattr(c,key),'unchanged checker semantic constant '+key)
        public=sys.modules.get('tools.splendorust_gpu_batchfill_20261105.audit');context=None if public is None else public.c
        module.c=c;require(public is None or public.c is context,'historical public checker untouched');_REUSED=module
    require(_REUSED.c is c,'private evidence context only');return _REUSED


def source_context_proof():
    load_reused()
    return dict(status='private_saved_checker_context_verified',library=c.artifact(LIBRARY),context=c.artifact(Path(c.__file__)),
                binding='Private module.c only; no historical file or public module mutation',models=0,games=0)


def support_admission(proof):
    passed=proof['passed'] is True
    return dict(status='support_admitted' if passed else 'support_not_admitted',passed=passed,
                support_status=proof['status'],exact_physical_outputs=passed,actual_gpu_entrypoint_qualified=passed,
                no_retry=True,scope='Admits only the predetermined full-decision qualification; no timing cells or games')


def qualification_gate(support,decision):
    passed=(support['passed'] is True and decision is not None and decision['status']=='passed' and decision['eligible'] and
        len(decision['case_rows'])==32 and all(x['completed'] and x['baseline_exact'] for x in decision['case_rows']))
    return dict(status='compiled_candidate_qualified' if passed else 'compiled_candidate_not_qualified',passed=bool(passed),
                support_status=support['status'],decision_status=None if decision is None else decision['status'],
                exact_complete_decisions=32 if passed else 0,no_retry=True,
                scope='Exact physical and complete scientific-decision qualification before any timing comparison')


def output_bits(rows):
    require(type(rows) is list and len(rows)==8 and all(type(row) is list and len(row)==83 for row in rows),
            'full physicalB8 by83 outputs retained')
    require(all(type(x) in (int,float) and math.isfinite(x) for row in rows for x in row),'finite exact saved network outputs')
    return bits([value for row in rows for value in row])


def first_difference(actual,expected,rows=None):
    output_bits(actual);output_bits(expected)
    for row in range(8) if rows is None else rows:
        for column in range(83):
            left,right=actual[row][column],expected[row][column]
            left_bits,right_bits=struct.pack('<f',left).hex(),struct.pack('<f',right).hex()
            if left!=right or left_bits!=right_bits:
                return dict(row=row,column=column,actual=left,expected=right,actual_float32_hex=left_bits,expected_float32_hex=right_bits)
    return None


def expected_physical_fixtures(plan):
    cases=plan['cases'];fixtures=[]
    groups=[list(range(i,i+8)) for i in range(0,32,8)]+[[i] for i in range(32)]
    for index,indexes in enumerate(groups):
        tokens=[0.]*11904;outputs=[[0.]*83 for _ in range(8)];sources=[]
        for i in indexes:
            case=cases[i];slot=i%8;require(case['slot']==slot,'original fixed native row')
            tokens[slot*1488:(slot+1)*1488]=case['expected_root_tokens']
            outputs[slot]=old.root_output(case,read(c.resolve(case['expected_results']['gpu']))['result'])
            sources.append(dict(slot=slot,case_index=i,source=case['expected_results']['gpu']))
        fixtures.append(dict(index=index,fixture_id=f'b{index:03d}',physical_batch=8,
                             kind='full_saved_roots' if index<4 else 'single_occupied_native_row',
                             case_indexes=indexes,occupied_slots=[i%8 for i in indexes],input_tokens=tokens,
                             input_sha256=digest(tokens),expected_occupied_outputs=outputs,expected_sources=sources))
    return fixtures


def timing_summary(cells):
    expected=['b1_eager','b1_compiled','b2_compiled','b2_eager']
    require([x['cell_id'] for x in cells]==expected[:len(cells)],'one predetermined timing prefix without replacement')
    require(len(cells)<=4,'fixed maximum four timing cells')
    complete=len(cells)==4 and all(x['status']=='passed' and x['eligible'] and len(x['case_rows'])==32 and
        all(r['completed'] and r['baseline_exact'] for r in x['case_rows']) for x in cells)
    pairs=[];pooled=None
    if complete:
        for block,control,candidate in ((1,cells[0],cells[1]),(2,cells[3],cells[2])):
            require(control['implementation']=='eager' and candidate['implementation']=='compiled' and control['block']==candidate['block']==block,
                    'matched unchangedGPU16 implementation intervention only')
            require(all(math.isfinite(x['wall_seconds']) and x['wall_seconds']>0 for x in (control,candidate)),'original positive complete-panel wall')
            ratio=control['wall_seconds']/candidate['wall_seconds']
            pairs.append(dict(block=block,eager_seconds=control['wall_seconds'],compiled_seconds=candidate['wall_seconds'],
                compiled_over_eager_completed_throughput=ratio,passed_useful_threshold=ratio>=1.05))
        pooled=(cells[0]['wall_seconds']+cells[3]['wall_seconds'])/(cells[1]['wall_seconds']+cells[2]['wall_seconds'])
    return dict(status='complete_timing_comparison' if complete else 'incomplete_timing_comparison',eligible=complete,
                useful_improvement=complete and all(x['passed_useful_threshold'] for x in pairs),
                minimum_each_block_throughput_ratio=1.05,pairs=pairs,pooled_compiled_over_eager_throughput=pooled,
                no_retry=True,scope='Descriptive fixed-workload complete-decision throughput; not statistical superiority, gameplay throughput or strength')


comparison_summary=timing_summary


def compilation_identity(folder,implementation):
    from . import compilation
    folder=Path(folder);identity=read(folder/'runtime_identity.json')
    require(identity['implementation']==implementation and identity['route']=='gpu' and identity['scheduler']=='row_fifo32' and
            identity['maximum_logical_clients']==32 and identity['physical_batch_unchanged']==8 and 'timing_observer_enabled' not in identity,
            'exact unobserved eager or compiled runtime')
    if implementation=='eager':
        require(identity['compilation_fingerprint'] is None and identity['compilation_receipt'] is None,
                'eager control has no compiled graph')
        return dict(implementation=implementation,compilation_fingerprint=None)
    binding=identity['compilation_receipt'];c.verify([binding]);receipt=read(c.resolve(binding))
    require(c.resolve(binding).resolve()==folder.with_name(folder.name+'_compilation').resolve()/'compilation.json',
            'one owned separate graph-evidence directory')
    require(receipt['schema']=='fixed_b8_trace_freeze_v1' and receipt['status']=='compiled_supported_pending_exact_qualification' and
            receipt['implementation']=='compiled' and receipt['retries']==0 and receipt['timed_panel'] is False and
            receipt['exact_outputs_qualified'] is False and receipt['original_weights_unchanged'] is True,'original one-shot compilation outcome')
    graphs=receipt['graphs'];require(set(graphs)=={'traced','frozen','traced_inventory','frozen_inventory'},'complete both graph text and canonical inventory')
    c.verify(list(graphs.values()));descriptions={name:read(c.resolve(graphs[name+'_inventory'])) for name in ('traced','frozen')}
    import re
    for name,description in descriptions.items():
        operators=[]
        def visit(block):
            for node in block['nodes']:
                require(node['kind'] in compilation.ALLOWED_OPS,'only declared graph operations')
                operators.append(node['kind'])
                for sub in node['blocks']:visit(sub)
        visit(description['body'])
        require(description['operators']==operators and description['operator_inventory']==dict(Counter(operators)) and
                description['math_sequence']==[x for x in operators if x in compilation.MATH_OPS] and
                description['fingerprint']==digest(description['body']),'canonical graph topology, math inventory and fingerprint')
        text=c.resolve(graphs[name]).read_text(encoding='utf-8')
        textual=re.findall(r'=\s*((?:aten|prim)::\w+)(?:\[|\()',text)
        require(textual==operators,'independently extracted textual graph operation sequence')
    compilation.validate_graph_pair(descriptions['traced'],descriptions['frozen'])
    expected=dict(options=compilation.OPTIONS,weights_sha256=identity['weights_sha256'],device=identity['device']['torch_device'],
                  frozen_graph_fingerprint=descriptions['frozen']['fingerprint'],mathematical_operator_sequence=descriptions['frozen']['math_sequence'])
    require(receipt['binding']==expected and receipt['compilation_fingerprint']==identity['compilation_fingerprint']==digest(expected),
            'actual original weights/device/options and complete compiled graph fingerprint')
    lib=old.load_saved_library(c.OUT)
    require(not any(lib.fallback(x['message']) for x in receipt['warnings']),'no successful hidden CPU fallback')
    return dict(implementation=implementation,compilation_fingerprint=identity['compilation_fingerprint'],compilation_receipt=binding)


def audit_mismatched_case(plan,case,spec,folder,requests):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import audit as ledger
    folder=Path(folder);reused=load_reused();attempt=read(folder/'attempt.json');started=attempt['started_perf_counter']
    require(read(folder/'case.json')==case and read(folder/'request.json')==case['request'] and attempt['case_id']==case['case_id'] and
            attempt['request_sha256']==case['request_sha256'] and case['index'] in spec['groups'][attempt['group_index']],
            'negative case binds original request, beforestate, native row and wave')
    client,rows=reused.case_requests(case,spec,requests,started);clock=read(folder/'clock.json')
    require(old.validate_clock(clock) and clock['started_perf_counter']==started,'exactness rejection follows completed original clock')
    require(rows and rows[0]['input_sha256']==digest(case['expected_root_tokens']),'actual corrected root callback in negative case')
    returned=read(folder/'service_calls.json');require(len(returned)==len(rows),'all completed candidate callbacks retained')
    for actual,saved in zip(returned,rows):
        require({k:v for k,v in actual.items() if k!='client_received_perf'}==saved and
                saved['complete_readback_perf']<=actual['client_received_perf']<saved['deadline_perf'],'unaltered original callback receipts/deadlines')
    result=read(folder/'result.json');output=old.root_output(case,result)
    expected=read(c.resolve(case['expected_results']['gpu']));after=read(folder/'after.json');root=q.validate_fixed_result(case,result)
    require(read(folder/'root.json')==root,'negative root metadata independently reconstructed')
    raw=read(folder/'raw_result.json')
    require(raw['result']==result and raw['observer']==case['observer'] and raw['action']==result['action'] and raw['diagnostics']==result['diagnostics'],
            'negative full raw result retained before any acceptance check')
    require(after==ledger.transition(case['snapshot'],result['action']),'negative actual action has lawful independent corrected afterstate')
    guard.check_training_transition(case['snapshot'],result['action'],after)
    require(old.callback_hashes(result)==[digest([r['input_sha256'],r['output_sha256'],1]) for r in rows] and
            all(r['complete_readback_perf']<=clock['ended_perf_counter'] for r in rows),'negative full scientific callback trace remains valid')
    baseline_output=old.root_output(case,expected['result'])
    exact=(result==expected['result'] and digest(result)==digest(expected['result']) and after==expected['after'] and bits(output)==bits(baseline_output))
    require(not exact,'valid negative independently proves real output/decision/afterstate mismatch')
    mismatch=read(folder/'exact_mismatch.json')
    require(mismatch==dict(kind='compiled_full_decision_exact_mismatch',actual_result=c.artifact(folder/'result.json'),
        expected_result=case['expected_results']['gpu'],actual_after=c.artifact(folder/'after.json'),actual_root=c.artifact(folder/'root.json'),
        completed_original_clock=c.artifact(folder/'clock.json'),no_retry=True),'exact saved rejection binds actual and frozen expected artifacts')
    return dict(case_id=case['case_id'],client_id=client,attempted=True,completed=True,callbacks=len(rows),baseline_exact=False,
                elapsed_seconds=clock['elapsed_seconds'],clock=clock,mismatch=c.artifact(folder/'exact_mismatch.json'),
                actual_result_sha256=digest(result),expected_result_sha256=digest(expected['result']),root_values_equal=output==baseline_output,
                root_float32_bytes_equal=bits(output)==bits(baseline_output))


def audit_negative_cell(plan,spec,folder):
    folder=Path(folder);reused=load_reused();execution,absent=reused.audit_stage_execution(plan,folder.parent,mode='cell',cell=spec)
    require(execution['status']=='exact_mismatch' and execution['worker_exit_code']==21 and execution['outer_deadline_observed'] is None and
            spec['implementation']=='compiled','compiled numerical refusal distinguished from technical/deadline failure')
    require(read(folder/'runtime_sources.json')['sources']==plan['source_guard'],'actual corrected source guard')
    events,requests,scheduling=reused.read_journal(folder/'runtime','gpu');warmup=old.audit_warmup(plan,folder,requests)
    closure=reused.audit_runtime(plan,folder/'runtime','gpu',events,requests,scheduling,'row_fifo32')
    require(closure['service_failure'] is None,'numerical refusal is not hidden poisoned service')
    summary=read(folder/'cell/cell_summary.json');worker=read(folder/'worker_result.json');rows=[];outcomes={}
    for case in plan['cases']:
        path=folder/'cell'/case['case_id']
        row=audit_mismatched_case(plan,case,spec,path,requests) if (path/'exact_mismatch.json').exists() else reused.audit_case(plan,case,spec,path,requests)
        row.pop('result',None);rows.append(row)
        if row['attempted']:
            outcome=read(path/'outcome.json');expected_status='completed' if row['baseline_exact'] else 'exact_mismatch'
            require(outcome['status']==expected_status and outcome['clock']==row['clock'] and outcome['case_id']==case['case_id'] and
                    outcome['index']==case['index'] and outcome['client_id']==row['client_id'] and outcome['decision_started'] is True and
                    outcome['request_sha256']==case['request_sha256'] and outcome['no_retry'] is True,'exact rejection/peer outcome original identity')
            if expected_status=='exact_mismatch':require(read(path/'failure.json')==outcome,'retained original negative exception and trace')
            else:
                require(outcome['result_sha256']==digest(read(path/'result.json')) and outcome['after_sha256']==digest(read(path/'after.json')),
                        'unaffected peer full results retain original hashes')
            outcomes[case['index']]=outcome
    groups=summary['groups'];require(1<=len(groups)<=2,'one exact completed wave prefix before refusal')
    indexes=[]
    for group_index,group in enumerate(groups):
        members=spec['groups'][group_index];indexes+=members;path=folder/'cell/groups'/f'{group_index:03d}'
        require(group==read(path/'summary.json') and group['case_indexes']==members and group['case_ids']==[plan['cases'][i]['case_id'] for i in members] and
                group['outcomes']==[outcomes[i] for i in members] and group['group_index']==group_index,'whole already-released wave preserved')
        require(read(path/'start.json')==dict(group_index=group_index,case_indexes=members,started_perf_counter=group['started_perf_counter'],
            started_unix=group['started_unix']) and group['wall_seconds']==group['ended_perf_counter']-group['started_perf_counter']>0,
            'original wave clock and exact barrier binding')
        require(group['attempted']==16 and group['completed']==sum(rows[i]['baseline_exact'] for i in members),'negative is completed scientifically but not accepted')
        require(all(group['started_perf_counter']<=rows[i]['clock']['started_perf_counter']<=rows[i]['clock']['ended_perf_counter']<=group['ended_perf_counter'] for i in members),
                'all affected and peer decisions inside original wave')
        wanted='exact_mismatch' if any(not rows[i]['baseline_exact'] for i in members) else 'passed'
        require(group['status']==wanted and (group_index==len(groups)-1 or wanted=='passed'),'stop immediately after first failed wave')
        if group_index:require(groups[group_index-1]['ended_perf_counter']<=group['started_perf_counter'],'ordered nonoverlapping waves')
    require(groups[-1]['status']=='exact_mismatch' and set(outcomes)==set(indexes),'some proven mismatch, no skipped or added request')
    require(summary['status']=='exact_mismatch' and summary['cell_id']==spec['cell_id'] and summary['implementation']=='compiled' and
            summary['phase']==spec['phase'] and summary['route']=='gpu' and summary['concurrency']==16 and summary['planned_case_count']==32 and
            summary['attempted_case_ids']==[x['case_id'] for x in rows if x['attempted']] and
            summary['completed_case_ids']==[x['case_id'] for x in rows if x['attempted'] and x['baseline_exact']] and
            summary['never_started_case_ids']==[x['case_id'] for x in rows if not x['attempted']] and summary['no_retry'] is True and
            summary['game_continuations']==0 and summary['full_cell_completed'] is False and summary['intentional_intergroup_gap_seconds']==0.,
            'exact original refusal summary and partition')
    require(summary['started_perf_counter']==groups[0]['started_perf_counter'] and summary['started_unix']==groups[0]['started_unix'] and
            summary['ended_perf_counter']>=groups[-1]['ended_perf_counter'] and summary['wall_seconds']==summary['ended_perf_counter']-summary['started_perf_counter'] and
            warmup['calls'][-1]['clock']['ended_perf_counter']<summary['started_perf_counter'],'original full cell timer and excluded warmup')
    ready=read(folder/'cell/adapters_ready.json')
    require(ready['count']==16 and ready['prepared_before_panel'] is True and ready['observed_perf_counter']<summary['started_perf_counter'] and
            ready['clients']==[dict(index=i,client_id=f'worker_{i}',slot=i%8) for i in range(16)],'same precreated persistent GPU16 clients')
    require(len(requests)==5+sum(x['callbacks'] for x in rows),'all physical requests accounted without retry')
    for event in scheduling:
        for request in event['ready']:
            origins=([x['clock']['started_perf_counter'] for x in warmup['calls']] if request['client_id']=='warmup' else
                [x['clock']['started_perf_counter'] for x in rows if x['attempted'] and x['client_id']==request['client_id']])
            require(sum(request['deadline_perf']==origin+60. and request['submitted_perf']>=origin for origin in origins)==1,
                    'every candidate scheduler request retains its own original clock')
    require(worker['mode']=='cell' and worker['cell_id']==spec['cell_id'] and worker['result']==summary and worker['exit_code']==21 and
            worker['reference_weights_unchanged'] is True and worker['no_retry'] is True and read(folder/'cleanup.json')['success'] is True,
            'actual numerical refusal exit, weights and exact cleanup')
    return dict(status='exact_mismatch',cell_id=spec['cell_id'],configuration=spec['configuration'],block=spec['block'],route='gpu',concurrency=16,
                scheduler='row_fifo32',eligible=False,completed_decisions_per_second=None,wall_seconds=summary['wall_seconds'],case_rows=rows,
                runtime_closure=closure,owned_processes_absent=absent,callback_count=len(requests)-5,warmup_calls=5,
                mismatch_case_ids=[x['case_id'] for x in rows if x.get('mismatch')],model_calls=0,new_games=0,
                scope='Completed legal scientific responses with independently proven exactness failure; no speed eligibility')


def audit_cell(plan,spec,worker_folder):
    folder=Path(worker_folder);reused=load_reused();execution=read(folder.parent/'execution.json')
    proof=audit_negative_cell(plan,spec,folder) if execution['status']=='exact_mismatch' else reused.audit_cell(plan,spec,folder)
    implementation=compilation_identity(folder/'runtime',spec['implementation'])
    initial=audit_initialization(folder,'runtime',spec['implementation'],True,read(folder/'warmup.json')['calls'][0]['clock'])
    return dict(**proof,implementation=spec['implementation'],phase=spec['phase'],compiled_identity=implementation,initialization=initial)


def audit_initialization(folder,name,implementation,succeeded,first_clock=None):
    folder=Path(folder);path=folder/'initialization'/f'{name}.json';value=read(path)
    worker=read(folder/'worker_identity.json');execution=read(folder.parent/'execution.json')
    require(value['schema']=='compiled_runtime_initialization_v1' and value['implementation']==implementation and
            Path(value['runtime_folder']).resolve()==(folder/name).resolve() and value['status']==('completed' if succeeded else 'failed') and
            (value['error'] is None if succeeded else bool(value['error'])) and value['no_retry'] is True and value['inside_panel_timing'] is False,
            'one durably preserved complete runtime initialization outcome')
    require(all(type(value[x]) in (int,float) and math.isfinite(value[x]) for x in ('started_perf_counter','ended_perf_counter','elapsed_seconds','started_unix','ended_unix')) and
            value['elapsed_seconds']==value['ended_perf_counter']-value['started_perf_counter']>=0 and
            worker['started_unix']<=value['started_unix']<=value['ended_unix']<=execution['unix_end'],'actual preparation interval inside owned stage')
    if first_clock is not None:
        require(succeeded and value['ended_perf_counter']<first_clock['started_perf_counter'],'complete preparation excluded from original support/panel requests')
    return dict(artifact=c.artifact(path),implementation=implementation,status=value['status'],elapsed_seconds=value['elapsed_seconds'],
                scope='Complete runtime preparation, including candidate compilation/graph checks; not isolated compiler timing')


def audit_scheduler_after_physical(events,requests,scheduling,physical_clocks):
    """Explicitly account original ticket allocation for prior physical requests."""
    from copy import deepcopy
    for clock in physical_clocks:require(old.validate_clock(clock) and clock['elapsed_seconds']<60.,'completed original physical prefix')
    require(all(a['ended_perf_counter']<=b['started_perf_counter'] for a,b in zip(physical_clocks,physical_clocks[1:])),
            'serial unreset physical request prefix')
    offset=len(physical_clocks);adjusted=deepcopy(scheduling)
    for event in adjusted:
        for row in event['ready']:
            require(type(row['enqueue_sequence']) is int and row['enqueue_sequence']>offset and
                    (not physical_clocks or row['submitted_perf']>physical_clocks[-1]['ended_perf_counter']),
                    'native requests strictly follow every authenticated physical ticket')
            row['enqueue_sequence']-=offset
    proof=load_reused().audit_scheduler(events,requests,adjusted,'row_fifo32')
    return dict(**proof,preceding_physical_requests=offset,raw_ticket_offset=offset,
                ticket_scope='Raw journal preserved; logical FIFO proof subtracts only authenticated serial physical prefix from every native ticket')


def physical_requests(fixtures):
    zero=[0.]*11904
    requests=[dict(phase='warmup',fixture_id='zero',repetition=i,input_tokens=zero,input_sha256=digest(zero),occupied_slots=[]) for i in range(5)]
    requests += [dict(fixture,phase='fixtures',repetition=repetition) for fixture in fixtures for repetition in range(2)]
    return requests


def classify_physical_refusal(value,clock):
    from . import compilation
    old.validate_clock(clock)
    require(value['schema']=='compiled_physical_support_refusal_v1' and value['implementation']=='compiled' and
            value['reason_kind']=='runtime_unsupported' and value['no_retry'] is True and value['traceback'] and
            clock['status']=='error' and clock['completed'] is False and clock['elapsed_seconds']<60.,
            'explicit unsupported compiled operation under original unexpired physical clock')
    chain=value['exception_chain'];require(chain,'original typed exception cause chain')
    wrappers={'Physical qualification inference failed','Inference service failed; no retry'};unsupported=[]
    for index,row in enumerate(chain):
        if index<len(chain)-1:
            require(row['type']=='RuntimeError' and row['module']=='builtins' and row['message'] in wrappers,
                    'only exact service wrappers above original unsupported cause');continue
        constructor={'RuntimeError':RuntimeError,'ValueError':ValueError,'NotImplementedError':NotImplementedError}.get(row['type'])
        require(row['module']=='builtins' and constructor is not None and
                compilation.rejection_kind(constructor(row['message']))=='unsupported_operation',
                'only actual backend unsupported operation, never resource/I/O/clock/checker failure')
        unsupported.append(row)
    require(unsupported,'independently recognized original unsupported cause')
    first=chain[0];constructor={'RuntimeError':RuntimeError,'ValueError':ValueError,'NotImplementedError':NotImplementedError}[first['type']]
    require(clock['error']==repr(constructor(first['message'])),'original clock error binds outer exception chain exactly')
    return dict(reason_kind='runtime_unsupported',original_unsupported_errors=unsupported)


def audit_physical_support(folder,implementation,fixtures,*,eager_folder=None,allow_mismatch=False,allow_runtime_refusal=False):
    folder=Path(folder);expected_requests=physical_requests(fixtures);results=[];comparisons=[];clocks=[];mismatch=None;refusal=None;expected_paths=[]
    for request in expected_requests:
        relative=Path('warmup')/f"{request['repetition']:03d}" if request['phase']=='warmup' else Path('fixtures')/request['fixture_id']/f"{request['repetition']:03d}"
        target=folder/relative
        if not target.exists():break
        require(mismatch is None,'no request after first candidate exactness refusal')
        require(read(target/'request.json')==request,'original predetermined physical request and input retained')
        if not (target/'result.json').exists():
            require(allow_runtime_refusal and implementation=='compiled','only explicitly classified compiled support refusal can lack an output')
            value=read(target/'failure.json');clock=read(target/'clock.json');classify_physical_refusal(value,clock)
            require(value['phase']==request['phase'] and value['fixture_id']==request['fixture_id'] and value['repetition']==request['repetition'] and
                    value['request']==c.artifact(target/'request.json') and value['clock']==c.artifact(target/'clock.json'),
                    'runtime refusal binds exact original request and unsuccessful clock')
            refusal=c.artifact(target/'failure.json');clocks.append(clock)
            expected_paths += [target/'request.json',target/'clock.json',target/'failure.json'];break
        result=read(target/'result.json');clock=read(target/'clock.json');old.validate_clock(clock)
        require(result['implementation']==implementation and result['phase']==request['phase'] and result['fixture_id']==request['fixture_id'] and
                result['repetition']==request['repetition'] and result['input_sha256']==request['input_sha256'] and result['physical_batch']==8 and
                result['occupied_slots']==request['occupied_slots'] and result['clock']==clock and clock['status']=='ok' and clock['elapsed_seconds']<60. and
                result['no_retry'] is True and result['output_float32_sha256']==hashlib.sha256(output_bits(result['outputs'])).hexdigest(),
                'all raw8x83 outputs/bytes and original complete clock preserved before comparison')
        artifact=c.artifact(target/'result.json');results.append(artifact);clocks.append(clock)
        expected_paths += [target/'request.json',target/'result.json',target/'clock.json']
        checks=[]
        if implementation=='compiled':
            expected_path=Path(eager_folder)/relative/'result.json';expected=read(expected_path)
            require(expected['implementation']=='eager' and expected['input_sha256']==request['input_sha256'],'same physical reference input')
            checks.append(('against_eager.json','cross_implementation_all_float32_rows',expected['outputs'],c.artifact(expected_path),None,[]))
        elif request['repetition']>0:
            path=folder/(Path('warmup')/'000' if request['phase']=='warmup' else Path('fixtures')/request['fixture_id']/'000')/'result.json'
            checks.append(('against_first.json','eager_repeat_all_float32_rows',read(path)['outputs'],c.artifact(path),None,[]))
        if request['phase']=='fixtures':
            checks.append(('against_saved.json','accepted_saved_occupied_rows',request['expected_occupied_outputs'],None,request['occupied_slots'],request['expected_sources']))
        for filename,kind,expected,expected_artifact,slots,sources in checks:
            difference=first_difference(result['outputs'],expected,slots)
            value=dict(status='passed' if difference is None else 'exact_mismatch',kind=kind,implementation=implementation,
                phase=request['phase'],fixture_id=request['fixture_id'],repetition=request['repetition'],actual=artifact,
                expected=expected_artifact,expected_sources=sources,occupied_slots=slots,difference=difference,
                all_rows_compared=slots is None,no_retry=True)
            require(read(target/filename)==value,'independently rederived exact comparison and first float32 difference')
            bound=c.artifact(target/filename);comparisons.append(bound);expected_paths.append(target/filename)
            if difference is not None:
                require(allow_mismatch and implementation=='compiled','unchanged eager prerequisite must remain exact')
                mismatch=bound;break
        if mismatch is not None:break
    require(results or refusal,'a completed physical prefix or explicit first-request refusal')
    require(all(a['ended_perf_counter']<=b['started_perf_counter'] for a,b in zip(clocks,clocks[1:])),
            'each separate physical request has one unreset ordered original clock')
    complete=len(results)==77 and mismatch is None and refusal is None
    if complete:
        summary=read(folder/'summary.json');expected_paths.append(folder/'summary.json')
        require(summary==dict(status='exact_physical_support_passed',implementation=implementation,warmup_calls=5,fixture_calls=72,
            complete_calls=77,results=results,comparisons=comparisons,outputs_retained_before_comparison=True,physical_batch=8,no_retry=True,
            ordinary_decisions=0,new_games=0,training_updates=0),'complete exact physical support summary independently reconstructed')
    else:require(((mismatch is not None and allow_mismatch) or (refusal is not None and allow_runtime_refusal)) and not (folder/'summary.json').exists(),
                 'only proven first exactness/support refusal is a valid prefix')
    actual_paths={p.resolve() for p in folder.rglob('*') if p.is_file()}
    require(actual_paths=={p.resolve() for p in expected_paths},'entire support inventory is exactly prescribed prefix; no hidden retry or missing comparison')
    return dict(status='passed' if complete else ('runtime_unsupported' if refusal else 'exact_output_mismatch'),implementation=implementation,complete_calls=len(results),
                warmup_calls=min(len(results),5),fixture_calls=max(0,len(results)-5),clocks=clocks,results=results,comparisons=comparisons,
                first_mismatch=mismatch,runtime_refusal=refusal,complete=complete)


def refused_runtime_closure(plan,folder,refusal):
    folder=Path(folder);identity=read(folder/'runtime_identity.json')
    require(identity['weights_sha256']==old.WEIGHTS and identity['checkpoint_sha256']==plan['runtime']['checkpoint']['sha256'] and
            identity['runtime_sources']==plan['source_guard'] and identity['implementation']=='compiled' and identity['route']=='gpu' and
            identity['execution_policy_id']==c.EXECUTION_POLICY_ID and identity['threads']==identity['interop_threads']==1,
            'rejected runtime actual original model/source identity')
    final,closed=read(folder/'runtime_final.json'),read(folder/'final_identity.json')
    require(final['unchanged_weights_sha256']==old.WEIGHTS and closed['weights_verified_unchanged'] is True and closed['service_closed'] is True and
            closed['active_operations']=={} and read(folder/'active_decisions.json')['active']==[],'rejected candidate exact original-weight and operation cleanup')
    leaf=refusal['exception_chain'][-1];constructor={'RuntimeError':RuntimeError,'ValueError':ValueError,'NotImplementedError':NotImplementedError}[leaf['type']]
    require({k:v for k,v in closed['identity'].items() if k!='warnings'}=={k:v for k,v in identity.items() if k!='warnings'} and
            final['service_batches']==closed['physical_batches']==0 and final['service_requests']==closed['completed_requests']==0 and
            final['service_failure']==repr(constructor(leaf['message'])),'poisoned physical-only candidate binds exact unsupported cause; no native service calls')
    return dict(initial_identity_verified=True,final_weights_verified=True,warnings=final['warnings'],service_failure=final['service_failure'],
                scope='Unsupported candidate refusal only; original weights and cleanup verified, unsupported warnings retained, no numerical/timing acceptance')


def audit_fresh_entrypoint(plan,folder,events,requests,scheduling,physical_clocks):
    folder=Path(folder);path=folder/'qualification_gpu';proof=q.verify_qualification(plan['runtime'],path)
    require(proof==read(path/'saved_qualification_audit.json'),'fresh67 actual compiled entrypoint reconstructed')
    lib=old.load_saved_library(folder.parent);forensic,_=q.read_forensics(path/'entrypoint_forensics.jsonl')
    hashes={'search':[],'reference':[]};clocks=[e for e in forensic if e.get('operation') in ('outer_clock','outer_ordinary_clock')]
    for clock in clocks:old.validate_clock(clock)
    for event in forensic:
        if event.get('event')=='result' and event.get('operation') in ('ordinary_worker_request','reference_choose','direct_same_quota_choose','exchange'):
            hashes[event['arm']].extend(lib.result_callback_hashes(event['result']))
    for arm,slot in (('search',0),('reference',1)):
        for row in lib.bind_client_callbacks('qualification_'+arm,hashes[arm],requests,slot=slot):
            matching=[clock for clock in clocks if row['deadline_perf']==clock['started_perf_counter']+60. and
                      clock['started_perf_counter']<=row['submitted_perf'] and row['complete_readback_perf']<=clock['started_perf_counter']+clock['elapsed_seconds']]
            require(len(matching)==1 and matching[0]['status']=='ok','every fresh native callback keeps original60s operation clock')
    require(sum(map(len,hashes.values()))==len(requests),'all native qualification callbacks and no extra searches')
    scheduler=audit_scheduler_after_physical(events,requests,scheduling,physical_clocks)
    load_reused().validate_client_sequences(requests)
    return dict(proof=proof,callbacks={k:len(v) for k,v in hashes.items()},scheduler=scheduler)


def audit_qualification(plan,container):
    from . import compilation
    container=Path(container);folder=container/'worker';reused=load_reused()
    execution,absent=audit_support_stage_execution(plan,container)
    require(execution['status']=='completed' and execution['worker_exit_code']==0 and
            read(folder/'runtime_sources.json')['sources']==plan['source_guard'],'actual support worker exit and corrected source guard')
    worker=read(folder/'worker_result.json');result=worker['result']
    require(worker['mode']=='qualification' and worker['cell_id'] is None and worker['exit_code']==0 and worker['no_retry'] is True and
            worker['reference_weights_unchanged'] is True and worker['new_games']==worker['training_updates']==0 and read(folder/'cleanup.json')['success'] is True,
            'successful owning process and unchanged weight/cleanup receipts, distinct from candidate acceptance')
    fixtures=expected_physical_fixtures(plan);require(read(folder/'support/fixtures.json')==fixtures,'all36 original fixed physical fixture bindings')
    eager=audit_physical_support(folder/'support/eager','eager',fixtures)
    require(eager['complete'],'all eager prerequisites independently accepted')
    events,requests,scheduling=reused.read_journal(folder/'eager','gpu')
    require(not events and not requests and not scheduling,'eager support has no native search calls')
    eager_closure=reused.audit_runtime(plan,folder/'eager','gpu',events,requests,scheduling,'row_fifo32')
    require(eager_closure['service_failure'] is None,'eager support never poisoned');compilation_identity(folder/'eager','eager')
    eager_initial=audit_initialization(folder,'eager','eager',True,eager['clocks'][0])
    require({p.name for p in (folder/'initialization').iterdir()}=={'eager.json','compiled.json'},'exactly one eager and one candidate preparation')
    require(read(folder/'initialization/compiled.json')['started_perf_counter']>eager['clocks'][-1]['ended_perf_counter'],
            'all eager support completes before candidate initialization')
    base=dict(eager=eager,eager_initialization=eager_initial,owned_processes_absent=absent,
              reused_qualification=plan['reused_qualification'],native_fallback_lines=execution['native_fallback_lines'],model_calls=0,new_games=0)
    c.verify([plan['reused_qualification']]);inherited=read(c.resolve(plan['reused_qualification']))
    require(inherited['status']=='saved_batchfill_diagnostic_verified','accepted prior GPU packing/collision and numerical bridge')
    if result['status']=='qualification_rejected':
        require(read(folder/'qualification_rejection.json')==result and result['fresh_decisions']==result['timing_cells_started']==result['panel_decisions']==0 and
                result['new_games']==result['training_updates']==0 and result['no_retry'] is True and result['traceback'] and
                result['completed_physical_support']=={'eager':read(folder/'support/eager/summary.json')},'original first refusal with no later qualification/timing work')
        c.verify([result['rejection']]);require(not (folder/'qualification_gpu').exists(),'no actual entrypoint after rejected support')
        if result['reason_kind']=='compile_rejected':
            path=folder/'compiled_compilation/compilation_failure.json'
            require(result['phase']=='initialize_compiled' and result['rejection']==c.artifact(path) and
                    compilation.classify_saved_compile_failure(read(path))=='candidate_rejected','independent source-pinned compilation refusal classification')
            require(not (folder/'compiled').exists() and not (folder/'support/compiled').exists(),'unsupported compile never creates a service or support calls')
            c.verify(list(read(path)['graphs'].values()))
            return dict(**base,status='compiled_unsupported',passed=False,reason_kind='compile_rejected',rejection=result['rejection'],
                        original_failure=read(path),compiled_identity=None,compilation_fingerprint=None,
                        compiled_initialization=audit_initialization(folder,'compiled','compiled',False))
        require(result['reason_kind'] in ('exact_output_mismatch','runtime_unsupported') and result['phase']=='compiled_fixed_physical_support','only declared compiled numerical/support refusal')
        is_runtime=result['reason_kind']=='runtime_unsupported'
        compiled=audit_physical_support(folder/'support/compiled','compiled',fixtures,eager_folder=folder/'support/eager',
                                        allow_mismatch=not is_runtime,allow_runtime_refusal=is_runtime)
        require(not compiled['complete'] and (compiled['runtime_refusal'] if is_runtime else compiled['first_mismatch'])==result['rejection'],
                'retained first candidate refusal independently reproduced')
        events,requests,scheduling=reused.read_journal(folder/'compiled','gpu');require(not events and not requests and not scheduling,'numerical refusal before native calls')
        closure=(refused_runtime_closure(plan,folder/'compiled',read(c.resolve(compiled['runtime_refusal']))) if is_runtime else reused.audit_runtime(plan,folder/'compiled','gpu',events,requests,scheduling,'row_fifo32'))
        if not is_runtime:require(closure['service_failure'] is None,'exact-output negative without poisoned service')
        identity=compilation_identity(folder/'compiled','compiled')
        return dict(**base,status=result['reason_kind'],passed=False,compiled=compiled,rejection=result['rejection'],runtime_closure=closure,
                    compiled_identity=c.artifact(folder/'compiled/runtime_identity.json'),compilation_fingerprint=identity['compilation_fingerprint'],
                    compiled_initialization=audit_initialization(folder,'compiled','compiled',True,compiled['clocks'][0]))
    require(result['status']=='qualification_passed','passed support or explicit classified refusal only')
    compiled=audit_physical_support(folder/'support/compiled','compiled',fixtures,eager_folder=folder/'support/eager')
    require(compiled['complete'],'all candidate physical fixtures exactly match eager and accepted occupied rows')
    events,requests,scheduling=reused.read_journal(folder/'compiled','gpu')
    closure=old.audit_runtime(plan,folder/'compiled','gpu',events,requests);require(closure['service_failure'] is None,'actual compiled qualifier runtime closed unpoisoned')
    identity=compilation_identity(folder/'compiled','compiled');entrypoint=audit_fresh_entrypoint(plan,folder,events,requests,scheduling,compiled['clocks'])
    require(result['physical_support']=={i:read(folder/'support'/i/'summary.json') for i in ('eager','compiled')} and
            result['gpu']==read(folder/'qualification_gpu/qualification.json') and result['fresh_decisions']==67 and
            result['reused_qualification']==plan['reused_qualification'] and result['compilation_fingerprint']==identity['compilation_fingerprint'] and
            result['timing_cells_started']==result['panel_decisions']==result['new_games']==result['training_updates']==0,'actual candidate support/entrypoint result matches raw evidence')
    return dict(**base,status='compiled_support_and_entrypoint_verified',passed=True,compiled=compiled,entrypoint=entrypoint,runtime_closure=closure,
                compiled_identity=c.artifact(folder/'compiled/runtime_identity.json'),compilation_fingerprint=identity['compilation_fingerprint'],
                compiled_initialization=audit_initialization(folder,'compiled','compiled',True,compiled['clocks'][0]))


audit_support=audit_qualification


def audit_support_stage_execution(plan,container):
    """Allow preserved stderr only for an independently proven candidate refusal."""
    container=Path(container);execution=read(container/'execution.json')
    if not execution['native_fallback_lines']:
        return load_reused().audit_stage_execution(plan,container,mode='qualification')
    owner=read(container/'owner.json');worker=read(container/'worker/worker_identity.json')
    outcome=read(container/'worker/worker_result.json')['result']
    require(outcome['status']=='qualification_rejected' and outcome['reason_kind'] in ('compile_rejected','runtime_unsupported'),
            'stderr fallback can only be retained as classified support refusal, never numerical or timing success')
    require(owner['identity']==execution['identity']==dict(pid=worker['pid'],created=worker['created']) and
            owner['stage_id']==execution['stage_id']=='qualification' and owner['mode']==execution['mode']==worker['mode']=='qualification' and
            worker['cell_id'] is None and owner['plan_sha256']==execution['plan_sha256']==c.sha(c.PLAN),'exact rejected support stage identity')
    require(owner['maximum_seconds']==execution['maximum_seconds']==plan['worker_limits']['qualification_seconds']==2700 and
            owner['monotonic_origin']==execution['monotonic_origin'] and owner['unix_origin']==execution['unix_origin'] and
            0<=execution['monotonic_end']-execution['monotonic_origin']<2700 and
            0<=execution['unix_end']-execution['unix_origin']<2700,'original rejected-stage dual clocks')
    require(execution['status']=='completed' and execution['worker_exit_code']==0 and execution['retries']==0 and not execution['cleanup_errors'] and
            0<=execution['peak_rss_bytes']<plan['maximum_worker_rss_bytes'] and worker['worker_affinity']==plan['worker_affinity'],
            'actual clean rejected support completion, resource bounds and placement')
    c.verify([owner['global_claim']]);claim=read(c.resolve(owner['global_claim']))
    require(worker['global_claim']==owner['global_claim'] and claim['plan_sha256']==execution['plan_sha256'] and
            claim['model_seconds']==plan['model_seconds']==c.MODEL_SECONDS and claim['execution_policy_id']==worker['execution_policy_id']==c.EXECUTION_POLICY_ID and
            claim['monotonic_origin']<=execution['monotonic_origin']<=execution['monotonic_end']<claim['monotonic_origin']+c.MODEL_SECONDS and
            claim['unix_origin']<=execution['unix_origin']<=execution['unix_end']<claim['unix_origin']+c.MODEL_SECONDS,'unextended global rejected-stage model clock')
    lib=old.load_saved_library(container);lines=(container/'worker/stderr.log').read_text(encoding='utf-8',errors='replace').splitlines()
    from tools.splendorust_gpu_probe_20261105.control import native_fallback_messages
    require(execution['native_fallback_lines']==native_fallback_messages(container/'worker/stderr.log') and
            all(lib.fallback(line) for line in execution['native_fallback_lines']),'exact preserved independently recognized native warnings')
    require(all(line in execution['native_fallback_lines'] for line in lines if lib.fallback(line)),'no omitted unsupported native warnings')
    absent=old.require_processes_absent(execution['observed_owned_identities']+[execution['identity']])
    return execution,absent


def validate_plan(plan):
    from . import design,backend,compilation
    require(plan['schema']=='splendorust_gpu_compiled_v1' and plan['cells']==design.cells() and
            plan['qualification_cell']==design.qualification_cell() and plan['support']==design.support(),'one predetermined compiled candidate and full sequence')
    require(plan['worker_limits']==dict(qualification_seconds=2700,cell_seconds=900) and
            plan['total_seconds']==c.TOTAL_SECONDS==14400 and plan['model_seconds']==c.MODEL_SECONDS==10800 and
            plan['internal_seconds']==60. and plan['outer_seconds']==75. and plan['execution_policy_id']==c.EXECUTION_POLICY_ID,
            'unchanged complete decision and bounded actual run clocks')
    require(plan['maximum_worker_rss_bytes']==c.MAX_RSS and plan['maximum_output_bytes']==c.MAX_OUTPUT and plan['minimum_free_disk_bytes']==c.MIN_DISK and
            plan['available_ram_reserve_bytes']==0 and plan['forecast_admission_gate'] is False,'actual resource caps; no reserve/forecast admission')
    require(plan['worker_affinity']==list(range(19)) and plan['supervisor_affinity']==[19] and plan['group_gap_seconds']==0. and
            plan['timing_observer'] is False,'unchanged host placement and no cost observer')
    require(plan['compilation']==dict(method='torch.jit.trace_then_freeze',physical_batch=8,example='all_zero_float32_on_selected_GPU_shape_8_31_48',
        check_trace=False,strict=True,optimize_numerics=False,alternative_options=0,precision='float32',
        primitive_source='tools/splendorust_gpu_probe_20261105/worker.py:make_primitive_forward',require_same_mathematical_operator_order=True,
        require_exact_outputs=True),'single unchanged float32 primitive compiler choice, no numerical relaxation')
    require(plan['maximum_panel_decisions']==160 and plan['timing_decisions']==128 and plan['qualification_decisions_per_route']==67 and
            plan['qualification_routes']==['compiled_gpu'] and plan['retries']==plan['new_games']==plan['training_updates']==0 and
            plan['larger_data_deferred'] is True and plan['automatic_followup_games'] is False and plan['automatic_followup_training'] is False,'fixed saved-only bounded sample')
    require(plan['useful_signal']==dict(minimum_throughput_ratio_each_pair=1.05,complete_exact_cells_required=4,statistical_significance_claim=False,
        automatic_promotion=False) and plan['late_completed_responses_accepted'] is True and plan['descriptive_latency_threshold_seconds']==[1.,9.,10.],
        'prospective descriptive throughput criterion and unchanged deadline policy')
    require(plan['compilation_source_equivalence']==backend.source_equivalence(),'independently rederived complete unchanged eager/backend source')
    for artifact,expected in ((plan['prior']['plan'],design.PLAN_SHA),(plan['prior']['audit'],design.AUDIT_SHA),(plan['prior']['closeout'],design.CLOSEOUT_SHA)):
        require(artifact['sha256']==expected,'accepted predecessor identity');c.verify([artifact])
    previous=read(c.resolve(plan['prior']['plan']))
    require(plan['runtime']==previous['runtime'] and plan['cases']==previous['cases'] and plan['warmup']==previous['warmup'] and
            plan['source_guard']==previous['source_guard'] and plan['prior']['frozen_inputs']==previous['inputs'] and
            plan['reused_qualification']==previous['reused_scheduler_qualification'],'exact frozen original model, rules, fixtures and preserved evidence')
    require(plan['runtime']['qualification']==q.build_plan(),'unchanged actual scientific entrypoint qualification')
    load_reused().validate_cases(plan['cases']);source_context_proof();return True


def verify_admission(plan,spec,container,support,gate,gate_path):
    container=Path(container);owner=read(container/'owner.json');support_only=spec['phase']=='decision_qualification'
    option='--support-sha256' if support_only else '--qualification-sha256';args=owner['args']
    require(args.count(option)==1 and args[args.index(option)+1]==c.sha(gate_path),'actual fresh worker binds original admission SHA')
    expected=dict(admission=c.artifact(gate_path),source_audits=gate['source_audits'],passed=True,independently_recomputed=True,
        verified_before_model_creation=True,compiled_identity=support['compiled_identity'],compilation_fingerprint=support['compilation_fingerprint'])
    filename='support_admission_binding.json' if support_only else 'qualification_gate_binding.json'
    require(read(container/'worker'/filename)==expected,'worker independently verified original complete admission before models')
    if spec['implementation']=='compiled':
        actual=c.artifact(container/'worker/runtime/runtime_identity.json');identity=read(c.resolve(actual))
        require(identity['compilation_fingerprint']==support['compilation_fingerprint'] and
            read(container/'worker/compilation_admission.json')==dict(actual=actual,qualified=support['compiled_identity'],
                compilation_fingerprint=support['compilation_fingerprint'],checked_before_requests=True),'each fresh compiled graph exactly equals qualified fingerprint before requests')


def audit_study(plan,output,*,expected_plan_sha,controller_session,controller_exit_code,controller_chunk,allow_failure=False):
    output=Path(output);require(output.resolve()==c.OUT.resolve() and c.sha(c.PLAN)==expected_plan_sha,'exact frozen plan and owned evidence')
    validate_plan(plan);c.verify(plan['inputs'])
    require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<plan['maximum_output_bytes'],'actual final retained evidence cap')
    claim,execution=read(output/'claim.json'),read(output/'execution.json')
    require(claim['plan_sha256']==execution['plan_sha256']==expected_plan_sha and claim['total_seconds']==plan['total_seconds'] and
            claim['model_seconds']==execution['model_seconds']==plan['model_seconds'] and claim['monotonic_origin']==execution['monotonic_origin'] and
            claim['unix_origin']==execution['unix_origin'] and 0<=execution['monotonic_end']-execution['monotonic_origin']<plan['total_seconds'] and
            0<=execution['unix_end']-execution['unix_origin']<plan['total_seconds'],'original one global dual clock, unextended')
    require(execution['retries']==execution['new_games']==execution['game_routes_started']==execution['training_updates']==0 and
            not execution['cleanup_errors'] and execution['execution_policy_id']==c.EXECUTION_POLICY_ID,'no extra work/retry and exact cleanup')
    absent=old.require_processes_absent(execution['observed_owned_identities']+[dict(pid=claim['pid'],created=claim['created'])])
    require(type(controller_session) is int and controller_session>0 and bool(controller_chunk),'actual retained controller exit attribution')
    base=dict(plan_sha256=expected_plan_sha,auditor=c.artifact(Path(__file__)),source_context=source_context_proof(),
        frozen_inputs_unchanged=len(plan['inputs']),exact_owned_processes_absent=absent,
        retained_controller_exit=dict(session_id=controller_session,exit_code=controller_exit_code,tool_chunk=controller_chunk,
            source='Actual retained execution supplied by root; exact owner absence independently checked'),
        auditor_models_launched=0,auditor_games_launched=0,new_games=0,training_updates=0)
    if execution['status']=='failed':
        require(allow_failure and controller_exit_code!=0 and read(output/'failure.json')==execution and execution['error'],
                'technical failure only in explicit preservation-only scope')
        return dict(**base,status='preserved_failed_attempt_verified',eligible=False,failure=execution['error'],
                    scope='Saved failure preservation, original clock and owner absence; no candidate acceptance')
    require(controller_exit_code==0 and execution['error'] is None and not (output/'failure.json').exists(),'actual successful controller exit')
    support=audit_qualification(plan,output/'qualification')
    require(support==read(output/'qualification/saved_audit.json'),'support evidence independently rederived')
    artifacts=[c.artifact(output/'qualification/saved_audit.json')];admission=read(output/'support_admission.json')
    require(admission==dict(support_admission(support),plan_sha256=expected_plan_sha,source_audits=list(artifacts)),
            'original exact support admission binds first audit')
    stage_names=['qualification'];decision=None;timing_cells=[];rejection='qualification' if not admission['passed'] else None;qualified=False
    if admission['passed']:
        spec=plan['qualification_cell'];container=output/'cells'/spec['cell_id']
        verify_admission(plan,spec,container,support,admission,output/'support_admission.json')
        decision=audit_cell(plan,spec,container/'worker');require(decision==read(container/'saved_audit.json'),'full-decision qualification independently reconstructed')
        stage_names.append(spec['cell_id']);artifacts.append(c.artifact(container/'saved_audit.json'));gate=read(output/'qualification_gate.json')
        require(gate==dict(qualification_gate(support,decision),plan_sha256=expected_plan_sha,source_audits=list(artifacts)),
                'timing admission independently rederived from complete exact qualification')
        qualified=gate['passed']
        if not qualified:rejection=spec['cell_id']
        else:
            for spec in plan['cells']:
                container=output/'cells'/spec['cell_id'];verify_admission(plan,spec,container,support,gate,output/'qualification_gate.json')
                proof=audit_cell(plan,spec,container/'worker');require(proof==read(container/'saved_audit.json'),'exact fresh timing evidence reconstructed')
                timing_cells.append(proof);artifacts.append(c.artifact(container/'saved_audit.json'));stage_names.append(spec['cell_id'])
                if proof['status']!='passed':rejection=spec['cell_id'];break
    else:require(not (output/'qualification_gate.json').exists(),'failed support never creates a timing gate')
    planned=[plan['qualification_cell']]+plan['cells']
    require({p.name for p in (output/'cells').iterdir() if p.is_dir()}==set(stage_names[1:]) if (output/'cells').exists() else len(stage_names)==1,
            'only predetermined stage prefix exists; no hidden retry or later cells')
    stages=[]
    for name in stage_names:
        container=output/'qualification' if name=='qualification' else output/'cells'/name;actual=read(container/'execution.json')
        stages.append(dict(stage_id=name,status=actual['status'],worker_exit_code=actual['worker_exit_code']))
    expected_status='compiled_study_negative_pending_independent_acceptance' if rejection else 'completed_compiled_comparison_pending_independent_acceptance'
    require(execution['status']==expected_status and execution['rejection_stage']==rejection and execution['qualified'] is qualified and
            execution['stages']==stages,'exact full sequence or first refusal prefix without replacement')
    comparison=comparison_summary(timing_cells)
    if rejection is None:require(read(output/'comparison.json')==comparison and comparison['eligible'],'complete comparison rederived from original wall clocks')
    else:require(not (output/'comparison.json').exists(),'no useful-throughput claim after any qualification or timing refusal')
    completion=read(output/'completion.json')
    require(completion['status']==expected_status and completion['execution']==c.artifact(output/'execution.json') and completion['plan']==c.artifact(c.PLAN) and
            completion['audited_stages']==len(artifacts) and completion['stage_audits']==artifacts and completion['qualified'] is qualified and
            completion['rejection_stage']==rejection and completion['new_games']==completion['training_updates']==0 and completion['inputs_unchanged'] is True,
            'completion binds actual exact owner/stages/outcome')
    c.verify(plan['inputs'])
    outcome=('inconclusive_control_deadline' if timing_cells and timing_cells[-1]['implementation']=='eager' and timing_cells[-1]['status']=='deadline_failed'
             else 'candidate_not_admitted' if rejection else 'completed_exact_comparison')
    return dict(**base,status='saved_compiled_study_negative_verified' if rejection else 'saved_compiled_comparison_verified',
        qualification=support,decision_qualification=decision,timing_cells=timing_cells,comparison=comparison,rejection_stage=rejection,
        scientific_outcome=outcome,attempted_panel_decisions=sum(sum(r['attempted'] for r in x['case_rows']) for x in timing_cells+([] if decision is None else [decision])),
        scope='One exact implementation candidate on saved positions; no new games, precision/search relaxation, automatic promotion, strength or collector claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-plan-sha',required=True);parser.add_argument('--controller-session',type=int,required=True)
    parser.add_argument('--controller-exit-code',type=int,required=True);parser.add_argument('--controller-chunk',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--allow-failure',action='store_true');args=parser.parse_args()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources();plan=read(c.PLAN);require(plan['source_guard']==sources,'saved corrected-source startup guard')
    result=audit_study(plan,c.OUT,expected_plan_sha=args.expected_plan_sha,controller_session=args.controller_session,
        controller_exit_code=args.controller_exit_code,controller_chunk=args.controller_chunk,allow_failure=args.allow_failure)
    destination=Path(args.output).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    require(not destination.is_relative_to(c.OUT),'independent final acceptance outside frozen study evidence')
    with destination.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(c.canonical(dict(status=result['status'],output=c.artifact(destination),models=0,games=0)));return 0


if __name__=='__main__':raise SystemExit(main())

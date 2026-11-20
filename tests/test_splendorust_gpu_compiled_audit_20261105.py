"""Model-free saved-evidence falsification for the one compiled GPU candidate."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib
import json

import pytest

from tools.splendorust_gpu_compiled_20261105 import audit as a,common as c,design,qualification as q


def cell_proofs(walls=(100.,94.,94.,100.)):
    return [dict(spec,status='passed',eligible=True,wall_seconds=wall,
            case_rows=[dict(completed=True,baseline_exact=True) for _ in range(32)]) for spec,wall in zip(design.cells(),walls)]


def test_each_pair_complete_throughput_gate_not_pooled_latency_gate():
    result=a.timing_summary(cell_proofs());assert result['useful_improvement'] and result['pooled_compiled_over_eager_throughput']==100/94
    result=a.timing_summary(cell_proofs((100.,80.,99.,100.)))
    assert result['pooled_compiled_over_eager_throughput']>1.05 and not result['useful_improvement']
    values=cell_proofs();values[-1].update(status='deadline_failed',eligible=False)
    assert not a.timing_summary(values)['eligible'] and a.timing_summary(values)['pooled_compiled_over_eager_throughput'] is None


def test_no_prefix_replacement_or_unqualified_gate():
    values=cell_proofs();assert not a.timing_summary(values[:2])['eligible']
    with pytest.raises(ValueError):a.timing_summary(list(reversed(values)))
    support=dict(passed=True,status='support_passed');decision=dict(values[0],status='passed')
    assert a.support_admission(support)['passed'] and a.qualification_gate(support,decision)['passed']
    decision['case_rows'][0]['baseline_exact']=False
    assert not a.qualification_gate(support,decision)['passed']
    assert not a.qualification_gate(dict(passed=False,status='compiled_unsupported'),None)['passed']


def test_exact_float32_signed_zero_and_value_mismatch_are_not_tolerated():
    rows=[[0.]*83 for _ in range(8)];actual=deepcopy(rows);actual[3][4]=-0.
    assert actual==rows
    difference=a.first_difference(actual,rows)
    assert difference['row']==3 and difference['column']==4 and difference['actual_float32_hex']=='00000080'
    assert a.first_difference(actual,rows,[0,1]) is None
    actual[0][0]=1e-20
    assert a.first_difference(actual,rows)['row']==0
    actual[0][0]=float('nan')
    with pytest.raises(ValueError):a.first_difference(actual,rows)


def test_private_checker_does_not_change_public_frozen_context():
    public=importlib.import_module('tools.splendorust_gpu_batchfill_20261105.audit');before=public.c
    private=a.load_reused()
    assert private is not public and public.c is before and private.c is c
    assert c.sha(a.LIBRARY)==a.LIBRARY_SHA and a.source_context_proof()['models']==0


def clock(start=1.,status='ok',error=None):
    result=dict(status=status,completed=status=='ok',started_perf_counter=start,ended_perf_counter=start+.1,elapsed_seconds=(start+.1)-start,
        started_unix=start,ended_unix=start+.1,internal_seconds=60.,outer_seconds=75.,execution_policy_id=c.EXECUTION_POLICY_ID,no_retry=True)
    if error is not None:result['error']=error
    return result


def test_native_fifo_explicitly_accounts77_physical_prefix_without_mutating_journal():
    clocks=[clock(float(i+1)) for i in range(77)]
    item=dict(request_id='native0:1',client_id='native0',slot=0,enqueue_sequence=78,submitted_perf=80.,deadline_perf=140.)
    request={k:v for k,v in item.items() if k!='enqueue_sequence'};request.update(dispatch_perf=82.)
    events=[dict(batch_id=1,requests=[request])]
    scheduling=[dict(scheduler='row_fifo32',scheduler_dispatch_index=1,observed_perf=81.,ready=[item],
                     selected_request_ids=['native0:1'],deferred_request_ids=[])]
    original=deepcopy(scheduling)
    result=a.audit_scheduler_after_physical(events,{'native0:1':request},scheduling,clocks)
    assert result['raw_ticket_offset']==77 and scheduling==original and result['completed']==1
    with pytest.raises(ValueError):a.audit_scheduler_after_physical(events,{'native0:1':request},scheduling,clocks[:-1])
    scheduling[0]['ready'][0]['enqueue_sequence']=79
    with pytest.raises(ValueError):a.audit_scheduler_after_physical(events,{'native0:1':request},scheduling,clocks)


def refusal(message='PrivateUse1 operation is not supported',typ='NotImplementedError'):
    chain=[dict(type='RuntimeError',module='builtins',message='Physical qualification inference failed'),dict(type=typ,module='builtins',message=message)]
    return dict(schema='compiled_physical_support_refusal_v1',implementation='compiled',reason_kind='runtime_unsupported',
                no_retry=True,traceback='original exception trace',exception_chain=chain)


def test_runtime_refusal_requires_original_nonresource_error_and_unexpired_failed_clock():
    value=refusal();saved=clock(status='error',error="RuntimeError('Physical qualification inference failed')")
    assert a.classify_physical_refusal(value,saved)['reason_kind']=='runtime_unsupported'
    for message,typ in [('out of memory: operation not supported','RuntimeError'),('disk full','OSError'),('not supported','TimeoutError'),('checker mismatch','ValueError')]:
        with pytest.raises(ValueError):a.classify_physical_refusal(refusal(message,typ),saved)
    changed=deepcopy(saved);changed['error']="RuntimeError('different error')"
    with pytest.raises(ValueError):a.classify_physical_refusal(value,changed)
    changed=deepcopy(saved);changed.update(status='timeout',elapsed_seconds=75.);changed['ended_perf_counter']=76.
    with pytest.raises(ValueError):a.classify_physical_refusal(value,changed)


def fixtures():
    return [dict(index=i,fixture_id=f'b{i:03d}',physical_batch=8,kind='full_saved_roots' if i<4 else 'single_occupied_native_row',
        case_indexes=list(range(i*8,i*8+8)) if i<4 else [i-4],occupied_slots=list(range(8)) if i<4 else [(i-4)%8],
        input_tokens=[float(i+1)]*11904,input_sha256=c.digest([float(i+1)]*11904),expected_occupied_outputs=[[float(i+1)]*83 for _ in range(8)],
        expected_sources=[]) for i in range(36)]


def fake_runtime(implementation,mode=None):
    count=[0]
    def forward(tokens,*,deadline):
        count[0]+=1
        if mode=='unsupported' and count[0]==1:raise NotImplementedError('PrivateUse1 operation is not supported')
        result=[[tokens[0]]*83 for _ in range(8)]
        if mode=='mismatch' and count[0]==1:result[0][0]=1.
        return result
    return SimpleNamespace(identity=dict(implementation=implementation,scheduler='row_fifo32'),route='gpu',adapters={},
        service=SimpleNamespace(clients={},pending={},physical_forward=forward),operation=lambda *args,**kwargs:nullcontext(),abort=lambda:None)


def test_actual_support_writer_to_independent_auditor_exact_and_both_negative_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);plan=dict(support=design.support());fixed=fixtures()
    eager=tmp_path/'eager';q.run_implementation(plan,fake_runtime('eager'),eager,fixed)
    assert a.audit_physical_support(eager,'eager',fixed)['complete_calls']==77
    compiled=tmp_path/'compiled';q.run_implementation(plan,fake_runtime('compiled'),compiled,fixed,eager_folder=eager)
    assert a.audit_physical_support(compiled,'compiled',fixed,eager_folder=eager)['complete']
    mismatch=tmp_path/'mismatch'
    with pytest.raises(q.SupportMismatch):q.run_implementation(plan,fake_runtime('compiled','mismatch'),mismatch,fixed,eager_folder=eager)
    value=a.audit_physical_support(mismatch,'compiled',fixed,eager_folder=eager,allow_mismatch=True)
    assert value['status']=='exact_output_mismatch' and value['complete_calls']==1 and value['first_mismatch']
    unsupported=tmp_path/'unsupported'
    with pytest.raises(q.SupportUnsupported):q.run_implementation(plan,fake_runtime('compiled','unsupported'),unsupported,fixed,eager_folder=eager)
    value=a.audit_physical_support(unsupported,'compiled',fixed,eager_folder=eager,allow_runtime_refusal=True)
    assert value['status']=='runtime_unsupported' and value['complete_calls']==0 and value['runtime_refusal']
    (mismatch/'hidden_retry.json').write_text('{}')
    with pytest.raises(ValueError,match='entire support inventory'):
        a.audit_physical_support(mismatch,'compiled',fixed,eager_folder=eager,allow_mismatch=True)


def test_physical_fixture_layout_reconstructed_from_accepted_source_rows():
    plan=c.read(c.ROOT/'outputs/splendorust_gpu_costs_20261105_01/plan.json');plan['support']=design.support()
    independent=a.expected_physical_fixtures(plan);actual=q.build_fixtures(plan)
    assert actual==independent and len(independent)==36
    for row in independent[4:]:
        index=row['index']-4;slot=index%8
        assert row['occupied_slots']==[slot]
        for other in range(8):
            if other!=slot:assert row['input_tokens'][other*1488:(other+1)*1488]==[0.]*1488


def test_valid_scientific_different_saved_cpu_result_proves_negative_without_weakening_checks(tmp_path,monkeypatch):
    plan=c.read(c.ROOT/'outputs/splendorust_gpu_batchfill_20261105_01/plan.json');case=plan['cases'][0]
    source=c.ROOT/'outputs/splendorust_gpu_batchfill_20261105_01/cells/b1_cpu8/worker/cell/p000'
    monkeypatch.setattr(c,'OUT',tmp_path);folder=tmp_path/'p000';folder.mkdir()
    for name in ('case.json','request.json','attempt.json','clock.json','result.json','root.json','raw_result.json','after.json','service_calls.json'):
        c.save(folder/name,c.read(source/name))
    expected=c.read(c.resolve(case['expected_results']['gpu']))
    assert c.read(folder/'result.json')!=expected['result']
    witness=dict(kind='compiled_full_decision_exact_mismatch',actual_result=c.artifact(folder/'result.json'),expected_result=case['expected_results']['gpu'],
        actual_after=c.artifact(folder/'after.json'),actual_root=c.artifact(folder/'root.json'),completed_original_clock=c.artifact(folder/'clock.json'),no_retry=True)
    c.save(folder/'exact_mismatch.json',witness)
    requests={row['request_id']:{k:v for k,v in row.items() if k!='client_received_perf'} for row in c.read(folder/'service_calls.json')}
    proof=a.audit_mismatched_case(plan,case,design.qualification_cell(),folder,requests)
    assert proof['completed'] and not proof['baseline_exact'] and proof['mismatch']
    witness['expected_result']=case['expected_results']['cpu'];(folder/'exact_mismatch.json').write_text(c.canonical(witness))
    with pytest.raises(ValueError,match='exact saved rejection'):
        a.audit_mismatched_case(plan,case,design.qualification_cell(),folder,requests)

"""Actual capture collector and actual data-loader qualification, no fitting."""
from copy import deepcopy
from pathlib import Path
import struct
from . import common as c, collector, data
from tools import future_training_rules_20261021 as guard
from tools.splendorust_gpu_throughput_20261105 import qualification as q
from tools.splendorust_selfplay_pilot_resume2_20261103 import data as accepted


def build_plan():
    return dict(schema='gpu_collector_qualification_v1',entrypoint=q.build_plan(),loader=accepted.build_qualification_cases(),
        actual_entrypoint_decisions=67,accepted_saved_decisions=32,actual_loader_decisions=16,
        official_fixture_continuation_decisions=1,decisions_per_route=116,
        ordinary_games=0,training_updates=0,full_loader_fixture='noble_final_round plus one official final-seat response',
        exact_capture_scientific_projection=True,original_clocks=True)


def fixture_spec(case,index):
    spec=dict(game_id='loader_'+case.get('id','private_hidden_repeat'),game_index=index,
              seed_decimal='0',start_player=0,purpose='engineering_qualification_only')
    if case.get('id')=='noble_final_round':spec['fixture_start']=case['snapshot']
    return spec


def _expect_rejection(operation,name):
    try:operation()
    except (ValueError,KeyError,TypeError,AssertionError):return dict(case=name,rejected=True)
    raise ValueError('Actual collector loader accepted '+name)


def full_loader_checks(document,spec,folder,boundary):
    """The real artifact/manifest/trajectory/target loader receives a real terminal fixture."""
    document_path=folder/'fixture_target_game.json';c.save(document_path,document)
    manifest=data.build_manifest([c.artifact(document_path)],[spec],route=document['route'],cell_id='qualification_fixture')
    c.save(folder/'fixture_corpus_manifest.json',manifest)
    captured=[]
    loaded=data.load_corpus(manifest,expected_specs=[spec],boundary=boundary,
                            sink=lambda spec,examples:captured.extend(examples))
    c.save(folder/'fixture_loaded_targets.json',dict(examples=captured,summary=loaded))
    outcome=data.signed_outcomes(document['final'],'completed',0)
    opposite=data.signed_outcomes(document['final'],'completed',1)
    c.require(opposite==[outcome[1],outcome[0]],'official outcome both canonical actor perspectives')
    negatives=data.qualify_manifest_rejections(manifest,[spec])
    for reason in ('cutoff','deadlock','timeout'):
        bad=deepcopy(document);bad['reason']=reason
        negatives.append(_expect_rejection(lambda doc=bad:data.load_game(doc,spec,boundary=boundary),reason+'_cannot_label'))
    bad=deepcopy(document);bad['final']=document['initial']
    negatives.append(_expect_rejection(lambda:data.load_game(bad,spec,boundary=boundary),'nonterminal_cannot_label'))
    for key in data.IDENTITY:
        bad=deepcopy(document);bad.pop(key)
        negatives.append(_expect_rejection(lambda doc=bad:data.load_game(doc,spec,boundary=boundary),'missing_game_'+key))
    result=dict(status='actual_complete_fixture_loader_passed',corpus=c.artifact(folder/'fixture_corpus_manifest.json'),
        loaded=c.artifact(folder/'fixture_loaded_targets.json'),eligible_roots=loaded['eligible_roots'],
        official_winner=document['winner'],actor0_value=outcome,actor1_value=opposite,negative_checks=negatives,
        ordinary_games=0,engineering_fixture_only=True,optimizer_updates=0)
    c.save(folder/'full_loader_checks.json',result)
    return result


def run_qualification(plan,route,runtime,folder,boundary=lambda:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    selected=plan['qualification'];c.require(selected==build_plan() and runtime.route==route,'fixed collector qualification')
    structural=q.run_qualification(plan['runtime'],runtime,folder/'entrypoint',boundary)
    parity=[]
    # Reuse only eight adapters and fixed physical slots, preserving CPU8 registration.
    adapters={}
    try:
        for slot in range(8):adapters[slot]=runtime.make_adapter('parity_'+str(slot),slot,512)
        for case in plan['cases']:
            boundary();adapter=adapters[case['slot']]
            spec=dict(game_id=case['case_id'],game_index=case['index'],purpose='engineering_qualification_only')
            row=collector.decide_snapshot(runtime,adapter,case['snapshot'],spec,0,folder/'parity'/case['case_id'],kind='capture_parity')
            c.verify([case['expected_results'][route]])
            expected=c.read(c.resolve(case['expected_results'][route]))
            root=q.validate_fixed_result(dict(case,observer_sha256=c.digest(case['observer'])),row['result'])
            expected_output=expected['result']['diagnostics']['phases'][0]['inference'][0]['output83']
            exact=row['result']==expected['result'] and row['after']==expected['after'] and (
                struct.pack('<83f',*root['output83'])==struct.pack('<83f',*expected_output))
            proof=dict(case_id=case['case_id'],actual=c.artifact(folder/'parity'/case['case_id']/'moves'/'000000.json'),
                       expected=case['expected_results'][route],exact_full_result_and_after=exact)
            c.save(folder/'parity'/case['case_id']/'comparison.json',proof)
            c.require(exact,'capture addition must preserve exact accepted scientific response')
            parity.append(proof)
    finally:
        for adapter in adapters.values():adapter.close()
    loader_plan=selected['loader'];cases=loader_plan['cases']+[loader_plan['private_variant']]
    rows=[];adapter=runtime.make_adapter('loader',0,512)
    try:
        for index,case in enumerate(cases):
            boundary();spec=fixture_spec(case,index)
            row=collector.decide_snapshot(runtime,adapter,case['snapshot'],spec,0,folder/'loader'/f'{index:02d}',kind='loader_counterexample')
            rows.append(row)
        rowproof=data.qualify_captured_rows(loader_plan,rows,boundary=boundary)
        c.save(folder/'loader_row_checks.json',rowproof)
        index=next(i for i,x in enumerate(cases) if x.get('id')=='noble_final_round')
        first=rows[index];spec=first['spec']
        c.require(not guard.decode_training_snapshot(first['after']).terminal,'one last official seat remains')
        boundary();last=collector.decide_snapshot(runtime,adapter,first['after'],spec,1,
            folder/'loader_terminal',kind='official_fixture_continuation')
        winner=guard.require_official_terminal_target(last['after'],'completed')
        document=dict(schema=data.GAME_SCHEMA,**data.IDENTITY,route=route,spec=spec,reason='completed',
            initial=first['before'],final=last['after'],winner=winner,
            rows=[c.artifact(folder/'loader'/f'{index:02d}'/'moves'/'000000.json'),
                  c.artifact(folder/'loader_terminal'/'moves'/'000001.json')])
        fullproof=full_loader_checks(document,spec,folder,boundary)
    finally:adapter.close()
    summary=dict(status='qualified',route=route,actual_entrypoint_decisions=67,accepted_saved_decisions=len(parity),
        actual_loader_decisions=len(rows),official_fixture_continuation_decisions=1,complete_decisions=116,
        entrypoint=c.artifact(folder/'entrypoint'/'qualification.json'),parity=parity,
        loader_rows=c.artifact(folder/'loader_row_checks.json'),full_loader=c.artifact(folder/'full_loader_checks.json'),
        actual_collector_and_loader_qualified=True,ordinary_games=0,training_updates=0,no_retry=True)
    c.save(folder/'qualification.json',summary)
    return summary

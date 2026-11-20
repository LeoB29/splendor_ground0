"""Model-free preservation/freeze contracts; no worker or experimental run."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest
from tools.splendorust_search128_resume_20261029 import preparation as p


def pin(path, sha='a'*64, size=1):
    return dict(path=Path(path).relative_to(p.c.ROOT).as_posix(), sha256=sha, bytes=size)


def manifest():
    bindings = dict(old_freeze=pin(p.c.OLD_OUT/'freeze.json', p.c.OLD_FREEZE_SHA),
        recovery1_freeze=pin(p.c.FIRST_RECOVERY_OUT/'freeze.json', p.c.FIRST_RECOVERY_SHA),
        original_claim=pin(p.c.OLD_OUT/'guardian_claim.json'),
        bridge=pin(p.c.FIRST_RECOVERY_OUT/'recovered_bridge.json'),
        closure_check=pin(p.c.CLOSURE_OUT/'closure_check.json', p.c.CLOSURE_SHA),
        checker_os_exit=pin(p.c.CLOSURE_OUT/'checker_os_exit.json'))
    trees = dict(original=[bindings['old_freeze'], bindings['original_claim']],
        recovery1=[bindings['recovery1_freeze'], bindings['bridge']],
        closure=[bindings['closure_check'], bindings['checker_os_exit']])
    return p.make_manifest(trees, bindings, {},
        dict(status='passed_saved_receipt_binding', historical_attempts_accepted=False))


def actual_receipt_data():
    return [p.c.read(p.c.OLD_OUT/'completion.json'),
        p.c.read(p.c.FIRST_RECOVERY_OUT/'completion.json'),
        p.c.read(p.c.CLOSURE_OUT/'closure_check.json'),
        p.c.read(p.c.CLOSURE_OUT/'checker_os_exit.json'),
        p.c.read(p.c.CLOSURE_OUT/'checker_launch.json')]


def test_real_saved_disposition_and_owned_exit_without_science_rerun():
    data = actual_receipt_data()
    result = p.validate_closure_disposition(*data)
    assert result['continuation_game_index'] == result['completed_main_games'] == 39
    assert result['original_completed_main_games'] == 38
    assert result['historical_attempts_accepted'] is False
    assert data[0]['status'] == data[1]['status'] == 'incomplete'


@pytest.mark.parametrize('mutation', [
    'old_accepted','recovery_accepted','wrong_freeze','accepted_benchmark','failed_exit','wrong_pid',
    'wrong_created','wrong_observer','long_checker','replayed_game','bridge_in_timing','new_model',
    'bad_graph_count','bad_startup_count','missing_worker','ordinary_count','changed_clock','bad_suite'
])
def test_saved_disposition_corruption_rejected(mutation):
    old, recovery, closure, exit_record, launch = actual_receipt_data()
    if mutation == 'old_accepted': old['status'] = 'completed'
    elif mutation == 'recovery_accepted': recovery['status'] = 'completed'
    elif mutation == 'wrong_freeze': closure['freeze_sha256'] = '0'*64
    elif mutation == 'accepted_benchmark': closure['benchmark_accepted'] = True
    elif mutation == 'failed_exit': exit_record['exit_code'] = 1
    elif mutation == 'wrong_pid': exit_record['checker_pid'] += 1
    elif mutation == 'wrong_created': exit_record['checker_created'] += 1
    elif mutation == 'wrong_observer': exit_record['observed_by'] = 'inferred from absent PID'
    elif mutation == 'long_checker': exit_record['elapsed_seconds'] = 901.
    elif mutation == 'replayed_game': closure['segmented_game']['game_rerun'] = True
    elif mutation == 'bridge_in_timing': closure['segmented_game']['proof']['recovered_move_in_ordinary_timing'] = True
    elif mutation == 'new_model': closure['new_model_calls'] = 1
    elif mutation == 'bad_graph_count': closure['graph_cases'] = 11
    elif mutation == 'bad_startup_count': closure['actual_entrypoint_decisions'] = 104
    elif mutation == 'missing_worker': closure['closed_workers'].pop()
    elif mutation == 'ordinary_count': closure['closed_workers'][0]['raw']['requests'] = 8
    elif mutation == 'changed_clock': closure['original_clocks_rewritten'] = True
    elif mutation == 'bad_suite': closure['actual_entrypoint_verification'][0]['complete_decisions'] = 52
    with pytest.raises(ValueError):p.validate_closure_disposition(old,recovery,closure,exit_record,launch)


def test_three_trees_counts_and_exact_prospective_lifecycles():
    value = manifest()
    assert p.validate_manifest(value) is True
    assert [value[x+'_tree_files'] for x in ('original','recovery1','closure')] == [2,2,2]
    assert value['combined_qualification_decisions'] == 441
    assert value['new_entrypoint_decisions'] == 119 and value['qualification_ordinary_requests'] == 84
    assert value['prior_final_seconds'] == 190 and value['prior_ram_wait_seconds'] == 200
    assert value['declared_worker_lifecycles'] == {
        'qualification_resume_research':dict(route='research',game_index=39,graph_request_id='request-00000001',
            entrypoint_request_id='request-00000002',ordinary_first_request_id='request-00000003'),
        'qualification_resume_frozen':dict(route='frozen',game_index=39,graph_request_id='request-00000001',
            entrypoint_request_id='request-00000002',ordinary_first_request_id='request-00000003')}


@pytest.mark.parametrize('mutation', [
    'missing_tree','duplicate_file','outside_root','file_count','byte_count','wrong_freeze',
    'reset_clock','reset_final','reset_ram','game_retry','wrong_next_game','extra_worker',
    'suite_after_play','wrong_combined_count','relabel_status','new_failed_move_call'
])
def test_manifest_corruption_rejected(mutation):
    value = manifest()
    if mutation == 'missing_tree':value['closure_tree'] = []
    elif mutation == 'duplicate_file':value['recovery1_tree'].append(deepcopy(value['recovery1_tree'][0]))
    elif mutation == 'outside_root':value['closure_tree'][0]['path'] = 'outputs/other/closure_check.json'
    elif mutation == 'file_count':value['original_tree_files'] += 1
    elif mutation == 'byte_count':value['recovery1_tree_bytes'] += 1
    elif mutation == 'wrong_freeze':value['old_freeze']['sha256'] = 'f'*64
    elif mutation == 'reset_clock':value['original_monotonic_origin'] += 10
    elif mutation == 'reset_final':value['prior_final_seconds'] = 0
    elif mutation == 'reset_ram':value['prior_ram_wait_seconds'] = 0
    elif mutation == 'game_retry':value['completed_game_retries'] = 1
    elif mutation == 'wrong_next_game':value['continuation_game_index'] = 38
    elif mutation == 'extra_worker':value['declared_worker_lifecycles']['main_g39_challenger'] = {}
    elif mutation == 'suite_after_play':value['declared_worker_lifecycles']['qualification_resume_research']['entrypoint_request_id'] = 'request-00000003'
    elif mutation == 'wrong_combined_count':value['combined_qualification_decisions'] = 322
    elif mutation == 'relabel_status':value['first_recovery_status'] = 'accepted'
    elif mutation == 'new_failed_move_call':value['new_call_for_failed_move'] = True
    with pytest.raises(ValueError):p.validate_manifest(value)


def test_freeze_preserves_every_scientific_field_and_does_not_mutate_old():
    value = manifest(); old = dict(schema='old',main_games=400,operational_games=6,seconds_per_move=10.,
        internal_seconds=9.,seed_domain='immutable',profiles={'64':{'quota':64},'128':{'quota':128}},
        quota64=64,quota128=128,world_pool=3,weights_sha256='b'*64,precision='float32',
        job_seconds=86400,collection_cutoff_seconds=72000,max_evidence_bytes=16*p.c.GiB,
        final_reserve_bytes=4*p.c.GiB,prefinal_evidence_bytes=12*p.c.GiB,
        automatic_retries=0,training=False,inputs=[])
    prior = deepcopy(old)
    frozen = p.build_freeze(old,pin(p.c.OUT/'recovery_manifest.json'),value,
        [pin(p.c.OUT/'source.py')],pin(p.c.PROTOCOL))
    assert old == prior
    for key in old.keys()-{'schema','inputs'}:assert frozen[key] == old[key]
    assert frozen['schema'] == p.FREEZE_SCHEMA
    assert frozen['max_workers'] == frozen['max_graph_builds'] == 864
    assert frozen['source_attempts_remain_incomplete'] is True
    assert frozen['new_entrypoint_decisions'] == 119 and frozen['combined_qualification_decisions'] == 441
    assert 'resume_manifest' in frozen and 'recovery_manifest' not in frozen


def test_input_union_retains_all_and_rejects_same_path_changes():
    a=pin(p.c.OLD_OUT/'a.json');b=pin(p.c.FIRST_RECOVERY_OUT/'b.json')
    result=p.merge_inputs([a],[a,b])
    assert len(result)==2 and {x['path'] for x in result}=={a['path'],b['path']}
    changed={**a,'bytes':2}
    with pytest.raises(ValueError,match='collision'):p.merge_inputs([a],[changed])


def test_inventory_preserves_all_files_including_cache_named_files(tmp_path):
    root=tmp_path/'root';tree=root/'tree';(tree/'__pycache__').mkdir(parents=True)
    (tree/'value.json').write_text('{}');(tree/'__pycache__/evidence.bin').write_bytes(b'unchanged')
    reader=p.ReviewRuntime(root,root/'out',default_seconds=30.)
    items=p._inventory(tree,reader)
    assert {x['path'] for x in items} == {'tree/value.json','tree/__pycache__/evidence.bin'}
    (tree/'extra').write_bytes(b'new')
    with pytest.raises(ValueError,match='Exact preserved'):reader.exact_tree(tree,{x['path'] for x in items})


def test_inventory_cap_is_finite(tmp_path,monkeypatch):
    tree=tmp_path/'tree';tree.mkdir();(tree/'a').write_bytes(b'a');(tree/'b').write_bytes(b'b')
    monkeypatch.setattr(p,'MAX_TREE_FILES',1)
    with pytest.raises(ValueError,match='Bounded preserved'):p._inventory(tree,p.ReviewRuntime(tmp_path,tmp_path/'out',default_seconds=30.))


def test_preparation_has_no_model_or_experiment_execution_call():
    tree=ast.parse(Path(p.__file__).read_text())
    names={node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id
           for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,(ast.Name,ast.Attribute))}
    assert not names & {'Popen','make_worker','play_game','predict','forward','run_entrypoint_checks','actual_entrypoint'}
    assert 'verify_runtime_sources' in names

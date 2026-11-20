"""Model-free regression tests for selection, schedules and strict entry points."""
from copy import deepcopy
import random
import sys
import pytest
from . import selection as s


def fixture_selection():
    refs=[[g,p] for g in range(1024) for p in range(1+(g%7))]
    _,subsets=s.subset_definitions(refs,list(range(len(refs))))
    global_orders={str(rep):sorted(range(len(refs)),key=lambda i:s.digest(['accepted',rep,refs[i]])) for rep in (0,1)}
    for subset in subsets.values():
        lookup={tuple(r):i for i,r in enumerate(subset['root_ids'])}
        subset['first_pass_orders']={rep:[lookup[tuple(refs[i])] for i in order if tuple(refs[i]) in lookup]
            for rep,order in global_orders.items()}
    return dict(subsets=subsets),refs,global_orders


def test_game_selection_is_exact_nested_and_independent_of_input_order():
    forward=s.game_order(list(range(1024)));reverse=s.game_order(list(reversed(range(1024))))
    assert forward==reverse
    assert len(set(forward[:256]))==256 and set(forward[:256])<set(forward[:512])<set(forward)


@pytest.mark.parametrize('bad',[list(range(1023)),list(range(1,1025)),[False,*range(1,1024)]])
def test_no_heldout_missing_or_boolean_game_identity(bad):
    with pytest.raises(ValueError):s.game_order(bad)


def test_every_selected_game_keeps_all_its_roots_and_exact_cache_offsets():
    selection,refs,_=fixture_selection()
    for size in s.SIZES:
        subset=selection['subsets'][str(size)];games=set(subset['game_ids'])
        expected=[i for i,r in enumerate(refs) if r[0] in games]
        assert subset['cache_indices']==expected
        assert subset['root_ids']==[refs[i] for i in expected]
        assert len(games)==size


def test_filtering_preserves_each_original_reference_order():
    selection,refs,global_orders=fixture_selection()
    for rep in (0,1):
        for size in s.SIZES:
            subset=selection['subsets'][str(size)];schedule=s.build_schedules(selection,rep,size)
            observed=[subset['root_ids'][i] for b in schedule['batches'][:schedule['one_pass_step']] for i in b]
            expected=[refs[i] for i in global_orders[str(rep)] if refs[i][0] in subset['game_ids']]
            assert observed==expected


def test_every_full_pass_is_without_replacement_and_partial_end_is_genuine():
    selection,_,_=fixture_selection();schedule=s.build_schedules(selection,0,256)
    n=selection['subsets']['256']['roots']
    assert len(schedule['batches'])==1782
    for item in schedule['pass_boundaries']:
        batches=schedule['batches'][item['start_step']-1:item['stop_step']]
        flat=[i for batch in batches for i in batch]
        assert len(flat)==len(set(flat))
        if item['complete']:
            assert sorted(flat)==list(range(n))
            assert all(len(b)==32 for b in batches[:-1])
            assert len(batches[-1])==(n%32 or 32)
        else:
            assert item is schedule['pass_boundaries'][-1]
            assert all(len(b)==32 for b in batches)
    assert schedule['one_pass_presentations']==n
    assert schedule['presentations']==sum(map(len,schedule['batches']))==sum(schedule['presentation_count_by_game'].values())
    assert schedule['presentation_difference_from_reference']==schedule['presentations']-57022


def test_schedule_does_not_consume_model_rng_and_seeds_change_order_only():
    selection,_,_=fixture_selection();before=random.getstate()
    a=s.build_schedules(selection,0,512);b=s.build_schedules(selection,1,512)
    assert before==random.getstate()
    assert a['batches']!=b['batches']
    assert a['presentations']==b['presentations'] and a['one_pass_step']==b['one_pass_step']
    assert a['checkpoint_steps']==sorted({a['one_pass_step']-1,a['one_pass_step'],1781,1782})
    assert len(a['batches'])==len(b['batches'])==1782


def test_mutated_order_or_claimed_count_cannot_validate():
    selection,_,_=fixture_selection();schedule=s.build_schedules(selection,0,256)
    for key in ('batches','presentations','checkpoint_steps','game_count'):
        changed=deepcopy(schedule)
        if key=='batches':changed[key][0][0],changed[key][0][1]=changed[key][0][1],changed[key][0][0]
        elif key=='checkpoint_steps':changed[key].append(100)
        else:changed[key]+=1
        with pytest.raises(ValueError):s.validate_schedules(changed,selection,0,256)


def test_invalid_repetition_or_population_cannot_enter_schedule():
    selection,_,_=fixture_selection()
    for rep,size in ((True,256),(2,256),(0,1024),(0,True)):
        with pytest.raises(ValueError):s.build_schedules(selection,rep,size)


def test_changed_first_pass_order_or_heldout_ref_is_rejected():
    selection,refs,_=fixture_selection();changed=deepcopy(selection)
    changed['subsets']['256']['first_pass_orders']['0'][0]=-1
    with pytest.raises(ValueError):s.build_schedules(changed,0,256)
    changed_refs=deepcopy(refs);changed_refs[-1][0]=1024
    with pytest.raises(ValueError):s.subset_definitions(changed_refs,list(range(len(changed_refs))))


def test_source_pins_and_selection_code_do_not_import_tensor_runtime():
    assert s.SOURCE_SELECTION['sha256']=='07f2b18cc99c0e51393d9e1cc5f670c2f0eae9e4389f51ab133bc40939593527'
    assert s.SOURCE_ACCEPTANCE['sha256']=='447163bcb1ff900bf4a77d53a433faf9e2ecdd7c9105123858f7371818142e03'
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('mutation',['legacy_identity','heldout_cache_root','source_hash'])
def test_actual_subset_loader_rejects_changed_identity_before_cache_release(monkeypatch,mutation):
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    from tools.splendorust_full_corpus_20261110 import corpus
    source=s.build_selection(runtime_sources=SOURCE_HASHES)
    changed=deepcopy(source)
    if mutation=='legacy_identity':changed.pop('ruleset_id')
    elif mutation=='heldout_cache_root':changed['subsets']['256']['root_ids'][0]=[1024,0]
    else:changed['source_cache_manifest']['sha256']='0'*64
    def forbidden(*args,**kwargs):raise AssertionError('Invalid metadata reached the cache release')
    monkeypatch.setattr(corpus,'load_cache',forbidden)
    with pytest.raises(ValueError):s.load_selected(changed,0,256)

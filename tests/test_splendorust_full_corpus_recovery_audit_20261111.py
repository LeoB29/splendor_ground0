"""Model-free counterexamples for merged histories, disclosure and unchanged statistics."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION','1')
os.environ.setdefault('FULL_CORPUS_MATCHUP','terminal_parent')
from tools.splendorust_full_corpus_recovery_20261111 import common as c, family, fit, audit
from tools.splendorust_full_corpus_recovery_20261111.strength import recovered_audit as saved, profile_parity as parity, report

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_full_corpus_family_fixture',ROOT/'tests/test_full_corpus_family_20261110.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def disclosed_record(group,**kwargs):
    row=fixtures.record(group,**kwargs)
    row['acceptance'].update(amended_recovered_study=True,original_attempt_status='incomplete',
        authorized_potentially_interrupted_retry_count=1 if group=='terminal_parent' else 0)
    return row


def test_amendment_retains_three_distinct_effects_and_original_thresholds():
    rows={g:disclosed_record(g) for g in family.GROUPS}
    result=family.synthesize(rows)
    assert result['main_games']==1200 and result['operational_games']==36
    assert result['combined_win_rate_across_opponents'] is None
    assert [r['interval_confidence'] for r in result['comparisons']]==[.95,.975,.975]
    assert result['authorized_potentially_interrupted_retry_count']==1
    assert result['original_attempt_status']=='incomplete'
    assert result['no_automatic_promotion'] is True


@pytest.mark.parametrize('key,value', [('amended_recovered_study',False),('original_attempt_status','complete'),
                                      ('authorized_potentially_interrupted_retry_count',0),
                                      ('authorized_potentially_interrupted_retry_count',2)])
def test_hidden_or_extra_retry_cannot_receive_family_acceptance(key,value):
    row=disclosed_record('terminal_parent');row['acceptance'][key]=value
    with pytest.raises(ValueError):family.validate_group('terminal_parent',**row)


def test_parent_claim_still_requires_adjusted_interval():
    row=disclosed_record('mixed_parent',official=fixtures.score(ci975=(.49,.63)))
    assert row['analysis']['unadjusted_95_criteria_met'] is True
    assert family.validate_group('mixed_parent',**row)['useful_gain_criteria_met'] is False


def test_primary_retains_single_95_interval():
    row=disclosed_record('mixed_terminal',official=fixtures.score(ci975=(.49,.63)))
    assert family.validate_group('mixed_terminal',**row)['useful_gain_criteria_met'] is True


def test_original_55_percent_threshold_is_not_lowered():
    row=disclosed_record('mixed_terminal',official=fixtures.score(wins=218,ci95=(.51,.58),ci975=(.505,.59)))
    assert family.validate_group('mixed_terminal',**row)['useful_gain_criteria_met'] is False


def failed_execution():
    return dict(status='failed',worker_exit_code=0,cleanup_errors=["ValueError('Exact previous process still alive')"],
                error='ValueError: FULL_CORPUS: Actual free disk floor')


def test_saved_failed_exit_zero_is_not_treated_as_completed():
    assert saved.validate_interrupted_execution(failed_execution(),False,False)


@pytest.mark.parametrize('key,value',[('status','passed'),('worker_exit_code',1),('cleanup_errors',[]),
                                     ('error','wrong model action')])
def test_only_exact_original_failure_qualifies_for_disclosed_missing_closure(key,value):
    value0=failed_execution();value0[key]=value
    with pytest.raises(ValueError):saved.validate_interrupted_execution(value0,False,False)


@pytest.mark.parametrize('result,complete',[(True,False),(False,True),(True,True)])
def test_original_interruption_must_never_be_relabelled_success(result,complete):
    with pytest.raises(ValueError):saved.validate_interrupted_execution(failed_execution(),result,complete)


def test_partial_game_uses_old_pair_then_new_pair_for_following_games(monkeypatch):
    monkeypatch.setattr(saved.lineage,'partial',lambda s:s['game_index']==32)
    first=saved.expected_pair({'stage':'main','game_index':32},{'reason':'natural'},30)
    assert first==(32,{'control':'main_g032_control','candidate':'main_g032_candidate'},
                     {'control':'main_g030_control','candidate':'main_g030_candidate'})
    following=saved.expected_pair({'stage':'main','game_index':33},{'reason':'natural'},32)
    assert following[0]==32 and following[1]['control']=='main_g032_control'
    forfeited=saved.expected_pair({'stage':'main','game_index':33},{'reason':'forfeit'},32)
    assert forfeited[0]==33


def test_partial_and_post_forfeit_workers_bind_resume_segment(monkeypatch):
    monkeypatch.setattr(saved.lineage,'partial_cohort',lambda:True)
    for index in (32,33,39):
        assert saved.worker_segment(f'main_g{index:03d}_control','new')==('resume_partial','resume_partial',None)
    assert saved.worker_segment('main_g030_control','old')==('main_030','main_block',30)
    assert saved.worker_segment('main_g040_control','new')==('main_040','main_block',40)


def test_merged_inventory_has_no_copy_or_gap(tmp_path,monkeypatch):
    old=tmp_path/'old';new=tmp_path/'new'
    for root,indices in ((old,range(22)),(new,range(22,25))):
        (root/'moves').mkdir(parents=True)
        for i in indices:(root/'moves'/f'{i:06d}.json').write_text('{}')
    monkeypatch.setattr(saved.lineage,'game_folder',lambda s:new)
    monkeypatch.setattr(saved.lineage,'old_game_folder',lambda s:old)
    monkeypatch.setattr(saved.lineage,'partial',lambda s:True)
    saved.game_inventory({},dict(plies=25))
    (new/'moves/000021.json').write_text('{}')
    with pytest.raises(ValueError):saved.game_inventory({},dict(plies=25))
    (new/'moves/000021.json').unlink();(new/'moves/000023.json').unlink()
    with pytest.raises(ValueError):saved.game_inventory({},dict(plies=25))


def test_profile_relocation_only_changes_proven_provenance():
    olditem=dict(path='oldmanifest',bytes=1,sha256='a')
    newitem=dict(path='newmanifest',bytes=2,sha256='b')
    old=dict(checkpoint=dict(path='oldmodel',bytes=3,sha256='c'),tensor_manifest=dict(path='oldtensors',bytes=4,sha256='d'))
    new=dict(checkpoint=dict(path='newmodel',bytes=3,sha256='c'),tensor_manifest=dict(path='newtensors',bytes=4,sha256='d'))
    source=dict(checkpoint='oldmodel',learned_checkpoint_manifest=olditem,native_reference_model=old['checkpoint'],
                artifacts=[olditem,old['checkpoint'],old['tensor_manifest']],simulations=128,worlds=3)
    target=dict(checkpoint='newmodel',learned_checkpoint_manifest=newitem,native_reference_model=new['checkpoint'],
                artifacts=[newitem,new['checkpoint'],new['tensor_manifest']],simulations=128,worlds=3)
    assert parity.relocate_profile(source,target,olditem,newitem,old,new)
    target['simulations']=129
    with pytest.raises(ValueError):parity.relocate_profile(source,target,olditem,newitem,old,new)


def test_saved_training_worker_rejects_fresh_training(monkeypatch):
    monkeypatch.setattr(sys,'argv',['fit','--mode','fit','--plan-sha256','x','--output','x'])
    with pytest.raises(SystemExit) as err:fit.main()
    assert err.value.code==2


def test_original_numerical_fit_verifier_is_reused():
    from tools.splendorust_full_corpus_20261110 import fit as old
    assert fit.verify_fit is old.verify_fit
    assert fit.validate_commitment is old.validate_commitment


def test_strength_statistics_are_unmodified_ast():
    def functions(path):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)}
    old=functions(ROOT/'tools/splendorust_full_corpus_20261110/strength/analysis.py')
    new=functions(ROOT/'tools/splendorust_full_corpus_recovery_20261111/strength/analysis.py')
    assert old==new


def test_report_reads_each_completed_old_game_once_and_new_suffix(tmp_path,monkeypatch):
    old=tmp_path/'old';new=tmp_path/'new';group='terminal_parent'
    monkeypatch.setattr(c,'ORIGINAL_OUT',old)
    monkeypatch.setattr(report.a,'MATCHUP',group)
    monkeypatch.setattr(report.a,'MATCHUP_OUT',new/'strength'/group)
    for rep in (0,1):
        count=c.COMPLETED_PREFIXES[group][rep]
        for index in range(200):
            root=old if index<count else new
            p=root/'strength'/group/f'r{rep}'/'games/main'/f'g{index:03d}'/'result.json'
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(rep=rep,index=index,source='old' if root==old else 'new')))
    result=report.main_games()
    assert len(result[0])==len(result[1])==200
    assert sum(x['source']=='old' for rs in result.values() for x in rs)==72
    assert result[1][32]['source']=='new' and result[1][31]['source']=='old'

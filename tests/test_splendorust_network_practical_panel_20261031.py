"""Saved-only fresh-panel selection, exclusions and tensor provenance tests."""
from copy import deepcopy
from collections import Counter
import pytest
from tools.splendorust_network_practical_20261031 import panel as p,common as c


def corpus():
    rows=[]
    for phase,ply in [('early',2),('middle',22),('late',42)]:
        for seat in (0,1):
            for j in range(15):
                i=len(rows)
                rows.append(dict(game_index=i,ply=ply,seat=seat,progress=phase,game_plies=60,
                    original_receipt_seconds=.5+j*.05,ignored_score=(i%2)))
    return rows


def binder(row,index):
    tokens=[float(row['game_index'])]*1488
    item=dict(tokens=tokens,input_sha256=c.digest(tokens),output83=[.1]*83,batch=1)
    return dict(**row,root_id=f'r{index}',panel_id=f'r{index}',
        trace=dict(phases=[dict(inference=[item])]),
        source_row=dict(path=f"row{row['game_index']}",bytes=1,sha256='a'*64),
        source_clock=dict(path='clock',bytes=1,sha256='b'*64),
        source_input=dict(path='input',bytes=1,sha256='c'*64))


def test_fixed_rule_selects36_distinct_games_and_balanced_cells():
    rows=corpus();roots,extras,skipped=p.select_bound(rows,binder,{0,15},{})
    assert len(roots)==len(extras)==18 and not skipped
    assert len({r['game_index'] for r in roots+extras})==36
    assert not ({0,15}&{r['game_index'] for r in roots+extras})
    for phase in p.PROGRESS:
        for seat in (0,1):
            assert Counter(r['group'] for r in roots+extras if r['progress']==phase and r['seat']==seat)==Counter(systematic=2,tail=1,numerical=3)
    # Independent first-cell priority assertion; source outcomes never rank.
    firstcell=[r for r in rows if r['progress']=='early' and r['seat']==0 and r['game_index']!=0]
    expected=sorted(firstcell,key=lambda r:(p.priority(r,'timing'),r['game_index'],r['ply']))[:2]
    assert [r['game_index'] for r in roots[:2]]==[r['game_index'] for r in expected]
    reverse=deepcopy(rows)
    for row in reverse:row['ignored_score']=100000-row['ignored_score']
    altered=p.select_bound(list(reversed(reverse)),binder,{0,15},{})
    expected=p.select_bound(rows,binder,{0,15},{})
    for left,right in zip(altered[:2],expected[:2],strict=True):
        assert [(r['game_index'],r['ply'],r['group']) for r in left]==[(r['game_index'],r['ply'],r['group']) for r in right]
    assert altered[2]==expected[2]


def test_tails_take_maximum_remaining_response_per_cell():
    rows=corpus();roots,extras,_=p.select_bound(rows,binder,set(),set())
    used={r['game_index'] for r in roots[:12]}
    for tail in roots[12:]:
        available=[r for r in rows if r['progress']==tail['progress'] and r['seat']==tail['seat'] and r['game_index'] not in used]
        expected=sorted(available,key=lambda r:(-r['original_receipt_seconds'],r['game_index'],r['ply']))[0]
        assert tail['game_index']==expected['game_index'];used.add(tail['game_index'])


def test_development_tensor_exclusion_advances_without_replacement_sampling():
    rows=corpus();first=p.select_bound(rows,binder,set(),set())[0][0]
    excluded=p.first_tensor(first)['input_sha256']
    roots,extras,skipped=p.select_bound(rows,binder,set(),{excluded})
    assert excluded not in {p.first_tensor(r)['input_sha256'] for r in roots+extras}
    assert skipped and skipped[0]['game_index']==first['game_index']
    assert all(x['reason']=='duplicate_or_development_tensor' for x in skipped)


def test_missing_full_tensor_advances_and_is_disclosed():
    rows=corpus();bad=p.select_bound(rows,binder,set(),set())[0][0]['game_index']
    def limited(row,index):
        root=binder(row,index)
        if row['game_index']==bad:root['trace']['phases'][0]['inference'][0].pop('tokens')
        return root
    roots,extras,skipped=p.select_bound(rows,limited,set(),set())
    assert bad not in {r['game_index'] for r in roots+extras}
    assert any(r['reason']=='no_full_tensor' for r in skipped)


def test_numerical_plan_exact38_cases_and_independently_check_composites():
    roots,extras,_=p.select_bound(corpus(),binder,set(),set());plan=p.numerical_plan(roots,extras)
    fixtures=plan['fixtures'];assert len(fixtures)==38
    assert all(set(f)=={'fixture_id','batch','tokens'} for f in fixtures)
    assert [f['batch'] for f in fixtures]==[1]*36+[2,8]
    assert len({c.digest(f['tokens']) for f in fixtures[:36]})==36
    assert fixtures[36]['tokens']==fixtures[0]['tokens']+fixtures[1]['tokens']
    assert fixtures[37]['tokens']==sum([f['tokens'] for f in fixtures[:8]],[])
    for f,provenance in zip(fixtures[:36],plan['provenance'],strict=True):
        assert c.digest(f['tokens'])==provenance['input_sha256']
        assert provenance['phase_index']==provenance['callback_index']==0
    assert plan['new_models']==plan['new_games']==0


@pytest.mark.parametrize('change',['duplicate','seat','progress','late','nan'])
def test_malformed_census_fails_closed(change):
    rows=corpus()
    if change=='duplicate':rows.append(deepcopy(rows[0]))
    elif change=='seat':rows[0]['seat']=True
    elif change=='progress':rows[0]['progress']='late'
    elif change=='late':rows[0]['original_receipt_seconds']=10.1
    else:rows[0]['original_receipt_seconds']=float('nan')
    with pytest.raises(ValueError):p.select_bound(rows,binder,set(),set())


def test_precommit_required_before_source_reads(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path)
    with pytest.raises(ValueError,match='saved before extraction'):p.build_panel()


def test_actual_saved_tensor_shape_and_missing_midpoint_arrays():
    old=c.read(p.OLD_PANEL);root=old['roots'][0];selected=p.first_tensor(root)
    assert len(selected['tokens'])==1488 and len(selected['expected_output83'])==83
    assert selected['input_sha256']==c.digest(selected['tokens'])
    first_phase=root['trace']['phases'][0]['inference']
    assert 'tokens' in first_phase[0] and 'tokens' not in first_phase[len(first_phase)//2]

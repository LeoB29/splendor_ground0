"""Synthetic score fixtures only: pairing, forfeits, scale, strata and fixed coverage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tools.matched_strength_20261020 import statistics as s


def games(n=10,score=1):
    return [{'game':g,'pair':g//2,'fit':s.FITS[(g//2)%3],
        'local_seat':1-((g//2+g%2)%2),'local_score_halfpoints':score,
        'termination':'natural','forfeiting_player':None} for g in range(6*n)]


def test_opposite_seat_results_are_one_pair_not_two_observations():
    rows=games()
    for row in rows:row['local_score_halfpoints']=2 if row['local_seat']==0 else 0
    summary=s.statistics(rows,pairs_per_fit=10)
    assert summary['pairs']==30 and summary['games']==60
    assert summary['primary']['mean_local_game_score']==.5
    assert summary['primary']['ci95']==[.5,.5]
    assert summary['primary']['comparison_to_half']=='interval_includes_half'
    assert summary['by_local_seat']['0']['mean_local_game_score']==1
    assert summary['by_local_seat']['1']['mean_local_game_score']==0
    assert summary['pair_score_histogram']=={'0':0,'1':0,'2':30,'3':0,'4':0}


def test_fit_strata_remain_fixed_in_every_bootstrap_draw():
    rows=games()
    for row in rows:row['local_score_halfpoints']={161001:0,161002:1,161003:2}[row['fit']]
    result=s.statistics(rows,pairs_per_fit=10)
    assert result['primary']['mean_local_game_score']==.5 and result['primary']['ci95']==[.5,.5]
    assert [result['by_fit'][str(fit)]['mean_local_game_score'] for fit in s.FITS]==[0,.5,1]


@pytest.mark.parametrize('score,mean,claim',[(0,0.,'local_below_half'),(1,.5,'interval_includes_half'),(2,1.,'local_above_half')])
def test_integer_halfpoint_scale_and_equality_threshold(score,mean,claim):
    result=s.statistics(games(score=score),pairs_per_fit=10)
    assert result['primary']['mean_local_game_score']==mean
    assert result['primary']['ci95']==[mean,mean]
    assert result['primary']['comparison_to_half']==claim
    assert result['primary']['integer_pair_numerator_total']==60*score


def test_forfeits_count_as_losses_with_no_discarding_or_draw_substitution():
    rows=games()
    rows[0].update(termination='forfeit',forfeiting_player='local',local_score_halfpoints=0)
    rows[1].update(termination='forfeit',forfeiting_player='opponent',local_score_halfpoints=2)
    result=s.statistics(rows,pairs_per_fit=10)
    assert result['overall']['games']==60 and result['overall']['forfeits']=={'local':1,'opponent':1}
    assert result['overall']['local_wins']==1 and result['overall']['local_losses']==1 and result['overall']['draws']==58
    assert result['primary']['mean_local_game_score']==.5 and result['primary']['ci95']==[.5,.5]


def test_varying_pair_distribution_matches_independent_weighted_draw_calculation():
    rows=games(n=2)
    # Per-fit pair numerators are [0,4], [1,3], [2,2]. Exact primary mean is 1/2.
    halfpoints=[(0,0),(0,1),(1,1),(2,2),(1,2),(1,1)]
    for pair,(first,second) in enumerate(halfpoints):
        rows[2*pair]['local_score_halfpoints']=first;rows[2*pair+1]['local_score_halfpoints']=second
    random=np.random.Generator(np.random.PCG64(s.BOOTSTRAP_SEED))
    reference=[]
    # Different reduction: resampling multiplicities dotted with fixed pair scores.
    for start in range(0,50000,1000):
        block=np.zeros(1000,dtype=np.int64)
        for values in ([0,4],[1,3],[2,2]):
            index=random.integers(0,2,size=(1000,2))
            for i,draw in enumerate(index):block[i]+=int(np.dot(np.bincount(draw,minlength=2),values))
        reference.extend(block.tolist())
    expected=np.asarray(reference,dtype='<i8')
    result=s.statistics(rows,pairs_per_fit=2)
    assert result['primary']['integer_pair_numerator_total']==12
    assert result['primary']['mean_local_game_score']==.5
    assert result['primary']['ci95']==(np.quantile(expected,[.025,.975],method='linear')/24).tolist()
    assert result['bootstrap']['totals_sha256']==hashlib.sha256(expected.tobytes()).hexdigest()


def test_input_order_cannot_change_summary_or_random_schedule():
    rows=games(n=1);rows[0]['local_score_halfpoints']=0
    first=s.statistics(rows,pairs_per_fit=1)
    assert first==s.statistics(list(reversed(rows)),pairs_per_fit=1)
    assert first['primary']['mean_local_game_score']==5/12


@pytest.mark.parametrize('change,label',[
    (lambda rows:rows.pop(),'Complete fixed game sample'),
    (lambda rows:rows.append(deepcopy(rows[0])),'Complete fixed game sample'),
    (lambda rows:rows.__setitem__(1,deepcopy(rows[0])),'Unique complete fixed game IDs'),
    (lambda rows:rows[0].update(pair=1),'Frozen paired schedule identity'),
    (lambda rows:rows[0].update(fit=161002),'Frozen paired schedule identity'),
    (lambda rows:rows[0].update(local_seat=0),'Frozen paired schedule identity'),
    (lambda rows:rows[0].update(game=True),'Exact integer'),
    (lambda rows:rows[0].update(local_score_halfpoints=True),'Exact integer'),
    (lambda rows:rows[0].update(local_score_halfpoints=.5),'Exact integer'),
    (lambda rows:rows[0].update(local_score_halfpoints=3),'Game score'),
    (lambda rows:rows[0].update(termination='resource_hold'),'Only resolved'),
    (lambda rows:rows[0].update(forfeiting_player='local'),'Natural ending'),
    (lambda rows:rows[0].update(termination='forfeit',forfeiting_player='local'),'Genuine forfeit'),
    (lambda rows:rows[0].update(termination='forfeit',forfeiting_player='opponent'),'Genuine forfeit'),
    (lambda rows:rows[0].update(termination='forfeit',forfeiting_player='unknown'),'Genuine forfeit'),
    (lambda rows:rows[0].update(seed=123),'Exact verified-game score schema'),
])
def test_invalid_missing_or_replaced_games_block_statistics(change,label):
    rows=games();change(rows)
    with pytest.raises(ValueError,match=label):s.statistics(rows,pairs_per_fit=10)


@pytest.mark.parametrize('n',[0,-1,True,1.0,101])
def test_invalid_fixed_allocation(n):
    with pytest.raises(ValueError,match='Fixed positive pairs'):s.statistics(games(),pairs_per_fit=n)


@pytest.mark.parametrize('seed',[-1,True,1.,2**64])
def test_invalid_rng_seed(seed):
    with pytest.raises(ValueError,match='Fixed unsigned64'):s.statistics(games(),pairs_per_fit=10,bootstrap_seed=seed)


def test_statistics_remain_interruptible_at_bounded_bootstrap_chunks():
    calls=[]
    def boundary():
        calls.append(1)
        if len(calls)==4:raise RuntimeError('global resource deadline')
    with pytest.raises(RuntimeError,match='global resource deadline'):
        s.statistics(games(),pairs_per_fit=10,boundary=boundary)
    assert len(calls)==4


def test_runtime_contract_pins_actual_rng_quantile_and_interpreter_files():
    result=s.runtime_contract()
    assert json.loads(json.dumps(result))==result
    assert result['schema']=='matched_strength_statistics_runtime_v1'
    assert result['numpy_version']==np.__version__
    assert result['python']['version']==sys.version
    roles={item['role'] for item in result['files']}
    assert {'python_executable','numpy_package','quantile_implementation','pcg64',
            'generator','bit_generator','mtrand','random_common','array_core'} <= roles
    if sys.platform=='win32':assert 'python_shared_library' in roles
    for item in result['files']:
        path=Path(item['path'])
        assert path.is_absolute() and path.is_file()
        assert item['bytes']==path.stat().st_size
        with path.open('rb') as handle:
            assert item['sha256']==hashlib.file_digest(handle,'sha256').hexdigest()


def test_runtime_change_is_detected_without_version_change_or_cached_hash(tmp_path,monkeypatch):
    source=tmp_path/'synthetic_numerical_runtime.bin'
    source.write_bytes(b'first')
    monkeypatch.setattr(s,'_runtime_paths',lambda:[('synthetic',source)])
    before=s.runtime_contract()
    source.write_bytes(b'other')
    after=s.runtime_contract()
    assert before['numpy_version']==after['numpy_version']
    assert before['files'][0]['bytes']==after['files'][0]['bytes']
    assert before!=after
    assert after['files'][0]['sha256']==hashlib.sha256(b'other').hexdigest()


def test_missing_runtime_file_cannot_produce_a_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'_runtime_paths',lambda:[('missing',tmp_path/'absent')])
    with pytest.raises(FileNotFoundError):s.runtime_contract()

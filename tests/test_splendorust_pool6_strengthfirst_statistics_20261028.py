"""Model-free schedule, paired arithmetic and administrative-guard counterexamples."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from tools.splendorust_pool6_strengthfirst_20261028 import analysis as stats, schedule
from tools.splendorust_pool6_live_20261027 import schedule as old_schedule


def games(outcome=lambda pair, game, seat: 1):
    rows=[]
    for spec in schedule.stage_schedule('main'):
        seat=spec['challenger_seat']; value=outcome(spec['pair_index'],spec['game_index']%2,seat)
        winner=None if value==1 else seat if value==2 else 1-seat
        rows.append(dict(spec,complete=True,winner=winner,reason='natural'))
    return rows


def administrative(row, loser, kind='research_outer_hard_guard'):
    row.update(reason='forfeit',loser=loser,winner=1-loser,failed_seat=loser,
               failed_role='challenger' if loser==row['challenger_seat'] else 'opponent',
               administrative_forfeit=True,failure_policy=stats.FAILURE_POLICY,failure=kind)


def test_exact_fresh400_main6_operational_pairing_and_domain():
    assert schedule.DOMAIN=='splendorust_pool6_strengthfirst_20261028_v1'
    assert stats.PAIRS==schedule.PAIRS==200 and stats.SEED==2026102806
    main=schedule.stage_schedule('main'); ops=schedule.stage_schedule('operational')
    assert len(main)==400 and len(ops)==6
    assert schedule.validate_stage_schedule(main,'main')==main
    assert schedule.validate_stage_schedule(ops,'operational')==ops
    assert schedule.validate_schedule(schedule.make_schedule())==schedule.make_schedule()
    seeds=set()
    for stage,rows in [('main',main),('operational',ops)]:
        for pair in range(len(rows)//2):
            left,right=rows[2*pair:2*pair+2]
            encoded=json.dumps([schedule.DOMAIN,stage,pair],separators=(',',':')).encode()
            seed=str(int.from_bytes(hashlib.sha256(encoded).digest(),'big'))
            assert left['seed_decimal']==right['seed_decimal']==seed
            assert {left['challenger_seat'],right['challenger_seat']}=={0,1}
            assert left['challenger_seat']==pair%2
            assert seed not in seeds;seeds.add(seed)
    old={r['seed_decimal'] for s in ('main','operational') for r in old_schedule.stage_schedule(s)}
    assert len(seeds)==203 and not seeds.intersection(old)


@pytest.mark.parametrize('mutation', ['missing','extra','reversed','duplicate','old_seed','bool_index','changed_quota','extra_field'])
def test_complete_frozen_schedule_rejects_changes(mutation):
    rows=schedule.stage_schedule('main')
    if mutation=='missing':rows.pop()
    elif mutation=='extra':rows.append(deepcopy(rows[-1]))
    elif mutation=='reversed':rows.reverse()
    elif mutation=='duplicate':rows[-1]=deepcopy(rows[0])
    elif mutation=='old_seed':rows[0]['seed_decimal']=old_schedule.expected_seed('main',0)
    elif mutation=='bool_index':rows[0]['pair_index']=False
    elif mutation=='changed_quota':rows[0]['challenger_quota']=32
    elif mutation=='extra_field':rows[0]['posthoc_selection']=True
    with pytest.raises(ValueError):schedule.validate_stage_schedule(rows,'main')


def test_full400_arithmetic_and_independent200_pair_bootstrap():
    rows=games(lambda pair,game,seat:(pair+game)%3)
    administrative(rows[0],rows[0]['challenger_seat'])
    administrative(rows[7],1-rows[7]['challenger_seat'],'research_internal_hard_guard')
    result=stats.analyze_games(rows,True)
    halfpoints=[1 if r['winner'] is None else 2*int(r['winner']==r['challenger_seat']) for r in rows]
    pair_points=[halfpoints[i]+halfpoints[i+1] for i in range(0,400,2)]
    assert result['games']==400 and result['pairs']==200
    assert result['denominator']==800 and result['numerator']==sum(halfpoints)
    assert result['pair_numerators']==pair_points
    assert result['search_mean_score']==sum(halfpoints)/800
    assert result['by_search_seat']=={str(s):sum(v for r,v in zip(rows,halfpoints) if r['challenger_seat']==s)/400 for s in (0,1)}
    # A separate full replicate-major stream catches stale100-pair and denominator code.
    rng=np.random.Generator(np.random.PCG64(2026102806)); totals=[]; stream=hashlib.sha256()
    for _ in range(50000):
        indices=rng.integers(0,200,size=200,dtype=np.int64)
        stream.update(indices.astype('<i8').tobytes())
        totals.append(sum(pair_points[int(i)] for i in indices))
    ordered=sorted(totals)
    def percentile(q):
        offset=49999*q;lo=math.floor(offset);hi=math.ceil(offset)
        return (ordered[lo]+(ordered[hi]-ordered[lo])*(offset-lo))/800
    assert result['paired_ci95']==[percentile(.025),percentile(.975)]
    assert result['bootstrap']['indices_sha256']==stream.hexdigest()
    assert result['bootstrap']['draw_order']=='replicate_major_200_int64_indices'
    assert result['administrative_forfeits']==2 and result['natural_games']==398
    assert result['administrative_forfeits_by_player']=={'challenger':1,'opponent':1}
    assert result['administrative_forfeits_may_affect_strength_interpretation'] is True
    assert result['one_second_deployability_established'] is False
    assert result['timing_exceedances_seconds_are_descriptive_only']==[.7,.9,1.]
    assert stats.verify_analysis(list(reversed(rows)),result)['independent_full_sample_sensitivity']


@pytest.mark.parametrize('points,expected',[(2,'useful_evidence_for_pool6_under_research_guard'),
                                           (0,'useful_evidence_for_pool3_under_research_guard'),
                                           (1,'inconclusive_or_small')])
def test_extremes_draws_and_degenerate_pair_safeguard(points,expected):
    result=stats.analyze_games(games(lambda *args:points),True)
    assert result['numerator']==400*points and result['pair_numerators']==[2*points]*200
    assert result['paired_ci95']==[points/2,points/2]
    assert result['bootstrap_degenerate'] is True
    assert result['classification']==expected
    assert result['administrative_forfeits']==0 and result['natural_games']==400
    bounds=result['administrative_forfeit_sensitivity']
    assert bounds['all_administrative_candidate_losses']==bounds['all_administrative_candidate_wins']==points/2


@pytest.mark.parametrize('mean,ci,seats,expected',[
    (.55,[.50001,.60],{'0':.5,'1':.6},'useful_evidence_for_pool6_under_research_guard'),
    (.549999,[.51,.6],{'0':.5,'1':.6},'inconclusive_or_small'),
    (.55,[.5,.6],{'0':.5,'1':.6},'inconclusive_or_small'),
    (.70,[.6,.8],{'0':1.,'1':.4},'inconclusive_or_small'),
    (.45,[.40,.49999],{'0':.5,'1':.4},'useful_evidence_for_pool3_under_research_guard'),
    (.45,[.4,.5],{'0':.5,'1':.4},'inconclusive_or_small'),
])
def test_exact_prospective_useful_effect_boundaries(mean,ci,seats,expected):
    assert stats.classify_score(mean,ci,seats,[0,1],False)==expected
    assert stats.classify_score(mean,ci,seats,[0,1],True)=='inconclusive_or_small'


def test_administrative_sensitivity_keeps_full400_denominator_and_no_natural_only_estimate():
    rows=games();administrative(rows[0],rows[0]['challenger_seat']);administrative(rows[1],1-rows[1]['challenger_seat'])
    bound=stats.administrative_sensitivity(stats.validated_games(rows))
    assert bound['denominator']==800 and bound['complete_sample_games']==400
    assert bound['lower_numerator']==398 and bound['upper_numerator']==402
    assert bound['all_administrative_candidate_losses']==398/800
    assert bound['all_administrative_candidate_wins']==402/800
    assert bound['natural_only_strength_estimate'] is False


@pytest.mark.parametrize('field,value',[
    ('administrative_forfeit',False),('administrative_forfeit',1),
    ('failure_policy','one_second_timeout_loss'),('failure','timeout'),('failure','illegal'),
    ('failure','infrastructure'),('winner',0),('failed_role','opponent'),('failed_seat',1),('loser',True)
])
def test_only_authentic_coherent_research_guard_forfeits_are_counted(field,value):
    rows=games();administrative(rows[0],0);rows[0][field]=value
    with pytest.raises(ValueError):stats.validated_games(rows)


@pytest.mark.parametrize('field,value',[
    ('complete',False),('complete',1),('winner',True),('reason','safety_cutoff'),
    ('loser',0),('failed_role','challenger'),('administrative_forfeit',True),('failure','timeout'),
    ('failure_policy',stats.FAILURE_POLICY),('search_seat',1),('challenger_simulations',32),
    ('challenger_mode','frozen'),('opponent_quota',32)
])
def test_changed_identity_or_unofficial_natural_outcome_rejected(field,value):
    rows=games();rows[0][field]=value
    with pytest.raises(ValueError):stats.validated_games(rows)


@pytest.mark.parametrize('field',['winner','complete','seed_decimal','challenger_mode','reason'])
def test_required_result_fields_cannot_be_omitted(field):
    rows=games();del rows[0][field]
    with pytest.raises(ValueError):stats.validated_games(rows)


def test_no_partial_old200_sample_operational_substitution_or_unverified_result():
    rows=games()
    for subset in (rows[:200],rows[:-1],rows+rows[:1],tuple(rows)):
        with pytest.raises(ValueError,match='400'):stats.validated_games(subset)
    with pytest.raises(ValueError,match='duplicate'):stats.validated_games(rows[:-1]+rows[:1])
    bad=deepcopy(rows);bad[0]=dict(schedule.stage_schedule('operational')[0],winner=None,reason='natural',complete=True)
    with pytest.raises(ValueError,match='Operational'):stats.validated_games(bad)
    for gate in (False,None,1):
        with pytest.raises(ValueError,match='gates'):stats.analyze_games(rows,gate)


def test_resource_boundary_interrupts_before_any_partial_primary_result():
    class Stop(Exception):pass
    def stop():raise Stop('bound')
    with pytest.raises(Stop):stats.analyze_games(games(),True,boundary=stop)


@pytest.mark.parametrize('field', ['denominator','by_search_seat','administrative_forfeit_sensitivity','classification'])
def test_independent_verifier_rejects_tampered_reporting(field):
    rows=games();result=stats.analyze_games(rows,True)
    result[field]=None
    with pytest.raises((ValueError,TypeError)):stats.verify_analysis(rows,result)

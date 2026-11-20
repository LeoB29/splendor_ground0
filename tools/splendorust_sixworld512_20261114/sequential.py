"""Frozen pair-level group-sequential decisions; no models or game execution.

The O'Brien--Fleming-shaped confidence limits are approximate and simulation-
qualified on the prespecified bounded pentanomial family. They are not claimed
to be a distribution-free confidence sequence. Calibration is independent of
every game in the prospective comparison.
"""
from __future__ import annotations
import hashlib
import json
import math
from statistics import NormalDist

SCHEMA='sixworld512_sequential_v2'
MAX_PAIRS=1800
LOOKS=tuple(range(200,1801,200))
SCORES=(0.,.25,.5,.75,1.)
OBF_CONSTANT=2.18
VARIANCE_FLOOR=.05
ALPHA=.025
TEACHER_THRESHOLD=.48
PLAYER_THRESHOLD=.5
TARGET_SCORE=1/(1+10**(-17/400))
FUTILITY_THRESHOLD=.1
FUTILITY_MIN_PAIRS=400
ND=NormalDist()


def require(value,message):
    if not value:raise ValueError('SIXWORLD_SEQUENTIAL: '+message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def configuration():
    return dict(schema=SCHEMA,max_pairs=MAX_PAIRS,max_games=2*MAX_PAIRS,
        looks_pairs=list(LOOKS),looks_games=[2*n for n in LOOKS],
        pair_score_support=list(SCORES),unit='complete_swapped_seat_pair',
        mean_target='expected_candidate_game_score',obf_constant=OBF_CONSTANT,
        sample_pair_variance_floor=VARIANCE_FLOOR,
        skewness_guard='conservative_first_order_studentized_tail_inflation',
        student_t_guard='add_(z^3+z)/(4*(n-1))_to_both_tails',
        critical_values=[OBF_CONSTANT*math.sqrt(MAX_PAIRS/n) for n in LOOKS],
        lower_tail_alpha=ALPHA,upper_tail_alpha=ALPHA,
        interval='simulation_qualified_skew_guarded_studentized_OBF_shaped_approximation',
        distribution_free_guarantee=False,teacher_threshold=TEACHER_THRESHOLD,
        player_threshold=PLAYER_THRESHOLD,material_harm_threshold=TEACHER_THRESHOLD,
        teacher_margin=.02,observed_55_percent_gate=False,
        target_elo=17,target_score=TARGET_SCORE,
        futility_min_pairs=FUTILITY_MIN_PAIRS,futility_probability=FUTILITY_THRESHOLD,
        futility='final_look_normal_conditional_power_both_goals',
        futility_treatment='nonbinding_for_type_I_calibration_but_executed_as_frozen',
        carry_teacher_certificate=True,
        precedence=['superiority','material_harm','maximum_sample','protected_futility','continue'])


def validate_calibration(calibration):
    require(type(calibration) is dict and calibration.get('schema')=='sixworld512_statistical_qualification_v2',
        'Source-bound statistical qualification required')
    require(calibration.get('status')=='passed' and calibration.get('configuration')==configuration()
        and calibration.get('configuration_sha256')==digest(configuration()),'Exact qualified configuration')
    proof=calibration.get('acceptance',{})
    require(proof.get('null_family_simultaneous_upper_max',1)<=ALPHA
        and proof.get('all_null_directions_pass') is True
        and proof.get('power_simultaneous_lower_min',0)>=.78
        and proof.get('both_power_cases_pass') is True
        and proof.get('calibration_validation_seeds_disjoint') is True
        and proof.get('production_scalar_vector_agreement') is True,
        'Null, power and independent implementation qualification passed')
    return True


def _numerators(pair_scores):
    require(type(pair_scores) in (list,tuple),'Explicit complete-pair score sequence')
    require(len(pair_scores)<=MAX_PAIRS,'No extension beyond1800 pairs')
    result=[]
    for score in pair_scores:
        require(type(score) in (int,float) and math.isfinite(score) and score in SCORES,
            'Each complete pair scores0,.25,.5,.75 or1; booleans are invalid')
        result.append(int(score*4))
    return result


def pair_score(game_scores):
    require(type(game_scores) in (list,tuple) and len(game_scores)==2,'Exactly two swapped-seat games per pair')
    require(all(type(x) in (int,float) and math.isfinite(x) and x in (0.,.5,1.) for x in game_scores),
        'Per-game win/draw/loss credit is1,.5,0')
    return (game_scores[0]+game_scores[1])/2


def _moments(numerators):
    n=len(numerators);total=sum(numerators);square=sum(x*x for x in numerators)
    # Integer sufficient statistics avoid cancellation and ordering-dependent means.
    mean=total/(4*n)
    raw_var=(n*square-total*total)/(16*n*(n-1))
    require(raw_var>=0,'Nonnegative exact sample variance')
    variance=max(raw_var,VARIANCE_FLOOR)
    return mean,raw_var,variance


def _skewness(numerators,variance):
    n=len(numerators);total=sum(numerators);square=sum(x*x for x in numerators);cube=sum(x*x*x for x in numerators)
    # Exact integer numerator for the empirical central third moment.
    third=(n*n*cube-3*n*total*square+2*total**3)/(64*n**3)
    return third,third/(variance**1.5)


def _tail_criticals(n,skewness):
    z=OBF_CONSTANT*math.sqrt(MAX_PAIRS/n)
    heavy=(z**3+z)/(4*(n-1))
    skew_scale=(2*z*z+1)/(6*math.sqrt(n))
    # Only widen the vulnerable side of the first-order studentized-mean
    # Cornish--Fisher correction; never narrow either original OBF tail.
    return z,z+heavy+max(0.,-skewness)*skew_scale,z+heavy+max(0.,skewness)*skew_scale


def _conditional_power(mean,variance,n,threshold,future_score,skewness=0.):
    """Conditional chance of crossing the FINAL boundary under fixed future mean.

    This is a prospective futility approximation, not a posterior probability or
    an assertion that the assumed future score is true. It can only stop work;
    the confidence bound alone authorizes either positive conclusion.
    """
    final_critical=_tail_criticals(MAX_PAIRS,skewness)[1]
    if n==MAX_PAIRS:
        return float(mean-final_critical*math.sqrt(variance/n)>threshold)
    expected_sum=n*mean+(MAX_PAIRS-n)*future_score
    required_sum=MAX_PAIRS*threshold+final_critical*math.sqrt(MAX_PAIRS*variance)
    z=(expected_sum-required_sum)/math.sqrt((MAX_PAIRS-n)*variance)
    return ND.cdf(z)


def _look(numerators,teacher_certified=False):
    n=len(numerators);require(n in LOOKS,'Only a prospectively scheduled complete-pair look')
    mean,raw_var,variance=_moments(numerators)
    third,skewness=_skewness(numerators,variance)
    critical,lower_critical,upper_critical=_tail_criticals(n,skewness)
    se=math.sqrt(variance/n)
    lower,upper=max(0.,mean-lower_critical*se),min(1.,mean+upper_critical*se)
    superior=lower>PLAYER_THRESHOLD
    teacher_now=lower>TEACHER_THRESHOLD
    harm=upper<TEACHER_THRESHOLD
    teacher_ever=bool(teacher_certified or teacher_now)
    cp_player=_conditional_power(mean,variance,n,PLAYER_THRESHOLD,TARGET_SCORE,skewness)
    cp_teacher=_conditional_power(mean,variance,n,TEACHER_THRESHOLD,.5,skewness)
    futile=(FUTILITY_MIN_PAIRS<=n<MAX_PAIRS and cp_player<FUTILITY_THRESHOLD
        and (teacher_ever or cp_teacher<FUTILITY_THRESHOLD))
    if superior:stop,reason,classification=True,'superiority','superior'
    elif harm:stop,reason,classification=True,'material_harm','materially_worse'
    elif n==MAX_PAIRS:
        stop,reason,classification=True,'maximum_sample','teacher_noninferior' if teacher_ever else 'unresolved'
    elif futile:
        stop,reason,classification=True,'protected_futility','teacher_noninferior' if teacher_ever else 'unresolved'
    else:stop,reason,classification=False,'continue','teacher_noninferior_pending' if teacher_ever else 'unresolved_pending'
    return dict(pairs=n,games=2*n,score=mean,pair_numerator_sum=sum(numerators),
        pair_numerator_square_sum=sum(x*x for x in numerators),sample_pair_variance=raw_var,
        pair_numerator_cube_sum=sum(x*x*x for x in numerators),third_central_moment=third,
        standardized_third_moment=skewness,lower_critical_value=lower_critical,upper_critical_value=upper_critical,
        variance_used=variance,variance_floor_applied=raw_var<VARIANCE_FLOOR,
        critical_value=critical,lower=lower,upper=upper,superiority_pass=superior,
        teacher_pass_this_look=teacher_now,teacher_certified_ever=teacher_ever,
        material_harm_pass=harm,conditional_power_player=cp_player,
        conditional_power_teacher=cp_teacher,futility_pass=futile,
        stop=stop,stop_reason=reason,classification=classification,
        player_eligible=superior,teacher_eligible=teacher_ever and not harm)


def analyze_pairs(pair_scores,calibration):
    validate_calibration(calibration)
    nums=_numerators(pair_scores)
    require(len(nums) in LOOKS,'Analysis only at400-game scheduled looks, never extra peeks')
    looks=[];certified=False
    for n in LOOKS:
        if n>len(nums):break
        look=_look(nums[:n],certified);looks.append(look);certified=look['teacher_certified_ever']
        if look['stop']:
            require(n==len(nums),'No games may be added after an earlier frozen stopping decision')
            break
    last=looks[-1]
    return dict(schema=SCHEMA,status='stop' if last['stop'] else 'continue',
        configuration_sha256=digest(configuration()),calibration_sha256=digest(calibration),
        pair_scores_sha256=digest(list(pair_scores)),pair_numerators=nums,
        games=len(nums)*2,pairs=len(nums),look=last,looks=looks,
        classification=last['classification'],stop=last['stop'],stop_reason=last['stop_reason'],
        player_eligible=last['player_eligible'],teacher_eligible=last['teacher_eligible'],
        promotion_requires_full_verification=True,new_games=0,models=0)


def verify_analysis(pair_scores,calibration,reported):
    """Independent arithmetic path: rational moments and independently expressed tails."""
    from fractions import Fraction
    validate_calibration(calibration);nums=_numerators(pair_scores)
    require(type(reported) is dict and reported.get('schema')==SCHEMA,'Sequential report schema')
    require(reported['pair_numerators']==nums and reported['pair_scores_sha256']==digest(list(pair_scores))
        and reported['configuration_sha256']==digest(configuration())
        and reported['calibration_sha256']==digest(calibration),'Exact pairs/configuration/calibration binding')
    require(len(nums) in LOOKS,'Audited only at fixed complete-pair look')
    seen_teacher=False;last=None;expected_looks=0
    for n in LOOKS:
        if n>len(nums):break
        expected_looks+=1;r=reported['looks'][expected_looks-1]
        x=[Fraction(q,4) for q in nums[:n]];avg=sum(x,Fraction())/n
        raw=sum((v-avg)**2 for v in x)/ (n-1)
        third=sum((v-avg)**3 for v in x)/n
        variance=max(float(raw),.05);se=math.sqrt(variance/n)
        skewness=float(third)/(variance**1.5)
        z=2.18/math.sqrt(n/1800)
        heavy=(z**3+z)/(4*(n-1));scale=(2*z*z+1)/(6*math.sqrt(n))
        zl=z+heavy+max(0.,-skewness)*scale;zu=z+heavy+max(0.,skewness)*scale
        lo=max(0.,float(avg)-zl*se);hi=min(1.,float(avg)+zu*se)
        for key,val in [('score',float(avg)),('sample_pair_variance',float(raw)),('variance_used',variance),
                        ('critical_value',z),('lower',lo),('upper',hi),('third_central_moment',float(third)),
                        ('standardized_third_moment',skewness),('lower_critical_value',zl),('upper_critical_value',zu)]:
            require(math.isclose(r[key],val,rel_tol=1e-12,abs_tol=1e-14),'Independent arithmetic: '+key)
        require(r['pair_numerator_sum']==sum(nums[:n]) and r['pair_numerator_square_sum']==sum(q*q for q in nums[:n])
            and r['pair_numerator_cube_sum']==sum(q*q*q for q in nums[:n]),
            'Independent integer moments')
        now=lo>.48;seen_teacher=seen_teacher or now
        sup=lo>.5;harm=hi<.48
        powers=[]
        for threshold,assumed in ((.5,1/(1+10**(-17/400))),(.48,.5)):
            if n==1800:prob=float(lo>threshold)
            else:
                finalz=2.18+(2.18**3+2.18)/(4*1799)+max(0.,-skewness)*(2*2.18**2+1)/(6*math.sqrt(1800))
                required=1800*threshold+finalz*math.sqrt(1800*variance)
                future_mean=n*float(avg)+(1800-n)*assumed
                prob=.5*math.erfc((required-future_mean)/math.sqrt(2*(1800-n)*variance))
            powers.append(prob)
        for key,val in zip(('conditional_power_player','conditional_power_teacher'),powers):
            require(math.isclose(r[key],val,rel_tol=1e-10,abs_tol=2e-15),'Independent conditional power')
        futile=400<=n<1800 and powers[0]<.1 and (seen_teacher or powers[1]<.1)
        if sup:stop,reason,label=True,'superiority','superior'
        elif harm:stop,reason,label=True,'material_harm','materially_worse'
        elif n==1800:stop,reason,label=True,'maximum_sample','teacher_noninferior' if seen_teacher else 'unresolved'
        elif futile:stop,reason,label=True,'protected_futility','teacher_noninferior' if seen_teacher else 'unresolved'
        else:stop,reason,label=False,'continue','teacher_noninferior_pending' if seen_teacher else 'unresolved_pending'
        expected=dict(pairs=n,games=2*n,variance_floor_applied=float(raw)<.05,superiority_pass=sup,
            teacher_pass_this_look=now,teacher_certified_ever=seen_teacher,material_harm_pass=harm,
            futility_pass=futile,stop=stop,stop_reason=reason,classification=label,
            player_eligible=sup,teacher_eligible=seen_teacher and not harm)
        require(all(r[k]==v for k,v in expected.items()),'Independent boundary and stop classification')
        last=r
        if stop:
            require(n==len(nums),'Independent no-post-stop extension')
            break
    require(len(reported['looks'])==expected_looks and reported['look']==last,'Exact complete interim history')
    for key in ('stop','stop_reason','classification','player_eligible','teacher_eligible'):
        require(reported[key]==last[key],'Top-level decision consistent')
    require(reported['status']==('stop' if last['stop'] else 'continue') and
        reported['games']==2*len(nums) and reported['pairs']==len(nums)
        and reported['promotion_requires_full_verification'] is True,'No premature eligibility application')
    return dict(status='passed',independent_fraction_moments=True,independent_erfc_tails=True,
        pairs=len(nums),games=len(nums)*2,scheduled_looks=expected_looks,models=0,new_games=0,
        report_sha256=digest(reported))

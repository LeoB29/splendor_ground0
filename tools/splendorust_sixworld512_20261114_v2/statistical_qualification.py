"""Bounded, model-free finite-pair qualification of the frozen sequential rule.

The prespecified constant is not fitted to prospective game scores. Discovery
and validation use disjoint seeds; validation has simultaneous binomial upper
bounds over its fixed distribution grid. A passed grid is empirical validation,
not a mathematical uniform guarantee over every possible pair distribution.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import traceback
from itertools import combinations
from . import sequential as s

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/splendorust_sixworld512_20261114_01/prelaunch'
BASELINE=ROOT/'outputs/splendorust_search512_20261101_01/main_analysis.json'
SCHEMA='sixworld512_statistical_qualification_v2'


def plan():
    return dict(schema='sixworld512_statistical_qualification_plan_v2',
        constant_prespecified=2.18,variance_floor_prespecified=.05,
        revision='conservative_skewness_and_student_t_tail_guards',
        discovery_paths_per_case=30000,validation_paths_per_case=200000,
        power_paths_per_case=400000,simulation_batch=10000,
        grid_seed=202611142801,discovery_seed=202611142802,
        validation_seed=202611142803,power_seed=202611142804,
        grid_means=[.02,.1,.25,.48,.5,.52,.75,.9,.98],
        mixture_weights=[.25,.5,.75],random_convex_mixtures_per_mean=16,
        validation_family_error=.01,power_family_error=.01,
        maximum_simultaneous_null_upper=.025,minimum_simultaneous_power_lower=.78,
        critical_null_means=[.48,.5],null_both_directions=True,
        null_error_ignores_futility=True,stop_rule_in_power=True,
        numerical_boundary_seconds=1200,models=0,new_games=0)


def _artifact(path):
    path=Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _cp_bound(successes,trials,error,upper):
    from scipy.stats import beta
    if upper:return 1. if successes==trials else float(beta.ppf(1-error,successes+1,trials-successes))
    return 0. if successes==0 else float(beta.ppf(error,successes,trials-successes+1))


def distributions():
    import numpy as np
    cfg=plan();rng=np.random.default_rng(cfg['grid_seed']);support=np.arange(5)/4
    result=[]
    for mean in cfg['grid_means']:
        vertices=[]
        for i,j in combinations(range(5),2):
            if support[i]<=mean<=support[j]:
                p=np.zeros(5);p[j]=(mean-support[i])/(support[j]-support[i]);p[i]=1-p[j]
                if not any(np.allclose(p,q,atol=1e-14,rtol=0) for q in vertices):vertices.append(p)
        rows=vertices.copy()
        for a,b in combinations(vertices,2):
            for weight in cfg['mixture_weights']:
                p=weight*a+(1-weight)*b
                if not any(np.allclose(p,q,atol=1e-14,rtol=0) for q in rows):rows.append(p)
        for _ in range(cfg['random_convex_mixtures_per_mean']):
            rows.append(rng.dirichlet(np.ones(len(vertices)))@np.array(vertices))
        for index,p in enumerate(rows):
            s.require(abs(float(p@support)-mean)<1e-12 and abs(float(sum(p))-1)<1e-12,'Exact null-grid moments')
            result.append(dict(case_id=f'm{mean:.2f}_{index:03d}',mean=mean,probabilities=[float(x) for x in p],
                variance=float(p@((support-mean)**2)),vertex_or_mixture_index=index))
    return result


def _vector(counts):
    """Simulation path independent of production scalar integer-moment code."""
    import numpy as np
    from scipy.special import ndtr
    n=np.array(s.LOOKS,dtype=float);support=np.arange(5)/4
    mean=(counts@support)/n
    var=np.maximum(((counts@(support*support))-n*mean*mean)/(n-1),s.VARIANCE_FLOOR)
    third=counts@(support**3)/n-3*mean*(counts@(support**2)/n)+2*mean**3
    skewness=third/(var**1.5)
    critical=s.OBF_CONSTANT*np.sqrt(s.MAX_PAIRS/n)
    heavy=(critical**3+critical)/(4*(n-1));scale=(2*critical**2+1)/(6*np.sqrt(n))
    lc=critical+heavy+np.maximum(0,-skewness)*scale;uc=critical+heavy+np.maximum(0,skewness)*scale
    lower=np.maximum(0,mean-lc*np.sqrt(var/n));upper=np.minimum(1,mean+uc*np.sqrt(var/n))
    teacher_now=lower>.48;teacher=np.maximum.accumulate(teacher_now,axis=1)
    superior=lower>.5;harm=upper<.48
    remaining=s.MAX_PAIRS-n
    denominator=np.sqrt(np.maximum(1,remaining)*var)
    finalc=s.OBF_CONSTANT+(s.OBF_CONSTANT**3+s.OBF_CONSTANT)/(4*(s.MAX_PAIRS-1))+np.maximum(0,-skewness)*(2*s.OBF_CONSTANT**2+1)/(6*np.sqrt(s.MAX_PAIRS))
    player_cp=ndtr((n*mean+remaining*s.TARGET_SCORE-s.MAX_PAIRS*.5-finalc*np.sqrt(s.MAX_PAIRS*var))/denominator)
    teacher_cp=ndtr((n*mean+remaining*.5-s.MAX_PAIRS*.48-finalc*np.sqrt(s.MAX_PAIRS*var))/denominator)
    player_cp[:,-1]=(lower[:,-1]>.5);teacher_cp[:,-1]=(lower[:,-1]>.48)
    futility=(n>=400)&(n<1800)&(player_cp<.1)&(teacher|(teacher_cp<.1))
    stop=superior|harm|futility;stop[:,-1]=True
    first=np.argmax(stop,axis=1)
    return dict(mean=mean,variance=var,third=third,skewness=skewness,lc=lc,uc=uc,lower=lower,upper=upper,teacher=teacher,superior=superior,
        harm=harm,player_cp=player_cp,teacher_cp=teacher_cp,futility=futility,first=first)


def _simulate_null(case,paths,seed,boundary):
    import numpy as np
    rng=np.random.default_rng(seed);lower=upper=0;done=0
    while done<paths:
        boundary();size=min(plan()['simulation_batch'],paths-done)
        counts=rng.multinomial(200,case['probabilities'],size=(size,9)).cumsum(axis=1)
        value=_vector(counts)
        lower+=int(np.count_nonzero(np.any(value['lower']>case['mean'],axis=1)))
        upper+=int(np.count_nonzero(np.any(value['upper']<case['mean'],axis=1)))
        done+=size
    return dict(**case,paths=paths,lower_errors=lower,upper_errors=upper,
        lower_error_estimate=lower/paths,upper_error_estimate=upper/paths,seed=seed)


def _tilt(probabilities,mean):
    import numpy as np
    p=np.array(probabilities,dtype=float);support=np.arange(5)/4;lo,hi=-80.,80.
    for _ in range(100):
        mid=(lo+hi)/2;weights=p*np.exp(mid*(support-.5));weights/=weights.sum()
        if float(weights@support)<mean:lo=mid
        else:hi=mid
    weights=p*np.exp(((lo+hi)/2)*(support-.5));weights/=weights.sum()
    return [float(x) for x in weights]


def power_cases():
    import numpy as np
    baseline=json.loads(BASELINE.read_text(encoding='utf-8'))
    nums=baseline['official']['pair_numerators']
    s.require(len(nums)==200 and baseline['complete_verification'] is True,'Accepted historical planning input')
    hist=np.bincount(nums,minlength=5)/len(nums)
    output=[]
    for name,mean in [('gain17',s.TARGET_SCORE),('equal',.5)]:
        iid=[(1-mean)**2,0.,2*mean*(1-mean),0.,mean**2]
        output.append(dict(case_id='independent_game_pair_'+name,mean=mean,probabilities=iid,power_acceptance=name=='gain17'))
        output.append(dict(case_id='historical_tilt_'+name,mean=mean,probabilities=_tilt(hist,mean),power_acceptance=name=='gain17'))
    return output


def _simulate_power(case,paths,seed,boundary):
    import numpy as np
    rng=np.random.default_rng(seed);wins=teachers=harm=unresolved=futility=0;pair_sum=0;done=0
    stop_hist=np.zeros(9,dtype=np.int64)
    for start in range(0,paths,plan()['simulation_batch']):
        boundary();size=min(plan()['simulation_batch'],paths-start)
        counts=rng.multinomial(200,case['probabilities'],size=(size,9)).cumsum(axis=1)
        v=_vector(counts);row=np.arange(size);ix=v['first'];stop_hist+=np.bincount(ix,minlength=9)
        won=v['superior'][row,ix];lost=v['harm'][row,ix];t=v['teacher'][row,ix]&~lost
        wins+=int(np.count_nonzero(won));teachers+=int(np.count_nonzero(t));harm+=int(np.count_nonzero(lost))
        unresolved+=int(np.count_nonzero(~won&~lost&~t))
        futility+=int(np.count_nonzero(v['futility'][row,ix]&~won&~lost))
        pair_sum+=int(np.sum((ix+1)*200));done+=size
    return dict(**case,paths=paths,seed=seed,superior=wins,teacher_eligible=teachers,material_harm=harm,
        unresolved=unresolved,futile=futility,power=wins/paths,teacher_probability=teachers/paths,
        expected_games=pair_sum*2/paths,stop_histogram=[int(x) for x in stop_hist])


def implementation_agreement(boundary):
    import numpy as np
    rng=np.random.default_rng(202611142805)
    checked=0
    for p in ([.25,0,.5,0,.25],[.3,.05,.3,.05,.3],[0,.6933333333333334,0,0,.30666666666666664],
              [.01,.52,.24,0,.23],[0,0,1,0,0],[0,0,0,0,1]):
        increments=rng.multinomial(200,p,size=(20,9));counts=increments.cumsum(axis=1);vec=_vector(counts)
        for trial in range(20):
            scores=[];seen=False
            for j,n in enumerate(s.LOOKS):
                boundary();scores.extend(q/4 for q,count in enumerate(increments[trial,j]) for _ in range(int(count)))
                scalar=s._look(s._numerators(scores),seen);seen=scalar['teacher_certified_ever']
                fields={'score':'mean','variance_used':'variance','lower':'lower','upper':'upper',
                    'third_central_moment':'third','standardized_third_moment':'skewness',
                    'lower_critical_value':'lc','upper_critical_value':'uc',
                    'conditional_power_player':'player_cp','conditional_power_teacher':'teacher_cp'}
                for key,array in fields.items():
                    s.require(math.isclose(scalar[key],float(vec[array][trial,j]),rel_tol=1e-9,abs_tol=2e-13),
                        'Independent scalar/vector numeric agreement: '+key)
                for key,array in [('teacher_certified_ever','teacher'),('superiority_pass','superior'),
                                  ('material_harm_pass','harm'),('futility_pass','futility')]:
                    s.require(scalar[key]==bool(vec[array][trial,j]),'Independent scalar/vector decision agreement')
                checked+=1
    return dict(passed=True,scheduled_looks_checked=checked,seed=202611142805)


def qualify(boundary=lambda:None,progress=lambda stage,row:None):
    """Run the prespecified finite model-free qualification, returning failure too."""
    import numpy as np
    import scipy
    started=time.monotonic();cfg=plan();config=s.configuration();grid=distributions()
    def bounded():
        boundary();s.require(time.monotonic()-started<cfg['numerical_boundary_seconds'],'Bounded statistical qualification clock')
    sources=[_artifact(Path(__file__)),_artifact(Path(s.__file__)),_artifact(BASELINE)]
    # The preliminary stage is descriptive and cannot change the fixed constant.
    discovery=[]
    for i,case in enumerate(r for r in grid if r['mean'] in (.48,.5)):
        row=_simulate_null(case,cfg['discovery_paths_per_case'],cfg['discovery_seed']+i*1009,bounded)
        discovery.append(row);progress('discovery',row)
    validation=[];family_error=cfg['validation_family_error']/(2*len(grid))
    for i,case in enumerate(grid):
        row=_simulate_null(case,cfg['validation_paths_per_case'],cfg['validation_seed']+i*1009,bounded)
        for direction in ('lower','upper'):
            row[direction+'_error_simultaneous_upper']=_cp_bound(row[direction+'_errors'],row['paths'],family_error,True)
        validation.append(row);progress('validation',row)
    powers=[];cases=power_cases();power_error=cfg['power_family_error']/2
    for i,case in enumerate(cases):
        row=_simulate_power(case,cfg['power_paths_per_case'],cfg['power_seed']+i*1009,bounded)
        row['power_simultaneous_lower']=_cp_bound(row['superior'],row['paths'],power_error,False)
        powers.append(row);progress('power',row)
    agreement=implementation_agreement(bounded)
    max_error=max(row[key] for row in validation for key in ('lower_error_simultaneous_upper','upper_error_simultaneous_upper'))
    min_power=min(row['power_simultaneous_lower'] for row in powers if row['power_acceptance'])
    acceptance=dict(null_family_simultaneous_upper_max=max_error,all_null_directions_pass=max_error<=.025,
        power_simultaneous_lower_min=min_power,both_power_cases_pass=min_power>=.78,
        calibration_validation_seeds_disjoint=not ({r['seed'] for r in discovery}&{r['seed'] for r in validation}),
        production_scalar_vector_agreement=agreement['passed'])
    passed=all(acceptance[k] for k in ('all_null_directions_pass','both_power_cases_pass',
        'calibration_validation_seeds_disjoint','production_scalar_vector_agreement'))
    result=dict(schema=SCHEMA,status='passed' if passed else 'failed',configuration=config,
        configuration_sha256=s.digest(config),qualification_plan=cfg,qualification_plan_sha256=s.digest(cfg),
        null_grid_sha256=s.digest(grid),null_grid_cases=len(grid),discovery=discovery,validation=validation,
        power=powers,acceptance=acceptance,implementation_agreement=agreement,inputs=sources,
        numpy_version=np.__version__,scipy_version=scipy.__version__,elapsed_seconds=time.monotonic()-started,
        confidence_scope='Approximate intervals, independently simulated on prespecified finite grids; not distribution-free.',
        scalar_verifier_uses='Fraction moments and erfc conditional tails',models=0,new_games=0)
    verify_qualification(result)
    return result


def verify_qualification(report):
    """Recheck MC arithmetic, its finite inventory, thresholds and no seed reuse."""
    cfg=plan();grid=distributions()
    s.require(report['schema']==SCHEMA and report['configuration']==s.configuration()
        and report['configuration_sha256']==s.digest(s.configuration()),'Qualified exact rule')
    s.require(report['qualification_plan']==cfg and report['qualification_plan_sha256']==s.digest(cfg)
        and report['null_grid_sha256']==s.digest(grid) and report['null_grid_cases']==len(grid),'Exact prospective simulation plan')
    s.require(len(report['validation'])==len(grid),'Complete validation distribution family')
    discovery=[case for case in grid if case['mean'] in (.48,.5)]
    s.require(len(report['discovery'])==len(discovery),'Complete descriptive calibration family')
    for i,(case,row) in enumerate(zip(discovery,report['discovery'],strict=True)):
        s.require(all(row[k]==v for k,v in case.items()) and row['paths']==cfg['discovery_paths_per_case']
            and row['seed']==cfg['discovery_seed']+i*1009,'Exact descriptive calibration cases and disjoint seeds')
    upper=[]
    for i,(case,row) in enumerate(zip(grid,report['validation'],strict=True)):
        s.require(all(row[k]==v for k,v in case.items()) and row['paths']==cfg['validation_paths_per_case']
            and row['seed']==cfg['validation_seed']+i*1009,'Exact independent validation cases and counts')
        for d in ('lower','upper'):
            failures=row[d+'_errors'];s.require(type(failures) is int and 0<=failures<=row['paths'],'Finite simulated failure count')
            bound=_cp_bound(failures,row['paths'],cfg['validation_family_error']/(2*len(grid)),True)
            s.require(row[d+'_error_estimate']==failures/row['paths'] and row[d+'_error_simultaneous_upper']==bound,
                'Independent binomial error arithmetic')
            upper.append(bound)
    lower=[]
    s.require(len(report['power'])==4,'Four fixed alternative/equality power cases')
    for i,(case,row) in enumerate(zip(power_cases(),report['power'],strict=True)):
        s.require(all(row[k]==v for k,v in case.items()) and row['paths']==cfg['power_paths_per_case']
            and row['seed']==cfg['power_seed']+i*1009,'Exact power case and seed')
        successes=row['superior'];s.require(type(successes) is int and 0<=successes<=row['paths'],'Finite power success count')
        bound=_cp_bound(successes,row['paths'],cfg['power_family_error']/2,False)
        s.require(row['power']==successes/row['paths'] and row['power_simultaneous_lower']==bound
            and len(row['stop_histogram'])==9 and sum(row['stop_histogram'])==row['paths'],'Independent power/stop inventory')
        s.require(row['expected_games']==sum((j+1)*400*x for j,x in enumerate(row['stop_histogram']))/row['paths']
            and row['teacher_probability']==row['teacher_eligible']/row['paths']
            and row['superior']<=row['teacher_eligible']<=row['paths']
            and row['teacher_eligible']+row['material_harm']+row['unresolved']==row['paths'],'Independent outcome/expected-cost arithmetic')
        if case['power_acceptance']:lower.append(bound)
    a=report['acceptance']
    disjoint=not ({r['seed'] for r in report['discovery']}&{r['seed'] for r in report['validation']})
    s.require(a['calibration_validation_seeds_disjoint']==disjoint
        and report['implementation_agreement']==dict(passed=True,scheduled_looks_checked=1080,seed=202611142805)
        and a['production_scalar_vector_agreement'] is True,'Independent seed and scalar/vector qualification inventory')
    s.require(a['null_family_simultaneous_upper_max']==max(upper) and a['power_simultaneous_lower_min']==min(lower),
        'Exact worst-case qualification summaries')
    passed=max(upper)<=.025 and min(lower)>=.78 and a['calibration_validation_seeds_disjoint'] and a['production_scalar_vector_agreement']
    s.require(report['status']==('passed' if passed else 'failed'),'Failed qualification never relabelled passed')
    return dict(status='passed',qualification_passed=passed,models=0,new_games=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    s.require(args.label.isalnum(),'Exclusive simple qualification label');OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/(args.label+'.json');s.require(not path.exists(),'Preserve previous numerical qualification')
    evidence=OUT/(args.label+'_evidence');evidence.mkdir(exist_ok=False)
    snapshots=[]
    for source in (Path(__file__),Path(s.__file__),Path(__file__).with_name('tests_sequential.py')):
        original=_artifact(source);target=evidence/source.name
        with target.open('xb') as stream:stream.write(source.read_bytes());stream.flush();os.fsync(stream.fileno())
        snapshots.append(dict(original=original,snapshot=_artifact(target)))
    with (evidence/'plan.json').open('x',encoding='utf-8') as stream:
        stream.write(s.canonical(dict(plan=plan(),configuration=s.configuration(),source_snapshots=snapshots))+'\n')
    journals={stage:(evidence/(stage+'.jsonl')).open('x',encoding='utf-8',newline='\n')
        for stage in ('discovery','validation','power')}
    def progress(stage,row):
        f=journals[stage];f.write(s.canonical(row)+'\n');f.flush();os.fsync(f.fileno())
    try:
        from tools.future_training_rules_20261021 import verify_runtime_sources
        sources=verify_runtime_sources()  # Once in this actual statistical worker.
        result=qualify(progress=progress)
        result['corrected_source_receipt']=sources
        result['source_snapshots']=snapshots
        for item in snapshots:s.require(_artifact(ROOT/item['original']['path'])==item['original'],'Qualification sources unchanged')
    except BaseException:
        result=dict(schema=SCHEMA,status='failed',error=traceback.format_exc(),source_snapshots=snapshots,
            qualification_plan=plan(),configuration=s.configuration(),models=0,new_games=0)
    finally:
        for stream in journals.values():stream.close()
    result['journal_artifacts']=[_artifact(evidence/(stage+'.jsonl')) for stage in journals]
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(s.canonical(result)+'\n');stream.flush();os.fsync(stream.fileno())
    print(s.canonical(dict(path=path.relative_to(ROOT).as_posix(),status=result['status'],acceptance=result.get('acceptance'),
        seconds=result.get('elapsed_seconds'),null_cases=result.get('null_grid_cases'),error=result.get('error'))),flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

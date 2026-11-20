"""Game-paired calibration of a fallible same-network deep-search surrogate.

Only independently seeded reference searches grade candidate Main actions.
Unvisited actions never acquire invented Q values. These diagnostics do not
measure actual regret, calibrated win probabilities, Elo or network-training gains.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import statistics
from . import common as c

QUOTAS = (64, 128, 256, 512)
CONTROL_SEEDS = ('A', 'B')
REFERENCE_SEEDS = ('R', 'S')
REFERENCE_QUOTA = 2048
CONDITIONS = tuple((q,s) for q in QUOTAS for s in CONTROL_SEEDS) + tuple((2048,s) for s in REFERENCE_SEEDS)
DRAW_COUNT = 20000
BOOTSTRAP_SEED = 202611135120
FAMILY_CONFIDENCE = 1. - .05/3.
ROOT_COUNT, GAME_COUNT, ROOTS_PER_GAME = 512, 128, 4
MIN_ROOT_COVERAGE, MIN_GAME_COVERAGE = .90, 100
ADJACENT = ((64,128),(128,256),(256,512))
require = c.require


def _metadata(panel):
    roots = panel['roots']
    require(type(roots) is list and all(type(row) is dict for row in roots), 'Panel roots list')
    out = {}
    for row in roots:
        rid = row['root_id']
        require(type(rid) is str and rid and rid not in out, 'Unique exact root identities')
        host = panel.get('host_metadata', {}).get(rid, row)
        game, split = host['game_index'], host['split']
        require(type(game) is int and split in ('validation','offline'), 'Declared held-out game/split')
        out[rid] = dict(root_id=rid, game_index=game, split=split)
    return out


def validate_record(row, meta):
    require(type(row) is dict and row.get('root_id') == meta['root_id'], 'Exact record root')
    if 'game_index' in row:
        require(type(row['game_index']) is int and row['game_index']==meta['game_index'], 'Host game binding')
    if 'split' in row:
        require(row['split']==meta['split'], 'Host split binding')
    q, seed = row.get('quota'), row.get('seed_label')
    require(type(q) is int and type(seed) is str and (q,seed) in CONDITIONS, 'Fixed quota/seed condition')
    action = row.get('main_action_index')
    require(type(action) is int and 0 <= action < 67, 'Canonical Main-action index')
    visits, values = row.get('visits67'), row.get('q67')
    require(type(visits) is list and type(values) is list and len(visits)==len(values)==67,
            'Complete canonical root statistics')
    require(all(type(n) is int and n>=0 for n in visits) and sum(visits)==q, 'Actual fixed root visit quota')
    for n, value in zip(visits,values,strict=True):
        require((value is None) if n==0 else
                type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,
                'Unvisited Q is missing; visited Q is finite win credit')
    require(visits[action]>0 and values[action] is not None and visits[action]==max(visits),
            'Selected Main action attains maximum visits')
    if 'elapsed_seconds' in row:
        require(type(row['elapsed_seconds']) in (int,float) and math.isfinite(row['elapsed_seconds'])
                and row['elapsed_seconds']>=0, 'Finite actual response time')
    return q,seed


def _inventory(records, panel, *, strict=True, boundary=lambda:None):
    meta = _metadata(panel)
    by_game = defaultdict(list)
    for rid, row in meta.items():by_game[row['game_index']].append(rid)
    if strict:
        require(len(meta)==ROOT_COUNT and len(by_game)==GAME_COUNT and all(len(x)==ROOTS_PER_GAME for x in by_game.values()),
                'Fixed 512 roots, four per 128 games')
        require(set(by_game)==set(range(1024,1152)), 'Exact accepted held-out game inventory')
    splits={}
    for row in meta.values():
        require(row['game_index'] not in splits or splits[row['game_index']]==row['split'], 'Game belongs to one split')
        splits[row['game_index']]=row['split']
    if strict:
        require(sum(s=='validation' for s in splits.values())==64 and sum(s=='offline' for s in splits.values())==64,
                '64 validation and 64 reused test games')
        require(all(s==('validation' if g<1088 else 'offline') for g,s in splits.items()),
                'Exact accepted source split assignments')
    require(type(records) is list and len(records)==len(meta)*len(CONDITIONS), 'Complete fixed decision inventory')
    book={rid:{} for rid in meta}
    for i,row in enumerate(records):
        if i%64==0:boundary()
        require(row.get('root_id') in meta, 'No unplanned decision root')
        rid=row['root_id'];key=validate_record(row,meta[rid])
        require(key not in book[rid], 'No duplicate decision cell')
        book[rid][key]=row
    require(all(set(rows)==set(CONDITIONS) for rows in book.values()), 'Every root has all ten fixed conditions')
    return meta,book,splits


def _root_scores(meta,book,*,boundary=lambda:None):
    rows=[]
    for rid in sorted(meta):
        boundary();cell=book[rid]
        actions={str(q):[cell[q,s]['main_action_index'] for s in CONTROL_SEEDS] for q in QUOTAS}
        chosen=set(a for values in actions.values() for a in values)
        refs={s:cell[2048,s] for s in REFERENCE_SEEDS}
        missing={s:sorted(a for a in chosen if refs[s]['visits67'][a]==0) for s in REFERENCE_SEEDS}
        eligible=not any(missing.values())
        reference_q={s:{} for s in REFERENCE_SEEDS}
        q_scores={};gaps={};agreement={}
        anchor=statistics.fmean(ref['q67'][ref['main_action_index']] for ref in refs.values())
        for q in QUOTAS:
            key=str(q)
            agreement[key]=statistics.fmean(float(a==ref['main_action_index']) for a in actions[key] for ref in refs.values())
            if eligible:
                for s in REFERENCE_SEEDS:
                    reference_q[s][key]=statistics.fmean(refs[s]['q67'][a] for a in actions[key])
                q_scores[key]=statistics.fmean(reference_q[s][key] for s in REFERENCE_SEEDS)
                gaps[key]=anchor-q_scores[key]
            else:
                for s in REFERENCE_SEEDS:reference_q[s][key]=None
                q_scores[key]=gaps[key]=None
        visits=[refs[s]['visits67'][a] for s in REFERENCE_SEEDS for a in sorted(chosen)]
        rows.append(dict(**meta[rid],control_actions=actions,reference_actions={s:refs[s]['main_action_index'] for s in REFERENCE_SEEDS},
            joint_supported=eligible,missing_action_indices=missing,minimum_reference_visits=min(visits),
            reference_q=reference_q,q_scores=q_scores,reference_selected_q_gap=gaps,agreement=agreement,
            references_agree=refs['R']['main_action_index']==refs['S']['main_action_index']))
    return rows


def _games(rows, field, *, supported=False):
    values=defaultdict(list);splits={}
    for row in rows:
        if supported and not row['joint_supported']:continue
        splits[row['game_index']]=row['split'];values[row['game_index']].append(row[field])
    return [dict(game_index=g,split=splits[g],roots=len(v),
        values=[statistics.fmean(x[str(q)] for x in v) for q in QUOTAS]) for g,v in sorted(values.items())]


def _reference_games(rows,label):
    projected=[{**r,'values':r['reference_q'][label]} for r in rows if r['joint_supported']]
    return _games(projected,'values')


def _bootstrap(games, *, draws,seed,boundary=lambda:None):
    import numpy as np
    if not games:return None,{}
    require({r['split'] for r in games}=={'validation','offline'}, 'Both source strata represented')
    rng=np.random.Generator(np.random.PCG64(seed));result=np.zeros((draws,4),dtype=np.float64);metadata={}
    for split in ('validation','offline'):
        chosen=[r for r in games if r['split']==split];matrix=np.asarray([r['values'] for r in chosen],dtype=np.float64)
        count=len(chosen);indices=rng.integers(0,count,size=(draws,count),dtype=np.int64)
        for offset in range(0,draws,256):
            boundary();result[offset:offset+256]+=matrix[indices[offset:offset+256]].sum(axis=1)
        metadata[split]=dict(game_ids=[r['game_index'] for r in chosen],games=count,
            index_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
    result/=len(games)
    return result,metadata


def _ci(values,confidence=.95):
    import numpy as np
    alpha=(1-confidence)/2
    return np.quantile(values,[alpha,1-alpha],method='linear').tolist()


def _estimates(games,samples):
    if not games:return {str(q):dict(mean=None,ci95=None,games=0) for q in QUOTAS}
    means=[statistics.fmean(r['values'][i] for r in games) for i in range(4)]
    return {str(q):dict(mean=means[i],ci95=_ci(samples[:,i]),games=len(games)) for i,q in enumerate(QUOTAS)}


def _contrasts(games,samples):
    pairs=(*ADJACENT,(64,512));out=[]
    for low,high in pairs:
        i,j=QUOTAS.index(low),QUOTAS.index(high)
        out.append(dict(low=low,high=high,difference=None if not games else statistics.fmean(r['values'][j]-r['values'][i] for r in games),
            ci95=None if samples is None else _ci(samples[:,j]-samples[:,i]),
            family_ci=None if samples is None or (low,high) not in ADJACENT else _ci(samples[:,j]-samples[:,i],FAMILY_CONFIDENCE),
            family_confidence=FAMILY_CONFIDENCE if (low,high) in ADJACENT else None))
    return out


def classification(coverage,contrasts,reference_contrasts):
    """Predeclared evidence categories; uncertainty is never proof of invalidity."""
    if not coverage['sufficient']:
        return dict(status='unresolved_coverage',reason='The common supported subset is below the frozen coverage minimum.')
    adjacent=contrasts[:3];end=contrasts[3]
    contrary=[r for r in adjacent if r['family_ci'][1]<0]
    if contrary:
        return dict(status='contradicted',reason='At least one adjacent search-budget increase has an opposite-signed family-adjusted interval.',
            contradictory_pairs=[[r['low'],r['high']] for r in contrary])
    monotone=all(r['difference']>=0 for r in adjacent)
    references_positive=all(reference_contrasts[s][3]['difference']>0 for s in REFERENCE_SEEDS)
    broad=monotone and end['ci95'][0]>0 and references_positive
    if broad and all(r['family_ci'][0]>0 for r in adjacent):
        return dict(status='strong_positive_control_support',reason='All three adjacent contrasts are positive after family adjustment, with consistent broad reference directions.')
    if broad:
        return dict(status='provisional_broad_support',reason='Means are ordered and the broad contrast is positive; at least one adjacent contrast remains uncertain.')
    return dict(status='unresolved',reason='The fixed positive controls are not clearly contradicted, but the predeclared ordering requirements are not established.')


def _summarize(root_rows,records,*,draws,seed,boundary=lambda:None):
    import numpy as np
    supported=[r for r in root_rows if r['joint_supported']]
    all_games={r['game_index'] for r in root_rows};supported_games={r['game_index'] for r in supported}
    split_counts={s:len({r['game_index'] for r in supported if r['split']==s}) for s in ('validation','offline')}
    coverage=dict(planned_roots=len(root_rows),supported_roots=len(supported),root_fraction=len(supported)/len(root_rows),
        planned_games=len(all_games),represented_games=len(supported_games),missing_root_ids=[r['root_id'] for r in root_rows if not r['joint_supported']],
        missing_game_ids=sorted(all_games-supported_games),represented_by_split=split_counts,
        minimum_root_fraction=MIN_ROOT_COVERAGE,minimum_games=MIN_GAME_COVERAGE,
        sufficient=len(supported)/len(root_rows)>=MIN_ROOT_COVERAGE and len(supported_games)>=MIN_GAME_COVERAGE and all(split_counts.values()))
    primary_games=_games(root_rows,'q_scores',supported=True)
    usable=bool(primary_games) and all(split_counts.values())
    samples,strata=_bootstrap(primary_games,draws=draws,seed=seed,boundary=boundary) if usable else (None,{})
    if not usable:primary_games=[]
    agreement_games=_games(root_rows,'agreement')
    agreement_samples,agreement_strata=_bootstrap(agreement_games,draws=draws,seed=seed,boundary=boundary)
    gap_games=_games(root_rows,'reference_selected_q_gap',supported=True) if usable else []
    gap_samples,_=_bootstrap(gap_games,draws=draws,seed=seed,boundary=boundary) if usable else (None,{})
    reference_contrasts={};reference_estimates={}
    for label in REFERENCE_SEEDS:
        games=_reference_games(root_rows,label) if usable else []
        sample,_=_bootstrap(games,draws=draws,seed=seed,boundary=boundary) if usable else (None,{})
        reference_estimates[label]=_estimates(games,sample);reference_contrasts[label]=_contrasts(games,sample)
    contrasts=_contrasts(primary_games,samples)
    missing_by_reference={s:sum(bool(r['missing_action_indices'][s]) for r in root_rows) for s in REFERENCE_SEEDS}
    timing={}
    for q in (*QUOTAS,REFERENCE_QUOTA):
        values=[r['elapsed_seconds'] for r in records if r['quota']==q and 'elapsed_seconds' in r]
        timing[str(q)]=dict(requests=len(values),mean=statistics.fmean(values) if values else None,
            median=statistics.median(values) if values else None,p95=float(np.quantile(values,.95)) if values else None,
            maximum=max(values) if values else None,over_one_second=sum(value>1. for value in values))
    return dict(schema='proxy_calibration_analysis_v1',status='fixed_positive_controls_analyzed',
        roots=root_rows,coverage=coverage,primary_estimates=_estimates(primary_games,samples),
        primary_contrasts=contrasts,reference_estimates=reference_estimates,reference_contrasts=reference_contrasts,
        reference_selected_q_gap=_estimates(gap_games,gap_samples),agreement_estimates=_estimates(agreement_games,agreement_samples),
        agreement_contrasts=_contrasts(agreement_games,agreement_samples),
        reference_stability=dict(agree_roots=sum(r['references_agree'] for r in root_rows),roots=len(root_rows)),
        support=dict(missing_roots_by_reference=missing_by_reference,
            supported_minimum_visits={str(n):sum(r['minimum_reference_visits']>=n for r in root_rows) for n in (1,2,4,8,16)}),
        game_summaries=dict(primary=primary_games,agreement=agreement_games),timing=timing,
        calibration=classification(coverage,contrasts,reference_contrasts),
        bootstrap=dict(draws=draws,seed=seed,generator='numpy.PCG64',unit='whole_game',
            primary_strata=strata,agreement_strata=agreement_strata,adjacent_family_confidence=FAMILY_CONFIDENCE,
            intervals='paired_percentile_descriptive_with_three_adjacent_Bonferroni_intervals'),
        limitations=[
            'The 128 held-out games have been inspected before and form reused development evidence.',
            'Both deep references use EF8A. Separate random seeds do not create an independent model lineage or a ground-truth oracle.',
            'Primary Q scores condition on the shared supported-root subset; missing actions are never imputed as zero.',
            'Few root-edge visits can leave noisy Q estimates even when a comparison is supported.',
            'The proxy grades Main actions and does not independently grade every auxiliary compound-turn choice.',
            'Positive-control ordering does not establish validity for changed network weights or an exchange rate to score or Elo.',
            'No new playing-strength, human-strength or superhuman claim, training decision or model promotion follows automatically.'],
        reference_promoted=False,followup_launched=False,playing_strength_established=False,new_games=0)


def analyze(records,panel,*,draws=DRAW_COUNT,seed=BOOTSTRAP_SEED,boundary=lambda:None):
    require(type(draws) is int and draws==DRAW_COUNT and type(seed) is int and seed==BOOTSTRAP_SEED,
            'Frozen production uncertainty settings')
    meta,book,_=_inventory(records,panel,boundary=boundary)
    return _summarize(_root_scores(meta,book,boundary=boundary),records,draws=draws,seed=seed,boundary=boundary)


def _close(actual,expected,message):
    if expected is None:require(actual is None,message);return
    if isinstance(expected,(int,bool,str)):
        require(type(actual) is type(expected) and actual==expected,message);return
    if isinstance(expected,float):
        require(type(actual) in (float,int) and math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-12),message);return
    if isinstance(expected,list):
        require(type(actual) is list and len(actual)==len(expected),message)
        for a,b in zip(actual,expected,strict=True):_close(a,b,message)
        return
    require(type(expected) is dict and type(actual) is dict and set(actual)==set(expected),message)
    for key in expected:_close(actual[key],expected[key],message+' '+key)


def verify_arithmetic(records,panel,reported,*,boundary=lambda:None):
    """Independent scalar/root reductions and multinomial-count bootstrap reader."""
    import numpy as np
    require(reported['schema']=='proxy_calibration_analysis_v1' and reported['status']=='fixed_positive_controls_analyzed',
            'Declared analysis schema')
    settings=reported['bootstrap']
    require(settings['draws']==DRAW_COUNT and type(settings['draws']) is int and settings['seed']==BOOTSTRAP_SEED
            and type(settings['seed']) is int and settings['generator']=='numpy.PCG64'
            and settings['unit']=='whole_game' and settings['adjacent_family_confidence']==FAMILY_CONFIDENCE,
            'Independent frozen uncertainty specification')
    meta,book,splits=_inventory(records,panel,boundary=boundary)
    require(len(reported['roots'])==len(meta) and [r['root_id'] for r in reported['roots']]==sorted(meta), 'Complete ordered roots')
    matrices={};independent_rows=[]
    for rid in sorted(meta):
        boundary();cell=book[rid]
        actions=np.asarray([[cell[q,s]['main_action_index'] for s in CONTROL_SEEDS] for q in QUOTAS],dtype=np.int64)
        reference_values=[];reference_visits=[];reference_actions=[];missing={}
        for label in REFERENCE_SEEDS:
            r=cell[2048,label];reference_values.append([np.nan if v is None else v for v in r['q67']]);reference_visits.append(r['visits67'])
            reference_actions.append(r['main_action_index'])
            missing[label]=sorted(set(int(a) for a in actions.flat if r['visits67'][a]==0))
        values=np.asarray(reference_values);visits=np.asarray(reference_visits)
        selected=values[:,actions];support=bool(np.all(visits[:,actions]>0))
        agreement=(actions[None,:,:]==np.asarray(reference_actions)[:,None,None]).mean(axis=(0,2))
        primary=selected.mean(axis=(0,2)) if support else np.full(4,np.nan)
        refmeans=selected.mean(axis=2) if support else np.full((2,4),np.nan)
        anchor=float(np.mean(values[np.arange(2),reference_actions]))
        row=dict(**meta[rid],control_actions={str(q):actions[i].tolist() for i,q in enumerate(QUOTAS)},
            reference_actions=dict(zip(REFERENCE_SEEDS,reference_actions,strict=True)),joint_supported=support,
            missing_action_indices=missing,minimum_reference_visits=int(visits[:,np.unique(actions)].min()),
            reference_q={label:{str(q):float(refmeans[r,i]) if support else None for i,q in enumerate(QUOTAS)} for r,label in enumerate(REFERENCE_SEEDS)},
            q_scores={str(q):float(primary[i]) if support else None for i,q in enumerate(QUOTAS)},
            reference_selected_q_gap={str(q):anchor-float(primary[i]) if support else None for i,q in enumerate(QUOTAS)},
            agreement={str(q):float(agreement[i]) for i,q in enumerate(QUOTAS)},references_agree=reference_actions[0]==reference_actions[1])
        independent_rows.append(row)
        matrices[rid]=dict(primary=primary,agreement=agreement,R=refmeans[0],S=refmeans[1],gap=anchor-primary,supported=support)
    _close(reported['roots'],independent_rows,'Independent root arithmetic')
    eligible=[r for r in independent_rows if r['joint_supported']]
    covered_games={r['game_index'] for r in eligible}
    sufficient=(len(eligible)/len(meta)>=MIN_ROOT_COVERAGE and len(covered_games)>=MIN_GAME_COVERAGE and
        all(any(r['split']==s for r in eligible) for s in ('validation','offline')))
    expected_coverage=dict(planned_roots=len(meta),supported_roots=len(eligible),root_fraction=len(eligible)/len(meta),
        planned_games=len(splits),represented_games=len(covered_games),missing_root_ids=[r['root_id'] for r in independent_rows if not r['joint_supported']],
        missing_game_ids=sorted(set(splits)-covered_games),represented_by_split={s:len({r['game_index'] for r in eligible if r['split']==s}) for s in ('validation','offline')},
        minimum_root_fraction=MIN_ROOT_COVERAGE,minimum_games=MIN_GAME_COVERAGE,sufficient=sufficient)
    _close(reported['coverage'],expected_coverage,'Independent support coverage')
    _close(reported['reference_stability'],dict(agree_roots=sum(r['references_agree'] for r in independent_rows),
        roots=len(meta)), 'Independent reference action stability')
    _close(reported['support'],dict(
        missing_roots_by_reference={s:sum(len(r['missing_action_indices'][s])>0 for r in independent_rows)
                                    for s in REFERENCE_SEEDS},
        supported_minimum_visits={str(n):sum(r['minimum_reference_visits']>=n for r in independent_rows)
                                 for n in (1,2,4,8,16)}), 'Independent root-edge support')
    timing={}
    for quota in (*QUOTAS,REFERENCE_QUOTA):
        values=np.asarray([r['elapsed_seconds'] for r in records if r['quota']==quota and 'elapsed_seconds' in r],dtype=np.float64)
        timing[str(quota)]=dict(requests=len(values),mean=float(values.mean()) if len(values) else None,
            median=float(np.median(values)) if len(values) else None,p95=float(np.percentile(values,95)) if len(values) else None,
            maximum=float(values.max()) if len(values) else None,over_one_second=int(np.count_nonzero(values>1.)))
    _close(reported['timing'],timing,'Independent actual timings')

    def independently_reduce(field,all_roots=False):
        grouped=defaultdict(list)
        for rid in sorted(meta):
            if all_roots or matrices[rid]['supported']:grouped[meta[rid]['game_index']].append(matrices[rid][field])
        if not grouped or {splits[g] for g in grouped} != {'validation','offline'}:return [],None,{}
        games=[dict(game_index=g,split=splits[g],roots=len(v),values=np.asarray(v).mean(axis=0).tolist()) for g,v in sorted(grouped.items())]
        rng=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED));sample=np.zeros((DRAW_COUNT,4),dtype=np.float64);strata={}
        for split in ('validation','offline'):
            subset=[r for r in games if r['split']==split];matrix=np.asarray([r['values'] for r in subset]);n=len(subset)
            indices=rng.integers(0,n,size=(DRAW_COUNT,n),dtype=np.int64)
            counts=np.zeros((DRAW_COUNT,n),dtype=np.int64)
            for offset in range(0,DRAW_COUNT,256):
                boundary()
                for j in range(offset,min(offset+256,DRAW_COUNT)):counts[j]=np.bincount(indices[j],minlength=n)
            sample+=counts@matrix
            strata[split]=dict(game_ids=[r['game_index'] for r in subset],games=n,
                index_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
        sample/=len(games)
        return games,sample,strata

    def independently_check(games,samples,estimates,contrasts=None):
        matrix=np.asarray([r['values'] for r in games]);means=None if not games else matrix.mean(axis=0)
        for i,q in enumerate(QUOTAS):
            expected=dict(mean=None if means is None else float(means[i]),ci95=None if samples is None else
                np.percentile(samples[:,i],[2.5,97.5],method='linear').tolist(),games=len(games))
            _close(estimates[str(q)],expected,'Independent estimate '+str(q))
        if contrasts is not None:
            require(len(contrasts)==4,'Fixed contrast inventory')
            for row,(low,high) in zip(contrasts,(*ADJACENT,(64,512)),strict=True):
                i,j=QUOTAS.index(low),QUOTAS.index(high);adjacent=(low,high) in ADJACENT
                lo=(1-FAMILY_CONFIDENCE)*50
                expected=dict(low=low,high=high,difference=None if means is None else float(np.mean(matrix[:,j]-matrix[:,i])),
                    ci95=None if samples is None else np.percentile(samples[:,j]-samples[:,i],[2.5,97.5],method='linear').tolist(),
                    family_ci=None if samples is None or not adjacent else np.percentile(samples[:,j]-samples[:,i],[lo,100-lo],method='linear').tolist(),
                    family_confidence=FAMILY_CONFIDENCE if adjacent else None)
                _close(row,expected,'Independent paired contrast')

    for field,all_roots,estkey,contrastkey,stratakey in (
        ('primary',False,'primary_estimates','primary_contrasts','primary_strata'),
        ('agreement',True,'agreement_estimates','agreement_contrasts','agreement_strata'),
        ('gap',False,'reference_selected_q_gap',None,None)):
        games,sample,strata=independently_reduce(field,all_roots)
        independently_check(games,sample,reported[estkey],reported[contrastkey] if contrastkey else None)
        if stratakey:_close(settings[stratakey],strata,'Independent bootstrap draw identities')
        if field in ('primary','agreement'):_close(reported['game_summaries'][field],games,'Independent equal-game means')
    for label in REFERENCE_SEEDS:
        games,sample,_=independently_reduce(label)
        independently_check(games,sample,reported['reference_estimates'][label],reported['reference_contrasts'][label])
    # Independently reconstruct the fixed interpretation without calling its producer.
    adjacent=reported['primary_contrasts'][:3];end=reported['primary_contrasts'][3]
    if not sufficient:category='unresolved_coverage'
    elif any(r['family_ci'][1]<0 for r in adjacent):category='contradicted'
    elif all(r['difference']>=0 for r in adjacent) and end['ci95'][0]>0 and all(reported['reference_contrasts'][s][3]['difference']>0 for s in REFERENCE_SEEDS):
        category='strong_positive_control_support' if all(r['family_ci'][0]>0 for r in adjacent) else 'provisional_broad_support'
    else:category='unresolved'
    require(reported['calibration']['status']==category,'Independent fixed classification')
    require(reported['reference_promoted'] is False and reported['followup_launched'] is False and
            reported['playing_strength_established'] is False and reported['new_games']==0,'No unearned strength or followup claim')
    return dict(status='independent_proxy_arithmetic_verified',roots=len(meta),decisions=len(records),
        games=len(splits),models=0,new_games=0,independent_q_and_agreement_reduction=True,
        independent_count_weighted_bootstrap=True,independent_coverage_and_classification=True,
        draws=DRAW_COUNT,seed=BOOTSTRAP_SEED)


def render_charts(result,folder,historical_controls=None,*,boundary=lambda:None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    folder=Path(folder)
    require(folder.resolve().is_relative_to(c.OUT),'New chart output only')
    folder.mkdir(parents=True,exist_ok=True)
    colors=('#0f766e','#c2410c');x=np.arange(4);artifacts=[]
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.8))
    primary=result['primary_contrasts'];est=result['primary_estimates']
    if est['64']['mean'] is not None:
        means=[100*est[str(q)]['mean'] for q in QUOTAS]
        lower=[100*est[str(q)]['ci95'][0] for q in QUOTAS]
        upper=[100*est[str(q)]['ci95'][1] for q in QUOTAS]
        axes[0].plot(x,means,'o-',color=colors[0],lw=2.5)
        axes[0].fill_between(x,lower,upper,color=colors[0],alpha=.15)
    else:axes[0].text(.5,.5,'Insufficient shared Q support',transform=axes[0].transAxes,ha='center')
    agreement=result['agreement_estimates'];mean=[100*agreement[str(q)]['mean'] for q in QUOTAS]
    lo=[100*agreement[str(q)]['ci95'][0] for q in QUOTAS];hi=[100*agreement[str(q)]['ci95'][1] for q in QUOTAS]
    axes[1].plot(x,mean,'o-',color=colors[1],lw=2.5);axes[1].fill_between(x,lo,hi,color=colors[1],alpha=.15)
    for ax in axes:
        ax.set_xticks(x,QUOTAS);ax.set_xlabel('EF8A simulations per decision');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Deep-reference credit of selected actions',fontsize=11)
    axes[0].set_ylabel('Reference credit points (higher is better)')
    axes[1].set_title('Agreement with independent deep searches',fontsize=11);axes[1].set_ylabel('Matching Main actions (%)')
    fig.suptitle('Can the cheaper diagnostic recognize known search improvements?',fontsize=14)
    fig.text(.5,.025,f"Q uses {result['coverage']['supported_roots']}/512 jointly supported roots; agreement uses all 512. Two control seeds, two 2048-simulation references.\n"
        'Same-network surrogate, not measured win probability. Nominal 95% intervals; paired contrasts appear in the companion chart.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.12,1,.93));path=folder/'proxy_budget_calibration.png';require(not path.exists(),'Exclusive chart')
    fig.savefig(path,dpi=160);plt.close(fig);artifacts.append(c.artifact(path));boundary()
    if historical_controls is not None:
        if type(historical_controls) is dict:
            historical_controls=historical_controls['comparisons']
        require(type(historical_controls) is list and len(historical_controls)==3,'Three historical adjacent controls')
        fig,axes=plt.subplots(1,2,figsize=(11.2,4.8));labels=[]
        for j,(record,pair) in enumerate(zip(historical_controls,ADJACENT,strict=True)):
            require((record['reference_simulations'],record['candidate_simulations'])==pair and 0<=record['score']<=1 and
                len(record['paired_ci95'])==2 and 0<=record['paired_ci95'][0]<=record['paired_ci95'][1]<=1,'Exact historical score context')
            labels.append(f'{pair[0]} to {pair[1]}')
            axes[0].plot([100*v for v in record['paired_ci95']],[j,j],color=colors[0],lw=2)
            axes[0].plot(100*record['score'],j,'o',color=colors[0])
            row=primary[j]
            if row['difference'] is not None:
                axes[1].plot([100*v for v in row['family_ci']],[j,j],color=colors[1],lw=2)
                axes[1].plot(100*row['difference'],j,'o',color=colors[1])
        for ax in axes:
            ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        if all(row['difference'] is None for row in primary):
            axes[1].text(.5,.5,'No supported primary contrasts',transform=axes[1].transAxes,ha='center')
        axes[0].axvline(50,color='#64748b',ls='--');axes[1].axvline(0,color='#64748b',ls='--')
        axes[0].set_title('Previously verified live comparisons');axes[0].set_xlabel('Higher-budget player score (%) | 95% intervals')
        axes[1].set_title('New saved-position Q contrasts');axes[1].set_xlabel('Difference in Q credit points | 98.33% intervals')
        fig.suptitle('Positive controls: agreement of direction, not an exchange rate',fontsize=14)
        fig.text(.5,.025,'Historical live games and this reused saved-position panel are different evidence.\n'
            'These plots do not convert proxy differences into win rates or Elo, and do not validate changed network weights.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.13,1,.93));path=folder/'proxy_historical_controls.png';require(not path.exists(),'Exclusive chart')
        fig.savefig(path,dpi=160);plt.close(fig);artifacts.append(c.artifact(path));boundary()
    return artifacts

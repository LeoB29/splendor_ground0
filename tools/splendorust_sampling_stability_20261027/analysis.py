"""Descriptive sampling stability on fixed development roots, never a strength oracle."""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
from tools.splendorust_decision_quality_20261027_v2.analysis import _measure, distribution
from . import common as c
from .design import LABELS, POOLS


def stability(actions):
    values=[c.canonical(a) for a in actions]
    pairs=list(combinations(values,2))
    unequal=sum(a!=b for a,b in pairs)
    return dict(n=len(values),pair_count=len(pairs),disagreeing_pairs=unequal,
        disagreement_rate=unequal/len(pairs) if pairs else None,
        unique_actions=len(set(values)),unanimous=len(set(values))==1 if values else None)


def paired_stability(pool3_actions,pool6_actions):
    left,right=stability(pool3_actions),stability(pool6_actions)
    valid=len(pool3_actions)==len(pool6_actions)==6
    delta=right['disagreement_rate']-left['disagreement_rate'] if valid else None
    return dict(pool3=left,pool6=right,complete_paired_root=valid,
        pool6_minus_pool3=delta,
        direction=('lower' if delta<0 else 'higher' if delta>0 else 'equal') if valid else None)


def _aggregate(entries,field):
    full=[e[field] for e in entries if e[field]['complete_paired_root']]
    return dict(planned_roots=len(entries),complete_paired_roots=len(full),
        mean_pool6_minus_pool3=sum(x['pool6_minus_pool3'] for x in full)/len(full) if full else None,
        roots_lower=sum(x['direction']=='lower' for x in full),
        roots_equal=sum(x['direction']=='equal' for x in full),
        roots_higher=sum(x['direction']=='higher' for x in full),
        pool3_mean_disagreement=sum(x['pool3']['disagreement_rate'] for x in full)/len(full) if full else None,
        pool6_mean_disagreement=sum(x['pool6']['disagreement_rate'] for x in full)/len(full) if full else None,
        pool3_all_six_agree=sum(x['pool3']['unanimous'] for x in full),
        pool6_all_six_agree=sum(x['pool6']['unanimous'] for x in full),
        unit='root; the15 within-root pairs are not independent observations')


def analyze(records,panel,matrix,*,complete):
    values=defaultdict(dict);baseline=set();failures=[];times=defaultdict(list);calls=defaultdict(list)
    for row in records:
        spec=row['spec']
        if not row['completed']:
            failures.append(dict(sequence=spec['sequence'],root_id=spec['root_id'],status=row['clock']['status']))
            continue
        if spec['stage']=='frozen_reference_reproduction':baseline.add(spec['root_id'])
        elif spec['stage'] in ('research_pilot','research_main'):
            key=(spec['world_pool'],spec['seed_family']);rid=spec['root_id']
            c.require(key not in values[rid],'Duplicate sampling condition')
            measured=_measure(row);values[rid][key]=measured
            label='pool'+str(key[0]);times[label].append(measured['complete_response_seconds'])
            calls[label].append(measured['callback_count'])
    roots=[]
    for descriptor in panel['roots']:
        rid=descriptor['root_id'];available=values[rid]
        main={p:[available[p,s]['main_action'] for s in LABELS if (p,s) in available] for p in POOLS}
        full={p:[available[p,s]['complete_action'] for s in LABELS if (p,s) in available] for p in POOLS}
        paired_seeds=[s for s in LABELS if (3,s) in available and (6,s) in available]
        roots.append(dict(root_id=rid,cell=str(descriptor['phase'])+'_seat'+str(descriptor['seat']),
            phase=descriptor['phase'],seat=descriptor['seat'],ply=descriptor['ply'],
            original64_reproduced=rid in baseline,
            conditions={f'pool{p}-{s}':available[p,s] for p in POOLS for s in LABELS if (p,s) in available},
            missing_conditions=[f'pool{p}-{s}' for p in POOLS for s in LABELS if (p,s) not in available],
            main_stability=paired_stability(main[3],main[6]),
            complete_action_stability=paired_stability(full[3],full[6]),
            matched_seed_changes=dict(available=len(paired_seeds),
                main=sum(c.canonical(available[3,s]['main_action'])!=c.canonical(available[6,s]['main_action']) for s in paired_seeds),
                complete_action=sum(c.canonical(available[3,s]['complete_action'])!=c.canonical(available[6,s]['complete_action']) for s in paired_seeds))))
    cells={cell:dict(main=_aggregate([r for r in roots if r['cell']==cell],'main_stability'),
        complete_action=_aggregate([r for r in roots if r['cell']==cell],'complete_action_stability'))
        for cell in sorted(set(r['cell'] for r in roots))}
    return dict(schema='splendorust_sampling_stability_descriptive_v1',complete=bool(complete),
        planned_roots=24,complete_research_roots=sum(not r['missing_conditions'] for r in roots),
        first_failure=failures[0] if failures else None,
        primary_main_stability=_aggregate(roots,'main_stability'),
        secondary_complete_action_stability=_aggregate(roots,'complete_action_stability'),
        matched_seed_changes=dict(available=sum(r['matched_seed_changes']['available'] for r in roots),
            main=sum(r['matched_seed_changes']['main'] for r in roots),
            complete_action=sum(r['matched_seed_changes']['complete_action'] for r in roots)),
        cell_results=cells,timing={k:distribution(v) for k,v in times.items()},
        callbacks={k:distribution(v) for k,v in calls.items()},roots=roots,
        automatic_promotion=False,strength_claim=False,
        next_study_recommendation=('Jointly interpret stability, timing and changed actions; any candidate needs fresh paired one-second games.'
            if complete else 'Resolve the preserved operational or evidence failure; incomplete coverage cannot settle this diagnostic.'),
        caveats=['Fixed24 historical development positions, not independent strength confirmation.',
            'Six seeds yield15 correlated comparisons per root;24 roots are the descriptive reporting units.',
            'Pool6 changes root-world generation and later shared-RNG neural-input hidden completions.',
            'More stable choices are not necessarily more accurate or stronger.',
            'Q values and initial/post-search values are retained internal estimates, not factual correctness.',
            'No root/seed selection, tuning, population inference, training or automatic player promotion.'])

"""Descriptive teacher and update signals from accepted saved predictions only."""
import math
import struct
from collections import defaultdict
from pathlib import Path
from .common import Context, ROOT, require

ROW_FIELDS = {'game_index','ply','split','root_input_sha256','legal_mask81','policy_target81','value_target2','output83'}
SPLITS = {'validation': (range(192,224),1802,57), 'offline': (range(224,256),1755,55)}
WEIGHTS = {'parent':'32cdd9466befc527d265a356be65200c3666a4fa8a82cfefdc40b327b2f529db',
           'candidate':'5eda6dfd8b137f69f378e449d3a75d5e328cb5303a27e66b154af615242cfd7d'}


def validate_row(row):
    require(type(row) is dict and set(row)==ROW_FIELDS,'Complete saved prediction row schema')
    require(type(row['game_index']) is int and type(row['ply']) is int and 0<=row['ply']<2000,
            'Game and ply identity')
    require(row['split'] in SPLITS and row['game_index'] in SPLITS[row['split']][0],'Heldout split identity')
    digest=row['root_input_sha256']
    require(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),'Input hash')
    mask,target,output,value=(row[k] for k in ('legal_mask81','policy_target81','output83','value_target2'))
    require(type(mask) is list and len(mask)==81 and all(type(x) is bool for x in mask) and any(mask),'Full legal81 mask')
    require(type(target) is list and len(target)==81 and
        all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in target) and
        abs(math.fsum(target)-1.)<=1e-12 and all(ok or t==0 for ok,t in zip(mask,target)),'Normalized legal teacher target')
    require(type(output) is list and len(output)==83 and
        all(type(x) in (int,float) and math.isfinite(x) for x in output),'Complete finite83 outputs')
    try:
        exact=all(struct.unpack('<f',struct.pack('<f',x))[0]==x for x in output)
    except (OverflowError,struct.error):exact=False
    require(exact and all(-1<=x<=1 for x in output[81:]),'Unchanged float32 outputs and signed value range')
    require(type(value) is list and len(value)==2 and all(type(x) in (int,float) and x in (-1.,0.,1.) for x in value)
        and math.fsum(value)==0.,'Two official signed outcome targets')


def masked_distribution(logits,mask):
    require(len(logits)==len(mask)==81 and any(mask),'Full nonempty masked distribution')
    legal=[i for i,ok in enumerate(mask) if ok]
    maximum=max(logits[i] for i in legal)
    normalizer=maximum+math.log(math.fsum(math.exp(logits[i]-maximum) for i in legal))
    logp=[logits[i]-normalizer if mask[i] else None for i in range(81)]
    probability=[math.exp(logp[i]) if mask[i] else 0. for i in range(81)]
    require(abs(math.fsum(probability)-1.)<=1e-12,'Stable legal softmax normalization')
    return probability,logp


def row_metrics(parent,candidate):
    validate_row(parent);validate_row(candidate)
    require({k:v for k,v in parent.items() if k!='output83'}==
            {k:v for k,v in candidate.items() if k!='output83'},'Exact same input, mask and targets for both predictions')
    mask=parent['legal_mask81'];target=parent['policy_target81']
    legal=[i for i,ok in enumerate(mask) if ok];visited=[i for i,x in enumerate(target) if x>0]
    top_mass=max(target);teacher_top={i for i in visited if target[i]==top_mass}
    entropy=-math.fsum(target[i]*math.log(target[i]) for i in visited)
    metrics=dict(legal_support_count=float(len(legal)),visited_support_count=float(len(visited)),
        visited_fraction_of_legal=len(visited)/len(legal),teacher_entropy_nats=entropy,
        teacher_effective_support=math.exp(entropy),teacher_inverse_simpson_support=1/math.fsum(x*x for x in target),
        teacher_entropy_fraction_of_log_legal=entropy/math.log(len(legal)) if len(legal)>1 else 0.,
        teacher_top_mass=top_mass,teacher_top_tie_count=float(len(teacher_top)),
        teacher_single_visited_action=float(len(visited)==1))
    distributions={};top_choices={}
    for name,row in (('parent',parent),('candidate',candidate)):
        output=row['output83'];p,logp=masked_distribution(output[:81],mask);distributions[name]=(p,logp)
        maximum=max(output[i] for i in legal);top_set={i for i in legal if output[i]==maximum}
        chosen=min(top_set);top_choices[name]=chosen
        ce=-math.fsum(target[i]*logp[i] for i in visited)
        kl=math.fsum(target[i]*(math.log(target[i])-logp[i]) for i in visited)
        require(kl>=-1e-12 and abs((ce-entropy)-kl)<=1e-12,'Teacher cross-entropy minus entropy equals KL')
        metrics.update({name+'_policy_crossentropy_nats':ce,name+'_teacher_to_policy_kl_nats':max(0.,kl),
            name+'_policy_entropy_nats':-math.fsum(p[i]*logp[i] for i in legal if p[i]),
            name+'_teacher_mode_agreement':float(chosen in teacher_top),
            name+'_teacher_model_top_set_overlap':float(bool(top_set & teacher_top)),
            name+'_policy_top_tie_count':float(len(top_set)),
            name+'_teacher_top_set_probability':math.fsum(p[i] for i in teacher_top),
            name+'_teacher_visited_support_probability':math.fsum(p[i] for i in visited),
            name+'_teacher_expected_probability':math.fsum(target[i]*p[i] for i in visited),
            name+'_teacher_mass_on_policy_argmax':target[chosen]})
    pp,lp=distributions['parent'];cp,lc=distributions['candidate']
    forward=math.fsum(pp[i]*(lp[i]-lc[i]) for i in legal if pp[i])
    reverse=math.fsum(cp[i]*(lc[i]-lp[i]) for i in legal if cp[i])
    require(forward>=-1e-12 and reverse>=-1e-12,'Nonnegative same-support policy divergence')
    metrics.update(policy_argmax_changed=float(top_choices['parent']!=top_choices['candidate']),
        parent_to_candidate_kl_nats=max(0.,forward),candidate_to_parent_kl_nats=max(0.,reverse),
        policy_total_variation=.5*math.fsum(abs(cp[i]-pp[i]) for i in legal),
        policy_max_absolute_probability_shift=max(abs(cp[i]-pp[i]) for i in legal))
    for metric in ('policy_crossentropy_nats','teacher_to_policy_kl_nats','teacher_mode_agreement',
                   'teacher_top_set_probability','teacher_visited_support_probability','teacher_expected_probability',
                   'teacher_mass_on_policy_argmax'):
        metrics['candidate_minus_parent_'+metric]=metrics['candidate_'+metric]-metrics['parent_'+metric]
    shifts=[candidate['output83'][81+i]-parent['output83'][81+i] for i in range(2)]
    for i,shift in enumerate(shifts):
        metrics[f'value_slot{i}_signed_shift']=shift
        metrics[f'value_slot{i}_absolute_shift']=abs(shift)
    metrics['value_mean_absolute_shift']=math.fsum(abs(x) for x in shifts)/2
    metrics['value_root_rms_shift']=math.sqrt(math.fsum(x*x for x in shifts)/2)
    require(all(math.isfinite(v) for v in metrics.values()),'All descriptive metrics finite')
    return metrics


def aggregate_games(rows):
    """Each row is (game_index, metrics); first average roots within each game."""
    by_game=defaultdict(list)
    for game,metrics in rows:by_game[game].append(metrics)
    require(by_game and all(by_game.values()),'Nonempty whole-game aggregate')
    fields=set(next(iter(by_game.values()))[0])
    require(all(set(m)==fields for entries in by_game.values() for m in entries),'Same complete metrics for every root')
    games=[]
    for game,entries in sorted(by_game.items()):
        games.append(dict(game_index=game,roots=len(entries),metrics={
            k:math.fsum(m[k] for m in entries)/len(entries) for k in sorted(fields)}))
    return dict(game_count=len(games),root_count=sum(g['roots'] for g in games),games=games,
        mean_game_metrics={k:math.fsum(g['metrics'][k] for g in games)/len(games) for k in sorted(fields)})


def read_predictions(ctx):
    folder=ctx.source/'fit/heldout_diagnostics';summary=ctx.read(folder/'result.json')
    require(summary['schema']=='selfplay_heldout_diagnostics_v1' and
        summary['status']=='fixed_checkpoint_diagnostics_complete' and summary['all_eligible_roots'] is True and
        summary['checkpoint_selection'] is False and summary['forward_calls']==224,'Accepted complete heldout provenance')
    checkpoint=summary['candidate_manifest']
    require(ctx.bind(ROOT/checkpoint['path'])==checkpoint,'Exact fixed candidate manifest')
    descriptors={d['path']:d for d in summary['batches']}
    require(len(descriptors)==len(summary['batches'])==224,'Unique complete224-batch inventory')
    predictions={};seen_files=set()
    for name in ('parent','candidate'):
        for split,(expected_games,root_count,batch_count) in SPLITS.items():
            files=sorted((folder/name/split).glob('*.json'))
            require([p.name for p in files]==[f'{i:06d}.json' for i in range(batch_count)],'Exact saved batch filenames')
            rows=[];by_key={}
            for i,path in enumerate(files):
                ctx.check();item=ctx.bind(path)
                require(descriptors.get(item['path'])==item,'Batch hash bound to accepted diagnostic result')
                seen_files.add(item['path']);document=ctx.read(path)
                require(set(document)=={'batch_index','candidate_manifest_fixed','model','rows','split','weights_sha256'} and
                    document['batch_index']==i and document['model']==name and document['split']==split and
                    document['candidate_manifest_fixed']==checkpoint and document['weights_sha256']==WEIGHTS[name],
                    'Exact batch/checkpoint/weight/split identity')
                expected_size=min(32,root_count-32*i)
                require(len(document['rows'])==expected_size,'All recorded rows in every exact batch')
                for row in document['rows']:
                    validate_row(row);key=(row['game_index'],row['ply'])
                    require(row['split']==split and key not in by_key,'No duplicate or cross-split heldout root')
                    by_key[key]=row;rows.append(row)
            require(len(rows)==root_count and {r['game_index'] for r in rows}==set(expected_games),
                    'Every eligible root from each of the32 fixed games')
            require([(r['game_index'],r['ply']) for r in rows]==sorted(by_key),'Original game/ply order preserved')
            predictions[(name,split)]=by_key
    require(seen_files==set(descriptors),'No omitted or extra accepted batch')
    return predictions,summary


def analyze(ctx):
    predictions,saved=read_predictions(ctx);result={};pooled=[]
    for split,(expected_games,_,_) in SPLITS.items():
        parent=predictions[('parent',split)];candidate=predictions[('candidate',split)]
        require(set(parent)==set(candidate),'Both models have identical root identities')
        values=[]
        for key in sorted(parent):
            ctx.check();metrics=row_metrics(parent[key],candidate[key]);values.append((key[0],metrics))
        summary=aggregate_games(values);pooled.extend(values)
        require([g['game_index'] for g in summary['games']]==list(expected_games),'Exact gameweighted split coverage')
        for name in ('parent','candidate'):
            old=saved['summaries'][name][split]
            require(old['game_count']==summary['game_count'] and old['root_count']==summary['root_count'],
                    'Accepted diagnostic root/game totals reproduced')
            require(abs(summary['mean_game_metrics'][name+'_policy_crossentropy_nats']-
                old['mean_game_losses']['policy'])<=1e-12,'Independent saved policy cross-entropy reproduction')
            original_games={g['game_index']:g for g in old['games']}
            for game in summary['games']:
                previous=original_games[game['game_index']]
                require(previous['roots']==game['roots'] and abs(previous['policy']-
                    game['metrics'][name+'_policy_crossentropy_nats'])<=1e-12,'Per-game policy loss reproduction')
        result[split]=summary
    pooled_summary=aggregate_games(pooled)
    require(pooled_summary['game_count']==64 and pooled_summary['root_count']==3557,'All64 games and3557 distinct heldout roots')
    return dict(component='teacher_targets_and_prediction_changes',splits=result,pooled64=pooled_summary,
        aggregation='Arithmetic mean of root metrics within each game, then equal weight for every game',
        probability_definition='Stable float64 softmax of saved float32 policy logits on the full authoritative legal81 mask',
        entropy_and_kl_units='natural logarithms (nats)',
        teacher_top_definition='All actions attaining the exact maximum normalized visit target; no arbitrary teacher tie break',
        policy_argmax_definition='Lowest legal action index attaining the exact maximum saved policy logit',
        teacher_mode_agreement_definition='The deterministic network argmax belongs to the complete teacher top-action set',
        visited_support_definition='Actions with strictly positive normalized teacher visit targets; not retained-but-unvisited actions',
        value_slot_definition='Actor-relative signed outputs: slot0 current actor, slot1 opponent; shifts candidate minus parent',
        effective_support_definition='Per-root exp(teacher entropy), averaged within game then across games',
        original_policy_losses_independently_reproduced=True,
        limitations=['Already-opened64-game diagnostic; no fresh confirmation or strength claim',
            'Network argmax agreement is not searched-move agreement',
            'Retained-but-zero-visit search support is absent from these batch records',
            'Differences do not identify causal policy/value interference or prove the best training change'])


def main():
    ctx=Context();print(ctx.finish('targets.json',analyze(ctx)))


if __name__=='__main__':main()

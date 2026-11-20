"""Exploratory saved-evidence diagnostics; no gameplay, inference or training.

No full-replay claim. Paired roots share identical state, proposals and worlds.
World halves diagnose sampling stability at16 worlds, not32-world repeatability.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import numpy as np

BASE = Path('outputs/learned_continuation_158')
OUT = BASE/'diagnostics_saved_evidence_20260914'
BINDING = '06fcd7d298a3099b11206ff3cc5c4343b5c9c1fb01b3ef975c87fafa015d40ab'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/name).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)


def measures(matrix, partitions):
    means = matrix.mean(axis=0)
    choice = int(means.argmax())
    k = matrix.shape[1]
    halves = matrix[partitions].mean(axis=1)
    other = (matrix.sum(axis=0)[None]-16*halves)/16
    left, right = halves.argmax(axis=1), other.argmax(axis=1)
    slots = np.arange(len(partitions))
    in_gain = ((halves[slots,left]-halves[:,0])+(other[slots,right]-other[:,0]))/4
    out_gain = ((other[slots,left]-other[:,0])+(halves[slots,right]-halves[:,0]))/4
    order = np.argsort(-means, kind='stable')
    gap = 0. if k == 1 else float((means[order[0]]-means[order[1]])/2)
    variance = 0. if k == 1 else float(np.var((matrix[:,1:]-matrix[:,[0]])/2,axis=0,ddof=1).mean())
    return {'support':k,'choice':choice,'gap_score':gap,'all_mean_tie':bool(np.ptp(means)==0),
            'halves_disagree':float((left!=right).mean()),
            'halves_in_sample_gain_score':float(in_gain.mean()),
            'halves_heldout_gain_score':float(out_gain.mean()),
            'mean_pair_difference_variance':variance}


def summary(rows):
    if not rows:
        return {'roots':0}
    keys=('gap_score','halves_disagree','halves_in_sample_gain_score',
          'halves_heldout_gain_score','mean_pair_difference_variance','all_mean_tie')
    return {'roots':len(rows), **{key:float(np.mean([r[key] for r in rows])) for key in keys},
            'choices':dict(Counter(str(r['choice']) for r in rows))}


def paired_summary(rows):
    if not rows:
        return {'roots':0}
    changed=[r for r in rows if r['changed']]
    return {'roots':len(rows),'changed':len(changed),'changed_fraction':len(changed)/len(rows),
            'outcome_disagreement':float(np.mean([r['outcome_disagreement'] for r in rows])),
            'candidate':summary([r['candidate'] for r in rows]),
            'reference':summary([r['reference'] for r in rows]),
            'mean_selected_advantage_under_candidate_score':float(np.mean([r['advantage_candidate'] for r in rows])),
            'mean_selected_advantage_under_reference_score':float(np.mean([r['advantage_reference'] for r in rows])),
            'changed_strict_opposite_rankings':sum(r['advantage_candidate']>0 and r['advantage_reference']<0 for r in changed)}


def main():
    wall=time.perf_counter()
    assert not (OUT/'summary.json').exists()
    result_bytes=(BASE/'result.json').read_bytes()
    assert sha(result_bytes)=='2abae0ec098780c23dcdfd9d82e78044e99d5d2e36ca0021f8e7ee373d3c4fb4'
    lock_bytes=(BASE/'main_games_lock.json').read_bytes()
    assert sha(lock_bytes)=='b535ff50e868020d8e1392f9ae95526e53f7cf1e866cdf6dec50299708b0b06a'
    lock=json.loads(lock_bytes)
    assert lock['manifest']==BINDING
    rng=np.random.default_rng(1584000000)
    partitions=np.stack([rng.permutation(32)[:16] for _ in range(20)])
    all_rows=[]
    matched=[]
    input_hashes={}
    for pair in range(201):
        pair_games=[]
        for seat in (0,1):
            index=2*pair+seat
            raw=(BASE/'main/games'/f'game_{index:04d}.json').read_bytes()
            assert sha(raw)==lock['files'][str(index)]
            game=json.loads(raw)
            assert game['spec']['candidate_seat']==seat and game['manifest']==BINDING
            items={}
            for move in game['trace']:
                path=BASE/'main/roots'/f'game_{index:04d}'/f"turn_{move['turn']:03d}.json"
                raw=path.read_bytes()
                assert sha(raw)==move['root_sha256']
                input_hashes[path.as_posix()]=move['root_sha256']
                record=json.loads(raw)
                assert record['manifest']==BINDING and 'failure' not in record
                matrix=np.array([[o['wdl'] for o in w['outcomes']] for w in record['worlds']])
                assert matrix.shape==(32,len(record['candidates']['actions']))
                assert np.isin(matrix,[-1.,0.,1.]).all()
                assert np.array_equal(matrix.mean(axis=0),record['means'])
                assert int(matrix.mean(axis=0).argmax())==record['choice']
                metrics=measures(matrix,partitions)
                kind='candidate' if move['actor']==seat else 'reference'
                assert ('continuation_model' in record)==(kind=='candidate')
                row={**metrics,'index':index,'pair':pair,'turn':move['turn'],'planner':kind,
                     'phase':'early' if move['turn']<20 else 'middle' if move['turn']<40 else 'late'}
                all_rows.append(row)
                items[move['turn']]=(move,record,matrix,row)
            pair_games.append(items)
        for turn in sorted(set(pair_games[0])&set(pair_games[1])):
            first,second=pair_games[0][turn],pair_games[1][turn]
            if first[0]['state_sha256']!=second[0]['state_sha256']:
                continue
            candidate,reference=(first,second) if first[3]['planner']=='candidate' else (second,first)
            c,r=candidate[1],reference[1]
            assert c['snapshot']==r['snapshot'] and c['candidates']==r['candidates'] and c['schedule']==r['schedule']
            assert [(w['seed'],w['state_sha256']) for w in c['worlds']]==[(w['seed'],w['state_sha256']) for w in r['worlds']]
            ci,ri=c['choice'],r['choice']
            matched.append({'pair':pair,'turn':turn,'model':c['continuation_model'],
                            'changed':ci!=ri,'candidate':candidate[3],'reference':reference[3],
                            'candidate_means':c['means'],'reference_means':r['means'],
                            'advantage_candidate':(c['means'][ci]-c['means'][ri])/2,
                            'advantage_reference':(r['means'][ci]-r['means'][ri])/2,
                            'outcome_disagreement':float((candidate[2]!=reference[2]).mean())})
        if (pair+1)%25==0:
            print(json.dumps({'pairs':pair+1,'roots':len(all_rows),'matched_roots':len(matched)}),flush=True)
    assert len(all_rows)==24496 and len(input_hashes)==24496
    initial=[r for r in matched if r['turn']==0]
    assert len(initial)==201
    data={'time':datetime.now(timezone.utc).isoformat(),'exploratory':True,'full_replay_complete':False,
          'new_simulations':0,'new_model_inference':0,'bootstrap_or_split_seed':1584000000,
          'split_method':'20 fixed random balanced disjoint16/16 world halves per root; both crossfit directions averaged',
          'units':'gain/gap in win=1 draw=.5 loss=0 score units; variances in squared score units',
          'limits':['halves measure16-world stability, not independent32-world repeatability',
                    'heldout scores use each policy own continuation objective, not actual planner strength',
                    'all-root planner distributions differ; matched-root subset includes behavior-selected later positions',
                    'initial201 roots form the fixed same-state panel; no new statistical success gate',
                    'world variation confounds hidden-card uncertainty and learned proposal randomness',
                    'hash checks and algebraic reconstruction do not substitute for incomplete full replay'],
          'source_result_sha256':sha(result_bytes),'source_games_lock_sha256':sha(lock_bytes),
          'source_root_hashes_digest':sha(json.dumps(input_hashes,sort_keys=True,separators=(',',':')).encode()),
          'script_sha256':sha(Path(__file__).read_bytes()),
          'all_roots':{kind:summary([r for r in all_rows if r['planner']==kind]) for kind in ('candidate','reference')},
          'support4':{kind:summary([r for r in all_rows if r['planner']==kind and r['support']==4]) for kind in ('candidate','reference')},
          'phase':{phase:{kind:summary([r for r in all_rows if r['planner']==kind and r['phase']==phase])
                          for kind in ('candidate','reference')} for phase in ('early','middle','late')},
          'matched_initial':paired_summary(initial),'matched_all':paired_summary(matched),
          'matched_initial_by_model':{model:paired_summary([r for r in initial if r['model']==model])
                                      for model in sorted({r['model'] for r in initial})},
          'elapsed_seconds':time.perf_counter()-wall}
    save('roots.json',all_rows)
    save('matched_roots.json',matched)
    save('summary.json',data)
    print(json.dumps({key:data[key] for key in ('elapsed_seconds','all_roots','matched_initial','matched_all')},indent=2),flush=True)


if __name__=='__main__':
    main()

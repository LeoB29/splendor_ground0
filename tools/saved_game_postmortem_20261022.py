"""Read-only arithmetic on the accepted amended benchmark; no model or simulation.

New artifacts only. `prepare` fixes the panel before `analyze` opens its values.
The historical benchmark remains immutable and is now development evidence.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import psutil
from tools import future_training_rules_20261021 as guard

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/matched_strength_recovery_20261022_01'
OUT = ROOT / 'outputs/matched_postmortem_20261022_01'
FINAL_SHA = '20508936bc7033b1c2da8030163b895a09db2d118b4281989a3653a89ec22335'
FREEZE_SHA = '70b1e8f5921db9e3d61306bd6933ccc6ffa529150d4322cdf4a680b9568c9713'
DOMAIN = 'splendor_saved_postmortem_20261022_panel_v1'
START = time.monotonic()
READS = {}
INDEX = {}
MAX_SECONDS = 900
MAX_RSS = 1024**3


def check(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def boundary():
    check(time.monotonic()-START < MAX_SECONDS, '15-minute analysis-process cap')
    check(psutil.Process().memory_info().rss < MAX_RSS, '1-GiB analysis memory cap')


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path, expected=None, indexed=True):
    boundary()
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    key = relative(path)
    check(path.stat().st_size <= 32*1024**2, 'Bounded JSON input size')
    data = path.read_bytes()
    item = {'path': key, 'bytes': len(data), 'sha256': sha(data)}
    if expected is not None:
        check(item == expected, 'Changed linked artifact: '+key)
    if indexed:
        check(INDEX.get(key) == item, 'Not bound by accepted evidence: '+key)
    if key in READS:
        check(READS[key] == item, 'Input changed during analysis: '+key)
    READS[key] = item
    return json.loads(data)


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
    check(len(data) < 32*1024**2, 'Output file size bound')
    check(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(data)<64*1024**2,'64-MiB total new-artifact cap')
    with path.open('xb') as f:
        f.write(data)
    return {'path': relative(path), 'bytes': len(data), 'sha256': sha(data)}


def authenticate():
    runtime = guard.verify_runtime_sources()  # once per new worker/startup
    final = read(BASE/'finalization.json', indexed=False)
    check(READS[relative(BASE/'finalization.json')]['sha256'] == FINAL_SHA, 'Accepted finalization identity')
    check(final['status'] == 'accepted_amended', 'Accepted amended scope')
    freeze = read(BASE/'freeze.json', indexed=False)
    check(READS[relative(BASE/'freeze.json')]['sha256'] == FREEZE_SHA, 'Frozen source identity')
    for item in freeze['inputs'] + final['artifacts']:
        check(item['path'] not in INDEX or INDEX[item['path']] == item, 'Conflicting provenance')
        INDEX[item['path']] = item
    return runtime


def phase(ply):
    return 'early' if ply < 16 else 'middle' if ply < 32 else 'late'


def panel_select(roots):
    result, used = [], set()
    for fit in (161001, 161002, 161003):
        for seat in (0, 1):
            for band in ('early', 'middle', 'late'):
                eligible = [r for r in roots if (r['fit'], r['seat'], r['phase']) == (fit, seat, band)]
                ranked = sorted(eligible, key=lambda r: (r['selection_hash'], r['id']))
                selected = next((r for r in ranked if r['game'] not in used), None)
                check(selected is not None, 'Empty distinct-game panel cell')
                result.append({**selected, 'cell_eligible_roots': len(eligible)})
                used.add(selected['game'])
    check(len(result) == 18 and len(used) == 18, 'Exactly 18 distinct games')
    return result


def prepare(runtime):
    plan = {
        'schema': 'saved_game_postmortem_plan_v1', 'time': datetime.now(timezone.utc).isoformat(),
        'scope': 'post_hoc_development_analysis_not_fresh_evaluation',
        'source_scope': 'amended_recovered_benchmark', 'finalization_sha256': FINAL_SHA,
        'recovery_freeze_sha256': FREEZE_SHA, 'runtime_sources': runtime,
        'selection_domain': DOMAIN, 'phase_by_absolute_zero_based_ply': {'early': [0,15], 'middle': [16,31], 'late': [32,399]},
        'panel_rule': 'Completed local moves only; fit ascending, seat ascending, phase early/middle/late; choose minimum SHA256(domain|game|ply) whose game has not already been selected. 18 distinct games. No values/actions/outcomes enter selection.',
        'prior_inspection': 'Known 0-60 result and g000/t0001 schema example were viewed before this rule; panel is post hoc, never claimed preregistered confirmation.',
        'coverage': 'All 60 game records and every completed local root; retain timeout record separately. Detailed traces only the fixed 18 roots.',
        'metrics': ['offered/legal canonical support and action types', 'legal purchases and immediate points omitted versus offered but rejected', 'raw035/fourth-slot/search disagreements', 'first64/last64 ranking agreement and cross-half conditional gain over first candidate', 'selected forecast by phase; not a calibrated SplendoRust win probability', 'panel trace continuation lengths and reasons', 'next-reply exact action agreement only for no-reveal chosen action and no deck reserves on either side'],
        'reply_comparison': 'Only TAKE_TOKENS or BUY_RESERVED with no deck-sourced reserve for either player; actual next move must be completed opponent. All other roots excluded from same-observation comparisons.',
        'limits': 'Excluded actions have no saved outcome values. Immediate scoring and action resemblance are not optimality. No causal attribution or new pass/fail success gate. No unrecorded counterfactuals.',
        'decision': 'Recommend candidate coverage or evaluation as the next controlled intervention; do not promote a player from this report.',
        'resources': {'analysis_process_wall_seconds': MAX_SECONDS, 'analysis_process_rss_bytes': MAX_RSS, 'new_artifact_cap_bytes': 64*1024**2, 'new_games': 0, 'new_rollouts': 0, 'model_inference': 0, 'training': 0},
        'training_pipeline_qualified': False, 'larger_data_deferred': True,
    }
    save('plan.json', plan)
    roots = []
    for game in range(60):
        for path in sorted((BASE/f'sealed/games/{game:03d}/moves').glob('*.json')):
            move = read(path)
            if move['player'] != 'local' or move['status'] != 'completed':
                continue
            roots.append({'id': move['id'], 'game': game, 'ply': move['ply'], 'fit': move['fit'],
                          'seat': move['actor'], 'phase': phase(move['ply']), 'move': READS[relative(path)],
                          'selection_hash': sha(f"{DOMAIN}|{game}|{move['ply']}".encode())})
    save('root_inventory.json', roots)
    save('panel.json', panel_select(roots))
    save('preparation_receipt.json', {'plan': artifact(OUT/'plan.json'), 'panel': artifact(OUT/'panel.json'),
          'root_inventory': artifact(OUT/'root_inventory.json'), 'completed_local_roots': len(roots),
          'inputs': list(READS.values()), 'analysis_source': artifact(Path(__file__)),
          'seconds': time.monotonic()-START, 'status': 'prepared'})
    print(json.dumps({'status': 'prepared', 'roots': len(roots), 'panel': [(r['game'], r['ply']) for r in panel_select(roots)]}))


def artifact(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return {'path': relative(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def avg(values):
    return statistics.mean(values) if values else None


def distribution(values):
    return {'n': len(values), 'mean': avg(values), 'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def first_max(values):
    return max(range(len(values)), key=lambda i: values[i])


def action_details(action, obs, cards):
    kind = action['action_type']
    cid = None
    if kind in ('BUY_VISIBLE', 'RESERVE_VISIBLE'):
        cid = obs['visible_tier_cards'][str(action['tier'])][action['market_index']]
    elif kind == 'BUY_RESERVED':
        cid = obs['players'][obs['current_player']]['reserves'][action['reserved_index']]['card_id']
    card = cards.get(cid)
    purchase = kind.startswith('BUY_')
    points = (card.points if purchase and card else 0) + (3 if action['noble_id'] else 0)
    return {'type': kind, 'card_id': cid, 'tier': card.tier if card else action['tier'],
            'card_points': card.points if card else None, 'immediate_points': points,
            'noble_id': action['noble_id'], 'take': action['take_tokens'], 'returns': action['return_tokens']}


def root_metrics(move, receipt, cards):
    decision = receipt['result']['decision']; r = decision['record']; p = r['proposal']
    check(r['ruleset_id'] == guard.RULESET_ID and r['catalogue_sha256'] == guard.CATALOGUE_SHA256, 'Root rules identity')
    check(r['root_identity'][-2:] == [move['game'], move['ply']], 'Root identity')
    offered = r['candidates']['actions']; legal = p['legal_actions']; n = len(offered)
    check(len(r['worlds']) == 128 and 1 <= n <= 4, 'Exact saved support/world count')
    values = [[o['wdl'] for o in w['outcomes']] for w in r['worlds']]
    check(all(len(w) == n and all(v in (-1,0,1) for v in w) for w in values), 'Saved WDL domain')
    means = [avg([w[i] for w in values]) for i in range(n)]
    check(means == r['means'] and first_max(means) == r['choice'], 'Saved selection arithmetic')
    check(decision['action'] == offered[r['choice']], 'Played offered action')
    check(len(legal) == len(r['scores']) == r['candidates']['legal_support'], 'Legal score alignment')
    obs = move['observer']['observation']
    detail = [action_details(a, obs, cards) for a in offered]
    ld = [action_details(a, obs, cards) for a in legal]
    half = [[avg([w[i] for w in values[lo:hi]]) for i in range(n)] for lo,hi in ((0,64),(64,128))]
    choices = [first_max(x) for x in half]
    cross_gain = ((half[1][choices[0]]-half[1][0])+(half[0][choices[1]]-half[0][0]))/4
    raw_idx = first_max(r['scores']); chosen = r['choice']
    check(offered[0] == legal[raw_idx], 'First candidate is exact raw035 argmax')
    buys = [x for x in ld if x['type'].startswith('BUY_')]
    offered_buys = [x for x in detail if x['type'].startswith('BUY_')]
    maxpoints = max(x['immediate_points'] for x in ld)
    offeredpoints = max(x['immediate_points'] for x in detail)
    same_max = sum(v == max(means) for v in means)
    legal_types = sorted({x['type'] for x in ld}); offered_types = sorted({x['type'] for x in detail})
    row = {'id': move['id'], 'game': move['game'], 'ply': move['ply'], 'fit': move['fit'], 'seat': move['actor'],
           'phase': phase(move['ply']), 'legal_support': len(legal), 'canonical_support': r['candidates']['canonical_support'],
           'offered_support': n, 'canonical_fraction': n/r['candidates']['canonical_support'],
           'legal_types': legal_types, 'offered_types': offered_types, 'omitted_types': sorted(set(legal_types)-set(offered_types)),
           'choice': chosen, 'selected_action':offered[chosen], 'selected': detail[chosen], 'raw_top': action_details(legal[raw_idx],obs,cards),
           'differs_raw_top_exact': offered[chosen] != legal[raw_idx], 'chosen_fourth': chosen == 3,
           'fourth_changed_from_reference': p['reference']['keys'] != r['candidates']['keys'],
           'selected_score': (means[chosen]+1)/2, 'raw_first_candidate_score': (means[0]+1)/2,
           'score_margin_over_runnerup': (sorted(means,reverse=True)[0]-sorted(means,reverse=True)[1])/2 if n>1 else None,
           'max_ties': same_max, 'half_choices': choices, 'half_same': choices[0] == choices[1],
           'cross_half_gain_score': cross_gain, 'half_means_score': [[(v+1)/2 for v in h] for h in half],
           'candidate_scores': [(v+1)/2 for v in means], 'candidates': detail,
           'legal_buy': bool(buys), 'offered_buy': bool(offered_buys), 'chosen_buy': detail[chosen]['type'].startswith('BUY_'),
           'legal_max_immediate_points': maxpoints, 'offered_max_immediate_points': offeredpoints,
           'chosen_immediate_points': detail[chosen]['immediate_points'],
           'score_gap': obs['players'][move['actor']]['score']-obs['players'][1-move['actor']]['score'],
           'scores': [q['score'] for q in obs['players']], 'bonuses': [sum(q['bonuses'].values()) for q in obs['players']],
           'reserves': [len(q['reserves']) for q in obs['players']],
           'rollout_reasons': dict(Counter(o['reason'] for w in r['worlds'] for o in w['outcomes'])),
           'selected_continuation_turns': distribution([w['outcomes'][chosen]['turns'] for w in r['worlds']]),
           'clock_seconds': move['clock']['elapsed_seconds'],
           'source_move': READS[relative(BASE/f"sealed/games/{move['game']:03d}/moves/{move['ply']:04d}.json")],
           'source_receipt': READS[move['receipt']['path']]}
    return row


def summarize(rows):
    return {'roots': len(rows), 'canonical_support': distribution([r['canonical_support'] for r in rows]),
            'canonical_fraction': distribution([r['canonical_fraction'] for r in rows]),
            'raw_top_changes': sum(r['differs_raw_top_exact'] for r in rows),
            'fourth_selected': sum(r['chosen_fourth'] for r in rows),
            'fourth_changed': sum(r['fourth_changed_from_reference'] for r in rows),
            'half_same': sum(r['half_same'] for r in rows), 'full_top_tie': sum(r['max_ties']>1 for r in rows),
            'cross_half_gain_score': distribution([r['cross_half_gain_score'] for r in rows]),
            'selected_score': distribution([r['selected_score'] for r in rows]),
            'forecast_above_half': sum(r['selected_score']>.5 for r in rows),
            'forecast_at_least_three_quarters': sum(r['selected_score']>=.75 for r in rows),
            'buy_legal': sum(r['legal_buy'] for r in rows),
            'buy_legal_but_none_offered': sum(r['legal_buy'] and not r['offered_buy'] for r in rows),
            'buy_offered_but_not_chosen': sum(r['offered_buy'] and not r['chosen_buy'] for r in rows),
            'positive_points_legal': sum(r['legal_max_immediate_points']>0 for r in rows),
            'positive_points_legal_but_none_offered': sum(r['legal_max_immediate_points']>0 and r['offered_max_immediate_points']==0 for r in rows),
            'positive_points_offered_but_zero_chosen': sum(r['offered_max_immediate_points']>0 and r['chosen_immediate_points']==0 for r in rows),
            'selected_types': dict(Counter(r['selected']['type'] for r in rows)),
            'rollout_reasons': dict(sum((Counter(r['rollout_reasons']) for r in rows), Counter()))}


def trace_metrics(row, move, moves):
    path = ROOT/move['trace']['path']
    check(artifact(path) == move['trace'] and INDEX.get(relative(path)) == move['trace'], 'Authenticated saved trace')
    READS[relative(path)] = move['trace']
    obs = move['observer']['observation']
    comparable = row['selected']['type'] in ('TAKE_TOKENS','BUY_RESERVED') and all(
        q['source'] != 'deck' for p in obs['players'] for q in p['reserves'])
    nextmove = moves.get((move['game'],move['ply']+1))
    comparable = comparable and nextmove is not None and nextmove['player']=='opponent' and nextmove['status']=='completed'
    actual = None
    if comparable:
        actual = read(nextmove['receipt']['path'],nextmove['receipt'])['result']['action']
    reply = Counter(); total = 0; chosen = 0; turntypes = Counter(); continuation_actions = Counter(); fingerprints = set(); seen = set()
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            boundary(); value = json.loads(line); total += 1
            check(value['candidate_index'] < row['offered_support'] and value['world_index'] < 128, 'Trace index')
            identity=(value['world_index'],value['candidate_index'])
            check(identity not in seen,'Unique trace world/candidate');seen.add(identity)
            if value['candidate_index'] != row['choice']:
                continue
            chosen += 1
            check(value['steps'][0]['forced'] and value['steps'][0]['turn']==move['ply'], 'Trace root')
            check(value['steps'][0]['action']==row['selected_action'], 'Trace selected candidate')
            for step in value['steps'][1:]:
                continuation_actions[('local' if step['actor']==move['actor'] else 'opponent')+'_'+step['action']['action_type']] += 1
            if len(value['steps'])>1:
                step=value['steps'][1]
                check(step['actor']==1-move['actor'] and step['turn']==move['ply']+1 and not step['forced'], 'Next simulated opponent')
                reply[canonical(step['action'])] += 1
                turntypes[step['action']['action_type']] += 1
                fingerprints.add(canonical([step['action'],step['scores_sha256'],step['selected_index'],step['legal_count']]))
    check(chosen==128 and total==128*row['offered_support'], 'Complete saved panel trace')
    if comparable:
        check(sum(reply.values())==128 and len(fingerprints)==1,'Same-observation reply invariance across all 128 worlds')
    return {'trace': move['trace'], 'records': total, 'selected_records': chosen,
            'comparable_no_reveal_no_private': bool(comparable), 'reply_count': sum(reply.values()),
            'actual_next_reply': actual, 'exact_actual_reply_count': reply[canonical(actual)] if comparable else None,
            'reply_types': dict(turntypes), 'reply_fingerprints':len(fingerprints), 'selected_continuation_action_mix': dict(continuation_actions),
            'reply_distribution': [{'action': json.loads(k), 'count': v} for k,v in reply.most_common()]}


def analyze(runtime):
    prep = read(OUT/'preparation_receipt.json',indexed=False)
    check(prep['analysis_source']==artifact(Path(__file__)), 'Analysis implementation changed after preparation')
    plan=read(OUT/'plan.json',prep['plan'],indexed=False)
    panel=read(OUT/'panel.json',prep['panel'],indexed=False)
    inventory=read(OUT/'root_inventory.json',prep['root_inventory'],indexed=False)
    check(panel_select(inventory)==panel, 'Panel selection reproduced')
    cards=guard.rules.corrected_cards_by_id()  # component lookup only; no environment/state decoder
    moves={}; results=[]; failures=[]
    for game in range(60):
        result=read(BASE/f'sealed/games/{game:03d}/result.json')
        check(result['complete'] and result['game']==game, 'All 60 accepted games')
        final=result['final']; check(final['ruleset_id']==guard.RULESET_ID and final['catalogue_sha256']==guard.CATALOGUE_SHA256,'Final rules identity')
        state=final['state']; seat=result['local_seat']
        results.append({'game': game,'fit': result['fit'],'seat': seat,'reason': result['reason'],
                        'winner': result['winner'],'score_gap': state['players'][seat]['score']-state['players'][1-seat]['score'],
                        'scores': [p['score'] for p in state['players']], 'plies': state['turn_index']})
        for path in sorted((BASE/f'sealed/games/{game:03d}/moves').glob('*.json')):
            m=read(path); moves[(game,m['ply'])]=m
            if m['status']!='completed':
                failures.append({'id':m['id'],'status':m['status'],'receipt':m['receipt']})
    rows=[]; panels=[]; selected={r['id'] for r in panel}
    for item in inventory:
        m=moves[(item['game'],item['ply'])]
        check(item['move']==READS[relative(BASE/f"sealed/games/{item['game']:03d}/moves/{item['ply']:04d}.json")], 'Prepared root unchanged')
        receipt=read(m['receipt']['path'],m['receipt'])
        row=root_metrics(m,receipt,cards); rows.append(row)
        if row['id'] in selected:
            panels.append({**row, 'trace_analysis':trace_metrics(row,m,moves)})
    check(len(panels)==18, 'Full fixed panel')
    last=[max((r for r in rows if r['game']==g['game']),key=lambda r:r['ply']) for g in results if g['reason']=='natural']
    game_rows=[{'game':g['game'],'fit':g['fit'],'seat':g['seat'],
                'mean_cross_half_gain_score':avg([r['cross_half_gain_score'] for r in rows if r['game']==g['game']]),
                'mean_selected_score':avg([r['selected_score'] for r in rows if r['game']==g['game']]),
                'half_agreement_fraction':avg([r['half_same'] for r in rows if r['game']==g['game']])} for g in results]
    summary={'scope':plan['scope'],'games':len(results),'natural_games':sum(g['reason']=='natural' for g in results),
             'failures':failures,'all_roots':summarize(rows),'panel':summarize(panels),
             'by_phase':{p:summarize([r for r in rows if r['phase']==p]) for p in ('early','middle','late')},
             'by_fit':{str(f):summarize([r for r in rows if r['fit']==f]) for f in (161001,161002,161003)},
             'last_local_natural_roots':summarize(last), 'natural_score_gap':distribution([g['score_gap'] for g in results if g['reason']=='natural']),
             'natural_game_plies':distribution([g['plies'] for g in results if g['reason']=='natural']),
             'game_weighted_cross_half_gain_score':distribution([g['mean_cross_half_gain_score'] for g in game_rows]),
             'game_weighted_selected_score':distribution([g['mean_selected_score'] for g in game_rows]),
             'panel_comparable_reply_roots':sum(p['trace_analysis']['comparable_no_reveal_no_private'] for p in panels),
             'panel_comparable_zero_actual_reply_roots':sum(p['trace_analysis']['comparable_no_reveal_no_private'] and p['trace_analysis']['exact_actual_reply_count']==0 for p in panels)}
    save('games.json',results);save('game_root_summaries.json',game_rows);save('roots.json',rows);save('panel_analysis.json',panels);save('summary.json',summary)
    for item in list(READS.values()):
        boundary();check(artifact(ROOT/item['path'])==item,'Consumed evidence changed: '+item['path'])
    outputs=[artifact(OUT/name) for name in ('plan.json','preparation_receipt.json','panel.json','root_inventory.json','games.json','game_root_summaries.json','roots.json','panel_analysis.json','summary.json')]
    save('completion.json',{'status':'completed_read_only_postmortem','time':datetime.now(timezone.utc).isoformat(),
         'runtime_sources':runtime,'inputs_reverified':list(READS.values()),'outputs':outputs,
         'seconds':time.monotonic()-START,'rss_bytes':psutil.Process().memory_info().rss,'new_games':0,'new_rollouts':0,'model_inference':0,'training':0,
         'larger_data_deferred':True,'training_pipeline_qualified':False})
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','analyze']);args=parser.parse_args()
    try:
        runtime=authenticate()
        (prepare if args.command=='prepare' else analyze)(runtime)
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        save('failure_'+str(time.time_ns())+'.json',{'command':args.command,'traceback':traceback.format_exc(),'seconds':time.monotonic()-START})
        raise

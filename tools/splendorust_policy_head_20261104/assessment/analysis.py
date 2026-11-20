"""Fixed400-game paired strength assessment of fixed learned candidate versus historical warm-start parent at128 each.

Outcome arithmetic only: all complete game evidence must be verified first.
No games, models, timing-based selection, sample extension or promotion.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math

MAIN_PAIRS = 200
REPLICATES = 50000
SEED = 20261104128
FAILURE_POLICY = 'research_hard_guard_9s_internal_10s_complete_response'
ADMIN_FAILURES = ('research_internal_hard_guard', 'research_outer_hard_guard')



def require(value, message):
    if not value:
        raise ValueError('SELFPLAY_ASSESSMENT_STATISTICS: ' + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def runtime_contract():
    from tools.matched_strength_20261020.statistics import runtime_contract as contract
    return contract()


def validated_games(games):
    """Strict complete sample and official results, independent of input order."""
    from . import design as schedule
    require(c.SCOPE == 'pilot' and c.MAIN_GAMES == 400, 'engineering qualification has no main strength estimate')
    expected = schedule.make_schedule('main')
    require(type(games) is list and len(games) == len(expected),
            'complete fixed400 main-game sample required')
    wanted = {_digest(spec) for spec in expected}
    observed = []
    ordered = []
    for game in games:
        require(type(game) is dict, 'game object')
        spec = schedule.validate_spec(game)
        require(spec['stage'] == 'main' and spec['candidate'] == 'candidate',
                'wrong stage or candidate')
        require(spec['challenger_mode'] == spec['candidate'] and
                spec['opponent_mode'] == 'control' and
                type(spec['challenger_quota']) is int and spec['challenger_quota'] == 128 and
                type(spec['opponent_quota']) is int and spec['opponent_quota'] == 128,
                'exact learned versus parent identities at matched128 search')
        observed.append(_digest(spec))
        seat = spec['challenger_seat']
        require(type(seat) is int and seat in (0, 1), 'candidate seat')
        require(game.get('complete') is True and 'winner' in game, 'incomplete or missing result')
        winner, reason, loser = game['winner'], game.get('reason'), game.get('loser')
        require(winner is None or type(winner) is int and winner in (0, 1), 'winner type/seat')
        require(reason in ('natural', 'forfeit'), 'unofficial or unresolved ending')
        if reason == 'forfeit':
            require(type(loser) is int and loser in (0, 1) and winner == 1 - loser,
                    'forfeit must lose for the responsible player')
            require(game.get('administrative_forfeit') is True and
                    game.get('failure_policy') == FAILURE_POLICY and
                    game.get('failure') in ADMIN_FAILURES,
                    'only classified research hard-guard forfeits enter the sample')
            require(type(game.get('failed_seat')) is int and game['failed_seat'] == loser and
                    game.get('failed_role') == ('challenger' if loser == seat else 'opponent'),
                    'administrative forfeit role/seat inconsistent')
        else:
            require(loser is None and game.get('administrative_forfeit', False) is False and
                    all(game.get(key) is None for key in
                        ('failure_policy', 'failure', 'failed_role', 'failed_seat')),
                    'natural result carries contradictory administrative failure')
        halfpoints = 1 if winner is None else 2 * int(winner == seat)
        for key in ('candidate_score_halfpoints', 'search_score_halfpoints'):
            if key in game:
                require(type(game[key]) is int and game[key] == halfpoints, 'derived score differs')
        ordered.append(dict(spec, winner=winner, reason=reason, loser=loser,
                            candidate_score_halfpoints=halfpoints,
                            conservative_halfpoints=0 if reason == 'forfeit' else halfpoints,
                            administrative_forfeit=reason == 'forfeit',
                            failure_policy=game.get('failure_policy'), failure=game.get('failure'),
                            failed_role=game.get('failed_role'), failed_seat=game.get('failed_seat')))
    require(len(wanted) == len(expected) and len(set(observed)) == len(expected) and
            set(observed) == wanted, 'missing, duplicate, substituted, or extra fixed game identity')
    ordered.sort(key=lambda row: (row['pair_index'], row['game_index']))
    require(len(ordered) == 2 * MAIN_PAIRS, 'fixed main sample size')
    seeds = []
    for pair in range(MAIN_PAIRS):
        members = ordered[2 * pair:2 * pair + 2]
        require(len(members) == 2 and {row['challenger_seat'] for row in members} == {0, 1} and
                len({row['seed_decimal'] for row in members}) == 1 and
                all(row['pair_index'] == pair for row in members),
                'swapped-seat pair shares one initial-world seed')
        seeds.append(members[0]['seed_decimal'])
    require(len(set(seeds)) == MAIN_PAIRS, 'independent setup seed per pair')
    return ordered


def _score(rows, conservative=False):
    key = 'conservative_halfpoints' if conservative else 'candidate_score_halfpoints'
    count = len(rows)
    require(count > 0 and count % 2 == 0, 'nonempty paired rows')
    n_pairs = count // 2
    points = [sum(row[key] for row in rows if row['pair_index'] == pair) for pair in range(n_pairs)]
    numerator = sum(points)
    seat_numerators = {str(seat): sum(row[key] for row in rows if row['challenger_seat'] == seat)
                       for seat in (0, 1)}
    counts = Counter(row[key] for row in rows)
    return dict(games=count, pairs=n_pairs, pair_numerators=points, numerator=numerator,
                denominator=2 * count, mean_score=numerator / (2 * count),
                by_candidate_seat={seat: value / count for seat, value in seat_numerators.items()},
                seat_numerators=seat_numerators, seat_denominator=count,
                wins=counts[2], draws=counts[1], losses=counts[0])


def _useful_gates(score):
    # Integer inequalities avoid an adjustable or rounded threshold.
    return dict(score_at_least_55pct=100 * score['numerator'] >= 55 * score['denominator'],
                each_seat_score_at_least_50pct=all(2 * value >= score['seat_denominator']
                                                 for value in score['seat_numerators'].values()))


def _administrative(rows):
    admin = [row for row in rows if row['administrative_forfeit']]
    natural = sum(row['candidate_score_halfpoints'] for row in rows if not row['administrative_forfeit'])
    return dict(administrative_games=len(admin), natural_games=len(rows) - len(admin),
                by_failed_player={role: sum(row['failed_role'] == role for row in admin)
                                  for role in ('challenger', 'opponent')},
                lower_numerator=natural, upper_numerator=natural + 2 * len(admin),
                denominator=2 * len(rows),
                all_administrative_candidate_losses=natural / (2 * len(rows)),
                all_administrative_candidate_wins=(natural + 2 * len(admin)) / (2 * len(rows)),
                natural_only_strength_estimate=False,
                interpretation='Full-sample reassignment sensitivity, not recovered counterfactual outcomes.')


def _bootstrap(official_pairs, conservative_pairs, boundary):
    import numpy as np
    values = np.asarray([official_pairs, conservative_pairs], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = np.empty((2, REPLICATES), dtype=np.int64)
    stream = hashlib.sha256()
    for start in range(0, REPLICATES, 500):
        boundary()
        n = min(500, REPLICATES - start)
        indices = rng.integers(0, MAIN_PAIRS, size=(n, MAIN_PAIRS), dtype=np.int64)
        stream.update(indices.astype('<i8').tobytes(order='C'))
        totals[:, start:start + n] = values[:, indices].sum(axis=2, dtype=np.int64)
    return totals, stream.hexdigest()


def _main_score(score, totals):
    import numpy as np
    result = dict(score)
    ci = (np.quantile(totals, [.025, .975], method='linear') / score['denominator']).tolist()
    radius = math.sqrt(math.log(40) / (2 * MAIN_PAIRS))
    hoeffding = [max(0., score['mean_score'] - radius), min(1., score['mean_score'] + radius)]
    degenerate = ci[0] == ci[1]
    gates = dict(_useful_gates(score), paired_lower_bound_above_50pct=ci[0] > .5,
                 degenerate_interval_safeguard=not degenerate or hoeffding[0] > .5)
    result.update(paired_ci95=ci, bounded_pair_hoeffding_95=hoeffding, bootstrap_degenerate=degenerate,
                  useful_gain_criteria=gates, useful_gain=all(gates.values()),
                  integer_totals_sha256=hashlib.sha256(totals.astype('<i8').tobytes()).hexdigest())
    return result


def analyze_main(games, evidence_gates_passed, *, boundary=lambda: None):
    import numpy as np
    require(evidence_gates_passed is True, 'all main evidence gates must pass before analysis')
    boundary()
    ordered = validated_games(games)
    official, conservative = _score(ordered), _score(ordered, True)
    totals, stream_hash = _bootstrap(official['pair_numerators'], conservative['pair_numerators'], boundary)
    official, conservative = _main_score(official, totals[0]), _main_score(conservative, totals[1])
    useful = official['useful_gain'] and conservative['useful_gain']
    classification = ('useful_gain_against_parent_at_matched128' if useful else
                      'official_useful_gain_not_robust_to_guard_forfeits' if official['useful_gain'] else
                      'useful_gain_not_established')
    boundary()
    return dict(schema='splendorust_selfplay_pilot_strength_analysis_v1', status='complete_verified_main',
                candidate='candidate', candidate_cpuct=.4, reference_cpuct=.4,
                candidate_simulations=128, reference_simulations=128, world_pool_each=3,
                games=400, pairs=200, official=official, conservative=conservative,
                administrative_sensitivity=_administrative(ordered), classification=classification,
                useful_gain_established=useful, complete_verification=True, evidence_gates_passed=True,
                verified_score_rows_sha256=_digest(ordered),
                bootstrap=dict(replicates=REPLICATES, seed=SEED, generator='numpy.PCG64', numpy_version=np.__version__,
                               resampling_unit='swapped_seat_pair', draw_order='replicate_major_200_int64_indices',
                               same_indices_for_official_and_conservative=True, quantile_method='linear',
                               indices_sha256=stream_hash),
                research_guard_seconds=dict(internal=9., complete_response=10.),
                timing_exceedances_seconds_are_descriptive_only=[.7, .9, 1.],
                one_second_deployability_established=False, automatic_promotion=False,
                other_study_games_pooled=False, operational_games_pooled=False, no_sample_extension=True,
                interpretation='Fixed final learned checkpoint against its unchanged historical parent at128 simulations each on fresh paired setups. Official guard-forfeit score and conservative sensitivity must both pass the frozen useful-gain rules. This does not establish broader opponent generalization, one-second deployability, or automatic promotion.')


def _independent_arithmetic(rows, score, conservative):
    """Direct winner/seat arithmetic, not the primary derived-score accumulation."""
    values = [0 if conservative and row['reason'] == 'forfeit' else
              1 if row['winner'] is None else 2 if row['winner'] == row['challenger_seat'] else 0
              for row in rows]
    n_pairs = len(rows) // 2
    pairs = [sum(value for row, value in zip(rows, values) if row['pair_index'] == pair)
             for pair in range(n_pairs)]
    seats = {str(seat): sum(value for row, value in zip(rows, values) if row['challenger_seat'] == seat)
             for seat in (0, 1)}
    require(score['pair_numerators'] == pairs and score['numerator'] == sum(values) and
            score['denominator'] == 2 * len(rows) and score['mean_score'] == sum(values) / (2 * len(rows)) and
            score['seat_numerators'] == seats and score['by_candidate_seat'] ==
            {seat: value / len(rows) for seat, value in seats.items()}, 'independent integer/seat arithmetic differs')
    return pairs


def verify_main(games, result, *, boundary=lambda: None):
    """Fresh per-replicate draw, Python integer sums and manual percentiles."""
    import numpy as np
    ordered = validated_games(games)
    pairs = [_independent_arithmetic(ordered, result[key], conservative)
             for key, conservative in (('official', False), ('conservative', True))]
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = [[], []]
    stream = hashlib.sha256()
    for replicate in range(REPLICATES):
        if replicate % 500 == 0:
            boundary()
        indices = rng.integers(0, MAIN_PAIRS, size=MAIN_PAIRS, dtype=np.int64)
        stream.update(indices.astype('<i8').tobytes())
        for route in (0, 1):
            totals[route].append(sum(pairs[route][int(index)] for index in indices))
    for route, key in enumerate(('official', 'conservative')):
        sorted_totals = sorted(totals[route])
        interval = []
        for quantile in (.025, .975):
            offset = (REPLICATES - 1) * quantile
            lo, hi = math.floor(offset), math.ceil(offset)
            interval.append((sorted_totals[lo] + (sorted_totals[hi] - sorted_totals[lo]) * (offset - lo)) / 800)
        require(result[key]['paired_ci95'] == interval, 'independent percentile interpolation differs')
        require(result[key]['integer_totals_sha256'] ==
                hashlib.sha256(np.asarray(totals[route], dtype='<i8').tobytes()).hexdigest(),
                'independent bootstrap integer totals differ')
    require(result['bootstrap']['indices_sha256'] == stream.hexdigest(), 'independent bootstrap stream differs')
    require(result == analyze_main(games, True, boundary=boundary),
            'main assessment report changed')
    return dict(status='passed', replicates=REPLICATES, independent_python_integer_sums=True,
                independent_linear_percentiles=True, independent_conservative_sensitivity=True,
                index_stream_sha256=stream.hexdigest(), verified_result_sha256=_digest(result),
                shared_trust_boundary='frozen NumPy PCG64 and bounded-integer primitive')


# Independent corrected ledger and same-route model reproduction.
from collections import Counter,defaultdict
from copy import deepcopy
import math
import statistics
from tools.external_strength_screen_20261018 import audit as ledger
from tools.future_training_rules_20261021 import decode_training_snapshot
from . import scientific
from . import common as c,design


def same(a,b):return c.canonical(a)==c.canonical(b)


def validate_completed(row):
    variant=row.get('variant');require(variant in c.VARIANTS,'known route')
    quota=c.ROUTE_QUOTAS[variant]
    require(row.get('route')==row.get('mode')==variant and type(row.get('quota')) is int and row['quota']==quota,
        'externally fixed route/quota')
    received=scientific.check_clock(row['clock'],limit=10.)
    require(row.get('completed') is True and row['clock']['status']=='ok' and
        row['clock']['deadline_seconds']==10. and received is not None,'complete original research response')
    result=row['clock']['result']
    require(same(row.get('action'),result['action']) and same(row.get('trace'),result['diagnostics']),
        'saved action/trace binds actual response')
    ledger.check_observer(row['before'],row['observer'])
    scientific.validate_result(result,row['observer'],'search',expected_quota=quota)
    scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',before=row['before'],after=row['after'],expected_quota=quota)
    require(same(ledger.transition(row['before'],row['action']),row['after']),'independent corrected ledger successor')
    decode_training_snapshot(row['after'])
    return received


def audit_game(initial,rows,ending,spec,*,freeze=None,boundary=lambda:None):
    """Authenticate every move and official ending independently of env.step."""
    spec=design.validate_spec(spec)
    require(same(initial,design.initial_snapshot(spec)),'exact corrected seed-bound initial state')
    require(type(rows) is list and 0<len(rows)<=c.MAX_PLIES,'bounded nonempty game rows')
    require(type(ending) is dict and all(k in ending and same(ending[k],v) for k,v in spec.items()),'ending specification')
    require(ending.get('complete') is True and type(ending.get('plies')) is int and ending['plies']==len(rows),
        'exact complete game move count')
    current=deepcopy(initial);decode_training_snapshot(current);failure=None;transitions=0
    latency={v:[] for v in c.VARIANTS}
    for ply,row in enumerate(rows):
        boundary();require(failure is None and not current['state']['terminal'],'no move after terminal/failure')
        require(design.validate_spec(row)==spec and type(row.get('ply')) is int and row['ply']==ply and
            same(row.get('before'),current),'contiguous full game identity and state')
        seat=current['state']['current_player'];role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        variant=spec[role+'_variant']
        require(type(row.get('seat')) is int and row['seat']==seat and row.get('player_role')==role and
            row.get('variant')==variant and row.get('arm')=='search' and type(row.get('quota')) is int and row['quota']==c.ROUTE_QUOTAS[variant] and
            row.get('route')==row.get('mode')==variant,'actual acting seat/variant/scientific identity')
        require(row.get('placement_mode')=='isolated' and not any(k in row for k in
            ('transition_error','placement_error','profile_capture_error')),'no hidden infrastructure failure')
        ledger.check_observer(current,row['observer'])
        require(row['clock']['deadline_seconds']==10.,'unshortened research deadline')
        if row['clock']['status']=='ok':
            elapsed=validate_completed(row);current=row['after'];latency[variant].append(elapsed);transitions+=1
            if not current['state']['terminal']:
                require(not same(row['before'],row['after']),'unresolved unchanged-state deadlock')
                require(not(ply and row['action']['action_type']=='PASS' and rows[ply-1]['action'] and
                    rows[ply-1]['action']['action_type']=='PASS'),'unresolved consecutive-pass deadlock')
        else:
            require(row.get('completed') is False and row.get('action') is None and row.get('trace') is None and
                row.get('after') is None,'failed request cannot invent a move')
            failure=dict(seat=seat,role=role,variant=variant,status=scientific.classify_failure(row['clock']))
    boundary();require(same(ending.get('final'),current),'final snapshot binds ledger')
    if failure is None:
        require(ending.get('reason')=='natural' and current['state']['terminal'] and
            same(ending.get('winner'),current['state']['winner']) and ending.get('loser') is None and
            ending.get('failed_role') is None and ending.get('failed_seat') is None and
            ending.get('failure') is None and ending.get('failure_policy') is None and
            ending.get('administrative_forfeit',False) is False,'true natural terminal and winner')
    else:
        require(ending.get('reason')=='forfeit' and ending.get('administrative_forfeit') is True and
            ending.get('failure_policy')=='research_hard_guard_9s_internal_10s_complete_response' and
            not current['state']['terminal'] and type(ending.get('winner')) is int and ending['winner']==1-failure['seat'] and
            type(ending.get('loser')) is int and ending['loser']==failure['seat'] and
            ending.get('failed_role')==failure['role'] and ending.get('failed_seat')==failure['seat'],
            'explicit hard-guard forfeit attribution')
        detail=ending.get('failure')
        require(detail==failure['status'] or type(detail) is dict and detail.get('status')==failure['status'],
            'exact preserved failure classification')
    return dict(status='passed',stage=spec['stage'],game_index=spec['game_index'],transitions=transitions,failed_moves=int(failure is not None),
        final_sha256=c.digest(current),latencies=latency,natural_terminal=failure is None,
        transition_oracle='independent corrected dictionary ledger, never env.step',
        trust_boundary='shared pinned corrected catalogue; raw transport/variant identity and process closure checked separately')


def verify_replay(original,replayed,selection):
    """A fresh worker must reproduce its own variant exactly, not its opponent."""
    require(original.get('completed') is True,'only completed played moves can be reproduced')
    expected={k:original[k] for k in ('stage','game_index','pair_index','ply','variant','player_role','seat')}
    require(all(selection.get(k)==v for k,v in expected.items()) and
        selection.get('observer_sha256')==c.digest(original['observer']),'source selection binding')
    require(replayed.get('variant')==original['variant'] and replayed.get('route')==original['variant'],
        'reproduction same fixed execution variant')
    for key in ('observer','before','action','trace','after'):
        require(same(original[key],replayed.get(key)),'exact same-variant replay '+key)
    require(same(original['clock']['result'],replayed['clock']['result']),'full scientific replay output equality')
    validate_completed(original);validate_completed(replayed)
    seed=int(c.digest(original['observer'])[:16],16)
    require(original['trace']['observer_seed']==replayed['trace']['observer_seed']==seed,'unchanged observer-derived world/search seed')
    return dict(status='passed',stage=original['stage'],game_index=original['game_index'],ply=original['ply'],variant=original['variant'],
        observer_sha256=c.digest(original['observer']),result_sha256=c.digest(original['clock']['result']),
        full_scientific_output_exact=True,cross_variant_equality_required=False)


def distribution(values):
    values=sorted(values)
    require(all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=10 for x in values),'valid completed response times')
    if not values:return dict(count=0,minimum=None,maximum=None,mean=None,median=None,p95=None,
        completed_over_700ms=0,completed_over_900ms=0,completed_over_one_second=0)
    q=(len(values)-1)*.95;lo=math.floor(q);hi=math.ceil(q)
    return dict(count=len(values),minimum=values[0],maximum=values[-1],mean=statistics.fmean(values),median=statistics.median(values),
        p95=values[lo]+(values[hi]-values[lo])*(q-lo),completed_over_700ms=sum(x>.7 for x in values),
        completed_over_900ms=sum(x>.9 for x in values),completed_over_one_second=sum(x>1 for x in values))


def summarize_timing(rows,*,boundary=lambda:None):
    """One streaming pass; retain timing scalars only, never scientific trees."""
    timing={(stage,variant):[] for stage in design.PAIRS for variant in c.VARIANTS}
    failures=[];seen=set();count=0
    for row in rows:
        boundary();spec=design.validate_spec(row);stage=spec['stage'];variant=row.get('variant')
        ply=row.get('ply');identity=(stage,spec['game_index'],ply)
        require(type(ply) is int and 0<=ply<c.MAX_PLIES and identity not in seen,'unique bounded timing decision')
        seen.add(identity)
        seat=row.get('seat');role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        require(type(seat) is int and seat in (0,1) and variant==spec[role+'_variant'] and
            row.get('player_role')==role and row.get('route')==row.get('mode')==variant and
            type(row.get('quota')) is int and row['quota']==c.ROUTE_QUOTAS[variant], 'timing role binding')
        require(row['clock']['deadline_seconds']==10.,'original complete-response guard')
        elapsed=scientific.check_clock(row['clock'],limit=10.)
        if row['clock']['status']=='ok':
            require(row.get('completed') is True,'completed timing declaration')
            timing[stage,variant].append(elapsed)
        else:
            require(row.get('completed') is False,'failed timing declaration')
            failures.append(dict(stage=stage,game_index=spec['game_index'],ply=ply,variant=variant,
                classification=scientific.classify_failure(row['clock'])))
        count+=1
    return dict(schema='selfplay_assessment_timing_summary_v1',decisions=count,failed_decisions=len(failures),failures=failures,
        stages={stage:{variant:distribution(timing[stage,variant]) for variant in c.VARIANTS} for stage in design.PAIRS},
        timings={variant:distribution(timing['operational',variant]+timing['main',variant]) for variant in c.VARIANTS},
        one_second_is_descriptive=True,seven_hundred_ms_is_descriptive=True,
        timing_used_for_strength_selection=False,one_second_deployability_established=False,
        interpretation='Main and operational timings are separate. Live states may diverge; aggregate times are not a controlled implementation speed ratio. All late completed moves remain in the sample under the unchanged9/10-second guards.')


def summarize_operational(games,rows,schedule=None,*,boundary=lambda:None):
    """Only six natural operational results admit the fixed main assessment."""
    expected=design.make_schedule('operational')
    if schedule is not None:require(schedule==expected,'fixed operational schedule')
    require(type(games) is list and len(games)==6,'six complete operational games')
    from itertools import groupby
    proofs=[];wins=Counter();timing={variant:[] for variant in c.VARIANTS}
    for index,(key,group) in enumerate(groupby(rows,key=lambda row:(row.get('stage'),row.get('game_index')))):
        require(index<6 and key==('operational',index),'complete ordered operational move inventories')
        spec,game=expected[index],games[index]
        require(design.validate_spec(game)==spec and game.get('reason')=='natural','fixed natural operational order')
        game_rows=list(group)
        proof=audit_game(design.initial_snapshot(spec),game_rows,game,spec,boundary=boundary)
        proofs.append(proof)
        for variant in c.VARIANTS:timing[variant].extend(proof['latencies'][variant])
        wins['draw' if game['winner'] is None else 'candidate' if game['winner']==spec['challenger_seat'] else 'control']+=1
        del game_rows
    require(len(proofs)==6,'all six operational move inventories')
    decisions=sum(len(values) for values in timing.values())
    return dict(schema='selfplay_assessment_operational_summary_v1',status='operational_games_passed',
        complete_planned_games=True,operational_games_passed=True,planned_games=6,completed_games=6,completed_pairs=3,
        decisions=decisions,completed_decisions=decisions,natural_games=6,administrative_forfeits=0,
        descriptive_wins={key:wins[key] for key in ('control','candidate','draw')},
        timings={variant:distribution(timing[variant]) for variant in c.VARIANTS},game_proofs=proofs,strength_test=False,
        reference_promoted=False,one_second_is_descriptive=True,seven_hundred_ms_is_descriptive=True)


select_replay_sample=design.select_replay_sample

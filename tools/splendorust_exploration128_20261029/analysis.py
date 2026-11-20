"""Prospectively fixed development selection and independent confirmation.

Outcome arithmetic only: the owner must verify complete game evidence before
calling either stage. No player, game, training, timing-based selection, partial
sample result, sample extension, or promotion is implemented here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

CANDIDATES = ('low', 'high')
CPUCT = {'low': .2, 'high': .8, 'frozen': .4}
DEVELOPMENT_PAIRS = 50
CONFIRMATION_PAIRS = 200
REPLICATES = 50000
SEED = 20261029128
FAILURE_POLICY = 'research_hard_guard_9s_internal_10s_complete_response'
ADMIN_FAILURES = ('research_internal_hard_guard', 'research_outer_hard_guard')


def require(value, message):
    if not value:
        raise ValueError('EXPLORATION128_STATISTICS: ' + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def runtime_contract():
    from tools.matched_strength_20261020.statistics import runtime_contract as contract
    return contract()


def validated_games(games, stage, candidate=None):
    """Strict complete sample and official results, independent of input order."""
    from . import schedule
    require(stage in ('development', 'confirmation'), 'analysis stage')
    require(candidate is None if stage == 'development' else candidate in CANDIDATES,
            'development has both candidates; confirmation has one selected candidate')
    expected = schedule.make_schedule(stage, candidate)
    require(type(games) is list and len(games) == len(expected),
            'complete fixed ' + stage + ' game sample required')
    wanted = {_digest(spec) for spec in expected}
    observed = []
    ordered = []
    for game in games:
        require(type(game) is dict, 'game object')
        spec = schedule.validate_spec(game)
        require(spec['stage'] == stage and spec['candidate'] in CANDIDATES,
                'wrong stage or candidate')
        require(spec['challenger_mode'] == spec['candidate'] and
                spec['opponent_mode'] == 'frozen' and
                type(spec['challenger_quota']) is int and spec['challenger_quota'] == 128 and
                type(spec['opponent_quota']) is int and spec['opponent_quota'] == 128,
                'exact fixed128 player identities')
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
    ordered.sort(key=lambda row: (CANDIDATES.index(row['candidate']), row['pair_index'], row['game_index']))
    n_pairs = DEVELOPMENT_PAIRS if stage == 'development' else CONFIRMATION_PAIRS
    candidates = CANDIDATES if stage == 'development' else (candidate,)
    seeds = {}
    for route in candidates:
        rows = [row for row in ordered if row['candidate'] == route]
        require(len(rows) == 2 * n_pairs, 'candidate sample size')
        for pair in range(n_pairs):
            members = [row for row in rows if row['pair_index'] == pair]
            require(len(members) == 2 and {row['challenger_seat'] for row in members} == {0, 1} and
                    len({row['seed_decimal'] for row in members}) == 1,
                    'swapped-seat pair must share one initial-world seed')
        seeds[route] = [rows[2 * pair]['seed_decimal'] for pair in range(n_pairs)]
        require(len(set(seeds[route])) == n_pairs, 'initial-world seeds repeated across pairs')
    if stage == 'development':
        require(seeds['low'] == seeds['high'], 'development candidates must share all50 setups')
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


def _development_gates(score):
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


def select_candidate(development_games, evidence_gates_passed, *, boundary=lambda: None):
    """One predeclared selection, only after all200 development games are verified."""
    require(evidence_gates_passed is True, 'all development evidence gates must pass before selection')
    boundary()
    ordered = validated_games(development_games, 'development')
    results = {}
    for candidate in CANDIDATES:
        rows = [row for row in ordered if row['candidate'] == candidate]
        official, conservative = _score(rows), _score(rows, True)
        official_gates, conservative_gates = _development_gates(official), _development_gates(conservative)
        results[candidate] = dict(cpuct=CPUCT[candidate], official=official, conservative=conservative,
                                  official_gates=official_gates, conservative_gates=conservative_gates,
                                  eligible=all(official_gates.values()) and all(conservative_gates.values()),
                                  administrative_sensitivity=_administrative(rows))
    eligible = [candidate for candidate in CANDIDATES if results[candidate]['eligible']]
    selected = min(eligible, key=lambda candidate: (-results[candidate]['conservative']['numerator'],
                                                   CANDIDATES.index(candidate))) if eligible else None
    boundary()
    return dict(schema='splendorust_exploration128_selection_v1', status='complete_verified_development',
                games=200, paired_setups=50, games_per_candidate=100, candidates=results,
                selected_candidate=selected, selected_cpuct=None if selected is None else CPUCT[selected],
                selection_rule='Require official and all-administrative-as-candidate-loss score>=55% and each seat>=50%; rank conservative score only; fixed low0.2 wins exact ties.',
                evidence_gates_passed=True, verified_score_rows_sha256=_digest(ordered),
                development_confidence_gate=False, development_is_strength_confirmation=False,
                no_timing_or_q_selection=True, no_sample_extension=True, automatic_promotion=False,
                interpretation='Noisy development screen of two preset alternatives, with common paired setups. Selection is not evidence of a confirmed gain; only the selected candidate can enter the disjoint fixed400-game confirmation.')


def _bootstrap(official_pairs, conservative_pairs, boundary):
    import numpy as np
    values = np.asarray([official_pairs, conservative_pairs], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = np.empty((2, REPLICATES), dtype=np.int64)
    stream = hashlib.sha256()
    for start in range(0, REPLICATES, 500):
        boundary()
        n = min(500, REPLICATES - start)
        indices = rng.integers(0, CONFIRMATION_PAIRS, size=(n, CONFIRMATION_PAIRS), dtype=np.int64)
        stream.update(indices.astype('<i8').tobytes(order='C'))
        totals[:, start:start + n] = values[:, indices].sum(axis=2, dtype=np.int64)
    return totals, stream.hexdigest()


def _confirmation_score(score, totals):
    import numpy as np
    result = dict(score)
    ci = (np.quantile(totals, [.025, .975], method='linear') / score['denominator']).tolist()
    radius = math.sqrt(math.log(40) / (2 * CONFIRMATION_PAIRS))
    hoeffding = [max(0., score['mean_score'] - radius), min(1., score['mean_score'] + radius)]
    degenerate = ci[0] == ci[1]
    gates = dict(_development_gates(score), paired_lower_bound_above_50pct=ci[0] > .5,
                 degenerate_interval_safeguard=not degenerate or hoeffding[0] > .5)
    result.update(paired_ci95=ci, bounded_pair_hoeffding_95=hoeffding, bootstrap_degenerate=degenerate,
                  useful_gain_criteria=gates, useful_gain=all(gates.values()),
                  integer_totals_sha256=hashlib.sha256(totals.astype('<i8').tobytes()).hexdigest())
    return result


def analyze_confirmation(confirmation_games, selected_candidate, evidence_gates_passed, *, boundary=lambda: None):
    import numpy as np
    require(evidence_gates_passed is True, 'all confirmation evidence gates must pass before analysis')
    require(selected_candidate in CANDIDATES, 'one prospectively selected candidate required')
    boundary()
    ordered = validated_games(confirmation_games, 'confirmation', selected_candidate)
    official, conservative = _score(ordered), _score(ordered, True)
    totals, stream_hash = _bootstrap(official['pair_numerators'], conservative['pair_numerators'], boundary)
    official, conservative = _confirmation_score(official, totals[0]), _confirmation_score(conservative, totals[1])
    useful = official['useful_gain'] and conservative['useful_gain']
    classification = ('useful_confirmed_gain_against_128_reference' if useful else
                      'official_useful_gain_not_robust_to_guard_forfeits' if official['useful_gain'] else
                      'useful_gain_not_established')
    boundary()
    return dict(schema='splendorust_exploration128_confirmation_v1', status='complete_verified_confirmation',
                selected_candidate=selected_candidate, candidate_cpuct=CPUCT[selected_candidate], reference_cpuct=.4,
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
                development_games_pooled=False, no_sample_extension=True,
                interpretation='Independent fixed-sample confirmation against unchanged128 on fresh paired setups. Official guard-forfeit score and conservative sensitivity must both pass the frozen useful-gain rules. This does not establish broader opponent generalization, one-second deployability, or automatic promotion.')


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


def verify_selection(development_games, result, *, boundary=lambda: None):
    ordered = validated_games(development_games, 'development')
    eligible = []
    for candidate in CANDIDATES:
        boundary()
        rows = [row for row in ordered if row['candidate'] == candidate]
        item = result['candidates'][candidate]
        _independent_arithmetic(rows, item['official'], False)
        conservative = _independent_arithmetic(rows, item['conservative'], True)
        seats = [sum(0 if row['reason'] == 'forfeit' else 1 if row['winner'] is None else
                     2 * int(row['winner'] == seat) for row in rows if row['challenger_seat'] == seat)
                 for seat in (0, 1)]
        if sum(conservative) >= 110 and min(seats) >= 50:
            eligible.append((sum(conservative), candidate))
    selected = sorted(eligible, key=lambda item: (-item[0], CANDIDATES.index(item[1])))[0][1] if eligible else None
    require(result['selected_candidate'] == selected, 'independent conservative eligibility/ranking differs')
    require(result == select_candidate(development_games, True, boundary=boundary), 'development report changed')
    return dict(status='passed', independent_integer_scores=True, independent_conservative_selection=True,
                selected_candidate=selected, verified_result_sha256=_digest(result))


def verify_confirmation(confirmation_games, selected_candidate, result, *, boundary=lambda: None):
    """Fresh per-replicate draw, Python integer sums and manual percentiles."""
    import numpy as np
    ordered = validated_games(confirmation_games, 'confirmation', selected_candidate)
    pairs = [_independent_arithmetic(ordered, result[key], conservative)
             for key, conservative in (('official', False), ('conservative', True))]
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = [[], []]
    stream = hashlib.sha256()
    for replicate in range(REPLICATES):
        if replicate % 500 == 0:
            boundary()
        indices = rng.integers(0, CONFIRMATION_PAIRS, size=CONFIRMATION_PAIRS, dtype=np.int64)
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
    require(result == analyze_confirmation(confirmation_games, selected_candidate, True, boundary=boundary),
            'confirmation report changed')
    return dict(status='passed', replicates=REPLICATES, independent_python_integer_sums=True,
                independent_linear_percentiles=True, independent_conservative_sensitivity=True,
                index_stream_sha256=stream.hexdigest(), verified_result_sha256=_digest(result),
                shared_trust_boundary='frozen NumPy PCG64 and bounded-integer primitive')

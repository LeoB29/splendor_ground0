"""Fixed complete-sample paired analysis for optimized 64 versus optimized 32.

The caller must seal and independently verify all games first. No partial
sample, optional stopping, quota selection or model/game execution is exposed.
"""
from collections import Counter
import hashlib
import json
import math

from .schedule import PAIRS, DOMAIN, expected_seed, validate_spec

REPLICATES = 50000
SEED = 2026102403


def require(value, message):
    if not value:
        raise ValueError('SEARCH_SCALE_OPT_STATISTICS: ' + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def runtime_contract():
    # Hash the actual loaded NumPy implementation and Python DLL.
    from tools.matched_strength_20261020.statistics import runtime_contract as contract
    return contract()


def validated_games(games):
    require(type(games) is list and len(games) == 2*PAIRS, 'complete 200-game sample required')
    ordered = []
    for value in games:
        spec = validate_spec(value)
        require(spec['stage'] == 'main', 'operational game in primary sample')
        seat = spec['challenger_seat']
        winner, reason = value.get('winner'), value.get('reason')
        require('winner' in value and (winner is None or type(winner) is int and winner in (0, 1)),
                'winner type/seat')
        require(reason in ('natural', 'forfeit'), 'unresolved or nonofficial ending')
        loser = value.get('loser')
        if reason == 'forfeit':
            require(type(loser) is int and loser in (0, 1) and winner == 1-loser,
                    'forfeit must lose for the responsible player')
            if 'failed_seat' in value:
                require(type(value['failed_seat']) is int and value['failed_seat'] == loser,
                        'forfeit failed seat differs')
            if 'failed_role' in value:
                require(value['failed_role'] == ('challenger' if loser == seat else 'opponent'),
                        'forfeit failed role differs')
        else:
            require(loser is None and value.get('failed_seat') is None and value.get('failed_role') is None,
                    'natural result has no forfeiting player')
        if 'complete' in value:
            require(value['complete'] is True, 'incomplete game')
        halfpoints = 1 if winner is None else 2*int(winner == seat)
        if 'challenger_score_halfpoints' in value:
            require(type(value['challenger_score_halfpoints']) is int and
                    value['challenger_score_halfpoints'] == halfpoints, 'derived score disagreement')
        ordered.append({**spec, 'winner': winner, 'reason': reason, 'loser': loser,
                        'challenger_score_halfpoints': halfpoints})
    ordered.sort(key=lambda row: (row['pair_index'], row['game_index']))
    require([(row['pair_index'], row['game_index']) for row in ordered] ==
            [(pair, 2*pair+game) for pair in range(PAIRS) for game in (0, 1)],
            'missing/duplicate game identities')
    return ordered


def _bootstrap(pair_values, boundary):
    import numpy as np
    values = np.asarray(pair_values, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = np.empty(REPLICATES, dtype=np.int64)
    index_hash = hashlib.sha256()
    for start in range(0, REPLICATES, 500):
        boundary()
        count = min(500, REPLICATES-start)
        indices = rng.integers(0, PAIRS, size=(count, PAIRS), dtype=np.int64)
        index_hash.update(indices.astype('<i8').tobytes(order='C'))
        totals[start:start+count] = values[indices].sum(axis=1, dtype=np.int64)
    return totals, index_hash.hexdigest()


def analyze_games(games, evidence_gates_passed, *, boundary=lambda: None):
    import numpy as np
    require(evidence_gates_passed is True, 'all evidence gates must pass before primary analysis')
    ordered = validated_games(games)
    pairs = [ordered[2*pair]['challenger_score_halfpoints'] + ordered[2*pair+1]['challenger_score_halfpoints']
             for pair in range(PAIRS)]
    totals, stream_hash = _bootstrap(pairs, boundary)
    numerator = sum(pairs)
    mean = numerator/400
    ci = (np.quantile(totals, [.025, .975], method='linear')/400).tolist()
    radius = math.sqrt(math.log(40)/(2*PAIRS))
    hoeffding = [max(0., mean-radius), min(1., mean+radius)]
    seats = {str(seat): sum(row['challenger_score_halfpoints'] for row in ordered
                           if row['challenger_seat'] == seat)/200 for seat in (0, 1)}
    degenerate = ci[0] == ci[1]
    challenger = mean >= .55 and ci[0] > .5 and min(seats.values()) >= .5
    opponent = mean <= .45 and ci[1] < .5 and max(seats.values()) <= .5
    if degenerate:
        challenger = challenger and hoeffding[0] > .5
        opponent = opponent and hoeffding[1] < .5
    counts = Counter(row['challenger_score_halfpoints'] for row in ordered)
    boundary()
    return {'schema': 'splendorust_search_scale_opt_statistics_v1', 'status': 'complete_verified_fixed_sample',
            'games': 200, 'pairs': 100, 'challenger_quota': 64, 'opponent_quota': 32,
            'pair_numerators': pairs, 'numerator': numerator, 'denominator': 400,
            'challenger_mean_score': mean, 'paired_ci95': ci,
            'bounded_pair_hoeffding_95': hoeffding, 'bootstrap_degenerate': degenerate,
            'by_challenger_seat': seats, 'challenger_wins': counts[2], 'draws': counts[1],
            'challenger_losses': counts[0],
            'forfeits': [{key: row[key] for key in ('pair_index', 'game_index', 'loser', 'winner')}
                         for row in ordered if row['reason'] == 'forfeit'],
            'bootstrap': {'replicates': REPLICATES, 'seed': SEED, 'generator': 'numpy.PCG64',
                          'numpy_version': np.__version__, 'draw_order': 'replicate_major_100_int64_indices',
                          'resampling_unit': 'swapped_seat_pair', 'quantile_method': 'linear',
                          'indices_sha256': stream_hash,
                          'integer_totals_sha256': hashlib.sha256(totals.astype('<i8').tobytes()).hexdigest()},
            'classification': 'useful_evidence_for_more_search' if challenger else
                              'useful_evidence_for_opponent_puct32' if opponent else 'inconclusive_or_small',
            'evidence_gates_passed': True, 'verified_score_rows_sha256': _digest(ordered),
            'interpretation': 'This exact paired matchup and allowance only; no equivalence, Elo, general-strength or automatic-promotion claim.'}


def verify_analysis(games, result, *, boundary=lambda: None):
    """Independent scalar integer sums and sorted-order percentile arithmetic.

    The frozen NumPy PCG64 and bounded-integer primitive are explicitly shared;
    draws are consumed per replicate instead of the primary vectorized batches.
    """
    import numpy as np
    ordered = validated_games(games)
    require(result.get('status') == 'complete_verified_fixed_sample' and
            result.get('evidence_gates_passed') is True, 'unaccepted primary result')
    values = [sum(row['challenger_score_halfpoints'] for row in ordered if row['pair_index'] == pair)
              for pair in range(PAIRS)]
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = []
    index_hash = hashlib.sha256()
    for replicate in range(REPLICATES):
        if replicate % 500 == 0:
            boundary()
        indices = rng.integers(0, PAIRS, size=PAIRS, dtype=np.int64)
        index_hash.update(indices.astype('<i8').tobytes())
        totals.append(sum(values[int(index)] for index in indices))
    sorted_totals = sorted(totals)
    def percentile(q):
        offset = (REPLICATES-1)*q
        low, high = math.floor(offset), math.ceil(offset)
        return (sorted_totals[low] + (sorted_totals[high]-sorted_totals[low])*(offset-low))/400
    interval = [percentile(.025), percentile(.975)]
    require(result['pair_numerators'] == values and result['numerator'] == sum(values) and
            result['denominator'] == 400 and result['challenger_mean_score'] == sum(values)/400,
            'independent pair integer arithmetic')
    require(result['paired_ci95'] == interval, 'independent interval interpolation')
    require(result['bootstrap']['indices_sha256'] == index_hash.hexdigest() and
            result['bootstrap']['integer_totals_sha256'] ==
            hashlib.sha256(np.asarray(totals, dtype='<i8').tobytes()).hexdigest(),
            'independent bootstrap stream/totals')
    require(result == analyze_games(games, True, boundary=boundary), 'result/report fields changed')
    return {'status': 'passed', 'replicates': REPLICATES, 'index_stream_sha256': index_hash.hexdigest(),
            'independent_python_integer_sums': True, 'independent_linear_percentiles': True,
            'shared_trust_boundary': 'frozen NumPy PCG64 and bounded-integer primitive'}

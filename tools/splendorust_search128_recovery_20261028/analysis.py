"""Fixed 200-pair research-guard analysis; no game/model execution.

The caller must first seal and independently verify every completed game.
No partial-sample analysis or adjustable sample size is exposed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

PAIRS = 200
GAMES = 2 * PAIRS
HALFPOINT_DENOMINATOR = 2 * GAMES
REPLICATES = 50000
SEED = 20261028128
FAILURE_POLICY = 'research_hard_guard_9s_internal_10s_complete_response'
ADMIN_FAILURES = ('research_internal_hard_guard', 'research_outer_hard_guard')
from .schedule import expected_seed



def require(value, message):
    if not value:
        raise ValueError("SEARCH128_STRENGTH_STATISTICS: " + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def runtime_contract():
    # This inspected helper hashes the loaded NumPy implementation and Python DLL.
    from tools.matched_strength_20261020.statistics import runtime_contract as contract
    return contract()


def validated_games(games):
    # Live result fields retain their original identity; explicitly validate then
    # project into the established fixed-pair integer analysis representation.
    from .schedule import validate_spec
    require(type(games) is list and len(games) == GAMES, 'complete 400-game sample required')
    projected=[]
    for game in games:
        validate_spec(game)
        require(game.get('stage')=='main','Operational result cannot enter primary analysis')
        require(game.get('challenger_mode')=='research' and game.get('opponent_mode')=='frozen' and
                game.get('challenger_quota')==128 and game.get('opponent_quota')==64,'Exact live identities')
        if 'search_seat' in game:
            require(type(game['search_seat']) is int and game['search_seat']==game['challenger_seat'], 'Contradictory projected seat')
        if 'challenger_simulations' in game:
            require(type(game['challenger_simulations']) is int and game['challenger_simulations']==128, 'Contradictory projected quota')
        projected.append(dict(game,search_seat=game['challenger_seat'],challenger_simulations=128))
    games=projected
    require(type(games) is list and len(games) == GAMES, "complete 400-game sample required")
    quotas = [g.get("challenger_simulations") for g in games if type(g) is dict]
    require(len(quotas) == GAMES and all(type(q) is int and q == 128 for q in quotas)
            and len(set(quotas)) == 1, "one frozen challenger quota for the complete sample")
    ordered = []
    for value in games:
        require(type(value) is dict, "game object")
        pair, game, seat = (value.get(k) for k in ("pair_index", "game_index", "search_seat"))
        require(all(type(x) is int for x in (pair, game, seat)) and
                0 <= pair < PAIRS and 0 <= game < 2 * PAIRS and game // 2 == pair and
                seat == (pair + game % 2) % 2,
                "frozen pair, game and swapped-seat schedule")
        require(value.get("stage", "main") == "main", "operational game in primary sample")
        if "seed_decimal" in value:
            require(value["seed_decimal"] == expected_seed("main", pair), "referee seed differs")
        require('winner' in value and value.get('complete') is True, 'Missing winner or incomplete game')
        winner, reason = value.get("winner"), value.get("reason")
        require(winner is None or type(winner) is int and winner in (0, 1), "winner type/seat")
        require(reason in ("natural", "forfeit"), "unresolved or nonofficial ending")
        loser = value.get("loser")
        if reason == "forfeit":
            require(type(loser) is int and loser in (0, 1) and winner == 1 - loser,
                    "forfeit must lose for the responsible player")
            require(value.get('administrative_forfeit') is True and
                    value.get('failure_policy') == FAILURE_POLICY and
                    value.get('failure') in ADMIN_FAILURES,
                    'Only classified administrative research hard-guard forfeits enter score')
            require(type(value.get('failed_seat')) is int and value['failed_seat']==loser and
                    value.get('failed_role') == ('challenger' if loser==seat else 'opponent'),
                    'Administrative forfeit role/seat inconsistent')
        else:
            require(loser is None, "natural result has no forfeiting player")
            require(value.get('administrative_forfeit', False) is False and
                    all(value.get(key) is None for key in ('failure_policy','failure','failed_role','failed_seat')),
                    'Natural result cannot carry a failure or administrative forfeit')
        if "complete" in value:
            require(value["complete"] is True, "incomplete game")
        halfpoints = 1 if winner is None else 2 * int(winner == seat)
        if "search_score_halfpoints" in value:
            require(type(value["search_score_halfpoints"]) is int and
                    value["search_score_halfpoints"] == halfpoints, "derived score disagreement")
        ordered.append({"pair_index": pair, "game_index": game, "search_seat": seat,
                        "winner": winner, "reason": reason, "loser": loser,
                        "search_score_halfpoints": halfpoints, "challenger_simulations": value["challenger_simulations"],
                        "administrative_forfeit": reason=='forfeit',
                        "failure_policy": value.get('failure_policy'), "failure": value.get('failure'),
                        "failed_role": value.get('failed_role'), "failed_seat": value.get('failed_seat')})
    ordered.sort(key=lambda r: (r["pair_index"], r["game_index"]))
    require([(r["pair_index"], r["game_index"]) for r in ordered] ==
            [(p, 2*p + g) for p in range(PAIRS) for g in (0, 1)], "missing/duplicate game identities")
    return ordered


def classify_score(mean, ci, seats, hoeffding, degenerate):
    """Prospectively fixed useful-effect rules; no optional sample extension."""
    search = mean >= .55 and ci[0] > .5 and min(seats.values()) >= .5
    reference = mean <= .45 and ci[1] < .5 and max(seats.values()) <= .5
    if degenerate:
        search = search and hoeffding[0] > .5
        reference = reference and hoeffding[1] < .5
    if search:return 'useful_evidence_for_128_under_research_guard'
    if reference:return 'useful_evidence_for_64_under_research_guard'
    if ci[0] > .5 and (not degenerate or hoeffding[0] > .5):
        return 'supported_128_advantage_not_meeting_useful_criteria'
    if ci[1] < .5 and (not degenerate or hoeffding[1] < .5):
        return 'supported_64_advantage_not_meeting_useful_criteria'
    return 'inconclusive'


def administrative_sensitivity(ordered):
    """Full-sample score bounds, not natural-only estimates or confidence intervals."""
    admin = [r for r in ordered if r['administrative_forfeit']]
    fixed_points = sum(r['search_score_halfpoints'] for r in ordered if not r['administrative_forfeit'])
    lower, upper = fixed_points, fixed_points + 2*len(admin)
    return dict(administrative_games=len(admin), complete_sample_games=GAMES,
                denominator=HALFPOINT_DENOMINATOR, lower_numerator=lower, upper_numerator=upper,
                all_administrative_candidate_losses=lower/HALFPOINT_DENOMINATOR,
                all_administrative_candidate_wins=upper/HALFPOINT_DENOMINATOR,
                interpretation='Descriptive full-sample reassignment bounds; not counterfactual game outcomes, confidence intervals, or a replacement primary estimate.',
                natural_only_strength_estimate=False)


def _bootstrap(pair_values, boundary):
    import numpy as np
    values = np.asarray(pair_values, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = np.empty(REPLICATES, dtype=np.int64)
    index_hash = hashlib.sha256()
    for start in range(0, REPLICATES, 500):
        boundary()
        n = min(500, REPLICATES - start)
        indices = rng.integers(0, PAIRS, size=(n, PAIRS), dtype=np.int64)
        index_hash.update(indices.astype("<i8").tobytes(order="C"))
        totals[start:start + n] = values[indices].sum(axis=1, dtype=np.int64)
    return totals, index_hash.hexdigest()


def analyze_games(games, evidence_gates_passed, *, boundary=lambda: None):
    import numpy as np
    require(evidence_gates_passed is True, "all evidence gates must pass before primary analysis")
    ordered = validated_games(games)
    pairs = [ordered[2*p]["search_score_halfpoints"] + ordered[2*p+1]["search_score_halfpoints"]
             for p in range(PAIRS)]
    totals, stream_hash = _bootstrap(pairs, boundary)
    numerator = sum(pairs)
    mean = numerator / HALFPOINT_DENOMINATOR
    ci = (np.quantile(totals, [.025, .975], method="linear") / HALFPOINT_DENOMINATOR).tolist()
    radius = math.sqrt(math.log(40) / (2 * PAIRS))
    hoeffding = [max(0., mean - radius), min(1., mean + radius)]
    seats = {str(seat): sum(r["search_score_halfpoints"] for r in ordered
                           if r["search_seat"] == seat) / (2*PAIRS) for seat in (0, 1)}
    degenerate = ci[0] == ci[1]
    classification = classify_score(mean,ci,seats,hoeffding,degenerate)
    counts = Counter(row["search_score_halfpoints"] for row in ordered)
    admin = [row for row in ordered if row['administrative_forfeit']]
    sensitivity = administrative_sensitivity(ordered)
    boundary()
    return {"schema": "splendorust_search128_strength_statistics_v1", "status": "complete_verified_fixed_sample",
            "games": GAMES, "pairs": PAIRS, "challenger_simulations": ordered[0]["challenger_simulations"],
            "reference_simulations": 64, "challenger_world_pool": 3, "reference_world_pool": 3, "pair_numerators": pairs, "numerator": numerator,
            "denominator": HALFPOINT_DENOMINATOR, "search_mean_score": mean, "paired_ci95": ci,
            "bounded_pair_hoeffding_95": hoeffding, "bootstrap_degenerate": degenerate,
            "by_search_seat": seats, "search_wins": counts[2], "draws": counts[1],
            "search_losses": counts[0],
            "forfeits": [{k: r[k] for k in ("pair_index", "game_index", "loser", "winner", "failure", "failure_policy", "administrative_forfeit")}
                         for r in admin],
            "natural_games": GAMES-len(admin), "administrative_forfeits": len(admin),
            "administrative_forfeits_by_player": {role:sum(r['failed_role']==role for r in admin) for role in ('challenger','opponent')},
            "administrative_failure_policy": FAILURE_POLICY,
            "administrative_forfeit_sensitivity": sensitivity,
            "administrative_forfeits_may_affect_strength_interpretation": bool(admin),
            "research_guard_seconds": {"internal":9.0,"complete_response":10.0},
            "timing_exceedances_seconds_are_descriptive_only":[.7,.9,1.0],
            "one_second_deployability_established":False,"automatic_promotion":False,
            "bootstrap": {"replicates": REPLICATES, "seed": SEED, "generator": "numpy.PCG64",
                          "numpy_version": np.__version__, "draw_order": "replicate_major_200_int64_indices",
                          "resampling_unit": "swapped_seat_pair", "quantile_method": "linear",
                          "indices_sha256": stream_hash,
                          "integer_totals_sha256": hashlib.sha256(totals.astype("<i8").tobytes()).hexdigest()},
            "classification": classification,
            "comparison": "128 versus64 total simulations, three root worlds in both; increased compute, not fixed cost",
            "useful_candidate_criteria": {"score_at_least_55pct":mean>=.55,
                "paired_lower_bound_above_50pct":ci[0]>.5,
                "each_seat_score_at_least_50pct":min(seats.values())>=.5,
                "degenerate_interval_safeguard":not degenerate or hoeffding[0]>.5,
                "complete_verification":True},
            "evidence_gates_passed": True, "verified_score_rows_sha256": _digest(ordered),
            "interpretation": "Fixed-sample score for these exact configurations under a9s internal/10s complete-response administrative guard, including its forfeits. Not pure unconstrained strength, one-second deployability, equivalence, Elo, general strength, or automatic promotion. Any guard forfeits may confound move-quality interpretation; full-sample sensitivity bounds do not replace the primary score."}


def verify_analysis(games, result, *, boundary=lambda: None):
    """Separate per-replicate draw/explicit integer sum and hand percentile path.

    Shares the frozen NumPy PCG64 primitive but independently consumes its index
    stream, sums Python integers, and interpolates sorted integer order statistics.
    """
    import numpy as np
    ordered = validated_games(games)
    require(result.get("status") == "complete_verified_fixed_sample" and
            result.get("evidence_gates_passed") is True, "unaccepted primary result")
    values = [sum(r["search_score_halfpoints"] for r in ordered if r["pair_index"] == pair)
              for pair in range(PAIRS)]
    rng = np.random.Generator(np.random.PCG64(SEED))
    totals = []
    index_hash = hashlib.sha256()
    for replicate in range(REPLICATES):
        if replicate % 500 == 0:
            boundary()
        indices = rng.integers(0, PAIRS, size=PAIRS, dtype=np.int64)
        index_hash.update(indices.astype("<i8").tobytes())
        totals.append(sum(values[int(index)] for index in indices))
    sorted_totals = sorted(totals)
    def percentile(q):
        offset = (REPLICATES - 1) * q
        lo, hi = math.floor(offset), math.ceil(offset)
        return (sorted_totals[lo] + (sorted_totals[hi] - sorted_totals[lo]) * (offset - lo)) / 800
    interval = [percentile(.025), percentile(.975)]
    require(result["pair_numerators"] == values and result["numerator"] == sum(values) and
            result["games"]==400 and result['pairs']==200 and
            result["denominator"] == 800 and result["search_mean_score"] == sum(values) / 800,
            "independent pair integer arithmetic")
    require(result["paired_ci95"] == interval, "independent interval interpolation")
    require(result["bootstrap"]["indices_sha256"] == index_hash.hexdigest() and
            result["bootstrap"]["integer_totals_sha256"] ==
            hashlib.sha256(np.asarray(totals, dtype="<i8").tobytes()).hexdigest(),
            "independent bootstrap stream/totals")
    # Independent direct outcome arithmetic, including unchanged full-sample denominator.
    require(result['by_search_seat']=={str(s):sum(r['search_score_halfpoints'] for r in ordered if r['search_seat']==s)/400
                                      for s in (0,1)}, 'Independent seat arithmetic')
    admin_count=sum(r['reason']=='forfeit' for r in ordered)
    natural_points=sum((1 if r['winner'] is None else 2*int(r['winner']==r['search_seat']))
                       for r in ordered if r['reason']=='natural')
    sensitivity=result['administrative_forfeit_sensitivity']
    require(result['administrative_forfeits']==admin_count and result['natural_games']==400-admin_count and
            sensitivity['lower_numerator']==natural_points and
            sensitivity['upper_numerator']==natural_points+2*admin_count and
            sensitivity['denominator']==800 and
            sensitivity['all_administrative_candidate_losses']==natural_points/800 and
            sensitivity['all_administrative_candidate_wins']==(natural_points+2*admin_count)/800,
            'Independent full-sample administrative sensitivity')
    # Recheck every interpretation/reporting field after the independent path.
    require(result == analyze_games(games, True, boundary=boundary), "result/report fields changed")
    return {"status": "passed", "replicates": REPLICATES, "index_stream_sha256": index_hash.hexdigest(),
            "independent_python_integer_sums": True, "independent_linear_percentiles": True,
            "independent_full_sample_sensitivity":True,
            "shared_trust_boundary": "frozen NumPy PCG64 and bounded-integer primitive"}

"""Fixed 100-pair analysis for the one-second comparison; no game/model execution.

The caller must first seal and independently verify every completed game.
No partial-sample analysis or adjustable sample size is exposed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

PAIRS = 100
REPLICATES = 50000
SEED = 2026102402
DOMAIN = "splendorust_puct_budget_1s_20261024_v1"
OPERATIONAL_DOMAIN = "splendorust_puct_budget_1s_20261024_v1_operational"


def require(value, message):
    if not value:
        raise ValueError("SEARCH_SCALE_STATISTICS: " + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def runtime_contract():
    # This inspected helper hashes the loaded NumPy implementation and Python DLL.
    from tools.matched_strength_20261020.statistics import runtime_contract as contract
    return contract()


def expected_seed(stage, pair_index):
    require(stage in ("operational", "main") and type(pair_index) is int and
            0 <= pair_index < (3 if stage == "operational" else PAIRS), "schedule identity")
    encoded = json.dumps([OPERATIONAL_DOMAIN if stage == "operational" else DOMAIN, stage, pair_index], separators=(",", ":"), ensure_ascii=True)
    return str(int.from_bytes(hashlib.sha256(encoded.encode("utf-8")).digest(), "big"))


def validated_games(games):
    require(type(games) is list and len(games) == 2 * PAIRS, "complete 200-game sample required")
    quotas = [g.get("challenger_simulations") for g in games if type(g) is dict]
    require(len(quotas) == 200 and all(type(q) is int and q in (32, 64) for q in quotas)
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
        winner, reason = value.get("winner"), value.get("reason")
        require(winner is None or type(winner) is int and winner in (0, 1), "winner type/seat")
        require(reason in ("natural", "forfeit"), "unresolved or nonofficial ending")
        loser = value.get("loser")
        if reason == "forfeit":
            require(type(loser) is int and loser in (0, 1) and winner == 1 - loser,
                    "forfeit must lose for the responsible player")
        else:
            require(loser is None, "natural result has no forfeiting player")
        if "complete" in value:
            require(value["complete"] is True, "incomplete game")
        halfpoints = 1 if winner is None else 2 * int(winner == seat)
        if "search_score_halfpoints" in value:
            require(type(value["search_score_halfpoints"]) is int and
                    value["search_score_halfpoints"] == halfpoints, "derived score disagreement")
        ordered.append({"pair_index": pair, "game_index": game, "search_seat": seat,
                        "winner": winner, "reason": reason, "loser": loser,
                        "search_score_halfpoints": halfpoints, "challenger_simulations": value["challenger_simulations"]})
    ordered.sort(key=lambda r: (r["pair_index"], r["game_index"]))
    require([(r["pair_index"], r["game_index"]) for r in ordered] ==
            [(p, 2*p + g) for p in range(PAIRS) for g in (0, 1)], "missing/duplicate game identities")
    return ordered


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
    mean = numerator / 400
    ci = (np.quantile(totals, [.025, .975], method="linear") / 400).tolist()
    radius = math.sqrt(math.log(40) / (2 * PAIRS))
    hoeffding = [max(0., mean - radius), min(1., mean + radius)]
    seats = {str(seat): sum(r["search_score_halfpoints"] for r in ordered
                           if r["search_seat"] == seat) / 200 for seat in (0, 1)}
    degenerate = ci[0] == ci[1]
    search = mean >= .55 and ci[0] > .5 and min(seats.values()) >= .5
    reference = mean <= .45 and ci[1] < .5 and max(seats.values()) <= .5
    if degenerate:
        search = search and hoeffding[0] > .5
        reference = reference and hoeffding[1] < .5
    counts = Counter(row["search_score_halfpoints"] for row in ordered)
    boundary()
    return {"schema": "splendorust_search_scale_statistics_v1", "status": "complete_verified_fixed_sample",
            "games": 200, "pairs": 100, "challenger_simulations": ordered[0]["challenger_simulations"],
            "reference_simulations": 16, "pair_numerators": pairs, "numerator": numerator,
            "denominator": 400, "search_mean_score": mean, "paired_ci95": ci,
            "bounded_pair_hoeffding_95": hoeffding, "bootstrap_degenerate": degenerate,
            "by_search_seat": seats, "search_wins": counts[2], "draws": counts[1],
            "search_losses": counts[0],
            "forfeits": [{k: r[k] for k in ("pair_index", "game_index", "loser", "winner")}
                         for r in ordered if r["reason"] == "forfeit"],
            "bootstrap": {"replicates": REPLICATES, "seed": SEED, "generator": "numpy.PCG64",
                          "numpy_version": np.__version__, "draw_order": "replicate_major_100_int64_indices",
                          "resampling_unit": "swapped_seat_pair", "quantile_method": "linear",
                          "indices_sha256": stream_hash,
                          "integer_totals_sha256": hashlib.sha256(totals.astype("<i8").tobytes()).hexdigest()},
            "classification": "useful_evidence_for_more_search" if search else
                              "useful_evidence_for_reference_puct16" if reference else "inconclusive_or_small",
            "evidence_gates_passed": True, "verified_score_rows_sha256": _digest(ordered),
            "interpretation": "This exact paired matchup and allowance only; no equivalence, Elo, general-strength or automatic-promotion claim."}


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
        return (sorted_totals[lo] + (sorted_totals[hi] - sorted_totals[lo]) * (offset - lo)) / 400
    interval = [percentile(.025), percentile(.975)]
    require(result["pair_numerators"] == values and result["numerator"] == sum(values) and
            result["denominator"] == 400 and result["search_mean_score"] == sum(values) / 400,
            "independent pair integer arithmetic")
    require(result["paired_ci95"] == interval, "independent interval interpolation")
    require(result["bootstrap"]["indices_sha256"] == index_hash.hexdigest() and
            result["bootstrap"]["integer_totals_sha256"] ==
            hashlib.sha256(np.asarray(totals, dtype="<i8").tobytes()).hexdigest(),
            "independent bootstrap stream/totals")
    # Recheck every interpretation/reporting field after the independent path.
    require(result == analyze_games(games, True, boundary=boundary), "result/report fields changed")
    return {"status": "passed", "replicates": REPLICATES, "index_stream_sha256": index_hash.hexdigest(),
            "independent_python_integer_sums": True, "independent_linear_percentiles": True,
            "shared_trust_boundary": "frozen NumPy PCG64 and bounded-integer primitive"}

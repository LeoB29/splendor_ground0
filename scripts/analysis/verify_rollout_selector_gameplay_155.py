"""155 independent full-game, information-set, continuation and paired audit.

Shares the frozen engine/scorer and separately implemented 149 branch replay,
154 catalog-minus-known world builder and canonical candidate reconstruction.
Does not use the 155 selection, scheduling, gameplay or statistics implementation.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import math
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_rollout_selector_gameplay as p
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

spec = importlib.util.spec_from_file_location("audit154_for155", "scripts/analysis/verify_rollout_selector_154.py")
a154 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a154)
compare = a154.compare


def independent_spec(stage, index):
    assert stage in ("probe", "main") and 0 <= index < (8 if stage == "probe" else 400)
    return {"index": index, "pair": index//2, "candidate_seat": index%2,
            "environment_seed": (1559000000 if stage == "probe" else 1550000000)+index//2,
            "search_stream": p.old.digest(["155-independent-search-stream-v1", stage, index])}


def independent_schedule(state, stream):
    key = p.old.digest(canonical_public_belief_key(state, state.current_player))
    return {"belief_sha256": key, "candidate_nonce": p.old.digest(["155-candidate-v1", stream, key]),
            "world_seeds": ["155-world-v1|%s|%s|%02d" % (stream, key, i) for i in range(32)]}


def hidden_alternative(state):
    changed = state.copy_shallow()
    other = changed.players[1-state.current_player]
    for cards in changed.hidden_tier_decks.values():
        cards.reverse()
    for i, card in enumerate(other.reserved_cards):
        deck = changed.hidden_tier_decks[card.tier]
        if other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE and deck:
            other.reserved_cards[i], deck[0] = deck[0], other.reserved_cards[i]
    return changed


def audit_root(stage, index, state, move, stream, binding):
    path = p.OUTPUT/stage/"roots"/f"game_{index:04d}"/f"turn_{state.turn_index:03d}.json"
    assert p.old.file_hash(path) == move["root_sha256"]
    assert path.stat().st_mtime_ns >= (p.OUTPUT/f"{stage}_launch.json").stat().st_mtime_ns
    record = p.old.read_json(path)
    assert record["manifest"] == binding and "failure" not in record
    assert p.old.digest(record["snapshot"]) == p.old.digest(p.old.serialize_state_snapshot(state))
    schedule = independent_schedule(state, stream)
    assert record["schedule"] == schedule
    candidates = a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
    assert record["candidates"] == candidates
    legal = p.old.SplendorEnv().legal_actions(state)
    scores = [float(v) for v in p.old._RAW.score_actions(state, legal)]
    assert all(math.isfinite(v) for v in scores)
    compare(record["scores"], scores)
    assert candidates["actions"][0] == p.old.serialize_action(legal[max(range(len(legal)), key=lambda i: scores[i])])
    # Audit each actual root's information invariance, not just hand-picked cases.
    changed = hidden_alternative(state)
    assert independent_schedule(changed, stream) == schedule
    changed_legal = p.old.SplendorEnv().legal_actions(changed)
    assert changed_legal == legal
    compare([float(v) for v in p.old._RAW.score_actions(changed, changed_legal)], scores)
    assert a154.independent_candidates(changed, "155-live-v1", schedule["candidate_nonce"]) == candidates
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert record["privacy"] == privacy and len(record["worlds"]) == 32
    totals, reasons, moves = [0.]*len(candidates["actions"]), Counter(), 0
    for seed, row in zip(schedule["world_seeds"], record["worlds"]):
        assert row["seed"] == seed and len(row["outcomes"]) == len(totals)
        world = a154.independent_world(state, seed)
        world_snapshot = p.old.serialize_state_snapshot(world)
        assert p.old.digest(world_snapshot) == row["state_sha256"]
        assert p.old.serialize_state_snapshot(a154.independent_world(changed, seed)) == world_snapshot
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        for j, (action, outcome) in enumerate(zip(candidates["actions"], row["outcomes"])):
            decoded = p.old.decode_action(action)
            assert decoded in p.old.SplendorEnv().legal_actions(world)
            replayed = a154.a149.replay_branch(world, decoded, state.current_player)
            assert replayed == outcome and replayed["reason"] in ("completed", "forced_pass_deadlock")
            totals[j] += replayed["wdl"]
            reasons[replayed["reason"]] += 1
            moves += replayed["turns"]
    means = [v/32 for v in totals]
    choice = max(range(len(means)), key=lambda i: means[i])
    compare(record["means"], means)
    assert record["choice"] == choice and move["action"] == candidates["actions"][choice]
    for key in ("wall", "cpu"):
        assert math.isfinite(record["timing"][key]) and record["timing"][key] >= 0
    return {"index": index, "seat": state.current_player, "turn": state.turn_index,
            "choice": choice, "support": len(totals), "canonical_support": candidates["canonical_support"],
            "legal_support": len(legal), "privacy": privacy, "timing": record["timing"],
            "continuations": sum(reasons.values()), "moves": moves, "reasons": dict(reasons)}


def audit_game(stage, index, sha, binding):
    wall, cpu = time.perf_counter(), time.process_time()
    path = p.OUTPUT/stage/"games"/f"game_{index:04d}.json"
    assert p.old.file_hash(path) == sha and path.stat().st_mtime_ns >= (p.OUTPUT/f"{stage}_launch.json").stat().st_mtime_ns
    record = p.old.read_json(path)
    spec = independent_spec(stage, index)
    assert record["spec"] == spec and record["manifest"] == binding and "failure" not in record
    env = p.old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    initial_sha = p.old.digest(p.old.serialize_state_snapshot(state))
    assert record["initial_state_sha256"] == initial_sha
    visits, hasher, passes, idle = Counter(), hashlib.sha256(), 0, 0
    roots, raw = [], []
    for n, move in enumerate(record["trace"]):
        assert not state.terminal and state.turn_index < 400
        visits[p.old.state_signature(state)] += 1
        assert visits[p.old.state_signature(state)] < 4
        assert move["state_sha256"] == p.old.digest(p.old.serialize_state_snapshot(state))
        assert move["turn"] == state.turn_index == n and move["actor"] == state.current_player
        legal = env.legal_actions(state)
        if state.current_player == spec["candidate_seat"]:
            roots.append(audit_root(stage, index, state, move, spec["search_stream"], binding))
        else:
            assert "root_sha256" not in move
            scores = p.old._RAW.score_actions(state, legal)
            assert all(math.isfinite(float(v)) for v in scores)
            action = legal[max(range(len(legal)), key=lambda i: scores[i])]
            assert p.old.serialize_action(action) == move["action"]
            assert all(math.isfinite(move["timing"][k]) and move["timing"][k] >= 0 for k in ("wall", "cpu"))
            raw.append(move["timing"])
        action = p.old.decode_action(move["action"])
        assert action in legal
        hasher.update(p.old.digest(move["action"]).encode())
        nxt = env.step(state, action)
        passes = passes+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if p.old.is_progress_transition(state, nxt, action) else idle+1
        state = nxt
        if n < len(record["trace"])-1:
            assert passes < 2 and idle < 60
    assert record["turns"] == len(record["trace"]) and record["trace_sha256"] == hasher.hexdigest()
    assert p.old.digest(record["final_state"]) == p.old.digest(p.old.serialize_state_snapshot(state))
    assert record["final_state_sha256"] == p.old.digest(record["final_state"])
    if record["reason"] == "completed":
        assert state.terminal and idle < 60 and passes < 2
        winner = state.winner
    else:
        assert record["reason"] == "forced_pass_deadlock" and passes == 2
        # Independently apply score then fewest bought-card tiebreak.
        ranks = [(player.score, -len(player.purchased_cards)) for player in state.players]
        winner = None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)
    assert record["winner"] == winner
    seat = spec["candidate_seat"]
    score = .5 if winner is None else float(winner == seat)
    row = {"index": index, "pair": index//2, "seat": seat, "score": score,
           "margin": state.players[seat].score-state.players[1-seat].score,
           "turns": record["turns"], "reason": record["reason"], "timing": record["timing"]}
    assert all(math.isfinite(row["timing"][k]) and row["timing"][k] >= 0 for k in ("wall", "cpu"))
    return {"game": row, "roots": roots, "raw_timings": raw, "initial_sha256": initial_sha,
            "audit_timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}}


def linear_quantile(values, q):
    ordered = sorted(values)
    point = (len(ordered)-1)*q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(point-lo)


def independent_statistics(rows):
    assert len(rows) == 400 and [r["index"] for r in rows] == list(range(400))
    values = []
    for pair in range(200):
        a, b = rows[pair*2:pair*2+2]
        assert a["pair"] == b["pair"] == pair and a["seat"] == 0 and b["seat"] == 1
        assert a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.)
        values.append((a["score"]+b["score"])/2)
    rng, boot = np.random.default_rng(1553000000), []
    # Explicit resample occurrence counts; no matrix product / NumPy BLAS.
    for _ in range(50):
        indices = rng.integers(0, 200, size=(1000, 200))
        for sample in indices:
            counts = np.bincount(sample, minlength=200)
            boot.append(float((counts*np.asarray(values)).sum())/200)
    ci = [linear_quantile(boot, q) for q in (.025, .975)]
    def summary(part):
        return {"games": len(part), "share": sum(r["score"] for r in part)/len(part),
                "wdl": {"wins": sum(r["score"] == 1 for r in part), "draws": sum(r["score"] == .5 for r in part),
                        "losses": sum(r["score"] == 0 for r in part)},
                "mean_margin": sum(r["margin"] for r in part)/len(part),
                "mean_turns": sum(r["turns"] for r in part)/len(part)}
    result = summary(rows)
    gates = {"material_share": result["share"] >= .55, "positive_lower_bound": ci[0] > .50}
    return {**result, "pairs": 200, "ci95": ci, "statistical_gates": gates,
            "statistical_pass": all(gates.values()), "seats": {str(s): summary(rows[s::2]) for s in (0, 1)}}


def independent_operational(data):
    roots, games = data["roots"], data["games"]
    def counts(part):
        n, changes = len(part), sum(r["choice"] > 0 for r in part)
        return {"roots": n, "changes": changes, "change_fraction": changes/n if n else None,
                "support_counts": dict(Counter(str(r["support"]) for r in part)),
                "own_private_roots": sum(r["privacy"]["own"] > 0 for r in part),
                "opponent_private_roots": sum(r["privacy"]["opponent"] > 0 for r in part)}
    def distribution(values):
        assert values and all(math.isfinite(v) and v >= 0 for v in values)
        return {"count": len(values), "mean": sum(values)/len(values), "p95": linear_quantile(values, .95),
                "max": max(values), "sum": sum(values)}
    inner = Counter()
    for r in roots:
        inner.update(r["reasons"])
    return {**counts(roots), "continuations": sum(r["continuations"] for r in roots),
            "continuation_moves": sum(r["moves"] for r in roots), "inner_reasons": dict(inner),
            "outer_reasons": dict(Counter(g["reason"] for g in games)),
            "decision_time": {k: distribution([r["timing"][k] for r in roots]) for k in ("wall", "cpu")},
            "raw_decision_time": {k: distribution([r[k] for r in data["raw_timings"]]) for k in ("wall", "cpu")},
            "game_time": {k: distribution([r["timing"][k] for r in games]) for k in ("wall", "cpu")},
            "turns": distribution([g["turns"] for g in games]),
            "seats": {str(s): counts([r for r in roots if r["seat"] == s]) for s in (0, 1)},
            "phases": {str(s): counts([r for r in roots if (0 if r["turn"] < 20 else 1 if r["turn"] < 40 else 2) == s]) for s in range(3)},
            "privacy_strata": {str(s): counts([r for r in roots if bool(r["privacy"]["own"] or r["privacy"]["opponent"]) == s]) for s in (False, True)}}


def verify(stage):
    wall = time.perf_counter()
    binding = p.check_identity()
    schedule = {s: [independent_spec(s, i) for i in range(8 if s == "probe" else 400)] for s in ("probe", "main")}
    assert p.old.read_json(p.OUTPUT/"schedule.json") == schedule
    frozen = p.old.read_json(p.OUTPUT/"freeze.json")
    assert frozen["manifest"] == binding and frozen["schedule_sha256"] == p.old.file_hash(p.OUTPUT/"schedule.json")
    assert frozen["tests_sha256"] == p.verify_test_evidence()
    launch = p.old.read_json(p.OUTPUT/f"{stage}_launch.json")
    assert launch["manifest"] == binding and launch["schedule_sha256"] == frozen["schedule_sha256"]
    if stage == "probe":
        assert launch["upstream"] == {"freeze_sha256": p.old.file_hash(p.OUTPUT/"freeze.json")}
    else:
        assert launch["upstream"] == {"preflight_sha256": p.old.file_hash(p.OUTPUT/"preflight.json")}
        preflight = p.old.read_json(p.OUTPUT/"preflight.json")
        assert preflight["passed"] and preflight["manifest"] == binding
        assert preflight["audit_sha256"] == p.old.file_hash(p.OUTPUT/"probe_audit.json")
    lock_path = p.OUTPUT/f"{stage}_games_lock.json"
    lock = p.old.read_json(lock_path)
    n = len(schedule[stage])
    assert lock["manifest"] == binding and lock["launch_sha256"] == p.old.file_hash(p.OUTPUT/f"{stage}_launch.json")
    assert set(lock["files"]) == {str(i) for i in range(n)}
    assert set((p.OUTPUT/stage/"games").glob("*.json")) == {p.game_path(stage, i) for i in range(n)}
    audited = {}
    # Individual audit certificates allow explicit crash-resume without rerunning
    # already verified complete games; certificate hashes bind their entire tree.
    with ProcessPoolExecutor(max_workers=8, initializer=p.old.initialize_worker) as pool:
        pending = {}
        for i in range(n):
            certificate = p.OUTPUT/stage/"replay"/f"game_{i:04d}.json"
            if certificate.exists():
                saved = p.old.read_json(certificate)
                assert saved["manifest"] == binding and saved["game_sha256"] == lock["files"][str(i)]
                assert p.old.file_hash(p.game_path(stage, i)) == saved["game_sha256"]
                for row in p.old.read_json(p.game_path(stage, i))["trace"]:
                    if "root_sha256" in row:
                        assert p.old.file_hash(p.root_path(stage, i, row["turn"])) == row["root_sha256"]
                audited[i] = saved["reconstruction"]
            else:
                pending[pool.submit(audit_game, stage, i, lock["files"][str(i)], binding)] = i
        for future in as_completed(pending):
            i = pending[future]
            audited[i] = future.result()
            p.old.save_new(p.OUTPUT/stage/"replay"/f"game_{i:04d}.json",
                           {"manifest": binding, "game_sha256": lock["files"][str(i)], "reconstruction": audited[i]})
            p.progress(f"{stage}_replay", games_complete=len(audited), games_total=n, elapsed_seconds=time.perf_counter()-wall)
    for i in range(0, n, 2):
        assert audited[i]["initial_sha256"] == audited[i+1]["initial_sha256"]
    data = {"games": [audited[i]["game"] for i in range(n)],
            "roots": [r for i in range(n) for r in audited[i]["roots"]],
            "raw_timings": [r for i in range(n) for r in audited[i]["raw_timings"]]}
    expected_roots = {p.root_path(stage, r["index"], r["turn"]) for r in data["roots"]}
    assert set((p.OUTPUT/stage/"roots").rglob("*.json")) == expected_roots
    output = {"passed": True, "manifest": binding, "games_lock_sha256": p.old.file_hash(lock_path),
              "games": n, "outer_moves": sum(g["turns"] for g in data["games"]), "roots": len(data["roots"]),
              "continuations": sum(r["continuations"] for r in data["roots"]),
              "continuation_moves": sum(r["moves"] for r in data["roots"]),
              "wall_seconds": time.perf_counter()-wall,
              "worker_cpu_seconds": sum(v["audit_timing"]["cpu"] for v in audited.values()),
              "shared_frozen_engine": True, "every_root_information_invariance": True,
              "replay_files": {str(i): p.old.file_hash(p.OUTPUT/stage/"replay"/f"game_{i:04d}.json") for i in range(n)}}
    if stage == "main":
        compare(data, p.old.read_json(p.OUTPUT/"observations.json"))
        result = p.old.read_json(p.OUTPUT/"result.json")
        stats = independent_statistics(data["games"])
        compare(stats, {k: result[k] for k in stats})
        compare(independent_operational(data), result["operational"])
        assert output["continuations"] <= 10240000
        output.update(result_sha256=p.old.file_hash(p.OUTPUT/"result.json"), observations_sha256=p.old.file_hash(p.OUTPUT/"observations.json"))
    p.check_identity()
    p.old.save_new(p.OUTPUT/f"{stage}_audit.json", output)
    p.progress(f"{stage}_audit_passed", games=n, roots=output["roots"], continuations=output["continuations"], wall_seconds=output["wall_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("probe", "main"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify(args.stage)

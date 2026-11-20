"""160 independent outer-game audit and prospectively sampled inner replay.

Every outer action, legal state, proposal set, hidden world, outcome field,
mean and chosen action is checked. Exactly two hash-selected worlds per root
receive complete independent replay of all offered actions. This is explicitly
not a full replay of every inner trajectory.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import math
import multiprocessing
import re
import sys
import time

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY))

from splendor_ai.alphazero import run_rollout_selector_gameplay as baseline
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE
from scripts.analysis import verify_learned_continuation_158 as a158

old = baseline.old
OUTPUT = Path("outputs/world_budget_160")
PRODUCTION_OUTPUT = OUTPUT.resolve()
compare = a158.compare


def independent_spec(stage, index):
    assert stage in ("probe", "main", "test")
    assert type(index) is int and 0 <= index < {"probe": 8, "main": 400, "test": 2}[stage]
    pair = index//2
    return {"index": index, "pair": pair, "candidate_seat": index % 2,
            "environment_seed": {"probe": 1609000000, "main": 1600000000, "test": 1609800000}[stage]+pair,
            "search_stream": old.digest(["160-root-stream-v1", stage, pair])}


def independent_schedule(state, stream, worlds):
    assert worlds in (32, 128)
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    return {"belief_sha256": belief,
            "candidate_nonce": old.digest(["155-candidate-v1", stream, belief]),
            "world_seeds": [f"155-world-v1|{stream}|{belief}|{j:02d}" for j in range(worlds)]}


def replay_indices(stage, index, turn, worlds):
    """The exact prospective hash namespace is fixed in the160 protocol."""
    assert worlds in (32, 128)
    return sorted(sorted(range(worlds), key=lambda wi: old.digest(
        ["160-audit-world-v1", stage, index, turn, wi]))[:2])


def _sha(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _timing(values):
    assert set(values) == {"wall", "cpu"}
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())


def _outcome(outcome, actor, start_turn):
    assert set(outcome) == {"reason", "winner", "turns", "trace_sha256", "final_state_sha256", "wdl"}
    assert outcome["reason"] in ("completed", "forced_pass_deadlock")
    assert outcome["winner"] is None or type(outcome["winner"]) is int and outcome["winner"] in (0, 1)
    assert type(outcome["turns"]) is int and 1 <= outcome["turns"] <= 400-start_turn
    assert type(outcome["wdl"]) in (int, float) and math.isfinite(outcome["wdl"])
    assert outcome["wdl"] == (0. if outcome["winner"] is None else 1. if outcome["winner"] == actor else -1.)
    _sha(outcome["trace_sha256"])
    _sha(outcome["final_state_sha256"])


def audit_root(stage, index, state, move, spec, binding, base=OUTPUT):
    path = Path(base)/stage/"roots"/f"game_{index:04d}"/f"turn_{state.turn_index:03d}.json"
    assert old.file_hash(path) == move["root_sha256"]
    assert path.stat().st_mtime_ns >= (Path(base)/f"{stage}_launch.json").stat().st_mtime_ns
    root = old.read_json(path)
    assert root["manifest"] == binding and "failure" not in root
    assert "continuation_model" not in root
    assert old.digest(root["snapshot"]) == old.digest(old.serialize_state_snapshot(state))
    worlds = 128 if state.current_player == spec["candidate_seat"] else 32
    schedule = independent_schedule(state, spec["search_stream"], worlds)
    assert root["schedule"] == schedule
    offered = a158.a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
    assert root["candidates"] == offered and 1 <= len(offered["actions"]) <= 4
    obs, features, _, _, legal = a158.a157.input_invariance(state, old._RAW._model.config)
    with torch.no_grad():
        logits, _ = old._RAW._model(obs[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all()
    scores = logits[0].tolist()
    compare(root["scores"], scores)
    assert offered["actions"][0] == old.serialize_action(legal[max(range(len(legal)), key=lambda j: scores[j])])
    alternative = a158.hidden_alternative(state)
    assert independent_schedule(alternative, spec["search_stream"], worlds) == schedule
    assert a158.a154.independent_candidates(alternative, "155-live-v1", schedule["candidate_nonce"]) == offered
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert root["privacy"] == privacy and len(root["worlds"]) == worlds
    selected = replay_indices(stage, index, state.turn_index, worlds)
    totals, moves, replayed, reasons = [0.]*len(offered["actions"]), 0, 0, Counter()
    for wi, (seed, row) in enumerate(zip(schedule["world_seeds"], root["worlds"])):
        assert row["seed"] == seed and len(row["outcomes"]) == len(totals)
        world = a158.a154.independent_world(state, seed)
        world_sha = old.digest(old.serialize_state_snapshot(world))
        assert row["state_sha256"] == world_sha
        assert old.digest(old.serialize_state_snapshot(a158.a154.independent_world(alternative, seed))) == world_sha
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        world_legal = old.SplendorEnv().legal_actions(world)
        for j, (payload, outcome) in enumerate(zip(offered["actions"], row["outcomes"])):
            action = old.decode_action(payload)
            assert action in world_legal
            _outcome(outcome, state.current_player, state.turn_index)
            if wi in selected:
                expected = a158.replay_branch(world, action, state.current_player, spec["search_stream"], wi)
                assert outcome == expected, (stage, index, state.turn_index, wi, j, outcome, expected)
                replayed += 1
            totals[j] += outcome["wdl"]
            reasons[outcome["reason"]] += 1
            moves += outcome["turns"]
    means = [total/worlds for total in totals]
    choice = max(range(len(means)), key=lambda j: means[j])
    compare(root["means"], means)
    assert type(root["choice"]) is int and root["choice"] == choice
    assert move["action"] == offered["actions"][choice]
    _timing(root["timing"])
    return {"index": index, "seat": state.current_player, "turn": state.turn_index,
            "planner": "candidate" if worlds == 128 else "reference", "worlds": worlds,
            "choice": choice, "support": len(totals), "canonical_support": offered["canonical_support"],
            "legal_support": len(legal), "privacy": privacy, "timing": root["timing"],
            "continuations": sum(reasons.values()), "moves": moves, "reasons": dict(reasons),
            "replay_world_indices": selected, "replayed_branches": replayed}


def audit_game(stage, index, expected_sha, binding, base=OUTPUT, test_initial_snapshot=None):
    """Reconstruct a complete outer game; source policies are independently checked."""
    wall, cpu = time.perf_counter(), time.process_time()
    base = Path(base)
    spec = independent_spec(stage, index)
    if stage == "test":
        resolved = base.resolve()
        assert resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents
        assert test_initial_snapshot is not None
        state = old.deserialize_state_snapshot(test_initial_snapshot)
        env = old.SplendorEnv()
    else:
        assert test_initial_snapshot is None
        env = old.SplendorEnv(seed=spec["environment_seed"])
        state = env.initial_state()
    path = base/stage/"games"/f"game_{index:04d}.json"
    assert old.file_hash(path) == expected_sha
    assert path.stat().st_mtime_ns >= (base/f"{stage}_launch.json").stat().st_mtime_ns
    game = old.read_json(path)
    assert game["manifest"] == binding and game["spec"] == spec and "failure" not in game
    if stage == "test":
        assert old.digest(game["test_initial_snapshot"]) == old.digest(test_initial_snapshot)
    else:
        assert "test_initial_snapshot" not in game
    initial_sha = old.digest(old.serialize_state_snapshot(state))
    assert game["initial_state_sha256"] == initial_sha
    start_turn = state.turn_index
    visits, passes, idle, hasher, roots = Counter(), 0, 0, hashlib.sha256(), []
    for n, move in enumerate(game["trace"]):
        assert not state.terminal and state.turn_index < 400
        visits[old.state_signature(state)] += 1
        assert visits[old.state_signature(state)] < 4
        assert move["turn"] == state.turn_index == start_turn+n and move["actor"] == state.current_player
        assert move["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        legal = env.legal_actions(state)
        assert legal
        roots.append(audit_root(stage, index, state, move, spec, binding, base))
        action = old.decode_action(move["action"])
        assert action in legal
        hasher.update(old.digest(move["action"]).encode())
        successor = env.step(state, action)
        passes = passes+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(state, successor, action) else idle+1
        state = successor
        if n < len(game["trace"])-1:
            assert passes < 2 and idle < 60
    assert game["turns"] == len(game["trace"]) and game["trace_sha256"] == hasher.hexdigest()
    assert old.digest(game["final_state"]) == old.digest(old.serialize_state_snapshot(state))
    assert game["final_state_sha256"] == old.digest(game["final_state"])
    if game["reason"] == "completed":
        assert state.terminal and passes < 2 and idle < 60
        winner = state.winner
    else:
        assert game["reason"] == "forced_pass_deadlock" and passes == 2
        winner = a158.adjudicate(state)
    assert game["winner"] == winner
    seat = spec["candidate_seat"]
    score = .5 if winner is None else float(winner == seat)
    _timing(game["timing"])
    row = {"index": index, "pair": spec["pair"], "seat": seat, "score": score,
           "margin": state.players[seat].score-state.players[1-seat].score,
           "turns": game["turns"], "reason": game["reason"], "timing": game["timing"]}
    return {"game": row, "roots": roots, "initial_sha256": initial_sha,
            "audit_timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}}


def linear_quantile(values, q):
    ordered = sorted(values)
    point = (len(ordered)-1)*q
    low, high = math.floor(point), math.ceil(point)
    return ordered[low]+(ordered[high]-ordered[low])*(point-low)


def independent_statistics(rows):
    """Independent whole-seed-pair bootstrap with exact half-point score sums."""
    count = len(rows)
    assert count >= 2 and count % 2 == 0
    assert [r["index"] for r in rows] == list(range(count))
    pairs = count//2
    units = []
    for pair in range(pairs):
        left, right = rows[2*pair:2*pair+2]
        assert left["pair"] == right["pair"] == pair
        assert left["seat"] == 0 and right["seat"] == 1
        assert all(type(r["score"]) in (int, float) and r["score"] in (0., .5, 1.) for r in (left, right))
        assert all(math.isfinite(r["margin"]) and type(r["turns"]) is int and r["turns"] >= 0 for r in (left, right))
        units.append(int(2*left["score"]+2*right["score"]))
    rng = np.random.default_rng(1603000000)
    boot = []
    unit_array = np.array(units, dtype=np.int64)
    for _ in range(50):
        sampled = rng.integers(0, pairs, size=(1000, pairs))
        # Occurrence counts independently recover the integer sums; no BLAS.
        for sample in sampled:
            counts = np.bincount(sample, minlength=pairs)
            boot.append(int((counts*unit_array).sum())/(2*count))
    share = sum(units)/(2*count)
    interval = [float(linear_quantile(boot, q)) for q in (.025, .975)]
    label = ("positive" if share >= .55 and interval[0] > .50 else
             "below_material_target" if interval[1] < .55 else "inconclusive")
    seats = {}
    for seat in (0, 1):
        part = rows[seat::2]
        seats[str(seat)] = {"games": len(part), "score_share": sum(r["score"] for r in part)/len(part),
                            "mean_score_margin": sum(r["margin"] for r in part)/len(part)}
    return {"games": count, "pairs": pairs,
            "wins": sum(row["score"] == 1 for row in rows),
            "draws": sum(row["score"] == .5 for row in rows),
            "losses": sum(row["score"] == 0 for row in rows),
            "score_share": share, "ci95": interval,
            "bootstrap_resamples": 50000, "bootstrap_seed": 1603000000,
            "classification": label, "seats": seats,
            "mean_score_margin": sum(row["margin"] for row in rows)/count}


def _fixture_worker(fixture, index):
    from splendor_ai.alphazero import windowless_workers_160 as windowless
    base = Path(fixture["base_dir"])
    health = windowless.assert_windowless_worker(base/"spawn_worker_stdout.txt", base/"spawn_worker_stderr.txt")
    checked = audit_game("test", index, fixture["games"][index]["sha256"], fixture["binding"],
                         base, fixture["initial_snapshot"])
    return {"health": health, "reconstruction": checked}


def spawn_probe(fixture_path):
    """Exercise this exact CLI/module in real hidden Windows-spawn audit workers."""
    from concurrent.futures import ProcessPoolExecutor
    from splendor_ai.alphazero import windowless_workers_160 as windowless
    fixture = old.read_json(Path(fixture_path))
    assert fixture["kind"] == "160-spawn-test-fixture-v1"
    base = Path(fixture["base_dir"]).resolve()
    assert base != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in base.parents
    assert [row["index"] for row in fixture["games"]] == [0, 1]
    windowless.assert_windowless_worker(base/"spawn_parent_stdout.txt", base/"spawn_parent_stderr.txt")
    configuration = windowless.configure_spawn()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                             initializer=old.initialize_worker) as pool:
        futures = [pool.submit(_fixture_worker, fixture, index) for index in (0, 1)]
        completed = [future.result() for future in futures]
    checks = [row["reconstruction"] for row in completed]
    assert checks[0]["initial_sha256"] == checks[1]["initial_sha256"]
    result = {"passed": True, "games": 2, "configuration": configuration,
              "worker_health": [row["health"] for row in completed],
              "roots": sum(len(row["roots"]) for row in checks),
              "continuations": sum(root["continuations"] for row in checks for root in row["roots"]),
              "replayed_branches": sum(root["replayed_branches"] for row in checks for root in row["roots"])}
    old.save_new(base/"spawn_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("spawn-probe",))
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()
    return spawn_probe(Path(args.fixture))


if __name__ == "__main__":
    main()

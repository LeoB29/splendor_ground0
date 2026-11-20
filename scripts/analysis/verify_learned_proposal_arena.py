"""Separate reconstruction of proposals, full games, every branch and statistics."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.analysis import learned_proposal_arena as p
from scripts.analysis import verify_learned_proposal_screen as a_screen
from scripts.analysis import verify_rollout_selector_154 as a154
from scripts.analysis import verify_afterstate_advantage_149 as a149
from scripts.analysis import verify_fast_selector_transfer_157 as a157
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

old, OUTPUT, PRODUCTION_OUTPUT = p.old, p.OUTPUT, p.PRODUCTION_OUTPUT
compare = a_screen.compare


def independent_specs(stage):
    assert stage in ("probe", "main", "test")
    count = {"probe": 6, "main": 201, "test": 3}[stage]
    ranked = sorted(range(count), key=lambda pair: old.digest(["learned-proposal-arena-fit-v1", stage, pair]))
    result = []
    for pair in range(count):
        fit = (161001, 161002, 161003)[ranked.index(pair) % 3]
        for seat in (0, 1):
            result.append({"index": 2*pair+seat, "pair": pair, "candidate_seat": seat, "fit": fit,
                           "environment_seed": {"probe": 2049000000, "main": 2040000000, "test": 2049800000}[stage]+pair,
                           "search_stream": old.digest(["learned-proposal-arena-stream-v1", stage, pair])})
    return result


def independent_spec(stage, index):
    rows = independent_specs(stage)
    assert type(index) is int and 0 <= index < len(rows)
    return rows[index]


def independent_schedule(state, stream):
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    return {"belief_sha256": belief, "candidate_nonce": old.digest(["155-candidate-v1", stream, belief]),
            "world_seeds": [f"155-world-v1|{stream}|{belief}|{j:02d}" for j in range(32)]}


def independent_proposal(state, stream, fit):
    # Existing independent screen implementation reconstructs groups, concrete
    # logits, original representatives and all three rankings without production
    # proposal helpers. Select the assigned frozen fit, never a favorable fit.
    all_fits = a_screen.independent_proposals(state, stream)
    indices = all_fits["arm_indices"][f"learned_{fit}"]
    reference = all_fits["reference"]
    return {"schedule": all_fits["schedule"], "reference": reference,
            "candidates": {"keys": all_fits["arm_keys"][f"learned_{fit}"],
                           "actions": [all_fits["union_actions"][j] for j in indices],
                           "canonical_support": reference["canonical_support"], "legal_support": reference["legal_support"]},
            "fit": fit, "group_keys": all_fits["group_keys"], "legal_actions": all_fits["legal_actions"],
            "learned_logits": all_fits["learned_logits"][str(fit)],
            "learned_group_scores": all_fits["learned_group_scores"][str(fit)]}


def adjudicate(state):
    ranks = [(player.score, -len(player.purchased_cards)) for player in state.players]
    return None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)


def _timing(values):
    assert set(values) == {"wall", "cpu"}
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())


def audit_root(stage, index, state, move, spec, binding, base=OUTPUT):
    path = Path(base) / stage / "roots" / f"game_{index:04d}" / f"turn_{state.turn_index:03d}.json"
    assert old.file_hash(path) == move["root_sha256"]
    assert path.stat().st_mtime_ns >= (Path(base) / f"{stage}_launch.json").stat().st_mtime_ns
    root = old.read_json(path)
    assert root["manifest"] == binding and "failure" not in root
    assert old.digest(root["snapshot"]) == old.digest(old.serialize_state_snapshot(state))
    schedule = independent_schedule(state, spec["search_stream"])
    assert root["schedule"] == schedule
    obs, features, _, _, legal = a157.input_invariance(state, old._RAW._model.config)
    with torch.no_grad():
        logits, _ = old._RAW._model(obs[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all()
    compare(root["scores"], logits[0].tolist())
    altered = a157.hidden_alternative(state)
    assert independent_schedule(altered, spec["search_stream"]) == schedule
    candidate = state.current_player == spec["candidate_seat"]
    if candidate:
        expected = independent_proposal(state, spec["search_stream"], spec["fit"])
        compare({key: value for key, value in root["proposal"].items() if key != "timing"}, expected)
        a157.input_invariance(state, p.screen._MODELS[spec["fit"]].config)
        compare(expected, independent_proposal(altered, spec["search_stream"], spec["fit"]))
        offered = expected["candidates"]
        assert offered["keys"][:3] == expected["reference"]["keys"][:3]
    else:
        assert "proposal" not in root
        offered = a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
        assert a154.independent_candidates(altered, "155-live-v1", schedule["candidate_nonce"]) == offered
    assert root["candidates"] == offered and len(offered["actions"]) == min(4, offered["canonical_support"])
    assert len(set(offered["keys"])) == len(offered["actions"])
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert root["privacy"] == privacy and len(root["worlds"]) == 32
    totals, moves, reasons = [0.] * len(offered["actions"]), 0, Counter()
    for wi, (seed, saved) in enumerate(zip(schedule["world_seeds"], root["worlds"])):
        assert saved["seed"] == seed and len(saved["outcomes"]) == len(totals)
        world = a154.independent_world(state, seed)
        world_sha = old.digest(old.serialize_state_snapshot(world))
        assert saved["state_sha256"] == world_sha
        assert old.digest(old.serialize_state_snapshot(a154.independent_world(altered, seed))) == world_sha
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        legal_world = old.SplendorEnv().legal_actions(world)
        for j, (payload, outcome) in enumerate(zip(offered["actions"], saved["outcomes"])):
            action = old.decode_action(payload)
            assert action in legal_world
            rebuilt = a149.replay_branch(world, action, state.current_player)
            compare(outcome, rebuilt)
            assert rebuilt["reason"] in ("completed", "forced_pass_deadlock")
            totals[j] += rebuilt["wdl"]
            moves += rebuilt["turns"]
            reasons[rebuilt["reason"]] += 1
    means = [v / 32 for v in totals]
    choice = min(range(len(means)), key=lambda j: (-means[j], j))
    compare(root["means"], means)
    assert type(root["choice"]) is int and root["choice"] == choice
    assert move["action"] == offered["actions"][choice]
    _timing(root["timing"])
    return {"index": index, "turn": state.turn_index, "seat": state.current_player,
            "planner": "candidate" if candidate else "reference", "fit": spec["fit"] if candidate else None,
            "sha256": move["root_sha256"], "choice": choice, "support": len(totals),
            "privacy": privacy, "timing": root["timing"], "continuations": sum(reasons.values()),
            "replayed_branches": sum(reasons.values()), "moves": moves, "reasons": dict(reasons)}


def linear_quantile(values, q):
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def independent_statistics(rows):
    assert len(rows) == 402 and [r["index"] for r in rows] == list(range(402))
    specs = independent_specs("main")
    strata = {seed: [] for seed in (161001, 161002, 161003)}
    for i in range(0, 402, 2):
        a, b = rows[i:i+2]
        assert (a["seat"], b["seat"]) == (0, 1) and a["pair"] == b["pair"] == i // 2
        assert a["fit"] == b["fit"] == specs[i]["fit"]
        assert a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.)
        strata[a["fit"]].append(int(2*a["score"]+2*b["score"]))
    assert all(len(v) == 67 for v in strata.values())
    rng = np.random.default_rng(2043000000)
    boot = []
    for _ in range(50):
        draws = [rng.integers(0, 67, size=(1000, 67)) for _ in range(3)]
        for i in range(1000):
            total = 0
            for values, indices in zip(strata.values(), draws):
                counts = np.bincount(indices[i], minlength=67)
                total += sum(int(n)*value for n, value in zip(counts, values))
            boot.append(total / 804)
    share = sum(r["score"] for r in rows) / 402
    interval = [float(linear_quantile(boot, q)) for q in (.025, .975)]
    superiority = interval[0] > .5
    practical = 10 * sum(int(2*r["score"]) for r in rows) >= 11 * 402 and superiority
    label = ("positive_practical_progress" if practical else
             "positive_below_observed_target" if superiority else
             "below_material_target" if interval[1] < .55 else "inconclusive")
    return {"games": 402, "pairs": 201, "share": share, "ci95": interval,
            "wins": sum(r["score"] == 1 for r in rows), "draws": sum(r["score"] == .5 for r in rows),
            "losses": sum(r["score"] == 0 for r in rows), "superiority_established": superiority,
            "practical_progress": practical, "material_target_excluded": interval[1] < .55, "classification": label}


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
        winner = adjudicate(state)
    assert game["winner"] == winner
    seat = spec["candidate_seat"]
    score = .5 if winner is None else float(winner == seat)
    _timing(game["timing"])
    row = {"index": index, "pair": spec["pair"], "seat": seat, "fit": spec["fit"], "score": score,
           "margin": state.players[seat].score-state.players[1-seat].score,
           "turns": game["turns"], "reason": game["reason"], "timing": game["timing"]}
    return {"game": row, "roots": roots, "initial_sha256": initial_sha,
            "audit_timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}}


def _fixture_job(kind, fixture, index):
    base = Path(fixture["base"])
    p.check_stage("test", base)
    if kind == "games":
        return p.play_game("test", independent_spec("test", index), fixture["binding"], base, fixture["snapshot"])
    assert kind == "audits"
    item = fixture["games"][index]
    return audit_game("test", index, item["sha256"], fixture["binding"], base, fixture["snapshot"])


def fixture_pipeline(path):
    """Explicit synthetic spawn test; cannot target production or use real starts."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    fixture = old.read_json(Path(path))
    assert fixture["kind"] == "learned-proposal-arena-synthetic-v1"
    base = Path(fixture["base"])
    p.check_stage("test", base)
    p.windows.assert_windowless_worker(base / "parent_stdout.txt", base / "parent_stderr.txt")
    p.windows.configure_spawn()
    old.save_new(base / "test_launch.json", {"manifest": fixture["binding"]})
    for kind in ("games", "audits"):
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=p.worker_initialize, initargs=(f"test_{kind}", fixture["binding"], base)) as pool:
            values = list(pool.map(_fixture_job, [kind]*6, [fixture]*6, range(6)))
        if kind == "games":
            fixture["games"] = values
    old.save_new(base / "fixture_result.json", {"passed": True, "games": 6,
                 "roots": sum(len(row["roots"]) for row in values),
                 "continuations": sum(root["continuations"] for row in values for root in row["roots"]),
                 "replayed_branches": sum(root["replayed_branches"] for row in values for root in row["roots"])})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explicit isolated synthetic Windows spawn check only")
    parser.add_argument("--fixture", required=True)
    fixture_pipeline(parser.parse_args().fixture)

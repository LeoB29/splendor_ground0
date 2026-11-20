"""158 independent full-game, full-continuation and stratified-pair audit.

Shares the frozen engine, serialization, canonical information/action keys,
observer encoders and neural models. Frozen154 independently reconstructs root
proposals and worlds; frozen157 independently reconstructs learned decisions
and checks hidden-information invariance. The trajectory, game, scheduling,
selection, statistical and operational loops below are independent of158 play.
Every continuation decision, including reference035 decisions, checks lawful
inputs. Decision hashes bind the entire157 evidence, not merely played actions.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import importlib.util
import math
import multiprocessing
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import run_learned_continuation as p
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

old = p.old
REPOSITORY = Path(__file__).resolve().parents[2]
FIXED_MODELS = ("treatment_157001", "treatment_157002", "treatment_157003")


def _frozen_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / "scripts" / "analysis" / filename)
    module = importlib.util.module_from_spec(spec)
    # Top-level registration also runs in Windows-spawn children.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a157 = _frozen_module("audit157_for158", "verify_fast_selector_transfer_157.py")
a154 = a157.a154
sys.modules[a157.spec154.name] = a154
sys.modules[a154.spec.name] = a154.a149
compare = a157.compare
hidden_alternative = a157.hidden_alternative
independent_decision = a157.independent_decision


@lru_cache(maxsize=2)
def independent_specs(stage):
    assert stage in ("probe", "main")
    assert tuple(p.MODELS) == FIXED_MODELS
    pairs = 6 if stage == "probe" else 201
    ranked = sorted(range(pairs), key=lambda pair: old.digest(["158-model-assignment-v1", stage, pair]))
    assignment = {pair: FIXED_MODELS[rank % 3] for rank, pair in enumerate(ranked)}
    return tuple({"index": index, "pair": index // 2, "candidate_seat": index % 2,
                  "environment_seed": (1589000000 if stage == "probe" else 1580000000) + index // 2,
                  "search_stream": old.digest(["158-root-stream-v1", stage, index // 2]),
                  "model_id": assignment[index // 2]} for index in range(2 * pairs))


def independent_spec(stage, index):
    assert isinstance(index, int) and 0 <= index < (12 if stage == "probe" else 402)
    return dict(independent_specs(stage)[index])


def test_spec(index):
    """A separate namespace usable only with an explicit external test fixture."""
    assert index in (0, 1)
    return {"index": index, "pair": 0, "candidate_seat": index,
            "environment_seed": 1589800000,
            "search_stream": old.digest(["158-test-root-stream-v1", 0]),
            "model_id": FIXED_MODELS[0]}


def independent_schedule(state, stream):
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    return {"belief_sha256": belief,
            "candidate_nonce": old.digest(["155-candidate-v1", stream, belief]),
            "world_seeds": ["155-world-v1|%s|%s|%02d" % (stream, belief, i) for i in range(32)]}


def continuation_stream(root_stream, world_index, step, actor):
    # Exogenous slot randomness contains no ancestor/root actor's private view.
    # Exact157 adds only the current simulated actor's own lawful belief hash.
    return old.digest(["158-continuation-v1", root_stream, world_index, step, actor])


def adjudicate(state):
    ranks = [(player.score, -len(player.purchased_cards)) for player in state.players]
    return None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)


def raw_decision(state):
    # Check both observer inputs and all legal concrete-action features on every
    # reference decision. Forward once, directly through the frozen model, after
    # reconstructing inputs independently of its public score_actions wrapper.
    config = old._RAW._model.config
    def inputs(position):
        legal = old.SplendorEnv().legal_actions(position)
        assert legal
        obs = torch.tensor(a157.fit_feature_vector(
            a157.encode_public_observation_tensor(position, position.current_player).vector,
            config.observation_size), dtype=torch.float32)
        features = torch.tensor([a157.fit_feature_vector(row.vector, config.action_feature_size)
                                 for row in a157.encode_legal_action_features(
                                     state=position, legal_actions=legal, player_id=position.current_player)],
                                dtype=torch.float32)
        return obs, features, legal
    obs, features, legal = inputs(state)
    altered_obs, altered_features, altered_legal = inputs(hidden_alternative(state))
    assert torch.equal(obs, altered_obs) and torch.equal(features, altered_features)
    assert legal == altered_legal
    # Greedy035 uses no canonical groups or information-key RNG. Their expensive
    # reconstruction is unnecessary here: all actual inputs are compared above.
    with torch.no_grad():
        logits, _ = old._RAW._model(obs[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all()
    values = logits[0].tolist()
    return legal[max(range(len(legal)), key=lambda i: values[i])]


def replay_branch(world, action, actor, root_stream, world_index, model=None):
    """Independent natural-end loop; no calls to either production trajectory."""
    env = old.SplendorEnv()
    pending = a154.a149.deterministic_part(world, action)
    draw = None if pending.chance_kind is None else world.hidden_tier_decks[pending.chance_tier][0].card_id
    composed = pending.resolve(draw)
    assert old.digest(old.serialize_state_snapshot(composed)) == old.digest(old.serialize_state_snapshot(env.step(world, action)))
    current, count, passes, idle = world, 0, 0, 0
    visits, action_hash, decision_hash = Counter(), hashlib.sha256(), hashlib.sha256()
    reason, winner, learned = "completed", None, 0
    while not current.terminal:
        visits[old.state_signature(current)] += 1
        assert visits[old.state_signature(current)] < 4, "Repeated continuation state"
        assert current.turn_index < 400, "Continuation turn cap"
        legal = env.legal_actions(current)
        assert legal
        if count == 0:
            chosen, nxt = action, composed
        else:
            if model is None:
                chosen = raw_decision(current)
            else:
                stream = continuation_stream(root_stream, world_index, count, current.current_player)
                chosen, evidence = independent_decision(current, stream, model)
                decision_hash.update(old.digest({"step": count, "actor": current.current_player,
                                                "stream": stream, "decision": evidence}).encode())
                learned += 1
            nxt = env.step(current, chosen)
        assert chosen in legal
        action_hash.update(old.digest(old.serialize_action(chosen)).encode())
        passes = passes + 1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(current, nxt, chosen) else idle + 1
        current, count = nxt, count + 1
        if passes == 2:
            reason, winner = "forced_pass_deadlock", adjudicate(current)
            break
        assert idle < 60, "Continuation no-progress cap"
    if reason == "completed":
        assert current.terminal
        winner = current.winner
    result = {"reason": reason, "winner": winner, "turns": count,
              "trace_sha256": action_hash.hexdigest(),
              "final_state_sha256": old.digest(old.serialize_state_snapshot(current)),
              "wdl": 0. if winner is None else 1. if winner == actor else -1.}
    if model is not None:
        result.update(decision_sha256=decision_hash.hexdigest(), learned_decisions=learned)
    return result


def root_path(base, stage, index, turn):
    return base / stage / "roots" / f"game_{index:04d}" / f"turn_{turn:03d}.json"


def game_path(base, stage, index):
    return base / stage / "games" / f"game_{index:04d}.json"


def audit_root(stage, index, state, move, spec, binding, base=None):
    base = p.OUTPUT if base is None else Path(base)
    path = root_path(base, stage, index, state.turn_index)
    assert old.file_hash(path) == move["root_sha256"]
    assert path.stat().st_mtime_ns >= (base / f"{stage}_launch.json").stat().st_mtime_ns
    record = old.read_json(path)
    assert record["manifest"] == binding and "failure" not in record
    assert old.digest(record["snapshot"]) == old.digest(old.serialize_state_snapshot(state))
    candidate = state.current_player == spec["candidate_seat"]
    if candidate:
        assert record["continuation_model"] == spec["model_id"]
        model = p.h.MODELS_CACHE[spec["model_id"]]
    else:
        assert "continuation_model" not in record
        model = None
    schedule = independent_schedule(state, spec["search_stream"])
    assert record["schedule"] == schedule
    offered = a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
    assert record["candidates"] == offered
    obs, features, _, _, legal = a157.input_invariance(state, old._RAW._model.config)
    with torch.no_grad():
        logits, _ = old._RAW._model(obs[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all()
    scores = logits[0].tolist()
    compare(record["scores"], scores)
    assert offered["actions"][0] == old.serialize_action(legal[max(range(len(legal)), key=lambda i: scores[i])])
    alternative = hidden_alternative(state)
    assert independent_schedule(alternative, spec["search_stream"]) == schedule
    assert a154.independent_candidates(alternative, "155-live-v1", schedule["candidate_nonce"]) == offered
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert record["privacy"] == privacy and len(record["worlds"]) == 32
    totals, reasons, moves = [0.] * len(offered["actions"]), Counter(), 0
    for world_index, (seed, row) in enumerate(zip(schedule["world_seeds"], record["worlds"])):
        assert row["seed"] == seed and len(row["outcomes"]) == len(totals)
        world = a154.independent_world(state, seed)
        snapshot = old.serialize_state_snapshot(world)
        assert old.digest(snapshot) == row["state_sha256"]
        assert old.digest(old.serialize_state_snapshot(a154.independent_world(alternative, seed))) == old.digest(snapshot)
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        for j, (payload, outcome) in enumerate(zip(offered["actions"], row["outcomes"])):
            decoded = old.decode_action(payload)
            assert decoded in old.SplendorEnv().legal_actions(world)
            replayed = replay_branch(world, decoded, state.current_player, spec["search_stream"], world_index, model)
            assert replayed == outcome, (index, state.turn_index, world_index, j, replayed, outcome)
            assert replayed["reason"] in ("completed", "forced_pass_deadlock")
            totals[j] += replayed["wdl"]
            reasons[replayed["reason"]] += 1
            moves += replayed["turns"]
    means = [total / 32 for total in totals]
    choice = max(range(len(means)), key=lambda i: means[i])
    compare(record["means"], means)
    assert record["choice"] == choice and move["action"] == offered["actions"][choice]
    assert all(math.isfinite(record["timing"][key]) and record["timing"][key] >= 0 for key in ("wall", "cpu"))
    return {"index": index, "model_id": spec["model_id"], "planner": "candidate" if candidate else "reference",
            "seat": state.current_player, "turn": state.turn_index, "choice": choice, "support": len(totals),
            "canonical_support": offered["canonical_support"], "legal_support": len(legal),
            "privacy": privacy, "timing": record["timing"], "continuations": sum(reasons.values()),
            "moves": moves, "reasons": dict(reasons)}


def audit_game(stage, index, expected_sha, binding, test_context=None):
    """Actual spawned worker; test fixtures exercise this same complete replay.

    Only the distinct test stage may replace the initial snapshot or output
    directory. It cannot read/write a production158 output subtree.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    if stage == "test":
        assert test_context is not None
        base = Path(test_context["base_dir"]).resolve()
        assert base != p.OUTPUT.resolve() and p.OUTPUT.resolve() not in base.parents
        spec = test_spec(index)
        assert test_context["spec"] == spec
        state = old.deserialize_state_snapshot(test_context["initial_snapshot"])
        env = old.SplendorEnv()
    else:
        assert stage in ("probe", "main") and test_context is None
        base, spec = p.OUTPUT, independent_spec(stage, index)
        env = old.SplendorEnv(seed=spec["environment_seed"])
        state = env.initial_state()
    path = game_path(base, stage, index)
    assert old.file_hash(path) == expected_sha
    assert path.stat().st_mtime_ns >= (base / f"{stage}_launch.json").stat().st_mtime_ns
    record = old.read_json(path)
    assert record["manifest"] == binding and record["spec"] == spec and "failure" not in record
    initial_sha = old.digest(old.serialize_state_snapshot(state))
    assert record["initial_state_sha256"] == initial_sha
    start_turn = state.turn_index
    visits, hasher, passes, idle = Counter(), hashlib.sha256(), 0, 0
    roots = []
    for n, move in enumerate(record["trace"]):
        assert not state.terminal and state.turn_index < 400
        visits[old.state_signature(state)] += 1
        assert visits[old.state_signature(state)] < 4
        assert move["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        assert move["turn"] == state.turn_index == start_turn + n and move["actor"] == state.current_player
        legal = env.legal_actions(state)
        assert legal
        roots.append(audit_root(stage, index, state, move, spec, binding, base))
        action = old.decode_action(move["action"])
        assert action in legal
        hasher.update(old.digest(move["action"]).encode())
        nxt = env.step(state, action)
        passes = passes + 1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(state, nxt, action) else idle + 1
        state = nxt
        if n < len(record["trace"]) - 1:
            assert passes < 2 and idle < 60
    assert record["turns"] == len(record["trace"]) and record["trace_sha256"] == hasher.hexdigest()
    assert old.digest(record["final_state"]) == old.digest(old.serialize_state_snapshot(state))
    assert record["final_state_sha256"] == old.digest(record["final_state"])
    if record["reason"] == "completed":
        assert state.terminal and idle < 60 and passes < 2
        winner = state.winner
    else:
        assert record["reason"] == "forced_pass_deadlock" and passes == 2
        winner = adjudicate(state)
    assert record["winner"] == winner
    seat = spec["candidate_seat"]
    score = .5 if winner is None else float(winner == seat)
    row = {"index": index, "pair": spec["pair"], "model_id": spec["model_id"], "seat": seat, "score": score,
           "margin": state.players[seat].score - state.players[1-seat].score,
           "turns": record["turns"], "reason": record["reason"], "timing": record["timing"]}
    assert all(math.isfinite(row["timing"][k]) and row["timing"][k] >= 0 for k in ("wall", "cpu"))
    return {"game": row, "roots": roots, "initial_sha256": initial_sha,
            "audit_timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}}


def linear_quantile(values, q):
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def independent_statistics(rows):
    assert len(rows) == 402 and [r["index"] for r in rows] == list(range(402))
    strata = {model: [] for model in FIXED_MODELS}
    for pair in range(201):
        a, b = rows[2*pair:2*pair+2]
        expected = independent_spec("main", pair*2)
        assert a["pair"] == b["pair"] == pair and (a["seat"], b["seat"]) == (0, 1)
        assert a["model_id"] == b["model_id"] == expected["model_id"]
        assert a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.)
        strata[a["model_id"]].append(int(2 * a["score"] + 2 * b["score"]))
    assert all(len(values) == 67 for values in strata.values())
    rng, boot = np.random.default_rng(1583000000), []
    # Independent count-based computation, preserving the frozen RNG call order.
    for _ in range(50):
        chunk = np.zeros(1000, dtype=np.int64)
        for model in FIXED_MODELS:
            values = np.asarray(strata[model], dtype=np.int64)
            indices = rng.integers(0, 67, size=(1000, 67))
            for j, sample in enumerate(indices):
                counts = np.bincount(sample, minlength=67)
                chunk[j] += int((counts * values).sum())
        # Equal stratum sizes make this the exact three-stratum mean; a single
        # final division also preserves the strict50% CI boundary.
        boot.extend((chunk / 804).tolist())
    ci = [linear_quantile(boot, q) for q in (.025, .975)]
    def summary(part):
        return {"games": len(part), "share": sum(r["score"] for r in part) / len(part),
                "wdl": {"wins": sum(r["score"] == 1 for r in part), "draws": sum(r["score"] == .5 for r in part),
                        "losses": sum(r["score"] == 0 for r in part)},
                "mean_margin": sum(r["margin"] for r in part) / len(part),
                "mean_turns": sum(r["turns"] for r in part) / len(part)}
    result = summary(rows)
    half_wins = sum(int(2 * r["score"]) for r in rows)
    gates = {"material_share": half_wins * 100 >= 55 * 2 * len(rows),
             "positive_lower_bound": ci[0] > .50,
             "positive_margin": sum(r["margin"] for r in rows) > 0,
             "both_seats": all(sum(int(2*r["score"]) for r in rows[s::2]) >= 201 for s in (0, 1))}
    return {**result, "pairs": 201, "ci95": ci, "statistical_gates": gates,
            "statistical_pass": all(gates.values()), "seats": {str(s): summary(rows[s::2]) for s in (0, 1)},
            "models": {model: summary([r for r in rows if r["model_id"] == model]) for model in FIXED_MODELS},
            "estimand": "uniform mixture of three fixed157 treatment planners"}


def independent_operational(data):
    roots, games = data["roots"], data["games"]
    def distribution(values):
        assert values and all(math.isfinite(v) and v >= 0 for v in values)
        return {"count": len(values), "mean": sum(values)/len(values), "p95": linear_quantile(values, .95),
                "max": max(values), "sum": sum(values)}
    def reasons(part):
        counts = Counter()
        for row in part:
            counts.update(row["reasons"])
        return dict(counts)
    def planner(part):
        n, changes = len(part), sum(row["choice"] != 0 for row in part)
        return {"roots": n, "changes": changes, "change_fraction": changes/n if n else None,
                "support_counts": dict(Counter(str(row["support"]) for row in part)),
                "own_private_roots": sum(row["privacy"]["own"] > 0 for row in part),
                "opponent_private_roots": sum(row["privacy"]["opponent"] > 0 for row in part),
                "continuations": sum(row["continuations"] for row in part),
                "continuation_moves": sum(row["moves"] for row in part), "inner_reasons": reasons(part),
                "decision_time": {key: distribution([row["timing"][key] for row in part]) for key in ("wall", "cpu")}}
    return {"roots": len(roots), "continuations": sum(row["continuations"] for row in roots),
            "continuation_moves": sum(row["moves"] for row in roots), "inner_reasons": reasons(roots),
            "outer_reasons": dict(Counter(game["reason"] for game in games)),
            "game_time": {key: distribution([game["timing"][key] for game in games]) for key in ("wall", "cpu")},
            "turns": distribution([game["turns"] for game in games]),
            "planners": {kind: planner([row for row in roots if row["planner"] == kind]) for kind in ("candidate", "reference")}}


def verify(stage):
    wall = time.perf_counter()
    with old.process_lock(p.OUTPUT / "audit.lock"):
        assert not (p.OUTPUT / f"{stage}_audit.json").exists(), "Completed audit must not be rerun automatically"
        binding = p.validate_freeze()
        schedules = {s: list(independent_specs(s)) for s in ("probe", "main")}
        assert old.read_json(p.OUTPUT / "schedule.json") == schedules
        frozen = old.read_json(p.OUTPUT / "freeze.json")
        assert frozen["manifest"] == binding and frozen["tests_sha256"] == p.verify_test_evidence()
        assert frozen["schedule_sha256"] == old.file_hash(p.OUTPUT / "schedule.json")
        launch_path = p.OUTPUT / f"{stage}_launch.json"
        launch = old.read_json(launch_path)
        assert launch["manifest"] == binding and launch["schedule_sha256"] == frozen["schedule_sha256"]
        upstream = "freeze.json" if stage == "probe" else "preflight.json"
        upstream_key = "freeze_sha256" if stage == "probe" else "preflight_sha256"
        assert launch["upstream"] == {upstream_key: old.file_hash(p.OUTPUT / upstream)}
        if stage == "main":
            preflight = old.read_json(p.OUTPUT / "preflight.json")
            assert preflight["passed"] and preflight["manifest"] == binding
            assert preflight["audit_sha256"] == old.file_hash(p.OUTPUT / "probe_audit.json")
        lock_path = p.OUTPUT / f"{stage}_games_lock.json"
        lock = old.read_json(lock_path)
        n = len(schedules[stage])
        assert lock["manifest"] == binding and lock["launch_sha256"] == old.file_hash(launch_path)
        assert set(lock["files"]) == {str(i) for i in range(n)}
        assert set((p.OUTPUT / stage / "games").glob("*.json")) == {game_path(p.OUTPUT, stage, i) for i in range(n)}
        assert not list((p.OUTPUT / stage / "replay").glob("*.json")), "Partial audit requires explicit review"
        audited = {}
        with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=p.h.initialize_worker) as pool:
            futures = {pool.submit(audit_game, stage, i, lock["files"][str(i)], binding): i for i in range(n)}
            try:
                for future in as_completed(futures):
                    i = futures[future]
                    audited[i] = future.result()
                    old.save_new(p.OUTPUT / stage / "replay" / f"game_{i:04d}.json",
                                 {"manifest": binding, "game_sha256": lock["files"][str(i)], "reconstruction": audited[i]})
                    p.progress(f"{stage}_replay", games_complete=len(audited), games_total=n,
                               elapsed_seconds=time.perf_counter()-wall)
            except BaseException:
                for future in futures:
                    future.cancel()
                old.save_new(p.OUTPUT / f"audit_failure_{stage}_{time.time_ns()}.json",
                             {"time": old.now(), "manifest": binding, "completed": sorted(audited),
                              "traceback": traceback.format_exc()})
                raise
        for i in range(0, n, 2):
            assert audited[i]["initial_sha256"] == audited[i+1]["initial_sha256"]
        data = {"games": [audited[i]["game"] for i in range(n)],
                "roots": [root for i in range(n) for root in audited[i]["roots"]]}
        expected_roots = {root_path(p.OUTPUT, stage, root["index"], root["turn"]) for root in data["roots"]}
        assert set((p.OUTPUT / stage / "roots").rglob("*.json")) == expected_roots
        compare(data, p.observations(stage))
        output = {"passed": True, "manifest": binding, "games_lock_sha256": old.file_hash(lock_path),
                  "games": n, "roots": len(data["roots"]), "outer_moves": sum(game["turns"] for game in data["games"]),
                  "continuations": sum(root["continuations"] for root in data["roots"]),
                  "continuation_moves": sum(root["moves"] for root in data["roots"]),
                  "wall_seconds": time.perf_counter()-wall,
                  "worker_cpu_seconds": sum(row["audit_timing"]["cpu"] for row in audited.values()),
                  "shared_frozen_engine": True, "every_root_information_invariance": True,
                  "every_continuation_information_invariance": True,
                  "every_learned_decision_evidence_replayed": True,
                  "replay_files": {str(i): old.file_hash(p.OUTPUT / stage / "replay" / f"game_{i:04d}.json") for i in range(n)}}
        if stage == "main":
            compare(data, old.read_json(p.OUTPUT / "observations.json"))
            result = old.read_json(p.OUTPUT / "result.json")
            stats = independent_statistics(data["games"])
            compare(stats, {key: result[key] for key in stats})
            compare(independent_operational(data), result["operational"])
            output.update(result_sha256=old.file_hash(p.OUTPUT / "result.json"),
                          observations_sha256=old.file_hash(p.OUTPUT / "observations.json"))
        assert p.validate_freeze() == binding
        old.save_new(p.OUTPUT / f"{stage}_audit.json", output)
        p.progress(f"{stage}_audit_passed", games=n, roots=output["roots"], continuations=output["continuations"],
                   wall_seconds=output["wall_seconds"])
        return output


def spawn_probe(fixture_path):
    """Dispatch the real successful game auditor in fresh Windows-style workers."""
    fixture = old.read_json(Path(fixture_path))
    assert fixture["kind"] == "158-spawn-test-fixture-v1" and len(fixture["games"]) == 2
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                             initializer=p.h.initialize_worker) as pool:
        futures = [pool.submit(audit_game, "test", row["index"], row["sha256"], fixture["binding"],
                               {"base_dir": fixture["base_dir"], "initial_snapshot": fixture["initial_snapshot"],
                                "spec": test_spec(row["index"])}) for row in fixture["games"]]
        rows = [future.result() for future in futures]
    assert [row["game"]["index"] for row in rows] == [0, 1]
    assert rows[0]["initial_sha256"] == rows[1]["initial_sha256"]
    return {"passed": True, "games": len(rows), "roots": sum(len(row["roots"]) for row in rows),
            "continuations": sum(root["continuations"] for row in rows for root in row["roots"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("probe", "main", "spawn-probe"))
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage == "spawn-probe":
        assert args.fixture is not None
        print(spawn_probe(args.fixture), flush=True)
    else:
        assert args.fixture is None
        verify(args.stage)


if __name__ == "__main__":
    main()

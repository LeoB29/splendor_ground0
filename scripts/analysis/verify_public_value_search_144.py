"""Replay all 144 trajectories; reproduce prespecified first/last searches."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_public_value_search as exp
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig
from splendor_ai.encoding import actions_by_canonical_successor_key, canonical_successor_action_key
from splendor_ai.engine.constants import TOKEN_COLORS
from splendor_ai.training.replay import collect_game_replay, serialize_state_snapshot, _serialize_action


def independent_value(state):
    actor = state.players[state.current_player]
    other = state.players[1-state.current_player]
    values = [(actor.score-other.score)/15]
    values.extend((actor.bonuses[c]-other.bonuses[c])/7 for c in TOKEN_COLORS)
    values.extend((actor.tokens[c]-other.tokens[c])/7 for c in TOKEN_COLORS)
    values.extend([(actor.tokens["gold"]-other.tokens["gold"])/5,
                   (len(actor.purchased_cards)-len(other.purchased_cards))/20,
                   (len(actor.reserved_cards)-len(other.reserved_cards))/3,
                   (len(actor.nobles)-len(other.nobles))/3])
    coefficients = torch.tensor(exp.read_json(exp.SIMPLE)["coefficients"], dtype=torch.float64)
    return float(torch.dot(torch.tensor(values, dtype=torch.float32).double(), coefficients).clamp(-1, 1))


def audit_root(state, legal, row):
    grouped = actions_by_canonical_successor_key(state, legal)
    edges = row["edges"]
    keys = [repr(key) for key in grouped]
    assert [edge["key"] for edge in edges] == keys
    assert row["mode"] == "real"
    assert row["raw_network_value"] == independent_value(state)
    scores = exp._RAW.score_actions(state, legal)
    by_key = {}
    for score, action in zip(scores, legal, strict=True):
        key = repr(canonical_successor_action_key(state, action))
        by_key[key] = max(score, by_key.get(key, -math.inf))
    largest = max(by_key.values())
    weights = [math.exp(by_key[key]-largest) for key in keys]
    priors = [w/sum(weights) for w in weights]
    total = sum(edge["visits"] for edge in edges)
    assert total == row["simulations"] >= 1
    for edge, prior in zip(edges, priors, strict=True):
        assert isinstance(edge["visits"], int) and edge["visits"] >= 0
        assert edge["probability"] == edge["visits"]/total
        assert edge["prior"] == edge["network_prior"] == prior
        assert math.isfinite(edge["q"]) and -1.000000000001 <= edge["q"] <= 1.000000000001
    best = max(range(len(edges)), key=lambda i: (edges[i]["visits"], edges[i]["q"], edges[i]["prior"], edges[i]["key"]))
    assert row["selected_row"] == best and row["action_key"] == edges[best]["key"]
    assert row["root_value"] == sum(e["q"]*e["visits"] for e in edges)/total
    assert row["expansions"] == row["inference_calls"] >= 1
    assert 0 <= row["max_depth"] <= exp.CONFIG["search"]["max_depth"]
    assert row["decision_seconds"] >= row["search_seconds"] >= exp.CONFIG["decision_seconds"]


def audit_game(path, binding, probe=False):
    if exp._VALUE is None:
        exp.initialize_worker()
    path = Path(path)
    record = exp.read_json(path)
    index = record["index"]
    expected = (exp.CONFIG["probe_seed"]+index//2, index%2) if probe else exp.schedule(index)
    assert (record["seed"], record["candidate_seat"]) == expected
    assert record["probe"] == probe and record["manifest"] == binding
    trace = record["trace"]
    search_turns = [row["turn"] for row in trace if row["candidate"]]
    selected = {search_turns[0], search_turns[-1]}
    reproduced, raw_count, search_count = 0, 0, 0

    class TapeBot:
        cursor = 0

        def choose_action(self, env, state, legal_actions=None):
            nonlocal reproduced, raw_count, search_count
            legal = env.legal_actions(state) if legal_actions is None else legal_actions
            row = trace[self.cursor]
            self.cursor += 1
            assert row["state_hash"] == exp.digest(serialize_state_snapshot(state))
            assert row["turn"] == state.turn_index and row["player"] == state.current_player
            assert row["candidate"] == (state.current_player == record["candidate_seat"])
            assert row["legal_count"] == len(legal)
            assert math.isfinite(row["decision_seconds"]) and row["decision_seconds"] > 0
            assert math.isfinite(row["cpu_seconds"]) and row["cpu_seconds"] >= 0
            matching = [a for a in legal if _serialize_action(a) == row["action"]]
            assert len(matching) == 1
            action = matching[0]
            assert repr(canonical_successor_action_key(state, action)) == row["action_key"]
            if row["candidate"]:
                search_count += 1
                assert row["search_seed"] == exp.move_seed(record["seed"], record["candidate_seat"], state.turn_index)
                audit_root(state, legal, row)
                if state.turn_index in selected:
                    config = replace(SearchConfig(**exp.CONFIG["search"]), simulations=row["simulations"])
                    result = AlphaZeroSearch(exp._VALUE, config).run(env, state, legal, self_play=False, seed=row["search_seed"])
                    assert _serialize_action(result.action) == row["action"]
                    for key, value in exp.search_record(result).items():
                        assert value == row[key], (path.name, state.turn_index, key)
                    reproduced += 1
            else:
                raw_count += 1
                assert exp._RAW.choose_action(env, state, legal) == action
            return action

    tape = TapeBot()
    replay = collect_game_replay(tape, tape, seed=record["seed"], max_turns=exp.CONFIG["max_turns"],
                                 repetition_limit=exp.CONFIG["repetition_limit"], no_progress_limit=exp.CONFIG["no_progress_limit"])
    assert tape.cursor == len(trace)
    for key, value in exp.replay_summary(replay).items():
        assert exp.digest(value) == exp.digest(record[key]), (path.name, key)
    points = .5 if record["winner"] is None else float(record["winner"] == record["candidate_seat"])
    assert record["candidate_points"] == points
    return {"index": index, "decisions": len(trace), "raw_reproduced": raw_count,
            "roots_audited": search_count, "searches_reproduced": reproduced,
            "valid_natural": record["valid_natural"], "sha256": exp.file_hash(path)}


def independent_primary(records):
    # Independently recover point scores and resample paired units in smaller blocks.
    pairs = []
    for i in range(0, len(records), 2):
        a, b = records[i:i+2]
        assert a["seed"] == b["seed"] and (a["candidate_seat"], b["candidate_seat"]) == (0, 1)
        scores = [.5 if r["winner"] is None else int(r["winner"] == r["candidate_seat"]) for r in (a, b)]
        pairs.append(sum(scores)/2)
    pair_values = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(exp.CONFIG["bootstrap_seed"])
    draws = []
    for _ in range(exp.CONFIG["bootstrap_samples"]//500):
        indices = rng.integers(len(pairs), size=(500, len(pairs)))
        draws.extend(pair_values[indices].mean(axis=1).tolist())
    return float(pair_values.mean()), [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def main():
    started = time.perf_counter()
    with exp.process_lock(exp.OUTPUT / "process.lock"):
        current = exp.identity()
        assert current == exp.read_json(exp.OUTPUT / "manifest.json")
        binding = exp.digest(current)
        result = exp.read_json(exp.OUTPUT / "result.json")
        assert result["manifest"] == binding
        paths = [exp.OUTPUT / "games" / f"game_{i:04d}.json" for i in range(exp.CONFIG["games"])]
        assert len(list((exp.OUTPUT / "games").glob("*.json"))) == len(paths)
        checks = []
        with ProcessPoolExecutor(max_workers=exp.CONFIG["workers"], initializer=exp.initialize_worker) as pool:
            futures = [pool.submit(audit_game, p, binding) for p in paths]
            for completed, future in enumerate(as_completed(futures), 1):
                checks.append(future.result())
                if completed % 40 == 0:
                    print(json.dumps({"audited": completed, "total": len(paths)}), flush=True)
        records = [exp.read_json(p) for p in paths]
        assert result["game_hashes"] == {p.name: exp.file_hash(p) for p in paths}
        assert result["summary"] == exp.summarize(records)
        share, interval = independent_primary(records)
        assert share == result["summary"]["candidate_score_share"]
        assert interval == result["summary"]["paired_ci95"]
        assert exp.identity() == current
        report = {"passed": True, "manifest": binding, "result_sha256": exp.file_hash(exp.OUTPUT / "result.json"),
                  "games": len(paths), "decisions": sum(c["decisions"] for c in checks),
                  "raw_reproduced": sum(c["raw_reproduced"] for c in checks),
                  "roots_audited": sum(c["roots_audited"] for c in checks),
                  "searches_reproduced": sum(c["searches_reproduced"] for c in checks),
                  "dependencies": len(current["dependencies"]),
                  "prior_dependencies": len(exp.read_json(exp.PRIOR)["dependencies"]),
                  "independent_score_share": share, "independent_paired_ci95": interval,
                  "elapsed_seconds": time.perf_counter()-started}
        exp.atomic_write(exp.OUTPUT / "verification.json", report)
        exp.atomic_write(exp.OUTPUT / "final_status.json", {
            "status": "complete", "decision": result["summary"]["decision"],
            "manifest": binding, "result_sha256": report["result_sha256"],
            "verification_sha256": exp.file_hash(exp.OUTPUT / "verification.json")})
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

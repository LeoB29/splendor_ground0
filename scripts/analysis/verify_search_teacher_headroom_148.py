"""Independent 148 trajectory, original policy, public value and endpoint audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
from pathlib import Path
import runpy
import time

import numpy as np

from splendor_ai.alphazero import run_search_teacher_headroom as exp
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig
from splendor_ai.encoding import actions_by_canonical_successor_key, canonical_successor_action_key
from splendor_ai.training.replay import collect_game_replay, serialize_state_snapshot, _serialize_action

independent_value = runpy.run_path("scripts/analysis/verify_public_value_search_144.py")["independent_value"]


def initialize():
    exp.initialize_worker()


def audit_root(state, legal, row):
    grouped = actions_by_canonical_successor_key(state, legal)
    edges = row["edges"]
    keys = [repr(key) for key in grouped]
    assert [edge["key"] for edge in edges] == keys
    assert row["mode"] == "real" and row["arm"] in ("long", "short")
    assert row["raw_network_value"] == independent_value(state)
    by_key = {}
    for score, action in zip(exp._RAW.score_actions(state, legal), legal, strict=True):
        key = repr(canonical_successor_action_key(state, action))
        by_key[key] = max(score, by_key.get(key, -math.inf))
    largest = max(by_key.values())
    weights = [math.exp(by_key[key]-largest) for key in keys]
    priors = [w/sum(weights) for w in weights]
    total = sum(edge["visits"] for edge in edges)
    assert total == row["simulations"] >= 1
    for edge, prior in zip(edges, priors, strict=True):
        assert type(edge["visits"]) is int and edge["visits"] >= 0
        assert edge["probability"] == edge["visits"]/total
        assert edge["prior"] == edge["network_prior"] == prior
        assert math.isfinite(edge["q"]) and -1.000000000001 <= edge["q"] <= 1.000000000001
    best = max(range(len(edges)), key=lambda i: (edges[i]["visits"], edges[i]["q"], edges[i]["prior"], edges[i]["key"]))
    assert row["selected_row"] == best and row["action_key"] == edges[best]["key"]
    assert row["root_value"] == sum(e["q"]*e["visits"] for e in edges)/total
    assert row["expansions"] == row["inference_calls"] >= 1
    assert 0 <= row["max_depth"] <= exp.CONFIG["search"]["max_depth"]
    allowance = exp.CONFIG["seconds_by_arm"][row["arm"]]
    assert row["allowance_seconds"] == allowance
    assert row["decision_seconds"] >= row["search_seconds"] >= allowance


def audit_game(path, binding, panel, probe=False):
    if exp._RAW is None:
        initialize()
    path = Path(path)
    record = exp.read_json(path)
    index = record["index"]
    assert path.name == f"game_{index:04d}.json"
    assert (record["seed"], record["candidate_seat"]) == exp.schedule(panel, index, probe)
    assert record["panel"] == panel and record["probe"] == probe and record["manifest"] == binding
    trace = record["trace"]
    selected = set()
    timed_arms = [arm for arm in exp.arms(panel) if arm != "raw"]
    for arm in timed_arms:
        turns = [row["turn"] for row in trace if row["arm"] == arm]
        assert turns
        selected.update([turns[0], turns[-1]])
    reproduced = {arm: 0 for arm in timed_arms}
    root_counts = {arm: 0 for arm in timed_arms}
    raw_count = 0

    class TapeBot:
        cursor = 0

        def choose_action(self, env, state, legal_actions=None):
            nonlocal raw_count
            legal = env.legal_actions(state) if legal_actions is None else legal_actions
            row = trace[self.cursor]
            self.cursor += 1
            assert row["state_hash"] == exp.digest(serialize_state_snapshot(state))
            assert row["turn"] == state.turn_index and row["player"] == state.current_player
            assert row["arm"] == ("long" if state.current_player == record["candidate_seat"] else exp.arms(panel)[1])
            assert row["legal_count"] == len(legal)
            assert math.isfinite(row["decision_seconds"]) and row["decision_seconds"] > 0
            assert math.isfinite(row["cpu_seconds"]) and row["cpu_seconds"] >= 0
            matching = [a for a in legal if _serialize_action(a) == row["action"]]
            assert len(matching) == 1
            action = matching[0]
            assert repr(canonical_successor_action_key(state, action)) == row["action_key"]
            if row["arm"] == "raw":
                assert _serialize_action(exp._RAW.choose_action(env, state, legal)) == row["action"]
                assert all(key not in row for key in ("search_seed", "search_seconds", "edges", "allowance_seconds"))
                raw_count += 1
            else:
                assert row["search_seed"] == exp.move_seed(record["seed"], state.current_player, state.turn_index)
                audit_root(state, legal, row)
                root_counts[row["arm"]] += 1
                if state.turn_index in selected:
                    config = replace(SearchConfig(**exp.CONFIG["search"]), simulations=row["simulations"])
                    result = AlphaZeroSearch(exp._VALUE, config).run(env, state, legal, self_play=False, seed=row["search_seed"])
                    assert _serialize_action(result.action) == row["action"]
                    for key, value in exp.search_record(result).items():
                        assert value == row[key], (path.name, state.turn_index, key)
                    reproduced[row["arm"]] += 1
            return action

    tape = TapeBot()
    replay = collect_game_replay(tape, tape, seed=record["seed"], max_turns=exp.CONFIG["max_turns"],
                                 repetition_limit=exp.CONFIG["repetition_limit"], no_progress_limit=exp.CONFIG["no_progress_limit"])
    assert tape.cursor == len(trace)
    for key, value in exp.replay_summary(replay).items():
        assert exp.digest(value) == exp.digest(record[key]), (path.name, key)
    points = .5 if record["winner"] is None else float(record["winner"] == record["candidate_seat"])
    assert record["candidate_points"] == points
    return {"panel": panel, "index": index, "decisions": len(trace), "roots_audited": root_counts,
            "searches_reproduced": reproduced, "raw_actions_reproduced": raw_count,
            "valid_natural": record["valid_natural"], "sha256": exp.file_hash(path)}


def independent_primary(panel, records):
    pairs = []
    for i in range(0, len(records), 2):
        a, b = records[i:i+2]
        assert a["seed"] == b["seed"] and (a["candidate_seat"], b["candidate_seat"]) == (0, 1)
        scores = [.5 if r["winner"] is None else int(r["winner"] == r["candidate_seat"]) for r in (a, b)]
        pairs.append(sum(scores)/2)
    values = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(exp.CONFIG["panels"][panel]["bootstrap_seed"])
    draws = []
    for _ in range(exp.CONFIG["bootstrap_samples"]//500):
        draws.extend(values[rng.integers(len(pairs), size=(500, len(pairs)))].mean(axis=1).tolist())
    return float(values.mean()), [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def independent_gate(panel, records, share, interval):
    natural = all(r["valid_natural"] for r in records)
    time_ok = True
    for arm in exp.arms(panel):
        if arm == "raw":
            continue
        rows = [t for r in records for t in r["trace"] if t["arm"] == arm]
        allowance = exp.CONFIG["seconds_by_arm"][arm]
        seconds = [t["decision_seconds"] for t in rows]
        time_ok &= bool(rows and sum(seconds)/len(seconds) <= allowance*1.10
                        and float(np.percentile(seconds, 95)) <= allowance*1.25
                        and all(t["allowance_seconds"] == allowance and
                                t["decision_seconds"] >= t["search_seconds"] >= allowance for t in rows))
    if not (natural and time_ok):
        return "invalid_integrity_or_budget"
    if share >= .55 and interval[0] > .5:
        return "material_superiority"
    if interval[1] < .5:
        return "candidate_weaker"
    if interval[0] > .5:
        return "positive_below_material_threshold"
    return "improvement_not_established"


def verify(panel):
    started = time.perf_counter()
    root = exp.OUTPUT / panel
    with exp.process_lock(root / "audit.lock"):
        if (root / "verification.json").exists():
            raise FileExistsError("Audit already completed; preserve evidence")
        current = exp.identity()
        assert current == exp.read_json(exp.OUTPUT / "manifest.json")
        binding = exp.digest(current)
        if panel == "competence":
            assert exp.audited_panel_passed("compute", binding)
        result = exp.read_json(root / "result.json")
        assert result["manifest"] == binding
        paths = [root / "games" / f"game_{i:04d}.json" for i in range(exp.CONFIG["games_per_panel"])]
        assert len(list((root / "games").glob("*.json"))) == len(paths)
        checks = []
        with ProcessPoolExecutor(max_workers=exp.CONFIG["workers"], initializer=initialize) as pool:
            futures = [pool.submit(audit_game, p, binding, panel) for p in paths]
            for completed, future in enumerate(as_completed(futures), 1):
                checks.append(future.result())
                if completed % 40 == 0:
                    print(json.dumps({"panel": panel, "audited": completed, "total": len(paths)}), flush=True)
        records = [exp.read_json(p) for p in paths]
        assert result["game_hashes"] == {p.name: exp.file_hash(p) for p in paths}
        assert result["summary"] == exp.summarize(panel, records)
        share, interval = independent_primary(panel, records)
        assert share == result["summary"]["candidate_score_share"]
        assert interval == result["summary"]["paired_ci95"]
        independent_decision = independent_gate(panel, records, share, interval)
        assert independent_decision == result["summary"]["decision"]
        assert exp.identity() == current
        timed_arms = [a for a in exp.arms(panel) if a != "raw"]
        report = {"passed": True, "panel": panel, "manifest": binding,
                  "result_sha256": exp.file_hash(root / "result.json"),
                  "games": len(paths), "decisions": sum(c["decisions"] for c in checks),
                  "roots_audited": {a: sum(c["roots_audited"][a] for c in checks) for a in timed_arms},
                  "searches_reproduced": {a: sum(c["searches_reproduced"][a] for c in checks) for a in timed_arms},
                  "raw_actions_reproduced": sum(c["raw_actions_reproduced"] for c in checks),
                  "dependencies": len(current["dependencies"]),
                  "prior_dependencies": len(exp.read_json(exp.PRIOR)["dependencies"]),
                  "independent_score_share": share, "independent_paired_ci95": interval,
                  "independent_decision": independent_decision, "elapsed_seconds": time.perf_counter()-started,
                  "completed_at": exp.utc_now()}
        exp.atomic_write(root / "verification.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=exp.PANELS)
    verify(parser.parse_args().panel)


if __name__ == "__main__":
    main()

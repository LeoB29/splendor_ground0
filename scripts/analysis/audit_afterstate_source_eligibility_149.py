"""Read-only development-source feasibility audit after 149's frozen stop.

No new policy inference, continuation outcomes, training or test games. The
matching is a feasibility witness, not an authorized replacement panel.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import torch
from splendor_ai.alphazero import run_afterstate_advantage as p

DEST = p.OUTPUT / "source_eligibility_audit.json"


def initialize():
    torch.set_num_threads(1)


def inspect(index, expected):
    path = p.CORPUS / "games" / f"game_{index:04d}.json"
    assert p.file_hash(path) == expected
    record = p.read_json(path)
    states = p.reconstruct_source(record)
    by_cell = [[s.turn_index for s in states if p.eligible(s, cell)] for cell in range(6)]
    all_positions = Counter(p.phase(s.turn_index)*2+s.current_player for s in states)
    return {"index": index, "eligible_turns": by_cell, "all_positions": [all_positions[c] for c in range(6)]}


def match(rows, quota):
    """Augmenting-path matching: each game capacity1; each cell capacity quota."""
    assigned = [[] for _ in range(6)]
    cells = {r["index"]: [c for c, turns in enumerate(r["eligible_turns"]) if turns] for r in rows}

    def place(game, visited):
        for cell in cells[game]:
            if cell in visited:
                continue
            visited.add(cell)
            if len(assigned[cell]) < quota:
                assigned[cell].append(game)
                return True
            for k, previous in enumerate(assigned[cell]):
                if place(previous, visited):
                    assigned[cell][k] = game
                    return True
        return False

    for row in rows:
        place(row["index"], set())
    flattened = [g for group in assigned for g in group]
    assert len(set(flattened)) == len(flattened)
    assert all(c in cells[g] for c, group in enumerate(assigned) for g in group)
    return {"counts": [len(a) for a in assigned], "feasible": all(len(a) == quota for a in assigned),
            "witness_game_ids_by_cell": assigned}


def main():
    started = time.perf_counter()
    p.check_identity()
    assert not (p.OUTPUT / "train_targets_lock.json").exists()
    sealed = p.read_json(p.CORPUS / "development_summary.json")["files"]
    rows = []
    with ProcessPoolExecutor(max_workers=8, initializer=initialize) as pool:
        jobs = [pool.submit(inspect, i, sealed[str(i)]) for i in range(4608)]
        for done, job in enumerate(as_completed(jobs), 1):
            rows.append(job.result())
            if done % 256 == 0 or done == len(jobs):
                print({"eligibility_audit": done, "total": len(jobs)}, flush=True)
    result = {}
    for split, start, stop, quota in (("train", 0, 4096, 128), ("validation", 4096, 4608, 32)):
        group = sorted([r for r in rows if start <= r["index"] < stop], key=lambda r: r["index"])
        eligible_games = [sum(bool(r["eligible_turns"][c]) for r in group) for c in range(6)]
        cyclic = [min(quota, sum(bool(r["eligible_turns"][c]) and (r["index"]-start)%6 == c for r in group)) for c in range(6)]
        result[split] = {"games": len(group), "eligible_games_by_cell": eligible_games,
            "eligible_positions_by_cell": [sum(len(r["eligible_turns"][c]) for r in group) for c in range(6)],
            "all_positions_by_cell": [sum(r["all_positions"][c] for r in group) for c in range(6)],
            "frozen_cyclic_counts": cyclic, "balanced_matching_feasibility": match(group, quota)}
    p.check_identity()
    p.save_new(DEST, {"seconds": time.perf_counter()-started, "passed": True, "new_inference": False,
              "new_continuations": False, "training": False, "test_games": False, "splits": result})
    print({split: {k: v for k, v in info.items() if k != "balanced_matching_feasibility"}
           | {"balanced_matching_feasible": info["balanced_matching_feasibility"]["feasible"]}
           for split, info in result.items()}, flush=True)


if __name__ == "__main__":
    main()

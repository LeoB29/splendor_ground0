"""Read-only saved-outcome diagnostics supporting the post160 recommendation.

No game/inference modules, simulation, fitting or gameplay. No frozen files edited.
The chosen roots precede their own selection outcomes: one hash-assigned source
seat per seed pair, first candidate128 decision in each phase. All findings are
exploratory reused-data diagnostics, not independent128-label quality tests.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/world_budget_160"
OUTPUT = SOURCE / "review_sampling_20260916"
BUDGETS = (8, 16, 32, 64)
PARTITIONS = 64
BOOTSTRAPS = 10000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    assert not OUTPUT.exists()
    final, _ = read(SOURCE / "final_status.json")
    result, result_sha = read(SOURCE / "result.json")
    assert final["complete"] and final["prespecified_audit_pass"]
    assert result_sha == final["result_sha256"]
    lock, lock_sha = read(SOURCE / "main_collect_lock.json")
    assert lock["manifest"] == final["manifest"] and len(lock["files"]) == 400
    permutations = np.array([np.random.default_rng(16061600 + i).permutation(128)
                             for i in range(PARTITIONS)])
    roots, pair_values, label_changes = [], [], []
    all128_roots = 0
    for pair in range(200):
        assigned_seat = int(digest(["post160-review-source-seat-v1", pair]), 16) % 2
        source_games = []
        for seat in (0, 1):
            index = pair * 2 + seat
            game, sha = read(SOURCE / "main/games" / f"game_{index:04d}.json")
            assert sha == lock["files"][str(index)] and game["manifest"] == final["manifest"]
            assert game["spec"]["index"] == index and game["spec"]["candidate_seat"] == seat
            assert "failure" not in game
            all128_roots += sum(move["actor"] == seat for move in game["trace"])
            source_games.append((game, sha))
        game, game_sha = source_games[assigned_seat]
        chosen_phases, values, changes = set(), [], []
        for move in game["trace"]:
            phase = min(move["turn"] // 20, 2)
            if move["actor"] != assigned_seat or phase in chosen_phases:
                continue
            chosen_phases.add(phase)
            path = SOURCE / "main/roots" / f"game_{pair*2+assigned_seat:04d}" / f"turn_{move['turn']:03d}.json"
            record, sha = read(path)
            assert sha == move["root_sha256"] and record["manifest"] == final["manifest"]
            assert "failure" not in record and len(record["worlds"]) == 128
            scores = (1 + np.array([[item["wdl"] for item in world["outcomes"]]
                                   for world in record["worlds"]], dtype=float)) / 2
            assert np.isin(scores, [0., .5, 1.]).all()
            assert 1 <= scores.shape[1] <= 4
            selected128 = int(scores.mean(axis=0).argmax())
            selected32 = int(scores[:32].mean(axis=0).argmax())
            assert selected128 == record["choice"]
            assert record["candidates"]["actions"][selected128] == move["action"]
            heldout = scores[permutations[:, 64:]].mean(axis=1)
            gains = []
            for budget in BUDGETS:
                choices = scores[permutations[:, :budget]].mean(axis=1).argmax(axis=1)
                gains.append(float(np.mean(heldout[np.arange(PARTITIONS), choices] - heldout[:, 0])))
            halves = scores[permutations[:, :64]].mean(axis=1).argmax(axis=1)
            row = {"pair": pair, "source_index": pair*2+assigned_seat, "actor": assigned_seat,
                   "phase": phase, "turn": move["turn"], "game_sha256": game_sha,
                   "source_path": path.relative_to(ROOT).as_posix(), "source_sha256": sha,
                   "support": scores.shape[1], "choice32": selected32, "choice128": selected128,
                   "label_changed": selected32 != selected128, "heldout_gain_score": gains,
                   "two64_halves_disagreement": float(np.mean(halves != heldout.argmax(axis=1)))}
            roots.append(row)
            values.append(gains)
            changes.append(float(row["label_changed"]))
        assert values
        pair_values.append(np.mean(values, axis=0))
        label_changes.append(np.mean(changes))
    observed = np.array(pair_values)
    rng = np.random.default_rng(16061601)
    indices = rng.integers(200, size=(BOOTSTRAPS, 200))
    bootstrap = observed[indices].mean(axis=1)
    increments = np.diff(observed, axis=1)
    bootstrap_increments = np.diff(bootstrap, axis=1)
    summary = {
        "status": "exploratory_saved_data_only; not a new strength or128-label validation experiment",
        "source_result_sha256": result_sha, "source_game_lock_sha256": lock_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_games_hashed": 400, "available_128_teacher_roots": all128_roots,
        "roots": len(roots), "source_pairs": 200,
        "phase_counts": {str(phase): sum(r["phase"] == phase for r in roots) for phase in range(3)},
        "source_seat_counts": {str(seat): sum(r["actor"] == seat for r in roots) for seat in (0, 1)},
        "selection_budgets": BUDGETS, "heldout_worlds": 64, "partitions_per_root": PARTITIONS,
        "pair_bootstrap_resamples": BOOTSTRAPS,
        "heldout_gain_over_proposal0_pp": (100*observed.mean(axis=0)).tolist(),
        "exploratory_pointwise_ci95_pp": (100*np.quantile(bootstrap, [.025, .975], axis=0).T).tolist(),
        "successive_budget_increment_pp": (100*increments.mean(axis=0)).tolist(),
        "exploratory_increment_ci95_pp": (100*np.quantile(bootstrap_increments, [.025, .975], axis=0).T).tolist(),
        "32prefix_vs128_label_change_fraction_pair_weighted": float(np.mean(label_changes)),
        "32prefix_vs128_label_changes": sum(r["label_changed"] for r in roots),
        "two64_halves_disagreement": float(np.mean([r["two64_halves_disagreement"] for r in roots])),
        "limitations": [
            "The128-selected move is not ground truth; label changes are descriptive, not accuracy gains.",
            "Disjoint64/64 partitions evaluate conditional choices up to64 samples, not independent128-label quality.",
            "All partitions/phases remain within200 source-pair units; reuse does not create extra independent observations.",
            "Visited states arise from the160 matchup; these are not fresh gameplay scores or label learnability guarantees.",
            "The three first-eligible phase positions per hash-assigned source game are a narrow diagnostic panel, not all training roots.",
        ],
    }
    OUTPUT.mkdir()
    save(OUTPUT / "roots.json", roots)
    summary["roots_sha256"] = hashlib.sha256((OUTPUT / "roots.json").read_bytes()).hexdigest()
    save(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

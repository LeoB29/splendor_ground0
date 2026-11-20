"""153 independent public-effect arithmetic, target sums, SVD fits and metrics."""
import argparse
from collections import Counter
import hashlib
import json
import math
import time

import torch

from splendor_ai.alphazero import run_public_feature_advantage as p
from splendor_ai.engine.constants import TOKEN_COLORS


def compare(a, b, tolerance=1e-12):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for k in a:
            compare(a[k], b[k], tolerance)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y, tolerance)
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, (a, b)
    else:
        assert a == b, (a, b)


def public_effects(state, action):
    """No engine step, decomposition, chance resolution or hidden card access."""
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    tokens = dict(own.tokens)
    bonuses = dict(own.bonuses)
    score, bought, reserved, nobles = own.score, len(own.purchased_cards), len(own.reserved_cards), len(own.nobles)
    kind = action.action_type.name
    if kind in ("TAKE_TOKENS", "RESERVE_VISIBLE", "RESERVE_DECK"):
        for c in action.take_tokens:
            tokens[c] += 1
        for c in action.return_tokens:
            tokens[c] -= 1
        if kind.startswith("RESERVE"):
            reserved += 1
    elif kind in ("BUY_VISIBLE", "BUY_RESERVED"):
        if kind == "BUY_VISIBLE":
            card = state.visible_tier_cards[action.tier][action.market_index]
        else:
            card = own.reserved_cards[action.reserved_index]
            reserved -= 1
        bought += 1
        score += card.points
        bonuses[card.bonus_color] += 1
        for c in action.spend_tokens:
            tokens[c] -= 1
        if action.noble_id is not None:
            noble = next(n for n in state.nobles if n.noble_id == action.noble_id)
            assert all(bonuses[c] >= n for c, n in noble.requirement.items())
            score += noble.points
            nobles += 1
    else:
        assert kind == "PASS"
    values = [(score-other.score)/15]
    values.extend((bonuses[c]-other.bonuses[c])/7 for c in TOKEN_COLORS)
    values.extend((tokens[c]-other.tokens[c])/7 for c in TOKEN_COLORS)
    values.extend([(tokens["gold"]-other.tokens["gold"])/5,
                   (bought-len(other.purchased_cards))/20,
                   (reserved-len(other.reserved_cards))/3, (nobles-len(other.nobles))/3])
    return values


def polynomial(phi):
    result = list(phi)
    for i, value in enumerate(phi):
        result.extend(value*other for other in phi[i:])
    return result


def independent_folds(samples):
    rows = [None]*len(samples)
    for cell in range(6):
        sources = []
        for i, sample in enumerate(samples):
            if sample["cell"] == cell:
                message = json.dumps(["153-fold-v1", sample["source_seed"]], separators=(",", ":"))
                sources.append((hashlib.sha256(message.encode()).hexdigest(), sample["source_seed"], i))
        for rank, (_, seed, i) in enumerate(sorted(sources)):
            rows[i] = {"row": i, "source_seed": seed, "cell": cell,
                       "sample_sha256": p.old.digest(samples[i]), "fold": (rank+cell*2)%5}
    return rows


def target_row(base, split, sample, lock, worlds):
    path = base/"targets"/split/f"state_{sample['index']:04d}.json"
    assert p.old.file_hash(path) == lock["files"][str(sample["index"])]["sha256"]
    record = p.old.read_json(path)
    assert record["index"] == sample["index"] and record["split"] == split
    assert record["sample_sha256"] == p.old.digest(sample)
    assert len(record["worlds"]) == {"train": 32, "validation": 64, "new": 8}[split]
    n = len(sample["actions"])
    sums = [0.]*n
    start = {"train": 1490100000, "validation": 1491100000, "new": 1521000000}[split]
    for i, world in enumerate(record["worlds"][:worlds]):
        assert world["seed"] == start+1000*sample["index"]+i
        assert len(world["outcomes"]) == n
        for j, row in enumerate(world["outcomes"]):
            assert row["reason"] in ("completed", "forced_pass_deadlock")
            expected = 0. if row["winner"] is None else 1. if row["winner"] == sample["actor"] else -1.
            assert row["wdl"] == expected
            sums[j] += expected
    q = [v/worlds for v in sums]
    return q+[q[0]]*(4-n)


def verify_data():
    freeze = p.old.read_json(p.OUTPUT/"freeze.json")
    assert freeze["manifest"] == p.check_identity()
    assert freeze["fold_map_sha256"] == p.old.file_hash(p.OUTPUT/"fold_map.json")
    lock = p.old.read_json(p.OUTPUT/"data_lock.json")
    assert lock["freeze_sha256"] == p.old.file_hash(p.OUTPUT/"freeze.json")
    data = {s: p.load_data(s) for s in ("train", "validation")}
    for split, n in (("train", 3072), ("validation", 192)):
        d = data[split]
        assert set(d) == {"phi", "x", "q", "y", "mask"}
        for key, shape in (("phi", (n, 4, 15)), ("x", (n, 4, 135)), ("q", (n, 4)), ("y", (n, 4))):
            assert d[key].dtype == torch.float64 and tuple(d[key].shape) == shape and torch.isfinite(d[key]).all()
        assert d["mask"].dtype == torch.bool and tuple(d["mask"].shape) == (n, 4)
    samples = {s: [] for s in data}
    counts = Counter()
    for base, split, output, offset, count, worlds in (
        (p.old.OUTPUT, "train", "train", 0, 768, 8),
        (p.parent.OUTPUT, "new", "train", 768, 2304, 8),
        (p.old.OUTPUT, "validation", "validation", 0, 192, 64)):
        panel = p.old.read_json(base/f"{split}_panel.json")
        targets = p.old.read_json(base/f"{split}_targets_lock.json")
        assert targets["all_natural"] and len(panel["samples"]) == count
        assert set(targets["files"]) == {str(i) for i in range(count)}
        for i, s in enumerate(panel["samples"]):
            assert s["index"] == i and targets["files"][str(i)]["natural"]
            samples[output].append(s)
            state = p.old.deserialize_state_snapshot(s["snapshot"])
            assert state.current_player == s["actor"] and state.turn_index == s["turn"]
            phase = 0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2
            assert s["cell"] == 2*phase+s["actor"] and not state.terminal
            before = p.old.serialize_state_snapshot(state)
            phi = []
            for a in s["actions"]:
                action = p.old.decode_action(a)
                counts[action.action_type.name] += 1
                counts["noble_actions"] += action.noble_id is not None
                phi.append(public_effects(state, action))
            assert p.old.serialize_state_snapshot(state) == before
            phi += [phi[0]]*(4-len(phi))
            psi = [polynomial(v) for v in phi]
            x = [[a-b for a, b in zip(row, psi[0])] for row in psi]
            q = target_row(base, split, s, targets, worlds)
            rebuilt = {"phi": torch.tensor(phi, dtype=torch.float64), "x": torch.tensor(x, dtype=torch.float64),
                       "q": torch.tensor(q, dtype=torch.float64), "y": torch.tensor([v-q[0] for v in q], dtype=torch.float64),
                       "mask": torch.tensor([0 < j < len(s["actions"]) for j in range(4)])}
            for key, value in rebuilt.items():
                assert torch.equal(value, data[output][key][i+offset]), (split, i, key)
        print({"audit": "public_effects_targets", "split": split, "states": count}, flush=True)
    sources = [s["source_seed"] for rows in samples.values() for s in rows]
    assert len(sources) == len(set(sources)) == 3264
    assert Counter(s["cell"] for s in samples["train"]) == {i: 512 for i in range(6)}
    assert Counter(s["cell"] for s in samples["validation"]) == {i: 32 for i in range(6)}
    folds = independent_folds(samples["train"])
    assert folds == p.old.read_json(p.OUTPUT/"fold_map.json")
    for cell in range(6):
        n = Counter(r["fold"] for r in folds if r["cell"] == cell)
        assert max(n.values())-min(n.values()) <= 1 and len(n) == 5
    for split, name in (("train", "C"), ("validation", "validation")):
        original = p.parent.load_data(name)
        for k in ("q", "y", "mask"):
            assert torch.equal(data[split][k], original[k].to(data[split][k].dtype))
    return data, folds, dict(counts)


def independent_solve(data, ridge):
    x, y, masks = data["x"].tolist(), data["y"].tolist(), data["mask"].tolist()
    width = len(x[0][0])
    rows, targets = [], []
    for values, labels, mask in zip(x, y, masks):
        weight = 1/math.sqrt(len(y)*max(sum(mask), 1))
        for j in range(len(mask)):
            if mask[j]:
                rows.append([v*weight for v in values[j]])
                targets.append(labels[j]*weight)
    for k in range(width):
        rows.append([math.sqrt(ridge) if j == k else 0. for j in range(width)])
        targets.append(0.)
    a = torch.tensor(rows, dtype=torch.float64)
    b = torch.tensor(targets, dtype=torch.float64)
    fitted = torch.linalg.lstsq(a, b, rcond=1e-14, driver="gelsd")
    assert int(fitted.rank) == width
    coefficients = fitted.solution
    residual = float((a.T@(a@coefficients-b)).abs().max())
    assert residual <= 1e-10 and torch.isfinite(coefficients).all()
    return coefficients, residual


def independent_prediction(data, coefficients):
    # Same sealed polynomial differences; independent scalar dot products.
    w = coefficients.tolist()
    values = [[sum(v*c for v, c in zip(row, w)) for row in state] for state in data["x"].tolist()]
    choice = [max([0]+[j for j in range(1, 4) if mask[j]], key=lambda j: pred[j])
              for pred, mask in zip(values, data["mask"].tolist())]
    assert all(row[0] == 0. for row in values)
    return {"advantages": values, "choices": choice}


def independent_metric(data, row):
    y, q, masks = (data[k].tolist() for k in ("y", "q", "mask"))
    n = len(y)
    error = sum(sum((pred[j]-label[j])**2 for j in range(4) if mask[j])/max(sum(mask), 1)
                for pred, label, mask in zip(row["advantages"], y, masks))/n
    gain = sum(values[j]-values[0] for values, j in zip(q, row["choices"]))/n
    return {"mse": error, "wdl_gain_over_035": gain, "baseline_fraction": row["choices"].count(0)/n}


def independent_collisions(data):
    result = {"states": len(data["y"]), "real_alternates": 0, "real_pairs": 0, "aliased_pairs": 0,
              "states_with_aliases": 0, "zero_difference_alternates": 0, "all_candidates_same_states": 0, "baseline_only_states": 0}
    for phi, mask in zip(data["phi"].tolist(), data["mask"].tolist()):
        rows = [phi[0]]+[phi[j] for j in range(1, 4) if mask[j]]
        frequencies = Counter(tuple(v) for v in rows)
        result["real_alternates"] += len(rows)-1
        result["real_pairs"] += len(rows)*(len(rows)-1)//2
        result["aliased_pairs"] += sum(n*(n-1)//2 for n in frequencies.values())
        result["states_with_aliases"] += len(frequencies) < len(rows)
        result["zero_difference_alternates"] += sum(row == rows[0] for row in rows[1:])
        result["all_candidates_same_states"] += len(frequencies) == 1
        result["baseline_only_states"] += len(rows) == 1
    return result


def verify_result(data, rows, training, saved, selected):
    val = data["validation"]
    public = independent_metric(val, rows["public"])
    controls = [{"seed": r["seed"], "selected_update": r["selected_update"], **independent_metric(val, r)} for r in rows["C"]]
    assert [r["seed"] for r in controls] == [149001, 149002, 149003]
    mean = {k: sum(r[k] for r in controls)/3 for k in public}
    neutral = {"advantages": [[0.]*4 for _ in val["y"]], "choices": [0]*len(val["y"])}
    zero = independent_metric(val, neutral)["mse"]
    reductions = {"versus_C": 1-public["mse"]/mean["mse"], "versus_zero": 1-public["mse"]/zero}
    checks = {"mse_versus_C": reductions["versus_C"] >= .03, "mse_versus_zero": reductions["versus_zero"] >= .03,
              "wdl_versus_035": public["wdl_gain_over_035"] >= .02, "wdl_versus_C": public["wdl_gain_over_035"] > mean["wdl_gain_over_035"]}
    contrasts = {name: {"mse_difference": public["mse"]-r["mse"], "wdl_difference": public["wdl_gain_over_035"]-r["wdl_gain_over_035"]}
                 for name, r in [(str(r["seed"]), r) for r in controls]+[("C_mean", mean)]}
    rebuilt = {"public": public, "C": {"per_seed": controls, "mean": mean}, "zero_mse": zero,
               "relative_mse_reduction": reductions, "development_gates": checks, "development_passed": all(checks.values()),
               "public_minus_control": contrasts, "development_only": True, "confirmatory": False,
               "training": independent_metric(data["train"], training), "selected_penalty": selected,
               "collisions": {s: independent_collisions(d) for s, d in data.items()}}
    compare(rebuilt, saved)
    previous = p.old.read_json(p.parent.OUTPUT/"result.json")
    compare(mean, previous["arms"]["C"]["means"])
    compare(zero, previous["zero_mse"])


def verify_final(data, folds):
    launch = p.old.read_json(p.OUTPUT/"launch.json")
    assert launch["manifest"] == p.check_identity()
    assert launch["preflight_sha256"] == p.old.file_hash(p.OUTPUT/"preflight_audit.json")
    preflight = p.old.read_json(p.OUTPUT/"preflight_audit.json")
    assert preflight["passed"] and preflight["data_lock_sha256"] == p.old.file_hash(p.OUTPUT/"data_lock.json")
    selection = p.old.read_json(p.OUTPUT/"selection_lock.json")
    assert selection["manifest"] == launch["manifest"] and selection["solves"] == 26
    for field, name in (("fits_sha256", "fits.pt"), ("data_lock_sha256", "data_lock.json"), ("fold_map_sha256", "fold_map.json")):
        assert selection[field] == p.old.file_hash(p.OUTPUT/name)
    assert (p.OUTPUT/"selection_lock.json").stat().st_mtime_ns < (p.OUTPUT/"validation_predictions.json").stat().st_mtime_ns
    fitted = torch.load(p.OUTPUT/"fits.pt", weights_only=True)
    assert len(fitted["fits"]) == 26 and len(fitted["oof_predictions"]) == 5
    train = data["train"]
    fold_ids = [r["fold"] for r in folds]
    cv, details, worst = [], [], 0.
    for penalty_index, ridge in enumerate([.001, .01, .1, 1., 10.]):
        oof = torch.zeros_like(train["y"])
        fold_rows = []
        for f in range(5):
            included = [i for i, v in enumerate(fold_ids) if v != f]
            held = [i for i, v in enumerate(fold_ids) if v == f]
            w, residual = independent_solve({k: v[included] for k, v in train.items()}, ridge)
            saved = fitted["fits"][penalty_index*5+f]
            assert saved["fold"] == f and saved["penalty"] == ridge and saved["normal_residual"] <= 1e-10
            torch.testing.assert_close(w, saved["coefficients"], atol=1e-9, rtol=1e-8)
            error = float((w-saved["coefficients"]).abs().max())
            worst = max(worst, error)
            d = {k: v[held] for k, v in train.items()}
            row = independent_prediction(d, w)
            oof[held] = torch.tensor(row["advantages"], dtype=torch.float64)
            fold_rows.append({"fold": f, "train_states": len(included), "held_states": len(held), "mse": independent_metric(d, row)["mse"]})
            details.append({"penalty": ridge, "fold": f, "coefficient_max_difference": error, "augmented_residual": residual})
        torch.testing.assert_close(oof, fitted["oof_predictions"][penalty_index], atol=1e-9, rtol=0)
        pooled = independent_metric(train, {"advantages": oof.tolist(), "choices": [0]*len(oof)})["mse"]
        cv.append({"penalty": ridge, "oof_mse": pooled, "folds": fold_rows})
        print({"audit": "five_cv_fits_reproduced", "penalty": ridge}, flush=True)
    compare(cv, fitted["cv"])
    compare(cv, selection["cv"])
    selected = sorted(cv, key=lambda r: (r["oof_mse"], -r["penalty"]))[0]["penalty"]
    assert selected == selection["selected_penalty"] == fitted["selected_penalty"]
    w, residual = independent_solve(train, selected)
    last = fitted["fits"][-1]
    assert last["fold"] == "all" and last["penalty"] == selected and last["normal_residual"] <= 1e-10
    torch.testing.assert_close(w, last["coefficients"], atol=1e-9, rtol=1e-8)
    error = float((w-last["coefficients"]).abs().max())
    worst = max(worst, error)
    details.append({"penalty": selected, "fold": "all", "coefficient_max_difference": error, "augmented_residual": residual})
    rows = p.old.read_json(p.OUTPUT/"validation_predictions.json")
    training = p.old.read_json(p.OUTPUT/"training_predictions.json")
    compare(independent_prediction(data["validation"], w), rows["public"], 1e-9)
    compare(independent_prediction(train, w), training, 1e-9)
    assert rows["C"] == p.old.read_json(p.parent.OUTPUT/"validation_predictions.json")["C"]
    for row in rows["C"]:
        rebuilt = [max([0]+[j for j in range(1, 4) if m[j]], key=lambda j: v[j]) for m, v in zip(data["validation"]["mask"].tolist(), row["advantages"])]
        assert row["choices"] == rebuilt
    verify_result(data, rows, training, p.old.read_json(p.OUTPUT/"result.json"), selected)
    return {"independent_solves": 26, "coefficient_max_difference": worst, "fits": details,
            "result_sha256": p.old.file_hash(p.OUTPUT/"result.json"), "selection_sha256": p.old.file_hash(p.OUTPUT/"selection_lock.json"),
            "predictions_sha256": p.old.file_hash(p.OUTPUT/"validation_predictions.json"),
            "training_predictions_sha256": p.old.file_hash(p.OUTPUT/"training_predictions.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "final"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    binding = p.check_identity()
    if args.stage == "preflight":
        assert not (p.OUTPUT/"fits.pt").exists() and not (p.OUTPUT/"launch.json").exists()
    data, folds, counts = verify_data()
    result = verify_final(data, folds) if args.stage == "final" else {}
    assert p.check_identity() == binding
    p.old.save_new(p.OUTPUT/f"{args.stage}_audit.json", {"passed": True, "manifest": binding,
        "seconds": time.perf_counter()-started, "states_verified": 3264, "public_action_counts": counts,
        "data_lock_sha256": p.old.file_hash(p.OUTPUT/"data_lock.json"), "parent_replay_audit_sha256": p.PARENT_AUDIT,
        "new_simulations": 0, **result})


if __name__ == "__main__":
    main()

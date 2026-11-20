"""153: fixed public quadratic potential, training-only ridge CV, no simulations."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from splendor_ai.alphazero import run_data_allocation as parent
from splendor_ai.alphazero.afterstate_149 import deterministic_part
from splendor_ai.engine.constants import TOKEN_COLORS

old = parent.old
OUTPUT = Path("outputs/public_feature_advantage_153")
PROTOCOL = Path("docs/engineering/public_feature_advantage_153.md")
VERIFIER = Path("scripts/analysis/verify_public_feature_advantage_153.py")
CONFIG = {
    "train_states": 3072, "validation_states": 192, "features": 135,
    "penalties": [.001, .01, .1, 1., 10.], "folds": 5,
    "fold_tag": "153-fold-v1", "fold_cell_offset": "2*cell",
    "dtype": "float64", "threads": 1, "solver": "normal_equation_solve",
    "audit_solver": "augmented_lstsq_gelsd_rcond_1e-14",
    "coefficient_atol": 1e-9, "coefficient_rtol": 1e-8,
    "prediction_atol": 1e-9, "metric_atol": 1e-12,
    "normal_residual_atol": 1e-10,
    "minimum_mse_reduction": .03, "minimum_wdl_gain": .02,
    "development_only": True, "new_simulations": 0, "neural_refits": 0,
    "fresh_performance_test": False, "arena": False, "promotion": False,
    "automatic_successor": False,
}
PARENT_BINDING = "015aef64dac1775b8e1403092d6f630514dcc7900e9b7348810be78d8499c1a8"
PARENT_AUDIT = "e430a94ae2c307ab532ebf0314434eec5a81dbc6fad3ece059b10c8bc52e765f"
PAIRS = tuple((i, j) for i in range(15) for j in range(i, 15))


def parent_inputs():
    if parent.check_identity() != PARENT_BINDING or old.file_hash(parent.OUTPUT/"final_audit.json") != PARENT_AUDIT:
        raise ValueError("152 identity/audit changed")
    audit = old.read_json(parent.OUTPUT/"final_audit.json")
    final = old.read_json(parent.OUTPUT/"final_status.json")
    if not audit["passed"] or not final["complete"] or final["audit_sha256"] != PARENT_AUDIT:
        raise ValueError("152 not closed")
    paths = [parent.OUTPUT/name for name in (
        "manifest.json", "final_audit.json", "final_status.json", "result.json",
        "selection_lock.json", "validation_predictions.json", "development_lock.json",
        "panel_lock.json", "new_panel.json", "new_targets_lock.json", "sources_lock.json",
        "panel_audit.json", "probe_audit.json")]
    for field, name in (("result_sha256", "result.json"), ("selection_sha256", "selection_lock.json"),
                        ("predictions_sha256", "validation_predictions.json"),
                        ("panel_audit_sha256", "panel_audit.json"), ("probe_audit_sha256", "probe_audit.json")):
        if audit[field] != old.file_hash(parent.OUTPUT/name):
            raise ValueError("152 audited artifact changed")
    selection = old.read_json(parent.OUTPUT/"selection_lock.json")
    for field, name in (("development_lock_sha256", "development_lock.json"),
                        ("new_targets_lock_sha256", "new_targets_lock.json"), ("panel_lock_sha256", "panel_lock.json")):
        if selection[field] != old.file_hash(parent.OUTPUT/name):
            raise ValueError("152 selection binding changed")
    panel_lock = old.read_json(parent.OUTPUT/"panel_lock.json")
    for field, name in (("panel_sha256", "new_panel.json"), ("sources_sha256", "sources_lock.json")):
        if panel_lock[field] != old.file_hash(parent.OUTPUT/name):
            raise ValueError("152 panel lineage changed")
    for name in ("C", "validation"):
        path = parent.OUTPUT/f"{name}_data.pt"
        if old.file_hash(path) != old.read_json(parent.OUTPUT/"development_lock.json")[name]:
            raise ValueError("152 data changed")
        paths.append(path)
    for info in selection["models"]["C"].values():
        path = Path(info["path"])
        if old.file_hash(path) != info["sha256"]:
            raise ValueError("C model changed")
        paths.append(path)
    # Original development panel/target hashes are already checked by the
    # explicit 150 -> 151 -> 152 inventory, never the old dynamic 149 identity.
    paths += [old.OUTPUT/f"{split}_{name}.json" for split in ("train", "validation")
              for name in ("panel", "targets_lock")]
    return {"binding": PARENT_BINDING, "files": {p.as_posix(): old.file_hash(p) for p in paths}}


def identity():
    files = [Path(__file__).relative_to(Path.cwd()), PROTOCOL, VERIFIER,
             Path("run_public_feature_advantage_153.py"), Path("tests/test_public_feature_advantage_153.py")]
    return {"format": "public_feature_advantage_153_v1", "config": CONFIG,
            "parent": parent_inputs(), "dependencies": {p.as_posix(): old.file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__)}


def check_identity():
    saved = old.read_json(OUTPUT/"manifest.json")
    if saved != identity():
        raise ValueError("153 frozen identity changed")
    return old.digest(saved)


def panels():
    train = old.read_json(old.OUTPUT/"train_panel.json")["samples"]
    extra = old.read_json(parent.OUTPUT/"new_panel.json")["samples"]
    validation = old.read_json(old.OUTPUT/"validation_panel.json")["samples"]
    assert (len(train), len(extra), len(validation)) == (768, 2304, 192)
    all_samples = train+extra+validation
    assert len({s["source_seed"] for s in all_samples}) == len(all_samples)
    return {"train": train+extra, "validation": validation}


def fold_map(samples):
    if len({s["source_seed"] for s in samples}) != len(samples):
        raise ValueError("One state per source game required")
    rows = [None]*len(samples)
    for cell in range(6):
        indices = sorted((i for i, s in enumerate(samples) if s["cell"] == cell),
                         key=lambda i: (old.digest([CONFIG["fold_tag"], samples[i]["source_seed"]]), samples[i]["source_seed"]))
        for rank, i in enumerate(indices):
            rows[i] = {"row": i, "source_seed": samples[i]["source_seed"], "cell": cell,
                       "sample_sha256": old.digest(samples[i]), "fold": (rank+2*cell)%5}
    if any(row is None for row in rows):
        raise ValueError("Unknown cell")
    return rows


def public_features(state, action):
    after = deterministic_part(state, action)
    own, other = after._partial.players[after.actor], after._partial.players[1-after.actor]
    phi = [(own.score-other.score)/15]
    phi += [(own.bonuses[c]-other.bonuses[c])/7 for c in TOKEN_COLORS]
    phi += [(own.tokens[c]-other.tokens[c])/7 for c in TOKEN_COLORS]
    phi += [(own.tokens["gold"]-other.tokens["gold"])/5,
            (len(own.purchased_cards)-len(other.purchased_cards))/20,
            (len(own.reserved_cards)+int(after.chance_kind == "private")-len(other.reserved_cards))/3,
            (len(own.nobles)-len(other.nobles))/3]
    return torch.tensor(phi, dtype=torch.float64)


def basis(phi):
    return torch.cat((phi, torch.stack([phi[..., i]*phi[..., j] for i, j in PAIRS], -1)), -1)


def project(samples, source):
    rows, masks = [], []
    for s in samples:
        state = old.deserialize_state_snapshot(s["snapshot"])
        assert state.current_player == s["actor"] and old.phase(state.turn_index)*2+s["actor"] == s["cell"]
        row = [public_features(state, old.decode_action(a)) for a in s["actions"]]
        rows.append(torch.stack(row+[row[0]]*(4-len(row))))
        masks.append([0 < j < len(row) for j in range(4)])
    phi = torch.stack(rows)
    psi = basis(phi)
    mask = torch.tensor(masks)
    assert torch.equal(mask, source["mask"])
    return {"phi": phi, "x": psi-psi[:, :1], "mask": mask,
            "q": source["q"].double(), "y": source["y"].double()}


def losses(pred, y, mask):
    return ((pred-y).square()*mask).sum(-1)/mask.sum(-1).clamp_min(1)


def solve(data, ridge):
    x, y, mask = data["x"], data["y"], data["mask"]
    weights = (mask/mask.sum(-1).clamp_min(1).double()[:, None]/len(y))[mask]
    a, b = x[mask], y[mask]
    gram = a.T@(a*weights[:, None])+ridge*torch.eye(x.shape[-1], dtype=torch.float64)
    rhs = a.T@(b*weights)
    w = torch.linalg.solve(gram, rhs)
    residual = float((gram@w-rhs).abs().max())
    assert torch.isfinite(w).all() and residual <= CONFIG["normal_residual_atol"]
    return w, residual


def subset(data, selection):
    return {k: v[selection] for k, v in data.items()}


def choose_penalty(cv):
    return min(cv, key=lambda row: (row["oof_mse"], -row["penalty"]))["penalty"]


def fit_cv(train, folds):
    fold = torch.tensor([r["fold"] for r in folds])
    cv, fits, oof = [], [], []
    for ridge in CONFIG["penalties"]:
        pred = torch.zeros_like(train["y"])
        details = []
        for f in range(5):
            included, held = fold != f, fold == f
            w, residual = solve(subset(train, included), ridge)
            pred[held] = train["x"][held]@w
            error = float(losses(pred[held], train["y"][held], train["mask"][held]).mean())
            details.append({"fold": f, "train_states": int(included.sum()), "held_states": int(held.sum()), "mse": error})
            fits.append({"penalty": ridge, "fold": f, "coefficients": w, "normal_residual": residual})
        cv.append({"penalty": ridge, "oof_mse": float(losses(pred, train["y"], train["mask"]).mean()), "folds": details})
        oof.append(pred)
    selected = choose_penalty(cv)
    w, residual = solve(train, selected)
    fits.append({"penalty": selected, "fold": "all", "coefficients": w, "normal_residual": residual})
    return {"cv": cv, "selected_penalty": selected, "fits": fits, "oof_predictions": oof}


def prediction(data, coefficients):
    pred = data["x"]@coefficients
    valid = data["mask"].clone()
    valid[:, 0] = True
    choices = pred.masked_fill(~valid, -torch.inf).argmax(-1)
    assert torch.equal(pred[:, 0], torch.zeros(len(pred), dtype=torch.float64))
    return {"advantages": pred.tolist(), "choices": choices.tolist()}


def metric(data, row):
    pred = torch.tensor(row["advantages"], dtype=torch.float64)
    choice = torch.tensor(row["choices"])
    return {"mse": float(losses(pred, data["y"], data["mask"]).mean()),
            "wdl_gain_over_035": float((data["q"][torch.arange(len(choice)), choice]-data["q"][:, 0]).mean()),
            "baseline_fraction": float((choice == 0).double().mean())}


def gates(treatment, control, zero):
    reductions = {"versus_C": 1-treatment["mse"]/control["mse"], "versus_zero": 1-treatment["mse"]/zero}
    checks = {"mse_versus_C": reductions["versus_C"] >= .03, "mse_versus_zero": reductions["versus_zero"] >= .03,
              "wdl_versus_035": treatment["wdl_gain_over_035"] >= .02,
              "wdl_versus_C": treatment["wdl_gain_over_035"] > control["wdl_gain_over_035"]}
    return {"relative_mse_reduction": reductions, "development_gates": checks, "development_passed": all(checks.values())}


def statistics(data, rows):
    treatment = metric(data, rows["public"])
    controls = [{"seed": r["seed"], "selected_update": r["selected_update"], **metric(data, r)} for r in rows["C"]]
    mean = {k: sum(r[k] for r in controls)/3 for k in treatment}
    zero = float(losses(torch.zeros_like(data["y"]), data["y"], data["mask"]).mean())
    contrasts = {str(r["seed"]): {"mse_difference": treatment["mse"]-r["mse"],
                 "wdl_difference": treatment["wdl_gain_over_035"]-r["wdl_gain_over_035"]} for r in controls}
    contrasts["C_mean"] = {"mse_difference": treatment["mse"]-mean["mse"],
                            "wdl_difference": treatment["wdl_gain_over_035"]-mean["wdl_gain_over_035"]}
    return {"public": treatment, "C": {"per_seed": controls, "mean": mean}, "zero_mse": zero,
            "public_minus_control": contrasts, **gates(treatment, mean, zero),
            "development_only": True, "confirmatory": False}


def collisions(data):
    pairs = aliased_pairs = zero_alternates = aliased_states = all_same = alternates = 0
    for phi, mask in zip(data["phi"], data["mask"]):
        rows = [phi[0]]+[phi[j] for j in range(1, 4) if mask[j]]
        unique = len({tuple(v.tolist()) for v in rows})
        all_same += unique == 1
        aliased_states += unique < len(rows)
        for j in range(1, len(rows)):
            alternates += 1
            zero_alternates += torch.equal(rows[j], rows[0])
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                pairs += 1
                aliased_pairs += torch.equal(rows[i], rows[j])
    return {"states": len(data["y"]), "real_alternates": alternates, "real_pairs": pairs,
            "aliased_pairs": aliased_pairs, "states_with_aliases": aliased_states,
            "zero_difference_alternates": zero_alternates, "all_candidates_same_states": all_same,
            "baseline_only_states": int((data["mask"].sum(-1) == 0).sum())}


def audit_stage(stage):
    subprocess.run([sys.executable, str(VERIFIER), stage], check=True)
    audit = old.read_json(OUTPUT/f"{stage}_audit.json")
    assert audit["passed"] and audit["manifest"] == check_identity()
    return audit


def prepare():
    old.save_new(OUTPUT/"manifest.json", identity())
    samples = panels()
    old.save_new(OUTPUT/"fold_map.json", fold_map(samples["train"]))
    old.save_new(OUTPUT/"freeze.json", {"time": old.now(), "manifest": check_identity(),
                 "fold_map_sha256": old.file_hash(OUTPUT/"fold_map.json")})
    for split, name in (("train", "C"), ("validation", "validation")):
        data = project(samples[split], parent.load_data(name))
        old.tensor_save(OUTPUT/f"{split}_data.pt", data)
        print({"stage": "public_features", "split": split, "states": len(data["y"])}, flush=True)
    old.save_new(OUTPUT/"data_lock.json", {"freeze_sha256": old.file_hash(OUTPUT/"freeze.json"),
                 **{s: old.file_hash(OUTPUT/f"{s}_data.pt") for s in ("train", "validation")}})
    audit_stage("preflight")


def load_data(split):
    if split not in ("train", "validation"):
        raise ValueError("153 has no protected performance split")
    path = OUTPUT/f"{split}_data.pt"
    assert old.file_hash(path) == old.read_json(OUTPUT/"data_lock.json")[split]
    return torch.load(path, weights_only=True)


def run():
    binding = check_identity()
    preflight = old.read_json(OUTPUT/"preflight_audit.json")
    assert preflight["passed"] and preflight["manifest"] == binding
    assert preflight["data_lock_sha256"] == old.file_hash(OUTPUT/"data_lock.json")
    freeze = old.read_json(OUTPUT/"freeze.json")
    assert freeze["manifest"] == binding
    assert freeze["fold_map_sha256"] == old.file_hash(OUTPUT/"fold_map.json")
    assert old.read_json(OUTPUT/"data_lock.json")["freeze_sha256"] == old.file_hash(OUTPUT/"freeze.json")
    started = time.perf_counter()
    old.save_new(OUTPUT/"launch.json", {"time": old.now(), "manifest": binding,
                 "preflight_sha256": old.file_hash(OUTPUT/"preflight_audit.json")})
    train = load_data("train")
    fitted = fit_cv(train, old.read_json(OUTPUT/"fold_map.json"))
    old.tensor_save(OUTPUT/"fits.pt", fitted)
    old.save_new(OUTPUT/"selection_lock.json", {"time": old.now(), "manifest": binding,
        "fits_sha256": old.file_hash(OUTPUT/"fits.pt"), "data_lock_sha256": old.file_hash(OUTPUT/"data_lock.json"),
        "fold_map_sha256": old.file_hash(OUTPUT/"fold_map.json"),
        "selected_penalty": fitted["selected_penalty"], "cv": fitted["cv"], "solves": len(fitted["fits"])})
    val = load_data("validation")
    w = fitted["fits"][-1]["coefficients"]
    rows = {"public": prediction(val, w), "C": old.read_json(parent.OUTPUT/"validation_predictions.json")["C"]}
    old.save_new(OUTPUT/"validation_predictions.json", rows)
    training = prediction(train, w)
    old.save_new(OUTPUT/"training_predictions.json", training)
    result = {**statistics(val, rows), "training": metric(train, training),
              "selected_penalty": fitted["selected_penalty"],
              "collisions": {"train": collisions(train), "validation": collisions(val)}}
    old.save_new(OUTPUT/"result.json", result)
    print({"stage": "development_result", **result}, flush=True)
    audit = audit_stage("final")
    for field, name in (("result_sha256", "result.json"), ("selection_sha256", "selection_lock.json"),
                        ("predictions_sha256", "validation_predictions.json"), ("training_predictions_sha256", "training_predictions.json")):
        assert audit[field] == old.file_hash(OUTPUT/name)
    old.save_new(OUTPUT/"final_status.json", {"complete": True, "finished": old.now(),
        "seconds": time.perf_counter()-started, "manifest": binding,
        "development_passed": result["development_passed"], "audit_sha256": old.file_hash(OUTPUT/"final_audit.json"),
        "result_sha256": old.file_hash(OUTPUT/"result.json"), "solves": 26, "independent_solves": 26,
        **{k: CONFIG[k] for k in ("development_only", "new_simulations", "neural_refits", "fresh_performance_test", "arena", "promotion", "automatic_successor")}})


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    with old.process_lock(OUTPUT/"run.lock"):
        try:
            {"prepare": prepare, "run": run}[args.mode]()
        except Exception:
            old.save_new(OUTPUT/f"failure_{args.mode}.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()

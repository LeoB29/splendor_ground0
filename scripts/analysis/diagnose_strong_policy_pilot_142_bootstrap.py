"""Post-stop diagnostic reconstruction; never continue the 142 pipeline.

The frozen trainer raises its gate before persisting epoch metrics. Re-execute
ONLY the identical fixed warmup with its identical data/seed/settings, capture
the failed frame, and save diagnostics. No tuning, new data, parent promotion,
self-play, joint update or arena is permitted here. Frozen sources stay intact.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_strong_policy_pilot as pilot
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, save_model


def main():
    torch.set_num_threads(1)
    manifest = pilot.read_json(pilot.OUTPUT / "manifest.json")
    assert manifest == pilot.identity()
    assert not (pilot.OUTPUT / "parent.pt").exists()
    assert not (pilot.OUTPUT / "selfplay").exists()
    directory = pilot.OUTPUT / "bootstrap_failure_diagnostic"
    directory.mkdir(exist_ok=False)
    captured = {}
    def trace(frame, event, arg):
        if frame.f_code is pilot.train.__code__ and event == "exception":
            exception = arg[1]
            if str(exception) == "Value bootstrap failed retention or held-out improvement gate":
                captured.update(frame.f_locals)
        return trace
    started = time.perf_counter()
    sys.settrace(trace)
    try:
        pilot.train("bootstrap", pilot.CHECKPOINT, directory / "unexpected_success.pt", manifest)
    except RuntimeError as error:
        assert str(error) == "Value bootstrap failed retention or held-out improvement gate"
    else:
        raise RuntimeError("Failed gate did not reproduce; do not continue")
    finally:
        sys.settrace(None)
    assert captured, "Missing failure-frame diagnostics"
    report = {
        "manifest": pilot.digest(manifest), "status": "bootstrap_gate_failure_reproduced",
        "same_data_seed_settings": True, "diagnostic_replay_only": True,
        "source_checkpoint_sha256": pilot.file_hash(pilot.CHECKPOINT),
        "data_summary_sha256": pilot.file_hash(pilot.OUTPUT / "bootstrap/summary.json"),
        "history": captured["history"],
        "before_policy_sha256": captured["before_policy"],
        "after_policy_sha256": captured["after_policy"],
        "policy_retention_passed": captured["before_policy"] == captured["after_policy"],
        "train_game_indices": captured["train_ids"], "validation_game_indices": captured["val_ids"],
        "training_positions": len(captured["training"]), "validation_positions": len(captured["validation"]),
        "trainable_parameters": sum(p.numel() for p in captured["model"].parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in captured["model"].parameters()),
        "diagnostic_wall_seconds": time.perf_counter() - started,
    }
    model = captured["model"]
    with torch.no_grad():
        for name in ("train", "val"):
            predictions = model.value_head(captured[f"{name}_x"]).squeeze(-1)
            targets = captured[f"{name}_y"]
            report[f"{name}_diagnostics"] = {
                "mse": float(((predictions-targets)**2).mean()),
                "zero_predictor_mse": float((targets**2).mean()),
                "target_mean": float(targets.mean()), "prediction_mean": float(predictions.mean()),
                "prediction_std": float(predictions.std()),
                "sign_accuracy": float(((predictions > 0) == (targets > 0)).float().mean()),
            }
    checkpoint = directory / "failed_value_head.pt"
    save_model(checkpoint, model, -142, {**report, "eligible_for_use": False})
    report["failed_checkpoint_sha256"] = pilot.file_hash(checkpoint)
    pilot.atomic_write(directory / "fit_diagnostics.json", report)
    print(report, flush=True)
    # Independently replay the completed corpus and reconstruct every model input.
    audit = runpy.run_path("scripts/analysis/verify_strong_policy_pilot_142.py")["audit_game"]
    summary = pilot.read_json(pilot.OUTPUT / "bootstrap/summary.json")
    positions = 0
    for index in range(256):
        path = pilot.OUTPUT / "bootstrap" / f"game_{index:04d}.json"
        assert pilot.file_hash(path) == summary["files"][path.name]
        record = audit(path, pilot.CONFIG, model.config)
        assert record["index"] == index and record["seed"] == pilot.CONFIG["bootstrap_seed"] + index
        assert record["input_hash"] == pilot.digest(summary["binding"])
        assert record["valid_natural"] and record["termination_reason"] == "completed"
        positions += record["positions"]
        if (index+1) % 32 == 0:
            print(f"verified bootstrap {index+1}/256", flush=True)
    assert positions == 16328
    assert pilot.policy_hash(StrongPolicyInference(pilot.CHECKPOINT).model) == pilot.policy_hash(model)
    assert not set(report["train_game_indices"]) & set(report["validation_game_indices"])
    assert sorted(report["train_game_indices"] + report["validation_game_indices"]) == list(range(256))
    assert pilot.identity() == manifest
    verification = {
        "status": "passed", "manifest": pilot.digest(manifest), "games": 256, "positions": positions,
        "all_trajectories_legal": True, "all_features_groups_and_outcome_labels_reconstructed": True,
        "all_games_naturally_completed": True, "policy_exactly_035": True,
        "fit_diagnostics_sha256": pilot.file_hash(directory / "fit_diagnostics.json"),
        "diagnostic_script_sha256": pilot.file_hash(Path(__file__)),
        "frozen_dependencies_unchanged": len(manifest["dependencies"]),
        "selfplay_joint_training_and_arena_started": False,
    }
    pilot.atomic_write(directory / "verification.json", verification)
    print(verification, flush=True)


if __name__ == "__main__":
    main()

"""Reporting-only recovery after 143 saved all 512 sealed test predictions.

Frozen source, data, models, selection and statistical rules are unchanged.
Reproduce the frozen evaluator, require exact equality to EVERY saved prediction,
and convert NumPy scalar types to their identical Python values for JSON only.
No game generation, fitting, model selection or new comparison is performed.
"""
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from splendor_ai.alphazero import run_value_generalization_scale as exp


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def main():
    torch.set_num_threads(1)
    manifest = exp.read_json(exp.OUTPUT / "manifest.json")
    assert exp.identity() == manifest
    failure = exp.read_json(exp.OUTPUT / "failure.json")
    assert "bool_ is not JSON serializable" in failure["error"]
    assert not (exp.OUTPUT / "result.json").exists()
    assert len(list((exp.OUTPUT / "games").glob("game_*.json"))) == 5120
    assert len(list((exp.OUTPUT / "test_predictions").glob("game_*.pt"))) == 512
    lock = exp.check_selection_lock(manifest)
    archive = exp.OUTPUT / "serialization_failure"
    archive.mkdir(exist_ok=False)
    for name in ("result.json.tmp", "failure.json", "run.log"):
        path = exp.OUTPUT / name
        if path.exists():
            shutil.copy2(path, archive / name)
    original_write, original_tensor_save, original_json = exp.atomic_write, exp.tensor_save, exp.json
    verified = []
    def require_identical_existing_tensor(path, payload):
        prior = torch.load(path, weights_only=True)
        assert prior.keys() == payload.keys()
        for key in prior:
            if isinstance(prior[key], torch.Tensor):
                assert torch.equal(prior[key], payload[key]), (path, key)
            else:
                assert prior[key] == payload[key], (path, key)
        verified.append(str(path))
    try:
        exp.atomic_write = lambda path, payload: original_write(path, json_safe(payload))
        exp.tensor_save = require_identical_existing_tensor
        exp.json = SimpleNamespace(dumps=lambda payload, **kwargs: original_json.dumps(json_safe(payload), **kwargs))
        result = exp.evaluate_test(manifest)
    finally:
        exp.atomic_write, exp.tensor_save, exp.json = original_write, original_tensor_save, original_json
    assert len(verified) == len(set(verified)) == 512
    assert exp.check_selection_lock(manifest) == lock
    assert exp.identity() == manifest
    saved = exp.read_json(exp.OUTPUT / "result.json")
    assert saved == json_safe(result)
    report = {"status": "passed", "scope": "JSON scalar-type conversion only",
              "manifest": exp.digest(manifest), "saved_test_predictions_reproduced_exactly": 512,
              "training_or_new_games": False, "changed_frozen_sources": False,
              "model_selection_unchanged": True, "statistical_rules_unchanged": True,
              "original_failure_sha256": exp.file_hash(archive / "failure.json"),
              "result_sha256": exp.file_hash(exp.OUTPUT / "result.json"),
              "recovery_script_sha256": exp.file_hash(Path(__file__))}
    exp.atomic_write(exp.OUTPUT / "serialization_recovery.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()

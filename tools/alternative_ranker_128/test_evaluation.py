"""Synthetic checks only; these tests never open the running study's outputs."""
from collections import Counter
import json

import numpy as np
import pytest

from . import evaluation as e


def pair(index=0, outsiders=("a", "b"), ridge="b", hgb="a"):
    fit, seat, phase = e.CELLS[index // e.PER_CELL]
    spec = {"index": index, "game_index": 10000 + index, "fit": fit, "seat": seat,
            "phase": phase, "snapshot_sha256": f"snapshot-{index}"}
    if not outsiders:
        ridge = hgb = None
    elif len(outsiders) == 1:
        ridge = hgb = outsiders[0]
    anchor = outsiders[0] if outsiders else None
    prediction = dict(spec, outsider_keys=list(outsiders), anchor_key=anchor,
                      scores={"ridge": [float(k == ridge) for k in outsiders],
                              "hgb": [float(k == hgb) for k in outsiders]},
                      choices={"ridge": ridge, "hgb": hgb})
    first = ["first0", "first1", "first2"]
    union = first + list(outsiders)
    root = {"manifest": "study", "stage": "main", "spec": spec, "union": {"keys": union},
            "candidates": {"anchor": {"keys": first + ([anchor] if anchor else [])},
                           "teacher": {"keys": union, "indices": list(range(len(union)))}},
            "proposals": {"anchor": {"group_keys": union,
                                     "learned_group_scores": [5., 4., 3.] + [2. - j for j in range(len(outsiders))]}},
            "selection": {"worlds": [{"outcomes": [{"wdl": (1 if key == "b" else 0),
                                                    "reason": "completed", "winner": None} for key in union]}
                                      for _ in range(e.WORLDS)]}}
    return prediction, root


def panel_rows():
    return [e.row_from_root(*pair(i)) for i in range(e.STATES)]


def test_fixed_choice_value_and_shared_action_cancel():
    row = e.row_from_root(*pair())
    assert row["delta_numerators"] == {"ridge": [128, 128], "hgb": [0, 0]}
    assert row["anchor_regret_numerators"] == [128, 128]


@pytest.mark.parametrize("outsiders", [(), ("a",)])
def test_zero_singleton_support_retained_as_zero_even_if_unknown(outsiders):
    prediction, root = pair(outsiders=outsiders)
    for world in root["selection"]["worlds"]:
        for outcome in world["outcomes"]:
            outcome.update(wdl=None, reason="repetition_cutoff", winner=None)
    row = e.row_from_root(prediction, root)
    assert row["delta_numerators"] == {"ridge": [0, 0], "hgb": [0, 0]}
    assert row["anchor_regret_numerators"] == [0, 0]


def test_unknown_bounds_not_draw_imputation():
    prediction, root = pair()
    for world in root["selection"]["worlds"]:
        world["outcomes"][3].update(wdl=None, reason="repetition_cutoff", winner=None)
    row = e.row_from_root(prediction, root)
    assert row["delta_numerators"] == {"ridge": [0, 256], "hgb": [0, 0]}
    assert row["unknown_selection_outcomes"] == 128
    assert row["anchor_regret_numerators"] == [0, 256]


@pytest.mark.parametrize("corruption", ["identity", "support", "choice", "nan", "anchor", "worlds", "unknown"])
def test_bad_prediction_or_coverage_rejected(corruption):
    prediction, root = pair()
    if corruption == "identity":
        prediction["snapshot_sha256"] = "other"
    elif corruption == "support":
        prediction["outsider_keys"].reverse()
    elif corruption == "choice":
        prediction["choices"]["ridge"] = "a"
    elif corruption == "nan":
        prediction["scores"]["ridge"][0] = float("nan")
    elif corruption == "anchor":
        prediction["anchor_key"] = "b"
    elif corruption == "worlds":
        root["selection"]["worlds"].pop()
    else:
        root["selection"]["worlds"][0]["outcomes"][0]["wdl"] = None
    with pytest.raises(ValueError):
        e.row_from_root(prediction, root)


def test_equal_logits_choose_first_teacher_outsider():
    prediction, root = pair()
    prediction["scores"]["ridge"] = [0., 0.]
    prediction["choices"]["ridge"] = "a"
    assert e.row_from_root(prediction, root)["delta_numerators"]["ridge"] == [0, 0]


def test_complete_stratified_panel_and_zero_regret_capture():
    rows = panel_rows()
    result = e.evaluate_rows(rows, bootstrap_samples=50)
    ridge = result["contrasts"]["ridge_minus_anchor"]
    assert ridge["mean_score_difference"] == .5
    assert ridge["ci975"] == [.5, .5]
    assert ridge["supplementary_capture_bounds"] == [1., 1.]
    assert result["contrasts"]["hgb_minus_anchor"]["superiority_established"] is False
    for row in rows:
        row["anchor_regret_numerators"] = [0, 0]
    assert e.evaluate_rows(rows, bootstrap_samples=50)["contrasts"]["ridge_minus_anchor"]["supplementary_capture_bounds"] is None


@pytest.mark.parametrize("problem", ["missing", "duplicate_game", "cell", "coupling"])
def test_no_panel_selection_or_broken_pairing(problem):
    rows = panel_rows()
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate_game":
        rows[0]["game_index"] = rows[1]["game_index"]
    elif problem == "cell":
        rows[0]["phase"] = 1
    else:
        rows[0]["delta_numerators"]["hgb"] = [-1, 1]
    with pytest.raises(ValueError):
        e.evaluate_rows(rows, bootstrap_samples=20)


def test_bootstrap_matches_independent_frequency_accumulation():
    rows = panel_rows()
    for row in rows:
        i = row["index"]
        row["delta_numerators"]["ridge"] = [i % 11 - 7, i % 11 - 3]
        row["choices"]["hgb"] = "b"
        row["delta_numerators"]["hgb"] = [4 - i % 13, 4 - i % 13]
    count, seed = 73, 1234
    actual = e.evaluate_rows(rows, bootstrap_samples=count, bootstrap_seed=seed, bootstrap_chunk=19)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, count, 19):
        size = min(19, count-start)
        accum = np.zeros((size, 4), dtype=np.int64)
        for cell in e.CELLS:
            members = [row for row in rows if (row["fit"], row["seat"], row["phase"]) == cell]
            selections = rng.integers(0, 40, size=(size, 40))
            for draw_index, selected in enumerate(selections):
                frequencies = Counter(int(k) for k in selected)
                for k, frequency in frequencies.items():
                    v = members[k]["delta_numerators"]
                    accum[draw_index] += frequency * np.array(v["ridge"] + v["hgb"])
        values.extend(accum.tolist())
    values = np.asarray(values) / (720 * 256)
    for j, name in enumerate(e.MODELS):
        def quantile(column, probability):
            ordered = sorted(column)
            at = (len(ordered)-1)*probability
            low, high = int(np.floor(at)), int(np.ceil(at))
            return ordered[low]+(ordered[high]-ordered[low])*(at-low)
        expected = [quantile(values[:, 2*j], .0125), quantile(values[:, 2*j+1], .9875)]
        assert actual["contrasts"][name+"_minus_anchor"]["ci975"] == pytest.approx(expected)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"path": path.name, "sha256": e._sha(path)}


def prediction_files(tmp_path):
    models = save(tmp_path / "models.json", {})
    selection = save(tmp_path / "selection.json", {})
    doc = {"schema": "alternative_ranker_predictions_v1", "manifest": "new", "study_binding": "study",
           "models_lock": models, "selection": selection,
           "rows": [pair(i)[0] for i in range(720)]}
    reference = save(tmp_path / "predictions.json", doc)
    seal = {"manifest": "new", "study_binding": "study", "predictions": reference,
            "models_lock": models, "selection": selection, "predictions_locked_before_outcome_read": True}
    save(tmp_path / "prediction_seal.json", seal)
    return tmp_path / "predictions.json", tmp_path / "prediction_seal.json"


def test_seal_checks_before_any_outcome_access(tmp_path, monkeypatch):
    predictions, seal = prediction_files(tmp_path)
    predictions.write_text(predictions.read_text()+" ")
    def prohibited(*args, **kwargs):
        pytest.fail("Outcome access occurred before a valid prediction seal")
    monkeypatch.setattr(e, "_authenticated_panel", prohibited)
    with pytest.raises(ValueError, match="hash mismatch"):
        e.evaluate_sealed(predictions, seal, repo_root=tmp_path, study_root=tmp_path / "study",
                          manifest="new", study_binding="study", bootstrap_samples=10)


def test_incomplete_study_does_not_open_any_root(tmp_path, monkeypatch):
    predictions, seal = prediction_files(tmp_path)
    save(tmp_path / "study" / "final_status.json", {"manifest": "study", "complete": False})
    with pytest.raises(ValueError, match="must be complete"):
        e.evaluate_sealed(predictions, seal, repo_root=tmp_path, study_root=tmp_path / "study",
                          manifest="new", study_binding="study", bootstrap_samples=10)


def test_acquisition_stop_is_not_a_zero_or_missing_panel(tmp_path):
    predictions, seal = prediction_files(tmp_path)
    save(tmp_path / "study" / "final_status.json", {"manifest": "study", "complete": True,
         "all_prescribed_audits_pass": True, "classification": "acquisition_not_established"})
    with pytest.raises(ValueError, match="No complete fresh panel"):
        e.evaluate_sealed(predictions, seal, repo_root=tmp_path, study_root=tmp_path / "study",
                          manifest="new", study_binding="study", bootstrap_samples=10)


def test_negative_interval_reports_deterioration():
    rows = panel_rows()
    for row in rows:
        row["delta_numerators"]["ridge"] = [-12, -12]
    result = e.evaluate_rows(rows, bootstrap_samples=20)
    assert result["contrasts"]["ridge_minus_anchor"]["classification"] == "restricted_ranking_deterioration"


def test_teacher_support_is_outcome_blind_and_uses_fixed_tie_order():
    from .prediction import teacher_support
    keys = [str(i) for i in range(12)]
    # Reference fourth is random and is replaced by the highest remaining anchor score.
    result = teacher_support(keys, [0.] * 12, ["0", "1", "2", "11"], 7,
                             lambda payload: str(20-int(payload[-1])).zfill(2))
    assert result == {"first_three": ["0", "1", "2"],
                      "teacher_keys": ["0", "1", "2", "3", "4", "5", "11", "10"],
                      "outsider_keys": ["3", "4", "5", "11", "10"], "anchor_key": "3"}
    for size in (1, 2, 3, 4, 5, 7):
        small = teacher_support(keys[:size], [0.] * size, keys[:min(4, size)], 7, lambda payload: "0")
        assert small["teacher_keys"] == keys[:size]
        assert len(small["outsider_keys"]) == max(0, size-3)


def test_source_snapshot_reads_only_authenticated_source_files(tmp_path, monkeypatch):
    from .prediction import source_snapshot
    source_spec = {"index": 42, "stage": "main", "fit": 161001, "seat": 1,
                   "phase": 0, "search_stream": "fixed"}
    snapshot = {"source_only": 10}
    game = {"manifest": "study", "spec": source_spec,
            "trace": [{"turn": 10, "snapshot": snapshot}], "winner": "not_used"}
    game_ref = save(tmp_path / "source_game.json", game)
    audit_ref = save(tmp_path / "source_audit.json", {"manifest": "study", "passed": True,
                    "spec": source_spec, "game": game_ref})
    spec = dict(source_spec, index=0, game_index=42, pair=42, source_stage="main", turn=10,
                snapshot_sha256="fixed_digest", source_game=game_ref, source_audit=audit_ref)
    original = e._read
    opened = []
    def guarded(path):
        opened.append(Path(path).name)
        assert Path(path).name in {"source_game.json", "source_audit.json"}
        return original(path)
    from pathlib import Path
    monkeypatch.setattr(e, "_read", guarded)
    assert source_snapshot(spec, repo_root=tmp_path, study_binding="study",
                           digest=lambda value: "fixed_digest") == snapshot
    assert opened == ["source_game.json", "source_audit.json"]


def test_prediction_lifecycle_seals_before_outcomes_with_mocked_inference(tmp_path, monkeypatch):
    from . import common as c, learning, prediction
    from .features import FEATURE_CONFIG
    base, study = tmp_path / "outputs" / "alternative_ranker_128", tmp_path / "outputs" / "structured"
    base.mkdir(parents=True)
    study.mkdir(parents=True)
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OUTPUT", base)
    monkeypatch.setattr(c, "STUDY_ROOT", study)
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return c.artifact(path)
    dataset = put(base / "dataset.json", {"train_only": True})
    model_refs = {name: put(base / f"{name}.pkl", {"fixture": name}) for name in e.MODELS}
    report = {"config": learning.CONFIG, "feature_config": FEATURE_CONFIG,
              "fresh_outcomes_read": False, "validation_read": False, "new_simulations": 0,
              "hyperparameter_selection": False, "dataset": dataset,
              "models": {name: {"model": ref, "identical_second_fit": True, "selected_fit": "first",
                                "parameter_sha256": "a"*64, "all_train_prediction_sha256": "b"*64}
                         for name, ref in model_refs.items()}}
    fitted = put(base / "models_report.json", report)
    put(base / "models_lock.json", {"manifest": "new", "models": model_refs,
                                    "learning": fitted, "dataset": dataset})
    anchor_refs = {f"anchor_{fit}": dict(put(study / f"anchor_{fit}.json", {"fake": fit}), source_fit=fit)
                   for fit in e.FITS}
    original_data = put(study / "data_lock.json", {"parent": {"models": anchor_refs}})
    registration = put(base / "manifest.json", {"study_binding": "study", "source": {"data": original_data}})
    monkeypatch.setattr(c, "validate_freeze", lambda: {"binding": "new", "manifest": registration})
    monkeypatch.setattr(c, "configure_compute", lambda: None)
    monkeypatch.setattr(c, "resource_gate", lambda: {})
    monkeypatch.setattr(prediction, "_resource_boundary", lambda *args: None)
    # These intentionally are not JSON: prediction may check their existence, not parse outcomes.
    (study / "final_status.json").write_text("MUST NOT READ OUTCOME STATUS")
    (study / "main_complete.json").write_text("MUST NOT READ OUTCOME SUMMARY")
    specs = [pair(i)[1]["spec"] for i in range(720)]
    selection = put(study / "selection.json", {"manifest": "study", "main": specs})
    put(study / "selection_seal.json", {"manifest": "study", "selection": selection})
    put(study / "models_lock.json", {"manifest": "study", "models": anchor_refs})
    calls = []
    def mock_load(ref, *, expected_recipe, expected_dataset):
        assert ref == model_refs[expected_recipe] and expected_dataset == dataset
        calls.append(expected_recipe)
        return {"recipe": expected_recipe}
    monkeypatch.setattr(learning, "load_model", mock_load)
    monkeypatch.setattr(prediction, "_load_anchors", lambda *args: {fit: object() for fit in e.FITS})
    monkeypatch.setattr(prediction, "_initialize_reference", lambda: None)
    monkeypatch.setattr(prediction, "source_snapshot", lambda *args, **kwargs: {"snapshot": True})
    monkeypatch.setattr(prediction, "prediction_row", lambda spec, *args: pair(spec["index"])[0])
    made = prediction.create_predictions(repo_root=tmp_path, base=base, study_root=study, manifest="new")
    assert calls == ["ridge", "hgb"] and made["states"] == 720
    assert made["teacher_outcomes_accessed"] is False
    doc = e.validate_prediction_seal(base / "predictions.json", base / "predictions_seal.json",
                                    repo_root=tmp_path, manifest="new", study_binding="study")
    assert len(doc["rows"]) == 720
    with pytest.raises(ValueError, match="One-shot"):
        prediction.create_predictions(repo_root=tmp_path, base=base, study_root=study, manifest="new")

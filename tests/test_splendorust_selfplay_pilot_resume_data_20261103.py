"""Saved-only continuation routing; no Torch import, model, fit or new game."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_selfplay_pilot_resume_20261103 import data as d, training as t

WORKSPACE = Path(__file__).resolve().parents[1]
OLD = d.CONTINUATION_ORIGINAL_OUTPUT
NEW = "outputs/splendorust_selfplay_pilot_resume_20261103_01"


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(t, "ROOT", tmp_path)
    specs = json.loads((WORKSPACE / OLD / "game_schedule.json").read_text())
    schedule = write(tmp_path, OLD + "/game_schedule.json", specs)
    binding = {"source": "current_frozen_job", "plan": write(tmp_path, OLD + "/qualification_plan.json", {"loader_cases": []}),
               "row_paths": [OLD + f"/capture_fixtures/capture_on/{i:03d}.json" for i in range(16)],
               "proof_path": OLD + "/capture_qualification.json"}
    wrapper = {"training_plan": t.pilot_plan(), "source_specs": specs,
               "learned_entrypoint": {}, "input_qualification": binding}
    fit_plan = write(tmp_path, OLD + "/fit_plan.json", wrapper)
    original = {"scope": "pilot", "phase": "pilot", "teacher_simulations": 512, "new_source_games": 256,
                **{key: d.IDENTITY[key] for key in ("ruleset_id", "catalogue_sha256", "turn_contract")},
                "historical_parent_checkpoint": {"sha256": t.PARENT_SHA256}, "game_schedule": schedule, "fit_plan": fit_plan}
    oldfreeze = write(tmp_path, OLD + "/freeze.json", original)
    monkeypatch.setattr(d, "CONTINUATION_ORIGINAL_FREEZE_SHA256", oldfreeze["sha256"])
    games = [{"path": (OLD if i < 32 else NEW) + f"/games/g{i:03d}/game.json", "bytes": 2,
              "sha256": hashlib.sha256(b"{}").hexdigest()} for i in range(256)]
    manifest = d.build_manifest(games, specs)
    declaration = {"schema": "selfplay_saved_source_prefix_v1", "completed_games": 32,
                   "original_freeze": oldfreeze, "original_schedule": schedule, "games": deepcopy(games[:32])}
    return SimpleNamespace(root=tmp_path, specs=specs, original=original, wrapper=wrapper,
                           declaration=declaration, manifest=manifest, output=tmp_path / NEW, binding=binding)


def bind(f):
    return d.validate_source_continuation(f.manifest, f.specs, f.declaration, current_output=f.output)


def rehash_manifest(f):
    f.manifest = d.build_manifest(f.manifest["games"], f.specs)


def test_exact_32_original_plus_224_new_is_provenance_not_acceptance(fixture):
    proof = bind(fixture)
    assert (proof["preserved_source_games"], proof["new_source_games"], proof["total_source_games"]) == (32, 224, 256)
    assert proof["all_original_loader_checks_required"]
    assert not proof["original_attempt_accepted"] and not proof["reused_checkpoint"]
    assert proof["qualification_games_reused"] == proof["game_replacements"] == 0
    # This guard binds descriptors only. The unchanged loader authenticates bytes when opening each split.
    assert not (fixture.root / fixture.manifest["games"][0]["path"]).exists()


@pytest.mark.parametrize("change", ["count", "bool_count", "extra", "missing", "schema", "freeze_hash", "freeze_path",
                                    "schedule", "prefix_hash", "prefix_copy", "prefix_swap", "suffix_old", "suffix_traversal",
                                    "suffix_hash", "suffix_size", "duplicate", "spec_seed", "spec_split"])
def test_rejects_continuation_substitutions(fixture, change):
    f = fixture
    if change == "count": f.declaration["completed_games"] = 31
    elif change == "bool_count": f.declaration["completed_games"] = True
    elif change == "extra": f.declaration["accepted"] = True
    elif change == "missing": del f.declaration["games"]
    elif change == "schema": f.declaration["schema"] = "historical_relabel"
    elif change == "freeze_hash": f.declaration["original_freeze"]["sha256"] = "0" * 64
    elif change == "freeze_path": f.declaration["original_freeze"]["path"] = NEW + "/freeze.json"
    elif change == "schedule": f.declaration["original_schedule"]["sha256"] = "1" * 64
    elif change == "prefix_hash": f.manifest["games"][0]["sha256"] = "1" * 64
    elif change == "prefix_copy": f.manifest["games"][0]["path"] = NEW + "/games/g000/game.json"
    elif change == "prefix_swap": f.manifest["games"][:2] = list(reversed(f.manifest["games"][:2]))
    elif change == "suffix_old": f.manifest["games"][32]["path"] = OLD + "/games/g032/game.json"
    elif change == "suffix_traversal": f.manifest["games"][32]["path"] = NEW + "/../escape.json"
    elif change == "suffix_hash": f.manifest["games"][32]["sha256"] = "G" * 64
    elif change == "suffix_size": f.manifest["games"][32]["bytes"] = True
    elif change == "duplicate": f.manifest["games"][32] = deepcopy(f.manifest["games"][31])
    elif change == "spec_seed": f.specs[32]["seed_decimal"] = "1234567"
    elif change == "spec_split": f.specs[31]["split"], f.specs[200]["split"] = f.specs[200]["split"], f.specs[31]["split"]
    rehash_manifest(f)
    with pytest.raises((ValueError, KeyError)): bind(f)


@pytest.mark.parametrize("field,value", [("scope", "qualification"), ("teacher_simulations", 256),
                                          ("new_source_games", 255), ("ruleset_id", "historical")])
def test_original_scientific_contract_cannot_change(fixture, monkeypatch, field, value):
    f = fixture
    f.original[field] = value
    f.declaration["original_freeze"] = write(f.root, OLD + "/freeze.json", f.original)
    monkeypatch.setattr(d, "CONTINUATION_ORIGINAL_FREEZE_SHA256", f.declaration["original_freeze"]["sha256"])
    with pytest.raises(ValueError): bind(f)


def test_changed_game_bytes_are_rejected_by_unchanged_artifact_reader(fixture):
    item = fixture.manifest["games"][0]
    path = fixture.root / item["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="hash differs"): d.read_artifact(item)


def test_only_original_frozen_fixture_binding_is_allowed(fixture):
    proof = d.validate_preserved_input_qualification(fixture.binding, fixture.declaration)
    assert proof["training_examples_added"] == proof["source_games_relabelled"] == 0
    altered = deepcopy(fixture.binding)
    altered["row_paths"][0] = NEW + "/capture_fixtures/capture_on/000.json"
    with pytest.raises(ValueError, match="only exact original"):
        d.validate_preserved_input_qualification(altered, fixture.declaration)


def cli_fixture(f):
    fit = write(f.root, NEW + "/fit_plan.json", f.wrapper)
    parity = write(f.root, NEW + "/fit_parity_fixtures.json", {"fixtures": []})
    write(f.root, NEW + "/corpus_manifest.json", f.manifest)
    freeze = write(f.root, NEW + "/freeze.json", {"fit_plan": fit, "fit_parity_fixtures": parity,
                                                  "phase": "pilot", "source_continuation": f.declaration})
    common = SimpleNamespace(OUT=f.output, SCOPE="pilot", verify=lambda items: [d.read_artifact(x) for x in items])
    args = SimpleNamespace(mode="pilot", out=f.output / "fit", plan=f.root / fit["path"],
                           fixtures=f.root / parity["path"], manifest=f.output / "corpus_manifest.json")
    env = {"STUDY_RESEARCH_FREEZE": str(f.output / "freeze.json"), "STUDY_RESEARCH_FREEZE_SHA256": freeze["sha256"]}
    return args, common, env


def test_real_trainer_cli_admits_only_bound_prefix_without_opening_models(fixture):
    args, common, env = cli_fixture(fixture)
    folder, wrapper, manifest, fixtures = t.admit_fit_cli(args, common=common, environment=env)
    assert folder == fixture.output / "fit" and wrapper == fixture.wrapper and manifest == fixture.manifest and fixtures == []


@pytest.mark.parametrize("change", ["prefix", "binding", "missing_declaration"])
def test_real_trainer_cli_rejects_bad_preserved_routes(fixture, change):
    f = fixture
    if change == "prefix":
        f.manifest["games"][0]["sha256"] = "0" * 64
        rehash_manifest(f)
    elif change == "binding": f.wrapper["input_qualification"]["row_paths"][0] = NEW + "/invented.json"
    args, common, env = cli_fixture(f)
    if change == "missing_declaration":
        freeze_path = f.output / "freeze.json"
        value = json.loads(freeze_path.read_text()); del value["source_continuation"]
        env["STUDY_RESEARCH_FREEZE_SHA256"] = write(f.root, NEW + "/freeze.json", value)["sha256"]
    with pytest.raises((ValueError, KeyError)): t.admit_fit_cli(args, common=common, environment=env)


def test_actual_saved_original_row_decodes_with_same_corrected_targets_and_provenance():
    path = WORKSPACE / OLD / "games/g000/moves/000000.json"
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    row = json.loads(path.read_bytes())
    decoded = d.decode_row(row, expected_spec=row["spec"], expected_ply=0)
    assert decoded["eligible"] and decoded["root_input_sha256"] == row["training_capture"]["root_input_sha256"]
    assert row["worker_folder"].startswith(OLD + "/workers/")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash


def test_actual_preserved_sixteen_loader_cases_remain_required(monkeypatch):
    original_freeze_path = WORKSPACE / OLD / "freeze.json"
    raw = original_freeze_path.read_bytes()
    old = json.loads(raw)
    binding = d.read_artifact(old["fit_plan"])["input_qualification"]
    declaration = {"schema": "selfplay_saved_source_prefix_v1", "completed_games": 32,
                   "original_freeze": {"path": OLD + "/freeze.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                   "original_schedule": old["game_schedule"], "games": []}
    saved_read = t._read
    monkeypatch.setattr(t, "_read", lambda path: {"source_continuation": declaration}
                        if Path(path) == WORKSPACE / NEW / "freeze.json" else saved_read(path))
    result = t.qualify_saved_tensor_inputs(WORKSPACE / NEW / "fit", binding, materialize=False)
    assert result["status"] == "actual_fitter_tensor_counterexamples_passed"
    assert len(result["sources"]) == 16 and result["frozen_input_binding"] == binding
    assert result["precollection_proof"]["path"] == OLD + "/capture_qualification.json"
    assert len(result["six_corrected_faces"]) == 6


@pytest.mark.parametrize("filename,allowed", [("data.py", {"validate_source_continuation", "validate_preserved_input_qualification"}),
                                               ("training.py", {"admit_fit_cli", "qualify_saved_tensor_inputs"}),
                                               ("corpus.py", set())])
def test_all_original_scientific_functions_are_identical(filename, allowed):
    base = (WORKSPACE / "tools/splendorust_selfplay_pilot_budget_20261102" / filename).read_text()
    new = (WORKSPACE / "tools/splendorust_selfplay_pilot_resume_20261103" / filename).read_text()
    base = base.replace("splendorust_selfplay_pilot_budget_20261102", "splendorust_selfplay_pilot_resume_20261103")
    nodes = lambda text: {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(text).body
                          if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
    old_nodes, new_nodes = nodes(base), nodes(new)
    assert set(new_nodes) - set(old_nodes) <= allowed
    assert set(old_nodes) <= set(new_nodes)
    for name in old_nodes:
        if name not in allowed: assert old_nodes[name] == new_nodes[name], name


def test_training_recipe_and_both_modes_remain_exact():
    from tools.splendorust_selfplay_pilot_budget_20261102 import training as old
    assert t.RECIPE == old.RECIPE and t.pilot_plan() == old.pilot_plan() and t.qualification_plan() == old.qualification_plan()
    assert t.PARENT_SHA256 == old.PARENT_SHA256 and t.PARAMETERS == old.PARAMETERS

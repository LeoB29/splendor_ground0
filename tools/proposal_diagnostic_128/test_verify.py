"""Independent proposal arithmetic, frozen replay panels and mutation checks."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from tools.proposal_diagnostic_128 import verify as v
from tools.proposal_diagnostic_128 import engine
from tools.proposal_diagnostic_128.test_engine import private_state


def synthetic_specs(count=201):
    return [{"index": i, "pair": i, "fit": v.FITS[i//67] if count == 201 else v.FITS[i % 3],
             "role": ("treatment", "anchor")[(i//3) % 2], "phase": i % 3,
             "privacy": {"own": i % 2, "opponent": (i//2) % 2}, "canonical_support": 1+i % 5}
            for i in range(count)]


def test_sample_plan_is_bounded_deterministic_and_covers_all_available_cells():
    specs = synthetic_specs()
    plan = v.sample_plan(specs, "main", "binding")
    assert plan == v.sample_plan(copy.deepcopy(specs), "main", "binding")
    stress = {row["index"] for row in plan["selections"] if row["stress"]}
    assert 0 < len(stress) <= 39
    assert not plan["coverage"]["missing_cells"]
    assert all(row["selected_index"] in stress for row in plan["stress_cells"])
    for row in plan["selections"]:
        for block in v.BLOCKS:
            assert len(row["ordinary"][block]) == 8
            assert len(row["replay"][block]) == (128 if row["stress"] else 8)
    changed = copy.deepcopy(specs)
    for spec in changed:
        spec["irrelevant_outcome"] = -999
    other = v.sample_plan(changed, "main", "other binding")
    assert other["selections"] == plan["selections"]
    assert other["stress_cells"] == plan["stress_cells"]
    probe = v.sample_plan(synthetic_specs(12), "probe", "binding")
    assert all(row["stress"] for row in probe["selections"])
    assert probe["coverage"]["replayed_worlds"] == 12*256


def test_independent_world_namespace_exact_and_global_disjoint():
    seen = set()
    for stage, count in (("probe", 12), ("main", 201), ("test", 12)):
        for index in range(count):
            for block in v.BLOCKS:
                seeds = v._world_seeds(stage, index, block)
                assert seeds == engine.world_seeds(stage, index, block)
                assert not (set(seeds) & seen)
                seen.update(seeds)


def test_separate_arithmetic_preserves_arm_ties_and_does_not_select_on_assessment():
    keys = list("abcde")
    candidates = {arm: {"keys": keys[:3]+(["e"] if arm == "treatment" else ["d"])} for arm in v.PROPOSERS}
    candidates["first_three"] = {"keys": keys[:3]}
    candidates["union"] = {"keys": keys}
    # d is preferred on selection but e on assessment; selection ties keep a.
    selection = [{"outcomes": [{"wdl": x} for x in [0, 0, -1, 1, 0]]} for _ in range(128)]
    assessment = [{"outcomes": [{"wdl": x} for x in [0, -1, -1, -1, 1]]} for _ in range(128)]
    proposed = {arm: 4 if arm == "treatment" else 3 for arm in v.PROPOSERS}
    summary = v._summary(selection, assessment, candidates, keys, proposed)
    assert summary["choices"]["treatment"] == 0
    assert summary["choices"]["union"] == 3
    assert summary["arm_scores"]["treatment"] == .5
    assert summary["fixed_proposal_scores"]["treatment"] == 1.
    assert summary["arm_scores"]["union"] == 0.


def statistic_rows():
    rows = []
    for i, spec in enumerate(synthetic_specs()):
        numerators = {"anchor": 128, "treatment": 128+i % 11-5, "replay": 126,
                      "random": 128+i % 7-3, "first_three": 128, "union": 132}
        rows.append({**{key: spec[key] for key in ("index", "pair", "fit", "role", "phase")},
                     "arm_scores": {key: value/256 for key, value in numerators.items()}})
    return rows


def test_statistics_match_separate_direct_resampling():
    rows = statistic_rows()
    result = v.statistics(rows)
    matrices = [np.array([[r["arm_scores"][a]-r["arm_scores"][b] for _, a, b in v.CONTRASTS]
                         for r in rows if r["fit"] == fit]) for fit in v.FITS]
    rng = np.random.default_rng(2083000000)
    samples = []
    for _ in range(50):
        total = np.zeros((1000, 5))
        for matrix in matrices:
            total += matrix[rng.integers(0, 67, size=(1000, 67))].sum(axis=1)
        samples.append(total/201)
    boot = np.concatenate(samples)
    for j, (name, _, _) in enumerate(v.CONTRASTS):
        assert result["contrasts"][name]["mean_score_difference"] == pytest.approx(sum(m[:, j].sum() for m in matrices)/201, abs=1e-12)
        assert result["contrasts"][name]["ci95"] == pytest.approx(np.quantile(boot[:, j], [.025, .975]), abs=1e-12)
    assert result["classification"] == "conditional_direction_inconclusive"
    bad = copy.deepcopy(rows)
    bad[1]["pair"] = bad[0]["pair"]
    with pytest.raises(AssertionError):
        v.statistics(bad)
    bad = copy.deepcopy(rows)
    bad[1]["arm_scores"]["anchor"] = .51
    with pytest.raises(AssertionError):
        v.statistics(bad)


@pytest.fixture(scope="module")
def fixture_artifacts(tmp_path_factory):
    engine.initialize_models()
    base = tmp_path_factory.mktemp("independent-diagnostic")
    state = private_state()
    snapshot = json.loads(json.dumps(v.old.serialize_state_snapshot(state)))
    legal = v.old.SplendorEnv().legal_actions(state)
    common = {"fit": 161001, "role": "treatment", "phase": 0, "turn": state.turn_index,
              "privacy": engine.baseline.privacy(state),
              "canonical_support": len({repr(v.old.canonical_successor_action_key(state, a)) for a in legal}),
              "source_stage": "test", "search_stream": "independent-diagnostic-fixture",
              "snapshot_sha256": v.old.digest(snapshot), "fixture_snapshot": snapshot}
    specs = [{**common, "index": i, "pair": i, "game_index": 2*i} for i in range(2)]
    sample = v.sample_plan(specs, "test", "fixture")
    v.old.save_new(base/"test"/"audit_sample.json", sample)
    items = [engine.collect("test", spec, "fixture", base) for spec in specs]
    return base, specs, sample, items


def test_real_private_fixture_independent_full_and_sampled_replays(fixture_artifacts):
    base, specs, plan, items = fixture_artifacts
    assert {row["stress"] for row in plan["selections"]} == {True, False}
    for spec, selection, item in zip(specs, plan["selections"], items, strict=True):
        receipt = v.c.verified(v.audit("test", spec, item, "fixture", base))
        assert receipt["passed"]
        assert receipt["recorded_worlds"] == 256
        assert receipt["replayed_worlds"] == (256 if selection["stress"] else 16)
        assert receipt["row"] == engine.statistics_row(v.c.verified(item))
        assert receipt["timing"]["ordinary_replay_wall"] > 0
        assert receipt["timing"]["structural_wall"] > 0


def clone_fixture(fixture_artifacts, tmp_path):
    source, specs, plan, items = fixture_artifacts
    index = next(row["index"] for row in plan["selections"] if not row["stress"])
    spec = specs[index]
    root = copy.deepcopy(v.c.verified(items[index]))
    v.old.save_new(tmp_path/"test"/"audit_sample.json", plan)
    seal_path = tmp_path/"test"/"selections"/f"state_{index:04d}_selection.json"
    seal = copy.deepcopy(v.c.verified(root["selection_artifact"]))
    # Clone the already generated evidence, preserving logical seal order.
    seal["selection_lock"]["created_at_ns"] = time.time_ns()
    root["selection_lock"] = seal["selection_lock"]
    v.old.save_new(seal_path, seal)
    root["selection_artifact"] = v.c.artifact(seal_path)
    root["assessment_started_at_ns"] = time.time_ns()
    return spec, root


@pytest.mark.parametrize("damage", ["model", "summary", "schedule", "world", "outcome", "sampled_trace", "seal_time", "sample_plan"])
def test_independent_audit_rejects_rehashed_corruptions(fixture_artifacts, tmp_path, damage):
    spec, root = clone_fixture(fixture_artifacts, tmp_path)
    block, index = "assessment", spec["index"]
    wi = v.ordinary_indices("test", index, block)[0]
    if damage == "model":
        root["proposals"]["treatment"]["model_id"] = "replay_161001"
    elif damage == "summary":
        root["summary"]["arm_scores"]["treatment"] += 1/256
    elif damage == "schedule":
        root[block]["world_seeds"][0] += 1
    elif damage == "world":
        root[block]["worlds"][0]["state_sha256"] = "0"*64
    elif damage == "outcome":
        unselected = next(i for i in range(128) if i not in v.ordinary_indices("test", index, block))
        root[block]["worlds"][unselected]["outcomes"][0]["wdl"] = .5
    elif damage == "sampled_trace":
        root[block]["worlds"][wi]["outcomes"][0]["trace_sha256"] = "0"*64
    elif damage == "seal_time":
        root["assessment_started_at_ns"] = root["selection_lock"]["created_at_ns"]-1
    else:
        path = tmp_path/"test"/"audit_sample.json"
        value = v.old.read_json(path)
        value["selections"][index]["replay"][block] = []
        path.write_text(json.dumps(value), encoding="utf-8")
    path = tmp_path/"test"/"roots"/f"state_{index:04d}.json"
    v.old.save_new(path, root)
    with pytest.raises((AssertionError, ValueError)):
        v.audit("test", spec, v.c.artifact(path), "fixture", tmp_path)


def test_synthetic_source_cannot_enter_production():
    with pytest.raises(AssertionError):
        v._source_state("test", {"fixture_snapshot": {}}, v.OUTPUT)
    with pytest.raises(AssertionError):
        v._source_state("main", {"fixture_snapshot": {}}, Path("anywhere"))

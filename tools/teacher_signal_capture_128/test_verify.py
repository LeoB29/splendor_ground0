"""Checks of independent arithmetic, pair sampling, model/world laws and seals."""
import copy
import itertools
import pytest
from . import verify as v, engine as e, statistics as stats
from .test_engine import worlds, proposals, candidates, outcome, fixture_registry, real_fixture_root, fixture_spec


def specs(stage="main"):
    result, pair = [], 0
    for split in (("train", "validation") if stage == "main" else ("train",)):
        for fit in v.FITS:
            for rank in range(14 if stage == "main" else 3):
                phases = [p for p in range(3) if p != rank % 3] if stage == "main" else [rank]
                for leg in (0, 1):
                    for phase in phases:
                        result.append({"index": len(result), "pair": pair, "game_index": pair*2+leg,
                            "fit": fit, "split": split, "seat": leg, "phase": phase,
                            "canonical_support": 1+rank % 8, "privacy": {"own": rank % 2, "opponent": (rank//2) % 2}})
                pair += 1
    return result


def test_sampling_covers_physics_privacy_short_support_and_all_probe():
    rows = specs()
    plan = v.sample_plan(rows, "main", "fixture")
    assert 18 <= plan["coverage"]["full_stress_roots"] <= 39 and not plan["coverage"]["missing_cells"]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["irrelevant_result"] = -999
    assert v.sample_plan(changed, "main", "changed")["selections"] == plan["selections"]
    assert all(len(row["ordinary"]["assessment"]) == 8 for row in plan["selections"])
    assert all(len(row["replay"]["assessment"]) == (128 if row["stress"] else 8) for row in plan["selections"])
    probe = v.sample_plan(specs("probe"), "probe", "fixture")
    assert probe["coverage"]["full_inner_replay"] and probe["coverage"]["replayed_worlds"] == 18*128


@pytest.mark.parametrize("damage", ["duplicate", "split", "pair", "fit", "phase", "privacy", "support"])
def test_pair_or_population_tamper_fails(damage):
    rows = specs()
    if damage == "duplicate": rows[1] = dict(rows[0], index=1)
    elif damage == "split": rows[0]["split"] = "validation"
    elif damage == "pair": rows[0]["pair"] += 1
    elif damage == "fit": rows[0]["fit"] = 161003
    elif damage == "phase": rows[0]["phase"] = rows[1]["phase"]
    elif damage == "privacy": rows[0]["privacy"]["own"] = -1
    else: rows[0]["canonical_support"] = 0
    with pytest.raises(AssertionError): v.sample_plan(rows, "main", "fixture")


@pytest.mark.parametrize("support", range(1, 9))
def test_original_ties_short_support_and_summaries_match_independently(support):
    source, prediction = worlds([0]*support), proposals(support)
    selected = v.independent_selection(source, prediction, support)
    assert selected == e.rebuild_choices(source, prediction, candidates(support))
    assert selected["choices"]["teacher"] == 0
    assert selected["choices"]["teacher_outsider"] == (3 if support > 3 else None)
    fresh = worlds([1]+[0]*(support-1))
    assert v.independent_summary(fresh, selected["choices"]) == e.summarize(fresh, selected["choices"])


def test_pairwise_certificates_agree_with_all_unknown_completions():
    for offset in range(-2, 3):
        saved = worlds([0, 0, 0, -1])
        for wi in range(abs(offset)):
            saved[wi]["outcomes"][0] = outcome(1 if offset > 0 else -1)
        saved[7]["outcomes"][0] = outcome(None)
        saved[9]["outcomes"][1] = outcome(None)
        result = v.independent_selection(saved, proposals(4), 4)
        possible = {max(range(4), key=lambda j: [offset+a, b, 0, -128][j]) for a, b in itertools.product((-1, 0, 1), repeat=2)}
        assert result["choices"]["teacher"] == (next(iter(possible)) if len(possible) == 1 else None)


def test_new_outcomes_cannot_reselect_or_impute_locked_choices():
    original = worlds([0, 0, 0, 1, -1])
    selected = v.independent_selection(original, proposals(5, [0, 0, 0, 0, 10]), 5)
    fresh = worlds([1, 0, 0, 0, 0])
    fresh[0]["outcomes"][3] = outcome(None)
    summary = v.independent_summary(fresh, selected["choices"])
    assert summary == e.summarize(fresh, selected["choices"])
    assert summary["choices"]["teacher"] == 3 and summary["fresh_selection"]["teacher_certificate"]["choice"] == 0
    assert summary["arm_scores"]["teacher"] is None
    assert summary["delta_bounds"]["teacher_minus_anchor"] == [0., 0.]
    assert summary["delta_bounds"]["teacher_outsider_minus_broad_outsider"] == [-1/256, 1/256]
    assert "fresh_teacher" not in summary["arm_scores"]


def test_unknown_replay_added_once_and_no_imputation_allowed():
    saved = worlds([1, -1, 0])
    saved[1]["outcomes"][0] = outcome(None)
    saved[99]["outcomes"][2] = outcome(None)
    cells = v.replay_cells(saved, [1, 3, 5, 7, 9, 11, 13, 15])
    assert len(cells["all"]) == 25 and cells["extra"] == {(99, 2)}
    saved[99]["outcomes"][2]["wdl"] = 0
    with pytest.raises(AssertionError): v._bounded_columns(saved, 3)


def test_world_namespaces_are_distinct_and_independently_equal():
    seen = set()
    for stage, count in (("main", 336), ("probe", 18), ("test", 18)):
        for index in range(count):
            seeds = v._world_seeds(stage, index)
            assert seeds == e.world_seeds(stage, index) and seen.isdisjoint(seeds)
            seen.update(seeds)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    e.initialize_models(tmp_path_factory.mktemp("signal-independent-models"), fixture_registry())


@pytest.mark.parametrize("fit", v.FITS)
def test_full_legal_inference_is_independent_and_private(models, fit, monkeypatch):
    original = real_fixture_root()
    state = e.old.deserialize_state_snapshot(original["snapshot"])
    expected = e.frozen_proposals(state, original["candidates"], fit)
    monkeypatch.setattr(e, "frozen_proposals", lambda *a, **k: pytest.fail("Auditor called collector"))
    alternate = v.input_audit.hidden_alternative(state)
    for family in v.PROPOSERS:
        actual = v.independent_proposal(state, original["candidates"]["keys"], f"{family}_{fit}")
        v.compare(actual, {k: value for k, value in expected[family].items() if k != "timing"})
        v.compare(actual, v.independent_proposal(alternate, original["candidates"]["keys"], f"{family}_{fit}"))


def test_real_fixture_complete_independent_audit_and_evidence_tamper(models, tmp_path, monkeypatch):
    spec = fixture_spec()
    state = e.old.deserialize_state_snapshot(spec["fixture_root"]["snapshot"])
    spec.update(canonical_support=len(e.old.actions_by_canonical_successor_key(state, e.old.SplendorEnv().legal_actions(state))),
                privacy=spec["fixture_root"]["privacy"])
    registry = fixture_registry()
    registry["manifest"] = "fixture"
    e.old.save_new(tmp_path/"models_lock.json", registry)
    e.old.save_new(tmp_path/"test"/"audit_sample.json", v.sample_plan([spec], "test", "fixture"))
    item = e.collect("test", spec, "fixture", tmp_path)
    expected_row = e.statistics_row(e.c.verified(item))
    monkeypatch.setattr(e, "frozen_proposals", lambda *a, **k: pytest.fail("Called collector inference"))
    monkeypatch.setattr(e, "rebuild_choices", lambda *a, **k: pytest.fail("Called collector selection"))
    monkeypatch.setattr(e, "summarize", lambda *a, **k: pytest.fail("Called collector summary"))
    checked = v.audit("test", spec, item, "fixture", tmp_path)
    receipt = e.c.verified(checked)
    v.compare(receipt["row"], expected_row)
    assert receipt["passed"] and receipt["recorded_continuations"] == 1024
    assert receipt["replayed_continuations"] == 1024  # Single fixture occupies a stress cell.
    assert v.summarize([checked], [spec], "fixture", tmp_path)["full_inner_replay"]
    with pytest.raises(AssertionError, match="Preserve"):
        v.audit("test", spec, item, "fixture", tmp_path)
    changed = dict(item, sha256="0"*64)
    with pytest.raises(ValueError): e.c.verify_hash(changed)


def population_rows():
    summaries = []
    prediction = proposals(5, [0, 0, 0, 0, 10])
    source = worlds([0, 0, 0, 0, 1])
    choices = v.independent_selection(source, prediction, 5)["choices"]
    for variant in range(7):
        fresh = worlds([0, 0, 0, 0, 0])
        for wi in range(variant): fresh[wi]["outcomes"][4] = outcome(1)
        if variant == 6: fresh[127]["outcomes"][4] = outcome(None)
        summaries.append(v.independent_summary(fresh, choices))
    return [v._row(spec, summaries[(spec["pair"]+spec["phase"]+(2 if spec["split"] == "validation" else 0)) % 7]) for spec in specs()]


def test_independent_pair_bootstrap_matches_all_contrasts_and_unknown_envelopes():
    rows = population_rows()
    result = stats.summarize(rows)
    v.compare(result, v.independent_statistics(rows))
    assert result["pairs"] == 84 and result["resampling_unit"] == "whole_source_pair_four_roots"
    assert result["groups"]["train"]["contrasts"]["teacher_minus_anchor"]["mean"] is None
    assert result["classification"] == "diagnostic_only" and result["promotion"] is False
    assert not result["train_validation_difference_is_causal"]
    assert result["validation_previously_used_for_checkpoint_selection"]


@pytest.mark.parametrize("damage", ["missing", "split", "phase", "pair", "scale", "choice", "score"])
def test_stats_rejects_broken_units_or_arithmetic(damage):
    rows = population_rows()
    if damage == "missing": rows.pop()
    elif damage == "split": rows[0]["split"] = "validation"
    elif damage == "phase": rows[0]["phase"] = rows[1]["phase"]
    elif damage == "pair": rows[0]["pair"] = rows[4]["pair"]
    elif damage == "scale": rows[0]["delta_bounds"]["teacher_minus_anchor"] = [.001, .001]
    elif damage == "choice": rows[0]["choices"]["teacher"] = 99
    else: rows[0]["arm_scores"]["teacher"] = -1
    with pytest.raises((AssertionError, ValueError)): v.independent_statistics(rows)
    if damage not in ("choice", "score"):
        with pytest.raises((AssertionError, ValueError)): stats.summarize(rows)

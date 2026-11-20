"""Independent conditional proposal/value checks and bounded sampled replay.

Shares the mature engine, observer encoders and neural arithmetic. Collection
proposal helpers, world constructors, aggregate/choice helpers are not used.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
import time

import numpy as np

from tools.proposal_diagnostic_128 import common as c
from tools.proposal_iteration_128 import arena as parent_arena
from tools.proposal_iteration_128 import verify as parent_verify
from scripts.analysis import verify_rollout_selector_154 as world_audit
from scripts.analysis import verify_afterstate_advantage_149 as continuation_audit
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit

old, OUTPUT, FITS = c.old, c.OUTPUT, c.FITS
compare = parent_verify.compare
ARMS = ("anchor", "treatment", "replay", "random", "first_three", "union")
PROPOSERS = ("anchor", "treatment", "replay", "random")
BLOCKS = ("selection", "assessment")
CONTRASTS = (
    ("treatment_minus_anchor", "treatment", "anchor"),
    ("treatment_minus_replay", "treatment", "replay"),
    ("random_minus_anchor", "random", "anchor"),
    ("first_three_minus_anchor", "first_three", "anchor"),
    ("union_minus_anchor", "union", "anchor"),
)
SAMPLE_CONFIG = {
    "seed": 2084000000,
    "world_tag": "proposal-diagnostic-128-audit-world-v1",
    "stress_tag": "proposal-diagnostic-128-audit-stress-v1",
    "worlds_per_block": 128,
    "ordinary_worlds_per_block": 8,
    "maximum_main_stress_roots": 39,
    "probe_full_replay": True,
}


def ordinary_indices(stage, index, block):
    assert stage in ("main", "probe", "test") and block in BLOCKS
    assert type(index) is int and index >= 0
    return sorted(sorted(range(128), key=lambda wi: (
        old.digest([SAMPLE_CONFIG["world_tag"], 2084000000, stage, index, block, wi]), wi))[:8])


def _privacy_category(spec):
    value = spec["privacy"]
    assert set(value) == {"own", "opponent"}
    assert all(type(n) is int and n >= 0 for n in value.values())
    return int(bool(value["own"])) + 2*int(bool(value["opponent"]))


def sample_plan(specs, stage, binding):
    """Select whole blocks using only frozen source identities/lawful metadata."""
    assert stage in ("main", "probe", "test") and specs
    assert [s["index"] for s in specs] == list(range(len(specs)))
    if stage != "test":
        assert len(specs) == {"main": 201, "probe": 12}[stage]
    for spec in specs:
        assert spec["fit"] in FITS and spec["role"] in ("treatment", "anchor")
        assert spec["phase"] in (0, 1, 2) and spec["canonical_support"] >= 1
        _privacy_category(spec)
    def priority(spec):
        return old.digest([SAMPLE_CONFIG["stress_tag"], 2084000000, stage, spec["index"]]), spec["index"]
    cells, selected = [], set()
    def add_cell(name, eligible):
        choice = min(eligible, key=priority)["index"] if eligible else None
        cells.append({"cell": name, "eligible_count": len(eligible), "selected_index": choice})
        if choice is not None:
            selected.add(choice)
    for fit in FITS:
        group = [s for s in specs if s["fit"] == fit]
        for role in ("treatment", "anchor"):
            for phase in range(3):
                add_cell(f"fit_role_phase|{fit}|{role}|{phase}",
                         [s for s in group if s["role"] == role and s["phase"] == phase])
        for privacy in range(4):
            add_cell(f"fit_privacy|{fit}|{privacy}", [s for s in group if _privacy_category(s) == privacy])
        for support in (1, 2, 3):
            add_cell(f"fit_short_support|{fit}|{support}", [s for s in group if s["canonical_support"] == support])
    assert len(cells) == 39 and len(selected) <= 39
    if stage == "probe":
        selected = {s["index"] for s in specs}
    selections = []
    for spec in specs:
        ordinary = {block: ordinary_indices(stage, spec["index"], block) for block in BLOCKS}
        stress = spec["index"] in selected
        selections.append({"index": spec["index"], "stress": stress, "ordinary": ordinary,
                           "replay": {block: list(range(128)) if stress else ordinary[block] for block in BLOCKS}})
    return {"manifest": binding, "stage": stage, "config": SAMPLE_CONFIG,
            "specs_sha256": old.digest(specs), "selections": selections, "stress_cells": cells,
            "coverage": {"states": len(specs), "full_stress_roots": len(selected), "stress_cells": 39,
                         "missing_cells": [row["cell"] for row in cells if row["selected_index"] is None],
                         "recorded_worlds": len(specs)*256,
                         "replayed_worlds": sum(len(row["replay"][block]) for row in selections for block in BLOCKS),
                         "full_inner_replay": stage == "probe"}}


def _checked_sample(stage, spec, binding, base):
    path = Path(base) / stage / "audit_sample.json"
    sample = old.read_json(path)
    assert sample["manifest"] == binding and sample["stage"] == stage and sample["config"] == SAMPLE_CONFIG
    assert sample["selections"][spec["index"]]["index"] == spec["index"]
    row = sample["selections"][spec["index"]]
    assert type(row["stress"]) is bool
    for block in BLOCKS:
        assert row["ordinary"][block] == ordinary_indices(stage, spec["index"], block)
        assert row["replay"][block] == (list(range(128)) if row["stress"] else row["ordinary"][block])
    return row, c.artifact(path)


def _source_state(stage, spec, base):
    if stage == "test":
        resolved = Path(base).resolve()
        assert not resolved.is_relative_to(OUTPUT.resolve())
        return old.deserialize_state_snapshot(spec["fixture_snapshot"])
    assert "fixture_snapshot" not in spec
    from tools.proposal_diagnostic_128 import data
    snapshot = data.replay_selected_state(spec)
    return old.deserialize_state_snapshot(snapshot)


def _summary(selection_worlds, assessment_worlds, candidates, union_keys, proposed_indices):
    """Independent arithmetic, preserving each arm's offered-order tie rule."""
    count = len(union_keys)
    assert len(selection_worlds) == len(assessment_worlds) == 128
    select_totals = [sum(world["outcomes"][j]["wdl"] for world in selection_worlds) for j in range(count)]
    assess_totals = [sum(world["outcomes"][j]["wdl"] for world in assessment_worlds) for j in range(count)]
    choices = {}
    for arm in ARMS:
        indices = [union_keys.index(key) for key in candidates[arm]["keys"]]
        assert indices and len(set(indices)) == len(indices)
        choices[arm] = max(indices, key=lambda j: select_totals[j])
    return {"selection_means": [v/128 for v in select_totals],
            "assessment_means_wdl": [v/128 for v in assess_totals],
            "arm_wdl": {arm: assess_totals[index]/128 for arm, index in choices.items()},
            "arm_scores": {arm: (128+assess_totals[index])/256 for arm, index in choices.items()},
            "fixed_proposal_wdl": {arm: None if index is None else assess_totals[index]/128
                                   for arm, index in proposed_indices.items()},
            "fixed_proposal_scores": {arm: None if index is None else (128+assess_totals[index])/256
                                      for arm, index in proposed_indices.items()},
            "choices": choices, "proposed_indices": proposed_indices}


def _row(spec, summary):
    return {key: spec[key] for key in ("index", "pair", "fit", "role", "phase")} | {
        key: summary[key] for key in ("arm_scores", "fixed_proposal_scores", "choices", "proposed_indices")}


def _world_seeds(stage, index, block):
    assert stage in ("probe", "main", "test") and block in BLOCKS
    assert type(index) is int and 0 <= index < (201 if stage == "main" else 12)
    first = 2080000000 + {"probe": 0, "main": 1000000, "test": 2000000}[stage] + 512*index
    if block == "assessment":
        first += 256
    return list(range(first, first+128))


def _proposals(state, spec, stage):
    """Separate legal enumeration and candidate assembly, with mature model math."""
    lock = old.read_json(c.SOURCE / "models_lock.json")
    assert lock["manifest"] == c.SOURCE_BINDING
    assert parent_arena._MODEL_ARTIFACTS == lock["models"]
    legal = old.SplendorEnv().legal_actions(state)
    group_keys, representatives = [], {}
    for action in legal:
        key = repr(old.canonical_successor_action_key(state, action))
        if key not in representatives:
            group_keys.append(key)
            representatives[key] = old.serialize_action(action)
    assert len(group_keys) == spec["canonical_support"]
    altered = input_audit.hidden_alternative(state)
    input_audit.input_invariance(state, old._RAW._model.config)
    proposals = {}
    for arm in PROPOSERS[:3]:
        model_id = f"{arm}_{spec['fit']}"
        c.verify_hash(lock["models"][model_id])
        input_audit.input_invariance(state, parent_arena._MODELS[model_id].config)
        expected = parent_verify.independent_proposal(state, spec["search_stream"], model_id)
        compare(expected, parent_verify.independent_proposal(altered, spec["search_stream"], model_id))
        proposals[arm] = expected
    reference = proposals["anchor"]["reference"]
    assert all(value["reference"] == reference for value in proposals.values())
    assert all(value["group_keys"] == group_keys for value in proposals.values())
    for key, action in zip(reference["keys"], reference["actions"], strict=True):
        representatives[key] = action
    first_three = reference["keys"][:3]
    keys_by_arm = {arm: list(proposals[arm]["candidates"]["keys"]) for arm in PROPOSERS[:3]}
    remaining = [key for key in group_keys if key not in first_three]
    random_fourth = sorted(remaining, key=lambda key: (
        old.digest(["proposal-diagnostic-128-random-fourth-v1", stage, spec["index"], key]), key))[:1]
    keys_by_arm["random"] = first_three + random_fourth
    union_keys = list(first_three)
    for arm in PROPOSERS:
        assert keys_by_arm[arm][:3] == first_three
        for key in keys_by_arm[arm]:
            if key not in union_keys:
                union_keys.append(key)
    assert 1 <= len(union_keys) <= 7
    keys_by_arm["first_three"], keys_by_arm["union"] = first_three, list(union_keys)
    candidates = {arm: {"keys": keys, "actions": [representatives[k] for k in keys],
                        "indices": [union_keys.index(k) for k in keys]} for arm, keys in keys_by_arm.items()}
    proposed_indices = {arm: union_keys.index(keys_by_arm[arm][3]) if len(keys_by_arm[arm]) == 4 else None
                        for arm in PROPOSERS}
    return proposals, candidates, {"keys": union_keys, "actions": [representatives[k] for k in union_keys]}, proposed_indices


def audit(stage, spec, item, binding, base=OUTPUT):
    """Check all evidence and independently replay the prospectively sealed panel."""
    start, cpu = time.perf_counter(), time.process_time()
    base = Path(base)
    index = spec["index"]
    _world_seeds(stage, index, "selection")
    assert spec["fit"] in FITS and spec["role"] in ("treatment", "anchor")
    path = base/stage/"roots"/f"state_{index:04d}.json"
    receipt_path = base/stage/"audits"/f"state_{index:04d}.json"
    assert not receipt_path.exists(), "Preserve existing audit evidence"
    assert c.verify_hash(item).resolve() == path.resolve()
    root = c.verified(item)
    assert root["manifest"] == binding and root["stage"] == stage and root["spec"] == spec
    assert "failure" not in root
    sample, sample_artifact = _checked_sample(stage, spec, binding, base)
    state = _source_state(stage, spec, base)
    snapshot = old.serialize_state_snapshot(state)
    assert old.digest(snapshot) == spec["snapshot_sha256"] == old.digest(root["snapshot"])
    assert not state.terminal and state.turn_index == spec["turn"]
    assert spec["phase"] == min(state.turn_index//20, 2)
    own, opponent = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(opponent.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(opponent.reserved_cards)))}
    assert privacy == spec["privacy"] == root["privacy"]
    expected, candidates, union, proposed = _proposals(state, spec, stage)
    assert set(root["proposals"]) == set(PROPOSERS[:3])
    for arm, proposal in expected.items():
        compare({k: v for k, v in root["proposals"][arm].items() if k != "timing"}, proposal)
        timing = root["proposals"][arm]["timing"]
        assert set(timing) == {"wall", "learned_extra_wall"}
        assert all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in timing.values())
        assert timing["learned_extra_wall"] <= timing["wall"]
    assert root["candidates"] == candidates and root["union"] == union and root["proposed_indices"] == proposed
    # The persisted snapshot has JSON string keys; its digest was compared with
    # the independently replayed state above before including this wire form.
    identity = {"spec": spec, "snapshot": root["snapshot"], "privacy": privacy, "proposals": expected,
                "candidates": candidates, "union": union, "proposed_indices": proposed}
    lock = root["selection_lock"]
    assert lock["manifest"] == binding and lock["stage"] == stage and lock["index"] == index
    assert lock["identity_digest"] == old.digest(identity)
    assert lock["selection_sha256"] == old.digest(root["selection"])
    assert lock["value_scale"] == "terminal_wdl"
    seal_path = base/stage/"selections"/f"state_{index:04d}_selection.json"
    assert c.verify_hash(root["selection_artifact"]).resolve() == seal_path.resolve()
    seal = c.verified(root["selection_artifact"])
    compare(seal, {"manifest": binding, "stage": stage, "index": index, "identity": identity,
                   "selection": root["selection"], "selection_lock": lock})
    created, assessment_start = lock["created_at_ns"], root["assessment_started_at_ns"]
    assert type(created) is type(assessment_start) is int and 0 < created <= assessment_start
    assert seal_path.stat().st_mtime_ns <= assessment_start
    assert c.verify_hash(sample_artifact).stat().st_mtime_ns <= created
    assert path.stat().st_mtime_ns >= assessment_start
    altered = input_audit.hidden_alternative(state)
    actions = [old.decode_action(value) for value in union["actions"]]
    ordinary_wall, extra_wall, replayed_worlds = 0., 0., 0
    moves, reasons = 0, Counter()
    for block in BLOCKS:
        data = root[block]
        seeds = _world_seeds(stage, index, block)
        assert set(data) == {"world_seeds", "worlds", "timing"}
        assert data["world_seeds"] == seeds and len(data["worlds"]) == 128
        parent_verify._timing(data["timing"])
        for wi, (seed, saved) in enumerate(zip(seeds, data["worlds"], strict=True)):
            c.budget_boundary(base)
            assert set(saved) == {"seed", "state_sha256", "outcomes"}
            assert saved["seed"] == seed and len(saved["outcomes"]) == len(actions)
            world = world_audit.independent_world(state, seed)
            assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world))
            assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world_audit.independent_world(altered, seed)))
            assert world.players[state.current_player].reserved_cards == own.reserved_cards
            legal = old.SplendorEnv().legal_actions(world)
            replay_this = wi in sample["replay"][block]
            replayed_worlds += int(replay_this)
            for action, outcome in zip(actions, saved["outcomes"], strict=True):
                assert action in legal
                parent_verify._check_outcome(outcome, state.current_player, state.turn_index)
                if replay_this:
                    replay_start = time.perf_counter()
                    rebuilt = continuation_audit.replay_branch(world, action, state.current_player)
                    compare(outcome, rebuilt)
                    elapsed = time.perf_counter()-replay_start
                    if wi in sample["ordinary"][block]:
                        ordinary_wall += elapsed
                    else:
                        extra_wall += elapsed
                    moves += rebuilt["turns"]
                    reasons[rebuilt["reason"]] += 1
    summary = _summary(root["selection"]["worlds"], root["assessment"]["worlds"], candidates, union["keys"], proposed)
    compare(lock["means"], summary.pop("selection_means"))
    assert lock["choices"] == summary["choices"]
    compare(root["summary"], summary)
    parent_verify._timing(root["timing"])
    replay_wall, wall = ordinary_wall+extra_wall, time.perf_counter()-start
    result = {"manifest": binding, "stage": stage, "spec": spec, "passed": True,
              "root": c.artifact(path), "selection_artifact": c.artifact(seal_path), "sample_artifact": sample_artifact,
              "row": _row(spec, summary), "privacy": privacy, "support": len(actions),
              "stress": sample["stress"], "ordinary_world_indices": sample["ordinary"],
              "replay_world_indices": sample["replay"], "recorded_worlds": 256,
              "recorded_continuations": 256*len(actions), "replayed_worlds": replayed_worlds,
              "replayed_continuations": replayed_worlds*len(actions), "moves": moves, "reasons": dict(reasons),
              "full_inner_replay": replayed_worlds == 256,
              "timing": {"wall": wall, "cpu": time.process_time()-cpu, "replay_wall": replay_wall,
                         "structural_wall": max(0., wall-replay_wall), "ordinary_replay_wall": ordinary_wall,
                         "full_extra_replay_wall": extra_wall}}
    assert sum(reasons.values()) == result["replayed_continuations"]
    old.save_new(receipt_path, result)
    return c.artifact(receipt_path, index=index)


def statistics(rows):
    """Receipt-derived bootstrap using histogram counts, not indexed resampling."""
    assert len(rows) == 201 and [r["index"] for r in rows] == list(range(201))
    assert len({r["pair"] for r in rows}) == 201
    strata = []
    for fit in FITS:
        subset = [row for row in rows if row["fit"] == fit]
        assert len(subset) == 67
        values = []
        for row in subset:
            assert set(row["arm_scores"]) == set(ARMS)
            numerator = {}
            for arm, value in row["arm_scores"].items():
                assert type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                num = int(round(256*value))
                assert value == num/256
                numerator[arm] = num
            values.append([numerator[left]-numerator[right] for _, left, right in CONTRASTS])
        strata.append(np.asarray(values, dtype=np.int64))
    totals = sum((matrix.sum(axis=0) for matrix in strata), np.zeros(len(CONTRASTS), dtype=np.int64))
    rng = np.random.default_rng(2083000000)
    bootstrap = np.zeros((50000, len(CONTRASTS)), dtype=np.int64)
    for start in range(0, 50000, 1000):
        for matrix in strata:
            draws = rng.integers(0, 67, size=(1000, 67))
            counts = np.stack([np.bincount(indices, minlength=67) for indices in draws])
            bootstrap[start:start+1000] += counts @ matrix
    contrasts = {}
    for j, (name, _, _) in enumerate(CONTRASTS):
        ordered = sorted(int(v) for v in bootstrap[:, j])
        def quantile(q):
            point = (len(ordered)-1)*q
            lo, hi = math.floor(point), math.ceil(point)
            return (ordered[lo]+(ordered[hi]-ordered[lo])*(point-lo))/(201*256)
        estimate = int(totals[j])/(201*256)
        interval = [quantile(.025), quantile(.975)]
        contrasts[name] = {"mean_score_difference": estimate, "score_percentage_points": 100*estimate,
                           "ci95": interval, "ci95_percentage_points": [100*x for x in interval]}
    primary = contrasts["treatment_minus_anchor"]
    label = ("conditional_direction_positive" if primary["ci95"][0] > 0 else
             "conditional_direction_negative" if primary["ci95"][1] < 0 else "conditional_direction_inconclusive")
    return {"states": 201, "source_pairs": 201, "fits": list(FITS), "bootstrap_samples": 50000,
            "bootstrap_seed": 2083000000, "primary": "treatment_minus_anchor", "contrasts": contrasts,
            "classification": label, "scope": "Conditional expected terminal scores under greedy035 continuations; not whole-game win rates."}


def summarize(items, specs, binding, base=OUTPUT):
    c.budget_boundary(base)
    assert len(items) == len(specs) == 201
    assert [spec["index"] for spec in specs] == list(range(201))
    rows, coverage = [], Counter()
    for item, spec in zip(items, specs):
        c.budget_boundary(base)
        receipt = c.verified(item)
        assert receipt["manifest"] == binding and receipt["passed"] and receipt["spec"] == spec
        c.verify_hash(receipt["root"])
        c.verify_hash(receipt["selection_artifact"])
        c.verify_hash(receipt["sample_artifact"])
        assert receipt["row"]["index"] == spec["index"]
        rows.append(receipt["row"])
        coverage["states"] += 1
        coverage["full_stress_roots"] += int(receipt["stress"])
        for key in ("recorded_continuations", "replayed_continuations", "recorded_worlds", "replayed_worlds"):
            coverage[key] += receipt[key]
    c.budget_boundary(base)
    result = statistics(rows)
    c.budget_boundary(base)
    fixed = {}
    for arm in PROPOSERS:
        values = [row["fixed_proposal_scores"][arm] for row in rows if row["fixed_proposal_scores"][arm] is not None]
        fixed[arm] = {"eligible_states": len(values), "mean_score": sum(values)/len(values) if values else None}
    changes = {}
    for arm in ("treatment", "replay", "random"):
        eligible = [row for row in rows if row["proposed_indices"][arm] is not None]
        changes[arm] = {
            "eligible_states": len(eligible),
            "fourth_changed_vs_anchor": sum(row["proposed_indices"][arm] != row["proposed_indices"]["anchor"] for row in eligible),
            "choice_changed_vs_anchor": sum(row["choices"][arm] != row["choices"]["anchor"] for row in rows),
            "fixed_fourth_minus_anchor_mean": (sum(row["fixed_proposal_scores"][arm]-row["fixed_proposal_scores"]["anchor"]
                                                      for row in eligible)/len(eligible) if eligible else None),
        }
    fourth_choices = {}
    for arm in PROPOSERS:
        eligible = [row for row in rows if row["proposed_indices"][arm] is not None]
        count = sum(row["choices"][arm] == row["proposed_indices"][arm] for row in eligible)
        fourth_choices[arm] = {"eligible_states": len(eligible), "fourth_chosen": count,
                               "fourth_chosen_fraction": count/len(eligible) if eligible else None}
    strata = {}
    for field in ("fit", "role", "phase"):
        strata[field] = {}
        for value in sorted({row[field] for row in rows}):
            subset = [row for row in rows if row[field] == value]
            strata[field][str(value)] = {"states": len(subset), "mean_score_difference": {
                name: sum(row["arm_scores"][left]-row["arm_scores"][right] for row in subset)/len(subset)
                for name, left, right in CONTRASTS}}
    return {"manifest": binding, "statistics": result, "rows": rows, "coverage": dict(coverage),
            "fixed_proposal_scores_descriptive": fixed, "changes_descriptive": changes,
            "fourth_choice_descriptive": fourth_choices, "strata_descriptive": strata,
            "full_inner_replay": False}

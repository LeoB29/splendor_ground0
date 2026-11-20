"""Independent specialist checks, sparse assessment and sampled replay.

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
import torch

from tools.proposal_specialist_128 import common as c
from tools.proposal_specialist_128 import engine
from tools.proposal_iteration_128 import verify as parent_verify
from scripts.analysis import verify_rollout_selector_154 as world_audit
from scripts.analysis import verify_afterstate_advantage_149 as continuation_audit
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector

old, OUTPUT, FITS = c.old, c.OUTPUT, c.FITS
compare = parent_verify.compare
ARMS = ("anchor", "specialist", "control")
PROPOSERS = ARMS
BLOCKS = ("selection", "assessment")
CONTRASTS = (
    ("specialist_minus_anchor", "specialist", "anchor"),
    ("specialist_minus_control", "specialist", "control"),
    ("control_minus_anchor", "control", "anchor"),
)
CELLS = tuple((fit, seat, phase) for fit in FITS for seat in (0, 1) for phase in range(3))
SAMPLE_CONFIG = {
    "seed": 2134000000,
    "world_tag": "proposal-specialist-128-audit-world-v1",
    "stress_tag": "proposal-specialist-128-audit-stress-v1",
    "worlds_per_block": 128,
    "ordinary_worlds_per_block": 8,
    "maximum_main_stress_roots": 39,
    "probe_full_replay": True,
}


def ordinary_indices(stage, index, block):
    assert stage in ("main", "probe", "test") and block in BLOCKS
    assert type(index) is int and index >= 0
    return sorted(sorted(range(128), key=lambda wi: (
        old.digest([SAMPLE_CONFIG["world_tag"], 2134000000, stage, index, block, wi]), wi))[:8])


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
        assert len(specs) == {"main": 360, "probe": 18}[stage]
    for spec in specs:
        assert spec["fit"] in FITS and spec["seat"] in (0, 1)
        assert spec["phase"] in (0, 1, 2) and spec["canonical_support"] >= 1
        _privacy_category(spec)
    assert len({spec["game_index"] for spec in specs}) == len(specs)
    assert all(spec["pair"] == spec["game_index"] for spec in specs)
    if stage != "test":
        assert Counter((s["fit"], s["seat"], s["phase"]) for s in specs) == {
            cell: 20 if stage == "main" else 1 for cell in CELLS}
    def priority(spec):
        return old.digest([SAMPLE_CONFIG["stress_tag"], 2134000000, stage, spec["index"]]), spec["index"]
    cells, selected = [], set()
    def add_cell(name, eligible):
        choice = min(eligible, key=priority)["index"] if eligible else None
        cells.append({"cell": name, "eligible_count": len(eligible), "selected_index": choice})
        if choice is not None:
            selected.add(choice)
    for fit in FITS:
        group = [s for s in specs if s["fit"] == fit]
        for seat in (0, 1):
            for phase in range(3):
                add_cell(f"fit_seat_phase|{fit}|{seat}|{phase}",
                         [s for s in group if s["seat"] == seat and s["phase"] == phase])
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
        if "fixture_snapshot" in spec:
            return old.deserialize_state_snapshot(spec["fixture_snapshot"])
    else:
        assert "fixture_snapshot" not in spec
    from tools.proposal_specialist_128 import source
    snapshot = source.replay_selected_state(spec, base)
    return old.deserialize_state_snapshot(snapshot)


def _summary(selection_worlds, assessment_worlds, candidates, union_keys, proposed_indices, assessment_indices):
    """Independent arithmetic, preserving each arm's offered-order tie rule."""
    count = len(union_keys)
    assert len(selection_worlds) == len(assessment_worlds) == 128
    select_totals = [sum(world["outcomes"][j]["wdl"] for world in selection_worlds) for j in range(count)]
    choices = {}
    for arm in ARMS:
        indices = [union_keys.index(key) for key in candidates[arm]["keys"]]
        assert indices and len(set(indices)) == len(indices)
        choices[arm] = max(indices, key=lambda j: select_totals[j])
    assert assessment_indices == sorted(set(choices.values()))
    assert all(len(world["outcomes"]) == len(assessment_indices) for world in assessment_worlds)
    assess_totals = {index: sum(world["outcomes"][column]["wdl"] for world in assessment_worlds)
                     for column, index in enumerate(assessment_indices)}
    return {"selection_means": [v/128 for v in select_totals],
            "assessment_indices": assessment_indices,
            "assessment_means_wdl": [assess_totals[index]/128 for index in assessment_indices],
            "arm_wdl": {arm: assess_totals[index]/128 for arm, index in choices.items()},
            "arm_scores": {arm: (128+assess_totals[index])/256 for arm, index in choices.items()},
            "choices": choices, "proposed_indices": proposed_indices}


def _row(spec, summary):
    return {key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "privacy")} | {
        key: summary[key] for key in ("arm_scores", "choices", "proposed_indices")}


def _world_seeds(stage, index, block):
    assert stage in ("probe", "main", "test") and block in BLOCKS
    assert type(index) is int and 0 <= index < (360 if stage == "main" else 18)
    first = 2130000000 + {"probe": 0, "main": 1000000, "test": 2000000}[stage] + 512*index
    if block == "assessment":
        first += 256
    return list(range(first, first+128))


def independent_proposal(state, stream, model_id):
    """Rebuild raw support, canonical grouping, learned ranks and tie rules."""
    legal = old.SplendorEnv().legal_actions(state)
    group_keys, members, representatives = [], {}, {}
    for j, action in enumerate(legal):
        key = repr(old.canonical_successor_action_key(state, action))
        if key not in members:
            group_keys.append(key)
            members[key] = []
            representatives[key] = old.serialize_action(action)
        members[key].append(j)
    raw_scores = list(map(float, old._RAW.score_actions(state, legal)))
    raw_order, raw_reps = [], {}
    for j in sorted(range(len(legal)), key=lambda j: (-raw_scores[j], j)):
        key = repr(old.canonical_successor_action_key(state, legal[j]))
        if key not in raw_reps:
            raw_order.append(key)
            raw_reps[key] = old.serialize_action(legal[j])
    first = raw_order[:2]
    representatives.update({key: raw_reps[key] for key in first})
    schedule = parent_verify.independent_schedule(state, stream)
    remaining = [key for key in group_keys if key not in first]
    rng = random.Random(f"149-candidates-v1|155-live-v1|{schedule['candidate_nonce']}")
    reference_keys = first + rng.sample(remaining, min(2, len(remaining)))
    reference = {"keys": reference_keys, "actions": [representatives[k] for k in reference_keys],
                 "canonical_support": len(group_keys), "legal_support": len(legal)}
    model = engine._MODELS[model_id]
    obs = encode_public_observation_tensor(state, state.current_player)
    rows = encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)
    observation = torch.tensor(fit_feature_vector(obs.vector, model.config.observation_size), dtype=torch.float32)
    features = torch.tensor([fit_feature_vector(r.vector, model.config.action_feature_size) for r in rows], dtype=torch.float32)
    with torch.no_grad():
        logits, _ = model(observation[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all().item()
    values = logits[0].tolist()
    group_values = [max(values[j] for j in members[key]) for key in group_keys]
    offered = list(reference_keys)
    if len(offered) == 4:
        eligible = [j for j, key in enumerate(group_keys) if key not in offered[:3]]
        best = sorted(eligible, key=lambda j: (-group_values[j], j))[0]
        offered = offered[:3] + [group_keys[best]]
    identity = engine._MODEL_ARTIFACTS[model_id]
    return {"schedule": schedule, "reference": reference,
            "candidates": {"keys": offered, "actions": [representatives[k] for k in offered],
                           "canonical_support": len(group_keys), "legal_support": len(legal)},
            "model_id": model_id, "model_sha256": identity["sha256"], "fit": identity["source_fit"],
            "group_keys": group_keys, "legal_actions": [old.serialize_action(a) for a in legal],
            "learned_logits": values, "learned_group_scores": group_values}


def _proposals(state, spec, binding, base):
    """Separate legal enumeration and candidate assembly, with mature model math."""
    lock = old.read_json(Path(base) / "models_lock.json")
    assert lock["manifest"] == binding
    assert engine._MODEL_ARTIFACTS == lock["models"]
    assert set(lock["models"]) == {f"{arm}_{fit}" for fit in FITS for arm in ARMS}
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
    for arm in PROPOSERS:
        model_id = f"{arm}_{spec['fit']}"
        c.verify_hash(lock["models"][model_id])
        input_audit.input_invariance(state, engine._MODELS[model_id].config)
        expected = independent_proposal(state, spec["search_stream"], model_id)
        compare(expected, independent_proposal(altered, spec["search_stream"], model_id))
        proposals[arm] = expected
    reference = proposals["anchor"]["reference"]
    assert all(value["reference"] == reference for value in proposals.values())
    assert all(value["group_keys"] == group_keys for value in proposals.values())
    for key, action in zip(reference["keys"], reference["actions"], strict=True):
        representatives[key] = action
    first_three = reference["keys"][:3]
    keys_by_arm = {arm: list(proposals[arm]["candidates"]["keys"]) for arm in PROPOSERS}
    union_keys = list(first_three)
    for arm in PROPOSERS:
        assert keys_by_arm[arm][:3] == first_three
        for key in keys_by_arm[arm]:
            if key not in union_keys:
                union_keys.append(key)
    assert 1 <= len(union_keys) <= 6
    candidates = {arm: {"keys": keys, "actions": [representatives[k] for k in keys],
                        "indices": [union_keys.index(k) for k in keys]} for arm, keys in keys_by_arm.items()}
    proposed_indices = {arm: union_keys.index(keys_by_arm[arm][3]) if len(keys_by_arm[arm]) == 4 else None
                        for arm in PROPOSERS}
    return proposals, candidates, {"keys": union_keys, "actions": [representatives[k] for k in union_keys]}, proposed_indices


def audit(stage, spec, item, binding, base=OUTPUT):
    """Check all evidence and independently replay the prospectively sealed panel."""
    c.budget_boundary(base)
    start, cpu = time.perf_counter(), time.process_time()
    base = Path(base)
    index = spec["index"]
    _world_seeds(stage, index, "selection")
    assert spec["fit"] in FITS and spec["seat"] in (0, 1)
    path = base/stage/"roots"/f"state_{index:04d}.json"
    receipt_path = base/stage/"audits"/f"state_{index:04d}.json"
    assert not receipt_path.exists(), "Preserve existing audit evidence"
    assert c.verify_hash(item).resolve() == path.resolve()
    root = c.verified(item)
    assert root["manifest"] == binding and root["stage"] == stage and root["spec"] == spec
    assert "failure" not in root
    sample, sample_artifact = _checked_sample(stage, spec, binding, base)
    if "fixture_snapshot" not in spec:
        assert spec["source_stage"] == stage
        source_game, source_audit = c.verified(spec["source_game"]), c.verified(spec["source_audit"])
        assert source_game["manifest"] == source_audit["manifest"] == binding
        assert source_audit["passed"] and source_audit["game"] == spec["source_game"]
    state = _source_state(stage, spec, base)
    snapshot = old.serialize_state_snapshot(state)
    assert old.digest(snapshot) == spec["snapshot_sha256"] == old.digest(root["snapshot"])
    assert not state.terminal and state.turn_index == spec["turn"] and state.current_player == spec["seat"]
    assert spec["phase"] == min(state.turn_index//20, 2)
    own, opponent = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(opponent.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(opponent.reserved_cards)))}
    assert privacy == spec["privacy"] == root["privacy"]
    expected, candidates, union, proposed = _proposals(state, spec, binding, base)
    assert set(root["proposals"]) == set(PROPOSERS)
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
    assert len(root["selection"]["worlds"]) == 128
    assert all(len(w["outcomes"]) == len(actions) for w in root["selection"]["worlds"])
    totals = [sum(w["outcomes"][j]["wdl"] for w in root["selection"]["worlds"]) for j in range(len(actions))]
    choices = {arm: max([union["keys"].index(k) for k in candidates[arm]["keys"]], key=lambda j: totals[j]) for arm in ARMS}
    assert lock["choices"] == choices
    assert all(type(value) is int for value in lock["choices"].values())
    expected_indices = {"selection": list(range(len(actions))), "assessment": sorted(set(choices.values()))}
    assert lock["assessment_indices"] == expected_indices["assessment"]
    ordinary_wall, extra_wall = 0., 0.
    replay_by_block, counts = Counter(), Counter()
    moves, reasons = 0, Counter()
    for block in BLOCKS:
        data = root[block]
        seeds = _world_seeds(stage, index, block)
        assert set(data) == {"world_seeds", "worlds", "timing", "candidate_indices"}
        assert data["world_seeds"] == seeds and len(data["worlds"]) == 128
        assert data["candidate_indices"] == expected_indices[block]
        assert all(type(j) is int for j in data["candidate_indices"])
        block_actions = [actions[j] for j in expected_indices[block]]
        counts[f"recorded_{block}_worlds"] = 128
        counts[f"replayed_{block}_worlds"] = len(sample["replay"][block])
        counts[f"recorded_{block}_continuations"] = 128*len(block_actions)
        counts[f"replayed_{block}_continuations"] = len(sample["replay"][block])*len(block_actions)
        parent_verify._timing(data["timing"])
        for wi, (seed, saved) in enumerate(zip(seeds, data["worlds"], strict=True)):
            c.budget_boundary(base)
            assert set(saved) == {"seed", "state_sha256", "outcomes"}
            assert saved["seed"] == seed and len(saved["outcomes"]) == len(block_actions)
            world = world_audit.independent_world(state, seed)
            assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world))
            assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world_audit.independent_world(altered, seed)))
            assert world.players[state.current_player].reserved_cards == own.reserved_cards
            legal = old.SplendorEnv().legal_actions(world)
            replay_this = wi in sample["replay"][block]
            counts["replayed_worlds"] += int(replay_this)
            for action, outcome in zip(block_actions, saved["outcomes"], strict=True):
                assert action in legal
                parent_verify._check_outcome(outcome, state.current_player, state.turn_index)
                if replay_this:
                    replay_start = time.perf_counter()
                    rebuilt = continuation_audit.replay_branch(world, action, state.current_player)
                    compare(outcome, rebuilt)
                    elapsed = time.perf_counter()-replay_start
                    replay_by_block[block] += elapsed
                    if wi in sample["ordinary"][block]:
                        ordinary_wall += elapsed
                    else:
                        extra_wall += elapsed
                    moves += rebuilt["turns"]
                    reasons[rebuilt["reason"]] += 1
    summary = _summary(root["selection"]["worlds"], root["assessment"]["worlds"], candidates, union["keys"], proposed,
                       root["assessment"]["candidate_indices"])
    compare(lock["means"], summary.pop("selection_means"))
    assert lock["choices"] == summary["choices"]
    compare(root["summary"], summary)
    parent_verify._timing(root["timing"])
    replay_wall, wall = ordinary_wall+extra_wall, time.perf_counter()-start
    result = {"manifest": binding, "stage": stage, "spec": spec, "passed": True,
              "root": c.artifact(path), "selection_artifact": c.artifact(seal_path), "sample_artifact": sample_artifact,
              "row": _row(spec, summary), "privacy": privacy, "support_selection": len(actions),
              "support_assessment": len(expected_indices["assessment"]),
              "stress": sample["stress"], "ordinary_world_indices": sample["ordinary"],
              "replay_world_indices": sample["replay"], "recorded_worlds": 256,
              "recorded_continuations": sum(counts[f"recorded_{block}_continuations"] for block in BLOCKS),
              "replayed_continuations": sum(counts[f"replayed_{block}_continuations"] for block in BLOCKS),
              **dict(counts), "moves": moves, "reasons": dict(reasons),
              "full_inner_replay": counts["replayed_worlds"] == 256,
              "timing": {"wall": wall, "cpu": time.process_time()-cpu, "replay_wall": replay_wall,
                         "structural_wall": max(0., wall-replay_wall), "ordinary_replay_wall": ordinary_wall,
                         "full_extra_replay_wall": extra_wall,
                         "selection_replay_wall": replay_by_block["selection"],
                         "assessment_replay_wall": replay_by_block["assessment"]}}
    assert sum(reasons.values()) == result["replayed_continuations"]
    old.save_new(receipt_path, result)
    return c.artifact(receipt_path, index=index)


def statistics(rows):
    """Receipt-derived bootstrap using histogram counts, not indexed resampling."""
    assert len(rows) == 360 and [r["index"] for r in rows] == list(range(360))
    assert len({r["game_index"] for r in rows}) == 360
    assert all(r["pair"] == r["game_index"] for r in rows)
    strata = []
    for cell in CELLS:
        subset = [row for row in rows if (row["fit"], row["seat"], row["phase"]) == cell]
        assert len(subset) == 20
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
    rng = np.random.default_rng(2133000000)
    bootstrap = np.zeros((50000, len(CONTRASTS)), dtype=np.int64)
    for start in range(0, 50000, 1000):
        for matrix in strata:
            draws = rng.integers(0, 20, size=(1000, 20))
            counts = np.stack([np.bincount(indices, minlength=20) for indices in draws])
            bootstrap[start:start+1000] += counts @ matrix
    contrasts = {}
    for j, (name, _, _) in enumerate(CONTRASTS):
        ordered = sorted(int(v) for v in bootstrap[:, j])
        def quantile(q):
            point = (len(ordered)-1)*q
            lo, hi = math.floor(point), math.ceil(point)
            return (ordered[lo]+(ordered[hi]-ordered[lo])*(point-lo))/(360*256)
        estimate = int(totals[j])/(360*256)
        interval = [quantile(.025), quantile(.975)]
        contrasts[name] = {"mean_score_difference": estimate, "score_percentage_points": 100*estimate,
                           "ci95": interval, "ci95_percentage_points": [100*x for x in interval]}
    gates = {"anchor_practical_gain": int(totals[0])*200 >= 360*256,
             "anchor_superiority": contrasts["specialist_minus_anchor"]["ci95"][0] > 0,
             "control_superiority": contrasts["specialist_minus_control"]["ci95"][0] > 0}
    passed = all(gates.values())
    label = "positive_conditional_specialist_contribution" if passed else "practical_progress_not_established"
    return {"states": 360, "source_games": 360, "fits": list(FITS), "cells": [list(cell) for cell in CELLS],
            "states_per_cell": 20, "bootstrap_samples": 50000, "bootstrap_seed": 2133000000,
            "primary": "specialist_minus_anchor", "attribution": "specialist_minus_control", "secondary": "control_minus_anchor", "contrasts": contrasts,
            **gates, "gate_pass": passed, "classification": label,
            "scope": "Conditional expected terminal scores on fresh fast-student positions under greedy035 continuations; not whole-game win rates."}


def summarize(items, specs, binding, base=OUTPUT):
    c.budget_boundary(base)
    assert len(items) == len(specs) == 360
    assert [spec["index"] for spec in specs] == list(range(360))
    rows, coverage = [], Counter()
    for item, spec in zip(items, specs):
        c.budget_boundary(base)
        receipt = c.verified(item)
        assert receipt["manifest"] == binding and receipt["passed"] and receipt["spec"] == spec
        c.verify_hash(receipt["root"])
        c.verify_hash(receipt["selection_artifact"])
        c.verify_hash(receipt["sample_artifact"])
        assert {key: receipt["row"][key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "privacy")} == {
            key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "privacy")}
        rows.append(receipt["row"])
        coverage["states"] += 1
        coverage["full_stress_roots"] += int(receipt["stress"])
        for key in ("recorded_continuations", "replayed_continuations", "recorded_worlds", "replayed_worlds",
                    "recorded_selection_continuations", "recorded_assessment_continuations",
                    "replayed_selection_continuations", "replayed_assessment_continuations"):
            coverage[key] += receipt[key]
    c.budget_boundary(base)
    result = statistics(rows)
    c.budget_boundary(base)
    changes = {}
    for arm in ("specialist", "control"):
        eligible = [row for row in rows if row["proposed_indices"][arm] is not None]
        changes[arm] = {
            "eligible_states": len(eligible),
            "fourth_changed_vs_anchor": sum(row["proposed_indices"][arm] != row["proposed_indices"]["anchor"] for row in eligible),
            "choice_changed_vs_anchor": sum(row["choices"][arm] != row["choices"]["anchor"] for row in rows),
        }
    fourth_choices = {}
    for arm in PROPOSERS:
        eligible = [row for row in rows if row["proposed_indices"][arm] is not None]
        count = sum(row["choices"][arm] == row["proposed_indices"][arm] for row in eligible)
        fourth_choices[arm] = {"eligible_states": len(eligible), "fourth_chosen": count,
                               "fourth_chosen_fraction": count/len(eligible) if eligible else None}
    strata = {}
    for field in ("fit", "seat", "phase"):
        strata[field] = {}
        for value in sorted({row[field] for row in rows}):
            subset = [row for row in rows if row[field] == value]
            strata[field][str(value)] = {"states": len(subset), "mean_score_difference": {
                name: sum(row["arm_scores"][left]-row["arm_scores"][right] for row in subset)/len(subset)
                for name, left, right in CONTRASTS}}
    strata["privacy"] = {}
    for privacy in range(4):
        subset = [row for row in rows if _privacy_category(row) == privacy]
        strata["privacy"][str(privacy)] = {"states": len(subset), "mean_score_difference": {
            name: sum(row["arm_scores"][left]-row["arm_scores"][right] for row in subset)/len(subset) if subset else None
            for name, left, right in CONTRASTS}}
    return {"manifest": binding, "statistics": result, "rows": rows, "coverage": dict(coverage),
            "changes_descriptive": changes,
            "fourth_choice_descriptive": fourth_choices, "strata_descriptive": strata,
            "full_inner_replay": False}

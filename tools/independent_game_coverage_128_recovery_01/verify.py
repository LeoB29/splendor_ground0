"""Independent frozen-learner transfer audit, sparse assessment and censored bounds.

Reuses mature rules, observation encoders, neural arithmetic and independent
continuation replay. Does not call collector proposal, selection or summary code.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from tools.independent_game_coverage_128_recovery_01 import common as c
from tools.independent_game_coverage_128_recovery_01 import engine
from tools.proposal_iteration_128 import verify as parent_verify
from tools.equal_budget_breadth_arena_128_recovery_02 import verify as repetition_audit
from scripts.analysis import verify_rollout_selector_154 as world_audit
from scripts.analysis import verify_afterstate_advantage_149 as continuation_audit
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector

old, OUTPUT, FITS = c.old, c.OUTPUT, c.FITS
compare = parent_verify.compare
ARMS = ("anchor", "narrow", "broad", "lowrate", "soft", "teacher")
PROPOSERS = ARMS[:-1]
BLOCKS = ("selection", "assessment")
CONTRASTS = (
    ("broad_minus_anchor", "broad", "anchor"),
    ("broad_minus_narrow", "broad", "narrow"),
    ("broad_minus_lowrate", "broad", "lowrate"),
    ("broad_minus_soft", "broad", "soft"),
    ("narrow_minus_anchor", "narrow", "anchor"),
    ("narrow_minus_soft", "narrow", "soft"),
    ("teacher_minus_anchor", "teacher", "anchor"),
)
CELLS = tuple((fit, seat, phase) for fit in FITS for seat in (0, 1) for phase in range(3))
SAMPLE_CONFIG = {"seed": 2254000000,
                 "world_tag": "independent-game-coverage-128-audit-world-v1",
                 "stress_tag": "independent-game-coverage-128-audit-stress-v1",
                 "worlds_per_block": 128, "ordinary_worlds_per_block": 8,
                 "maximum_main_stress_roots": 39, "probe_full_replay": True,
                 "all_unresolved_cells_replayed": True}


def ordinary_indices(stage, index, block):
    assert stage in ("main", "probe", "test") and block in BLOCKS
    assert type(index) is int and index >= 0
    return sorted(sorted(range(128), key=lambda wi: (
        old.digest([SAMPLE_CONFIG["world_tag"], SAMPLE_CONFIG["seed"], stage, index, block, wi]), wi))[:8])


def _privacy_category(spec):
    value = spec["privacy"]
    assert set(value) == {"own", "opponent"}
    assert all(type(n) is int and n >= 0 for n in value.values())
    return int(bool(value["own"])) + 2*int(bool(value["opponent"]))


def validate_specs(specs, stage):
    assert stage in ("main", "probe", "test") and specs
    assert [s["index"] for s in specs] == list(range(len(specs)))
    assert len({s["game_index"] for s in specs}) == len(specs)
    assert all(type(s["game_index"]) is int and s["game_index"] >= 0
               and s["pair"] == s["game_index"] for s in specs)
    for spec in specs:
        assert spec["fit"] in FITS and spec["seat"] in (0, 1) and spec["phase"] in (0, 1, 2)
        assert type(spec["canonical_support"]) is int and spec["canonical_support"] >= 1
        _privacy_category(spec)
    if stage != "test":
        assert len(specs) == {"main": 360, "probe": 18}[stage]
        assert Counter((s["fit"], s["seat"], s["phase"]) for s in specs) == {
            cell: 20 if stage == "main" else 1 for cell in CELLS}


def sample_plan(specs, stage, binding):
    """Identity-only whole-world sample, plus metadata-selected stress roots."""
    validate_specs(specs, stage)
    def priority(spec):
        return old.digest([SAMPLE_CONFIG["stress_tag"], SAMPLE_CONFIG["seed"], stage, spec["index"]]), spec["index"]
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
                         "missing_cells": [r["cell"] for r in cells if r["selected_index"] is None],
                         "recorded_worlds": len(specs)*256,
                         "replayed_worlds": sum(len(r["replay"][b]) for r in selections for b in BLOCKS),
                         "full_inner_replay": stage == "probe"}}


def _checked_sample(stage, spec, binding, base):
    path = Path(base)/stage/"audit_sample.json"
    sample = old.read_json(path)
    assert sample["manifest"] == binding and sample["stage"] == stage and sample["config"] == SAMPLE_CONFIG
    row = sample["selections"][spec["index"]]
    assert row["index"] == spec["index"] and type(row["stress"]) is bool
    for block in BLOCKS:
        assert row["ordinary"][block] == ordinary_indices(stage, spec["index"], block)
        assert row["replay"][block] == (list(range(128)) if row["stress"] else row["ordinary"][block])
    return row, c.artifact(path)


def _source_state(stage, spec, base):
    if stage == "test":
        assert not Path(base).resolve().is_relative_to(OUTPUT.resolve())
        if "fixture_snapshot" in spec:
            return old.deserialize_state_snapshot(spec["fixture_snapshot"])
    else:
        assert "fixture_snapshot" not in spec
    from tools.independent_game_coverage_128_recovery_01 import source
    return old.deserialize_state_snapshot(source.replay_selected_state(spec, base))


def _world_seeds(stage, index, block):
    assert stage in ("probe", "main", "test") and block in BLOCKS
    assert type(index) is int and 0 <= index < (360 if stage == "main" else 18)
    first = c.CONFIG["world_seed_base"] + {"probe": 0, "main": 1000000, "test": 2000000}[stage] + 512*index
    first += 256 if block == "assessment" else 0
    return list(range(first, first+128))


def _bounded_columns(worlds, support):
    assert type(support) is int and 1 <= support <= 12 and len(worlds) == 128
    assert all(len(w["outcomes"]) == support for w in worlds)
    totals, missing, cells = [0]*support, [0]*support, []
    for wi, world in enumerate(worlds):
        for ci, outcome in enumerate(world["outcomes"]):
            value = outcome["wdl"]
            if value is None:
                assert outcome["reason"] == "repetition_cutoff" and outcome["winner"] is None
                missing[ci] += 1
                cells.append({"world": wi, "candidate": ci})
            else:
                assert outcome["reason"] in ("completed", "forced_pass_deadlock")
                assert type(value) in (float, int) and value in (-1, 0, 1)
                totals[ci] += int(value)
    return {"means": [total/128 if not missing[j] else None for j, total in enumerate(totals)],
            "means_bounds": [[(total-missing[j])/128, (total+missing[j])/128] for j, total in enumerate(totals)],
            "sums_wdl": totals, "unknown_counts": missing, "unresolved_cells": cells}


def independent_selection(worlds, candidates, support, proofs=None):
    """Exact integer pairwise guarantees, with original per-arm order on ties."""
    assert set(candidates) == set(ARMS)
    value = _bounded_columns(worlds, support)
    cells = value["unresolved_cells"]
    if proofs is None:
        groups = [{"context_sha256": None, "cells": [cell]} for cell in cells]
    else:
        assert [(p["world"], p["candidate"]) for p in proofs] == [(x["world"], x["candidate"]) for x in cells]
        contexts, grouped = {}, {}
        for proof in proofs:
            key = old.digest(proof["context"])
            assert proof["context_sha256"] == key
            assert key not in contexts or contexts[key] == proof["context"]
            contexts[key] = proof["context"]
            grouped.setdefault(key, []).append({"world": proof["world"], "candidate": proof["candidate"]})
        groups = [{"context_sha256": key, "cells": grouped[key]} for key in sorted(grouped)]
    multiplicities = [Counter(cell["candidate"] for cell in group["cells"]) for group in groups]
    choices, certificates = {}, {}
    for arm in ARMS:
        indices = candidates[arm]["indices"]
        assert indices and len(indices) == len(set(indices))
        assert all(type(j) is int and 0 <= j < support for j in indices)
        candidates_that_win = []
        for position, candidate in enumerate(indices):
            margins = [value["sums_wdl"][candidate]-value["sums_wdl"][other]
                       -sum(abs(group[candidate]-group[other]) for group in multiplicities) for other in indices]
            if all(margin >= (1 if other_position < position else 0)
                   for other_position, margin in enumerate(margins)):
                candidates_that_win.append((candidate, margins))
        assert len(candidates_that_win) <= 1
        choices[arm] = candidates_that_win[0][0] if candidates_that_win else None
        certificates[arm] = {"indices": indices, "choice": choices[arm],
                             "pairwise_lower_numerators": candidates_that_win[0][1] if candidates_that_win else None}
    value.update(choices=choices, certificate={
        "rule": "bounded_repetition_original_order_v1" if proofs is None else "coupled_repetition_original_order_v1",
        "unknown_groups": groups, "arm_certificates": certificates})
    return value


def independent_summary(worlds, assessment_indices, choices, proposed, support):
    assert set(choices) == set(proposed) == set(ARMS)
    assert all(type(j) is int and 0 <= j < support for j in choices.values())
    assert all(j is None or type(j) is int and 0 <= j < support for j in proposed.values())
    assert assessment_indices == sorted(set(choices.values())) and len(assessment_indices) <= 6
    values = _bounded_columns(worlds, len(assessment_indices))
    lookup = dict(zip(assessment_indices, range(len(assessment_indices)), strict=True))
    wdl = {arm: values["means"][lookup[j]] for arm, j in choices.items()}
    bounds = {arm: [(128+values["sums_wdl"][lookup[j]]-values["unknown_counts"][lookup[j]])/256,
                    (128+values["sums_wdl"][lookup[j]]+values["unknown_counts"][lookup[j]])/256]
              for arm, j in choices.items()}
    differences = {}
    for name, left, right in CONTRASTS:
        if choices[left] == choices[right]:
            differences[name] = [0., 0.]
        else:
            differences[name] = [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
    return {"assessment_indices": assessment_indices, "assessment_means_wdl": values["means"],
            "assessment_wdl_bounds": values["means_bounds"], "assessment_unknown_counts": values["unknown_counts"],
            "arm_wdl": wdl, "arm_scores": {a: (1+x)/2 if x is not None else None for a,x in wdl.items()},
            "arm_score_bounds": bounds, "delta_bounds": differences, "choices": choices, "proposed_indices": proposed}


def replay_cells(worlds, candidate_indices, sampled_worlds):
    """Count ordinary/stress cells once and add every otherwise-unsampled unknown."""
    assert len(set(candidate_indices)) == len(candidate_indices)
    sampled, unknown = set(), set()
    for wi, world in enumerate(worlds):
        assert len(world["outcomes"]) == len(candidate_indices)
        for ci, outcome in zip(candidate_indices, world["outcomes"], strict=True):
            cell = (wi, ci)
            if wi in sampled_worlds:
                sampled.add(cell)
            if outcome["wdl"] is None:
                assert outcome["reason"] == "repetition_cutoff" and outcome["winner"] is None
                unknown.add(cell)
    return {"sampled": sampled, "unknown": unknown, "extra": unknown-sampled, "all": sampled|unknown}


def _row(spec, summary):
    return {key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "privacy")} | {
        key: summary[key] for key in ("arm_scores", "arm_score_bounds", "delta_bounds", "choices", "proposed_indices")}


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


def _proposals(state, spec, binding, base, stage):
    lock = old.read_json(Path(base)/"models_lock.json")
    assert lock["manifest"] == binding
    assert engine._MODEL_ARTIFACTS == lock["models"]
    assert set(lock["models"]) == {f"{arm}_{fit}" for arm in PROPOSERS for fit in FITS}
    legal = old.SplendorEnv().legal_actions(state)
    group_keys, representatives = [], {}
    for action in legal:
        key = repr(old.canonical_successor_action_key(state, action))
        if key not in representatives:
            group_keys.append(key)
            representatives[key] = old.serialize_action(action)
    assert len(group_keys) == spec["canonical_support"]
    alternate = input_audit.hidden_alternative(state)
    input_audit.input_invariance(state, old._RAW._model.config)
    proposals = {}
    for arm in PROPOSERS:
        model_id = f"{arm}_{spec['fit']}"
        c.verify_hash(lock["models"][model_id])
        input_audit.input_invariance(state, engine._MODELS[model_id].config)
        expected = independent_proposal(state, spec["search_stream"], model_id)
        compare(expected, independent_proposal(alternate, spec["search_stream"], model_id))
        proposals[arm] = expected
    reference = proposals["anchor"]["reference"]
    assert all(p["reference"] == reference and p["group_keys"] == group_keys for p in proposals.values())
    representatives.update(zip(reference["keys"], reference["actions"], strict=True))
    first = reference["keys"][:3]
    by_arm = {arm: list(proposals[arm]["candidates"]["keys"]) for arm in PROPOSERS}
    assert all(keys[:3] == first for keys in by_arm.values())
    ranked_indices = sorted([j for j, key in enumerate(group_keys) if key not in by_arm["anchor"]],
                            key=lambda j: (-proposals["anchor"]["learned_group_scores"][j], j))
    ranked_indices = ranked_indices[:min(2, min(8, len(group_keys))-len(by_arm["anchor"]))]
    random_indices = [j for j, key in enumerate(group_keys) if key not in by_arm["anchor"] and j not in ranked_indices]
    random_indices.sort(key=lambda j: (old.digest([c.CONFIG["teacher_random_tag"], stage, spec["index"], group_keys[j]]), j))
    random_indices = random_indices[:min(8, len(group_keys))-len(by_arm["anchor"])-len(ranked_indices)]
    by_arm["teacher"] = by_arm["anchor"] + [group_keys[j] for j in ranked_indices+random_indices]
    union_keys = list(first)
    for arm in ARMS:
        for key in by_arm[arm]:
            if key not in union_keys:
                union_keys.append(key)
    assert 1 <= len(union_keys) <= 12
    candidates = {arm: {"keys": keys, "actions": [representatives[k] for k in keys],
                        "indices": [union_keys.index(k) for k in keys]} for arm, keys in by_arm.items()}
    assert all(len(candidates[a]["keys"]) == min(4, len(group_keys)) for a in PROPOSERS)
    assert len(candidates["teacher"]["keys"]) == min(8, len(group_keys))
    proposed = {arm: union_keys.index(keys[3]) if len(keys) >= 4 else None for arm, keys in by_arm.items()}
    added = {"ranked": [union_keys.index(group_keys[j]) for j in ranked_indices],
             "random": [union_keys.index(group_keys[j]) for j in random_indices]}
    return proposals, candidates, {"keys": union_keys, "actions": [representatives[k] for k in union_keys]}, proposed, added


def audit(stage, spec, item, binding, base=OUTPUT):
    """Authenticate every root, legal input, sparse budget and committed decision."""
    c.budget_boundary(base)
    start, cpu, base = time.perf_counter(), time.process_time(), Path(base)
    index = spec["index"]
    _world_seeds(stage, index, "selection")
    assert spec["fit"] in FITS and spec["seat"] in (0, 1)
    path, receipt_path = base/stage/"roots"/f"state_{index:04d}.json", base/stage/"audits"/f"state_{index:04d}.json"
    assert not receipt_path.exists(), "Preserve existing audit evidence"
    assert c.verify_hash(item).resolve() == path.resolve()
    root = c.verified(item)
    assert root["manifest"] == binding and root["stage"] == stage and root["spec"] == spec and "failure" not in root
    sample, sample_artifact = _checked_sample(stage, spec, binding, base)
    state = _source_state(stage, spec, base)
    snapshot = old.serialize_state_snapshot(state)
    assert old.digest(snapshot) == spec["snapshot_sha256"] == old.digest(root["snapshot"])
    assert not state.terminal and state.turn_index == spec["turn"] and state.current_player == spec["seat"]
    assert spec["phase"] == min(state.turn_index//20, 2)
    own, opponent = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(opponent.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(opponent.reserved_cards)))}
    assert privacy == spec["privacy"] == root["privacy"]
    expected, candidates, union, proposed, added = _proposals(state, spec, binding, base, stage)
    assert set(root["proposals"]) == set(PROPOSERS)
    for arm, proposal in expected.items():
        compare({k: v for k, v in root["proposals"][arm].items() if k != "timing"}, proposal)
        timing = root["proposals"][arm]["timing"]
        assert set(timing) == {"wall", "learned_extra_wall"}
        assert all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in timing.values())
        assert timing["learned_extra_wall"] <= timing["wall"]
    assert root["candidates"] == candidates and root["union"] == union and root["proposed_indices"] == proposed
    assert root["teacher_added_indices"] == added
    identity = {"spec": spec, "snapshot": root["snapshot"], "privacy": privacy, "proposals": expected,
                "candidates": candidates, "union": union, "proposed_indices": proposed, "teacher_added_indices": added}
    lock = root["selection_lock"]
    assert lock["manifest"] == binding and lock["stage"] == stage and lock["index"] == index
    assert lock["identity_digest"] == old.digest(identity) and lock["selection_sha256"] == old.digest(root["selection"])
    proofs = root.get("equivalence_proofs")
    assert lock["proofs_sha256"] == old.digest(proofs if proofs is not None else [])
    assert lock["value_scale"] == "terminal_wdl"
    seal_path = base/stage/"selections"/f"state_{index:04d}_selection.json"
    assert c.verify_hash(root["selection_artifact"]).resolve() == seal_path.resolve()
    seal = c.verified(root["selection_artifact"])
    compare(seal, {"manifest": binding, "stage": stage, "index": index, "identity": identity,
                   "selection": root["selection"], "equivalence_proofs": proofs if proofs is not None else [], "selection_lock": lock})
    created, assessment_start = lock["created_at_ns"], root["assessment_started_at_ns"]
    assert type(created) is type(assessment_start) is int and 0 < created <= assessment_start
    assert seal_path.stat().st_mtime_ns <= assessment_start
    assert c.verify_hash(sample_artifact).stat().st_mtime_ns <= created
    assert path.stat().st_mtime_ns >= assessment_start
    actions = [old.decode_action(value) for value in union["actions"]]
    selection = independent_selection(root["selection"]["worlds"], candidates, len(actions), proofs)
    assert all(type(value) is int for value in selection["choices"].values()), "Ambiguous selection must hold"
    if proofs is not None:
        # Coupling is a fallback only; never introduce new correlations without need.
        assert any(x is None for x in independent_selection(root["selection"]["worlds"], candidates, len(actions))["choices"].values())
        assert proofs and len(proofs) == len(selection["unresolved_cells"])
        assert set(root["proof_timing"]) == {"wall", "cpu", "replayed_continuations"}
        parent_verify._timing({k: root["proof_timing"][k] for k in ("wall", "cpu")})
        assert root["proof_timing"]["replayed_continuations"] == len(proofs)
    else:
        assert "proof_timing" not in root
    compare(selection, lock["selection_statistics"])
    assert lock["choices"] == selection["choices"]
    indices = {"selection": list(range(len(actions))), "assessment": sorted(set(selection["choices"].values()))}
    assert lock["assessment_indices"] == indices["assessment"]
    assert len(indices["assessment"]) <= 6
    alternate = input_audit.hidden_alternative(state)
    ordinary_wall, extra_wall, moves = 0., 0., 0
    replay_wall_by_block, counts, reasons = Counter(), Counter(), Counter()
    proof_by_cell = {(p["world"], p["candidate"]): p for p in proofs} if proofs is not None else {}
    assert len(proof_by_cell) == len(proofs or [])
    coverage_cells, independently_replayed_proofs = {}, []
    for block in BLOCKS:
        data = root[block]
        seeds = _world_seeds(stage, index, block)
        assert set(data) == {"world_seeds", "worlds", "timing", "candidate_indices"}
        assert data["world_seeds"] == seeds and len(data["worlds"]) == 128
        assert data["candidate_indices"] == indices[block] and all(type(j) is int for j in data["candidate_indices"])
        parent_verify._timing(data["timing"])
        cells = replay_cells(data["worlds"], indices[block], sample["replay"][block])
        coverage_cells[block] = {"unknown": sorted(cells["unknown"]), "extra": sorted(cells["extra"])}
        counts[f"recorded_{block}_worlds"] = 128
        counts[f"replayed_{block}_worlds"] = len({wi for wi, _ in cells["all"]})
        counts[f"recorded_{block}_continuations"] = 128*len(indices[block])
        counts[f"replayed_{block}_continuations"] = len(cells["all"])
        counts[f"unresolved_{block}_continuations"] = len(cells["unknown"])
        counts[f"additional_unknown_{block}_replays"] = len(cells["extra"])
        if block == "selection" and proofs is not None:
            assert set(proof_by_cell) == cells["unknown"]
        for wi, (seed, saved) in enumerate(zip(seeds, data["worlds"], strict=True)):
            c.budget_boundary(base)
            assert set(saved) == {"seed", "state_sha256", "outcomes"} and saved["seed"] == seed
            world = world_audit.independent_world(state, seed)
            assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world))
            assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world_audit.independent_world(alternate, seed)))
            assert world.players[state.current_player].reserved_cards == own.reserved_cards
            legal = old.SplendorEnv().legal_actions(world)
            for ci, outcome in zip(indices[block], saved["outcomes"], strict=True):
                action = actions[ci]
                assert action in legal
                repetition_audit._check_outcome(outcome, state.current_player, state.turn_index)
                if (wi, ci) in cells["all"]:
                    replay_start = time.perf_counter()
                    if block == "selection" and (wi, ci) in proof_by_cell:
                        proof = repetition_audit.independent_equivalence_proof(world, action, state.current_player, wi, ci)
                        assert old.digest(proof) == old.digest(proof_by_cell[(wi, ci)])
                        independently_replayed_proofs.append(proof)
                        rebuilt = {k: proof["replay"][k] for k in outcome if k != "wdl"} | {"wdl": None}
                    else:
                        rebuilt = continuation_audit.replay_branch(world, action, state.current_player)
                    compare(outcome, rebuilt)
                    elapsed = time.perf_counter()-replay_start
                    replay_wall_by_block[block] += elapsed
                    if wi in sample["ordinary"][block]:
                        ordinary_wall += elapsed
                    else:
                        extra_wall += elapsed
                    moves += rebuilt["turns"]
                    reasons[rebuilt["reason"]] += 1
    if proofs is not None:
        compare(selection, independent_selection(root["selection"]["worlds"], candidates, len(actions), independently_replayed_proofs))
    summary = independent_summary(root["assessment"]["worlds"], indices["assessment"], selection["choices"], proposed, len(actions))
    compare(root["summary"], summary)
    parent_verify._timing(root["timing"])
    recorded = sum(counts[f"recorded_{block}_continuations"] for block in BLOCKS)
    replayed = sum(counts[f"replayed_{block}_continuations"] for block in BLOCKS)
    unknown = sum(counts[f"unresolved_{block}_continuations"] for block in BLOCKS)
    replay_wall, wall = ordinary_wall+extra_wall, time.perf_counter()-start
    result = {"manifest": binding, "stage": stage, "spec": spec, "passed": True,
              "root": c.artifact(path), "selection_artifact": c.artifact(seal_path), "sample_artifact": sample_artifact,
              "row": _row(spec, summary), "privacy": privacy, "support_selection": len(actions),
              "support_assessment": len(indices["assessment"]), "stress": sample["stress"],
              "ordinary_world_indices": sample["ordinary"], "replay_world_indices": sample["replay"],
              "unresolved_replay_cells": coverage_cells, "recorded_worlds": 256,
              "replayed_worlds": sum(counts[f"replayed_{block}_worlds"] for block in BLOCKS),
              "recorded_continuations": recorded, "replayed_continuations": replayed,
              "unresolved_continuations": unknown, "terminal_continuations": recorded-unknown,
              "coupled_certified_roots": int(proofs is not None), "equivalence_proof_cells": len(proofs or []),
              "selection_certified_unknown_roots": int(bool(selection["unresolved_cells"])),
              **dict(counts), "moves": moves, "reasons": dict(reasons), "full_inner_replay": replayed == recorded,
              "timing": {"wall": wall, "cpu": time.process_time()-cpu, "replay_wall": replay_wall,
                         "structural_wall": max(0., wall-replay_wall), "ordinary_replay_wall": ordinary_wall,
                         "full_extra_replay_wall": extra_wall,
                         "selection_replay_wall": replay_wall_by_block["selection"],
                         "assessment_replay_wall": replay_wall_by_block["assessment"]}}
    assert sum(reasons.values()) == replayed and reasons["repetition_cutoff"] == unknown
    old.save_new(receipt_path, result)
    return c.artifact(receipt_path, index=index)


def summarize(items, specs, binding, base=OUTPUT):
    """Receipt-derived rows and coverage, with histogram inference for the fresh main panel."""
    c.budget_boundary(base)
    assert len(items) == len(specs) and specs
    stage = c.verified(items[0])["stage"]
    validate_specs(specs, stage)
    sample = old.read_json(Path(base)/stage/"audit_sample.json")
    compare(sample, sample_plan(specs, stage, binding))
    rows, coverage = [], Counter()
    for item, spec in zip(items, specs, strict=True):
        c.budget_boundary(base)
        receipt = c.verified(item)
        assert receipt["manifest"] == binding and receipt["stage"] == stage and receipt["passed"] and receipt["spec"] == spec
        for key in ("root", "selection_artifact", "sample_artifact"):
            c.verify_hash(receipt[key])
        assert {k: receipt["row"][k] for k in ("index", "pair", "game_index", "fit", "seat", "phase", "privacy")} == {
            k: spec[k] for k in ("index", "pair", "game_index", "fit", "seat", "phase", "privacy")}
        root = c.verified(receipt["root"])
        selection = independent_selection(root["selection"]["worlds"], root["candidates"], len(root["union"]["keys"]), root.get("equivalence_proofs"))
        compare(selection, root["selection_lock"]["selection_statistics"])
        summary = independent_summary(root["assessment"]["worlds"], root["assessment"]["candidate_indices"],
                                      selection["choices"], root["proposed_indices"], len(root["union"]["keys"]))
        compare(receipt["row"], _row(spec, summary))
        rows.append(receipt["row"])
        coverage["states"] += 1
        coverage["full_stress_roots"] += int(receipt["stress"])
        for key in ("recorded_continuations", "replayed_continuations", "recorded_worlds", "replayed_worlds",
                    "unresolved_continuations", "terminal_continuations", "coupled_certified_roots", "equivalence_proof_cells",
                    "selection_certified_unknown_roots", "recorded_selection_continuations", "recorded_assessment_continuations",
                    "replayed_selection_continuations", "replayed_assessment_continuations",
                    "unresolved_selection_continuations", "unresolved_assessment_continuations"):
            coverage[key] += receipt[key]
    return {"manifest": binding, "rows": rows, "coverage": dict(coverage),
            "statistics": statistics(rows) if stage == "main" else None,
            "full_inner_replay": coverage["recorded_continuations"] == coverage["replayed_continuations"]}




def statistics(rows):
    """Paired whole-source-game bootstrap using independent frequency histograms."""
    assert len(rows) == 360 and [r["index"] for r in rows] == list(range(360))
    assert len({r["game_index"] for r in rows}) == 360
    assert all(type(r["game_index"]) is int and r["game_index"] >= 0 and r["pair"] == r["game_index"] for r in rows)
    assert Counter((r["fit"], r["seat"], r["phase"]) for r in rows) == {cell:20 for cell in CELLS}
    strata = []
    for cell in CELLS:
        subset = [r for r in rows if (r["fit"], r["seat"], r["phase"]) == cell]
        values = []
        for row in subset:
            assert set(row["arm_scores"]) == set(row["arm_score_bounds"]) == set(row["choices"]) == set(ARMS)
            assert set(row["delta_bounds"]) == {name for name, _, _ in CONTRASTS}
            for arm in ARMS:
                bounds = row["arm_score_bounds"][arm]
                assert type(row["choices"][arm]) is int and 0 <= row["choices"][arm] < 12
                assert len(bounds) == 2 and 0 <= bounds[0] <= bounds[1] <= 1
                assert all(type(x) in (int,float) and math.isfinite(x) and x*256 == int(x*256) for x in bounds)
                assert row["arm_scores"][arm] == (bounds[0] if bounds[0] == bounds[1] else None)
            for left in ARMS:
                for right in ARMS:
                    if row["choices"][left] == row["choices"][right]:
                        assert row["arm_score_bounds"][left] == row["arm_score_bounds"][right]
            integers = []
            for name, left, right in CONTRASTS:
                bound = row["delta_bounds"][name]
                expected = [0.,0.] if row["choices"][left] == row["choices"][right] else [
                    row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][1],
                    row["arm_score_bounds"][left][1]-row["arm_score_bounds"][right][0]]
                assert bound == expected and len(bound) == 2 and -1 <= bound[0] <= bound[1] <= 1
                assert all(type(x) in (int,float) and math.isfinite(x) and x*256 == int(x*256) for x in bound)
                integers.extend(int(x*256) for x in bound)
            values.append(integers)
        strata.append(np.asarray(values,dtype=np.int64))
    totals = sum((matrix.sum(axis=0) for matrix in strata),np.zeros(2*len(CONTRASTS),dtype=np.int64))
    count, chunk = c.CONFIG["bootstrap_samples"], c.CONFIG["bootstrap_chunk"]
    rng = np.random.default_rng(c.CONFIG["bootstrap_seed"])
    bootstrap = np.zeros((count,2*len(CONTRASTS)),dtype=np.int64)
    for start in range(0,count,chunk):
        size = min(chunk,count-start)
        for matrix in strata:
            draws = rng.integers(0,20,size=(size,20))
            histograms = np.stack([np.bincount(draw,minlength=20) for draw in draws])
            bootstrap[start:start+size] += histograms @ matrix
    denominator, contrasts = 360*256, {}
    def quantile(column,fraction):
        ordered = np.sort(column)
        position = (len(ordered)-1)*fraction
        lower,upper = math.floor(position),math.ceil(position)
        return (int(ordered[lower])+(int(ordered[upper])-int(ordered[lower]))*(position-lower))/denominator
    for j,(name,_,_) in enumerate(CONTRASTS):
        bounds = [int(totals[2*j])/denominator,int(totals[2*j+1])/denominator]
        interval = [quantile(bootstrap[:,2*j],.025),quantile(bootstrap[:,2*j+1],.975)]
        point = bounds[0] if bounds[0] == bounds[1] else None
        contrasts[name] = {"mean_score_difference":point,"mean_bounds":bounds,
                           "score_percentage_points":None if point is None else 100*point,
                           "ci95":interval,"ci95_percentage_points":[100*x for x in interval]}
    gates = {"anchor_practical_gain":int(totals[0])*200 >= denominator,
             "anchor_superiority":contrasts["broad_minus_anchor"]["ci95"][0] > 0,
             "lowrate_superiority":contrasts["broad_minus_lowrate"]["ci95"][0] > 0}
    passed = all(gates.values())
    return {"states":360,"source_games":360,"fits":list(FITS),"cells":[list(cell) for cell in CELLS],
            "states_per_cell":20,"bootstrap_samples":count,"bootstrap_seed":c.CONFIG["bootstrap_seed"],
            "primary":"broad_minus_anchor","required_control":"broad_minus_lowrate","attribution":"broad_minus_narrow",
            "contrasts":contrasts,**gates,"gate_pass":passed,
            "breadth_attribution":contrasts["broad_minus_narrow"]["ci95"][0] > 0,
            "classification":"positive_conditional_data_package" if passed else "practical_progress_not_established",
            "unknown_outcomes_imputed":False,
            "scope":"Conditional expected terminal scores on fresh fast-student positions under greedy035 continuations; not whole-game win rates."}

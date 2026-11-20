"""Independent choices, inference, fresh-world arithmetic and sampled replay.

No collector choice, grouping, world-construction or summary helper is used.
The audited original corpus is inherited; every new world law and fixed choice
is reconstructed, with full probe/stress and all unknown continuation replay.
"""
from collections import Counter
import math
from pathlib import Path
import time
import numpy as np
import torch
from . import common as c
from .statistics import ARMS, PROPOSERS, CONTRASTS, NAMES, PRIMARY, SPLITS
from tools.proposal_iteration_128 import verify as parent_verify
from tools.equal_budget_breadth_arena_128_recovery_02 import verify as repetition_audit
from scripts.analysis import verify_rollout_selector_154 as world_audit
from scripts.analysis import verify_afterstate_advantage_149 as continuation_audit
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector

old, OUTPUT, FITS = c.old, c.OUTPUT, c.FITS
compare = parent_verify.compare
SAMPLE_CONFIG = {"seed": 2224000000,
    "world_tag": "teacher-signal-capture-128-audit-world-v1",
    "stress_tag": "teacher-signal-capture-128-audit-stress-v1",
    "worlds_per_block": 128, "ordinary_worlds_per_block": 8,
    "maximum_main_stress_roots": 39, "probe_full_replay": True,
    "all_unresolved_cells_replayed": True}


def ordinary_indices(stage, index):
    assert stage in ("main", "probe", "test") and type(index) is int and index >= 0
    return sorted(sorted(range(128), key=lambda wi: (
        old.digest([SAMPLE_CONFIG["world_tag"], SAMPLE_CONFIG["seed"], stage, index, wi]), wi))[:8])


def validate_specs(specs, stage):
    assert stage in ("main", "probe", "test") and specs
    assert [s["index"] for s in specs] == list(range(len(specs)))
    assert len({(s["game_index"], s["phase"]) for s in specs}) == len(specs)
    for spec in specs:
        assert spec["fit"] in FITS and spec["seat"] in (0, 1) and spec["phase"] in range(3)
        assert spec["split"] in SPLITS and spec["pair"] == spec["game_index"]//2
        assert type(spec["canonical_support"]) is int and spec["canonical_support"] >= 1
        assert set(spec["privacy"]) == {"own", "opponent"}
        assert all(type(v) is int and v >= 0 for v in spec["privacy"].values())
    if stage == "test":
        return
    assert len(specs) == {"main": 336, "probe": 18}[stage]
    counts = Counter()
    for pair in sorted({s["pair"] for s in specs}):
        rows = [s for s in specs if s["pair"] == pair]
        assert len(rows) == (4 if stage == "main" else 2)
        assert all((r["fit"], r["split"]) == (rows[0]["fit"], rows[0]["split"]) for r in rows)
        assert {r["game_index"] for r in rows} == {2*pair, 2*pair+1}
        assert {r["seat"] for r in rows} == {0, 1}
        phase_sets = [{r["phase"] for r in rows if r["game_index"] == g} for g in (2*pair, 2*pair+1)]
        assert phase_sets[0] == phase_sets[1] and len(phase_sets[0]) == (2 if stage == "main" else 1)
        counts[(rows[0]["split"], rows[0]["fit"])] += 1
    if stage == "main":
        assert counts == {(s, f): 14 for s in SPLITS for f in FITS}
    else:
        assert counts == {("train", f): 3 for f in FITS}


def sample_plan(specs, stage, binding):
    validate_specs(specs, stage)
    def priority(spec):
        return old.digest([SAMPLE_CONFIG["stress_tag"], SAMPLE_CONFIG["seed"], stage, spec["index"]]), spec["index"]
    cells, selected = [], set()
    def add_cell(name, eligible):
        choice = min(eligible, key=priority)["index"] if eligible else None
        cells.append({"cell": name, "eligible_count": len(eligible), "selected_index": choice})
        if choice is not None:
            selected.add(choice)
    # Mature 39-cell physics/privacy/short-support stress recipe, identity-only.
    for fit in FITS:
        group = [s for s in specs if s["fit"] == fit]
        for seat in (0, 1):
            for phase in range(3):
                add_cell(f"fit_seat_phase|{fit}|{seat}|{phase}", [s for s in group if (s["seat"], s["phase"]) == (seat, phase)])
        for privacy in range(4):
            add_cell(f"fit_privacy|{fit}|{privacy}", [s for s in group if int(bool(s["privacy"]["own"]))+2*int(bool(s["privacy"]["opponent"])) == privacy])
        for support in (1, 2, 3):
            add_cell(f"fit_short_support|{fit}|{support}", [s for s in group if s["canonical_support"] == support])
    if stage == "probe":
        selected = {s["index"] for s in specs}
    selections = [{"index": s["index"], "stress": s["index"] in selected,
                   "ordinary": {"assessment": ordinary_indices(stage, s["index"])},
                   "replay": {"assessment": list(range(128)) if s["index"] in selected else ordinary_indices(stage, s["index"])}} for s in specs]
    return {"manifest": binding, "stage": stage, "config": SAMPLE_CONFIG,
            "specs_sha256": old.digest(specs), "selections": selections, "stress_cells": cells,
            "coverage": {"states": len(specs), "full_stress_roots": len(selected), "stress_cells": 39,
                         "missing_cells": [r["cell"] for r in cells if r["selected_index"] is None],
                         "recorded_worlds": len(specs)*128,
                         "replayed_worlds": sum(len(r["replay"]["assessment"]) for r in selections),
                         "full_inner_replay": stage == "probe"}}


def _bounded_columns(worlds, support, count=128):
    assert type(support) is int and 1 <= support <= 8 and len(worlds) == count
    assert all(len(w["outcomes"]) == support for w in worlds)
    totals, unknowns, cells = [0]*support, [0]*support, []
    for wi, world in enumerate(worlds):
        for ci, outcome in enumerate(world["outcomes"]):
            value = outcome["wdl"]
            if value is None:
                assert outcome["reason"] == "repetition_cutoff" and outcome["winner"] is None
                unknowns[ci] += 1
                cells.append({"world": wi, "candidate": ci})
            else:
                assert outcome["reason"] in ("completed", "forced_pass_deadlock")
                assert type(value) in (float, int) and value in (-1, 0, 1)
                totals[ci] += int(value)
    return {"means": [v/count if not unknowns[j] else None for j, v in enumerate(totals)],
            "means_bounds": [[(v-unknowns[j])/count, (v+unknowns[j])/count] for j, v in enumerate(totals)],
            "sums_wdl": totals, "unknown_counts": unknowns, "unresolved_cells": cells}


def _certificate(values, indices):
    assert len(set(indices)) == len(indices)
    certified = []
    for position, candidate in enumerate(indices):
        margins = [0 if candidate == other else values["sums_wdl"][candidate]-values["sums_wdl"][other]
                   -values["unknown_counts"][candidate]-values["unknown_counts"][other] for other in indices]
        if all(margin >= (1 if op < position else 0) for op, margin in enumerate(margins)):
            certified.append((candidate, margins))
    assert len(certified) <= 1
    return {"indices": list(indices), "choice": certified[0][0] if certified else None,
            "pairwise_lower_numerators": certified[0][1] if certified else None}


def independent_selection(worlds, proposals, support):
    assert set(proposals) == set(PROPOSERS)
    values = _bounded_columns(worlds, support)
    supports = {"anchor": list(range(min(4, support))), "teacher": list(range(support)),
                "teacher_outsider": list(range(3, support))}
    choices, certificates = {"original_fourth": 3 if support > 3 else None}, {}
    supports["original_fourth"] = [3] if support > 3 else []
    for arm in ("anchor", "teacher", "teacher_outsider"):
        certificates[arm] = _certificate(values, supports[arm])
        choices[arm] = certificates[arm]["choice"]
    for family in PROPOSERS:
        scores = proposals[family]["offered_group_scores"]
        assert len(scores) == support and all(type(x) in (int, float) and math.isfinite(x) for x in scores)
        for kind, indices in (("direct", list(range(support))), ("outsider", list(range(3, support)))):
            arm = f"{family}_{kind}"
            supports[arm] = indices
            choices[arm] = min(indices, key=lambda j: (-scores[j], j)) if indices else None
        arm = f"{family}_supported"
        supports[arm] = list(range(min(3, support)))
        outsider = choices[f"{family}_outsider"]
        if outsider is not None:
            supports[arm].append(outsider)
        certificates[arm] = _certificate(values, supports[arm])
        choices[arm] = certificates[arm]["choice"]
    values.update(choices=choices, supports=supports, rollout_certificates=certificates)
    assert choices["anchor_outsider"] == choices["original_fourth"]
    assert choices["anchor_supported"] == choices["anchor"]
    return values


def _scores(worlds, choices, support, count):
    value = _bounded_columns(worlds, support, count)
    bounds = {arm: None if j is None else [
        (count+value["sums_wdl"][j]-value["unknown_counts"][j])/(2*count),
        (count+value["sums_wdl"][j]+value["unknown_counts"][j])/(2*count)] for arm, j in choices.items()}
    scores = {arm: b[0] if b is not None and b[0] == b[1] else None for arm, b in bounds.items()}
    deltas = {}
    for name, left, right in CONTRASTS:
        if choices[left] == choices[right]:
            deltas[name] = [0., 0.]
        else:
            assert bounds[left] is not None and bounds[right] is not None
            deltas[name] = [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
    return value, scores, bounds, deltas


def replay_cells(worlds, sampled_worlds):
    sampled, unknown = set(), set()
    for wi, world in enumerate(worlds):
        for ci, outcome in enumerate(world["outcomes"]):
            if wi in sampled_worlds:
                sampled.add((wi, ci))
            if outcome["wdl"] is None:
                assert outcome["reason"] == "repetition_cutoff" and outcome["winner"] is None
                unknown.add((wi, ci))
    return {"sampled": sampled, "unknown": unknown, "extra": unknown-sampled, "all": sampled|unknown}


def independent_proposal(state, candidate_keys, model_id):
    from . import engine
    legal = old.SplendorEnv().legal_actions(state)
    keys, members = [], {}
    for index, action in enumerate(legal):
        key = repr(old.canonical_successor_action_key(state, action))
        if key not in members:
            keys.append(key)
            members[key] = []
        members[key].append(index)
    model = engine._MODELS[model_id]
    observation = encode_public_observation_tensor(state, state.current_player)
    action_rows = encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)
    obs = torch.tensor(fit_feature_vector(observation.vector, model.config.observation_size), dtype=torch.float32)
    actions = torch.tensor([fit_feature_vector(r.vector, model.config.action_feature_size) for r in action_rows], dtype=torch.float32)
    with torch.no_grad():
        logits, _ = model(obs[None], actions[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all().item()
    values = logits[0].tolist()
    group_values = [max(values[j] for j in members[key]) for key in keys]
    assert all(key in keys for key in candidate_keys)
    artifact = engine._MODEL_ARTIFACTS[model_id]
    return {"model_id": model_id, "model_sha256": artifact["sha256"], "fit": artifact["source_fit"],
            "group_keys": keys, "legal_actions": [old.serialize_action(a) for a in legal],
            "learned_logits": values, "learned_group_scores": group_values,
            "offered_group_scores": [group_values[keys.index(key)] for key in candidate_keys]}


def _world_seeds(stage, index):
    assert stage in ("probe", "main", "test") and type(index) is int
    assert 0 <= index < {"probe": 18, "main": 336, "test": 18}[stage]
    first = 2210000000 + {"probe": 0, "main": 1000000, "test": 2000000}[stage] + 512*index
    return list(range(first, first+128))


def _row(spec, summary):
    return {key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "split", "privacy")} | {
        key: summary[key] for key in ("choices", "arm_scores", "arm_score_bounds", "delta_bounds",
                                     "fresh_selection", "outsider_available")} | {"support": len(summary["assessment_indices"])}


def independent_summary(worlds, choices):
    support = len(worlds[0]["outcomes"])
    assert set(choices) == set(ARMS)
    assert all((j is None) == (support <= 3 and (arm.endswith("_outsider") or arm == "original_fourth"))
               for arm, j in choices.items())
    value, scores, bounds, deltas = _scores(worlds, choices, support, 128)
    teacher = _certificate(value, list(range(support)))
    outsider = _certificate(value, list(range(3, support)))
    return {"outsider_available": support > 3, "assessment_indices": list(range(support)),
            "assessment_means_wdl": value["means"], "assessment_wdl_bounds": value["means_bounds"],
            "assessment_unknown_counts": value["unknown_counts"], "choices": choices,
            "arm_scores": scores, "arm_score_bounds": bounds, "delta_bounds": deltas,
            "fresh_selection": {"teacher_certificate": teacher, "outsider_certificate": outsider,
                "teacher_agreement": None if teacher["choice"] is None else teacher["choice"] == choices["teacher"],
                "outsider_agreement": None if outsider["choice"] is None else outsider["choice"] == choices["teacher_outsider"]}}


def audit(stage, spec, item, binding, base=OUTPUT):
    from . import engine, source
    c.budget_boundary(base)
    started, cpu, base = time.perf_counter(), time.process_time(), Path(base)
    index = spec["index"]
    seeds = _world_seeds(stage, index)
    path = base/stage/"roots"/f"state_{index:04d}.json"
    receipt_path = base/stage/"audits"/f"state_{index:04d}.json"
    assert not receipt_path.exists(), "Preserve existing independent evidence"
    assert c.verify_hash(item).resolve() == path.resolve()
    root = c.verified(item)
    assert root["manifest"] == binding and root["stage"] == stage and old.digest(root["spec"]) == old.digest(spec) and "failure" not in root
    sample_path = base/stage/"audit_sample.json"
    sample_artifact = c.artifact(sample_path)
    sample = c.verified(sample_artifact)
    assert sample["manifest"] == binding and sample["stage"] == stage and sample["config"] == SAMPLE_CONFIG
    selected = sample["selections"][index]
    assert selected["index"] == index and type(selected["stress"]) is bool
    assert selected["ordinary"] == {"assessment": ordinary_indices(stage, index)}
    assert selected["replay"] == {"assessment": list(range(128)) if selected["stress"] else ordinary_indices(stage, index)}
    if stage == "test" and "fixture_root" in spec:
        assert not base.resolve().is_relative_to(OUTPUT.resolve())
        original = spec["fixture_root"]
    else:
        assert "fixture_root" not in spec
        original = source.load_root(spec, base)
    assert "failure" not in original and not original.get("equivalence_proofs")
    assert old.digest(original["snapshot"]) == spec["snapshot_sha256"] == old.digest(root["snapshot"])
    state = old.deserialize_state_snapshot(original["snapshot"])
    assert not state.terminal and state.current_player == spec["seat"] and state.turn_index == spec["turn"]
    assert min(state.turn_index//20, 2) == spec["phase"]
    own, opponent = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(opponent.reserved_card_source(i) != input_audit.RESERVE_SOURCE_VISIBLE for i in range(len(opponent.reserved_cards)))}
    assert privacy == root["privacy"] == original["privacy"] == spec["privacy"]
    candidates = original["candidates"]
    assert root["candidates"] == candidates
    keys, actions = candidates["keys"], [old.decode_action(a) for a in candidates["actions"]]
    support = len(keys)
    assert 1 <= support <= 8 and support == len(actions) == len(set(keys))
    legal = old.SplendorEnv().legal_actions(state)
    assert all(action in legal for action in actions)
    assert [repr(old.canonical_successor_action_key(state, a)) for a in actions] == keys
    all_keys = list(dict.fromkeys(repr(old.canonical_successor_action_key(state, a)) for a in legal))
    assert len(all_keys) == spec["canonical_support"] and support == min(8, len(all_keys))
    old_block = {"candidate_indices": list(range(support)), "world_seeds": original["schedule"]["world_seeds"], "worlds": original["worlds"]}
    assert root["old_selection"] == old_block and root["old_teacher_choice"] == original["choice"] and root["old_means"] == original["means"]
    assert len(old_block["world_seeds"]) == len(set(old_block["world_seeds"])) == 128
    assert [w["seed"] for w in old_block["worlds"]] == old_block["world_seeds"]
    assert set(old_block["world_seeds"]).isdisjoint(seeds)
    registry = old.read_json(base/"models_lock.json")
    assert registry["manifest"] == binding and registry["models"] == engine._MODEL_ARTIFACTS
    assert set(registry["models"]) == {f"{p}_{f}" for p in PROPOSERS for f in FITS}
    alternate = input_audit.hidden_alternative(state)
    input_audit.input_invariance(state, old._RAW._model.config)
    proposals = {}
    assert set(root["proposals"]) == set(PROPOSERS)
    for family in PROPOSERS:
        model_id = f"{family}_{spec['fit']}"
        c.verify_hash(registry["models"][model_id])
        input_audit.input_invariance(state, engine._MODELS[model_id].config)
        expected = independent_proposal(state, keys, model_id)
        compare(expected, independent_proposal(alternate, keys, model_id))
        proposals[family] = expected
        saved = root["proposals"][family]
        compare({k: v for k, v in saved.items() if k != "timing"}, expected)
        assert set(saved["timing"]) == {"wall"} and type(saved["timing"]["wall"]) in (int, float)
        assert math.isfinite(saved["timing"]["wall"]) and saved["timing"]["wall"] >= 0
    identity = {"spec": root["spec"], "snapshot": root["snapshot"], "privacy": privacy, "candidates": candidates,
                "old_teacher_choice": original["choice"], "old_means": original["means"], "proposals": proposals}
    lock = root["selection_lock"]
    assert lock["manifest"] == binding and lock["stage"] == stage and lock["index"] == index
    assert lock["identity_digest"] == old.digest(identity) and lock["old_selection_sha256"] == old.digest(old_block)
    seal_path = base/stage/"selections"/f"state_{index:04d}_selection.json"
    assert c.verify_hash(root["selection_artifact"]).resolve() == seal_path.resolve()
    seal = c.verified(root["selection_artifact"])
    compare(seal, {"manifest": binding, "stage": stage, "index": index, "identity": identity,
                   "old_selection": old_block, "selection_lock": lock})
    created, assessment_start = lock["created_at_ns"], root["assessment_started_at_ns"]
    assert type(created) is type(assessment_start) is int and 0 < created <= assessment_start
    assert sample_path.stat().st_mtime_ns <= created and seal_path.stat().st_mtime_ns <= assessment_start <= path.stat().st_mtime_ns
    selection = independent_selection(old_block["worlds"], proposals, support)
    assert all(choice is not None or not selection["supports"][arm] for arm, choice in selection["choices"].items()), "Ambiguous old decision must hold"
    compare(selection, lock["selection_statistics"])
    assert selection["choices"] == lock["choices"] and selection["choices"]["teacher"] == original["choice"]
    assert selection["means"] == original["means"] and lock["assessment_indices"] == list(range(support))
    assessment = root["assessment"]
    assert set(assessment) == {"candidate_indices", "world_seeds", "worlds", "timing"}
    assert assessment["candidate_indices"] == list(range(support)) and assessment["world_seeds"] == seeds
    assert len(assessment["worlds"]) == 128
    parent_verify._timing(assessment["timing"])
    cells = replay_cells(assessment["worlds"], selected["replay"]["assessment"])
    wall_replay, ordinary_wall, moves, reasons = 0., 0., 0, Counter()
    for wi, (seed, saved) in enumerate(zip(seeds, assessment["worlds"], strict=True)):
        c.budget_boundary(base)
        assert set(saved) == {"seed", "state_sha256", "outcomes"} and saved["seed"] == seed and len(saved["outcomes"]) == support
        world = world_audit.independent_world(state, seed)
        assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world))
        assert saved["state_sha256"] == old.digest(old.serialize_state_snapshot(world_audit.independent_world(alternate, seed)))
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        legal_world = old.SplendorEnv().legal_actions(world)
        for ci, (action, outcome) in enumerate(zip(actions, saved["outcomes"], strict=True)):
            assert action in legal_world
            repetition_audit._check_outcome(outcome, state.current_player, state.turn_index)
            if (wi, ci) in cells["all"]:
                replay_start = time.perf_counter()
                rebuilt = continuation_audit.replay_branch(world, action, state.current_player)
                compare(outcome, rebuilt)
                elapsed = time.perf_counter()-replay_start
                wall_replay += elapsed
                ordinary_wall += elapsed if wi in selected["ordinary"]["assessment"] else 0.
                moves += rebuilt["turns"]
                reasons[rebuilt["reason"]] += 1
    summary = independent_summary(assessment["worlds"], selection["choices"])
    compare(summary, root["summary"])
    parent_verify._timing(root["timing"])
    wall, recorded, replayed, unknown = time.perf_counter()-started, 128*support, len(cells["all"]), len(cells["unknown"])
    assert sum(reasons.values()) == replayed and reasons["repetition_cutoff"] == unknown
    result = {"manifest": binding, "stage": stage, "spec": spec, "passed": True,
              "root": c.artifact(path), "selection_artifact": c.artifact(seal_path), "sample_artifact": sample_artifact,
              "row": _row(spec, summary), "privacy": privacy, "support_assessment": support, "stress": selected["stress"],
              "ordinary_world_indices": selected["ordinary"], "replay_world_indices": selected["replay"],
              "unresolved_replay_cells": {"unknown": sorted(cells["unknown"]), "extra": sorted(cells["extra"])},
              "recorded_worlds": 128, "replayed_worlds": len({wi for wi, _ in cells["all"]}),
              "recorded_continuations": recorded, "replayed_continuations": replayed,
              "recorded_assessment_continuations": recorded, "replayed_assessment_continuations": replayed,
              "unresolved_continuations": unknown, "terminal_continuations": recorded-unknown,
              "unresolved_assessment_continuations": unknown, "additional_unknown_assessment_replays": len(cells["extra"]),
              "selection_certified_unknown_roots": int(bool(selection["unresolved_cells"])),
              "moves": moves, "reasons": dict(reasons), "full_inner_replay": replayed == recorded,
              "timing": {"wall": wall, "cpu": time.process_time()-cpu, "replay_wall": wall_replay,
                         "structural_wall": max(0., wall-wall_replay), "ordinary_replay_wall": ordinary_wall,
                         "full_extra_replay_wall": wall_replay-ordinary_wall, "assessment_replay_wall": wall_replay}}
    old.save_new(receipt_path, result)
    return c.artifact(receipt_path, index=index)


def summarize(items, specs, binding, base=OUTPUT):
    assert len(items) == len(specs) and specs
    stage = "main" if len(specs) == 336 else "probe" if len(specs) == 18 else "test"
    validate_specs(specs, stage)
    sample = old.read_json(Path(base)/stage/"audit_sample.json")
    compare(sample, sample_plan(specs, stage, binding))
    rows, coverage = [], Counter()
    for item, spec in zip(items, specs, strict=True):
        c.budget_boundary(base)
        receipt = c.verified(item)
        assert receipt["manifest"] == binding and receipt["stage"] == stage and receipt["passed"] and old.digest(receipt["spec"]) == old.digest(spec)
        for key in ("root", "selection_artifact", "sample_artifact"):
            c.verify_hash(receipt[key])
        root = c.verified(receipt["root"])
        selection = independent_selection(root["old_selection"]["worlds"], root["proposals"], len(root["candidates"]["keys"]))
        compare(selection, root["selection_lock"]["selection_statistics"])
        summary = independent_summary(root["assessment"]["worlds"], selection["choices"])
        compare(receipt["row"], _row(spec, summary))
        rows.append(receipt["row"])
        coverage["states"] += 1
        coverage["full_stress_roots"] += int(receipt["stress"])
        for key in ("recorded_continuations", "replayed_continuations", "recorded_worlds", "replayed_worlds",
                    "recorded_assessment_continuations", "replayed_assessment_continuations", "unresolved_assessment_continuations",
                    "unresolved_continuations", "terminal_continuations", "selection_certified_unknown_roots"):
            coverage[key] += receipt[key]
    return {"rows": rows, "coverage": dict(coverage), "full_inner_replay": coverage["recorded_continuations"] == coverage["replayed_continuations"]}


def independent_statistics(rows):
    """Independent histogram bootstrap with explicit integer endpoint arithmetic."""
    assert len(rows) == 336 and [r["index"] for r in rows] == list(range(336))
    pair_ids = sorted({r["pair"] for r in rows})
    assert len(pair_ids) == 84
    pair_values, pair_cells = {}, {}
    for pair in pair_ids:
        records = [r for r in rows if r["pair"] == pair]
        assert len(records) == 4 and {r["game_index"] for r in records} == {2*pair, 2*pair+1}
        assert {r["seat"] for r in records} == {0, 1}
        assert all((r["fit"], r["split"]) == (records[0]["fit"], records[0]["split"]) for r in records)
        for game in (2*pair, 2*pair+1):
            leg = [r for r in records if r["game_index"] == game]
            assert len(leg) == 2 and len({r["phase"] for r in leg}) == 2
        assert {r["phase"] for r in records if r["game_index"] == 2*pair} == {r["phase"] for r in records if r["game_index"] == 2*pair+1}
        pair_cells[pair] = (records[0]["split"], records[0]["fit"])
        values = np.zeros(len(CONTRASTS)*2, dtype=np.int64)
        for row in records:
            assert set(row["choices"]) == set(row["arm_scores"]) == set(row["arm_score_bounds"]) == set(ARMS)
            assert type(row["support"]) is int and 1 <= row["support"] <= 8
            assert row["outsider_available"] == (row["support"] > 3)
            for arm in ARMS:
                choice = row["choices"][arm]
                bound = row["arm_score_bounds"][arm]
                if choice is None:
                    assert row["support"] <= 3 and (arm.endswith("_outsider") or arm == "original_fourth")
                    assert bound is row["arm_scores"][arm] is None
                else:
                    assert type(choice) is int and 0 <= choice < row["support"]
                    assert len(bound) == 2 and 0 <= bound[0] <= bound[1] <= 1
                    assert all(type(x) in (int, float) and math.isfinite(x) and x*256 == int(x*256) for x in bound)
                    assert row["arm_scores"][arm] == (bound[0] if bound[0] == bound[1] else None)
            assert row["choices"]["anchor_outsider"] == row["choices"]["original_fourth"]
            assert row["choices"]["anchor_supported"] == row["choices"]["anchor"]
            assert set(row["delta_bounds"]) == set(NAMES)
            for j, (name, left, right) in enumerate(CONTRASTS):
                bounds = row["delta_bounds"][name]
                expected = [0., 0.] if row["choices"][left] == row["choices"][right] else [
                    row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][1],
                    row["arm_score_bounds"][left][1]-row["arm_score_bounds"][right][0]]
                assert bounds == expected and len(bounds) == 2 and -1 <= bounds[0] <= bounds[1] <= 1
                for endpoint, value in enumerate(bounds):
                    assert type(value) in (int, float) and math.isfinite(value) and value*256 == int(value*256)
                    values[2*j+endpoint] += int(value*256)
        pair_values[pair] = values
    strata = []
    for split in ("train", "validation"):
        for fit in FITS:
            members = [p for p in pair_ids if pair_cells[p] == (split, fit)]
            assert len(members) == 14
            strata.append(np.stack([pair_values[p] for p in members]))
    totals = [sum((a.sum(axis=0) for a in strata[i:i+3]), np.zeros(2*len(CONTRASTS), dtype=np.int64)) for i in (0, 3)]
    rng = np.random.default_rng(2213000000)
    boot = np.zeros((50000, 2, len(CONTRASTS)*2), dtype=np.int64)
    for start in range(0, 50000, 1000):
        for si, matrix in enumerate(strata):
            indices = rng.integers(0, 14, size=(1000, 14))
            frequencies = np.stack([np.bincount(draw, minlength=14) for draw in indices])
            boot[start:start+1000, si//3] += frequencies @ matrix
    denominator = 168*256
    def quantile(column, fraction):
        ordered = np.sort(column)
        position = (len(ordered)-1)*fraction
        lower, upper = math.floor(position), math.ceil(position)
        return (int(ordered[lower])+(int(ordered[upper])-int(ordered[lower]))*(position-lower))/denominator
    def entry(lo, hi, low_samples, high_samples):
        bounds = [int(lo)/denominator, int(hi)/denominator]
        interval = [quantile(low_samples, .025), quantile(high_samples, .975)]
        material = bool(int(lo)*200 >= denominator and interval[0] > 0)
        return {"mean": bounds[0] if lo == hi else None, "mean_bounds": bounds, "ci95": interval,
                "material_signal": material, "material_target_excluded": bool(interval[1] < .005),
                "positive_signal": bool(interval[0] > 0), "negative_signal": bool(interval[1] < 0),
                "classification": "material_signal" if material else "target_not_established"}
    groups = {}
    for si, split in enumerate(("train", "validation")):
        groups[split] = {"states": 168, "pairs": 42, "strata": 3, "contrasts": {
            name: entry(totals[si][2*j], totals[si][2*j+1], boot[:, si, 2*j], boot[:, si, 2*j+1])
            for j, (name, _, _) in enumerate(CONTRASTS)}}
        repeatability = {}
        for kind in ("teacher", "outsider"):
            counts = Counter()
            for row in rows:
                if row["split"] != split:
                    continue
                if kind == "outsider" and not row["outsider_available"]:
                    assert row["fresh_selection"]["outsider_agreement"] is None
                    counts["absent"] += 1
                    continue
                counts["available"] += 1
                choice = row["fresh_selection"][f"{kind}_certificate"]["choice"]
                locked = row["choices"]["teacher" if kind == "teacher" else "teacher_outsider"]
                expected = None if choice is None else choice == locked
                assert row["fresh_selection"][f"{kind}_agreement"] is expected
                counts["unresolved" if expected is None else "agreed" if expected else "disagreed"] += 1
            assert counts["available"]+counts["absent"] == 168
            checked = counts["agreed"]+counts["disagreed"]
            repeatability[kind] = {"states": 168, **{key: counts[key] for key in ("available", "absent", "agreed", "disagreed", "unresolved")},
                                  "agreement_rate_among_certified": counts["agreed"]/checked if checked else None,
                                  "diagnostic_only": True}
        groups[split]["repeatability"] = repeatability
    groups["validation_minus_train"] = {"contrasts": {
        name: entry(totals[1][2*j]-totals[0][2*j+1], totals[1][2*j+1]-totals[0][2*j],
                    boot[:, 1, 2*j]-boot[:, 0, 2*j+1], boot[:, 1, 2*j+1]-boot[:, 0, 2*j])
        for j, (name, _, _) in enumerate(CONTRASTS)}}
    return {"states": 336, "pairs": 84, "strata": 6, "bootstrap_samples": 50000, "bootstrap_seed": 2213000000,
            "resampling_unit": "whole_source_pair_four_roots", "groups": groups,
            "primary_contrasts": list(PRIMARY), "unknown_outcomes_imputed": False,
            "classification": "diagnostic_only", "promotion": False,
            "train_validation_difference_is_causal": False,
            "validation_previously_used_for_checkpoint_selection": True}

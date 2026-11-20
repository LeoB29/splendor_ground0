"""Outcome-blind sampling and authenticated membership in the actual teacher corpus.

Historical trajectory and tensor audits are inherited; this module never plays
a game, loads a policy, evaluates logits or computes a new scientific endpoint.
"""
from collections import Counter, defaultdict
from pathlib import Path

import torch

from . import common as c

DATA_SHA = "da7f4df92cfc5cd247cdc32bf4d6fec026bb23235de22ec745b9031ec9a11fc1"
ARENA_RESULT_SHA = "e554788ba3ff580651464a6007d31c953e1ac639dc54d90e6d247d406e1ceb2e"
ARENA_AUDIT_SHA = "e82b556f009fc6a70a01d101eeb98d2f09924414c2aea205269e1694b03fc591"
LEARNING_RESULT_SHA = "7aba7182dfb585ddb183fa09e1a0ebcb13b61b3bf3fc6ea20639f1b0b2f75989"
LEARNING_AUDIT_SHA = "34a7e1f0f23cccb0cedb486cef4544895dccce57bbad4c3aabd63232c49041e1"
SPLIT_TAG = "proposal-breadth-learning-128-source-pair-split-v1"
TRAIN_TAG = "teacher-signal-capture-128-train-pairs-v1"
MATCH_TAG = "teacher-signal-capture-128-match-pairs-v1"
PROBE_TAG = "teacher-signal-capture-128-probe-pairs-v1"
ROOT_TAG = "teacher-signal-capture-128-root-v1"


def phase(turn):
    c.require(type(turn) is int and turn >= 0, "Invalid source turn")
    return min(turn//20, 2)


def _rank(tag, *parts):
    return c.old.digest([tag, *parts])


def _plain(ref):
    return {key: ref[key] for key in ("path", "sha256")}


def original_splits(schedule):
    """Reproduce the already-frozen 53/14 whole-pair split independently."""
    result = {}
    c.require(len(schedule) == 402 and [s["index"] for s in schedule] == list(range(402)), "Incomplete source schedule")
    for fit in c.FITS:
        pairs = {row["pair"] for row in schedule if row["fit"] == fit}
        c.require(len(pairs) == 67, "Unexpected source fit size")
        ranked = sorted(pairs, key=lambda pair: (_rank(SPLIT_TAG, fit, pair), pair))
        result.update({pair: "validation" if pair in ranked[:14] else "train" for pair in pairs})
    c.require(len(result) == 201, "Source pair identities overlap fits")
    return result


def allocations(games, splits):
    """Consume only game identities and candidate-side turn availability."""
    c.require(len(games) == 402 and {g["spec"]["index"] for g in games} == set(range(402)), "Incomplete game population")
    ordered = sorted(games, key=lambda g: g["spec"]["index"])
    c.require(splits == original_splits([g["spec"] for g in ordered]), "Changed original training split")
    pairs = defaultdict(list)
    for game in games:
        spec = game["spec"]
        c.require(spec["fit"] in c.FITS and spec["pair"] == spec["index"]//2
                  and spec["candidate_seat"] == spec["index"] % 2, "Changed source seat/pair identity")
        pairs[spec["pair"]].append(game)
    for legs in pairs.values():
        c.require(len(legs) == 2 and {g["spec"]["candidate_seat"] for g in legs} == {0, 1}
                  and len({(g["spec"]["fit"], g["spec"]["environment_seed"], g["spec"]["search_stream"]) for g in legs}) == 1,
                  "Incomplete reversed-seat source pair")
    result = {"probe": [], "main": []}

    def append(stage, pair, fit, pair_rank, phases, omitted):
        for game in sorted(pairs[pair], key=lambda g: g["spec"]["index"]):
            gs = game["spec"]
            for ph in sorted(phases):
                eligible = [move for move in game["trace"] if move["actor"] == gs["candidate_seat"] and phase(move["turn"]) == ph]
                c.require(bool(eligible), "Fixed source phase unavailable; no pair substitution")
                move = min(eligible, key=lambda m: (_rank(ROOT_TAG, stage, gs["index"], ph, m["turn"]), m["turn"]))
                result[stage].append({"index": len(result[stage]), "pair": pair, "game_index": gs["index"],
                    "fit": fit, "seat": gs["candidate_seat"], "phase": ph, "split": splits[pair],
                    "source_arm": "candidate", "turn": move["turn"], "pair_rank": pair_rank,
                    "omitted_phase": omitted, "environment_seed": gs["environment_seed"],
                    "search_stream": gs["search_stream"], "snapshot_sha256": move["state_sha256"],
                    "root_sha256": move["root_sha256"], "source_rowid": f"breadth128:{gs['index']}:{move['turn']}"})

    for fi, fit in enumerate(c.FITS):
        fit_pairs = [pair for pair, legs in pairs.items() if legs[0]["spec"]["fit"] == fit]
        valid = [pair for pair in fit_pairs if splits[pair] == "validation"]
        train = sorted([pair for pair in fit_pairs if splits[pair] == "train"], key=lambda p: (_rank(TRAIN_TAG, fit, p), p))
        c.require(len(valid) == 14 and len(train) == 53, "Changed source split counts")
        chosen = train[:14]
        for split, selected in (("train", chosen), ("validation", valid)):
            matched = sorted(selected, key=lambda p: (_rank(MATCH_TAG, fit, split, p), p))
            for position, pair in enumerate(matched):
                omitted = (position+fi) % 3
                append("main", pair, fit, position, [p for p in range(3) if p != omitted], omitted)
        probe = sorted(train[14:], key=lambda p: (_rank(PROBE_TAG, fit, p), p))[:3]
        for position, pair in enumerate(probe):
            append("probe", pair, fit, position, [(position+fi) % 3], None)
    validate_schedule(result)
    return result


def validate_schedule(schedule):
    c.require(set(schedule) == {"probe", "main"}, "Unexpected source stages")
    pairsets = {}
    for stage, records in schedule.items():
        c.require([r["index"] for r in records] == list(range(18 if stage == "probe" else 336)), "Wrong panel indices")
        c.require(len({r["source_rowid"] for r in records}) == len(records), "Repeated source row")
        grouped = defaultdict(list)
        for row in records:
            c.require(row["fit"] in c.FITS and row["split"] in ("train", "validation")
                      and row["source_arm"] == "candidate" and row["seat"] in (0, 1)
                      and row["game_index"]//2 == row["pair"] and row["seat"] == row["game_index"] % 2
                      and phase(row["turn"]) == row["phase"]
                      and row["source_rowid"] == f"breadth128:{row['game_index']}:{row['turn']}", "Invalid source row identity")
            grouped[row["pair"]].append(row)
        c.require(len(grouped) == (9 if stage == "probe" else 84), "Wrong pair count")
        for pair, rows in grouped.items():
            head = rows[0]
            c.require(len(rows) == (2 if stage == "probe" else 4)
                      and {r["game_index"] for r in rows} == {2*pair, 2*pair+1}
                      and len({(r["fit"], r["split"], r["pair_rank"], r["omitted_phase"],
                                r["environment_seed"], r["search_stream"]) for r in rows}) == 1,
                      "Broken whole source pair")
            fi = list(c.FITS).index(head["fit"])
            if stage == "main":
                c.require(type(head["pair_rank"]) is int and 0 <= head["pair_rank"] < 14
                          and head["omitted_phase"] == (head["pair_rank"]+fi) % 3, "Changed main phase assignment")
                expected = {(seat, ph) for seat in (0, 1) for ph in range(3) if ph != head["omitted_phase"]}
            else:
                c.require(head["split"] == "train" and type(head["pair_rank"]) is int and 0 <= head["pair_rank"] < 3
                          and head["omitted_phase"] is None, "Probe must use remaining TRAIN pairs")
                expected = {(seat, (head["pair_rank"]+fi) % 3) for seat in (0, 1)}
            c.require({(r["seat"], r["phase"]) for r in rows} == expected, "Changed pair phase/seat coverage")
        if stage == "main":
            c.require(Counter((rows[0]["fit"], rows[0]["split"]) for rows in grouped.values()) ==
                      Counter({(fit, split): 14 for fit in c.FITS for split in ("train", "validation")}), "Main split balance")
            c.require(Counter((r["fit"], r["split"], r["pair_rank"]) for r in records) ==
                      Counter({(fit, split, rank): 4 for fit in c.FITS for split in ("train", "validation") for rank in range(14)}), "Main pair ranks")
            c.require(Counter(r["phase"] for r in records) == {0: 112, 1: 112, 2: 112}, "Main phase balance")
        else:
            c.require(Counter((r["fit"], r["phase"], r["seat"]) for r in records) ==
                      Counter({(fit, ph, seat): 1 for fit in c.FITS for ph in range(3) for seat in (0, 1)}), "Probe balance")
        pairsets[stage] = set(grouped)
    c.require(pairsets["probe"].isdisjoint(pairsets["main"]), "Probe/main source-pair leakage")
    return True


def closed_sources(data_ref=None):
    """Authenticate source closure, full population metadata and export ancestry."""
    refs = {}
    def add(ref, read=True):
        item = _plain(ref)
        c.require(item["path"] not in refs or refs[item["path"]] == item, "Conflicting source hashes")
        refs[item["path"]] = item
        return c.verified(ref) if read else c.verify_hash(ref)
    def close(folder, binding, result_sha, audit_sha):
        seal = add(c.artifact(folder/"freeze.json")); manifest = add(seal["manifest"])
        c.require(seal["owner_authorized"] is True and seal["binding"] == binding == c.old.digest(manifest), "Source binding changed")
        status = add(c.artifact(folder/"final_status.json"))
        c.require(status["manifest"] == binding and status["complete"] is True and status["all_prescribed_audits_pass"] is True,
                  "Source is not closed")
        c.require(status["result"]["sha256"] == result_sha and status["audit"]["sha256"] == audit_sha, "Source closure changed")
        result, audit = add(status["result"]), add(status["audit"])
        add(status["report"], False)
        c.require(result["manifest"] == audit["manifest"] == binding and audit["passed"] is True
                  and audit["result"] == status["result"], "Source final audit linkage")
        return result, audit
    _, learning_audit = close(c.LEARNING, c.LEARNING_BINDING, LEARNING_RESULT_SHA, LEARNING_AUDIT_SHA)
    learning = add(learning_audit["learning"])
    selected_data = learning["data"]
    c.require(selected_data["sha256"] == DATA_SHA and (data_ref is None or data_ref == selected_data), "Wrong trained corpus")
    data, data_audit = add(selected_data), None
    data_audit = add(data["audit"])
    c.require(data["manifest"] == data_audit["manifest"] == c.LEARNING_BINDING and data_audit["passed"] is True
              and data_audit["independent_breadth_tensor_target_reconstruction"] is True
              and data_audit["retained_tensors_targets_splits_unchanged"] is True
              and data_audit["source_evidence"] == data["source_evidence"], "Learning data audit changed")
    c.require(data["rows"] == 35886 and data["config"]["breadth_rows"] == 11950
              and len(data["exports"]) == len(data_audit["breadth_audits"]) == 402, "Training population changed")
    for ref in data["source_evidence"]:
        add(ref, False)
    for key in ("parent_data_lock", "parent_data_audit"):
        add(data[key], False)
    _, arena_audit = close(c.SOURCE, c.SOURCE_BINDING, ARENA_RESULT_SHA, ARENA_AUDIT_SHA)
    complete = add(arena_audit["stages"]["main"])
    c.require(complete["all_prescribed_audits_pass"] is True and complete["manifest"] == c.SOURCE_BINDING, "Arena main incomplete")
    game_lock, audit_lock = add(complete["game_lock"]), add(complete["audit_lock"])
    add(complete["sample"], False)
    c.require(game_lock["manifest"] == audit_lock["manifest"] == c.SOURCE_BINDING, "Arena locks differ")
    games, audits = game_lock["files"], audit_lock["files"]
    c.require([r["index"] for r in games] == [r["index"] for r in audits] == list(range(402)), "Missing source games/audits")
    schedule = add(c.artifact(c.SOURCE/"schedule.json"))["main"]
    split_map = original_splits(schedule)
    c.require({pair: data["splits"][f"breadth128:{pair}"] for pair in split_map} == split_map, "Original split seal changed")
    metadata, export_audits, row_count = [], [], 0
    for i, (game_ref, audit_ref, export_ref, export_audit_ref, gs) in enumerate(zip(
            games, audits, data["exports"], data_audit["breadth_audits"], schedule, strict=True)):
        game, audit, ea = add(game_ref), add(audit_ref), add(export_audit_ref)
        add(export_ref, False)
        c.require(export_ref["index"] == export_audit_ref["index"] == i and ea["export"] == export_ref,
                  "Export identity changed")
        c.require(ea["manifest"] == c.LEARNING_BINDING and ea["passed"] is True and ea["actual_game_replayed"] is True
                  and ea["every_row_hidden_input_invariance"] is True and ea["independent_targets"] is True,
                  "Missing inherited export verification")
        c.require(ea["source"] == {"game": game_ref, "audit": audit_ref, "spec": gs, "parent_binding": c.SOURCE_BINDING},
                  "Export source ancestry changed")
        c.require(game["spec"] == gs and game["manifest"] == audit["manifest"] == c.SOURCE_BINDING
                  and audit["passed"] is True and audit["game_artifact"] == game_ref
                  and "failure" not in game and game["reason"] in c.old.VALID_NATURAL_TERMINATIONS,
                  "Invalid inherited source game")
        c.require(len(game["trace"]) == len(audit["roots"]), "Incomplete game audit")
        projected = []
        source_roots = {ref["turn"]: ref for ref in ea["source_roots"]}
        c.require(len(source_roots) == export_ref["rows"], "Export row count changed")
        for move, checked in zip(game["trace"], audit["roots"], strict=True):
            c.require(move["turn"] == checked["turn"] and move["root_sha256"] == checked["sha256"], "Source trace/audit mismatch")
            if move["actor"] == gs["candidate_seat"]:
                c.require(checked["planner"] == "candidate" and checked["fit"] == gs["fit"]
                          and move["turn"] in source_roots and source_roots[move["turn"]]["sha256"] == move["root_sha256"],
                          "Candidate source row ancestry changed")
                projected.append({key: move[key] for key in ("actor", "turn", "state_sha256", "root_sha256")})
        c.require(len(projected) == len(source_roots), "Missing candidate export rows")
        row_count += len(projected)
        metadata.append({"spec": gs, "trace": projected})
        export_audits.append(ea)
    c.require(row_count == 11950, "Candidate corpus must include all 11,950 rows")
    return {"metadata": metadata, "splits": split_map, "games": games, "audits": audits,
            "exports": data["exports"], "export_audit_refs": data_audit["breadth_audits"], "export_audits": export_audits,
            "data": selected_data, "evidence": [refs[key] for key in sorted(refs)],
            "inherited_coverage": complete["coverage"], "source_rows": row_count}


def _export_row(spec):
    values = torch.load(c.verify_hash(spec["source_export"]), map_location="cpu", weights_only=True)
    c.require(len(values) == spec["source_export"]["rows"], "Changed export length")
    selected = [row for row in values if row["id"] == spec["source_rowid"]]
    c.require(len(selected) == 1, "Source row absent or duplicated in actual training export")
    return selected[0]


def _verify_export(spec, root, row):
    c.require(row["id"] == spec["source_rowid"] and row["pair"] == f"breadth128:{spec['pair']}"
              and row["split"] == spec["split"] and row["seat"] == spec["seat"] and row["phase"] == spec["phase"]
              and row["privacy"] == spec["privacy"] == root["privacy"]
              and row["teacher"] == root["choice"] and row["narrow_support"] == min(4, len(root["candidates"]["keys"])),
              "Source tensor row/target metadata differs")
    state = c.old.deserialize_state_snapshot(root["snapshot"])
    legal = c.old.SplendorEnv().legal_actions(state)
    group_keys = [repr(key) for key in c.old.actions_by_canonical_successor_key(state, legal)]
    encoded = [c.old.serialize_action(action) for action in legal]
    c.require(c.old.digest(encoded) == c.old.digest(root["proposal"]["legal_actions"]) and group_keys == root["proposal"]["group_keys"]
              and len(group_keys) == spec["canonical_support"] == row["group_count"], "Source legal/canonical support differs")
    memberships = [group_keys.index(repr(c.old.canonical_successor_action_key(state, action))) for action in legal]
    c.require(row["groups"].tolist() == memberships and len(row["features"]) == len(legal), "Export legal feature membership differs")
    expected = [[i for i, group in enumerate(memberships) if group_keys[group] == key] for key in root["candidates"]["keys"]]
    c.require([member.tolist() for member in row["members"]] == expected and all(expected), "Export offered action membership differs")
    c.require(c.baseline.privacy(state) == spec["privacy"], "Private reserve metadata changed")
    return state


def load_root(spec, base=None):
    """Authenticate source root, actual trajectory, tensor row and prior receipts."""
    if base is not None:
        c.budget_boundary(base)
    game, root, audit, ea = (c.verified(spec[key]) for key in
                             ("source_game", "source_root", "source_audit", "source_export_audit"))
    gs = game["spec"]
    c.require(gs["index"] == spec["game_index"] and gs["pair"] == spec["pair"] and gs["fit"] == spec["fit"]
              and gs["candidate_seat"] == spec["seat"] and gs["search_stream"] == spec["search_stream"]
              and gs["environment_seed"] == spec["environment_seed"], "Source game specification differs")
    c.require(game["manifest"] == root["manifest"] == audit["manifest"] == c.SOURCE_BINDING
              and "failure" not in game and "failure" not in root and audit["passed"] is True
              and audit["game_artifact"] == spec["source_game"] and game["reason"] in c.old.VALID_NATURAL_TERMINATIONS,
              "Source game/root audit linkage")
    c.require(spec["source_arm"] == root["planner"] == "candidate"
              and root["root_identity"] == ["main", spec["game_index"], spec["turn"]]
              and 0 <= spec["turn"] < len(game["trace"]), "Wrong source root identity")
    move = game["trace"][spec["turn"]]
    c.require(move["turn"] == spec["turn"] and move["actor"] == spec["seat"] and phase(move["turn"]) == spec["phase"]
              and move["state_sha256"] == spec["snapshot_sha256"] == c.old.digest(root["snapshot"])
              and move["root_sha256"] == spec["source_root"]["sha256"], "Source snapshot/trace identity differs")
    checked = [row for row in audit["roots"] if row["turn"] == spec["turn"]]
    c.require(len(checked) == 1 and checked[0]["sha256"] == spec["source_root"]["sha256"]
              and checked[0]["index"] == spec["game_index"] and checked[0]["seat"] == spec["seat"]
              and checked[0]["assigned_fit"] == spec["fit"] and checked[0]["planner"] == "candidate"
              and checked[0]["canonical_support"] == spec["canonical_support"], "Source root audit differs")
    c.require(ea["manifest"] == c.LEARNING_BINDING and ea["passed"] is True and ea["actual_game_replayed"] is True
              and ea["independent_targets"] is True and ea["every_row_hidden_input_invariance"] is True
              and ea["export"] == spec["source_export"]
              and ea["source"] == {"game": spec["source_game"], "audit": spec["source_audit"], "spec": gs, "parent_binding": c.SOURCE_BINDING},
              "Source export audit differs")
    roots = [ref for ref in ea["source_roots"] if ref["turn"] == spec["turn"]]
    c.require(len(roots) == 1 and _plain(roots[0]) == _plain(spec["source_root"]), "Tensor export refers to another root")
    c.require(root["snapshot"]["turn_index"] == spec["turn"] and root["snapshot"]["current_player"] == spec["seat"]
              and not root["snapshot"]["terminal"]
              and root["candidates"]["actions"][root["choice"]] == move["action"], "Source actual chosen move/state differs")
    _verify_export(spec, root, _export_row(spec))
    return root


def build(data_ref=None, base=None):
    """Build immutable schedule/evidence in memory; caller writes the new seal."""
    parent = closed_sources(data_ref)
    schedule = allocations(parent["metadata"], parent["splits"])
    evidence = {ref["path"]: ref for ref in parent["evidence"]}
    for records in schedule.values():
        for spec in records:
            i, turn = spec["game_index"], spec["turn"]
            ea = parent["export_audits"][i]
            roots = [ref for ref in ea["source_roots"] if ref["turn"] == turn]
            c.require(len(roots) == 1 and roots[0]["sha256"] == spec.pop("root_sha256"), "Selected source root changed")
            spec.update(source_root=roots[0], source_game=parent["games"][i], source_audit=parent["audits"][i],
                        source_export=parent["exports"][i], source_export_audit=parent["export_audit_refs"][i])
            audit = c.verified(spec["source_audit"])
            checked = [row for row in audit["roots"] if row["turn"] == turn]
            c.require(len(checked) == 1, "Missing selected structural audit")
            spec.update(canonical_support=checked[0]["canonical_support"], privacy=checked[0]["privacy"])
            load_root(spec, base)
            evidence[spec["source_root"]["path"]] = _plain(spec["source_root"])
    validate_schedule(schedule)
    return {"schedule": schedule, "source_evidence": [evidence[key] for key in sorted(evidence)],
            "inventory": {key: evidence[key]["sha256"] for key in sorted(evidence)},
            "source_binding": c.SOURCE_BINDING, "learning_binding": c.LEARNING_BINDING,
            "training_data": parent["data"], "source_rows": parent["source_rows"],
            "selection_uses_outcomes": False, "new_source_games": 0, "historical_trajectory_replays": 0,
            "all_validation_pairs_included": True, "probe_pairs_disjoint": True,
            "phase_rule": "matched_pair_rank_omit_(rank+fit_index)%3; identity_root_within_each_remaining_phase",
            "inherited_coverage": parent["inherited_coverage"]}

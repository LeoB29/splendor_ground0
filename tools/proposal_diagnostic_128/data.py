"""Identity-only source selection and descriptive saved common-prefix evidence.

No model inference or continuation simulation occurs here. Source trajectories
are independently replayed using the trusted engine. Saved-world contrasts are
explicitly descriptive and are never independent assessment of selected maxima.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features
from . import common as c

ROOT, OUTPUT, SOURCE = c.ROOT, c.OUTPUT, c.SOURCE
old, require, artifact, verified, verify_hash = c.old, c.require, c.artifact, c.verified, c.verify_hash
FITS = c.FITS
RULES = {"pair_tag": "proposal-diagnostic-128-source-pair-v1",
         "cell_tag": "proposal-diagnostic-128-source-cell-v1",
         "root_tag": "proposal-diagnostic-128-source-root-v1",
         "probe_game_tag": "proposal-diagnostic-128-probe-game-v1",
         "probe_cell_tag": "proposal-diagnostic-128-probe-cell-v1",
         "main_pairs_per_fit": 67, "main_cells": 6,
         "probe_games_per_fit": 4, "missing_cell": "hold_no_fallback"}
CELLS = [(role, phase) for role in ("treatment", "anchor") for phase in range(3)]


def _trace_hash(trace):
    hasher = hashlib.sha256()
    for move in trace:
        hasher.update(old.digest(move["action"]).encode())
    return hasher.hexdigest()


def _role(game, move):
    return "treatment" if move["actor"] == game["spec"]["candidate_seat"] else "anchor"


def _root_item(stage, game_index, move):
    path = SOURCE / stage / "roots" / f"game_{game_index:04d}" / f"turn_{move['turn']:03d}.json"
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": move["root_sha256"]}


def _inventory():
    """Read every source-game/audit receipt, but only selected roots later."""
    refs = []

    def read(path):
        item = artifact(path)
        refs.append(item)
        return verified(item)

    final = read(SOURCE / "final_audit.json")
    require(final["manifest"] == c.SOURCE_BINDING and final["passed"], "Unverified parent study")
    require(not final["full_inner_replay"], "Unexpected source audit contract")
    final_stage = verified(final["stages"]["stage_a"])
    refs.append(final["stages"]["stage_a"])
    require(final_stage["passed"] and final_stage["manifest"] == c.SOURCE_BINDING, "Source stage audit")
    schedule = read(SOURCE / "schedule.json")
    records = {}
    for stage, count in (("stage_a", 402), ("probe_a", 12)):
        complete_item = artifact(SOURCE / f"{stage}_complete.json")
        refs.append(complete_item)
        complete = verified(complete_item)
        if stage == "stage_a":
            require(final_stage["complete"] == complete_item, "Source final/completion linkage")
        require(complete["manifest"] == c.SOURCE_BINDING and complete["games"] == count, "Source completion identity")
        refs.extend([complete["game_lock"], complete["audit_lock"]])
        games, audits = verified(complete["game_lock"]), verified(complete["audit_lock"])
        require(games["manifest"] == audits["manifest"] == c.SOURCE_BINDING, "Source lock binding")
        require(len(games["files"]) == len(audits["files"]) == len(schedule[stage]) == count, "Source count")
        entries = []
        for index, (gi, ai) in enumerate(zip(games["files"], audits["files"], strict=True)):
            require(gi["index"] == ai["index"] == index, "Source game/audit order")
            game, audit = verified(gi), verified(ai)
            require(game["manifest"] == audit["manifest"] == c.SOURCE_BINDING, "Source game/audit binding")
            require(game["spec"] == schedule[stage][index], "Source game schedule")
            require(audit["passed"] and audit["game_artifact"] == gi, "Source receipt linkage")
            require("failure" not in game and game["reason"] in old.VALID_NATURAL_TERMINATIONS, "Invalid source game")
            require(len(game["trace"]) == len(audit["roots"]) == game["turns"], "Source trace/audit length")
            require(_trace_hash(game["trace"]) == game["trace_sha256"], "Source trace hash")
            for turn, (move, checked) in enumerate(zip(game["trace"], audit["roots"], strict=True)):
                require(move["turn"] == checked["turn"] == turn and move["root_sha256"] == checked["sha256"], "Source root linkage")
                role = _role(game, move)
                require(checked["model_id"] == f"{role}_{game['spec']['fit']}", "Source model role")
                require(checked["seat"] == move["actor"] and checked["assigned_fit"] == game["spec"]["fit"], "Source seat/fit")
            refs.extend([gi, ai])
            entries.append({"game": game, "audit": audit, "game_item": gi, "audit_item": ai})
        records[stage] = entries
    return records, refs


def _eligible(entries, role, phase):
    return [(entry, move) for entry in entries for move in entry["game"]["trace"]
            if _role(entry["game"], move) == role and min(move["turn"]//20, 2) == phase]


def selected_identities(records):
    """Production assignment; reads only identities/actors/turns, never outcomes."""
    result = {"main": [], "probe": []}
    for fit in FITS:
        entries = [entry for entry in records["stage_a"] if entry["game"]["spec"]["fit"] == fit]
        pairs = sorted({entry["game"]["spec"]["pair"] for entry in entries},
                       key=lambda pair: old.digest([RULES["pair_tag"], fit, pair]))
        require(len(pairs) == 67 and len(entries) == 134, "Main source fit quota")
        cells = sorted(CELLS, key=lambda cell: old.digest([RULES["cell_tag"], fit, list(cell)]))
        for rank, pair in enumerate(pairs):
            role, phase = cells[rank % 6]
            pair_entries = [e for e in entries if e["game"]["spec"]["pair"] == pair]
            require(len(pair_entries) == 2, "Source pair lacks both seats")
            choices = _eligible(pair_entries, role, phase)
            require(bool(choices), "Assigned main source cell missing; no fallback")
            entry, move = min(choices, key=lambda value: old.digest([RULES["root_tag"], "main", fit, pair,
                                value[0]["game"]["spec"]["index"], value[1]["turn"]]))
            result["main"].append((entry, move, role, phase))
    result["main"].sort(key=lambda x: x[0]["game"]["spec"]["pair"])
    cells = sorted(CELLS, key=lambda cell: old.digest([RULES["probe_cell_tag"], list(cell)]))
    for fit_index, fit in enumerate(FITS):
        entries = sorted((e for e in records["probe_a"] if e["game"]["spec"]["fit"] == fit),
                         key=lambda e: old.digest([RULES["probe_game_tag"], fit, e["game"]["spec"]["index"]]))
        require(len(entries) == 4, "Probe fit game quota")
        for rank, entry in enumerate(entries):
            role, phase = cells[(2*fit_index+rank) % 6]
            choices = _eligible([entry], role, phase)
            require(bool(choices), "Assigned probe source cell missing; no fallback")
            _, move = min(choices, key=lambda value: old.digest([RULES["root_tag"], "probe", fit,
                            entry["game"]["spec"]["pair"], entry["game"]["spec"]["index"], value[1]["turn"]]))
            result["probe"].append((entry, move, role, phase))
    result["probe"].sort(key=lambda x: x[0]["game"]["spec"]["index"])
    return result


def independent_identities(records):
    """Separately implement quotas, eligible-pool enumeration and hash minimum."""
    answer = {"main": [], "probe": []}
    possible = [(r, p) for r in ["treatment", "anchor"] for p in [0, 1, 2]]
    for fit in (161001, 161002, 161003):
        grouped = {}
        for entry in records["stage_a"]:
            spec = entry["game"]["spec"]
            if spec["fit"] == fit:
                grouped.setdefault(spec["pair"], []).append(entry)
        require(len(grouped) == 67 and all(len(v) == 2 for v in grouped.values()), "Independent source quota")
        order = sorted(grouped, key=lambda p: old.digest(["proposal-diagnostic-128-source-pair-v1", fit, p]))
        cell_order = sorted(possible, key=lambda x: old.digest(["proposal-diagnostic-128-source-cell-v1", fit, list(x)]))
        for ordinal, pair in enumerate(order):
            desired = cell_order[ordinal % len(possible)]
            options = []
            for entry in grouped[pair]:
                spec = entry["game"]["spec"]
                for move in entry["game"]["trace"]:
                    role = "treatment" if move["actor"] == spec["candidate_seat"] else "anchor"
                    phase = 0 if move["turn"] < 20 else 1 if move["turn"] < 40 else 2
                    if (role, phase) == desired:
                        digest = old.digest(["proposal-diagnostic-128-source-root-v1", "main", fit, pair, spec["index"], move["turn"]])
                        options.append((digest, spec["index"], move["turn"], entry, move))
            require(bool(options), "Independent main cell missing")
            option = sorted(options, key=lambda x: x[:3])[0]
            answer["main"].append((option[3], option[4], *desired))
    probe_cells = sorted(possible, key=lambda x: old.digest(["proposal-diagnostic-128-probe-cell-v1", list(x)]))
    for offset, fit in enumerate((161001, 161002, 161003)):
        games = {e["game"]["spec"]["index"]: e for e in records["probe_a"] if e["game"]["spec"]["fit"] == fit}
        require(len(games) == 4, "Independent probe quota")
        ordering = sorted(games, key=lambda i: old.digest(["proposal-diagnostic-128-probe-game-v1", fit, i]))
        for ordinal, index in enumerate(ordering):
            desired = probe_cells[(offset*2+ordinal) % 6]
            entry = games[index]
            spec = entry["game"]["spec"]
            options = []
            for move in entry["game"]["trace"]:
                role = "treatment" if spec["candidate_seat"] == move["actor"] else "anchor"
                phase = 0 if move["turn"] < 20 else 1 if move["turn"] < 40 else 2
                if (role, phase) == desired:
                    options.append((old.digest(["proposal-diagnostic-128-source-root-v1", "probe", fit, spec["pair"], index, move["turn"]]), move))
            require(bool(options), "Independent probe cell missing")
            move = sorted(options, key=lambda x: (x[0], x[1]["turn"]))[0][1]
            answer["probe"].append((entry, move, *desired))
    for stage, key in (("main", "pair"), ("probe", "index")):
        answer[stage].sort(key=lambda row: row[0]["game"]["spec"][key])
    return answer


def _spec(index, diagnostic_stage, picked):
    entry, move, role, phase = picked
    game_spec = entry["game"]["spec"]
    source_stage = "stage_a" if diagnostic_stage == "main" else "probe_a"
    root_item = _root_item(source_stage, game_spec["index"], move)
    root = verified(root_item)
    checked = entry["audit"]["roots"][move["turn"]]
    require(root["manifest"] == c.SOURCE_BINDING and "failure" not in root, "Selected source root binding")
    require(old.digest(root["snapshot"]) == move["state_sha256"], "Selected source snapshot hash")
    require(root["proposal"]["model_id"] == f"{role}_{game_spec['fit']}", "Selected source model role")
    require(root["privacy"] == checked["privacy"], "Selected source privacy")
    require(root["candidates"]["actions"][root["choice"]] == move["action"], "Selected source choice link")
    return {"index": index, "pair": game_spec["pair"], "fit": game_spec["fit"], "role": role, "phase": phase,
            "game_index": game_spec["index"], "turn": move["turn"], "source_stage": source_stage,
            "source_game": entry["game_item"], "source_root": root_item, "source_audit": entry["audit_item"],
            "search_stream": game_spec["search_stream"], "snapshot_sha256": move["state_sha256"],
            "privacy": root["privacy"], "canonical_support": root["candidates"]["canonical_support"]}


def _source_files(refs, specs):
    by_path = {}
    for item in refs + [spec["source_root"] for stage in ("main", "probe") for spec in specs[stage]]:
        basic = {"path": item["path"], "sha256": item["sha256"]}
        require(basic["path"] not in by_path or by_path[basic["path"]] == basic, "Conflicting source hash")
        by_path[basic["path"]] = basic
    return [by_path[key] for key in sorted(by_path)]


def prepare_selection():
    records, refs = _inventory()
    identities = selected_identities(records)
    specs = {stage: [_spec(i, stage, row) for i, row in enumerate(identities[stage])] for stage in ("main", "probe")}
    return {**specs, "source_files": _source_files(refs, specs), "source_binding": c.SOURCE_BINDING, "rules": RULES}


def validate_selection(selection):
    require(selection["source_binding"] == c.SOURCE_BINDING and selection["rules"] == RULES, "Selection binding/rules")
    records, refs = _inventory()
    identities = independent_identities(records)
    expected = {stage: [_spec(i, stage, row) for i, row in enumerate(identities[stage])] for stage in ("main", "probe")}
    require(selection["main"] == expected["main"] and selection["probe"] == expected["probe"], "Independent selection differs")
    require(selection["source_files"] == _source_files(refs, expected), "Selection source inventory differs")
    require(len(expected["main"]) == 201 and len(expected["probe"]) == 12, "Selection counts")
    require(len({spec["pair"] for spec in expected["main"]}) == 201, "Main must have one state per pair")
    return True


def replay_selected_state(spec):
    """Independent actual source trajectory through the selected pre-decision root."""
    game, audit, root = (verified(spec[key]) for key in ("source_game", "source_audit", "source_root"))
    require(game["manifest"] == audit["manifest"] == root["manifest"] == c.SOURCE_BINDING, "Selected parent binding")
    require(audit["passed"] and audit["game_artifact"] == spec["source_game"], "Selected source receipt")
    require(game["spec"]["index"] == spec["game_index"] and game["spec"]["pair"] == spec["pair"], "Selected source identity")
    require(game["spec"]["fit"] == spec["fit"] and game["spec"]["search_stream"] == spec["search_stream"], "Selected fit/stream")
    require(_trace_hash(game["trace"]) == game["trace_sha256"], "Selected source trace hash")
    require(len(game["trace"]) == len(audit["roots"]) and 0 <= spec["turn"] < len(game["trace"]), "Selected turn")
    env = old.SplendorEnv(seed=game["spec"]["environment_seed"])
    state = env.initial_state()
    for move, checked in zip(game["trace"], audit["roots"], strict=True):
        require(move["turn"] == state.turn_index == checked["turn"] and move["actor"] == state.current_player, "Actual source sequence")
        require(move["state_sha256"] == old.digest(old.serialize_state_snapshot(state)), "Actual source state")
        require(move["root_sha256"] == checked["sha256"], "Actual source audit linkage")
        if state.turn_index == spec["turn"]:
            require(move["root_sha256"] == spec["source_root"]["sha256"], "Selected root hash")
            require(_role(game, move) == spec["role"] and min(state.turn_index//20, 2) == spec["phase"], "Selected cell")
            require(root["proposal"]["model_id"] == checked["model_id"] == f"{spec['role']}_{spec['fit']}", "Selected model")
            require(old.digest(root["snapshot"]) == move["state_sha256"] == spec["snapshot_sha256"], "Selected snapshot")
            require(root["privacy"] == checked["privacy"] == spec["privacy"], "Selected privacy")
            require(root["candidates"]["canonical_support"] == spec["canonical_support"], "Selected support")
            require(root["candidates"]["actions"][root["choice"]] == move["action"], "Selected action")
            return old.serialize_state_snapshot(state)
        action = old.decode_action(move["action"])
        require(action in env.legal_actions(state), "Illegal actual source move")
        state = env.step(state, action)
    raise ValueError("Selected source turn not reached")


def describe_matched_roots(treatment, anchor, fit):
    """Validate exact matching and summarize fixed-proposal saved-world contrasts."""
    require(old.digest(treatment["snapshot"]) == old.digest(anchor["snapshot"]), "Matched snapshots differ")
    require(treatment["manifest"] == anchor["manifest"] == c.SOURCE_BINDING, "Matched parent binding")
    require(treatment["schedule"] == anchor["schedule"], "Matched world schedule differs")
    states = [old.deserialize_state_snapshot(r["snapshot"]) for r in (treatment, anchor)]
    inputs = []
    for root, state, role in zip((treatment, anchor), states, ("treatment", "anchor"), strict=True):
        legal = old.SplendorEnv().legal_actions(state)
        observation = encode_public_observation_tensor(state, state.current_player).vector
        features = [value.vector for value in encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)]
        serialized = [old.serialize_action(action) for action in legal]
        proposal = root["proposal"]
        require(proposal["model_id"] == f"{role}_{fit}" and proposal["fit"] == fit, "Matched proposal model")
        require(serialized == proposal["legal_actions"], "Matched legal actions differ")
        require(root["candidates"] == proposal["candidates"] and root["schedule"] == proposal["schedule"], "Matched proposal linkage")
        groups = old.actions_by_canonical_successor_key(state, legal)
        keys = [repr(key) for key in groups]
        require(keys == proposal["group_keys"], "Matched canonical groups")
        logits = proposal["learned_logits"]
        require(len(logits) == len(legal) and all(math.isfinite(v) for v in logits), "Saved logits invalid")
        action_keys = [repr(old.canonical_successor_action_key(state, action)) for action in legal]
        values = [max(logits[i] for i, action_key in enumerate(action_keys) if action_key == key) for key in keys]
        require(values == proposal["learned_group_scores"], "Saved group maxima differ")
        reference = proposal["reference"]["keys"]
        expected = list(reference)
        if len(reference) == 4:
            available = [i for i, key in enumerate(keys) if key not in reference[:3]]
            expected = reference[:3] + [keys[max(available, key=lambda i: values[i])]]
        require(root["candidates"]["keys"] == expected, "Saved fourth proposal is not eligible maximum")
        require(len(root["worlds"]) == len(root["schedule"]["world_seeds"]) == 128, "Saved world budget")
        size = len(expected)
        require(all(len(w["outcomes"]) == size for w in root["worlds"]), "Saved outcome support")
        require([w["seed"] for w in root["worlds"]] == root["schedule"]["world_seeds"], "Saved world seed linkage")
        require(all(o["reason"] in old.VALID_NATURAL_TERMINATIONS and o["wdl"] in (-1., 0., 1.) for w in root["worlds"] for o in w["outcomes"]), "Saved outcome value")
        means = [sum(w["outcomes"][i]["wdl"] for w in root["worlds"])/128 for i in range(size)]
        require(means == root["means"] and means.index(max(means)) == root["choice"], "Saved choice arithmetic")
        inputs.append(old.digest([observation, features, serialized]))
    require(inputs[0] == inputs[1], "Lawful matched inputs differ")
    require(treatment["proposal"]["reference"] == anchor["proposal"]["reference"], "Matched original candidate set differs")
    tc, ac = treatment["candidates"], anchor["candidates"]
    require(tc["keys"][:3] == ac["keys"][:3] and tc["actions"][:3] == ac["actions"][:3], "Matched first three differ")
    common = set(tc["keys"]) & set(ac["keys"])
    for key in common:
        require(tc["actions"][tc["keys"].index(key)] == ac["actions"][ac["keys"].index(key)], "Matched representative differs")
    for tw, aw in zip(treatment["worlds"], anchor["worlds"], strict=True):
        require(tw["seed"] == aw["seed"] and tw["state_sha256"] == aw["state_sha256"], "Matched world identity differs")
        for key in common:
            require(tw["outcomes"][tc["keys"].index(key)] == aw["outcomes"][ac["keys"].index(key)], "Common candidate outcomes differ")
    fourth = len(tc["keys"]) == len(ac["keys"]) == 4
    require(len(tc["keys"]) == len(ac["keys"]), "Matched supports differ")
    return {"turn": states[0].turn_index, "fit": fit, "phase": min(states[0].turn_index//20, 2),
            "snapshot_sha256": old.digest(treatment["snapshot"]), "lawful_input_sha256": inputs[0],
            "privacy": treatment["privacy"], "has_fourth": fourth,
            "proposal_changed": fourth and tc["keys"][3] != ac["keys"][3],
            "decision_changed": tc["keys"][treatment["choice"]] != ac["keys"][anchor["choice"]],
            "treatment_fourth_key": tc["keys"][3] if fourth else None,
            "anchor_fourth_key": ac["keys"][3] if fourth else None,
            "fixed_fourth_wdl_difference": treatment["means"][3]-anchor["means"][3] if fourth else None,
            "treatment_selected_fourth": fourth and treatment["choice"] == 3,
            "anchor_selected_fourth": fourth and anchor["choice"] == 3,
            "independent_assessment": False}


def _summary(rows):
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair"], []).append(row)
    values = [row["fixed_fourth_wdl_difference"] for row in rows if row["has_fourth"]]
    pair_values = [sum(v)/len(v) for group in by_pair.values()
                   if (v := [row["fixed_fourth_wdl_difference"] for row in group if row["has_fourth"]])]
    return {"roots": len(rows), "source_pairs": len(by_pair),
            "fourth_available": len(values), "proposal_changed": sum(r["proposal_changed"] for r in rows),
            "decision_changed": sum(r["decision_changed"] for r in rows),
            "fixed_fourth_wdl_mean_by_root": sum(values)/len(values) if values else None,
            "fixed_fourth_wdl_mean_by_pair": sum(pair_values)/len(pair_values) if pair_values else None,
            "treatment_selected_fourth": sum(r["treatment_selected_fourth"] for r in rows),
            "anchor_selected_fourth": sum(r["anchor_selected_fourth"] for r in rows),
            "fit_roots": dict(Counter(str(r["fit"]) for r in rows)),
            "phase_roots": dict(Counter(str(r["phase"]) for r in rows))}


def cached_analysis(selection, binding, base=OUTPUT):
    """Every opening and full common prefix, INCLUDING the first divergent root."""
    base = Path(base)
    require(not (base / "cached_analysis.json").exists(), "One-shot cached analysis")
    records, refs = _inventory()
    require(selection["source_binding"] == c.SOURCE_BINDING, "Cached source binding")
    bound = {item["path"]: item["sha256"] for item in selection["source_files"]}
    require(all(bound.get(item["path"]) == item["sha256"] for item in refs), "Cached source not in selection seal")
    pairs = []
    for pair in range(201):
        c.budget_boundary(base)
        entries = [entry for entry in records["stage_a"] if entry["game"]["spec"]["pair"] == pair]
        require(len(entries) == 2, "Cached pair missing")
        entries.sort(key=lambda e: e["game"]["spec"]["candidate_seat"])
        g0, g1 = [e["game"] for e in entries]
        fit = g0["spec"]["fit"]
        require(g1["spec"]["fit"] == fit and g0["spec"]["environment_seed"] == g1["spec"]["environment_seed"], "Cached pair source identity")
        require(g0["spec"]["search_stream"] == g1["spec"]["search_stream"], "Cached pair stream")
        env = old.SplendorEnv(seed=g0["spec"]["environment_seed"])
        state = env.initial_state()
        roots, diverged = [], False
        for m0, m1 in zip(g0["trace"], g1["trace"]):
            require(m0["turn"] == m1["turn"] == state.turn_index and m0["actor"] == m1["actor"] == state.current_player, "Cached actual prefix sequence")
            require(m0["state_sha256"] == m1["state_sha256"] == old.digest(old.serialize_state_snapshot(state)), "Cached common actual state")
            root_items = [_root_item("stage_a", entry["game"]["spec"]["index"], move) for entry, move in zip(entries, (m0, m1), strict=True)]
            saved = [verified(item) for item in root_items]
            for root, move in zip(saved, (m0, m1), strict=True):
                require(old.digest(root["snapshot"]) == move["state_sha256"], "Cached root snapshot link")
                require(root["candidates"]["actions"][root["choice"]] == move["action"], "Cached root action link")
            treatment_index = 0 if _role(g0, m0) == "treatment" else 1
            report = describe_matched_roots(saved[treatment_index], saved[1-treatment_index], fit)
            report.update(pair=pair, source_roots=root_items)
            roots.append(report)
            actions_differ = m0["action"] != m1["action"]
            require(actions_differ == report["decision_changed"], "Canonical and actual divergence differ")
            if actions_differ:
                diverged = True
                break
            action = old.decode_action(m0["action"])
            require(action in env.legal_actions(state), "Illegal cached actual prefix action")
            state = env.step(state, action)
        require(bool(roots) and roots[0]["turn"] == 0, "Cached opening missing")
        if not diverged:
            require(len(g0["trace"]) == len(g1["trace"]) == len(roots), "Unexplained source prefix length")
            require(old.digest(g0["final_state"]) == old.digest(g1["final_state"]) == old.digest(old.serialize_state_snapshot(state)), "Identical decisions produced different final state")
        pairs.append({"pair": pair, "fit": fit, "roots": roots, "first_divergent_turn": roots[-1]["turn"] if diverged else None,
                      "common_prefix_includes_first_divergence": True})
        if (pair+1) % 20 == 0:
            print({"cached_source_pairs": pair+1, "total": 201}, flush=True)
    opening = [row["roots"][0] for row in pairs]
    later = [root for row in pairs for root in row["roots"][1:]]
    value = {"manifest": binding, "source_binding": c.SOURCE_BINDING, "source_files": refs,
             "selection_sha256": old.digest(selection), "pairs": pairs,
             "opening": _summary(opening), "later_common_prefix": _summary(later),
             "all_common_prefix": _summary(opening+later), "independent_assessment": False,
             "new_model_inference": False, "new_continuation_simulations": 0,
             "caveats": ["Same saved worlds are coupled descriptive evidence, not independent assessment.",
                         "Only fixed neural fourth values are contrasted; rollout-selected maxima are not independently assessed here.",
                         "Later common prefixes favor early/agreed trajectories; opening and later summaries are separate.",
                         "Whole-pair accounting; no post-hoc confidence interval or strength claim."]}
    old.save_new(base / "cached_analysis.json", value)
    return artifact(base / "cached_analysis.json")

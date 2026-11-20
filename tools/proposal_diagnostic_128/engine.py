"""Same-state proposal comparisons with sealed selection before fresh assessment."""
from __future__ import annotations

from pathlib import Path
import random
import time
import traceback

from tools.proposal_diagnostic_128 import common as c
from tools.proposal_iteration_128 import arena as parent

ROOT, OUTPUT, SOURCE = c.ROOT, c.OUTPUT, c.SOURCE
old, baseline, CONFIG, require = c.old, c.baseline, c.CONFIG, c.require
ARMS = ("anchor", "treatment", "replay", "random", "first_three", "union")
PROPOSERS = ("anchor", "treatment", "replay", "random")


def initialize_models():
    require(old.read_json(SOURCE / "models_lock.json")["manifest"] == c.SOURCE_BINDING,
            "Wrong completed proposal model lineage")
    parent.initialize_models(base=SOURCE)


def check_stage(stage, index, base=OUTPUT):
    require(stage in ("probe", "main", "test"), "Unknown diagnostic stage")
    require(type(index) is int and index >= 0, "Invalid diagnostic index")
    if stage == "test":
        resolved = Path(base).resolve()
        require(not resolved.is_relative_to(OUTPUT.resolve()), "Synthetic source cannot target production")
        require(index < 12, "Fixture index outside isolated schedule")
    else:
        require(index < CONFIG["states"][stage], "Diagnostic outside frozen schedule")


def root_path(stage, index, base=OUTPUT):
    check_stage(stage, index, base)
    return Path(base) / stage / "roots" / f"state_{index:04d}.json"


def selection_path(stage, index, base=OUTPUT):
    check_stage(stage, index, base)
    return Path(base) / stage / "selections" / f"state_{index:04d}_selection.json"


def world_seeds(stage, index, block):
    """Nonoverlapping integer ranges; no outcome, checkpoint or source hash enters."""
    require(stage in ("probe", "main", "test") and block in ("selection", "assessment"), "World namespace")
    limit = 12 if stage == "test" else CONFIG["states"][stage]
    require(type(index) is int and 0 <= index < limit, "World index outside schedule")
    offset = {"probe": 0, "main": 1000000, "test": 2000000}[stage]
    offset += index * 512 + (0 if block == "selection" else 256)
    return [CONFIG["world_seed_base"] + offset + wi for wi in range(128)]


def assemble_union(reference, proposals, group_keys, representatives, identity):
    """Order by common first three then anchor/treatment/replay/random challengers."""
    stage, index = identity
    require(stage in ("probe", "main", "test") and type(index) is int and index >= 0, "Proposal identity")
    require(set(proposals) == {"anchor", "treatment", "replay"}, "Missing model proposal")
    require(len(set(group_keys)) == len(group_keys) and set(representatives) == set(group_keys), "Canonical support")
    require(len(reference["keys"]) == min(4, len(group_keys)), "Reference support")
    first = list(reference["keys"][:3])
    require(len(first) == len(set(first)), "Duplicate retained candidate")
    by_arm = {}
    for arm in PROPOSERS[:3]:
        value = proposals[arm]["candidates"]
        require(len(value["keys"]) == min(4, len(group_keys)) and value["keys"][:3] == first,
                "Model changed retained candidate support")
        require(len(set(value["keys"])) == len(value["keys"]), "Duplicate model proposal")
        require(all(representatives[key] == action for key, action in zip(value["keys"], value["actions"], strict=True)),
                "Inconsistent concrete representative")
        by_arm[arm] = list(value["keys"])
    remaining = [key for key in group_keys if key not in first]
    random_key = min(remaining, key=lambda key: (old.digest([CONFIG["random_tag"], stage, index, key]), key)) if remaining else None
    by_arm["random"] = first + ([random_key] if random_key is not None else [])
    union = list(first)
    for arm in PROPOSERS:
        for key in by_arm[arm]:
            if key not in union:
                union.append(key)
    require(1 <= len(union) <= 7, "Union exceeds seven-candidate ceiling")
    by_arm["first_three"], by_arm["union"] = first, list(union)
    candidates = {arm: {"keys": keys, "actions": [representatives[key] for key in keys],
                         "indices": [union.index(key) for key in keys]} for arm, keys in by_arm.items()}
    proposed = {arm: union.index(by_arm[arm][3]) if len(by_arm[arm]) == 4 else None for arm in PROPOSERS}
    return {"candidates": candidates, "union": {"keys": union, "actions": [representatives[key] for key in union]},
            "proposed_indices": proposed}


def proposal_union(state, stream, fit, identity):
    require(fit in c.FITS, "Unknown assigned fit")
    proposals = {arm: parent.learned_proposal(state, stream, f"{arm}_{fit}") for arm in PROPOSERS[:3]}
    reference = proposals["anchor"]["reference"]
    require(all(value["reference"] == reference for value in proposals.values()), "Unmatched raw reference")
    legal = old.SplendorEnv().legal_actions(state)
    groups = old.actions_by_canonical_successor_key(state, legal)
    keys = [repr(key) for key in groups]
    reps = {repr(key): old.serialize_action(action) for key, action in groups.items()}
    reps.update(zip(reference["keys"], reference["actions"], strict=True))
    return {"proposals": proposals, **assemble_union(reference, proposals, keys, reps, identity)}


def means(worlds, support):
    require(len(worlds) == 128 and type(support) is int and 1 <= support <= 7, "Root budget")
    require(all(len(row["outcomes"]) == support for row in worlds), "Missing or extra candidate outcome")
    values = [[row["outcomes"][j]["wdl"] for row in worlds] for j in range(support)]
    require(all(type(v) in (int, float) and v in (-1., 0., 1.) for column in values for v in column), "Invalid terminal value")
    return [sum(column) / 128 for column in values]


def choose(worlds, candidates, support):
    require(set(candidates) == set(ARMS), "Selection arm completeness")
    values = means(worlds, support)
    choices = {}
    for arm in ARMS:
        indices = candidates[arm]["indices"]
        require(indices and len(set(indices)) == len(indices) and all(type(i) is int and 0 <= i < support for i in indices),
                "Invalid offered selection support")
        # Keep each arm's original candidate order for exact ties.
        choices[arm] = max(indices, key=lambda j: values[j])
    return values, choices


def summarize(worlds, choices, proposed_indices, support):
    require(set(choices) == set(ARMS) and set(proposed_indices) == set(PROPOSERS), "Summary arm completeness")
    values = means(worlds, support)
    require(all(type(i) is int and 0 <= i < support for i in choices.values()), "Choice outside union")
    require(all(i is None or type(i) is int and 0 <= i < support for i in proposed_indices.values()), "Proposal outside union")
    return {"assessment_means_wdl": values,
            "arm_wdl": {arm: values[index] for arm, index in choices.items()},
            "arm_scores": {arm: (1 + values[index]) / 2 for arm, index in choices.items()},
            "fixed_proposal_wdl": {arm: None if index is None else values[index]
                                    for arm, index in proposed_indices.items()},
            "fixed_proposal_scores": {arm: None if index is None else (1 + values[index]) / 2
                                      for arm, index in proposed_indices.items()},
            "choices": choices, "proposed_indices": proposed_indices}


def statistics_row(record):
    """Recompute diagnostic endpoints from raw blocks, never cached score fields."""
    require("failure" not in record, "Failed diagnostic cannot enter statistics")
    support = len(record["union"]["keys"])
    values, choices = choose(record["selection"]["worlds"], record["candidates"], support)
    require(values == record["selection_lock"]["means"] and choices == record["selection_lock"]["choices"],
            "Statistics differ from sealed selection")
    result = summarize(record["assessment"]["worlds"], choices, record["proposed_indices"], support)
    require(result == record["summary"], "Stored summary differs from raw assessment")
    return {**{key: record["spec"][key] for key in ("index", "pair", "fit", "role", "phase")},
            **{key: result[key] for key in ("arm_scores", "fixed_proposal_scores", "choices", "proposed_indices")}}


def proposal_identity(record):
    """Exclude timings; bind every scientific input available before selection."""
    return {"spec": record["spec"], "snapshot": record["snapshot"], "privacy": record["privacy"],
            "proposals": {arm: {key: value for key, value in proposal.items() if key != "timing"}
                          for arm, proposal in record["proposals"].items()},
            "candidates": record["candidates"], "union": record["union"],
            "proposed_indices": record["proposed_indices"]}


def run_block(state, actions, seeds, block, base):
    started, cpu = time.perf_counter(), time.process_time()
    c.budget_boundary(base)
    for seed in seeds:
        c.budget_boundary(base)
        world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
        saved = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": []}
        block["worlds"].append(saved)
        legal = old.SplendorEnv().legal_actions(world)
        for action in actions:
            require(action in legal, "Illegal forced diagnostic candidate")
            result = old.trajectory(world, action)
            result["wdl"] = old.outcome_value(result["winner"], state.current_player) if result["reason"] in old.VALID_NATURAL_TERMINATIONS else None
            saved["outcomes"].append(result)
            require(result["wdl"] is not None, "Invalid diagnostic continuation ending")
    block["timing"] = {"wall": time.perf_counter()-started, "cpu": time.process_time()-cpu}


def collect(stage, spec, binding, base=OUTPUT):
    index = spec["index"]
    path, seal_path = root_path(stage, index, base), selection_path(stage, index, base)
    require(not path.exists() and not seal_path.exists(), "One-shot diagnostic; preserve prior root and selection")
    started, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": binding, "stage": stage, "spec": spec, "time": old.now()}
    try:
        c.budget_boundary(base)
        if "fixture_snapshot" in spec:
            require(stage == "test", "Synthetic snapshot override outside isolated test")
            source = {"snapshot": spec["fixture_snapshot"]}
        else:
            source = c.verified(spec["source_root"])
        require(old.digest(source["snapshot"]) == spec["snapshot_sha256"], "Source snapshot mismatch")
        if stage != "test":
            require(source["manifest"] == c.SOURCE_BINDING and "failure" not in source, "Invalid closed source")
        state = old.deserialize_state_snapshot(source["snapshot"])
        require(not state.terminal and state.turn_index == spec["turn"], "Invalid source decision state")
        record.update(snapshot=source["snapshot"], privacy=baseline.privacy(state))
        record.update(proposal_union(state, spec["search_stream"], spec["fit"], (stage, index)))
        actions = [old.decode_action(value) for value in record["union"]["actions"]]
        record["selection"] = {"world_seeds": world_seeds(stage, index, "selection"), "worlds": []}
        run_block(state, actions, record["selection"]["world_seeds"], record["selection"], base)
        values, choices = choose(record["selection"]["worlds"], record["candidates"], len(actions))
        selection_lock = {"manifest": binding, "stage": stage, "index": index,
                          "identity_digest": old.digest(proposal_identity(record)),
                          "selection_sha256": old.digest(record["selection"]),
                          "means": values, "value_scale": "terminal_wdl", "choices": choices,
                          "time": old.now(), "created_at_ns": time.time_ns()}
        record["selection_lock"] = selection_lock
        # The complete selection evidence is persisted before any assessment is made.
        old.save_new(seal_path, {"manifest": binding, "stage": stage, "index": index,
                                 "identity": proposal_identity(record), "selection": record["selection"],
                                 "selection_lock": selection_lock})
        record["selection_artifact"] = c.artifact(seal_path)
        record["assessment_started_at_ns"] = time.time_ns()
        record["assessment"] = {"world_seeds": world_seeds(stage, index, "assessment"), "worlds": []}
        require(set(record["selection"]["world_seeds"]).isdisjoint(record["assessment"]["world_seeds"]), "Assessment reuses selection worlds")
        run_block(state, actions, record["assessment"]["world_seeds"], record["assessment"], base)
        record["summary"] = summarize(record["assessment"]["worlds"], choices, record["proposed_indices"], len(actions))
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter()-started, "cpu": time.process_time()-cpu}
        old.save_new(path, record)
    return c.artifact(path, index=index, continuations=256*len(actions))

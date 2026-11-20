"""Closed historical labels and an outcome-blind independent-game intervention.

Only the 720 common 128-world selection matrices become new TRAIN supervision.
Old assessment/extension arrays never enter an exported fitting row. Original
11,950 tensors and split audits are inherited byte-for-byte. New public inputs,
canonical memberships and target arithmetic are checked without playing games.
"""
from collections import Counter, defaultdict
from pathlib import Path

import torch

from scripts.analysis import verify_fast_selector_transfer_157 as independent
from splendor_ai.alphazero.strong_policy import encode_position
from splendor_ai.training.model import LegalActionAttentionScorerConfig
from tools.soft_teacher_targets_128 import source as inherited
from tools.soft_teacher_targets_128 import common as soft_common
from tools.proposal_breadth_128 import source as breadth_source
from tools.equal_budget_breadth_128 import source as equal_source
from . import common as c

SOFT_BINDING = "1b4b30451d1d51b75fa1388ac4f5482d20167cf0d8095d4bdca90e8ff94030c6"
SOFT_RESULT = "45a87c83b9a9c155ae5884eb908d73dcde70a633797395104efd71950d55c3d3"
SOFT_AUDIT = "123da25eeab20709c5601820ff913653c3d2823c97434ef1a809369dfc95dbb2"
SOFT_MODELS = {
    161001: "37613abe392e2344bbf16dfa7364fd3975fa5cc8a13af20c18b610504a4ebfd6",
    161002: "033f61565fd63865a966ea7a12e7c742a64bb720e223888f1b7adbddc7ec89a4",
    161003: "c9f5669dcaa302bdda22a30906b15b97ccd02aeda56a66a7446b0d241f91e5ad",
}
PANELS = ("proposal_breadth_128", "equal_budget_breadth_128")
PANEL_IDENTITIES = {
    PANELS[0]: ("231bdd640ac88af2470c24af38a04e28248a7edeba8e376173df5b356fd27a74",
                "2592d17b5ed5f1ca2a7f463ffa27b9e803db3600b0b3f62ea177aab26e2a2570",
                "f9273a1612636b4c13be6e47768a8ba05a0159f3b176aeddea5aca076faabf55"),
    PANELS[1]: ("00d5f1b0224b91806a056e9d9e1678e34fc2ddb9b2f07e607ff7e06fbba9a108",
                "99e7a00941de58d5364ca66219393f1380b134bce24402790c81780fb395292e",
                "32b5104c7f1ef2b6b5ba1f73ffdc967a4bad4241e22175727813164e63172d60"),
}
PANEL_MODULES = dict(zip(PANELS, (breadth_source, equal_source)))
EXTRA_FIELDS = {"source_panel", "source_game", "environment_seed", "cell"}
row_digest = inherited.row_digest
_plain = inherited._plain


def _soft_parent(add, parent):
    base = c.ROOT / "outputs/soft_teacher_targets_128"
    freeze = add(c.artifact(base / "freeze.json"))
    manifest = add(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True and freeze["binding"] == SOFT_BINDING
              == c.old.digest(manifest) and manifest["source_parent"] == parent
              and manifest["identity"]["config"] == soft_common.CONFIG, "Soft source freeze changed")
    for path, sha in manifest["identity"]["dependencies"].items():
        add({"path": path, "sha256": sha}, False)
    checks = add(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"], "Soft checks changed")
    for ref in checks["logs"]:
        add(ref, False)
    status = add(c.artifact(base / "final_status.json"))
    c.require(status["manifest"] == SOFT_BINDING and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True
              and status["result"]["sha256"] == SOFT_RESULT
              and status["audit"]["sha256"] == SOFT_AUDIT, "Soft completed source changed")
    result, audit = add(status["result"]), add(status["audit"])
    c.require(result["manifest"] == audit["manifest"] == SOFT_BINDING
              and result["all_prescribed_audits_pass"] is audit["passed"] is True
              and audit["training_reproduced"] is True and audit["result"] == status["result"]
              and audit["assessment"] == result["assessment_audit"], "Soft completion audit changed")
    data, original = add(result["data"]), add(parent["reused_data"])
    c.require(data["parent"] == parent and data["manifest"] == SOFT_BINDING
              and data["tensor_bytes_reused"] is data["inherited_conversion_audit"] is True
              and all(data[k] == original[k] for k in inherited.DATA_FIELDS), "Soft data ancestry changed")
    registry = add(result["models"])
    c.require(registry["manifest"] == SOFT_BINDING and registry["fixed_final"] is True
              and registry["all_fit_audits_pass"] is True and registry["audits"] == audit["fits"]
              and set(registry["models"]) == {f"focused_{f}" for f in c.FITS}
              and [r["fit"] for r in registry["fits"]] == list(c.FITS)
              == [r["fit"] for r in registry["audits"]], "Soft models incomplete")
    selected = {}
    for fit, fit_ref, audit_ref in zip(c.FITS, registry["fits"], registry["audits"], strict=True):
        trained, checked = add(fit_ref), add(audit_ref)
        model = registry["models"][f"focused_{fit}"]
        c.require(trained["manifest"] == checked["manifest"] == SOFT_BINDING
                  and trained["config"] == soft_common.CONFIG and trained["fit"] == checked["fit"] == fit
                  and checked["result"] == _plain(fit_ref)
                  and trained["initializer"] == parent["models"][f"anchor_{fit}"]
                  and trained["selected_model"] == checked["selected_model"] == _plain(model)
                  and model["source_fit"] == fit and model["sha256"] == SOFT_MODELS[fit]
                  and trained["selected_update"] == checked["selected_update"] == model["selected_update"] == 1784
                  and trained["row_exposures"] == checked["row_exposures"] == 114044
                  and trained["train_rows"] == 9461
                  and trained["validation_read"] is checked["validation_read"] is False
                  and trained["assessment_read"] is checked["assessment_read"] is False,
                  "Soft selected model/fit identity changed")
        for flag in ("passed", "complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
                     "value_head_frozen", "final_always_selected", "matched_high_rate_control_verified",
                     "optimization_stability_independently_reproduced", "soft_target_arithmetic_independently_reproduced"):
            c.require(checked[flag] is True, "Incomplete soft fitting audit")
        add(model, False)
        selected[f"soft_{fit}"] = model
    # These are authenticated provenance only, never used to choose new rows.
    for key in ("assessment", "assessment_audit", "validation_old_targets"):
        add(result[key], False)
    return selected, {"result": status["result"], "audit": status["audit"],
                      "data": result["data"], "models": result["models"]}


def _panel_parent(panel, add, anchors):
    binding, result_sha, audit_sha = PANEL_IDENTITIES[panel]
    base, module = c.ROOT / "outputs" / panel, PANEL_MODULES[panel]
    freeze = add(c.artifact(base / "freeze.json"))
    manifest = add(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True and freeze["binding"] == binding == c.old.digest(manifest),
              "Added source freeze changed")
    for path, sha in manifest["identity"]["dependencies"].items():
        add({"path": path, "sha256": sha}, False)
    status = add(c.artifact(base / "final_status.json"))
    c.require(status["manifest"] == binding and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True
              and status["result"]["sha256"] == result_sha and status["audit"]["sha256"] == audit_sha,
              "Added source is not the completed audited panel")
    result, audit = add(status["result"]), add(status["audit"])
    c.require(result["manifest"] == audit["manifest"] == binding and audit["passed"] is True
              and result["all_prescribed_audits_pass"] is True and audit["result"] == status["result"],
              "Added source completion linkage changed")
    registry = add(result["models"])
    c.require(registry["manifest"] == binding and all(_plain(registry["models"][f"anchor_{f}"])
              == _plain(anchors[f"anchor_{f}"]) for f in c.FITS), "Added source anchor population changed")
    seal = add(result["selection"])
    selection = add(seal["selection"])
    c.require(seal["manifest"] == selection["manifest"] == binding and len(selection["main"]) == 360,
              "Added panel selection changed")
    for ref in seal["source_locks"]:
        add(ref, False)
    complete = add(audit["stages"]["main"])
    collection, receipts = add(complete["collection"]), add(complete["audits"])
    c.require(complete["manifest"] == collection["manifest"] == receipts["manifest"] == binding
              and complete["all_prescribed_audits_pass"] is True and complete["states"] == 360
              and [r["index"] for r in collection["files"]] == list(range(360))
              == [r["index"] for r in receipts["files"]], "Added root audit population changed")
    descriptors = []
    for spec, root_ref, audit_ref in zip(selection["main"], collection["files"], receipts["files"], strict=True):
        receipt = add(audit_ref)
        c.require(receipt["manifest"] == binding and receipt["passed"] is True and receipt["stage"] == "main"
                  and receipt["spec"] == spec and receipt["root"] == _plain(root_ref)
                  and receipt["recorded_selection_worlds"] == 128, "Added label audit changed")
        add(root_ref, False)
        add(receipt["selection_artifact"], False)
        add(receipt["sample_artifact"], False)
        source_audit = add(spec["source_audit"])
        source_spec = module.independent_spec("main", spec["game_index"])
        c.require(source_audit["passed"] is True and source_audit["manifest"] == binding
                  and source_audit["spec"] == source_spec and source_audit["game"] == spec["source_game"]
                  and source_audit["every_move_hidden_input_invariance"] is True
                  and all(spec[k] == source_spec[k] for k in ("fit", "seat", "phase", "search_stream")),
                  "Added source-game audit changed")
        add(spec["source_game"], False)
        descriptors.append({"source_panel": panel, "binding": binding, "spec": spec,
                            "source_spec": source_spec, "selection": receipt["selection_artifact"],
                            "root": _plain(root_ref), "audit": _plain(audit_ref)})
    c.require(Counter((d["spec"]["fit"], d["spec"]["seat"], d["spec"]["phase"]) for d in descriptors)
              == {cell: 20 for cell in module.CELLS}, "Added 18-cell allocation changed")
    return descriptors, {"result": status["result"], "audit": status["audit"], "selection": seal["selection"]}


def disjointness(descriptors, historical_games, corpus):
    """Whole-game seeds include all 160/learned/breadth ancestor TRAIN and VAL."""
    c.require(len(historical_games) == 1204
              and Counter(g["family"] for g in historical_games)
              == {"world_budget_160": 400, "learned_proposal_128": 402, "proposal_breadth_arena_128": 402},
              "Historical learning-game population changed")
    historic = {g["environment_seed"] for g in historical_games}
    added = [d["source_spec"]["environment_seed"] for d in descriptors]
    games = [(d["source_panel"], d["spec"]["game_index"]) for d in descriptors]
    c.require(len(added) == len(set(added)) == len(set(games)) == 720
              and set(added).isdisjoint(historic), "Added games overlap inherited TRAIN/VAL ancestry")
    by_split = {split: sorted({shard["spec"]["environment_seed"] for shard in corpus["shards"]
                              if corpus["splits"][str(shard["spec"]["pair"])] == split})
                for split in ("train", "validation")}
    c.require(len(by_split["train"]) == 159 and len(by_split["validation"]) == 42
              and set(by_split["train"]).isdisjoint(by_split["validation"])
              and set(by_split["train"] + by_split["validation"]) <= historic, "Inherited whole-pair split changed")
    # Also forbid unselected/probe source games from both paid panels. No old
    # trajectory is loaded to obtain these independently reconstructed namespaces.
    prior_pools = {module.independent_spec(stage, i)["environment_seed"] for module in PANEL_MODULES.values()
                   for stage, n in (("main", 1080), ("probe", 54)) for i in range(n)}
    forbidden = sorted(historic | prior_pools)
    fresh = {c.CONFIG["source_seed"][stage] + i for stage in ("main", "probe")
             for i in range(c.CONFIG["source_games"][stage])}
    c.require(fresh.isdisjoint(forbidden), "Prospective source namespace overlaps history")
    return {"historical_games": 1204, "historical_metadata_sha256": c.old.digest(historical_games),
            "historical_environment_seeds": sorted(historic), "old_split_environment_seeds": by_split,
            "added_environment_seeds": sorted(added), "added_distinct_games": 720,
            "all_inherited_train_validation_and_anchor_ancestry_checked": True,
            "forbidden_environment_seeds": forbidden, "prospective_source_seed_disjoint": True}


def closed_parent():
    prior = inherited.closed_parent()
    refs = {r["path"]: _plain(r) for r in prior["evidence"]}
    def add(ref, read=True):
        item = _plain(ref)
        c.require(item["path"] not in refs or refs[item["path"]] == item, "Conflicting source artifacts")
        present = item["path"] in refs
        refs[item["path"]] = item
        return (c.old.read_json(c.ROOT/item["path"]) if present else c.verified(item)) if read else (
            c.ROOT/item["path"] if present else c.verify_hash(item))
    soft_models, soft = _soft_parent(add, prior)
    models = {f"{arm}_{fit}": prior["models"][f"{arm}_{fit}"]
              for arm in ("anchor", "lowrate") for fit in c.FITS} | soft_models
    descriptors, panels = [], {}
    for panel in PANELS:
        selected, receipt = _panel_parent(panel, add, models)
        descriptors.extend(selected)
        panels[panel] = receipt
    training_data = add(prior["corpus"]["data"])
    historical_refs = [r for r in training_data["source_evidence"] if "/main/games/game_" in r["path"]]
    historical = []
    for ref in historical_refs:
        game = add(ref)
        historical.append({"family": ref["path"].split("/")[1], "path": ref["path"],
                           "environment_seed": game["spec"]["environment_seed"]})
    proof = disjointness(descriptors, historical, prior["corpus"])
    dependencies = {r["path"]: r["sha256"] for r in refs.values() if not r["path"].startswith("outputs/")}
    retirement = {"source_panels": list(PANELS), "rows": 720,
                  "source_selections": {p: panels[p]["selection"] for p in PANELS},
                  "reason": "Consumed as TRAIN; permanently excluded from future evaluation evidence",
                  "ids": [row_identity(d)["id"] for d in descriptors], "permanent": True}
    return {"models": models, "model_config": prior["model_config"], "reused_data": prior["reused_data"],
            "corpus": prior["corpus"], "soft": soft, "panels": panels, "added_sources": descriptors,
            "training_disjointness": proof, "forbidden_environment_seeds": proof["forbidden_environment_seeds"],
            "retired_evaluation_panels": retirement, "dependencies": dependencies,
            "evidence": [refs[p] for p in sorted(refs)], "inventory": {p: refs[p]["sha256"] for p in sorted(refs)},
            "historical_fit_replays": 0, "historical_trajectory_replays": 0,
            "old_conversion_audit_inherited": True, "selection_uses_outcomes": False}


def row_identity(descriptor):
    spec, panel = descriptor["spec"], descriptor["source_panel"]
    return {"id": f"{panel}:{spec['game_index']}:{spec['turn']}", "pair": f"{panel}:{spec['game_index']}",
            "split": "train", "source_fit": spec["fit"], "seat": spec["seat"], "phase": spec["phase"],
            "privacy": spec["privacy"], "source_panel": panel, "source_game": spec["game_index"],
            "environment_seed": descriptor["source_spec"]["environment_seed"],
            "cell": [panel, spec["fit"], spec["seat"], spec["phase"]]}


def partition(identities):
    c.require(len(identities) == len({r["id"] for r in identities}) == 720, "Added identity population changed")
    cells = defaultdict(list)
    for i, row in enumerate(identities):
        c.require(row["cell"] == [row["source_panel"], row["source_fit"], row["seat"], row["phase"]],
                  "Added cell metadata changed")
        cells[tuple(row["cell"])].append(i)
    expected = {(p, f, seat, phase) for p in PANELS for f in c.FITS for seat in (0, 1) for phase in range(3)}
    c.require(set(cells) == expected and all(len(v) == 20 for v in cells.values()), "Added cell population changed")
    blocks = [[] for _ in range(4)]
    for cell in sorted(cells):
        ranked = sorted(cells[cell], key=lambda i: (c.old.digest([c.CONFIG["partition_tag"], list(cell), identities[i]["id"]]), identities[i]["id"]))
        for block in range(4):
            blocks[block].extend(ranked[5*block:5*(block+1)])
    blocks = [sorted(v) for v in blocks]
    c.require(all(len(v) == len(set(v)) == 180 for v in blocks)
              and sorted(i for block in blocks for i in block) == list(range(720)), "Four-block partition changed")
    return {"tag": c.CONFIG["partition_tag"], "blocks": blocks,
            "narrow_indices": {str(f): blocks[i] for i, f in enumerate(c.FITS)},
            "outcome_blind": True, "all_source_fits_pooled": True}


def selection_targets(saved, descriptor):
    identity, block, lock = saved["identity"], saved["selection"], saved["selection_lock"]
    panel, spec, binding = descriptor["source_panel"], descriptor["spec"], descriptor["binding"]
    arm = "breadth" if panel == PANELS[0] else "teacher"
    offered = identity["candidates"][arm]
    keys = offered["keys"]
    n = len(keys)
    c.require(saved["manifest"] == binding and saved["stage"] == "main" and saved["index"] == spec["index"]
              and identity["spec"] == spec and lock["identity_digest"] == c.old.digest(identity)
              and lock["selection_sha256"] == c.old.digest(block)
              and lock["value_scale"] == "terminal_wdl" and identity["union"]["keys"] == keys
              and offered["indices"] == block["candidate_indices"] == list(range(n))
              and identity["candidates"]["anchor"]["keys"] == keys[:min(4, n)]
              and 1 <= n <= 8 and len(keys) == len(set(keys)) == len(offered["actions"])
              and len(block["worlds"]) == len(block["world_seeds"]) == len(set(block["world_seeds"])) == 128,
              "Common128 selection identity/support changed")
    sums = [0] * n
    for world in block["worlds"]:
        c.require(len(world["outcomes"]) == n, "Missing common teacher outcome")
        for i, outcome in enumerate(world["outcomes"]):
            value = outcome["wdl"]
            c.require(type(value) in (int, float) and value in (-1, 0, 1)
                      and outcome["reason"] in c.old.VALID_NATURAL_TERMINATIONS, "Censored/invalid teacher outcome")
            sums[i] += int(value)
    # Independent column reduction checks all score sums, not just the argmax.
    second = [sum(int(w["outcomes"][i]["wdl"]) for w in block["worlds"]) for i in range(n)]
    teacher = max(range(n), key=sums.__getitem__)
    outside = max(range(3, n), key=sums.__getitem__) if n > 3 else -1
    c.require(second == sums and lock["means"][arm] == [v/128 for v in sums]
              and lock["choices"][arm] == teacher, "Common128 old teacher arithmetic changed")
    return {"old_sums_wdl": sums, "teacher": teacher, "narrow_support": min(4, n),
            "narrow_teacher": max(range(min(4, n)), key=sums.__getitem__),
            "outsider_start": min(3, n), "outsider_support": max(0, n-3),
            "outsider_teacher": outside, "outsider_label": outside-3 if outside >= 0 else -1}


def reconstruct_added(descriptor, index, config):
    saved = c.verified(descriptor["selection"])
    spec, identity = descriptor["spec"], saved["identity"]
    game, checked = c.verified(spec["source_game"]), c.verified(spec["source_audit"])
    moves = [m for m in game["trace"] if m["turn"] == spec["turn"]]
    c.require(game["manifest"] == checked["manifest"] == descriptor["binding"]
              and game["spec"] == checked["spec"] == descriptor["source_spec"]
              and checked["passed"] is True and checked["game"] == spec["source_game"]
              and game["trace_sha256"] == checked["trace_sha256"] == c.old.digest(game["trace"])
              and len(moves) == 1 and moves[0]["actor"] == spec["seat"]
              and moves[0]["state_sha256"] == spec["snapshot_sha256"] == c.old.digest(identity["snapshot"])
              and moves[0]["snapshot"] == identity["snapshot"], "Added row is not its audited source-game turn")
    target = selection_targets(saved, descriptor)
    state = c.old.deserialize_state_snapshot(identity["snapshot"])
    encoded, keys, actions = encode_position(state, config)
    obs, features, groups, independent_keys, legal = independent.input_invariance(state, config)
    c.require(torch.equal(encoded["observation"], obs) and torch.equal(encoded["features"], features)
              and encoded["groups"].tolist() == groups and keys == independent_keys and actions == legal
              and state.current_player == spec["seat"] and min(state.turn_index//20, 2) == spec["phase"]
              and c.baseline.privacy(state) == spec["privacy"] == identity["privacy"],
              "New lawful/private-invariant encoding failed")
    offered = identity["union"]
    members = []
    for key, encoded_action in zip(offered["keys"], offered["actions"], strict=True):
        action = c.old.decode_action(encoded_action)
        c.require(action in legal and repr(c.old.canonical_successor_action_key(state, action)) == key,
                  "Offered action differs from canonical teacher group")
        locations = [i for i, a in enumerate(legal) if repr(c.old.canonical_successor_action_key(state, a)) == key]
        c.require(locations, "Teacher candidate absent from full legal input")
        members.append(torch.tensor(locations, dtype=torch.long))
    row = {**encoded, **row_identity(descriptor), **target, "index": 11950 + index, "members": members}
    validate_added_row(row)
    return row


def validate_added_row(row):
    c.require(set(row) == inherited.ROW_FIELDS | EXTRA_FIELDS and 11950 <= row["index"] < 12670,
              "Unexpected added fitting payload")
    # Mature canonical tensor/target checks apply after translating only the new
    # namespace. There is no change to old input/target or alias semantics.
    core = {key: row[key] for key in inherited.ROW_FIELDS}
    core["index"] = row["index"] - 11950
    inherited.validate_row(core)
    c.require(row["split"] == "train" and row["cell"] == [row["source_panel"], row["source_fit"], row["seat"], row["phase"]]
              and row["source_panel"] in PANELS and type(row["source_game"]) is int
              and type(row["environment_seed"]) is int
              and row["id"].startswith(f"{row['source_panel']}:{row['source_game']}:")
              and row["pair"] == f"{row['source_panel']}:{row['source_game']}", "Added row identity/split changed")
    return True


def prepare(binding, base=c.OUTPUT, parent=None):
    base, parent = Path(base), closed_parent() if parent is None else parent
    c.require(not (base/"data_lock.json").exists(), "One-shot training source preparation")
    original = c.verified(parent["reused_data"])
    c.require(original["manifest"] == inherited.LOWRATE_BINDING
              and parent["reused_data"]["sha256"] == inherited.LOWRATE_DATA_SHA, "Original full data seal changed")
    config = LegalActionAttentionScorerConfig(**parent["model_config"])
    added = [reconstruct_added(d, i, config) for i, d in enumerate(parent["added_sources"])]
    identities = [row_identity(d) for d in parent["added_sources"]]
    allocation = partition(identities)
    files = {"old": original["files"]["train"], "validation": original["files"]["validation"],
             "added": inherited.previous._write_tensor(base/"added.pt", added)}
    digests = {"old": original["row_digests"]["train"], "validation": original["row_digests"]["validation"],
               "added": row_digest(added)}
    audit = {"manifest": binding, "passed": True, "added_rows": 720,
             "independent_public_private_invariant_encoding": True, "all_canonical_memberships_checked": True,
             "every128_world_target_independently_summed": True, "actual_source_turn_membership_checked": True,
             "inherited_tensor_bytes_unchanged": True, "assessment_in_payload": False,
             "historical_trajectory_replays": 0, "new_simulations": 0,
             "files": files, "row_digests": digests, "partition": allocation,
             "training_disjointness": parent["training_disjointness"],
             "retired_evaluation_panels": parent["retired_evaluation_panels"]}
    c.old.save_new(base/"training_source_audit.json", audit)
    value = {"manifest": binding, "parent": parent, "inherited_data": parent["reused_data"],
             "files": files, "row_digests": digests, "counts": {"old": 9461, "added": 720, "validation": 2489},
             "added_identities": identities, "partition": allocation, "audit": c.artifact(base/"training_source_audit.json"),
             "assessment_in_payload": False, "historical_replay": False, "tensor_bytes_reused": True,
             "retired_evaluation_panels": parent["retired_evaluation_panels"]}
    c.old.save_new(base/"data_lock.json", value)
    return c.artifact(base/"data_lock.json")


def _lock(binding, base):
    base = Path(base)
    value = c.old.read_json(base/"data_lock.json")
    c.require(value["manifest"] == binding and value["counts"] == {"old": 9461, "added": 720, "validation": 2489}
              and value["assessment_in_payload"] is value["historical_replay"] is False
              and value["tensor_bytes_reused"] is True, "Training data seal changed")
    if (base/"freeze.json").exists():
        freeze = c.old.read_json(base/"freeze.json")
        manifest = c.verified(freeze["manifest"])
        c.require(freeze["binding"] == binding == c.old.digest(manifest)
                  and manifest["source_parent"] == value["parent"], "Training ancestry differs from frozen source")
    else:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Production training source lacks freeze")
    original = c.verified(value["inherited_data"])
    c.require(value["inherited_data"] == value["parent"]["reused_data"]
              and value["inherited_data"]["sha256"] == inherited.LOWRATE_DATA_SHA
              and original["manifest"] == inherited.LOWRATE_BINDING
              and all(value["files"][new] == original["files"][old]
                      and value["row_digests"][new] == original["row_digests"][old]
                      for new, old in (("old", "train"), ("validation", "validation")))
              and value["added_identities"] == [row_identity(d) for d in value["parent"]["added_sources"]]
              and value["partition"] == partition(value["added_identities"])
              and value["retired_evaluation_panels"] == value["parent"]["retired_evaluation_panels"],
              "Training tensor/split/partition identity changed")
    audit = c.verified(value["audit"])
    c.require(audit["manifest"] == binding and audit["passed"] is True
              and audit["files"] == value["files"] and audit["row_digests"] == value["row_digests"]
              and audit["partition"] == value["partition"] and audit["assessment_in_payload"] is False
              and audit["training_disjointness"] == value["parent"]["training_disjointness"]
              and audit["retired_evaluation_panels"] == value["retired_evaluation_panels"]
              and all(audit[k] is True for k in ("independent_public_private_invariant_encoding",
                  "all_canonical_memberships_checked", "every128_world_target_independently_summed",
                  "actual_source_turn_membership_checked", "inherited_tensor_bytes_unchanged")),
              "New data conversion audit changed")
    return value


def load_rows(binding, split, base=c.OUTPUT):
    c.require(split in ("old", "added", "validation"), "Unknown training data split")
    lock = _lock(binding, base)
    rows = torch.load(c.verify_hash(lock["files"][split]), map_location="cpu", weights_only=True)
    c.require(len(rows) == lock["counts"][split] and row_digest(rows) == lock["row_digests"][split],
              "Training tensor content changed")
    if split == "added":
        for i, (row, identity) in enumerate(zip(rows, lock["added_identities"], strict=True)):
            validate_added_row(row)
            c.require(row["index"] == 11950+i and all(row[k] == v for k, v in identity.items()), "Added row order changed")
    else:
        expected_split = "train" if split == "old" else "validation"
        expected = []
        corpus = lock["parent"]["corpus"]
        for shard in corpus["shards"]:
            spec = shard["spec"]
            if corpus["splits"][str(spec["pair"])] == expected_split:
                expected.extend(f"breadth128:{spec['index']}:{r['turn']}" for r in shard["roots"])
        c.require([row["id"] for row in rows] == expected, "Original tensor order/membership changed")
        for row in rows:
            inherited.validate_row(row)
            c.require(row["split"] == expected_split, "Validation crossed training firewall")
    return rows


def narrow_indices(fit, binding, base=c.OUTPUT):
    c.require(fit in c.FITS, "Unknown partition fit")
    return _lock(binding, base)["partition"]["narrow_indices"][str(fit)]


def control_models(binding, base=c.OUTPUT):
    return _lock(binding, base)["parent"]["models"]


def initializer(fit, base=c.OUTPUT):
    c.require(fit in c.FITS, "Unknown initializer")
    lock = c.old.read_json(Path(base)/"data_lock.json")
    ref = _lock(lock["manifest"], base)["parent"]["models"][f"anchor_{fit}"]
    c.require(ref["sha256"] == inherited.inherited_models.EXPECTED[f"anchor_{fit}"]
              and ref["source_fit"] == fit, "Original initializer changed")
    c.verify_hash(ref)
    return ref


def validate(binding, base=c.OUTPUT, authenticate_parent=True):
    lock = _lock(binding, base)
    if authenticate_parent:
        c.require(lock["parent"] == closed_parent(), "Closed source ancestry changed")
    # After the exhaustive sealed conversion, runtime verification is hashes and
    # receipt linkage. Re-encoding or re-reading 128-world targets per fit would
    # add no independent evidence and is deliberately avoided.
    for ref in lock["files"].values():
        c.verify_hash(ref)
    return lock

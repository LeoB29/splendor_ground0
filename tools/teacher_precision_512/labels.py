"""Recoverable additional-world supervision and independent sampled replay.

The original128 outcomes stay immutable. Three separately durable128-world
blocks add384 worlds. Unknown terminal values hold; none become target draws.
"""
from pathlib import Path
import random
import time
import traceback
import uuid

from scripts.analysis import verify_rollout_selector_154 as world_audit
from scripts.analysis import verify_afterstate_advantage_149 as continuation_audit
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit
from tools.equal_budget_breadth_arena_128_recovery_02 import verify as repetition_audit
from . import common as c
from . import label_source


class UnresolvedTarget(c.BudgetExceeded):
    """An unknown continuation cannot be silently converted into a soft label."""


def root_path(index, base=c.OUTPUT):
    return Path(base)/"labels"/"roots"/f"state_{index:04d}.json"


def audit_path(index, base=c.OUTPUT):
    return Path(base)/"labels"/"audits"/f"state_{index:04d}.json"


def block_path(index, block, base=c.OUTPUT):
    c.require(block in (1, 2, 3), "Wrong additional label block")
    return Path(base)/"labels"/"blocks"/f"state_{index:04d}"/f"block_{block}.json"


def block_receipt_path(index, block, base=c.OUTPUT):
    return block_path(index, block, base).with_name(f"block_{block}_complete.json")


def _seal_block(spec, binding, block, base):
    path = block_path(spec["index"], block, base)
    receipt = {"manifest": binding, "index": spec["index"], "row_id": spec["row_id"],
               "spec_sha256": c.old.digest(spec), "block": block, "artifact": c.artifact(path)}
    c.old.save_new(block_receipt_path(spec["index"], block, base), receipt)


def _authenticated_block(spec, binding, block, base):
    path, seal = block_path(spec["index"], block, base), block_receipt_path(spec["index"], block, base)
    if not path.exists() or not seal.exists():
        raise c.BudgetExceeded("Incomplete label-block completion receipt; preserve evidence for explicit recovery")
    receipt = c.old.read_json(seal)
    c.require(receipt["manifest"] == binding and receipt["index"] == spec["index"]
              and receipt["row_id"] == spec["row_id"] and receipt["spec_sha256"] == c.old.digest(spec)
              and receipt["block"] == block and c.verify_hash(receipt["artifact"]).resolve() == path.resolve(),
              "Completed label block differs from durable receipt")
    record = c.verified(receipt["artifact"])
    _validate_block(record, spec, binding, block)
    return record, c.artifact(path, block=block, completion=c.artifact(seal))


def world_seeds(index, block):
    c.require(type(index) is int and 0 <= index < 720 and block in (1, 2, 3), "Wrong label namespace")
    first = c.CONFIG["label_world_seed_base"]+index*1024+block*128
    return list(range(first, first+128))


def ordinary_indices(index, block):
    c.require(type(index) is int and 0 <= index < 720 and block in (1, 2, 3), "Wrong label sample identity")
    return sorted(sorted(range(128), key=lambda w: (
        c.old.digest([c.CONFIG["label_audit_tag"], c.CONFIG["audit_seed"], index, block, w]), w))[:8])


def sums(worlds, support, *, require_resolved=True):
    c.require(len(worlds) == 128 and type(support) is int and 1 <= support <= 8, "Wrong label block dimensions")
    totals, unknown = [0]*support, []
    for wi, world in enumerate(worlds):
        c.require(len(world["outcomes"]) == support, "Missing label candidate")
        for ai, outcome in enumerate(world["outcomes"]):
            value = outcome["wdl"]
            if value is None:
                c.require(outcome["reason"] == "repetition_cutoff" and outcome["winner"] is None, "Invalid censored label outcome")
                unknown.append([wi, ai])
            else:
                c.require(type(value) in (int, float) and value in (-1, 0, 1)
                          and outcome["reason"] in c.old.VALID_NATURAL_TERMINATIONS, "Invalid terminal label outcome")
                totals[ai] += int(value)
    if require_resolved and unknown:
        raise UnresolvedTarget("Unresolved label outcomes: preserve blocks and hold without imputation")
    return totals, unknown


def independent_sums(worlds, support):
    """Separate column-wise integer reduction; no collector aggregates used."""
    assert len(worlds) == 128 and all(len(w["outcomes"]) == support for w in worlds)
    result = []
    for column in range(support):
        values = [w["outcomes"][column] for w in worlds]
        assert all(type(o["wdl"]) in (int, float) and o["wdl"] in (-1, 0, 1)
                   and o["reason"] in c.old.VALID_NATURAL_TERMINATIONS for o in values), "Unresolved/invalid targets"
        result.append(sum(int(o["wdl"]) for o in values))
    return result


def _validate_block(record, spec, binding, block):
    c.require(record["manifest"] == binding and record["index"] == spec["index"]
              and record["row_id"] == spec["row_id"] and record["spec_sha256"] == c.old.digest(spec)
              and record["block"] == block and record["world_tag"] == c.CONFIG["label_world_tag"]
              and record["world_seeds"] == world_seeds(spec["index"], block)
              and record["candidate_indices"] == list(range(spec["support"]))
              and record["complete"] is True and record["assessment_read"] is False
              and [w["seed"] for w in record["worlds"]] == record["world_seeds"], "Changed label block identity/budget")
    total, unknown = sums(record["worlds"], spec["support"], require_resolved=False)
    c.require(record["sums_wdl"] == total and record["unknown_cells"] == unknown, "Changed block arithmetic")
    c.require(set(record["timing"]) == {"wall", "cpu"}
              and all(type(x) in (int, float) and 0 <= x < float("inf") for x in record["timing"].values()), "Invalid block timing")
    return unknown


def _check_unknown(record, state, actions, spec, binding, base):
    """Audit every unresolved branch before raising the scientific hold."""
    if not record["unknown_cells"]:
        return
    rows = []
    for wi, ai in record["unknown_cells"]:
        c.budget_boundary(base)
        seed = record["world_seeds"][wi]
        world = world_audit.independent_world(state, seed)
        rebuilt = continuation_audit.replay_branch(world, actions[ai], state.current_player)
        c.require(rebuilt == record["worlds"][wi]["outcomes"][ai], "Unresolved label replay mismatch")
        rows.append({"world": wi, "candidate": ai, "outcome": rebuilt})
    path = Path(base)/"labels"/"unresolved"/f"state_{spec['index']:04d}_block_{record['block']}.json"
    value = {"manifest": binding, "index": spec["index"], "block": record["block"],
             "rows": rows, "all_unknown_branches_replayed": True, "imputed": False}
    if path.exists():
        c.require(c.old.read_json(path) == value, "Unresolved audit changed")
    else:
        c.old.save_new(path, value)
    raise UnresolvedTarget("Unresolved label outcomes independently reproduced; cannot form512-world targets")


def collect(spec, binding, base=c.OUTPUT):
    base, started, cpu = Path(base), time.perf_counter(), time.process_time()
    c.budget_boundary(base)
    if root_path(spec["index"], base).exists():
        validate_root(spec, binding, base)
        return c.artifact(root_path(spec["index"], base), index=spec["index"])
    source = label_source.load_source(spec, binding, base)
    state = c.old.deserialize_state_snapshot(source["snapshot"])
    actions = [c.old.decode_action(a) for a in source["union"]["actions"]]
    old_sums, old_unknown = sums(source["old_block"]["worlds"], len(actions))
    c.require(old_sums == spec["old_sums_wdl"] and not old_unknown, "Inherited128 target changed")
    old_seeds = set(source["old_block"]["world_seeds"])
    blocks, reused_wall, reused_cpu = [], 0., 0.
    for block in (1, 2, 3):
        c.budget_boundary(base)
        path = block_path(spec["index"], block, base)
        if path.exists():
            record, block_ref = _authenticated_block(spec, binding, block, base)
            reused_wall += record["timing"]["wall"]
            reused_cpu += record["timing"]["cpu"]
        else:
            c.require(not block_receipt_path(spec["index"], block, base).exists(), "Preserve orphaned block completion receipt")
            begin, proc = time.perf_counter(), time.process_time()
            seeds = world_seeds(spec["index"], block)
            c.require(old_seeds.isdisjoint(seeds), "New labels overlap inherited world seeds")
            record = {"manifest": binding, "index": spec["index"], "row_id": spec["row_id"],
                      "spec_sha256": c.old.digest(spec), "block": block,
                      "world_tag": c.CONFIG["label_world_tag"], "world_seeds": seeds,
                      "candidate_indices": list(range(len(actions))), "worlds": [], "assessment_read": False}
            try:
                for seed in seeds:
                    c.budget_boundary(base)
                    world = c.old.determinize_for_observer(state, state.current_player, random.Random(seed))
                    saved = {"seed": seed, "state_sha256": c.old.digest(c.old.serialize_state_snapshot(world)), "outcomes": []}
                    record["worlds"].append(saved)
                    legal = c.old.SplendorEnv().legal_actions(world)
                    for action in actions:
                        c.require(action in legal, "Inherited candidate became illegal in lawful label world")
                        out = c.old.trajectory(world, action)
                        out["wdl"] = (c.old.outcome_value(out["winner"], state.current_player)
                                      if out["reason"] in c.old.VALID_NATURAL_TERMINATIONS else None)
                        repetition_audit._check_outcome(out, state.current_player, state.turn_index)
                        saved["outcomes"].append(out)
                totals, unknown = sums(record["worlds"], len(actions), require_resolved=False)
                record.update(complete=True, sums_wdl=totals, unknown_cells=unknown,
                              timing={"wall": time.perf_counter()-begin, "cpu": time.process_time()-proc})
                c.old.save_new(path, record)
                _seal_block(spec, binding, block, base)
                record, block_ref = _authenticated_block(spec, binding, block, base)
            except BaseException:
                record["failure"] = traceback.format_exc()
                record["complete"] = False
                c.old.save_new(path.with_name(f"block_{block}_partial_{uuid.uuid4().hex}.json"), record)
                raise
        _check_unknown(record, state, actions, spec, binding, base)
        blocks.append(block_ref)
    records = [c.verified(r) for r in blocks]
    totals = [old_sums[j]+sum(r["sums_wdl"][j] for r in records) for j in range(len(actions))]
    result = {"manifest": binding, "index": spec["index"], "row_id": spec["row_id"], "spec": spec,
              "complete": True, "old_sums_wdl": old_sums, "sums_wdl512": totals, "worlds": 512,
              "new_worlds": 384, "blocks": blocks, "old_selection": source["old_selection"],
              "source_root": source["source_root"], "source_audit": source["source_audit"],
              "unknown_outcomes": 0, "assessment_read": False, "time": c.old.now(),
              "recorded_continuations": 384*len(actions), "inherited_continuations": 128*len(actions),
              "timing": {"wall": time.perf_counter()-started+reused_wall,
                         "cpu": time.process_time()-cpu+reused_cpu}}
    c.old.save_new(root_path(spec["index"], base), result)
    return c.artifact(root_path(spec["index"], base), index=spec["index"])


def validate_root(spec, binding, base=c.OUTPUT, *, independent=False):
    """Read-only source/link/512-label reconstruction; no trajectories or inference."""
    source = label_source.load_source(spec, binding, base)
    root = c.old.read_json(root_path(spec["index"], base))
    c.require(root["manifest"] == binding and root["index"] == spec["index"] and root["row_id"] == spec["row_id"]
              and root["spec"] == spec and root["complete"] is True and root["worlds"] == 512
              and root["new_worlds"] == 384 and root["unknown_outcomes"] == 0
              and root["assessment_read"] is False and len(root["blocks"]) == 3
              and all(root[k] == source[k] for k in ("old_selection", "source_root", "source_audit")), "Changed complete label root")
    columns = [independent_sums(source["old_block"]["worlds"], spec["support"])]
    used = set(source["old_block"]["world_seeds"])
    for block, ref in enumerate(root["blocks"], 1):
        c.require(ref["block"] == block and c.verify_hash(ref).resolve() == block_path(spec["index"], block, base).resolve(), "Wrong block artifact")
        c.require(c.verify_hash(ref["completion"]).resolve() == block_receipt_path(spec["index"], block, base).resolve(),
                  "Wrong block completion receipt")
        record, authenticated = _authenticated_block(spec, binding, block, base)
        c.require(authenticated == ref, "Block/receipt differs from complete root seal")
        c.require(not _validate_block(record, spec, binding, block), "Unknown target must hold")
        c.require(used.isdisjoint(record["world_seeds"]), "Teacher world namespaces overlap")
        used.update(record["world_seeds"])
        columns.append(independent_sums(record["worlds"], spec["support"]) if independent else record["sums_wdl"])
    total = [sum(col[j] for col in columns) for j in range(spec["support"])]
    c.require(root["old_sums_wdl"] == spec["old_sums_wdl"] == columns[0]
              and root["sums_wdl512"] == total and len(used) == 512
              and root["recorded_continuations"] == 384*spec["support"]
              and root["inherited_continuations"] == 128*spec["support"], "Changed512 label arithmetic/accounting")
    return root


def audit(spec, binding, base=c.OUTPUT, *, full=False):
    base, started, cpu = Path(base), time.perf_counter(), time.process_time()
    c.budget_boundary(base)
    path = audit_path(spec["index"], base)
    root = validate_root(spec, binding, base, independent=True)
    root_ref = c.artifact(root_path(spec["index"], base))
    source = label_source.load_source(spec, binding, base)
    if "fixture_payload" in spec:
        forced_full, plan_ref = full, None
    else:
        plan = label_source.load_plan(binding, base)
        forced_full = spec["index"] in plan["sample"]["full_replay_indices"]
        c.require(not full or spec["index"] in plan["sample"]["probe_indices"], "Unplanned full timing replay")
        plan_ref = c.artifact(Path(base)/"label_plan.json")
    if path.exists():
        receipt = c.old.read_json(path)
        c.require(receipt["manifest"] == binding and receipt["passed"] is True and receipt["root"] == root_ref
                  and receipt["full_inner_replay"] == bool(forced_full) and receipt["plan"] == plan_ref,
                  "Prior label audit cannot be reused")
        return c.artifact(path, index=spec["index"])
    state = c.old.deserialize_state_snapshot(source["snapshot"])
    alternate = input_audit.hidden_alternative(state)
    actions = [c.old.decode_action(a) for a in source["union"]["actions"]]
    replayed, replay_wall, turns, replay_indices = 0, 0., 0, {}
    for block, ref in enumerate(root["blocks"], 1):
        record = c.verified(ref)
        expected = [2310000000+spec["index"]*1024+128*block+i for i in range(128)]
        assert record["world_seeds"] == expected
        selected = list(range(128)) if forced_full else ordinary_indices(spec["index"], block)
        replay_indices[str(block)] = selected
        for wi, saved in enumerate(record["worlds"]):
            c.budget_boundary(base)
            assert set(saved) == {"seed", "state_sha256", "outcomes"} and saved["seed"] == expected[wi]
            world = world_audit.independent_world(state, expected[wi])
            assert saved["state_sha256"] == c.old.digest(c.old.serialize_state_snapshot(world))
            assert saved["state_sha256"] == c.old.digest(c.old.serialize_state_snapshot(world_audit.independent_world(alternate, expected[wi])))
            legal = c.old.SplendorEnv().legal_actions(world)
            for action, outcome in zip(actions, saved["outcomes"], strict=True):
                assert action in legal
                repetition_audit._check_outcome(outcome, state.current_player, state.turn_index)
                assert outcome["wdl"] is not None
                if wi in selected:
                    now = time.perf_counter()
                    rebuilt = continuation_audit.replay_branch(world, action, state.current_player)
                    assert outcome == rebuilt, (spec["index"], block, wi, outcome, rebuilt)
                    replay_wall += time.perf_counter()-now
                    replayed += 1
                    turns += rebuilt["turns"]
    wall = time.perf_counter()-started
    result = {"manifest": binding, "index": spec["index"], "row_id": spec["row_id"], "spec": spec,
              "root": root_ref, "plan": plan_ref, "passed": True, "old128_authenticated": True,
              "all512_sums_independently_checked": True, "assessment_read": False, "unknown_outcomes": 0,
              "old_trajectory_replays": 0, "recorded_worlds": 384,
              "replayed_worlds": sum(len(v) for v in replay_indices.values()),
              "recorded_continuations": 384*len(actions), "replayed_continuations": replayed,
              "replay_world_indices": replay_indices, "full_inner_replay": bool(forced_full), "moves": turns,
              "timing": {"wall": wall, "cpu": time.process_time()-cpu, "replay_wall": replay_wall,
                         "structural_wall": max(0., wall-replay_wall)}}
    c.old.save_new(path, result)
    return c.artifact(path, index=spec["index"])


def build_lock(binding, base=c.OUTPUT):
    """Seal complete ordered roots and audits, including reused full timing roots."""
    base = Path(base)
    plan = label_source.load_plan(binding, base)
    roots, audits = [], []
    for spec in plan["specs"]:
        c.budget_boundary(base)
        root = validate_root(spec, binding, base, independent=True)
        ref, audit_ref = c.artifact(root_path(spec["index"], base), index=spec["index"]), c.artifact(audit_path(spec["index"], base), index=spec["index"])
        checked = c.verified(audit_ref)
        c.require(checked["manifest"] == binding and checked["spec"] == spec and checked["passed"] is True
                  and checked["root"] == {k: ref[k] for k in ("path", "sha256")}
                  and checked["all512_sums_independently_checked"] is checked["old128_authenticated"] is True
                  and checked["unknown_outcomes"] == 0 and checked["assessment_read"] is False,
                  "Incomplete label audit population")
        roots.append(ref); audits.append(audit_ref)
    outputs = []
    for name, files in (("labels_lock.json", roots), ("labels_audits_lock.json", audits)):
        value = {"manifest": binding, "complete": True, "files": files, "states": 720,
                 "worlds_per_root": 512, "new_worlds_per_root": 384, "assessment_read": False,
                 "all_prescribed_audits_pass": True, "plan": c.artifact(base/"label_plan.json")}
        c.old.save_new(base/name, value)
        outputs.append(c.artifact(base/name))
    return tuple(outputs)


def crossfit_row(block_sums):
    """128-vs384 teaching choices, each scored on its excluded128 block."""
    assert len(block_sums) == 4 and block_sums[0]
    support = len(block_sums[0])
    assert 1 <= support <= 8 and all(len(row) == support for row in block_sums)
    assert all(type(v) is int and -128 <= v <= 128 for row in block_sums for v in row)
    folds = []
    for h in range(4):
        small_block = (h+1) % 4
        if support <= 3:
            small = large = None
            delta = 0
        else:
            small = max(range(3, support), key=lambda a: block_sums[small_block][a])
            large = max(range(3, support), key=lambda a: sum(block_sums[b][a] for b in range(4) if b != h))
            delta = block_sums[h][large]-block_sums[h][small]
        folds.append({"excluded_block": h, "single_block": small_block, "choice128": small,
                      "choice384": large, "heldout_wdl_sum_difference": delta,
                      "score_gain": delta/256})
    # Independent column-major first-strict-improvement reconstruction.
    columns = list(zip(*block_sums, strict=True))
    check = []
    for h in range(4):
        winner_small = winner_large = None
        score_small = score_large = -1000
        for a in range(3, support):
            low = columns[a][(h+1) % 4]
            high = sum(columns[a])-columns[a][h]
            if low > score_small: winner_small, score_small = a, low
            if high > score_large: winner_large, score_large = a, high
        delta = 0 if winner_large is None else columns[winner_large][h]-columns[winner_small][h]
        check.append((winner_small, winner_large, delta))
    assert check == [(r["choice128"], r["choice384"], r["heldout_wdl_sum_difference"]) for r in folds]
    numerator = sum(r["heldout_wdl_sum_difference"] for r in folds)
    return {"folds": folds, "sum_heldout_wdl_differences": numerator,
            "mean_score_gain": numerator/1024,
            "choice_disagreements": sum(r["choice128"] != r["choice384"] for r in folds),
            "outsider_support": max(0, support-3)}


def crossfit_diagnostic(binding, base=c.OUTPUT):
    """No inference, resampling, new trajectories, model selection or gates."""
    base = Path(base)
    collection_ref, audits_ref = c.artifact(base/"labels_lock.json"), c.artifact(base/"labels_audits_lock.json")
    collection, audits = c.verified(collection_ref), c.verified(audits_ref)
    c.require(collection["manifest"] == audits["manifest"] == binding
              and collection["complete"] is audits["complete"] is True
              and len(collection["files"]) == len(audits["files"]) == 720,
              "Incomplete audited label corpus")
    rows = []
    for i, (ref, audit_ref) in enumerate(zip(collection["files"], audits["files"], strict=True)):
        c.budget_boundary(base)
        root, checked = c.verified(ref), c.verified(audit_ref)
        c.require(root["index"] == checked["index"] == ref["index"] == audit_ref["index"] == i
                  and root["manifest"] == checked["manifest"] == binding and checked["passed"] is True
                  and checked["root"] == {k: ref[k] for k in ("path", "sha256")}, "Wrong crossfit input/audit")
        source = label_source.load_source(root["spec"], binding, base)
        blocks = [source["old_block"]]+[c.verified(b) for b in root["blocks"]]
        columns = [independent_sums(b["worlds"], root["spec"]["support"]) for b in blocks]
        c.require([sum(col[j] for col in columns) for j in range(root["spec"]["support"])]
                  == root["sums_wdl512"], "Crossfit512 sum changed")
        rows.append({"index": i, "row_id": root["row_id"], **crossfit_row(columns)})
    numerator = sum(r["sum_heldout_wdl_differences"] for r in rows)
    value = {"manifest": binding, "collection": collection_ref, "audits": audits_ref,
             "rows": rows, "states": 720, "folds_per_root": 4,
             "mean_score_gain": numerator/(720*1024), "score_percentage_points": 100*numerator/(720*1024),
             "choice_disagreements": sum(r["choice_disagreements"] for r in rows),
             "nonempty_outsider_roots": sum(r["outsider_support"] > 0 for r in rows),
             "independent_columnar_arithmetic_checked": True, "assessment_read": False,
             "descriptive_only": True, "model_gate": False, "resampling": False,
             "new_simulations": 0, "historical_trajectory_replays": 0,
             "scope": "TRAIN-block crossfit128 versus384. Not independent evaluation of a512-selected maximum or fresh-state generalization."}
    path = base/"label_crossfit_diagnostic.json"
    c.old.save_new(path, value)
    return c.artifact(path)

"""Exact, authenticated mapping of the 720 retired TRAIN roots to new labels.

Only old selection artifacts are opened. Historical assessment payloads are
authenticated by file hash, never read into supervision. No model is evaluated.
"""
from collections import Counter
from pathlib import Path

from tools.independent_game_coverage_128_recovery_01 import training_source as inherited
from . import common as c

CELLS = tuple((f, s, p) for f in c.FITS for s in (0, 1) for p in range(3))
COVERAGE_DATA_SHA = "7013b84d738a87c1fa9b0586ee79e758f1db130f0b3cbdb9dbe4406dc8e49812"


def _plan_path(base):
    return Path(base)/"label_plan.json"


def validate_specs(specs, *, fixture=False):
    c.require(specs and [s["index"] for s in specs] == list(range(len(specs))), "Label index/order changed")
    c.require(len({s["row_id"] for s in specs}) == len(specs), "Duplicate label row")
    for s in specs:
        c.require(s["fit"] in c.FITS and s["seat"] in (0, 1) and s["phase"] in range(3), "Wrong label cell")
        c.require(type(s["support"]) is int and 1 <= s["support"] <= 8, "Wrong label support")
        c.require(set(s["privacy"]) == {"own", "opponent"}
                  and all(type(v) is int and v >= 0 for v in s["privacy"].values()), "Invalid label privacy")
    if not fixture:
        c.require(len(specs) == 720 and Counter((s["fit"], s["seat"], s["phase"]) for s in specs)
                  == {cell: 40 for cell in CELLS}, "720 TRAIN population changed")
        c.require(Counter(s["descriptor"]["source_panel"] for s in specs)
                  == {p: 360 for p in inherited.PANELS}, "TRAIN panel mixture changed")
        for s in specs:
            d, old_spec = s["descriptor"], s["descriptor"]["spec"]
            c.require(s["row_id"] == inherited.row_identity(d)["id"]
                      and s["row_index"] == 11950+s["index"]
                      and s["support"] == len(s["old_sums_wdl"])
                      and all(s[k] == old_spec[k] for k in ("fit", "seat", "phase", "turn", "privacy", "snapshot_sha256")),
                      "Label plan metadata differs from immutable source descriptor")


def sample_plan(specs, *, fixture=False):
    """Precommitted identity/metadata selections; no outcome/timing fields used."""
    validate_specs(specs, fixture=fixture)
    priority = lambda s: (c.old.digest([c.CONFIG["label_stress_tag"], c.CONFIG["audit_seed"], s["row_id"]]), s["index"])
    probe_priority = lambda s: (c.old.digest([c.CONFIG["label_probe_tag"], s["row_id"]]), s["index"])
    probe, stress, cells = set(), set(), []
    def choose(name, eligible, *, timing=False):
        chosen = min(eligible, key=probe_priority if timing else priority)["index"] if eligible else None
        cells.append({"cell": name, "eligible_count": len(eligible), "selected_index": chosen, "timing": timing})
        if chosen is not None:
            (probe if timing else stress).add(chosen)
    for fit, seat, phase in CELLS:
        eligible = [s for s in specs if (s["fit"], s["seat"], s["phase"]) == (fit, seat, phase)]
        choose(f"probe|{fit}|{seat}|{phase}", eligible, timing=True)
        choose(f"phase|{fit}|{seat}|{phase}", eligible)
    for fit in c.FITS:
        group = [s for s in specs if s["fit"] == fit]
        for privacy in range(4):
            choose(f"privacy|{fit}|{privacy}", [s for s in group if int(bool(s["privacy"]["own"]))
                   + 2*int(bool(s["privacy"]["opponent"])) == privacy])
        for support in (1, 2, 3):
            choose(f"short|{fit}|{support}", [s for s in group if s["support"] == support])
    if not fixture:
        c.require(len(probe) == 18 and len(stress) <= 39, "Probe/stress bounds changed")
    return {"probe_indices": sorted(probe), "stress_indices": sorted(stress),
            "full_replay_indices": sorted(probe | stress), "cells": cells,
            "missing_cells": [x["cell"] for x in cells if x["selected_index"] is None],
            "outcome_blind": True, "ordinary_worlds_per_block": 8}


def prepare(binding, base=c.OUTPUT):
    base = Path(base)
    data_ref = c.artifact(base/"data_lock.json")
    data = c.verified(data_ref)
    c.require(data["manifest"] == binding and data["counts"]["added"] == 720, "Wrong inherited data seal")
    parent = c.verified(data["parent"]["coverage_data"])
    c.require(data["parent"]["coverage_data"]["sha256"] == COVERAGE_DATA_SHA
              and parent["manifest"] == c.PARENT_BINDING, "Unrecognized inherited720 corpus")
    descriptors = parent["parent"]["added_sources"]
    c.require(data["added_identities"] == [inherited.row_identity(d) for d in descriptors]
              and data["retired_evaluation_panels"]["permanent"] is True
              and data["retired_evaluation_panels"]["ids"] == [r["id"] for r in data["added_identities"]],
              "Label sources are not exact retired TRAIN rows")
    specs = []
    for i, descriptor in enumerate(descriptors):
        saved = c.verified(descriptor["selection"])
        target = inherited.selection_targets(saved, descriptor)
        identity, spec = inherited.row_identity(descriptor), descriptor["spec"]
        value = {"index": i, "row_index": 11950+i, "row_id": identity["id"],
                 "fit": spec["fit"], "seat": spec["seat"], "phase": spec["phase"],
                 "turn": spec["turn"], "privacy": spec["privacy"],
                 "support": len(target["old_sums_wdl"]), "descriptor": descriptor,
                 "snapshot_sha256": spec["snapshot_sha256"],
                 "old_sums_wdl": target["old_sums_wdl"]}
        load_source(value, binding, base, check_plan=False)
        specs.append(value)
    sample = sample_plan(specs)
    plan = {"manifest": binding, "data": data_ref, "specs": specs, "sample": sample,
            "assessment_read": False, "old128_authenticated": True,
            "row_order": "inherited_added_tensor_index11950_plus_index", "time": c.old.now()}
    c.old.save_new(_plan_path(base), plan)
    return c.artifact(_plan_path(base))


def load_plan(binding, base=c.OUTPUT):
    plan = c.old.read_json(_plan_path(base))
    c.require(plan["manifest"] == binding and plan["assessment_read"] is False
              and plan["old128_authenticated"] is True, "Invalid label plan")
    data = c.verified(plan["data"])
    c.require(data["manifest"] == binding and c.verify_hash(plan["data"]).resolve()
              == (Path(base)/"data_lock.json").resolve(), "Label/data binding mismatch")
    validate_specs(plan["specs"])
    c.require(plan["sample"] == sample_plan(plan["specs"]), "Label audit sample changed")
    parent = c.verified(data["parent"]["coverage_data"])
    c.require(data["parent"]["coverage_data"]["sha256"] == COVERAGE_DATA_SHA
              and parent["manifest"] == c.PARENT_BINDING, "Unrecognized inherited720 corpus")
    c.require([s["descriptor"] for s in plan["specs"]] == parent["parent"]["added_sources"]
              and [s["row_id"] for s in plan["specs"]] == [r["id"] for r in data["added_identities"]],
              "Label plan left inherited source population")
    return plan


def probe_specs(plan):
    return [plan["specs"][i] for i in plan["sample"]["probe_indices"]]


def load_source(spec, binding, base=c.OUTPUT, *, check_plan=True):
    """Authenticate old selection, actual reached state, actions and old sums."""
    base = Path(base)
    if "fixture_payload" in spec:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Fixture label source targets production")
        return spec["fixture_payload"]
    if check_plan:
        plan = load_plan(binding, base)
        c.require(plan["specs"][spec["index"]] == spec, "Unplanned TRAIN label root")
    d = spec["descriptor"]
    old_spec = d["spec"]
    saved = c.verified(d["selection"])
    target = inherited.selection_targets(saved, d)
    c.require(spec["row_id"] == inherited.row_identity(d)["id"] and spec["row_index"] == 11950+spec["index"]
              and spec["old_sums_wdl"] == target["old_sums_wdl"]
              and all(spec[k] == old_spec[k] for k in ("fit", "seat", "phase", "turn", "privacy", "snapshot_sha256")),
              "Label metadata/old score mismatch")
    receipt = c.verified(d["audit"])
    c.verify_hash(d["root"])
    c.require(receipt["passed"] is True and receipt["manifest"] == d["binding"]
              and receipt["spec"] == old_spec and receipt["root"] == d["root"]
              and receipt["selection_artifact"] == d["selection"], "Old128 independent audit changed")
    game, audit = c.verified(old_spec["source_game"]), c.verified(old_spec["source_audit"])
    moves = [m for m in game["trace"] if m["turn"] == old_spec["turn"]]
    snapshot = saved["identity"]["snapshot"]
    c.require(game["manifest"] == audit["manifest"] == d["binding"] and audit["passed"] is True
              and game["spec"] == audit["spec"] == d["source_spec"] and audit["game"] == old_spec["source_game"]
              and game["trace_sha256"] == audit["trace_sha256"] == c.old.digest(game["trace"])
              and len(moves) == 1 and moves[0]["snapshot"] == snapshot
              and moves[0]["actor"] == spec["seat"] and moves[0]["state_sha256"] == spec["snapshot_sha256"],
              "Label source differs from audited actual turn")
    state = c.old.deserialize_state_snapshot(snapshot)
    offered = saved["identity"]["union"]
    actions = [c.old.decode_action(a) for a in offered["actions"]]
    legal = c.old.SplendorEnv().legal_actions(state)
    c.require(not state.terminal and state.current_player == spec["seat"] and state.turn_index == spec["turn"]
              and c.old.digest(snapshot) == spec["snapshot_sha256"] and len(actions) == spec["support"]
              and all(a in legal for a in actions)
              and [repr(c.old.canonical_successor_action_key(state, a)) for a in actions] == offered["keys"],
              "Label actions, state or canonical order changed")
    return {"snapshot": snapshot, "union": offered, "old_selection": d["selection"],
            "source_root": d["root"], "source_audit": d["audit"],
            "old_block": saved["selection"], "old_sums_wdl": target["old_sums_wdl"]}

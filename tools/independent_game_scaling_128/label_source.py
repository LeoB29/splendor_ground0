"""Outcome-blind nested TRAIN identities and bounded label-audit sample."""
from collections import Counter
from pathlib import Path

from . import common as c

CELLS = tuple((f, s, p) for f in c.FITS for s in (0, 1) for p in range(3))
_PLAN_CACHE = {}


def make_spec(source_spec):
    return {**source_spec, "row_index": 12670+source_spec["index"],
            "row_id": f"independent_game_scaling_128:{source_spec['game_index']}:{source_spec['turn']}",
            "support": min(8, source_spec["canonical_support"])}


def validate_specs(specs, *, fixture=False):
    c.require(specs and [s["index"] for s in specs] == list(range(len(specs))), "New label index/order differs")
    c.require(len({s["game_index"] for s in specs}) == len({s["row_id"] for s in specs}) == len(specs),
              "More than one TRAIN root from a source game")
    for s in specs:
        c.require(s["fit"] in c.FITS and s["seat"] in (0, 1) and s["phase"] in range(3)
                  and s["row_index"] == 12670+s["index"] and s["source_stage"] == "train"
                  and s["pair"] == s["game_index"] and s["row_id"] == f"independent_game_scaling_128:{s['game_index']}:{s['turn']}"
                  and type(s["support"]) is int and s["support"] == min(8, s["canonical_support"])
                  and 1 <= s["support"] <= 8, "New label identity/support differs")
        c.require(set(s["privacy"]) == {"own", "opponent"}
                  and all(type(v) is int and v >= 0 for v in s["privacy"].values()), "Invalid label privacy")
    if not fixture:
        c.require(len(specs) == 10800 and Counter((s["fit"],s["seat"],s["phase"]) for s in specs)
                  == {cell:600 for cell in CELLS}, "New10800 TRAIN balance changed")


def partitions(specs):
    """The original720 core plus identity-nested120/600 new games per cell."""
    validate_specs(specs)
    medium = []
    for cell in CELLS:
        selected = [s for s in specs if (s["fit"],s["seat"],s["phase"]) == cell]
        ranked = sorted(selected,key=lambda s:(c.old.digest([c.CONFIG["partition_tag"],list(cell),s["row_id"]]),s["row_id"]))
        medium.extend(720+s["index"] for s in ranked[:120])
    small = list(range(720))
    result = {"small":small,"medium":small+sorted(medium),"large":list(range(11520))}
    c.require({k:len(v) for k,v in result.items()} == c.CONFIG["sizes"]
              and set(result["small"]) < set(result["medium"]) < set(result["large"]), "Nested sample sizes changed")
    return result


def sample_plan(specs, *, fixture=False):
    validate_specs(specs,fixture=fixture)
    priority = lambda s:(c.old.digest([c.CONFIG["label_stress_tag"],c.CONFIG["audit_seed"],s["row_id"]]),s["index"])
    probe_priority = lambda s:(c.old.digest([c.CONFIG["label_probe_tag"],s["row_id"]]),s["index"])
    probe,stress,cells = set(),set(),[]
    def choose(name,eligible,timing=False):
        selected = min(eligible,key=probe_priority if timing else priority)["index"] if eligible else None
        cells.append({"cell":name,"eligible_count":len(eligible),"selected_index":selected,"timing":timing})
        if selected is not None: (probe if timing else stress).add(selected)
    for f,s,p in CELLS:
        group = [v for v in specs if (v["fit"],v["seat"],v["phase"]) == (f,s,p)]
        choose(f"probe|{f}|{s}|{p}",group,True)
        choose(f"phase|{f}|{s}|{p}",group)
    for f in c.FITS:
        group = [v for v in specs if v["fit"] == f]
        for privacy in range(4):
            choose(f"privacy|{f}|{privacy}",[v for v in group if int(bool(v["privacy"]["own"]))+2*int(bool(v["privacy"]["opponent"])) == privacy])
        for support in (1,2,3): choose(f"short|{f}|{support}",[v for v in group if v["support"] == support])
    if not fixture: c.require(len(probe) == 18 and len(stress) <= 39, "Label probe/stress bound changed")
    return {"probe_indices":sorted(probe),"stress_indices":sorted(stress),"full_replay_indices":sorted(probe|stress),
            "cells":cells,"missing_cells":[v["cell"] for v in cells if v["selected_index"] is None],
            "outcome_blind":True,"ordinary_worlds_per_block":8}


def prepare(binding, source_refs, audit_refs, base=c.OUTPUT):
    from . import source, training_source
    base = Path(base)
    selection_path = base/"train_selection.json"
    if selection_path.exists():
        selection_ref = c.artifact(selection_path)
        selection = c.verified(selection_ref)
        c.require(selection["manifest"] == binding and selection["sources"]["train"]
                  == {"games":source_refs,"audits":audit_refs}, "Existing training selection differs")
        source.validate_train_selection(selection,binding,base)
    else:
        selection_ref = source.prepare_train_selection(source_refs,audit_refs,binding,base)
        selection = c.verified(selection_ref)
    specs = [make_spec(s) for s in selection["train"]]
    validate_specs(specs)
    inherited = training_source.inherited_lock(binding,base)
    historic = set(inherited["parent"]["training_disjointness"]["forbidden_environment_seeds"])
    historic.update(i["environment_seed"] for i in inherited["added_identities"])
    seeds = [c.CONFIG["source_seed"]["train"]+s["game_index"] for s in specs]
    future = {c.CONFIG["source_seed"][stage]+i for stage in ("probe","main") for i in range(c.CONFIG["source_games"][stage])}
    c.require(len(set(seeds)) == 10800 and set(seeds).isdisjoint(historic|future), "New TRAIN games overlap historical/future games")
    c.require(Counter((r["source_fit"],r["seat"],r["phase"]) for r in inherited["added_identities"])
              == {cell:40 for cell in CELLS}, "Original720 core balance changed")
    value = {"manifest":binding,"selection":selection_ref,"inherited":c.artifact(base/"inherited_source_lock.json"),
             "specs":specs,"partition":partitions(specs),"sample":sample_plan(specs),
             "assessment_read":False,"selection_uses_outcomes":False,"old128_only":True,
             "distinct_new_training_games":10800,"new_source_seed_disjoint":True,"time":c.old.now()}
    path = base/"label_plan.json"
    if path.exists():
        old = c.old.read_json(path)
        c.require({k:v for k,v in old.items() if k != "time"} == {k:v for k,v in value.items() if k != "time"}, "Existing label plan differs")
    else: c.old.save_new(path,value)
    return c.artifact(path)


def load_plan(binding, base=c.OUTPUT):
    """Authenticate and derive the complete plan once per worker, not per root."""
    base = Path(base)
    seal_path = base/"label_plan_seal.json"
    if seal_path.exists():
        seal = c.old.read_json(seal_path)
        c.require(seal["manifest"] == binding, "Label plan seal binding differs")
        ref = seal["plan"]
    else:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Production label plan is not sealed")
        ref = c.artifact(base/"label_plan.json")
    c.require((c.ROOT/ref["path"]).resolve() == (base/"label_plan.json").resolve(), "Label plan seal redirected")
    key = (str(base.resolve()),binding,ref["sha256"])
    if key not in _PLAN_CACHE:
        value = c.verified(ref)
        selection = c.verified(value["selection"])
        c.require(value["manifest"] == selection["manifest"] == binding
                  and value["specs"] == [make_spec(s) for s in selection["train"]]
                  and value["partition"] == partitions(value["specs"])
                  and value["sample"] == sample_plan(value["specs"])
                  and value["assessment_read"] is value["selection_uses_outcomes"] is False
                  and value["old128_only"] is value["new_source_seed_disjoint"] is True,
                  "Frozen TRAIN identities/sample/partition changed")
        c.require(c.verify_hash(value["inherited"]).resolve() == (base/"inherited_source_lock.json").resolve(), "Label source seal differs")
        _PLAN_CACHE[key] = value
    return _PLAN_CACHE[key]


def probe_specs(plan):
    return [plan["specs"][i] for i in plan["sample"]["probe_indices"]]


def plan_ref(binding, base=c.OUTPUT):
    load_plan(binding,base)
    path = Path(base)/"label_plan_seal.json"
    return c.old.read_json(path)["plan"] if path.exists() else c.artifact(Path(base)/"label_plan.json")


def load_source(spec, binding, base=c.OUTPUT):
    from . import source
    base = Path(base)
    if "fixture_payload" in spec:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Fixture TRAIN root targets production")
        return spec["fixture_payload"]["snapshot"]
    plan = load_plan(binding,base)
    c.require(plan["specs"][spec["index"]] == spec, "Unplanned training root")
    state = source.load_state(spec,binding,base)
    snapshot = state if isinstance(state,dict) else c.old.serialize_state_snapshot(state)
    c.require(c.old.digest(snapshot) == spec["snapshot_sha256"], "Selected TRAIN snapshot changed")
    return snapshot

"""Outcome-blind whole-pair sampling from the completed, audited U/A arena."""
from collections import Counter, defaultdict
from pathlib import Path
from . import common as c
from tools.equal_budget_breadth_arena_128_recovery_02 import common as arena

def phase(turn):
    return 0 if turn < 20 else 1 if turn < 40 else 2

def rank(*identity):
    return c.old.digest([c.CONFIG["selection_tag"], *identity])

def closed_arena():
    refs = {}
    def read(ref):
        refs[ref["path"]] = {k: ref[k] for k in ("path", "sha256")}
        return c.verified(ref)
    freeze = read(c.artifact(c.SOURCE/"freeze.json"))
    manifest = read(freeze["manifest"])
    c.require(c.old.digest(manifest) == freeze["binding"] == c.SOURCE_BINDING, "Arena binding changed")
    status = read(c.artifact(c.SOURCE/"final_status.json"))
    c.require(status["complete"] and status["all_prescribed_audits_pass"]
              and status["manifest"] == c.SOURCE_BINDING, "Arena incomplete")
    c.require(status["result"]["sha256"] == "2dde6af8330cb6ed3c5afa0369cca3a48c7646087cf903d753f07d93ddc73d03"
              and status["audit"]["sha256"] == "22f36a08694dd9160c14c2cc0110c07f9909bbb0bf5462ddd21d89000d5c9572",
              "Arena completion changed")
    result, audit = read(status["result"]), read(status["audit"])
    c.require(audit["passed"] and audit["result"] == status["result"]
              and result["all_prescribed_audits_pass"]
              and audit["manifest"] == result["manifest"] == c.SOURCE_BINDING
              and result["games"] == audit["coverage"]["games"] == 402 and result["pairs"] == 201,
              "Arena result not audited")
    inputs, audits = read(audit["inputs"]), read(audit["audit_lock"])
    c.require(audits["audit_recovery_binding"] == c.SOURCE_BINDING, "Wrong source audit generation")
    games = inputs["games"]
    c.require([x["index"] for x in games] == list(range(402))
              and [x["index"] for x in audits["files"]] == list(range(402)), "Source inventory incomplete")
    return {"evidence": [refs[k] for k in sorted(refs)], "games": games, "audits": audits["files"]}

def allocations(games):
    """Consume identity/availability metadata only, never game outcomes."""
    c.require(len(games) == 402 and sorted(g["spec"]["index"] for g in games) == list(range(402)),
              "Incomplete source game identities")
    by_fit = defaultdict(dict)
    for item in games:
        spec = item["spec"]
        c.require(spec["fit"] in c.FITS and spec["pair"] == spec["index"]//2
                  and spec["candidate_seat"] in (0, 1), "Invalid source pair identity")
        by_fit[spec["fit"]].setdefault(spec["pair"], []).append(item)
    result = {"probe": [], "main": []}
    for fit in c.FITS:
        pairs = by_fit[fit]
        c.require(len(pairs) == 67, "Unexpected arena fit population")
        ordered = sorted(pairs, key=lambda p: (rank("pair", fit, p), p))
        for pos, pair in enumerate(ordered[:63]):
            stage = "probe" if pos < 3 else "main"
            if stage == "probe":
                ph = pos
                arm = ("candidate", "reference")[int(rank("probe-arm", fit, pair), 16) % 2]
            else:
                cell = (pos-3) % 6
                ph, arm = cell//2, ("candidate", "reference")[cell%2]
            legs = sorted(pairs[pair], key=lambda g: g["spec"]["index"])
            c.require(len(legs) == 2 and {g["spec"]["candidate_seat"] for g in legs} == {0,1}
                      and len({g["spec"]["environment_seed"] for g in legs}) == 1
                      and len({g["spec"]["search_stream"] for g in legs}) == 1, "Incomplete seat pair")
            for game in legs:
                gs = game["spec"]
                actor = gs["candidate_seat"] if arm == "candidate" else 1-gs["candidate_seat"]
                eligible = [r for r in game["trace"] if r["actor"] == actor and phase(r["turn"]) == ph]
                c.require(bool(eligible), "Fixed source cell unavailable; do not substitute a pair")
                selected = min(eligible, key=lambda r: (rank("root", stage, gs["index"], r["turn"]), r["turn"]))
                result[stage].append({"index": len(result[stage]), "pair": pair,
                    "game_index": gs["index"], "fit": fit, "seat": actor, "phase": ph,
                    "source_arm": arm, "turn": selected["turn"], "search_stream": gs["search_stream"],
                    "environment_seed": gs["environment_seed"],
                    "snapshot_sha256": selected["state_sha256"], "root_sha256": selected["root_sha256"]})
    return result

def training_seeds(data_ref):
    """Read authenticated game metadata already covered by the closed data audit."""
    data = c.verified(data_ref)
    c.require(data["manifest"] == c.LEARNING_BINDING, "Wrong historical corpus")
    refs = [r for r in data["source_evidence"] if "/main/games/game_" in r["path"] and r["path"].endswith(".json")]
    c.require(len(refs) == len({r["path"] for r in refs}) == 1204, "Unexpected historical corpus game population")
    seeds = {c.verified(r)["spec"]["environment_seed"] for r in refs}
    return {"game_refs_sha256": c.old.digest(refs), "games": len(refs), "environment_seeds": sorted(seeds)}

def build(data_ref):
    parent = closed_arena()
    games = [c.verified(r) for r in parent["games"]]
    # Projection deliberately excludes winner, reason, final scores and policy outputs.
    metadata = [{"spec": g["spec"], "trace": [{k:r[k] for k in ("actor","turn","state_sha256","root_sha256")}
                 for r in g["trace"]]} for g in games]
    schedule = allocations(metadata)
    training = training_seeds(data_ref)
    c.require(set(training["environment_seeds"]).isdisjoint(g["spec"]["environment_seed"] for g in games),
              "Arena seeds overlap training/validation games")
    for specs in schedule.values():
        for spec in specs:
            index, turn = spec["game_index"], spec["turn"]
            path = arena.resolve_root("main", index, turn)
            c.require(path is not None, "Missing audited source root")
            spec["source_game"], spec["source_audit"] = parent["games"][index], parent["audits"][index]
            spec["source_root"] = c.artifact(path)
            c.require(spec["source_root"]["sha256"] == spec.pop("root_sha256"), "Source trace root mismatch")
            snapshot = load_snapshot(spec)
            state = c.old.deserialize_state_snapshot(snapshot)
            legal = c.old.SplendorEnv().legal_actions(state)
            spec["canonical_support"] = len(c.old.actions_by_canonical_successor_key(state, legal))
            spec["privacy"] = c.baseline.privacy(state)
    validate_schedule(schedule)
    return {"schedule": schedule, "source_evidence": parent["evidence"],
            "training_disjointness": training, "new_source_games": 0,
            "source_binding": c.SOURCE_BINDING, "selection_uses_outcomes": False}

def validate_schedule(schedule):
    c.require(set(schedule) == {"probe", "main"}, "Unexpected panel stages")
    pairsets = {}
    for stage, specs in schedule.items():
        c.require([s["index"] for s in specs] == list(range(c.CONFIG["states"][stage])), "Panel indices")
        pairs = defaultdict(list)
        c.require(len({s["game_index"] for s in specs}) == len(specs), "Repeated source game")
        for s in specs:
            c.require(s["fit"] in c.FITS and s["phase"] in (0, 1, 2)
                      and s["source_arm"] in ("candidate", "reference")
                      and s["game_index"]//2 == s["pair"] and s["turn"] >= 0
                      and phase(s["turn"]) == s["phase"], "Invalid source cell")
            pairs[s["pair"]].append(s)
        c.require(len(pairs) == c.CONFIG["source_pairs"][stage], "Pair count")
        for rows in pairs.values():
            c.require(len(rows) == 2 and {s["seat"] for s in rows} == {0,1}
                      and len({(s["fit"],s["phase"],s["source_arm"]) for s in rows}) == 1,
                      "Whole pair/cell broken")
        cells = Counter((r[0]["fit"],r[0]["phase"],r[0]["source_arm"]) for r in pairs.values())
        if stage == "main":
            c.require(len(cells) == 18 and set(cells.values()) == {10}, "Main cell balance")
        else:
            c.require(Counter((r[0]["fit"],r[0]["phase"]) for r in pairs.values()) ==
                      Counter({(f,p):1 for f in c.FITS for p in range(3)}), "Probe cell balance")
        pairsets[stage] = set(pairs)
    c.require(pairsets["probe"].isdisjoint(pairsets["main"]), "Probe/main pair leakage")
    return True

def load_snapshot(spec, base=c.OUTPUT):
    c.require(type(spec["turn"]) is int and spec["turn"] >= 0
              and spec["source_arm"] in ("candidate", "reference"), "Invalid source identity")
    game, root, audit = (c.verified(spec[k]) for k in ("source_game","source_root","source_audit"))
    gs = game["spec"]
    c.require(audit["passed"] and audit["game_artifact"] == spec["source_game"]
              and audit["manifest"] == game["manifest"] == root["manifest"] == arena.ORIGINAL_BINDING
              and "failure" not in game and "failure" not in root
              and game["reason"] in c.old.VALID_NATURAL_TERMINATIONS, "Source audit linkage")
    c.require(gs["index"] == spec["game_index"] and gs["fit"] == spec["fit"]
              and gs["pair"] == spec["pair"] and gs["search_stream"] == spec["search_stream"]
              and gs["environment_seed"] == spec["environment_seed"], "Source spec identity")
    c.require(root["root_identity"] == ["main",spec["game_index"],spec["turn"]], "Source root identity")
    c.require(spec["turn"] < len(game["trace"]), "Source turn outside completed game")
    trace = game["trace"][spec["turn"]]
    c.require(trace["actor"] == spec["seat"] and trace["turn"] == spec["turn"]
              and phase(spec["turn"]) == spec["phase"], "Source actor/phase")
    actor = gs["candidate_seat"] if spec["source_arm"] == "candidate" else 1-gs["candidate_seat"]
    c.require(actor == spec["seat"] and root["planner"] == spec["source_arm"], "Source arm mismatch")
    c.require(c.old.digest(root["snapshot"]) == spec["snapshot_sha256"] == trace["state_sha256"]
              and spec["source_root"]["sha256"] == trace["root_sha256"], "Source snapshot/trace changed")
    matches = [r for r in audit["roots"] if r["turn"] == spec["turn"]]
    c.require(len(matches) == 1 and matches[0]["sha256"] == spec["source_root"]["sha256"]
              and matches[0]["index"] == spec["game_index"] and matches[0]["seat"] == spec["seat"]
              and matches[0]["assigned_fit"] == spec["fit"] and matches[0]["planner"] == spec["source_arm"],
              "Source root audit linkage")
    state = c.old.deserialize_state_snapshot(root["snapshot"])
    c.require(not state.terminal and state.current_player == spec["seat"] and state.turn_index == spec["turn"],
              "Source saved state identity")
    return root["snapshot"]

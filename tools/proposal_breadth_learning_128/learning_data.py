"""Retained verified tensors and independently rebuilt breadth supervision.

Original evidence remains read-only. New rows are checked against actual game
reconstruction and independent lawful encoders, without new inner simulations.
"""
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import torch

from tools.proposal_iteration_128 import learning as previous
from tools.proposal_breadth_arena_128 import arena as teacher
from tools.proposal_breadth_arena_128 import verify as teacher_audit
from tools.proposal_breadth_arena_128 import models as teacher_models
from scripts.analysis import verify_fast_selector_transfer_157 as independent
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from . import common as c

ROOT, OUTPUT, SOURCE = c.ROOT, c.OUTPUT, c.SOURCE
BREADTH_SOURCE, BREADTH_BINDING = c.BREADTH_SOURCE, c.BREADTH_BINDING
old, f = c.old, previous.f
CONFIG = {"retained_rows": 23936, "breadth_rows": 11950, "total_rows": 35886,
          "retained_train_rows": 19047, "retained_validation_rows": 4889,
          "new_training_pairs_per_source_fit": 53, "new_validation_pairs_per_source_fit": 14,
          "split_tag": "proposal-breadth-learning-128-source-pair-split-v1",
          "row_order": "retained_old_then_retained_new_then_breadth_game_turn",
          "old_splits_unchanged": True, "new_continuation_simulations": 0}
PARENT_DATA_SHA = "65a053738cbd396391d499524b87252d26b3132250ba14b1c682043f9547b165"
BREADTH_RESULT_SHA = "e554788ba3ff580651464a6007d31c953e1ac639dc54d90e6d247d406e1ceb2e"
BREADTH_AUDIT_SHA = "e82b556f009fc6a70a01d101eeb98d2f09924414c2aea205269e1694b03fc591"


@lru_cache(maxsize=1)
def splits():
    result = dict(previous.splits())
    schedule = teacher.specs("main")
    for fit in c.FITS:
        pairs = {row["pair"] for row in schedule if row["fit"] == fit}
        assert len(pairs) == 67
        order = sorted(pairs, key=lambda pair: old.digest([CONFIG["split_tag"], fit, pair]))
        result.update({f"breadth128:{pair}": "validation" if pair in order[:14] else "train" for pair in pairs})
    assert len(result) == 602
    return result


@lru_cache(maxsize=1)
def independent_splits():
    result = dict(previous.independent_splits())
    schedule = teacher_audit.independent_specs("main")
    for fit in (161001, 161002, 161003):
        pairs = sorted({row["pair"] for row in schedule if row["fit"] == fit})
        keys = [(old.digest(["proposal-breadth-learning-128-source-pair-split-v1", fit, pair]), pair) for pair in pairs]
        valid = {pair for _, pair in sorted(keys)[:14]}
        result.update({"breadth128:%s" % pair: "validation" if pair in valid else "train" for pair in pairs})
    return result


def _parent_data():
    item = c.artifact(SOURCE / "data_lock.json")
    assert item["sha256"] == PARENT_DATA_SHA
    lock = c.verified(item)
    audit = c.verified(lock["audit"])
    assert lock["manifest"] == audit["manifest"] == c.SOURCE_BINDING and audit["passed"]
    assert lock["config"] == previous.CONFIG and lock["splits"] == previous.independent_splits()
    assert lock["rows"] == 23936 and lock["train_rows"] == 19047 and lock["validation_rows"] == 4889
    for kind, count in (("old", 400), ("new", 402)):
        assert len(lock[f"{kind}_exports"]) == len(audit[f"{kind}_export_audits"]) == count
    return item, lock, audit


def source_evidence():
    """Prelaunch hash inventory; no corpus extraction, training or root scan."""
    found = {}
    def add(item, read=False):
        c.verify_hash(item)
        value = {key: item[key] for key in ("path", "sha256")}
        assert item["path"] not in found or found[item["path"]] == value
        found[item["path"]] = value
        return c.verified(item) if read else None
    parent_item, parent, audit = _parent_data()
    add(parent_item); add(parent["audit"])
    for item in parent["source_dependencies"]:
        add(item)
    for kind in ("old", "new"):
        for item, checked in zip(parent[f"{kind}_exports"], audit[f"{kind}_export_audits"], strict=True):
            add(item)
            receipt = add(checked, True)
            assert receipt["passed"] and receipt["export"] == item
            source = receipt["source"]
            if kind == "old":
                add({"path": source["path"], "sha256": source["sha256"]})
            else:
                add(source["game"]); add(source["audit"])
    # This closed predecessor helper has no ancestor runtime-budget checks.
    ancestor = teacher_models._closed_parent()
    for item in ancestor["source_evidence"]:
        add(item)
    freeze = add(c.artifact(BREADTH_SOURCE / "freeze.json"), True)
    manifest = add(freeze["manifest"], True)
    assert freeze["binding"] == BREADTH_BINDING == old.digest(manifest) and freeze["owner_authorized"]
    status = add(c.artifact(BREADTH_SOURCE / "final_status.json"), True)
    assert status["manifest"] == BREADTH_BINDING and status["complete"] and status["all_prescribed_audits_pass"]
    assert status["classification"] == "positive_practical_progress"
    assert status["result"]["sha256"] == BREADTH_RESULT_SHA and status["audit"]["sha256"] == BREADTH_AUDIT_SHA
    result, final = add(status["result"], True), add(status["audit"], True)
    add(status["report"])
    assert result["manifest"] == final["manifest"] == BREADTH_BINDING and final["passed"]
    assert result["practical_progress"] and result["all_prescribed_audits_pass"] and final["result"] == status["result"]
    registry_item = c.artifact(BREADTH_SOURCE / "models_lock.json")
    registry = add(registry_item, True)
    assert final["models"] == result["models"] == registry_item and registry["manifest"] == BREADTH_BINDING
    assert all(registry[key] == value for key, value in ancestor.items())
    assert not registry["training"] and not registry["ancestor_learning_replayed"]
    for fit in c.FITS:
        item = previous.initializer(fit)
        assert registry["models"][f"anchor_{fit}"] == {**item, "unchanged_from_anchor": True}
        add(item)
    add(registry["continuation_model"])
    for stage in ("probe", "main"):
        complete_item = c.artifact(BREADTH_SOURCE / f"{stage}_complete.json")
        assert final["stages"][stage] == complete_item
        complete = add(complete_item, True)
        assert complete["manifest"] == BREADTH_BINDING and complete["all_prescribed_audits_pass"]
        for name in ("game_lock", "audit_lock", "sample"):
            lock = add(complete[name], True)
            assert lock["manifest"] == BREADTH_BINDING
            if stage == "main" and name != "sample":
                assert len(lock["files"]) == 402
                assert [item["index"] for item in lock["files"]] == list(range(402))
                for item in lock["files"]:
                    add(item)
    add(c.artifact(BREADTH_SOURCE / "schedule.json"))
    return [found[key] for key in sorted(found)]


def source_games():
    complete = old.read_json(BREADTH_SOURCE / "main_complete.json")
    games, audits = c.verified(complete["game_lock"]), c.verified(complete["audit_lock"])
    assert complete["manifest"] == games["manifest"] == audits["manifest"] == BREADTH_BINDING
    assert complete["all_prescribed_audits_pass"] and len(games["files"]) == len(audits["files"]) == 402
    schedule = teacher.specs("main")
    assert schedule == teacher_audit.independent_specs("main") == old.read_json(BREADTH_SOURCE / "schedule.json")["main"]
    return [{"game": game, "audit": checked, "spec": spec, "parent_binding": BREADTH_BINDING}
            for game, checked, spec in zip(games["files"], audits["files"], schedule, strict=True)]


def targets(root):
    candidates, prefix = root["candidates"], root["proposal"]["anchor_candidates"]
    size, narrow = len(candidates["actions"]), len(prefix["actions"])
    assert 1 <= size <= 8 and narrow == min(size, 4)
    assert len(candidates["keys"]) == len(set(candidates["keys"])) == size
    assert prefix["keys"] == candidates["keys"][:narrow] and prefix["actions"] == candidates["actions"][:narrow]
    assert len(root["worlds"]) == 128
    assert all(len(world["outcomes"]) == size for world in root["worlds"])
    assert all(type(outcome["wdl"]) in (int,float) and outcome["wdl"] in (-1.,0.,1.)
               and outcome["reason"] in old.VALID_NATURAL_TERMINATIONS
               for world in root["worlds"] for outcome in world["outcomes"])
    totals = [sum(world["outcomes"][i]["wdl"] for world in root["worlds"]) for i in range(size)]
    broad = totals.index(max(totals))
    assert root["means"] == [value/128 for value in totals] and root["choice"] == broad
    return {"teacher": broad, "narrow_teacher": totals[:narrow].index(max(totals[:narrow])), "narrow_support": narrow}


def independent_targets(root):
    size = len(root["candidates"]["keys"])
    assert size == len(root["candidates"]["actions"]) and 1 <= size <= 8
    prefix = root["proposal"]["anchor_candidates"]
    for key in ("keys", "actions"):
        assert prefix[key] == root["candidates"][key][:min(4,size)]
    assert len(root["worlds"]) == 128 and len(set(root["candidates"]["keys"])) == size
    sums = [0]*size
    for world in root["worlds"]:
        assert len(world["outcomes"]) == size
        for i, outcome in enumerate(world["outcomes"]):
            assert type(outcome["wdl"]) in (int,float) and outcome["wdl"] in (-1,0,1)
            assert outcome["reason"] in old.VALID_NATURAL_TERMINATIONS
            sums[i] += int(outcome["wdl"])
    broad = max(range(size), key=lambda i: sums[i])
    narrow = max(range(min(size,4)), key=lambda i: sums[i])
    assert root["choice"] == broad and root["means"] == [value/128 for value in sums]
    return {"teacher": broad, "narrow_teacher": narrow, "narrow_support": min(size,4)}


def make_row(state, root, spec, config):
    row, keys, actions = f.encode_position(state, config)
    offered = [old.canonical_successor_action_key(state, old.decode_action(action)) for action in root["candidates"]["actions"]]
    assert len(set(offered)) == len(offered) and 1 <= len(offered) <= 8
    assert [repr(key) for key in offered] == root["candidates"]["keys"]
    row["members"] = [torch.nonzero(row["groups"] == keys.index(key), as_tuple=True)[0] for key in offered]
    assert all(len(group) for group in row["members"])
    pair = f"breadth128:{spec['pair']}"
    row.update(id=f"breadth128:{spec['index']}:{state.turn_index}", pair=pair, split=splits()[pair],
               seat=state.current_player, phase=min(state.turn_index//20,2), privacy=root["privacy"],
               **targets(root))
    return row


def inspect_row(row, state, root, spec, config):
    obs, features, groups, keys, legal = independent.input_invariance(state, config)
    members = [[i for i, action in enumerate(legal) if repr(old.canonical_successor_action_key(state,action)) == key]
               for key in root["candidates"]["keys"]]
    assert all(members) and 1 <= len(members) <= 8
    assert [repr(old.canonical_successor_action_key(state,old.decode_action(action)))
            for action in root["candidates"]["actions"]] == root["candidates"]["keys"]
    assert torch.equal(row["observation"],obs) and torch.equal(row["features"],features)
    assert row["groups"].tolist() == groups and row["group_count"] == len(keys)
    assert [member.tolist() for member in row["members"]] == members
    privacy = {role: sum(player.reserved_card_source(i) != independent.RESERVE_SOURCE_VISIBLE
                         for i in range(len(player.reserved_cards)))
               for role,player in (("own",state.players[state.current_player]),("opponent",state.players[1-state.current_player]))}
    pair = "breadth128:%s" % spec["pair"]
    metadata = {"id": "breadth128:%s:%s" % (spec["index"],state.turn_index), "pair":pair,
                "split":independent_splits()[pair], "seat":state.current_player,
                "phase":min(state.turn_index//20,2), "privacy":privacy, **independent_targets(root)}
    assert {key:row[key] for key in metadata} == metadata and privacy == root["privacy"]
    assert set(row) == set(metadata) | {"observation","features","groups","group_count","members"}
    return {"rows":1,row["split"]:1,f"fit_{spec['fit']}":1,f"support_{len(members)}":1,
            "different_targets":int(row["teacher"] != row["narrow_teacher"]),
            "offered_actions":len(members),"candidate_continuations":128*len(members)}


def export_game(source, binding, base=OUTPUT):
    c.budget_boundary(base)
    game, receipt = c.verified(source["game"]), c.verified(source["audit"])
    spec = source["spec"]
    assert spec == teacher_audit.independent_spec("main",spec["index"])
    assert game["spec"] == spec and game["manifest"] == receipt["manifest"] == source["parent_binding"] == BREADTH_BINDING
    assert source["game"]["index"] == source["audit"]["index"] == spec["index"]
    assert receipt["passed"] and receipt["game_artifact"] == source["game"] and not receipt["full_inner_replay"]
    assert "failure" not in game and game["reason"] in old.VALID_NATURAL_TERMINATIONS
    assert len(game["trace"]) == len(receipt["roots"])
    rows, roots = [], []
    for move, checked in zip(game["trace"],receipt["roots"],strict=True):
        c.budget_boundary(base)
        assert move["turn"] == checked["turn"] and move["root_sha256"] == checked["sha256"]
        if move["actor"] != spec["candidate_seat"]:
            continue
        assert checked["planner"] == "candidate" and checked["fit"] == spec["fit"]
        item = c.artifact(teacher.root_path("main",spec["index"],move["turn"],BREADTH_SOURCE))
        assert item["sha256"] == checked["sha256"]
        root = c.verified(item)
        assert root["manifest"] == BREADTH_BINDING and "failure" not in root and root["planner"] == "candidate"
        assert root["root_identity"] == ["main",spec["index"],move["turn"]]
        assert old.digest(root["snapshot"]) == move["state_sha256"]
        assert root["candidates"]["actions"][root["choice"]] == move["action"]
        state = old.deserialize_state_snapshot(root["snapshot"])
        rows.append(make_row(state,root,spec,old._RAW._model.config))
        roots.append({**item,"turn":move["turn"]})
    path = Path(base)/"data/exports"/f"game_{spec['index']:04d}.pt"
    assert not path.exists()
    old.tensor_save(path,rows)
    exported = c.artifact(path,index=spec["index"],rows=len(rows))
    saved = torch.load(c.verify_hash(exported),map_location="cpu",weights_only=True)
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state, index, coverage = env.initial_state(), 0, Counter()
    for move in game["trace"]:
        c.budget_boundary(base)
        assert (move["turn"],move["actor"]) == (state.turn_index,state.current_player)
        assert old.digest(old.serialize_state_snapshot(state)) == move["state_sha256"]
        if state.current_player == spec["candidate_seat"]:
            assert roots[index]["sha256"] == move["root_sha256"]
            root = c.verified(roots[index])
            assert old.digest(root["snapshot"]) == move["state_sha256"]
            coverage.update(inspect_row(saved[index],state,root,spec,old._RAW._model.config))
            assert root["candidates"]["actions"][saved[index]["teacher"]] == move["action"]
            index += 1
        action = old.decode_action(move["action"])
        assert action in env.legal_actions(state)
        state = env.step(state,action)
    assert index == len(rows) == len(saved)
    assert old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"]) == game["final_state_sha256"]
    if game["reason"] == "completed":
        assert state.terminal and state.winner == game["winner"]
    else:
        assert old.adjudicate_scores(state) == game["winner"]
    audit_path = Path(base)/"data/audits"/f"game_{spec['index']:04d}.json"
    old.save_new(audit_path,{"manifest":binding,"passed":True,"source":source,"export":exported,
                 "source_roots":roots,"coverage":dict(coverage),"actual_game_replayed":True,
                 "every_row_hidden_input_invariance":True,"independent_targets":True,"continuations_replayed":0,
                 "source_continuation_verification":"inherited_completed_sampled_breadth_arena_audit"})
    return exported,c.artifact(audit_path,index=spec["index"])


def retained_shard(kind, item, checked, base=OUTPUT):
    c.budget_boundary(base)
    assert kind in ("old","new")
    receipt = c.verified(checked)
    assert receipt["passed"] and receipt["export"] == item
    values = torch.load(c.verify_hash(item),map_location="cpu",weights_only=True)
    assert len(values) == item["rows"]
    rows = []
    for row in values:
        target = row["labels"]["world128"] if kind == "old" else row["teacher"]
        assert 1 <= len(row["members"]) <= 4 and 0 <= target < len(row["members"])
        assert row["split"] == previous.independent_splits()[row["pair"]]
        rows.append({**row,"teacher":target,"narrow_teacher":target,"narrow_support":len(row["members"])})
    return rows


def retained_rows(base=OUTPUT):
    _, lock, audit = _parent_data()
    rows = []
    for kind in ("old","new"):
        for item,checked in zip(lock[f"{kind}_exports"],audit[f"{kind}_export_audits"],strict=True):
            rows.extend(retained_shard(kind,item,checked,base))
    assert len(rows) == len({row["id"] for row in rows}) == 23936
    assert Counter(row["split"] for row in rows) == {"train":19047,"validation":4889}
    return rows


def prepare_data(binding, base=OUTPUT):
    c.budget_boundary(base)
    base = Path(base)
    assert not (base/"data_lock.json").exists()
    old.initialize_worker()
    for fit_id in c.FITS:
        payload = torch.load(c.verify_hash(previous.initializer(fit_id)),map_location="cpu",weights_only=True)
        assert payload["model_config"] == asdict(old._RAW._model.config), "Source/model encoding configuration mismatch"
    assert splits() == independent_splits()
    evidence = source_evidence()
    parent_item, parent, _ = _parent_data()
    retained = retained_rows(base)
    exports,audits,coverage = [],[],Counter()
    reporter = StatusReporter(base/"data",{"manifest":binding})
    for source in source_games():
        c.budget_boundary(base)
        item,checked = export_game(source,binding,base)
        exports.append(item); audits.append(checked)
        coverage.update(c.verified(checked)["coverage"])
        if len(exports) % 20 == 0:
            reporter.event("breadth_learning_export",completed=len(exports),total=402)
    assert coverage["rows"] == 11950 and coverage["offered_actions"] == 88384
    assert coverage["candidate_continuations"] == 11313152 and coverage["support_8"] == 9819
    assert sum(coverage[f"support_{n}"] for n in range(5,9)) == 11037
    assert Counter(value for key,value in splits().items() if key.startswith("breadth128:")) == {"train":159,"validation":42}
    all_rows = retained + [row for item in exports for row in torch.load(c.verify_hash(item),map_location="cpu",weights_only=True)]
    assert len(all_rows) == len({row["id"] for row in all_rows}) == 35886
    audit_path = base/"data_audit.json"
    old.save_new(audit_path,{"manifest":binding,"passed":True,"parent_data_lock":parent_item,
                 "parent_data_audit":parent["audit"],"source_evidence":evidence,"breadth_audits":audits,
                 "coverage":dict(coverage),"retained_rows":23936,"retained_tensors_targets_splits_unchanged":True,
                 "independent_breadth_tensor_target_reconstruction":True,"continuations_replayed":0})
    old.save_new(base/"data_lock.json",{"manifest":binding,"config":CONFIG,"parent_data_lock":parent_item,
                 "parent_data_audit":parent["audit"],"source_evidence":evidence,"audit":c.artifact(audit_path),
                 "exports":exports,"splits":splits(),"rows":35886,
                 "train_rows":19047+coverage["train"],"validation_rows":4889+coverage["validation"],
                 "ordered_row_ids_sha256":old.digest([row["id"] for row in all_rows])})
    return c.artifact(base/"data_lock.json")


def validate_data(binding, base=OUTPUT):
    c.budget_boundary(base)
    lock = old.read_json(Path(base)/"data_lock.json")
    assert lock["manifest"] == binding and lock["config"] == CONFIG and lock["splits"] == splits() == independent_splits()
    parent_item,parent,_ = _parent_data()
    assert lock["parent_data_lock"] == parent_item and lock["parent_data_audit"] == parent["audit"]
    audit = c.verified(lock["audit"])
    assert audit["manifest"] == binding and audit["passed"] and audit["parent_data_lock"] == parent_item
    assert audit["source_evidence"] == lock["source_evidence"] and audit["retained_rows"] == 23936
    assert audit["retained_tensors_targets_splits_unchanged"] and audit["independent_breadth_tensor_target_reconstruction"]
    for item in lock["source_evidence"]:
        c.verify_hash(item)
    assert len(lock["exports"]) == len(audit["breadth_audits"]) == 402
    for index,(item,checked) in enumerate(zip(lock["exports"],audit["breadth_audits"],strict=True)):
        c.budget_boundary(base)
        receipt = c.verified(checked)
        assert item["index"] == checked["index"] == index and receipt["export"] == item
        assert receipt["manifest"] == binding and receipt["passed"] and receipt["actual_game_replayed"]
        assert receipt["independent_targets"] and receipt["every_row_hidden_input_invariance"]
        c.verify_hash(item)
        c.verify_hash(receipt["source"]["game"]); c.verify_hash(receipt["source"]["audit"])
    assert lock["rows"] == 35886 and lock["train_rows"]+lock["validation_rows"] == 35886
    return lock


def load_data(binding, base=OUTPUT):
    lock = validate_data(binding,base)
    rows = retained_rows(base)
    for item in lock["exports"]:
        values = torch.load(c.verify_hash(item),weights_only=True,map_location="cpu")
        assert len(values) == item["rows"]
        rows.extend(values)
    assert len(rows) == len({row["id"] for row in rows}) == lock["rows"]
    assert old.digest([row["id"] for row in rows]) == lock["ordered_row_ids_sha256"]
    assert all(row["split"] == lock["splits"][row["pair"]] for row in rows)
    assert Counter(row["split"] for row in rows) == {"train":lock["train_rows"],"validation":lock["validation_rows"]}
    return rows

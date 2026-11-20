"""Strict corrected-game loader for the first, separately qualified learning pilot.

Imports and preparation are model-free. No historical training loader is used.
The caller owns the frozen game inventory and all real-clock/resource guards.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit, rules
from tools.external_strength_screen_20261018_02 import bridge

ROOT = Path(__file__).resolve().parents[2]
GAME_SCHEMA = "selfplay_training_game_v1"
CORPUS_SCHEMA = "selfplay_corpus_v1"
QUALIFICATION_SPLITS = ("fixture_train", "fixture_validation", "fixture_assessment")
PILOT_SPLITS = ("train", "validation", "offline")
IDENTITY = {"ruleset_id": guard.RULESET_ID, "catalogue_sha256": guard.CATALOGUE_SHA256,
            "turn_contract": guard.TURN_CONTRACT, "origin": "generated_under_declared_rules"}
MAX_GAME_BYTES = 64 * 1024**2
MAX_PLIES = 2000


def require(value, message):
    if not value:
        raise ValueError("Self-play data: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def verify_identity(document):
    require(type(document) is dict, "declared corrected identity required")
    for key, value in IDENTITY.items():
        require(document.get(key) == value, "missing or incompatible " + key)


def read_artifact(item, *, boundary=lambda: None):
    require(type(item) is dict and set(item) == {"path", "bytes", "sha256"}, "exact artifact fields")
    require(type(item["bytes"]) is int and 0 <= item["bytes"] <= MAX_GAME_BYTES, "artifact size bound")
    path = Path(item["path"])
    require(not path.is_absolute(), "workspace-relative artifact required")
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.relative_to(ROOT).as_posix() == item["path"],
            "canonical workspace artifact path")
    boundary()
    before = path.stat()
    require(before.st_size == item["bytes"], "artifact size differs")
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "artifact changed while reading")
    require(hashlib.sha256(raw).hexdigest() == item["sha256"], "artifact hash differs")
    boundary()
    return json.loads(raw)


def build_manifest(game_descriptors, expected_specs, *, qualification_only=False):
    """Bind exact scheduled files, never infer a corpus by globbing successes."""
    core = {"schema": CORPUS_SCHEMA, "qualification_only": qualification_only,
            "teacher_simulations": 512, "specs": deepcopy(expected_specs),
            "games": deepcopy(game_descriptors)}
    metadata = {"schema": "corrected_training_data_v1", **IDENTITY, "data_sha256": digest(core)}
    return {**core, "metadata": metadata}


def validate_manifest(manifest, expected_specs):
    require(type(manifest) is dict and set(manifest) ==
            {"schema", "qualification_only", "teacher_simulations", "specs", "games", "metadata"},
            "exact corpus manifest fields")
    require(manifest["schema"] == CORPUS_SCHEMA and type(manifest["qualification_only"]) is bool,
            "corpus schema and qualification scope")
    require(type(manifest["teacher_simulations"]) is int and manifest["teacher_simulations"] == 512,
            "fixed512 source teacher")
    guard.require_data_contract(manifest["metadata"])
    core = {key: value for key, value in manifest.items() if key != "metadata"}
    require(digest(core) == manifest["metadata"]["data_sha256"], "corpus inventory digest")
    require(canonical(manifest["specs"]) == canonical(expected_specs), "exact prospectively frozen game specs")
    require(type(expected_specs) is list and bool(expected_specs), "nonempty scheduled corpus")
    require(len(manifest["games"]) == len(expected_specs), "every scheduled game has one file")
    require(len({x["path"] for x in manifest["games"]}) == len(expected_specs), "no duplicate game files")
    indices, seeds = [], []
    for spec in expected_specs:
        require(type(spec) is dict and type(spec.get("game_index")) is int and spec["game_index"] >= 0,
                "explicit game identity")
        require(type(spec.get("seed_decimal")) is str and spec["seed_decimal"].isdigit(), "decimal setup seed")
        require(spec.get("teacher_simulations") == 512, "spec teacher quota")
        require(spec.get("split") in (QUALIFICATION_SPLITS if manifest["qualification_only"] else PILOT_SPLITS),
                "explicit disjoint data split")
        if manifest["qualification_only"]:
            require(spec.get("purpose") == "engineering_qualification_only", "qualification data never pilot data")
        else:
            require(spec.get("purpose") == "single_generation_learning_pilot", "fresh pilot data purpose")
        indices.append(spec["game_index"]); seeds.append(int(spec["seed_decimal"]))
    require(len(set(indices)) == len(indices) and len(set(seeds)) == len(seeds), "independent game/seed inventory")
    expected_counts = {"fixture_train": 2, "fixture_validation": 1, "fixture_assessment": 1} if manifest["qualification_only"] else {
        "train": 192, "validation": 32, "offline": 32}
    require(dict(Counter(s["split"] for s in expected_specs)) == expected_counts, "fixed split game counts")
    return core


def signed_outcomes(final_snapshot, reason, actor):
    require(type(actor) is int and actor in (0, 1), "absolute actor seat")
    winner = guard.require_official_terminal_target(final_snapshot, reason)
    require(winner is None or type(winner) is int and winner in (0, 1), "official winner or draw")
    outcome = 0.0 if winner is None else (1.0 if winner == actor else -1.0)
    # Native transfer::encode puts the current actor in canonical player slot0.
    return [outcome, -outcome if outcome else 0.0]


def check_encoded_costs(snapshot, tokens):
    """Inspect actual loader-bound token offsets, not a separate legacy encoder."""
    state = guard.decode_training_snapshot(snapshot)
    require(type(tokens) is list and len(tokens) == 31 * 48 and all(
        type(x) in (int, float) and math.isfinite(x) for x in tokens), "finite31x48 root tensor")
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    checks = []
    for tier in (1, 2, 3):
        for slot, card in enumerate(state.visible_tier_cards[tier]):
            token = 3 + 4 * (tier - 1) + slot
            expected = [f32(f32(card.cost.get(c, 0)) * f32(.1)) for c in bridge.COLORS]
            require(tokens[token * 48:token * 48 + 5] == expected,
                    "actual root tensor corrected costs: " + card.card_id)
            checks.append({"card_id": card.card_id, "costs": expected, "token_index": token})
    return checks


def decode_row(row, *, expected_spec=None, expected_ply=None, expected_quota=512):
    """Actual collector/loader path, shared by qualification and later fitting."""
    from .capture_adapter import validate_capture, public_root_tokens
    require(type(row) is dict, "game row object")
    required = {"spec", "ply", "before", "observer", "action", "after", "training_capture"}
    require(required <= set(row), "complete row and root capture required")
    if expected_spec is not None:
        require(canonical(row["spec"]) == canonical(expected_spec), "row spec binding")
    require(type(row["ply"]) is int and row["ply"] >= 0, "nonnegative ply")
    if expected_ply is not None:
        require(row["ply"] == expected_ply, "contiguous row ply")
    state = guard.decode_training_snapshot(row["before"])
    require(not state.terminal, "no decision after terminal")
    observer = rules.observation_payload(state)
    require(canonical(row["observer"]) == canonical(observer), "actual corrected lawful observer")
    audit.check_observer(row["before"], row["observer"])
    legal = guard.legal_actions_for_snapshot(row["before"])
    action = rules.deserialize_action(row["action"])
    require(action in legal, "complete corrected legal action")
    guard.check_training_transition(row["before"], row["action"], row["after"])
    capture = validate_capture(row["training_capture"], observer, expected_quota=expected_quota)
    require(type(capture) is dict and type(capture.get("eligible")) is bool, "capture eligibility contract")
    trace = row.get("trace")
    require(type(trace) is dict and type(trace.get("phases")) is list, "complete source scientific trace")
    if capture["exclusion_reason"] == "forced_pass":
        require(trace.get("forced_pass") is True and trace["phases"] == [], "forced-pass capture/trace binding")
    else:
        require(bool(trace["phases"]), "source Main trace")
        main = trace["phases"][0]
        require(main["action"] == capture["main_action"] == bridge._main(action) and
                main["neural_callback_count"] == capture["callback_count"] and
                main["simulations"] == capture["native_capture"]["simulations"],
                "same-source captured action, callbacks and native work")
    if not capture["eligible"]:
        return {"eligible": False, "reason": capture["exclusion_reason"], "ply": row["ply"], "actor": state.current_player}
    tokens = capture["root_tokens"]
    callbacks = main.get("inference")
    require(type(callbacks) is list and len(callbacks) == capture["callback_count"] and callbacks[0]["batch"] == 1,
            "complete raw callback evidence includes actual root")
    root = callbacks[0]
    require(root["tokens"] == tokens and root["outputs"] == [capture["root_output83"]] and
            root["input_sha256"] == digest(tokens) and root["output_sha256"] == digest(root["outputs"]),
            "captured root arrays bind original same-decision inference evidence")
    require(tokens == public_root_tokens(observer), "independent lawful public root encoding")
    checks = check_encoded_costs(row["before"], tokens)
    require(capture["authoritative_actions"] == bridge.main_actions(observer), "loader's independent Main support")
    mask, targets = capture["legal_mask81"], capture["policy_target81"]
    require(type(mask) is list and len(mask) == 81 and all(type(x) is bool for x in mask) and any(mask),
            "full legal Main mask")
    require(type(targets) is list and len(targets) == 81 and all(type(x) in (int, float) and
            math.isfinite(x) and x >= 0 for x in targets), "finite policy targets")
    require(abs(sum(targets) - 1.0) <= 1e-12 and all(ok or v == 0 for ok, v in zip(mask, targets)),
            "normalized target supported on all legal Main channels")
    return {"eligible": True, "ply": row["ply"], "actor": state.current_player,
            "tokens": deepcopy(tokens), "policy": deepcopy(targets), "mask": deepcopy(mask),
            "canonical_players": [state.current_player, 1 - state.current_player],
            "observer_sha256": digest(observer), "row_sha256": digest(row),
            "root_input_sha256": capture["root_input_sha256"], "encoded_cost_checks": checks,
            "authoritative_legal_actions": [rules.serialize_action(a) for a in legal]}


def load_game(document, expected_spec, *, boundary=lambda: None):
    verify_identity(document)
    require(document.get("schema") == GAME_SCHEMA, "game schema")
    require(canonical(document.get("spec")) == canonical(expected_spec), "game belongs to frozen split/seed")
    require(document.get("reason") == "completed", "only official natural completed games can train")
    initial = document["initial"]
    guard.decode_training_snapshot(initial)
    require(initial == rules.serialize_snapshot(rules.CorrectedEnv(seed=int(expected_spec["seed_decimal"])).initial_state()),
            "actual source initial state belongs to the frozen independent seed")
    rows = document["rows"]
    require(type(rows) is list and 0 < len(rows) <= MAX_PLIES, "complete bounded nonempty game")
    state, examples, exclusions = initial, [], Counter()
    for ply, row in enumerate(rows):
        boundary()
        require(canonical(row["before"]) == canonical(state), "contiguous independent trajectory")
        require(type(row.get("clock")) is dict and row["clock"].get("status") == "ok" and
                row["clock"].get("accepted") is True, "completed accepted teacher decision")
        raw = row["clock"].get("result")
        require(type(raw) is dict and raw.get("action") == row["action"] and
                raw.get("diagnostics") == row.get("trace"), "raw action and full diagnostics binding")
        decoded = decode_row(row, expected_spec=expected_spec, expected_ply=ply)
        if decoded["eligible"]:
            decoded.update(game_index=expected_spec["game_index"], split=expected_spec["split"])
            examples.append(decoded)
        else:
            exclusions[decoded["reason"]] += 1
        state = row["after"]
    require(canonical(state) == canonical(document["final"]), "exact final snapshot")
    winner = guard.require_official_terminal_target(state, document["reason"])
    require(document.get("winner") == winner and
            (winner is None or type(document.get("winner")) is int), "verified final winner")
    require(bool(examples), "every scheduled source game must have at least one eligible root")
    for example in examples:
        example["value"] = signed_outcomes(state, document["reason"], example["actor"])
    return {"spec": deepcopy(expected_spec), "examples": examples, "plies": len(rows),
            "eligible_roots": len(examples), "exclusions": dict(exclusions), "winner": winner,
            "initial_sha256": digest(initial), "final_sha256": digest(state), "game_sha256": digest(document)}


def load_corpus(manifest, *, expected_specs, expected_quota=512, include_splits=None, boundary=lambda: None):
    require(expected_quota == 512, "fixed512 teacher")
    validate_manifest(manifest, expected_specs)
    allowed = QUALIFICATION_SPLITS if manifest["qualification_only"] else PILOT_SPLITS
    selected = allowed if include_splits is None else tuple(include_splits)
    require(bool(selected) and len(set(selected)) == len(selected) and set(selected) <= set(allowed), "explicit loader split selection")
    games, examples = [], []
    for spec, descriptor in zip(expected_specs, manifest["games"], strict=True):
        if spec["split"] not in selected:
            continue
        boundary()
        result = load_game(read_artifact(descriptor, boundary=boundary), spec, boundary=boundary)
        examples.extend(result.pop("examples")); games.append(result)
    require(len(games) == sum(s["split"] in selected for s in expected_specs), "no dropped or substituted selected games")
    return {"schema": "selfplay_loaded_corpus_v1", "qualification_only": manifest["qualification_only"],
            "manifest_sha256": digest(manifest), "manifest": deepcopy(manifest),
            "opened_splits": list(selected),
            "games": games, "examples": examples, "game_counts": dict(Counter(x["spec"]["split"] for x in games)),
            "example_counts": dict(Counter(x["split"] for x in examples)), **IDENTITY}


def build_qualification_cases():
    from tools.splendorust_search512_20261101.corpus import build_fixtures, hidden_variant
    cases = deepcopy(build_fixtures())
    for case in cases:
        case["observer"] = rules.observation_payload(guard.decode_training_snapshot(case["snapshot"]))
    private = next(x for x in cases if "private" in x["id"])
    changed = hidden_variant(guard.decode_training_snapshot(private["snapshot"]))
    require(rules.observation_payload(changed) == private["observer"], "private invariance fixture")
    return {"schema": "selfplay_loader_qualification_cases_v1", "cases": cases,
            "private_variant": {"source_id": private["id"], "snapshot": rules.serialize_snapshot(changed),
                                "observer": private["observer"]},
            "qualification_only": True, "pilot_data": False, **IDENTITY}


def qualify_captured_rows(plan, rows, *, boundary=lambda: None):
    """Actual captured rows must traverse the real loader, not a helper-only test."""
    require(canonical(plan) == canonical(build_qualification_cases()), "fixed actual-loader counterexamples")
    require(type(rows) is list and len(rows) == len(plan["cases"]) + 1, "all structural cases and hidden repeat")
    evidence, face_ids, eligible = [], set(), []
    for index, case in enumerate(plan["cases"] + [plan["private_variant"]]):
        boundary(); row = rows[index]
        require(row["before"] == case["snapshot"], "actual qualification snapshot binding")
        decoded = decode_row(row)
        if decoded["eligible"]:
            eligible.append(decoded)
            face_ids.update(x["card_id"] for x in decoded["encoded_cost_checks"] if x["card_id"] in rules.CORRECTED_COSTS)
        evidence.append({"case_id": case.get("id", "private_hidden_repeat"),
                         "eligible": decoded["eligible"], "loader_result_sha256": digest(decoded),
                         "root_input_sha256": decoded.get("root_input_sha256"),
                         "encoded_cost_checks": decoded.get("encoded_cost_checks", [])})
    require(face_ids == set(rules.CORRECTED_COSTS), "all six corrected faces checked in actual bound tensors")
    original = next(i for i, x in enumerate(plan["cases"]) if x["id"] == plan["private_variant"]["source_id"])
    require(rows[original]["training_capture"] == rows[-1]["training_capture"], "actual loader private-world invariance")
    # Exactly the collator used by fitting; these are inputs/policy targets only,
    # not fabricated terminal outcomes or extra optimization examples.
    batch = collate_arrays(eligible, [1.] * len(eligible))
    actual_rows = batch["tokens"].reshape(len(eligible), -1).tolist()
    require(actual_rows == [x["tokens"] for x in eligible] and batch["mask"].tolist() == [x["mask"] for x in eligible],
            "actual encoded training minibatch matches decoded root inputs and masks")
    rejected = qualify_row_rejections(rows[0], boundary=boundary)
    return {"status": "actual_loader_counterexamples_passed", "rows": evidence,
            "six_corrected_faces": sorted(face_ids), "actual_training_loader_used": True,
            "historical_training_loader_used": False, "model_calls_in_verifier": 0,
            "actual_input_policy_batch": {"shape": list(batch["tokens"].shape), "rows": len(eligible),
                "tokens_sha256": digest(actual_rows), "terminal_labels_fabricated": False, "optimizer_updates": 0},
            "actual_loader_rejections": rejected}


def collate_arrays(examples, weights):
    """Shared input collation; the supervisor never imports Torch or a model."""
    import numpy as np
    require(0 < len(examples) == len(weights) <= 32, "bounded actual training batch")
    tokens = np.asarray([x["tokens"] for x in examples], dtype=np.float32).reshape(-1, 31, 48)
    policy = np.asarray([x["policy"] for x in examples], dtype=np.float32)
    mask = np.asarray([x["mask"] for x in examples], dtype=np.bool_)
    w = np.asarray(weights, dtype=np.float32)
    require(policy.shape == mask.shape == (len(examples), 81) and w.shape == (len(examples),), "actual batch shapes")
    require(np.isfinite(tokens).all() and np.isfinite(policy).all() and np.isfinite(w).all() and (w > 0).all(), "finite batch")
    require(mask.any(axis=1).all() and (policy >= 0).all() and (policy[~mask] == 0).all() and
            np.allclose(policy.sum(axis=1), np.ones(len(examples)), rtol=0, atol=1e-6), "all legal Main mask and normalized targets")
    return {"tokens": tokens, "policy": policy, "mask": mask, "weights": w}


def qualify_row_rejections(source, *, boundary=lambda: None):
    """Run fixed negative mutations through the actual decode_row entry point."""
    cases = []
    for key in ("schema", "ruleset_id", "catalogue_sha256"):
        bad = deepcopy(source); bad["before"].pop(key); cases.append(("missing_snapshot_" + key, bad))
        bad = deepcopy(source); bad["before"][key] = "historical"; cases.append(("old_snapshot_" + key, bad))
    bad = deepcopy(source); bad["before"] = bad["before"]["state"]; cases.append(("bare_historical_snapshot", bad))
    bad = deepcopy(source); bad["observer"]["catalogue_sha256"] = "0" * 64; cases.append(("wrong_observer_catalogue", bad))
    bad = deepcopy(source); bad["training_capture"]["root_input_sha256"] = "0" * 64; cases.append(("wrong_root_hash", bad))
    bad = deepcopy(source); bad["training_capture"]["root_tokens"][0] += .25; cases.append(("changed_root_tensor", bad))
    bad = deepcopy(source); bad["training_capture"]["legal_mask81"][0] = not bad["training_capture"]["legal_mask81"][0]; cases.append(("wrong_legal_mask", bad))
    bad = deepcopy(source); bad["training_capture"]["visits67"][0] += 1; cases.append(("changed_visit_targets", bad))
    bad = deepcopy(source); bad["training_capture"]["root_output83"][0] += .125; cases.append(("unbound_root_output", bad))
    result = []
    for name, bad in cases:
        boundary()
        try:
            decode_row(bad)
        except (ValueError, KeyError, AssertionError, TypeError):
            result.append({"case": name, "rejected": True})
        else:
            raise ValueError("Self-play data: actual loader accepted " + name)
    return result


def qualify_manifest_rejections(manifest, expected_specs, *, boundary=lambda: None):
    cases = []
    for key in ("ruleset_id", "catalogue_sha256", "turn_contract", "origin", "data_sha256"):
        bad = deepcopy(manifest); bad["metadata"].pop(key)
        cases.append(("missing_corpus_" + key, bad))
    bad = deepcopy(manifest); bad["games"][0]["sha256"] = "0" * 64
    bad["metadata"]["data_sha256"] = digest({k: v for k, v in bad.items() if k != "metadata"})
    cases.append(("actual_corpus_file_hash_mismatch", bad))
    result = []
    for name, bad in cases:
        boundary()
        try:
            load_corpus(bad, expected_specs=expected_specs, include_splits=["fixture_train"], boundary=boundary)
        except (ValueError, KeyError, AssertionError, TypeError):
            result.append({"case": name, "rejected": True})
        else:
            raise ValueError("Self-play data: actual loader accepted " + name)
    return result

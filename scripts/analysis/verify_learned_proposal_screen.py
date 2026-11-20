"""Independent source/proposal/world/full-trajectory/statistics reconstruction.

Shares the frozen rules engine, canonical grouping, encoding primitives and
neural arithmetic. Does not use the production proposer, selector or statistics.
"""
from collections import Counter
import hashlib
import math
import random
import time

import numpy as np
import torch

from scripts.analysis import learned_proposal_screen as p
from scripts.analysis import verify_afterstate_advantage_149 as replay
from scripts.analysis import verify_rollout_selector_154 as precedent
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE


def compare(left, right):
    if isinstance(left, dict):
        p.require(set(left) == set(right), "Independent key mismatch")
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        p.require(len(left) == len(right), "Independent list length")
        for a, b in zip(left, right):
            compare(a, b)
    elif isinstance(left, float):
        p.require(math.isfinite(left) and math.isfinite(right) and abs(left - right) <= 1e-12,
                  f"Independent numeric mismatch: {left}, {right}")
    else:
        p.require(left == right, f"Independent mismatch: {left}, {right}")


def source_check(stage, item, binding):
    start, cpu = time.perf_counter(), time.process_time()
    record = p.verified(item)
    index = item["index"]
    expected_seed = (2039000000 if stage == "probe" else 2030000000) + index
    p.require(record["manifest"] == binding and record["stage"] == stage
              and record["index"] == index and record["seed"] == expected_seed, "Source identity")
    env = p.old.SplendorEnv(seed=expected_seed)
    state = env.initial_state()
    counts, seen, hasher, best = [0] * 6, Counter(), hashlib.sha256(), {}
    passes = idle = 0
    for i, row in enumerate(record["trace"]):
        p.require(not state.terminal and state.turn_index < 400, "Source continues after ending")
        seen[p.old.state_signature(state)] += 1
        p.require(seen[p.old.state_signature(state)] < 4, "Source repetition")
        snapshot = p.old.serialize_state_snapshot(state)
        p.require(row["state_hash"] == p.old.digest(snapshot) and row["turn"] == state.turn_index
                  and row["actor"] == state.current_player, "Source state identity")
        if all(player.reserved_card_source(j) == RESERVE_SOURCE_VISIBLE for player in state.players for j in range(len(player.reserved_cards))):
            phase = 0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2
            cell = phase * 2 + state.current_player
            counts[cell] += 1
            key = p.old.digest(["learned-proposal-position-v1", stage, index, state.turn_index])
            if str(cell) not in best or key < best[str(cell)]["key"]:
                best[str(cell)] = {"key": key, "turn": state.turn_index, "snapshot_sha256": p.old.digest(snapshot)}
        legal = env.legal_actions(state)
        values = p.old._RAW.score_actions(state, legal)
        p.require(legal and len(values) == len(legal) and all(math.isfinite(v) for v in values), "Invalid source scores")
        action = legal[min(range(len(legal)), key=lambda j: (-values[j], j))]
        p.require(p.old.serialize_action(action) == row["action"], "Source not greedy035")
        hasher.update(p.old.digest(row["action"]).encode())
        successor = env.step(state, action)
        passes = passes + 1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if p.old.is_progress_transition(state, successor, action) else idle + 1
        state = successor
        if i < len(record["trace"]) - 1:
            p.require(passes < 2 and idle < 60, "Source ignored cutoff")
    p.require(len(record["trace"]) == record["turns"] and record["trace_sha256"] == hasher.hexdigest(), "Source trace hash")
    p.require(p.old.digest(p.old.serialize_state_snapshot(state)) == p.old.digest(record["final_state"])
              == record["final_state_sha256"], "Source final state")
    if record["reason"] == "completed":
        p.require(state.terminal and idle < 60, "False completion")
        winner = state.winner
    else:
        p.require(record["reason"] == "forced_pass_deadlock" and passes == 2, "Invalid source termination")
        winner = p.old.adjudicate_scores(state)
    p.require(winner == record["winner"], "Wrong source winner")
    path = p.path_for(stage, "sources_audit", index)
    p.old.save_new(path, {"manifest": binding, "index": index, "source": item, "passed": True,
                         "eligible": counts, "best": best, "moves": record["turns"],
                         "timing": {"wall": time.perf_counter() - start, "cpu": time.process_time() - cpu}})
    return p.artifact(path, index=index)


def verify_matching(stage, panel, audit_items, source_items):
    quota = 3 if stage == "probe" else 48
    buckets, allowed, prefix = [[] for _ in range(6)], {}, []
    audits = [p.verified(item) for item in audit_items]

    def visit(game, touched):
        for cell in allowed[game]:
            if cell in touched:
                continue
            touched.add(cell)
            if len(buckets[cell]) < quota:
                buckets[cell].append(game)
                return True
            for slot, previous in enumerate(buckets[cell]):
                if visit(previous, touched):
                    buckets[cell][slot] = game
                    return True
        return False

    for index, row in enumerate(audits):
        p.require(row["index"] == index and row["passed"] and row["source"] == source_items[index], "Source audit link")
        allowed[index] = [c for c in range(6) if row["eligible"][c] > 0]
        prefix.append({"source_index": index, "eligible_by_cell": row["eligible"], "sha256": row["source"]["sha256"]})
        visit(index, set())
        if all(len(bucket) == quota for bucket in buckets):
            break
    p.require([len(bucket) for bucket in buckets] == [quota] * 6, "Independent source matching failed")
    assignments = sorted((i, c) for c, bucket in enumerate(buckets) for i in bucket)
    compare(panel["filtering"], prefix)
    compare(panel["counts"], [quota] * 6)
    p.require(len(panel["samples"]) == quota * 6 and len({s["source_seed"] for s in panel["samples"]}) == quota * 6,
              "Panel source independence")
    for index, (sample, (source_index, cell)) in enumerate(zip(panel["samples"], assignments)):
        expected_seed = (2039000000 if stage == "probe" else 2030000000) + source_index
        best = audits[source_index]["best"][str(cell)]
        p.require(sample["index"] == index and sample["source_index"] == source_index
                  and sample["source_seed"] == expected_seed and sample["cell"] == cell
                  and sample["stage"] == stage and sample["turn"] == best["turn"]
                  and p.old.digest(sample["snapshot"]) == best["snapshot_sha256"]
                  and sample["actor"] == cell % 2
                  and sample["stream"] == p.old.digest(["learned-proposal-root-v1", stage, source_index]), "Panel sample identity")


def independent_proposals(state, stream):
    legal = p.old.SplendorEnv().legal_actions(state)
    raw = list(map(float, p.old._RAW.score_actions(state, legal)))
    group_keys, members, reps = [], {}, {}
    for index, action in enumerate(legal):
        key = repr(p.old.canonical_successor_action_key(state, action))
        if key not in members:
            group_keys.append(key)
            members[key] = []
            reps[key] = p.old.serialize_action(action)
        members[key].append(index)
    order = []
    top_reps = {}
    for index in sorted(range(len(legal)), key=lambda j: (-raw[j], j)):
        key = repr(p.old.canonical_successor_action_key(state, legal[index]))
        if key not in order:
            order.append(key)
            top_reps[key] = p.old.serialize_action(legal[index])
    belief = p.old.digest(canonical_public_belief_key(state, state.current_player))
    nonce = p.old.digest(["155-candidate-v1", stream, belief])
    schedule = {"belief_sha256": belief, "candidate_nonce": nonce,
                "world_seeds": [f"155-world-v1|{stream}|{belief}|{j:02d}" for j in range(32)]}
    chosen = order[:2]
    reps.update({key: top_reps[key] for key in chosen})
    remaining = [key for key in group_keys if key not in chosen]
    rng = random.Random(f"149-candidates-v1|155-live-v1|{nonce}")
    chosen += rng.sample(remaining, min(2, len(remaining)))
    reference = {"keys": chosen, "actions": [reps[key] for key in chosen],
                 "canonical_support": len(group_keys), "legal_support": len(legal)}
    logits, grouped = {}, {}
    for seed, model in p._MODELS.items():
        obs = encode_public_observation_tensor(state, state.current_player)
        features = encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)
        observation = torch.tensor(fit_feature_vector(obs.vector, model.config.observation_size), dtype=torch.float32)
        inputs = torch.tensor([fit_feature_vector(row.vector, model.config.action_feature_size) for row in features], dtype=torch.float32)
        with torch.no_grad():
            values, _ = model(observation[None], inputs[None], torch.ones((1, len(legal)), dtype=torch.bool))
        p.require(torch.isfinite(values).all().item(), "Invalid independent logits")
        logits[str(seed)] = values[0].tolist()
        grouped[str(seed)] = [max(logits[str(seed)][j] for j in members[key]) for key in group_keys]
    arm_keys = {"reference": list(chosen)}
    if len(chosen) < 4:
        for arm in p.ARMS[1:]:
            arm_keys[arm] = list(chosen)
    else:
        common = chosen[:3]
        arm_keys["raw_control"] = common + [next(key for key in order if key not in common)]
        for seed in p.MODELS:
            choices = [j for j, key in enumerate(group_keys) if key not in common]
            j = min(choices, key=lambda j: (-grouped[str(seed)][j], j))
            arm_keys[f"learned_{seed}"] = common + [group_keys[j]]
    union = []
    for arm in p.ARMS:
        for key in arm_keys[arm]:
            if key not in union:
                union.append(key)
    return {"schedule": schedule, "reference": reference, "group_keys": group_keys,
            "representatives": [reps[key] for key in group_keys], "raw_rank": order,
            "raw_scores": raw, "learned_logits": logits, "learned_group_scores": grouped,
            "legal_actions": [p.old.serialize_action(a) for a in legal],
            "union_keys": union, "union_actions": [reps[key] for key in union],
            "arm_keys": arm_keys, "arm_indices": {arm: [union.index(key) for key in arm_keys[arm]] for arm in p.ARMS}}


def proposal_check(stage, sample, item, binding):
    start, cpu = time.perf_counter(), time.process_time()
    record = p.verified(item)
    p.require(record["manifest"] == binding and record["index"] == sample["index"]
              and record["sample_sha256"] == p.old.digest(sample), "Proposal identity")
    state = p.old.deserialize_state_snapshot(sample["snapshot"])
    expected = independent_proposals(state, sample["stream"])
    compare({key: record[key] for key in expected}, expected)
    # A changed lawful hidden allocation/deck ordering cannot alter proposals/logits.
    other = precedent.independent_world(state, 2039300000 + (1000 if stage == "main" else 0) + sample["index"])
    compare(expected, independent_proposals(other, sample["stream"]))
    p.require(all(len(v) == min(4, len(expected["group_keys"])) and len(set(v)) == len(v)
                  for v in expected["arm_indices"].values()), "Duplicate/unequal offered support")
    path = p.path_for(stage, "proposals_audit", sample["index"])
    p.old.save_new(path, {"manifest": binding, "index": sample["index"], "proposal": item,
                         "passed": True, "lawful_hidden_world_invariance": True,
                         "timing": {"wall": time.perf_counter() - start, "cpu": time.process_time() - cpu}})
    return p.artifact(path, index=sample["index"])


def independent_choice(record, proposal):
    p.require(record["action_indices"] == list(range(len(proposal["union_actions"])))
              and len(record["worlds"]) == 32, "Independent selection budget")
    means = []
    for action in record["action_indices"]:
        scores = [world["outcomes"][action]["wdl"] for world in record["worlds"]]
        p.require(all(v in (-1., 0., 1.) for v in scores), "Invalid selection score")
        means.append(sum(scores) / 32)
    selected = {}
    for arm in p.ARMS:
        candidates = proposal["arm_indices"][arm]
        # Iterate arm order, not union order, on exact ties.
        chosen = candidates[0]
        for action in candidates[1:]:
            if means[action] > means[chosen]:
                chosen = action
        selected[arm] = chosen
    return {"choices": selected, "union_selection_wdl": means}


def continuation_check(stage, kind, sample, proposal_item, target_item, choices, binding):
    start, cpu = time.perf_counter(), time.process_time()
    offered, record = p.verified(proposal_item), p.verified(target_item)
    p.require(record["manifest"] == binding and record["index"] == sample["index"]
              and record["sample_sha256"] == p.old.digest(sample) and record["proposal"] == proposal_item
              and record["stage"] == stage and record["kind"] == kind and record["locked_choices"] == choices,
              "Target identity/dependencies")
    if kind == "selection":
        p.require(choices is None, "Future choice leakage")
        indices, worlds = list(range(len(offered["union_actions"]))), 32
    else:
        sealed = p.old.read_json(p.OUTPUT / stage / "choices.json")
        p.require(sealed["manifest"] == binding and sealed["rows"][sample["index"]]["choices"] == choices,
                  "Evaluation not bound to global selection lock")
        indices, worlds = sorted(set(choices.values())), 128
    p.require(record["action_indices"] == indices and len(record["worlds"]) == worlds, "Target budget/support")
    seed_base = {("probe", "selection"): 2039100000, ("probe", "evaluation"): 2039200000,
                 ("main", "selection"): 2031000000, ("main", "evaluation"): 2032000000}[stage, kind]
    state = p.old.deserialize_state_snapshot(sample["snapshot"])
    moves = 0
    for index, saved in enumerate(record["worlds"]):
        seed = seed_base + sample["index"] * 1000 + index
        p.require(saved["seed"] == seed and len(saved["outcomes"]) == len(indices), "Target world identity")
        world = precedent.independent_world(state, seed)
        p.require(p.old.digest(p.old.serialize_state_snapshot(world)) == saved["world_sha256"], "Independent world mismatch")
        for slot, action_index in enumerate(indices):
            action = p.old.decode_action(offered["union_actions"][action_index])
            p.require(action in p.old.SplendorEnv().legal_actions(world), "Illegal forced action")
            rebuilt = replay.replay_branch(world, action, state.current_player)
            compare(rebuilt, saved["outcomes"][slot])
            p.require(rebuilt["reason"] in p.old.VALID_NATURAL_TERMINATIONS, "Invalid inner termination")
            moves += rebuilt["turns"]
    path = p.path_for(stage, f"{kind}_audit", sample["index"])
    p.old.save_new(path, {"manifest": binding, "index": sample["index"], "target": target_item,
                         "proposal": proposal_item, "passed": True, "continuations": len(indices) * worlds,
                         "moves": moves, "full_inner_replay": True,
                         "timing": {"wall": time.perf_counter() - start, "cpu": time.process_time() - cpu}})
    return p.artifact(path, index=sample["index"], continuations=len(indices) * worlds, moves=moves)


def independent_rows(samples, offers, choices, evaluations):
    result = []
    for sample, offered, choice, record in zip(samples, offers, choices, evaluations):
        positions = {action: j for j, action in enumerate(record["action_indices"])}
        scores = {}
        for arm in p.ARMS:
            slot = positions[choice["choices"][arm]]
            halfpoints = sum(int(w["outcomes"][slot]["wdl"] + 1) for w in record["worlds"])
            scores[arm] = halfpoints / 256
        average = sum(scores[arm] for arm in p.ARMS[2:]) / 3
        result.append({"index": sample["index"], "source_seed": sample["source_seed"], "cell": sample["cell"],
                       "scores": scores, "learned_mean": average, "primary": average - scores["reference"],
                       "attribution": average - scores["raw_control"], "choices": choice["choices"],
                       "canonical_support": offered["reference"]["canonical_support"],
                       "legal_support": offered["reference"]["legal_support"], "arm_indices": offered["arm_indices"],
                       "proposal_timing": offered["timing"],
                       "fourth_raw_ranks": {arm: offered["raw_rank"].index(offered["arm_keys"][arm][-1]) + 1 for arm in p.ARMS}})
    return result


def independent_statistics(rows):
    p.require(len(rows) == 288 and len({r["source_seed"] for r in rows}) == 288, "Stats source units")
    strata = [[r for r in rows if r["cell"] == c] for c in range(6)]
    p.require(all(len(s) == 48 for s in strata), "Stats strata")
    data = [np.array([[r["primary"], r["attribution"]] for r in s]) for s in strata]
    rng = np.random.default_rng(2033000000)
    bootstrap = np.zeros((50000, 2))
    for start in range(0, 50000, 1000):
        # Reconstruct via occurrence counts, not production advanced-index averaging.
        for array in data:
            draws = rng.integers(48, size=(1000, 48))
            counts = np.zeros((1000, 48), dtype=np.int64)
            for j in range(48):
                counts[:, j] = np.sum(draws == j, axis=1)
            # Explicit weighted sums avoid introducing BLAS/OpenMP initialization
            # into a process already using Torch's CPU runtime on this host.
            for column in range(2):
                bootstrap[start:start + 1000, column] += (counts * array[:, column]).sum(axis=1) / (48 * 6)
    output = {}
    for column, name in enumerate(("primary", "attribution")):
        values = sorted(bootstrap[:, column].tolist())
        bounds = []
        for probability in (.025, .975):
            index = probability * (len(values) - 1)
            lower = int(index)
            bounds.append(100 * (values[lower] + (index - lower) * (values[lower + 1] - values[lower])))
        output[name] = {"delta_pp": 100 * sum(r[name] for r in rows) / 288, "ci95_pp": bounds}
    passed = all(output[name]["ci95_pp"][0] > 0 for name in ("primary", "attribution"))
    output["conditional_learning_contribution_established"] = passed
    output["classification"] = "positive_conditional_learning_contribution" if passed else "learning_contribution_not_established"
    return output


def final_audit(binding):
    samples = p.old.read_json(p.OUTPUT / "main/panel.json")["samples"]
    offers = [p.verified(item) for item in p.old.read_json(p.OUTPUT / "main_proposals_lock.json")["files"]]
    choices = p.old.read_json(p.OUTPUT / "main/choices.json")["rows"]
    evaluation = [p.verified(item) for item in p.old.read_json(p.OUTPUT / "main_evaluation_lock.json")["files"]]
    expected_rows = independent_rows(samples, offers, choices, evaluation)
    compare(expected_rows, p.old.read_json(p.OUTPUT / "state_results.json"))
    expected = independent_statistics(expected_rows)
    result = p.old.read_json(p.OUTPUT / "result.json")
    compare(expected, {key: result[key] for key in expected})
    full_counts = {}
    for stage in ("probe", "main"):
        counts = {}
        for kind in ("selection", "evaluation"):
            collection = p.verify_stage(f"{stage}_{kind}", binding)
            audit = p.verify_stage(f"{stage}_{kind}_audit", binding)
            p.require(len(collection["files"]) == len(audit["files"]) == (18 if stage == "probe" else 288), "Complete root audit count")
            counts[kind] = sum(row["continuations"] for row in collection["files"])
            p.require(counts[kind] == sum(row["continuations"] for row in audit["files"]), "Collected/replayed count mismatch")
            for original, receipt in zip(collection["files"], audit["files"]):
                row = p.verified(receipt)
                p.require(row["passed"] and row["full_inner_replay"] and row["target"] == original, "Missing continuation receipt")
        p.require(sum(counts.values()) <= (16128 if stage == "probe" else 258048), "Full budget exceeded")
        full_counts[stage] = counts
    p.old.save_new(p.OUTPUT / "final_audit.json", {"manifest": binding, "passed": True, "time": p.old.now(),
                   "states": 288, "all_three_fits": True, "full_inner_replay": True, "continuations": full_counts,
                   "result_sha256": p.old.file_hash(p.OUTPUT / "result.json"),
                   "state_results_sha256": p.old.file_hash(p.OUTPUT / "state_results.json"),
                   "independent_statistics": expected})

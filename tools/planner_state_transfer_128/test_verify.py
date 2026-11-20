"""Independent population, interval, privacy, sparse replay and corruption checks."""
import copy
import itertools
import json
from pathlib import Path
import time

import pytest

from tools.planner_state_transfer_128 import verify as v
from tools.planner_state_transfer_128 import engine as e


def synthetic_specs(stage="main"):
    specs, pair = [], 0
    for fit in v.FITS:
        for phase in range(3):
            for source_arm in (("candidate", "reference") if stage == "main" else ("candidate",)):
                for _ in range(10 if stage == "main" else 1):
                    for leg in (0, 1):
                        i = len(specs)
                        specs.append({"index": i, "pair": pair, "game_index": 2*pair+leg,
                                      "fit": fit, "phase": phase, "seat": leg, "source_arm": source_arm,
                                      "privacy": {"own": (i//2) % 2, "opponent": (i//4) % 2},
                                      "canonical_support": 1+(i//2) % 5})
                    pair += 1
    return specs


def test_prescribed_sampling_is_pair_valid_identity_only_and_probe_full():
    specs = synthetic_specs()
    plan = v.sample_plan(specs, "main", "fixture")
    assert plan == v.sample_plan(copy.deepcopy(specs), "main", "fixture")
    assert len({s["pair"] for s in specs}) == 180
    assert 18 <= plan["coverage"]["full_stress_roots"] <= 39
    assert not plan["coverage"]["missing_cells"]
    changed = copy.deepcopy(specs)
    for spec in changed:
        spec["outcome_not_used"] = -1234
    other = v.sample_plan(changed, "main", "other binding")
    assert other["selections"] == plan["selections"] and other["stress_cells"] == plan["stress_cells"]
    for row in plan["selections"]:
        assert all(len(row["ordinary"][b]) == 8 for b in v.BLOCKS)
        assert all(len(row["replay"][b]) == (128 if row["stress"] else 8) for b in v.BLOCKS)
    probe = v.sample_plan(synthetic_specs("probe"), "probe", "fixture")
    assert probe["coverage"]["replayed_worlds"] == 18*256
    assert all(row["stress"] for row in probe["selections"])


@pytest.mark.parametrize("damage", ["duplicate_game", "pair", "fit", "phase", "role", "seat", "support", "privacy"])
def test_source_pair_or_metadata_corruption_rejected(damage):
    specs = synthetic_specs()
    if damage == "duplicate_game":
        specs[1]["game_index"] = specs[0]["game_index"]
    elif damage == "pair":
        specs[1]["pair"] += 1
    elif damage == "fit":
        specs[1]["fit"] = v.FITS[1]
    elif damage == "phase":
        specs[1]["phase"] = 1
    elif damage == "role":
        specs[1]["source_arm"] = "reference"
    elif damage == "seat":
        specs[1]["seat"] = specs[0]["seat"]
    elif damage == "support":
        specs[1]["canonical_support"] = 0
    else:
        specs[1]["privacy"]["own"] = -1
    with pytest.raises(AssertionError):
        v.sample_plan(specs, "main", "fixture")


def test_world_namespaces_independently_reconstructed():
    seen = set()
    for stage, count in (("main", 360), ("probe", 18), ("test", 18)):
        for index in range(count):
            for block in v.BLOCKS:
                seeds = v._world_seeds(stage, index, block)
                assert seeds == e.world_seeds(stage, index, block) and not seen.intersection(seeds)
                seen.update(seeds)


def worlds(values):
    return [{"outcomes": [{"wdl": value, "reason": "repetition_cutoff" if value is None else "completed",
                           "winner": None if value in (None, 0) else 0 if value == 1 else 1}
                          for value in values]} for _ in range(128)]


def candidates(indices):
    return {arm: {"indices": list(indices)} for arm in v.ARMS}


@pytest.mark.parametrize("indices", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_bounded_selection_matches_all_admissible_unknown_completions(indices):
    data = worlds([0, 0, 0])
    # The early candidate can win at an exact bound only if its tie order permits.
    for offset in (-2, -1, 0, 1, 2):
        changed = copy.deepcopy(data)
        for wi in range(abs(offset)):
            changed[wi]["outcomes"][0] = worlds([1 if offset > 0 else -1])[0]["outcomes"][0]
        changed[7]["outcomes"][0] = worlds([None])[0]["outcomes"][0]
        changed[9]["outcomes"][1] = worlds([None])[0]["outcomes"][0]
        actual = v.independent_selection(changed, candidates(indices), 3)
        winners = set()
        for a, b in itertools.product((-1, 0, 1), repeat=2):
            totals = [offset+a, b, 0]
            winners.add(max(indices, key=lambda j: totals[j]))
        expected = next(iter(winners)) if len(winners) == 1 else None
        assert actual["choices"] == dict.fromkeys(v.ARMS, expected)
        if expected is not None:
            assert actual == e.choose(changed, candidates(indices), 3)
        else:
            with pytest.raises(e.AmbiguousDecision):
                e.choose(changed, candidates(indices), 3)


def test_coupled_selection_preserves_unknowns_and_requires_equal_contexts():
    data = worlds([0, 0])
    data[0]["outcomes"] = worlds([None, None])[0]["outcomes"]
    assert v.independent_selection(data, candidates([0, 1]), 2)["choices"] == dict.fromkeys(v.ARMS, None)
    context = {"fixture": "equalfuture"}
    proofs = [{"world": 0, "candidate": j, "context": context, "context_sha256": v.old.digest(context)} for j in range(2)]
    result = v.independent_selection(data, candidates([0, 1]), 2, proofs)
    assert result == e.choose(data, candidates([0, 1]), 2, proofs)
    assert result["choices"] == dict.fromkeys(v.ARMS, 0)
    assert result["means"] == [None, None]
    assert result["certificate"]["arm_certificates"]["anchor"]["pairwise_lower_numerators"] == [0, 0]
    proofs[1]["context"] = {"fixture": "differentfuture"}
    proofs[1]["context_sha256"] = v.old.digest(proofs[1]["context"])
    assert v.independent_selection(data, candidates([0, 1]), 2, proofs)["choices"] == dict.fromkeys(v.ARMS, None)
    proofs[1]["context_sha256"] = "0"*64
    with pytest.raises(AssertionError):
        v.independent_selection(data, candidates([0, 1]), 2, proofs)


def test_real_saved_coupled_proofs_independently_reconstruct_json_guard_context(monkeypatch):
    saved = v.old.read_json(v.c.ROOT/"outputs/equal_budget_breadth_arena_128/recovery_02/main/roots/game_0310/turn_058.json")
    def no_new_continuation(*args, **kwargs):
        raise AssertionError("Saved-trace fixture must not collect new continuations")
    monkeypatch.setattr(v.old, "trajectory", no_new_continuation)
    original = [v.repetition_audit.independent_saved_proof(saved, p) for p in saved["equivalence_proofs"]]
    proofs = []
    for offset in (0, 64):
        for item in original:
            copied = copy.deepcopy(item)
            copied["world"] += offset
            proofs.append(copied)
    data = copy.deepcopy(saved["worlds"])*2
    selection = v.independent_selection(data, candidates(range(8)), 8, proofs)
    assert selection == e.choose(data, candidates(range(8)), 8, proofs)
    assert selection["choices"] == dict.fromkeys(v.ARMS, 0)
    assert len(selection["certificate"]["unknown_groups"]) == 1
    assert selection["means"][:2] == [None, None]
    damaged = copy.deepcopy(original[0])
    damaged["context"]["no_progress_count"] += 1
    damaged["context_sha256"] = v.old.digest(damaged["context"])
    with pytest.raises(AssertionError):
        v.repetition_audit.independent_saved_proof(saved, damaged)


@pytest.mark.parametrize("size", [1, 2, 3])
def test_short_support_and_sparse_assessment_are_exact(size):
    data = worlds([0]*size)
    selection = v.independent_selection(data, candidates(range(size)), size)
    assert selection["choices"] == dict.fromkeys(v.ARMS, 0)
    summary = v.independent_summary(worlds([1]), [0], selection["choices"], dict.fromkeys(v.ARMS), size)
    assert summary["arm_scores"] == dict.fromkeys(v.ARMS, 1.)
    assert all(bounds == [0., 0.] for bounds in summary["delta_bounds"].values())


def test_sparse_indices_bounds_and_same_unknown_action_delta_zero():
    choices = {"anchor": 0, "broad": 0, "narrow": 3, "teacher": 9}
    proposed = dict.fromkeys(v.ARMS)
    data = worlds([0, -1, 1])
    data[13]["outcomes"][0] = worlds([None])[0]["outcomes"][0]
    summary = v.independent_summary(data, [0, 3, 9], choices, proposed, 10)
    assert summary == e.summarize(data, [0, 3, 9], choices, proposed, 10)
    assert summary["arm_scores"]["anchor"] is summary["arm_scores"]["broad"] is None
    assert summary["arm_score_bounds"]["anchor"] == [127/256, 129/256]
    assert summary["delta_bounds"]["broad_minus_anchor"] == [0., 0.]
    assert summary["delta_bounds"]["broad_minus_narrow"] == [127/256, 129/256]
    with pytest.raises(AssertionError):
        v.independent_summary(data, [0, 2, 9], choices, proposed, 10)


def test_all_unknown_cells_added_without_replaying_known_unselected_cells():
    data = worlds([1, -1, 0])
    data[1]["outcomes"][0] = worlds([None])[0]["outcomes"][0]
    data[99]["outcomes"][2] = worlds([None])[0]["outcomes"][0]
    cells = v.replay_cells(data, [0, 4, 9], [1, 3, 5, 7, 9, 11, 13, 15])
    assert len(cells["sampled"]) == 24 and len(cells["all"]) == 25
    assert cells["unknown"] == {(1, 0), (99, 9)} and cells["extra"] == {(99, 9)}
    assert (99, 0) not in cells["all"]


@pytest.mark.parametrize("reason,value", [("repetition_cutoff", 0), ("completed", None), ("completed", True), ("no_progress_cutoff", None)])
def test_imputed_or_unsupported_outcome_rejected(reason, value):
    data = worlds([0])
    data[0]["outcomes"][0].update(reason=reason, wdl=value)
    with pytest.raises(AssertionError):
        v._bounded_columns(data, 1)


def private_state():
    env = v.old.SplendorEnv(seed=2189810000)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = dict.fromkeys(player.bonuses, 5)
    return state


def short_state(support):
    state = v.old.SplendorEnv(seed=2189820000+support).initial_state()
    cards = [card for tier in state.visible_tier_cards for card in state.visible_tier_cards[tier]+state.hidden_tier_decks[tier]]
    reserved = sorted([card for card in cards if card.points > 0], key=lambda card: card.card_id)[:support]
    state.visible_tier_cards = {tier: [] for tier in state.visible_tier_cards}
    state.hidden_tier_decks = {tier: [] for tier in state.hidden_tier_decks}
    state.deck_counts = dict.fromkeys(state.deck_counts, 0)
    state.bank_tokens = dict.fromkeys(state.bank_tokens, 0)
    state.nobles = []
    for player in state.players:
        player.score = 14
        player.bonuses = dict.fromkeys(player.bonuses, 10)
    state.players[0].reserved_cards = reserved
    state.players[0].reserved_card_sources = ["deck"]*support
    state.players[1].purchased_cards = [card for card in cards if card not in reserved]
    return state


def fixture_spec(state, index):
    snapshot = json.loads(json.dumps(v.old.serialize_state_snapshot(state)))
    legal = v.old.SplendorEnv().legal_actions(state)
    return {"index": index, "pair": index//2, "game_index": index, "fit": 161001,
            "seat": state.current_player, "phase": min(state.turn_index//20, 2), "turn": state.turn_index,
            "source_arm": "candidate", "privacy": e.baseline.privacy(state),
            "canonical_support": len({repr(v.old.canonical_successor_action_key(state, a)) for a in legal}),
            "search_stream": "planner-state-transfer-independent-fixture",
            "snapshot_sha256": v.old.digest(snapshot), "fixture_snapshot": snapshot}


@pytest.fixture(scope="module")
def fixture_artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("planner-transfer-verifier")
    registry = v.old.read_json(v.c.ROOT/"outputs/proposal_breadth_learning_128/models_lock.json")
    registry["manifest"] = "fixture"
    v.old.save_new(base/"models_lock.json", registry)
    e.initialize_models(base)
    specs = [fixture_spec(private_state(), i) for i in range(2)]
    specs += [fixture_spec(short_state(n), n+1) for n in (1, 2, 3)]
    assert [s["canonical_support"] for s in specs[2:]] == [1, 2, 3]
    plan = v.sample_plan(specs, "test", "fixture")
    v.old.save_new(base/"test"/"audit_sample.json", plan)
    items = [e.collect("test", spec, "fixture", base) for spec in specs]
    return base, specs, plan, items


def test_actual_json_private_and_short_support_roots_replay_exact_budgets(fixture_artifacts):
    base, specs, plan, items = fixture_artifacts
    assert any(not row["stress"] for row in plan["selections"])
    receipts = []
    for spec, sample, item in zip(specs, plan["selections"], items, strict=True):
        ref = v.audit("test", spec, item, "fixture", base)
        receipts.append(ref)
        receipt = v.c.verified(ref)
        assert receipt["row"] == e.statistics_row(v.c.verified(item))
        assert receipt["recorded_continuations"] == 128*(receipt["support_selection"]+receipt["support_assessment"])
        assert receipt["replayed_worlds"] == (256 if sample["stress"] else 16)
        if spec["canonical_support"] <= 3:
            assert receipt["support_assessment"] == 1 and receipt["row"]["proposed_indices"] == dict.fromkeys(v.ARMS)
    combined = v.summarize(receipts, specs, "fixture", base)
    assert len(combined["rows"]) == 5 and combined["coverage"]["states"] == 5


def clone_fixture(fixture_artifacts, tmp_path):
    base, specs, plan, items = fixture_artifacts
    index = next(row["index"] for row in plan["selections"] if not row["stress"])
    spec, root = specs[index], copy.deepcopy(v.c.verified(items[index]))
    v.old.save_new(tmp_path/"models_lock.json", v.old.read_json(base/"models_lock.json"))
    v.old.save_new(tmp_path/"test"/"audit_sample.json", plan)
    seal_path = tmp_path/"test"/"selections"/f"state_{index:04d}_selection.json"
    seal = copy.deepcopy(v.c.verified(root["selection_artifact"]))
    seal["selection_lock"]["created_at_ns"] = time.time_ns()
    root["selection_lock"] = seal["selection_lock"]
    v.old.save_new(seal_path, seal)
    root["selection_artifact"] = v.c.artifact(seal_path)
    root["assessment_started_at_ns"] = time.time_ns()
    return spec, root


@pytest.mark.parametrize("damage", ["none", "model", "model_hash", "summary", "delta_bounds", "assessment_indices", "selection_indices",
                                    "schedule", "world", "outcome", "sampled_trace", "hidden_unknown", "seal_time", "sample_plan", "teacher_added"])
def test_rehashed_corruptions_fail_and_pristine_clone_passes(fixture_artifacts, tmp_path, damage):
    spec, root = clone_fixture(fixture_artifacts, tmp_path)
    index, block = spec["index"], "assessment"
    wi = v.ordinary_indices("test", index, block)[0]
    if damage == "model":
        root["proposals"]["broad"]["model_id"] = "narrow_161001"
    elif damage == "model_hash":
        root["proposals"]["broad"]["model_sha256"] = "0"*64
    elif damage == "summary":
        root["summary"]["arm_scores"]["broad"] += 1/256
    elif damage == "delta_bounds":
        root["summary"]["delta_bounds"]["broad_minus_anchor"][0] += 1/256
    elif damage == "assessment_indices":
        root[block]["candidate_indices"] = []
    elif damage == "selection_indices":
        root["selection"]["candidate_indices"] = root["selection"]["candidate_indices"][1:]
    elif damage == "schedule":
        root[block]["world_seeds"][0] += 1
    elif damage == "world":
        root[block]["worlds"][0]["state_sha256"] = "0"*64
    elif damage == "outcome":
        wi = next(i for i in range(128) if i not in v.ordinary_indices("test", index, block))
        root[block]["worlds"][wi]["outcomes"][0]["wdl"] = .5
    elif damage == "sampled_trace":
        root[block]["worlds"][wi]["outcomes"][0]["trace_sha256"] = "0"*64
    elif damage == "hidden_unknown":
        wi = next(i for i in range(128) if i not in v.ordinary_indices("test", index, block))
        root[block]["worlds"][wi]["outcomes"][0].update(reason="repetition_cutoff", winner=None, wdl=None)
        root["summary"] = v.independent_summary(root[block]["worlds"], root[block]["candidate_indices"],
            root["selection_lock"]["choices"], root["proposed_indices"], len(root["union"]["keys"]))
    elif damage == "seal_time":
        root["assessment_started_at_ns"] = root["selection_lock"]["created_at_ns"]-1
    elif damage == "sample_plan":
        path = tmp_path/"test"/"audit_sample.json"
        value = v.old.read_json(path)
        value["selections"][index]["replay"][block] = []
        path.write_text(json.dumps(value), encoding="utf-8")
    elif damage == "teacher_added":
        root["teacher_added_indices"]["ranked"] = []
    path = tmp_path/"test"/"roots"/f"state_{index:04d}.json"
    v.old.save_new(path, root)
    if damage == "none":
        assert v.c.verified(v.audit("test", spec, v.c.artifact(path), "fixture", tmp_path))["passed"]
    else:
        with pytest.raises((AssertionError, ValueError)):
            v.audit("test", spec, v.c.artifact(path), "fixture", tmp_path)


def test_source_fixture_override_cannot_target_production():
    with pytest.raises(AssertionError):
        v._source_state("test", {"fixture_snapshot": {}}, v.OUTPUT)
    with pytest.raises(AssertionError):
        v._source_state("main", {"fixture_snapshot": {}}, Path("elsewhere"))


def statistic_rows(broad=2, narrow=0, unknown=False, varying=False):
    rows = []
    for spec in synthetic_specs():
        anchor = 128
        effect = broad+(spec["pair"] % 10-4 if varying else 0)
        scores = {"anchor": anchor, "broad": anchor+effect, "narrow": anchor+narrow, "teacher": 140}
        bounds = {arm: [value/256, value/256] for arm, value in scores.items()}
        if unknown:
            bounds["broad"] = [(scores["broad"]-1)/256, (scores["broad"]+1)/256]
        choices = dict(zip(v.ARMS, range(4)))
        row = {**spec, "choices": choices, "arm_score_bounds": bounds,
               "arm_scores": {arm: value[0] if value[0] == value[1] else None for arm, value in bounds.items()},
               "delta_bounds": {name: [bounds[a][0]-bounds[b][1], bounds[a][1]-bounds[b][0]]
                                for name, a, b in v.CONTRASTS}}
        rows.append(row)
    return rows


def test_independent_bootstrap_matches_indexed_implementation_and_pair_unit():
    from tools.planner_state_transfer_128 import statistics
    rows = statistic_rows(varying=True)
    result = v.independent_statistics(rows)
    v.compare(result, statistics.summarize(rows))
    assert result["pairs"] == 180 and result["strata"] == 18
    assert result["contrasts"]["broad_minus_anchor"]["mean"] == 2.5/256
    # Both legs carry the identical pair effect: splitting them into 360 units
    # would reduce uncertainty, so verify against a direct pair bootstrap too.
    import numpy as np
    rng = np.random.default_rng(2183000000)
    samples = []
    effects = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    for _ in range(50):
        total = np.zeros(1000, dtype=np.int64)
        for _ in range(18):
            total += effects[rng.integers(0, 10, size=(1000, 10))].sum(axis=1)
        samples.extend(total/(180*256))
    assert result["contrasts"]["broad_minus_anchor"]["ci95"] == pytest.approx(np.quantile(samples, [.025, .975]), abs=1e-14)


@pytest.mark.parametrize("broad,narrow,unknown,passed", [(2,0,False,True), (1,0,False,False), (2,2,False,False),
                                                        (0,0,False,False), (2,0,True,False), (3,0,True,True)])
def test_statistics_gates_and_unknown_envelopes_have_closed_form(broad, narrow, unknown, passed):
    from tools.planner_state_transfer_128 import statistics
    rows = statistic_rows(broad=broad, narrow=narrow, unknown=unknown)
    result = v.independent_statistics(rows)
    v.compare(result, statistics.summarize(rows))
    delta = 1 if unknown else 0
    bounds = [(broad-delta)/256, (broad+delta)/256]
    assert result["contrasts"]["broad_minus_anchor"]["mean_bounds"] == bounds
    assert result["contrasts"]["broad_minus_anchor"]["ci95"] == bounds
    assert result["contrasts"]["broad_minus_anchor"]["mean"] == (None if unknown else broad/256)
    assert result["unknown_outcomes_imputed"] is False and result["gate_pass"] is passed


@pytest.mark.parametrize("damage", ["duplicate", "split_pair", "unbalanced", "imputed", "delta_grid", "delta_inconsistent"])
def test_statistics_rejects_pair_and_censoring_corruption(damage):
    rows = statistic_rows(unknown=True)
    if damage == "duplicate":
        rows[1]["game_index"] = rows[0]["game_index"]
    elif damage == "split_pair":
        rows[1]["pair"] += 1
    elif damage == "unbalanced":
        rows[0]["source_arm"] = rows[1]["source_arm"] = "reference"
    elif damage == "imputed":
        rows[0]["arm_scores"]["broad"] = .5
    elif damage == "delta_grid":
        rows[0]["delta_bounds"]["broad_minus_anchor"][0] += .00001
    else:
        rows[0]["delta_bounds"]["teacher_minus_anchor"][0] -= 1/256
    with pytest.raises(AssertionError):
        v.independent_statistics(rows)

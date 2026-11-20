"""Independent population, interval, privacy, sparse replay and corruption checks."""
import copy
import itertools
import json
from pathlib import Path
import time

import pytest

from tools.independent_game_coverage_128 import verify as v
from tools.independent_game_coverage_128 import engine as e


def synthetic_specs(stage="main"):
    specs = []
    for i in range(360 if stage == "main" else 18):
        fit, seat, phase = v.CELLS[i % 18]
        specs.append({"index":i,"pair":3*i,"game_index":3*i,"fit":fit,"seat":seat,"phase":phase,
                      "privacy":{"own":i%2,"opponent":(i//2)%2},"canonical_support":1+i%5})
    return specs


def test_prescribed_sampling_is_game_valid_identity_only_and_probe_full():
    specs = synthetic_specs()
    plan = v.sample_plan(specs, "main", "fixture")
    assert plan == v.sample_plan(copy.deepcopy(specs), "main", "fixture")
    assert len({s["pair"] for s in specs}) == 360
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


@pytest.mark.parametrize("damage", ["duplicate_game", "pair", "fit", "phase", "seat", "support", "privacy"])
def test_source_pair_or_metadata_corruption_rejected(damage):
    specs = synthetic_specs()
    if damage == "duplicate_game":
        specs[1]["game_index"] = specs[0]["game_index"]
    elif damage == "pair":
        specs[1]["pair"] += 1
    elif damage == "fit":
        specs[1]["fit"] = v.FITS[1]
    elif damage == "phase":
        specs[1]["phase"] = 0
    elif damage == "seat":
        specs[1]["seat"] = 1
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
    choices = {"anchor":0,"broad":0,"narrow":3,"teacher":11,"lowrate":9,"soft":7}
    proposed = dict.fromkeys(v.ARMS)
    data = worlds([0,-1,0,0,1])
    data[13]["outcomes"][0] = worlds([None])[0]["outcomes"][0]
    summary = v.independent_summary(data, [0,3,7,9,11], choices, proposed, 12)
    assert summary == e.summarize(data, [0,3,7,9,11], choices, proposed, 12)
    assert summary["arm_scores"]["anchor"] is summary["arm_scores"]["broad"] is None
    assert summary["arm_score_bounds"]["anchor"] == [127/256, 129/256]
    assert summary["delta_bounds"]["broad_minus_anchor"] == [0., 0.]
    assert summary["delta_bounds"]["broad_minus_narrow"] == [127/256, 129/256]
    with pytest.raises(AssertionError):
        v.independent_summary(data, [0,2,7,9,11], choices, proposed, 12)


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
    return {"index": index, "pair": index, "game_index": index, "fit": 161001,
            "seat": state.current_player, "phase": min(state.turn_index//20, 2), "turn": state.turn_index,
            "privacy": e.baseline.privacy(state),
            "canonical_support": len({repr(v.old.canonical_successor_action_key(state, a)) for a in legal}),
            "search_stream": "planner-state-transfer-independent-fixture",
            "snapshot_sha256": v.old.digest(snapshot), "fixture_snapshot": snapshot}


@pytest.fixture(scope="module")
def fixture_artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("planner-transfer-verifier")
    from .test_engine import fixture_registry
    registry = fixture_registry()
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


def test_six_distinct_assessed_choices_in_twelve_candidate_union():
    indices = [0,2,4,7,9,11]
    choices = dict(zip(v.ARMS,indices,strict=True))
    data = worlds([-1,0,1,-1,0,1])
    proposed = {arm:3 for arm in v.ARMS}
    actual = v.independent_summary(data,indices,choices,proposed,12)
    assert actual == e.summarize(data,indices,choices,proposed,12)
    assert actual["arm_scores"] == dict(zip(v.ARMS,[0.,.5,1.,0.,.5,1.],strict=True))
    assert actual["delta_bounds"]["broad_minus_lowrate"] == [1.,1.]
    assert actual["delta_bounds"]["narrow_minus_soft"] == [0.,0.]


def statistic_rows(broad=3, narrow=0, lowrate=0, soft=0, unknown=0, vary=True):
    """One independent game per row, with endpoints in exact 1/256 units."""
    rows = []
    for spec in synthetic_specs():
        i = spec["index"]
        variation = (i // 18 % 5 - 2) if vary else 0
        scores = dict(anchor=128, narrow=128+narrow, broad=128+broad+variation,
                      lowrate=128+lowrate, soft=128+soft, teacher=140)
        bounds = {arm:[score/256,score/256] for arm,score in scores.items()}
        bounds["broad"] = [(scores["broad"]-unknown)/256,(scores["broad"]+unknown)/256]
        choices = dict(zip(v.ARMS,range(6),strict=True))
        rows.append({**spec,"arm_scores":{arm:lo if lo == hi else None for arm,(lo,hi) in bounds.items()},
                     "arm_score_bounds":bounds,"choices":choices,
                     "proposed_indices":{arm:choices[arm] for arm in v.PROPOSERS},
                     "delta_bounds":{name:[bounds[left][0]-bounds[right][1],bounds[left][1]-bounds[right][0]]
                                     for name,left,right in v.CONTRASTS}})
    return rows


def direct_bootstrap(rows):
    """Independent indexed gathering, deliberately no frequency histograms."""
    np = v.np
    matrices = [np.asarray([[row["delta_bounds"][name] for name,_,_ in v.CONTRASTS]
                            for row in rows if (row["fit"],row["seat"],row["phase"]) == cell])
                for cell in v.CELLS]
    rng = np.random.default_rng(v.c.CONFIG["bootstrap_seed"])
    draws = []
    for start in range(0,v.c.CONFIG["bootstrap_samples"],v.c.CONFIG["bootstrap_chunk"]):
        size = min(v.c.CONFIG["bootstrap_chunk"],v.c.CONFIG["bootstrap_samples"]-start)
        total = np.zeros((size,len(v.CONTRASTS),2))
        for matrix in matrices:
            indices = rng.integers(0,20,size=(size,20))
            total += matrix[indices].sum(axis=1)
        draws.append(total/360)
    array = np.concatenate(draws)
    return {name:[float(np.quantile(array[:,j,0],.025)),float(np.quantile(array[:,j,1],.975))]
            for j,(name,_,_) in enumerate(v.CONTRASTS)}


@pytest.mark.parametrize("unknown", [0,1,3])
def test_histogram_statistics_match_independent_indexed_interval_bootstrap(monkeypatch,unknown):
    monkeypatch.setitem(v.c.CONFIG,"bootstrap_samples",97)
    monkeypatch.setitem(v.c.CONFIG,"bootstrap_chunk",23)
    rows = statistic_rows(unknown=unknown)
    result, expected = v.statistics(rows), direct_bootstrap(rows)
    assert result["states"] == result["source_games"] == 360
    assert result["states_per_cell"] == 20 and not result["unknown_outcomes_imputed"]
    for name,left,right in v.CONTRASTS:
        actual = result["contrasts"][name]
        assert actual["ci95"] == pytest.approx(expected[name],abs=1e-15)
        assert actual["mean_bounds"] == pytest.approx([
            sum(row["delta_bounds"][name][endpoint] for row in rows)/360 for endpoint in (0,1)],abs=1e-15)
        if unknown and left == "broad":
            assert actual["mean_score_difference"] is None and actual["score_percentage_points"] is None


@pytest.mark.parametrize("broad,narrow,lowrate,unknown,gate,attribution", [
    (2,2,0,0,True,False), (2,0,2,0,False,True), (2,0,0,0,True,True),
    (1,0,0,0,False,True), (2,0,0,1,False,True), (2,0,0,2,False,False),
])
def test_practical_gate_is_separate_from_breadth_attribution(monkeypatch,broad,narrow,lowrate,unknown,gate,attribution):
    monkeypatch.setitem(v.c.CONFIG,"bootstrap_samples",31)
    result = v.statistics(statistic_rows(broad,narrow,lowrate,unknown=unknown,vary=False))
    assert result["gate_pass"] is gate and result["breadth_attribution"] is attribution
    assert result["classification"] == ("positive_conditional_data_package" if gate else "practical_progress_not_established")


def test_shared_unknown_action_cancels_exactly_in_paired_statistics(monkeypatch):
    monkeypatch.setitem(v.c.CONFIG,"bootstrap_samples",31)
    rows = statistic_rows(unknown=2)
    for row in rows:
        row["choices"]["anchor"] = row["choices"]["broad"]
        row["arm_score_bounds"]["anchor"] = row["arm_score_bounds"]["broad"].copy()
        row["arm_scores"]["anchor"] = None
        for name,left,right in v.CONTRASTS:
            row["delta_bounds"][name] = [0.,0.] if row["choices"][left] == row["choices"][right] else [
                row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][1],
                row["arm_score_bounds"][left][1]-row["arm_score_bounds"][right][0]]
    result = v.statistics(rows)
    assert result["contrasts"]["broad_minus_anchor"]["ci95"] == [0.,0.]
    assert result["contrasts"]["broad_minus_anchor"]["mean_score_difference"] == 0.
    assert not result["gate_pass"]


@pytest.mark.parametrize("damage", ["game", "pair", "cell", "index", "imputation", "grid", "delta", "same_action", "missing_arm"])
def test_statistics_rejects_population_and_bound_corruption(damage):
    rows = statistic_rows(unknown=1)
    row = rows[0]
    if damage == "game": row["game_index"] = rows[1]["game_index"]
    elif damage == "pair": row["pair"] += 1
    elif damage == "cell": row["fit"] = v.FITS[1]
    elif damage == "index": row["index"] = 1
    elif damage == "imputation": row["arm_scores"]["broad"] = .5
    elif damage == "grid": row["arm_score_bounds"]["broad"][0] += 1/512
    elif damage == "delta": row["delta_bounds"]["broad_minus_anchor"][0] += 1/256
    elif damage == "same_action": row["choices"]["broad"] = row["choices"]["anchor"]
    else: row["arm_scores"].pop("soft")
    with pytest.raises(AssertionError):
        v.statistics(rows)



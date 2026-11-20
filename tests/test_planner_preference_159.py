"""159 diagnostic contracts and semantic-tamper tests, using isolated fixtures."""
import copy
import math
from pathlib import Path
import subprocess
import sys

import pytest

from splendor_ai.alphazero import learned_continuation_158 as h
from splendor_ai.alphazero import planner_preference_159 as p
from splendor_ai.engine.actions import ActionType
from scripts.analysis import verify_planner_preference_159 as a


@pytest.fixture(scope="module", autouse=True)
def fixed_models():
    h.initialize_worker()


def tiny_sample():
    """Legal one-move endings keep complete trajectory replay tests inexpensive."""
    env = h.old.SplendorEnv(seed=1599800000)
    state = env.initial_state()
    state.current_player, state.turn_index = 1, 1
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    actions = [move for move in env.legal_actions(state)
               if move.action_type == ActionType.BUY_VISIBLE
               and state.visible_tier_cards[move.tier][move.market_index].points > 0][:2]
    assert len(actions) == 2
    return {"index": 0, "pair": 0, "model_id": h.MODELS[0], "phase": 0,
            "actor": 1, "source_planner": "GG", "cell": 2, "source_index": 0,
            "turn": 1, "snapshot": h.old.serialize_state_snapshot(state),
            "candidates": {"actions": [h.old.serialize_action(action) for action in actions]}}


def fixture_record(sample, stage, selection=None, real=False):
    state = h.old.deserialize_state_snapshot(sample["snapshot"])
    indices = (list(range(len(sample["candidates"]["actions"]))) if stage == "selection"
               else list(dict.fromkeys([selection["choices"]["LL"], selection["choices"]["GG"]])))
    same = stage == "evaluation" and selection["choices"]["LL"] == selection["choices"]["GG"]
    record = {"manifest": "a"*64, "index": sample["index"], "pair": sample["pair"],
              "stage": stage, "stream": a.independent_stream(sample, stage),
              "action_indices": indices, "selection_sha256": None, "worlds": [],
              "timing": {"wall": 0., "cpu": 0.}}
    for wi, seed in enumerate([] if same else a.independent_seeds(sample, stage)):
        world = a.a158.a154.independent_world(state, seed)
        row = {"seed": seed, "state_sha256": h.old.digest(h.old.serialize_state_snapshot(world)),
               "outcomes": {"GG": [], "LL": []}}
        for policy in a.POLICIES:
            for j in indices:
                if real:
                    action = h.old.decode_action(sample["candidates"]["actions"][j])
                    outcome = (h.old.trajectory(world, action) if policy == "GG" else
                               h.trajectory(world, action, record["stream"], wi, sample["model_id"]))
                    outcome["wdl"] = h.old.outcome_value(outcome["winner"], sample["actor"])
                else:
                    # Distinct policies select distinct actions; all values and
                    # natural-result fields are structurally valid fixtures.
                    winner = sample["actor"] if (policy == "GG") == (j == 0) else 1-sample["actor"]
                    outcome = {"reason": "completed", "winner": winner, "turns": 1,
                               "trace_sha256": "b"*64, "final_state_sha256": "c"*64,
                               "wdl": 1. if winner == sample["actor"] else -1.}
                    if policy == "LL":
                        outcome.update(decision_sha256="d"*64, learned_decisions=0)
                row["outcomes"][policy].append(outcome)
        record["worlds"].append(row)
    if stage == "selection":
        record["means"] = {policy: [sum(row["outcomes"][policy][j]["wdl"] for row in record["worlds"])/32
                                      for j in range(len(indices))] for policy in a.POLICIES}
        record["choices"] = {policy: max(range(len(indices)), key=lambda j: record["means"][policy][j])
                               for policy in a.POLICIES}
    else:
        record["same_choice"] = same
    return record


def source_sample(tmp_path):
    spec = a.a158.independent_spec("main", 0)
    env = h.old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    initial = h.old.serialize_state_snapshot(state)
    first = env.legal_actions(state)[0]
    first_move = {"turn": 0, "actor": 0, "state_sha256": h.old.digest(initial),
                  "action": h.old.serialize_action(first), "root_sha256": "f"*64}
    state = env.step(state, first)
    schedule = a.a158.independent_schedule(state, spec["search_stream"])
    offered = a.a158.a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
    snapshot = h.old.serialize_state_snapshot(state)
    root_path = tmp_path/"main"/"roots"/"game_0000"/"turn_001.json"
    root = {"manifest": a.SOURCE_BINDING, "snapshot": snapshot, "schedule": schedule,
            "candidates": offered, "choice": 0}
    h.old.save_new(root_path, root)
    move = {"turn": 1, "actor": 1, "state_sha256": h.old.digest(snapshot),
            "root_sha256": h.old.file_hash(root_path), "action": offered["actions"][0]}
    game = {"manifest": a.SOURCE_BINDING, "spec": spec, "initial_state_sha256": h.old.digest(initial),
            "trace": [first_move, move]}
    game_path = tmp_path/"main"/"games"/"game_0000.json"
    h.old.save_new(game_path, game)
    return {"index": 0, "pair": 0, "model_id": spec["model_id"], "phase": 0,
            "actor": 1, "source_planner": "GG", "cell": 2, "source_index": 0,
            "source_game_path": game_path.as_posix(), "source_game_sha256": h.old.file_hash(game_path),
            "source_root_path": root_path.as_posix(), "source_root_sha256": h.old.file_hash(root_path),
            "turn": 1, "snapshot": snapshot, "candidates": offered, "source_schedule": schedule,
            "source_spec": spec}


def test_independent_world_streams_are_fresh_lawful_and_separate():
    sample = tiny_sample()
    selection, evaluation = a.independent_seeds(sample, "selection"), a.independent_seeds(sample, "evaluation")
    assert len(set(selection)) == 32 and len(set(evaluation)) == 128
    assert not set(selection) & set(evaluation)
    original = a.a158.independent_schedule(h.old.deserialize_state_snapshot(sample["snapshot"]), "source")
    assert not (set(selection) | set(evaluation)) & set(original["world_seeds"])
    altered = copy.deepcopy(sample)
    altered["snapshot"] = h.old.serialize_state_snapshot(a.a158.hidden_alternative(
        h.old.deserialize_state_snapshot(sample["snapshot"])))
    assert a.independent_seeds(altered, "selection") == selection
    # Neither root action nor model identity enters the shared world stream.
    altered["candidates"]["actions"].reverse()
    altered["model_id"] = h.MODELS[1]
    assert a.independent_seeds(altered, "evaluation") == evaluation
    with pytest.raises(AssertionError):
        a.independent_stream(sample, "adaptive_extension")


def test_source_prefix_snapshot_proposals_and_seals_are_independently_verified(tmp_path):
    sample = source_sample(tmp_path)
    observed = a.verify_sample(sample)
    assert observed["prefix_moves"] == 1 and observed["support"] == 4
    for field, value in (("source_game_sha256", "0"*64), ("source_root_sha256", "0"*64),
                         ("source_planner", "LL"), ("pair", 1), ("phase", 1), ("actor", 0), ("cell", 3)):
        altered = copy.deepcopy(sample)
        altered[field] = value
        with pytest.raises(AssertionError):
            a.verify_sample(altered)
    altered = copy.deepcopy(sample)
    altered["candidates"]["actions"].reverse()
    with pytest.raises(AssertionError):
        a.verify_sample(altered)
    altered = copy.deepcopy(sample)
    altered["snapshot"]["turn_index"] += 1
    with pytest.raises(AssertionError):
        a.verify_sample(altered)


def test_source_prefix_cannot_be_rebound_to_an_illegal_move(tmp_path):
    sample = source_sample(tmp_path)
    path = Path(sample["source_game_path"])
    game = h.old.read_json(path)
    game["trace"][0]["action"]["take_tokens"] = ["gold"]*9
    h.old.atomic_write(path, game)
    sample["source_game_sha256"] = h.old.file_hash(path)
    with pytest.raises(AssertionError):
        a.verify_sample(sample)


def test_record_audit_recomputes_means_and_freezes_ordered_selection():
    sample = tiny_sample()
    selected = fixture_record(sample, "selection")
    evaluated = fixture_record(sample, "evaluation", selected)
    first = a.verify_record(sample, selected, "selection")
    second = a.verify_record(sample, evaluated, "evaluation", selected)
    assert selected["choices"] == {"GG": 0, "LL": 1}
    assert evaluated["action_indices"] == [1, 0]
    assert first["continuations"] == 128 and second["continuations"] == 512
    for mutate in (lambda r: r["means"]["GG"].__setitem__(0, .5),
                   lambda r: r["choices"].__setitem__("GG", 1),
                   lambda r: r["worlds"][0].__setitem__("seed", "selection-reused"),
                   lambda r: r["worlds"][0].__setitem__("state_sha256", "0"*64),
                   lambda r: r["worlds"][0]["outcomes"]["GG"][0].__setitem__("wdl", -1.),
                   lambda r: r["worlds"][0]["outcomes"]["LL"][0].__setitem__("learned_decisions", 1),
                   lambda r: r["worlds"][0]["outcomes"]["GG"][0].__setitem__("reason", "max_turns")):
        corrupted = copy.deepcopy(selected)
        mutate(corrupted)
        with pytest.raises(AssertionError):
            a.verify_record(sample, corrupted, "selection")
    corrupted = copy.deepcopy(evaluated)
    corrupted["action_indices"].reverse()
    with pytest.raises(AssertionError):
        a.verify_record(sample, corrupted, "evaluation", selected)


def test_tied_first_max_and_same_move_have_exact_zero_without_evaluation():
    sample = tiny_sample()
    selected = fixture_record(sample, "selection", real=True)
    assert selected["means"] == {"GG": [1., 1.], "LL": [1., 1.]}
    assert selected["choices"] == {"GG": 0, "LL": 0}
    evaluated = fixture_record(sample, "evaluation", selected)
    checked = a.verify_record(sample, evaluated, "evaluation", selected, replay=True)
    assert evaluated["worlds"] == [] and checked["continuations"] == checked["replayed_branches"] == 0
    selected["choices"]["LL"] = 1
    with pytest.raises(AssertionError):
        a.verify_record(sample, selected, "selection")


def test_actual_frozen_trajectories_match_independent_sampled_replay_and_detect_tamper():
    sample = tiny_sample()
    selected = fixture_record(sample, "selection", real=True)
    checked = a.verify_record(sample, selected, "selection", replay=True)
    assert checked["replayed_branches"] == 2*2*2
    corrupted = copy.deepcopy(selected)
    corrupted["worlds"][0]["outcomes"]["GG"][0]["final_state_sha256"] = "0"*64
    a.verify_record(sample, corrupted, "selection", replay=False)
    with pytest.raises(AssertionError):
        a.verify_record(sample, corrupted, "selection", replay=True)


def test_evaluation_replay_visits_exact_prespecified_worlds(monkeypatch):
    sample = tiny_sample()
    selected = fixture_record(sample, "selection")
    evaluated = fixture_record(sample, "evaluation", selected)
    calls = []
    def replay(world, action, actor, stream, world_index, model):
        assert stream == a.independent_stream(sample, "evaluation")
        policy = "GG" if model is None else "LL"
        action_index = sample["candidates"]["actions"].index(h.old.serialize_action(action))
        calls.append((world_index, policy, action_index))
        j = evaluated["action_indices"].index(action_index)
        return evaluated["worlds"][world_index]["outcomes"][policy][j]
    monkeypatch.setattr(a.a158, "replay_branch", replay)
    checked = a.verify_record(sample, evaluated, "evaluation", selected, replay=True)
    assert calls == [(wi, policy, j) for wi in (0, 64) for policy in ("GG", "LL") for j in (1, 0)]
    assert checked["replayed_branches"] == 8


def test_production_simulation_obeys_world_schedule_and_independent_replay():
    sample = tiny_sample()
    # Two real turns exercise an actual learned inner decision under159's
    # exogenous stream, followed by independent frozen157 decision replay.
    sample["actor"], sample["turn"] = 0, 0
    sample["snapshot"]["current_player"], sample["snapshot"]["turn_index"] = 0, 0
    assert p.world_seeds(sample, "selection") == a.independent_seeds(sample, "selection")
    assert p.world_seeds(sample, "evaluation") == a.independent_seeds(sample, "evaluation")
    selected = p.simulate(sample, "selection", "a"*64)
    checked = a.verify_record(sample, selected, "selection", replay=True)
    assert checked["replayed_branches"] == 8
    assert all(o["learned_decisions"] >= 1 for w in selected["worlds"] for o in w["outcomes"]["LL"])
    evaluated = p.simulate(sample, "evaluation", "a"*64, selected)
    a.verify_record(sample, evaluated, "evaluation", selected, replay=True)
    a.a158.compare(p.record_metrics(selected, evaluated), a.independent_metrics(selected, evaluated))


def test_metrics_have_correct_score_units_sign_halves_and_zero_denominator():
    sample = tiny_sample()
    selected = fixture_record(sample, "selection")
    evaluated = fixture_record(sample, "evaluation", selected)
    metric = p.record_metrics(selected, evaluated)
    a.a158.compare(metric, a.independent_metrics(selected, evaluated))
    assert metric["d_GG"] == -1. and metric["d_LL"] == 1.
    assert metric["interaction"] == 2. and metric["interaction_crossproduct"] == 4.
    # Each endpoint uses both64-world halves. Changing only one half cannot
    # silently become a full-panel effect or an action re-selection.
    for world in evaluated["worlds"][:64]:
        world["outcomes"]["LL"].reverse()
    metric = p.record_metrics(selected, evaluated)
    a.a158.compare(metric, a.independent_metrics(selected, evaluated))
    assert metric["d_LL"] == 0. and metric["first_half_d_LL"] == -1. and metric["second_half_d_LL"] == 1.
    assert metric["interaction_crossproduct"] == 0.
    tied = fixture_record(sample, "selection", real=True)
    empty = p.simulate(sample, "evaluation", "a"*64, tied)
    metric = p.record_metrics(tied, empty)
    assert all(value == 0 for value in metric.values())
    a.a158.compare(metric, a.independent_metrics(tied, empty))


def stats_rows(variable=True):
    rows = []
    for cell in range(36):
        for within in range(5):
            g = (.1 if cell % 2 else -.1) + ((within-2)*.01 if variable else 0)
            l = -g
            rows.append({"index": len(rows), "pair": len(rows), "cell": cell,
                         "model_id": h.MODELS[cell//12], "phase": (cell//4)%3,
                         "actor": (cell//2)%2, "source_planner": p.POLICIES[cell%2],
                         "d_GG": g, "d_LL": l, "interaction": l-g,
                         "interaction_crossproduct": (l-g)**2, "changed": True})
    return rows


@pytest.mark.parametrize("variable", [False, True])
def test_stratified_statistics_match_independent_reconstruction(variable):
    rows = stats_rows(variable)
    stats = p.statistics(rows)
    a.a158.compare(stats, a.independent_statistics(rows))
    assert abs(stats["estimates"]["d_GG"]["mean"]) < 1e-12
    if not variable:
        # Different stratum means must not create sampling variance when every
        # root within each fixed stratum has the same response.
        assert stats["estimates"]["d_GG"]["half_width"] < 1e-12
    else:
        # Five deviations (-.02,-.01,0,.01,.02) have sample variance.00025.
        expected_se = math.sqrt(.00025/(5*36))
        assert abs(stats["estimates"]["d_GG"]["standard_error"]-expected_se) < 1e-12
        assert abs(stats["estimates"]["d_GG"]["degrees_of_freedom"]-144.) < 1e-9
    damaged = copy.deepcopy(rows)
    damaged[1]["pair"] = damaged[0]["pair"]
    with pytest.raises(AssertionError):
        p.statistics(damaged)
    with pytest.raises(AssertionError):
        a.independent_statistics(damaged)
    with pytest.raises(AssertionError):
        p.statistics(rows[:-1])


def test_collector_requires_selection_seal_and_refuses_existing_record(tmp_path):
    sample = tiny_sample()
    with pytest.raises(FileNotFoundError):
        p.collect_task(tmp_path, sample, "evaluation", "a"*64)
    assert not p.record_path(tmp_path, "evaluation", 0).exists()
    selected = p.collect_task(tmp_path, sample, "selection", "a"*64)
    path = p.record_path(tmp_path, "selection", 0)
    original = path.read_bytes()
    with pytest.raises(AssertionError):
        p.collect_task(tmp_path, sample, "selection", "a"*64)
    assert path.read_bytes() == original
    h.old.save_new(tmp_path/"selection_lock.json", {"manifest": "a"*64, "files": {"0": selected["sha256"]}})
    evaluated = p.collect_task(tmp_path, sample, "evaluation", "a"*64)
    record = h.old.read_json(p.record_path(tmp_path, "evaluation", 0))
    assert record["selection_sha256"] == selected["sha256"]
    assert record["worlds"] == [] and evaluated["sha256"] == h.old.file_hash(p.record_path(tmp_path, "evaluation", 0))


def test_actual_windows_spawn_imports_production_and_independent_auditor(tmp_path):
    sample = tiny_sample()
    fixture = tmp_path/"sample.json"
    h.old.save_new(fixture, sample)
    driver = tmp_path/"spawn_fixture.py"
    repository = Path(__file__).resolve().parents[1]
    driver.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repository)!r})\n"
        "import multiprocessing\n"
        "from pathlib import Path\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from splendor_ai.alphazero import planner_preference_159 as p\n"
        "def exercise(path):\n"
        "    sample = p.old.read_json(Path(path))\n"
        "    selected = p.simulate(sample, 'selection', 'a'*64)\n"
        "    return p.auditor().verify_record(sample, selected, 'selection', replay=True)\n"
        "if __name__ == '__main__':\n"
        "    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=p.h.initialize_worker) as pool:\n"
        "        result = pool.submit(exercise, sys.argv[1]).result()\n"
        "    assert result['replayed_branches'] == 8\n"
        "    print('159 actual spawn and independent replay passed')\n", encoding="utf-8")
    run = subprocess.run([sys.executable, str(driver), str(fixture)], cwd=repository,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout+run.stderr
    assert "159 actual spawn and independent replay passed" in run.stdout


@pytest.mark.parametrize("g,l,width,label", [
    ((-.08, -.04), (.04, .08), .02, "material_conditional_preference_reversal"),
    ((.04, .08), (-.08, -.04), .02, "material_reverse_direction_conditional_preference_reversal"),
    ((.04, .08), (.04, .08), .02, "learned_selected_moves_better_under_both_conditional_objectives"),
    ((-.08, -.04), (-.08, -.04), .02, "greedy_selected_moves_better_under_both_conditional_objectives"),
    ((-.01, .01), (-.01, .01), .01, "small_average_fixed_choice_effects_only"),
    ((-.025, .01), (-.01, .01), .0175, "inconclusive_for_material_classification"),
    ((-.12, -.04), (.04, .12), .04, "inconclusive_precision_target_not_met"),
])
def test_classification_requires_precision_and_strict_material_bounds(g, l, width, label):
    estimates = {"d_GG": {"simultaneous_ci95": list(g), "half_width": width},
                 "d_LL": {"simultaneous_ci95": list(l), "half_width": width}}
    actual = p.classification(estimates)
    assert actual == a.independent_classification(estimates)
    assert actual["label"] == label


@pytest.mark.parametrize("fail_audit", [False, True])
def test_pipeline_seals_and_audits_all_selections_before_evaluation(tmp_path, monkeypatch, fail_audit):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_freeze", lambda: "a"*64)
    h.old.save_new(tmp_path/"panel.json", {"samples": [tiny_sample()]})
    calls = []
    def tasks(base, samples, stage, binding, audit=False):
        calls.append((stage, audit))
        if fail_audit and audit:
            raise ValueError("Deliberate failed selection audit")
    monkeypatch.setattr(p, "run_tasks", tasks)
    monkeypatch.setattr(p, "finish", lambda samples, binding: calls.append(("finish", False)))
    if fail_audit:
        with pytest.raises(ValueError, match="Deliberate"):
            p.run()
        assert calls == [("selection", False), ("selection", True)]
        assert (tmp_path/"failure.json").exists()
    else:
        p.run()
        assert calls == [("selection", False), ("selection", True),
                         ("evaluation", False), ("evaluation", True), ("finish", False)]
    with pytest.raises(AssertionError, match="One-shot"):
        p.run()

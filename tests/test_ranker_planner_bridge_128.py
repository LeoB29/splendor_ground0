"""Artificial saved-world cases; never open any experiment or model artifact."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.ranker_planner_bridge_128 import analysis, independent


CELLS = tuple((fit, seat, phase) for fit in (161001, 161002, 161003)
              for seat in (0, 1) for phase in (0, 1, 2))
MODELS = ("ridge", "hgb")


def noop():
    pass


def artificial_pair(index=0, outsiders=("a", "b", "c"), ridge="b", hgb="c",
                    common_count=3, halves=None):
    """A genuine root-shaped fixture with independently specified half payoffs."""
    fit, seat, phase = CELLS[index // 40]
    spec = {"index": index, "game_index": 10000 + index, "fit": fit,
            "seat": seat, "phase": phase, "snapshot_sha256": f"{index:064x}"}
    first = [f"common{j}" for j in range(common_count)]
    keys = first + list(outsiders)
    anchor = outsiders[0] if outsiders else None
    if len(outsiders) < 2:
        ridge = hgb = anchor
    prediction = {**spec, "outsider_keys": list(outsiders), "anchor_key": anchor,
                  "scores": {name: [float(key == chosen) for key in outsiders]
                             for name, chosen in (("ridge", ridge), ("hgb", hgb))},
                  "choices": {"ridge": ridge, "hgb": hgb}}
    actions = [{"type": "artificial", "key": key} for key in keys]

    def candidate(offered):
        indices = [keys.index(key) for key in offered]
        return {"keys": offered, "indices": indices,
                "actions": [deepcopy(actions[j]) for j in indices]}

    anchor_keys = first + ([anchor] if anchor is not None else [])
    seeds = list(range(50000, 50128))
    if halves is None:
        payoffs = {key: -1 for key in keys}
        payoffs.update({"a": 0, "b": 1, "c": -1})
        halves = (payoffs, payoffs)
    root = {"manifest": "artificial-study", "stage": "main", "spec": deepcopy(spec),
            "union": {"keys": keys, "actions": actions},
            "candidates": {"anchor": candidate(anchor_keys), "raw": candidate(anchor_keys),
                           "relations": candidate(anchor_keys), "teacher": candidate(keys)},
            "proposals": {"anchor": {"group_keys": list(keys),
                "learned_group_scores": [20. - j for j in range(len(keys))]}},
            "selection": {"candidate_indices": list(range(len(keys))),
                "world_seeds": seeds,
                "worlds": [{"seed": seed, "outcomes": [
                    {"wdl": halves[wi // 64][key], "reason": "completed", "winner": None}
                    for key in keys]} for wi, seed in enumerate(seeds)]}}
    return prediction, root


def checked_row(prediction, root):
    row = analysis.row_from_root(prediction, root)
    assert independent.reconstruct(prediction, root) == row
    return row


def assert_rejected(prediction, root):
    for function in (analysis.row_from_root, independent.reconstruct):
        with pytest.raises(ValueError):
            function(prediction, root)


def test_opposite_half_rankings_select_then_assess_on_the_other_half():
    first = {"common0": 0, "common1": -1, "common2": -1, "a": 1, "b": -1}
    second = {**first, "a": -1, "b": 1}
    row = checked_row(*artificial_pair(outsiders=("a", "b"), hgb="a",
                                       halves=(first, second)))
    a, b = row["directions"]
    assert a["selection_range"] == [0, 64]
    assert a["assessment_range"] == [64, 128]
    assert b["selection_range"] == [64, 128]
    assert b["assessment_range"] == [0, 64]
    assert a["choices"] == {"anchor": "a", "ridge": "common0", "hgb": "a"}
    assert b["choices"] == {"anchor": "common0", "ridge": "b", "hgb": "common0"}
    assert a["delta_wdl_sums"] == {"ridge": 64, "hgb": 0}
    assert b["delta_wdl_sums"] == {"ridge": -64, "hgb": 0}
    assert row["delta_numerators"] == {"ridge": 0, "hgb": 0}


def test_signed_score_conversion_is_half_a_point_per_wdl_unit():
    row = checked_row(*artificial_pair())
    assert row["proposals"] == {"anchor": "a", "ridge": "b", "hgb": "c"}
    assert row["candidates"]["ridge"] == ["common0", "common1", "common2", "b"]
    for direction in row["directions"]:
        assert direction["choices"] == {"anchor": "a", "ridge": "b", "hgb": "common0"}
        assert direction["delta_wdl_sums"] == {"ridge": 64, "hgb": -64}
    assert row["delta_numerators"] == {"ridge": 128, "hgb": -128}


def test_first_offered_action_wins_planner_ties_and_first_outsider_wins_score_ties():
    prediction, root = artificial_pair()
    for world in root["selection"]["worlds"]:
        for outcome in world["outcomes"]:
            outcome["wdl"] = 0
    prediction["scores"]["ridge"] = [3., 3., 3.]
    prediction["choices"]["ridge"] = "a"
    row = checked_row(prediction, root)
    assert row["proposals"]["ridge"] == "a"
    assert all(set(direction["choices"].values()) == {"common0"}
               for direction in row["directions"])
    assert row["delta_numerators"] == {"ridge": 0, "hgb": 0}
    prediction["choices"]["ridge"] = "b"
    assert_rejected(prediction, root)


def test_changed_outsiders_can_both_lose_to_the_same_common_action():
    prediction, root = artificial_pair()
    for world in root["selection"]["worlds"]:
        for key, outcome in zip(root["union"]["keys"], world["outcomes"], strict=True):
            outcome["wdl"] = 1 if key == "common1" else -1
    row = checked_row(prediction, root)
    assert len(set(row["proposals"].values())) == 3
    assert all(set(direction["choices"].values()) == {"common1"}
               for direction in row["directions"])
    assert row["delta_numerators"] == {"ridge": 0, "hgb": 0}


@pytest.mark.parametrize("outsiders", [(), ("a",)])
def test_empty_and_singleton_outsider_cases_are_retained(outsiders):
    row = checked_row(*artificial_pair(outsiders=outsiders))
    assert row["outsider_count"] == len(outsiders)
    assert row["delta_numerators"] == {"ridge": 0, "hgb": 0}
    assert row["unknown_selection_outcomes"] == 0
    assert row["candidates"]["ridge"] == row["candidates"]["anchor"]


@pytest.mark.parametrize("common_count", [1, 2])
def test_short_legal_support_keeps_only_the_recorded_actions(common_count):
    row = checked_row(*artificial_pair(outsiders=(), common_count=common_count))
    assert row["candidates"]["anchor"] == [f"common{j}" for j in range(common_count)]
    assert row["proposals"] == dict.fromkeys(("anchor", "ridge", "hgb"))
    assert all(set(direction["choices"].values()) == {"common0"}
               for direction in row["directions"])


@pytest.mark.parametrize("corruption", ["unknown", "missing_outcome", "missing_world",
                                        "extra_world", "out_of_range", "boolean", "cutoff"])
def test_unresolved_or_incomplete_worlds_are_rejected_even_without_outsiders(corruption):
    prediction, root = artificial_pair(outsiders=())
    worlds = root["selection"]["worlds"]
    if corruption == "unknown":
        worlds[0]["outcomes"][0]["wdl"] = None
    elif corruption == "missing_outcome":
        worlds[0]["outcomes"].pop()
    elif corruption == "missing_world":
        worlds.pop()
    elif corruption == "extra_world":
        worlds.append(deepcopy(worlds[-1]))
    elif corruption == "out_of_range":
        worlds[0]["outcomes"][0]["wdl"] = 2
    elif corruption == "boolean":
        worlds[0]["outcomes"][0]["wdl"] = True
    else:
        worlds[0]["outcomes"][0]["reason"] = "repetition_cutoff"
    assert_rejected(prediction, root)


@pytest.mark.parametrize("field", ["index", "game_index", "fit", "seat", "phase", "snapshot_sha256"])
def test_prediction_identity_must_match_its_root(field):
    prediction, root = artificial_pair()
    prediction[field] = "different" if field == "snapshot_sha256" else prediction[field] + 1
    assert_rejected(prediction, root)


@pytest.mark.parametrize("corruption", ["choice", "score_nan", "anchor", "outsider_order",
                                        "duplicate_union", "candidate_index", "candidate_key",
                                        "candidate_action", "selection_index", "world_seed"])
def test_sealed_choices_and_union_key_index_action_alignment(corruption):
    prediction, root = artificial_pair()
    if corruption == "choice":
        prediction["choices"]["ridge"] = "c"
    elif corruption == "score_nan":
        prediction["scores"]["ridge"][0] = float("nan")
    elif corruption == "anchor":
        prediction["anchor_key"] = "b"
    elif corruption == "outsider_order":
        prediction["outsider_keys"].reverse()
    elif corruption == "duplicate_union":
        root["union"]["keys"][1] = root["union"]["keys"][0]
    elif corruption == "candidate_index":
        root["candidates"]["anchor"]["indices"][0] = 1
    elif corruption == "candidate_key":
        root["candidates"]["teacher"]["keys"] = list(reversed(root["union"]["keys"]))
    elif corruption == "candidate_action":
        root["candidates"]["anchor"]["actions"][0] = {"type": "different"}
    elif corruption == "selection_index":
        root["selection"]["candidate_indices"][0] = 1
    else:
        root["selection"]["worlds"][0]["seed"] = -1
    assert_rejected(prediction, root)


def test_assessment_worlds_cannot_affect_this_diagnostic():
    prediction, root = artificial_pair()
    expected = checked_row(prediction, root)
    root["assessment"] = {"worlds": "deliberately unusable; not an input"}
    assert checked_row(prediction, root) == expected


def artificial_panel(mixed=True):
    positive = checked_row(*artificial_pair())
    negative = checked_row(*artificial_pair(ridge="c", hgb="b"))
    rows = []
    for index in range(720):
        row = deepcopy(positive if not mixed or index % 2 == 0 else negative)
        fit, seat, phase = CELLS[index // 40]
        row.update(index=index, game_index=10000 + index, fit=fit, seat=seat,
                   phase=phase, snapshot_sha256=f"{index:064x}")
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def mixed_panel_summary():
    rows = artificial_panel()
    summary = analysis.statistics(rows, boundary=noop)
    independent.verify_statistics(rows, summary, boundary=noop)
    return rows, summary


def test_bootstrap_keeps_folds_in_roots_and_models_jointly_paired(mixed_panel_summary):
    rows, summary = mixed_panel_summary
    assert len(rows) == summary["states"] == summary["source_games"] == 720
    assert summary["states_per_cell"] == 40
    assert summary["cell_counts"] == [40] * 18
    assert summary["selector_worlds"] == 64
    assert summary["directions_per_root"] == 2
    assert summary["root_score_denominator"] == 256
    assert summary["bootstrap_samples"] == 50000
    assert summary["bootstrap_seed"] == 2813000000
    assert summary["bootstrap_chunk"] == 1000
    assert summary["bootstrap_totals_shape"] == [50000, 2]
    ridge = summary["contrasts"]["ridge_minus_anchor"]
    hgb = summary["contrasts"]["hgb_minus_anchor"]
    for contrast in (ridge, hgb):
        assert contrast["integer_numerator_total"] == 0
        assert contrast["mean_score_difference"] == contrast["mean_percentage_points"] == 0
        assert contrast["ci975"][0] < 0 < contrast["ci975"][1]
        assert contrast["ci975_percentage_points"] == pytest.approx(
            [100 * endpoint for endpoint in contrast["ci975"]])
    assert ridge["ci975"] == pytest.approx([-hgb["ci975"][1], -hgb["ci975"][0]])
    assert summary["fourth_selected"]["anchor"] == {"direction_choices": 1440, "fraction": 1.}
    for model in MODELS:
        assert summary["proposal_changes"][model] == {"roots": 720, "fraction": 1.}
        assert summary["final_action_changes"][model] == {"direction_choices": 1440, "fraction": 1.}
        assert summary["fourth_selected"][model] == {"direction_choices": 720, "fraction": .5}


def test_constant_half_point_effect_has_exact_mean_and_degenerate_interval():
    rows = artificial_panel(mixed=False)
    summary = analysis.statistics(rows, boundary=noop)
    independent.verify_statistics(rows, summary, boundary=noop)
    for model, effect in (("ridge", .5), ("hgb", -.5)):
        contrast = summary["contrasts"][model + "_minus_anchor"]
        assert contrast["integer_numerator_total"] == int(effect * 720 * 256)
        assert contrast["mean_score_difference"] == effect
        assert contrast["mean_percentage_points"] == effect * 100
        assert contrast["ci975"] == [effect, effect]
        assert contrast["ci975_percentage_points"] == [effect * 100, effect * 100]


def test_cancelling_folds_are_not_bootstrapped_as_independent_observations():
    first = {"common0": 0, "common1": -1, "common2": -1, "a": 1, "b": -1}
    second = {**first, "a": -1, "b": 1}
    template = checked_row(*artificial_pair(outsiders=("a", "b"), hgb="a",
                                            halves=(first, second)))
    rows = artificial_panel()
    for row in rows:
        identity = {key: row[key] for key in
                    ("index", "game_index", "fit", "seat", "phase", "snapshot_sha256")}
        row.clear()
        row.update(deepcopy(template), **identity)
    summary = analysis.statistics(rows, boundary=noop)
    independent.verify_statistics(rows, summary, boundary=noop)
    for model in MODELS:
        contrast = summary["contrasts"][model + "_minus_anchor"]
        assert contrast["mean_score_difference"] == 0
        assert contrast["ci975"] == [0., 0.]


def test_degenerate_roots_remain_in_summary_denominators():
    rows = artificial_panel(mixed=False)
    rows[0] = checked_row(*artificial_pair(0, outsiders=()))
    rows[1] = checked_row(*artificial_pair(1, outsiders=("a",)))
    summary = analysis.statistics(rows, boundary=noop)
    independent.verify_statistics(rows, summary, boundary=noop)
    assert summary["outsider_support_counts"] == {"0": 1, "1": 1, "2": 0,
                                                  "3": 718, "4": 0, "5": 0}
    assert summary["degenerate_support_roots_retained"] == 2
    assert summary["unknown_selection_outcomes"] == 0
    for model, sign in (("ridge", 1), ("hgb", -1)):
        contrast = summary["contrasts"][model + "_minus_anchor"]
        assert contrast["integer_numerator_total"] == sign * 718 * 128
        assert contrast["mean_score_difference"] == pytest.approx(sign * 718 / 1440)
        assert summary["proposal_changes"][model]["roots"] == 718
        assert summary["proposal_changes"][model]["fraction"] == pytest.approx(718 / 720)


@pytest.mark.parametrize("corruption", ["missing", "duplicate_index", "out_of_range_index",
                                        "boolean_index", "duplicate_game", "negative_game",
                                        "invalid_fit", "unbalanced_cell", "row_order"])
def test_panel_ids_source_games_and_strata_cannot_change(mixed_panel_summary, corruption):
    original, summary = mixed_panel_summary
    rows = deepcopy(original)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate_index":
        rows[0]["index"] = 1
    elif corruption == "out_of_range_index":
        rows[-1]["index"] = 720
    elif corruption == "boolean_index":
        rows[0]["index"] = False
    elif corruption == "duplicate_game":
        rows[0]["game_index"] = rows[1]["game_index"]
    elif corruption == "negative_game":
        rows[0]["game_index"] = -1
    elif corruption == "invalid_fit":
        rows[0]["fit"] = 161999
    elif corruption == "unbalanced_cell":
        rows[0]["phase"] = 1
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError):
        analysis.statistics(rows, boundary=noop)
    with pytest.raises(ValueError):
        independent.verify_statistics(rows, summary, boundary=noop)


def test_independent_statistics_reject_changed_arithmetic(mixed_panel_summary):
    rows, original = mixed_panel_summary
    summary = deepcopy(original)
    summary["contrasts"]["ridge_minus_anchor"]["mean_score_difference"] = .01
    with pytest.raises(ValueError):
        independent.verify_statistics(rows, summary, boundary=noop)


def test_input_inventory_rejects_changed_hashes_and_workspace_escape(tmp_path, monkeypatch):
    from tools.ranker_planner_bridge_128 import run
    workspace = tmp_path / "artificial_workspace"
    workspace.mkdir()
    monkeypatch.setattr(run, "ROOT", workspace)
    source = workspace / "input.json"
    source.write_text('{"artificial": true}', encoding="utf-8")
    inventory = run.Inputs(noop)
    good = {"path": "input.json", "sha256": run.sha(source)}
    with pytest.raises(ValueError):
        inventory.add({**good, "sha256": "0" * 64})
    assert inventory.add(good) == source
    with pytest.raises(ValueError):
        inventory.add({**good, "sha256": "1" * 64})
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        inventory.add({"path": "../outside.json", "sha256": run.sha(outside)})
    source.write_text('{"artificial": false}', encoding="utf-8")
    with pytest.raises(ValueError):
        inventory.revalidate()


def test_new_output_is_exclusive_and_cannot_escape_namespace(tmp_path, monkeypatch):
    from tools.ranker_planner_bridge_128 import run
    namespace = tmp_path / "artificial_diagnostic"
    namespace.mkdir()
    monkeypatch.setattr(run, "BASE", namespace)
    destination = namespace / "receipt.json"
    run.new_json(destination, {"original": 1})
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        run.new_json(destination, {"replacement": 2})
    assert destination.read_bytes() == original
    with pytest.raises(ValueError):
        run.new_json(tmp_path / "outside.json", {"outside": True})
    assert not (tmp_path / "outside.json").exists()


def test_output_ceiling_reserves_space_for_a_control_stop_receipt(tmp_path, monkeypatch):
    from tools.ranker_planner_bridge_128 import run
    namespace = tmp_path / "artificial_diagnostic"
    namespace.mkdir()
    monkeypatch.setattr(run, "BASE", namespace)
    monkeypatch.setattr(run, "output_bytes", lambda: run.MAX_OUTPUT - run.RECEIPT_RESERVE)
    with pytest.raises(ValueError):
        run.new_json(namespace / "result.json", {"ordinary": True})
    run.new_json(namespace / "stop.json", {"complete": False}, control=True)
    assert not (namespace / "result.json").exists()
    assert run.read(namespace / "stop.json") == {"complete": False}


def test_supervisor_authentication_requires_the_token_and_exact_live_parent(monkeypatch):
    import psutil
    from tools.ranker_planner_bridge_128 import run
    lease = {"pid": 12345, "created": 456.25, "token": "artificial-token"}
    monkeypatch.setattr(run, "read", lambda path: dict(lease))
    monkeypatch.setattr(run.os, "getppid", lambda: lease["pid"])
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(
        is_running=lambda: True, create_time=lambda: lease["created"]))
    run.validate_supervisor(dict(lease))
    for corruption in ("wrong_token", "different_parent", "dead_parent", "reused_pid"):
        supplied = dict(lease)
        with monkeypatch.context() as patch:
            if corruption == "wrong_token":
                supplied["token"] = "different-token"
            elif corruption == "different_parent":
                patch.setattr(run.os, "getppid", lambda: lease["pid"] + 1)
            else:
                patch.setattr(psutil, "Process", lambda pid: SimpleNamespace(
                    is_running=lambda: corruption != "dead_parent",
                    create_time=lambda: lease["created"] + (corruption == "reused_pid")))
            with pytest.raises(ValueError):
                run.validate_supervisor(supplied)


@pytest.mark.parametrize("bound", ["elapsed", "rss", "output", "ram", "commit", "disk"])
def test_resource_guard_rejects_each_bound_without_accessing_host_resources(monkeypatch, bound):
    from tools.alternative_ranker_128 import common
    from tools.ranker_planner_bridge_128 import run
    snapshot = {"available_bytes": 4 * 1024**3, "commit_headroom_bytes": 4 * 1024**3,
                "disk_free_bytes": 10 * 1024**3}
    elapsed, rss, output_size = 100, 128 * 1024**2, 0
    if bound == "elapsed":
        elapsed = run.MAX_SECONDS + 1
    elif bound == "rss":
        rss = run.MAX_RSS + 1
    elif bound == "output":
        output_size = run.MAX_OUTPUT + 1
    elif bound == "ram":
        snapshot["available_bytes"] = 128 * 1024**2 - 1
    elif bound == "commit":
        snapshot["commit_headroom_bytes"] = 1024**3 - 1
    else:
        snapshot["disk_free_bytes"] = run.DISK_RESERVE - 1
    monkeypatch.setattr(run.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(common, "resource_snapshot", lambda: dict(snapshot))
    monkeypatch.setattr(run, "output_bytes", lambda: output_size)
    fake_process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss))
    with pytest.raises(ValueError):
        run.resource_check(0, process=fake_process)

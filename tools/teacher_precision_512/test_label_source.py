"""TRAIN-only identity, fixed metadata sampling and inherited-source checks."""
import copy

import pytest

from . import label_source as p


def metadata_specs():
    rows = []
    for panel in p.inherited.PANELS:
        for fit, seat, phase in p.CELLS:
            for ordinal in range(20):
                index = len(rows)
                rows.append({"index": index, "row_id": f"{panel}:{index}:0", "fit": fit,
                             "seat": seat, "phase": phase, "support": ordinal % 8+1,
                             "privacy": {"own": ordinal % 2, "opponent": (ordinal//2) % 2},
                             "descriptor": {"source_panel": panel}})
                row = rows[-1]
                row.update(row_index=11950+index, turn=0, snapshot_sha256="0"*64,
                           old_sums_wdl=[0]*row["support"])
                row["descriptor"].update(spec={"game_index": index, **{k: row[k] for k in
                    ("fit", "seat", "phase", "turn", "privacy", "snapshot_sha256")}},
                    source_spec={"environment_seed": index})
    return rows


def test_metadata_sampling_is_outcome_blind_and_bounded():
    rows = metadata_specs()
    expected = p.sample_plan(rows)
    assert len(expected["probe_indices"]) == 18
    assert len(expected["stress_indices"]) <= 39
    assert len(expected["full_replay_indices"]) <= 57
    assert {(rows[i]["fit"], rows[i]["seat"], rows[i]["phase"]) for i in expected["probe_indices"]} == set(p.CELLS)
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(score=-999, wins=999, timing=1e99, outcomes=[1])
    assert p.sample_plan(changed) == expected
    for i in expected["full_replay_indices"]:
        assert 0 <= i < 720


@pytest.mark.parametrize("damage", ["duplicate", "cell", "order", "support"])
def test_bad_population_rejected(damage):
    rows = metadata_specs()
    if damage == "duplicate": rows[1]["row_id"] = rows[0]["row_id"]
    if damage == "cell": rows[0]["phase"] = 1
    if damage == "order": rows[0], rows[1] = rows[1], rows[0]
    if damage == "support": rows[0]["support"] = 9
    with pytest.raises((ValueError, AssertionError)):
        p.sample_plan(rows)


@pytest.mark.parametrize("index", [0, 360, 719])
def test_real_inherited_selection_mapping_and_tamper(tmp_path, index):
    """Read closed evidence only; no historical rollouts/inference are repeated."""
    data = p.c.old.read_json(p.c.PARENT/"data_lock.json")
    descriptor = data["parent"]["added_sources"][index]
    saved = p.c.verified(descriptor["selection"])
    target = p.inherited.selection_targets(saved, descriptor)
    old_spec = descriptor["spec"]
    spec = {"index": index, "row_index": 11950+index,
            "row_id": p.inherited.row_identity(descriptor)["id"], "descriptor": descriptor,
            "support": len(target["old_sums_wdl"]), "old_sums_wdl": target["old_sums_wdl"],
            **{k: old_spec[k] for k in ("fit", "seat", "phase", "turn", "privacy", "snapshot_sha256")}}
    source = p.load_source(spec, "fixture", tmp_path, check_plan=False)
    assert source["old_sums_wdl"] == spec["old_sums_wdl"]
    assert len(source["old_block"]["worlds"]) == 128
    assert set(source) == {"snapshot", "union", "old_selection", "source_root", "source_audit", "old_block", "old_sums_wdl"}
    damaged = copy.deepcopy(spec)
    damaged["old_sums_wdl"][0] += 1
    with pytest.raises((ValueError, AssertionError)):
        p.load_source(damaged, "fixture", tmp_path, check_plan=False)


def test_fixture_source_cannot_target_production():
    with pytest.raises((ValueError, AssertionError)):
        p.load_source({"fixture_payload": {}}, "fixture", p.c.OUTPUT)


def test_all720_real_sources_prepare_in_isolated_directory_and_seeds_are_disjoint(tmp_path):
    """Authenticate all old sources; no production labels, inference or replay."""
    from . import training_source, labels, engine
    parent = training_source.closed_parent()
    training_source.prepare("fixture", tmp_path, parent=parent)
    plan_ref = p.prepare("fixture", tmp_path)
    plan = p.load_plan("fixture", tmp_path)
    assert p.c.verified(plan_ref) == plan
    assert len(plan["specs"]) == 720 and len(p.probe_specs(plan)) == 18
    assert [s["row_index"] for s in plan["specs"]] == list(range(11950, 12670))
    assert all(s["row_id"] == parent["retired_evaluation_panels"]["ids"][i] for i, s in enumerate(plan["specs"]))
    old_world_seeds = set()
    for spec in plan["specs"]:
        saved = p.c.verified(spec["descriptor"]["selection"])
        seeds = saved["selection"]["world_seeds"]
        assert len(seeds) == 128 and old_world_seeds.isdisjoint(seeds)
        old_world_seeds.update(seeds)
    new_world_seeds = {seed for i in range(720) for b in (1, 2, 3) for seed in labels.world_seeds(i, b)}
    fresh_evaluation_seeds = {seed for stage, count in (("probe", 18), ("main", 360), ("test", 18))
                              for i in range(count) for block in ("selection", "assessment")
                              for seed in engine.world_seeds(stage, i, block)}
    fresh_source_seeds = {p.c.CONFIG["source_seed"][stage]+i for stage, count in (("probe", 54), ("main", 1080))
                          for i in range(count)}
    assert len(old_world_seeds) == 720*128 and len(new_world_seeds) == 720*384
    assert old_world_seeds.isdisjoint(new_world_seeds | fresh_evaluation_seeds | fresh_source_seeds)
    assert new_world_seeds.isdisjoint(fresh_evaluation_seeds | fresh_source_seeds)
    damaged = copy.deepcopy(plan)
    damaged["specs"][0]["support"] = 1 if damaged["specs"][0]["support"] != 1 else 2
    p._plan_path(tmp_path).write_text(__import__("json").dumps(damaged))
    with pytest.raises((ValueError, AssertionError)):
        p.load_plan("fixture", tmp_path)

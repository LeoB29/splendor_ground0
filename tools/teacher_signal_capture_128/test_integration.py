"""Authenticate every proposed source and check fixed choices without assessment."""
from . import common as c, source, models, engine


def test_complete_original_corpus_panel_and_frozen_choices_are_compatible(tmp_path, monkeypatch):
    parent = models.closed_parent()
    selected = source.build(parent["training_data"])
    assert selected["training_data"] == parent["training_data"]
    assert len(selected["inventory"]) == len(selected["source_evidence"])
    engine.initialize_models(tmp_path, {"models": parent["models"]})
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not generate a production assessment")
    monkeypatch.setattr(c.old, "trajectory", forbidden)
    all_old, all_new = set(), set()
    for stage, specs in selected["schedule"].items():
        assert len(specs) == c.CONFIG["states"][stage]
        for spec in specs:
            original = source.load_root(spec)
            state = c.old.deserialize_state_snapshot(original["snapshot"])
            proposals = engine.frozen_proposals(state, original["candidates"], spec["fit"])
            locked = engine.rebuild_choices(original["worlds"], proposals, original["candidates"])
            assert locked["choices"]["teacher"] == original["choice"]
            assert locked["means"] == original["means"]
            assert locked["choices"]["anchor_supported"] == locked["choices"]["anchor"]
            assert locked["choices"]["anchor_outsider"] == locked["choices"]["original_fourth"]
            all_old.update(original["schedule"]["world_seeds"])
            seeds = engine.world_seeds(stage, spec["index"])
            assert all_new.isdisjoint(seeds)
            all_new.update(seeds)
    assert len(all_new) == 354*128 and all_new.isdisjoint(all_old)

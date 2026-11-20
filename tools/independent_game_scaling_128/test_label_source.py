"""Identity-only nested independent-game population and bounded replay samples."""
import copy
from collections import Counter

import pytest

from . import label_source as p


def metadata_specs():
    rows = []
    for fit,seat,phase in p.CELLS:
        for ordinal in range(600):
            index = len(rows)
            rows.append(p.make_spec({"index":index,"game_index":index*3,"pair":index*3,
                "source_stage":"train","fit":fit,"seat":seat,"phase":phase,"turn":phase*20+seat,
                "privacy":{"own":ordinal%2,"opponent":(ordinal//2)%2},
                "canonical_support":ordinal%12+1,"snapshot_sha256":"0"*64,"search_stream":f"fixture:{index}"}))
    return rows


def test_nested_population_balanced_identity_only_and_full_probe_reused():
    specs = metadata_specs()
    partition,sample = p.partitions(specs),p.sample_plan(specs)
    assert partition["small"] == list(range(720)) and partition["large"] == list(range(11520))
    assert len(partition["medium"]) == 2880 and set(partition["small"]) < set(partition["medium"]) < set(partition["large"])
    cells = Counter((specs[i-720]["fit"],specs[i-720]["seat"],specs[i-720]["phase"]) for i in partition["medium"][720:])
    assert cells == {cell:120 for cell in p.CELLS}
    assert len(sample["probe_indices"]) == 18 and len(sample["stress_indices"]) <= 39 and len(sample["full_replay_indices"]) <= 57
    assert {(specs[i]["fit"],specs[i]["seat"],specs[i]["phase"]) for i in sample["probe_indices"]} == set(p.CELLS)
    assert set(sample["probe_indices"]) <= set(sample["full_replay_indices"])
    changed = copy.deepcopy(specs)
    for spec in changed: spec.update(outcome=-999,score=999,timing=1e50)
    assert p.partitions(changed) == partition and p.sample_plan(changed) == sample


@pytest.mark.parametrize("damage",["game","cell","order","support","rowid","global_index"])
def test_bad_new_population_cannot_be_sealed(damage):
    specs = metadata_specs()
    if damage == "game": specs[1]["game_index"] = specs[0]["game_index"]
    elif damage == "cell": specs[0]["phase"] = 2
    elif damage == "order": specs[0],specs[1] = specs[1],specs[0]
    elif damage == "support": specs[0]["support"] = 8
    elif damage == "rowid": specs[0]["row_id"] = "retired:0"
    else: specs[0]["row_index"] = 11950
    with pytest.raises((ValueError,AssertionError)): p.partitions(specs)


def test_fixture_cannot_target_production():
    with pytest.raises(ValueError): p.load_source({"fixture_payload":{}},"fixture",p.c.OUTPUT)


def test_all_seed_namespaces_separate_from_inherited_selection_worlds():
    from . import engine,labels,training_source
    data = p.c.old.read_json(p.c.PARENT/"data_lock.json")
    oldseeds = set()
    for descriptor in data["parent"]["added_sources"]:
        saved = p.c.verified(descriptor["selection"])
        seeds = saved["selection"]["world_seeds"]
        assert len(seeds) == 128 and oldseeds.isdisjoint(seeds)
        oldseeds.update(seeds)
    assert len(oldseeds) == 720*128
    fresh = {seed for stage,count in (("probe",18),("main",720),("test",18)) for i in range(count)
             for block in ("selection","assessment") for seed in engine.world_seeds(stage,i,block)}
    sources = {p.c.CONFIG["source_seed"][stage]+i for stage,count in p.c.CONFIG["source_games"].items() for i in range(count)}
    labels_set = {seed for i in range(10800) for seed in labels.world_seeds(i)}
    assert len(labels_set) == 10800*128 and oldseeds.isdisjoint(fresh|sources|labels_set)
    assert fresh.isdisjoint(sources|labels_set) and sources.isdisjoint(labels_set)
    historical = set(data["parent"]["training_disjointness"]["forbidden_environment_seeds"])
    assert sources.isdisjoint(historical|{i["environment_seed"] for i in data["added_identities"]})

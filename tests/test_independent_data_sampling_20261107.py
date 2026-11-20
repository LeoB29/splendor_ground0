from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import sys
import pytest

from tools.splendorust_independent_data_20261107 import sampling as s, schedule


@pytest.fixture(scope="module")
def selection():
    games=[]
    for spec in schedule.collection_schedule():
        game=spec["game_index"]
        roots=[dict(game_index=game,ply=i,split=spec["split"],row_sha256=s.digest([game,i,"row"]),
                    root_input_sha256=s.digest([game,i,"root"])) for i in range(23)]
        games.append(dict(spec=spec,artifact=dict(path=f"fake/{game}.json",bytes=1,sha256="a"*64),roots=roots))
    return s.build_selection(games,{"accepted":True})


def test_fixed_breadth_counts_and_complete_pass(selection):
    assert selection["narrow_game_ids"]==schedule.narrow_ids()
    assert Counter(x%2 for x in selection["narrow_game_ids"])=={0:96,1:96}
    assert selection["game_counts"]=={"train":1024,"validation":64,"offline":64}
    for cell in selection["cells"]:
        assert cell["phase_counts"]==[1024]*3
        assert len(cell["root_ids"])==3072
        assert sorted(i for b in cell["batches"] for i in b)==list(range(3072))
        assert len(cell["batches"])==96 and all(len(b)==32 for b in cell["batches"])
        count=Counter(g for g,_ in cell["root_ids"])
        assert set(count.values())=={16 if cell["breadth"]=="narrow" else 3}
        assert len(count)==(192 if cell["breadth"]=="narrow" else 1024)
    assert selection["cells"][0]["game_ids"]==selection["cells"][2]["game_ids"]
    assert selection["cells"][0]["root_ids"]!=selection["cells"][2]["root_ids"]


def test_thirds_integer_boundary():
    assert s.relative_thirds(list(range(8)))==([0,1],[2,3,4],[5,6,7])


def test_hash_targets_do_not_choose_roots(selection):
    games=deepcopy(selection["games"])
    for game in games:
        for root in game["roots"]:root["row_sha256"]="f"*64;root["root_input_sha256"]="e"*64
    new=s.build_selection(games,{"different_outcomes_not_inputs":True})
    assert new["cells"]==selection["cells"]
    assert new["narrow_game_ids"]==selection["narrow_game_ids"]


@pytest.mark.parametrize("mutation",[
    lambda x:x["cells"][0]["batches"][0].reverse(),
    lambda x:x["cells"][0]["batches"][0].__setitem__(0,True),
    lambda x:x["cells"][0].__setitem__("repetition",0.),
    lambda x:x["cells"][0]["root_ids"][0].__setitem__(0,1024),
    lambda x:x["games"][0]["spec"].__setitem__("game_index",False),
    lambda x:x["games"][0]["spec"].__setitem__("quota",512.),
    lambda x:x["games"][0]["roots"][0].__setitem__("ply",False),
    lambda x:x.__setitem__("selection_uses_outcomes",True),
])
def test_typed_or_changed_selection_rejected(selection,mutation):
    bad=deepcopy(selection);mutation(bad)
    with pytest.raises((ValueError,KeyError,TypeError)):s.validate_selection(bad,authenticate=False)


def test_short_narrow_game_is_not_replaced(selection):
    games=deepcopy(selection["games"]);g=selection["narrow_game_ids"][0]
    games[g]["roots"]=games[g]["roots"][:14]
    with pytest.raises(ValueError,match="quota"):s.build_selection(games,selection["source_receipt"])


def test_train_loader_never_reads_heldout(selection,monkeypatch):
    seen=[]
    def load(game,receipt,boundary):
        seen.append(game["spec"]["game_index"])
        assert game["spec"]["split"]=="train"
        return {"examples":deepcopy(game["roots"])}
    fake=SimpleNamespace(authenticate_selection_source=lambda *a,**k:None,load_selected_game=load)
    import tools.splendorust_independent_data_20261107 as package
    monkeypatch.setitem(sys.modules,package.__name__+".corpus",fake)
    monkeypatch.setattr(package,"corpus",fake,raising=False)
    loaded=s.load_selected(selection,0,"narrow")
    assert sorted(seen)==selection["narrow_game_ids"]
    assert len(loaded["train_examples"])==3072 and "heldout_examples" not in loaded


def test_actual_loader_rejects_changed_root_hash(selection,monkeypatch):
    def load(game,receipt,boundary):
        rows=deepcopy(game["roots"]);rows[0]["row_sha256"]="0"*64
        return {"examples":rows}
    fake=SimpleNamespace(authenticate_selection_source=lambda *a,**k:None,load_selected_game=load)
    import tools.splendorust_independent_data_20261107 as package
    monkeypatch.setitem(sys.modules,package.__name__+".corpus",fake);monkeypatch.setattr(package,"corpus",fake,raising=False)
    with pytest.raises(ValueError,match="inventory"):s.load_selected(selection,0,"narrow")


def test_heldout_gate_precedes_first_loader_read(selection,monkeypatch):
    from tools.splendorust_independent_data_20261107 import fit
    seen=[]
    def refuse(*a,**k):raise ValueError("four fits not committed")
    monkeypatch.setattr(fit,"validate_commitment",refuse)
    fake=SimpleNamespace(load_selected_game=lambda *a,**k:seen.append(1))
    import tools.splendorust_independent_data_20261107 as package
    monkeypatch.setitem(sys.modules,package.__name__+".corpus",fake);monkeypatch.setattr(package,"corpus",fake,raising=False)
    with pytest.raises(ValueError,match="not committed"):s.load_heldout(selection,{})
    assert seen==[]

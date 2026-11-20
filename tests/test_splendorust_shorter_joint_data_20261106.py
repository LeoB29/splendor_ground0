"""Scientific subset and actual corrected-collator regressions; no models."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from tools.splendorust_shorter_joint_20261106 import data as d


@pytest.fixture(scope="module")
def selection():
    games = []
    for g in range(256):
        split = "train" if g < 192 else "validation" if g < 224 else "offline"
        count = (48 if g < 11 else 52 if g == 11 else 56) if split == "train" else (
            66 if g == 223 else 56) if split == "validation" else (50 if g == 255 else 55)
        spec = {"game_index": g, "split": split, "seed_decimal": str(g + 1),
                "teacher_simulations": 512, "purpose": "single_generation_learning_pilot"}
        roots = [{"game_index": g, "ply": p * 2, "split": split,
                  "row_sha256": d.digest([g, p, "row"]), "root_input_sha256": d.digest([g, p, "root"])}
                 for p in range(count)]
        games.append({"spec": spec, "artifact": {"path": f"fake/g{g:03d}.json", "bytes": 1, "sha256": d.digest(g)}, "roots": roots})
    return d._build_selection(games, {"inputs": [], "metadata": {"fixture": True}})


def test_original_sampling_and_complete_first_pass(selection):
    original = d._build_original_selection(selection["games"], selection["source_receipt"])
    assert len(selection["cells"]) == 2
    for rep, cell in enumerate(selection["cells"]):
        source = next(x for x in original["cells"] if x["repetition"] == rep and x["breadth"] == "broad")
        assert cell["root_ids"] == source["root_ids"]
        assert cell["game_ids"] == list(range(192))
        assert set(Counter(g for g, _ in cell["root_ids"]).values()) == {16}
        assert cell["phase_counts"] == [1024] * 3
        assert cell["batches"] == source["batches"][:96]
        assert cell["permutations"] == source["permutations"][:1]
        assert sorted(i for batch in cell["batches"] for i in batch) == list(range(3072))
        assert cell["probe_batches"] == []
        assert cell["optimizer_steps"] == 96 and cell["presentations"] == 3072
        assert cell["original_cell_sha256"] == d.digest(source)
    assert selection["study_id"] == d.STUDY
    assert selection["original_selection"] == d.ORIGINAL_SELECTION
    assert selection["original_selection_canonical_sha256"] == d.digest(original)
    assert selection["domain"] == d.DOMAIN
    assert selection["historical_data_relabelled"] is False


def test_rank_uses_ids_not_outcomes_or_policy_confidence(selection):
    # Hash/target contents may change source identity but cannot rank positions.
    games = deepcopy(selection["games"])
    for game in games:
        for root in game["roots"]:
            root["row_sha256"] = d.digest(["changed full row with different outcome", root["game_index"], root["ply"]])
            root["root_input_sha256"] = d.digest(["changed root confidence", root["game_index"], root["ply"]])
    different = d._build_selection(games, selection["source_receipt"])
    assert different["games"] != selection["games"]
    assert different["game_rank"] == selection["game_rank"]
    assert different["cells"] == selection["cells"]
    assert different["repetitions"] == selection["repetitions"]
    x = {**games[0]["roots"][0], "winner": 0, "value": [1, -1], "policy": [.01]}
    y = {**x, "winner": 1, "value": [-1, 1], "policy": [.99]}
    assert d.root_identity(x) == d.root_identity(y)


@pytest.mark.parametrize("mutation", [
    lambda x: x["cells"][0]["root_ids"][0].__setitem__(0, 192),
    lambda x: x["cells"][0]["batches"][0].reverse(),
    lambda x: x["cells"][0]["probe_batches"].append([0]),
    lambda x: x["cells"][1]["root_ids"].pop(),
    lambda x: x["repetitions"][1].__setitem__("dense_game_ids", x["repetitions"][0]["dense_game_ids"]),
    lambda x: x.__setitem__("catalogue_sha256", "0" * 64),
    lambda x: x["games"][0]["roots"][0].__setitem__("game_index", True),
    lambda x: x["games"][0]["roots"].reverse(),
    lambda x: x["games"][0]["spec"].__setitem__("split", "offline"),
])
def test_frozen_selection_rejects_changed_sample_orders_or_identity(selection, mutation):
    bad = deepcopy(selection); mutation(bad)
    with pytest.raises((ValueError, KeyError)):
        d.validate_selection(bad, authenticate=False)


def test_selected_loader_reads_each_needed_game_once_and_preserves_rows(selection, monkeypatch):
    calls = []

    def load_game(document, spec, *, boundary):
        calls.append(spec["game_index"]); boundary()
        roots = selection["games"][spec["game_index"]]["roots"]
        return {"examples": [{**root, "tokens": [0.] * 1488, "mask": [True] + [False] * 80,
                               "policy": [1.] + [0.] * 80, "value": [1., -1.]} for root in roots]}

    monkeypatch.setattr(d.accepted, "load_game", load_game)
    monkeypatch.setattr(d.accepted, "read_artifact", lambda item, boundary: {})
    loaded = d._load_selected(selection, 0, "broad")
    assert len(calls) == len(set(calls)) == 256
    assert calls == sorted(set(selection["cells"][0]["game_ids"]) | set(range(192, 256)))
    assert len(loaded["train_examples"]) == 3072 and set(loaded["train_weights"]) == {1.}
    assert len(loaded["heldout_examples"]) == 3557
    per_game = Counter()
    for example, weight in zip(loaded["heldout_examples"], loaded["heldout_weights"]):
        per_game[example["game_index"]] += weight
    assert all(abs(value - 3557 / 64) < 1e-10 for value in per_game.values())
    assert loaded["batches"] == selection["cells"][0]["batches"]
    bad = deepcopy(selection)
    g = calls[0]; bad["games"][g]["roots"][0]["row_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="decoded roots"):
        d._load_selected(bad, 0, "broad")


def test_actual_saved_corrected_fixtures_and_same_torch_collator_without_model():
    _, _, source = d._source_context()
    binding = source["input_qualification"]
    indexed = {x["path"]: x for x in source["inputs"]}
    cases = d.accepted.read_artifact(binding["plan"])["loader_cases"]
    rows = [d.accepted.read_artifact(indexed[path]) for path in binding["row_paths"]]
    proof = d.accepted.qualify_captured_rows(cases, rows)
    assert len(proof["six_corrected_faces"]) == 6 and proof["actual_training_loader_used"]
    examples = [d.accepted.decode_row(row) for row in rows]
    examples = [x for x in examples if x["eligible"]]
    arrays = d.accepted.collate_arrays(examples, [1.] * len(examples))
    batch = d.collate_inputs(examples, [1.] * len(examples))
    assert all(batch[k].tolist() == value.tolist() for k, value in arrays.items())
    assert any("noble" in x["case_id"] for x in proof["rows"])
    assert any("private" in x["case_id"] for x in proof["rows"])
    assert all(x["rejected"] for x in proof["actual_loader_rejections"])


def test_actual_collator_canonical_value_rejections_without_model():
    root = {"tokens": [0.] * 1488, "mask": [True] + [False] * 80,
            "policy": [1.] + [0.] * 80, "value": [1., -1.]}
    assert d.collate([root], [1.])["value"].tolist() == [[1., -1.]]
    for value in [[1., 1.], [.5, -.5], [float("nan"), 0.], [1.]]:
        bad = {**root, "value": value}
        with pytest.raises(ValueError):
            d.collate([bad], [1.])


def test_accepted_sources_unchanged_and_exclusive_receipt(tmp_path):
    for path, expected in d.SOURCE_CODE.items():
        assert hashlib.sha256((d.ROOT / path).read_bytes()).hexdigest() == expected
    path = tmp_path / "receipt.json"
    d._write_exclusive(path, {"a": 1})
    with pytest.raises(FileExistsError):
        d._write_exclusive(path, {"a": 2})
    assert path.read_text() == '{"a":1}\n'

"""Synthetic source qualification checks; no external imports or model inference."""

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from tools.external_strength_qualification import adapter


def _catalog() -> adapter.Catalog:
    cards = {}
    index = 0
    for tier, count in ((1, 40), (2, 30), (3, 20)):
        for i in range(count):
            index += 1
            cards[f"C{index}"] = adapter.CardFace(tier, "white", i % 6, (i, 0, 0, 0, 0))
    nobles = {f"N{i}": adapter.NobleFace(3, (i, 1, 0, 0, 0)) for i in range(10)}
    return adapter.Catalog(cards, nobles)


def _source_pair() -> tuple[bytes, bytes]:
    catalog = _catalog()
    color_map = dict(enumerate(adapter.COLORS))
    local_rows = [(c.tier - 1, c.points, 4, *c.cost) for c in catalog.cards.values()]
    local_nobles = [n.requirements for n in catalog.nobles.values()]
    local = (f"_LEGACY_GEM_INDEX_TO_COLOR={color_map!r}\n"
             f"_RAW_CARD_ROWS={local_rows!r}\n_RAW_NOBLE_ROWS={local_nobles!r}\n").encode()
    upstream_cards = {code: (c.bonus, dict(zip(adapter.COLORS, c.cost)), c.tier, c.points) for code, c in catalog.cards.items()}
    upstream_nobles = [(code, dict(zip(adapter.COLORS, n.requirements))) for code, n in catalog.nobles.items()]
    upstream = f"CARDS={upstream_cards!r}\nNOBLES={upstream_nobles!r}\n".encode()
    return local, upstream


def test_full_semantic_match_is_not_arena_qualification():
    local, upstream = _source_pair()
    result = adapter.compare_catalogs(adapter.parse_local_catalog(local), adapter.parse_upstream_catalog(upstream))
    assert result["catalog_match"] is True
    assert result["matched_card_count"] == 90
    assert result["matched_noble_count"] == 10
    assert result["arena_qualified"] is False
    assert "one_second_runtime_qualification" in result["remaining_gates"]


@pytest.mark.parametrize("field,value", [("cost", (99, 0, 0, 0, 0)), ("points", 8), ("bonus", "red")])
def test_cost_points_and_bonus_differences_block(field, value):
    local = _catalog()
    upstream = _catalog()
    upstream.cards["C1"] = replace(upstream.cards["C1"], **{field: value})
    result = adapter.compare_catalogs(local, upstream)
    assert result["status"] == "blocked_catalog_mismatch"
    assert result["matched_card_count"] == 89
    assert result["unmatched_local_cards"]["C1"] == local.cards["C1"].json()
    assert result["unmatched_upstream_cards"]["C1"] == upstream.cards["C1"].json()


def test_noble_difference_blocks():
    local = _catalog()
    upstream = _catalog()
    upstream.nobles["N0"] = adapter.NobleFace(4, upstream.nobles["N0"].requirements)
    result = adapter.compare_catalogs(local, upstream)
    assert result["matched_noble_count"] == 9
    assert result["catalog_match"] is False


def test_component_identifiers_do_not_substitute_for_semantics():
    local = _catalog()
    upstream = adapter.Catalog({"external_" + key: value for key, value in local.cards.items()}, local.nobles)
    result = adapter.compare_catalogs(local, upstream)
    assert result["card_join"]["C1"] == "external_C1"


@pytest.mark.parametrize("bad", [True, -1, 1.0])
def test_bad_component_values_are_rejected(bad):
    with pytest.raises(adapter.QualificationError):
        adapter._card(1, "white", 0, {"black": bad})


def test_duplicate_faces_and_wrong_counts_fail():
    catalog = _catalog()
    catalog.cards["C2"] = catalog.cards["C1"]
    with pytest.raises(adapter.QualificationError, match="Duplicate"):
        adapter.compare_catalogs(catalog, _catalog())
    catalog.cards.pop("C2")
    with pytest.raises(adapter.QualificationError, match="90 cards"):
        adapter.compare_catalogs(catalog, _catalog())


def test_parser_does_not_execute_source(tmp_path):
    _, upstream = _source_pair()
    marker = tmp_path / "executed"
    malicious_tail = f"\n__import__('pathlib').Path({str(marker)!r}).write_text('bad')\n".encode()
    assert len(adapter.parse_upstream_catalog(upstream + malicious_tail).cards) == 90
    assert not marker.exists()
    with pytest.raises(adapter.QualificationError, match="Nonliteral"):
        adapter.parse_upstream_catalog(b"CARDS=__import__('os').environ\nNOBLES=[]")


def test_duplicate_assignment_is_not_accepted():
    _, upstream = _source_pair()
    with pytest.raises(adapter.QualificationError, match="Duplicate assignment"):
        adapter.parse_upstream_catalog(upstream + b"\nCARDS={}\n")


def _write_sources(tmp_path, monkeypatch):
    local, upstream = _source_pair()
    for name, data in ((adapter.LOCAL_CATALOG_PATH, local), (adapter.UPSTREAM_CATALOG_PATH, upstream)):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    monkeypatch.setattr(adapter, "UPSTREAM_CATALOG_SHA256", hashlib.sha256(upstream).hexdigest())
    return tmp_path


def test_authentication_and_model_execution_flags(tmp_path, monkeypatch):
    root = _write_sources(tmp_path, monkeypatch)
    receipt = adapter.qualify_sources(root)
    assert receipt["model_deserialized"] is False
    assert receipt["external_source_executed"] is False
    assert receipt["model_inference_calls"] == receipt["arena_games"] == 0
    (root / adapter.UPSTREAM_CATALOG_PATH).write_bytes(b"changed")
    with pytest.raises(adapter.QualificationError, match="hash mismatch"):
        adapter.qualify_sources(root)


def test_adapter_remains_blocked_even_with_matching_synthetic_catalog(tmp_path, monkeypatch):
    root = _write_sources(tmp_path, monkeypatch)
    with pytest.raises(adapter.QualificationBlocked, match="alone") as caught:
        adapter.create_adapter(root)
    assert caught.value.receipt["arena_qualified"] is False


def test_catalog_mismatch_prevents_adapter_creation(tmp_path, monkeypatch):
    root = _write_sources(tmp_path, monkeypatch)
    source = (root / adapter.LOCAL_CATALOG_PATH).read_text()
    source = source.replace("(0, 0, 4, 0, 0, 0, 0, 0)", "(0, 0, 4, 99, 0, 0, 0, 0)")
    (root / adapter.LOCAL_CATALOG_PATH).write_text(source)
    with pytest.raises(adapter.QualificationBlocked, match="do not match") as caught:
        adapter.create_adapter(root)
    assert caught.value.receipt["matched_card_count"] == 89

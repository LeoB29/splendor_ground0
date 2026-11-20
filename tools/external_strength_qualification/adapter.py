"""Fail-closed source qualification for the pinned external PPO.

This is deliberately not an executable opponent: catalog qualification failed.
Source modules are parsed as data, never imported; no model is deserialized and
no inference or game is performed here. Matching catalogs alone would not waive
the outstanding observation, action, checkpoint and timing qualification gates.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


UPSTREAM_COMMIT = "95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b"
UPSTREAM_CATALOG_SHA256 = "f4318e25da33ff14acab12cdad3e04377f1b2f5f89ac674dbebe07d118631444"
UPSTREAM_CATALOG_PATH = (
    "outputs/external_strength_preparation_20261018_01/qualification_sources/"
    "roeey777/src/splendor/splendor/splendor_utils.py"
)
LOCAL_CATALOG_PATH = "src/splendor_ai/engine/data.py"
COLORS = ("black", "blue", "green", "red", "white")


class QualificationError(ValueError):
    """Source data are malformed or fail authentication."""


class QualificationBlocked(RuntimeError):
    """A prerequisite prevents faithful external-opponent qualification."""

    def __init__(self, reason: str, receipt: dict[str, Any]):
        super().__init__(reason)
        self.receipt = receipt


@dataclass(frozen=True)
class CardFace:
    tier: int
    bonus: str
    points: int
    cost: tuple[int, ...]

    def json(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "bonus": self.bonus,
            "points": self.points,
            "cost": {color: count for color, count in zip(COLORS, self.cost) if count},
        }


@dataclass(frozen=True)
class NobleFace:
    points: int
    requirements: tuple[int, ...]

    def json(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "requirements": {
                color: count for color, count in zip(COLORS, self.requirements) if count
            },
        }


@dataclass(frozen=True)
class Catalog:
    cards: dict[str, CardFace]
    nobles: dict[str, NobleFace]


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QualificationError(f"{name} must be an integer >= {minimum}")
    return value


def _cost(value: Any) -> tuple[int, ...]:
    if not isinstance(value, dict) or set(value) - set(COLORS):
        raise QualificationError("Component cost has unknown colors or is not a dictionary")
    return tuple(_integer(value.get(color, 0), f"cost {color}") for color in COLORS)


def _card(tier: Any, bonus: Any, points: Any, cost: Any) -> CardFace:
    if _integer(tier, "tier", 1) not in (1, 2, 3) or bonus not in COLORS:
        raise QualificationError("Invalid tier or card bonus")
    return CardFace(tier, bonus, _integer(points, "points"), _cost(cost))


def _assignments(source: bytes, names: set[str]) -> dict[str, Any]:
    """Extract only requested top-level literal assignments, without execution."""
    values: dict[str, Any] = {}
    for node in ast.parse(source.decode("utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                if target.id in values:
                    raise QualificationError(f"Duplicate assignment: {target.id}")
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise QualificationError(f"Nonliteral component data: {target.id}") from exc
    if set(values) != names:
        raise QualificationError(f"Missing component data: {sorted(names - set(values))}")
    return values


def _validate_catalog(catalog: Catalog) -> Catalog:
    if len(catalog.cards) != 90 or len(catalog.nobles) != 10:
        raise QualificationError("Expected exactly 90 cards and 10 nobles")
    if Counter(card.tier for card in catalog.cards.values()) != {1: 40, 2: 30, 3: 20}:
        raise QualificationError("Expected tier counts 40/30/20")
    if len(set(catalog.cards.values())) != 90 or len(set(catalog.nobles.values())) != 10:
        raise QualificationError("Duplicate component faces prevent a bijective join")
    return catalog


def parse_local_catalog(source: bytes) -> Catalog:
    values = _assignments(source, {"_RAW_CARD_ROWS", "_RAW_NOBLE_ROWS", "_LEGACY_GEM_INDEX_TO_COLOR"})
    color_map = values["_LEGACY_GEM_INDEX_TO_COLOR"]
    if set(color_map) != set(range(5)) or set(color_map.values()) != set(COLORS):
        raise QualificationError("Unexpected local component color map")
    cards: dict[str, CardFace] = {}
    for index, row in enumerate(values["_RAW_CARD_ROWS"], 1):
        if len(row) != 8:
            raise QualificationError("Malformed local card row")
        zero_tier, points, bonus_index, *cost = row
        _integer(zero_tier, "zero-based tier")
        _integer(bonus_index, "bonus index")
        if bonus_index not in color_map:
            raise QualificationError("Unknown local bonus index")
        tier = zero_tier + 1
        cards[f"T{tier}_{index:02d}"] = _card(
            tier, color_map[bonus_index], points,
            {color_map[i]: value for i, value in enumerate(cost)},
        )
    nobles: dict[str, NobleFace] = {}
    for index, row in enumerate(values["_RAW_NOBLE_ROWS"], 1):
        if len(row) != 5:
            raise QualificationError("Malformed local noble row")
        nobles[f"N{index:02d}"] = NobleFace(3, _cost({color_map[i]: value for i, value in enumerate(row)}))
    return _validate_catalog(Catalog(cards, nobles))


def parse_upstream_catalog(source: bytes) -> Catalog:
    values = _assignments(source, {"CARDS", "NOBLES"})
    cards = {
        code: _card(tier, bonus, points, cost)
        for code, (bonus, cost, tier, points) in values["CARDS"].items()
    }
    nobles = {code: NobleFace(3, _cost(cost)) for code, cost in values["NOBLES"]}
    return _validate_catalog(Catalog(cards, nobles))


def compare_catalogs(local: Catalog, upstream: Catalog) -> dict[str, Any]:
    """Compare full semantic faces; do not guess an ID substitution for a mismatch."""
    _validate_catalog(local)
    _validate_catalog(upstream)
    card_inverse = {face: code for code, face in upstream.cards.items()}
    noble_inverse = {face: code for code, face in upstream.nobles.items()}
    card_join = {code: card_inverse[face] for code, face in local.cards.items() if face in card_inverse}
    noble_join = {code: noble_inverse[face] for code, face in local.nobles.items() if face in noble_inverse}
    local_card_faces = set(local.cards.values())
    local_noble_faces = set(local.nobles.values())
    matched = len(card_join) == 90 and len(noble_join) == 10
    return {
        "catalog_match": matched,
        "status": "catalog_only_passed" if matched else "blocked_catalog_mismatch",
        "arena_qualified": False,
        "local_card_count": len(local.cards),
        "upstream_card_count": len(upstream.cards),
        "matched_card_count": len(card_join),
        "matched_noble_count": len(noble_join),
        "card_join": card_join,
        "noble_join": noble_join,
        "unmatched_local_cards": {code: face.json() for code, face in local.cards.items() if code not in card_join},
        "unmatched_upstream_cards": {code: face.json() for code, face in upstream.cards.items() if face not in local_card_faces},
        "unmatched_local_nobles": {code: face.json() for code, face in local.nobles.items() if code not in noble_join},
        "unmatched_upstream_nobles": {code: face.json() for code, face in upstream.nobles.items() if face not in local_noble_faces},
        "interpretation": "Catalog disagreement; this comparison does not adjudicate official component correctness.",
        "remaining_gates": ["catalog_adjudication" ] if not matched else [
            "observer_and_feature_equivalence", "legal_action_and_empty_mask_qualification",
            "safe_checkpoint_load_and_output_equivalence", "one_second_runtime_qualification",
        ],
    }


def qualify_sources(repo_root: Path) -> dict[str, Any]:
    """Read and authenticate local data plus the pinned source, without imports."""
    local = (repo_root / LOCAL_CATALOG_PATH).read_bytes()
    upstream = (repo_root / UPSTREAM_CATALOG_PATH).read_bytes()
    upstream_hash = hashlib.sha256(upstream).hexdigest()
    if upstream_hash != UPSTREAM_CATALOG_SHA256:
        raise QualificationError("Pinned upstream component source hash mismatch")
    receipt = compare_catalogs(parse_local_catalog(local), parse_upstream_catalog(upstream))
    receipt.update({
        "upstream_repository": "roeey777/Splendor-AI",
        "upstream_commit": UPSTREAM_COMMIT,
        "sources": [
            {"path": LOCAL_CATALOG_PATH, "sha256": hashlib.sha256(local).hexdigest(), "bytes": len(local)},
            {"path": UPSTREAM_CATALOG_PATH, "sha256": upstream_hash, "bytes": len(upstream),
             "source_url": f"https://raw.githubusercontent.com/roeey777/Splendor-AI/{UPSTREAM_COMMIT}/src/splendor/splendor/splendor_utils.py"},
        ],
        "external_source_executed": False,
        "model_deserialized": False,
        "model_inference_calls": 0,
        "arena_games": 0,
    })
    return receipt


def create_adapter(repo_root: Path) -> None:
    """Never construct an opponent while source qualification is incomplete."""
    receipt = qualify_sources(repo_root)
    if not receipt["catalog_match"]:
        raise QualificationBlocked("External PPO blocked: component catalogs do not match", receipt)
    raise QualificationBlocked("Catalog match alone does not qualify an external opponent", receipt)

from __future__ import annotations

from dataclasses import replace

import pytest

from splendor_ai.eval.run_action_family_root_filter import (
    ActionFamilyRootFilterConfig,
    BUY,
    FILTERS,
    PUBLIC_OPTIONAL_FAMILIES,
    _family_integrity,
    _parse_action_name,
    _provenance_matches,
    _replay_filter,
    _validate_config,
)


def _key(action: str, target: str | None = None) -> str:
    return repr((action, target, None, (("white", 0),), None))


def test_089_configuration_freezes_interpretable_filter_frontier() -> None:
    cfg = ActionFamilyRootFilterConfig()
    _validate_config(cfg)
    assert _provenance_matches(cfg)
    assert [name for name, _ in FILTERS] == [
        "buy_only",
        "reserve_visible_only",
        "take_tokens_only",
        "card_actions",
        "buy_or_take",
        "reserve_or_take",
        "all_public_non_deck",
    ]
    assert BUY == {"BUY_VISIBLE", "BUY_RESERVED"}
    assert cfg.maximum_root_ratio == 0.70


def test_089_configuration_refuses_post_registration_changes() -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ActionFamilyRootFilterConfig(), maximum_root_ratio=0.71))


def test_089_action_family_parser_is_strict() -> None:
    assert _parse_action_name(_key("BUY_VISIBLE", "card")) == "BUY_VISIBLE"
    with pytest.raises(ValueError, match="unknown action family"):
        _parse_action_name(_key("UNKNOWN"))
    with pytest.raises(ValueError, match="unexpected schema"):
        _parse_action_name(repr(("BUY_VISIBLE",)))


def test_089_filter_preserves_top_four_and_deep_matching_actions() -> None:
    row = {
        "sample_index": 0,
        "phase": "early",
        "champion_discovery_best_index": 1,
        "expanded_root_keys": [
            _key("TAKE_TOKENS"),
            _key("BUY_VISIBLE", "a"),
            _key("RESERVE_VISIBLE", "b"),
            _key("TAKE_TOKENS"),
            _key("TAKE_TOKENS"),
            _key("BUY_RESERVED", "c"),
            _key("RESERVE_VISIBLE", "d"),
        ],
        "expanded_discovery_values": [0.0, 0.2, 0.1, 0.0, 0.5, 0.4, 0.6],
        "expanded_evaluation_values": [0.0, 0.1, 0.1, 0.0, 0.5, 0.3, 0.7],
    }
    buy = _replay_filter([row], "buy_only", BUY, "085")[0]
    assert buy["retained_roots"] == 5
    assert buy["selected_index"] == 5
    assert buy["selected_action_family"] == "BUY_RESERVED"
    assert buy["evaluation_gain"] == pytest.approx(0.2)

    full = _replay_filter(
        [row],
        "all_public_non_deck",
        PUBLIC_OPTIONAL_FAMILIES,
        "085",
    )[0]
    assert full["retained_roots"] == 7
    assert full["selected_index"] == 6
    assert full["evaluation_gain"] == pytest.approx(0.6)


def test_089_family_integrity_rejects_optional_deck_reserve() -> None:
    valid = {
        "expanded_root_count": 5,
        "expanded_root_keys": [
            _key("RESERVE_DECK"),
            _key("BUY_VISIBLE"),
            _key("TAKE_TOKENS"),
            _key("RESERVE_VISIBLE"),
            _key("BUY_RESERVED"),
        ],
    }
    assert all(_family_integrity([valid]).values())
    invalid = {
        **valid,
        "expanded_root_keys": [*valid["expanded_root_keys"][:4], _key("RESERVE_DECK")],
    }
    assert not _family_integrity([invalid])[
        "every_optional_root_is_supported_public_non_deck_family"
    ]

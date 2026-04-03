from splendor_ai.engine.constants import (
    BANK_GEMS_PER_COLOR_2P,
    MAX_TOKENS_PER_PLAYER,
    NOBLES_IN_PLAY_2P,
    VISIBLE_CARDS_PER_TIER,
)
from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles
from splendor_ai.engine.env import SplendorEnv


def test_initial_state_bank_matches_two_player_setup() -> None:
    env = SplendorEnv(seed=123)
    state = env.initial_state()

    assert state.bank_tokens["gold"] == 5
    for color in ("white", "blue", "green", "red", "black"):
        assert state.bank_tokens[color] == BANK_GEMS_PER_COLOR_2P


def test_initial_state_has_two_players() -> None:
    env = SplendorEnv()
    state = env.initial_state()

    assert len(state.players) == 2
    assert state.current_player == 0
    assert state.turn_index == 0
    assert len(state.nobles) == NOBLES_IN_PLAY_2P


def test_constants_expose_expected_limits() -> None:
    assert MAX_TOKENS_PER_PLAYER == 10
    assert NOBLES_IN_PLAY_2P == 3


def test_base_component_counts_match_rulebook() -> None:
    deck_by_tier = build_base_deck_by_tier()
    nobles = build_base_nobles()

    assert len(deck_by_tier[1]) == 40
    assert len(deck_by_tier[2]) == 30
    assert len(deck_by_tier[3]) == 20
    assert len(nobles) == 10


def test_initial_state_reveals_four_cards_per_tier_and_tracks_hidden_decks() -> None:
    env = SplendorEnv(seed=7)
    state = env.initial_state()

    expected_remaining = {1: 36, 2: 26, 3: 16}
    for tier in (1, 2, 3):
        assert len(state.visible_tier_cards[tier]) == VISIBLE_CARDS_PER_TIER
        assert len(state.hidden_tier_decks[tier]) == expected_remaining[tier]
        assert state.deck_counts[tier] == expected_remaining[tier]


def test_initial_setup_partitions_each_tier_without_duplication() -> None:
    env = SplendorEnv(seed=11)
    state = env.initial_state()
    full_deck = build_base_deck_by_tier()

    for tier in (1, 2, 3):
        full_ids = {card.card_id for card in full_deck[tier]}
        setup_ids = {
            card.card_id
            for card in state.visible_tier_cards[tier] + state.hidden_tier_decks[tier]
        }
        assert setup_ids == full_ids
        assert len(setup_ids) == len(full_ids)


def test_initial_state_is_deterministic_for_fixed_seed() -> None:
    env_a = SplendorEnv(seed=99)
    env_b = SplendorEnv(seed=99)

    state_a = env_a.initial_state()
    state_b = env_b.initial_state()

    assert [card.card_id for card in state_a.visible_tier_cards[1]] == [
        card.card_id for card in state_b.visible_tier_cards[1]
    ]
    assert [noble.noble_id for noble in state_a.nobles] == [
        noble.noble_id for noble in state_b.nobles
    ]


def test_non_terminal_state_now_exposes_legal_actions() -> None:
    env = SplendorEnv()
    state = env.initial_state()

    assert env.legal_actions(state)

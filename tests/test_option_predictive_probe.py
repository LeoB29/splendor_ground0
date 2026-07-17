import torch

from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.run_option_predictive_probe import (
    OptionPredictiveProbeConfig,
    OptionProbeHead,
    _action_from_payload,
    build_parser,
    candidate_cards,
    option_probe_gates,
    split_game_ids,
)


def test_probe_head_zero_option_is_deterministic() -> None:
    torch.manual_seed(3)
    head = OptionProbeHead(8, 6, 12)
    state = torch.randn(2, 8)
    candidates = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    zeros = torch.zeros(2, 6)

    first = head(state, candidates, zeros, mask)
    second = head(state, candidates, zeros, mask)

    assert torch.equal(first, second)
    assert torch.isneginf(first[0, 3])
    assert torch.isneginf(first[1, 2:]).all()


def test_candidate_cards_match_canonical_board_then_self_reserve_order() -> None:
    state = SplendorEnv(seed=5).initial_state()
    reserved = state.visible_tier_cards[1].pop()
    state.players[0].reserve_card(reserved, "visible")

    cards = candidate_cards(state)

    assert [card.card_id for card in cards[:3]] == sorted(
        card.card_id for card in state.visible_tier_cards[1]
    )
    assert cards[-1].card_id == reserved.card_id


def test_action_payload_round_trip_for_take() -> None:
    action = _action_from_payload(
        {
            "action_type": "TAKE_TOKENS",
            "take_tokens": ["white", "blue", "green"],
            "return_tokens": ["red"],
            "spend_tokens": [],
            "metadata": [],
        }
    )

    assert action.action_type is ActionType.TAKE_TOKENS
    assert action.take_tokens == ("white", "blue", "green")
    assert action.return_tokens == ("red",)


def test_split_ids_leave_confirmation_disjoint(tmp_path) -> None:
    config = OptionPredictiveProbeConfig(
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        representation_checkpoint=tmp_path / "023.pt",
        policy_checkpoint=tmp_path / "035.pt",
        checkpoint_016=tmp_path / "016.pt",
        checkpoint_023=tmp_path / "023b.pt",
    )

    train, validation, confirmation = split_game_ids(config)

    assert (train.start, train.stop) == (0, 192)
    assert (validation.start, validation.stop) == (192, 256)
    assert (confirmation.start, confirmation.stop) == (256, 512)
    assert set(train).isdisjoint(validation)
    assert set(validation).isdisjoint(confirmation)


def test_gate_requires_incremental_multi_turn_signal(tmp_path) -> None:
    config = OptionPredictiveProbeConfig(
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        representation_checkpoint=tmp_path / "023.pt",
        policy_checkpoint=tmp_path / "035.pt",
        checkpoint_016=tmp_path / "016.pt",
        checkpoint_023=tmp_path / "023b.pt",
        minimum_confirmation_samples=2,
        minimum_multi_turn_samples=1,
        minimum_stratum_samples=1,
    )
    summary = {
        "samples": 2,
        "baseline": {"nll": 1.0, "top1": 0.4, "mrr": 0.5},
        "option": {"nll": 0.8, "top1": 0.5, "mrr": 0.6},
        "relative_nll_improvement": 0.2,
        "nll_delta_ci": {"ci95_high": -0.1},
        "top1_delta_ci": {"ci95_low": 0.01},
        "mrr_delta_ci": {"ci95_low": 0.01},
    }
    multi = {
        "samples": 1,
        "baseline": {"nll": 1.0, "top1": 0.4, "mrr": 0.5},
        "option": {"nll": 0.8, "top1": 0.5, "mrr": 0.6},
        "relative_nll_improvement": 0.2,
    }
    stratum = {
        "samples": 2,
        "baseline": {"nll": 1.0},
        "option": {"nll": 0.9},
    }
    confirmation = {
        **summary,
        "multi_turn": multi,
        "breakdowns": {
            "opponent_seat": {
                f"{opponent}:seat{seat}": dict(stratum)
                for opponent in ("035", "016", "shallow", "023")
                for seat in (0, 1)
            },
            "phase": {
                "0-14": dict(stratum),
                "15-29": dict(stratum),
                "30-44": dict(stratum),
            },
        },
    }

    assert all(option_probe_gates(confirmation, config).values())
    confirmation["multi_turn"]["option"]["mrr"] = 0.5
    assert not option_probe_gates(confirmation, config)[
        "multi_turn_mrr_improves_at_least_2pt"
    ]


def test_parser_freezes_cpu_probe_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.games == 512
    assert args.workers == 4
    assert args.epochs == 30
    assert args.device == "cpu"

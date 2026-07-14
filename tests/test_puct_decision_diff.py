from splendor_ai.bots.puct_bot import PuctSearchStats
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.run_puct_decision_diff import (
    _DecisionDiffBot,
    _action_card_payload,
    _changed_root_signal,
    summarize_audit,
)


class _FakePuct:
    def __init__(self, action_offset: int, n_step_evaluations: int) -> None:
        self.action_offset = action_offset
        self.calls = 0
        self.last_search_stats = PuctSearchStats(
            simulations=8,
            n_step_leaf_evaluations=n_step_evaluations,
        )

    def choose_action(self, env, state, legal_actions=None):
        self.calls += 1
        return legal_actions[self.action_offset]


def test_diff_bot_skips_plain_search_when_candidate_did_not_use_n_step() -> None:
    env = SplendorEnv(seed=1)
    state = env.initial_state()
    actions = env.legal_actions(state)
    plain = _FakePuct(action_offset=1, n_step_evaluations=0)
    candidate = _FakePuct(action_offset=0, n_step_evaluations=0)
    bot = _DecisionDiffBot(
        plain=plain,
        candidate=candidate,
        trajectory_policy="plain",
        root_top_k=4,
    )

    selected = bot.choose_action(env, state, actions)

    assert selected == actions[0]
    assert plain.calls == 0
    assert candidate.calls == 1
    assert bot.rows[0]["plain_search_inferred_identical"] is True


def test_action_card_payload_identifies_visible_card() -> None:
    env = SplendorEnv(seed=2)
    state = env.initial_state()
    action = next(
        action
        for action in env.legal_actions(state)
        if action.action_type.name == "RESERVE_VISIBLE"
    )

    payload = _action_card_payload(state, action)

    assert payload is not None
    assert payload["card_id"] == state.visible_tier_cards[action.tier][
        action.market_index
    ].card_id


def test_changed_root_signal_measures_candidate_q_uplift() -> None:
    rows = [
        {
            "outcome_flip": "regressed",
            "plain_action_index": 1,
            "candidate_action_index": 2,
            "plain_root": [
                {"action_index": 1, "prior": 0.7, "rank": 1, "q": -0.4, "visit_share": 0.7},
                {"action_index": 2, "prior": 0.1, "rank": 3, "q": -0.5, "visit_share": 0.1},
            ],
            "candidate_root": [
                {"action_index": 1, "prior": 0.7, "rank": 2, "q": -0.45, "visit_share": 0.2},
                {"action_index": 2, "prior": 0.1, "rank": 1, "q": 0.1, "visit_share": 0.7},
            ],
        }
    ]

    summary = _changed_root_signal(rows)

    assert summary["candidate_lower_prior_count"] == 1
    assert summary["mean_candidate_plain_rank"] == 3
    assert summary["mean_candidate_q_uplift"] == 0.6
    assert summary["by_outcome_flip"]["regressed"]["roots"] == 1


def test_audit_summary_groups_changed_transitions_and_outcome_flips() -> None:
    conditions = [
        {
            "outcome_flip": "regressed",
            "score_margin_delta": -4,
        },
        {
            "outcome_flip": "improved",
            "score_margin_delta": 2,
        },
    ]
    rows = [
        {
            "seed": 10,
            "puct_seat": 0,
            "trajectory_policy": "plain",
            "phase": "late",
            "changed": True,
            "transition": "TAKE_TOKENS -> BUY_VISIBLE",
            "outcome_flip": "regressed",
            "score_margin_delta": -4,
        },
        {
            "seed": 11,
            "puct_seat": 1,
            "trajectory_policy": "candidate",
            "phase": "late",
            "changed": True,
            "transition": "TAKE_TOKENS -> BUY_VISIBLE",
            "outcome_flip": "improved",
            "score_margin_delta": 2,
        },
        {
            "seed": 11,
            "puct_seat": 1,
            "trajectory_policy": "candidate",
            "phase": "middle",
            "changed": False,
            "transition": "BUY_VISIBLE -> BUY_VISIBLE",
            "outcome_flip": "improved",
            "score_margin_delta": 2,
        },
    ]

    summary = summarize_audit(conditions, rows)

    assert summary["candidate_improved_conditions"] == 1
    assert summary["candidate_regressed_conditions"] == 1
    assert summary["mean_candidate_score_margin_delta"] == -1
    assert summary["changed_roots"] == 2
    assert summary["changed_root_rate"] == 2 / 3
    transition = summary["changed_action_transitions"][
        "TAKE_TOKENS -> BUY_VISIBLE"
    ]
    assert transition["count"] == 2
    assert transition["conditions"] == 2
    assert transition["improved_count"] == 1
    assert transition["regressed_count"] == 1

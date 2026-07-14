from pathlib import Path

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.run_human_game_forensic_audit import (
    HumanGameForensicAuditConfig,
    _candidate_indices,
    _diagnosis,
    _guarantees_root_win,
    _standard_error,
    build_parser,
)


class _Scored:
    def __init__(self, action_index, action):
        self.action_index = action_index
        self.action = action


def test_forensic_parser_accepts_required_paths() -> None:
    args = build_parser().parse_args(
        [
            "--replay-path", "human.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--output-path", "audit.json",
            "--report-path", "audit.md",
        ]
    )
    assert args.max_rollout_states == 40


def test_guaranteed_win_detects_second_player_terminal_purchase() -> None:
    env = SplendorEnv(seed=7)
    state = env.initial_state()
    state.current_player = 1
    state.start_player = 0
    state.players[1].score = 14
    state.players[1].bonuses = {color: 10 for color in state.players[1].bonuses}
    action = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.BUY_VISIBLE
        and state.visible_tier_cards[action.tier][action.market_index].points > 0
    )
    assert _guarantees_root_win(env, state, action, root_player=1)
    assert ActionCodec().encode(state, action) >= 0


def test_diagnosis_prefers_tactical_only_with_repeated_exact_errors() -> None:
    tactical = [{}, {}, {}]
    result = _diagnosis(tactical, [], __import__("collections").Counter())
    assert result["primary"] == "tactical"
    assert _standard_error([1.0, 1.0]) == 0.0


def test_rollout_candidates_cover_each_available_action_type(tmp_path: Path) -> None:
    env = SplendorEnv(seed=11)
    state = env.initial_state()
    actions = env.legal_actions(state)
    action_map = {index: action for index, action in enumerate(actions)}
    scored = tuple(_Scored(index, action) for index, action in action_map.items())
    selected = _candidate_indices(
        {
            "chosen_action_index": 0,
            "priority_alternative_action_indices": [],
        },
        state,
        action_map,
        scored,
        HumanGameForensicAuditConfig(
            replay_path=tmp_path / "replay.jsonl",
            policy_checkpoint=tmp_path / "policy.pt",
            output_path=tmp_path / "audit.json",
            report_path=tmp_path / "audit.md",
            max_candidates_per_state=8,
        ),
    )

    available_types = {action.action_type for action in actions}
    selected_types = {action_map[index].action_type for index in selected}
    assert selected_types == available_types

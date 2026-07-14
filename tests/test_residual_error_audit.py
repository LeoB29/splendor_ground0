import json

import pytest

from splendor_ai.training.run_residual_error_audit import (
    _classify_state,
    _excluded_state_keys,
    _paired_margin,
    _root_visit_choice,
    build_parser,
)


def test_paired_margin_uses_common_rollout_seeds() -> None:
    mean, standard_error = _paired_margin([0.8, 0.6, 0.7], [0.3, 0.1, 0.2])

    assert mean == 0.5
    assert standard_error == pytest.approx(0.0)


def test_error_attribution_separates_puct_error_from_determinization_overlay() -> None:
    assert _classify_state(
        resolved=True,
        puct_choice=3,
        oracle_action=2,
        prior_action=3,
    ) == "prior_miss"
    assert _classify_state(
        resolved=True,
        puct_choice=3,
        oracle_action=2,
        prior_action=4,
    ) == "backup_search_miss"


def test_root_visit_choice_uses_visits_then_action_index() -> None:
    payload = {
        "root_visit_distribution": [
            {"action_index": 9, "visits": 4},
            {"action_index": 7, "visits": 4},
            {"action_index": 5, "visits": 5},
        ]
    }

    assert _root_visit_choice(payload, {7, 9}) == 7
    assert _root_visit_choice(payload, {5, 7, 9}) == 5


def test_parser_accepts_required_residual_audit_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--replay-path", "replay.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--output-path", "audit.json",
        ]
    )

    assert args.max_states == 90


def test_excluded_audit_state_keys_are_loaded_from_prior_artifact(tmp_path) -> None:
    path = tmp_path / "prior_audit.json"
    path.write_text(
        json.dumps(
            {
                "states": [
                    {"game_seed": 7, "turn_index": 42, "player_id": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _excluded_state_keys(path) == {(7, 42, 1)}

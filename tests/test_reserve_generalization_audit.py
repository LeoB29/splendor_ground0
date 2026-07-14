from pathlib import Path

from splendor_ai.training.run_reserve_generalization_audit import (
    ReserveGeneralizationAuditConfig,
    _is_nonhuman_policy_row,
    _passes_bonferroni_robustness,
    _reservoir_add,
    _summarize,
    _turn_band_label,
    _wilson_interval,
    build_parser,
)


def test_parser_uses_preregistered_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--replay-path", "replays.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--output-path", "audit.json",
            "--report-path", "audit.md",
        ]
    )
    assert args.samples_per_band == 15
    assert args.rollout_determinizations == 3
    assert args.rollouts_per_determinization == 8
def test_provenance_requires_nonhuman_policy035_actor() -> None:
    payload = {
        "player_id": 0,
        "game_bot_seats": ["LegalActionAttentionSamplingBot", "OtherBot"],
        "game_bot_metadata": [
            {"checkpoint_path": "outputs/legal_action_scorer_035_legal_observation/best.pt"},
            {},
        ],
    }
    assert _is_nonhuman_policy_row(payload, "legal_action_scorer_035_legal_observation")
    payload["human_player_id"] = 1
    assert not _is_nonhuman_policy_row(payload, "legal_action_scorer_035_legal_observation")


def test_turn_bands_and_wilson_interval() -> None:
    assert _turn_band_label(0) == "0-4"
    assert _turn_band_label(19) == "15-19"
    lower, upper = _wilson_interval(6, 60)
    assert 0.03 < lower < 0.10 < upper


def test_reservoir_keeps_fixed_capacity() -> None:
    import random

    pool = []
    rng = random.Random(7)
    for seen in range(1, 101):
        _reservoir_add(pool, {"value": seen}, seen=seen, limit=10, rng=rng)
    assert len(pool) == 10
    assert len({row["value"] for row in pool}) == 10


def test_summary_passes_preregistered_generalization_gate(tmp_path: Path) -> None:
    config = ReserveGeneralizationAuditConfig(
        replay_path=tmp_path / "replays.jsonl",
        policy_checkpoint=tmp_path / "policy.pt",
        output_path=tmp_path / "audit.json",
        report_path=tmp_path / "audit.md",
    )
    results = []
    for index in range(60):
        resolved = index in {0, 1, 2, 15, 16, 17}
        band = ("0-4", "5-9", "10-14", "15-19")[index // 15]
        actions = [
            {
                "action_index": 1,
                "action": {"action_type": "RESERVE_VISIBLE"},
                "prior_score": 2.0,
                "rollout_count": 24,
            },
            {
                "action_index": 2,
                "action": {"action_type": "TAKE_TOKENS"},
                "prior_score": 0.0,
                "rollout_count": 24,
            },
        ]
        results.append(
            {
                "resolved_mistake": resolved,
                "turn_band": band,
                "candidate_count": 2,
                "chosen_action_index": 1,
                "best_alternative_action_index": 2,
                "paired_return_delta": 0.3,
                "paired_return_delta_standard_error": 0.05,
                "determinization_deltas": [0.2, 0.3, 0.4],
                "actions": actions,
                "recorded_action_matches_policy": True,
            }
        )
    summary = _summarize(
        config,
        {"selected_states": 60, "unique_games": 60},
        results,
    )
    assert summary["generalizes"]
    assert summary["resolved_bands"] == 2
    assert summary["bonferroni_robust_resolved_mistakes"] == 6


def test_bonferroni_robustness_accounts_for_selected_alternative(tmp_path: Path) -> None:
    config = ReserveGeneralizationAuditConfig(
        replay_path=tmp_path / "replays.jsonl",
        policy_checkpoint=tmp_path / "policy.pt",
        output_path=tmp_path / "audit.json",
        report_path=tmp_path / "audit.md",
    )
    result = {
        "candidate_count": 5,
        "paired_return_delta": 0.24,
        "paired_return_delta_standard_error": 0.1,
        "determinization_deltas": [0.1, 0.2, 0.3],
    }
    assert not _passes_bonferroni_robustness(result, config)
    result["paired_return_delta"] = 0.35
    assert _passes_bonferroni_robustness(result, config)

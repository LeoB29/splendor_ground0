from __future__ import annotations

from splendor_ai.training.run_policy_delta_causal_audit import (
    OPPONENTS,
    PolicyDeltaCausalAuditConfig,
    _diagnose,
    _source_assignment,
    outcome_blind_replay_projection,
    select_balanced_unique_game_disagreements,
    summarize_policy_delta_states,
)


def test_source_assignment_balances_opponents_and_seats() -> None:
    cells = [_source_assignment(index) for index in range(40)]
    assert set(cells) == {(opponent, seat) for opponent in OPPONENTS for seat in (0, 1)}
    assert set(cells.count(cell) for cell in set(cells)) == {5}


def test_outcome_blind_projection_ignores_all_result_fields() -> None:
    base = {
        "game_seed": 7,
        "turn_index": 11,
        "player_id": 1,
        "state_snapshot": {"turn_index": 11},
        "game_winner": 0,
        "game_final_scores": [15, 2],
        "final_value": -1.0,
    }
    changed = {
        **base,
        "game_winner": 1,
        "game_final_scores": [0, 99],
        "final_value": 1.0,
    }
    assert outcome_blind_replay_projection(base) == outcome_blind_replay_projection(changed)


def test_balanced_selection_uses_one_state_per_game() -> None:
    rows = []
    game_index = 0
    for opponent in OPPONENTS:
        for seat in (0, 1):
            for phase in ("early", "mid", "late"):
                stratum = f"{opponent}:seat{seat}:{phase}"
                for offset in range(3):
                    rows.append(
                        {
                            "stratum": stratum,
                            "source_cell": f"{opponent}:seat{seat}",
                            "phase": phase,
                            "source_game_index": game_index,
                            "source_game_seed": 1000 + game_index,
                            "turn_index": offset,
                            "parent_action_index": 1,
                            "candidate_action_index": 2,
                        }
                    )
                    game_index += 1
    selected, _ = select_balanced_unique_game_disagreements(
        rows,
        states_per_opponent_seat=2,
        seed=9,
    )
    assert len(selected) == len(OPPONENTS) * 2 * 2
    assert len({row["source_game_index"] for row in selected}) == len(selected)
    assert set({
        cell: sum(row["source_cell"] == cell for row in selected)
        for cell in {row["source_cell"] for row in rows}
    }.values()) == {2}


def _summary_config() -> PolicyDeltaCausalAuditConfig:
    return PolicyDeltaCausalAuditConfig(
        output_dir=None,  # type: ignore[arg-type]
        data_dir=None,  # type: ignore[arg-type]
        parent_checkpoint=None,  # type: ignore[arg-type]
        candidate_checkpoint=None,  # type: ignore[arg-type]
        checkpoint_016=None,  # type: ignore[arg-type]
        checkpoint_023=None,  # type: ignore[arg-type]
        minimum_split_sign_states=2,
        bootstrap_samples=200,
    )


def test_summary_reports_candidate_minus_parent_and_resolution() -> None:
    rows = []
    for index, value in enumerate((0.08, 0.06, -0.02, 0.04)):
        rows.append(
            {
                "phase": ("early", "mid", "late")[index % 3],
                "opponent": OPPONENTS[index],
                "actor_seat": index % 2,
                "stratum": f"{OPPONENTS[index]}:seat{index % 2}:early",
                "candidate_minus_parent_first": value,
                "candidate_minus_parent_second": value * 0.9,
                "candidate_minus_parent_full": value * 0.95,
                "resolved": index < 2,
                "terminal_continuations": 48,
                "total_continuations": 48,
            }
        )
    summary = summarize_policy_delta_states(rows, _summary_config())
    assert summary["overall"]["mean_candidate_minus_parent_return"] > 0.0
    assert summary["overall"]["resolved_states"] == 2
    assert summary["overall"]["continuation_cutoff_fraction"] == 0.0
    assert summary["overall"]["split_sign_agreement"] == 1.0


def test_diagnosis_never_authorizes_training_or_promotion() -> None:
    stable = {"stability": {"stable_action_signal": True}}
    assert _diagnose({**stable, "overall": {"mean_ci95": [-0.2, -0.01]}}) == (
        "stable_local_harm_close_awr"
    )
    assert _diagnose({**stable, "overall": {"mean_ci95": [0.01, 0.2]}}) == (
        "stable_local_benefit_test_compounding"
    )
    assert _diagnose(
        {
            "stability": {"stable_action_signal": False},
            "overall": {"mean_ci95": [0.01, 0.2]},
        }
    ) == "local_effect_unresolved_close_awr"

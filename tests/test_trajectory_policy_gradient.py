from dataclasses import asdict
import json
import math

import torch

from splendor_ai.bots import (
    LegalActionAttentionSamplingBot,
    RandomLegalBot,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.dataset import LegalActionReplayDataset
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.replay import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.run_trajectory_advantage_corpus import (
    TrajectoryAdvantageCorpusConfig,
    build_trajectory_advantage_corpus,
)
from splendor_ai.training.run_trajectory_policy_gradient import (
    _exploration_probabilities,
    compute_clipped_trajectory_loss,
)
from splendor_ai.training.run_group_relative_trajectory_corpus import (
    GroupRelativeTrajectoryCorpusConfig,
    build_group_relative_trajectory_corpus,
)


def _write_attention_checkpoint(tmp_path):
    model = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=16,
            state_depth=1,
            action_depth=1,
            attention_layers=1,
            attention_heads=1,
            attention_feedforward_size=16,
            dropout=0.0,
        )
    )
    path = tmp_path / "attention.pt"
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        path,
    )
    return path


def test_sampling_bot_records_exact_behavior_probability(tmp_path) -> None:
    checkpoint = _write_attention_checkpoint(tmp_path)
    bot = LegalActionAttentionSamplingBot(
        checkpoint,
        seed=7,
        epsilon=0.2,
        temperature=0.5,
        top_k=3,
    )
    env = SplendorEnv(seed=8)
    state = env.initial_state()

    action = bot.choose_action(env, state, env.legal_actions(state))
    metadata = bot.last_action_metadata

    assert action is not None
    assert metadata is not None
    assert 0.0 < metadata["behavior_probability"] <= 1.0
    assert math.isclose(
        metadata["behavior_log_probability"],
        math.log(metadata["behavior_probability"]),
    )
    assert metadata["behavior_support_size"] == 3


def test_advantage_corpus_keeps_only_sampling_turns(tmp_path) -> None:
    checkpoint = _write_attention_checkpoint(tmp_path)
    sampling_bot = LegalActionAttentionSamplingBot(checkpoint, seed=9)
    replay = collect_game_replay(
        bot_seat_0=sampling_bot,
        bot_seat_1=RandomLegalBot(seed=10),
        seed=11,
        max_turns=4,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "replay.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["game_final_scores"] = [15, 10]
        row["final_value"] = 1.0 if row["player_id"] == 0 else -1.0
        row["game_timed_out"] = False
        row["game_stalled"] = False
    replay_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_path = tmp_path / "advantages.jsonl"
    summary = build_trajectory_advantage_corpus(
        TrajectoryAdvantageCorpusConfig(
            replay_paths=(replay_path,),
            output_path=output_path,
            summary_path=tmp_path / "summary.json",
            log_every=0,
        )
    )
    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["rows_written"] == 2
    assert summary["positive_rows"] == 2
    assert summary["skipped_missing_behavior_probability"] == 2
    assert all(row["trajectory_advantage"] > 0.0 for row in output_rows)
    dataset = LegalActionReplayDataset(output_path)
    assert dataset[0].trajectory_mask
    assert dataset[0].behavior_log_probability < 0.0


def test_clipped_trajectory_loss_matches_behavior_policy() -> None:
    action_scores = torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
        requires_grad=True,
    )
    reference_scores = action_scores.detach().clone()
    legal_mask = torch.ones((2, 3), dtype=torch.bool)
    target_rows = torch.tensor([0, 1], dtype=torch.long)
    current_probs, reference_probs = _exploration_probabilities(
        action_scores=action_scores,
        reference_scores=reference_scores,
        legal_action_mask=legal_mask,
        epsilon=0.2,
        temperature=0.5,
        top_k=3,
    )
    behavior_log_probs = reference_probs[
        torch.arange(2), target_rows
    ].log()
    losses = compute_clipped_trajectory_loss(
        action_scores=action_scores,
        reference_scores=reference_scores,
        legal_action_mask=legal_mask,
        target_action_row=target_rows,
        advantages=torch.tensor([1.0, -1.0]),
        behavior_log_probability=behavior_log_probs,
        sample_weight=torch.ones(2),
        clip_ratio=0.1,
        kl_weight=1.0,
        entropy_weight=0.001,
        sampling_epsilon=0.2,
        sampling_temperature=0.5,
        sampling_top_k=3,
    )

    assert torch.isclose(losses["approx_ratio"], torch.tensor(1.0))
    assert torch.isclose(losses["kl_loss"], torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(
        losses["behavior_log_prob_mae"],
        torch.tensor(0.0),
        atol=1e-7,
    )
    losses["total_loss"].backward()
    assert torch.isfinite(action_scores.grad).all()


def test_group_relative_corpus_centers_repeated_seed_returns(tmp_path) -> None:
    paths = []
    for index, outcome in enumerate((1.0, -1.0, 1.0)):
        path = tmp_path / f"rollout_{index}.jsonl"
        row = {
            "game_seed": 100,
            "player_id": 0,
            "game_final_scores": [15, 10] if outcome > 0 else [10, 15],
            "final_value": outcome,
            "game_timed_out": False,
            "game_stalled": False,
            "behavior_log_probability": -0.1,
            "game_bot_metadata": [
                {"bot_class": "LegalActionAttentionSamplingBot"},
                {"bot_class": "ShallowSearchBot"},
            ],
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths.append(path)

    output_path = tmp_path / "group_relative.jsonl"
    summary = build_group_relative_trajectory_corpus(
        GroupRelativeTrajectoryCorpusConfig(
            replay_paths=tuple(paths),
            output_path=output_path,
            summary_path=tmp_path / "summary.json",
            min_group_trajectories=3,
        )
    )
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    advantages = [row["trajectory_advantage"] for row in rows]

    assert summary["groups_total"] == 1
    assert summary["groups_informative"] == 1
    assert summary["trajectories_written"] == 3
    assert len(rows) == 3
    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-12)
    assert advantages[1] < 0.0 < advantages[0]
    assert all(
        row["trajectory_metadata"]["baseline_mode"] == "leave_one_out"
        for row in rows
    )

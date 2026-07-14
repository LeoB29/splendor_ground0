import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from splendor_ai.bots import (
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    LegalActionSetScorerBot,
    RandomLegalBot,
)
from splendor_ai.training import (
    LegalActionReplayDataset,
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    LegalActionScorerConfig,
    LegalActionScorerModel,
    LegalActionSetScorerConfig,
    LegalActionSetScorerModel,
    SupervisedTrainConfig,
    collate_legal_action_replay_samples,
    collect_game_replay,
    compute_legal_action_supervised_losses,
    compute_pairwise_preference_loss,
    compute_root_visit_distillation_loss,
    compute_search_score_distillation_loss,
    export_replay_games_jsonl,
    fit_legal_action_scorer,
    masked_legal_action_scores,
    split_legal_action_replay_dataset,
)


def _write_snapshot_replay(tmp_path, max_turns: int = 8):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=max_turns,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "snapshot_replay.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    return replay_path


def _write_snapshot_replays(tmp_path, max_turns: int = 8, count: int = 4):
    replays = [
        collect_game_replay(
            bot_seat_0=RandomLegalBot(seed=10 + index),
            bot_seat_1=RandomLegalBot(seed=20 + index),
            seed=30 + index,
            max_turns=max_turns,
            include_state_snapshots=True,
        )
        for index in range(count)
    ]
    replay_path = tmp_path / "snapshot_replays.jsonl"
    export_replay_games_jsonl(replay_path, replays)
    return replay_path


def test_compute_legal_action_supervised_losses_returns_finite_values() -> None:
    action_scores = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, 1.0, -5.0]],
        dtype=torch.float32,
    )
    value_pred = torch.tensor([0.25, -0.5], dtype=torch.float32)
    target_action_row = torch.tensor([2, 0], dtype=torch.long)
    legal_action_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    value_target = torch.tensor([1.0, -1.0], dtype=torch.float32)

    policy_loss, value_loss, total_loss = compute_legal_action_supervised_losses(
        action_scores=action_scores,
        value_pred=value_pred,
        target_action_row=target_action_row,
        legal_action_mask=legal_action_mask,
        value_target=value_target,
        value_loss_weight=2.0,
        sample_weight=torch.tensor([2.0, 0.5], dtype=torch.float32),
    )

    assert float(policy_loss) > 0.0
    assert float(value_loss) > 0.0
    assert float(total_loss) > float(policy_loss)
    assert masked_legal_action_scores(action_scores, legal_action_mask)[1, 2] < -1e30


def test_search_score_distillation_loss_prefers_search_ranking() -> None:
    legal_action_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
    search_value_target = torch.tensor([[100.0, 0.0, -100.0]], dtype=torch.float32)
    search_value_mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    aligned_loss = compute_search_score_distillation_loss(
        action_scores=torch.tensor([[2.0, 0.0, -2.0]], dtype=torch.float32),
        legal_action_mask=legal_action_mask,
        search_value_target=search_value_target,
        search_value_mask=search_value_mask,
        temperature=100.0,
    )
    reversed_loss = compute_search_score_distillation_loss(
        action_scores=torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float32),
        legal_action_mask=legal_action_mask,
        search_value_target=search_value_target,
        search_value_mask=search_value_mask,
        temperature=100.0,
    )

    assert torch.isfinite(aligned_loss)
    assert torch.isfinite(reversed_loss)
    assert float(aligned_loss) < float(reversed_loss)


def test_root_visit_distillation_loss_prefers_visit_distribution() -> None:
    legal_action_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
    root_visit_target = torch.tensor([[0.7, 0.25, 0.05]], dtype=torch.float32)
    root_visit_mask = torch.tensor([True], dtype=torch.bool)

    aligned_loss = compute_root_visit_distillation_loss(
        action_scores=torch.tensor([[2.0, 0.0, -2.0]], dtype=torch.float32),
        legal_action_mask=legal_action_mask,
        root_visit_target=root_visit_target,
        root_visit_mask=root_visit_mask,
    )
    reversed_loss = compute_root_visit_distillation_loss(
        action_scores=torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float32),
        legal_action_mask=legal_action_mask,
        root_visit_target=root_visit_target,
        root_visit_mask=root_visit_mask,
    )

    assert torch.isfinite(aligned_loss)
    assert float(aligned_loss) < float(reversed_loss)


def test_pairwise_preference_loss_prefers_teacher_above_competitor() -> None:
    legal_action_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
    target_action_row = torch.tensor([0], dtype=torch.long)
    competitor_row = torch.tensor([2], dtype=torch.long)
    preference_mask = torch.tensor([True], dtype=torch.bool)

    aligned_loss = compute_pairwise_preference_loss(
        action_scores=torch.tensor([[3.0, 0.0, 1.0]], dtype=torch.float32),
        legal_action_mask=legal_action_mask,
        target_action_row=target_action_row,
        preference_competitor_row=competitor_row,
        preference_mask=preference_mask,
        margin=1.0,
    )
    reversed_loss = compute_pairwise_preference_loss(
        action_scores=torch.tensor([[1.0, 0.0, 3.0]], dtype=torch.float32),
        legal_action_mask=legal_action_mask,
        target_action_row=target_action_row,
        preference_competitor_row=competitor_row,
        preference_mask=preference_mask,
        margin=1.0,
    )

    assert torch.isfinite(aligned_loss)
    assert torch.isfinite(reversed_loss)
    assert float(aligned_loss) < float(reversed_loss)


def test_legal_action_replay_dataset_reads_search_score_targets(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
    ]
    first_legal_indices = rows[0]["legal_action_indices"]
    rows[0]["search_action_scores"] = [
        {"action_index": first_legal_indices[0], "value": 100.0},
        {"action_index": first_legal_indices[1], "value": 50.0},
    ]
    rows[0]["training_weight"] = 2.5
    scored_path = tmp_path / "scored_replay.jsonl"
    scored_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = LegalActionReplayDataset(scored_path)
    sample = dataset[0]
    batch = collate_legal_action_replay_samples([sample])

    assert sample.search_value_mask.sum().item() == 2
    assert sample.search_value_target[0].item() == 100.0
    assert sample.search_value_target[1].item() == 50.0
    assert sample.sample_weight == 2.5
    assert batch["search_value_mask"].shape == batch["legal_action_mask"].shape
    assert batch["search_value_target"][0, 0].item() == 100.0
    assert batch["sample_weight"][0].item() == 2.5


def test_legal_action_replay_dataset_reads_root_visit_targets(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
    ]
    legal_indices = rows[0]["legal_action_indices"]
    rows[0]["root_visit_distribution"] = [
        {"action_index": legal_indices[0], "visits": 7, "probability": 0.7},
        {"action_index": legal_indices[1], "visits": 3, "probability": 0.3},
    ]
    visit_path = tmp_path / "visit_replay.jsonl"
    visit_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    sample = LegalActionReplayDataset(visit_path)[0]
    batch = collate_legal_action_replay_samples([sample])

    assert sample.root_visit_mask
    assert sample.root_visit_target[0].item() == pytest.approx(0.7)
    assert sample.root_visit_target[1].item() == pytest.approx(0.3)
    assert batch["root_visit_mask"][0].item()
    assert batch["root_visit_target"][0, :2].sum().item() == pytest.approx(1.0)


def test_legal_action_replay_dataset_reads_preference_pairs(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
    ]
    first_legal_indices = rows[0]["legal_action_indices"]
    competitor_index = next(
        action_index
        for action_index in first_legal_indices
        if action_index != rows[0]["action_index"]
    )
    competitor_row = first_legal_indices.index(competitor_index)
    rows[0]["model_top_action_index"] = competitor_index
    rows[0]["training_weight"] = 2.0
    preference_path = tmp_path / "preference_replay.jsonl"
    preference_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = LegalActionReplayDataset(
        preference_path,
        preference_rows_policy_weight=0.0,
        preference_rows_distill_weight=0.25,
        preference_rows_preference_weight=3.0,
    )
    sample = dataset[0]
    batch = collate_legal_action_replay_samples([sample])

    assert sample.preference_mask
    assert sample.preference_competitor_row == competitor_row
    assert sample.policy_sample_weight == 0.0
    assert sample.distill_sample_weight == 0.5
    assert sample.preference_sample_weight == 6.0
    assert batch["preference_mask"][0].item()
    assert batch["preference_competitor_row"][0].item() == competitor_row
    assert batch["policy_sample_weight"][0].item() == 0.0
    assert batch["distill_sample_weight"][0].item() == 0.5
    assert batch["preference_sample_weight"][0].item() == 6.0


def test_legal_action_replay_dataset_can_filter_winner_actions(tmp_path) -> None:
    replay_path = _write_snapshot_replays(tmp_path)
    rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
    ]
    winner_rows = sum(1 for row in rows if float(row["final_value"]) > 0.0)

    dataset = LegalActionReplayDataset(
        replay_path,
        policy_target_filter="winner",
    )

    assert len(dataset) == winner_rows
    assert len(dataset) > 0


def test_legal_action_replay_dataset_applies_outcome_weights(tmp_path) -> None:
    replay_path = _write_snapshot_replays(tmp_path)
    rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["training_weight"] = 2.0
    weighted_path = tmp_path / "weighted_snapshot_replays.jsonl"
    weighted_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    dataset = LegalActionReplayDataset(
        weighted_path,
        winner_action_weight=3.0,
        loser_action_weight=0.5,
        draw_action_weight=1.0,
    )

    for row_index, row in enumerate(rows):
        if float(row["final_value"]) > 0.0:
            assert dataset[row_index].sample_weight == 6.0
            break
    else:
        raise AssertionError("Expected at least one winner row.")

    for row_index, row in enumerate(rows):
        if float(row["final_value"]) < 0.0:
            assert dataset[row_index].sample_weight == 1.0
            break
    else:
        raise AssertionError("Expected at least one loser row.")


def test_split_legal_action_replay_dataset_by_game_keeps_games_disjoint(tmp_path) -> None:
    replay_path = _write_snapshot_replays(tmp_path, count=4)
    dataset = LegalActionReplayDataset(replay_path)

    train_dataset, validation_dataset = split_legal_action_replay_dataset(
        dataset,
        validation_fraction=0.5,
        seed=0,
        validation_split_unit="game",
    )

    assert validation_dataset is not None
    train_keys = {
        dataset.validation_group_key_for_index(int(index), "game")
        for index in train_dataset.indices
    }
    validation_keys = {
        dataset.validation_group_key_for_index(int(index), "game")
        for index in validation_dataset.indices
    }
    assert train_keys
    assert validation_keys
    assert train_keys.isdisjoint(validation_keys)


def test_fit_legal_action_scorer_runs_on_snapshot_replay(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=32,
            action_hidden_size=16,
            scorer_hidden_size=24,
            state_depth=2,
            action_depth=2,
        )
    )

    trained_model, metrics = fit_legal_action_scorer(
        replay_path=replay_path,
        config=SupervisedTrainConfig(
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            value_loss_weight=1.0,
            epochs=1,
            device="cpu",
            shuffle=False,
        ),
        model=model,
    )

    assert isinstance(trained_model, LegalActionScorerModel)
    assert len(metrics) == 1
    assert metrics[0].train.samples == len(LegalActionReplayDataset(replay_path))
    assert metrics[0].train.batches > 0


def test_legal_action_scorer_bot_loads_checkpoint_and_chooses_legal_action(tmp_path) -> None:
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=12,
            state_depth=1,
            action_depth=1,
        )
    )
    checkpoint_path = tmp_path / "legal_action_scorer.pt"
    torch.save(
        {
            "model_kind": "legal_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )

    from splendor_ai.engine.env import SplendorEnv

    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    bot = LegalActionScorerBot(checkpoint_path, device="cpu")

    chosen_action = bot.choose_action(env, state, legal_actions)

    assert chosen_action in legal_actions


def test_legal_action_set_scorer_bot_loads_checkpoint_and_chooses_legal_action(tmp_path) -> None:
    model = LegalActionSetScorerModel(
        LegalActionSetScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            set_context_hidden_size=6,
            scorer_hidden_size=12,
            state_depth=1,
            action_depth=1,
            set_context_depth=1,
        )
    )
    checkpoint_path = tmp_path / "legal_action_set_scorer.pt"
    torch.save(
        {
            "model_kind": "legal_action_set_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )

    from splendor_ai.engine.env import SplendorEnv

    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    bot = LegalActionSetScorerBot(checkpoint_path, device="cpu")

    chosen_action = bot.choose_action(env, state, legal_actions)

    assert chosen_action in legal_actions


def test_legal_action_attention_scorer_bot_loads_checkpoint_and_chooses_legal_action(tmp_path) -> None:
    model = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=12,
            state_depth=1,
            action_depth=1,
            attention_layers=1,
            attention_heads=2,
            attention_feedforward_size=16,
        )
    )
    checkpoint_path = tmp_path / "legal_action_attention_scorer.pt"
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )

    from splendor_ai.engine.env import SplendorEnv

    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    bot = LegalActionAttentionScorerBot(checkpoint_path, device="cpu")

    chosen_action = bot.choose_action(env, state, legal_actions)

    assert chosen_action in legal_actions


def test_run_legal_action_supervised_cli_writes_checkpoint_and_metrics(tmp_path) -> None:
    replay_path = _write_snapshot_replays(tmp_path)
    output_dir = tmp_path / "legal_output"
    result = subprocess.run(
        [
            sys.executable,
            "run_legal_action_supervised.py",
            "--replay-path",
            str(replay_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--state-hidden-size",
            "16",
            "--action-hidden-size",
            "8",
            "--scorer-hidden-size",
            "12",
            "--state-depth",
            "1",
            "--action-depth",
            "1",
            "--validation-fraction",
            "0.5",
            "--validation-split-unit",
            "game",
            "--policy-target-filter",
            "winner",
            "--winner-action-weight",
            "2.0",
            "--search-distill-loss-weight",
            "0.1",
            "--search-distill-temperature",
            "1000",
            "--preference-loss-weight",
            "0.2",
            "--preference-margin",
            "0.5",
            "--preference-rows-policy-weight",
            "0.0",
            "--preference-rows-distill-weight",
            "0.5",
            "--preference-rows-preference-weight",
            "2.0",
            "--log-every-batches",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Legal-action scorer training complete" in result.stdout
    assert (output_dir / "legal_action_scorer.pt").exists()
    metrics = json.loads((output_dir / "legal_action_scorer_metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["total_samples"] == len(
        LegalActionReplayDataset(replay_path, policy_target_filter="winner")
    )
    assert metrics["dataset"]["validation_split_unit"] == "game"
    assert metrics["dataset"]["policy_target_filter"] == "winner"
    assert metrics["dataset"]["winner_action_weight"] == 2.0
    assert metrics["dataset"]["search_distill_loss_weight"] == 0.1
    assert metrics["dataset"]["search_distill_temperature"] == 1000
    assert metrics["dataset"]["preference_loss_weight"] == 0.2
    assert metrics["dataset"]["preference_margin"] == 0.5
    assert metrics["dataset"]["preference_rows_policy_weight"] == 0.0
    assert metrics["dataset"]["preference_rows_distill_weight"] == 0.5
    assert metrics["dataset"]["preference_rows_preference_weight"] == 2.0
    assert metrics["checkpoints"]["final_checkpoint_path"] == "legal_action_scorer.pt"


def test_run_legal_action_supervised_cli_writes_set_scorer_checkpoint(tmp_path) -> None:
    replay_path = _write_snapshot_replays(tmp_path)
    output_dir = tmp_path / "legal_set_output"
    result = subprocess.run(
        [
            sys.executable,
            "run_legal_action_supervised.py",
            "--replay-path",
            str(replay_path),
            "--output-dir",
            str(output_dir),
            "--architecture",
            "set",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--state-hidden-size",
            "16",
            "--action-hidden-size",
            "8",
            "--set-context-hidden-size",
            "6",
            "--scorer-hidden-size",
            "12",
            "--state-depth",
            "1",
            "--action-depth",
            "1",
            "--set-context-depth",
            "1",
            "--validation-fraction",
            "0.5",
            "--validation-split-unit",
            "game",
            "--log-every-batches",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Legal-action scorer training complete" in result.stdout
    assert (output_dir / "legal_action_set_scorer.pt").exists()
    metrics = json.loads((output_dir / "legal_action_set_scorer_metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["architecture"] == "set"
    assert metrics["dataset"]["model_kind"] == "legal_action_set_scorer"
    assert metrics["checkpoints"]["final_checkpoint_path"] == "legal_action_set_scorer.pt"


def test_run_legal_action_supervised_cli_writes_attention_scorer_checkpoint(tmp_path) -> None:
    replay_path = _write_snapshot_replays(tmp_path)
    output_dir = tmp_path / "legal_attention_output"
    result = subprocess.run(
        [
            sys.executable,
            "run_legal_action_supervised.py",
            "--replay-path",
            str(replay_path),
            "--output-dir",
            str(output_dir),
            "--architecture",
            "attention",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--state-hidden-size",
            "16",
            "--action-hidden-size",
            "8",
            "--scorer-hidden-size",
            "12",
            "--state-depth",
            "1",
            "--action-depth",
            "1",
            "--attention-layers",
            "1",
            "--attention-heads",
            "2",
            "--attention-feedforward-size",
            "16",
            "--validation-fraction",
            "0.5",
            "--validation-split-unit",
            "game",
            "--log-every-batches",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Legal-action scorer training complete" in result.stdout
    assert (output_dir / "legal_action_attention_scorer.pt").exists()
    metrics = json.loads(
        (output_dir / "legal_action_attention_scorer_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["dataset"]["architecture"] == "attention"
    assert metrics["dataset"]["model_kind"] == "legal_action_attention_scorer"
    assert metrics["checkpoints"]["final_checkpoint_path"] == "legal_action_attention_scorer.pt"

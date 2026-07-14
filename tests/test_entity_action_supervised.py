from __future__ import annotations

from splendor_ai.bots import GreedyHeuristicBot
from splendor_ai.training.corpus import generate_replay_corpus, write_replay_corpus
from splendor_ai.training.entity_action_supervised import (
    EntityActionReplayDataset,
    create_entity_action_replay_dataloader,
    run_entity_action_epoch,
)
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)


def test_entity_replay_dataset_and_training_epoch(tmp_path) -> None:
    games, summary = generate_replay_corpus(
        games=2,
        bot_seat_0_factory=lambda: GreedyHeuristicBot(seed=1),
        bot_seat_1_factory=lambda: GreedyHeuristicBot(seed=2),
        seed_start=100,
        include_state_snapshots=True,
    )
    replay_path, _summary_path = write_replay_corpus(tmp_path, games, summary)
    dataset = EntityActionReplayDataset(replay_path)
    dataloader = create_entity_action_replay_dataloader(dataset, batch_size=16, shuffle=False)
    model = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=32,
            action_hidden_size=32,
            scorer_hidden_size=32,
            entity_layers=1,
            action_layers=1,
            attention_heads=4,
            attention_feedforward_size=64,
            id_embedding_size=8,
        )
    )

    metrics = run_entity_action_epoch(model, dataloader, device="cpu")

    assert metrics.samples == len(dataset)
    assert metrics.policy_loss > 0.0
    assert metrics.uniform_policy_loss > 0.0

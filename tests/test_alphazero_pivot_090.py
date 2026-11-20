from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from splendor_ai.alphazero.generation import SandboxConfig, run_vertical_slice
from splendor_ai.alphazero.network import (
    AlphaZeroInference,
    create_initial_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from splendor_ai.alphazero.replay import (
    read_replay_jsonl,
    replay_row_from_search,
    write_replay_jsonl,
)
from splendor_ai.alphazero.search import (
    AlphaZeroSearch,
    SearchConfig,
    _Edge,
    _Node,
    _PathStep,
)
from splendor_ai.alphazero.training import alpha_zero_loss, collate_replay_rows
from splendor_ai.encoding import (
    actions_by_canonical_successor_key,
    encode_canonical_state_entities,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "generation_000.pt"
    create_initial_checkpoint(path, seed=900090)
    return path


def test_search_covers_every_successor_and_accounts_visits(checkpoint: Path) -> None:
    env = SplendorEnv(seed=7)
    state = env.initial_state()
    legal = env.legal_actions(state)
    expected = actions_by_canonical_successor_key(state, legal)
    search = AlphaZeroSearch(
        AlphaZeroInference(checkpoint),
        SearchConfig(
            simulations=5,
            mode="debug",
            root_noise_fraction=0.0,
            seed=77,
        ),
    )
    result = search.run(env, state, legal, self_play=False)

    assert len(result.root_edges) == len(expected)
    assert {edge.action_key for edge in result.root_edges} == set(expected)
    assert sum(edge.visits for edge in result.root_edges) == 5
    assert sum(result.visit_probabilities) == pytest.approx(1.0)
    assert result.action in expected.values()


def test_real_search_is_invariant_to_true_hidden_deck_order(checkpoint: Path) -> None:
    env = SplendorEnv(seed=8)
    state_a = env.initial_state()
    state_b = state_a.copy_shallow()
    for tier in (1, 2, 3):
        state_b.hidden_tier_decks[tier] = list(reversed(state_b.hidden_tier_decks[tier]))
    config = SearchConfig(
        simulations=4,
        mode="real",
        root_noise_fraction=0.0,
        seed=88,
    )
    result_a = AlphaZeroSearch(AlphaZeroInference(checkpoint), config).run(
        env, state_a, self_play=False
    )
    result_b = AlphaZeroSearch(AlphaZeroInference(checkpoint), config).run(
        env, state_b, self_play=False
    )

    projection_a = [
        (edge.action_key, edge.visits, edge.network_prior, edge.search_prior, edge.q)
        for edge in result_a.root_edges
    ]
    projection_b = [
        (edge.action_key, edge.visits, edge.network_prior, edge.search_prior, edge.q)
        for edge in result_b.root_edges
    ]
    assert projection_a == projection_b
    assert result_a.raw_network_value == result_b.raw_network_value


def test_blind_reserve_is_private_to_opponent_but_visible_to_owner() -> None:
    env = SplendorEnv(seed=9)
    state_a = env.initial_state()
    state_b = state_a.copy_shallow()
    card_a = state_a.hidden_tier_decks[1][0]
    card_b = state_a.hidden_tier_decks[1][1]
    state_a.players[0].reserve_card(card_a, RESERVE_SOURCE_DECK)
    state_b.players[0].reserve_card(card_b, RESERVE_SOURCE_DECK)

    assert encode_canonical_state_entities(state_a, 1) == encode_canonical_state_entities(
        state_b, 1
    )
    assert encode_canonical_state_entities(state_a, 0) != encode_canonical_state_entities(
        state_b, 0
    )


def test_reserve_deck_remains_in_search_distribution(checkpoint: Path) -> None:
    env = SplendorEnv(seed=10)
    state = env.initial_state()
    result = AlphaZeroSearch(
        AlphaZeroInference(checkpoint),
        SearchConfig(simulations=2, mode="debug", root_noise_fraction=0.0),
    ).run(env, state, self_play=False)
    assert any(edge.action.action_type is ActionType.RESERVE_DECK for edge in result.root_edges)


def test_backup_flips_value_between_players() -> None:
    edge = _Edge(prior=1.0)
    node = _Node(player_id=0, edges={})
    path = [_PathStep(node=node, edge=edge, parent_player=0, child_player=1)]
    AlphaZeroSearch._backup(path, 1.0)
    assert edge.visits == 1
    assert edge.q == -1.0
    assert node.visits == 1


def test_search_is_seed_deterministic(checkpoint: Path) -> None:
    env = SplendorEnv(seed=11)
    state = env.initial_state()
    config = SearchConfig(simulations=4, mode="real", seed=111)
    first = AlphaZeroSearch(AlphaZeroInference(checkpoint), config).run(
        env, state, self_play=True, seed=12345
    )
    second = AlphaZeroSearch(AlphaZeroInference(checkpoint), config).run(
        env, state, self_play=True, seed=12345
    )
    assert first.action == second.action
    assert [
        (edge.action_key, edge.visits, edge.network_prior, edge.search_prior, edge.q)
        for edge in first.root_edges
    ] == [
        (edge.action_key, edge.visits, edge.network_prior, edge.search_prior, edge.q)
        for edge in second.root_edges
    ]


def test_replay_round_trip_and_finite_loss(checkpoint: Path, tmp_path: Path) -> None:
    env = SplendorEnv(seed=12)
    state = env.initial_state()
    inference = AlphaZeroInference(checkpoint)
    config = SearchConfig(simulations=3, mode="debug", root_noise_fraction=0.0)
    result = AlphaZeroSearch(inference, config).run(env, state, self_play=False)
    row = replay_row_from_search(
        generation=0,
        game_seed=12,
        state=state,
        result=result,
        checkpoint_path=str(checkpoint),
        checkpoint_generation=0,
        search_config=asdict(config),
    ).with_outcome(winner=0, final_scores=(15, 10), termination_reason="completed")
    path = write_replay_jsonl(tmp_path / "replay.jsonl", [row])
    loaded = read_replay_jsonl(path)
    assert loaded == [row]

    batch = collate_replay_rows(loaded)
    model, _identity = load_checkpoint(checkpoint, device="cpu")
    logits, values = model(
        **{key: value for key, value in batch.items() if key not in {"policy_target", "value_target"}}
    )
    losses = alpha_zero_loss(
        logits,
        values,
        batch["action_mask"],
        batch["policy_target"],
        batch["value_target"],
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_checkpoint_reload_has_identical_predictions(checkpoint: Path, tmp_path: Path) -> None:
    env = SplendorEnv(seed=13)
    state = env.initial_state()
    actions = list(actions_by_canonical_successor_key(state, env.legal_actions(state)).values())
    original = AlphaZeroInference(checkpoint).predict_one(state, actions)
    model, identity = load_checkpoint(checkpoint, device="cpu")
    copy_path = tmp_path / "copy.pt"
    save_checkpoint(
        copy_path,
        model=model,
        generation=identity.generation,
        metadata=identity.metadata,
    )
    reloaded = AlphaZeroInference(copy_path).predict_one(state, actions)
    assert original == reloaded


def test_joint_loss_decreases_on_repeated_toy_batch(checkpoint: Path) -> None:
    env = SplendorEnv(seed=14)
    state = env.initial_state()
    inference = AlphaZeroInference(checkpoint)
    config = SearchConfig(simulations=3, mode="debug", root_noise_fraction=0.0)
    result = AlphaZeroSearch(inference, config).run(env, state, self_play=False)
    row = replay_row_from_search(
        generation=0,
        game_seed=14,
        state=state,
        result=result,
        checkpoint_path=str(checkpoint),
        checkpoint_generation=0,
        search_config=asdict(config),
    ).with_outcome(winner=0, final_scores=(15, 9), termination_reason="completed")
    batch = collate_replay_rows([row])
    model, _identity = load_checkpoint(checkpoint, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    def loss() -> torch.Tensor:
        logits, values = model(
            **{
                key: value
                for key, value in batch.items()
                if key not in {"policy_target", "value_target"}
            }
        )
        return alpha_zero_loss(
            logits,
            values,
            batch["action_mask"],
            batch["policy_target"],
            batch["value_target"],
        )["total_loss"]

    initial = float(loss().item())
    model.train()
    for _ in range(12):
        objective = loss()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    model.eval()
    final = float(loss().item())
    assert final < initial


def test_vertical_slice_trains_and_reloads_generation_one(tmp_path: Path) -> None:
    result = run_vertical_slice(
        SandboxConfig(
            output_dir=tmp_path / "az090",
            max_turns=4,
            arena_pairs=1,
        )
    )
    assert result["status"] == "complete"
    assert result["generation_one_actually_used"] is True
    assert result["training"]["parent_generation"] == 0
    assert result["training"]["child_generation"] == 1
    assert Path(result["generation_one_checkpoint"]).exists()


def test_core_modules_have_no_035_dependency() -> None:
    package = Path("src/splendor_ai/alphazero")
    for name in ("network.py", "search.py", "replay.py", "selfplay.py", "training.py"):
        text = (package / name).read_text(encoding="utf-8")
        assert "035" not in text

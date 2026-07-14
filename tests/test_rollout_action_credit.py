import json
from types import SimpleNamespace

import torch

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card
from splendor_ai.training.replay import serialize_action, serialize_state_snapshot
from splendor_ai.training.run_rollout_action_credit import (
    RolloutActionCreditConfig,
    build_rollout_action_credit_corpus,
    select_root_candidates,
)
from splendor_ai.training.run_rollout_credit_training import (
    compute_rollout_credit_loss,
)


class _FakeScorer:
    def __init__(self, *args, **kwargs) -> None:
        self.codec = ActionCodec()

    def score_actions(self, state, legal_actions):
        scores = [
            SimpleNamespace(
                action=action,
                action_index=self.codec.encode(state, action),
                score=(
                    10.0 - row * 0.01
                    if action.action_type == ActionType.RESERVE_VISIBLE
                    else float(-row)
                ),
            )
            for row, action in enumerate(legal_actions)
        ]
        return tuple(sorted(scores, key=lambda item: item.score, reverse=True))


def test_root_candidates_add_best_buy_actions() -> None:
    env = SplendorEnv(seed=1)
    state = env.initial_state()
    state.visible_tier_cards[1][0] = Card(
        card_id="free",
        tier=1,
        bonus_color="white",
        points=1,
        cost={},
    )
    legal_actions = env.legal_actions(state)
    scorer = _FakeScorer()
    scores = scorer.score_actions(state, legal_actions)

    candidates = select_root_candidates(scores, policy_top_k=2, max_candidates=4)

    assert len(candidates) <= 4
    assert any(
        item.action.action_type == ActionType.BUY_VISIBLE
        for item in candidates
    )


def test_rollout_credit_builder_writes_action_return_distribution(
    tmp_path,
    monkeypatch,
) -> None:
    env = SplendorEnv(seed=2)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    codec = ActionCodec()
    original = legal_actions[0]
    payload = {
        "game_seed": 2,
        "turn_index": 0,
        "player_id": 0,
        "action_index": codec.encode(state, original),
        "action_payload": serialize_action(original),
        "legal_action_indices": list(codec.legal_action_indices(state, legal_actions)),
        "observation_vector": [0.0],
        "final_value": 1.0,
        "game_final_scores": [15, 10],
        "game_bot_metadata": [
            {
                "bot_class": "LegalActionAttentionScorerBot",
                "checkpoint_path": (
                    "outputs/legal_action_scorer_022_attention_distill/model.pt"
                ),
            },
            {"bot_class": "ShallowSearchBot"},
        ],
        "state_snapshot": serialize_state_snapshot(state),
    }
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "splendor_ai.training.run_rollout_action_credit._LegalActionScorer",
        _FakeScorer,
    )
    monkeypatch.setattr(
        "splendor_ai.training.run_rollout_action_credit.rollout_action_return",
        lambda **kwargs: (
            0.75
            if kwargs["root_action"].action_type == ActionType.RESERVE_VISIBLE
            and kwargs["root_action"].market_index == 0
            else -0.25
        ),
    )

    output_path = tmp_path / "credit.jsonl"
    summary = build_rollout_action_credit_corpus(
        RolloutActionCreditConfig(
            checkpoint=tmp_path / "unused.pt",
            replay_paths=(replay_path,),
            output_path=output_path,
            summary_path=tmp_path / "summary.json",
            max_states=3,
            root_policy_top_k=2,
            max_root_candidates=3,
            rollouts_per_action=3,
            min_action_return_range=0.1,
            log_every_states=0,
        )
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["states_evaluated"] == 1
    assert summary["rows_written"] == 1
    assert len(row["search_action_scores"]) == 2
    assert row["search_action_scores"][0]["value"] == 0.75
    assert row["rollout_credit_metadata"]["action_return_range"] == 1.0


def test_rollout_credit_loss_uses_target_distribution_and_parent_kl() -> None:
    action_scores = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    parent_scores = action_scores.detach().clone()
    legal_mask = torch.ones((1, 3), dtype=torch.bool)
    rollout_mask = torch.tensor([[True, True, False]])
    rollout_returns = torch.tensor([[0.0, 1.0, 0.0]])

    losses = compute_rollout_credit_loss(
        action_scores=action_scores,
        parent_scores=parent_scores,
        legal_action_mask=legal_mask,
        rollout_return_target=rollout_returns,
        rollout_action_mask=rollout_mask,
        sample_weight=torch.ones(1),
        target_temperature=0.25,
        parent_kl_weight=2.0,
        entropy_weight=0.0,
    )

    assert torch.isclose(losses["parent_kl"], torch.tensor(0.0), atol=1e-7)
    assert losses["target_accuracy"].item() == 0.0
    losses["total_loss"].backward()
    assert torch.isfinite(action_scores.grad).all()

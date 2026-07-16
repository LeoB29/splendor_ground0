from __future__ import annotations

from collections import Counter

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import serialize_action, serialize_state_snapshot
from splendor_ai.training.run_entity_dagger import (
    DAGGER_TEACHER,
    LEAGUE,
    BalancedSourceDataset,
    EntityDaggerConfig,
    _confirmation_gates,
    _shallow_teacher,
    league_assignment,
    relabel_snapshot_row,
    split_league_seed_values,
)


class _Rows:
    def __init__(self, prefix: str, count: int) -> None:
        self.rows = [f"{prefix}{index}" for index in range(count)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _aggregate(loss: float, top1: float, mrr: float) -> dict[str, float]:
    return {
        "policy_loss": loss,
        "top1_accuracy": top1,
        "mean_reciprocal_rank": mrr,
    }


def test_league_schedule_is_balanced_by_profile_and_parent_seat() -> None:
    assignments = [league_assignment(index) for index in range(240)]

    assert Counter(opponent for opponent, _ in assignments) == Counter(
        {profile: 60 for profile in LEAGUE}
    )
    for profile in ("shallow", "016", "035"):
        seats = [seat for opponent, seat in assignments if opponent == profile]
        assert Counter(seats) == Counter({0: 30, 1: 30})
    assert all(seat is None for opponent, seat in assignments if opponent == "parent")


def test_league_split_is_reproducible_disjoint_and_profile_stratified() -> None:
    first = split_league_seed_values(290_000, 240, 192, 24, 24, seed=17)
    second = split_league_seed_values(290_000, 240, 192, 24, 24, seed=17)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 192,
        "validation": 24,
        "confirmation": 24,
    }
    seed_sets = [set(first[name]) for name in ("train", "validation", "confirmation")]
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])
    expected_per_profile = {"train": 48, "validation": 6, "confirmation": 6}
    for name, values in first.items():
        assert Counter((seed - 290_000) % 4 for seed in values) == Counter(
            {profile: expected_per_profile[name] for profile in range(4)}
        )


def test_balanced_source_dataset_has_exact_equal_exposure() -> None:
    dataset = BalancedSourceDataset(_Rows("old", 5), _Rows("new", 2))

    assert len(dataset) == 10
    assert Counter(dataset.source_for_index(index) for index in range(len(dataset))) == {
        "old": 5,
        "new": 5,
    }
    assert [dataset[index] for index in range(1, len(dataset), 2)] == [
        "new0",
        "new1",
        "new0",
        "new1",
        "new0",
    ]


def test_relabel_snapshot_row_preserves_rollout_action_and_uses_legal_teacher_action() -> None:
    env = SplendorEnv(seed=9)
    state = env.initial_state()
    actions = env.legal_actions(state)
    codec = ActionCodec()
    rollout = actions[0]
    row = {
        "state_snapshot": serialize_state_snapshot(state),
        "action_index": codec.encode(state, rollout),
        "action_payload": serialize_action(rollout),
    }

    relabeled, disagreed, transition = relabel_snapshot_row(
        row,
        teachers=(_shallow_teacher(), _shallow_teacher()),
        env=env,
        codec=codec,
        rollout_profile="parent",
        rollout_parent_seat=None,
    )

    assert row["action_index"] == codec.encode(state, rollout)
    assert relabeled["rollout_action_index"] == row["action_index"]
    assert relabeled["rollout_action_payload"] == row["action_payload"]
    assert relabeled["action_index"] in codec.legal_action_indices(state, actions)
    assert relabeled["target_source"] == DAGGER_TEACHER
    assert relabeled["rollout_profile"] == "parent"
    assert disagreed == (relabeled["action_index"] != row["action_index"])
    assert transition.endswith(f"->{relabeled['action_payload']['action_type']}")


def test_confirmation_gates_require_new_uptake_and_old_retention(tmp_path) -> None:
    config = EntityDaggerConfig(
        output_dir=tmp_path,
        old_source_dir=tmp_path,
        parent_checkpoint=tmp_path / "parent.pt",
        checkpoint_016=tmp_path / "016.pt",
        checkpoint_035=tmp_path / "035.pt",
    )
    evaluations = {
        "new_confirmation": {
            "parent": {"aggregate": _aggregate(2.0, 0.40, 0.60)},
            "candidate": {"aggregate": _aggregate(1.8, 0.44, 0.64)},
            "comparison": {
                "ci95_high": -0.05,
                "candidate_better_fraction": 0.75,
            },
        },
        "old_confirmation": {
            "parent": {"aggregate": _aggregate(1.0, 0.60, 0.75)},
            "candidate": {"aggregate": _aggregate(1.01, 0.59, 0.745)},
            "comparison": {},
        },
    }

    gates = _confirmation_gates(config, evaluations)

    assert all(gates.values())
    evaluations["old_confirmation"]["candidate"]["aggregate"]["policy_loss"] = 1.03
    assert _confirmation_gates(config, evaluations)[
        "old_policy_loss_regression_at_most_2pct"
    ] is False

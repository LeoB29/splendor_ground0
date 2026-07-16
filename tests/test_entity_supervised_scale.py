from __future__ import annotations

from splendor_ai.training.run_entity_supervised_scale import (
    EntityScaleConfig,
    _corpus_contract,
    _candidate_configs,
    split_game_heldout_indices,
)
from splendor_ai.training.run_entity_shallow_actor_ablation import (
    filter_indices_by_actor,
)


class _SeedDataset:
    def __init__(self, seeds: list[int]) -> None:
        self.seeds = seeds

    def __len__(self) -> int:
        return len(self.seeds)

    def game_seed_for_index(self, index: int) -> int:
        return self.seeds[index]

    def actor_label_for_index(self, index: int) -> str:
        return "ShallowSearchBot" if index % 2 else "GreedyHeuristicBot"


def test_three_way_game_split_is_disjoint_and_reproducible() -> None:
    dataset = _SeedDataset([seed for seed in range(10, 16) for _ in range(3)])

    first = split_game_heldout_indices(dataset, 4, 1, 1, seed=17)
    second = split_game_heldout_indices(dataset, 4, 1, 1, seed=17)

    assert first == second
    train_indices, validation_indices, confirmation_indices, split_seeds = first
    assert len(train_indices) == 12
    assert len(validation_indices) == 3
    assert len(confirmation_indices) == 3
    seed_sets = [set(split_seeds[name]) for name in ("train", "validation", "confirmation")]
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])
    assert set.union(*seed_sets) == set(range(10, 16))


def test_scale_candidates_only_change_preregistered_widths() -> None:
    configs = _candidate_configs()

    assert set(configs) == {"width64", "width128"}
    assert configs["width64"].entity_hidden_size == 64
    assert configs["width128"].entity_hidden_size == 128
    assert configs["width64"].entity_layers == configs["width128"].entity_layers == 1
    assert configs["width64"].action_layers == configs["width128"].action_layers == 1


def test_actor_filter_keeps_only_requested_source() -> None:
    dataset = _SeedDataset([1, 1, 2, 2, 3, 3])

    filtered = filter_indices_by_actor(dataset, list(range(6)), "ShallowSearchBot")

    assert filtered == [1, 3, 5]


def test_shallow_selfplay_contract_is_coherent(tmp_path) -> None:
    config = EntityScaleConfig(
        output_dir=tmp_path,
        parent_checkpoint=tmp_path / "unused.pt",
        pairing="shallow_shallow",
    )

    contract = _corpus_contract(config)

    assert contract["teacher"] == "information_legal_ShallowSearchBot_depth1_d1_selfplay"

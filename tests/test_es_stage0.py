from splendor_ai.training.run_es_stage0 import _aggregate_update, _calibrate_sigma, build_parser
from splendor_ai.training.es_adapter import normalized_direction


def test_es_stage0_parser_keeps_aggregate_gradient_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--replay-path", "replays.jsonl",
            "--output-dir", "outputs/es",
            "--output-path", "outputs/es/result.json",
        ]
    )

    assert args.directions == 16
    assert args.rank == 4
    assert args.fitness_pairs_per_sign == 50


def test_aggregate_update_uses_all_antithetic_direction_differences() -> None:
    directions = [normalized_direction(8, 1), normalized_direction(8, 2)]
    candidates = [
        {"direction_index": 0, "sign": 1, "fitness": {"mean": 0.2}},
        {"direction_index": 0, "sign": -1, "fitness": {"mean": -0.2}},
        {"direction_index": 1, "sign": 1, "fitness": {"mean": -0.1}},
        {"direction_index": 1, "sign": -1, "fitness": {"mean": 0.1}},
    ]

    update = _aggregate_update(directions, candidates, sigma=0.5)

    assert update.shape == (8,)
    assert abs(float(update.norm()) - 0.5) < 1e-6

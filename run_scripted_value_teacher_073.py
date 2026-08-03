"""Entry point for the gold-disciplined reserve-and-clear teacher panel (073).

Protocol: `docs/experiments/scripted_value_teacher_073.md`. This is the single
re-elicitation `072` authorized, and it is terminal for the scripted-strategy line.
"""

from pathlib import Path

from splendor_ai.eval.run_scripted_value_teacher import (
    TeacherPanelConfig,
    run_scripted_value_teacher,
)


PROTOCOL_073 = "docs/experiments/scripted_value_teacher_073.md"
SEED_START_073 = 760_000
BOOTSTRAP_SEED_073 = 760_073


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scripted value teacher 073.")
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/scripted_value_teacher_073")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    run_scripted_value_teacher(
        TeacherPanelConfig(
            checkpoint=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            pairs=args.pairs,
            seed_start=SEED_START_073,
            workers=args.workers,
            experiment="073",
            protocol=PROTOCOL_073,
            bootstrap_seed=BOOTSTRAP_SEED_073,
            protect_gold=True,
            reserve_and_clear=True,
        )
    )


if __name__ == "__main__":
    main()

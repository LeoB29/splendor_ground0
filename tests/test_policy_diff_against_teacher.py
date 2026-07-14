import json
from dataclasses import asdict

import torch

from splendor_ai.bots import RandomLegalBot
from splendor_ai.training import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.model import LegalActionScorerConfig, LegalActionScorerModel
from splendor_ai.training.run_policy_diff_against_teacher import (
    PolicyDiffTeacherConfig,
    compare_policies_against_teacher,
)


def _write_checkpoint(tmp_path):
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=12,
            state_depth=1,
            action_depth=1,
        )
    )
    checkpoint_path = tmp_path / "scorer.pt"
    torch.save(
        {
            "model_kind": "legal_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_model_tagged_replay(tmp_path):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=6,
        include_state_snapshots=True,
    )
    raw_path = tmp_path / "raw.jsonl"
    replay_path = tmp_path / "tagged.jsonl"
    export_replay_games_jsonl(raw_path, [replay])
    rows = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload["game_bot_seats"] = ["LegalActionScorerBot", "ShallowSearchBot"]
        rows.append(json.dumps(payload))
    replay_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return replay_path


def test_policy_diff_can_use_search_teacher_and_model_turn_filter(tmp_path) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    replay_path = _write_model_tagged_replay(tmp_path)
    output_path = tmp_path / "summary.json"

    summary = compare_policies_against_teacher(
        PolicyDiffTeacherConfig(
            before_checkpoint=checkpoint,
            before_model_kind="legal_action_scorer",
            after_checkpoint=checkpoint,
            after_model_kind="legal_action_scorer",
            replay_paths=(replay_path,),
            output_path=output_path,
            teacher_source="search",
            search_depth=1,
            search_max_branching=4,
            search_buy_branching=2,
            search_reserve_branching=1,
            search_take_branching=1,
            game_seed_filter=(3,),
            model_turns_only=True,
            log_every=0,
        )
    )

    assert output_path.exists()
    assert summary["rows_analyzed"] > 0
    assert summary["rows_skipped_by_model_turn"] > 0
    assert summary["teacher_source"] == "search"
    assert "before_teacher_match_rate" in summary["overall"]

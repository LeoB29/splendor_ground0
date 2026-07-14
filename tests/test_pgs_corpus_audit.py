import json
import subprocess
import sys

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_pgs_corpus_audit import (
    PGSCorpusAuditConfig,
    run_pgs_corpus_audit,
)


def _reserve_visible_payload(tier: int = 1, market_index: int = 0):
    return {
        "action_type": "RESERVE_VISIBLE",
        "tier": tier,
        "market_index": market_index,
        "reserved_index": None,
        "take_tokens": ["gold"],
        "return_tokens": [],
        "spend_tokens": [],
        "noble_id": None,
        "metadata": [],
    }


def _reserve_deck_payload(tier: int = 1):
    return {
        "action_type": "RESERVE_DECK",
        "tier": tier,
        "market_index": None,
        "reserved_index": None,
        "take_tokens": ["gold"],
        "return_tokens": [],
        "spend_tokens": [],
        "noble_id": None,
        "metadata": [],
    }


def _write_pgs_rows(tmp_path):
    state = SplendorEnv(seed=7).initial_state()
    base = {
        "game_seed": 7,
        "game_turns": 2,
        "game_winner": 0,
        "game_final_scores": [15, 10],
        "game_stalled": False,
        "game_timed_out": False,
        "turn_index": 0,
        "player_id": 0,
        "state_snapshot": serialize_state_snapshot(state),
        "final_value": 1.0,
        "winner": 0,
        "training_weight": 1.2,
    }
    risky = {
        **base,
        "action_index": 3528,
        "action_payload": _reserve_deck_payload(),
        "original_action_index": 2520,
        "original_action_payload": _reserve_visible_payload(),
        "improvement_metadata": {
            "changed_action": True,
            "search_margin": 87.0,
        },
    }
    anchor = {
        **base,
        "turn_index": 1,
        "action_index": 2520,
        "action_payload": _reserve_visible_payload(),
        "original_action_index": 2520,
        "original_action_payload": _reserve_visible_payload(),
        "improvement_metadata": {
            "changed_action": False,
            "search_margin": 0.0,
        },
        "training_weight": 0.8,
    }
    path = tmp_path / "pgs.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (risky, anchor)) + "\n",
        encoding="utf-8",
    )
    return path


def test_pgs_corpus_audit_filters_conservative_profile(tmp_path) -> None:
    input_path = _write_pgs_rows(tmp_path)
    output_path = tmp_path / "audit.json"
    filtered_path = tmp_path / "filtered.jsonl"

    summary = run_pgs_corpus_audit(
        PGSCorpusAuditConfig(
            input_paths=(input_path,),
            output_path=output_path,
            filtered_output_path=filtered_path,
            filter_profile="conservative_029",
            log_every=0,
        )
    )

    assert output_path.exists()
    assert filtered_path.exists()
    assert summary["rows_changed"] == 1
    assert summary["risky_early_changed_rows"] == 1
    assert summary["by_risk_reason"]["early_reserve_visible_to_deck"] == 1
    assert summary["filtered"]["rows_written"] == 1
    assert summary["filtered"]["rows_dropped"] == 1
    rows = [
        json.loads(line)
        for line in filtered_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["training_weight"] == 0.8


def test_run_pgs_corpus_audit_cli(tmp_path) -> None:
    input_path = _write_pgs_rows(tmp_path)
    output_path = tmp_path / "audit.json"

    result = subprocess.run(
        [
            sys.executable,
            "run_pgs_corpus_audit.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--log-every",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "[pgs-audit]" in result.stdout
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["rows_read"] == 2
    assert summary["by_transition"]["RESERVE_VISIBLE -> RESERVE_DECK"] == 1

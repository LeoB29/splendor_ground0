import json
import subprocess
import sys

from splendor_ai.training.run_manifest_report import print_manifest_report


def _write_manifest(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "source_type": "league",
                "files": {
                    "replays": "replays.jsonl",
                    "summary": "summary.json",
                },
                "seed_block": {
                    "seed_start": 100,
                    "seed_end_inclusive": 101,
                    "games": 2,
                },
                "results": {
                    "games": 2,
                    "total_steps": 130,
                    "seat0_wins": 1,
                    "seat1_wins": 1,
                    "draws": 0,
                    "stalled_games": 0,
                    "timed_out_games": 0,
                    "loop_fallback_triggers": 0,
                    "loop_fallback_games": 0,
                    "average_turns": 65.0,
                    "average_final_score_seat0": 14.0,
                    "average_final_score_seat1": 13.5,
                },
                "matchups": [
                    {
                        "seat0": {
                            "bot_class": "LegalActionScorerBot",
                            "label": (
                                "LegalActionScorerBot:legal_action_scorer:"
                                "outputs/legal_action_scorer_001/legal_action_scorer_best.pt"
                            ),
                            "metadata": {},
                        },
                        "seat1": {
                            "bot_class": "GreedyHeuristicBot",
                            "label": "GreedyHeuristicBot",
                            "metadata": {},
                        },
                        "games": 2,
                        "seat0_wins": 1,
                        "seat1_wins": 1,
                        "draws": 0,
                        "stalled_games": 0,
                        "timed_out_games": 0,
                        "loop_fallback_triggers": 0,
                        "seed_start": 100,
                        "seed_end_inclusive": 101,
                    }
                ],
                "metadata": {
                    "pairings": ["checkpoint:greedy"],
                },
            }
        ),
        encoding="utf-8",
    )
    return corpus_dir, manifest_path


def test_print_manifest_report_outputs_summary_and_matchups(tmp_path, capsys) -> None:
    _corpus_dir, manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print_manifest_report([(manifest_path, manifest)])

    output = capsys.readouterr().out
    assert "source" in output
    assert "league" in output
    assert "100-101" in output
    assert "1-1-0" in output
    assert "LegalActionScorerBot:legal_action_scorer:legal_action_scorer_001" in output


def test_run_manifest_report_cli_accepts_corpus_directory(tmp_path) -> None:
    corpus_dir, _manifest_path = _write_manifest(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "run_manifest_report.py",
            str(corpus_dir),
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "league" in result.stdout
    assert "100-101" in result.stdout
    assert "checkpoint:greedy" not in result.stdout

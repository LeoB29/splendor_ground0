import json

from splendor_ai.training.run_candidate_harness import (
    build_command,
    evaluate_gate,
    run_harness,
    value_at_path,
)


def test_value_at_path_reads_nested_dicts_and_lists() -> None:
    payload = {
        "overall": {"rate": 0.75},
        "opponents": [{"model_wins": 12}],
    }

    assert value_at_path(payload, "overall.rate") == 0.75
    assert value_at_path(payload, "opponents.0.model_wins") == 12


def test_evaluate_gate_compares_numeric_values() -> None:
    payload = {"overall": {"before": 0.7, "after": 0.72}}

    passed = evaluate_gate(
        payload,
        {
            "name": "after-rate",
            "path": "overall.after",
            "op": ">=",
            "value": 0.71,
        },
    )
    failed = evaluate_gate(
        payload,
        {
            "name": "before-rate",
            "path": "overall.before",
            "op": ">",
            "value": 0.75,
        },
    )

    assert passed.passed
    assert not failed.passed
    assert failed.actual == 0.7


def test_build_command_uses_module_invocation() -> None:
    command = build_command(
        {
            "module": "splendor_ai.training.example",
            "args": ["--flag", "value"],
        }
    )

    assert command[-3:] == ["splendor_ai.training.example", "--flag", "value"]
    assert command[-4] == "-m"


def test_run_harness_skips_benchmarks_after_failed_diagnostic(tmp_path) -> None:
    diagnostic_output = tmp_path / "diagnostic.json"
    diagnostic_output.write_text(
        json.dumps({"overall": {"after_teacher_match_rate": 0.5}}),
        encoding="utf-8",
    )
    benchmark_output = tmp_path / "benchmark.json"
    config = {
        "variants": [
            {
                "name": "variant",
                "diagnostics": [
                    {
                        "name": "diagnostic",
                        "command": ["python", "-c", "pass"],
                        "output_path": str(diagnostic_output),
                        "skip_if_outputs_exist": [str(diagnostic_output)],
                        "gates": [
                            {
                                "name": "min-match",
                                "path": "overall.after_teacher_match_rate",
                                "op": ">=",
                                "value": 0.6,
                            }
                        ],
                    }
                ],
                "benchmarks": [
                    {
                        "name": "benchmark",
                        "command": ["python", "-c", "pass"],
                        "output_path": str(benchmark_output),
                    }
                ],
            }
        ]
    }

    summary = run_harness(config, force=False, dry_run=False)

    variant = summary["variants"][0]
    assert variant["status"] == "diagnostic_failed"
    assert variant["diagnostics"][0]["status"] == "gate_failed"
    assert variant["benchmarks"] == []

"""Diagnostic-gated candidate training and evaluation harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    actual: object
    expected: object
    op: str
    path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run training variants through diagnostics and gated benchmarks."
    )
    parser.add_argument("--config", required=True, help="JSON harness config.")
    parser.add_argument("--output-path", required=True, help="Summary JSON path.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun stages even when their declared output paths already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and write a summary without running them.",
    )
    parser.add_argument(
        "--stop-after-diagnostics",
        action="store_true",
        help="Skip benchmarks even if diagnostics pass.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = run_harness(
        config=config,
        force=args.force,
        dry_run=args.dry_run,
        stop_after_diagnostics=args.stop_after_diagnostics,
    )
    summary["config_path"] = str(config_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[harness] wrote summary: {output_path}")
    failed = [
        variant
        for variant in summary["variants"]
        if variant["status"] in {"command_failed", "diagnostic_failed", "benchmark_failed"}
    ]
    if failed:
        print(f"[harness] failed/rejected variants: {', '.join(item['name'] for item in failed)}")


def run_harness(
    config: dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
    stop_after_diagnostics: bool = False,
) -> dict[str, Any]:
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Harness config requires a non-empty 'variants' list.")
    common_env = {
        str(key): str(value)
        for key, value in config.get("env", {}).items()
        if value is not None
    }
    started = time.perf_counter()
    variant_summaries = []
    for variant in variants:
        variant_summaries.append(
            _run_variant(
                variant=variant,
                common_env=common_env,
                force=force,
                dry_run=dry_run,
                stop_after_diagnostics=stop_after_diagnostics,
            )
        )
    return {
        "variants": variant_summaries,
        "timing": {
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def _run_variant(
    variant: dict[str, Any],
    common_env: dict[str, str],
    force: bool,
    dry_run: bool,
    stop_after_diagnostics: bool,
) -> dict[str, Any]:
    name = str(variant.get("name", "unnamed"))
    print(f"[harness:{name}] starting")
    summary: dict[str, Any] = {
        "name": name,
        "status": "pending",
        "train": None,
        "diagnostics": [],
        "benchmarks": [],
    }
    train_spec = variant.get("train")
    if train_spec is not None:
        train_result = _run_stage(
            stage_name=f"{name}:train",
            spec=train_spec,
            common_env=common_env,
            force=force,
            dry_run=dry_run,
        )
        summary["train"] = train_result
        if train_result["status"] == "failed":
            summary["status"] = "command_failed"
            return summary

    diagnostics_passed = True
    for diagnostic in variant.get("diagnostics", []):
        diagnostic_result = _run_stage_with_gates(
            stage_name=f"{name}:diagnostic:{diagnostic.get('name', 'unnamed')}",
            spec=diagnostic,
            common_env=common_env,
            force=force,
            dry_run=dry_run,
        )
        summary["diagnostics"].append(diagnostic_result)
        if diagnostic_result["status"] == "failed":
            summary["status"] = "command_failed"
            return summary
        if diagnostic_result["status"] == "gate_failed":
            diagnostics_passed = False
    if not diagnostics_passed:
        summary["status"] = "diagnostic_failed"
        print(f"[harness:{name}] diagnostics failed; skipping benchmarks")
        return summary
    if stop_after_diagnostics:
        summary["status"] = "diagnostics_passed"
        print(f"[harness:{name}] diagnostics passed; benchmarks skipped by flag")
        return summary

    benchmarks_passed = True
    for benchmark in variant.get("benchmarks", []):
        benchmark_result = _run_stage_with_gates(
            stage_name=f"{name}:benchmark:{benchmark.get('name', 'unnamed')}",
            spec=benchmark,
            common_env=common_env,
            force=force,
            dry_run=dry_run,
        )
        summary["benchmarks"].append(benchmark_result)
        if benchmark_result["status"] == "failed":
            summary["status"] = "command_failed"
            return summary
        if benchmark_result["status"] == "gate_failed":
            benchmarks_passed = False
    summary["status"] = "passed" if benchmarks_passed else "benchmark_failed"
    return summary


def _run_stage_with_gates(
    stage_name: str,
    spec: dict[str, Any],
    common_env: dict[str, str],
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    result = _run_stage(
        stage_name=stage_name,
        spec=spec,
        common_env=common_env,
        force=force,
        dry_run=dry_run,
    )
    output_path = result.get("output_path")
    if result["status"] == "failed" or dry_run:
        return result
    if output_path is None:
        if spec.get("gates"):
            raise ValueError(f"Stage with gates requires output_path: {stage_name}")
        return result
    payload = json.loads(Path(str(output_path)).read_text(encoding="utf-8"))
    gate_results = evaluate_gates(payload, spec.get("gates", []))
    result["gates"] = [asdict(gate_result) for gate_result in gate_results]
    if any(not gate_result.passed for gate_result in gate_results):
        result["status"] = "gate_failed"
    return result


def _run_stage(
    stage_name: str,
    spec: dict[str, Any],
    common_env: dict[str, str],
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    command = build_command(spec)
    output_path = spec.get("output_path")
    declared_outputs = [Path(str(path)) for path in spec.get("skip_if_outputs_exist", [])]
    if output_path is not None:
        declared_outputs.append(Path(str(output_path)))
    skipped = bool(declared_outputs) and all(path.exists() for path in declared_outputs) and not force
    result: dict[str, Any] = {
        "name": spec.get("name", stage_name),
        "status": "skipped" if skipped else "planned" if dry_run else "pending",
        "command": command,
        "output_path": output_path,
        "gates": [],
    }
    if skipped:
        print(f"[harness:{stage_name}] skipping; declared outputs exist")
        return result
    if dry_run:
        print(f"[harness:{stage_name}] dry-run: {' '.join(command)}")
        return result

    print(f"[harness:{stage_name}] running: {' '.join(command)}")
    started = time.perf_counter()
    env = os.environ.copy()
    env.update(common_env)
    env.update(
        {
            str(key): str(value)
            for key, value in spec.get("env", {}).items()
            if value is not None
        }
    )
    completed = subprocess.run(
        command,
        cwd=spec.get("cwd"),
        env=env,
        text=True,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    result["returncode"] = completed.returncode
    result["status"] = "completed" if completed.returncode == 0 else "failed"
    return result


def build_command(spec: dict[str, Any]) -> list[str]:
    if "command" in spec:
        command = spec["command"]
        if not isinstance(command, list) or not command:
            raise ValueError("'command' must be a non-empty list.")
        return [str(part) for part in command]
    module = spec.get("module")
    if not module:
        raise ValueError("Stage requires either 'command' or 'module'.")
    args = spec.get("args", [])
    if not isinstance(args, list):
        raise ValueError("'args' must be a list.")
    return [sys.executable, "-m", str(module), *[str(arg) for arg in args]]


def evaluate_gates(payload: dict[str, Any], gates: list[dict[str, Any]]) -> list[GateResult]:
    return [evaluate_gate(payload, gate) for gate in gates]


def evaluate_gate(payload: dict[str, Any], gate: dict[str, Any]) -> GateResult:
    name = str(gate.get("name", gate.get("path", "unnamed")))
    path = str(gate["path"])
    op = str(gate.get("op", ">="))
    expected = gate.get("value")
    actual = value_at_path(payload, path)
    passed = _compare(actual, expected, op)
    return GateResult(
        name=name,
        passed=passed,
        actual=actual,
        expected=expected,
        op=op,
        path=path,
    )


def value_at_path(payload: object, path: str) -> object:
    current = payload
    if not path:
        return current
    for raw_part in path.split("."):
        if isinstance(current, list):
            index = int(raw_part)
            current = current[index]
        elif isinstance(current, dict):
            current = current[raw_part]
        else:
            raise KeyError(f"Cannot descend into {type(current).__name__} at {raw_part!r}.")
    return current


def _compare(actual: object, expected: object, op: str) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op in {">", ">=", "<", "<="}:
        actual_number = float(actual)  # type: ignore[arg-type]
        expected_number = float(expected)  # type: ignore[arg-type]
        if op == ">":
            return actual_number > expected_number
        if op == ">=":
            return actual_number >= expected_number
        if op == "<":
            return actual_number < expected_number
        return actual_number <= expected_number
    raise ValueError(f"Unsupported gate op: {op}")


if __name__ == "__main__":
    main()

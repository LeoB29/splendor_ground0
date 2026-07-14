"""Summarize fixed objective and capacity ablations for reserve corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize fixed reserve counterfactual objective/capacity ablations."
    )
    parser.add_argument("--soft-audit-path", required=True)
    parser.add_argument("--hard-head-audit-path", required=True)
    parser.add_argument("--hard-full-audit-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path", required=True)
    return parser


def summarize_reserve_counterfactual_diagnostic(
    *,
    soft_audit_path: Path,
    hard_head_audit_path: Path,
    hard_full_audit_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    audits = {
        "soft_head": _load(soft_audit_path),
        "hard_head": _load(hard_head_audit_path),
        "hard_full": _load(hard_full_audit_path),
    }
    rows = {name: _row(audit) for name, audit in audits.items()}
    gates = {
        "hard_target_changes_heldout_top_one": (
            rows["hard_head"]["heldout_best_action_delta_ci95_low"] > 0.0
        ),
        "full_model_changes_train_top_one": (
            rows["hard_full"]["train_best_action_delta_ci95_low"] > 0.0
        ),
        "full_model_changes_heldout_top_one": (
            rows["hard_full"]["heldout_best_action_delta_ci95_low"] > 0.0
        ),
        "full_model_preserves_broad_kl": rows["hard_full"]["broad_mean_parent_kl"] <= 0.01,
    }
    conclusion = {
        "objective": (
            "Hard top-action supervision did not improve training or held-out top-one "
            "accuracy when the score head was the only trainable component."
        ),
        "capacity": (
            "Unfreezing the full model greatly improved calibration but did not establish "
            "training or held-out top-one improvement and exceeded the broad KL budget."
        ),
        "decision": (
            "Close this one-step matched-return imitation family. The next learner must "
            "change the credit or decision process, not retune target sharpness or model scope."
        ),
    }
    payload = {
        "format": "reserve_counterfactual_objective_capacity_diagnostic_v1",
        "audits": {
            "soft_head": str(soft_audit_path),
            "hard_head": str(hard_head_audit_path),
            "hard_full": str(hard_full_audit_path),
        },
        "comparison": rows,
        "gates": gates,
        "conclusion": conclusion,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render(payload), encoding="utf-8")
    return payload


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(audit: dict[str, Any]) -> dict[str, Any]:
    def metric(section: str, name: str) -> dict[str, float]:
        return audit[section][name]
    return {
        "target_objective": audit["config"]["target_objective"],
        "train_scope": audit["config"]["train_scope"],
        "train_best_action_delta": metric("train", "best_action_accuracy_delta")["mean"],
        "train_best_action_delta_ci95_low": metric("train", "best_action_accuracy_delta")["ci95_low"],
        "heldout_soft_ce_delta": metric("heldout", "soft_ce_delta_candidate_minus_baseline")["mean"],
        "heldout_best_action_delta": metric("heldout", "best_action_accuracy_delta")["mean"],
        "heldout_best_action_delta_ci95_low": metric("heldout", "best_action_accuracy_delta")["ci95_low"],
        "external_best_action_delta": metric("external", "best_action_accuracy_delta")["mean"],
        "heldout_top_action_change_rate": audit["heldout"]["full_top_action_change_rate"],
        "broad_top_action_change_rate": audit["broad_drift"]["top_action_change_rate"],
        "broad_mean_parent_kl": audit["broad_drift"]["mean_parent_kl"],
        "passes_its_uptake_gate": bool(audit["pass"]),
    }


def _render(payload: dict[str, Any]) -> str:
    comparison = payload["comparison"]
    lines = [
        "# Reserve Counterfactual Objective/Capacity Diagnostic",
        "",
        "| Arm | Target | Scope | Train top-one delta | Held-out top-one delta | Held-out CE delta | Broad KL |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name in ("soft_head", "hard_head", "hard_full"):
        row = comparison[name]
        lines.append(
            f"| {name} | {row['target_objective']} | {row['train_scope']} | "
            f"{row['train_best_action_delta']:+.3f} | {row['heldout_best_action_delta']:+.3f} | "
            f"{row['heldout_soft_ce_delta']:+.3f} | {row['broad_mean_parent_kl']:.4f} |"
        )
    lines.extend(["", "## Fixed Checks", ""])
    for name, passed in payload["gates"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Conclusion", ""])
    for name, text in payload["conclusion"].items():
        lines.append(f"- {name}: {text}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    payload = summarize_reserve_counterfactual_diagnostic(
        soft_audit_path=Path(args.soft_audit_path),
        hard_head_audit_path=Path(args.hard_head_audit_path),
        hard_full_audit_path=Path(args.hard_full_audit_path),
        output_path=Path(args.output_path),
        report_path=Path(args.report_path),
    )
    print(
        "[reserve-counterfactual-diagnostic]"
        f" objective={payload['gates']['hard_target_changes_heldout_top_one']}"
        f" capacity={payload['gates']['full_model_changes_heldout_top_one']}"
    )


if __name__ == "__main__":
    main()

"""Run the preregistered fresh policy-delta causal audit 034."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splendor_ai.training.run_policy_delta_causal_audit import (
    PolicyDeltaCausalAuditConfig,
    run_policy_delta_causal_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_policy_delta_causal_audit(
        PolicyDeltaCausalAuditConfig(
            output_dir=Path("outputs/policy_delta_causal_audit_034"),
            data_dir=Path("data/policy_delta_causal_audit_034"),
            parent_checkpoint=Path(
                "outputs/legal_action_scorer_035_legal_observation/"
                "legal_action_attention_scorer_best.pt"
            ),
            candidate_checkpoint=Path(
                "outputs/trust_region_generation_033/projected/"
                "legal_action_attention_scorer_projected.pt"
            ),
            checkpoint_016=Path(
                "outputs/legal_action_scorer_016_policy_guided_mix/"
                "legal_action_scorer_best.pt"
            ),
            checkpoint_023=Path(
                "outputs/canonical_entity_coherent_023/"
                "canonical_entity_action_scorer_coherent_width64_best.pt"
            ),
            workers=args.workers,
            device=args.device,
        )
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "diagnostic_only": result["diagnostic_only"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


"""Run the owner-authorized post-hoc 087 strength continuation."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval.run_expanded_root_resolver_champion_challenge import (
    ExpandedRootChampionChallengeConfig,
    _atomic_write,
    _identity,
    _operational_gate_names,
    _run_strength,
    _strength_gates,
    _validate_config,
)


WAIVER = Path(
    "docs/experiments/expanded_root_resolver_champion_challenge_087_owner_waiver.md"
)
ORIGINAL_RESULT_SHA256 = (
    "d2aa943c968b1fbaa11039c181bf3019bb9e7e91a5238d8c4c952c56ba98fee2"
)
ORIGINAL_SMOKE_SHA256 = (
    "0e5540f9793903b4dc86566eb4df51f3cb6c0bb6e6c7964614da2a622d63316e"
)


def run_owner_waived_strength() -> dict[str, Any]:
    cfg = ExpandedRootChampionChallengeConfig()
    _validate_config(cfg)
    original_result_path = cfg.output_dir / "result.json"
    smoke_path = cfg.output_dir / "smoke.json"
    if _sha256_file(original_result_path) != ORIGINAL_RESULT_SHA256:
        raise ValueError("Original 087 result changed after the owner waiver.")
    if _sha256_file(smoke_path) != ORIGINAL_SMOKE_SHA256:
        raise ValueError("Original 087 smoke changed after the owner waiver.")
    original = json.loads(original_result_path.read_text(encoding="utf-8"))
    smoke_gates = original["smoke"]["gates"]
    failed_smoke_gates = sorted(name for name, passed in smoke_gates.items() if not passed)
    if failed_smoke_gates != ["zero_rollout_fallback_triggers"]:
        raise ValueError("Owner waiver applies only to the one recorded smoke gate.")

    identity = _identity(cfg)
    started = time.perf_counter()
    strength = _run_strength(cfg, identity)
    gates = _strength_gates(cfg, strength)
    frozen_operational_pass = all(gates[name] for name in _operational_gate_names())
    frozen_all_strength_gates_pass = all(gates.values())
    non_time_gates = {
        name: passed for name, passed in gates.items() if name != "within_72_active_wall_hours"
    }
    payload = {
        "experiment": "087_owner_waived_strength_continuation",
        "status": "posthoc_owner_authorized_not_preregistered_promotion_evidence",
        "waiver": str(WAIVER).replace("\\", "/"),
        "waiver_sha256": _sha256_file(WAIVER),
        "original_result_sha256": _sha256_file(original_result_path),
        "original_smoke_sha256": _sha256_file(smoke_path),
        "original_087_decision": original["decision"],
        "waived_smoke_gate": "zero_rollout_fallback_triggers",
        "identity": identity,
        "strength": strength,
        "original_strength_gates": gates,
        "original_operational_gates_pass": frozen_operational_pass,
        "original_all_strength_gates_pass": frozen_all_strength_gates_pass,
        "all_non_time_strength_gates_pass": all(non_time_gates.values()),
        "formal_087_retroactive_pass": False,
        "automatic_research_champion_promotion_authorized": False,
        "research_champion": "public_belief_resolver_050_051",
        "fast_champion": "035",
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    _atomic_write(cfg.output_dir / "owner_waived_strength_result.json", payload)
    print("087 owner-waived strength continuation complete", flush=True)
    return payload


def main() -> None:
    run_owner_waived_strength()


if __name__ == "__main__":
    main()

"""Full consolidated-policy / parent-value transfer test for experiment 109."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .external_benchmark import independent_paired_difference_bootstrap
from .network import HybridAlphaZeroInference, load_checkpoint
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearchBot, SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap


@dataclass(frozen=True, slots=True)
class HybridTransferConfig:
    output_dir: Path
    result_108: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    consolidation_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "e315ee1a6c1faa3f41ce129562c2d2b9a67b65fb6f000f615ec20c6cdd5f7599"
EXPECTED_HASHES = {
    "result_108": "57392ef95e50b54ad239b870213240e37935a1a5a037cacb0442cf42f344aca4",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "consolidation_checkpoint": "c21c0ed3765e09820eb415c7369d0be0733913915e29d288db825cbcf9461c89",
}
SEED_START = 1_980_000
PAIRED_SEEDS = 100
BOOTSTRAP_SAMPLES = 50_000


def hybrid_transfer_decision(searched_assessment: dict[str, Any]) -> str:
    if searched_assessment["passed"]:
        return "hybrid_internal_pass_value_update_implicated_design_anchored_training_no_install"
    return "hybrid_internal_transfer_failed_close_consolidation_branch_no_install"


def run_hybrid_transfer_109(config: HybridTransferConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 109 result is not a valid completed result.")

    search = _evaluation_search()
    result: dict[str, Any] = {
        "kind": "alphazero_hybrid_transfer_109",
        "status": "running",
        "active_stage": "raw_full_consolidated_policy_vs_raw_generation_11",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "confirmed_search": asdict(search),
        "panels": {},
        "comparisons_with_108": {},
        "hybrid_identity": {
            "policy_checkpoint": str(config.consolidation_checkpoint),
            "policy_checkpoint_sha256": EXPECTED_HASHES["consolidation_checkpoint"],
            "value_checkpoint": str(config.generation_11_checkpoint),
            "value_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        },
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    specs = (
        (
            "raw_full_consolidated_policy_vs_raw_generation_11",
            alpha_zero_raw_factory(config.consolidation_checkpoint, config.device),
            alpha_zero_raw_factory(config.generation_11_checkpoint, config.device),
        ),
        (
            "searched_hybrid_vs_searched_generation_11",
            _hybrid_search_factory(
                config.consolidation_checkpoint,
                config.generation_11_checkpoint,
                config.device,
                search,
            ),
            alpha_zero_search_factory(config.generation_11_checkpoint, config.device, search),
        ),
    )
    for index, (name, candidate_factory, baseline_factory) in enumerate(specs):
        result["active_stage"] = name
        _write_progress(output, result, started)
        result["panels"][name] = _run_panel(
            output=output / "arenas",
            name=name,
            candidate=candidate_factory,
            baseline=baseline_factory,
            bootstrap_seed=1_090_300 + index,
            max_turns=config.max_turns,
        )
        _write_progress(output, result, started)

    prior = json.loads(config.result_108.read_text(encoding="utf-8"))
    raw = result["panels"]["raw_full_consolidated_policy_vs_raw_generation_11"]["arena"]
    searched = result["panels"]["searched_hybrid_vs_searched_generation_11"]["arena"]
    prior_raw = prior["internal_panels"]["raw_selected_repair_vs_raw_generation_11"]["arena"]
    prior_searched = prior["internal_panels"][
        "searched_selected_repair_vs_searched_generation_11"
    ]["arena"]
    result["comparisons_with_108"] = {
        "raw_full_policy_minus_lambda_025_vs_generation_11": independent_paired_difference_bootstrap(
            raw, prior_raw, samples=BOOTSTRAP_SAMPLES, seed=1_090_400
        ),
        "searched_hybrid_minus_lambda_025_vs_generation_11": independent_paired_difference_bootstrap(
            searched, prior_searched, samples=BOOTSTRAP_SAMPLES, seed=1_090_401
        ),
    }
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = hybrid_transfer_decision(
        result["panels"]["searched_hybrid_vs_searched_generation_11"]["assessment"]
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def _hybrid_search_factory(
    policy_checkpoint: Path,
    value_checkpoint: Path,
    device: str,
    config: SearchConfig,
):
    inference = HybridAlphaZeroInference(
        policy_checkpoint, value_checkpoint, device=device
    )
    evaluation = replace(config, root_noise_fraction=0.0)

    def factory(seed: int, seat: int) -> AlphaZeroSearchBot:
        return AlphaZeroSearchBot(
            config=replace(evaluation, seed=seed * 2 + seat),
            self_play=False,
            inference=inference,  # type: ignore[arg-type]
        )

    return factory


def _run_panel(
    *,
    output: Path,
    name: str,
    candidate: Any,
    baseline: Any,
    bootstrap_seed: int,
    max_turns: int,
) -> dict[str, Any]:
    arena = _arena_dict(
        run_paired_arena(
            name=name,
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=SEED_START,
            paired_seeds=PAIRED_SEEDS,
            max_turns=max_turns,
            progress_path=output / f"{name}.json",
            resume=True,
        )
    )
    return {
        "arena": arena,
        "paired_bootstrap": paired_score_share_bootstrap(
            arena, samples=BOOTSTRAP_SAMPLES, seed=bootstrap_seed
        ),
        "assessment": practical_assessment(arena),
    }


def _evaluation_search() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_090_109,
    )


def _verify_inputs(config: HybridTransferConfig) -> None:
    paths = {
        "result_108": config.result_108,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "consolidation_checkpoint": config.consolidation_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 109 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_108.read_text(encoding="utf-8"))
    raw = prior.get("internal_panels", {}).get(
        "raw_selected_repair_vs_raw_generation_11", {}
    ).get("arena", {})
    searched = prior.get("internal_panels", {}).get(
        "searched_selected_repair_vs_searched_generation_11", {}
    ).get("arena", {})
    hybrid_gate = prior.get("offline_audit", {}).get("variants", {}).get(
        "hybrid_candidate_policy_parent_value", {}
    ).get("retention_gate", {})
    if (
        prior.get("status") != "complete"
        or prior.get("decision") != "coherent_repair_retained_but_internal_gameplay_failed_no_install"
        or prior.get("selection")
        != {
            "lambda": 0.25,
            "pathway": "coherent_interpolation",
            "variant": "interpolation_2500",
        }
        or (raw.get("candidate_wins"), raw.get("draws"), raw.get("candidate_losses"))
        != (98, 1, 101)
        or (searched.get("candidate_wins"), searched.get("draws"), searched.get("candidate_losses"))
        != (98, 5, 97)
        or hybrid_gate.get("passed") is not True
    ):
        raise ValueError("Frozen 108 result lacks its exact selection, failure, or hybrid pass.")
    parent_model, parent_identity = load_checkpoint(
        config.generation_11_checkpoint, device=config.device
    )
    child_model, child_identity = load_checkpoint(
        config.consolidation_checkpoint, device=config.device
    )
    if (
        parent_identity.generation != 11
        or parent_identity.metadata.get("parent_generation") != 10
        or child_identity.generation != 12
        or child_identity.metadata.get("parent_generation") != 11
        or parent_identity.model_config != child_identity.model_config
        or parent_model.state_dict().keys() != child_model.state_dict().keys()
    ):
        raise ValueError("Frozen 109 checkpoints lack exact compatible lineage.")
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]:
        raise ValueError("Frozen 108 result did not retain exact generation 2.")


def _jsonable_config(config: HybridTransferConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )

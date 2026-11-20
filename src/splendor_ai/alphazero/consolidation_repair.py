"""Parent-safe interpolation and value-isolation audit for experiment 108."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import time
from typing import Any, Iterator

import torch

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .network import HybridAlphaZeroInference, load_checkpoint, save_checkpoint
from .replay import AlphaZeroReplayRow, _row_from_payload
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearchBot, SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .training import collate_replay_rows


@dataclass(frozen=True, slots=True)
class ConsolidationRepairConfig:
    output_dir: Path
    result_107: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    consolidation_checkpoint: Path
    replay_7: Path
    replay_8: Path
    replay_9: Path
    replay_10: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "e28d6ad21b505cd3a59768ae2899db5a406485fee642a57c6a38c14a5f8c92b8"
EXPECTED_HASHES = {
    "result_107": "d0bba71d37eb6883096f7c532f5abd25cd0bbcf5623c9fc10980dbb24a52e157",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "consolidation_checkpoint": "c21c0ed3765e09820eb415c7369d0be0733913915e29d288db825cbcf9461c89",
    "replay_7": "06ee156543da64de3126eed00cb363ed5f2d4c0e49e093550024fffc2ace5a1c",
    "replay_8": "96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10",
    "replay_9": "1b13ef038ccc6b3f695d7c7860a2284272ae18c8e75d44028e4cbb6f69914f0c",
    "replay_10": "09279d9593acfd944a32571716a6527c0be40e41055b6fa1f5bc25c88f43a60c",
}
INTERPOLATION_LAMBDAS = (0.0625, 0.125, 0.25, 0.5, 0.75)
NEWEST_OBJECTIVE_TOLERANCE = 0.005
REPRODUCTION_TOLERANCE = 1e-5
SEED_START = 1_970_000
PAIRED_SEEDS = 100
BOOTSTRAP_SAMPLES = 50_000
_GAME_SEED_PATTERN = re.compile(r'"game_seed"\s*:\s*(\d+)')


def interpolation_name(weight: float) -> str:
    return f"interpolation_{int(round(weight * 10_000)):04d}"


def interpolate_state_dicts(
    parent: dict[str, torch.Tensor],
    child: dict[str, torch.Tensor],
    weight: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Interpolation weight must be in [0,1].")
    if parent.keys() != child.keys():
        raise ValueError("Interpolation checkpoints have different state keys.")
    result: dict[str, torch.Tensor] = {}
    for key in parent:
        left = parent[key]
        right = child[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Interpolation tensor mismatch for {key}.")
        if left.dtype.is_floating_point or left.dtype.is_complex:
            result[key] = torch.lerp(left, right, weight)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"Non-floating interpolation tensor changed for {key}.")
            result[key] = left.clone()
    return result


def retention_gate(variant: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    corpus: dict[str, Any] = {}
    for generation in range(7, 11):
        current = variant["corpora"][str(generation)]
        anchor = parent["corpora"][str(generation)]
        corpus[str(generation)] = {
            "rows": int(current["rows"]),
            "parent_objective_loss": float(anchor["objective_loss"]),
            "variant_objective_loss": float(current["objective_loss"]),
            "delta_variant_minus_parent": float(current["objective_loss"])
            - float(anchor["objective_loss"]),
        }
    aggregate_delta = float(variant["aggregate"]["objective_loss"]) - float(
        parent["aggregate"]["objective_loss"]
    )
    checks = {
        "aggregate_objective_improved": aggregate_delta < 0.0,
        "replay_7_objective_improved": corpus["7"]["delta_variant_minus_parent"] < 0.0,
        "replay_8_objective_improved": corpus["8"]["delta_variant_minus_parent"] < 0.0,
        "replay_10_not_meaningfully_regressed": (
            corpus["10"]["delta_variant_minus_parent"]
            <= NEWEST_OBJECTIVE_TOLERANCE
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "aggregate_objective_delta_variant_minus_parent": aggregate_delta,
        "newest_objective_tolerance": NEWEST_OBJECTIVE_TOLERANCE,
        "corpus_objective_comparison": corpus,
    }


def select_repair(variants: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    coherent = [
        (name, metrics)
        for name, metrics in variants.items()
        if metrics.get("kind") == "coherent_interpolation"
        and metrics["retention_gate"]["passed"]
    ]
    if coherent:
        name, metrics = min(
            coherent,
            key=lambda item: (
                float(item[1]["aggregate"]["objective_loss"]),
                float(item[1]["lambda"]),
            ),
        )
        return {
            "pathway": "coherent_interpolation",
            "variant": name,
            "lambda": float(metrics["lambda"]),
        }
    hybrid_name = "hybrid_candidate_policy_parent_value"
    if variants[hybrid_name]["retention_gate"]["passed"]:
        return {"pathway": "hybrid_value_isolation", "variant": hybrid_name}
    return None


def repair_decision(selection: dict[str, Any] | None, searched: dict[str, Any] | None) -> str:
    if selection is None:
        return "no_repair_variant_passed_retention_no_gameplay_no_install"
    if searched is None:
        raise ValueError("A selected repair requires a searched assessment.")
    if selection["pathway"] == "coherent_interpolation":
        if searched["passed"]:
            return "coherent_repair_internal_pass_design_parent_safe_training_no_install"
        return "coherent_repair_retained_but_internal_gameplay_failed_no_install"
    if searched["passed"]:
        return "value_blocker_confirmed_by_hybrid_internal_pass_no_install"
    return "hybrid_retention_did_not_transfer_to_gameplay_no_install"


def run_consolidation_repair_108(config: ConsolidationRepairConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 108 result is not a valid completed result.")

    result: dict[str, Any] = {
        "kind": "alphazero_consolidation_repair_108",
        "status": "running",
        "active_stage": "streaming_common_holdout_audit",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "offline_audit": None,
        "selection": None,
        "derived_checkpoint": None,
        "internal_panels": {},
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    audit = _run_or_load_offline_audit(config)
    result["offline_audit"] = audit
    result["selection"] = audit["selection"]
    _write_progress(output, result, started)

    if result["selection"] is None:
        result["status"] = "complete"
        result["active_stage"] = "complete"
        result["decision"] = repair_decision(None, None)
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(sealed_path, result)
        return result

    selection = result["selection"]
    if selection["pathway"] == "coherent_interpolation":
        derived = _run_or_load_derived_checkpoint(config, selection)
        result["derived_checkpoint"] = derived
        policy_checkpoint = Path(derived["checkpoint"])
        raw_factory = alpha_zero_raw_factory(policy_checkpoint, config.device)
        searched_factory = alpha_zero_search_factory(
            policy_checkpoint, config.device, _evaluation_search()
        )
    else:
        raw_factory = alpha_zero_raw_factory(config.consolidation_checkpoint, config.device)
        searched_factory = _hybrid_search_factory(
            config.consolidation_checkpoint,
            config.generation_11_checkpoint,
            config.device,
            _evaluation_search(),
        )
    specs = (
        (
            "raw_selected_repair_vs_raw_generation_11",
            raw_factory,
            alpha_zero_raw_factory(config.generation_11_checkpoint, config.device),
        ),
        (
            "searched_selected_repair_vs_searched_generation_11",
            searched_factory,
            alpha_zero_search_factory(
                config.generation_11_checkpoint, config.device, _evaluation_search()
            ),
        ),
    )
    for index, (name, candidate_factory, baseline_factory) in enumerate(specs):
        result["active_stage"] = name
        _write_progress(output, result, started)
        result["internal_panels"][name] = _run_panel(
            output=output / "arenas",
            name=name,
            candidate=candidate_factory,
            baseline=baseline_factory,
            bootstrap_seed=1_080_300 + index,
            max_turns=config.max_turns,
        )
        _write_progress(output, result, started)
    searched_assessment = result["internal_panels"][
        "searched_selected_repair_vs_searched_generation_11"
    ]["assessment"]
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = repair_decision(selection, searched_assessment)
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def _run_or_load_offline_audit(config: ConsolidationRepairConfig) -> dict[str, Any]:
    path = config.output_dir / "offline_audit.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("Offline audit protocol hash mismatch.")
        if payload.get("input_hashes") != EXPECTED_HASHES:
            raise ValueError("Offline audit input hash mismatch.")
        return payload

    parent_model, _ = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    child_model, child_identity = load_checkpoint(
        config.consolidation_checkpoint, device=config.device
    )
    parent_state = parent_model.state_dict()
    child_state = child_model.state_dict()
    models: dict[str, torch.nn.Module] = {
        "parent": parent_model,
        "child": child_model,
    }
    for weight in INTERPOLATION_LAMBDAS:
        model = deepcopy(parent_model)
        model.load_state_dict(interpolate_state_dicts(parent_state, child_state, weight))
        model.eval()
        models[interpolation_name(weight)] = model
    variant_sources: dict[str, tuple[str, str, str, float | None]] = {
        "parent": ("parent", "parent", "anchor", 0.0),
        "full_candidate": ("child", "child", "anchor", 1.0),
        "hybrid_candidate_policy_parent_value": (
            "child",
            "parent",
            "hybrid_value_isolation",
            None,
        ),
    }
    for weight in INTERPOLATION_LAMBDAS:
        name = interpolation_name(weight)
        variant_sources[name] = (name, name, "coherent_interpolation", weight)
    validation_games = _validation_games(child_identity.metadata)
    replay_paths = {
        7: config.replay_7,
        8: config.replay_8,
        9: config.replay_9,
        10: config.replay_10,
    }
    variants = _stream_variant_metrics(
        replay_paths=replay_paths,
        validation_games=validation_games,
        models=models,
        variant_sources=variant_sources,
        device=config.device,
    )
    parent = variants["parent"]
    for name, metrics in variants.items():
        metrics["retention_gate"] = retention_gate(metrics, parent)
    prior = json.loads(config.result_107.read_text(encoding="utf-8"))
    reproduction = _reproduction_assessment(variants, prior)
    if not reproduction["passed"]:
        raise RuntimeError("108 failed to reproduce the exact 107 holdout metrics.")
    selection = select_repair(variants)
    payload = {
        "kind": "alphazero_consolidation_repair_offline_audit_108",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": dict(EXPECTED_HASHES),
        "interpolation_lambdas": list(INTERPOLATION_LAMBDAS),
        "validation_games": {
            str(generation): sorted(seeds)
            for generation, seeds in validation_games.items()
        },
        "validation_game_count": sum(len(seeds) for seeds in validation_games.values()),
        "variants": variants,
        "reproduction_assessment": reproduction,
        "selection": selection,
        "training_performed": False,
    }
    _write_json(path, payload)
    return payload


def _stream_variant_metrics(
    *,
    replay_paths: dict[int, Path],
    validation_games: dict[int, set[int]],
    models: dict[str, torch.nn.Module],
    variant_sources: dict[str, tuple[str, str, str, float | None]],
    device: str,
    batch_size: int = 256,
) -> dict[str, dict[str, Any]]:
    totals = {
        name: {
            "aggregate": _empty_totals(),
            "corpora": {str(generation): _empty_totals() for generation in replay_paths},
        }
        for name in variant_sources
    }
    for generation, path in replay_paths.items():
        batch: list[AlphaZeroReplayRow] = []
        for row in _selected_rows(path, generation, validation_games[generation]):
            batch.append(row)
            if len(batch) == batch_size:
                _accumulate_batch(batch, generation, models, variant_sources, totals, device)
                batch.clear()
        if batch:
            _accumulate_batch(batch, generation, models, variant_sources, totals, device)
    result: dict[str, dict[str, Any]] = {}
    for name, source in variant_sources.items():
        policy_source, value_source, kind, weight = source
        result[name] = {
            "kind": kind,
            "lambda": weight,
            "policy_source": policy_source,
            "value_source": value_source,
            "aggregate": _finish_totals(totals[name]["aggregate"]),
            "corpora": {
                generation: _finish_totals(values)
                for generation, values in totals[name]["corpora"].items()
            },
        }
    return result


def _selected_rows(path: Path, generation: int, selected_seeds: set[int]) -> Iterator[AlphaZeroReplayRow]:
    found: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _GAME_SEED_PATTERN.search(line)
            if match is None:
                raise ValueError("Replay row is missing game_seed.")
            seed = int(match.group(1))
            if seed not in selected_seeds:
                continue
            row = _row_from_payload(json.loads(line))
            if (
                row.generation != generation
                or row.checkpoint_generation != generation
                or row.value_target is None
                or row.termination_reason != "completed"
            ):
                raise ValueError(f"Selected replay row violates clean generation {generation} identity.")
            found.add(seed)
            yield row
    if found != selected_seeds:
        raise ValueError(f"Replay generation {generation} lacks an exact validation game.")


def _accumulate_batch(
    rows: list[AlphaZeroReplayRow],
    generation: int,
    models: dict[str, torch.nn.Module],
    variant_sources: dict[str, tuple[str, str, str, float | None]],
    totals: dict[str, dict[str, Any]],
    device: str,
) -> None:
    batch = collate_replay_rows(rows)
    moved = {key: value.to(device) for key, value in batch.items()}
    inputs = {
        key: value
        for key, value in moved.items()
        if key not in {"policy_target", "value_target"}
    }
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for name, model in models.items():
            outputs[name] = model(**inputs)
    for name, (policy_source, value_source, _kind, _weight) in variant_sources.items():
        logits = outputs[policy_source][0]
        values = outputs[value_source][1]
        masked = logits.masked_fill(
            ~moved["action_mask"], torch.finfo(logits.dtype).min
        )
        log_policy = torch.log_softmax(masked, dim=1)
        policy = -(moved["policy_target"] * log_policy).sum(dim=1)
        value = (values - moved["value_target"]) ** 2
        for target in (totals[name]["aggregate"], totals[name]["corpora"][str(generation)]):
            target["rows"] += len(rows)
            target["policy_sum"] += float(policy.sum().item())
            target["value_sum"] += float(value.sum().item())


def _empty_totals() -> dict[str, float | int]:
    return {"rows": 0, "policy_sum": 0.0, "value_sum": 0.0}


def _finish_totals(values: dict[str, float | int]) -> dict[str, Any]:
    rows = int(values["rows"])
    if rows <= 0:
        raise ValueError("A holdout metric contains no rows.")
    policy = float(values["policy_sum"]) / rows
    value = float(values["value_sum"]) / rows
    return {
        "rows": rows,
        "policy_cross_entropy": policy,
        "value_mse": value,
        "objective_loss": policy + 0.5 * value,
    }


def _validation_games(metadata: dict[str, Any]) -> dict[int, set[int]]:
    result = {generation: set() for generation in range(7, 11)}
    entries = metadata.get("validation_games")
    if not isinstance(entries, list):
        raise ValueError("107 checkpoint lacks validation-game metadata.")
    for entry in entries:
        generation_text, seed_text = str(entry).split(":", maxsplit=1)
        generation = int(generation_text)
        if generation not in result:
            raise ValueError("107 validation split contains an unexpected generation.")
        result[generation].add(int(seed_text))
    if sum(len(seeds) for seeds in result.values()) != 205 or any(not seeds for seeds in result.values()):
        raise ValueError("107 validation split does not contain the exact 205 games.")
    return result


def _reproduction_assessment(
    variants: dict[str, dict[str, Any]], prior: dict[str, Any]
) -> dict[str, Any]:
    expected_aggregate = {
        "parent": prior["training"]["training"]["parent_validation_metrics"],
        "full_candidate": prior["training"]["training"]["child_validation_metrics"],
    }
    checks: dict[str, bool] = {}
    errors: dict[str, float] = {}
    for variant, expected in expected_aggregate.items():
        actual = variants[variant]["aggregate"]
        for metric, expected_key in (
            ("policy_cross_entropy", "policy_loss"),
            ("value_mse", "value_loss"),
            ("objective_loss", "objective_loss"),
        ):
            key = f"{variant}_aggregate_{metric}"
            error = abs(float(actual[metric]) - float(expected[expected_key]))
            errors[key] = error
            checks[key] = error <= REPRODUCTION_TOLERANCE
    prior_corpora = prior["retention_assessment"]["held_out_corpus_metrics"]
    for generation in range(7, 11):
        for variant, expected_key in (
            ("parent", "parent_objective_loss"),
            ("full_candidate", "candidate_objective_loss"),
        ):
            key = f"{variant}_replay_{generation}_objective"
            error = abs(
                float(variants[variant]["corpora"][str(generation)]["objective_loss"])
                - float(prior_corpora[str(generation)][expected_key])
            )
            errors[key] = error
            checks[key] = error <= REPRODUCTION_TOLERANCE
    return {
        "passed": all(checks.values()),
        "tolerance": REPRODUCTION_TOLERANCE,
        "checks": checks,
        "absolute_errors": errors,
    }


def _run_or_load_derived_checkpoint(
    config: ConsolidationRepairConfig, selection: dict[str, Any]
) -> dict[str, Any]:
    manifest = config.output_dir / "checkpoints" / "selected_interpolation.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("selection") != selection
            or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        ):
            raise ValueError("Derived checkpoint manifest mismatch.")
        _, identity = load_checkpoint(payload["checkpoint"], device=config.device)
        if identity.generation != 12 or identity.metadata.get("parent_generation") != 11:
            raise ValueError("Derived checkpoint lineage mismatch.")
        return payload
    weight = float(selection["lambda"])
    parent, _ = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    child, _ = load_checkpoint(config.consolidation_checkpoint, device=config.device)
    parent.load_state_dict(interpolate_state_dicts(parent.state_dict(), child.state_dict(), weight))
    checkpoint = config.output_dir / "checkpoints" / f"selected_{interpolation_name(weight)}.pt"
    save_checkpoint(
        checkpoint,
        model=parent,
        generation=12,
        metadata={
            "kind": "alphazero_consolidation_interpolation_108",
            "parent_generation": 11,
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "consolidation_checkpoint": str(config.consolidation_checkpoint),
            "consolidation_checkpoint_sha256": EXPECTED_HASHES["consolidation_checkpoint"],
            "interpolation_lambda": weight,
            "protocol_sha256": PROTOCOL_SHA256,
        },
    )
    _, identity = load_checkpoint(checkpoint, device=config.device)
    if identity.generation != 12 or identity.metadata.get("interpolation_lambda") != weight:
        raise RuntimeError("Derived checkpoint failed round-trip validation.")
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "selection": selection,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    _write_json(manifest, payload)
    return payload


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
        seed=1_080_108,
    )


def _verify_inputs(config: ConsolidationRepairConfig) -> None:
    paths = {
        "result_107": config.result_107,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "consolidation_checkpoint": config.consolidation_checkpoint,
        "replay_7": config.replay_7,
        "replay_8": config.replay_8,
        "replay_9": config.replay_9,
        "replay_10": config.replay_10,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 108 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_107.read_text(encoding="utf-8"))
    if (
        prior.get("status") != "complete"
        or prior.get("decision") != "consolidation_retention_gate_failed_no_gameplay_no_install"
        or prior["retention_assessment"]["passed"] is not False
        or prior["internal_panels"] != {}
        or prior["external_panels"] != {}
    ):
        raise ValueError("Frozen 107 result lacks its exact gated stop.")
    expected_delta = 0.00611787877462566
    actual_delta = float(
        prior["retention_assessment"]["aggregate_objective_delta_candidate_minus_parent"]
    )
    if abs(actual_delta - expected_delta) > 1e-12:
        raise ValueError("Frozen 107 result has the wrong aggregate retention loss.")
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
        raise ValueError("Frozen 108 checkpoints lack exact compatible lineage.")
    _validation_games(child_identity.metadata)
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]:
        raise ValueError("Frozen 107 result did not retain exact generation 2.")


def _jsonable_config(config: ConsolidationRepairConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )

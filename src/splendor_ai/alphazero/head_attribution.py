"""Offline and hybrid-gameplay head attribution for programme 096."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from .arena import alpha_zero_search_factory, run_paired_arena
from .network import HybridAlphaZeroInference, load_checkpoint
from .replay import AlphaZeroReplayRow, read_replay_jsonl
from .search import AlphaZeroSearchBot, SearchConfig
from .search_primary import _arena_dict, _percentile, _seat_score_shares, paired_score_share_bootstrap
from .training import collate_replay_rows


@dataclass(frozen=True, slots=True)
class HeadAttributionConfig:
    output_dir: Path
    parent_checkpoint: Path
    child_checkpoint: Path
    replay_095: Path
    result_095: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "c3b78e215c254182c93117c76b4683ce4397f3743e0acbb4d68cad1eac46a638"
EXPECTED_HASHES = {
    "parent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "child_checkpoint": "36476fa8333a208c6d14559894cc491f9aa8f75a0a3c0c1b2972090d073090a4",
    "replay_095": "c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47",
    "result_095": "6d2ec319982ced65855d526e1fe2c692c1f4bda49d8e3f60c473b693b4714655",
}
VARIANTS = (
    "full_child",
    "child_policy_parent_value",
    "parent_policy_child_value",
)


def audit_replay_heads(
    *,
    parent_checkpoint: Path,
    child_checkpoint: Path,
    replay_path: Path,
    device: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    rows = read_replay_jsonl(replay_path)
    if len(rows) != 4106 or any(row.value_target is None for row in rows):
        raise ValueError("096 requires the exact complete 4,106-row 095 replay.")
    parent, _parent_identity = load_checkpoint(parent_checkpoint, device=device)
    child, _child_identity = load_checkpoint(child_checkpoint, device=device)
    records: list[dict[str, Any]] = []
    parent_reproduction_tv: list[float] = []
    parent_reproduction_max_error: list[float] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = collate_replay_rows(chunk)
        moved = {key: value.to(device) for key, value in batch.items()}
        inputs = {
            key: value
            for key, value in moved.items()
            if key not in {"policy_target", "value_target"}
        }
        with torch.no_grad():
            parent_logits, parent_values = parent(**inputs)
            child_logits, child_values = child(**inputs)
        floor = torch.finfo(parent_logits.dtype).min
        mask = moved["action_mask"]
        parent_probs = torch.softmax(parent_logits.masked_fill(~mask, floor), dim=1)
        child_probs = torch.softmax(child_logits.masked_fill(~mask, floor), dim=1)
        for local, row in enumerate(chunk):
            count = len(row.policy_target)
            target = [float(value) for value in row.policy_target]
            pp = [float(value) for value in parent_probs[local, :count].tolist()]
            cp = [float(value) for value in child_probs[local, :count].tolist()]
            stored = [float(value) for value in row.root_network_priors]
            parent_reproduction_tv.append(
                0.5 * sum(abs(a - b) for a, b in zip(pp, stored, strict=True))
            )
            parent_reproduction_max_error.append(
                max(abs(a - b) for a, b in zip(pp, stored, strict=True))
            )
            visit_top = _argmax(target)
            raw_top = int(row.raw_policy_top_row)
            selected_family = str(row.action_payloads[visit_top]["action_type"])
            records.append(
                {
                    "phase": _phase(row.turn_index),
                    "outcome": _outcome(float(row.value_target)),
                    "action_family": selected_family,
                    "search_changed_raw": visit_top != raw_top,
                    "parent_confidence": max(stored),
                    "target": target,
                    "visit_top": visit_top,
                    "parent_probs": pp,
                    "child_probs": cp,
                    "parent_value": float(parent_values[local].item()),
                    "child_value": float(child_values[local].item()),
                    "value_target": float(row.value_target),
                }
            )
    thresholds = _quartile_thresholds([row["parent_confidence"] for row in records])
    for record in records:
        record["confidence_quartile"] = _quartile_label(
            record["parent_confidence"], thresholds
        )
    strata: dict[str, dict[str, Any]] = {"overall": _summarize_records(records)}
    for field in (
        "phase",
        "outcome",
        "action_family",
        "search_changed_raw",
        "confidence_quartile",
    ):
        for value in sorted({str(record[field]) for record in records}):
            subset = [record for record in records if str(record[field]) == value]
            strata[f"{field}:{value}"] = _summarize_records(subset)
    decisive = [record for record in records if record["search_changed_raw"]]
    return {
        "rows": len(records),
        "games": len({row.game_seed for row in rows}),
        "confidence_quartile_thresholds": thresholds,
        "parent_prior_reproduction": {
            "mean_total_variation": sum(parent_reproduction_tv)
            / len(parent_reproduction_tv),
            "max_total_variation": max(parent_reproduction_tv),
            "mean_max_action_error": sum(parent_reproduction_max_error)
            / len(parent_reproduction_max_error),
            "max_action_error": max(parent_reproduction_max_error),
        },
        "strata": strata,
        "decisive_states": {
            "count": len(decisive),
            "fraction": len(decisive) / len(records),
            "summary": _summarize_records(decisive),
        },
    }


def paired_arena_difference_bootstrap(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    left_clusters = _arena_clusters(left)
    right_clusters = _arena_clusters(right)
    if set(left_clusters) != set(right_clusters):
        raise ValueError("Difference bootstrap requires identical paired seeds.")
    differences = [
        left_clusters[key] - right_clusters[key] for key in sorted(left_clusters)
    ]
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return {
        "paired_seeds": len(differences),
        "samples": samples,
        "seed": seed,
        "mean_difference": sum(differences) / len(differences),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def attribution_decision(
    value_swap: dict[str, Any], policy_swap: dict[str, Any]
) -> str:
    value_positive = float(value_swap["ci95_low"]) > 0.0
    policy_positive = float(policy_swap["ci95_low"]) > 0.0
    if value_positive and not policy_positive:
        return "child_value_head_primary_blocker"
    if policy_positive and not value_positive:
        return "child_policy_head_primary_blocker"
    if value_positive and policy_positive:
        return "both_heads_or_joint_incompatibility"
    return "no_clear_gameplay_head_attribution"


def run_head_attribution_096(config: HeadAttributionConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    prior = json.loads(config.result_095.read_text(encoding="utf-8"))
    if prior.get("decision") != "searched_confirmation_failed_keep_generation_2":
        raise ValueError("Hash-locked 095 result has the wrong decision.")
    result: dict[str, Any] = {
        "kind": "alphazero_head_attribution_096",
        "status": "running",
        "strength_claim": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "offline_audit": None,
        "gameplay": [],
        "head_swap_comparisons": None,
        "attribution": None,
        "installed_checkpoint": str(config.parent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.parent_checkpoint),
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "decision": "running",
    }
    _write_progress(output, result, started)
    result["offline_audit"] = audit_replay_heads(
        parent_checkpoint=config.parent_checkpoint,
        child_checkpoint=config.child_checkpoint,
        replay_path=config.replay_095,
        device=config.device,
    )
    _write_progress(output, result, started)
    search = _evaluation_search()
    factories = {
        "full_child": alpha_zero_search_factory(
            config.child_checkpoint, config.device, search
        ),
        "child_policy_parent_value": _hybrid_search_factory(
            config.child_checkpoint,
            config.parent_checkpoint,
            config.device,
            search,
        ),
        "parent_policy_child_value": _hybrid_search_factory(
            config.parent_checkpoint,
            config.child_checkpoint,
            config.device,
            search,
        ),
    }
    parent_factory = alpha_zero_search_factory(
        config.parent_checkpoint, config.device, search
    )
    bootstrap_seeds = (960_096, 960_097, 960_098)
    for order, variant in enumerate(VARIANTS):
        arena = _arena_dict(
            run_paired_arena(
                name=f"{variant}_vs_full_parent_search",
                candidate_factory=factories[variant],
                baseline_factory=parent_factory,
                seed_start=1_670_000,
                paired_seeds=24,
                max_turns=config.max_turns,
                progress_path=output / "arenas" / f"{variant}.json",
                resume=True,
            )
        )
        bootstrap = paired_score_share_bootstrap(
            arena, samples=50_000, seed=bootstrap_seeds[order]
        )
        result["gameplay"].append(
            {
                "order": order,
                "variant": variant,
                "policy_checkpoint": str(
                    config.child_checkpoint
                    if variant != "parent_policy_child_value"
                    else config.parent_checkpoint
                ),
                "value_checkpoint": str(
                    config.parent_checkpoint
                    if variant == "child_policy_parent_value"
                    else config.child_checkpoint
                ),
                "arena": arena,
                "paired_bootstrap": bootstrap,
                "seat_score_shares": {
                    str(key): value for key, value in _seat_score_shares(arena).items()
                },
                "restores_confirmed_transfer": float(bootstrap["ci95_low"]) > 0.5,
            }
        )
        _write_progress(output, result, started)
    by_name = {row["variant"]: row for row in result["gameplay"]}
    full = by_name["full_child"]["arena"]
    value_swap = paired_arena_difference_bootstrap(
        by_name["child_policy_parent_value"]["arena"],
        full,
        samples=50_000,
        seed=961_096,
    )
    policy_swap = paired_arena_difference_bootstrap(
        by_name["parent_policy_child_value"]["arena"],
        full,
        samples=50_000,
        seed=961_097,
    )
    decision = attribution_decision(value_swap, policy_swap)
    result["head_swap_comparisons"] = {
        "value_swap_minus_full_child": value_swap,
        "policy_swap_minus_full_child": policy_swap,
    }
    result["attribution"] = {
        "decision": decision,
        "rubric": "strict paired-bootstrap CI95 lower bound above zero",
    }
    return _finish(output, result, started, decision=decision)


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


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    networks = {}
    for name in ("parent", "child"):
        policy_key = f"{name}_probs"
        value_key = f"{name}_value"
        ces = [_cross_entropy(row["target"], row[policy_key]) for row in records]
        tvs = [_tv(row["target"], row[policy_key]) for row in records]
        jsds = [_jsd(row["target"], row[policy_key]) for row in records]
        errors = [row[value_key] - row["value_target"] for row in records]
        decisive = [row for row in records if row["search_changed_raw"]]
        networks[name] = {
            "policy_cross_entropy": sum(ces) / len(ces),
            "policy_target_top_agreement": sum(
                _argmax(row[policy_key]) == row["visit_top"] for row in records
            )
            / len(records),
            "policy_target_total_variation": sum(tvs) / len(tvs),
            "policy_target_jsd": sum(jsds) / len(jsds),
            "value_mean": sum(row[value_key] for row in records) / len(records),
            "value_bias": sum(errors) / len(errors),
            "value_mae": sum(abs(value) for value in errors) / len(errors),
            "value_mse": sum(value * value for value in errors) / len(errors),
            "value_sign_accuracy": sum(
                _sign(row[value_key]) == _sign(row["value_target"])
                for row in records
            )
            / len(records),
            "decisive_visit_top_agreement": (
                sum(_argmax(row[policy_key]) == row["visit_top"] for row in decisive)
                / len(decisive)
                if decisive
                else None
            ),
            "decisive_visit_top_probability": (
                sum(row[policy_key][row["visit_top"]] for row in decisive)
                / len(decisive)
                if decisive
                else None
            ),
        }
    parent = networks["parent"]
    child = networks["child"]
    return {
        "count": len(records),
        "parent": parent,
        "child": child,
        "delta_child_minus_parent": {
            key: child[key] - parent[key]
            for key in (
                "policy_cross_entropy",
                "policy_target_top_agreement",
                "policy_target_total_variation",
                "policy_target_jsd",
                "value_bias",
                "value_mae",
                "value_mse",
                "value_sign_accuracy",
                "decisive_visit_top_agreement",
                "decisive_visit_top_probability",
            )
            if child[key] is not None and parent[key] is not None
        },
    }


def _arena_clusters(arena: dict[str, Any]) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for row in arena["rows"]:
        values.setdefault(int(row["seed"]), []).append(float(row["candidate_points"]))
    if not values or any(len(points) != 2 for points in values.values()):
        raise ValueError("Arena requires exactly two candidate seats per seed.")
    return {seed: sum(points) / 2.0 for seed, points in values.items()}


def _evaluation_search() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=960_096,
    )


def _phase(turn: int) -> str:
    return "early_lt20" if turn < 20 else ("middle_20_49" if turn < 50 else "late_ge50")


def _outcome(value: float) -> str:
    return "win" if value > 0 else ("loss" if value < 0 else "draw")


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _cross_entropy(target: list[float], predicted: list[float]) -> float:
    return -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, predicted, strict=True))


def _tv(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def _jsd(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right, strict=True)]
    return 0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint)


def _kl(left: list[float], right: list[float]) -> float:
    return sum(a * math.log(a / b) for a, b in zip(left, right, strict=True) if a > 0 and b > 0)


def _quartile_thresholds(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [_percentile(ordered, probability) for probability in (0.25, 0.5, 0.75)]


def _quartile_label(value: float, thresholds: list[float]) -> str:
    if value <= thresholds[0]:
        return "q1"
    if value <= thresholds[1]:
        return "q2"
    if value <= thresholds[2]:
        return "q3"
    return "q4"


def _sign(value: float) -> int:
    return 0 if math.isclose(value, 0.0, abs_tol=1e-12) else (1 if value > 0 else -1)


def _verify_inputs(config: HeadAttributionConfig) -> None:
    for name, path in {
        "parent_checkpoint": config.parent_checkpoint,
        "child_checkpoint": config.child_checkpoint,
        "replay_095": config.replay_095,
        "result_095": config.result_095,
    }.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 096 input hash mismatch for {name}: {actual}")


def _jsonable_config(config: HeadAttributionConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "parent_checkpoint": str(config.parent_checkpoint),
        "child_checkpoint": str(config.child_checkpoint),
        "replay_095": str(config.replay_095),
        "result_095": str(config.result_095),
    }


def _finish(output: Path, result: dict[str, Any], started: float, *, decision: str) -> dict[str, Any]:
    result["status"] = "complete"
    result["decision"] = decision
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Post-hoc disagreement and logit audit for rejected experiment 052."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


RESULT_SHA256 = "60458c1bea4328b64f33ad8238529d074a25d1034fa2845e0b339deee3827153"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
STUDENT_SHA256 = "32236a090513dd2e7a4bb4e249f11f8f5a3de8f4387a6feadb9b1684dea3773b"
TARGET_SHA256 = {
    "train": "6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5",
    "validation": "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052",
    "confirmation": "8625d9049957ce266fa544896f0262cb37ccea6d168ed0c60a847ef5325cce8f",
}
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


@dataclass(frozen=True, slots=True)
class ResolverDistillationAuditConfig:
    source_dir: Path = Path("outputs/resolver_selfplay_distillation_052")
    parent_checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    student_checkpoint: Path = Path(
        "outputs/resolver_selfplay_distillation_052/"
        "legal_action_attention_scorer_best.pt"
    )
    output_dir: Path = Path("outputs/resolver_distillation_disagreement_audit_052a")
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 520_052
    probability_tolerance: float = 1e-6
    device: str = "cpu"


def run_audit(
    config: ResolverDistillationAuditConfig = ResolverDistillationAuditConfig(),
) -> dict[str, Any]:
    """Run the frozen 052a diagnostic and persist its summary."""

    identity = _validate_identity(config)
    parent = _LegalActionScorer(
        config.parent_checkpoint,
        config.device,
        "legal_action_attention_scorer",
    )
    student = _LegalActionScorer(
        config.student_checkpoint,
        config.device,
        "legal_action_attention_scorer",
    )
    env = SplendorEnv()
    codec = ActionCodec()
    splits: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(("train", "validation", "confirmation")):
        print(f"[resolver052a:{split}] evaluating frozen rows", flush=True)
        path = config.source_dir / f"{split}_targets.jsonl"
        split_rows, integrity = _evaluate_rows(
            path,
            parent=parent,
            student=student,
            env=env,
            codec=codec,
            tolerance=config.probability_tolerance,
        )
        splits[split] = _summarize_split(
            split_rows,
            integrity,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed + split_index * 10_000,
        )
        print(
            f"[resolver052a:{split}] rows={splits[split]['rows']}"
            f" raw_disagreements={splits[split]['populations']['raw_resolver']['rows']}"
            f" target_disagreements={splits[split]['populations']['blended_target']['rows']}",
            flush=True,
        )
    payload = {
        "experiment": "052a_resolver_distillation_disagreement_logit_audit",
        "protocol": "docs/experiments/resolver_distillation_disagreement_audit_052a.md",
        "configuration": _config_payload(config),
        "identity": identity,
        "splits": splits,
        "integrity_passed": all(split["integrity"]["passed"] for split in splits.values()),
        "training_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(config.output_dir / "result.json", payload)
    return payload


def _validate_identity(config: ResolverDistillationAuditConfig) -> dict[str, Any]:
    paths = {
        "result_052": config.source_dir / "result.json",
        "parent_035": config.parent_checkpoint,
        "student_052": config.student_checkpoint,
        **{
            f"{split}_targets": config.source_dir / f"{split}_targets.jsonl"
            for split in TARGET_SHA256
        },
    }
    expected = {
        "result_052": RESULT_SHA256,
        "parent_035": PARENT_SHA256,
        "student_052": STUDENT_SHA256,
        **{f"{split}_targets": value for split, value in TARGET_SHA256.items()},
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256_file(path)
        if observed[name] != expected[name]:
            raise ValueError(
                f"Frozen 052a identity mismatch for {name}: "
                f"expected {expected[name]}, observed {observed[name]}."
            )
    result = json.loads(paths["result_052"].read_text(encoding="utf-8"))
    if result.get("decision") != "offline_distillation_gate_failed" or result.get("passed"):
        raise ValueError("052a requires the frozen rejected 052 result.")
    return {"expected_sha256": expected, "observed_sha256": observed}


def _evaluate_rows(
    path: Path,
    *,
    parent: _LegalActionScorer,
    student: _LegalActionScorer,
    env: SplendorEnv,
    codec: ActionCodec,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    checks = {
        "legal_support_exact": True,
        "target_distribution_valid": True,
        "parent_argmax_reproduced": True,
        "stored_parent_distribution_reproduced": True,
        "referenced_actions_legal": True,
        "resolver_values_aligned": True,
        "finite_scores": True,
    }
    maximum_parent_probability_error = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            state = deserialize_state_snapshot(row["state_snapshot"])
            legal = env.legal_actions(state)
            legal_indices = [codec.encode(state, action) for action in legal]
            target_items = row["resolver_policy_distribution"]
            stored_indices = [int(item["action_index"]) for item in target_items]
            checks["legal_support_exact"] &= legal_indices == stored_indices
            target_by_index = {
                int(item["action_index"]): float(item["probability"])
                for item in target_items
            }
            stored_parent_by_index = {
                int(item["action_index"]): float(item["parent_probability"])
                for item in target_items
            }
            target = [target_by_index.get(index, math.nan) for index in legal_indices]
            checks["target_distribution_valid"] &= (
                all(math.isfinite(value) and value >= 0.0 for value in target)
                and abs(sum(target) - 1.0) <= tolerance
            )
            parent_scores = _aligned_scores(parent, state, legal, legal_indices)
            student_scores = _aligned_scores(student, state, legal, legal_indices)
            checks["finite_scores"] &= all(
                math.isfinite(value) for value in parent_scores + student_scores
            )
            parent_probs = _softmax(parent_scores)
            for index, probability in zip(legal_indices, parent_probs, strict=True):
                error = abs(probability - stored_parent_by_index.get(index, math.nan))
                maximum_parent_probability_error = max(
                    maximum_parent_probability_error,
                    error if math.isfinite(error) else math.inf,
                )
            checks["stored_parent_distribution_reproduced"] &= (
                maximum_parent_probability_error <= tolerance
            )
            parent_row = _argmax_row(parent_scores)
            student_row = _argmax_row(student_scores)
            target_row = _argmax_row(target)
            parent_index = legal_indices[parent_row]
            student_index = legal_indices[student_row]
            target_index = legal_indices[target_row]
            resolver_index = int(row["action_index"])
            stored_parent_index = int(row["raw_parent_action_index"])
            legal_index_set = set(legal_indices)
            checks["referenced_actions_legal"] &= {
                resolver_index,
                stored_parent_index,
                target_index,
                student_index,
            } <= legal_index_set
            checks["parent_argmax_reproduced"] &= parent_index == stored_parent_index
            by_index = {index: offset for offset, index in enumerate(legal_indices)}
            group_by_action, value_by_action, groups_aligned = _resolver_value_maps(
                row.get("resolver_action_values", [])
            )
            if bool(row["target_available"]):
                checks["resolver_values_aligned"] &= (
                    groups_aligned
                    and resolver_index in group_by_action
                    and stored_parent_index in group_by_action
                )
            base = {
                "game_seed": int(row["game_seed"]),
                "turn_index": int(row["turn_index"]),
                "phase": str(row["phase"]),
                "seat": int(row["player_id"]),
                "outcome": str(row["outcome_role"]),
                "target_available": bool(row["target_available"]),
                "parent_index": parent_index,
                "student_index": student_index,
                "target_index": target_index,
                "resolver_index": resolver_index,
            }
            if not bool(row["target_available"]):
                records.append(
                    {
                        **base,
                        "kind": "fallback",
                        "fallback_kl": _kl(parent_probs, _softmax(student_scores)),
                        "fallback_moved": student_index != parent_index,
                        "fallback_entropy_delta": (
                            _entropy(_softmax(student_scores)) - _entropy(parent_probs)
                        ),
                    }
                )
                continue
            action_by_index = {
                index: action for index, action in zip(legal_indices, legal, strict=True)
            }
            raw_pair = None
            if resolver_index != parent_index:
                raw_pair = _pair_record(
                    base,
                    preferred_index=resolver_index,
                    parent_scores=parent_scores,
                    student_scores=student_scores,
                    target_by_index=target_by_index,
                    by_index=by_index,
                    action_by_index=action_by_index,
                )
                parent_group = group_by_action.get(parent_index)
                resolver_group = group_by_action.get(resolver_index)
                raw_pair.update(
                    {
                        "canonical_disagreement": (
                            parent_group is not None
                            and resolver_group is not None
                            and parent_group != resolver_group
                        ),
                        "alias_only": (
                            parent_group is not None and parent_group == resolver_group
                        ),
                        "resolver_value_advantage": (
                            value_by_action[resolver_index] - value_by_action[parent_index]
                            if resolver_index in value_by_action
                            and parent_index in value_by_action
                            else None
                        ),
                    }
                )
            target_pair = None
            if target_index != parent_index:
                target_pair = _pair_record(
                    base,
                    preferred_index=target_index,
                    parent_scores=parent_scores,
                    student_scores=student_scores,
                    target_by_index=target_by_index,
                    by_index=by_index,
                    action_by_index=action_by_index,
                )
            records.append(
                {
                    **base,
                    "kind": "resolved",
                    "raw_pair": raw_pair,
                    "target_pair": target_pair,
                    "raw_and_target_preferred_same": resolver_index == target_index,
                }
            )
    integrity = {
        **checks,
        "maximum_parent_probability_error": maximum_parent_probability_error,
        "rows": len(records),
        "passed": all(checks.values()),
    }
    return records, integrity


def _pair_record(
    base: dict[str, Any],
    *,
    preferred_index: int,
    parent_scores: list[float],
    student_scores: list[float],
    target_by_index: dict[int, float],
    by_index: dict[int, int],
    action_by_index: dict[int, Action],
) -> dict[str, Any]:
    parent_index = int(base["parent_index"])
    preferred_row = by_index[preferred_index]
    parent_row = by_index[parent_index]
    parent_gap = parent_scores[preferred_row] - parent_scores[parent_row]
    student_gap = student_scores[preferred_row] - student_scores[parent_row]
    student_index = int(base["student_index"])
    return {
        **{key: base[key] for key in ("game_seed", "turn_index", "phase", "seat", "outcome")},
        "preferred_index": preferred_index,
        "parent_index": parent_index,
        "student_index": student_index,
        "preferred_action_type": action_by_index[preferred_index].action_type.value,
        "parent_gap": parent_gap,
        "student_gap": student_gap,
        "gap_delta": student_gap - parent_gap,
        "directional_movement": student_gap > parent_gap,
        "crossed": parent_gap <= 0.0 and student_gap > 0.0,
        "captured": student_index == preferred_index,
        "retained_parent": student_index == parent_index,
        "moved_third": student_index not in {preferred_index, parent_index},
        "parent_rank": _rank(parent_scores, preferred_row),
        "student_rank": _rank(student_scores, preferred_row),
        "target_probability_advantage": (
            target_by_index[preferred_index] - target_by_index[parent_index]
        ),
        "parent_gap_bin": _parent_gap_bin(parent_gap),
    }


def _resolver_value_maps(
    values: Iterable[dict[str, Any]],
) -> tuple[dict[int, int], dict[int, float], bool]:
    group_by_action: dict[int, int] = {}
    value_by_action: dict[int, float] = {}
    aligned = True
    for group, item in enumerate(values):
        value = float(item["value"])
        aligned &= math.isfinite(value)
        for raw_index in item["action_indices"]:
            index = int(raw_index)
            aligned &= index not in group_by_action
            group_by_action[index] = group
            value_by_action[index] = value
    return group_by_action, value_by_action, aligned


def _aligned_scores(
    scorer: _LegalActionScorer,
    state: Any,
    legal: list[Action],
    legal_indices: list[int],
) -> list[float]:
    scored = scorer.score_actions(state, legal)
    by_index = {int(item.action_index): float(item.score) for item in scored}
    if set(by_index) != set(legal_indices) or len(by_index) != len(legal_indices):
        raise ValueError("Model score support does not match regenerated legal actions.")
    return [by_index[index] for index in legal_indices]


def _summarize_split(
    records: list[dict[str, Any]],
    integrity: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    resolved = [row for row in records if row["kind"] == "resolved"]
    fallback = [row for row in records if row["kind"] == "fallback"]
    raw = [row["raw_pair"] for row in resolved if row["raw_pair"] is not None]
    canonical = [row for row in raw if row["canonical_disagreement"]]
    alias_only = [row for row in raw if row["alias_only"]]
    target = [row["target_pair"] for row in resolved if row["target_pair"] is not None]
    populations = {
        "raw_resolver": _population_report(
            raw, bootstrap_samples, bootstrap_seed + 1
        ),
        "canonical_resolver": _population_report(
            canonical, bootstrap_samples, bootstrap_seed + 2
        ),
        "alias_only_resolver": _population_report(
            alias_only, bootstrap_samples, bootstrap_seed + 3
        ),
        "blended_target": _population_report(
            target, bootstrap_samples, bootstrap_seed + 4
        ),
    }
    return {
        "rows": len(records),
        "resolved_rows": len(resolved),
        "fallback_rows": len(fallback),
        "raw_resolver_disagreement_rate": len(raw) / len(resolved) if resolved else 0.0,
        "canonical_resolver_disagreement_rate": (
            len(canonical) / len(resolved) if resolved else 0.0
        ),
        "blended_target_disagreement_rate": (
            len(target) / len(resolved) if resolved else 0.0
        ),
        "raw_resolver_equals_blended_target_rate": (
            statistics.fmean(row["raw_and_target_preferred_same"] for row in resolved)
            if resolved
            else 0.0
        ),
        "populations": populations,
        "fallback": _fallback_report(fallback),
        "integrity": integrity,
    }


def _population_report(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary = _pair_summary(rows)
    summary["bootstrap_ci95"] = _cluster_bootstrap(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    seeds = sorted({int(row["game_seed"]) for row in rows})
    midpoint = len(seeds) // 2
    first = set(seeds[:midpoint])
    second = set(seeds[midpoint:])
    summary["seed_ordered_halves"] = {
        "first": _pair_summary([row for row in rows if row["game_seed"] in first]),
        "second": _pair_summary([row for row in rows if row["game_seed"] in second]),
    }
    summary["strata"] = {
        "phase": _stratify(rows, "phase"),
        "seat": _stratify(rows, "seat"),
        "outcome": _stratify(rows, "outcome"),
        "preferred_action_type": _stratify(rows, "preferred_action_type"),
        "parent_gap_bin": _stratify(rows, "parent_gap_bin"),
    }
    return summary


def _pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "directional_movement": 0.0,
            "crossed": 0.0,
            "preferred_action_capture": 0.0,
            "retained_parent": 0.0,
            "moved_third": 0.0,
        }
    numeric = (
        "parent_gap",
        "student_gap",
        "gap_delta",
        "target_probability_advantage",
        "parent_rank",
        "student_rank",
    )
    summary: dict[str, Any] = {
        "rows": len(rows),
        "directional_movement": statistics.fmean(row["directional_movement"] for row in rows),
        "crossed": statistics.fmean(row["crossed"] for row in rows),
        "preferred_action_capture": statistics.fmean(row["captured"] for row in rows),
        "retained_parent": statistics.fmean(row["retained_parent"] for row in rows),
        "moved_third": statistics.fmean(row["moved_third"] for row in rows),
    }
    for key in numeric:
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "quantiles": _quantile_map(values),
        }
    value_advantages = [
        float(row["resolver_value_advantage"])
        for row in rows
        if row.get("resolver_value_advantage") is not None
    ]
    summary["resolver_value_advantage"] = (
        {
            "rows": len(value_advantages),
            "mean": statistics.fmean(value_advantages),
            "median": statistics.median(value_advantages),
            "quantiles": _quantile_map(value_advantages),
        }
        if value_advantages
        else {"rows": 0}
    )
    return summary


def _stratify(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[key])].append(row)
    return {
        value: _pair_summary(stratum)
        for value, stratum in sorted(values.items(), key=lambda item: item[0])
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows or samples <= 0:
        return {}
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["game_seed"])].append(row)
    clusters = []
    for seed_rows in by_seed.values():
        clusters.append(
            (
                len(seed_rows),
                sum(float(row["gap_delta"]) for row in seed_rows),
                sum(bool(row["directional_movement"]) for row in seed_rows),
                sum(bool(row["captured"]) for row in seed_rows),
            )
        )
    rng = random.Random(seed)
    distributions = {
        "mean_gap_delta": [],
        "directional_movement": [],
        "preferred_action_capture": [],
    }
    for _ in range(samples):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        count = sum(item[0] for item in chosen)
        distributions["mean_gap_delta"].append(sum(item[1] for item in chosen) / count)
        distributions["directional_movement"].append(sum(item[2] for item in chosen) / count)
        distributions["preferred_action_capture"].append(sum(item[3] for item in chosen) / count)
    return {
        key: {
            "lower": _quantile(values, 0.025),
            "upper": _quantile(values, 0.975),
        }
        for key, values in distributions.items()
    }


def _fallback_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"rows": 0}
        return {
            "rows": len(selected),
            "mean_kl": statistics.fmean(row["fallback_kl"] for row in selected),
            "action_movement": statistics.fmean(row["fallback_moved"] for row in selected),
            "mean_entropy_delta": statistics.fmean(
                row["fallback_entropy_delta"] for row in selected
            ),
        }

    return {
        **summarize(rows),
        "phase": {
            phase: summarize([row for row in rows if row["phase"] == phase])
            for phase in ("early", "mid", "late")
        },
        "seat": {
            str(seat): summarize([row for row in rows if row["seat"] == seat])
            for seat in (0, 1)
        },
    }


def _parent_gap_bin(value: float) -> str:
    if value <= -1.0:
        return "(-inf,-1]"
    if value <= -0.5:
        return "(-1,-0.5]"
    if value <= -0.25:
        return "(-0.5,-0.25]"
    if value <= -0.1:
        return "(-0.25,-0.1]"
    if value <= -0.05:
        return "(-0.1,-0.05]"
    return "(-0.05,0]"


def _rank(scores: list[float], row: int) -> int:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return order.index(row) + 1


def _argmax_row(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def _kl(target: list[float], predicted: list[float]) -> float:
    return sum(
        value * (math.log(value) - math.log(max(prediction, 1e-30)))
        for value, prediction in zip(target, predicted, strict=True)
        if value > 0.0
    )


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _quantile_map(values: list[float]) -> dict[str, float]:
    return {f"{probability:.2f}": _quantile(values, probability) for probability in QUANTILES}


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _config_payload(config: ResolverDistillationAuditConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    result = run_audit()
    confirmation = result["splits"]["confirmation"]["populations"]
    print(
        "052a result:"
        f" integrity={result['integrity_passed']}"
        f" raw_direction={confirmation['raw_resolver']['directional_movement']:.4f}"
        f" target_capture={confirmation['blended_target']['preferred_action_capture']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

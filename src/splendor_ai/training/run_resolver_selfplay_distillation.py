"""Run preregistered experiment 052: synchronous resolver self-play distillation."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Literal

import torch

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.cfr import TwoPlyResolution
from splendor_ai.encoding import ActionCodec, canonical_successor_action_key
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT_SHA256,
    _sha256_file,
)
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
    PHASES,
    VALID_TERMINATIONS,
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
    _frozen_contract_matches_049,
    _integrity,
    _seat_breakdown,
    _summarize_telemetry,
)
from splendor_ai.eval.run_shallow_operational_gate import (
    SHALLOW_CONTRACT,
    summarize_pair_rows,
)
from splendor_ai.training.dataset import LegalActionReplayDataset
from splendor_ai.training.legal_action_supervised import (
    create_legal_action_replay_dataloader,
    fit_legal_action_scorer_dataloaders_with_artifacts,
)
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.replay import (
    deserialize_state_snapshot,
    serialize_action,
    serialize_state_snapshot,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer
from splendor_ai.training.supervised import SupervisedTrainConfig


RESULT_051_SHA256 = "1a731e5edf3c63cffb2b46d8adaebba9fc50b81e490367bb0c8448ef6a885ed4"
CHECKPOINT_016_SHA256 = "876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056"
CHECKPOINT_023_SHA256 = "29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070"
Opponent = Literal["035", "shallow", "016", "023"]
BROAD_BLOCKS: tuple[tuple[Opponent, int], ...] = (
    ("shallow", 610_700),
    ("016", 610_800),
    ("023", 610_900),
)


@dataclass(frozen=True, slots=True)
class ResolverDistillationConfig:
    output_dir: Path = Path("outputs/resolver_selfplay_distillation_052")
    result_051: Path = Path("outputs/broad_public_belief_resolver_confirmation_051/result.json")
    checkpoint_035: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    checkpoint_016: Path = Path(
        "outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt"
    )
    checkpoint_023: Path = Path(
        "outputs/canonical_entity_coherent_023/"
        "canonical_entity_action_scorer_coherent_width64_best.pt"
    )
    smoke_seed: int = 610_000
    train_seed_start: int = 610_100
    train_games: int = 80
    validation_seed_start: int = 610_180
    validation_games: int = 20
    confirmation_seed_start: int = 610_300
    confirmation_games: int = 20
    direct_seed_start: int = 610_500
    direct_pairs: int = 100
    broad_pairs: int = 50
    parent_temperature: float = 1.0
    resolver_temperature: float = 0.10
    resolver_blend: float = 0.75
    resolved_training_weight: float = 1.0
    fallback_training_weight: float = 0.25
    epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    shuffle_seed: int = 52
    workers: int = 10
    bootstrap_samples: int = 20_000
    maximum_loop_adjudicated_fraction: float = 0.001
    minimum_train_rows: int = 4_000
    minimum_eval_rows: int = 1_000
    minimum_stratum_rows: int = 50
    maximum_kl_ratio: float = 0.80
    minimum_top_action_gain: float = 0.05
    minimum_action_movement: float = 0.05
    maximum_action_movement: float = 0.50
    maximum_fallback_anchor_kl: float = 0.02
    minimum_score_share: float = 0.55
    minimum_seat_score_share: float = 0.50
    device: str = "cpu"


class RecordingResolverBot(FrozenPublicBeliefResolverBot):
    """Observation-only recorder around the frozen resolver decision path."""

    def __init__(self, config: ResolverDistillationConfig, seed: int, seat: int) -> None:
        self._distill_config = config
        self._recording_seed = seed
        self._codec = ActionCodec()
        self.rows: list[dict[str, Any]] = []
        super().__init__(ResolverGameplayConfig(), seed, seat)

    def _on_resolved_decision(
        self,
        *,
        state: SplendorState,
        legal_actions: list[Action],
        all_scores: tuple[Any, ...],
        parent: Action,
        selected: Action,
        decision_index: int,
        resolution: TwoPlyResolution,
    ) -> None:
        self.rows.append(
            build_recording_row(
                self._distill_config,
                self._codec,
                self._recording_seed,
                state,
                legal_actions,
                all_scores,
                parent,
                selected,
                decision_index,
                resolution=resolution,
                fallback_reason=None,
            )
        )

    def _on_fallback_decision(
        self,
        *,
        state: SplendorState,
        legal_actions: list[Action],
        all_scores: tuple[Any, ...],
        parent: Action,
        decision_index: int,
        reason: str,
    ) -> None:
        self.rows.append(
            build_recording_row(
                self._distill_config,
                self._codec,
                self._recording_seed,
                state,
                legal_actions,
                all_scores,
                parent,
                parent,
                decision_index,
                resolution=None,
                fallback_reason=reason,
            )
        )


def build_recording_row(
    cfg: ResolverDistillationConfig,
    codec: ActionCodec,
    game_seed: int,
    state: SplendorState,
    legal_actions: list[Action],
    all_scores: tuple[Any, ...],
    parent: Action,
    selected: Action,
    decision_index: int,
    *,
    resolution: TwoPlyResolution | None,
    fallback_reason: str | None,
) -> dict[str, Any]:
    action_indices = [codec.encode(state, action) for action in legal_actions]
    if len(set(action_indices)) != len(action_indices):
        raise RuntimeError("052 requires unique legal action codec indices per state.")
    score_by_identity = {id(item.action): float(item.score) for item in all_scores}
    scores = []
    for action in legal_actions:
        value = score_by_identity.get(id(action))
        if value is None:
            matches = [float(item.score) for item in all_scores if item.action == action]
            if len(matches) != 1:
                raise RuntimeError("Could not align frozen 035 score with a legal action.")
            value = matches[0]
        scores.append(value)
    parent_probs = _softmax(scores, cfg.parent_temperature)
    target_probs = list(parent_probs)
    action_values: list[dict[str, Any]] = []
    target_available = resolution is not None
    if resolution is not None:
        root_probs = _softmax(list(resolution.action_values), cfg.resolver_temperature)
        resolver_probs = [0.0] * len(legal_actions)
        keys = [canonical_successor_action_key(state, action) for action in legal_actions]
        for root_key, root_value, root_probability in zip(
            resolution.root_actions,
            resolution.action_values,
            root_probs,
            strict=True,
        ):
            positions = [index for index, key in enumerate(keys) if key == root_key]
            if not positions:
                raise RuntimeError("Resolved canonical successor is absent from legal actions.")
            conditional_total = sum(parent_probs[index] for index in positions)
            for index in positions:
                conditional = (
                    parent_probs[index] / conditional_total
                    if conditional_total > 0.0
                    else 1.0 / len(positions)
                )
                resolver_probs[index] += root_probability * conditional
            action_values.append(
                {
                    "canonical_successor": repr(root_key),
                    "value": float(root_value),
                    "action_indices": [action_indices[index] for index in positions],
                }
            )
        target_probs = [
            cfg.resolver_blend * resolver + (1.0 - cfg.resolver_blend) * parent_prob
            for resolver, parent_prob in zip(resolver_probs, parent_probs, strict=True)
        ]
        target_probs = _normalize(target_probs)
    parent_index = codec.encode(state, parent)
    selected_index = codec.encode(state, selected)
    return {
        "format": "resolver_policy_target_v1",
        "game_seed": game_seed,
        "turn_index": state.turn_index,
        "decision_index": decision_index,
        "player_id": state.current_player,
        "phase": _phase(state.turn_index),
        "state_snapshot": serialize_state_snapshot(state),
        "action": serialize_action(selected),
        "action_index": selected_index,
        "raw_parent_action_index": parent_index,
        "target_available": target_available,
        "fallback_reason": fallback_reason,
        "training_weight": (
            cfg.resolved_training_weight if target_available else cfg.fallback_training_weight
        ),
        "resolver_policy_distribution": [
            {
                "action_index": action_index,
                "probability": float(probability),
                "parent_probability": float(parent_probability),
            }
            for action_index, probability, parent_probability in zip(
                action_indices,
                target_probs,
                parent_probs,
                strict=True,
            )
        ],
        "resolver_action_values": action_values,
    }


def _softmax(values: list[float], temperature: float) -> list[float]:
    if temperature <= 0.0 or not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Softmax requires finite values and positive temperature.")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    return _normalize(weights)


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Policy target has invalid total mass.")
    return [value / total for value in values]


def run_experiment(config: ResolverDistillationConfig | None = None) -> dict[str, Any]:
    cfg = config or ResolverDistillationConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 052 result does not match the frozen protocol.")
        return result

    smoke = _run_smoke(cfg)
    if not smoke["passed"]:
        return _finish(cfg, smoke=smoke, decision="recording_smoke_failed")

    train = _collect_split(
        cfg,
        "train",
        range(cfg.train_seed_start, cfg.train_seed_start + cfg.train_games),
        cfg.minimum_train_rows,
    )
    validation = _collect_split(
        cfg,
        "validation",
        range(cfg.validation_seed_start, cfg.validation_seed_start + cfg.validation_games),
        cfg.minimum_eval_rows,
    )
    if not train["passed"] or not validation["passed"]:
        return _finish(
            cfg,
            smoke=smoke,
            train=train,
            validation=validation,
            decision="development_collection_failed",
        )

    training = _train_student(cfg)
    confirmation = _collect_split(
        cfg,
        "confirmation",
        range(
            cfg.confirmation_seed_start,
            cfg.confirmation_seed_start + cfg.confirmation_games,
        ),
        cfg.minimum_eval_rows,
    )
    if not confirmation["passed"]:
        return _finish(
            cfg,
            smoke=smoke,
            train=train,
            validation=validation,
            training=training,
            confirmation=confirmation,
            decision="confirmation_collection_failed",
        )

    offline = _run_offline_gate(cfg)
    if not offline["passed"]:
        return _finish(
            cfg,
            smoke=smoke,
            train=train,
            validation=validation,
            training=training,
            confirmation=confirmation,
            offline=offline,
            decision="offline_distillation_gate_failed",
        )

    direct = _run_gameplay_block(
        cfg,
        "direct_vs_035",
        "035",
        cfg.direct_seed_start,
        cfg.direct_pairs,
    )
    direct_gates = _gameplay_gates(cfg, direct)
    if not all(direct_gates.values()):
        return _finish(
            cfg,
            smoke=smoke,
            train=train,
            validation=validation,
            training=training,
            confirmation=confirmation,
            offline=offline,
            direct=direct,
            direct_gates=direct_gates,
            decision="paired_parent_gate_failed",
        )

    broad: dict[str, Any] = {}
    broad_gates: dict[str, Any] = {}
    for opponent, seed_start in BROAD_BLOCKS:
        block = _run_gameplay_block(
            cfg,
            f"broad_vs_{opponent}",
            opponent,
            seed_start,
            cfg.broad_pairs,
        )
        broad[opponent] = block
        broad_gates[opponent] = _gameplay_gates(cfg, block)
    passed = all(all(gates.values()) for gates in broad_gates.values())
    return _finish(
        cfg,
        smoke=smoke,
        train=train,
        validation=validation,
        training=training,
        confirmation=confirmation,
        offline=offline,
        direct=direct,
        direct_gates=direct_gates,
        broad=broad,
        broad_gates=broad_gates,
        passed=passed,
        decision=(
            "promote_student_authorize_preregistered_generation_2"
            if passed
            else "broad_promotion_gate_failed"
        ),
    )


def _play_teacher_game(cfg: ResolverDistillationConfig, seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    bots = (RecordingResolverBot(cfg, seed, 0), RecordingResolverBot(cfg, seed, 1))
    game = play_game(
        bots[0],
        bots[1],
        seed=seed,
        max_turns=400,
        repetition_limit=4,
        no_progress_limit=60,
    )
    rows = sorted(bots[0].rows + bots[1].rows, key=lambda row: int(row["turn_index"]))
    for row in rows:
        actor = int(row["player_id"])
        winner = game.winner
        row["game_seed"] = seed
        row["winner"] = winner
        row["final_value"] = 0.0 if winner is None else 1.0 if winner == actor else -1.0
        row["final_score_margin"] = (
            game.final_scores[actor] - game.final_scores[1 - actor]
        ) / 20.0
        row["game_stalled"] = game.stalled
        row["game_timed_out"] = game.timed_out
        row["game_termination_reason"] = game.termination_reason
        row["game_model_loop_fallback_triggers"] = 0
        row["outcome_role"] = (
            "draw" if winner is None else "winner" if winner == actor else "loser"
        )
    return {
        "seed": seed,
        "game": asdict(game),
        "rows": rows,
        "telemetry": [bots[0].telemetry, bots[1].telemetry],
    }


def _run_smoke(cfg: ResolverDistillationConfig) -> dict[str, Any]:
    path = cfg.output_dir / "smoke.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("[resolver052:smoke] running repeated from-zero recording games", flush=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_play_teacher_game, cfg, cfg.smoke_seed) for _ in range(2)]
        first, second = [future.result() for future in futures]
    identical = first == second
    summary = _summarize_collection(cfg, "smoke", [first], minimum_rows=1)
    payload = {
        "first": first,
        "second": second,
        "repeat_identical": identical,
        "summary": summary,
        "passed": identical and summary["passed"],
    }
    _atomic_write(path, payload)
    return payload


def _collect_split(
    cfg: ResolverDistillationConfig,
    name: str,
    seeds: Iterable[int],
    minimum_rows: int,
) -> dict[str, Any]:
    seed_list = list(seeds)
    stage_dir = cfg.output_dir / "games" / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        seed
        for seed in seed_list
        if (stage_dir / f"{seed}.json").exists()
    }
    pending = [seed for seed in seed_list if seed not in completed]
    print(
        f"[resolver052:{name}] games={len(seed_list)} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_play_teacher_game, cfg, seed): seed for seed in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                _atomic_write(stage_dir / f"{seed}.json", future.result())
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver052:{name}] complete={len(completed) + index}/{len(seed_list)} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )
    games = [
        json.loads((stage_dir / f"{seed}.json").read_text(encoding="utf-8"))
        for seed in seed_list
    ]
    summary = _summarize_collection(cfg, name, games, minimum_rows)
    _write_jsonl(cfg.output_dir / f"{name}_targets.jsonl", (row for game in games for row in game["rows"]))
    _atomic_write(cfg.output_dir / f"{name}_summary.json", summary)
    return summary


def _summarize_collection(
    cfg: ResolverDistillationConfig,
    name: str,
    games: list[dict[str, Any]],
    minimum_rows: int,
) -> dict[str, Any]:
    rows = [row for game in games for row in game["rows"]]
    pair_like = [{"candidate_telemetry": game["telemetry"]} for game in games]
    telemetry = _summarize_telemetry(pair_like)
    game_rows = [{"games": [game["game"]]} for game in games]
    integrity = _integrity(game_rows)
    target_checks = _validate_target_rows(rows)
    privacy = _privacy_audit(cfg)
    breakdown = _row_breakdown(rows)
    strata_ok = True
    if name != "smoke":
        expected_strata = [
            "seat:0",
            "seat:1",
            "phase:early",
            "phase:mid",
            "phase:late",
            "outcome:winner",
            "outcome:loser",
        ]
        strata_ok = all(
            int(breakdown["completed_search_strata"].get(key, 0))
            >= cfg.minimum_stratum_rows
            for key in expected_strata
        )
    gates = {
        "minimum_rows": len(rows) >= minimum_rows,
        "clean_games": integrity["games"] == len(games) and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "decision_row_accounting": len(rows) == int(telemetry["decisions"]),
        "exact_branch_accounting": bool(telemetry["exact_branch_accounting"]),
        "all_continuations_resolved": bool(telemetry["all_continuations_resolved"]),
        "zero_hard_caps": int(telemetry["hard_caps"]) == 0,
        "bounded_loop_adjudication": float(telemetry["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction,
        "complete_valid_targets": all(target_checks.values()),
        "information_legal_recording": privacy["passed"],
        "minimum_completed_search_strata": strata_ok,
    }
    return {
        "name": name,
        "games": len(games),
        "rows": len(rows),
        "integrity": integrity,
        "telemetry": telemetry,
        "target_checks": target_checks,
        "privacy": privacy,
        "breakdown": breakdown,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _validate_target_rows(rows: list[dict[str, Any]]) -> dict[str, bool]:
    env = SplendorEnv()
    codec = ActionCodec()
    support_ok = probability_ok = values_ok = True
    for row in rows:
        state = deserialize_state_snapshot(row["state_snapshot"])
        legal = env.legal_actions(state)
        regenerated = [codec.encode(state, action) for action in legal]
        distribution = row.get("resolver_policy_distribution", [])
        stored = [int(item["action_index"]) for item in distribution]
        probabilities = [float(item["probability"]) for item in distribution]
        support_ok &= regenerated == stored and len(set(stored)) == len(stored)
        probability_ok &= (
            all(math.isfinite(value) and value >= 0.0 for value in probabilities)
            and abs(sum(probabilities) - 1.0) <= 1e-6
        )
        target_available = bool(row["target_available"])
        action_values = row.get("resolver_action_values", [])
        values_ok &= (
            len(action_values) == 4
            and len({item["canonical_successor"] for item in action_values}) == 4
            and all(math.isfinite(float(item["value"])) for item in action_values)
            if target_available
            else len(action_values) == 0
        )
    return {
        "exact_legal_support": support_ok,
        "finite_normalized_probabilities": probability_ok,
        "exact_resolved_value_width": values_ok,
    }


def _row_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: Counter[str] = Counter()
    for row in rows:
        if not bool(row["target_available"]):
            continue
        strata[f"seat:{row['player_id']}"] += 1
        strata[f"phase:{row['phase']}"] += 1
        if row["outcome_role"] in {"winner", "loser"}:
            strata[f"outcome:{row['outcome_role']}"] += 1
    return {
        "completed_search_rows": sum(bool(row["target_available"]) for row in rows),
        "fallback_rows": sum(not bool(row["target_available"]) for row in rows),
        "completed_search_strata": dict(sorted(strata.items())),
    }


def _privacy_audit(cfg: ResolverDistillationConfig) -> dict[str, Any]:
    scorer = _LegalActionScorer(
        cfg.checkpoint_035,
        cfg.device,
        "legal_action_attention_scorer",
    )
    codec = ActionCodec()
    env_a = SplendorEnv(seed=610_052)
    original = env_a.initial_state()
    permuted = original.copy_shallow()
    for tier in permuted.hidden_tier_decks:
        permuted.hidden_tier_decks[tier].reverse()
    original_actions = env_a.legal_actions(original)
    permuted_actions = env_a.legal_actions(permuted)
    hidden_indices_equal = [codec.encode(original, action) for action in original_actions] == [
        codec.encode(permuted, action) for action in permuted_actions
    ]
    hidden_probs_equal = _probability_maps_equal(
        codec,
        original,
        original_actions,
        _scorer_probs(scorer, original, original_actions),
        permuted,
        permuted_actions,
        _scorer_probs(scorer, permuted, permuted_actions),
    )

    blind_a = original.copy_shallow()
    blind_b = original.copy_shallow()
    blind_b.hidden_tier_decks[1][0], blind_b.hidden_tier_decks[1][1] = (
        blind_b.hidden_tier_decks[1][1],
        blind_b.hidden_tier_decks[1][0],
    )
    reserve_a = next(
        action
        for action in env_a.legal_actions(blind_a)
        if action.action_type == ActionType.RESERVE_DECK and action.tier == 1
    )
    reserve_b = next(
        action
        for action in env_a.legal_actions(blind_b)
        if action.action_type == ActionType.RESERVE_DECK and action.tier == 1
    )
    after_a = env_a.step(blind_a, reserve_a)
    env_b = SplendorEnv(seed=610_052)
    after_b = env_b.step(blind_b, reserve_b)
    actions_a = env_a.legal_actions(after_a)
    actions_b = env_b.legal_actions(after_b)
    blind_indices_equal = [codec.encode(after_a, action) for action in actions_a] == [
        codec.encode(after_b, action) for action in actions_b
    ]
    blind_probs_equal = _probability_maps_equal(
        codec,
        after_a,
        actions_a,
        _scorer_probs(scorer, after_a, actions_a),
        after_b,
        actions_b,
        _scorer_probs(scorer, after_b, actions_b),
    )
    checks = {
        "hidden_deck_legal_indices_equal": hidden_indices_equal,
        "hidden_deck_parent_distribution_equal": hidden_probs_equal,
        "opponent_blind_reserve_legal_indices_equal": blind_indices_equal,
        "opponent_blind_reserve_parent_distribution_equal": blind_probs_equal,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _probability_maps_equal(
    codec: ActionCodec,
    state_a: SplendorState,
    actions_a: list[Action],
    probabilities_a: list[float],
    state_b: SplendorState,
    actions_b: list[Action],
    probabilities_b: list[float],
) -> bool:
    first = {
        codec.encode(state_a, action): probability
        for action, probability in zip(actions_a, probabilities_a, strict=True)
    }
    second = {
        codec.encode(state_b, action): probability
        for action, probability in zip(actions_b, probabilities_b, strict=True)
    }
    return first.keys() == second.keys() and all(
        math.isclose(first[key], second[key], rel_tol=0.0, abs_tol=1e-7)
        for key in first
    )


def _train_student(cfg: ResolverDistillationConfig) -> dict[str, Any]:
    path = cfg.output_dir / "training.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    checkpoint = torch.load(cfg.checkpoint_035, map_location="cpu", weights_only=True)
    model = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(**checkpoint.get("model_config", {}))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    train_dataset = LegalActionReplayDataset(
        cfg.output_dir / "train_targets.jsonl",
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    validation_dataset = LegalActionReplayDataset(
        cfg.output_dir / "validation_targets.jsonl",
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    torch.manual_seed(cfg.shuffle_seed)
    train_loader = create_legal_action_replay_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    validation_loader = create_legal_action_replay_dataloader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    initial = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    started = time.perf_counter()
    artifacts = fit_legal_action_scorer_dataloaders_with_artifacts(
        train_loader,
        validation_loader,
        SupervisedTrainConfig(
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            value_loss_weight=0.0,
            epochs=cfg.epochs,
            device=cfg.device,
        ),
        model=model,
        root_visit_loss_weight=1.0,
        policy_loss_weight=0.0,
        log_every_batches=10,
    )
    if artifacts.best_model_state_dict is None or artifacts.best_epoch_index is None:
        raise RuntimeError("052 training did not select a validation epoch.")
    final_path = cfg.output_dir / "legal_action_attention_scorer.pt"
    best_path = cfg.output_dir / "legal_action_attention_scorer_best.pt"
    train_payload = _training_payload(cfg)
    _save_checkpoint(final_path, artifacts.model, train_payload)
    best_model = LegalActionAttentionScorerModel(artifacts.model.config)
    best_model.load_state_dict(artifacts.best_model_state_dict)
    _save_checkpoint(best_path, best_model, train_payload)
    movement = _parameter_movement(initial, artifacts.best_model_state_dict)
    payload = {
        "best_epoch": artifacts.best_epoch_index,
        "epochs": [
            {"train": asdict(item.train), "validation": asdict(item.validation)}
            for item in artifacts.metrics
        ],
        "movement": movement,
        "training_seconds": time.perf_counter() - started,
        "final_checkpoint": str(final_path),
        "best_checkpoint": str(best_path),
        "recipe": train_payload,
    }
    _atomic_write(path, payload)
    return payload


def _run_offline_gate(cfg: ResolverDistillationConfig) -> dict[str, Any]:
    path = cfg.output_dir / "offline_gate.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    best = cfg.output_dir / "legal_action_attention_scorer_best.pt"
    validation = _evaluate_distribution_file(cfg, cfg.output_dir / "validation_targets.jsonl", best)
    confirmation = _evaluate_distribution_file(cfg, cfg.output_dir / "confirmation_targets.jsonl", best)
    gates = {
        "validation_kl_ratio": validation["resolved"]["kl_ratio"] <= cfg.maximum_kl_ratio,
        "confirmation_kl_ratio": confirmation["resolved"]["kl_ratio"] <= cfg.maximum_kl_ratio,
        "confirmation_top_action_gain": confirmation["resolved"]["top_action_gain"]
        >= cfg.minimum_top_action_gain,
        "bounded_action_movement": cfg.minimum_action_movement
        <= confirmation["all"]["action_movement"]
        <= cfg.maximum_action_movement,
        "all_resolved_strata_improve": all(
            int(item["rows"]) >= cfg.minimum_stratum_rows
            and float(item["student_kl"]) < float(item["parent_kl"])
            for item in confirmation["strata"].values()
        ),
        "fallback_anchor_preserved": confirmation["fallback"]["student_kl"]
        <= cfg.maximum_fallback_anchor_kl,
        "finite_and_legal": bool(validation["finite_and_legal"])
        and bool(confirmation["finite_and_legal"]),
    }
    payload = {
        "validation": validation,
        "confirmation": confirmation,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_write(path, payload)
    return payload


def _evaluate_distribution_file(
    cfg: ResolverDistillationConfig,
    row_path: Path,
    student_checkpoint: Path,
) -> dict[str, Any]:
    parent = _LegalActionScorer(cfg.checkpoint_035, cfg.device, "legal_action_attention_scorer")
    student = _LegalActionScorer(student_checkpoint, cfg.device, "legal_action_attention_scorer")
    env = SplendorEnv()
    codec = ActionCodec()
    records: list[dict[str, Any]] = []
    finite_and_legal = True
    for row in _read_jsonl(row_path):
        state = deserialize_state_snapshot(row["state_snapshot"])
        legal = env.legal_actions(state)
        indices = [codec.encode(state, action) for action in legal]
        target_map = {
            int(item["action_index"]): float(item["probability"])
            for item in row["resolver_policy_distribution"]
        }
        target = [target_map[index] for index in indices]
        parent_probs = _scorer_probs(parent, state, legal)
        student_probs = _scorer_probs(student, state, legal)
        finite_and_legal &= all(
            math.isfinite(value) and value > 0.0
            for value in parent_probs + student_probs
        )
        target_top = max(range(len(target)), key=lambda index: (target[index], -index))
        parent_top = max(range(len(parent_probs)), key=lambda index: (parent_probs[index], -index))
        student_top = max(range(len(student_probs)), key=lambda index: (student_probs[index], -index))
        records.append(
            {
                "target_available": bool(row["target_available"]),
                "phase": row["phase"],
                "seat": int(row["player_id"]),
                "outcome": row["outcome_role"],
                "parent_kl": _kl(target, parent_probs),
                "student_kl": _kl(target, student_probs),
                "parent_agrees": parent_top == target_top,
                "student_agrees": student_top == target_top,
                "moved": student_top != parent_top,
            }
        )
    resolved = [row for row in records if row["target_available"]]
    fallback = [row for row in records if not row["target_available"]]
    strata = {
        **{f"phase:{phase}": _metric_rows([row for row in resolved if row["phase"] == phase]) for phase in PHASES},
        **{f"seat:{seat}": _metric_rows([row for row in resolved if row["seat"] == seat]) for seat in (0, 1)},
        **{
            f"outcome:{role}": _metric_rows([row for row in resolved if row["outcome"] == role])
            for role in ("winner", "loser")
        },
    }
    return {
        "rows": len(records),
        "all": _metric_rows(records),
        "resolved": _metric_rows(resolved),
        "fallback": _metric_rows(fallback),
        "strata": strata,
        "finite_and_legal": finite_and_legal,
    }


def _metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "parent_kl": math.inf,
            "student_kl": math.inf,
            "kl_ratio": math.inf,
            "parent_top_agreement": 0.0,
            "student_top_agreement": 0.0,
            "top_action_gain": 0.0,
            "action_movement": 0.0,
        }
    parent_kl = statistics.fmean(float(row["parent_kl"]) for row in rows)
    student_kl = statistics.fmean(float(row["student_kl"]) for row in rows)
    parent_agreement = statistics.fmean(bool(row["parent_agrees"]) for row in rows)
    student_agreement = statistics.fmean(bool(row["student_agrees"]) for row in rows)
    return {
        "rows": len(rows),
        "parent_kl": parent_kl,
        "student_kl": student_kl,
        "kl_ratio": student_kl / parent_kl if parent_kl > 0.0 else math.inf,
        "parent_top_agreement": parent_agreement,
        "student_top_agreement": student_agreement,
        "top_action_gain": student_agreement - parent_agreement,
        "action_movement": statistics.fmean(bool(row["moved"]) for row in rows),
    }


def _scorer_probs(scorer: Any, state: SplendorState, legal: list[Action]) -> list[float]:
    rows = scorer.score_actions(state, legal)
    by_identity = {id(row.action): float(row.score) for row in rows}
    scores = []
    for action in legal:
        value = by_identity.get(id(action))
        if value is None:
            matches = [float(row.score) for row in rows if row.action == action]
            if len(matches) != 1:
                raise RuntimeError("Could not align evaluated policy score.")
            value = matches[0]
        scores.append(value)
    return _softmax(scores, 1.0)


def _kl(target: list[float], predicted: list[float]) -> float:
    return sum(
        value * (math.log(value) - math.log(max(prediction, 1e-30)))
        for value, prediction in zip(target, predicted, strict=True)
        if value > 0.0
    )


def _run_gameplay_block(
    cfg: ResolverDistillationConfig,
    name: str,
    opponent: Opponent,
    seed_start: int,
    pairs: int,
) -> dict[str, Any]:
    path = cfg.output_dir / f"{name}.json"
    progress_path = cfg.output_dir / f"{name}.progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = {int(row["seed"]): row for row in progress.get("pairs", [])}
    seeds = list(range(seed_start, seed_start + pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(f"[resolver052:{name}] pairs={pairs} complete={len(completed)}", flush=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {
                executor.submit(_play_student_pair, cfg, opponent, seed): seed
                for seed in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                completed[seed] = future.result()
                _atomic_write(
                    progress_path,
                    {"completed": False, "pairs": [completed[key] for key in sorted(completed)]},
                )
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver052:{name}] complete={len(completed)}/{pairs} eta_seconds={eta:.0f}",
                    flush=True,
                )
    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, cfg.bootstrap_samples, seed_start)
    summary["seat_breakdown"] = _seat_breakdown(rows)
    payload = {
        "name": name,
        "opponent": opponent,
        "seed_start": seed_start,
        "pairs_count": pairs,
        "summary": summary,
        "integrity": _integrity(rows),
        "pairs": rows,
    }
    _atomic_write(path, payload)
    _atomic_write(progress_path, {"completed": True, "pairs": rows})
    return payload


def _play_student_pair(
    cfg: ResolverDistillationConfig,
    opponent: Opponent,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    games = []
    for candidate_seat in (0, 1):
        candidate = LegalActionAttentionScorerBot(
            cfg.output_dir / "legal_action_attention_scorer_best.pt",
            cfg.device,
        )
        baseline = _gameplay_opponent(cfg, opponent)
        bots = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
        games.append(
            asdict(
                play_game(
                    bots[0],
                    bots[1],
                    seed=seed,
                    max_turns=400,
                    repetition_limit=4,
                    no_progress_limit=60,
                )
            )
        )
    return {"seed": seed, "candidate_seats": [0, 1], "games": games}


def _gameplay_opponent(cfg: ResolverDistillationConfig, opponent: Opponent) -> Any:
    if opponent == "035":
        return LegalActionAttentionScorerBot(cfg.checkpoint_035, cfg.device)
    if opponent == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    if opponent == "016":
        return LegalActionScorerBot(cfg.checkpoint_016, cfg.device)
    if opponent == "023":
        return CanonicalEntityActionScorerBot(cfg.checkpoint_023, cfg.device)
    raise ValueError(opponent)


def _gameplay_gates(cfg: ResolverDistillationConfig, block: dict[str, Any]) -> dict[str, bool]:
    summary = block["summary"]
    integrity = block["integrity"]
    return {
        "exact_clean_games": integrity["games"] == int(block["pairs_count"]) * 2
        and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "score_share_at_least_55pct": float(summary["score_share"]["mean"])
        >= cfg.minimum_score_share,
        "score_share_ci95_lower_above_50pct": float(summary["score_share"]["ci95_low"])
        > 0.50,
        "positive_mean_margin": float(summary["final_score_margin"]["mean"]) > 0.0,
        "no_seat_collapse": all(
            float(item["score_share"]) >= cfg.minimum_seat_score_share
            for item in summary["seat_breakdown"].values()
        ),
    }


def _finish(cfg: ResolverDistillationConfig, *, decision: str, passed: bool = False, **stages: Any) -> dict[str, Any]:
    payload = {
        "experiment": "052_synchronous_resolver_selfplay_distillation",
        "protocol": "docs/experiments/resolver_selfplay_distillation_052.md",
        "identity": _identity(cfg),
        **stages,
        "passed": passed,
        "decision": decision,
        "gpu_rental_authorized": False,
        "generation_2_authorized": passed,
    }
    _atomic_write(cfg.output_dir / "result.json", payload)
    print(f"052 result: {'PASS' if passed else 'FAIL'} decision={decision}", flush=True)
    return payload


def _identity(cfg: ResolverDistillationConfig) -> dict[str, Any]:
    return {
        "configuration": _config_payload(cfg),
        "hashes": {
            "result_051": _sha256_file(cfg.result_051),
            "checkpoint_035": _sha256_file(cfg.checkpoint_035),
            "checkpoint_016": _sha256_file(cfg.checkpoint_016),
            "checkpoint_023": _sha256_file(cfg.checkpoint_023),
        },
        "resolver_configuration": _config_payload(ResolverGameplayConfig()),
        "expected_branches_per_search": EXPECTED_BRANCHES_PER_SEARCH,
        "shallow_contract": SHALLOW_CONTRACT,
    }


def _validate_config(cfg: ResolverDistillationConfig) -> None:
    if cfg != ResolverDistillationConfig():
        raise ValueError("Experiment 052 permits only its preregistered defaults.")
    for path in (cfg.result_051, cfg.checkpoint_035, cfg.checkpoint_016, cfg.checkpoint_023):
        if not path.exists():
            raise FileNotFoundError(path)
    result_051 = json.loads(cfg.result_051.read_text(encoding="utf-8"))
    if (
        _sha256_file(cfg.result_051) != RESULT_051_SHA256
        or not bool(result_051.get("passed"))
        or result_051.get("decision")
        != "authorize_separately_preregistered_resolver_teacher_corpus"
        or _sha256_file(cfg.checkpoint_035) != CHECKPOINT_SHA256
        or _sha256_file(cfg.checkpoint_016) != CHECKPOINT_016_SHA256
        or _sha256_file(cfg.checkpoint_023) != CHECKPOINT_023_SHA256
        or not _frozen_contract_matches_049(ResolverGameplayConfig())
    ):
        raise ValueError("Experiment 052 prerequisite or frozen identity mismatch.")


def _training_payload(cfg: ResolverDistillationConfig) -> dict[str, Any]:
    return {
        "parent_checkpoint": str(cfg.checkpoint_035),
        "parent_sha256": CHECKPOINT_SHA256,
        "target_kind": "resolver_value_derived_complete_policy_distribution",
        "hard_policy_loss_weight": 0.0,
        "value_loss_weight": 0.0,
        "policy_distribution_loss_weight": 1.0,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "shuffle_seed": cfg.shuffle_seed,
        "device": cfg.device,
    }


def _save_checkpoint(path: Path, model: LegalActionAttentionScorerModel, recipe: dict[str, Any]) -> None:
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "train_config": recipe,
        },
        path,
    )


def _parameter_movement(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, float]:
    squared_delta = 0.0
    squared_before = 0.0
    max_abs = 0.0
    for name, previous in before.items():
        current = after[name]
        delta = current.double() - previous.double()
        squared_delta += float((delta * delta).sum())
        squared_before += float((previous.double() * previous.double()).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "l2": math.sqrt(squared_delta),
        "relative_l2": math.sqrt(squared_delta / squared_before) if squared_before else 0.0,
        "max_abs": max_abs,
    }


def _phase(turn: int) -> str:
    return "early" if turn <= 19 else "mid" if turn <= 49 else "late"


def _config_payload(cfg: Any) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()

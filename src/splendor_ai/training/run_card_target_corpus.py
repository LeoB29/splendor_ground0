"""Generate the preregistered fresh target-only corpus for experiment 082."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
from typing import Any

import torch

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.card_target_executor import target_executor_action
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.card_option_q import enumerate_option_targets
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_state_snapshot
from splendor_ai.training.run_action_value_training import _spearman
from splendor_ai.training.run_card_option_q_corpus import _rollout_option_share
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


@dataclass(frozen=True, slots=True)
class CardTargetCorpusConfig:
    checkpoint: Path
    output_path: Path
    progress_path: Path
    source_seed_start: int = 785_500
    source_games: int = 600
    validation_states: int = 200
    confirmation_states: int = 200
    source_epsilon: float = 0.25
    determinizations: int = 2
    continuations_per_determinization: int = 4
    continuation_epsilon: float = 0.10
    continuation_temperature: float = 0.50
    continuation_top_k: int = 3
    max_additional_turns: int = 120
    bootstrap_samples: int = 20_000
    split_seed: int = 782_082
    workers: int = 10
    device: str = "cpu"

    @property
    def selected_states(self) -> int:
        return self.validation_states + self.confirmation_states


_WORKER_SCORER: _LegalActionScorer | None = None


def build_card_target_corpus(config: CardTargetCorpusConfig) -> dict[str, Any]:
    _validate_config(config)
    contract = _config_payload(config)
    header, completed = _load_progress(config.progress_path)
    if header is None:
        scorer = _LegalActionScorer(
            config.checkpoint, config.device, "legal_action_attention_scorer"
        )
        selected, source = _collect_source_states(config, scorer)
        header = {"contract": contract, "selected_states": selected, "source": source}
        config.progress_path.parent.mkdir(parents=True, exist_ok=True)
        config.progress_path.write_text(
            json.dumps({"header": header}) + "\n", encoding="utf-8"
        )
    elif header.get("contract") != contract:
        raise ValueError("Existing 082 corpus progress has a different contract.")
    selected = header["selected_states"]
    missing = [(index, row) for index, row in enumerate(selected) if index not in completed]
    print(
        f"[082:corpus] selected={len(selected)} completed={len(completed)} "
        f"missing={len(missing)}",
        flush=True,
    )
    if missing:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_worker,
            initargs=(str(config.checkpoint), config.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_saved_state, item, config): item[0]
                for item in missing
            }
            with config.progress_path.open("a", encoding="utf-8") as handle:
                for done, future in enumerate(as_completed(futures), 1):
                    state_index, row = future.result()
                    completed[state_index] = row
                    handle.write(json.dumps({"state_index": state_index, "row": row}) + "\n")
                    handle.flush()
                    if done % 10 == 0 or len(completed) == len(selected):
                        print(
                            f"[082:corpus] completed={len(completed)}/{len(selected)}",
                            flush=True,
                        )
    rows = [completed[index] for index in range(len(selected))]
    summary = _summarize(rows, config)
    split_counts = Counter(row["split"] for row in rows)
    phase_by_split = {
        split: Counter(row["phase"] for row in rows if row["split"] == split)
        for split in ("validation", "confirmation")
    }
    seat_by_split = {
        split: Counter(str(row["actor_seat"]) for row in rows if row["split"] == split)
        for split in ("validation", "confirmation")
    }
    exact_targets = all(
        row["target_count"] == len(row["targets"])
        and row["target_count"] >= 13
        and sum(bool(target["baseline"]) for target in row["targets"]) == 1
        and all(
            target["total_continuations"]
            == config.determinizations * config.continuations_per_determinization
            for target in row["targets"]
        )
        for row in rows
    )
    gates = {
        "exact_state_split_phase_seat_coverage": (
            len(rows) == 400
            and split_counts == {"validation": 200, "confirmation": 200}
            and all(phase_by_split[split] == {"early": 67, "mid": 67, "late": 66}
                    for split in phase_by_split)
            and all(seat_by_split[split] == {"0": 100, "1": 100}
                    for split in seat_by_split)
            and header["source"]["selected_games"] == 400
        ),
        "exact_target_determinization_coverage": exact_targets,
        "zero_illegal_null_errors": summary["errors"] == 0,
        "continuation_cutoff_fraction_safe": summary["cutoff_fraction"] <= 0.01,
        "split_half_target_rank_signal_positive": summary["split_spearman_ci95"][0] > 0.0,
    }
    payload = {
        "corpus": "card_target_counterfactual_082_v1",
        "contract": contract,
        "source": header["source"],
        "split_counts": dict(split_counts),
        "phase_counts_by_split": {
            key: dict(value) for key, value in phase_by_split.items()
        },
        "seat_counts_by_split": {
            key: dict(value) for key, value in seat_by_split.items()
        },
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "human_data_used": False,
        "states": rows,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _desired_cells() -> Counter[tuple[str, str, int]]:
    counts: Counter[tuple[str, str, int]] = Counter()
    frozen = {
        "validation": {
            ("early", 0): 34, ("early", 1): 33,
            ("mid", 0): 33, ("mid", 1): 34,
            ("late", 0): 33, ("late", 1): 33,
        },
        "confirmation": {
            ("early", 0): 33, ("early", 1): 34,
            ("mid", 0): 34, ("mid", 1): 33,
            ("late", 0): 33, ("late", 1): 33,
        },
    }
    for split, cells in frozen.items():
        for (phase, seat), count in cells.items():
            counts[(split, phase, seat)] = count
    return counts


def _collect_source_states(
    config: CardTargetCorpusConfig,
    scorer: _LegalActionScorer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    remaining = _desired_cells()
    rng = random.Random(config.split_seed + 17)
    selected: list[dict[str, Any]] = []
    clean_games = 0
    for offset in range(config.source_games):
        seed = config.source_seed_start + offset
        env = SplendorEnv(seed=seed)
        state = env.initial_state()
        candidates: dict[tuple[str, int], list[SplendorState]] = {}
        seen: Counter[tuple[object, ...]] = Counter()
        no_progress = 0
        clean = True
        while not state.terminal and state.turn_index < 400:
            signature = state_signature(state)
            seen[signature] += 1
            if seen[signature] >= 4 or no_progress >= 60:
                clean = False
                break
            cell = (_phase(state.turn_index), state.current_player)
            candidates.setdefault(cell, []).append(state)
            actions = list(
                actions_by_canonical_successor_key(
                    state, env.legal_actions(state)
                ).values()
            )
            scores = scorer.score_actions(state, actions)
            action_rng = random.Random(
                config.source_seed_start * 1_000_003 + seed * 10_007 + state.turn_index
            )
            action = (
                actions[action_rng.randrange(len(actions))]
                if action_rng.random() < config.source_epsilon
                else scores[0].action
            )
            next_state = env.step(state, action)
            no_progress = (
                0 if is_progress_transition(state, next_state, action) else no_progress + 1
            )
            state = next_state
        clean_games += int(clean and state.terminal)
        available = [
            cell for cell, count in remaining.items()
            if count > 0 and candidates.get((cell[1], cell[2]))
        ]
        if available:
            available.sort()
            split, phase, seat = available[rng.randrange(len(available))]
            states = candidates[(phase, seat)]
            chosen = states[rng.randrange(len(states))]
            selected.append(
                {
                    "source_game_seed": seed,
                    "split": split,
                    "phase": phase,
                    "actor_seat": seat,
                    "state_snapshot": serialize_state_snapshot(chosen),
                }
            )
            remaining[(split, phase, seat)] -= 1
        if len(selected) == config.selected_states:
            break
    if len(selected) != config.selected_states or any(remaining.values()):
        raise RuntimeError(
            f"Fresh source games could not fill frozen split/phase/seat cells: {dict(remaining)}"
        )
    return selected, {
        "source_games_requested": config.source_games,
        "source_games_consumed": offset + 1,
        "clean_source_games": clean_games,
        "selected_games": len(selected),
        "outcomes_stored_or_used": False,
    }


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(
        Path(checkpoint), device, "legal_action_attention_scorer"
    )


def _evaluate_saved_state(
    item: tuple[int, dict[str, Any]],
    config: CardTargetCorpusConfig,
) -> tuple[int, dict[str, Any]]:
    state_index, selected = item
    if _WORKER_SCORER is None:
        _initialize_worker(str(config.checkpoint), config.device)
    assert _WORKER_SCORER is not None
    state = deserialize_state_snapshot(selected["state_snapshot"])
    env = SplendorEnv()
    codec = ActionCodec()
    actions = list(
        actions_by_canonical_successor_key(state, env.legal_actions(state)).values()
    )
    scores = _WORKER_SCORER.score_actions(state, actions)
    targets = enumerate_option_targets(state, state.current_player)
    target_rows = []
    errors = 0
    for target in targets:
        root_action = (
            scores[0].action
            if target.card is None
            else target_executor_action(state, actions, scores, target.card)
        )
        action_index = codec.encode(state, root_action)
        first: list[float] = []
        second: list[float] = []
        terminal = 0
        for det_index in range(config.determinizations):
            det = determinize_for_observer(
                state,
                state.current_player,
                random.Random(config.split_seed + state_index * 1_000_003 + det_index),
            )
            det_actions = list(
                actions_by_canonical_successor_key(det, env.legal_actions(det)).values()
            )
            action_by_index = {codec.encode(det, action): action for action in det_actions}
            det_root = action_by_index.get(action_index)
            if det_root is None:
                errors += 1
                continue
            for continuation_index in range(config.continuations_per_determinization):
                value, complete = _rollout_option_share(
                    env,
                    _WORKER_SCORER,
                    det,
                    det_root,
                    target.key,
                    state.current_player,
                    seed=(
                        config.split_seed
                        + state_index * 10_000_019
                        + det_index * 100_003
                        + continuation_index
                    ),
                    config=config,
                )
                (first if continuation_index < 2 else second).append(value)
                terminal += int(complete)
        values = first + second
        target_rows.append(
            {
                "target_key": target.key,
                "target_source": target.source,
                "executor_action_index": action_index,
                "first_half_mean": statistics.fmean(first) if first else 0.0,
                "second_half_mean": statistics.fmean(second) if second else 0.0,
                "full_mean": statistics.fmean(values) if values else 0.0,
                "terminal_continuations": terminal,
                "total_continuations": len(values),
                "baseline": target.key == "CLEAR",
            }
        )
    return state_index, {
        **selected,
        "target_count": len(target_rows),
        "split_spearman": _spearman(
            [row["first_half_mean"] for row in target_rows],
            [row["second_half_mean"] for row in target_rows],
        ),
        "errors": errors,
        "targets": target_rows,
    }


def _summarize(
    rows: list[dict[str, Any]],
    config: CardTargetCorpusConfig,
) -> dict[str, Any]:
    spearmans = [float(row["split_spearman"]) for row in rows]
    total = sum(
        target["total_continuations"] for row in rows for target in row["targets"]
    )
    terminal = sum(
        target["terminal_continuations"] for row in rows for target in row["targets"]
    )
    return {
        "states": len(rows),
        "targets": sum(row["target_count"] for row in rows),
        "continuations": total,
        "cutoffs": total - terminal,
        "cutoff_fraction": (total - terminal) / max(total, 1),
        "errors": sum(row["errors"] for row in rows),
        "mean_split_spearman": statistics.fmean(spearmans),
        "split_spearman_ci95": _bootstrap_ci(
            spearmans, config.bootstrap_samples, config.split_seed + 901
        ),
    }


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def _load_progress(
    path: Path,
) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    if not path.exists():
        return None, {}
    header = None
    completed: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "header" in payload:
                header = payload["header"]
            elif "state_index" in payload:
                completed[int(payload["state_index"])] = payload["row"]
    return header, completed


def _phase(turn: int) -> str:
    return "early" if turn < 20 else "mid" if turn < 40 else "late"


def _config_payload(config: CardTargetCorpusConfig) -> dict[str, Any]:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    payload["selected_states"] = config.selected_states
    return payload


def _validate_config(config: CardTargetCorpusConfig) -> None:
    if (
        config.source_seed_start,
        config.source_games,
        config.validation_states,
        config.confirmation_states,
        config.split_seed,
    ) != (785_500, 600, 200, 200, 782_082):
        raise ValueError("082 fresh source and split contract is frozen.")
    if (config.determinizations, config.continuations_per_determinization) != (2, 4):
        raise ValueError("082 requires two determinizations and four continuations each.")
    if config.workers <= 0:
        raise ValueError("workers must be positive.")

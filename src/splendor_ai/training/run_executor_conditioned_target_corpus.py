"""Generate experiment 083's fresh target-only evidence corpus."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from splendor_ai.training.run_card_target_corpus import (
    _collect_source_states,
    _evaluate_saved_state,
    _initialize_worker,
    _load_progress,
    _summarize,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


@dataclass(frozen=True, slots=True)
class ExecutorConditionedCorpusConfig:
    checkpoint: Path
    output_path: Path
    progress_path: Path
    source_seed_start: int = 789_500
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
    split_seed: int = 783_083
    workers: int = 10
    device: str = "cpu"

    @property
    def selected_states(self) -> int:
        return self.validation_states + self.confirmation_states


def build_executor_conditioned_corpus(
    config: ExecutorConditionedCorpusConfig,
) -> dict[str, Any]:
    _validate_corpus_config(config)
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
        raise ValueError("Existing 083 corpus progress has a different contract.")
    selected = header["selected_states"]
    missing = [(index, row) for index, row in enumerate(selected) if index not in completed]
    print(
        f"[083:corpus] selected={len(selected)} completed={len(completed)} "
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
                            f"[083:corpus] completed={len(completed)}/{len(selected)}",
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
            and all(
                phase_by_split[split] == {"early": 67, "mid": 67, "late": 66}
                for split in phase_by_split
            )
            and all(
                seat_by_split[split] == {"0": 100, "1": 100}
                for split in seat_by_split
            )
            and header["source"]["selected_games"] == 400
        ),
        "exact_target_determinization_coverage": exact_targets,
        "zero_illegal_null_errors": summary["errors"] == 0,
        "continuation_cutoff_fraction_safe": summary["cutoff_fraction"] <= 0.01,
        "split_half_target_rank_signal_positive": (
            summary["split_spearman_ci95"][0] > 0.0
        ),
    }
    payload = {
        "corpus": "executor_conditioned_target_counterfactual_083_v1",
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


def _config_payload(config: ExecutorConditionedCorpusConfig) -> dict[str, Any]:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    payload["selected_states"] = config.selected_states
    return payload


def _validate_corpus_config(config: ExecutorConditionedCorpusConfig) -> None:
    if (
        config.source_seed_start,
        config.source_games,
        config.validation_states,
        config.confirmation_states,
        config.split_seed,
    ) != (789_500, 600, 200, 200, 783_083):
        raise ValueError("083 fresh source and split contract is frozen.")
    if (config.determinizations, config.continuations_per_determinization) != (2, 4):
        raise ValueError("083 requires two determinizations and four continuations each.")
    if config.workers <= 0:
        raise ValueError("workers must be positive.")

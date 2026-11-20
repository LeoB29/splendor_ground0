"""Diagnostics-only crash recovery for the frozen programme-101 run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .clean_continuation import PROTOCOL_SHA256 as CONTINUATION_PROTOCOL_SHA256
from .network import load_checkpoint
from .scaled_loop import _diagnostics, _trajectory, training_validity


@dataclass(frozen=True, slots=True)
class CleanContinuationRecoveryConfig:
    output_dir: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


RECOVERY_PROTOCOL_SHA256 = "2acf976b2b3b3baee45fadad294300327205df70f13b3cc00e3393aa31afb3f9"
EXPECTED_HASHES = {
    "progress": "4e70ff2d2c83b4db90cbf454c4243228693493cce44ada3ec61a2e6150013417",
    "generation_4_checkpoint": "0cd7782714ca0b2c081847af92dc731711ecd15b14c72bb9fa25efb039d35669",
    "generation_5_checkpoint": "201e34128c9588722ce78edc85e8672675b1075dc0696651484718ddd7fa2444",
    "sanitized_generation_3": "e39440b4eeeba28b9c33e0c0fa222c5d89afb81d9285530e1829748f206b06ce",
    "sanitized_generation_4": "7e15847d2cdf51ecd467a2bb6f38f13e1901bd7337971bbe8f4817ace1dec244",
    "training_generation_4": "0c77bf5e0c89d1d13ba08e98e008288a40928a1459c1cc1f40303da2bc9eb7f1",
    "training_generation_5": "a61c4acaa92d7f8c0f15e8a2d60eebd7fdd3be3ac25176601ecbf2f45045d2a3",
    "raw_generation_5_diagnostic": "10476b0dc5743bc09a12317769729e78b19626f3f10c6ddbc1b60bbb42cae069",
    "partial_searched_generation_5_diagnostic": "b0811df44ad6ce22e2c95292238309596b10d83979bdc8418cf008fee5d3b482",
}


def arena_progress_valid(
    payload: dict[str, Any],
    *,
    expected_name: str,
    expected_rows: int,
    expected_completed: bool,
) -> bool:
    identity = payload.get("identity") or {}
    rows = payload.get("rows") or []
    schedule = [
        (int(identity.get("seed_start", -1)) + offset, seat)
        for offset in range(int(identity.get("paired_seeds", 0)))
        for seat in (0, 1)
    ]
    observed = [(int(row["seed"]), int(row["candidate_seat"])) for row in rows]
    return bool(
        identity.get("name") == expected_name
        and identity.get("seed_start") == (
            1_780_000 if "raw_child_vs_raw_parent" in expected_name else 1_780_100
        )
        and identity.get("paired_seeds") == 8
        and identity.get("max_turns") == 200
        and payload.get("target_games") == 16
        and len(rows) == expected_rows
        and payload.get("completed_games") == expected_rows
        and payload.get("completed") is expected_completed
        and observed == schedule[:expected_rows]
    )


def run_clean_continuation_recovery(
    config: CleanContinuationRecoveryConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    paths = _paths(output)
    _verify_hashes(paths)
    if paths["result"].exists():
        raise ValueError("A sealed 101 result already exists; recovery is not applicable.")
    state = json.loads(paths["progress"].read_text(encoding="utf-8"))
    training_5 = json.loads(
        paths["training_generation_5"].read_text(encoding="utf-8")
    )
    _verify_structure(state, training_5, paths, config)
    precrash_elapsed = float(state.get("elapsed_seconds", 0.0))
    state["crash_recovery"] = {
        "status": "running",
        "protocol_sha256": RECOVERY_PROTOCOL_SHA256,
        "input_hashes": dict(EXPECTED_HASHES),
        "started_from_stage": "diagnostics_generation_5",
        "precrash_snapshot_elapsed_seconds": precrash_elapsed,
    }
    state["active_stage"] = "diagnostics_generation_5_crash_recovery"
    _write_json(paths["progress"], state)

    diagnostics = _diagnostics(
        output=output,
        config=config,  # type: ignore[arg-type]
        iteration=2,
        parent=paths["generation_4_checkpoint"],
        child=paths["generation_5_checkpoint"],
    )
    validity = training_validity(training_5, expected_parent=4)
    validity["checks"]["checkpoint_roundtrip_generation"] = True
    validity["valid"] = bool(validity["valid"])
    sanitation_4 = state["sanitation"]["generation_4"]
    generation_5 = {
        "iteration": 3,
        "parent_generation": 4,
        "parent_checkpoint": str(paths["generation_4_checkpoint"]),
        "parent_checkpoint_sha256": EXPECTED_HASHES["generation_4_checkpoint"],
        "selfplay": sanitation_4["summary"],
        "selfplay_validity": {
            "valid": True,
            "checks": sanitation_4["summary"]["checks"],
        },
        "replay_path": str(paths["sanitized_generation_4"]),
        "replay_sha256": EXPECTED_HASHES["sanitized_generation_4"],
        "replay_window": [
            str(paths["sanitized_generation_3"]),
            str(paths["sanitized_generation_4"]),
        ],
        "child_generation": 5,
        "child_checkpoint": str(paths["generation_5_checkpoint"]),
        "child_checkpoint_sha256": EXPECTED_HASHES["generation_5_checkpoint"],
        "training": training_5,
        "training_validity": validity,
        "diagnostics": diagnostics,
    }
    state["continued_generations"].append(generation_5)
    all_generations = [
        state["inherited_generation_3"],
        *state["continued_generations"],
    ]
    recovery_elapsed = time.perf_counter() - started
    state["status"] = "complete"
    state["decision"] = "clean_continuation_complete_review_trajectory"
    state["active_stage"] = "complete"
    state["final_sandbox_checkpoint"] = str(paths["generation_5_checkpoint"])
    state["final_sandbox_checkpoint_sha256"] = EXPECTED_HASHES[
        "generation_5_checkpoint"
    ]
    state["trajectory"] = _trajectory(all_generations)
    state["crash_recovery"].update(
        status="complete",
        recovery_elapsed_seconds=recovery_elapsed,
        note=(
            "Total elapsed excludes unsnapshotted time between the last main "
            "progress write and the host crash."
        ),
    )
    state["elapsed_seconds"] = precrash_elapsed + recovery_elapsed
    _write_json(paths["progress"], state)
    _write_json(paths["result"], state)
    return state


def _paths(output: Path) -> dict[str, Path]:
    return {
        "progress": output / "progress.json",
        "result": output / "result.json",
        "generation_4_checkpoint": output / "checkpoints" / "generation_004.pt",
        "generation_5_checkpoint": output / "checkpoints" / "generation_005.pt",
        "sanitized_generation_3": output / "selfplay" / "generation_003_sanitized.jsonl",
        "sanitized_generation_4": output / "selfplay" / "generation_004_sanitized.jsonl",
        "training_generation_4": output / "training" / "generation_004.json",
        "training_generation_5": output / "training" / "generation_005.json",
        "raw_generation_5_diagnostic": output
        / "diagnostics"
        / "generation_005"
        / "raw_child_vs_raw_parent.json",
        "partial_searched_generation_5_diagnostic": output
        / "diagnostics"
        / "generation_005"
        / "searched_child_vs_searched_parent.json",
        "child_search_generation_5_diagnostic": output
        / "diagnostics"
        / "generation_005"
        / "child_search_vs_child_raw.json",
        "generation_4_selfplay_progress": output
        / "selfplay"
        / "generation_004_raw.progress.json",
    }


def _verify_hashes(paths: dict[str, Path]) -> None:
    for name, expected in EXPECTED_HASHES.items():
        actual = _sha256(paths[name])
        if actual != expected:
            raise ValueError(f"Frozen 101 recovery hash mismatch for {name}: {actual}")


def _verify_structure(
    state: dict[str, Any],
    training_5: dict[str, Any],
    paths: dict[str, Path],
    config: CleanContinuationRecoveryConfig,
) -> None:
    if state.get("protocol_sha256") != CONTINUATION_PROTOCOL_SHA256:
        raise ValueError("Progress has the wrong 101 protocol identity.")
    if state.get("status") != "running" or state.get("active_stage") != "diagnostics_generation_5":
        raise ValueError("Progress is not at the frozen recovery stage.")
    continued = state.get("continued_generations") or []
    if len(continued) != 1 or continued[0].get("child_checkpoint_sha256") != EXPECTED_HASHES["generation_4_checkpoint"]:
        raise ValueError("Progress lacks the exact completed generation-4 record.")
    sanitation = state.get("sanitation") or {}
    if sanitation.get("generation_3", {}).get("sanitized_sha256") != EXPECTED_HASHES["sanitized_generation_3"]:
        raise ValueError("Progress has the wrong sanitized generation-3 identity.")
    if sanitation.get("generation_4", {}).get("sanitized_sha256") != EXPECTED_HASHES["sanitized_generation_4"]:
        raise ValueError("Progress has the wrong sanitized generation-4 identity.")
    if not sanitation["generation_4"]["summary"].get("valid"):
        raise ValueError("Sanitized generation-4 replay was not valid.")
    model_4, identity_4 = load_checkpoint(paths["generation_4_checkpoint"], device=config.device)
    del model_4
    model_5, identity_5 = load_checkpoint(paths["generation_5_checkpoint"], device=config.device)
    del model_5
    if identity_4.generation != 4 or identity_5.generation != 5:
        raise ValueError("Recovery checkpoints have the wrong generations.")
    if identity_5.metadata.get("parent_generation") != 4:
        raise ValueError("Generation 5 has the wrong parent generation.")
    if not training_validity(training_5, expected_parent=4)["valid"]:
        raise ValueError("Generation-5 training record is invalid.")
    raw = json.loads(paths["raw_generation_5_diagnostic"].read_text(encoding="utf-8"))
    searched = json.loads(
        paths["partial_searched_generation_5_diagnostic"].read_text(encoding="utf-8")
    )
    if not arena_progress_valid(
        raw,
        expected_name="generation_005_raw_child_vs_raw_parent",
        expected_rows=16,
        expected_completed=True,
    ):
        raise ValueError("Completed generation-5 raw diagnostic is invalid.")
    if not arena_progress_valid(
        searched,
        expected_name="generation_005_searched_child_vs_searched_parent",
        expected_rows=3,
        expected_completed=False,
    ):
        raise ValueError("Partial generation-5 searched diagnostic is invalid.")
    if paths["child_search_generation_5_diagnostic"].exists():
        raise ValueError("Unexpected generation-5 search-vs-raw progress exists.")
    selfplay = json.loads(
        paths["generation_4_selfplay_progress"].read_text(encoding="utf-8")
    )
    if selfplay.get("completed_games") != 256 or any(
        game.get("termination_reason") != "completed" for game in selfplay.get("games", [])
    ):
        raise ValueError("Generation-4 self-play did not complete cleanly.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

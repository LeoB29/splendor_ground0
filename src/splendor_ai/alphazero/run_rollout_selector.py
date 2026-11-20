"""154: fresh, prospectively locked terminal-rollout move selection."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import run_public_feature_advantage as parent

old = parent.old
OUTPUT = Path("outputs/rollout_selector_154")
PROTOCOL = Path("docs/engineering/rollout_selector_154.md")
VERIFIER = Path("scripts/analysis/verify_rollout_selector_154.py")
PARENT_BINDING = "0627752a8399ac74484584025135282dcbff41e376122a362a8cf32218554d3f"
PARENT_AUDIT = "128a026b4ce8df849d64b8673b12ac32f02c26f7a789a4262a77ced392f8e7c4"
CONFIG = {
    "states": 768, "cells": 6, "quota": 128, "source_limit": 4096,
    "source_seed": 1_540_000_000, "discovery_seed": 1_541_000_000,
    "evaluation_seed": 1_542_000_000, "bootstrap_seed": 1_543_000_000,
    "probe_source_seed": 1_549_000_000, "probe_sources": 2,
    "probe_discovery_seed": 1_549_100_000, "probe_evaluation_seed": 1_549_200_000,
    "privacy_seed": 1_549_300_000,
    "discovery_worlds": 32, "evaluation_worlds": 128, "probe_worlds": 4,
    "probe_states": 6, "workers": 8, "source_batch": 8,
    "discovery_cap": 98304, "evaluation_cap": 196608, "main_cap": 294912,
    "bootstrap_replicates": 50000, "bootstrap_chunk": 1000,
    "minimum_gain": .02, "maximum_half_width": .02,
    "statistic_tolerance": 1e-12, "source_policy": "exact_greedy035",
    "position_tag": "154-position-v1", "candidate_split": "154-main",
    "training": False, "arena": False, "promotion": False, "automatic_successor": False,
}


def parent_inputs():
    assert parent.check_identity() == PARENT_BINDING
    assert old.file_hash(parent.OUTPUT/"final_audit.json") == PARENT_AUDIT
    final = old.read_json(parent.OUTPUT/"final_status.json")
    audit = old.read_json(parent.OUTPUT/"final_audit.json")
    assert final["complete"] and audit["passed"] and final["audit_sha256"] == PARENT_AUDIT
    assert final["result_sha256"] == audit["result_sha256"] == old.file_hash(parent.OUTPUT/"result.json")
    paths = [parent.OUTPUT/n for n in ("manifest.json", "final_audit.json", "final_status.json", "result.json")]
    paths += [old.OUTPUT/"probe_panel.json", Path("scripts/analysis/verify_afterstate_advantage_149.py")]
    return {"binding": PARENT_BINDING, "files": {p.as_posix(): old.file_hash(p) for p in paths}}


def identity():
    files = [Path(__file__).relative_to(Path.cwd()), PROTOCOL, VERIFIER,
             Path("run_rollout_selector_154.py"), Path("tests/test_rollout_selector_154.py")]
    return {"format": "rollout_selector_154_v1", "config": CONFIG, "parent": parent_inputs(),
            "dependencies": {p.as_posix(): old.file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def check_identity():
    saved = old.read_json(OUTPUT/"manifest.json")
    if saved != identity():
        raise ValueError("154 frozen identity changed")
    return old.digest(saved)


def progress(stage, **values):
    row = {"time": old.now(), "stage": stage, **values}
    old.atomic_write(OUTPUT/"progress.json", row)
    print(row, flush=True)


def save_identical_or_new(path, value):
    """Only recover a completed deterministic stage; never replace evidence."""
    if path.exists():
        assert old.read_json(path) == value
    else:
        old.save_new(path, value)


def source_path(index, probe=False):
    return OUTPUT/("probe_sources" if probe else "sources")/f"game_{index:04d}.json"


def source_task(index, probe=False):
    assert 0 <= index < CONFIG["probe_sources" if probe else "source_limit"]
    path = source_path(index, probe)
    seed = CONFIG["probe_source_seed" if probe else "source_seed"]+index
    if not path.exists():
        wall, cpu = time.perf_counter(), time.process_time()
        record = old.trajectory(old.SplendorEnv(seed=seed).initial_state(), include_trace=True)
        old.save_new(path, {"seed": seed, "index": index, **record,
                           "timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}})
    record = old.read_json(path)
    assert record["seed"] == seed and record["index"] == index
    return {"index": index, "sha256": old.file_hash(path), "reason": record["reason"], "turns": record["turns"], "timing": record["timing"]}


def select_panel(records):
    assigned, cells, positions, filtering = [[] for _ in range(6)], {}, {}, []
    for record in records:
        i = len(filtering)
        assert record["index"] == i and record["seed"] == CONFIG["source_seed"]+i
        if record["reason"] not in old.VALID_NATURAL_TERMINATIONS:
            raise ValueError("Non-natural source: no replacement")
        states = old.reconstruct_source(record)
        eligible = [[s for s in states if old.eligible(s, c)] for c in range(6)]
        cells[i] = [c for c in range(6) if eligible[c]]
        positions[i] = {c: min(eligible[c], key=lambda s: old.digest([CONFIG["position_tag"], i, s.turn_index])) for c in cells[i]}
        filtering.append({"source_index": i, "eligible_by_cell": [len(v) for v in eligible],
                          "sha256": old.file_hash(source_path(i))})
        old.place_source(i, cells, assigned, CONFIG["quota"])
        if len(filtering) % 128 == 0:
            progress("source_matching", examined=len(filtering), counts=[len(v) for v in assigned])
        if all(len(v) == CONFIG["quota"] for v in assigned):
            break
    if [len(v) for v in assigned] != [CONFIG["quota"]]*6:
        raise ValueError("Frozen source schedule exhausted")
    choices = {i: c for c, indices in enumerate(assigned) for i in indices}
    assert len(choices) == CONFIG["states"]
    samples = []
    for i, c in sorted(choices.items()):
        state = positions[i][c]
        samples.append({"index": len(samples), "source_index": i, "source_seed": CONFIG["source_seed"]+i,
                       "cell": c, "actor": state.current_player, "turn": state.turn_index,
                       "snapshot": old.serialize_state_snapshot(state), **old.candidates(state, CONFIG["candidate_split"], i)})
    return {"samples": samples, "filtering": filtering, "counts": [CONFIG["quota"]]*6}


def generate_panel():
    attempted = {}
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=old.initialize_worker) as pool:
        def records():
            for start in range(0, CONFIG["source_limit"], CONFIG["source_batch"]):
                batch = list(pool.map(source_task, range(start, min(start+CONFIG["source_batch"], CONFIG["source_limit"]))))
                attempted.update((str(r["index"]), r) for r in batch)
                if any(r["reason"] not in old.VALID_NATURAL_TERMINATIONS for r in batch):
                    raise ValueError("Non-natural source, including prefetched suffix")
                for r in batch:
                    yield old.read_json(source_path(r["index"]))
        panel = select_panel(records())
    save_identical_or_new(OUTPUT/"sources_lock.json", {"files": attempted, "selected_prefix": len(panel["filtering"]),
                 "unused_prefetched": len(attempted)-len(panel["filtering"])})
    return panel


def stage_spec(stage):
    if stage not in ("discovery", "evaluation", "probe_discovery", "probe_evaluation"):
        raise ValueError("Unknown stage")
    probe = stage.startswith("probe_")
    worlds = CONFIG["probe_worlds"] if probe else CONFIG[f"{stage}_worlds"]
    return CONFIG[f"{stage}_seed"], worlds, CONFIG["probe_states"] if probe else CONFIG["states"]


def world_seed(stage, index, world):
    start, count, states = stage_spec(stage)
    if not 0 <= index < states or not 0 <= world < count:
        raise ValueError("World outside frozen schedule")
    return start+1000*index+world


def action_indices(stage, sample, choice=None):
    stage_spec(stage)
    if stage.endswith("discovery"):
        return list(range(len(sample["actions"])))
    if not isinstance(choice, int) or not 0 <= choice < len(sample["actions"]):
        raise ValueError("Evaluation requires a valid locked choice")
    return [0] if choice == 0 else [0, choice]


def record_info(path):
    record = old.read_json(path)
    outcomes = [o for world in record["worlds"] for o in world["outcomes"]]
    return {"index": record["index"], "sha256": old.file_hash(path),
            "natural": all(o["reason"] in old.VALID_NATURAL_TERMINATIONS and o["wdl"] is not None for o in outcomes),
            "continuations": len(outcomes), "moves": sum(o["turns"] for o in outcomes),
            "reasons": dict(Counter(o["reason"] for o in outcomes)), "timing": record["timing"]}


def collect_task(stage, sample, choice, upstream):
    path = OUTPUT/"targets"/stage/f"state_{sample['index']:04d}.json"
    indices = action_indices(stage, sample, choice)
    _, count, _ = stage_spec(stage)
    if not path.exists():
        wall, cpu = time.perf_counter(), time.process_time()
        state = old.deserialize_state_snapshot(sample["snapshot"])
        worlds = []
        for w in range(count):
            seed = world_seed(stage, sample["index"], w)
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            outcomes = []
            for j in indices:
                result = old.trajectory(world, old.decode_action(sample["actions"][j]))
                result["wdl"] = old.outcome_value(result["winner"], state.current_player) if result["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                outcomes.append(result)
            worlds.append({"seed": seed, "world_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": outcomes})
        old.save_new(path, {"stage": stage, "index": sample["index"], "sample_sha256": old.digest(sample),
                     "upstream_sha256": upstream, "action_indices": indices, "worlds": worlds,
                     "timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}})
    saved = old.read_json(path)
    assert saved["stage"] == stage and saved["index"] == sample["index"] and saved["sample_sha256"] == old.digest(sample)
    assert saved["upstream_sha256"] == upstream and saved["action_indices"] == indices and len(saved["worlds"]) == count
    return record_info(path)


def collect(stage, samples, choices, upstream):
    _, count, expected = stage_spec(stage)
    assert len(samples) == expected and [s["index"] for s in samples] == list(range(expected))
    budget = sum(len(action_indices(stage, s, choices[i] if choices else None))*count for i, s in enumerate(samples))
    cap = expected*(4 if stage.endswith("discovery") else 2)*count
    assert budget <= cap
    files = {}
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=old.initialize_worker) as pool:
        jobs = [pool.submit(collect_task, stage, s, choices[i] if choices else None, upstream) for i, s in enumerate(samples)]
        for i, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            files[str(row["index"])] = row
            if i % 16 == 0 or i == len(jobs):
                progress(stage, states=i, total=len(jobs))
    assert sum(r["continuations"] for r in files.values()) == budget
    old.save_new(OUTPUT/f"{stage}_targets_lock.json", {"files": files, "all_natural": all(r["natural"] for r in files.values()), "upstream_sha256": upstream})
    if not all(r["natural"] for r in files.values()):
        raise ValueError("Invalid continuation: no replacement or exclusions")


def read_records(stage, samples, upstream):
    lock = old.read_json(OUTPUT/f"{stage}_targets_lock.json")
    assert lock["all_natural"] and lock["upstream_sha256"] == upstream
    assert set(lock["files"]) == {str(s["index"]) for s in samples}
    records = []
    for sample in samples:
        path = OUTPUT/"targets"/stage/f"state_{sample['index']:04d}.json"
        assert old.file_hash(path) == lock["files"][str(sample["index"])]["sha256"]
        saved = old.read_json(path)
        assert saved["upstream_sha256"] == upstream and saved["sample_sha256"] == old.digest(sample)
        records.append(saved)
    return records


def select_choices(samples, records):
    rows = []
    for sample, record in zip(samples, records):
        assert record["action_indices"] == list(range(len(sample["actions"])))
        sums = [sum(w["outcomes"][j]["wdl"] for w in record["worlds"]) for j in record["action_indices"]]
        q = [v/len(record["worlds"]) for v in sums]
        choice = max(range(len(q)), key=lambda j: q[j])
        rows.append({"index": sample["index"], "sample_sha256": old.digest(sample), "choice": choice,
                     "action": sample["actions"][choice], "key": sample["keys"][choice], "discovery_q": q})
    return rows


def per_state(samples, choices, evaluation):
    rows = []
    for sample, selected, record in zip(samples, choices, evaluation):
        choice = selected["choice"]
        assert record["action_indices"] == ([0] if choice == 0 else [0, choice])
        q = [sum(w["outcomes"][j]["wdl"] for w in record["worlds"])/len(record["worlds"]) for j in range(len(record["action_indices"]))]
        rows.append({"index": sample["index"], "source_seed": sample["source_seed"], "cell": sample["cell"],
                     "choice": choice, "candidate_count": len(sample["actions"]), "baseline_q": q[0], "selected_q": q[-1],
                     "evaluation_gain": q[-1]-q[0], "discovery_gain": selected["discovery_q"][choice]-selected["discovery_q"][0]})
    return rows


def statistics(rows):
    assert len({r["source_seed"] for r in rows}) == len(rows)
    arrays = [np.asarray([r["evaluation_gain"] for r in rows if r["cell"] == c], dtype=float) for c in range(6)]
    assert len({len(a) for a in arrays}) == 1 and len(arrays[0]) > 0
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = []
    for _ in range(CONFIG["bootstrap_replicates"]//CONFIG["bootstrap_chunk"]):
        draws = sum(a[rng.integers(len(a), size=(CONFIG["bootstrap_chunk"], len(a)))].mean(-1) for a in arrays)/6
        boot.append(draws)
    interval = np.quantile(np.concatenate(boot), [.025, .975]).tolist()
    gain = sum(r["evaluation_gain"] for r in rows)/len(rows)
    discovery = sum(r["discovery_gain"] for r in rows)/len(rows)
    half = (interval[1]-interval[0])/2
    gates = {"material_gain": gain >= .02, "positive_lower_bound": interval[0] > 0, "precision": half <= .02}
    def summary(part):
        return {"states": len(part), "gain": sum(r["evaluation_gain"] for r in part)/len(part),
                "discovery_gain": sum(r["discovery_gain"] for r in part)/len(part),
                "switch_fraction": sum(r["choice"] != 0 for r in part)/len(part)}
    return {"states": len(rows), "gain": gain, "ci95": interval, "half_width": half,
            "apparent_discovery_gain": discovery, "discovery_minus_evaluation": discovery-gain,
            "switch_fraction": sum(r["choice"] != 0 for r in rows)/len(rows),
            "baseline_only_states": sum(r["candidate_count"] == 1 for r in rows),
            "coincident_choices": sum(r["choice"] == 0 for r in rows),
            "cells": {str(c): summary([r for r in rows if r["cell"] == c]) for c in range(6)},
            "phases": {str(c): summary([r for r in rows if r["cell"]//2 == c]) for c in range(3)},
            "seats": {str(c): summary([r for r in rows if r["cell"]%2 == c]) for c in range(2)},
            "statistical_gates": gates, "statistical_pass": all(gates.values()),
            "conditional_move_evidence_only": True, "arena": False, "training": False}


def audit_stage(stage):
    path = OUTPUT/f"{stage}_audit.json"
    if not path.exists():
        subprocess.run([sys.executable, str(VERIFIER), stage], check=True)
    result = old.read_json(path)
    assert result["passed"] and result["manifest"] == check_identity()
    return result


def prepare():
    old.save_new(OUTPUT/"manifest.json", identity())
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=old.initialize_worker) as pool:
        sources = list(pool.map(source_task, range(CONFIG["probe_sources"]), [True]*CONFIG["probe_sources"]))
    assert all(r["reason"] in old.VALID_NATURAL_TERMINATIONS for r in sources)
    old.save_new(OUTPUT/"probe_sources_lock.json", {"files": {str(r["index"]): r for r in sources}})
    panel = old.read_json(old.OUTPUT/"probe_panel.json")
    old.save_new(OUTPUT/"probe_panel.json", panel)
    collect("probe_discovery", panel["samples"], None, old.file_hash(OUTPUT/"probe_panel.json"))
    records = read_records("probe_discovery", panel["samples"], old.file_hash(OUTPUT/"probe_panel.json"))
    rows = select_choices(panel["samples"], records)
    old.save_new(OUTPUT/"probe_selection_lock.json", {"rows": rows, "targets_sha256": old.file_hash(OUTPUT/"probe_discovery_targets_lock.json")})
    collect("probe_evaluation", panel["samples"], [r["choice"] for r in rows], old.file_hash(OUTPUT/"probe_selection_lock.json"))
    audit = audit_stage("probe")
    infos = [r for stage in ("probe_discovery", "probe_evaluation") for r in old.read_json(OUTPUT/f"{stage}_targets_lock.json")["files"].values()]
    count = sum(r["continuations"] for r in infos)
    cpu_rate = sum(r["timing"]["cpu"] for r in infos)/count
    wall_rate = sum(r["timing"]["wall"] for r in infos)/count
    source_cpu = sum(r["timing"]["cpu"] for r in sources)/len(sources)
    estimate = {"outcome_blind": True, "probe_continuations": count,
        "cpu_seconds_per_continuation": cpu_rate, "task_wall_seconds_per_continuation": wall_rate,
        "collection_projection_seconds_cpu": cpu_rate*CONFIG["main_cap"]/CONFIG["workers"],
        "collection_projection_seconds_wall": wall_rate*CONFIG["main_cap"]/CONFIG["workers"],
        "source_ceiling_seconds_cpu": source_cpu*CONFIG["source_limit"]/CONFIG["workers"],
        "audit_projection_seconds_cpu": audit["continuation_cpu_seconds"]/count*CONFIG["main_cap"]/CONFIG["workers"],
        "caveat": "Small probes; fixed population/budgets unchanged. Startup, source matching, hashing and contention add cost."}
    old.save_new(OUTPUT/"timing_probe.json", estimate)
    old.save_new(OUTPUT/"preflight.json", {"passed": True, "manifest": check_identity(), "time": old.now(),
                 "probe_audit_sha256": old.file_hash(OUTPUT/"probe_audit.json"), "timing_sha256": old.file_hash(OUTPUT/"timing_probe.json")})
    progress("preflight_passed", **estimate)


def run(resume=False):
    if resume and list(OUTPUT.glob("failure_*.json")):
        raise ValueError("Preserved failure requires review, not automatic resume")
    binding = check_identity()
    preflight = old.read_json(OUTPUT/"preflight.json")
    assert preflight["passed"] and preflight["manifest"] == binding
    for field, name in (("probe_audit_sha256", "probe_audit.json"), ("timing_sha256", "timing_probe.json")):
        assert preflight[field] == old.file_hash(OUTPUT/name)
    if (OUTPUT/"launch.json").exists():
        if not resume:
            raise ValueError("Existing launch; explicit identity-matched resume required")
        launch = old.read_json(OUTPUT/"launch.json")
        assert launch["manifest"] == binding and launch["preflight_sha256"] == old.file_hash(OUTPUT/"preflight.json")
    else:
        old.save_new(OUTPUT/"launch.json", {"manifest": binding, "time": old.now(), "preflight_sha256": old.file_hash(OUTPUT/"preflight.json")})
    if (OUTPUT/"final_status.json").exists():
        raise ValueError("Experiment already closed")
    old.initialize_worker()
    if not (OUTPUT/"panel_lock.json").exists():
        panel = generate_panel()
        save_identical_or_new(OUTPUT/"panel.json", panel)
        old.save_new(OUTPUT/"panel_lock.json", {"manifest": binding, "time": old.now(),
                     "panel_sha256": old.file_hash(OUTPUT/"panel.json"), "sources_sha256": old.file_hash(OUTPUT/"sources_lock.json")})
    panel = old.read_json(OUTPUT/"panel.json")
    panel_lock = old.read_json(OUTPUT/"panel_lock.json")
    assert panel_lock["manifest"] == binding and panel_lock["panel_sha256"] == old.file_hash(OUTPUT/"panel.json")
    assert panel_lock["sources_sha256"] == old.file_hash(OUTPUT/"sources_lock.json")
    panel_audit = audit_stage("panel")
    assert panel_audit["panel_lock_sha256"] == old.file_hash(OUTPUT/"panel_lock.json")
    if not (OUTPUT/"discovery_launch.json").exists():
        old.save_new(OUTPUT/"discovery_launch.json", {"time": old.now(), "panel_audit_sha256": old.file_hash(OUTPUT/"panel_audit.json")})
    if not (OUTPUT/"discovery_targets_lock.json").exists():
        collect("discovery", panel["samples"], None, old.file_hash(OUTPUT/"panel_lock.json"))
    discovery = read_records("discovery", panel["samples"], old.file_hash(OUTPUT/"panel_lock.json"))
    rows = select_choices(panel["samples"], discovery)
    if not (OUTPUT/"selection_lock.json").exists():
        old.save_new(OUTPUT/"selection_lock.json", {"manifest": binding, "time": old.now(), "rows": rows,
                     "panel_lock_sha256": old.file_hash(OUTPUT/"panel_lock.json"), "targets_sha256": old.file_hash(OUTPUT/"discovery_targets_lock.json")})
    selection = old.read_json(OUTPUT/"selection_lock.json")
    assert selection["rows"] == rows and selection["manifest"] == binding and selection["targets_sha256"] == old.file_hash(OUTPUT/"discovery_targets_lock.json")
    discovery_audit = audit_stage("discovery")
    assert discovery_audit["selection_sha256"] == old.file_hash(OUTPUT/"selection_lock.json")
    if not (OUTPUT/"evaluation_launch.json").exists():
        old.save_new(OUTPUT/"evaluation_launch.json", {"time": old.now(), "discovery_audit_sha256": old.file_hash(OUTPUT/"discovery_audit.json"),
                     "selection_sha256": old.file_hash(OUTPUT/"selection_lock.json")})
    if not (OUTPUT/"evaluation_targets_lock.json").exists():
        collect("evaluation", panel["samples"], [r["choice"] for r in rows], old.file_hash(OUTPUT/"selection_lock.json"))
    evaluation = read_records("evaluation", panel["samples"], old.file_hash(OUTPUT/"selection_lock.json"))
    observations = per_state(panel["samples"], rows, evaluation)
    if not (OUTPUT/"state_results.json").exists():
        old.save_new(OUTPUT/"state_results.json", observations)
    result = statistics(observations)
    if not (OUTPUT/"result.json").exists():
        old.save_new(OUTPUT/"result.json", result)
    assert old.read_json(OUTPUT/"result.json") == result and old.read_json(OUTPUT/"state_results.json") == observations
    progress("results_collected", statistical_pass=result["statistical_pass"], gain=result["gain"], ci95=result["ci95"])
    audit = audit_stage("final")
    assert audit["result_sha256"] == old.file_hash(OUTPUT/"result.json")
    assert audit["state_results_sha256"] == old.file_hash(OUTPUT/"state_results.json")
    assert audit["evaluation_targets_sha256"] == old.file_hash(OUTPUT/"evaluation_targets_lock.json")
    finished = old.now()
    elapsed = (datetime.fromisoformat(finished)-datetime.fromisoformat(old.read_json(OUTPUT/"launch.json")["time"])).total_seconds()
    old.save_new(OUTPUT/"final_status.json", {"complete": True, "finished": finished, "elapsed_seconds": elapsed,
        "manifest": binding, "result_sha256": old.file_hash(OUTPUT/"result.json"), "audit_sha256": old.file_hash(OUTPUT/"final_audit.json"),
        "statistical_pass": result["statistical_pass"], "advance_gate_passed": result["statistical_pass"] and audit["passed"],
        "main_continuations": audit["main_continuations"], "conditional_move_evidence_only": True,
        **{k: CONFIG[k] for k in ("training", "arena", "promotion", "automatic_successor")}})
    progress("complete", statistical_pass=result["statistical_pass"], elapsed_seconds=elapsed)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "resume"))
    args = parser.parse_args()
    with old.process_lock(OUTPUT/"run.lock"):
        try:
            prepare() if args.mode == "prepare" else run(resume=args.mode == "resume")
        except Exception:
            old.save_new(OUTPUT/f"failure_{args.mode}_{time.time_ns()}.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()

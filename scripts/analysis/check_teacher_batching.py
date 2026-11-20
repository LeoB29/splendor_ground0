"""Explicit isolated equivalence/profile probe; never modifies or uses162 data."""
import argparse
import copy
from pathlib import Path
import time

import torch

from splendor_ai.alphazero import teacher_batching as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = batch.old
    root = Path("outputs/teacher_batching_checks").resolve()
    target = args.output.resolve()
    if root not in target.parents or target.exists():
        raise ValueError("Choose a new file below outputs/teacher_batching_checks")
    # Do not compete with the ongoing162 pipeline. There is no wait/auto-resume.
    status = Path("outputs/student_refresh_162/final_status.json")
    if not status.exists() or not old.read_json(status).get("complete"):
        raise ValueError("Defer throughput profiling until162 finishes")
    old.initialize_worker()
    work = []
    for seed in (2000500000, 2000500001, 2000500002):
        env = old.SplendorEnv(seed=seed)
        state = env.initial_state()
        # Fixed early, middle and late snapshots, never selected by an outcome.
        for turn in range(41):
            if turn in (0, 20, 40):
                work.append((copy.deepcopy(state), None))
            if state.terminal:
                break
            state = env.step(state, old._RAW.choose_action(env, state))
    before = time.perf_counter()
    serial = [old.trajectory(copy.deepcopy(s), a, include_trace=True) for s, a in work]
    serial_seconds = time.perf_counter()-before
    results = []
    for size in (1, 4, 16):
        policy = batch.Batched035(old._RAW)
        started = time.perf_counter()
        parallel = batch.trajectories(work, policy, size, include_trace=True)
        elapsed = time.perf_counter()-started
        mismatch = [i for i, (a, b) in enumerate(zip(serial, parallel)) if a != b]
        results.append({"batch_size": size, "seconds": elapsed, "metrics": policy.metrics,
                        "mismatched_jobs": mismatch, "full_trace_equal": not mismatch,
                        "serial_over_batch_time": serial_seconds/elapsed,
                        "records": parallel})
    evidence = {"checkpoint_sha256": old.file_hash(old.CHECKPOINT), "torch": str(torch.__version__),
                "threads": torch.get_num_threads(), "serial_seconds": serial_seconds,
                "source_snapshots": [old.serialize_state_snapshot(s) for s, _ in work],
                "serial_records": serial, "batches": results,
                "production_enabled": False,
                "limitations": "Small fixed workload, serial-first ordering; not a reliable speed estimate or universal equivalence proof."}
    old.save_new(target, evidence)
    print({"output": str(target), "jobs": len(work),
           "all_trace_equal": all(r["full_trace_equal"] for r in results), "production_enabled": False})


if __name__ == "__main__":
    main()

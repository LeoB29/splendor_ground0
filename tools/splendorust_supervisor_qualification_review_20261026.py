"""Bounded saved-data review; prints JSON and never writes experiment evidence.

Run only after the owning qualification job has completed. No model, game,
decision reproduction or test is launched. A caller may redirect stdout to a
new supplementary receipt. Imported validators operate on already saved data.
"""
import psutil

psutil.Process().cpu_affinity(list(range(2, 20)))

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
import traceback
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/splendorust_supervisor_qualification_20261026_01"
EXPECTED_FREEZE = "cc5f19859865e41844689deee48cf1e089db59fdbb6b46af14da3711ac34d2e4"
SEALED_DIRS = ("games", "workers", "timing_rows", "timing_worker_segments",
               "control_timeouts", "repeated_decisions", "placement_requests", "placement_acks")


class Review:
    def __init__(self, seconds):
        self.begun = time.perf_counter()
        self.deadline = self.begun + seconds
        self.consumed = {}
        self.checks = []

    def guard(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Independent saved-data review budget exhausted")

    def path(self, value):
        p = pathlib.Path(value)
        p = p if p.is_absolute() else ROOT / p
        p = p.resolve()
        if not p.is_relative_to(ROOT):
            raise ValueError("Review input escaped repository")
        return p

    def fingerprint(self, value):
        self.guard()
        p = self.path(value)
        digest = hashlib.sha256()
        size = 0
        with p.open("rb") as stream:
            while data := stream.read(1024 * 1024):
                self.guard()
                size += len(data)
                digest.update(data)
        item = dict(path=p.relative_to(ROOT).as_posix(), bytes=size, sha256=digest.hexdigest())
        self.consumed[item["path"]] = item
        return item

    def read(self, value):
        item = self.fingerprint(value)
        self.guard()
        if item["bytes"] > 32 * 1024 * 1024:
            raise ValueError("Unexpectedly large JSON review input")
        data = self.path(value).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
        return json.loads(data)

    def local(self, name):
        return self.read(OUT / name)

    def check_row_files(self, row):
        clock = self.path(row["worker_clock_file"])
        assert clock.is_relative_to(OUT / "workers")
        assert self.read(clock) == row["clock"]
        assert clock.name == row["clock"]["request_id"] + ".json"
        assert self.read(clock.parent / "inputs" / clock.name) == dict(
            player="search", observer=row["observer"], before=row["before"])
        for field, expected in (("parent_profile", clock.parent / "parent_profiles" / clock.name),
                                ("child_profile", clock.parent / (clock.stem + "_child_profile.json"))):
            name = row.get(field + "_file")
            if name is None:
                assert row.get(field) is None and not row["completed"]
            else:
                assert self.path(name) == expected and self.read(name) == row[field]
        assert self.read(clock.parent / "placement_requests" / clock.name) == row["placement_evidence"]

    def run(self):
        freeze = self.local("freeze.json")
        assert self.fingerprint(OUT / "freeze.json")["sha256"] == EXPECTED_FREEZE
        completion, execution = self.local("completion.json"), self.local("execution.json")
        assert completion["freeze_sha256"] == execution["freeze_sha256"] == EXPECTED_FREEZE
        if completion["status"] == "incomplete":
            return dict(status="run_incomplete_not_accepted", completion_status="incomplete",
                        failure=completion.get("failure"), exit_code=execution.get("exit_code"),
                        full_seal_or_prefix_acceptance_claim=False)
        assert execution["status"] == "completed" and execution["exit_code"] == 0
        assert not execution["cleanup_errors"] and completion["failure"] is None
        assert execution["charged_job_seconds"] <= freeze["job_seconds"]
        self.checks.append("Completed owning execution and preserved original freeze identity")

        seal = self.local("completed_evidence_seal.json")["artifacts"]
        assert len(seal) == len({x["path"] for x in seal})
        for item in seal:
            assert self.path(item["path"]).is_relative_to(OUT)
            assert self.fingerprint(item["path"]) == item
        actual = {p.relative_to(ROOT).as_posix() for name in SEALED_DIRS
                  for p in (OUT / name).rglob("*") if p.is_file()}
        assert actual == {item["path"] for item in seal}
        self.checks.append("Exact completed-seal inventory and every sealed artifact hash")

        sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
        from tools.splendorust_supervisor_qualification_20261026 import (
            assessment, panel, runner, game_audit as audit, preparation)
        from tools.splendorust_search_scale_20261024 import audit as science
        # Bind the actually imported repository code, without rereading the whole ancestry.
        frozen = {x["path"]: x for x in freeze["inputs"]}
        loaded = set()
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            p = pathlib.Path(filename).resolve()
            if p.suffix == ".py" and p.is_relative_to(ROOT) and p != pathlib.Path(__file__).resolve():
                relative = p.relative_to(ROOT).as_posix()
                assert relative in frozen, "Unbound imported review validator: " + relative
                assert self.fingerprint(p) == frozen[relative]
                loaded.add(relative)

        plan = self.local("panel.json")
        paths = sorted((OUT / "timing_rows").glob("*.json"))
        assert [p.name for p in paths] == [f"{i:04d}.json" for i in range(len(paths))]
        rows = [self.read(p) for p in paths]
        for row in rows:
            self.guard()
            self.check_row_files(row)
        stop_path = OUT / "timing_stop.json"
        timing_stop = self.read(stop_path) if stop_path.exists() else None
        validation = assessment.validate_timing_rows(rows, plan, boundary=self.guard,
                                                      allow_prefix=timing_stop is not None)
        assert validation == self.local("timing_validation.json")
        gate = panel.timing_gate(rows)
        assert gate == self.local("timing_gate.json")
        self.local("control_timeout_manifest.json")
        lifecycle = runner.verify_timing_lifecycle(rows, plan, freeze, self)
        assert lifecycle == self.local("timing_lifecycle_verification.json")
        if timing_stop is not None:
            assert timing_stop == runner.timing_failure_record(rows[-1])
            assert rows[-1]["placement_mode"] == "isolated"
        self.checks.append("Recomputed timing full science, historical matches, gate and retained-control lifecycle")

        schedule = preparation.validate_schedule(self.local("schedule.json"))
        assert schedule == preparation.operational_schedule()
        games, records, game_receipts = [], [], []
        for spec in schedule:
            folder = runner.game_folder(spec)
            if not (folder / "result.json").exists():
                break
            initial, result = self.read(folder / "initial.json"), self.read(folder / "result.json")
            assert all(initial[k] == v for k, v in spec.items())
            move_paths = sorted((folder / "moves").glob("*.json"))
            assert [p.name for p in move_paths] == [f"{i:04d}.json" for i in range(result["plies"])]
            moves = [self.read(p) for p in move_paths]
            for row in moves:
                self.guard()
                self.check_row_files(row)
                assert row["placement_mode"] == "isolated"
            game_receipts.append(audit.audit_game(initial["initial"], moves, result, spec, boundary=self.guard))
            games.append(result)
            records.extend(moves)
        game_folders = {p.parent.resolve() for p in (OUT / "games").rglob("initial.json")}
        assert game_folders == {runner.game_folder(s).resolve() for s in schedule[:len(games)]}
        fault_path = OUT / "collection_stop.json"
        fault = self.read(fault_path) if fault_path.exists() else None
        coverage = self.local("coverage.json")
        if gate["qualified"]:
            expected_coverage = audit.validate_stopped_prefix(schedule, games, fault)
            assert all(coverage[k] == v for k, v in expected_coverage.items())
        else:
            assert not games and fault is None
            assert coverage["status"] == ("timing_failure_verified" if timing_stop else "timing_gate_negative_verified")
        assert self.local("all_game_replay.json") == dict(status="passed", games=len(games), receipts=game_receipts)
        assert self.local("performance.json") == audit.performance(records, games)
        sample = audit.select_replay_sample(records, maximum=48, allow_operational=True)
        assert sample == self.local("replay_sample_manifest.json")
        selected = sample["decisions"]
        repeated_paths = sorted((OUT / "repeated_decisions").glob("*.json"))
        assert [p.name for p in repeated_paths] == [f"{i:02d}.json" for i in range(len(selected))]
        for item, path in zip(selected, repeated_paths, strict=True):
            self.guard()
            original = next(row for row in records if all(row[k] == item[k]
                            for k in ("stage", "pair_index", "game_index", "ply", "player_role")))
            repeated = self.read(path)
            self.check_row_files(repeated)
            audit.validate_boundary_row(repeated)
            science.check_clock(repeated["clock"])
            assert original["completed"] and repeated["completed"]
            assert repeated["clock"]["deadline_seconds"] == 1.
            assert all(repeated[k] == original[k] for k in ("action", "trace", "before", "observer", "after"))
            assert repeated["clock"]["result"] == original["clock"]["result"]
        repeated_summary = self.local("repeated_decision_summary.json")
        assert repeated_summary == dict(status="passed", requests=len(selected), identities=selected,
            no_failed_decision_retries=True, selection_scope="completed operational cells only")
        self.checks.append("Saved operational ledgers and science; deterministic reproduction selection and exact saved outputs")

        integrity = self.local("all_worker_integrity.json")
        expected_names = {f"qualification_{mode}_{quota}" for mode in ("original", "isolated") for quota in (64, 32)}
        expected_names.update(row["timing_worker_name"] for row in rows)
        expected_names.update(f"{spec['stage']}_g{spec['game_index']}_{role}"
                              for spec in schedule[:len(games)] for role in ("challenger", "opponent"))
        expected_names.update(f"audit_{i:02d}" for i in range(len(selected)))
        assert expected_names == {p.name for p in (OUT / "workers").iterdir() if p.is_dir()}
        assert integrity["status"] == "passed" and integrity["workers"] == len(expected_names)
        assert len(integrity["receipts"]) == len(expected_names) <= 164
        assert {r["logs"]["worker"] for r in integrity["receipts"]} == expected_names
        graphs, request_count, owned = 0, 0, []
        for receipt in integrity["receipts"]:
            folder = OUT / "workers" / receipt["logs"]["worker"]
            cleanup = self.read(folder / "cleanup.json")
            assert cleanup["status"] == "passed" and not cleanup["alive"] and not cleanup["errors"]
            owned.extend(cleanup["confirmed"])
            assert receipt["logs"]["status"] == receipt["raw"]["status"] == "passed"
            names = {p.name for p in folder.glob("request-*.json") if re.fullmatch(r"request-\d{8}\.json", p.name)}
            assert names == {p.name for p in (folder / "parent_profiles").glob("*.json")}
            assert names == {p.name for p in (folder / "placement_requests").glob("*.json")}
            assert len(names) == receipt["raw"]["requests"] == receipt["placement_requests"]
            request_count += len(names)
            if receipt["graph"] is not None:
                assert receipt["graph"]["status"] == "passed" and receipt["graph"]["fixtures"] == 11
                graphs += 1
        assert graphs == 2 + lifecycle["graph_workers"] + len(games) + sum(x["quota"] == 64 for x in selected)
        assert request_count == 4 + graphs + len(rows) + len(records) + len(selected)
        qualification = self.local("qualification_summary.json")
        assert qualification["status"] == "passed" and qualification["actual_decisions"] == 208 and qualification["workers"] == 4
        for mode in ("original", "isolated"):
            for quota in (64, 32):
                value = self.local(f"entrypoint_checks_{mode}_{quota}.json")
                assert self.read(OUT / "workers" / f"qualification_{mode}_{quota}" / (value["request_id"] + ".json")) == value
                assert value["status"] == "ok" and value["result"]["counts"]["complete_decisions"] == 52
        margin = len(games) == 6 and fault is None and all(
            r["completed"] and r["clock"]["receipt_elapsed_seconds"] < .7 for r in records)
        qualified = gate["qualified"] and margin
        assert coverage["isolated_qualified"] == qualified and coverage["operational_margin_passed"] == margin
        assert coverage["timing_control_timeouts"] == lifecycle["control_timeouts"]
        assert coverage["timing_continuation_workers"] == lifecycle["continuation_workers"]
        assert coverage["timing_all_attempted_decisions_completed"] == all(row["completed"] for row in rows)
        assert completion["coverage"] == coverage
        verification = self.local("verification_summary.json")
        assert verification["status"] == "passed" and verification["timing_requests"] == len(rows)
        assert verification["games"] == len(games) and verification["live_requests"] == len(records)
        assert verification["workers"] == len(expected_names) and verification["graph_workers"] == graphs
        assert verification["reproductions"] == len(selected) and verification["isolated_qualified"] == qualified
        expected_status = ("isolated_qualified_verified" if qualified else
            "captured_fault_verified" if coverage["status"] == "captured_fault_verified_prefix" else
            coverage["status"] if coverage["status"].startswith("timing_") else "operational_margin_negative_verified")
        assert completion["status"] == expected_status
        launch, claim = self.local("launch.json"), self.local("watchdog_claim.json")
        owned.extend([launch, dict(pid=claim["watchdog_pid"], created=claim["watchdog_created"])])
        absent = []
        for pid, created in sorted({(x["pid"], x["created"]) for x in owned}):
            self.guard()
            try:
                still_owned = abs(psutil.Process(pid).create_time() - created) < .01
            except psutil.NoSuchProcess:
                still_owned = False
            assert not still_owned, f"Owned process identity still present: {pid}"
            absent.append(dict(pid=pid, created=created, same_identity_absent=True))
        self.checks.append("Bound final startup/raw/graph receipts, exact worker inventories and absent owned identities")
        before_readback = list(self.consumed.values())
        for item in before_readback:
            assert self.fingerprint(item["path"]) == item
        self.checks.append("Every consumed input hash read back unchanged")
        return dict(status="passed_scoped_review", completion_status=completion["status"],
            isolated_qualified=qualified, timing_validation=validation, timing_gate=gate,
            timing_lifecycle_summary={k: v for k, v in lifecycle.items() if k != "segments"},
            operational_games=len(games), operational_requests=len(records), reproductions=len(selected),
            worker_count=len(expected_names), graph_workers=graphs, worker_requests=request_count,
            actual_entrypoint_decisions=208, sealed_artifacts=len(seal), sealed_bytes=sum(x["bytes"] for x in seal),
            imported_validator_sources=len(loaded), owned_identities_absent=absent,
            charged_job_seconds=execution["charged_job_seconds"], exit_code=0, cleanup_errors=[],
            consumed_inputs_readback="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=55.)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 60:
        parser.error("Review budget must be between 1 and 60 seconds")
    review = Review(args.max_seconds)
    try:
        result = review.run()
    except Exception as error:
        result = dict(status="audit_incomplete" if isinstance(error, TimeoutError) else "audit_failed",
                      error=repr(error), traceback=traceback.format_exc(), no_acceptance_claim=True)
    result.update(schema="supervisor_qualification_independent_saved_review_v1",
        time=datetime.now(timezone.utc).isoformat(), freeze_sha256=EXPECTED_FREEZE,
        review_elapsed_seconds=time.perf_counter()-review.begun,
        scope="Saved evidence only; no models, new decisions, games, replays or tests. No evidence writes.",
        no_full_ancestry_reaudit=True, no_causal_or_strength_claim=True,
        graph_and_raw_verification_scope="Bound original final-validator receipts plus all sealed bytes; no optional graph-wrapper decoding.",
        completed_checks=review.checks, consumed_inputs_count=len(review.consumed),
        consumed_inputs=sorted(review.consumed.values(), key=lambda x: x["path"]))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed_scoped_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded, saved-only review of the frozen isolated 200-game reliability run.

Run ONLY after completion and owning watchdog exit, with Python -B. This reader
prints JSON and does not write evidence, load a model, start a worker, play a game,
repeat a decision, or execute tests. A caller can exclusively create a NEW review
receipt from stdout. Budget exhaustion yields no acceptance. The repository's
frozen scientific validators are reused; this is an independent evidence readback,
not a second implementation of the scientific oracle or a fresh evaluation.
"""
import sys

sys.dont_write_bytecode = True

import psutil

psutil.Process().cpu_affinity(list(range(2, 20)))

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/splendorust_isolated_reliability_20261026_01"
EXPECTED_FREEZE = "a7be16152ff398eb5feca4f68419539c5817148d9fe31d74c7441ae6a5b63497"
SEALED_DIRS = ("games", "workers", "repeated_decisions", "placement_requests", "placement_acks")
REQUEST_NAME = re.compile(r"request-\d{8}\.json")
CELL_KEYS = ("player_role", "phase", "seat", "private_category")
IDENTITY_KEYS = ("stage", "pair_index", "game_index", "ply", "player_role")


def require(value, message):
    if not value:
        raise ValueError("SAVED_REVIEW: " + message)


def game_folder(spec):
    return OUT / "games" / spec["stage"] / f"p{spec['pair_index']:03d}_g{spec['game_index']}"


class Review:
    def __init__(self, seconds):
        self.begun = time.perf_counter()
        self.deadline = self.begun + seconds
        self.consumed = {}
        self.sealed = {}
        self.checks = []
        self.worker_proofs = {}
        self.row_requests = set()
        self.stage_role_counts = {}
        self.owned = []
        self.loaded_sources = set()

    def guard(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Saved-data review budget exhausted; no acceptance")

    def path(self, value):
        p = Path(value)
        p = (p if p.is_absolute() else ROOT / p).resolve()
        require(p.is_relative_to(ROOT), "Input escaped repository")
        return p

    def register(self, path, size, sha):
        relative = path.relative_to(ROOT).as_posix()
        item = dict(path=relative, bytes=size, sha256=sha)
        if relative in self.consumed:
            require(self.consumed[relative] == item, "Input changed during review: " + relative)
        if relative in self.sealed:
            require(self.sealed[relative] == item, "Sealed artifact changed: " + relative)
        self.consumed[relative] = item
        return item

    def fingerprint(self, value):
        self.guard()
        p = self.path(value)
        size, digest = 0, hashlib.sha256()
        with p.open("rb") as stream:
            while data := stream.read(1024 * 1024):
                self.guard()
                size += len(data)
                digest.update(data)
        return self.register(p, size, digest.hexdigest())

    def read(self, value):
        self.guard()
        p = self.path(value)
        require(p.stat().st_size <= 64 * 1024 * 1024, "Unexpectedly large review JSON")
        data = p.read_bytes()
        self.register(p, len(data), hashlib.sha256(data).hexdigest())
        self.guard()
        return json.loads(data)

    def local(self, name):
        return self.read(OUT / name)

    def inventory(self, folder):
        found = set()
        for p in folder.rglob("*"):
            self.guard()
            if p.is_file():
                found.add(p.relative_to(ROOT).as_posix())
        return found

    def check_absent(self, identities):
        absent = []
        for pid, created in sorted({(x["pid"], x["created"]) for x in identities}):
            self.guard()
            require(type(pid) is int and pid > 0 and type(created) in (int, float), "Invalid owner identity")
            try:
                present = abs(psutil.Process(pid).create_time() - created) < .01
            except psutil.NoSuchProcess:
                present = False
            require(not present, f"Owned process identity still present: {pid}")
            absent.append(dict(pid=pid, created=created, same_identity_absent=True))
        return absent

    def authenticate_imports(self):
        frozen = {x["path"]: x for x in self.freeze["inputs"]}
        for module in list(sys.modules.values()):
            self.guard()
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            p = Path(filename).resolve()
            if p.suffix == ".py" and p.is_relative_to(ROOT) and p != Path(__file__).resolve():
                relative = p.relative_to(ROOT).as_posix()
                require(relative in frozen, "Imported validator is outside the freeze: " + relative)
                if relative not in self.loaded_sources:
                    require(self.fingerprint(p) == frozen[relative], "Imported validator source changed")
                    self.loaded_sources.add(relative)

    def worker_proof(self, folder):
        name = folder.name
        if name not in self.worker_proofs:
            self.guard()
            proof = self.read(folder / "placement_admission.json")
            self.read(folder / "initialization.json")
            verified = self.placement.verify_worker_placement(folder, self.freeze, "isolated")
            require(verified == proof, "Saved worker admission changed")
            require((proof["supervisor"]["pid"], proof["supervisor"]["created"]) ==
                    (self.launch["pid"], self.launch["created"]), "Worker has a different supervising owner")
            self.worker_proofs[name] = proof
        return self.worker_proofs[name]

    def placement_request(self, folder, request_id, parent_profile):
        self.guard()
        proof = self.worker_proof(folder)
        value = self.read(folder / "placement_requests" / f"{request_id}.json")
        self.placement.verify_saved_request(value, self.freeze, "isolated", request_id=request_id,
            player_pid=proof["player_pid"], supervisor_identity=proof["supervisor"], parent_profile=parent_profile)
        stamp = proof["supervisor"]["monotonic_ns"]
        intervals = [(i, x) for i, x in enumerate(self.timeline) if x["start"] <= stamp]
        require(bool(intervals), "Worker started before acknowledged supervisor placement")
        i, interval = intervals[-1]
        before, after = value["before"]["monotonic_ns"], value["after"]["monotonic_ns"]
        require(interval["start"] <= before <= after and
                (i + 1 == len(self.timeline) or after < self.timeline[i + 1]["transition_start"]),
                "Saved request overlaps an unacknowledged placement transition")
        return value

    def row_files(self, row, expected_worker):
        self.guard()
        require(row["placement_mode"] == "isolated" and row.get("observation_mode") == "profiled",
                "Every scientific request must use the qualified isolated observer")
        require("profile_capture_error" not in row and "placement_error" not in row, "Evidence failure is not a captured player fault")
        role = row["player_role"]
        require(role in ("challenger", "opponent") and row["arm"] == "search" and
                row["mode"] == ("candidate" if role == "challenger" else "baseline") and
                row["quota"] == (64 if role == "challenger" else 32), "Scientific role/backend/quota changed")
        clock = self.path(row["worker_clock_file"])
        require(clock.parent == OUT / "workers" / expected_worker, "Decision belongs to an unexpected worker")
        require(clock.name == row["clock"]["request_id"] + ".json", "Request filename binding changed")
        relative = clock.relative_to(ROOT).as_posix()
        require(relative not in self.row_requests, "A scientific request was reused as a new attempt")
        self.row_requests.add(relative)
        require(self.read(clock) == row["clock"], "Original trusted clock differs from move row")
        require(self.read(clock.parent / "inputs" / clock.name) == dict(
            player="search", observer=row["observer"], before=row["before"]), "Original saved input differs")
        for field, expected in (("parent_profile", clock.parent / "parent_profiles" / clock.name),
                                ("child_profile", clock.parent / (clock.stem + "_child_profile.json"))):
            name = row.get(field + "_file")
            if name is None:
                require(row.get(field) is None and not row["completed"], "Successful request lacks a sidecar")
            else:
                require(self.path(name) == expected and self.read(name) == row[field], "Timing sidecar binding changed")
        require(self.placement_request(clock.parent, clock.stem, row["parent_profile"]) == row["placement_evidence"],
                "Placement sidecar differs from move row")

    def exchanges(self):
        request_paths = sorted((OUT / "placement_requests").glob("*.json"))
        ack_paths = sorted((OUT / "placement_acks").glob("*.json"))
        names = [f"{i:04d}.json" for i in range(1, len(request_paths) + 1)]
        require(names and [p.name for p in request_paths] == names and [p.name for p in ack_paths] == names,
                "Supervisor handshake sequence differs")
        self.timeline = []
        for path, ack_path in zip(request_paths, ack_paths, strict=True):
            request, ack = self.read(path), self.read(ack_path)
            require(request["placement_mode"] == "isolated" and request["sequence"] == int(path.stem),
                    "Controller placement or sequence changed")
            self.common.validate_exchange(request, ack, self.freeze, self.watchdog,
                dict(pid=self.launch["pid"], created=self.launch["created"]))
            self.timeline.append(dict(start=ack["watchdog_receipt"]["monotonic_ns"],
                                      transition_start=request["runner_receipt"]["monotonic_ns"]))
        self.placement.verify_saved_runtime(self.local("watchdog_initial_placement.json"), self.freeze,
            "isolated", role="watchdog", identity=dict(pid=self.watchdog["watchdog_pid"], created=self.watchdog["watchdog_created"]))
        return dict(status="passed", exchanges=len(self.timeline), owned_watchdog_and_runner=True)

    def compact_timing(self, row):
        keys = ("clock", "request_id", "deadline_seconds", "elapsed_seconds", "receipt_elapsed_seconds",
                "received_before_deadline", "accepted", "status")
        clock = {key: row["clock"][key] for key in keys}
        clock["telemetry"] = {key: value for key, value in row["clock"].get("telemetry", {}).items()
                              if key in ("request_cpu_seconds", "rss_bytes", "peak_sampled_rss_bytes")}
        return {**{key: row[key] for key in ("stage", "placement_mode", "player_role", "arm", "mode", "quota")}, "clock": clock}

    def count_boundaries(self, row):
        group = self.stage_role_counts.setdefault((row["stage"], row["player_role"]), dict(
            attempted=0, completed=0, statuses=Counter(), above=0, at_or_above=0))
        group["attempted"] += 1
        group["statuses"][row["clock"]["status"]] += 1
        group["completed"] += int(row["completed"])
        if row["completed"]:
            arrived = row["clock"]["receipt_elapsed_seconds"]
            group["above"] += int(arrived > .7)
            group["at_or_above"] += int(arrived >= .7)

    def reconcile_boundaries(self, rows_count, stop):
        value = self.local("boundary_analysis.json")
        require(value["rows"] == rows_count and value["observed_faults"] == int(stop is not None) and
                value["all_collected_decisions_completed"] is (stop is None) and
                value["complete_planned_collection_inferred"] is False and value["deadline_adjustment_seconds"] == 0. and
                value["causal_bottleneck_proven"] is False and value["no_strength_claim"] is True,
                "Boundary summary scope or fault count changed")
        require(len(value["groups"]) == len(self.stage_role_counts), "Boundary group inventory differs")
        seen = set()
        for item in value["groups"]:
            key = item["stage"], item["player_role"]
            require(key not in seen and key in self.stage_role_counts, "Duplicate or unexpected boundary group")
            seen.add(key)
            actual = self.stage_role_counts[key]
            require(item["attempted_decisions"] == actual["attempted"] and item["completed_decisions"] == actual["completed"] and
                    item["statuses"] == actual["statuses"] and item["completed_above_700ms"] == actual["above"] and
                    item["completed_at_or_above_700ms"] == actual["at_or_above"], "Boundary counts differ from every live clock")
        require(len(value["failures"]) == int(stop is not None), "Boundary failure inventory differs")
        if stop is not None:
            failed = value["failures"][0]
            require(all(failed[key] == stop[key] for key in (*IDENTITY_KEYS, "seat", "request_id")) and
                    failed["classification"] == stop["failure_classification"] and
                    failed["receipt_elapsed_seconds"] == stop["clock"]["receipt_elapsed_seconds"] and
                    failed["partial_or_missing_evidence_not_promoted"] is True, "Boundary fault attribution differs")
        return value

    def run(self):
        self.freeze = self.local("freeze.json")
        require(self.consumed[(OUT / "freeze.json").relative_to(ROOT).as_posix()]["sha256"] == EXPECTED_FREEZE,
                "Wrong frozen attempt")
        completion, execution = self.local("completion.json"), self.local("execution.json")
        require(completion["freeze_sha256"] == execution["freeze_sha256"] == EXPECTED_FREEZE, "Completion/execution freeze changed")
        if completion["status"] == "incomplete":
            return dict(status="run_incomplete_not_accepted", completion_status="incomplete",
                failure=completion.get("failure"), exit_code=execution.get("exit_code"), no_acceptance_claim=True)
        require(completion["status"] in ("completed_verified", "captured_fault_verified") and completion["failure"] is None,
                "Not a verified complete workload or captured prefix")
        require(execution["status"] == "completed" and execution["exit_code"] == 0 and not execution["cleanup_errors"],
                "Owning execution or cleanup has not completed successfully")
        charged = execution["charged_job_seconds"]
        require(type(charged) in (int, float) and math.isfinite(charged) and 0 <= charged < self.freeze["job_seconds"] == 21600,
                "Execution exceeded its frozen global window")
        require(execution["cleanup_receipt"]["errors"] == [] and execution["cleanup_receipt"]["unconfirmed"] == [],
                "Owning cleanup has unconfirmed processes")
        self.launch, claim, self.watchdog = self.local("launch.json"), self.local("watchdog_claim.json"), self.local("watchdog.json")
        require(self.launch["freeze_sha256"] == claim["freeze_sha256"] == self.watchdog["freeze_sha256"] == EXPECTED_FREEZE,
                "Owner freeze identity differs")
        require((self.launch["pid"], self.launch["created"]) == (self.watchdog["runner_pid"], self.watchdog["runner_created"]) and
                (claim["watchdog_pid"], claim["watchdog_created"]) == (self.watchdog["watchdog_pid"], self.watchdog["watchdog_created"]),
                "Owned runner/watchdog identities differ")
        self.owned.extend([self.launch, dict(pid=claim["watchdog_pid"], created=claim["watchdog_created"])])
        self.owned.extend(execution["cleanup_receipt"]["confirmed"])
        self.check_absent(self.owned)
        self.checks.append("Exact attempt identity, successful owning exit and absent runner/watchdog identities")

        seal = self.local("completed_evidence_seal.json")["artifacts"]
        require(len(seal) == len({x["path"] for x in seal}), "Duplicate sealed paths")
        for item in seal:
            self.guard()
            p = self.path(item["path"])
            require(any(p.is_relative_to(OUT / name) for name in SEALED_DIRS), "Unexpected seal scope")
            self.sealed[item["path"]] = item
        actual = set().union(*(self.inventory(OUT / name) for name in SEALED_DIRS))
        require(actual == set(self.sealed), "Completed seal is not the exact evidence inventory")
        # Every sealed byte is read below or in the final readback, including raw logs.

        sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
        from tools.splendorust_isolated_reliability_20261026 import audit, common, placement, schedule as schedules
        self.audit, self.common, self.placement = audit, common, placement
        self.authenticate_imports()
        require(self.freeze["placement_mode"] == "isolated" and self.freeze["seconds_per_move"] == 1. and
                self.freeze["internal_seconds"] == .9 and self.freeze["total_games"] == 200 and
                self.freeze["operational_games"] == 6 and self.freeze["main_games"] == 194 and
                self.freeze["margin_seconds"] == .7 and self.freeze["margin_is_forfeit"] is False and
                self.freeze["margin_is_admission_gate"] is False and self.freeze["automatic_retries"] == 0 and
                self.freeze["failed_decision_retries"] == 0, "Frozen reliability design differs")
        exchanges = self.exchanges()
        schedule = self.local("schedule.json")
        schedules.validate_schedule(schedule)
        require(schedule == schedules.make_schedule(), "Fresh frozen schedule derivation differs")
        schedule_item = self.consumed[(OUT / "schedule.json").relative_to(ROOT).as_posix()]
        require(schedule_item in self.freeze["inputs"], "Saved schedule is not authenticated by the original freeze")
        stop_path = OUT / "collection_stop.json"
        stop = self.read(stop_path) if stop_path.exists() else None
        expected_status = "completed_verified" if stop is None else "captured_fault_verified"
        require(completion["status"] == expected_status, "Completion does not match fault-stop status")

        games, timing_rows, game_receipts, winners, winning_rows = [], [], [], {}, {}
        template = audit.select_replay_sample([], maximum=48, allow_operational=stop is not None)
        cells = [tuple(item[k] for k in CELL_KEYS) for item in template["empty_cells"]]
        last_row = None
        for spec in schedule:
            self.guard()
            folder = game_folder(spec)
            if not (folder / "result.json").exists():
                break
            initial, result = self.read(folder / "initial.json"), self.read(folder / "result.json")
            require(set(initial) == set(spec) | {"initial"} and all(initial[k] == v for k, v in spec.items()),
                    "Game initial schedule identity differs")
            paths = sorted((folder / "moves").glob("*.json"))
            require([p.name for p in paths] == [f"{i:04d}.json" for i in range(result["plies"])], "Game move inventory is not contiguous")
            moves = [self.read(p) for p in paths]
            for row in moves:
                self.row_files(row, f"{spec['stage']}_g{spec['game_index']}_{row['player_role']}")
                timing_rows.append(self.compact_timing(row))
                self.count_boundaries(row)
            checked = audit.audit_game(initial["initial"], moves, result, spec, boundary=self.guard)
            require(self.read(folder / "online_replay.json") == checked, "Online and independently reread game science differ")
            game_receipts.append(checked)
            games.append(result)
            last_row = moves[-1]
            if stop is not None or spec["stage"] == "main":
                part = audit.select_replay_sample(moves, maximum=48, allow_operational=stop is not None)
                by_identity = {tuple(row[k] for k in IDENTITY_KEYS): row for row in moves}
                for item in part["decisions"]:
                    cell = tuple(item[k] for k in CELL_KEYS)
                    if cell not in winners or item["selection_sha256"] < winners[cell]["selection_sha256"]:
                        winners[cell] = item
                        winning_rows[cell] = by_identity[tuple(item[k] for k in IDENTITY_KEYS)]
        actual_folders = {p.parent.resolve() for p in (OUT / "games").rglob("initial.json")}
        require(actual_folders == {game_folder(spec).resolve() for spec in schedule[:len(games)]}, "Extra or missing game outside the completed prefix")
        coverage = audit.validate_stopped_prefix(schedule, games, stop)
        require(coverage == self.local("coverage.json") == completion["coverage"], "Coverage differs from exact scheduled prefix")
        require(self.local("all_game_replay.json") == dict(status="passed", games=len(games), receipts=game_receipts),
                "Final game replay summary differs")
        if stop is not None:
            require(last_row is not None and last_row["completed"] is False and
                    stop == self.read(game_folder(schedule[len(games) - 1]) / "captured_fault.json") and
                    stop["clock"] == last_row["clock"] and stop["worker_clock_file"] == last_row["worker_clock_file"] and
                    all(stop[k] == last_row[k] for k in (*IDENTITY_KEYS, "seat")), "Captured fault is not the final original request")
        performance = audit.performance(timing_rows, games)
        require(performance == self.local("performance.json"), "Recomputed timing distributions differ")
        boundary_summary = self.reconcile_boundaries(len(timing_rows), stop)
        sample = {**template, "decisions": [winners[cell] for cell in cells if cell in winners],
                  "empty_cells": [dict(zip(CELL_KEYS, cell)) for cell in cells if cell not in winners]}
        require(sample == self.local("replay_sample_manifest.json"), "Deterministic all-available-cell selection differs")
        selected = sample["decisions"]
        require(len(selected) <= 48 and all(sum(x["player_role"] == role for x in selected) <= 24 for role in audit.ROLES),
                "Replay cap or role bound exceeded")
        paths = sorted((OUT / "repeated_decisions").glob("*.json"))
        require([p.name for p in paths] == [f"{i:02d}.json" for i in range(len(selected))], "Saved reproduction inventory differs")
        for index, (item, path) in enumerate(zip(selected, paths, strict=True)):
            self.guard()
            original = winning_rows[tuple(item[k] for k in CELL_KEYS)]
            repeated = self.read(path)
            self.row_files(repeated, f"audit_{index:02d}")
            audit.validate_boundary_row(repeated)
            audit.check_clock(repeated["clock"])
            require(original["completed"] and repeated["completed"] and repeated["clock"]["deadline_seconds"] == 1., "Incomplete or shortened reproduction")
            require(all(repeated[k] == original[k] for k in ("before", "observer", "action", "trace", "after")) and
                    repeated["clock"]["result"] == original["clock"]["result"], "Saved reproduction full science differs")
        require(self.local("repeated_decision_summary.json") == dict(status="passed", requests=len(selected), identities=selected,
                prefix_only=stop is not None), "Reproduction summary differs")
        self.checks.append("Every played game: full saved science, dictionary ledger, raw-clock/file binding, isolated placement and exact first-fault loss")
        self.checks.append("Every live timing distribution, descriptive inclusive700ms count and deterministic up-to48 saved full-output reproductions")

        qualification = self.local("qualification_summary.json")
        require(qualification["status"] == "passed" and qualification["actual_workers"] == 2 and qualification["complete_decisions"] == 104,
                "Actual startup qualification inventory differs")
        startup_ids = {}
        for quota in (32, 64):
            receipt = self.local(f"entrypoint_checks_{quota}.json")
            folder = OUT / "workers" / f"qualification_{quota}"
            require(receipt == self.read(folder / (receipt["request_id"] + ".json")) and receipt["status"] == "ok" and
                    receipt["result"]["status"] == "entrypoint_counterexamples_passed" and
                    receipt["result"]["challenger_quota"] == quota and receipt["result"]["counts"]["complete_decisions"] == 52,
                    "Preserved actual startup suite differs")
            startup_ids[folder.name] = receipt["request_id"]
        if len(games) > 6:
            gate = self.local("operational_gate.json")
            require(gate["passed"] is True and gate["operational_games"] == 6 and gate["remaining_games"] == 194 and
                    gate["total_games"] == 200 and gate["outcomes_used"] is False, "Main expansion lacks its outcome-blind resource gate")
            self.local("final_evidence_forecast.json")

        expected_names = {"qualification_32", "qualification_64"}
        expected_names.update(f"{spec['stage']}_g{spec['game_index']}_{role}"
                              for spec in schedule[:len(games)] for role in audit.ROLES)
        expected_names.update(f"audit_{i:02d}" for i in range(len(selected)))
        require(expected_names == {p.name for p in (OUT / "workers").iterdir() if p.is_dir()} and len(expected_names) <= 450,
                "Exact worker lifecycle inventory differs")
        expected_graphs = {"qualification_64"}
        expected_graphs.update(f"{spec['stage']}_g{spec['game_index']}_challenger" for spec in schedule[:len(games)])
        expected_graphs.update(f"audit_{i:02d}" for i, item in enumerate(selected) if item["mode"] == "candidate")
        require(len(expected_graphs) <= 249, "Graph worker cap exceeded")
        summaries = {key: self.local(name) for key, name in (
            ("logs", "all_worker_log_integrity.json"), ("raw", "all_raw_transport.json"),
            ("graphs", "all_graph_parity.json"), ("placement", "all_placement_integrity.json"))}
        receipts = {}
        for key, summary in summaries.items():
            expected = expected_graphs if key == "graphs" else expected_names
            values = summary["receipts"]
            require(summary["status"] == "passed" and summary["workers"] == len(values) == len(expected) and
                    {r["worker"] for r in values} == expected, "Final receipt worker inventory differs: " + key)
            receipts[key] = {r["worker"]: r for r in values}
        require(summaries["graphs"]["independent_full_outputs_and_float32_bits"] is True and
                summaries["placement"]["all_workers_isolated"] is True and
                summaries["placement"]["controller_exchanges"] == exchanges, "Final graph/placement proof scope differs")
        worker_request_count = raw_failed_count = 0
        for name in sorted(expected_names):
            self.guard()
            folder = OUT / "workers" / name
            self.worker_proof(folder)
            cleanup = self.read(folder / "cleanup.json")
            require(cleanup["status"] == "passed" and not cleanup["alive"] and not cleanup["errors"], "Worker cleanup incomplete")
            self.owned.extend(cleanup["confirmed"])
            log, raw, placed = (receipts[key][name] for key in ("logs", "raw", "placement"))
            require(log["status"] == raw["status"] == "passed" and placed["placement_mode"] == "isolated", "Final worker verification failed")
            require(log["cleanup_sha256"] == self.consumed[(folder / "cleanup.json").relative_to(ROOT).as_posix()]["sha256"],
                    "Log verifier cleanup binding changed")
            request_paths = sorted(p for p in folder.glob("request-*.json") if REQUEST_NAME.fullmatch(p.name))
            names = [p.name for p in request_paths]
            require(names == [p.name for p in sorted((folder / "parent_profiles").glob("*.json"))] ==
                    [p.name for p in sorted((folder / "placement_requests").glob("*.json"))], "Per-request observer/placement inventory differs")
            require(len(names) == raw["requests"] == placed["requests"], "Raw receipt request count differs")
            require(len(raw["requests_checked"]) == len(names) and
                    {x["request_id"] + ".json" for x in raw["requests_checked"]} == set(names), "Raw checked request identities differ")
            require(raw["exact_bytes_checked"] is True and raw["original_trusted_receipts_unchanged"] is True and
                    raw["failed_packets_remain_unaccepted"] is True, "Raw transport verifier scope differs")
            checked_ids = {x["request_id"]: x for x in raw["requests_checked"]}
            worker_request_count += len(names)
            raw_failed_count += raw["failed_requests"]
            for path in request_paths:
                self.guard()
                clock = self.read(path)
                require(clock["request_id"] == path.stem and clock["status"] == checked_ids[path.stem]["status"],
                        "Raw proof and original clock status differ")
                parent = self.read(folder / "parent_profiles" / path.name)
                self.placement_request(folder, path.stem, parent)
            if name in startup_ids:
                require(startup_ids[name] in checked_ids, "Actual startup request absent from verified transport")
            if name in expected_graphs:
                graph = receipts["graphs"][name]
                require(graph["status"] == "passed" and graph["fixtures"] == 11, "Graph fixture receipt differs")
        require(worker_request_count == 2 + len(expected_graphs) + len(timing_rows) + len(selected),
                "All worker requests do not reconcile to startup/graph/live/reproduction requests")
        require(raw_failed_count == int(stop is not None), "Raw failure count differs from the first-fault stop")
        verification = self.local("verification_summary.json")
        require(verification["status"] == "passed" and verification["games"] == len(games) and verification["requests"] == len(timing_rows) and
                verification["workers"] == len(expected_names) and verification["graph_workers"] == len(expected_graphs) and
                verification["reproductions"] == len(selected) and verification["complete_planned_workload"] is (stop is None) and
                verification["captured_fault"] is (stop is not None) and verification["all_workers_and_supervisors_isolated"] is True and
                verification["all_scientific_outputs_transitions_and_raw_receipts_verified"] is True and verification["no_strength_claim"] is True,
                "Final verification summary differs from independent readback")
        require(completion["automatic_retries"] == 0 and completion["larger_data_deferred"] is True and
                completion["no_strength_claim"] is True and completion["total_games_maximum"] == 200 and completion["main_games_maximum"] == 194,
                "Final policy/completion fields differ")
        self.authenticate_imports()
        self.checks.append("Exact worker/request/graph/startup inventories; saved full placement proofs; bound original raw/log/graph final-validator receipts")

        # Complete byte authentication without retaining decoded logs. Seal items
        # already parsed above are read again too; all consumed summaries and
        # validator sources receive the same final readback. Never re-audit ancestry.
        targets = {**self.sealed, **self.consumed}
        for item in targets.values():
            require(self.fingerprint(item["path"]) == item, "Final input readback differs")
        final_actual = set().union(*(self.inventory(OUT / name) for name in SEALED_DIRS))
        require(final_actual == set(self.sealed), "Evidence inventory changed during review")
        absent = self.check_absent(self.owned)
        self.guard()
        self.checks.append("Every sealed artifact and consumed summary/source hash read back; exact inventory unchanged; all recorded owners absent")
        return dict(status="passed_scoped_review", completion_status=completion["status"], coverage=coverage,
            performance=performance, games=len(games), live_decisions=len(timing_rows), actual_entrypoint_decisions=104,
            reproduced_completed_decisions=len(selected), replay_cap=48, replay_populated_cells=len(selected),
            replay_empty_cells=len(sample["empty_cells"]), worker_count=len(expected_names), graph_workers=len(expected_graphs),
            worker_requests=worker_request_count, preserved_player_faults=raw_failed_count,
            boundary_failures=boundary_summary["failures"], sealed_artifacts=len(seal), sealed_bytes=sum(x["bytes"] for x in seal),
            imported_validator_sources=len(self.loaded_sources), owned_identities_absent=absent,
            charged_job_seconds=charged, exit_code=0, cleanup_errors=[], consumed_inputs_readback="passed",
            maximum_sampled_combined_worker_rss_bytes=verification["maximum_sampled_combined_worker_rss_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=300.)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 300:
        parser.error("Review budget must be between 1 and 300 seconds")
    review = Review(args.max_seconds)
    try:
        result = review.run()
    except Exception as error:
        result = dict(status="audit_incomplete" if isinstance(error, TimeoutError) else "audit_failed",
            error=repr(error), traceback=traceback.format_exc(), no_acceptance_claim=True)
    result.update(schema="isolated_reliability_independent_saved_review_v1", time=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=EXPECTED_FREEZE, review_elapsed_seconds=time.perf_counter() - review.begun,
        review_budget_seconds=args.max_seconds, scope="Saved evidence only; no models, new decisions, games, replays, tests or evidence writes.",
        no_full_ancestry_reaudit=True, no_causal_or_strength_claim=True,
        science_scope="Every saved live decision and ledger recomputed using authenticated frozen validators; selected saved reproductions compared exactly.",
        initial_world_scope="Schedule identity and sealed initial world checked; seed-to-world reconstruction relies on the original final-validator receipt and is not rerun.",
        graph_and_raw_scope="Original final-validator receipts bound to exact worker/request inventories and every sealed byte; no graph forward, log decompression or raw-packet parser rerun.",
        boundary_scope="Every row's boundary evidence validated by the frozen game auditor; inclusive700ms counts and attribution reconciled; boundary-metric distributions not separately recomputed.",
        resource_scope="Global charged runtime and saved outcome-blind expansion gate checked; historical empirical forecast and RAM telemetry not independently remeasured.",
        completed_checks=review.checks, consumed_inputs_count=len(review.consumed),
        consumed_inputs=sorted(review.consumed.values(), key=lambda x: x["path"]))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed_scoped_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Saved-only scientific component of the separately bounded recovery audit.

No evidence writes, model calls, subprocesses, fresh games or decision retries.
The original attempt remains incomplete with exit1. Only the owning recovery
pipeline can issue separate recovered acceptance, after its remaining runtime,
ancestry, final snapshot readback and cleanup checks also pass. Scientific
validators are the authenticated frozen implementations, not a new oracle.
"""
import sys
import psutil
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/splendorust_isolated_reliability_20261026_01"
EXPECTED_FREEZE = "a7be16152ff398eb5feca4f68419539c5817148d9fe31d74c7441ae6a5b63497"
SEALED_DIRS = ("games", "workers", "repeated_decisions", "placement_requests", "placement_acks")
REQUEST_NAME = re.compile(r"request-\d{8}\.json")
CELL_KEYS = ("player_role", "phase", "seat", "private_category")
IDENTITY_KEYS = ("stage", "pair_index", "game_index", "ply", "player_role")
EXPECTED_SEALED_FILES = 74528
EXPECTED_SEALED_BYTES = 700262089


def require(value, message):
    if not value:
        raise ValueError("SAVED_REVIEW: " + message)


def game_folder(spec):
    return OUT / "games" / spec["stage"] / f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def reconstruct_initial(seed_decimal):
    """Exact original final-validator initial-world route; never play a move."""
    from tools.external_strength_screen_20261018 import rules
    return rules.serialize_snapshot(rules.CorrectedEnv(seed=int(seed_decimal)).initial_state())


def validate_original_interruption(completion, execution, failure, final_start, markers):
    """Recognize only this recorded final source-guard budget interruption."""
    require(markers == {"coverage": False, "verification": False, "final_end": False, "collection_stop": False},
            "Original missing-final-marker contract differs")
    require(completion.get("status") == "incomplete" and completion.get("stage") == "final" and
            completion.get("coverage") is None and completion.get("automatic_retries") == 0 and
            completion.get("freeze_sha256") == EXPECTED_FREEZE, "Wrong original incomplete attempt")
    require(execution.get("status") == "runner_failed" and execution.get("exit_code") == 1 and
            execution.get("freeze_sha256") == EXPECTED_FREEZE and execution.get("cleanup_errors") == [],
            "Original exit must be the preserved clean runner failure")
    cleanup = execution.get("cleanup_receipt", {})
    require(cleanup.get("errors") == [] and cleanup.get("unconfirmed") == [], "Original owning cleanup incomplete")
    require(failure.get("stage") == "final" and failure.get("preserved_partial_evidence") is True and
            failure.get("partial_evidence_not_accepted") is True and failure.get("error") == completion.get("failure"),
            "Failure and original completion disagree")
    error = failure.get("error")
    require(type(error) is str and error.endswith("ValueError: Stage/global real-clock deadline\n"),
            "Only the recorded final-stage clock interruption is recoverable here")
    frames = re.findall(r'File "([^\"]+)", line (\d+), in (\w+)', error)
    expected = [("runner.py", "763", "run"), ("runner.py", "748", "run_stages"),
                ("runner.py", "721", "final_checks"), ("runner.py", "199", "verify_runtime"),
                ("runner.py", "144", "guard"), ("common.py", "38", "require")]
    actual = [(Path(name).name, line, function) for name, line, function in frames]
    require(actual == expected and all(Path(name).resolve().parent == ROOT / "tools/splendorust_isolated_reliability_20261026"
                                      for name, _, _ in frames), "Unexpected failure locus or extra exception frame")
    require("verify_runtime(freeze,envelope)" in error and "envelope.guard()" in error,
            "Interruption was not the final frozen-source guard")
    require(final_start.get("cap_seconds") == 1800 and
            final_start["monotonic_deadline"] == final_start["monotonic"] + 1800 and
            final_start["unix_deadline"] == final_start["unix_time"] + 1800,
            "Original final-stage window changed")
    require(execution["monotonic_end"] >= final_start["monotonic_deadline"] or
            execution["unix_end"] >= final_start["unix_deadline"], "Recorded execution did not reach the final cap")
    require(0 <= execution["charged_job_seconds"] < 21600, "Original global cap or charged duration differs")
    return {"original_status": "incomplete", "original_exit_code": 1, "original_status_preserved": True,
            "recognized_interruption": "final_frozen_input_loop_guard_at_final_stage_budget",
            "original_final_input_loop_completed_index_unknown": True,
            "original_global_cap_exhausted": False}


def validate_saved_reproduction(original, repeated):
    """A saved successful repetition must match all scientific outputs exactly."""
    require(original.get("completed") is True and repeated.get("completed") is True and
            repeated["clock"]["deadline_seconds"] == 1., "Incomplete or shortened reproduction")
    require(all(repeated[key] == original[key] for key in ("before", "observer", "action", "trace", "after")) and
            repeated["clock"]["result"] == original["clock"]["result"], "Saved reproduction full science differs")


def merge_replay_cells(winners, winning_rows, selected, rows):
    """Streaming minimum per frozen cell is equivalent to one full-panel pass."""
    identities = {tuple(row[key] for key in IDENTITY_KEYS): row for row in rows}
    for item in selected:
        cell = tuple(item[key] for key in CELL_KEYS)
        if cell not in winners or item["selection_sha256"] < winners[cell]["selection_sha256"]:
            require(tuple(item[key] for key in IDENTITY_KEYS) in identities, "Replay selector invented a source row")
            winners[cell] = item
            winning_rows[cell] = identities[tuple(item[key] for key in IDENTITY_KEYS)]


def receipt_inventory(summary, expected_names):
    values = summary["receipts"]
    require(summary["status"] == "passed" and summary["workers"] == len(values) == len(expected_names) and
            {row["worker"] for row in values} == expected_names, "Final receipt worker inventory differs")
    return {row["worker"]: row for row in values}


def bind_closed_stream_receipts(folder, cleanup, log, raw, sealed):
    """Bind original decompression proofs to the exact compressed sealed bytes."""
    names = {"stdout.log", "stderr.log", "events.jsonl"}
    streams = log["decoded_streams"]
    require(log["streams"] == len(streams) == 3 and {item["name"] for item in streams} == names and
            set(cleanup["log_compression"]) == names, "Closed stream receipt inventory differs")
    fields = ("status", "codec", "path", "raw_bytes", "raw_sha256", "compressed_bytes", "compressed_sha256")
    by_name = {}
    for item in streams:
        name = item["name"]
        require(item["status"] == "verified" and item["path"] == name + ".gz", "Compressed stream identity differs")
        require(all(item[key] == cleanup["log_compression"][name][key] for key in fields),
                "Original cleanup and final decoded-stream proof differ")
        relative = (folder / item["path"]).relative_to(ROOT).as_posix()
        require(sealed.get(relative) == dict(path=relative, bytes=item["compressed_bytes"], sha256=item["compressed_sha256"]),
                "Decoded-stream proof does not bind the sealed compressed bytes")
        by_name[name] = item
    require(log["raw_bytes"] == sum(item["raw_bytes"] for item in streams) and
            log["compressed_bytes"] == sum(item["compressed_bytes"] for item in streams), "Log byte totals differ")
    require(all(raw["stdout_log"][key] == by_name["stdout.log"][key] for key in fields),
            "Raw packet verifier does not bind the same exact stdout stream")


def bind_graph_receipt(receipt, identity, graph_clock, saved_request, parity_summary, digest):
    """Authenticate existing graph proof identities without executing a graph."""
    require(receipt["status"] == "passed" and receipt["fixtures"] == 11 and
            receipt["graph_identity_sha256"] == digest(identity) and
            receipt["weights_sha256"] == identity["weights_before_sha256"] == identity["weights_after_sha256"],
            "Saved graph final receipt identity/weights binding differs")
    require(graph_clock == saved_request and graph_clock["status"] == "ok" and
            graph_clock["result"] == parity_summary and
            parity_summary["graph_identity_sha256"] == receipt["graph_identity_sha256"],
            "Saved graph final receipt does not bind its successful raw request")


class Review:
    def __init__(self, seconds, *, boundary, original_freeze, recovery_freeze):
        self.begun = time.perf_counter()
        self.deadline = self.begun + seconds
        self.boundary = boundary
        self.original_freeze = original_freeze
        self.recovery_freeze = recovery_freeze
        self.original_snapshot = {item["path"]: item for item in recovery_freeze["original_snapshot"]}
        require(len(self.original_snapshot) == len(recovery_freeze["original_snapshot"]), "Duplicate original snapshot path")
        self.recovery_inputs = {item["path"]: item for item in recovery_freeze["inputs"]}
        require(len(self.recovery_inputs) == len(recovery_freeze["inputs"]), "Duplicate recovery source path")
        self.consumed = {}
        self.sealed = {}
        self.checks = []
        self.worker_proofs = {}
        self.row_requests = set()
        self.stage_role_counts = {}
        self.owned = []
        self.loaded_sources = set()
        self.checked_placement_requests = set()

    def guard(self):
        self.boundary()
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
        if path.is_relative_to(OUT):
            require(relative in self.original_snapshot and self.original_snapshot[relative] == item,
                    "Original artifact differs from recovery snapshot: " + relative)
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
            if p.suffix == ".py" and p.is_relative_to(ROOT):
                relative = p.relative_to(ROOT).as_posix()
                expected = self.recovery_inputs.get(relative, frozen.get(relative))
                require(expected is not None, "Imported validator is outside both freezes: " + relative)
                if relative not in self.loaded_sources:
                    require(self.fingerprint(p) == expected, "Imported validator source changed")
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
        self.checked_placement_requests.add((folder.name, request_id))
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
        require(self.inventory(OUT) == set(self.original_snapshot), "Original directory inventory differs from recovery freeze")
        self.freeze = self.local("freeze.json")
        require(self.consumed[(OUT / "freeze.json").relative_to(ROOT).as_posix()]["sha256"] == EXPECTED_FREEZE,
                "Wrong frozen attempt")
        require(self.freeze == self.original_freeze, "Supplied original freeze differs from saved bytes")
        completion, execution = self.local("completion.json"), self.local("execution.json")
        interrupted = validate_original_interruption(completion, execution, self.local("failure.json"),
            self.local("stages/final_start.json"), {
                "coverage": (OUT / "coverage.json").exists(), "verification": (OUT / "verification_summary.json").exists(),
                "final_end": (OUT / "stages/final_end.json").exists(), "collection_stop": (OUT / "collection_stop.json").exists()})
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
        self.checks.append("Exact preserved incomplete/exit1 source-guard interruption, clean original cleanup and absent runner/watchdog identities")

        seal = self.local("completed_evidence_seal.json")["artifacts"]
        require(len(seal) == len({x["path"] for x in seal}) == EXPECTED_SEALED_FILES and
                sum(item["bytes"] for item in seal) == EXPECTED_SEALED_BYTES, "Original sealed inventory/count differs")
        for item in seal:
            self.guard()
            p = self.path(item["path"])
            require(any(p.is_relative_to(OUT / name) for name in SEALED_DIRS), "Unexpected seal scope")
            require(self.original_snapshot.get(item["path"]) == item, "Original seal differs from recovery snapshot")
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
        stop = None

        games, timing_rows, game_receipts, winners, winning_rows = [], [], [], {}, {}
        template = audit.select_replay_sample([], maximum=48, allow_operational=stop is not None)
        cells = [tuple(item[k] for k in CELL_KEYS) for item in template["empty_cells"]]
        for spec in schedule:
            self.guard()
            folder = game_folder(spec)
            if not (folder / "result.json").exists():
                break
            initial, result = self.read(folder / "initial.json"), self.read(folder / "result.json")
            require(set(initial) == set(spec) | {"initial"} and all(initial[k] == v for k, v in spec.items()),
                    "Game initial schedule identity differs")
            # Reconstruct only the saved initial world. No environment step,
            # action selection, native player or fresh game is invoked.
            expected_initial = reconstruct_initial(spec["seed_decimal"])
            require(initial["initial"] == expected_initial, "Saved initial world does not match the frozen seed")
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
            if spec["stage"] == "main":
                part = audit.select_replay_sample(moves, maximum=48, allow_operational=False)
                merge_replay_cells(winners, winning_rows, part["decisions"], moves)
        actual_folders = {p.parent.resolve() for p in (OUT / "games").rglob("initial.json")}
        require(actual_folders == {game_folder(spec).resolve() for spec in schedule[:len(games)]}, "Extra or missing game outside the completed prefix")
        coverage = audit.validate_stopped_prefix(schedule, games, stop)
        require(coverage["status"] == "full_workload_verified" and len(games) == 200 and len(timing_rows) == 11020,
                "Recovery requires the complete recorded200-game/11020-decision workload")
        require(self.local("all_game_replay.json") == dict(status="passed", games=len(games), receipts=game_receipts),
                "Final game replay summary differs")
        performance = audit.performance(timing_rows, games)
        require(performance == self.local("performance.json"), "Recomputed timing distributions differ")
        boundary_summary = self.reconcile_boundaries(len(timing_rows), stop)
        sample = {**template, "decisions": [winners[cell] for cell in cells if cell in winners],
                  "empty_cells": [dict(zip(CELL_KEYS, cell)) for cell in cells if cell not in winners]}
        require(sample == self.local("replay_sample_manifest.json"), "Deterministic all-available-cell selection differs")
        selected = sample["decisions"]
        require(len(selected) == 48 and all(sum(x["player_role"] == role for x in selected) == 24 for role in audit.ROLES),
                "Recovery requires all48 recorded occupied replay cells")
        paths = sorted((OUT / "repeated_decisions").glob("*.json"))
        require([p.name for p in paths] == [f"{i:02d}.json" for i in range(len(selected))], "Saved reproduction inventory differs")
        for index, (item, path) in enumerate(zip(selected, paths, strict=True)):
            self.guard()
            original = winning_rows[tuple(item[k] for k in CELL_KEYS)]
            repeated = self.read(path)
            self.row_files(repeated, f"audit_{index:02d}")
            audit.validate_boundary_row(repeated)
            audit.check_clock(repeated["clock"])
            validate_saved_reproduction(original, repeated)
        require(self.local("repeated_decision_summary.json") == dict(status="passed", requests=len(selected), identities=selected,
                prefix_only=stop is not None), "Reproduction summary differs")
        self.checks.append("Every saved initial seed world and200 natural games: full science, dictionary ledger, raw-clock/file binding and isolated placement")
        self.checks.append("All11020 live timing receipts, descriptive inclusive700ms counts and all48 deterministic saved full-output reproductions")

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
        require(expected_names == {p.name for p in (OUT / "workers").iterdir() if p.is_dir()} and len(expected_names) == 450,
                "Exact worker lifecycle inventory differs")
        expected_graphs = {"qualification_64"}
        expected_graphs.update(f"{spec['stage']}_g{spec['game_index']}_challenger" for spec in schedule[:len(games)])
        expected_graphs.update(f"audit_{i:02d}" for i, item in enumerate(selected) if item["mode"] == "candidate")
        require(len(expected_graphs) == 225, "Recorded graph worker inventory differs")
        summaries = {key: self.local(name) for key, name in (
            ("logs", "all_worker_log_integrity.json"), ("raw", "all_raw_transport.json"),
            ("graphs", "all_graph_parity.json"), ("placement", "all_placement_integrity.json"))}
        receipts = {}
        for key, summary in summaries.items():
            expected = expected_graphs if key == "graphs" else expected_names
            receipts[key] = receipt_inventory(summary, expected)
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
            bind_closed_stream_receipts(folder, cleanup, log, raw, self.sealed)
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
                if (folder.name, path.stem) not in self.checked_placement_requests:
                    parent = self.read(folder / "parent_profiles" / path.name)
                    self.placement_request(folder, path.stem, parent)
            if name in startup_ids:
                require(startup_ids[name] in checked_ids, "Actual startup request absent from verified transport")
            if name in expected_graphs:
                graph = receipts["graphs"][name]
                identity, graph_clock = self.read(folder / "graph_identity.json"), self.read(folder / "graph_parity_clock.json")
                # This is the saved wrapper's summary, not a model invocation
                # or a second numerical graph-parity assessment.
                parity = self.read(folder / "graph_parity.json")
                bind_graph_receipt(graph, identity, graph_clock,
                    self.read(folder / (graph_clock["request_id"] + ".json")), parity["summary"], self.common.digest)
        require(worker_request_count == 2 + len(expected_graphs) + len(timing_rows) + len(selected),
                "All worker requests do not reconcile to startup/graph/live/reproduction requests")
        require(worker_request_count == 11295 and raw_failed_count == 0, "Recorded complete request/failure count differs")
        require(completion["automatic_retries"] == 0 and completion["larger_data_deferred"] is True and
                completion["no_strength_claim"] is True and completion["total_games_maximum"] == 200 and completion["main_games_maximum"] == 194,
                "Final policy/completion fields differ")
        self.authenticate_imports()
        self.checks.append("Exact worker/request/graph/startup inventories; saved full placement proofs; bound original raw/log/graph final-validator receipts")

        # Each original byte is authenticated at least once here. The owning
        # recovery orchestrator must perform its final full snapshot readback
        # AFTER every component; this component does not claim that final step.
        for item in self.original_snapshot.values():
            if item["path"] not in self.consumed:
                require(self.fingerprint(item["path"]) == item, "Original byte authentication differs")
        require(self.inventory(OUT) == set(self.original_snapshot), "Original inventory changed during review")
        absent = self.check_absent(self.owned)
        self.guard()
        self.checks.append("Every original snapshot artifact authenticated, including74528 sealed artifacts; exact inventory unchanged; all recorded owners absent")
        return dict(status="passed_saved_evidence_review", original_interruption=interrupted,
            original_completion_status=completion["status"], coverage=coverage,
            performance=performance, games=len(games), live_decisions=len(timing_rows), actual_entrypoint_decisions=104,
            reproduced_completed_decisions=len(selected), replay_cap=48, replay_populated_cells=len(selected),
            replay_empty_cells=len(sample["empty_cells"]), worker_count=len(expected_names), graph_workers=len(expected_graphs),
            worker_requests=worker_request_count, preserved_player_faults=raw_failed_count,
            boundary_failures=boundary_summary["failures"], sealed_artifacts=len(seal), sealed_bytes=sum(x["bytes"] for x in seal),
            imported_validator_sources=len(self.loaded_sources), owned_identities_absent=absent,
            original_charged_job_seconds=charged, original_exit_code=1, original_cleanup_errors=[],
            original_snapshot_artifacts=len(self.original_snapshot), original_snapshot_byte_authentication="passed",
            final_snapshot_readback="required_from_owning_recovery_pipeline_before_acceptance",
            completed_checks=self.checks, review_elapsed_seconds=time.perf_counter() - self.begun,
            consumed_inputs=sorted(self.consumed.values(), key=lambda item: item["path"]),
            no_overall_recovered_acceptance_claim=True, no_causal_or_strength_claim=True,
            initial_world_scope="Every saved seed-to-initial-world reconstruction checked; no gameplay.",
            graph_and_raw_scope="Original final-validator receipts bound to exact request inventories and every sealed byte; no new graph/native call, raw-log decompression or packet parser run.",
            boundary_scope="Each row validated; inclusive700ms counts reconciled. Boundary metric distributions retained from authenticated original summary, not separately recomputed.",
            full_source_runtime_and_prior_acceptance="required_from_other_owning_recovery_components")


def run_review(envelope, original_freeze, recovery_freeze):
    """Return a component receipt; propagate failure to the owning one-shot job.

    The owner supplies both real-clock guards and performs final snapshot
    readback after all components. This function never declares recovered
    acceptance on its own and never catches an exhausted overall deadline.
    """
    envelope.guard()
    seconds = min(1800., envelope.remaining())
    require(seconds > 0, "No remaining recovery allowance")
    result = Review(seconds, boundary=envelope.guard, original_freeze=original_freeze,
                    recovery_freeze=recovery_freeze).run()
    envelope.guard()
    result.update(schema="splendorust_reliability_recovery_saved_review_v1", freeze_sha256=EXPECTED_FREEZE,
                  scope="Saved data only; no model calls, new moves, games, decision repetitions, subprocesses or evidence writes.")
    return result

"""Runner integrity and paired arithmetic using synthetic records only."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import runner, common


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _clock(status="ok"):
    return {"request_id": "request-00000001", "clock": "perf_counter_ns", "status": status,
            "accepted": status == "ok", "deadline_seconds": 1.0, "elapsed_seconds": 1.1,
            "receipt_elapsed_seconds": .2 if status == "ok" else None,
            "received_before_deadline": status == "ok", "result": None, "detail": "Synthetic fixture"}


def test_clock_accepts_timely_receipt_even_when_validation_finishes_later():
    runner.check_clock(_clock(), "request-00000001")
    runner.check_clock(_clock("timeout"), "request-00000001")


@pytest.mark.parametrize("changes", [
    {"accepted": False}, {"accepted": 1}, {"received_before_deadline": False},
    {"deadline_seconds": 0}, {"deadline_seconds": 1.01}, {"deadline_seconds": True},
    {"elapsed_seconds": -.1}, {"receipt_elapsed_seconds": float("nan")},
    {"receipt_elapsed_seconds": 1.01, "received_before_deadline": False},
    {"elapsed_seconds": .1, "receipt_elapsed_seconds": .2},
    {"request_id": "request-00000002"}, {"clock": "monotonic_ns"},
])
def test_clock_rejects_inconsistent_flags_bounds_and_identity(changes):
    receipt = {**_clock(), **changes}
    with pytest.raises(ValueError):
        runner.check_clock(receipt, "request-00000001")


def test_schedule_has_30_distinct_worlds_and_exact_seat_reversals():
    schedule = runner.make_schedule()
    assert [row["game"] for row in schedule] == list(range(60))
    assert len({row["seed"] for row in schedule}) == 30
    assert sum(row["opponent_seat"] == 0 for row in schedule) == 30
    for pair in range(30):
        first, second = schedule[2 * pair:2 * pair + 2]
        assert first["pair"] == second["pair"] == pair
        assert first["seed"] == second["seed"]
        assert first["opponent_seat"] == pair % 2 and second["opponent_seat"] == 1 - first["opponent_seat"]
        expected = str(int(hashlib.sha256(f"{runner.NAMESPACE}:pair:{pair}".encode()).hexdigest(), 16))
        assert first["seed"] == expected


def _summary_games(outcomes):
    games = []
    for spec in runner.make_schedule():
        score = outcomes[spec["pair"]][spec["game"] % 2]
        initial = {"synthetic_world": spec["pair"]}
        winner = None if score == .5 else spec["opponent_seat"] if score == 1 else 1 - spec["opponent_seat"]
        games.append({**spec, "complete": True, "initial": initial,
                      "initial_sha256": runner.c.digest(initial), "winner": winner,
                      "opponent_score": score, "reason": "natural"})
    return games


def test_bootstrap_resamples_whole_pairs_not_individual_games():
    games = _summary_games([(1., 0.)] * 30)
    result = runner.summarize(games)
    assert result["opponent_mean_score"] == .5
    assert result["paired_bootstrap_95_percent_interval"] == [.5, .5]
    assert (result["opponent_wins"], result["draws"], result["opponent_losses"]) == (30, 0, 30)
    assert result["by_opponent_seat"] == {"0": .5, "1": .5}
    assert result["bootstrap_replicates"] == 50000


def test_paired_bootstrap_nonconstant_arithmetic():
    # Twenty winning pairs and ten losing pairs: bootstrap wins are Binomial(30, 2/3).
    # Its stable 2.5%/97.5% count quantiles are 15 and 25, rather than 60-game quantiles.
    games = _summary_games([(1., 1.)] * 20 + [(0., 0.)] * 10)
    result = runner.summarize(games)
    assert result["opponent_mean_score"] == pytest.approx(2 / 3)
    assert result["paired_bootstrap_95_percent_interval"] == [15 / 30, 25 / 30]
    assert (result["opponent_wins"], result["draws"], result["opponent_losses"]) == (40, 0, 20)


@pytest.mark.parametrize("corruption", ["score", "hash", "seed", "duplicate", "pair_world", "winner_bool", "incomplete"])
def test_summary_rejects_tampered_score_hash_and_pair_identity(corruption):
    games = _summary_games([(1., 0.)] * 30)
    if corruption == "score":
        games[0]["opponent_score"] = 0.
    elif corruption == "hash":
        games[0]["initial_sha256"] = "0" * 64
    elif corruption == "seed":
        games[0]["seed"] = "123"
    elif corruption == "duplicate":
        games[1] = deepcopy(games[0])
    elif corruption == "pair_world":
        games[0]["initial"] = {"synthetic_world": "another"}
        games[0]["initial_sha256"] = runner.c.digest(games[0]["initial"])
    elif corruption == "winner_bool":
        games[0]["winner"] = False
    else:
        games[0]["complete"] = False
    with pytest.raises(ValueError):
        runner.summarize(games)


@pytest.fixture
def single_forfeit(tmp_path, monkeypatch):
    """Only a seeded initial state and synthetic timeout; no action is played."""
    monkeypatch.setattr(runner.c, "ROOT", tmp_path)
    base = tmp_path / "synthetic-screen"
    _write(base / "opponent_profile.json", {"profile_id": "synthetic-fixed-work"})
    spec = runner.make_schedule()[0]
    state = rules.CorrectedEnv(seed=int(spec["seed"])).initial_state()
    initial = rules.serialize_snapshot(state)
    clock = _clock("timeout")
    row = {"ply": 0, "actor": 0, "player": "opponent", "before_sha256": runner.c.digest(initial),
           "observer": rules.observation_payload(state), "clock": deepcopy(clock)}
    folder = base / "games" / "000"
    row_path = folder / "moves" / "0000.json"
    worker_path = base / "workers" / "game-000" / "request-00000001.json"
    _write(row_path, row)
    _write(worker_path, clock)
    game = {**spec, "complete": True, "ruleset_id": rules.RULESET_ID,
            "catalogue_sha256": rules.CATALOGUE_SHA256, "initial": initial,
            "initial_sha256": runner.c.digest(initial), "final": deepcopy(initial),
            "moves": [{"file": row_path.relative_to(tmp_path).as_posix(), "sha256": runner.c.file_hash(row_path)}],
            "reason": "player_forfeit_timeout", "winner": 1, "failing_player": "opponent", "opponent_score": 0.}
    _write(folder / "initial.json", {**spec, "initial": initial})
    _write(folder / "result.json", game)
    return {"base": base, "spec": spec, "game": game, "row": row, "row_path": row_path,
            "folder": folder, "worker_path": worker_path, "root": tmp_path}


def _save_forfeit(fixture, *, bind_row=True):
    _write(fixture["row_path"], fixture["row"])
    if bind_row:
        fixture["game"]["moves"][0]["sha256"] = runner.c.file_hash(fixture["row_path"])
    _write(fixture["folder"] / "result.json", fixture["game"])


def test_single_forfeit_replay_binds_worker_receipt_without_invented_transition(single_forfeit):
    f = single_forfeit
    game, count, latency = runner.audit_game(f["base"], f["spec"])
    assert game == f["game"] and count == 0
    assert latency == {"opponent": [], "raw035": []}
    assert game["initial"] == game["final"]


@pytest.mark.parametrize("corruption", ["worker_receipt", "move_hash", "winner", "failing_player", "actor",
    "player", "ply", "chain", "request_id", "resource_as_forfeit", "support_as_forfeit", "reason", "move_after_forfeit"])
def test_forfeit_audit_rejects_receipt_attribution_and_progression_corruption(single_forfeit, corruption):
    f = single_forfeit
    if corruption == "worker_receipt":
        changed = deepcopy(f["row"]["clock"])
        changed["detail"] = "Different preserved receipt"
        _write(f["worker_path"], changed)
    elif corruption == "move_hash":
        f["game"]["moves"][0]["sha256"] = "0" * 64
    elif corruption == "winner":
        f["game"]["winner"], f["game"]["opponent_score"] = 0, 1.
    elif corruption == "failing_player":
        f["game"]["failing_player"] = "raw035"
    elif corruption in ("actor", "ply"):
        f["row"][corruption] = 1
    elif corruption == "player":
        f["row"]["player"] = "raw035"
    elif corruption == "chain":
        f["row"]["before_sha256"] = "0" * 64
    elif corruption == "request_id":
        f["row"]["clock"]["request_id"] = "request-00000002"
        _write(f["worker_path"], f["row"]["clock"])
    elif corruption == "resource_as_forfeit":
        f["row"]["clock"]["status"] = "rss_limit"
        f["game"]["reason"] = "player_forfeit_rss_limit"
        _write(f["worker_path"], f["row"]["clock"])
    elif corruption == "support_as_forfeit":
        f["row"]["clock"]["detail"] = "Empty adapter support"
        _write(f["worker_path"], f["row"]["clock"])
    elif corruption == "reason":
        f["game"]["reason"] = "natural"
    else:
        extra = deepcopy(f["row"])
        extra["ply"] = 1
        path = f["folder"] / "moves" / "0001.json"
        _write(path, extra)
        f["game"]["moves"].append({"file": path.relative_to(f["root"]).as_posix(), "sha256": runner.c.file_hash(path)})
    _save_forfeit(f, bind_row=corruption != "move_hash")
    with pytest.raises(ValueError):
        runner.audit_game(f["base"], f["spec"])


def test_final_source_verification_overrides_completed_workflow(tmp_path, monkeypatch):
    base = tmp_path / "synthetic-finalization"
    _write(base / 'qualification/execution_start.json', {'execution_cap_seconds': 14400,
        'monotonic_start': runner.time.monotonic(), 'time': runner.c.now()})
    monkeypatch.setattr(runner.c, 'artifact', lambda path: {'path': str(path), 'sha256': runner.c.file_hash(path), 'bytes': path.stat().st_size})
    _write(base / "freeze.json", {"inputs": [], "topology": {"selected_logical_processor": 0},
        'execution_start': runner.c.artifact(base / 'qualification/execution_start.json')})
    _write(base / "schedule.json", [])
    after_workflow = False

    def verify(_items):
        if after_workflow:
            raise ValueError("Synthetic final source mutation")

    def boundary(*_args, **_kwargs):
        nonlocal after_workflow
        after_workflow = True

    monkeypatch.setattr(runner.c, "verify", verify)
    monkeypatch.setattr(runner.c, "execution_lease", nullcontext)
    monkeypatch.setattr(runner.c, "boundary", boundary)
    monkeypatch.setattr(runner, "qualify", lambda *_: {"passed": True, "synthetic": True})
    monkeypatch.setattr(runner, "summarize", lambda games: {"synthetic_games": len(games)})
    monkeypatch.setattr(runner, "audit_all_games", lambda *_: None)
    monkeypatch.setattr(runner, "play_game", lambda *_: pytest.fail("No gameplay permitted in this test"))
    assert runner.run(base) == 1
    completion = runner.c.read(base / "completion.json")
    assert completion["status"] == "failed" and completion["frozen_inputs_unchanged"] is False
    assert "Synthetic final source mutation" in completion["failure"]
    assert (base / "final_source_failure.json").exists()


def test_new_schedule_never_reuses_first_external_screen_seeds():
    from tools.external_strength_screen_20261018 import runner as prior
    assert not {r['seed'] for r in runner.make_schedule()} & {r['seed'] for r in prior.make_schedule()}


def test_profile_binds_artifact_hash_size_and_work_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    artifact = tmp_path / 'synthetic.bin'
    artifact.write_bytes(b'synthetic-no-model')
    profile = {'schema': common.PROFILE_SCHEMA, 'opponent_id': common.OPPONENT_ID,
               'profile_id': 'synthetic', 'deterministic_fixed_work': True,
               'internal_budget_seconds': .85, 'artifacts': [
                   {'path': artifact.name, 'sha256': common.file_hash(artifact), 'bytes': artifact.stat().st_size}]}
    path = tmp_path / 'profile.json'
    _write(path, profile)
    assert common.load_profile(path) == profile
    artifact.write_bytes(b'different')
    with pytest.raises(ValueError, match='Changed bound artifact'):
        common.load_profile(path)


class _Process:
    def __init__(self, pid, rss, cpu, affinity=None, role=None, parent_pid=1):
        self.pid, self.rss, self.cpu = pid, rss, cpu
        self.role = role or ('python' if pid == 1 else 'rust')
        self.parent_pid = parent_pid
        self.affinity = [2] if affinity is None else affinity
        self.children_list = []
    def create_time(self): return float(self.pid)
    def children(self, recursive): return self.children_list
    def cpu_times(self):
        from types import SimpleNamespace
        return SimpleNamespace(user=self.cpu, system=0.)
    def cpu_affinity(self, value=None):
        if value is not None: self.affinity = list(value)
        return self.affinity
    def memory_info(self):
        from types import SimpleNamespace
        return SimpleNamespace(rss=self.rss)
    def num_threads(self): return 1
    def exe(self): return str(Path('synthetic-processes/' + {'python': 'python.exe', 'rust': 'bridge.exe', 'console': 'conhost.exe', 'other': 'other.exe'}[self.role]).resolve())
    def name(self): return Path(self.exe()).name
    def ppid(self): return self.parent_pid


def _tree_worker(parent):
    worker = object.__new__(common.ProcessTreeWorker)
    worker._ps, worker.affinity = parent, [2]
    worker._tree_cpu, worker._tree_identities = {}, set()
    worker._tree_roles, worker._role_owners = {}, {}
    worker._console_startup_affinity, worker._initializing = {}, False
    worker._python_executable = Path('synthetic-processes/python.exe').resolve()
    worker._native_executable = Path('synthetic-processes/bridge.exe').resolve()
    worker._console_executable = Path('synthetic-processes/conhost.exe').resolve()
    worker._expected_executables = {p: 'synthetic-hash' for p in (worker._python_executable, worker._native_executable, worker._console_executable)}
    worker._verified_executables = dict(worker._expected_executables)
    worker._last_telemetry, worker._peak_rss = {}, 0
    worker._failure, worker._failure_detail = None, None
    worker.max_rss_bytes = 100
    return worker


def test_combined_descendant_memory_and_cpu_retained_after_child_exits():
    parent, child = _Process(1, 20, 2.), _Process(2, 50, 3.)
    parent.children_list = [child]
    worker = _tree_worker(parent)
    first = worker._sample()
    assert first['rss_bytes'] == first['peak_sampled_rss_bytes'] == 70
    assert first['cpu_seconds'] == 5 and first['processes_observed'] == 2
    parent.children_list, parent.cpu = [], 4.
    second = worker._sample()
    assert second['rss_bytes'] == 20 and second['peak_sampled_rss_bytes'] == 70
    assert second['cpu_seconds'] == 7 and worker._failure is None


@pytest.mark.parametrize('reason', ['rss', 'affinity', 'respawn'])
def test_descendant_guard_fails_closed(reason):
    parent, child = _Process(1, 20, 1.), _Process(2, 50, 2.)
    parent.children_list = [child]
    worker = _tree_worker(parent)
    worker._sample()
    if reason == 'rss': child.rss = 90
    elif reason == 'affinity': child.affinity = [2, 3]
    else: parent.children_list = [_Process(3, 30, 1.)]
    worker._sample()
    assert worker._failure == {'rss': 'rss_limit', 'affinity': 'affinity_changed', 'respawn': 'process_tree_limit'}[reason]


def test_only_exact_console_helpers_allowed_and_counted():
    parent, child = _Process(1, 20, 1.), _Process(2, 30, 2.)
    console_python = _Process(3, 7, .2, role='console', parent_pid=1)
    console_rust = _Process(4, 7, .3, role='console', parent_pid=2)
    parent.children_list = [console_python, console_rust, child]
    worker = _tree_worker(parent)
    row = worker._sample()
    assert worker._failure is None
    assert row['rss_bytes'] == 64 and row['cpu_seconds'] == 3.5 and row['processes_observed'] == 4
    assert {p['role'] for p in row['process_tree']} == {'python', 'rust', 'console:python', 'console:rust'}
    parent.children_list[0] = _Process(5, 7, .1, role='console', parent_pid=1)
    worker._sample()
    assert worker._failure == 'process_tree_limit'


def test_console_affinity_repaired_once_only_during_initialization():
    parent = _Process(1, 20, 1.)
    helper = _Process(2, 7, .1, affinity=[0, 1, 2, 3], role='console', parent_pid=1)
    parent.children_list = [helper]
    worker = _tree_worker(parent)
    worker._initializing = True
    receipt = worker._sample()
    assert worker._failure is None and helper.affinity == [2]
    console = receipt['process_tree'][1]
    assert console['startup_affinity_before'] == [0, 1, 2, 3]
    assert console['startup_affinity_after'] == [2] and console['startup_affinity_adjusted'] is True
    worker._initializing = False
    helper.affinity = [1, 2]
    worker._sample()
    assert worker._failure == 'affinity_changed'


def test_console_affinity_never_repaired_after_ready():
    parent = _Process(1, 20, 1.)
    helper = _Process(2, 7, .1, affinity=[1, 2], role='console', parent_pid=1)
    parent.children_list = [helper]
    worker = _tree_worker(parent)
    worker._sample()
    assert helper.affinity == [1, 2] and worker._failure == 'affinity_changed'


@pytest.mark.parametrize('role,parent_pid', [('other', 1), ('console', 99), ('python', 1)])
def test_unknown_executable_or_console_owner_rejected(role, parent_pid):
    parent = _Process(1, 20, 1.)
    parent.children_list = [_Process(2, 10, 1., role=role, parent_pid=parent_pid)]
    worker = _tree_worker(parent)
    worker._sample()
    assert worker._failure == 'process_tree_limit'


def test_common_execution_deadline_includes_qualification(tmp_path, monkeypatch):
    started = runner.time.monotonic() - 100
    from datetime import datetime, timedelta, timezone
    _write(tmp_path / 'qualification/execution_start.json', {'execution_cap_seconds': 14400,
        'monotonic_start': started, 'time': (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()})
    _, start, deadline, elapsed = runner.execution_window(tmp_path)
    assert start == started and deadline == started + 14400 and 100 <= elapsed < 101
    receipt = runner.c.read(tmp_path / 'qualification/execution_start.json')
    receipt['monotonic_start'] += 200
    _write(tmp_path / 'qualification/execution_start.json', receipt)
    with pytest.raises(ValueError, match='Expired or inconsistent'):
        runner.execution_window(tmp_path)


def _synthetic_diagnostics(turn=0):
    from tools import analyze_external_strength_screen_20261018_02 as independent
    observer = {'observation': {'turn_index': turn}}
    legal = [{'action_type': 'TAKE_TOKENS', 'take_tokens': ['white', 'blue', 'green'], 'return_tokens': [], 'noble_id': None},
             {'action_type': 'TAKE_TOKENS', 'take_tokens': ['white', 'blue', 'red'], 'return_tokens': [], 'noble_id': None}]
    profile = {'profile_id': 'synthetic', 'iterations': 16}
    heuristic = turn >= 124
    d = {'opponent_id': 'splendorust', 'profile_id': 'synthetic', 'observer_seed': int(independent.digest(observer)[:16], 16),
         'phases': [{'action': independent.main_action(legal[0]), 'simulations': 0 if heuristic else 16,
                     'leaf_calls': 0 if heuristic else 17, 'native_legal_count': 2, 'heuristic_phase': heuristic}],
         'inference_calls': 0 if heuristic else 17, 'inference_rows': 0 if heuristic else 17, 'forced_pass': False}
    return independent, d, observer, legal, profile


def test_independent_diagnostics_validate_fixed_work_and_late_heuristic():
    for turn in (0, 124, 180):
        audit, d, observer, legal, profile = _synthetic_diagnostics(turn)
        got = audit.check_diagnostics(d, observer, legal[0], legal, profile)
        assert got['searched_main_decisions'] == int(turn < 124)
        assert got['late_heuristic_main_decisions'] == int(turn >= 124)


@pytest.mark.parametrize('field,value', [('simulations', 15), ('simulations', True), ('leaf_calls', 18),
                                      ('leaf_calls', 0), ('heuristic_phase', True), ('native_legal_count', 1)])
def test_independent_diagnostics_reject_impossible_work(field, value):
    audit, d, observer, legal, profile = _synthetic_diagnostics()
    d['phases'][0][field] = value
    with pytest.raises(ValueError):
        audit.check_diagnostics(d, observer, legal[0], legal, profile)


def test_independent_diagnostics_bind_seed_output_and_inference_counts():
    for mutate in ('seed', 'action', 'rows', 'phase'):
        audit, d, observer, legal, profile = _synthetic_diagnostics()
        if mutate == 'seed': d['observer_seed'] += 1
        elif mutate == 'action': d['phases'][0]['action'] = audit.main_action(legal[1])
        elif mutate == 'rows': d['inference_rows'] += 1
        else: d['phases'].append(deepcopy(d['phases'][0]))
        with pytest.raises(ValueError):
            audit.check_diagnostics(d, observer, legal[0], legal, profile)

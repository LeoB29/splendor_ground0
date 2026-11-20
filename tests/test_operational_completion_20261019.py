"""Coordinator tests use fake child handles and synthetic operational evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tools import operational_completion_20261019 as c


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path); base = tmp_path/'outputs/stage'; base.mkdir(parents=True)
    monkeypatch.setattr(c, 'BASE', base)
    source = tmp_path/'tools/coordinator.py'; analyzer = tmp_path/'tools/frozen_analyzer.py'
    report = tmp_path/'docs/result.md'; report.parent.mkdir()
    for path in (source, analyzer): path.parent.mkdir(exist_ok=True); path.write_text('# synthetic identity only', encoding='utf-8')
    monkeypatch.setattr(c, 'SOURCE', source); monkeypatch.setattr(c, 'ANALYZER', analyzer); monkeypatch.setattr(c, 'REPORT', report)
    freeze = {'executable': sys.executable, 'inputs': [c.artifact(analyzer)]}
    write(base/'freeze.json', freeze); monkeypatch.setattr(c, 'FREEZE_SHA256', c.sha(base/'freeze.json'))
    write(base/'launch.json', {'epoch_seconds': c.EPOCH, 'monotonic_start': c.MONOTONIC, 'maximum_seconds': c.CAP, 'pid': c.RUNNER['pid']})
    command = [sys.executable, '-B', '-u', '-m', 'tools.live_game_operational_20261019.runner', 'run']
    write(base/'runner_process.json', {'identity': c.RUNNER, 'command': command, 'freeze_sha256': c.FREEZE_SHA256, 'outcomes_closed': True})
    write(base/'coordinator_plan.json', c.make_plan())
    coordinator = c.Coordinator(c.sha(source), c.FREEZE_SHA256, c.sha(base/'coordinator_plan.json'))
    return coordinator, c.read(base/'coordinator_plan.json')


def completion_files():
    process = c.read(c.BASE/'runner_process.json')
    execution = {'runner_identity': c.RUNNER, 'command': process['command'], 'exit_code': 0, 'outcomes_closed': True,
                 'started_epoch_seconds': c.EPOCH-10, 'finished_utc': c.now(), 'elapsed_seconds': 300}
    complete = {'status': 'completed_pending_independent_analysis', 'failure': None, 'completed_games': 18,
                'worker_processes_closed': True, 'outcomes_closed': True}
    write(c.BASE/'runner_execution.json', execution); write(c.BASE/'completion.json', complete)
    write(c.BASE/'final_input_check.json', {'all_hashes_and_sizes_match': True})
    return execution


def accepted():
    timings = {'mean': 20., 'p95': 40., 'max': 60.}
    return {'status': 'passed', 'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True, 'freeze_sha256': c.FREEZE_SHA256,
        'source_preservation': {'files': 100, 'bytes': 200, 'all_hashes_and_sizes_match': True},
        'operational_schedule': {'games': 18, 'pairs': 9, 'actual_dictionary_transitions': 600},
        'per_stage': {key: {'attempted': 300, 'statuses': {'completed': 300}, 'completed_receipt_seconds': timings}
                      for key in ('games:local', 'games:opponent')},
        'validation': {'requests': 12, 'exact_hidden_state_repeat_pairs': 6},
        'actual_continuation_verification': {'roots': 300, 'full_stress_roots': 30, 'sample_only_roots': 270,
            'recorded_worlds': 38400, 'recorded_continuations': 153600,
            'covered_cells': 35, 'cells': 81, 'replayed_union': {'worlds': 6000, 'continuations': 24000, 'transitions': 800000}},
        'resources': {'production_elapsed_seconds': 40000, 'including_final_analysis_seconds': 40100,
            'peak_sampled_combined_rss_bytes': 300*1024**2, 'evidence_bytes_before_this_receipt': 1024**3}}


def test_prospective_plan_pins_commands_boundaries_and_source(workspace):
    coordinator, plan = workspace
    assert coordinator.authenticate() == plan
    assert plan['maximum_seconds'] == 86400 and plan['final_reserve_seconds'] == 600 and plan['cpu'] == 2
    assert plan['automatic_retries'] == 0 and plan['analysis_attempts'] == 1
    assert plan['analyzer_command'][0] == sys.executable


@pytest.mark.parametrize('damage', ['source', 'freeze', 'plan', 'analyzer', 'runner_identity', 'launch'])
def test_changed_prospective_identity_is_rejected_before_any_launch(workspace, damage):
    coordinator, _ = workspace
    if damage in ('source', 'analyzer'):
        (c.SOURCE if damage == 'source' else c.ANALYZER).write_text('changed', encoding='utf-8')
    else:
        name = {'freeze': 'freeze.json', 'plan': 'coordinator_plan.json', 'runner_identity': 'runner_process.json', 'launch': 'launch.json'}[damage]
        value = c.read(c.BASE/name); value['changed'] = True
        if damage == 'runner_identity': value['identity']['created'] += 1
        if damage == 'launch': value['epoch_seconds'] += 1
        write(c.BASE/name, value)
    with pytest.raises(c.Hold): coordinator.authenticate()
    assert not (c.BASE/'coordinator_launch.json').exists()


@pytest.mark.parametrize('damage', ['exit', 'identity', 'command', 'incomplete', 'failure', 'alive'])
def test_runner_receipt_and_closed_identity_are_required(workspace, monkeypatch, damage):
    execution = completion_files(); monkeypatch.setattr(c, 'alive', lambda _: damage == 'alive')
    if damage == 'exit': execution['exit_code'] = 1
    elif damage == 'identity': execution['runner_identity'] = {**c.RUNNER, 'created': 1.}
    elif damage == 'command': execution['command'] = ['untrusted']
    elif damage in ('incomplete', 'failure'):
        complete = c.read(c.BASE/'completion.json')
        if damage == 'incomplete': complete['completed_games'] = 17
        else: complete['failure'] = {'category': 'hold'}
        write(c.BASE/'completion.json', complete)
    with pytest.raises(c.Hold): c.check_runner(execution)


def test_runner_receipt_delivery_waits_for_partial_json_only(workspace, monkeypatch):
    expected = completion_files(); path = c.BASE/'runner_execution.json'
    path.write_text('{', encoding='utf-8'); sleeps = []
    monkeypatch.setattr(c, 'resources', lambda **_: {})
    monkeypatch.setattr(c, 'remaining', lambda: 1000)
    def complete_delivery(seconds):
        sleeps.append(seconds); write(path, expected)
    monkeypatch.setattr(c.time, 'sleep', complete_delivery)
    assert c.delivered_runner_execution() == expected and sleeps == [.1]


def test_persistent_malformed_runner_receipt_is_bounded_and_held(workspace, monkeypatch):
    (c.BASE/'runner_execution.json').write_text('{', encoding='utf-8'); clock = [100.]; sleeps = []
    monkeypatch.setattr(c, 'resources', lambda **_: {})
    monkeypatch.setattr(c, 'remaining', lambda: 1000)
    monkeypatch.setattr(c.time, 'monotonic', lambda: clock[0])
    def advance(seconds): sleeps.append(seconds); clock[0] += seconds
    monkeypatch.setattr(c.time, 'sleep', advance)
    with pytest.raises(c.Hold, match='runner_execution_delivery_incomplete'): c.delivered_runner_execution()
    assert max(sleeps) <= .1 and sum(sleeps) == pytest.approx(5.)
    assert not (c.BASE/'independent_analysis_execution_01.log').exists()


def test_valid_json_semantic_failure_has_no_delivery_retry(workspace, monkeypatch):
    completion_files(); write(c.BASE/'runner_execution.json', {'invalid': True})
    monkeypatch.setattr(c.time, 'sleep', lambda _: pytest.fail('Semantic errors must not be retried'))
    execution = c.delivered_runner_execution()
    with pytest.raises(KeyError): c.check_runner(execution)


def test_delivery_wait_retains_original_deadline_reserve(workspace, monkeypatch):
    (c.BASE/'runner_execution.json').write_text('{', encoding='utf-8')
    def deadline(**_): raise c.Hold('original_deadline_reserve')
    monkeypatch.setattr(c, 'resources', deadline)
    monkeypatch.setattr(c.time, 'sleep', lambda _: pytest.fail('No wait after original deadline reserve'))
    with pytest.raises(c.Hold, match='original_deadline_reserve'): c.delivered_runner_execution()


def test_wall_and_monotonic_deadlines_both_retain_original_boundary(monkeypatch):
    monkeypatch.setattr(c.time, 'time', lambda: c.EPOCH+10)
    monkeypatch.setattr(c.time, 'monotonic', lambda: c.MONOTONIC+100)
    assert c.remaining() == 86300
    monkeypatch.setattr(c.time, 'time', lambda: c.EPOCH+86500)
    assert c.remaining() == -100


@pytest.mark.parametrize('fault', ['deadline', 'disk', 'evidence'])
def test_resource_limits_stop_before_final_reserve(workspace, monkeypatch, fault):
    monkeypatch.setattr(c, 'remaining', lambda: 600 if fault == 'deadline' else 1000)
    monkeypatch.setattr(c.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0 if fault == 'disk' else c.MIN_FREE))
    if fault == 'evidence': monkeypatch.setattr(c, 'MAX_BYTES', 32*1024**2)
    with pytest.raises(c.Hold): c.resources()


def test_report_reads_only_operational_allowlist_and_disclaims_strength(workspace):
    analysis = accepted(); analysis['sealed_winner_payload'] = 'DO_NOT_DISCLOSE'
    report = c.operational_report(analysis)
    assert 'DO_NOT_DISCLOSE' not in report and 'zero strength observations' in report
    assert 'not full replay' in report and 'tail-latency' in report and 'absent cells remain untested' in report
    assert '38400 recorded worlds' in report and '153600 recorded continuations' in report and 'shares frozen model' in report
    assert 'eighteen-game' in report and 'nine swapped-seat pairs' in report


@pytest.mark.parametrize('damage', ['status', 'closed', 'strength', 'wrong_stage', 'unexpected_status'])
def test_unaccepted_or_unexpected_analysis_is_not_reported(workspace, damage):
    value = accepted()
    if damage == 'status': value['status'] = 'failed'
    elif damage == 'closed': value['outcomes_closed'] = False
    elif damage == 'strength': value['strength_analysis_games'] = 18
    elif damage == 'wrong_stage': value['operational_schedule']['games'] = 6
    else: value['per_stage']['games:local']['statuses'] = {'untrusted': 1}
    with pytest.raises(c.Hold): c.operational_report(value)


class FakeChild:
    pid = 999999
    def __init__(self, code=0): self.returncode = None; self.code = code; self.killed = False; self.waited = 0
    def poll(self): return self.returncode
    def wait(self, timeout): self.waited += 1; self.returncode = self.code; return self.returncode
    def kill(self): self.killed = True; self.code = -9


def fake_process(pid=None):
    return SimpleNamespace(create_time=lambda: 123., children=lambda recursive: [],
        memory_info=lambda: SimpleNamespace(rss=1024), cpu_affinity=lambda values: None)


@pytest.mark.parametrize('outcome', ['success', 'failed_exit', 'guard_hold', 'exit_race'])
def test_exactly_one_owned_child_and_preserved_execution_receipt(workspace, monkeypatch, outcome):
    coordinator, plan = workspace; write(c.BASE/'coordinator_launch.json', {'claimed': True})
    monkeypatch.setattr(c, 'resources', lambda **_: {})
    monkeypatch.setattr(c, 'remaining', lambda: 1000)
    monkeypatch.setattr(c, 'alive', lambda identity: False)
    child = FakeChild(1 if outcome == 'failed_exit' else 0); calls = []
    def popen(command, **kwargs):
        calls.append(command)
        if outcome in ('success', 'exit_race'): write(c.BASE/'independent_analysis.json', accepted())
        return child
    monkeypatch.setattr(c.subprocess, 'Popen', popen)
    observed = [0]
    def process(pid=None):
        observed[0] += 1
        if observed[0] > 1 and outcome == 'guard_hold': raise c.Hold('synthetic_guard')
        if observed[0] > 1 and outcome == 'exit_race':
            child.returncode = 0; raise c.psutil.NoSuchProcess(child.pid)
        return fake_process(pid)
    monkeypatch.setattr(c.psutil, 'Process', process)
    if outcome in ('success', 'exit_race'): assert coordinator.analyze(plan)['status'] == 'passed'
    else:
        with pytest.raises(c.Hold): coordinator.analyze(plan)
    assert len(calls) == 1
    receipt = c.read(c.BASE/'independent_analysis_execution_01.json')
    assert receipt['analyzer_process_closed'] and receipt['analyzer_process'] == {'pid': child.pid, 'created': 123.}
    assert child.killed is (outcome == 'guard_hold')
    with pytest.raises(c.Hold, match='already_attempted'): coordinator.analyze(plan)
    assert len(calls) == 1


def test_successful_shell_chain_is_exclusive_and_does_not_rewrite_handoffs(workspace, monkeypatch):
    coordinator, _ = workspace; completion_files()
    monkeypatch.setattr(c, 'resources', lambda **_: {'remaining_seconds': 1000})
    monkeypatch.setattr(c, 'alive', lambda _: False)
    monkeypatch.setattr(c.psutil, 'Process', fake_process)
    def analyzed(plan):
        value = accepted(); write(c.BASE/'independent_analysis.json', value)
        write(c.BASE/'independent_analysis_execution_01.json', {'analyzer_process_closed': True})
        (c.BASE/'independent_analysis_execution_01.log').write_text('passed', encoding='utf-8')
        return value
    monkeypatch.setattr(coordinator, 'analyze', analyzed)
    coordinator.run()
    final = c.read(c.BASE/'finalization.json')
    assert final['status'] == 'operational_checks_passed_pending_finalization_validation' and final['handoffs_rewritten'] is False
    assert final['games'] == 18 and final['pairs'] == 9 and c.REPORT.exists()
    assert final['verified_actual_transitions'] == 600 and final['attempted_actual_decisions'] == 600
    assert final['larger_data_deferred'] is True
    assert c.read(c.BASE/'finalization_validation.json')['status'] == 'passed'
    before = (c.BASE/'coordinator_launch.json').read_bytes()
    with pytest.raises(c.Hold): coordinator.run()
    assert (c.BASE/'coordinator_launch.json').read_bytes() == before


def test_hold_keeps_exact_diagnostic_sealed_and_never_creates_success(workspace):
    coordinator, _ = workspace; coordinator.claimed = True
    write(c.BASE/'coordinator_launch.json', {'claimed': True})
    try: raise ValueError('SEALED_SYNTHETIC_DETAILS')
    except ValueError: coordinator.preserve_hold()
    assert 'SEALED_SYNTHETIC_DETAILS' in c.read(c.BASE/'sealed/coordinator_failure_01.json')['traceback']
    assert 'SEALED_SYNTHETIC_DETAILS' not in (c.BASE/'coordinator_hold.json').read_text(encoding='utf-8')
    assert not c.REPORT.exists() and not (c.BASE/'finalization.json').exists()


def test_cleanup_uses_recorded_creation_time_and_never_kills_reused_pid(monkeypatch):
    child = FakeChild(); killed = []; waited = []
    owner = {'pid': child.pid, 'created': 123.}
    descendants = {(222, 124.): {'pid': 222, 'created': 124.}, (333, 125.): {'pid': 333, 'created': 125.}}
    def process(pid):
        return SimpleNamespace(pid=pid, create_time=lambda: {child.pid:123., 222:124., 333:999.}[pid],
            children=lambda recursive: [], kill=lambda:killed.append(pid), wait=lambda timeout:waited.append(pid))
    monkeypatch.setattr(c.psutil, 'Process', process)
    c.close_owned(child, owner, descendants)
    assert killed == [222] and waited == [222] and child.killed


def test_unexpected_descendant_is_captured_and_cleaned_on_guard_hold(workspace, monkeypatch):
    coordinator, plan = workspace; write(c.BASE/'coordinator_launch.json', {'claimed': True})
    monkeypatch.setattr(c, 'resources', lambda **_: {}); monkeypatch.setattr(c, 'remaining', lambda: 1000)
    child = FakeChild(); killed = []; descendant = SimpleNamespace(pid=222, create_time=lambda:124.)
    def process(pid=None):
        return SimpleNamespace(pid=pid, create_time=lambda:123. if pid == child.pid else 124.,
            children=lambda recursive:[descendant], memory_info=lambda:SimpleNamespace(rss=10),
            kill=lambda:killed.append(pid), wait=lambda timeout:None)
    monkeypatch.setattr(c.psutil, 'Process', process); monkeypatch.setattr(c.subprocess, 'Popen', lambda *args, **kwargs:child)
    monkeypatch.setattr(c, 'alive', lambda _:False)
    with pytest.raises(c.Hold): coordinator.analyze(plan)
    receipt = c.read(c.BASE/'independent_analysis_execution_01.json')
    assert receipt['owned_descendants'] == [{'pid':222, 'created':124.}] and killed == [222]
    assert receipt['owned_descendants_closed'] and child.killed


def test_cleanup_wait_error_does_not_suppress_execution_receipt(workspace, monkeypatch):
    coordinator, plan = workspace; write(c.BASE/'coordinator_launch.json', {'claimed': True})
    monkeypatch.setattr(c, 'resources', lambda **_: {}); monkeypatch.setattr(c, 'remaining', lambda:1000)
    child = FakeChild(); observed = [0]
    def process(pid=None):
        observed[0] += 1
        if observed[0] > 1: raise c.Hold('guard_trigger')
        return fake_process(pid)
    def failed_wait(timeout): raise c.subprocess.TimeoutExpired('fake child', timeout)
    child.wait = failed_wait
    monkeypatch.setattr(c.psutil, 'Process', process); monkeypatch.setattr(c.subprocess, 'Popen', lambda *args, **kwargs:child)
    monkeypatch.setattr(c, 'alive', lambda _:True)
    with pytest.raises(c.Hold): coordinator.analyze(plan)
    receipt = c.read(c.BASE/'independent_analysis_execution_01.json')
    assert receipt['cleanup_error'] == 'Hold' and receipt['analyzer_process_closed'] is False and child.killed


def test_oversized_analysis_log_causes_preserved_hold(workspace, monkeypatch):
    coordinator, plan = workspace; write(c.BASE/'coordinator_launch.json', {'claimed': True})
    monkeypatch.setattr(c, 'resources', lambda **_: {}); monkeypatch.setattr(c, 'remaining', lambda:1000)
    monkeypatch.setattr(c, 'MAX_LOG', 2); monkeypatch.setattr(c.psutil, 'Process', fake_process)
    monkeypatch.setattr(c, 'alive', lambda _:False); child = FakeChild()
    plan = c.make_plan(); write(c.BASE/'coordinator_plan.json', plan)
    coordinator.expected_plan = c.sha(c.BASE/'coordinator_plan.json')
    def popen(*args, **kwargs): kwargs['stdout'].write(b'oversized'); kwargs['stdout'].flush(); return child
    monkeypatch.setattr(c.subprocess, 'Popen', popen)
    with pytest.raises(c.Hold): coordinator.analyze(plan)
    assert child.killed and c.read(c.BASE/'independent_analysis_execution_01.json')['error'] == 'Hold'


@pytest.mark.parametrize('failure_check', [4, 5])
def test_postwrite_failure_cannot_leave_unqualified_success(workspace, monkeypatch, failure_check):
    coordinator, _ = workspace; completion_files(); calls = [0]
    def resource(**kwargs):
        calls[0] += 1
        if calls[0] == failure_check: raise c.Hold('postwrite_bound')
        return {'remaining_seconds':1000}
    monkeypatch.setattr(c, 'resources', resource); monkeypatch.setattr(c, 'alive', lambda _:False)
    monkeypatch.setattr(c.psutil, 'Process', fake_process)
    def analyzed(plan):
        value = accepted(); write(c.BASE/'independent_analysis.json', value)
        write(c.BASE/'independent_analysis_execution_01.json', {'analyzer_process_closed':True})
        (c.BASE/'independent_analysis_execution_01.log').write_text('passed', encoding='utf-8')
        return value
    monkeypatch.setattr(coordinator, 'analyze', analyzed)
    try: coordinator.run()
    except c.Hold: coordinator.preserve_hold()
    else: pytest.fail('Expected a postwrite resource hold')
    assert c.read(c.BASE/'finalization.json')['status'] == 'operational_checks_passed_pending_finalization_validation'
    assert (c.BASE/'coordinator_hold.json').exists()
    assert (c.BASE/'finalization_validation.json').exists() is (failure_check == 5)

"""Coordinator tests use fake child handles and synthetic operational evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tools import operational_recovery_completion_20261020_02 as c

REAL_ADMISSION = c.analyzer_admission


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fake_admission(plan):
    log = c.BASE/'memory_admissions/analysis.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True); log.write_text('{"synthetic":true}\n', encoding='utf-8')
    value = {'schema': 'operational_memory_admission_v2', 'status': 'passed', 'name': 'analysis',
             'reserve_seconds': 600, 'request_seconds': 0, 'initial': False,
             'no_worker_during_admission': True, 'outcomes_closed': True,
             'charged_seconds': 1., 'cumulative_after_seconds': 1200.,
             'readings_log': {'path': log.relative_to(c.BASE).as_posix(), 'sha256': c.sha(log), 'bytes': log.stat().st_size}}
    path = c.BASE/'memory_admissions/analysis.json'; write(path, value)
    return {'receipt': c.artifact(path), 'readings_log': c.artifact(log),
            'charged_seconds': 1., 'cumulative_after_seconds': 1200.}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path); base = tmp_path/'outputs/stage'; base.mkdir(parents=True)
    monkeypatch.setattr(c, 'BASE', base)
    source = tmp_path/'tools/coordinator.py'; analyzer = tmp_path/'tools/frozen_analyzer.py'; memory = tmp_path/'tools/memory.py'
    report = tmp_path/'docs/result.md'; report.parent.mkdir()
    for path in (source, analyzer, memory): path.parent.mkdir(exist_ok=True); path.write_text('# synthetic identity only', encoding='utf-8')
    monkeypatch.setattr(c, 'SOURCE', source); monkeypatch.setattr(c, 'ANALYZER', analyzer); monkeypatch.setattr(c, 'REPORT', report)
    monkeypatch.setattr(c, 'MEMORY_SOURCE', memory)
    monkeypatch.setattr(c, 'analyzer_admission', fake_admission)
    original = tmp_path/'outputs/original'; monkeypatch.setattr(c, 'ORIGINAL_BASE', original)
    write(original/'freeze.json', {'synthetic': 'original freeze'})
    monkeypatch.setattr(c, 'ORIGINAL_FREEZE_SHA256', c.sha(original/'freeze.json'))
    write(original/'launch.json', {'epoch_seconds': 1000.})
    write(original/'completion.json', {'status': 'preserved_hold', 'completed_games': 5,
          'failure': {'category': 'ResourceHold'}, 'validation_passed': True, 'worker_processes_closed': True,
          'outcomes_closed': True, 'elapsed_seconds': 10500.937})
    write(original/'runner_execution.json', {'exit_code': 1, 'outcomes_closed': True,
          'runner_identity': {'pid': 999998, 'created': 990.}})
    write(original/'coordinator_hold.json', {'status': 'preserved_hold'})
    write(original/'final_input_check.json', {'all_hashes_and_sizes_match': True})
    prior = tmp_path/'outputs/prior'; monkeypatch.setattr(c, 'PRIOR_BASE', prior)
    write(prior/'freeze.json', {'synthetic': 'prior recovery freeze'})
    monkeypatch.setattr(c, 'PRIOR_FREEZE_SHA256', c.sha(prior/'freeze.json'))
    write(prior/'launch.json', {'epoch_seconds': 20000.})
    write(prior/'completion.json', {'status': 'preserved_hold', 'completed_games': 16,
          'imported_completed_games': 5, 'validation_reused': True, 'failure': {'category': 'ResourceHold'},
          'worker_processes_closed': True, 'outcomes_closed': True, 'elapsed_seconds': 20236.344})
    write(prior/'runner_execution.json', {'exit_code': 1, 'outcomes_closed': True,
          'runner_identity': {'pid': 999995, 'created': 19990.}})
    write(prior/'coordinator_hold.json', {'status': 'preserved_hold'})
    write(prior/'final_input_check.json', {'all_hashes_and_sizes_match': True})
    write(base/'source_snapshot.json', {'synthetic': 'preserved prior recovery inventory'})
    write(base/'recovery_plan.json', {'schema': 'live_operational_recovery02_plan_v1',
          'source_base': prior.relative_to(tmp_path).as_posix(), 'source_freeze': c.artifact(prior/'freeze.json'),
          'original_freeze': c.artifact(original/'freeze.json'),
          'source_snapshot': c.artifact(base/'source_snapshot.json'), 'completed_games': c.COMPLETED_GAMES,
          'partial_game': {'game': 21, 'next_ply': 11}, 'validation_reused': True})
    freeze = {'schema': 'live_operational_recovery02_freeze_v1', 'executable': sys.executable,
              'execution_cap_seconds': c.CAP, 'final_reserve_seconds': c.RESERVE,
              'evidence_cap_bytes': c.MAX_BYTES, 'minimum_free_bytes': c.MIN_FREE,
              'max_process_tree_rss_bytes': c.RSS_LIMIT, 'outcomes_closed': True, 'strength_analysis_games': 0,
              'inputs': [c.artifact(analyzer), c.artifact(source), c.artifact(memory), c.artifact(base/'source_snapshot.json'),
                         c.artifact(base/'recovery_plan.json')]+[c.artifact(p) for p in sorted(original.glob('*.json'))]+
                         [c.artifact(p) for p in sorted(prior.glob('*.json'))],
              'recovery_plan': c.artifact(base/'recovery_plan.json'),
              'source_freeze': c.artifact(prior/'freeze.json'), 'original_freeze': c.artifact(original/'freeze.json')}
    write(base/'freeze.json', freeze)
    write(base/'launch.json', {'epoch_seconds': 50000., 'monotonic_start': 10000.,
          'maximum_seconds': c.CAP, 'pid': 999997, 'outcomes_closed': True, 'freeze': c.artifact(base/'freeze.json'),
          'recovery_plan': c.artifact(base/'recovery_plan.json'), 'source_launch': c.artifact(prior/'launch.json'),
          'validation_reused': True, 'imported_completed_games': 16})
    command = [sys.executable, '-B', '-u', '-m', c.RUNNER_MODULE, 'run']
    write(base/'runner_process.json', {'identity': {'pid': 999997, 'created': 49990.}, 'command': command,
          'freeze_sha256': c.sha(c.BASE/'freeze.json'), 'outcomes_closed': True})
    monkeypatch.setattr(c, 'alive', lambda _: False)
    write(base/'coordinator_plan.json', c.make_plan(c.sha(c.BASE/'freeze.json'), c.sha(c.SOURCE)))
    coordinator = c.Coordinator(c.sha(source), c.sha(c.BASE/'freeze.json'), c.sha(base/'coordinator_plan.json'))
    return coordinator, c.read(base/'coordinator_plan.json')


def completion_files():
    process = c.read(c.BASE/'runner_process.json')
    execution = {'runner_identity': c.read(c.BASE/'runner_process.json')['identity'], 'command': process['command'], 'exit_code': 0, 'outcomes_closed': True,
                 'started_epoch_seconds': c.read(c.BASE/'launch.json')['epoch_seconds']-10, 'finished_utc': c.now(), 'elapsed_seconds': 300}
    complete = {'status': 'completed_pending_independent_analysis', 'failure': None, 'completed_games': 18,
                'worker_processes_closed': True, 'outcomes_closed': True, 'validation_reused': True,
                'imported_completed_games': 16}
    write(c.BASE/'runner_execution.json', execution); write(c.BASE/'completion.json', complete)
    write(c.BASE/'final_input_check.json', {'all_hashes_and_sizes_match': True})
    return execution


def accepted():
    timings = {'mean': 20., 'p95': 40., 'max': 60.}
    return {'status': 'passed', 'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True, 'freeze_sha256': c.sha(c.BASE/'freeze.json'),
        'source_preservation': {'files': 100, 'bytes': 200, 'all_hashes_and_sizes_match': True},
        'operational_schedule': {'games': 18, 'pairs': 9, 'actual_dictionary_transitions': 600},
        'per_stage': {
            'games:local': {'attempted': 304, 'statuses': {'completed': 300, 'forfeit_timeout': 4}, 'completed_receipt_seconds': timings},
            'games:opponent': {'attempted': 300, 'statuses': {'completed': 300}, 'completed_receipt_seconds': timings}},
        'validation': {'requests': 12, 'exact_hidden_state_repeat_pairs': 6,
                       'reused_from_first_attempt': True, 'new_validation_requests': 0},
        'scope': {'original_timeout_forfeits_preserved': 4, 'unstarted_hold_not_scored_as_player_failure': True},
        'first_attempt_lineage': {'completed_games_reused': [6, 8, 10, 7, 9], 'partial_game': 11,
                                 'next_unstarted_ply': 15, 'original_attempt_status': 'preserved_hold',
                                 'original_source_and_evidence_unchanged': True},
        'prior_recovery_lineage': {'completed_games_reused': c.COMPLETED_GAMES, 'partial_game': 21,
                                  'next_unstarted_ply': 11, 'prior_recovery_status': 'preserved_hold',
                                  'prior_recovery_source_and_evidence_unchanged': True},
        'memory_waiting': {'admissions': 24, 'stability_required_admissions': 2, 'charged_seconds': 1200.},
        'actual_continuation_verification': {'roots': 300, 'full_stress_roots': 30, 'sample_only_roots': 270,
            'recorded_worlds': 38400, 'recorded_continuations': 153600,
            'covered_cells': 35, 'cells': 81, 'replayed_union': {'worlds': 6000, 'continuations': 24000, 'transitions': 800000}},
        'resources': {'production_elapsed_seconds': 40000, 'including_final_analysis_seconds': 40100,
            'original_attempt_elapsed_seconds': 10500.937, 'first_recovery_elapsed_seconds': 20236.344,
            'sum_attempt_execution_seconds_including_analysis': 70837.281,
            'between_attempt_wall_gap_not_charged_to_fresh_cap': True,
            'peak_sampled_combined_rss_bytes': 300*1024**2, 'evidence_bytes_before_this_receipt': 1024**3}}


def test_prospective_plan_pins_commands_boundaries_and_source(workspace):
    coordinator, plan = workspace
    assert coordinator.authenticate() == plan
    assert plan['maximum_seconds'] == 43200 and plan['final_reserve_seconds'] == 600 and plan['cpu'] == 2
    assert plan['automatic_retries'] == 0 and plan['analysis_attempts'] == 1
    assert plan['analyzer_command'][0] == sys.executable


@pytest.mark.parametrize('damage', ['source', 'freeze', 'plan', 'analyzer', 'memory', 'runner_identity', 'launch'])
def test_changed_prospective_identity_is_rejected_before_any_launch(workspace, damage):
    coordinator, _ = workspace
    if damage in ('source', 'analyzer', 'memory'):
        {'source': c.SOURCE, 'analyzer': c.ANALYZER, 'memory': c.MEMORY_SOURCE}[damage].write_text('changed', encoding='utf-8')
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
    elif damage == 'identity': execution['runner_identity'] = {**c.read(c.BASE/'runner_process.json')['identity'], 'created': 1.}
    elif damage == 'command': execution['command'] = ['untrusted']
    elif damage in ('incomplete', 'failure'):
        complete = c.read(c.BASE/'completion.json')
        if damage == 'incomplete': complete['completed_games'] = 17
        else: complete['failure'] = {'category': 'hold'}
        write(c.BASE/'completion.json', complete)
    with pytest.raises(c.Hold): c.check_runner(execution, c.read(c.BASE/'coordinator_plan.json'))


def test_runner_receipt_delivery_waits_for_partial_json_only(workspace, monkeypatch):
    expected = completion_files(); path = c.BASE/'runner_execution.json'
    path.write_text('{', encoding='utf-8'); sleeps = []
    monkeypatch.setattr(c, 'resources', lambda *args, **_: {})
    monkeypatch.setattr(c, 'remaining', lambda *args: 1000)
    def complete_delivery(seconds):
        sleeps.append(seconds); write(path, expected)
    monkeypatch.setattr(c.time, 'sleep', complete_delivery)
    assert c.delivered_runner_execution(c.read(c.BASE/'coordinator_plan.json')) == expected and sleeps == [.1]


def test_persistent_malformed_runner_receipt_is_bounded_and_held(workspace, monkeypatch):
    (c.BASE/'runner_execution.json').write_text('{', encoding='utf-8'); clock = [100.]; sleeps = []
    monkeypatch.setattr(c, 'resources', lambda *args, **_: {})
    monkeypatch.setattr(c, 'remaining', lambda *args: 1000)
    monkeypatch.setattr(c.time, 'monotonic', lambda: clock[0])
    def advance(seconds): sleeps.append(seconds); clock[0] += seconds
    monkeypatch.setattr(c.time, 'sleep', advance)
    with pytest.raises(c.Hold, match='runner_execution_delivery_incomplete'): c.delivered_runner_execution(c.read(c.BASE/'coordinator_plan.json'))
    assert max(sleeps) <= .1 and sum(sleeps) == pytest.approx(5.)
    assert not (c.BASE/'independent_analysis_execution_01.log').exists()


def test_valid_json_semantic_failure_has_no_delivery_retry(workspace, monkeypatch):
    completion_files(); write(c.BASE/'runner_execution.json', {'invalid': True})
    monkeypatch.setattr(c.time, 'sleep', lambda _: pytest.fail('Semantic errors must not be retried'))
    execution = c.delivered_runner_execution(c.read(c.BASE/'coordinator_plan.json'))
    with pytest.raises(KeyError): c.check_runner(execution, c.read(c.BASE/'coordinator_plan.json'))


def test_delivery_wait_retains_recovery_deadline_reserve(workspace, monkeypatch):
    (c.BASE/'runner_execution.json').write_text('{', encoding='utf-8')
    def deadline(*args, **_): raise c.Hold('recovery_deadline_reserve')
    monkeypatch.setattr(c, 'resources', deadline)
    monkeypatch.setattr(c.time, 'sleep', lambda _: pytest.fail('No wait after original deadline reserve'))
    with pytest.raises(c.Hold, match='recovery_deadline_reserve'): c.delivered_runner_execution(c.read(c.BASE/'coordinator_plan.json'))


def test_wall_and_monotonic_deadlines_both_retain_recovery_boundary(workspace, monkeypatch):
    monkeypatch.setattr(c.time, 'time', lambda: c.read(c.BASE/'launch.json')['epoch_seconds']+10)
    monkeypatch.setattr(c.time, 'monotonic', lambda: c.read(c.BASE/'launch.json')['monotonic_start']+100)
    assert c.remaining(c.read(c.BASE/'coordinator_plan.json')) == 43100
    monkeypatch.setattr(c.time, 'time', lambda: c.read(c.BASE/'launch.json')['epoch_seconds']+43300)
    assert c.remaining(c.read(c.BASE/'coordinator_plan.json')) == -100


@pytest.mark.parametrize('fault', ['deadline', 'disk', 'evidence'])
def test_resource_limits_stop_before_final_reserve(workspace, monkeypatch, fault):
    monkeypatch.setattr(c, 'remaining', lambda *args: 600 if fault == 'deadline' else 1000)
    monkeypatch.setattr(c.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0 if fault == 'disk' else c.MIN_FREE))
    if fault == 'evidence': monkeypatch.setattr(c, 'MAX_BYTES', 32*1024**2)
    with pytest.raises(c.Hold): c.resources(c.read(c.BASE/'coordinator_plan.json'))


def test_report_reads_only_operational_allowlist_and_disclaims_strength(workspace):
    analysis = accepted(); analysis['sealed_winner_payload'] = 'DO_NOT_DISCLOSE'
    report = c.operational_report(analysis, c.read(c.BASE/'coordinator_plan.json'))
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
    with pytest.raises(c.Hold): c.operational_report(value, c.read(c.BASE/'coordinator_plan.json'))


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
    monkeypatch.setattr(c, 'resources', lambda *args, **_: {})
    monkeypatch.setattr(c, 'remaining', lambda *args: 1000)
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
    monkeypatch.setattr(c, 'resources', lambda *args, **_: {'remaining_seconds': 1000})
    monkeypatch.setattr(c, 'alive', lambda _: False)
    monkeypatch.setattr(c.psutil, 'Process', fake_process)
    def analyzed(plan):
        fake_admission(plan)
        value = accepted(); write(c.BASE/'independent_analysis.json', value)
        write(c.BASE/'independent_analysis_execution_01.json', {'analyzer_process_closed': True})
        (c.BASE/'independent_analysis_execution_01.log').write_text('passed', encoding='utf-8')
        return value
    monkeypatch.setattr(coordinator, 'analyze', analyzed)
    coordinator.run()
    final = c.read(c.BASE/'finalization.json')
    assert final['status'] == 'operational_checks_passed_pending_finalization_validation' and final['handoffs_rewritten'] is False
    assert final['games'] == 18 and final['pairs'] == 9 and c.REPORT.exists()
    assert final['verified_actual_transitions'] == 600 and final['attempted_actual_decisions'] == 604
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
    c.close_owned(child, owner, descendants, {'pid': 1234, 'created': 100.})
    assert killed == [222] and waited == [222] and child.killed


def test_unexpected_descendant_is_captured_and_cleaned_on_guard_hold(workspace, monkeypatch):
    coordinator, plan = workspace; write(c.BASE/'coordinator_launch.json', {'claimed': True})
    monkeypatch.setattr(c, 'resources', lambda *args, **_: {}); monkeypatch.setattr(c, 'remaining', lambda *args: 1000)
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
    monkeypatch.setattr(c, 'resources', lambda *args, **_: {}); monkeypatch.setattr(c, 'remaining', lambda *args:1000)
    child = FakeChild(); observed = [0]
    def process(pid=None):
        observed[0] += 1
        if observed[0] > 1: raise c.Hold('guard_trigger')
        return fake_process(pid)
    def failed_wait(timeout): raise c.subprocess.TimeoutExpired('fake child', timeout)
    child.wait = failed_wait
    monkeypatch.setattr(c.psutil, 'Process', process); monkeypatch.setattr(c.subprocess, 'Popen', lambda *args, **kwargs:child)
    monkeypatch.setattr(c, 'alive', lambda value:value['pid'] == child.pid)
    with pytest.raises(c.Hold): coordinator.analyze(plan)
    receipt = c.read(c.BASE/'independent_analysis_execution_01.json')
    assert receipt['cleanup_error'] == 'Hold' and receipt['analyzer_process_closed'] is False and child.killed


def test_oversized_analysis_log_causes_preserved_hold(workspace, monkeypatch):
    coordinator, plan = workspace; write(c.BASE/'coordinator_launch.json', {'claimed': True})
    monkeypatch.setattr(c, 'resources', lambda *args, **_: {}); monkeypatch.setattr(c, 'remaining', lambda *args:1000)
    monkeypatch.setattr(c, 'MAX_LOG', 2); monkeypatch.setattr(c.psutil, 'Process', fake_process)
    monkeypatch.setattr(c, 'alive', lambda _:False); child = FakeChild()
    plan = c.make_plan(c.sha(c.BASE/'freeze.json'), c.sha(c.SOURCE)); write(c.BASE/'coordinator_plan.json', plan)
    coordinator.expected_plan = c.sha(c.BASE/'coordinator_plan.json')
    def popen(*args, **kwargs): kwargs['stdout'].write(b'oversized'); kwargs['stdout'].flush(); return child
    monkeypatch.setattr(c.subprocess, 'Popen', popen)
    with pytest.raises(c.Hold): coordinator.analyze(plan)
    assert child.killed and c.read(c.BASE/'independent_analysis_execution_01.json')['error'] == 'Hold'


@pytest.mark.parametrize('failure_check', [4, 5])
def test_postwrite_failure_cannot_leave_unqualified_success(workspace, monkeypatch, failure_check):
    coordinator, _ = workspace; completion_files(); calls = [0]
    def resource(*args, **kwargs):
        calls[0] += 1
        if calls[0] == failure_check: raise c.Hold('postwrite_bound')
        return {'remaining_seconds':1000}
    monkeypatch.setattr(c, 'resources', resource); monkeypatch.setattr(c, 'alive', lambda _:False)
    monkeypatch.setattr(c.psutil, 'Process', fake_process)
    def analyzed(plan):
        fake_admission(plan)
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


def test_plan_uses_new_authenticated_launch_instead_of_historical_process_constants(workspace):
    coordinator, first = workspace
    launch = c.read(c.BASE/'launch.json'); process = c.read(c.BASE/'runner_process.json')
    launch.update(pid=999996, epoch_seconds=90000., monotonic_start=27000.)
    process['identity'] = {'pid': 999996, 'created': 89995.}
    write(c.BASE/'launch.json', launch); write(c.BASE/'runner_process.json', process)
    second = c.make_plan(coordinator.expected_freeze, coordinator.expected_source)
    assert second['runner_identity'] == process['identity'] != first['runner_identity']
    assert second['recovery_epoch_seconds'] == 90000.
    assert second['recovery_monotonic_start'] == 27000.
    assert second['preserved_original'] == first['preserved_original']
    assert second['original_epoch_seconds'] == 1000.


@pytest.mark.parametrize('missing', ['source', 'analyzer', 'original', 'snapshot'])
def test_coordinator_and_analyzer_and_preserved_inputs_must_be_scientifically_frozen(workspace, missing):
    coordinator, _ = workspace
    paths = {'source': c.SOURCE, 'analyzer': c.ANALYZER,
             'original': c.ORIGINAL_BASE/'completion.json', 'snapshot': c.BASE/'source_snapshot.json'}
    frozen = c.read(c.BASE/'freeze.json')
    frozen['inputs'] = [v for v in frozen['inputs'] if v['path'] != paths[missing].relative_to(c.ROOT).as_posix()]
    write(c.BASE/'freeze.json', frozen)
    launch = c.read(c.BASE/'launch.json'); launch['freeze'] = c.artifact(c.BASE/'freeze.json')
    process = c.read(c.BASE/'runner_process.json'); process['freeze_sha256'] = c.sha(c.BASE/'freeze.json')
    write(c.BASE/'launch.json', launch); write(c.BASE/'runner_process.json', process)
    with pytest.raises(c.Hold, match='frozen_recovery_input_identity'):
        c.make_plan(c.sha(c.BASE/'freeze.json'), coordinator.expected_source)


@pytest.mark.parametrize('filename', ['completion.json', 'runner_execution.json', 'coordinator_hold.json'])
def test_preserved_original_closure_changes_invalidate_new_plan(workspace, filename):
    coordinator, _ = workspace
    value = c.read(c.ORIGINAL_BASE/filename); value['changed'] = True
    write(c.ORIGINAL_BASE/filename, value)
    with pytest.raises(c.Hold, match='frozen_recovery_input_identity'):
        coordinator.authenticate()


@pytest.mark.parametrize('damage', ['validation_rerun', 'missing_timeouts', 'changed_partial', 'merged_cost'])
def test_recovery_report_requires_reused_validation_timeouts_prefix_and_separate_cost(workspace, damage):
    value = accepted()
    if damage == 'validation_rerun': value['validation']['new_validation_requests'] = 12
    elif damage == 'missing_timeouts': value['per_stage']['games:local']['statuses']['forfeit_timeout'] = 1
    elif damage == 'changed_partial': value['prior_recovery_lineage']['next_unstarted_ply'] = 12
    else: value['resources']['including_final_analysis_seconds'] += 10500.937
    with pytest.raises(c.Hold):
        c.operational_report(value, c.read(c.BASE/'coordinator_plan.json'))


def test_recovery_report_does_not_reset_history_or_describe_validation_as_rerun(workspace):
    report = c.operational_report(accepted(), c.read(c.BASE/'coordinator_plan.json'))
    assert report.startswith('# Operational recovery completed with recorded timing failures')
    assert 'does not qualify either player as failure-free' in report
    assert 'local planner exceeded 120 seconds at least four times' in report
    assert 'four earlier player timeouts remain' in report
    assert 'were not rerun' in report and 'both earlier attempts remain preserved holds' in report
    assert '2.917 hours' in report and '11.139 hours' in report and '19.677 hours' in report
    assert 'excluding the gap between attempts' in report
    assert '0.333 hours, already included' in report and '12-hour cap' in report
    assert 'active player clocks were never paused or retried' in report


def test_substituted_analyzer_command_is_rejected_without_child(workspace, monkeypatch):
    coordinator, plan = workspace; plan = deepcopy(plan); plan['analyzer_command'] = ['untrusted']
    monkeypatch.setattr(c.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unbound child must not launch'))
    with pytest.raises(c.Hold, match='analysis_plan_identity'):
        coordinator.analyze(plan)


def test_existing_recovery_hold_cannot_be_promoted_to_success(workspace, monkeypatch):
    coordinator, _ = workspace; write(c.BASE/'coordinator_hold.json', {'status': 'preserved_hold'})
    monkeypatch.setattr(c.psutil, 'Process', fake_process)
    with pytest.raises(c.Hold, match='finalization_already_exists'):
        coordinator.run()
    assert not (c.BASE/'coordinator_launch.json').exists() and not c.REPORT.exists()


@pytest.mark.parametrize('key,value', [('execution_cap_seconds', 90000), ('final_reserve_seconds', 599),
                                      ('max_process_tree_rss_bytes', 4*1024**3), ('strength_analysis_games', False)])
def test_new_hash_cannot_silently_change_the_coordinator_resource_or_closed_scope_contract(workspace, key, value):
    coordinator, _ = workspace
    freeze = c.read(c.BASE/'freeze.json'); freeze[key] = value; write(c.BASE/'freeze.json', freeze)
    with pytest.raises(c.Hold, match='recovery_freeze_resource_contract'):
        c.make_plan(c.sha(c.BASE/'freeze.json'), coordinator.expected_source)


@pytest.mark.parametrize('filename', ['audit_manifest.json', 'independent_analysis.json'])
def test_prior_recovery_cannot_gain_an_audit_or_analysis_after_preserved_hold(workspace, filename):
    coordinator, _ = workspace; write(c.PRIOR_BASE/filename, {'synthetic': True})
    with pytest.raises(c.Hold, match='prior_recovery_not_preserved_unaudited_hold'):
        coordinator.authenticate()


@pytest.mark.parametrize('damage', ['completion', 'closure', 'freeze'])
def test_first_recovery_is_bound_independently_of_original_attempt(workspace, damage):
    coordinator, _ = workspace
    path = c.PRIOR_BASE/{'completion': 'completion.json', 'closure': 'coordinator_hold.json', 'freeze': 'freeze.json'}[damage]
    value = c.read(path); value['tampered'] = True; write(path, value)
    with pytest.raises(c.Hold): coordinator.authenticate()


def test_analysis_wait_admission_uses_fresh_both_clock_deadline_and_binds_its_log(workspace, monkeypatch):
    _, plan = workspace; calls = []
    monkeypatch.setattr(c.time, 'monotonic', lambda: 1000.)
    monkeypatch.setattr(c.time, 'time', lambda: plan['recovery_epoch_seconds']+60)
    def wait(base, deadline, **kwargs):
        calls.append((base, deadline, kwargs)); fake_admission(plan)
        return c.read(c.BASE/'memory_admissions/analysis.json')
    monkeypatch.setitem(sys.modules, c.MEMORY_MODULE, SimpleNamespace(wait_for_memory=wait))
    result = REAL_ADMISSION(plan)
    assert calls == [(c.BASE, 44140., {'name': 'analysis', 'reserve': 600, 'request_seconds': 0})]
    assert result['receipt'] == c.artifact(c.BASE/'memory_admissions/analysis.json')
    assert result['readings_log'] == c.artifact(c.BASE/'memory_admissions/analysis.jsonl')


def test_failed_memory_wait_never_launches_or_retries_analyzer(workspace, monkeypatch):
    coordinator, plan = workspace
    monkeypatch.setattr(c, 'resources', lambda *a, **k: {})
    def wait(_): raise c.Hold('bounded_memory_wait_exhausted')
    monkeypatch.setattr(c, 'analyzer_admission', wait)
    monkeypatch.setattr(c.subprocess, 'Popen', lambda *a, **k: pytest.fail('No analyzer before RAM admission'))
    with pytest.raises(c.Hold, match='bounded_memory_wait_exhausted'): coordinator.analyze(plan)
    assert not (c.BASE/'independent_analysis_execution_01.log').exists()


def test_memory_admission_finishes_before_the_single_analyzer_starts(workspace, monkeypatch):
    coordinator, plan = workspace; write(c.BASE/'coordinator_launch.json', {'claimed': True})
    events = []; child = FakeChild()
    monkeypatch.setattr(c, 'resources', lambda *a, **k: {})
    monkeypatch.setattr(c, 'remaining', lambda *a: 1000.)
    monkeypatch.setattr(c.psutil, 'Process', fake_process)
    def wait(value): events.append('memory_admitted'); return fake_admission(value)
    def popen(*args, **kwargs):
        events.append('analyzer_started'); write(c.BASE/'independent_analysis.json', accepted()); return child
    monkeypatch.setattr(c, 'analyzer_admission', wait); monkeypatch.setattr(c.subprocess, 'Popen', popen)
    coordinator.analyze(plan)
    assert events == ['memory_admitted', 'analyzer_started']
    receipt = c.read(c.BASE/'independent_analysis_execution_01.json')
    assert receipt['memory_admission']['receipt'] == c.artifact(c.BASE/'memory_admissions/analysis.json')


@pytest.mark.parametrize('charged', [-1, 14400, float('nan')])
def test_report_never_exempts_memory_waits_from_the_new_cap(workspace, charged):
    value = accepted(); value['memory_waiting']['charged_seconds'] = charged
    with pytest.raises(c.Hold, match='charged_memory_waiting'):
        c.operational_report(value, c.read(c.BASE/'coordinator_plan.json'))

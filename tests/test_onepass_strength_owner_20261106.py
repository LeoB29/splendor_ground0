"""Model-free ownership, ordering and inherited-clock counterexamples."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('ONEPASS_REPETITION', '0')
import pytest
from tools.splendorust_onepass_strength_20261106 import common as c, owner, envelope


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path / 'new_study')
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    monkeypatch.setattr(owner.shutil, 'disk_usage', lambda path: SimpleNamespace(free=50 * c.GiB))
    now = {'mono': 1000., 'wall': 10000.}
    monkeypatch.setattr(owner.time, 'monotonic', lambda: now['mono'])
    monkeypatch.setattr(owner.time, 'time', lambda: now['wall'])
    return now


def documents(stage='main', mode='main_block', reader=False):
    freeze = c.save(c.OUT / 'freeze.json', {'test': 'no_models'})
    claim = dict(monotonic_origin=0., unix_origin=9000., freeze=freeze)
    claim_item = c.save(c.OUT / 'claim.json', claim)
    phase_item = owner.begin_phase(claim, stage)
    ident = owner.segment_id(mode, 0, 0 if mode == 'main_block' else None)
    segment = dict(segment_id=ident, mode=mode, repetition=0,
                   block_start=0 if mode == 'main_block' else None, claim=claim_item,
                   phase_clock=phase_item, reader_clock=None, freeze_sha256=freeze['sha256'])
    if reader:
        segment['reader_clock'] = c.save(c.OUT / 'reader_clock.json', dict(
            monotonic=1000., unix_time=10000., cap_seconds=c.INDEPENDENT_REVIEW_SECONDS,
            monotonic_deadline=1000.+c.INDEPENDENT_REVIEW_SECONDS,
            unix_deadline=10000.+c.INDEPENDENT_REVIEW_SECONDS,
            original_phase=phase_item, claim=claim_item))
    seg_item = c.save(c.OUT / 'segments' / ident / 'claim.json', segment)
    return claim, claim_item, phase_item, seg_item


def test_owner_phase_clips_to_original_model_deadline(isolated):
    isolated['mono'] = c.MODEL_CUTOFF_SECONDS - 10.
    isolated['wall'] = 9000. + isolated['mono']
    claim = dict(monotonic_origin=0., unix_origin=9000.)
    c.save(c.OUT / 'claim.json', claim)
    item = owner.begin_phase(claim, 'main')
    phase = c.read(c.ROOT / item['path'])
    assert phase['monotonic_deadline'] == c.MODEL_CUTOFF_SECONDS
    assert phase['unix_deadline'] == 9000. + c.MODEL_CUTOFF_SECONDS
    isolated['mono'] = c.MODEL_CUTOFF_SECONDS
    with pytest.raises(ValueError, match='expired'):
        owner.guard(claim, phase)


def test_final_phase_clips_to_global_not_new_72h(isolated):
    isolated['mono'] = c.JOB_SECONDS - 20.
    isolated['wall'] = 9000. + isolated['mono']
    claim = dict(monotonic_origin=0., unix_origin=9000.)
    c.save(c.OUT / 'claim.json', claim)
    item = owner.begin_phase(claim, 'final')
    phase = c.read(c.ROOT / item['path'])
    assert phase['monotonic_deadline'] == c.JOB_SECONDS
    assert phase['unix_deadline'] == 9000. + c.JOB_SECONDS


def test_segment_inherits_owner_clock_and_cannot_begin_again(isolated, monkeypatch):
    claim, claim_item, phase_item, seg_item = documents()
    monkeypatch.setattr(envelope.time, 'perf_counter', lambda: 9e12)
    env = envelope.SegmentEnvelope(claim_item, phase_item, seg_item)
    phase = c.read(c.ROOT / phase_item['path'])
    assert env.start == 0. and env.wall_start == 9000.
    assert env.deadline == phase['monotonic_deadline']
    assert env.wall_deadline == phase['unix_deadline']
    assert env.effective_deadline() < 1e6
    with pytest.raises(ValueError, match='root owner'):
        env.begin('main')


def test_segment_refuses_restarted_phase_deadline(isolated):
    claim, claim_item, phase_item, seg_item = documents()
    phase = c.read(c.ROOT / phase_item['path']); phase['monotonic_deadline'] += 1
    changed_phase = c.save(c.OUT / 'stages' / 'changed_start.json', phase)
    seg = c.read(c.ROOT / seg_item['path']); seg['phase_clock'] = changed_phase
    changed_seg = c.save(c.OUT / 'segments' / 'changed' / 'claim.json', seg)
    with pytest.raises(ValueError, match='Unreset'):
        envelope.SegmentEnvelope(claim_item, changed_phase, changed_seg)


def test_reader_subclock_is_enforced_inside_reader(isolated):
    claim, claim_item, phase_item, seg_item = documents('final', 'independent', reader=True)
    env = envelope.SegmentEnvelope(claim_item, phase_item, seg_item)
    assert env.deadline == 1000.+c.INDEPENDENT_REVIEW_SECONDS
    assert env.wall_deadline == 10000.+c.INDEPENDENT_REVIEW_SECONDS
    isolated['mono'] = env.deadline
    with pytest.raises(ValueError):
        env.guard()


def test_model_segment_cannot_use_saved_only_72h_phase(isolated):
    claim, claim_item, phase_item, seg_item = documents('final', 'main_block')
    with pytest.raises(ValueError, match='phase|mode|role'):
        envelope.SegmentEnvelope(claim_item, phase_item, seg_item)


def test_reader_cannot_extend_original_six_hour_allowance(isolated):
    claim, claim_item, phase_item, seg_item = documents('final', 'independent', reader=True)
    reader_path = c.OUT / 'reader_clock.json'
    reader = c.read(reader_path); reader['monotonic_deadline'] += 1
    reader_path.write_text(c.canonical(reader))
    segment_path = c.ROOT / seg_item['path']
    segment = c.read(segment_path); segment['reader_clock'] = c.artifact(reader_path)
    segment_path.write_text(c.canonical(segment))
    with pytest.raises(ValueError, match='reader clock'):
        envelope.SegmentEnvelope(claim_item, phase_item, c.artifact(segment_path))


def test_available_ram_zero_is_only_recorded_not_admission_gate(isolated, monkeypatch):
    claim, claim_item, phase_item, seg_item = documents()
    monkeypatch.setattr(envelope.psutil, 'virtual_memory', lambda: SimpleNamespace(available=0))
    env = envelope.SegmentEnvelope(claim_item, phase_item, seg_item)
    env.admit()
    record = c.read(c.OUT / 'memory_admissions' / 'main_block_r0_g000.json')
    assert record['observed_available_bytes'] == 0
    assert record['ram_reserve_policy'] == 'advisory_only'
    assert record['wait_seconds'] == record['stable_seconds'] == 0.


@pytest.mark.parametrize('mode,rep,offset', [
    ('main_block', False, 0), ('main_block', 0, False), ('main_block', 0, 0.),
    ('main_block', 0, 200), ('main_block', 0, 1), ('saved', 0, 0), ('other', 0, None),
])
def test_exact_segment_identity_rejects_aliases_and_extra_blocks(mode, rep, offset):
    with pytest.raises(ValueError):
        owner.segment_id(mode, rep, offset)


def fake_run_setup(monkeypatch, fail=None):
    from tools.splendorust_search256_recovery_20261030_02 import placement
    monkeypatch.setattr(placement, 'configure_execution', lambda *args: {'mode': 'isolated'})
    calls = []
    def run(claim, phase, expected, mode, repetition, block_start=None, reader_clock=None):
        calls.append((mode, repetition, block_start))
        if (mode, repetition) == fail:
            raise ValueError('saved artificial failure; no retry')
        return dict(status='passed', mode=mode, repetition=repetition)
    monkeypatch.setattr(owner, 'run_segment', run)
    c.save(c.OUT / 'report' / 'independent_acceptance.json', {'status': 'synthetic_test_only'})
    freeze = c.save(c.OUT / 'freeze.json', dict(owner.contract(), inputs=[]))
    return freeze['sha256'], calls


def test_both_qualification_and_operational_stages_precede_interleaved_main(isolated, monkeypatch):
    sha, calls = fake_run_setup(monkeypatch)
    assert owner.run(sha) == 0
    assert calls[:4] == [('qualification', 0, None), ('qualification', 1, None),
                         ('operational', 0, None), ('operational', 1, None)]
    expected_main = [('main_block', rep, block) for block in range(0, 200, 10) for rep in (0, 1)]
    assert calls[4:44] == expected_main
    assert calls[44:] == [('replay', 0, None), ('replay', 1, None), ('saved', 0, None),
                           ('saved', 1, None), ('aggregate', 0, None), ('independent', 0, None),
                           ('independent', 1, None), ('reader', 0, None)]
    complete = c.read(c.OUT / 'completion.json')
    assert complete['owner_exit_still_requires_observation'] is True


def test_qualification_failure_preserved_without_ops_or_retry(isolated, monkeypatch):
    sha, calls = fake_run_setup(monkeypatch, ('qualification', 1))
    assert owner.run(sha) == 1
    assert calls == [('qualification', 0, None), ('qualification', 1, None)]
    assert (c.OUT / 'failure.json').is_file()
    assert not (c.OUT / 'completion.json').exists()
    assert c.read(c.OUT / 'execution.json')['status'] == 'failed'


def test_operational_failure_preserves_attempt_and_prevents_all_main(isolated, monkeypatch):
    sha, calls = fake_run_setup(monkeypatch, ('operational', 1))
    assert owner.run(sha) == 1
    assert len(calls) == 4 and all(mode != 'main_block' for mode, _, _ in calls)
    assert (c.OUT / 'failure.json').is_file()


def test_owner_rejects_boolean_contract_substitution(isolated, monkeypatch):
    sha, calls = fake_run_setup(monkeypatch)
    path = c.OUT / 'freeze.json'
    value = c.read(path); value['automatic_retries'] = False
    path.write_text(c.canonical(value))
    with pytest.raises(ValueError, match='contract'):
        owner.run(c.file_hash(path))
    assert calls == []


def test_common_rejects_duplicate_json_and_wrong_size_types(isolated):
    p = c.OUT / 'duplicate.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match='Duplicate'):
        c.read(p)
    item = c.artifact(p); item['bytes'] = float(item['bytes'])
    with pytest.raises(ValueError, match='size'):
        c.verify([item])


def test_one_launch_claim_is_exclusive(isolated, monkeypatch):
    sha, calls = fake_run_setup(monkeypatch)
    assert owner.run(sha) == 0
    before = list(calls)
    with pytest.raises(FileExistsError):
        owner.run(sha)
    assert calls == before


def test_owner_cannot_hide_live_descendant_by_closing_job(isolated, monkeypatch):
    from tools.local_player_qualification_128 import clock
    freeze = c.save(c.OUT / 'freeze.json', {'test': 'mocked_no_process_launch'})
    claim = dict(monotonic_origin=0., unix_origin=9000., freeze=freeze)
    c.save(c.OUT / 'claim.json', claim)
    phase = owner.begin_phase(claim, 'qualification')
    closed = {'job': False}

    class Identity:
        def __init__(self, pid): self.pid = pid
        def create_time(self): return 111. if self.pid == 999 else 222.
        def children(self, recursive=True): return [Identity(888)]

    class Process:
        pid, returncode = 999, None
        def __init__(self, args, **kwargs):
            self.polls = 0
            folder = Path(args[args.index('--segment') + 1])
            c.save(folder / 'result.json', dict(status='passed', mode='qualification', repetition=0,
                                               pid=999, created=111.))
        def poll(self):
            self.polls += 1
            if self.polls == 1: return None
            self.returncode = 0
            return 0

    class Job:
        def __init__(self, pid): assert pid == 999
        def close(self): closed['job'] = True

    def absent(identity):
        if identity['pid'] == 888 and not closed['job']:
            raise ValueError('Unclean live descendant before forced Job cleanup')

    monkeypatch.setattr(owner.subprocess, 'Popen', Process)
    monkeypatch.setattr(owner.psutil, 'Process', Identity)
    monkeypatch.setattr(clock, '_WindowsJob', Job)
    monkeypatch.setattr(owner, 'require_absent', absent)
    monkeypatch.setattr(owner, 'finish_owned_process', lambda *args: None)
    monkeypatch.setattr(owner.time, 'sleep', lambda seconds: None)
    with pytest.raises(ValueError, match='Failed segment preserved'):
        owner.run_segment(claim, phase, freeze['sha256'], 'qualification', 0)
    receipt = c.read(c.OUT / 'segments' / 'qualification_r0' / 'execution.json')
    assert receipt['status'] == 'failed'
    assert 'Unclean live descendant' in receipt['error']
    assert closed['job'] is True  # Cleanup occurred, but the failure was not erased.


@pytest.mark.parametrize('field,bad', [('repetition', False), ('freeze_sha256', 'wrong')])
def test_segment_rejects_wrong_typed_identity_or_freeze_before_work(isolated, monkeypatch, field, bad):
    import sys
    from tools.splendorust_onepass_strength_20261106 import segment
    from tools.splendorust_onepass_strength_20261106.assessment import common as a
    monkeypatch.setattr(a, 'REPETITION', 0)
    claim, claim_item, phase_item, seg_item = documents('qualification', 'qualification')
    path = c.ROOT / seg_item['path']; local = c.read(path); local[field] = bad
    path.write_text(c.canonical(local))
    monkeypatch.setattr(sys, 'argv', ['segment', '--mode', 'qualification', '--segment', str(path.parent),
                                    '--freeze-sha256', c.file_hash(c.OUT / 'freeze.json')])
    with pytest.raises(ValueError):
        segment.main()
    assert not (path.parent / 'controller_identity.json').exists()


@pytest.mark.parametrize('field,value', [('block_start', 0.), ('repetition', False),
                                        ('segment_id', 'main_block_r1_g000'), ('freeze_sha256', 'wrong')])
def test_segment_claim_strict_identity_and_global_freeze(isolated, field, value):
    claim, claim_item, phase_item, seg_item = documents()
    path = c.ROOT / seg_item['path']; document = c.read(path)
    document[field] = value
    with pytest.raises(ValueError):
        owner.validate_segment_claim(document, path.parent, 'main_block', 0, 0,
                                     c.file_hash(c.OUT / 'freeze.json'))


def test_segment_claim_rejects_wrong_folder_and_extra_fields(isolated):
    claim, claim_item, phase_item, seg_item = documents()
    path = c.ROOT / seg_item['path']; document = c.read(path)
    expected = c.file_hash(c.OUT / 'freeze.json')
    with pytest.raises(ValueError, match='path'):
        owner.validate_segment_claim(document, path.parent.parent / 'other', 'main_block', 0, 0, expected)
    document['retry'] = True
    with pytest.raises(ValueError, match='fields'):
        owner.validate_segment_claim(document, path.parent, 'main_block', 0, 0, expected)


@pytest.mark.parametrize('same_path', [True, False])
def test_freeze_requires_distinct_independent_reviewers(isolated, same_path):
    log = c.OUT / 'prelaunch' / 'tests.log'; log.parent.mkdir(parents=True, exist_ok=True); log.write_text('passed')
    tests = c.OUT / 'prelaunch' / 'tests.json'
    c.save(tests, dict(exit_code=0, models=0, games=0, inputs=[], log=c.artifact(log)))
    reviews = []
    for index in range(3):
        path = c.OUT / 'prelaunch' / f'review{index}.json'
        c.save(path, dict(approved=True, blocking_findings=[], inputs=[], reviewer='same_person'))
        reviews.append(path)
    with pytest.raises(ValueError, match='Distinct|distinct|independent'):
        owner.freeze(tests, [reviews[0]] * 3 if same_path else reviews)

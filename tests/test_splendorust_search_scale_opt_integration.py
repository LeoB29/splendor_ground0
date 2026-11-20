"""Model-free resource and evidence-boundary integration counterexamples."""
from types import SimpleNamespace
import json
import pytest

from tools.splendorust_search_scale_opt_20261024 import common as c, runner as r, schedule
from tools.splendorust_search_scale_20261024 import compression


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'outputs/new_comparison')
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    c.OUT.mkdir(parents=True)
    return tmp_path


class Process:
    def __init__(self, pid, rss, children=(), created=1.):
        self.pid, self.rss, self.descendants, self.created = pid, rss, list(children), created
    def create_time(self):
        return self.created
    def children(self, recursive):
        assert recursive is True
        return self.descendants
    def memory_info(self):
        return SimpleNamespace(rss=self.rss)


def test_second_worker_constructor_sample_counts_other_resident_tree():
    envelope = r.Envelope({'monotonic_origin': 0., 'unix_origin': 0.})
    native = Process(11, c.GiB//2)
    root = Process(10, c.GiB, [native])
    envelope.resident[(10, 1.)] = root
    # The constructor has not yet returned, so the second worker is not owned
    # in resident; its inherited sample callback must supply its whole tree.
    new = SimpleNamespace(process=SimpleNamespace(pid=20))
    with pytest.raises(ValueError, match='Combined'):
        envelope.guard_worker_sample(new, {'rss_bytes': c.GiB})
    assert envelope.peak_combined_rss == 5*c.GiB//2


def test_registered_worker_sample_counts_own_tree_exactly_once():
    envelope = r.Envelope({'monotonic_origin': 0., 'unix_origin': 0.})
    envelope.resident[(10, 1.)] = Process(10, c.GiB//2)
    envelope.resident[(20, 1.)] = Process(20, c.GiB, [Process(21, c.GiB)])
    child = SimpleNamespace(process=SimpleNamespace(pid=20))
    envelope.guard_worker_sample(child, {'rss_bytes': c.GiB//2})
    assert envelope.peak_combined_rss == c.GiB


def test_unobservable_other_resident_is_not_silently_omitted():
    class Missing(Process):
        def children(self, recursive):
            raise r.psutil.NoSuchProcess(self.pid)
    envelope = r.Envelope({'monotonic_origin': 0., 'unix_origin': 0.})
    envelope.resident[(10, 1.)] = Missing(10, 0)
    with pytest.raises(ValueError, match='disappeared'):
        envelope.aggregate_worker_rss()


@pytest.fixture
def clock(monkeypatch, workspace):
    clock = SimpleNamespace(mono=0., wall=1000.)
    monkeypatch.setattr(r.time, 'monotonic', lambda: clock.mono)
    monkeypatch.setattr(r.time, 'time', lambda: clock.wall)
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda root: SimpleNamespace(free=10*c.GiB))
    monkeypatch.setattr(c, 'save', lambda path, value: None)
    return clock


def test_stage_and_absolute_cutoffs_never_reset_total_clock(clock):
    envelope = r.Envelope({'monotonic_origin': 0., 'unix_origin': 1000.})
    envelope.begin('qualification')
    assert envelope.deadline == 3600 and envelope.wall_deadline == 4600
    clock.mono, clock.wall = c.COLLECTION_CUTOFF_SECONDS-20, 1000+c.COLLECTION_CUTOFF_SECONDS-20
    envelope.begin('main')
    assert envelope.deadline == c.COLLECTION_CUTOFF_SECONDS
    assert envelope.remaining() == 20
    envelope.begin('final')
    assert envelope.deadline == clock.mono+c.STAGES['final']
    clock.mono, clock.wall = c.JOB_SECONDS-5, 1000+c.JOB_SECONDS-5
    envelope.begin('final')
    assert envelope.deadline == c.JOB_SECONDS and envelope.remaining() == 5


@pytest.mark.parametrize('expired_clock', ['mono', 'wall'])
def test_either_clock_can_expire_the_job(clock, expired_clock):
    envelope = r.Envelope({'monotonic_origin': 0., 'unix_origin': 1000.})
    envelope.begin('main')
    setattr(clock, expired_clock, c.JOB_SECONDS+(1000 if expired_clock == 'wall' else 0))
    with pytest.raises(ValueError, match='real-clock'):
        envelope.guard()


def test_prefinal_evidence_cannot_borrow_final_reserve(clock, monkeypatch):
    envelope = r.Envelope({'monotonic_origin': 0., 'unix_origin': 1000.})
    monkeypatch.setattr(envelope, 'accounted_bytes', lambda: 448*c.MiB)
    with pytest.raises(ValueError, match='evidence'):
        envelope.begin('main')
    envelope.begin('final')
    monkeypatch.setattr(envelope, 'accounted_bytes', lambda: 512*c.MiB)
    with pytest.raises(ValueError, match='evidence'):
        envelope.guard()


def test_runtime_validation_binds_statistical_implementation(monkeypatch):
    monkeypatch.setattr(c, 'verify', lambda items: None)
    monkeypatch.setattr(c, 'file_hash', lambda path: 'python-hash')
    monkeypatch.setattr(r, 'processor_topology', lambda: {'cpu': 1})
    monkeypatch.setattr(compression, 'runtime_contract', lambda: {'codec': 1})
    monkeypatch.setattr(r, 'snapshot_api_modules', lambda: {'modules': 1})
    monkeypatch.setattr(r.statistics, 'runtime_contract', lambda: {'numpy': 'frozen'})
    freeze = dict(inputs=[], python=r.sys.version, python_executable_sha256='python-hash',
                  topology={'cpu': 1}, torch_runtime=[], compression_runtime={'codec': 1},
                  external_runtime=[], sampler_api_modules={'modules': 1},
                  statistics_runtime={'numpy': 'frozen'})
    r.verify_runtime(freeze)
    monkeypatch.setattr(r.statistics, 'runtime_contract', lambda: {'numpy': 'changed'})
    with pytest.raises(ValueError, match='Statistics runtime'):
        r.verify_runtime(freeze)


@pytest.mark.parametrize('worker_count', [413, 414, 415])
def test_final_inventory_requires_both_workers_and_excludes_operations_from_score(workspace, monkeypatch, worker_count):
    """Synthetic ledgers exercise only orchestration; scientific checks are stubs."""
    specs = schedule.make_schedule()
    class Env:
        def __init__(self, seed):
            self.seed = seed
        def initial_state(self):
            return {'synthetic_seed': self.seed}
    monkeypatch.setattr(r.rules, 'CorrectedEnv', Env)
    monkeypatch.setattr(r.rules, 'serialize_snapshot', lambda state: state)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    for spec in specs:
        folder = r.game_folder(spec)
        initial = Env(int(spec['seed_decimal'])).initial_state()
        write(folder/'initial.json', {**spec, 'initial': initial})
        write(folder/'result.json', {**spec, 'plies': 1, 'reason': 'natural', 'winner': None,
                                     'complete': True, 'final': initial})
        write(folder/'moves/0000.json', {**spec, 'ply': 0, 'completed': True})
    for index in range(worker_count):
        (c.OUT/'workers'/f'worker{index:03d}').mkdir(parents=True)
    monkeypatch.setattr(r, 'verify_saved_decision', lambda row: None)
    monkeypatch.setattr(r.audit, 'audit_game', lambda *args, **kwargs: {'status': 'passed'})
    monkeypatch.setattr(r.audit, 'performance', lambda rows: {'decisions': len(rows)})
    monkeypatch.setattr(r.audit, 'select_replay_sample', lambda rows, maximum: {'decisions': []})
    monkeypatch.setattr(r, 'verify_worker_logs', lambda folder, **kwargs: {'status': 'passed'})
    monkeypatch.setattr(r, 'verify_runtime', lambda *args: None)
    monkeypatch.setattr(r, 'qualification_acceptance', lambda **kwargs: {'status': 'passed'})
    scored = []
    def analyze(rows, evidence_gates_passed, **kwargs):
        scored.extend(rows)
        assert evidence_gates_passed is True
        return {'synthetic': 'verified'}
    monkeypatch.setattr(r.statistics, 'analyze_games', analyze)
    monkeypatch.setattr(r.statistics, 'verify_analysis', lambda *args, **kwargs: {'status': 'passed'})
    envelope = SimpleNamespace(guard=lambda: None, effective_deadline=lambda: float('inf'))
    freeze = {'qualification_acceptance': {'status': 'passed'}}
    if worker_count != 414:
        with pytest.raises(ValueError, match='worker|Worker'):
            r.final_checks(envelope, freeze, specs)
        assert not scored
    else:
        assert r.final_checks(envelope, freeze, specs) == {'synthetic': 'verified'}
        assert len(scored) == 200 and {row['stage'] for row in scored} == {'main'}

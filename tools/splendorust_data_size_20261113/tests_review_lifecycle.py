"""Independent model-free integration checks for the frozen execution boundary."""
from copy import deepcopy
from pathlib import Path
import pytest
from . import common as c, control, owner, placement, worker


def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(c, 'PLAN', tmp_path/'freeze.json')
    monkeypatch.setattr(c, 'SELECTION', tmp_path/'selection.json')
    monkeypatch.setattr(c, 'COMMITMENT', tmp_path/'all_checkpoints_committed.json')
    ident = dict(pid=777, created=12.5)
    monkeypatch.setattr(c, 'identity', lambda: ident)
    return ident


@pytest.mark.parametrize('mismatch', [False, True])
def test_worker_source_guard_precedes_model_and_is_called_once(monkeypatch, tmp_path, mismatch):
    from tools import future_training_rules_20261021 as guard
    from . import fit
    workspace(monkeypatch, tmp_path)
    plan_sources = {'accepted': 'sources'}
    c.save(c.PLAN, dict(design=c.design(), worker_inputs=[], data_inputs=[], corrected_source_receipt=plan_sources))
    freeze = c.sha(c.PLAN); calls = []
    monkeypatch.setattr(control, 'handshake', lambda *args: {})
    monkeypatch.setattr(placement, 'set_own', lambda cpu: calls.append(('placement', cpu)))
    monkeypatch.setattr(control, 'Boundary', lambda *args: lambda: None)
    monkeypatch.setattr(c, 'verify', lambda *args, **kwargs: None)
    def source_guard():
        calls.append(('guard', None))
        return {'changed': 'sources'} if mismatch else plan_sources
    def model_entry(folder, **kwargs):
        assert calls[-1] == ('guard', None)
        assert kwargs['runtime_sources'] == plan_sources
        assert (Path(folder)/'runtime_sources.json').exists()
        calls.append(('model', None))
        return {'test_only': True}
    monkeypatch.setattr(guard, 'verify_runtime_sources', source_guard)
    monkeypatch.setattr(fit, 'run_qualification', model_entry)
    monkeypatch.setattr(fit, 'verify_qualification', lambda *args, **kwargs: {'status':'saved_qualification_verified'})
    assert worker.run('qualification', freeze) == int(mismatch)
    assert sum(name == 'guard' for name, _ in calls) == 1
    assert sum(name == 'model' for name, _ in calls) == int(not mismatch)
    result = c.read(tmp_path/'jobs/qualification/worker/result.json')
    assert result['status'] == ('failed' if mismatch else 'passed')
    assert (tmp_path/'jobs/qualification/worker/failure.json').exists() is mismatch


def test_failed_saved_qualification_blocks_admission_and_preserves_observations(monkeypatch, tmp_path):
    from tools import future_training_rules_20261021 as guard
    from . import fit
    workspace(monkeypatch, tmp_path)
    sources={'accepted':'sources'}
    c.save(c.PLAN,dict(design=c.design(),worker_inputs=[],data_inputs=[],corrected_source_receipt=sources))
    freeze=c.sha(c.PLAN);calls=[]
    monkeypatch.setattr(control,'handshake',lambda *args:{})
    monkeypatch.setattr(placement,'set_own',lambda cpu:None)
    monkeypatch.setattr(control,'Boundary',lambda *args:lambda:None)
    monkeypatch.setattr(c,'verify',lambda *args,**kwargs:None)
    monkeypatch.setattr(guard,'verify_runtime_sources',lambda:sources)
    def qualification(folder,**kwargs):
        calls.append('actual_qualification')
        c.save(Path(folder)/'qualification_result.json',dict(status='qualification_passed',observation='retained'))
        return dict(status='qualification_passed')
    def failed_verifier(folder,**kwargs):
        calls.append('saved_verifier')
        raise ValueError('saved qualification loss mismatch')
    monkeypatch.setattr(fit,'run_qualification',qualification)
    monkeypatch.setattr(fit,'verify_qualification',failed_verifier)
    assert worker.run('qualification',freeze)==1
    assert calls==['actual_qualification','saved_verifier']
    folder=tmp_path/'jobs/qualification/worker'
    assert c.read(folder/'qualification_result.json')['observation']=='retained'
    assert c.read(folder/'result.json')['status']=='failed'
    assert 'saved qualification loss mismatch' in c.read(folder/'failure.json')['error']
    assert not (folder/'qualification_saved_verification.json').exists()


def test_owner_preserves_failure_and_never_launches_next_stage(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    c.save(c.PLAN, dict(worker_inputs=[], data_inputs=[], placement={'fixed': True}))
    freeze = c.sha(c.PLAN); launched = []
    monkeypatch.setattr(control, 'handshake', lambda *args: {'limits': {'original': True}})
    monkeypatch.setattr(control, 'bounds', lambda *args: None)
    monkeypatch.setattr(control, 'clipped_limits', lambda *args, **kwargs: {'unchanged': True})
    monkeypatch.setattr(placement, 'set_own', lambda cpu: None)
    monkeypatch.setattr(placement, 'snapshot', lambda: {'fixed': True})
    monkeypatch.setattr(placement, 'preserve_calibration', lambda: {'untouched': True})
    monkeypatch.setattr(c, 'verify', lambda *args, **kwargs: None)
    def failed_launch(command, folder, limits, metadata, **kwargs):
        launched.append(metadata['role'])
        c.save(folder/'actual_exit.json', dict(status='failed', exit_code=9))
        raise ValueError('saved failure')
    monkeypatch.setattr(control, 'launch', failed_launch)
    assert owner.run(freeze) == 1
    assert launched == ['qualification']
    assert c.read(tmp_path/'failure.json')['status'] == 'incomplete'
    assert c.read(tmp_path/'jobs/qualification/actual_exit.json')['exit_code'] == 9
    assert not (tmp_path/'completion.json').exists()
    assert c.read(tmp_path/'stage.json')['status'] == 'failed'


@pytest.mark.parametrize('mutation', ['running', 'wrong_result', 'boolean_exit', 'wrong_freeze'])
def test_prior_job_needs_exact_clean_exit_and_bound_result(monkeypatch, tmp_path, mutation):
    ident = workspace(monkeypatch, tmp_path); c.save(c.PLAN, {'fixed': True}); freeze = c.sha(c.PLAN)
    role = 'qualification'; folder = tmp_path/'jobs'/role
    result = c.save(folder/'worker/result.json', dict(status='passed', error=None, role=role,
        identity=ident, freeze_sha256=freeze))
    execution = dict(result_artifact=result, status='passed', exit_code=0, actual_exit_observed=True,
        exact_owned_absence=True, cleanup_errors=[], error=None, role=role, freeze_sha256=freeze,
        identity=ident, observed_owned_identities=[ident])
    if mutation == 'wrong_result': execution['result_artifact'] = dict(result, sha256='0'*64)
    if mutation == 'boolean_exit': execution['exit_code'] = False
    if mutation == 'wrong_freeze': execution['freeze_sha256'] = '0'*64
    c.save(folder/'actual_exit.json', execution)
    monkeypatch.setattr(c, 'exact_alive', lambda item: mutation == 'running')
    with pytest.raises(ValueError): worker.passed_job(role)


def test_commits_all_eight_fixed_endpoints_in_evaluator_order(monkeypatch, tmp_path):
    from . import fit, offline
    workspace(monkeypatch, tmp_path)
    selected = {'subsets': {'256': {'one_pass_steps': 445}, '512': {'one_pass_steps': 889}}}
    c.save(c.SELECTION, selected)
    monkeypatch.setattr(c, 'SOURCE_SELECTION', tmp_path/'source_selection.json')
    monkeypatch.setattr(c, 'ORIGINAL_COMMITMENT', tmp_path/'source_commitment.json')
    c.save(c.SOURCE_SELECTION, {'old': True}); c.save(c.ORIGINAL_COMMITMENT, {'original': True})
    proofs = {}
    for size in (256, 512):
        for rep in (0, 1):
            role = f'g{size}_r{rep}'; steps = dict(onepass=445 if size == 256 else 889, matched=1782)
            checkpoints = {}
            for endpoint, step in steps.items():
                checkpoints[str(step)] = c.save(tmp_path/'fixtures'/f'{role}_{endpoint}.json',
                    dict(metadata={'weights_sha256': c.digest([role, endpoint])}))
            proofs[role] = dict(game_count=size, repetition=rep, optimizer_calls=1782,
                replay_optimizer_calls=2, endpoint_steps=steps, checkpoints=checkpoints)
    monkeypatch.setattr(worker, 'passed_job', lambda role: ({'role': role}, {}))
    monkeypatch.setattr(fit, 'verify_fit', lambda folder, **kwargs: deepcopy(proofs[Path(folder).parent.name]))
    result = worker.commit(tmp_path/'commit', lambda: None)
    saved = c.read(c.resolve(result['commitment']))
    assert [row['model_id'] for row in saved['models']] == list(offline.FRESH_IDS)
    assert [row['checkpoint_step'] for row in saved['models']] == [445, 445, 1782, 1782, 889, 889, 1782, 1782]
    assert saved['scientific_new_updates'] == 7128 and saved['replay_updates'] == 8
    assert saved['checkpoint_selection'] is False

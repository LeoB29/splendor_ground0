"""Model-free inference worker dispatch, public projection and timer contracts."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import tools.move_evaluation_20261023 as original
import tools.move_evaluation_inference_opt_20261023 as package
from tools.move_evaluation_inference_opt_20261023 import common as c, worker as w


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path/'workspace'
    out = root/'outputs/inference'
    out.mkdir(parents=True)
    monkeypatch.setattr(c, 'ROOT', root)
    monkeypatch.setattr(c, 'OUT', out)
    def artifact(path):
        return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(c, 'artifact', artifact)
    return root, out


def initialization_stubs(monkeypatch, *, identity=None):
    from tools import future_training_rules_20261021 as guard
    import tools.corrected_planner_qualification_20261018 as qpackage
    calls = []
    monkeypatch.setattr(guard, 'verify_runtime_sources', lambda: calls.append('guard') or {'verified': True})
    q = SimpleNamespace(initialize=lambda: calls.append('proposal_raw_models') or {'historical': True})
    monkeypatch.setitem(sys.modules, 'tools.corrected_planner_qualification_20261018.local', q)
    monkeypatch.setattr(qpackage, 'local', q, raising=False)
    def reference(folder):
        calls.append('original_reference')
        return SimpleNamespace(folder=folder, kind='original_reference')
    def factored(folder):
        calls.append('prebound_fused_transformer')
        return SimpleNamespace(folder=folder, kind='prebound_fused_transformer', eligibility_receipt={'synthetic_eligibility': True})
    old = SimpleNamespace(GreedyOpponent=reference)
    new = SimpleNamespace(PreboundOpponent=factored, IMPLEMENTATION_ID=identity or c.IMPLEMENTATION_ID)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.greedy_adapter', old)
    monkeypatch.setattr(original, 'greedy_adapter', old, raising=False)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_inference_opt_20261023.inference', new)
    monkeypatch.setattr(package, 'inference', new, raising=False)
    return calls


@pytest.mark.parametrize('mode', w.MODES)
def test_startup_guard_once_then_models_and_explicit_opponent(sandbox, monkeypatch, mode):
    _, out = sandbox
    calls = initialization_stubs(monkeypatch)
    folder = out/mode
    monkeypatch.setenv('STUDY_WORKER_DIR', str(folder))
    monkeypatch.setenv('STUDY_MODE', mode)
    ctx = w.initialize()
    expected = 'original_reference' if mode in ('audit', 'probe_reference') else 'prebound_fused_transformer'
    assert calls == ['guard', 'proposal_raw_models', expected]
    assert ctx['mode'] == mode and ctx['used'] is False and ctx['opponent'].kind == expected
    assert ctx['implementation'] == w.implementation_metadata(mode)
    assert json.loads((folder/'implementation.json').read_text()) == ctx['implementation']
    assert json.loads((folder/'runtime_sources.json').read_text())['sources'] == {'verified': True}
    assert json.loads((folder/'models.json').read_text()) == {'historical': True}
    assert ctx['implementation']['optimization_enabled'] is (mode not in ('audit', 'probe_reference'))
    assert ctx['eligibility'] == c.artifact(folder/'eligibility.json')
    assert ctx['implementation']['prediction_cache'] is False
    assert ctx['implementation']['scientific_policy_id'] == w.SCIENTIFIC_POLICY_ID


@pytest.mark.parametrize('path', ['outside', 'outputs/inference/../../escaped'])
def test_output_escape_rejected_before_guard_or_models(sandbox, monkeypatch, path):
    root, _ = sandbox
    calls = initialization_stubs(monkeypatch)
    monkeypatch.setenv('STUDY_WORKER_DIR', str(root/path))
    monkeypatch.setenv('STUDY_MODE', 'probe_plain')
    with pytest.raises(ValueError, match='Worker output boundary'):
        w.initialize()
    assert calls == []


@pytest.mark.parametrize('mode,identity,match', [('unknown', None, 'diagnostic mode'),
                                             ('probe_plain', 'wrong', 'implementation identity')])
def test_invalid_mode_or_factoring_identity_prevents_model_loading(sandbox, monkeypatch, mode, identity, match):
    _, out = sandbox
    calls = initialization_stubs(monkeypatch, identity=identity)
    monkeypatch.setenv('STUDY_WORKER_DIR', str(out/'worker'))
    monkeypatch.setenv('STUDY_MODE', mode)
    with pytest.raises(ValueError, match=match):
        w.initialize()
    assert calls == ['guard']


@pytest.mark.parametrize('mode', ['qualify_plain', 'qualify_timed'])
def test_actual_qualification_uses_supplied_opponent_and_original_raw(sandbox, monkeypatch, mode):
    calls = []
    raw = object()
    evaluator = SimpleNamespace(select_raw=raw)
    def qualify(opponent, raw_callback):
        calls.append((opponent, raw_callback))
        return {'actual_fixture': 'synthetic-result'}
    qualification = SimpleNamespace(run_entrypoint_checks=qualify)
    for name, value in [('evaluator', evaluator), ('qualification', qualification)]:
        monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.'+name, value)
        monkeypatch.setattr(original, name, value, raising=False)
    opponent = object()
    ctx = {'mode': mode, 'opponent': opponent}
    assert w.scientific_body(ctx, {'operation': 'entrypoint_checks'}) == {'actual_fixture': 'synthetic-result'}
    assert calls == [(opponent, raw)]
    with pytest.raises(ValueError, match='Qualification payload'):
        w.scientific_body(ctx, {'operation': 'entrypoint_checks', 'extra': 1})
    assert len(calls) == 1


@pytest.fixture
def public_panel(monkeypatch):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules
    from tools.move_evaluation_20261023 import qualification
    from tools.move_evaluation_20261023 import worker as original_worker
    fixture = next(f for f in qualification.build_fixtures() if f['id'] == 'private_reserves')
    state = guard.decode_training_snapshot(fixture['snapshot'])
    public = rules.observation_payload(state)
    roots = []
    for index in range(18):
        fit = (161001, 161002, 161003)[index//6]
        live = deepcopy(public)
        live.update(schema='corrected_live_planner_observer_v1',
                    context={'fit': fit, 'identity': ['synthetic', index], 'context_sentinel': 'removed'})
        roots.append({'id': f'root{index:02d}', 'fit': fit, 'observer': live,
                      'candidates': [], 'selection_schedule': [],
                      'not_public': {'hidden_referee_sentinel': True}})
    progress = []
    monkeypatch.setattr(original_worker, 'progress', lambda: progress.append(True))
    monkeypatch.setattr(c, 'read', lambda path: {'roots': roots})
    return roots, public, state, progress


def test_probe_144_rows_exact_public_projection_and_input_preservation(public_panel):
    from tools.external_strength_screen_20261018 import rules
    roots, public, state, progress = public_panel
    original_roots = deepcopy(roots)
    calls = []
    expected_action = rules.legal_actions(rules.state_from_observation(public))[0]
    def choose(observer, *, diagnostic_capture):
        assert diagnostic_capture is True
        assert observer == public
        assert set(observer) == {'schema', 'ruleset_id', 'catalogue_sha256', 'observation'}
        assert observer['schema'] == rules.OBSERVATION_SCHEMA
        assert observer['observation']['players'][1]['reserves'][0]['card_id'] is None
        assert observer['observation']['players'][0]['reserves'][0]['card_id'] == state.players[0].reserved_cards[0].card_id
        assert 'hidden_referee_sentinel' not in c.canonical(observer)
        assert 'context_sentinel' not in c.canonical(observer)
        calls.append(observer)
        return expected_action, {'synthetic_trace': True}
    result = w.probe_panel({'opponent': SimpleNamespace(choose=choose)},
                          {'operation': 'public_inference_panel', 'repetitions': 8})
    assert result['schema'] == 'entity_public_inference_panel_v1'
    assert result['repetitions'] == 8 and result['panel_size'] == 18
    assert len(result['rows']) == len(calls) == len(progress) == 144
    assert len({id(value) for value in calls}) == 144
    for ordinal, row in enumerate(result['rows']):
        repetition, index = divmod(ordinal, 18)
        assert row['repetition'] == repetition and row['index'] == index
        assert row['id'] == roots[index]['id'] and row['fit'] == roots[index]['fit']
        assert row['observer_sha256'] == c.digest(public)
        assert row['action'] == rules.serialize_action(expected_action)
    assert roots == original_roots


def test_mutating_opponent_fails_probe_without_mutating_locked_input(public_panel):
    from tools.external_strength_screen_20261018 import rules
    roots, public, _, progress = public_panel
    before = deepcopy(roots)
    action = rules.legal_actions(rules.state_from_observation(public))[0]
    def choose(observer, *, diagnostic_capture):
        observer['observation']['bank_tokens'].clear()
        return action, {}
    with pytest.raises(ValueError, match='changed its public input'):
        w.probe_panel({'opponent': SimpleNamespace(choose=choose)},
                      {'operation': 'public_inference_panel', 'repetitions': 8})
    assert len(progress) == 1 and roots == before


@pytest.mark.parametrize('payload', [{}, {'operation': 'public_inference_panel', 'repetitions': 7},
    {'operation': 'public_inference_panel', 'repetitions': 8, 'extra': True}])
def test_probe_rejects_changed_schedule_before_choice(public_panel, payload):
    calls = []
    with pytest.raises(ValueError, match='Fixed public inference panel'):
        w.probe_panel({'opponent': SimpleNamespace(choose=lambda *_a, **_k: calls.append(True))}, payload)
    assert calls == []


@pytest.mark.parametrize('corruption', ['count', 'fit', 'schema', 'context_fit', 'extra_envelope'])
def test_probe_rejects_changed_public_identity(public_panel, corruption):
    roots, _, _, _ = public_panel
    if corruption == 'count':
        roots.pop()
    elif corruption == 'fit':
        roots[0]['fit'] = 161002
    elif corruption == 'schema':
        roots[0]['observer']['schema'] = 'historical'
    elif corruption == 'context_fit':
        roots[0]['observer']['context']['fit'] = 161003
    else:
        roots[0]['observer']['hidden'] = {}
    calls = []
    with pytest.raises(ValueError, match='(Fixed panel identities|Qualified live public observer)'):
        w.probe_panel({'opponent': SimpleNamespace(choose=lambda *_a, **_k: calls.append(True))},
                      {'operation': 'public_inference_panel', 'repetitions': 8})
    assert calls == []


@pytest.mark.parametrize('mode', ['probe_reference', 'probe_plain', 'probe_timed'])
def test_probe_dispatch_is_explicit(sandbox, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(w, 'probe_panel', lambda ctx, payload: calls.append((ctx, payload)) or {'panel': True})
    ctx, payload = {'mode': mode}, {'exact': True}
    assert w.scientific_body(ctx, payload) == {'panel': True}
    assert calls == [(ctx, payload)]


@pytest.mark.parametrize('mode', w.MODES)
def test_handle_once_preserves_science_and_administrative_metadata(sandbox, monkeypatch, mode):
    _, out = sandbox
    calls = []
    def body(ctx, payload):
        calls.append(payload)
        return {'science': ['unchanged']}
    monkeypatch.setattr(w, 'scientific_body', body)
    monkeypatch.setattr(w, 'targets', lambda ctx: {'scientific_body': body})
    ctx = {'used': False, 'mode': mode, 'folder': out, 'implementation': w.implementation_metadata(mode), 'eligibility': {'saved_receipt': True}}
    result = w.handle(ctx, {'once': True})
    assert result['scientific'] == {'science': ['unchanged']}
    assert result['implementation'] == ctx['implementation']
    assert result['measurement']['mode'] == mode
    assert result['measurement']['scientific_body_seconds'] == result['measurement']['body_seconds']
    expected_schema = 'selected_python_regions_timing_v1' if mode == 'timed' or mode.endswith('_timed') else 'ordinary_body_wall_time_v1'
    assert result['measurement']['schema'] == expected_schema
    with pytest.raises(ValueError, match='Exactly one request'):
        w.handle(ctx, {})
    assert calls == [{'once': True}]


def test_timing_failure_keeps_original_exception_and_single_use(sandbox, monkeypatch):
    _, out = sandbox
    error = RuntimeError('synthetic application failure')
    def body(ctx, payload):
        raise error
    monkeypatch.setattr(w, 'scientific_body', body)
    monkeypatch.setattr(w, 'targets', lambda ctx: {'scientific_body': body})
    ctx = {'used': False, 'mode': 'probe_timed', 'folder': out, 'implementation': {}}
    with pytest.raises(RuntimeError) as observed:
        w.handle(ctx, {})
    assert observed.value is error and ctx['used'] is True
    assert json.loads((out/'timing_failure.json').read_text())['phase'] == 'application'


def test_targets_use_new_code_and_preserve_original_sources(monkeypatch):
    from tools.move_evaluation_reuse_20261023 import worker as reuse_worker
    from tools.move_evaluation_inference_opt_20261023 import inference
    from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
    original_code = Opponent._infer.__code__
    old_body = reuse_worker.scientific_body
    def spare(ctx):
        return ctx
    baseline = {'scientific_body': old_body, 'entity_infer': Opponent._infer, 'spare': spare}
    monkeypatch.setattr(reuse_worker, 'targets', lambda ctx: dict(baseline))
    targets = w.targets({})
    assert targets['scientific_body'] is w.scientific_body
    assert targets['entity_infer'] is inference.PreboundOpponent._infer
    assert targets['probe_panel'] is w.probe_panel
    assert all(targets[label] is fn for label, fn in inference.fine_targets().items())
    assert len({id(fn.__code__) for fn in targets.values()}) == len(targets)
    assert baseline['entity_infer'] is Opponent._infer and Opponent._infer.__code__ is original_code
    assert reuse_worker.scientific_body is old_body


def test_ambiguous_selected_code_fails_closed(monkeypatch):
    from tools.move_evaluation_reuse_20261023 import worker as reuse_worker
    monkeypatch.setattr(reuse_worker, 'targets', lambda ctx: {'duplicate': w.probe_panel})
    with pytest.raises(ValueError, match='Ambiguous inference'):
        w.targets({})


def fake_scientific_modules(monkeypatch):
    evaluator=SimpleNamespace(PROFILE='unchanged',select_raw=object())
    monkeypatch.setitem(sys.modules,'tools.move_evaluation_20261023.evaluator',evaluator)
    monkeypatch.setattr(original,'evaluator',evaluator,raising=False)
    return evaluator


@pytest.mark.parametrize('mode',['qualify_plain','qualify_timed'])
def test_actual_qualification_receives_optimized_opponent_unchanged(sandbox,monkeypatch,mode):
    _,out=sandbox
    evaluator=fake_scientific_modules(monkeypatch)
    opponent=object();calls=[]
    def qualify(policy,raw):
        calls.append((policy,raw));return {'accepted':'same'}
    qualification=SimpleNamespace(run_entrypoint_checks=qualify)
    monkeypatch.setitem(sys.modules,'tools.move_evaluation_20261023.qualification',qualification)
    monkeypatch.setattr(original,'qualification',qualification,raising=False)
    result=w.scientific_body({'mode':mode,'opponent':opponent},{'operation':'entrypoint_checks'})
    assert calls==[(opponent,evaluator.select_raw)] and result=={'accepted':'same'}


def test_audit_rejects_optimized_policy_before_evidence_or_replay(sandbox,monkeypatch):
    fake_scientific_modules(monkeypatch)
    with pytest.raises(ValueError,match='Original reference audit required'):
        w.scientific_body({'mode':'audit','implementation':{'active_opponent':'prebound_fused_transformer'}},{})


def test_audit_uses_original_policy_and_exact_trace(sandbox,monkeypatch):
    root,out=sandbox
    fake_scientific_modules(monkeypatch)
    path=out/'continuations.jsonl.gz';path.write_bytes(b'synthetic')
    checked=[];calls=[];opponent=object()
    monkeypatch.setattr(c,'verify',lambda items: checked.extend(items))
    def audit(selector,result,trace,policy,*,boundary):
        calls.append((selector,result,trace,policy));return {'status':'passed_synthetic'}
    module=SimpleNamespace(audit_selection=audit)
    monkeypatch.setitem(sys.modules,'tools.move_evaluation_20261023.audit',module)
    monkeypatch.setattr(original,'audit',module,raising=False)
    payload={'selector':{'fixed':True},'result':{'exact':True},'trace':c.artifact(path)}
    value=w.scientific_body({'mode':'audit','opponent':opponent,'implementation':{'active_opponent':'original_reference'}},payload)
    assert checked==[payload['trace']]
    assert calls==[(payload['selector'],payload['result'],path,opponent)]
    assert value=={'status':'passed_synthetic'}



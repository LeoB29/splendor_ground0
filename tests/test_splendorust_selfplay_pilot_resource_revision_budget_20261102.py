"""Model-free checks of the single prospective assessment-budget amendment."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import pytest

from tools.splendorust_selfplay_pilot_budget_20261102 import common as c, resources
from tools.splendorust_selfplay_pilot_budget_20261102 import qualification_control as owner

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / 'outputs/splendorust_selfplay_pilot_launch_preparation_20261102_01/resource_amendment_01.json'


def run_scope(scope, source):
    result = subprocess.run([sys.executable, '-B', '-c', source], cwd=ROOT,
        env={**os.environ, 'SELFPLAY_JOB_SCOPE': scope, 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize('scope', ['qualification', 'pilot'])
def test_real_scoped_import_changes_only_the_one_stage_and_preserves_schedules(scope):
    value = run_scope(scope, '''
import json, sys
from tools.splendorust_selfplay_pilot_v2_20261102 import common as old, design as old_design, training as old_fit
from tools.splendorust_selfplay_pilot_budget_20261102 import common as new, design as new_design, training as new_fit
from tools.splendorust_selfplay_pilot_v2_20261102.assessment import common as old_a, design as old_ad
from tools.splendorust_selfplay_pilot_budget_20261102.assessment import common as new_a, design as new_ad
from tools.splendorust_selfplay_pilot_budget_20261102 import qualification_control as owner
from tools.splendorust_selfplay_pilot_budget_20261102.assessment import control as assessment
changed={key:[old.STAGES[key],new.STAGES[key]] for key in old.STAGES if old.STAGES[key]!=new.STAGES[key]}
assert changed==({'assessment':[57600,72000]} if new.SCOPE=='pilot' else {})
assert old.JOB_SECONDS==new.JOB_SECONDS and old.MODEL_CUTOFF_SECONDS==new.MODEL_CUTOFF_SECONDS
assert old.FINAL_SECONDS==new.FINAL_SECONDS and old.OUT==new.OUT
assert old_a.STAGE_MAP==new_a.STAGE_MAP and new_a.STAGES['main']==new.STAGES['assessment']
assert owner.contract()['stage_seconds']['assessment']==new.STAGES['assessment']
assert assessment.contract()['stage_seconds']['main']==new.STAGES['assessment']
assert old_design.schedule()==new_design.schedule()
assert old_design.DOMAIN==new_design.DOMAIN and old_ad.DOMAIN==new_ad.DOMAIN
assert old_ad.make_schedule()==new_ad.make_schedule()
assert old_ad.replay_rule()==new_ad.replay_rule()
assert old_fit.qualification_plan()==new_fit.qualification_plan()
assert old_fit.pilot_plan()==new_fit.pilot_plan()
assert 'torch' not in sys.modules
print(json.dumps(dict(changed=changed,job=new.JOB_SECONDS,models=new.MODEL_CUTOFF_SECONDS,
 final=new.FINAL_SECONDS,main=new_a.STAGES['main'],sources=len(new_design.schedule()),
 assessment_games=len(new_ad.make_schedule()))))
''')
    if scope == 'pilot':
        assert value == dict(changed={'assessment': [57600, 72000]}, job=259200,
            models=216000, final=43200, main=72000, sources=256, assessment_games=406)
    else:
        assert value == dict(changed={}, job=9000, models=7200, final=1800,
            main=0, sources=4, assessment_games=6)


@pytest.mark.parametrize('elapsed,wall_elapsed,stage,expected,wall_expected', [
    (10, 10, 'assessment', 30, 30),
    (50, 50, 'assessment', 60, 60),
    (50, 59, 'assessment', 60, 60),
    (60, 60, 'final', 72, 72),
])
def test_real_envelope_clamps_new_stage_to_original_model_and_total_clocks(
        monkeypatch, elapsed, wall_elapsed, stage, expected, wall_expected):
    monkeypatch.setattr(c, 'JOB_SECONDS', 72 * 3600)
    monkeypatch.setattr(c, 'COLLECTION_CUTOFF_SECONDS', 60 * 3600)
    monkeypatch.setattr(c, 'STAGES', {**c.STAGES, 'assessment': 20 * 3600, 'final': 12 * 3600})
    monkeypatch.setattr(owner.time, 'monotonic', lambda: 1000 + elapsed * 3600)
    monkeypatch.setattr(owner.time, 'time', lambda: 2000 + wall_elapsed * 3600)
    records = []
    monkeypatch.setattr(c, 'atomic_save', lambda path, value: records.append((path, value)))
    envelope = owner.Envelope(dict(monotonic_origin=1000., unix_origin=2000.))
    # Resource sampling is irrelevant here; retain the production clock guard.
    monkeypatch.setattr(envelope, 'guard', envelope.time_guard)
    envelope.begin(stage)
    assert envelope.start == 1000. and envelope.wall_start == 2000.
    assert envelope.deadline == 1000 + expected * 3600
    assert envelope.wall_deadline == 2000 + wall_expected * 3600
    assert records[0][1]['cap_seconds'] == c.STAGES[stage]
    assert envelope.remaining() == min((expected-elapsed)*3600, (wall_expected-wall_elapsed)*3600)


def projection_from_saved_proposal(stage_seconds, *, late_seconds=0., evidence_override=None):
    receipt = json.loads(PROPOSAL.read_text())
    config = SimpleNamespace(**vars(c))
    config.SCOPE = 'pilot'
    config.MAX_SOURCE_GAMES = 256
    config.MAX_SOURCE_WORKER_LIFECYCLES = 34
    config.MAX_SOURCE_REPLAY_DECISIONS = 48
    config.MAX_ASSESSMENT_GAMES = 406
    config.MAX_ASSESSMENT_REPLAY_DECISIONS = 96
    config.MAX_REPLAY_DECISIONS = 144
    config.RAM_WAIT_SECONDS = 10800
    config.MODEL_CUTOFF_SECONDS = 216000
    config.JOB_SECONDS = 259200
    config.PREFINAL_EVIDENCE_BYTES = 40*c.GiB
    config.FINAL_RESERVE_BYTES = 8*c.GiB
    config.STAGES = {'collection': 34*3600, 'fit': 4*3600, 'assessment': stage_seconds}
    binding = receipt['embedded_plan_binding']
    plan = receipt['resource_plan']
    assert c.digest(plan) == binding['content_sha256']
    def verify(items):
        assert items == [binding] and c.digest(plan) == binding['content_sha256']
    def read(path):
        assert path == c.ROOT/binding['path']
        return deepcopy(plan)
    config.verify, config.read = verify, read
    function = FunctionType(resources.source_forecast.__code__,
        {**resources.source_forecast.__globals__, 'c': config}, 'saved_projection',
        resources.source_forecast.__defaults__, resources.source_forecast.__closure__)
    function.__kwdefaults__ = resources.source_forecast.__kwdefaults__
    snapshot = deepcopy(receipt['snapshot'])
    snapshot['monotonic_now'] += late_seconds
    snapshot['unix_now'] += late_seconds
    if evidence_override is not None:
        snapshot['accounted_bytes'] = evidence_override
    return function({'resource_plan': binding}, snapshot), receipt


def test_actual_saved_resource_failure_is_resolved_only_by_authorized_stage_limit():
    before, receipt = projection_from_saved_proposal(16*3600)
    after, _ = projection_from_saved_proposal(20*3600)
    assert before == receipt['before'] and after == receipt['after']
    assert before['checks']['assessment_stage'] is False and before['affordable'] is False
    assert after['checks']['assessment_stage'] is True and after['affordable'] is True
    normalized = deepcopy(before)
    normalized['checks']['assessment_stage'] = True
    normalized['affordable'] = True
    assert normalized == after
    assert 18*3600 < after['projected_main_assessment_seconds'] < 20*3600
    assert after['remaining_source_games'] == 256
    assert after['reserved_final_seconds'] == 12*3600
    assert not after['outcomes_used'] and not after['changes_to_sample_or_deadlines']


@pytest.mark.parametrize('late,bytes_override,failed', [
    (12*3600, None, 'model'), (0., 40*1024**3, 'evidence')])
def test_larger_nested_stage_cannot_bypass_shared_clock_or_evidence_limit(late, bytes_override, failed):
    value, _ = projection_from_saved_proposal(20*3600, late_seconds=late, evidence_override=bytes_override)
    assert value['checks'][failed] is False and value['affordable'] is False


def test_runtime_science_and_training_files_remain_qualified_after_namespace_normalization():
    old = ROOT/'tools/splendorust_selfplay_pilot_v2_20261102'
    new = ROOT/'tools/splendorust_selfplay_pilot_budget_20261102'
    freeze = json.loads((c.QUALIFICATION_OUT/'freeze.json').read_text())
    indexed = {x['path']: x for x in freeze['inputs']}
    changed = {'common.py', 'preparation.py'}
    checked = []
    for prior in sorted(old.rglob('*.py')):
        relative = prior.relative_to(old).as_posix()
        if relative in changed:
            continue
        c.verify([indexed[prior.relative_to(ROOT).as_posix()]])
        expected = prior.read_text().replace('tools.splendorust_selfplay_pilot_v2_20261102',
            'tools.splendorust_selfplay_pilot_budget_20261102').replace(
            'tools/splendorust_selfplay_pilot_v2_20261102', 'tools/splendorust_selfplay_pilot_budget_20261102')
        assert (new/relative).read_text() == expected, relative
        checked.append(relative)
    assert len(checked) == 36
    assert {'data.py', 'training.py', 'clock_contract.py', 'assessment/analysis.py',
            'assessment/design.py', 'capture_adapter.py', 'qualification_control.py'} <= set(checked)


def test_actual_equivalence_proof_reuses_only_accepted_qualification_not_its_data():
    from tools.splendorust_selfplay_pilot_budget_20261102 import resource_revision as revision
    proof = revision.build_resource_only_equivalence()
    assert proof['status'] == 'exact_resource_only_equivalence'
    assert len(proof['source_map']) == 38
    assert proof['delta'] == dict(field='pilot.STAGES.assessment', old_seconds=57600, new_seconds=72000)
    assert proof['qualification_acceptance']['sha256'] == 'd47a4a087133ba66820d37afc4813f98429c1d6379d7b06d3266acaed77b1352'
    assert proof['new_qualification_required'] is False
    assert proof['reused_training_games'] == proof['reused_training_examples'] == 0
    assert proof['reused_engineering_checkpoint'] is False
    assert proof['models'] == proof['games'] == 0


@pytest.mark.parametrize('relative', ['training.py', 'design.py', 'assessment/design.py',
                                     'clock_contract.py', 'common.py'])
def test_equivalence_rejects_any_unapproved_change_without_touching_files(monkeypatch, relative):
    from tools.splendorust_selfplay_pilot_budget_20261102 import resource_revision as revision
    changed = (revision.REVISED_PACKAGE/relative).resolve()
    original_read = Path.read_bytes
    def altered_read(path):
        content = original_read(path)
        return content+b'\n# unapproved mutation\n' if path.resolve() == changed else content
    monkeypatch.setattr(Path, 'read_bytes', altered_read)
    with pytest.raises(ValueError, match='Unapproved source change'):
        revision.build_resource_only_equivalence()


def test_equivalence_cannot_substitute_an_unaccepted_or_different_qualification(monkeypatch):
    from tools.splendorust_selfplay_pilot_budget_20261102 import resource_revision as revision
    original_hash = c.file_hash
    monkeypatch.setattr(c, 'file_hash', lambda path: 'f'*64 if Path(path).resolve() ==
        revision.ACCEPTANCE_PATH.resolve() else original_hash(path))
    with pytest.raises(ValueError, match='Exact qualified source'):
        revision.build_resource_only_equivalence()

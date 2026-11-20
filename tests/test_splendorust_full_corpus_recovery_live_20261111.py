"""Model-free adversarial checks for immutable live lineage and the one retry."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '1')
os.environ.setdefault('FULL_CORPUS_MATCHUP', 'terminal_parent')

from tools.splendorust_full_corpus_recovery_20261111.strength import common as c, lineage, checkpoint


@pytest.mark.parametrize('matchup,rep,count', [
    ('mixed_parent', 0, 40), ('mixed_parent', 1, 40),
    ('terminal_parent', 0, 40), ('terminal_parent', 1, 32),
    ('mixed_terminal', 0, 30), ('mixed_terminal', 1, 30)])
def test_exact_saved_prefix_routing(monkeypatch, tmp_path, matchup, rep, count):
    monkeypatch.setattr(c, 'MATCHUP', matchup)
    monkeypatch.setattr(c, 'REPETITION', rep)
    monkeypatch.setattr(c, 'SOURCE_MAIN_COUNT', count)
    monkeypatch.setattr(c, 'OUT', tmp_path/'new')
    monkeypatch.setattr(c, 'LIVE_SOURCE_OUT', tmp_path/'original')
    assert all(lineage.game_source({'stage': 'operational', 'game_index': i}) == 'old' for i in range(6))
    for i in range(200):
        spec = {'stage': 'main', 'game_index': i}
        expected = 'old' if i < count else 'new'
        assert lineage.game_source(spec) == expected
        path = lineage.move_path(spec, 0)
        if matchup == 'terminal_parent' and rep == 1 and i == 32:
            assert path.is_relative_to(tmp_path/'original')
            assert lineage.move_path(spec, 21).is_relative_to(tmp_path/'original')
            assert lineage.move_path(spec, 22).is_relative_to(tmp_path/'new')
        else:
            assert path.is_relative_to(tmp_path/('original' if expected == 'old' else 'new'))


@pytest.mark.parametrize('stage,index', [('main', -1), ('main', 200), ('operational', 6), ('main', True), ('replay', 0)])
def test_rejects_unreserved_game_identity(stage, index):
    with pytest.raises(ValueError):
        lineage.game_source({'stage': stage, 'game_index': index})


def test_original_commitment_is_not_relabelled():
    descriptor = checkpoint.commit_descriptor()
    assert descriptor == c.artifact(c.parent.LIVE_SOURCE_OUT/'all_fits_committed.json')
    assert checkpoint.SOURCE_STUDY == 'splendorust_full_corpus_20261110_v1'
    assert c.read(c.ROOT/descriptor['path'])['study_id'] == checkpoint.SOURCE_STUDY


def test_partial_previous_segment_is_recovery_not_failed_original(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'MATCHUP', 'terminal_parent')
    monkeypatch.setattr(c, 'REPETITION', 1)
    monkeypatch.setattr(c, 'SOURCE_MAIN_COUNT', 32)
    monkeypatch.setattr(c, 'OUT', tmp_path/'new')
    monkeypatch.setattr(c, 'LIVE_SOURCE_OUT', tmp_path/'old')
    assert lineage.prior_segment('main_020') == tmp_path/'old/segments/main_020/complete.json'
    assert lineage.prior_segment('main_030') == tmp_path/'new/segments/resume_partial/complete.json'
    assert lineage.prior_segment('main_040') == tmp_path/'new/segments/main_040/complete.json'


def _saved_retry_fixture():
    base=c.parent.LIVE_SOURCE_OUT/'strength/terminal_parent/r1/games/main/g032'
    return c.read(base/'spec.json'), c.read(base/'moves/000021.json')['after']


def test_retry_token_precedes_dispatch_and_failed_token_blocks_dispatch(monkeypatch):
    from tools.splendorust_full_corpus_recovery_20261111.strength import control
    from tools.splendorust_full_corpus_recovery_20261111 import boundary
    spec, before = _saved_retry_fixture()
    monkeypatch.setattr(c, 'MATCHUP', 'terminal_parent')
    monkeypatch.setattr(c, 'REPETITION', 1)
    events=[]
    def consume(*args, **kwargs):
        events.append(('consume', kwargs))
        assert kwargs['request_id'] == 'request-00000003'
    def dispatch(*args, **kwargs):
        events.append(('dispatch', None))
        raise RuntimeError('dispatch sentinel')
    monkeypatch.setattr(boundary, 'consume_authorized_retry', consume)
    monkeypatch.setattr(control.transport, 'ordinary', dispatch)
    child=SimpleNamespace(route='control',output_dir=c.OUT/'workers/main_g032_control',_counter=2)
    with pytest.raises(RuntimeError,match='dispatch sentinel'):
        control.decision(child,spec,before,SimpleNamespace(),{},ply=22)
    assert [e[0] for e in events] == ['consume','dispatch']
    def refuse(*args, **kwargs): raise ValueError('token already consumed')
    monkeypatch.setattr(boundary, 'consume_authorized_retry', refuse)
    events.clear()
    with pytest.raises(ValueError,match='already consumed'):
        control.decision(child,spec,before,SimpleNamespace(),{},ply=22)
    assert events == []


def test_exact_replay_does_not_consume_live_retry(monkeypatch):
    from tools.splendorust_full_corpus_recovery_20261111.strength import control
    from tools.splendorust_full_corpus_recovery_20261111 import boundary
    spec,before=_saved_retry_fixture()
    monkeypatch.setattr(c,'MATCHUP','terminal_parent')
    monkeypatch.setattr(c,'REPETITION',1)
    def forbidden(*args,**kwargs): raise AssertionError('Replay tried to consume live amendment')
    def dispatch(*args,**kwargs): raise RuntimeError('replay dispatch')
    monkeypatch.setattr(boundary,'consume_authorized_retry',forbidden)
    monkeypatch.setattr(control.transport,'ordinary',dispatch)
    child=SimpleNamespace(route='control',output_dir=c.OUT/'workers/replay_control',_counter=2)
    with pytest.raises(RuntimeError,match='replay dispatch'):
        control.decision(child,spec,before,SimpleNamespace(),{},ply=22)


def test_source_worker_paths_reject_unknown_or_traversal(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'new')
    monkeypatch.setattr(c,'LIVE_SOURCE_OUT',tmp_path/'old')
    assert lineage.source_of({'worker_folder':'old/workers/main_g030_control'}) == 'old'
    assert lineage.source_of({'worker_folder':'new/workers/main_g032_control'}) == 'new'
    with pytest.raises(ValueError): lineage.source_of({'worker_folder':'unknown/workers/x'})
    with pytest.raises(ValueError): lineage.worker_folder('../escape','old')


def test_partial_segment_is_single_cohort_and_new_blocks_only(monkeypatch):
    from tools.splendorust_full_corpus_recovery_20261111.strength import control
    monkeypatch.setattr(c,'MATCHUP','mixed_parent')
    monkeypatch.setattr(c,'REPETITION',1)
    with pytest.raises(ValueError): control.segment_id('resume_partial')
    monkeypatch.setattr(c,'MATCHUP','terminal_parent')
    assert control.segment_id('resume_partial') == 'resume_partial'
    with pytest.raises(ValueError): control.segment_id('resume_partial',30)


@pytest.mark.parametrize('route', ['candidate', 'control'])
def test_actual_two_trained_player_profiles_self_map_exactly(monkeypatch, route):
    """Exercise real imported descriptors, including the trained control route."""
    from tools.splendorust_full_corpus_recovery_20261111.strength import profile_parity
    monkeypatch.setattr(c,'MATCHUP','mixed_terminal')
    monkeypatch.setattr(c,'REPETITION',0)
    monkeypatch.setattr(c,'ROUTE_ARMS',{'candidate':'mixed','control':'terminal'})
    original=c.parent.LIVE_SOURCE_OUT/'strength/mixed_terminal/r0'
    monkeypatch.setattr(c,'LIVE_SOURCE_OUT',original)
    frozen=c.read(original/'freeze.json')
    proof=profile_parity.profile_map(frozen,frozen,route)
    assert proof['exact_model_bytes'] and proof['all_scientific_profile_settings_exact']
    assert all(row['source']==row['target'] for row in proof['mapping'].values())

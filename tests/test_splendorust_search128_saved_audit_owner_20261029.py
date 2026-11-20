"""New audit clocks and exit requirements cannot promote the failed attempt."""
import inspect
from pathlib import Path
from copy import deepcopy
import pytest
from tools.splendorust_search128_saved_audit_20261029 import common as c, audit, owner
from tools.splendorust_search128_resume_20261029 import launch_owner as source_owner
from tools.splendorust_search128_resume_20261029 import independent_review as frozen


def test_owned_process_observation_is_qualified_implementation():
    assert inspect.getsource(owner.observe)==inspect.getsource(source_owner.observe)
    assert c.JOB_SECONDS==source_owner.c.JOB_SECONDS==86400
    assert c.LIMIT==3600


def test_frozen_full_review_is_unmodified_reference():
    assert audit.frozen.review_all is frozen.review_all
    assert c.file_hash(c.SOURCE/'freeze.json')==c.SOURCE_SHA
    freeze=c.read(c.SOURCE/'freeze.json')
    item=next(x for x in freeze['inputs'] if x['path'].endswith('/splendorust_search128_resume_20261029/independent_review.py'))
    assert c.artifact(Path(frozen.__file__))==item


def test_new_clock_is_clipped_by_final_even_with_hour_remaining():
    final=dict(monotonic_deadline=c.ORIGINAL_MONO+50000,unix_deadline=c.ORIGINAL_UNIX+50000)
    assert c.deadline_pair(c.ORIGINAL_MONO+49900,c.ORIGINAL_UNIX+49900,final)==(final['monotonic_deadline'],final['unix_deadline'])


def test_new_clock_does_not_exceed_one_hour_or_global():
    far=dict(monotonic_deadline=c.ORIGINAL_MONO+100000,unix_deadline=c.ORIGINAL_UNIX+100000)
    assert c.deadline_pair(c.ORIGINAL_MONO,c.ORIGINAL_UNIX,far)==(c.ORIGINAL_MONO+3600,c.ORIGINAL_UNIX+3600)
    assert c.deadline_pair(c.ORIGINAL_MONO+86390,c.ORIGINAL_UNIX+86390,far)==(c.ORIGINAL_MONO+86400,c.ORIGINAL_UNIX+86400)


def terminal_fixture():
    script=c.artifact(Path(audit.__file__))
    value=dict(status='passed_saved_recovery_component',accepted=False,audit_freeze_sha256='a',
        source_freeze_sha256=c.SOURCE_SHA,reader=dict(pid=123,created=456.),reader_source=script,
        original_attempts_accepted=False,source_attempt_accepted=False,
        model_calls=0,new_games=0,search_calls=0,scientific_replays=0,bootstrap_executions=0)
    terminal=dict(freeze_sha256='a',exit_code=0,cleanup_reason=None,observed_by='owned Process handle',
        pid=123,created=456.,component_sha256='b',script_sha256=script['sha256'])
    class Reader:
        def pin(self,path):return c.artifact(path)
    return Reader(),value,dict(sha256='b'),terminal


def test_actual_successful_reader_required():
    r,value,pin,terminal=terminal_fixture()
    audit.check_reader_terminal(r,value,pin,terminal,'a')
    for field,bad in [('exit_code',1),('pid',999),('component_sha256','bad'),('script_sha256','bad'),('cleanup_reason','killed')]:
        wrong=deepcopy(terminal);wrong[field]=bad
        with pytest.raises(ValueError):audit.check_reader_terminal(r,value,pin,wrong,'a')


def test_old_failed_attempt_cannot_be_promoted():
    r,value,pin,terminal=terminal_fixture()
    for field in ('original_attempts_accepted','source_attempt_accepted'):
        wrong=deepcopy(value);wrong[field]=True
        with pytest.raises(ValueError):audit.check_reader_terminal(r,wrong,pin,terminal,'a')
    assert c.read(c.SOURCE/'postrun/independent_review_os_exit.json')['exit_code']==1
    assert c.read(c.SOURCE/'owner_failure.json')['status']=='not_accepted'
    assert c.read(c.OWNER_EXIT)['actual_owner_exit_code']==1


def test_model_or_replay_work_cannot_be_accepted():
    r,value,pin,terminal=terminal_fixture()
    for field in ('model_calls','new_games','search_calls','scientific_replays','bootstrap_executions'):
        wrong=deepcopy(value);wrong[field]=1
        with pytest.raises(ValueError):audit.check_reader_terminal(r,wrong,pin,terminal,'a')


def test_publication_cannot_write_into_frozen_output():
    with pytest.raises(ValueError,match='separate audit output'):
        c.save(c.SOURCE/'forbidden_test_publication.json',{})
    assert not(c.SOURCE/'forbidden_test_publication.json').exists()
